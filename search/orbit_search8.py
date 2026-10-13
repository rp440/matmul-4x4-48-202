"""Orbit search driver: apply sequences of exact orbit moves (orbit_moves.py) to
a starting circuit triple, re-optimize L, R and P independently with the LNS
(rebuild2/rebuild3), and keep the best. Multi-process, restartable, with batched
successive halving (stages A, B, C), exact rank-48 admission of every candidate,
and bounded circuit storage.

Budgets are LNS seconds PER PHASE PER SEED; transform/init/admission are extra.
Atomic Search steps may overrun. Interrupted work is never marked scored.
Promotions reuse in-memory winners (or matching top-ten checkpoints on resume).
Missing checkpoints reconstruct a new seed from prefix + full move sequence.
A candidate is "published" only if it is exactly admitted and costs < TARGET.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from fractions import Fraction as F
import gzip
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import time

import orbit
import orbit_moves as base
import normalize
import rebuild3 as r
import slp

FACTORS = base.FACTORS
STOP = None
TARGET = 202  # candidates strictly below this total are published
SOURCES = ('orbit_search8.py', 'orbit_moves.py', 'orbit.py', 'lowering.py', 'normalize.py',
           'rebuild3.py', 'rebuild2.py', 'rebuild.py', 'slp.py', 'verify_dag.py')


def admit48(progs):
    report = slp.verify_candidate(progs)
    if report['tensor']['rank'] != 48 or report['dag']['rank'] != 48:
        raise ValueError('orbit_search8 requires rank 48')
    return report


def apply_move(progs, move):
    """Delegate old moves unchanged; extend the identical GL4 shear formula."""
    if move['type'] != 'shear' or F(move['c']) in (1, -1):
        return base.apply_move(progs, move)
    axis, i, j, c = move['axis'], move['i'], move['j'], F(move['c'])
    if axis not in ('X', 'Y', 'Z') or not 0 <= i < 4 or not 0 <= j < 4 or i == j or not c:
        raise ValueError('invalid rational shear')
    admit48(progs)
    rows = {4*i+t: ((4*i+t, 1), (4*j+t, c)) for t in range(4)}
    cols = {4*t+j: ((4*t+j, 1), (4*t+i, -c)) for t in range(4)}
    result = list(progs)
    if axis == 'X':
        result[0] = orbit.input_map(result[0], 16, rows)
        result[2] = orbit.output_map(result[2],
            {4*i+t: ((4*i+t, 1), (4*j+t, -c)) for t in range(4)})
    elif axis == 'Y':
        result[0] = orbit.input_map(result[0], 16, cols)
        result[1] = orbit.input_map(result[1], 16, rows)
    else:
        result[1] = orbit.input_map(result[1], 16, cols)
        result[2] = orbit.output_map(result[2],
            {4*t+j: ((4*t+j, 1), (4*t+i, c)) for t in range(4)})
    return [normalize.cleanup(p, 48 if pi == 2 else 16, 16 if pi == 2 else 48)
            for pi, p in enumerate(result)]


def catalog(coefficients):
    # Mix families so neither the 1680 scales nor the 216 shears starve.
    shears = [dict(type='shear', axis=a, i=i, j=j, c=str(c))
        for c in coefficients for a in 'YXZ' for i in range(4)
        for j in range(4) if i != j]
    scales = [dict(type='scale', k=k, a=str(a), b=str(b))
        for a, b in itertools.product(FACTORS, repeat=2) if (a, b) != (1, 1)
        for k in sorted(range(48), key=lambda k: (k != 20, k))]
    moves = [dict(type='tau')]
    for i in range(max(len(scales), len(shears))):
        if i < len(shears): moves.append(shears[i])
        if i < len(scales): moves.append(scales[i])
    return moves


def json_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        for number, line in enumerate(f, 1):
            # Also accepts console logs with a loud prefix before JSON.
            pos = line.find('{')
            if pos < 0: continue
            try: row = json.loads(line[pos:])
            except json.JSONDecodeError: continue  # crashed/truncated tail
            if isinstance(row, dict): yield number, row


def disk_free(path):
    """df -k, POSIX output; injectable for regression tests."""
    text = subprocess.check_output(['df', '-Pk', str(path)], text=True)
    return int(text.strip().splitlines()[-1].split()[3])*1024


def disk_guard(path, minimum=5_000_000_000):
    free = disk_free(path)
    if free < minimum:
        raise OSError(f'DISK GUARD: {free} bytes free < {minimum}; stop cleanly')
    return free


class RotatingLog:
    def __init__(self, path, limit=200_000_000):
        self.path, self.limit = Path(path), limit
    def write(self, row):
        line = json.dumps(row, separators=(',', ':'))+'\n'
        with self.path.open('a') as f:
            f.write(line)
            f.flush()
        if self.path.stat().st_size > self.limit:
            number = 1
            while Path(f'{self.path}.{number:04d}.gz').exists(): number += 1
            target = Path(f'{self.path}.{number:04d}.gz')
            temp = Path(str(target)+'.tmp')
            with self.path.open('rb') as source, gzip.open(temp, 'wb') as dest:
                import shutil
                shutil.copyfileobj(source, dest)
            temp.replace(target)
            self.path.write_text('')


def point(prefix, moves, **extra):
    digest = hashlib.sha256(json.dumps([prefix, moves], sort_keys=True).encode()).hexdigest()[:24]
    return dict(point_id=digest, prefix=prefix, moves=moves, **extra)


def task_key(spec, stage, seed):
    return f'{spec["point_id"]}:{stage}:{seed}'


def stage_seeds(config, spec, stage):
    offset = int(spec['point_id'][:8], 16) % 1_000_000
    return [config['seed']+offset*100+10*'ABC'.index(stage)+i
            for i in range(config[f'{stage.lower()}_seeds'])]


def worker_init(stop):
    global STOP
    STOP = stop
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def interrupted(config):
    return time.monotonic() >= config['deadline'] or (STOP is not None and STOP.is_set())


def publish(progs, config, record):
    report = admit48(progs)  # rank check BEFORE any candidate file is written
    if report['total'] >= TARGET: return None
    directory = Path(config['candidates'])/f'orbit_search8_{base.circuit_hash(progs)}_{report["total"]}'
    path, checked = base.save_triple(progs, directory, record)
    temp = directory/f'.record.{os.getpid()}.tmp'
    temp.write_text(json.dumps(record, indent=2)+'\n')
    temp.replace(directory/'record.json')  # process-specific staging, same-DAG race safe
    if admit48(r.load_triple(directory)) != checked: raise ValueError('saved candidate changed')
    print('!!! EXACT ADMITTED RANK-48 CANDIDATE BELOW '+str(TARGET)+' !!! '+path, flush=True)
    return path


def evaluate(job):
    """Exact admission at every transform / accepted step / init / best."""
    config, spec, stage, seed, starting = job
    start = time.monotonic()
    seeded = r.load_triple(spec['prefix'])
    admit48(seeded)
    for move in spec['moves']:
        if interrupted(config): return None, None
        seeded = apply_move(seeded, move)
        admit48(seeded)
    reconstructed_hash = base.circuit_hash(seeded)
    if starting is not None:
        if slp.load_progs(*starting) != slp.load_progs(*seeded):
            raise ValueError('promotion checkpoint changes orbit phase maps')
        admit48(starting)
        seeded = starting
    report = admit48(seeded)
    result, phases, discoveries = list(seeded), [], []
    record_base = dict(point_id=spec['point_id'], prefix=spec['prefix'],
        move_sequence=spec['moves'], stage=stage, seed=seed,
        seeds=[seed], planned_stage_seeds=stage_seeds(config, spec, stage),
        phase_seeds={p: seed+i for i, p in enumerate('LRP')},
        seed_policy='promoted_winner' if starting is not None else 'reconstructed_new_seed',
        starting_circuit_hash=base.circuit_hash(seeded), reconstructed_circuit_hash=reconstructed_hash)
    def inspect(candidate, where):
        if sum(base.costs(candidate)) < TARGET:
            path = publish(candidate, config, dict(record_base, admission_stage=where))
            if path and path not in discoveries: discoveries.append(path)
    inspect(seeded, 'seeded')
    for pi, prog in enumerate(seeded):
        if interrupted(config): return None, None
        began = time.monotonic()
        search = r.Search(prog, 48 if pi == 2 else 16, seed+pi, config['orders'],
            config['temperature'], config['tabu'], config['weights'])
        result, _ = r.admit(result, 'LRP'[pi], search.best_prog)
        inspect(result, f'{"LRP"[pi]}_init')
        deadline = min(config['deadline'], time.monotonic()+config[f'{stage.lower()}_seconds'])
        while time.monotonic() < deadline and not interrupted(config):
            if search.step():
                admitted, _ = r.admit(result, 'LRP'[pi], search.current_prog)
                inspect(admitted, f'{"LRP"[pi]}_step')
        if interrupted(config): return None, None
        result, _ = r.admit(result, 'LRP'[pi], search.best_prog)
        inspect(result, f'{"LRP"[pi]}_best')
        phases.append(dict(phase='LRP'[pi], cost=search.best_cost,
            iterations=search.stats['iterations'], seconds=time.monotonic()-began,
            max_D=search.max_D))
        del search  # one cache/solver at a time
    verified = admit48(result)
    if verified['total'] > report['total']: raise ValueError('optimizer worsened admitted seed')
    inspect(result, 'final')
    row = dict(record_base, event='point_complete', task_key=task_key(spec, stage, seed),
        seeded_total=report['total'], optimized_total=verified['total'],
        optimized_costs=base.costs(result), seconds=time.monotonic()-start,
        phases=phases, worker_peak_mib=r.rss_mb(), pid=os.getpid(), discoveries=discoveries)
    return result, row


def score(spec, stage, config, records):
    rows = [records[task_key(spec, stage, seed)] for seed in stage_seeds(config, spec, stage)]
    return sum(min(row['optimized_costs'][pi] for row in rows) for pi in range(3))


def merge_phase_best(old, candidate):
    """Combine independent seed winners ONLY with exact identical phase maps."""
    admit48(old)
    admit48(candidate)
    if slp.load_progs(*old) != slp.load_progs(*candidate):
        raise ValueError('phase winners have different maps')
    result, changed = list(old), []
    for pi in range(3):
        if slp.count_ops(candidate[pi])['total'] < slp.count_ops(result[pi])['total']:
            result, _ = r.admit(result, 'LRP'[pi], candidate[pi])
            changed.append(pi)
    admit48(result)
    return result, changed


def promote(cohort, stage, fraction, config, records):
    return sorted(cohort, key=lambda s: (score(s, stage, config, records), s['point_id']))[
        :max(1, math.ceil(len(cohort)*fraction))] if cohort else []


def next_point(state, moves, depth):
    # Historical queues and fresh frontiers alternate from the first batch.
    state['generation_turn'] = state.get('generation_turn', 0)+1
    if state['generation_turn'] % 2 or not state['frontiers']:
        while state['priority']:
            spec = state['priority'].pop(0)
            if spec['point_id'] not in state['generated']:
                state['generated'].append(spec['point_id'])
                return spec
    while True:
        available = [f for f in state['frontiers'] if len(f['moves']) < depth and f['cursor'] < len(moves)]
        if not available:
            while state['priority']:
                spec = state['priority'].pop(0)
                if spec['point_id'] not in state['generated']:
                    state['generated'].append(spec['point_id'])
                    return spec
            return None
        frontier = min(available, key=lambda f: (f['cursor'], len(f['moves']), f['point_id']))
        move = moves[frontier['cursor']]
        frontier['cursor'] += 1
        spec = point(frontier['prefix'], frontier['moves']+[move])
        if spec['point_id'] not in state['generated']:
            state['generated'].append(spec['point_id'])
            return spec


def update_frontiers(state, cohort, stage, config, records):
    proposals = {f['point_id']: f for f in state['frontiers']}
    for spec in cohort:
        if len(spec['moves']) >= config['depth']: continue
        total = score(spec, stage, config, records)
        if spec['point_id'] in proposals:
            proposals[spec['point_id']]['total'] = min(total, proposals[spec['point_id']]['total'])
        else: proposals[spec['point_id']] = dict(spec, total=total, cursor=0)
    roots = [f for f in proposals.values() if not f['moves']]
    selected = []
    for d in range(1, config['depth']):
        selected.extend(sorted((f for f in proposals.values() if len(f['moves']) == d),
            key=lambda f: (f['total'], f['point_id']))[:config['beam']])
    state['frontiers'] = roots+selected


def save_top(progs, row, state, output):
    digest = base.circuit_hash(progs)
    if any(t['hash'] == digest for t in state['top']): return
    entry = dict(total=row['optimized_total'], hash=digest, point_id=row['point_id'], stage=row['stage'], seed=row['seed'])
    if len(state['top']) < 10:
        entry['slot'] = next(i for i in range(10) if i not in {t['slot'] for t in state['top']})
    else:
        worst = max(state['top'], key=lambda t: (t['total'], t['hash']))
        if (entry['total'], digest) >= (worst['total'], worst['hash']): return
        entry['slot'] = worst['slot']
        state['top'].remove(worst)
    base.save_triple(progs, output/'top10'/str(entry['slot']), row)
    # base writes a unique provenance per point. Replace it with ONE record.
    directory = output/'top10'/str(entry['slot'])
    for path in directory.glob('record_*.json'): path.unlink()
    r.atomic_json(directory/'record.json', row)
    state['top'].append(entry)


def run(args):
    output = Path(args.resume or args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    free = disk_guard(output)
    hashes = {name: hashlib.sha256((r.ROOT/name).read_bytes()).hexdigest() for name in SOURCES}
    if args.resume:
        manifest = json.loads((output/'manifest.json').read_text())
        if hashes != manifest['source_hashes']: raise ValueError('resume source hashes changed; review before resuming')
        config = dict(manifest['config'], procs=args.procs)
        state = json.loads((output/'state.json').read_text())
    else:
        if (output/'manifest.json').exists(): raise ValueError('use fresh --output or --resume')
        config = vars(args).copy()
        config['prefixes'] = [str((r.ROOT/p).resolve()) for p in args.prefixes]
        config['coefficients'] = [str(F(c)) for c in args.coefficients.split(',')]
        roots = [point(p, [], cursor=0, total=admit48(r.load_triple(p))['total']) for p in config['prefixes']]
        priority = [point(f['prefix'], []) for f in roots]
        state = dict(priority=priority, frontiers=roots, generated=[], cohort=[], stage='A', top=[], batches=0)
        r.atomic_json(output/'manifest.json', dict(config=config, source_hashes=hashes,
            started_at=datetime.now(timezone.utc).isoformat(),
            budget_policy='LNS seconds per independent phase per seed; initialization/admission extra'))
        snapshot = output/'source_snapshot'
        snapshot.mkdir()
        for name in hashes: (snapshot/name).write_bytes((r.ROOT/name).read_bytes())
    progress = RotatingLog(output/'progress.jsonl')
    console = RotatingLog(output/'points.log')
    records = {}
    for path in sorted(output.glob('progress.jsonl*.gz'))+[output/'progress.jsonl']:
        if path.exists():
            for _, row in json_rows(path):
                if row.get('event') == 'point_complete': records[row['task_key']] = row
    start = time.monotonic()
    config.update(deadline=start+args.wall_seconds, candidates=str(Path(config['candidates']).resolve()))
    moves = catalog([F(c) for c in config['coefficients']])
    ctx = mp.get_context('spawn')
    stop = ctx.Event()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous: signal.signal(sig, lambda *_: stop.set())
    initial_bytes = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    completed, errors, peaks = 0, [], {}
    session_points = set()
    winners = {}  # active batch only, <=batch_size small exact triples
    phase_sources = {}
    next_disk, reason = start+600, 'wall_limit'
    def checkpoint(): r.atomic_json(output/'state.json', state)
    checkpoint()
    try:
        with ProcessPoolExecutor(max_workers=args.procs, mp_context=ctx,
                initializer=worker_init, initargs=(stop,)) as pool:
            while not stop.is_set() and time.monotonic() < config['deadline']:
                if not state['cohort']:
                    for _ in range(config['batch_size']):
                        spec = next_point(state, moves, config['depth'])
                        if spec is None: break
                        state['cohort'].append(spec)
                    state['stage'] = 'A'
                    checkpoint()
                    if not state['cohort']:
                        reason = 'catalog_exhausted'
                        break
                stage = state['stage']
                for spec in state['cohort']:
                    if spec['point_id'] not in winners:
                        top = [t for t in state['top'] if t['point_id'] == spec['point_id']]
                        if top:
                            best = min(top, key=lambda t: t['total'])
                            saved = r.load_triple(output/'top10'/str(best['slot']))
                            if admit48(saved)['total'] != best['total'] or base.circuit_hash(saved) != best['hash']:
                                raise ValueError('top-ten checkpoint hash/recount changed')
                            winners[spec['point_id']] = (best['total'], saved)
                            provenance = json.loads((output/'top10'/str(best['slot'])/'record.json').read_text())
                            phase_sources[spec['point_id']] = provenance.get('phase_sources', {})
                jobs = [(config, spec, stage, seed, winners.get(spec['point_id'], (0, None))[1]) for spec in state['cohort']
                    for seed in stage_seeds(config, spec, stage)
                    if task_key(spec, stage, seed) not in records]
                pending, cursor = {}, 0
                while cursor < len(jobs) or pending:
                    now = time.monotonic()
                    if now >= next_disk:
                        try:
                            free = disk_guard(output)
                            disk_guard(config['candidates'] if Path(config['candidates']).exists() else output)
                            console.write(dict(event='disk_check', free_bytes=free))
                        except OSError as exc:
                            reason = str(exc)
                            stop.set()
                        next_disk = now+600
                    if now >= config['deadline']: stop.set()
                    while not stop.is_set() and cursor < len(jobs) and len(pending) < args.procs:
                        job = jobs[cursor]
                        pending[pool.submit(evaluate, job)] = job
                        cursor += 1
                    if not pending: break
                    done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                    for future in done:
                        job = pending.pop(future)
                        try: progs, row = future.result()
                        except Exception as exc:
                            errors.append(dict(point_id=job[1]['point_id'], stage=stage, seed=job[3], error=repr(exc)))
                            progress.write(dict(event='point_error', **errors[-1]))
                            reason = 'worker_error'
                            stop.set()
                            continue
                        if row is None: continue
                        row['timestamp'] = datetime.now(timezone.utc).isoformat()
                        pid = row['point_id']
                        sources = phase_sources.setdefault(pid, {})
                        if pid in winners:
                            combined, changed = merge_phase_best(winners[pid][1], progs)
                        else:
                            combined, changed = progs, range(3)
                        for pi in changed:
                            sources['LRP'[pi]] = dict(seed=row['seed'],
                                phase_seed=row.get('phase_seeds', {}).get('LRP'[pi], row['seed']+pi),
                                stage=row['stage'], task_key=row['task_key'])
                        combined_costs = base.costs(combined)
                        winners[pid] = (sum(combined_costs), combined)
                        merged_record = dict(row, optimized_total=sum(combined_costs), optimized_costs=combined_costs,
                            phase_sources=dict(sources), seeds=sorted({s['seed'] for s in sources.values()}))
                        if sum(combined_costs) < TARGET:
                            path = publish(combined, config, merged_record)
                            if path and path not in row['discoveries']: row['discoveries'].append(path)
                            console.write(dict(event='CANDIDATE_BELOW_202', path=path,
                                total=sum(combined_costs), point_id=pid))
                        row['combined_total'] = sum(combined_costs)
                        # Persist score first: crash => completed task remains skipped.
                        progress.write(row)
                        records[row['task_key']] = row
                        completed += 1
                        session_points.add(row['point_id'])
                        peaks[row['pid']] = max(peaks.get(row['pid'], 0), row['worker_peak_mib'])
                        save_top(combined, merged_record, state, output)
                        checkpoint()
                        display = {k: row[k] for k in ('point_id', 'stage', 'seed', 'move_sequence',
                            'optimized_total', 'optimized_costs', 'seconds', 'worker_peak_mib')}
                        console.write(display)
                        print(json.dumps(display), flush=True)
                    if stop.is_set() and not pending: break
                if stop.is_set(): break
                if any(task_key(s, stage, seed) not in records for s in state['cohort']
                       for seed in stage_seeds(config, s, stage)): break
                update_frontiers(state, state['cohort'], stage, config, records)
                if stage == 'A':
                    state['cohort'] = promote(state['cohort'], stage, config['a_fraction'], config, records)
                    state['stage'] = 'B'
                elif stage == 'B':
                    state['cohort'] = promote(state['cohort'], stage, config['b_fraction'], config, records)
                    state['stage'] = 'C'
                else:
                    state['cohort'] = []
                    state['batches'] += 1
                winners = {s['point_id']: winners[s['point_id']] for s in state['cohort'] if s['point_id'] in winners}
                phase_sources = {pid: phase_sources[pid] for pid in winners if pid in phase_sources}
                checkpoint()
    finally:
        checkpoint()
        for sig, handler in previous.items(): signal.signal(sig, handler)
    elapsed = time.monotonic()-start
    size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    summary = dict(event='run_complete', reason=reason, wall_seconds=elapsed,
        scored_tasks=completed, unique_points=len({row['point_id'] for row in records.values()}),
        tasks_per_hour=completed*3600/elapsed,
        points_per_hour=len(session_points)*3600/elapsed,
        best_total=min((x.get('combined_total', x['optimized_total']) for x in records.values()), default=202),
        worker_peak_mib=peaks, parent_peak_mib=r.rss_mb(), errors=errors,
        disk_growth_bytes=size-initial_bytes, disk_growth_bytes_per_hour=(size-initial_bytes)*3600/elapsed,
        free_bytes=disk_free(output),
        candidates=sorted({p for x in records.values() for p in x.get('discoveries', [])}),
        stages={s: sum(x['stage'] == s for x in records.values()) for s in 'ABC'})
    r.atomic_json(output/'summary.json', summary)
    print(json.dumps(summary), flush=True)
    if errors: raise RuntimeError('worker failed; completed work saved, failed work remains pending')
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefixes', nargs='+', default=['data/4x4x4_48_202', 'data/4x4x4_48_203', 'data/4x4x4_48_204'],
                   help='triple prefixes (PREFIX_L.slp etc.), relative to this directory')
    p.add_argument('--procs', type=int, default=8)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--beam', type=int, default=8, help='frontier width per depth')
    p.add_argument('--coefficients', default='1,-1,2,-2,1/2,-1/2')
    p.add_argument('--wall-seconds', type=float, default=36000)
    for stage, seconds, seeds in [('a', 60, 1), ('b', 300, 3), ('c', 1200, 2)]:
        p.add_argument(f'--{stage}-seconds', type=float, default=seconds)
        p.add_argument(f'--{stage}-seeds', type=int, default=seeds)
    p.add_argument('--a-fraction', type=float, default=.25)
    p.add_argument('--b-fraction', type=float, default=.25)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--orders', type=int, default=4)
    p.add_argument('--temperature', type=float, default=.35)
    p.add_argument('--tabu', type=int, default=1024)
    p.add_argument('--seed', type=int, default=26001)
    p.add_argument('--weights', nargs=6, type=float, default=r.DEFAULT_WEIGHTS)
    p.add_argument('--output', default='out/orbit_search8')
    p.add_argument('--resume', help='existing run directory; saved search settings are reused')
    p.add_argument('--candidates', default='out/candidates')
    return p


def main():
    p = parser()
    args = p.parse_args()
    try: coeffs = [F(c) for c in args.coefficients.split(',')]
    except (ValueError, ZeroDivisionError): p.error('invalid exact shear coefficients')
    if not coeffs or any(not c for c in coeffs): p.error('shear coefficients must be nonzero')
    if not 1 <= args.procs <= 8 or not 1 <= args.depth <= 4 or args.beam < 1 or args.batch_size < 1:
        p.error('invalid worker/depth/beam/batch limits')
    if min(args.a_seconds, args.b_seconds, args.c_seconds, args.wall_seconds) <= 0 or min(args.a_seeds, args.b_seeds, args.c_seeds) < 1:
        p.error('positive stage budgets/seeds required')
    if not 0 < args.a_fraction <= 1 or not 0 < args.b_fraction <= 1:
        p.error('invalid promotion fractions')
    if args.orders < 1 or args.temperature < 0 or args.tabu < 1 or min(args.weights) < 0 or sum(args.weights) <= 0:
        p.error('invalid rebuild3 parameters')
    run(args)


if __name__ == '__main__': main()
