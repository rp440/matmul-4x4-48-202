"""Independent phase LNS for any exactly admitted 4x4 triple (L, R or P).

Optimizes ONE phase of a circuit triple (a prefix like data/4x4x4_48_203) for a
wall-clock budget. The phase map is never changed; every improvement is
admitted through slp.verify_candidate (exact tensor + typed DAG + recount).
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
import traceback

import rebuild2 as r
import rebuild as base
import slp

ROOT = base.ROOT
Dictionary = base.Dictionary
key, direction, combine, bits = base.key, base.direction, base.combine, base.bits
fingerprint, atomic_json, write_slp = base.fingerprint, base.atomic_json, base.write_slp
rss_mb = base.rss_mb
Search = r.Search
FAMILIES = r.FAMILIES
DEFAULT_WEIGHTS = r.DEFAULT_WEIGHTS
SOURCES = ('rebuild3.py', 'rebuild2.py', 'rebuild.py', 'slp.py', 'verify_dag.py')


def load_triple(prefix):
    path = Path(prefix)
    paths = [path/f'{p}.slp' if path.is_dir() else Path(f'{prefix}_{p}.slp') for p in 'LRP']
    progs = [slp.parse_slp(p) for p in paths]
    slp.verify_candidate(progs)
    return progs


def admit(progs, phase, prog):
    pi = 'LRP'.index(phase)
    n = len(slp.eval_map(progs[0], 16)) if phase == 'P' else 16
    if slp.eval_map(prog, n) != slp.eval_map(progs[pi], n): raise ValueError('candidate changes phase map')
    candidate = list(progs)
    candidate[pi] = prog
    report = slp.verify_candidate(candidate)
    return candidate, report


def publish(progs, phase, prog, directory, announce, stage='step'):
    """Admit, write, re-read and re-admit a phase replacement."""
    candidate, verification = admit(progs, phase, prog)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, p in zip('LRP', candidate): write_slp(directory/f'{name}.slp', p)
    write_slp(directory/'best.slp', prog)
    atomic_json(directory/'verification.json', verification)
    # Verify exactly what was written before announcement.
    reread = [slp.parse_slp(directory/f'{name}.slp') for name in 'LRP']
    if slp.verify_candidate(reread)['total'] != verification['total']: raise ValueError('persisted recount changed')
    before = slp.verify_candidate(progs)['total']
    if verification['total'] < before:
        announce(dict(event='IMPROVED', stage=stage, phase=phase, before=before,
                      total=verification['total'], path=str(directory)))
    return verification


def worker(config, conn):
    start = time.monotonic()
    try:
        progs = load_triple(config["prefix"])
        pi = 'LRP'.index(config['phase'])
        search = Search(progs[pi], len(slp.eval_map(progs[0], 16)) if pi == 2 else 16, config['seed'], config['orders'],
                        config['temperature'], config['tabu'], config['weights'])
        directory = Path(config['output'])/f'worker_{config["seed"]}'
        def announce(row): conn.put(dict(row, seed=config['seed'], pid=os.getpid()))
        def commit(stage):
            return publish(progs, config['phase'], search.best_prog, directory, announce, stage)
        verification = commit('init')
        saved = search.best_cost
        next_log = 0
        while time.monotonic() < config['deadline']:
            move = search.step()
            if move:
                admit(progs, config['phase'], search.current_prog)
                search.stats['full_triple_admissions'] += 1
            if search.best_cost < saved:
                verification = commit('step')
                saved = search.best_cost
            elapsed = time.monotonic()-start
            if elapsed >= next_log:
                report = search.report(elapsed)
                report.pop('plateau_registers')
                announce(dict(event='progress', phase=config['phase'], elapsed_seconds=elapsed,
                              worker_peak_mib=rss_mb(), **report))
                next_log = elapsed+config['log_seconds']
        verification = commit('final')
        elapsed = time.monotonic()-start
        report = dict(event='worker_complete', seed=config['seed'], pid=os.getpid(), phase=config['phase'],
                      elapsed_seconds=elapsed, worker_peak_mib=rss_mb(), verification=verification, **search.report(elapsed))
        atomic_json(directory/'summary.json', report)
        conn.put(report)
    except Exception:
        conn.put(dict(event='worker_error', seed=config['seed'], pid=os.getpid(), error=traceback.format_exc()))
        raise


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists(): raise ValueError('use a fresh output directory')
    progs = load_triple(args.prefix)
    verification = slp.verify_candidate(progs)
    config = vars(args).copy()
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}
    atomic_json(output/'manifest.json', dict(config=config, baseline=verification,
                started_at=datetime.now(timezone.utc).isoformat(), source_hashes=hashes))
    ctx = mp.get_context('spawn')
    conn = ctx.Queue()
    start = time.monotonic()
    config['deadline'] = start+args.wall_seconds
    processes = [ctx.Process(target=worker, args=(dict(config, seed=args.seed+i), conn)) for i in range(args.procs)]
    for p in processes: p.start()
    rows, errors = [], []
    with (output/'progress.jsonl').open('a', buffering=1) as log:
        while len(rows)+len(errors) < len(processes):
            try: row = conn.get(timeout=1)
            except queue.Empty:
                for p in processes:
                    if p.exitcode is not None and p.pid not in {r['pid'] for r in rows+errors}:
                        errors.append(dict(event='worker_error', pid=p.pid, error=f'exit {p.exitcode} without report'))
                continue
            row['timestamp'] = datetime.now(timezone.utc).isoformat()
            log.write(json.dumps(row)+'\n')
            display = {k:v for k,v in row.items() if k != 'plateau_registers'}
            print(('!!! EXACT VERIFIED IMPROVEMENT !!! ' if row['event']=='IMPROVED' else '')+json.dumps(display), flush=True)
            if row['event'] == 'worker_complete': rows.append(row)
            elif row['event'] == 'worker_error': errors.append(row)
    for p in processes: p.join()
    elapsed = time.monotonic()-start
    merged = base.Cardinality()
    families = {f: Counter() for f in FAMILIES}
    for row in rows:
        for i, value in enumerate(row['plateau_registers']): merged.registers[i] = max(merged.registers[i], value)
        for f, st in row['family_yields'].items(): families[f].update(st)
    summary = dict(wall_seconds=elapsed, requested_wall_seconds=args.wall_seconds, workers=rows,
                   errors=errors, exitcodes=[p.exitcode for p in processes], parent_peak_mib=rss_mb(),
                   iterations=sum(r.get('iterations', 0) for r in rows),
                   best_cost=min((r['best_cost'] for r in rows), default=None),
                   best_total=min((r['verification']['total'] for r in rows), default=None),
                   merged_plateau_distinct_estimate=merged.estimate(), family_yields={f:dict(s) for f,s in families.items()},
                   source_hashes=hashes, completed_at=datetime.now(timezone.utc).isoformat())
    atomic_json(output/'summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'workers'}), flush=True)
    if errors or any(p.exitcode != 0 for p in processes): raise RuntimeError('worker failed')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=list('LRP'), default='L')
    p.add_argument('--prefix', default='data/4x4x4_48_203',
                   help='triple prefix: PREFIX_L.slp, PREFIX_R.slp, PREFIX_P.slp')
    p.add_argument('--procs', type=int, default=2)
    p.add_argument('--wall-seconds', type=float, default=900)
    p.add_argument('--orders', type=int, default=4)
    p.add_argument('--temperature', type=float, default=.2)
    p.add_argument('--tabu', type=int, default=256)
    p.add_argument('--seed', type=int, default=100)
    p.add_argument('--weights', type=float, nargs=6, default=DEFAULT_WEIGHTS, metavar='W')
    p.add_argument('--log-seconds', type=float, default=30)
    p.add_argument('--output', default='out/rebuild3')
    args = p.parse_args()
    if not 1 <= args.procs <= 4 or args.orders < 1 or args.wall_seconds <= 0 or args.tabu < 1 or args.temperature < 0:
        p.error('invalid limits')
    run(args)


if __name__ == '__main__': main()
