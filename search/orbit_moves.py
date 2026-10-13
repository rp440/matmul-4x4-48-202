"""Exact orbit moves on an L/R/P circuit triple, plus triple hashing/saving.

Moves (each is an exact isotropy of the rank-48 4x4 matmul tensor):
  tau    transpose symmetry: swaps the roles of L and R, transposes C.
  scale  multiply product k's L-output by a and R-output by b, and the k-th
         input of P by 1/(a*b).  a, b in {1,-1,2,-2,1/2,-1/2}.
  shear  GL4 elementary shear on axis X, Y or Z (rows/columns i,j; c = +-1 or
         any nonzero rational via orbit_search8.apply_move).
Every changed input/output costs explicitly charged gates; results are
normalized by normalize.cleanup, and callers re-verify with slp.verify_candidate.
"""
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import re

import lowering
import orbit
import rebuild3 as r
import normalize
import slp

FACTORS = (F(1), F(-1), F(2), F(-2), F(1, 2), F(-1, 2))


def costs(progs):
    return [slp.count_ops(p)['total'] for p in progs]


def circuit_hash(progs):
    # Charged typed DAG ignores SSA spelling and free aliases. Phase-map hashes
    # would collapse distinct plateau circuits; raw SLP hashes overcount copies.
    return hashlib.sha256(json.dumps(slp.to_dag(*progs), sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def map_hash(progs):
    return hashlib.sha256(repr(slp.load_progs(*progs)).encode()).hexdigest()


def tau(progs):
    """(L, R, P) -> (R with transposed inputs, L with transposed inputs, P with transposed outputs)."""
    def transpose_inputs(prog):
        return [(name, re.sub(r'\bi(\d+)\b', lambda m: f'i{4*(int(m[1])%4)+int(m[1])//4}', text))
                for name, text in prog]
    # Rename old outputs first to avoid sequential output alias overwrites.
    # Unique prefix also makes tau twice safe on the renamed circuit.
    prefix = 'tau_'
    while any(name.startswith(prefix) for name, _ in progs[2]): prefix += 'x_'
    renamed = {name: prefix+name for name, _ in progs[2]}
    p = [(renamed[name], re.sub(r'\b[A-Za-z_]\w*\b',
          lambda m: renamed.get(m[0], m[0]), text)) for name, text in progs[2]]
    p += [(f'o{k}', renamed[f'o{4*(k%4)+k//4}']) for k in range(16)]
    return [transpose_inputs(progs[1]), transpose_inputs(progs[0]), p]


def apply_move(progs, move):
    """Apply one orbit move with actual product rank taken from L."""
    rank = len(slp.eval_map(progs[0], 16))
    if move['type'] == 'identity':
        return list(progs)
    if move['type'] == 'tau':
        return tau(progs)
    result = list(progs)
    if move['type'] == 'scale':
        k, a, b = move['k'], F(move['a']), F(move['b'])
        if not 0 <= k < rank or a not in FACTORS or b not in FACTORS:
            raise ValueError('invalid scale')
        result[0] = orbit.output_scale(result[0], 16, k, a)
        result[1] = orbit.output_scale(result[1], 16, k, b)
        if a*b != 1:
            result[2] = orbit.input_map(result[2], rank, {k: ((k, 1/(a*b)),)})
    elif move['type'] == 'shear':
        axis, i, j, c = move['axis'], move['i'], move['j'], F(move['c'])
        if axis not in ('X', 'Y', 'Z') or not 0 <= i < 4 or not 0 <= j < 4 or i == j or c not in (-1, 1):
            raise ValueError('invalid elementary shear')
        rows = {4*i+t: ((4*i+t, 1), (4*j+t, c)) for t in range(4)}
        cols = {4*t+j: ((4*t+j, 1), (4*t+i, -c)) for t in range(4)}
        def output_map(prog, combinations):
            phase = lowering.lower(prog, rank)
            gates, outputs = dict(phase.gates), list(phase.outputs)
            idx = max(phase.vectors)+1
            for target, edges in combinations.items():
                gates[idx] = tuple((phase.outputs[u], F(v)) for u, v in edges)
                outputs[target] = idx
                idx += 1
            return orbit.emit_general(gates, tuple(outputs), rank)
        if axis == 'X':
            result[0] = orbit.input_map(result[0], 16, rows)
            result[2] = output_map(result[2], {4*i+t: ((4*i+t, 1), (4*j+t, -c)) for t in range(4)})
        elif axis == 'Y':
            result[0] = orbit.input_map(result[0], 16, cols)
            result[1] = orbit.input_map(result[1], 16, rows)
        else:
            result[1] = orbit.input_map(result[1], 16, cols)
            result[2] = output_map(result[2], {4*t+j: ((4*t+j, 1), (4*t+i, c)) for t in range(4)})
    else:
        raise ValueError('unknown orbit move')
    return [normalize.cleanup(p, rank if pi == 2 else 16, 16 if pi == 2 else rank)
            for pi, p in enumerate(result)]


def save_triple(progs, directory, record):
    """Admit, atomically save, reparse, re-admit before reporting a path."""
    report = slp.verify_candidate(progs)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for phase, prog in zip('LRP', progs):
        temp = directory/f'.{phase}.{os.getpid()}.tmp'
        r.write_slp(temp, prog)
        temp.replace(directory/f'{phase}.slp')
    def atomic_json(path, value):
        # Workers can discover the same circuit concurrently, so use a
        # process-specific staging file.
        temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
        temp.write_text(json.dumps(value, indent=2)+'\n')
        temp.replace(path)
    atomic_json(directory/'dag.json', slp.to_dag(*progs))
    atomic_json(directory/'verification.json', report)
    # Same digest can be discovered concurrently: provenance is per point.
    provenance = directory/f'record_{record.get("point_id", "initial")}_{record.get("stage", "final")}_{os.getpid()}.json'
    atomic_json(provenance, record)
    saved = r.load_triple(directory)
    if circuit_hash(saved) != circuit_hash(progs) or slp.verify_candidate(saved) != report:
        raise ValueError('persisted admission/hash changed')
    return str(directory), report
