"""Exact circuit transformations used by the orbit moves.

emit_general: normalize a weighted gate graph back to an SLP.
output_scale: scale one output of a phase by a constant (fusing scale chains).
input_map / output_map: substitute inputs or recombine outputs with
explicitly charged preprocessing / postprocessing gates.
"""
from fractions import Fraction as F

import lowering
import normalize

FACTORS = lowering.COEFFICIENTS


def emit_general(gates, outputs, n):
    """Normalize weighted first operands for the shared emitter, then signs/DCE."""
    gates = dict(gates)
    # Bypass unary ancestors, without rewriting their other consumers. DCE
    # later removes only chains that have actually become unused.
    for g, edges in list(gates.items()):
        if len(edges) == 1:
            u, c = edges[0]
            seen = {g}
            while u in gates and len(gates[u]) == 1:
                if u in seen: raise ValueError('Cyclic scalar chain')
                seen.add(u)
                u, d = gates[u][0]
                c *= d
            gates[g] = ((u, c),)
    idx = max([n-1, *gates])+1
    for g, edges in list(gates.items()):
        if len(edges) == 2 and edges[0][1] != 1:
            if edges[1][1] == 1:
                gates[g] = tuple(reversed(edges))
            else:
                u, c = edges[0]
                gates[idx] = ((u, c),)
                gates[g] = ((idx, F(1)), edges[1])
                idx += 1
    return normalize.cleanup(lowering.emit(gates, outputs, n), n, len(outputs))


def output_scale(prog, n, k, factor):
    phase = lowering.lower(prog, n)
    gates, outputs = dict(phase.gates), list(phase.outputs)
    idx = max(phase.vectors)+1
    def scaled(u, c):
        nonlocal idx
        # Fuse any existing chain of unary scales; preserve other consumers.
        if u in gates and len(gates[u]) == 1:
            v, d = gates[u][0]
            return scaled(v, c*d)
        if c == 1: return u
        result = idx
        gates[result] = ((u, c),)
        idx += 1
        return result
    outputs[k] = scaled(outputs[k], F(factor))
    return emit_general(gates, tuple(outputs), n)


def input_map(prog, n, combinations):
    """Substitute original inputs with explicitly charged preprocessing gates."""
    phase = lowering.lower(prog, n)
    gates = dict(phase.gates)
    idx = max(phase.vectors)+1
    aliases = {}
    for target, edges in combinations.items():
        gates[idx] = tuple((u, F(c)) for u, c in edges)
        aliases[target] = idx
        idx += 1
    # Rewrite only original circuit edges, not simultaneous pre-map definitions.
    for g, edges in phase.gates.items():
        gates[g] = tuple((aliases.get(u, u), c) for u, c in edges)
    outputs = tuple(aliases.get(g, g) for g in phase.outputs)
    return emit_general(gates, outputs, n)


def output_map(prog, combinations):
    phase = lowering.lower(prog, 48)
    gates, outputs = dict(phase.gates), list(phase.outputs)
    idx = max(phase.vectors)+1
    for target, edges in combinations.items():
        gates[idx] = tuple((phase.outputs[u], F(c)) for u, c in edges)
        outputs[target] = idx
        idx += 1
    return emit_general(gates, tuple(outputs), 48)

