"""Lowering of a parsed SLP phase to an SSA gate graph, and re-emission.

lower() gives one charged operation per gate (edges carry exact Fraction
weights); emit() renumbers topologically and drops dead gates.
"""
from dataclasses import dataclass
from fractions import Fraction as F

import slp

COEFFICIENTS = (F(1), F(-1), F(2), F(-2), F(1, 2), F(-1, 2))


@dataclass
class Phase:
    n: int
    gates: dict  # id -> tuple((operand id, Fraction weight), ...)
    outputs: tuple
    labels: dict
    vectors: dict


def lower(prog, n):
    """One charged operation per SSA gate; preserve sequential overwrites."""
    env = {f'i{k}': k for k in range(n)}
    gates, labels = {}, {}
    vectors = {k: tuple(F(j == k) for j in range(n)) for k in range(n)}

    def expression(tree, label):
        if isinstance(tree, str):
            return env[tree]
        op, *args = tree
        a = expression(args[0], label)
        if op in ('add', 'subtract'):
            edges = ((a, F(1)), (expression(args[1], label), F(1 if op == 'add' else -1)))
        else:
            c = F(-1) if op == 'negative' else args[1]
            if op == 'divide':
                c = 1/c
            if c == 1:
                return a
            edges = ((a, c),)
        idx = n + len(gates)
        gates[idx], labels[idx] = edges, label
        vectors[idx] = tuple(sum((c*vectors[u][j] for u, c in edges), F(0)) for j in range(n))
        return idx

    for name, text in prog:
        env[name] = expression(slp._tree(text), name)
    outputs = slp._outputs(env)
    if set(outputs) != set(range(len(outputs))):
        raise ValueError('Non-contiguous phase outputs')
    phase = Phase(n, gates, tuple(outputs[k] for k in range(len(outputs))), labels, vectors)
    if len(gates) != slp.count_ops(prog)['total']:
        raise ValueError('SSA/count_ops discrepancy')
    return phase


def emit(gates, outputs, n):
    """Topological renumbering and DCE; available later wires can move earlier."""
    names = {k: f'i{k}' for k in range(n)}
    active, prog = set(), []

    def visit(g):
        if g in names:
            return names[g]
        if g in active:
            raise ValueError('Cyclic splice')
        active.add(g)
        edges = gates[g]
        refs = [(visit(u), c) for u, c in edges]
        if len(refs) == 1:
            a, c = refs[0]
            text = f'-{a}' if c == -1 else f'{a}*({c})'
        else:
            (a, ca), (b, cb) = refs
            if ca != 1:
                raise ValueError('Unexpected first edge weight')
            text = f'{a}{"+" if cb > 0 else "-"}{b}' if abs(cb) == 1 else f'{a}+({cb})*{b}'
        name = f't{len(prog)}'
        prog.append((name, text))
        names[g] = name
        active.remove(g)
        return name

    # Do not append output aliases until all DFS temporary names are allocated.
    refs = [visit(g) for g in outputs]
    return prog + [(f'o{k}', name) for k, name in enumerate(refs)]
