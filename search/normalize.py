"""Exact SSA normalization of a linear SLP: sign propagation, subtraction
reordering, exact value reuse (interning) and dead-code elimination.

Never changes the linear map; the operation count is recounted by slp.count_ops.
"""
from fractions import Fraction as F

import slp


def combine(a, b, c=1):
    return tuple(x + c * y for x, y in zip(a, b))


class Builder:
    """Exact value interning; emit canonical subtracts, and charge negations."""
    def __init__(self, n):
        self.n = n
        self.prog = []
        self.names = [f'i{k}' for k in range(n)]
        self.values = [tuple(F(j == k) for j in range(n)) for k in range(n)]
        self.lookup = {v: i for i, v in enumerate(self.values)}

    def emit(self, expr, value):
        value = tuple(value)
        if value in self.lookup:
            return self.lookup[value]
        idx = len(self.values)
        name = f't{idx - self.n}'
        self.prog.append((name, expr))
        self.values.append(value)
        self.names.append(name)
        self.lookup[value] = idx
        return idx

    def scale(self, a, c):
        c = F(c)
        if c == 1:
            return a
        name = self.names[a]
        expr = f'-{name}' if c == -1 else f'{name}*({c})'
        return self.emit(expr, tuple(c*x for x in self.values[a]))

    def add(self, a, b, c=1):
        c = F(c)
        value = combine(self.values[a], self.values[b], c)
        if value in self.lookup:
            return self.lookup[value]
        b = self.scale(b, abs(c))
        op = '+' if c >= 0 else '-'
        return self.emit(f'{self.names[a]}{op}{self.names[b]}', value)


def _lower(prog, n, m):
    """Lower slp's parsed trees to SSA weighted edges, preserving overwrites."""
    env = {f'i{k}': k for k in range(n)}
    gates = []

    def expression(tree):
        if isinstance(tree, str):
            return env[tree]
        op, *args = tree
        a = expression(args[0])
        if op in ('add', 'subtract'):
            edges = [(a, F(1)), (expression(args[1]), F(1 if op == 'add' else -1))]
        else:
            c = F(-1) if op == 'negative' else args[1]
            if op == 'divide':
                c = 1/c
            if c == 1:
                return a
            edges = [(a, c)]
        idx = n + len(gates)
        gates.append(edges)
        return idx

    for name, text in prog:
        env[name] = expression(slp._tree(text))
    outputs = [env[f'o{k}'] for k in range(m)]
    return gates, outputs


def cleanup(prog, n, m):
    """SSA sign propagation, subtraction reordering, exact reuse and DCE."""
    gates, outputs = _lower(prog, n, m)
    live = set(outputs)
    for k in range(len(gates)-1, -1, -1):
        if n+k in live:
            live.update(a for a, _ in gates[k])
    builder = Builder(n)
    # Delay signs until a consumer: -a+b becomes b-a; -(a+b) stays a
    # signed reference, so no negation is paid unless an output needs it.
    refs = {k: (k, F(1)) for k in range(n)}
    for k, edges in enumerate(gates):
        if n+k not in live:
            continue
        if len(edges) == 1:
            a, c = edges[0]
            idx, sign = refs[a]
            c *= sign
            # A negative nonunit scalar is one charged operation already;
            # splitting it into positive scaling + output negation costs two.
            refs[n+k] = ((idx, c) if abs(c) == 1 else
                         (builder.scale(idx, c), F(1)))
        else:
            terms = [(refs[a][0], c*refs[a][1]) for a, c in edges]
            terms.sort(key=lambda item: item[1] < 0)
            (a, ca), (b, cb) = terms
            sign = F(-1 if ca < 0 else 1)
            idx = builder.add(builder.scale(a, abs(ca)),
                              builder.scale(b, abs(cb)), cb*sign/abs(cb))
            refs[n+k] = (idx, sign)
    for k, idx in enumerate(outputs):
        ref, sign = refs[idx]
        builder.prog.append((f'o{k}', builder.names[builder.scale(ref, sign)]))
    # Exact interning can make an earlier live path dead (e.g. (3*x-x)/2
    # becomes x). Repeat only when gates disappeared, so this terminates.
    live_gates = sum(n+k in live for k in range(len(gates)))
    if len(builder.values)-n < live_gates:
        return cleanup(builder.prog, n, m)
    return builder.prog
