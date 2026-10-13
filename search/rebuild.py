"""Exact dictionary / catalog machinery for the circuit LNS (used by rebuild2).

Directions identify ONLY v and -v. Magnitudes are never normalized away.
Dictionary construction uses Fractions; the production join and closure use
injective integer packing and integer dependency masks.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import sys
from fractions import Fraction as F

import slp

ROOT = Path(__file__).resolve().parent
MIB = 1024**2


def key(v):
    """Exact integer-scaled tuple: denominator followed by integer coordinates."""
    denominator = math.lcm(*(x.denominator for x in v))
    return (denominator, *(int(x * denominator) for x in v))


def direction(v):
    # Last nonzero positive agrees with the sign of balanced radix packing.
    sign = next((-1 if x < 0 else 1 for x in reversed(v) if x), 1)
    return (tuple(sign * x for x in v), sign)


def combine(a, b, ca=1, cb=1):
    return tuple(ca * x + cb * y for x, y in zip(a, b))


def bits(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def fingerprint(values):
    return hashlib.sha256(repr(sorted(key(v) for v in values)).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def write_slp(path, prog):
    Path(path).write_text(''.join(f'{name}:={expression};\n' for name, expression in prog))


class Dictionary:
    def __init__(self, n, values=(), targets=(), preferences=None, witness=None):
        self.n = n
        self.values = []
        self.exact_index = {}
        self.index = {}
        self.preferred = []
        self.witness = witness or {}
        for i in range(n):
            self.insert(tuple(F(i == j) for j in range(n)))
        for v in values:
            self.insert(v)
        self.targets = [self.insert(v) for v in targets]
        self.inputs = (1 << n) - 1
        self.required = self.inputs
        self.wants = {}
        for i, sign in self.targets:
            self.required |= 1 << i
            self.wants.setdefault(i, set()).add(sign)
        if preferences:
            for v, sign in preferences.items():
                self.preferred[self.index[key(v)]] = sign
        for i, signs in self.wants.items():
            self.preferred[i] = min(signs)

    def insert(self, vector):
        vector = tuple(F(x) for x in vector)
        if len(vector) != self.n:
            raise ValueError('vector dimension mismatch')
        v, sign = direction(vector)
        k = key(v)
        if k not in self.index:
            self.index[k] = len(self.values)
            self.values.append(v)
            self.preferred.append(sign)
        i = self.index[k]
        self.exact_index[key(vector)] = (i, sign)
        self.exact_index[key(tuple(-x for x in vector))] = (i, -sign if any(v) else 1)
        return i, sign

    @classmethod
    def from_program(cls, prog, n):
        """Expose every charged nested expression, not just assignment values."""
        pool = cls(n)
        env = {f'i{i}': (pool.values[i], i, 1) for i in range(n)}
        wanted = {}

        def evaluate(tree):
            if isinstance(tree, str):
                return env[tree]
            op, *args = tree
            u, ui, us = evaluate(args[0])
            if op in ('add', 'subtract'):
                v, vi, vs = evaluate(args[1])
                c = 1 if op == 'add' else -1
                result = combine(u, v, 1, c)
                x, xs = pool.insert(result)
                production = (ui, vi, xs * us, xs * c * vs)
            else:
                c = -1 if op == 'negative' else args[1]
                if op == 'divide':
                    c = 1 / c
                result = tuple(c * a for a in u)
                x, xs = pool.insert(result)
                production = (ui, -1, c * xs * us, 0)
            if x >= n and x not in pool.witness and x not in production[:2]:
                pool.witness[x] = production
                pool.preferred[x] = xs
            return result, x, xs

        for name, expression in prog:
            env[name] = evaluate(slp._tree(expression))
            if re.fullmatch(r'o\d+', name):
                wanted[int(name[1:])] = env[name][0]
        if set(wanted) != set(range(len(wanted))):
            raise ValueError('noncontiguous outputs')
        pool.targets = [pool.insert(wanted[i]) for i in range(len(wanted))]
        pool.inputs = (1 << n) - 1
        pool.required = pool.inputs
        pool.wants = {}
        for i, sign in pool.targets:
            pool.required |= 1 << i
            pool.wants.setdefault(i, set()).add(sign)
        for i, signs in pool.wants.items():
            pool.preferred[i] = min(signs)
        return pool

    @property
    def full(self):
        return (1 << len(self.values)) - 1


class Catalog:
    def __init__(self, pool):
        self.pool = pool
        self.options = [[] for _ in pool.values]
        self.masks = [[] for _ in pool.values]
        scale = math.lcm(*(x.denominator for v in pool.values for x in v))
        rows = [tuple(int(x * scale) for x in v) for v in pool.values]
        bound = max((abs(x) for row in rows for x in row), default=1)
        # Each side of a lookup has coordinates bounded by 2*bound. A
        # difference is <=4*bound, below this radix: no carries/collisions.
        radix = 4 * bound + 1
        powers = [radix**i for i in range(pool.n)]
        packed = [sum(x * p for x, p in zip(row, powers)) for row in rows]
        if len(set(packed)) != len(rows) or any(x < 0 for x in packed):
            raise AssertionError('noninjective or incorrectly oriented packing')
        lookup = {x: i for i, x in enumerate(packed)}
        self.packed_bits = max(map(int.bit_length, packed), default=0)

        def add(x, u, v, a, b):
            if x is not None and x >= pool.n and x not in (u, v):
                self.options[x].append((u, v, a, b))

        for u, a in enumerate(packed):
            add(lookup.get(2 * a), u, -1, 2, 0)
            # Test 2*x == u rather than truncating a packed half.
            if a % 2 == 0:
                add(lookup.get(a // 2), u, -1, F(1, 2), 0)
            for v in range(u, len(packed)):
                b = packed[v]
                add(lookup.get(a + b), u, v, 1, 1)
                diff = a - b
                sign = 1 if diff >= 0 else -1
                add(lookup.get(abs(diff)), u, v, sign, -sign)
        for x, options in enumerate(self.options):
            original = pool.witness.get(x)
            def priority(p):
                same = p == original or (p[1] >= 0 and
                       (p[1], p[0], p[3], p[2]) == original)
                return not same
            options.sort(key=priority)
            self.masks[x] = [(1 << u) | (1 << v if v >= 0 else 0)
                             for u, v, _, _ in options]
        self.count = sum(map(len, self.options))

    def closure(self, selected, trace=False):
        """Greedy monotone closure. This hot loop uses ONLY integer masks."""
        if selected & self.pool.required != self.pool.required:
            return None if trace else False
        closed = self.pool.inputs
        remaining = selected & ~closed
        order = [] if trace else None
        masks = self.masks
        while remaining:
            previous = remaining
            todo = remaining
            while todo:
                bit = todo & -todo
                todo ^= bit
                x = bit.bit_length() - 1
                for deps in masks[x]:
                    if deps & closed == deps:
                        if trace:
                            order.append((x, closed))
                        closed |= bit
                        remaining ^= bit
                        break
            if previous == remaining:
                return None if trace else False
        return order if trace else True

    def plan(self, selected):
        order = self.closure(selected, trace=True)
        if order is None:
            raise ValueError('infeasible selected value set')
        signs = [1] * len(self.pool.values)
        chosen = []
        for x, closed in order:
            desired = self.pool.preferred[x]
            fallback = None
            for p, deps in zip(self.options[x], self.masks[x]):
                if deps & closed != deps:
                    continue
                u, v, a, b = p
                if v < 0:
                    sign = desired  # signed scale costs exactly one
                else:
                    aa, bb = a * signs[u], b * signs[v]
                    # Subtraction admits both orientations by swapping its
                    # operands; addition admits just the physical sum sign.
                    sign = desired if aa != bb else aa
                candidate = (x, p, sign)
                if sign == desired:
                    fallback = candidate
                    break
                if fallback is None:
                    fallback = candidate
            chosen.append(fallback)
            signs[x] = fallback[2]
        fixes = {(i, sign) for i, sign in self.pool.targets
                 if signs[i] != sign and any(self.pool.values[i])}
        return selected.bit_count() - self.pool.n + len(fixes), chosen, signs, fixes

    def emit(self, selected, plan=None):
        cost, chosen, signs, fixes = plan or self.plan(selected)
        names = {i: f'i{i}' for i in range(self.pool.n)}
        prog = []
        for x, (u, v, a, b), sign in chosen:
            name = f'v{x}'
            aa = a * signs[u] * sign
            if v < 0:
                expression = f'{names[u]}*({aa})'
            else:
                bb = b * signs[v] * sign
                if aa == bb == 1:
                    expression = f'{names[u]}+{names[v]}'
                elif aa == 1 and bb == -1:
                    expression = f'{names[u]}-{names[v]}'
                elif aa == -1 and bb == 1:
                    expression = f'{names[v]}-{names[u]}'
                else:
                    raise AssertionError('unpaid physical negation')
            prog.append((name, expression))
            names[x] = name
        correction = {}
        for i, sign in sorted(fixes):
            name = f'neg{i}'
            prog.append((name, f'-{names[i]}'))
            correction[i, sign] = name
        for j, (i, sign) in enumerate(self.pool.targets):
            prog.append((f'o{j}', correction.get((i, sign), names[i])))
        if slp.count_ops(prog)['total'] != cost:
            raise AssertionError('direction/sign cost differs from charged recount')
        expected = {j: {i: c * sign for i, c in enumerate(self.pool.values[x]) if c}
                    for j, (x, sign) in enumerate(self.pool.targets)}
        if slp.eval_map(prog, self.pool.n) != expected:
            raise ValueError('emitted phase differs from exact target map')
        return prog


class Cardinality:
    """Fixed-memory HLL estimate; recent tabu-set size is separately exact."""
    def __init__(self):
        self.registers = bytearray(4096)

    def add(self, digest):
        h = int(digest[:16], 16)
        slot = h & 4095
        tail = h >> 12
        rank = 53 - tail.bit_length() if tail else 53
        self.registers[slot] = max(self.registers[slot], rank)

    def estimate(self):
        m = len(self.registers)
        estimate = .7213 / (1 + 1.079 / m) * m*m / sum(2.0**-r for r in self.registers)
        zero = self.registers.count(0)
        if estimate <= 2.5*m and zero:
            estimate = m * math.log(m/zero)
        return round(estimate)


def rss_mb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (MIB if sys.platform == 'darwin' else 1024)
