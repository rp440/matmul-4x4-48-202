"""Structured incumbent LNS (large-neighbourhood search) over exact circuits.

Each step proposes a few new vector directions in six families, closes the
dictionary under one-gate productions, and reverse-deletes to the cheapest
charged circuit. Costs are always recounted by slp.count_ops. Needs z3 for the
signed-production planner (Catalog.plan).
"""
from collections import Counter, defaultdict, deque
from fractions import Fraction as F
import math
import random

import rebuild as base
import slp

Dictionary = base.Dictionary
key, direction, combine, bits = base.key, base.direction, base.combine, base.bits
fingerprint = base.fingerprint
FAMILIES = ('target_residual', 'coordinated', 'sparse_pivot', 'internal_residual', 'dyadic', 'broad')
DEFAULT_WEIGHTS = (30, 25, 20, 10, 10, 5)


class Catalog(base.Catalog):
    def __init__(self, pool):
        super().__init__(pool)
        # Exact proportional groups index productions only. Dictionary values
        # still identify ONLY v and -v; scalar magnitudes are never collapsed.
        groups = defaultdict(list)
        for i, vector in enumerate(pool.values):
            pivot = next((x for x in vector if x), None)
            if pivot is not None:
                normalized = tuple(x/pivot for x in vector)
                groups[key(normalized)].append((i,pivot))
        for group in groups.values():
            for x, xp in group:
                if x < pool.n: continue
                for u, up in group:
                    if x == u: continue
                    production = (u,-1,xp/up,0)
                    if production not in self.options[x]:
                        self.options[x].append(production)
                        self.masks[x].append(1<<u)
        self.count = sum(map(len,self.options))

    def prune(self, selected, plan):
        """Remove unused chosen productions BEFORE cost/sign correction scoring."""
        _, chosen, signs, _ = plan
        live = self.pool.required
        for x, p, sign in reversed(chosen):
            if live >> x & 1:
                u, v, _, _ = p
                live |= (1 << u) | (1 << v if v >= 0 else 0)
        chosen = [c for c in chosen if live >> c[0] & 1]
        fixes = {(i, s) for i, s in self.pool.targets
                 if signs[i] != s and any(self.pool.values[i])}
        return live, (len(chosen) + len(fixes), chosen, signs, fixes)

    def plan(self, selected):
        initial = self.prune(selected, super().plan(selected))
        if not hasattr(self, 'signed_stats'): self.signed_stats = Counter()
        if not initial[1][3]:
            return initial[1]  # one gate/live direction is already a lower bound
        # Global signed production search: ranks enforce acyclicity, and every
        # binary production must have at least one physical positive operand.
        # Try the lower bound first, then paid corrections up to greedy cost.
        try:
            import z3
        except ImportError as ex:
            raise ImportError('rebuild2 needs the z3-solver package (pip install z3-solver)') from ex
        solver = z3.Solver()
        solver.set(timeout=200)
        ids = list(bits(selected))
        signs = {i: z3.Bool(f's{i}') for i in ids}
        ranks = {i: z3.Int(f'r{i}') for i in ids}
        admissible = {}
        for i in ids:
            if i < self.pool.n:
                solver.add(signs[i], ranks[i] == 0)
                continue
            solver.add(ranks[i] >= 1, ranks[i] <= len(ids))
            choices = []
            for p, deps in zip(self.options[i], self.masks[i]):
                if deps & selected != deps: continue
                u, v, a, b = p
                valid = ranks[i] > ranks[u]
                if v >= 0:
                    positive_a = signs[i] == (signs[u] if a == 1 else z3.Not(signs[u]))
                    positive_b = signs[i] == (signs[v] if b == 1 else z3.Not(signs[v]))
                    valid = z3.And(valid, ranks[i] > ranks[v], z3.Or(positive_a, positive_b))
                choices.append((p, valid))
            admissible[i] = choices
            solver.add(z3.Or([valid for p, valid in choices]))
        fixes = {(i, s) for i, s in self.pool.targets if any(self.pool.values[i])}
        penalties = [((z3.Not(signs[i]) if s == 1 else signs[i]), 1) for i, s in fixes]
        best = initial
        # Bounded exact search over the full catalog, rather than a fixed greedy
        # production ordering. UNKNOWN preserves the cheapest verified plan.
        for bound in range(len(initial[1][3])):
            solver.push()
            solver.add(z3.PbLe(penalties, bound) if penalties else z3.BoolVal(True))
            status = solver.check()
            self.signed_stats[str(status)] += 1
            if status == z3.sat:
                model = solver.model()
                actual = [1]*len(self.pool.values)
                for i in ids: actual[i] = 1 if z3.is_true(model.eval(signs[i])) else -1
                chosen = []
                for i in sorted(admissible, key=lambda i: model.eval(ranks[i]).as_long()):
                    p = next(p for p, valid in admissible[i] if z3.is_true(model.eval(valid)))
                    chosen.append((i, p, actual[i]))
                candidate = self.prune(selected, (0, chosen, actual, set()))
                if candidate[1][0] < best[1][0]: best = candidate
                solver.pop()
                break
            solver.pop()
        return best[1]

    def scored(self, selected):
        return self.prune(selected, self.plan(selected))


def reverse_delete(catalog, rng, repetitions=4, novel=0):
    if not catalog.closure(catalog.pool.full):
        raise ValueError('dictionary contains unreachable proposals')
    best, moved = None, None
    def keep(candidate):
        nonlocal best, moved
        if best is None or candidate[1][0] <= best[1][0]: best = candidate
        if candidate[0] & novel and (moved is None or candidate[1][0] <= moved[1][0]): moved = candidate
    for _ in range(repetitions):
        selected = catalog.pool.full
        current = catalog.scored(selected)
        keep(current)
        changed = True
        while changed:
            changed = False
            removable = list(bits(selected & ~catalog.pool.required))
            rng.shuffle(removable)
            for x in removable:
                trial = selected & ~(1 << x)
                if catalog.closure(trial):
                    selected = trial
                    changed = True
                    candidate = catalog.scored(selected)
                    # A feasible deletion can increase charged cost. Preserve
                    # the cheapest signed live DAG anywhere along the chain.
                    keep(candidate)
    # LNS keeps the cheapest distinct proposed state separately from the old
    # best. This permits cost-neutral and best+1 moves without losing a best.
    return moved if novel and moved is not None else best


class Proposals:
    """Six exact families. A package is committed whole, helpers included."""
    def __init__(self, prog, n, weights=DEFAULT_WEIGHTS):
        self.n, self.weights = n, tuple(weights)
        if len(weights) != 6 or min(weights) < 0 or not sum(weights):
            raise ValueError('need six nonnegative weights with positive sum')
        pool = Dictionary.from_program(prog, n)
        self.seed_values = pool.values
        self.targets = [pool.values[i] for i, _ in pool.targets if i >= n]
        self.consumers = defaultdict(set)
        for x, p in pool.witness.items():
            for u in p[:2]:
                if u >= 0:
                    self.consumers[pool.values[u]].add(pool.values[x])
        self.non_targets = [v for i, v in enumerate(pool.values) if i >= n and i not in pool.wants]
        # Downstream target cones include outputs used inside the circuit.
        self.cones = {v: set() for v in pool.values}
        for t in self.targets:
            stack, seen = [t], set()
            while stack:
                v = stack.pop()
                if v in seen: continue
                seen.add(v)
                self.cones[v].add(t)
                i = pool.index[key(v)]
                if i in pool.witness:
                    stack.extend(pool.values[u] for u in pool.witness[i][:2] if u >= 0)
        self.cache_digest = None
        self.cache = {}
        self.stats = {f: Counter() for f in FAMILIES}
        self.named = self.documented(prog)

    def documented(self, prog):
        # A fixed three-direction package taken from the published 204 circuit's
        # structure. Only installed when the dimension and output count match.
        if self.n != 16: return []
        env = slp.eval_map(prog, self.n)
        if len(env) != 48: return []
        out = [tuple(env[j].get(i, F(0)) for i in range(self.n)) for j in range(48)]
        q5 = combine(self.seed_values[11], out[21], 1, -1)
        q6 = combine(out[21], out[39], 1, -1)
        q4 = combine(out[11], out[33], 1, -1)
        return [[q5, q6, q4]]

    def refresh(self, values):
        digest = fingerprint(values)
        if digest == self.cache_digest: return
        self.cache = {}  # release previous-state cache before building the next
        known = set(values)
        def residuals(consumers):
            hits = defaultdict(set)
            for t in consumers:
                for v in values:
                    for c in (-1, 1):
                        q = direction(combine(t, v, 1, c))[0]
                        if any(q) and q not in known:
                            hits[q].add(t)
            return hits
        target_hits = residuals(self.targets)
        internal = set().union(*self.consumers.values()) if self.consumers else set()
        internal_hits = residuals(internal)
        def ranked(hits, minimum):
            return [(q, (1+len(h))**2/(1+sum(bool(x) for x in q)))
                    for q, h in hits.items() if len(h) >= minimum]
        sparse = set()
        inputs = values[:self.n]
        for group in (inputs, self.targets):
            for ai, a in enumerate(group):
                for b in group[ai+1:]:
                    for sign in (-1, 1):
                        q = direction(combine(a, b, 1, sign))[0]
                        if q not in known and any(q) and sum(bool(x) for x in q) <= 4:
                            sparse.add(q)
        self.cache = dict(target=ranked(target_hits, 3), internal=internal_hits,
                          sparse=list(sparse), target_hits=target_hits)
        self.cache_digest = digest

    @staticmethod
    def weighted(entries, rng):
        if not entries: return None
        return rng.choices([q for q, w in entries], [w for q, w in entries])[0]

    def package(self, family, values, rng):
        if family == 'target_residual':
            q = self.weighted(self.cache['target'], rng)
            return [q] if q else []
        if family == 'sparse_pivot':
            return [rng.choice(self.cache['sparse'])] if self.cache['sparse'] else []
        if family == 'internal_residual':
            shared = [(v, cs) for v, cs in self.consumers.items() if len(cs) >= 2]
            if not shared: return []
            v, cs = rng.choices(shared, [len(cs) for v, cs in shared])[0]
            entries = [(q, (1+len(h & cs))**2/(1+sum(bool(x) for x in q)))
                       for q, h in self.cache['internal'].items() if h & cs]
            q = self.weighted(entries, rng)
            return [q] if q else []
        if family == 'coordinated':
            if self.named and rng.random() < .2:
                return rng.choice(self.named)
            overlapping = [(a, b) for ai, a in enumerate(self.non_targets)
                           for b in self.non_targets[ai+1:] if self.cones[a] & self.cones[b]]
            if not overlapping: return []
            selected = list(rng.choice(overlapping))
            if rng.random() < .5:
                third = [v for v in self.non_targets if v not in selected and
                         any(self.cones[v] & self.cones[u] for u in selected)]
                if third: selected.append(rng.choice(third))
            package = []
            for u in selected:
                cs = self.consumers[u]
                entries = [(q, (1+len(h & cs))**2/(1+sum(bool(x) for x in q)))
                           for q, h in self.cache['internal'].items() if h & cs]
                q = self.weighted(entries, rng)
                if q: package.append(q)
            return package
        a, b = rng.choice(values), rng.choice(values)
        sign = rng.choice((-1, 1))
        numerator = combine(a, b, 1, sign)
        if family == 'dyadic':
            halves = [v for v in values if tuple(2*x for x in v) in values]
            if halves and rng.random() < .5:
                half = rng.choice(halves)
                numerator = tuple(2*x for x in half)
                # Alternate two-term decomposition of an incumbent numerator.
                a = rng.choice(values)
                b = combine(numerator, a, 1, -1)
                return [b, numerator, half]
            return [numerator, tuple(x/2 for x in numerator)]
        form = rng.randrange(5)
        if form == 0: return [numerator]
        if form in (1, 2):
            c = F(1, 2) if form == 1 else F(2)
            return [numerator, tuple(c*x for x in numerator)]
        if form == 3: return [tuple(2*x for x in b), combine(a, b, 1, 2*sign)]
        return [numerator, combine(numerator, rng.choice(values), 1, rng.choice((-1, 1)))]

    def draw(self, values, rng, k):
        self.refresh(values)
        known = set(values)
        proposals, families = [], set()
        while len(proposals) < k:
            family = rng.choices(FAMILIES, self.weights)[0]
            stats = self.stats[family]
            stats['draws'] += 1
            package = self.package(family, values, rng)
            new = []
            for v in package:
                q = direction(v)[0]
                if any(q) and q not in known and q not in new: new.append(q)
            if not new:
                stats['empty_resamples'] += 1
                continue
            if len(new) > k-len(proposals):
                stats['capacity_resamples'] += 1
                continue
            # Closure checks the package as a unit, including all new helpers.
            pool = Dictionary(self.n, values+proposals+new)
            if not Catalog(pool).closure(pool.full):
                stats['unreachable_resamples'] += 1
                continue
            proposals.extend(new)
            known.update(new)
            families.add(family)
            stats['packages'] += 1
            stats['directions'] += len(new)
        return proposals, families


class Search:
    def __init__(self, prog, n, seed=0, repetitions=4, temperature=.2, tabu_size=256,
                 weights=DEFAULT_WEIGHTS):
        pool = Dictionary.from_program(prog, n)
        catalog = Catalog(pool)
        selected, plan = reverse_delete(catalog, random.Random(seed), repetitions)
        rebuilt = catalog.emit(selected, plan)
        # Exact original is itself a signed realization; never degrade a seed.
        if slp.count_ops(prog)['total'] < plan[0]:
            rebuilt = prog
            pool = Dictionary.from_program(prog, n)
            selected = pool.full
            plan = (slp.count_ops(prog)['total'], [], pool.preferred, set())
        self.current = [pool.values[i] for i in bits(selected)]
        self.targets = [tuple(s*x for x in pool.values[i]) for i, s in pool.targets]
        self.preferences = {pool.values[i]: plan[2][i] for i in bits(selected)}
        self.current_cost = self.best_cost = slp.count_ops(rebuilt)['total']
        self.current_prog = self.best_prog = rebuilt
        self.incumbent_cost = slp.count_ops(prog)['total']
        self.n, self.repetitions, self.temperature, self.tabu_size = n, repetitions, temperature, tabu_size
        self.rng = random.Random(seed)
        self.proposals = Proposals(prog, n, weights)
        first = fingerprint(self.current)
        self.tabu, self.recent = deque([first]), {first}
        self.plateau = base.Cardinality()
        self.plateau.add(first)
        self.stats, self.costs = Counter(), Counter()
        self.max_D, self.max_productions = len(pool.values), catalog.count
        self.signed_stats = Counter(getattr(catalog, "signed_stats", {}))
        self.last_move = None

    def step(self, proposals=None, family='coordinated'):
        self.last_move = None
        if proposals is None:
            proposals, families = self.proposals.draw(self.current, self.rng, self.rng.randint(1, 8))
        else:
            families = {family}
        pool = Dictionary(self.n, self.current+proposals, self.targets, self.preferences)
        catalog = Catalog(pool)
        self.max_D, self.max_productions = max(self.max_D, len(pool.values)), max(self.max_productions, catalog.count)
        self.stats['iterations'] += 1
        self.stats['proposals_added'] += len(proposals)
        if not catalog.closure(pool.full):
            raise ValueError('atomic package was not reachable')
        novel = sum(1 << i for i,v in enumerate(pool.values) if v not in self.current)
        selected, plan = reverse_delete(catalog, self.rng, self.repetitions, novel)
        self.signed_stats.update(getattr(catalog, "signed_stats", {}))
        cost = plan[0]
        self.costs[cost] += 1
        values = [pool.values[i] for i in bits(selected)]
        digest = fingerprint(values)
        removed = len(self.current)-len(set(self.current) & set(values))
        added = len(set(values)-set(self.current))
        kind = None
        if cost < self.current_cost: kind = 'descent'
        elif cost == self.current_cost and digest not in self.recent: kind = 'plateau'
        elif (cost == self.best_cost+1 and digest not in self.recent and self.temperature > 0
              and self.rng.random() < math.exp(-1/self.temperature)): kind = 'annealing'
        for f in families:
            st = self.proposals.stats[f]
            st['rebuilt_batches'] += 1
            st['charged_gain'] += self.current_cost-cost
            st['deletion_surplus'] += removed-added
            if kind: st[kind+'_moves'] += 1
            if cost < self.best_cost: st['improvements'] += 1
        if kind:
            prog = catalog.emit(selected, plan)
            self.current, self.current_cost, self.current_prog = values, cost, prog
            self.preferences = {pool.values[i]: plan[2][i] for i in bits(selected)}
            # Families follow the actual accepted DAG, including internal
            # consumers introduced by previous plateau/annealing moves.
            previous_stats = self.proposals.stats
            self.proposals = Proposals(prog, self.n, self.proposals.weights)
            self.proposals.stats = previous_stats
            self.stats[kind+'_moves'] += 1
            if digest in self.recent: self.tabu.remove(digest)
            self.tabu.append(digest)
            self.recent.add(digest)
            if len(self.tabu) > self.tabu_size: self.recent.discard(self.tabu.popleft())
            self.last_move = dict(kind=kind, digest=digest, cost=cost, surplus=removed-added)
        else:
            self.stats['rejections'] += 1
            if digest in self.recent: self.stats['tabu_hits'] += 1
        if cost < self.best_cost:
            self.best_cost, self.best_prog = cost, self.current_prog
            self.plateau = base.Cardinality()
            self.stats['improvements'] += 1
        if cost == self.best_cost: self.plateau.add(digest)
        return self.last_move

    def report(self, elapsed):
        return dict(**self.stats, best_cost=self.best_cost, current_cost=self.current_cost,
                    plateau_size=len(self.recent), plateau_distinct_estimate=self.plateau.estimate(),
                    plateau_registers=list(self.plateau.registers), family_yields={f: dict(s) for f, s in self.proposals.stats.items()},
                    signed_search=dict(self.signed_stats), cost_histogram=dict(self.costs), max_D=self.max_D, max_productions=self.max_productions,
                    iterations_per_second=self.stats['iterations']/max(elapsed, 1e-9))
