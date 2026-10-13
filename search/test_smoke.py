"""Smoke tests (standard-library unittest, < 60 s).  Run:  python3 test_smoke.py

1. The published 202 circuit verifies exactly (4096 tensor identities, typed DAG).
2. Replaying the two orbit moves on the 204 circuit reproduces the exact L, R, P
   linear maps of the 203 and 202 circuits.
3. A 5 s optimizer run on phase L of the 203 circuit returns a circuit that is
   exactly admitted and no worse than its input (needs z3-solver).
"""
import json
from pathlib import Path
import sys
import time
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import orbit_moves  # noqa: E402
import slp  # noqa: E402

try:
    import z3  # noqa: E402,F401
except ImportError:
    z3 = None

DATA = HERE/'data'


def load(name):
    return [slp.parse_slp(DATA/f'{name}_{p}.slp') for p in 'LRP']


class Smoke(unittest.TestCase):
    def test_verify_202(self):
        report = slp.verify_candidate(load('4x4x4_48_202'))
        self.assertEqual(report['total'], 202)
        self.assertEqual([report['counts'][p]['total'] for p in 'LRP'], [56, 56, 90])
        self.assertEqual(report['tensor'], dict(equations_checked=4096, mismatches=0, rank=48))
        self.assertEqual(report['dag']['operation_counts']['linear_ops'], 202)

    def test_published_204_and_203(self):
        self.assertEqual(slp.verify_candidate(load('4x4x4_48_204'))['total'], 204)
        self.assertEqual(slp.verify_candidate(load('4x4x4_48_203'))['total'], 203)

    def test_replay_moves_reproduces_maps(self):
        progs = load('4x4x4_48_204')
        moves = json.loads((DATA/'moves_204_to_203.json').read_text())
        self.assertEqual([m['type'] for m in moves], ['scale', 'shear'])
        for move in moves:
            progs = orbit_moves.apply_move(progs, move)
            self.assertEqual(slp.verify_candidate(progs)['tensor']['mismatches'], 0)
        maps = slp.load_progs(*progs)
        self.assertEqual(maps, slp.load_progs(*load('4x4x4_48_203')))
        self.assertEqual(maps, slp.load_progs(*load('4x4x4_48_202')))
        # sanity: the maps are not those of the untransformed 204 circuit
        self.assertNotEqual(maps, slp.load_progs(*load('4x4x4_48_204')))

    @unittest.skipIf(z3 is None, 'z3-solver not installed')
    def test_optimizer_5s(self):
        import rebuild3
        progs = rebuild3.load_triple(DATA/'4x4x4_48_203')
        before = slp.count_ops(progs[0])['total']
        search = rebuild3.Search(progs[0], 16, seed=1803)
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            search.step()
        candidate, report = rebuild3.admit(progs, 'L', search.best_prog)
        self.assertLessEqual(slp.count_ops(candidate[0])['total'], before)
        self.assertLessEqual(report['total'], 203)
        self.assertEqual(report['tensor']['mismatches'], 0)
        self.assertEqual(slp.eval_map(candidate[0], 16), slp.eval_map(progs[0], 16))


if __name__ == '__main__':
    unittest.main(verbosity=2)
