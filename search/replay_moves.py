"""Replay a sequence of orbit moves from a circuit triple and compare phase maps.

Applies each move with orbit_moves.apply_move (exact; every intermediate triple
is admitted by slp.verify_candidate), then checks that the resulting L, R, P
linear maps (exact Fractions) equal those of --expect, if given. The replayed
circuit is the raw transformed one (before re-optimization), so its operation
count is usually higher than the optimized --expect circuit; only the maps must
agree.
"""
import argparse
import json
from pathlib import Path

import orbit_moves
import slp


def load(prefix):
    return [slp.parse_slp(f'{prefix}_{p}.slp') for p in 'LRP']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start', default='data/4x4x4_48_204', help='starting triple prefix')
    p.add_argument('--moves', default='data/moves_204_to_203.json', help='JSON list of moves')
    p.add_argument('--expect', default='data/4x4x4_48_202', help='triple whose maps must be reproduced')
    args = p.parse_args()
    progs = load(args.start)
    print(f'start: {slp.verify_candidate(progs)["total"]} ops')
    for move in json.loads(Path(args.moves).read_text()):
        progs = orbit_moves.apply_move(progs, move)
        print(f'after {json.dumps(move)}: {slp.verify_candidate(progs)["total"]} ops (exact tensor verified)')
    if args.expect:
        same = slp.load_progs(*progs) == slp.load_progs(*load(args.expect))
        print(f'maps equal to {args.expect}: {same}')
        raise SystemExit(0 if same else 1)


if __name__ == '__main__':
    main()
