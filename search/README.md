# Circuit search code

Code used to go from the published PLinOpt 204-operation circuits for the
rational <4x4x4:48> algorithm to the 202-operation circuits in the parent folder:

    204 -> (scale product 20 by (a, b) = (1, -1); shear Y = I + E_12) -> 203 -> re-optimize L -> 202

The first arrow is an exact change of basis (an isotropy of the matrix
multiplication tensor) followed by a local search over the addition circuits;
the last arrow is a local search over the addition circuit of one phase
(L, R or P) that keeps its linear map unchanged.
Indices are 0-based. Cost model: binary +/- costs 1, a scaling by a constant
other than +1 is 1, unary minus is 1, copies are free.

The code was written with AI coding assistants. Nothing is trusted from the
search itself: every circuit that is kept is checked exactly (all 4096 tensor
coefficients with `fractions.Fraction`, plus a typed-DAG check and an operation
recount) by `slp.verify_candidate`.

## Requirements

Python >= 3.10 (tested on 3.14), macOS or Linux (`resource` and `df` are used
by the search code). Verification and move replay use the standard library
only. The optimizer (`rebuild3.py`, `orbit_search8.py`) also needs `z3-solver`
(`pip install z3-solver`; tested with 4.16.0), used in `rebuild2.Catalog.plan`
for the signed-production planner.

## Files

| file | purpose |
|---|---|
| `slp.py` | parse PLinOpt `.slp` files, exact evaluation, operation count, tensor check, DAG export, `verify_candidate`. CLI: verify a triple. |
| `verify_dag.py` | typed-DAG verifier (A-, B- and product-linear spaces kept separate). |
| `replay_moves.py` | apply a list of orbit moves to a circuit and compare the resulting maps with another circuit. |
| `orbit_moves.py` | the orbit moves `tau`, `scale`, `shear` as exact circuit transformations; circuit hashing; saving. |
| `orbit.py`, `lowering.py`, `normalize.py` | helpers: input/output substitution with charged gates, SLP to gate-graph lowering, SSA clean-up and dead-code removal. |
| `rebuild.py`, `rebuild2.py` | the LNS optimizer: exact vector dictionary, one-gate production catalog, reverse deletion, six proposal families. |
| `rebuild3.py` | CLI: optimize one phase (L, R or P) of a triple for a wall-clock budget, with full-triple admission. |
| `orbit_search8.py` | multi-process driver: orbit move sequences, then independent L/R/P optimization, successive halving (stages A/B/C), exact rank-48 admission, restartable. |
| `test_smoke.py` | unit tests (a few seconds). |
| `data/` | circuits (see below) and `moves_204_to_203.json`. |

## Data

- `data/4x4x4_48_204_{L,R,P}.slp`: copied unchanged from PLinOpt
  (https://github.com/jgdumas/plinopt, `data/4x4x4_48_204_{L,R,P}.slp`), by
  Dumas, Pernet, Sedoglavic, Stapleton and Tichavsky. The algorithm and these
  circuits are theirs.
- `data/4x4x4_48_203_{L,R,P}.slp`: intermediate 203-operation circuits.
- `data/4x4x4_48_202_{L,R,P}.slp`: the 202-operation circuits (same as the parent folder).
- `data/moves_204_to_203.json`: the two moves.

A triple is addressed by a prefix such as `data/4x4x4_48_202`
(`PREFIX_L.slp`, `PREFIX_R.slp`, `PREFIX_P.slp`). Run everything from this folder.

## Commands

(a) Verify the 202 circuit exactly (about 1 s; standard library only):

    python3 slp.py data/4x4x4_48_202

Expected: `Total linear ops: 202` and `Tensor: {"equations_checked": 4096, "mismatches": 0, "rank": 48}`.
Also works for `data/4x4x4_48_203` and `data/4x4x4_48_204`.

(b) Replay the moves from the 204 circuit and compare the maps with the 202 circuit (about 1 s; standard library only):

    python3 replay_moves.py

Prints the operation count after each move (205, then 213 for the raw,
un-optimized transformed circuit) and `maps equal to data/4x4x4_48_202: True`.
The 203 and 202 circuits have the same L, R, P maps as the replayed circuit;
only their addition circuits differ.

(c) Re-optimize phase L of the 203 circuit for 60 s (needs z3; the result depends on timing):

    python3 rebuild3.py --phase L --prefix data/4x4x4_48_203 --procs 1 --seed 1803 --wall-seconds 60 --output out/L203

Prints `progress` rows and, on a strict improvement, an `IMPROVED` row
(`before` 203, `total` 202; L goes from 57 to 56 operations). With seed 1803 this
reached 202 within 60 s on the machine used here; other seeds or slower
machines may need longer. Output goes to `out/L203/`: `manifest.json`,
`progress.jsonl`, `summary.json`, and per worker `worker_<seed>/` with the admitted
triple (`L.slp`, `R.slp`, `P.slp`), `best.slp` and `verification.json`.

(d) Run the orbit search briefly (needs z3; about 40 s):

    python3 orbit_search8.py --prefixes data/4x4x4_48_203 --procs 2 --depth 1 --wall-seconds 40 \
        --a-seconds 3 --b-seconds 5 --c-seconds 8 --b-seeds 1 --c-seeds 1 --batch-size 4 --output out/orbit

This only exercises the machinery (a handful of points). Each printed line is
one scored (move sequence, seed) task with its L/R/P costs. Circuits strictly
below 202 would be saved under `out/candidates/` (none are expected). The
defaults (`--procs 8`, stage budgets of 60 s, 300 s and 1200 s per phase) are
those meant for long runs. Use `--resume out/orbit` to continue a run.

Tests:

    python3 test_smoke.py

## Known limitations

- The multi-stage driver does not deduplicate candidates that differ only by
  product signs or by transpose (`tau`), so equivalent circuits can be explored repeatedly.
- Optimization results depend on wall-clock time (time-limited LNS and 200 ms z3
  checks), so reruns with the same seed are not bit-for-bit reproducible.
- The raw count after replaying the moves (213) is not the count of the 203
  circuit, which came from LNS re-optimization of the transformed circuit.
- `orbit_search8.py` publishes only circuits strictly below 202 (`TARGET`).

## License

The code in this folder is CC0 / public domain. The circuit files in data/ are
copied from or derived from PLinOpt and are distributed under the CeCILL-B
licence; see ../NOTICE and ../LICENSES/CeCILL-B.txt.
