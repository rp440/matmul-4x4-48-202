# Shorter circuits for the rational rank-48 4x4 matrix multiplication algorithm

This folder contains straight-line programs (SLPs) for the rational ⟨4x4x4:48⟩
algorithm studied by Dumas, Pernet, Sedoglavic, Stapleton and Tichavský in
"Fast matrix multiplication via recursive ⟨4x4x4:48⟩ algorithms into practice"
(arXiv:2609.12027, https://arxiv.org/abs/2609.12027). The bilinear algorithm is
the same one; only the addition circuits around the 48 multiplications differ.
The published circuits come from PLinOpt (https://github.com/jgdumas/plinopt,
`data/4x4x4_48_204_{L,R,P}.slp`). All credit for the algorithm and the tooling
belongs to those authors.

## Results

| | published | this folder |
|---|---|---|
| linear operations, 4x4x4:48 (L + R + P) | 204 | 202 (56 + 56 + 90) |
| leading constant, 4x4 base case | 1 + 204/32 = 59/8 = 7.375 | 1 + 202/32 = 7.3125 |
| P core of the square alternative-basis variant (16ALT) | 85 ops | 84 ops |
| leading constant, 16ALT variant | 205/32 | 1 + (44 + 44 + 84)/32 = 51/8 |

Cost model: each binary addition or subtraction costs 1, each scaling by a
constant other than +1 or -1 costs 1 (the halvings here), copies are free.
The 202 circuit uses 54 additions/subtractions plus 2 halvings in each of L and
R, and 90 additions/subtractions in P. It needs 1/2 in the coefficient ring.

## Files

- `4x4x4_48_202_L.slp`: 48 linear forms in the entries of A (inputs `i0..i15`).
- `4x4x4_48_202_R.slp`: 48 linear forms in the entries of B (inputs `i0..i15`).
- `4x4x4_48_202_P.slp`: the 16 entries of C from the 48 products (inputs `i0..i47`, outputs `o0..o15`).
- `4x4x4_48_204-16ALT_P_84ops.slp`: 84-operation circuit for the same 16x48 linear map as PLinOpt `data/4x4x4_48_204-16ALT_P.slp` (85 ops).
- `matmul_4x4_48mult_202ops.py`: standalone Python implementation of the 202-operation algorithm.
- `check_alt_core.py`: checks the 84-operation file against the published one.

The SLPs use the PLinOpt format: one `name:=expression;` per line.

## How to verify

    python3 matmul_4x4_48mult_202ops.py
    python3 check_alt_core.py

The first script checks all 4096 coefficient identities (every pair of basis
matrices) with exact `Fraction` arithmetic, plus 1000 random integer products.
The second downloads the published `4x4x4_48_204-16ALT_P.slp` at a pinned
PLinOpt commit, parses both files into exact linear maps, asserts they are
identical, and counts operations under the cost model above. It also accepts a
local path to the reference file as an argument. Both need only the Python
standard library (3.8+).

## How it was found

Starting from the published algorithm, I moved along its isotropy orbit (product
20 rescaled by (a, b) = (1, -1) and a shear Y = I + E_12, 0-based indices),
then ran a local search that repeatedly rebuilds parts of the addition chains
of L, R and P, accepting rewrites that keep or lower the operation count. It ran on a
laptop, and the search code was written with AI coding assistants. Every result
was checked exactly against the matrix multiplication tensor.

## Conventions

- A, B, C are 4x4 and indexed row-major: `i0..i15` are `A[0][0], A[0][1], ..., A[3][3]` (resp. B).
- Product `k` is `m_k = L_k * R_k`, where `L_k` is output `ok` of the L program and `R_k` output `ok` of the R program.
- The P program takes `i0..i47` = `m_0..m_47` and returns the entries of C in row-major order as `o0..o15`.
- The circuits only use ring operations and multiplication by 1/2, so they apply to noncommutative entries (matrix blocks), with the product order `L_k * R_k` preserved.

## License

Circuits and code released under CC0 / public domain.
