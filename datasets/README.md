# Datasets

All files use the public textbook convention

\[
H=\sum_{i<j}S_i^T J_{ij}S_j+\sum_i h_i^T S_i,\qquad S=\sigma/2.
\]

Sparse exchange terms are encoded as `[i, j, J]`, where `J` is a scalar,
a length-three diagonal, or a 3-by-3 matrix. Missing fields are zero.
HamiltonZero expands this representation when loading a system.

`train/foundation_5000.jsonl` contains the 5,000 systems used by the released
foundation checkpoint, with sizes from 2 to 64 spins. Each row includes the
fixed WL1/FWL2 dispatch and, where available, `e_ed` in textbook energy units.
There are 3,169 non-null exact-diagonalization references.

The N≥256 evaluation systems are:

| File | Physical spins | Family |
|---|---:|---|
| `eval/ppp_ohno_n256.json` | 256 | PPP–Ohno C128H130 |
| `eval/ppp_ohno_n512.json` | 512 | PPP–Ohno C256H258 |
| `eval/ppp_ohno_n1024.json` | 1,024 | PPP–Ohno C512H514 |
| `eval/rudy_n256.json` | 256 | RUDY-12 MaxCut |
| `eval/rudy_n512.json` | 512 | RUDY-12 MaxCut |
| `eval/rudy_n1024.json` | 1,024 | RUDY-12 MaxCut |
| `eval/j1j2_45x45_obc.json` | 2,025 | square-lattice J1–J2 |
| `eval/triangular_42x42_pbc.json` | 1,764 | triangular Heisenberg |
| `eval/j1j2_64x64_obc.json` | 4,096 | square-lattice J1–J2 |
| `eval/j1j2_90x90_obc.json` | 8,100 | square-lattice J1–J2 |

Each evaluation record stores its fixed WL1 routing dispatch.

All stored couplings and reference energies use the textbook convention above.
Padding is reconstructed in memory and is not stored as zero-valued sites.

These datasets are distributed under the repository Apache-2.0 license.
