# Benchmark Results

The practical benchmark contains **200 sealed prompts**. Automatic PPQ evaluation and blind A/B comparison with the base Gemma model are stored in this directory.

## Automatic PPQ evaluation

| Metric | Result |
|---|---:|
| Route accuracy | 97.00% |
| Action accuracy | 94.50% |
| Runtime decision accuracy | 87.00% |
| Response decision accuracy | 86.50% |
| Family accuracy | 82.58% (128/155) |
| End-to-end accuracy | 81.00% (162/200) |

## Blind A/B comparison (185 comparable prompts)

| Metric | PPQ | Base Gemma |
|---|---:|---:|
| Task fulfilled | 80.00% | 20.00% |
| Authentic | 99.37% (157/158) | 45.16% (56/124) |
| Appropriate | 97.47% (154/158) | 55.49% (91/164) |
| Fabricated | 1.08% | 31.35% |

Pairwise preference: **PPQ 167**, **Base 11**, **Tie 7**. PPQ decisive win rate: **93.82% (167/178)**.

`authentic` and `appropriate` exclude `NA`, so their denominators may differ across systems.

These artifacts are frozen evaluation records. Do not tune the system using this benchmark if the reported numbers are to remain a held-out evaluation.
