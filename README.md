# Source code and results: Trust-Gated Robust Median (TGM), revision R2

Reproduction package for *"Trust-Gated Robust Median: Root-Free Defense Against
Backdoor Attacks in Non-IID Federated Learning"* (JISAS-D-26-05066, revised version).

Every number in the revised paper is regenerated from a per-seed raw result file
in `results_r2/`, and every table body in the paper is generated from those files
by `rfw/make_latex_r2.py` (no number is transcribed by hand).

## Contents

```
sourceCode/
├── requirements.txt         pinned environment (Python 3.9.6, NumPy 2.0.2, ...)
├── run_all.sh               one-command reproduction of all R2 results
├── data/
│   ├── unsw_nb15_training.csv     UNSW-NB15 training partition (Moustafa & Slay, 2015)
│   └── fashion_mnist/             official Fashion-MNIST release (MD5-checked in data.py)
├── rfw/
│   ├── model.py             NumPy MLP client model
│   ├── data.py              digits, UNSW-NB15 and Fashion-MNIST providers + triggers
│   ├── flsim.py             FL simulator: partitioning, attacks (A1/A2/A3, intermittent,
│   │                        heuristic adaptive, optimized adaptive incl. partial knowledge),
│   │                        all defenses (incl. original-recipe FLAME and AlignIns) and TGM
│   ├── run_r2.py            R2 experiment driver (raw per-seed CSVs, resumable blocks)
│   ├── make_tables_r2.py    summaries: mean, 95% CI, failed seeds, trust diagnostics
│   ├── make_latex_r2.py     LaTeX table bodies of the paper (+ R1->R2 change log)
│   ├── timing_r2.py         per-round server aggregation timing
│   ├── run_revision.py, run_fix.py, run_experiments.py   R1 drivers (kept for reference)
├── results/                 R1 result CSVs (kept for reference)
└── results_r2/              R2 raw per-seed results, summaries, LaTeX tables, timing
```

`digits` is loaded from scikit-learn at runtime. The R2 code reproduces the R1 code
bit-for-bit for every R1 configuration (new behaviour is opt-in through new
configuration fields), so the R1 drivers still run unchanged.

## How to reproduce

```bash
pip install -r requirements.txt
cd rfw
python run_r2.py all        # or a comma-separated list of blocks, e.g. alpha_all,failure50
python make_tables_r2.py
python make_latex_r2.py
python timing_r2.py
```

The full R2 battery takes about an hour on a 32-core CPU (Fashion-MNIST and the
optimized adaptive attacker dominate). Blocks are resumable: a block whose raw CSV
exists is skipped.

## Results map (raw CSV in `results_r2/` -> revised paper)

| Raw file | Paper location |
|---|---|
| `raw_alpha_all.csv`, `raw_faithful_digits.csv` (alpha rows) | Tables 2, 3, 4; Fig. 3 |
| `raw_failure50.csv` | Table 5 (failed runs over 50 seeds) |
| `raw_fmnist_main.csv`, `raw_faithful_fmnist.csv` | Table 6 (Fashion-MNIST) |
| `raw_fmnist_eta.csv`, `raw_fmnist_attacks.csv` | Table 7 |
| `raw_participation.csv` | Table 8; Fig. 4 |
| `raw_eta_all.csv`, `raw_faithful_digits.csv` (eta rows) | Table 9; Fig. 5 |
| `raw_variants_all.csv`, `raw_faithful_digits.csv` (atk rows) | Table 10 |
| `raw_adaptive.csv`, `raw_adaptive_pk.csv`, `raw_gate_tradeoff.csv` | Table 11; Section 6.7 |
| `raw_unsw.csv`, `raw_faithful_unsw.csv` | Table 12; UNSW ablation in Section 6.8 |
| `timing.csv` | Table 13 |
| `raw_quant.csv` | Table 14 |
| `raw_capacity.csv`, `raw_population.csv` | Table 15; Section 6.11 |
| `raw_warmup.csv` | Table 16 |
| `raw_intermittent.csv` | Table 17 |
| `raw_ablation.csv`, `raw_fmnist_ablation.csv` | Tables 18 and 19 |
| `raw_sensitivity.csv` | Table 20 |
| `raw_controls.csv`, `raw_faithful_digits.csv` (ctrl rows), `raw_fltrust_root.csv` | Section 5.5 (positive controls, FLTrust root size) |

`results_r2/latex/changes_r1_to_r2.csv` lists every cell whose value differs from the
submitted (R1) version of the paper.

## Notes

* Fashion-MNIST: official 60k/10k split; local learning rate 0.2 and batch 32 were
  chosen on clean, attack-free FedAvg accuracy before any defense was run.
* Results at Dirichlet alpha = 0.05 depend on the package versions; use the pinned
  `requirements.txt` to reproduce them exactly.
* All results are mean +- 95% CI (1.96 sd / sqrt(n)) over seeds 0-9; the
  failure-frequency analysis uses seeds 0-49.
