# Source code and results — Trust-Gated Robust Median (TGM)

Reproduction package for *"Trust-Gated Robust Median: Root-Free Defense Against
Backdoor Attacks in Non-IID Federated Learning."*

## Contents

```
sourceCode/
├── README.md               this file
├── requirements.txt        Python dependencies (NumPy/SciPy/scikit-learn/pandas)
├── rfw/                     simulation library + experiment drivers
│   ├── model.py            NumPy MLP client model (1- or 2-hidden-layer, no DL framework)
│   ├── data.py             dataset providers + backdoor triggers (digits, UNSW-NB15)
│   ├── flsim.py            Dirichlet partitioning; A1/A2/A3/adaptive/intermittent attacks;
│   │                       b-bit quantization; all defenses (Median, Trimmed-Mean, Multi-Krum,
│   │                       Bulyan, RLR, FLTrust, FLAME-style, AlignIns-style, RFW-ARA, RFW-TGM)
│   ├── run_revision.py      main driver — reproduces every table/figure in the paper
│   ├── run_fix.py           stable α=0.5 headline (Table 1) + AlignIns positive control
│   └── run_experiments.py   original 5-seed battery (kept for reference)
├── data/
│   └── unsw_nb15_training.csv   UNSW-NB15 training partition (Moustafa & Slay, 2015)
└── results/                key result CSVs — every number in the paper traces here
```

`digits` is loaded from scikit-learn at runtime; no download is needed.

## How to reproduce

```bash
pip install -r requirements.txt
cd rfw
python run_revision.py all      # writes all CSVs to ../results_rev/  (~1 h on 2 cores)
python run_fix.py               # writes the stable-α=0.5 headline + AlignIns control
```

Runs are parallelized across CPU cores; pass a comma-separated block list to
`run_revision.py` (e.g. `python run_revision.py eta,alpha,quant`) to reproduce a
subset. The reference outputs shipped in `results/` were produced this way
(10 seeds on the headline `digits` tables, 5 seeds on the auxiliary sweeps).

## Results map (CSV → paper)

| CSV in `results/`                         | Paper location |
|-------------------------------------------|----------------|
| `rev_digits_main_a05_STABLE.csv`          | Table 1 (moderate-heterogeneity headline) |
| `rev_digits_main_a05.csv`                 | Table 2 (extreme-heterogeneity corner) |
| `rev_digits_alpha.csv`                    | Table 3, Fig. 3 (heterogeneity sweep) |
| `rev_participation.csv`                   | Table 4, Fig. 4 (participation) |
| `rev_digits_eta.csv`                      | Table 5, Fig. 5 (compromise-rate sweep) |
| `rev_digits_variants.csv`                 | Table 6 (attack variants) |
| `rev_adaptive.csv`                        | Table 7 (adaptive attacker) |
| `rev_unsw_eta.csv`                        | Table 8 (UNSW-NB15) |
| `rev_quant_e02.csv`, `rev_quant_e03.csv`  | Table 10 (b-bit quantization) |
| `rev_capacity_eta.csv`                    | Table 11 (2×128 capacity model) |
| `rev_intermittent.csv`                    | Table 12 (intermittent attacker) |
| `rev_digits_ablation_a01/a05.csv`         | Ablation table (mechanism) |
| `rev_unsw_ablation.csv`, `rev_capacity_ablation.csv` | ablation replications |
| `rev_sensitivity.csv`                     | hyperparameter sensitivity |
| `rev_flame_iid.csv`, `rev_alignins_poscontrol.csv` | baseline faithfulness controls |
| `rev_pop_unsw_N100/N200.csv`, `rev_pop_digits_N100.csv` | population scaling |

## Environment

Python 3.9+ with NumPy, SciPy, scikit-learn, pandas. No PyTorch, no GPU.
All results are reported as mean ± 95% CI over the seeds noted above.
