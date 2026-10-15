#!/usr/bin/env bash
# One-command reproduction of every table in the revised (R2) paper.
set -e
pip install -r requirements.txt
cd rfw
python run_r2.py all          # raw per-seed results -> ../results_r2/raw_<block>.csv
python make_tables_r2.py      # summaries (mean, 95% CI, failed seeds) -> ../results_r2/summary_*.csv
python make_latex_r2.py       # LaTeX table bodies used in the paper -> ../results_r2/latex/
python timing_r2.py           # server aggregation timing (Table 13) -> ../results_r2/timing.csv
echo "Done. Raw and summary CSVs are in ../results_r2/"
