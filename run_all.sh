#!/usr/bin/env bash
# One-command reproduction of all paper results.
set -e
pip install -r requirements.txt
cd rfw
python run_revision.py all
python run_fix.py
echo "Done. Regenerated CSVs are in ../results_rev/ ; reference CSVs are in ../results/"
