"""Summarise the raw per-seed R2 results (../results_r2/raw_<block>.csv).

For every (block, label) cell: number of seeds, mean and 95% CI half-width
(1.96 * sd / sqrt(n), the convention used throughout the paper) of BA and
MTA, the number of seeds with BA >= 50% ('failures'), and, for TGM runs, the
trust diagnostics averaged over seeds. Output: ../results_r2/summary_<block>.csv
and a combined ../results_r2/summary_all.csv.

Usage: python make_tables_r2.py [block ...]
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R2 = os.path.join(HERE, '..', 'results_r2')


def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return x.mean(), 1.96 * sd / np.sqrt(len(x))


def summarise(block):
    df = pd.read_csv(os.path.join(R2, f'raw_{block}.csv'))
    rows = []
    for lbl, g in df.groupby('label', sort=False):
        ba, bac = ci95(g['BA'])
        mta, mtac = ci95(g['MTA'])
        row = dict(block=block, label=lbl, n=len(g), BA=round(ba, 1), BA_ci=round(bac, 1),
                   MTA=round(mta, 1), MTA_ci=round(mtac, 1),
                   fails=int((g['BA'] >= 50).sum()),
                   BA_median=round(float(np.median(g['BA'])), 1))
        for k in ['FPR', 'trust_ratio', 'mal_share', 'mal_frac_rd', 's_ben', 's_mal', 'hhat']:
            if k in g and g[k].notna().any():
                row[k] = round(float(np.nanmean(pd.to_numeric(g[k], errors='coerce'))), 3)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(R2, f'summary_{block}.csv'), index=False)
    return out


if __name__ == '__main__':
    blocks = sys.argv[1:] or [os.path.basename(p)[4:-4] for p in
                              sorted(glob.glob(os.path.join(R2, 'raw_*.csv')))]
    allr = [summarise(b) for b in blocks]
    if allr:
        pd.concat(allr).to_csv(os.path.join(R2, 'summary_all.csv'), index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 30)
    for s in allr:
        print(s.to_string(index=False))
        print()
