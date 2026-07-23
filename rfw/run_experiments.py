"""Run the full RFW small-scale validation and write result CSVs.

Every number reported in the paper's small-scale validation tables traces to a
CSV written here. Runs are averaged over 5 seeds {0,1,2,3,4}; we report
mean +/- 95% CI. Two real datasets are evaluated:

  * ``digits`` (sklearn 8x8 images, 10 classes) -- the primary testbed.
  * ``unsw``   (UNSW-NB15 network-intrusion detection, 10 classes) -- a second,
    applied network-security dataset (tabular MLP, feature-pattern backdoor).

Baselines now include the two trust/robust defenses the paper is motivated
against: **FLTrust** (root-dataset cosine trust) and a **FLAME-style** cosine-
clustering filter, in addition to FedAvg / Median / Trimmed-Mean / Multi-Krum /
RFW-ARA. We report the REAL measured numbers, including regimes where FLTrust or
Multi-Krum match or beat RFW-PGM (notably on UNSW-NB15).

Usage:
    python3 run_experiments.py            # full grid (parallel; a few minutes)
    python3 run_experiments.py --quick    # 1 seed, fewer configs (smoke)
    python3 run_experiments.py --serial   # disable multiprocessing
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from multiprocessing import Pool

import numpy as np

from flsim import Config, run

SEEDS = [0, 1, 2, 3, 4]
OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(OUT, exist_ok=True)

# Per-dataset base config. server_lr = 1.0 (global_flat += agg), as run.
BASE = dict(n_clients=50, rounds=60, participation=0.4, local_epochs=3,
            batch_size=32, local_lr=0.5, hidden=64, warmup=10, attack_start=15)
DS_OVERRIDES = {
    'digits': dict(local_lr=0.5, batch_size=32),
    'unsw':   dict(local_lr=0.3, batch_size=64),
}

HEAD_ETA = 0.30
HEAD_ALPHA = 0.1
# Per-dataset headline compromise rate: the point where methods actually
# separate. On digits the field breaks around eta=0.3; on the harder tabular
# UNSW-NB15 task every defense collapses by eta=0.2, so eta=0.1 is where the
# comparison is informative.
DS_HEAD_ETA = {'digits': 0.30, 'unsw': 0.10}
ALPHAS = [0.05, 0.1, 0.5, 1.0]
ETAS = [0.05, 0.1, 0.2, 0.3, 0.4]

DEFENSE_LABEL = {
    'fedavg': 'FedAvg', 'median': 'Coord-Median', 'trimmed_mean': 'Trimmed-Mean',
    'multikrum': 'Multi-Krum', 'fltrust': 'FLTrust', 'flame': 'FLAME (clustering)',
    'rfw': 'RFW-ARA (mean)', 'rfw_pgm': 'RFW-PGM (ours)',
}
ALL_DEF = ['fedavg', 'median', 'trimmed_mean', 'multikrum', 'fltrust', 'flame',
           'rfw', 'rfw_pgm']
ETA_DEF = ['median', 'trimmed_mean', 'multikrum', 'fltrust', 'flame', 'rfw',
           'rfw_pgm']


# ---------------------------------------------------------------------------
# task cache: run each unique config once (parallel), then assemble tables
# ---------------------------------------------------------------------------
_POOL = None


def _run_one(kw):
    r = run(Config(**kw))
    return kw['_key'], r


def _full_kw(dataset, **cfg):
    kw = dict(BASE)
    kw.update(DS_OVERRIDES[dataset])
    kw['dataset'] = dataset
    kw.update(cfg)
    return kw


class Cache:
    def __init__(self):
        self.tasks = {}   # key -> kw
        self.res = {}     # key -> result dict

    def request(self, dataset, seeds, **cfg):
        keys = []
        for s in seeds:
            kw = _full_kw(dataset, seed=s, **cfg)
            key = tuple(sorted((k, v) for k, v in kw.items()))
            kw2 = dict(kw); kw2['_key'] = key
            self.tasks[key] = kw2
            keys.append(key)
        return keys

    def compute(self, serial=False):
        todo = [kw for k, kw in self.tasks.items() if k not in self.res]
        if not todo:
            return
        if serial:
            for kw in todo:
                k, r = _run_one(kw)
                self.res[k] = r
        else:
            with Pool() as p:
                for k, r in p.imap_unordered(_run_one, todo, chunksize=1):
                    self.res[k] = r

    def get(self, keys, field):
        return [self.res[k][field] for k in keys]


def ci95(vals):
    a = np.asarray(vals, float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return float('nan'), float('nan')
    m = a.mean()
    sd = a.std(ddof=1) if len(a) > 1 else 0.0
    return m, 1.96 * sd / np.sqrt(len(a))


def write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            w.writerow(r)
    print('wrote', os.path.basename(path), flush=True)


# ---------------------------------------------------------------------------
# table builders (register all needed runs, compute once, then emit)
# ---------------------------------------------------------------------------
def build_all(cache, seeds, datasets, quick=False):
    reg = {}   # (name) -> lambda producing rows from cache, plus header/path

    alphas = [HEAD_ALPHA] if quick else ALPHAS
    etas = [0.1, 0.3] if quick else ETAS
    attacks = ['A3'] if quick else ['A1', 'A2', 'A3', 'adaptive']

    plan = []  # list of dicts describing each CSV

    for ds in datasets:
        tag = '' if ds == 'digits' else 'unsw_'
        head_eta = DS_HEAD_ETA[ds]
        # eta sweep (all defenses)
        keys = {(dfn, e): cache.request(ds, seeds, attack='A3', mal_frac=e,
                                        defense=dfn, alpha=HEAD_ALPHA)
                for dfn in ETA_DEF for e in etas}
        plan.append(dict(kind='eta', ds=ds, path=f'table_{tag}eta_sweep.csv'
                         if tag else 'table6_eta_sweep.csv',
                         defs=ETA_DEF, etas=etas, keys=keys))
        # main comparison at headline eta (all defenses) + FPR
        mkeys = {dfn: cache.request(ds, seeds, attack='A3', mal_frac=head_eta,
                                    defense=dfn, alpha=HEAD_ALPHA)
                 for dfn in ALL_DEF}
        plan.append(dict(kind='main', ds=ds, path=f'table_{tag}main.csv'
                         if tag else 'table1_main.csv',
                         defs=ALL_DEF, keys=mkeys, eta=head_eta))
        # alpha sweep (all defenses) at headline eta
        akeys = {(dfn, a): cache.request(ds, seeds, attack='A3', mal_frac=head_eta,
                                         defense=dfn, alpha=a)
                 for dfn in ALL_DEF for a in alphas}
        plan.append(dict(kind='alpha', ds=ds, path=f'table_{tag}alpha_sweep.csv'
                         if tag else 'table2_alpha_sweep.csv',
                         defs=ALL_DEF, alphas=alphas, keys=akeys))
        # attack variants
        vkeys = {(dfn, atk): cache.request(ds, seeds, attack=atk, mal_frac=head_eta,
                                           defense=dfn, alpha=HEAD_ALPHA)
                 for dfn in ALL_DEF for atk in attacks}
        plan.append(dict(kind='variants', ds=ds, path=f'table_{tag}attack_variants.csv'
                         if tag else 'table5_attack_variants.csv',
                         defs=ALL_DEF, attacks=attacks, keys=vkeys))
        # ablation
        abl = [
            ('FedAvg (no defense)', dict(defense='fedavg')),
            ('Coord-Median (unweighted)', dict(defense='median')),
            ('PGM minus all trust-gating (uniform-weight median)', dict(defense='pgm_no_prov')),
            ('PGM minus robust median (= RFW-ARA trust-weighted mean)', dict(defense='pgm_no_median')),
            ('PGM w/ geometric median instead of coord median', dict(defense='pgm_geo')),
            ('PGM HAS-only trust (lambda=1 no provenance)', dict(defense='rfw_pgm', balance_lambda=1.0)),
            ('PGM provenance-only trust (lambda=0 no HAS)', dict(defense='rfw_pgm', balance_lambda=0.0)),
            ('RFW-PGM full: HAS+provenance (lambda=0.5) (ours)', dict(defense='rfw_pgm')),
        ]
        abkeys = [(lbl, cache.request(ds, seeds, attack='A3', mal_frac=head_eta,
                                      alpha=HEAD_ALPHA, **extra)) for lbl, extra in abl]
        plan.append(dict(kind='ablation', ds=ds, path=f'table_{tag}ablation.csv'
                         if tag else 'table8_ablation.csv', rows=abkeys))

    # digits-only extras
    if 'digits' in datasets:
        ds = 'digits'
        # participation
        pkeys = {(dfn, rho): cache.request(ds, seeds, attack='A3', mal_frac=HEAD_ETA,
                                           defense=dfn, alpha=HEAD_ALPHA, participation=rho)
                 for dfn in ['fedavg', 'median', 'fltrust', 'rfw', 'rfw_pgm']
                 for rho in [0.2, 0.3, 0.4, 0.6]}
        plan.append(dict(kind='participation', ds=ds, path='table3_participation.csv',
                         defs=['fedavg', 'median', 'fltrust', 'rfw', 'rfw_pgm'],
                         rhos=[0.2, 0.3, 0.4, 0.6], keys=pkeys))
        # adaptive vs non-adaptive
        adkeys = []
        for e in [0.1, 0.2, 0.3]:
            na = cache.request(ds, seeds, attack='A3', mal_frac=e, defense='rfw_pgm', alpha=HEAD_ALPHA)
            ad = cache.request(ds, seeds, attack='adaptive', mal_frac=e, defense='rfw_pgm', alpha=HEAD_ALPHA)
            adkeys.append((e, na, ad))
        plan.append(dict(kind='adaptive', ds=ds, path='table6_adaptive.csv', rows=adkeys))
        # sensitivity
        sens = []
        for g in [1.0, 2.0, 4.0, 8.0]:
            sens.append(('gamma_c', g, cache.request(ds, seeds, attack='A3', mal_frac=HEAD_ETA,
                        defense='rfw_pgm', alpha=HEAD_ALPHA, gamma_c=g)))
        for l in [0.0, 0.25, 0.5, 0.75, 1.0]:
            sens.append(('lambda', l, cache.request(ds, seeds, attack='A3', mal_frac=HEAD_ETA,
                        defense='rfw_pgm', alpha=HEAD_ALPHA, balance_lambda=l)))
        for b in [0.7, 0.8, 0.9, 0.95]:
            sens.append(('beta', b, cache.request(ds, seeds, attack='A3', mal_frac=HEAD_ETA,
                        defense='rfw_pgm', alpha=HEAD_ALPHA, decay_beta=b)))
        plan.append(dict(kind='sensitivity', ds=ds, path='table9_sensitivity.csv', rows=sens))
        # FPR of RFW-PGM across alpha (satisfies the FPR metric definition)
        fkeys = {a: cache.request(ds, seeds, attack='A3', mal_frac=HEAD_ETA,
                                  defense='rfw_pgm', alpha=a) for a in alphas}
        plan.append(dict(kind='fpr', ds=ds, path='table_fpr.csv', alphas=alphas, keys=fkeys))

    return plan


def emit(cache, plan):
    for p in plan:
        kind = p['kind']; path = os.path.join(OUT, p['path'])
        if kind == 'eta':
            rows = []
            for dfn in p['defs']:
                row = [DEFENSE_LABEL[dfn]]
                for e in p['etas']:
                    m, _ = ci95(cache.get(p['keys'][(dfn, e)], 'BA'))
                    row.append(f'{m:.1f}')
                rows.append(row)
            write_csv(path, ['defense'] + [f'eta_{e}' for e in p['etas']], rows)
        elif kind == 'main':
            fedm, _ = ci95(cache.get(p['keys']['fedavg'], 'MTA'))
            rows = []
            for dfn in p['defs']:
                bm, bc = ci95(cache.get(p['keys'][dfn], 'BA'))
                mm, mc = ci95(cache.get(p['keys'][dfn], 'MTA'))
                fm, _ = ci95(cache.get(p['keys'][dfn], 'FPR'))
                cost = fedm - mm if dfn != 'fedavg' else 0.0
                rows.append([DEFENSE_LABEL[dfn], f'{bm:.1f}', f'{bc:.1f}',
                             f'{mm:.1f}', f'{mc:.1f}', f'{cost:.1f}',
                             '' if np.isnan(fm) else f'{fm:.1f}'])
            write_csv(path, ['defense', 'BA', 'BA_ci', 'MTA', 'MTA_ci',
                             'utility_cost', 'FPR'], rows)
        elif kind == 'alpha':
            header = ['defense']
            for a in p['alphas']:
                header += [f'BA_alpha_{a}', f'ci_alpha_{a}']
            rows = []
            for dfn in p['defs']:
                row = [DEFENSE_LABEL[dfn]]
                for a in p['alphas']:
                    bm, bc = ci95(cache.get(p['keys'][(dfn, a)], 'BA'))
                    row += [f'{bm:.1f}', f'{bc:.1f}']
                rows.append(row)
            write_csv(path, header, rows)
        elif kind == 'variants':
            header = ['defense']
            for atk in p['attacks']:
                header += [f'{atk}_BA', f'{atk}_ci']
            rows = []
            for dfn in p['defs']:
                row = [DEFENSE_LABEL[dfn]]
                for atk in p['attacks']:
                    bm, bc = ci95(cache.get(p['keys'][(dfn, atk)], 'BA'))
                    row += [f'{bm:.1f}', f'{bc:.1f}']
                rows.append(row)
            write_csv(path, header, rows)
        elif kind == 'ablation':
            rows = []
            for lbl, keys in p['rows']:
                bm, bc = ci95(cache.get(keys, 'BA'))
                mm, mc = ci95(cache.get(keys, 'MTA'))
                rows.append([lbl, f'{bm:.1f}', f'{bc:.1f}', f'{mm:.1f}', f'{mc:.1f}'])
            write_csv(path, ['configuration', 'BA', 'BA_ci', 'MTA', 'MTA_ci'], rows)
        elif kind == 'participation':
            rows = []
            for dfn in p['defs']:
                row = [DEFENSE_LABEL[dfn]]
                for rho in p['rhos']:
                    bm, _ = ci95(cache.get(p['keys'][(dfn, rho)], 'BA'))
                    mm, _ = ci95(cache.get(p['keys'][(dfn, rho)], 'MTA'))
                    row.append(f'{bm:.1f}/{mm:.1f}')
                rows.append(row)
            write_csv(path, ['defense'] + [f'rho_{r}' for r in p['rhos']], rows)
        elif kind == 'adaptive':
            rows = []
            for e, na, ad in p['rows']:
                nm, nc = ci95(cache.get(na, 'BA'))
                am, ac = ci95(cache.get(ad, 'BA'))
                amm, amc = ci95(cache.get(ad, 'MTA'))
                rows.append([e, f'{nm:.1f}', f'{nc:.1f}', f'{am:.1f}', f'{ac:.1f}',
                             f'{amm:.1f}', f'{amc:.1f}'])
            write_csv(path, ['eta', 'nonadaptive_BA', 'na_ci', 'adaptive_BA',
                             'ad_ci', 'adaptive_MTA', 'ad_mta_ci'], rows)
        elif kind == 'sensitivity':
            rows = []
            for hp, val, keys in p['rows']:
                bm, bc = ci95(cache.get(keys, 'BA'))
                mm, _ = ci95(cache.get(keys, 'MTA'))
                rows.append([hp, val, f'{bm:.1f}', f'{bc:.1f}', f'{mm:.1f}'])
            write_csv(path, ['hyperparameter', 'value', 'BA', 'BA_ci', 'MTA'], rows)
        elif kind == 'fpr':
            rows = []
            for a in p['alphas']:
                fm, fc = ci95(cache.get(p['keys'][a], 'FPR'))
                mm, _ = ci95(cache.get(p['keys'][a], 'MTA'))
                rows.append([a, f'{fm:.1f}', f'{fc:.1f}', f'{mm:.1f}'])
            write_csv(path, ['alpha', 'FPR', 'FPR_ci', 'MTA'], rows)


def main(quick=False, serial=False, datasets=('digits', 'unsw')):
    seeds = [0] if quick else SEEDS
    t0 = time.time()
    cache = Cache()
    plan = build_all(cache, seeds, list(datasets), quick=quick)
    n = len(cache.tasks)
    print(f'computing {n} unique runs on {"1 core" if serial else "all cores"} ...', flush=True)
    cache.compute(serial=serial)
    emit(cache, plan)
    print(f'ALL DONE in {time.time()-t0:.1f}s ({n} runs, {len(seeds)} seeds, '
          f'datasets={list(datasets)})', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--serial', action='store_true')
    ap.add_argument('--datasets', default='digits,unsw')
    args = ap.parse_args()
    main(quick=args.quick, serial=args.serial,
         datasets=tuple(args.datasets.split(',')))
