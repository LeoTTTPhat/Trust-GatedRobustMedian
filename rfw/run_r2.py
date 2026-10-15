"""Revision R2 experiment driver (JISAS-D-26-05066, second round).

Every block writes RAW per-seed results to ../results_r2/raw_<block>.csv (one
row per run), so every number quoted in the revised paper traces to a row.
Summary tables are produced from these files by make_tables_r2.py. Blocks are
resumable: a block whose raw CSV already exists is skipped.

Usage:  python run_r2.py block1,block2,...   (or 'all')

What each block answers (reviewer concern in brackets):
  participation, quant, sensitivity, intermittent, capacity, population, unsw,
  controls      -> every auxiliary table re-run on the paper-wide 10 seeds
                   [unify seed counts; 5-seed vs 10-seed inconsistency;
                    quantization reference point]
  quant also adds an unquantized (full-precision) reference row.
  ablation      -> clipping / even-cohort median controls behind the
                   'uniform-weight median' row, plus score-design alternatives
                   (direction-only, norm-only, cosine, other squashing maps)
                   [exact-equivalence claim; why these components]
  failure50     -> 50-seed failure frequency at the unstable corner
                   [bimodality across seeds]
  warmup        -> contaminated warm-up, no warm-up, warm-up-free (online)
                   scale [reliance on an uncontaminated warm-up]
  adaptive      -> defense-aware optimised adaptive attacker vs the heuristic
                   one [adaptive attacker is heuristic]
  fmnist_*      -> Fashion-MNIST, a conventional visual benchmark [scale]
  faithful_*    -> original-recipe FLAME (HDBSCAN) and AlignIns [-style baselines]
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from multiprocessing import Pool

from flsim import Config, run

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results_r2')
os.makedirs(OUT, exist_ok=True)

BASE = dict(n_clients=50, rounds=60, participation=0.4, local_epochs=3,
            batch_size=32, local_lr=0.5, hidden=64, warmup=10, attack_start=15)
# Per-dataset local-SGD settings. digits/UNSW as in the paper; Fashion-MNIST
# lr chosen on CLEAN FedAvg only (lr in {0.05,0.1,0.2} x batch in {32,64}).
DSO = {'digits': dict(local_lr=0.5, batch_size=32),
       'unsw': dict(local_lr=0.3, batch_size=64),
       'fmnist': dict(local_lr=0.2, batch_size=32)}

S10 = list(range(10))
S50 = list(range(50))

CORE = ['median', 'trimmed_mean', 'multikrum', 'bulyan', 'rlr', 'alignins',
        'fltrust', 'flame', 'rfw', 'rfw_pgm']
FIELDS = ['BA', 'MTA', 'FPR', 'hhat', 'w_ben', 'w_mal', 'trust_ratio',
          'mal_share', 'mal_frac_rd', 's_ben', 's_mal', 'seconds']


def cfg_of(dataset, **kw):
    d = dict(BASE)
    d.update(DSO[dataset])
    d['dataset'] = dataset
    d.update(kw)
    return d


def T(label, dataset, seeds, **kw):
    """Tasks for one table cell: same config, several seeds."""
    return [dict(label=label, cfg=cfg_of(dataset, seed=s, **kw)) for s in seeds]


def _one(task):
    t0 = time.time()
    cfg = dict(task['cfg'])
    if isinstance(cfg.get('hidden'), list):
        cfg['hidden'] = tuple(cfg['hidden'])
    r = run(Config(**cfg))
    r['seconds'] = time.time() - t0
    return task['_i'], r


def execute(block, tasks, procs=31):
    path = os.path.join(OUT, f'raw_{block}.csv')
    if os.path.exists(path):
        print(f'skip {block} (exists)', flush=True)
        return
    print(f'== {block}: {len(tasks)} runs ==', flush=True)
    t0 = time.time()
    for i, t in enumerate(tasks):
        t['_i'] = i
    res = [None] * len(tasks)
    # longest-first scheduling keeps all cores busy
    order = sorted(tasks, key=lambda t: -({'fmnist': 50, 'unsw': 3}.get(t['cfg']['dataset'], 1)
                                          * (8 if t['cfg'].get('attack') == 'adaptive_opt' else 1)))
    with Pool(procs) as p:
        for i, r in p.imap_unordered(_one, order, chunksize=1):
            res[i] = r
    keys = sorted({k for t in tasks for k in t['cfg']})
    with open(path + '.tmp', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['block', 'label'] + keys + FIELDS)
        for t, r in zip(tasks, res):
            row = [block, t['label']]
            for k in keys:
                v = t['cfg'].get(k, '')
                row.append(json.dumps(v) if isinstance(v, (list, tuple)) else v)
            row += [r.get(k, '') for k in FIELDS]
            w.writerow(row)
    os.replace(path + '.tmp', path)
    print(f'   wrote raw_{block}.csv in {time.time() - t0:.0f}s', flush=True)


# ---------------------------------------------------------------------------
# Block definitions
# ---------------------------------------------------------------------------
def b_participation():
    out = []
    for d in ['fedavg', 'median', 'alignins', 'fltrust', 'rfw', 'rfw_pgm']:
        for rho in [0.2, 0.3, 0.4, 0.6]:
            out += T(f'{d}|rho={rho}', 'digits', S10, defense=d, attack='A3',
                     alpha=0.1, mal_frac=0.3, participation=rho)
    return out


def b_quant():
    out = []
    for e in [0.2, 0.3]:
        for d in ['median', 'multikrum', 'alignins', 'fltrust', 'rfw_pgm']:
            for b in [0, 16, 8, 4, 2]:
                out += T(f'{d}|eta={e}|bits={b}', 'digits', S10, defense=d, attack='A3',
                         alpha=0.1, mal_frac=e, quant_bits=b)
    return out


def b_sensitivity():
    out = []
    grid = [('gamma_c', [1.0, 2.0, 4.0, 8.0]),
            ('balance_lambda', [0.0, 0.25, 0.5, 0.75, 1.0]),
            ('decay_beta', [0.7, 0.8, 0.9, 0.95]),
            ('hhat_scale', [0.25, 0.5, 1.0, 2.0, 4.0]),
            ('warmup', [5, 10])]
    for a in [0.1, 0.5]:
        for hp, vals in grid:
            for v in vals:
                out += T(f'{hp}={v}|alpha={a}', 'digits', S10, defense='rfw_pgm',
                         attack='A3', alpha=a, mal_frac=0.3, **{hp: v})
    return out


def b_intermittent():
    out = []
    cf = [('TGM HAS-only (lam=1)', dict(defense='rfw_pgm', balance_lambda=1.0)),
          ('TGM fusion (lam=0.5)', dict(defense='rfw_pgm', balance_lambda=0.5)),
          ('TGM prov-only (lam=0)', dict(defense='rfw_pgm', balance_lambda=0.0)),
          ('Coord-Median', dict(defense='median')),
          ('AlignIns', dict(defense='alignins'))]
    for lbl, ex in cf:
        for e in [0.1, 0.2, 0.3]:
            out += T(f'{lbl}|eta={e}', 'digits', S10, attack='intermittent', alpha=0.1,
                     mal_frac=e, intermittent_q=0.5, **ex)
    return out


def b_capacity():
    out = []
    for d in ['median', 'multikrum', 'bulyan', 'alignins', 'fltrust', 'rfw_pgm']:
        for e in [0.1, 0.2, 0.3, 0.4]:
            out += T(f'{d}|eta={e}', 'digits', S10, defense=d, attack='A3', alpha=0.1,
                     mal_frac=e, hidden=[128, 128])
    for d in ['median_clip', 'pgm_no_prov', 'pgm_no_median']:
        out += T(f'{d}|eta=0.3', 'digits', S10, defense=d, attack='A3', alpha=0.1,
                 mal_frac=0.3, hidden=[128, 128])
    return out


def b_population():
    out = []
    for N in [100, 200]:
        for d in ['median', 'multikrum', 'bulyan', 'alignins', 'fltrust', 'rfw_pgm']:
            for e in [0.05, 0.1, 0.2]:
                out += T(f'unsw|N={N}|{d}|eta={e}', 'unsw', S10, defense=d, attack='A3',
                         alpha=0.1, mal_frac=e, n_clients=N)
    for d in ['median', 'alignins', 'fltrust', 'rfw_pgm']:
        for e in [0.1, 0.2, 0.3]:
            out += T(f'digits|N=100|{d}|eta={e}', 'digits', S10, defense=d, attack='A3',
                     alpha=0.5, mal_frac=e, n_clients=100)
    return out


def b_unsw():
    out = []
    for d in CORE:
        for e in [0.05, 0.1, 0.2, 0.3, 0.4]:
            out += T(f'{d}|eta={e}', 'unsw', S10, defense=d, attack='A3', alpha=0.1, mal_frac=e)
    abl = [('median_clip', {}), ('pgm_no_prov', {}), ('pgm_no_median', {}), ('pgm_geo', {}),
           ('rfw_pgm', dict(balance_lambda=1.0)), ('rfw_pgm', dict(balance_lambda=0.0))]
    for d, ex in abl:
        tag = d + ('' if not ex else f"|lam={ex['balance_lambda']}")
        out += T(f'abl|{tag}|eta=0.1', 'unsw', S10, defense=d, attack='A3', alpha=0.1,
                 mal_frac=0.1, **ex)
    return out


def b_controls():
    """Positive controls: boosted A1 at eta=0.1 from near-IID to non-IID."""
    out = []
    for d in ['fedavg', 'median', 'flame', 'alignins', 'multikrum', 'rfw_pgm']:
        for a in [100.0, 10.0, 1.0, 0.1]:
            out += T(f'{d}|alpha={a}', 'digits', S10, defense=d, attack='A1', alpha=a, mal_frac=0.1)
    return out


ABL_ROWS = [('FedAvg', dict(defense='fedavg')),
            ('Coord-Median (midpoint, no clip)', dict(defense='median')),
            ('Coord-Median + clipping', dict(defense='median_clip')),
            ('Uniform weighted median, no clip', dict(defense='median_wm')),
            ('TGM - trust (uniform weighted median + clip)', dict(defense='pgm_no_prov')),
            ('TGM - median (trust-weighted mean, ARA)', dict(defense='pgm_no_median')),
            ('TGM geometric median', dict(defense='pgm_geo')),
            ('TGM HAS-only (lam=1)', dict(defense='rfw_pgm', balance_lambda=1.0)),
            ('TGM provenance-only (lam=0)', dict(defense='rfw_pgm', balance_lambda=0.0)),
            ('TGM full (lam=0.5)', dict(defense='rfw_pgm')),
            ('score: direction term only', dict(defense='rfw_pgm', has_variant='dir_only')),
            ('score: norm term only', dict(defense='rfw_pgm', has_variant='norm_only')),
            ('score: cosine distance + norm', dict(defense='rfw_pgm', has_variant='cosine')),
            ('squash: 1-exp(-r)', dict(defense='rfw_pgm', has_variant='squash_exp')),
            ('squash: none (s=r)', dict(defense='rfw_pgm', has_variant='no_squash'))]


def b_ablation():
    out = []
    for a in [0.05, 0.1, 0.5]:
        for lbl, ex in ABL_ROWS:
            out += T(f'{lbl}|alpha={a}', 'digits', S10, attack='A3', alpha=a, mal_frac=0.3, **ex)
    return out


def b_failure50():
    out = []
    defs = [('median', {}), ('median_clip', {}), ('bulyan', {}), ('alignins', {}),
            ('fltrust', {}), ('rfw_pgm', {}), ('rfw_pgm', dict(balance_lambda=1.0))]
    for a in [0.05, 0.1, 0.5]:
        for e in [0.2, 0.3]:
            for d, ex in defs:
                tag = d + ('|lam=1' if ex else '')
                out += T(f'{tag}|alpha={a}|eta={e}', 'digits', S50, defense=d, attack='A3',
                         alpha=a, mal_frac=e, **ex)
    return out


def b_warmup():
    out = []
    base_defs = ['median', 'bulyan', 'fltrust', 'alignins', 'rfw_pgm']
    for a in [0.1, 0.5]:
        for e in [0.2, 0.3]:
            for d in base_defs:
                out += T(f'clean|{d}|alpha={a}|eta={e}', 'digits', S10, defense=d,
                         attack='A3', alpha=a, mal_frac=e)
                out += T(f'contam|{d}|alpha={a}|eta={e}', 'digits', S10, defense=d,
                         attack='A3', alpha=a, mal_frac=e, attack_start=0,
                         warmup_agg='median')
                out += T(f'nowarm|{d}|alpha={a}|eta={e}', 'digits', S10, defense=d,
                         attack='A3', alpha=a, mal_frac=e, attack_start=0, warmup=0)
            for tag, ex in [('clean', {}),
                            ('contam', dict(attack_start=0, warmup_agg='median')),
                            ('nowarm', dict(attack_start=0, warmup=0))]:
                out += T(f'{tag}|rfw_pgm_online|alpha={a}|eta={e}', 'digits', S10,
                         defense='rfw_pgm', hhat_mode='online', attack='A3', alpha=a,
                         mal_frac=e, **ex)
    return out


def b_adaptive():
    out = []
    for a in [0.1, 0.5]:
        for e in [0.1, 0.2, 0.3]:
            for atk in ['A3', 'adaptive', 'adaptive_opt']:
                for d in ['median', 'bulyan', 'fltrust', 'alignins', 'rfw_pgm']:
                    out += T(f'{atk}|{d}|alpha={a}|eta={e}', 'digits', S10, defense=d,
                             attack=atk, alpha=a, mal_frac=e)
    return out


ALL_DEFS = ['fedavg'] + CORE


def b_fmnist_main():
    out = []
    for a in [0.1, 0.5, 1.0]:
        for d in ALL_DEFS:
            out += T(f'{d}|alpha={a}|eta=0.3', 'fmnist', S10, defense=d, attack='A3',
                     alpha=a, mal_frac=0.3)
    return out


def b_fmnist_eta():
    out = []
    for e in [0.1, 0.2, 0.4]:
        for d in ['median', 'multikrum', 'bulyan', 'alignins', 'fltrust', 'flame', 'rfw', 'rfw_pgm']:
            out += T(f'{d}|alpha=0.5|eta={e}', 'fmnist', S10, defense=d, attack='A3',
                     alpha=0.5, mal_frac=e)
    return out


def b_fmnist_ablation():
    out = []
    for lbl, ex in ABL_ROWS:
        if ex.get('defense') in ('fedavg', 'median', 'rfw_pgm') and 'has_variant' not in ex \
                and 'balance_lambda' not in ex:
            continue  # already in fmnist_main at alpha=0.5
        out += T(f'{lbl}|alpha=0.5', 'fmnist', S10, attack='A3', alpha=0.5, mal_frac=0.3, **ex)
    return out


def b_fmnist_attacks():
    out = []
    for atk in ['A1', 'A2', 'adaptive', 'adaptive_opt']:
        for d in ['median', 'bulyan', 'fltrust', 'alignins', 'rfw_pgm']:
            out += T(f'{atk}|{d}|alpha=0.5|eta=0.3', 'fmnist', S10, defense=d, attack=atk,
                     alpha=0.5, mal_frac=0.3)
    return out


def b_alpha_all():
    """Headline alpha-sweep (A3, eta=0.3) incl. the alpha=0.5 main table and the
    alpha=0.05 extreme-heterogeneity table, regenerated end-to-end with the R2
    code and pinned environment (raw per-seed rows, MTA and FPR included)."""
    out = []
    for a in [0.05, 0.1, 0.5, 1.0]:
        for d in ALL_DEFS:
            out += T(f'{d}|alpha={a}', 'digits', S10, defense=d, attack='A3', alpha=a, mal_frac=0.3)
    return out


def b_eta_all():
    out = []
    for e in [0.05, 0.1, 0.2, 0.3, 0.4]:
        for d in CORE:
            out += T(f'{d}|eta={e}', 'digits', S10, defense=d, attack='A3', alpha=0.1, mal_frac=e)
    return out


def b_variants_all():
    out = []
    for atk in ['A1', 'A2', 'A3', 'adaptive', 'adaptive_opt']:
        for d in ALL_DEFS:
            out += T(f'{atk}|{d}', 'digits', S10, defense=d, attack=atk, alpha=0.1, mal_frac=0.3)
    return out


def b_adaptive_pk():
    out = []
    for a in [0.1, 0.5]:
        for e in [0.2, 0.3]:
            for d in ['median', 'bulyan', 'fltrust', 'alignins', 'rfw_pgm']:
                out += T(f'adaptive_opt_pk|{d}|alpha={a}|eta={e}', 'digits', S10, defense=d,
                         attack='adaptive_opt_pk', alpha=a, mal_frac=e)
    return out


def b_gate_tradeoff():
    """Sharper trust gates vs the optimised adaptive attacker (and A3)."""
    out = []
    for a in [0.1, 0.5]:
        for e in [0.2, 0.3]:
            for g in [1.0, 2.0, 4.0, 8.0]:
                for atk in ['A3', 'adaptive_opt']:
                    out += T(f'{atk}|gamma_c={g}|alpha={a}|eta={e}', 'digits', S10, defense='rfw_pgm',
                             attack=atk, alpha=a, mal_frac=e, gamma_c=g)
    return out


def b_fltrust_root():
    """FLTrust root-set size: 30/60/100/200 samples (A3 eta=0.3 at alpha=0.5 and
    0.1; boosted A1 eta=0.1 near-IID positive control)."""
    out = []
    for rs in [30, 60, 100, 200]:
        for a in [0.1, 0.5]:
            out += T(f'A3|root={rs}|alpha={a}', 'digits', S10, defense='fltrust', root_size=rs,
                     attack='A3', alpha=a, mal_frac=0.3)
        out += T(f'A1ctrl|root={rs}|alpha=100.0', 'digits', S10, defense='fltrust', root_size=rs,
                 attack='A1', alpha=100.0, mal_frac=0.1)
    return out


FAITHFUL = ['flame_orig', 'alignins_orig', 'fltrust_r100']


def _fd(d):
    """Faithful-baseline defense name -> config (FLTrust with its default 100-sample root)."""
    return dict(defense='fltrust', root_size=100) if d == 'fltrust_r100' else dict(defense=d)



def b_faithful_digits():
    out = []
    for d in FAITHFUL:
        for a in [0.05, 0.1, 0.5, 1.0]:
            out += T(f'alpha|{d}|alpha={a}', 'digits', S10, **_fd(d), attack='A3', alpha=a, mal_frac=0.3)
        for e in [0.05, 0.1, 0.2, 0.4]:      # eta=0.3 is in the alpha rows
            out += T(f'eta|{d}|eta={e}', 'digits', S10, **_fd(d), attack='A3', alpha=0.1, mal_frac=e)
        for atk in ['A1', 'A2', 'adaptive', 'adaptive_opt']:
            out += T(f'atk|{d}|{atk}', 'digits', S10, **_fd(d), attack=atk, alpha=0.1, mal_frac=0.3)
        for a in [100.0, 10.0, 1.0, 0.1]:
            out += T(f'ctrl|{d}|alpha={a}', 'digits', S10, **_fd(d), attack='A1', alpha=a, mal_frac=0.1)
    return out


def b_faithful_unsw():
    out = []
    for d in FAITHFUL:
        for e in [0.05, 0.1, 0.2, 0.3, 0.4]:
            out += T(f'{d}|eta={e}', 'unsw', S10, **_fd(d), attack='A3', alpha=0.1, mal_frac=e)
    return out


def b_faithful_fmnist():
    out = []
    for d in FAITHFUL:
        for a in [0.1, 0.5, 1.0]:
            out += T(f'{d}|alpha={a}|eta=0.3', 'fmnist', S10, **_fd(d), attack='A3', alpha=a, mal_frac=0.3)
        for e in [0.1, 0.2, 0.4]:
            out += T(f'{d}|alpha=0.5|eta={e}', 'fmnist', S10, **_fd(d), attack='A3', alpha=0.5, mal_frac=e)
    return out


BLOCKS = dict(participation=b_participation, quant=b_quant, sensitivity=b_sensitivity,
              intermittent=b_intermittent, capacity=b_capacity, population=b_population,
              unsw=b_unsw, controls=b_controls, ablation=b_ablation, failure50=b_failure50,
              warmup=b_warmup, adaptive=b_adaptive, fmnist_main=b_fmnist_main,
              fmnist_eta=b_fmnist_eta, fmnist_ablation=b_fmnist_ablation,
              fmnist_attacks=b_fmnist_attacks, faithful_digits=b_faithful_digits,
              faithful_unsw=b_faithful_unsw, faithful_fmnist=b_faithful_fmnist,
              alpha_all=b_alpha_all, eta_all=b_eta_all, variants_all=b_variants_all,
              adaptive_pk=b_adaptive_pk, gate_tradeoff=b_gate_tradeoff,
              fltrust_root=b_fltrust_root)

if __name__ == '__main__':
    want = sys.argv[1].split(',') if len(sys.argv) > 1 else ['all']
    names = list(BLOCKS) if 'all' in want else want
    t0 = time.time()
    for n in names:
        execute(n, BLOCKS[n]())
    print(f'DONE {names} in {time.time() - t0:.0f}s', flush=True)
