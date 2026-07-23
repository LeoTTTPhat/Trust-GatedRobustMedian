"""Revision experiment driver (addresses reviewer concerns).

Writes result CSVs to ../results_rev/. Each block is RESUMABLE: if its output
CSV already exists it is skipped, so the driver can be run in chunks. New content
vs. the original run_experiments.py:

  * New run baselines: Bulyan, RLR, AlignIns (concern: missing closest baseline).
  * Bandwidth/quantization sweep (concern: title promises bandwidth axis).
  * Intermittent / slow-drip attacker to test whether provenance is load-bearing
    (concern: method named after a redundant component).
  * Larger-capacity model (2x128 MLP) and larger population (N=100/200) scaling
    (concern: small model / small population).
  * 10 seeds on the digits headline tables (concern: statistical power).
  * FLAME near-IID sanity check (concern: baseline faithfulness).
"""
from __future__ import annotations
import csv, os, sys, time
from multiprocessing import Pool
import numpy as np
from flsim import Config, run

OUT = os.path.join(os.path.dirname(__file__), '..', 'results_rev')
os.makedirs(OUT, exist_ok=True)

BASE = dict(n_clients=50, rounds=60, participation=0.4, local_epochs=3,
            batch_size=32, local_lr=0.5, hidden=64, warmup=10, attack_start=15)
DSO = {'digits': dict(local_lr=0.5, batch_size=32),
       'unsw': dict(local_lr=0.3, batch_size=64)}

SEEDS10 = list(range(10))
SEEDS5 = list(range(5))

LBL = {'fedavg':'FedAvg','median':'Coord-Median','trimmed_mean':'Trimmed-Mean',
       'multikrum':'Multi-Krum','bulyan':'Bulyan','rlr':'RLR','alignins':'AlignIns',
       'fltrust':'FLTrust','flame':'FLAME (clustering)','rfw':'RFW-ARA (mean)',
       'rfw_pgm':'RFW-TGM (ours)'}

def kw(dataset, **cfg):
    d = dict(BASE); d.update(DSO[dataset]); d['dataset'] = dataset; d.update(cfg)
    return d

def ci95(vals):
    a = np.asarray(vals, float); a = a[~np.isnan(a)]
    if len(a) == 0: return float('nan'), float('nan')
    m = a.mean(); sd = a.std(ddof=1) if len(a) > 1 else 0.0
    return m, 1.96 * sd / np.sqrt(len(a))

def _run_one(k):
    return k['_i'], run(Config(**{kk: vv for kk, vv in k.items() if kk != '_i'}))

def compute(tasks, serial=False):
    """tasks: list of kw dicts (with _i index). Return list of result dicts."""
    res = [None] * len(tasks)
    for i, t in enumerate(tasks): t['_i'] = i
    if serial:
        for t in tasks: i, r = _run_one(t); res[i] = r
    else:
        with Pool() as p:
            for i, r in p.imap_unordered(_run_one, tasks, chunksize=1):
                res[i] = r
    return res

def write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows: w.writerow(r)
    print('  wrote', os.path.basename(path), flush=True)

def done(path): return os.path.exists(os.path.join(OUT, path))

# ---------------------------------------------------------------------------
# Generic sweep: rows = defenses, cols = an axis of configs. field defaults BA.
# ---------------------------------------------------------------------------
def sweep(path, dataset, defs, axis_name, axis_vals, axis_key, seeds,
          fixed, field='BA', serial=False, extra_per_def=None):
    if done(path):
        print(f'skip {path}'); return
    print(f'== {path} ({dataset}, {len(defs)}x{len(axis_vals)}x{len(seeds)}) ==', flush=True)
    tasks = []; idx = {}
    for dfn in defs:
        for av in axis_vals:
            for s in seeds:
                cfg = dict(fixed); cfg[axis_key] = av; cfg['defense'] = dfn; cfg['seed'] = s
                if extra_per_def and dfn in extra_per_def: cfg.update(extra_per_def[dfn])
                idx[(dfn, av, s)] = len(tasks); tasks.append(kw(dataset, **cfg))
    res = compute(tasks, serial=serial)
    header = ['defense'] + [f'{axis_name}_{v}' for v in axis_vals]
    rows = []
    for dfn in defs:
        row = [LBL.get(dfn, dfn)]
        for av in axis_vals:
            m, c = ci95([res[idx[(dfn, av, s)]][field] for s in seeds])
            row.append(f'{m:.1f}±{c:.1f}')
        rows.append(row)
    write_csv(os.path.join(OUT, path), header, rows)

# ---------------------------------------------------------------------------
BLOCKS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['all']
def want(b): return 'all' in BLOCKS or b in BLOCKS

CORE = ['median','trimmed_mean','multikrum','bulyan','rlr','alignins','fltrust',
        'flame','rfw','rfw_pgm']

t0 = time.time()

# 1. digits eta-sweep with ALL baselines, 10 seeds, A3 alpha=0.1
if want('eta'):
    sweep('rev_digits_eta.csv','digits', CORE, 'eta',[0.05,0.1,0.2,0.3,0.4],'mal_frac',
          SEEDS10, dict(attack='A3', alpha=0.1))

# 2. digits alpha-sweep (NEW HEADLINE lives here) 10 seeds, A3 eta=0.3
if want('alpha'):
    sweep('rev_digits_alpha.csv','digits',['fedavg']+CORE,'alpha',[0.05,0.1,0.5,1.0],'alpha',
          SEEDS10, dict(attack='A3', mal_frac=0.3))

# 3. digits attack variants, 10 seeds, alpha=0.1 eta=0.3
if want('variants'):
    sweep('rev_digits_variants.csv','digits',['fedavg']+CORE,'atk',
          ['A1','A2','A3','adaptive'],'attack', SEEDS10, dict(alpha=0.1, mal_frac=0.3))

# 4. digits MAIN at extreme-heterogeneity headline alpha=0.05 eta=0.3 (+FPR), 10 seeds
if want('main05'):
    path='rev_digits_main_a05.csv'
    if not done(path):
        print('== '+path+' ==', flush=True)
        defs=['fedavg']+CORE; tasks=[]; idx={}
        for dfn in defs:
            for s in SEEDS10:
                idx[(dfn,s)]=len(tasks); tasks.append(kw('digits',defense=dfn,seed=s,
                    attack='A3', alpha=0.05, mal_frac=0.3))
        res=compute(tasks)
        fedm,_=ci95([res[idx[('fedavg',s)]]['MTA'] for s in SEEDS10])
        rows=[]
        for dfn in defs:
            bm,bc=ci95([res[idx[(dfn,s)]]['BA'] for s in SEEDS10])
            mm,mc=ci95([res[idx[(dfn,s)]]['MTA'] for s in SEEDS10])
            fm,_=ci95([res[idx[(dfn,s)]]['FPR'] for s in SEEDS10])
            cost= (fedm-mm) if dfn!='fedavg' else 0.0
            rows.append([LBL.get(dfn,dfn),f'{bm:.1f}',f'{bc:.1f}',f'{mm:.1f}',f'{mc:.1f}',
                         f'{cost:.1f}','' if np.isnan(fm) else f'{fm:.1f}'])
        write_csv(os.path.join(OUT,path),['defense','BA','BA_ci','MTA','MTA_ci','util','FPR'],rows)
    else: print('skip '+path)

# 5. digits ablation (renamed) 10 seeds at alpha=0.1 AND alpha=0.05
if want('ablation'):
    for a,tag in [(0.1,'a01'),(0.05,'a05')]:
        path=f'rev_digits_ablation_{tag}.csv'
        if done(path): print('skip '+path); continue
        print('== '+path+' ==', flush=True)
        abl=[('FedAvg (no defense)',dict(defense='fedavg')),
             ('Coord-Median (unweighted)',dict(defense='median')),
             ('TGM - all trust-gating (uniform-weight median)',dict(defense='pgm_no_prov')),
             ('TGM - robust median (= trust-weighted mean, ARA)',dict(defense='pgm_no_median')),
             ('TGM w/ geometric (not coordinate) median',dict(defense='pgm_geo')),
             ('TGM HAS-only trust (lambda=1)',dict(defense='rfw_pgm',balance_lambda=1.0)),
             ('TGM provenance-only trust (lambda=0)',dict(defense='rfw_pgm',balance_lambda=0.0)),
             ('RFW-TGM full: HAS+prov (lambda=0.5) (ours)',dict(defense='rfw_pgm'))]
        tasks=[]; idx={}
        for lbl,extra in abl:
            for s in SEEDS10:
                idx[(lbl,s)]=len(tasks); tasks.append(kw('digits',seed=s,attack='A3',
                    alpha=a, mal_frac=0.3, **extra))
        res=compute(tasks); rows=[]
        for lbl,_ in abl:
            bm,bc=ci95([res[idx[(lbl,s)]]['BA'] for s in SEEDS10])
            mm,mc=ci95([res[idx[(lbl,s)]]['MTA'] for s in SEEDS10])
            rows.append([lbl,f'{bm:.1f}',f'{bc:.1f}',f'{mm:.1f}',f'{mc:.1f}'])
        write_csv(os.path.join(OUT,path),['configuration','BA','BA_ci','MTA','MTA_ci'],rows)

# 6. QUANTIZATION sweep (bandwidth axis) digits alpha=0.1, eta=0.2 and 0.3, 5 seeds
if want('quant'):
    for e,tag in [(0.2,'e02'),(0.3,'e03')]:
        sweep(f'rev_quant_{tag}.csv','digits',
              ['median','multikrum','alignins','fltrust','rfw_pgm'],
              'bits',[16,8,4,2],'quant_bits', SEEDS5,
              dict(attack='A3', alpha=0.1, mal_frac=e))

# 7. INTERMITTENT attacker: provenance test, digits alpha=0.1, 5 seeds
if want('intermittent'):
    path='rev_intermittent.csv'
    if not done(path):
        print('== '+path+' ==', flush=True)
        etas=[0.1,0.2,0.3]; configs=[('TGM HAS-only (lam=1)',dict(defense='rfw_pgm',balance_lambda=1.0)),
            ('TGM fusion (lam=0.5)',dict(defense='rfw_pgm',balance_lambda=0.5)),
            ('TGM prov-only (lam=0)',dict(defense='rfw_pgm',balance_lambda=0.0)),
            ('Coord-Median',dict(defense='median')),
            ('AlignIns',dict(defense='alignins'))]
        tasks=[]; idx={}
        for lbl,extra in configs:
            for e in etas:
                for s in SEEDS5:
                    idx[(lbl,e,s)]=len(tasks); tasks.append(kw('digits',seed=s,
                        attack='intermittent', alpha=0.1, mal_frac=e, intermittent_q=0.5, **extra))
        res=compute(tasks); rows=[]
        for lbl,_ in configs:
            row=[lbl]
            for e in etas:
                m,c=ci95([res[idx[(lbl,e,s)]]['BA'] for s in SEEDS5]); row.append(f'{m:.1f}±{c:.1f}')
            rows.append(row)
        write_csv(os.path.join(OUT,path),['config']+[f'eta_{e}' for e in etas],rows)
    else: print('skip '+path)

# 8. MODEL CAPACITY: 2x128 MLP, digits, eta-sweep + ablation, 5 seeds
if want('capacity'):
    sweep('rev_capacity_eta.csv','digits',
          ['median','multikrum','bulyan','alignins','fltrust','rfw_pgm'],
          'eta',[0.1,0.2,0.3,0.4],'mal_frac', SEEDS5,
          dict(attack='A3', alpha=0.1, hidden=(128,128)))
    path='rev_capacity_ablation.csv'
    if not done(path):
        print('== '+path+' ==', flush=True)
        abl=[('Coord-Median',dict(defense='median')),
             ('TGM - trust-gating (uniform median)',dict(defense='pgm_no_prov')),
             ('TGM - median (trust-weighted mean)',dict(defense='pgm_no_median')),
             ('RFW-TGM full (ours)',dict(defense='rfw_pgm'))]
        tasks=[]; idx={}
        for lbl,extra in abl:
            for s in SEEDS5:
                idx[(lbl,s)]=len(tasks); tasks.append(kw('digits',seed=s,attack='A3',
                    alpha=0.1, mal_frac=0.3, hidden=(128,128), **extra))
        res=compute(tasks); rows=[]
        for lbl,_ in abl:
            bm,bc=ci95([res[idx[(lbl,s)]]['BA'] for s in SEEDS5])
            mm,mc=ci95([res[idx[(lbl,s)]]['MTA'] for s in SEEDS5])
            rows.append([lbl,f'{bm:.1f}',f'{bc:.1f}',f'{mm:.1f}',f'{mc:.1f}'])
        write_csv(os.path.join(OUT,path),['configuration','BA','BA_ci','MTA','MTA_ci'],rows)
    else: print('skip rev_capacity_ablation.csv')

# 9. POPULATION scaling on UNSW (plenty of samples): N=100,200 eta-sweep, 5 seeds
if want('population'):
    for N in [100,200]:
        sweep(f'rev_pop_unsw_N{N}.csv','unsw',
              ['median','multikrum','bulyan','alignins','fltrust','rfw_pgm'],
              'eta',[0.05,0.1,0.2],'mal_frac', SEEDS5,
              dict(attack='A3', alpha=0.1, n_clients=N))
    # digits population N=100 (higher alpha to keep clients non-degenerate)
    sweep('rev_pop_digits_N100.csv','digits',
          ['median','alignins','fltrust','rfw_pgm'],
          'eta',[0.1,0.2,0.3],'mal_frac', SEEDS5,
          dict(attack='A3', alpha=0.5, n_clients=100))

# 10. UNSW eta-sweep + ablation with new baselines, 5 seeds
if want('unsw'):
    sweep('rev_unsw_eta.csv','unsw', CORE, 'eta',[0.05,0.1,0.2,0.3,0.4],'mal_frac',
          SEEDS5, dict(attack='A3', alpha=0.1))
    path='rev_unsw_ablation.csv'
    if not done(path):
        print('== '+path+' ==', flush=True)
        abl=[('Coord-Median (unweighted)',dict(defense='median')),
             ('TGM - trust-gating (uniform median)',dict(defense='pgm_no_prov')),
             ('TGM - median (trust-weighted mean)',dict(defense='pgm_no_median')),
             ('TGM w/ geometric median',dict(defense='pgm_geo')),
             ('TGM HAS-only (lambda=1)',dict(defense='rfw_pgm',balance_lambda=1.0)),
             ('RFW-TGM full (ours)',dict(defense='rfw_pgm'))]
        tasks=[]; idx={}
        for lbl,extra in abl:
            for s in SEEDS5:
                idx[(lbl,s)]=len(tasks); tasks.append(kw('unsw',seed=s,attack='A3',
                    alpha=0.1, mal_frac=0.1, **extra))
        res=compute(tasks); rows=[]
        for lbl,_ in abl:
            bm,bc=ci95([res[idx[(lbl,s)]]['BA'] for s in SEEDS5])
            mm,mc=ci95([res[idx[(lbl,s)]]['MTA'] for s in SEEDS5])
            rows.append([lbl,f'{bm:.1f}',f'{bc:.1f}',f'{mm:.1f}',f'{mc:.1f}'])
        write_csv(os.path.join(OUT,path),['configuration','BA','BA_ci','MTA','MTA_ci'],rows)
    else: print('skip rev_unsw_ablation.csv')

# 11. FLAME faithfulness check: boosted A1 attack (what FLAME's clustering+clip
#     is designed for) at low eta=0.1 across heterogeneity. FLAME should filter
#     well near-IID (alpha=100) and degrade as data becomes non-IID.
if want('flame_iid'):
    sweep('rev_flame_iid.csv','digits',['fedavg','median','flame','multikrum','rfw_pgm'],
          'alpha',[100.0,10.0,1.0,0.1],'alpha', SEEDS5, dict(attack='A1', mal_frac=0.1))

# 12. participation (add alignins) digits, 5 seeds
if want('participation'):
    sweep('rev_participation.csv','digits',
          ['fedavg','median','alignins','fltrust','rfw','rfw_pgm'],
          'rho',[0.2,0.3,0.4,0.6],'participation', SEEDS5,
          dict(attack='A3', alpha=0.1, mal_frac=0.3))

# 13. sensitivity (gamma_c, lambda, beta) digits 5 seeds -> reuse for updated table
if want('sensitivity'):
    path='rev_sensitivity.csv'
    if not done(path):
        print('== '+path+' ==', flush=True)
        rows_spec=[('gamma_c',[1.0,2.0,4.0,8.0]),('balance_lambda',[0.0,0.25,0.5,0.75,1.0]),
                   ('decay_beta',[0.7,0.8,0.9,0.95])]
        tasks=[]; idx={}
        for hp,vals in rows_spec:
            for v in vals:
                for s in SEEDS5:
                    idx[(hp,v,s)]=len(tasks); tasks.append(kw('digits',seed=s,attack='A3',
                        alpha=0.1, mal_frac=0.3, defense='rfw_pgm', **{hp:v}))
        res=compute(tasks); rows=[]
        for hp,vals in rows_spec:
            for v in vals:
                bm,bc=ci95([res[idx[(hp,v,s)]]['BA'] for s in SEEDS5])
                mm,_=ci95([res[idx[(hp,v,s)]]['MTA'] for s in SEEDS5])
                rows.append([hp,v,f'{bm:.1f}',f'{bc:.1f}',f'{mm:.1f}'])
        write_csv(os.path.join(OUT,path),['hyperparameter','value','BA','BA_ci','MTA'],rows)
    else: print('skip '+path)

# 14. adaptive vs non-adaptive (renamed) digits 10 seeds
if want('adaptive'):
    path='rev_adaptive.csv'
    if not done(path):
        print('== '+path+' ==', flush=True)
        etas=[0.1,0.2,0.3]; tasks=[]; idx={}
        for e in etas:
            for s in SEEDS10:
                idx[('na',e,s)]=len(tasks); tasks.append(kw('digits',seed=s,attack='A3',
                    alpha=0.1, mal_frac=e, defense='rfw_pgm'))
                idx[('ad',e,s)]=len(tasks); tasks.append(kw('digits',seed=s,attack='adaptive',
                    alpha=0.1, mal_frac=e, defense='rfw_pgm'))
        res=compute(tasks); rows=[]
        for e in etas:
            nm,nc=ci95([res[idx[('na',e,s)]]['BA'] for s in SEEDS10])
            am,ac=ci95([res[idx[('ad',e,s)]]['BA'] for s in SEEDS10])
            amm,amc=ci95([res[idx[('ad',e,s)]]['MTA'] for s in SEEDS10])
            rows.append([e,f'{nm:.1f}',f'{nc:.1f}',f'{am:.1f}',f'{ac:.1f}',f'{amm:.1f}',f'{amc:.1f}'])
        write_csv(os.path.join(OUT,path),['eta','na_BA','na_ci','ad_BA','ad_ci','ad_MTA','ad_mta_ci'],rows)
    else: print('skip '+path)

print(f'BLOCKS {BLOCKS} DONE in {time.time()-t0:.1f}s', flush=True)
