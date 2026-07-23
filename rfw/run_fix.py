import csv, os
from multiprocessing import Pool
import numpy as np
from flsim import Config, run
OUT='../results_rev'
BASE=dict(n_clients=50,rounds=60,participation=0.4,local_epochs=3,batch_size=32,local_lr=0.5,hidden=64,warmup=10,attack_start=15)
def kw(**c):
    d=dict(BASE); d['dataset']='digits'; d.update(c); return d
def ci95(v):
    a=np.asarray(v,float); a=a[~np.isnan(a)]
    if len(a)==0: return float('nan'),float('nan')
    m=a.mean(); sd=a.std(ddof=1) if len(a)>1 else 0.0
    return m,1.96*sd/np.sqrt(len(a))
def one(k): return k['_i'], run(Config(**{x:y for x,y in k.items() if x!='_i'}))
def comp(tasks):
    for i,t in enumerate(tasks): t['_i']=i
    res=[None]*len(tasks)
    with Pool() as p:
        for i,r in p.imap_unordered(one,tasks,chunksize=1): res[i]=r
    return res
SEEDS=list(range(10))
# 1. main at alpha=0.5 (stable headline)
defs=['fedavg','median','trimmed_mean','multikrum','bulyan','rlr','alignins','fltrust','flame','rfw','rfw_pgm']
LBL={'fedavg':'FedAvg','median':'Coord-Median','trimmed_mean':'Trimmed-Mean','multikrum':'Multi-Krum','bulyan':'Bulyan','rlr':'RLR','alignins':'AlignIns','fltrust':'FLTrust','flame':'FLAME (clustering)','rfw':'RFW-ARA (mean)','rfw_pgm':'RFW-TGM (ours)'}
tasks=[]; idx={}
for d in defs:
    for s in SEEDS:
        idx[(d,s)]=len(tasks); tasks.append(kw(defense=d,seed=s,attack='A3',alpha=0.5,mal_frac=0.3))
res=comp(tasks)
fedm,_=ci95([res[idx[('fedavg',s)]]['MTA'] for s in SEEDS])
rows=[]
for d in defs:
    bm,bc=ci95([res[idx[(d,s)]]['BA'] for s in SEEDS]); mm,mc=ci95([res[idx[(d,s)]]['MTA'] for s in SEEDS]); fm,_=ci95([res[idx[(d,s)]]['FPR'] for s in SEEDS])
    cost=(fedm-mm) if d!='fedavg' else 0.0
    rows.append([LBL[d],f'{bm:.1f}',f'{bc:.1f}',f'{mm:.1f}',f'{mc:.1f}',f'{cost:.1f}','' if np.isnan(fm) else f'{fm:.1f}'])
with open(f'{OUT}/rev_digits_main_a05_STABLE.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['defense','BA','BA_ci','MTA','MTA_ci','util','FPR']); [w.writerow(r) for r in rows]
print('wrote stable main (alpha=0.5)')
# 2. AlignIns positive control: A1 boosted near-IID
tasks=[]; idx={}; defs2=['fedavg','alignins','median','rfw_pgm']; alphas=[100.0,10.0,1.0,0.1]
for d in defs2:
    for a in alphas:
        for s in range(5):
            idx[(d,a,s)]=len(tasks); tasks.append(kw(defense=d,seed=s,attack='A1',alpha=a,mal_frac=0.1))
res=comp(tasks); rows=[]
for d in defs2:
    row=[LBL[d]]
    for a in alphas:
        m,c=ci95([res[idx[(d,a,s)]]['BA'] for s in range(5)]); row.append(f'{m:.1f}±{c:.1f}')
    rows.append(row)
with open(f'{OUT}/rev_alignins_poscontrol.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['defense']+[f'alpha_{a}' for a in alphas]); [w.writerow(r) for r in rows]
print('wrote alignins positive control')
