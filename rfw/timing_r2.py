"""Per-round server aggregation time (R2): |S_t|=20 updates of a digits-size
model (d=4,810), 150 repetitions after 10 warm-up calls, single process.
Writes ../results_r2/timing.csv. Machine: see platform line in the output."""
import os, platform, time, csv, subprocess
import numpy as np
import flsim

rng = np.random.default_rng(0)
n, d = 20, 4810
U = rng.normal(0, 0.05, size=(n, d))
glob = rng.normal(0, 0.1, size=d)
sc = np.ones(n)

def ctx():
    return dict(hhat=4.0, psi=[0.5] * 50, decay_beta=0.9, balance_lambda=0.5, gamma_c=2.0,
                trim_frac=0.2, selected=list(range(n)), mal_frac=0.3, root_update=U.mean(0),
                rng=np.random.default_rng(1), mal=None, rlr_theta_frac=0.5, alignins_lam=2.5,
                has_variant='default', hhat_mode='warmup', hhat_cos=1.0, global_flat=glob)

defs = ['fedavg', 'trimmed_mean', 'median', 'multikrum', 'bulyan', 'alignins', 'alignins_orig',
        'flame', 'flame_orig', 'rfw', 'rfw_pgm']
rows = []
for dfn in defs:
    for _ in range(10):
        flsim._aggregate(dfn, U, sc, sc, ctx())
    ts = []
    for _ in range(150):
        c = ctx(); t0 = time.perf_counter(); flsim._aggregate(dfn, U, sc, sc, c); ts.append(time.perf_counter() - t0)
    rows.append((dfn, 1000 * float(np.mean(ts))))
base = rows[0][1]
cpu = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip()
print('platform:', platform.platform(), '| CPU:', cpu, '| numpy', np.__version__)
with open(os.path.join(os.path.dirname(__file__), '..', 'results_r2', 'timing.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['defense', 'ms', 'relative'])
    for dfn, ms in rows:
        w.writerow([dfn, f'{ms:.3f}', f'{ms / base:.0f}']); print(f'{dfn:14s} {ms:7.3f} ms  {ms / base:6.0f}x')
