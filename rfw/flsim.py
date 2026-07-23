"""Federated-learning backdoor simulation harness (real, runnable).

Implements, over two real datasets (sklearn ``digits`` image classification and
the UNSW-NB15 network-intrusion detection task; see ``data.py``) with a NumPy
MLP client model:
  * Dirichlet(alpha) non-IID label partitioning across N clients.
  * A backdoor trigger (pixel patch for digits / feature-pattern for UNSW) that
    relabels triggered inputs to a target class, with attack variants A1
    (model replacement / boosted), A2 (constrained / stealthy), A3
    (CollaPois-style shared-Trojan collaboration), and an ADAPTIVE attacker
    that optimizes against the full RFW pipeline (HAS residual + provenance).
  * Baseline defenses: FedAvg, coordinate-wise Median, Trimmed-Mean, Multi-Krum,
    and -- the two trust/robust baselines the paper is motivated against --
    **FLTrust** (Cao et al., root-dataset cosine-trust weighted mean with
    magnitude normalization) and a **FLAME-style** cosine-distance clustering
    filter (Nguyen et al.: keep the majority cosine cluster, median-norm clip,
    small calibrated noise).
  * RFW-ARA = HAS + UPT + softmax trust-weighted mean of clipped updates
    (the prior mechanism).
  * RFW-PGM (this paper's NEW mechanism): Provenance-Gated robust Median.
    A coordinate-wise WEIGHTED median whose per-client weights fuse current-round
    geometry (HAS) AND accumulated cross-round provenance (UPT). Ablations
    (pgm_no_prov, pgm_no_median, pgm_no_has, pgm_geo) isolate each ingredient.

Everything here is executed by run_experiments.py; numbers in the paper trace to
CSVs it writes.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from data import get_dataset
from model import MLP


# ---------------------------------------------------------------------------
# Data + trigger (dataset-agnostic; specifics live in data.py)
# ---------------------------------------------------------------------------
def stamp_trigger(X, trigger_idx, trigger_val):
    Xp = X.copy()
    Xp[:, trigger_idx] = trigger_val
    return Xp


def make_backdoor_set(X, y, trigger_idx, trigger_val, target, poison_frac=0.5,
                      rng=None):
    """Return a training set where ``poison_frac`` of samples carry the trigger
    and are relabeled to ``target`` (the rest are clean)."""
    n = X.shape[0]
    if rng is None:
        rng = np.random.default_rng(0)
    k = int(round(poison_frac * n))
    idx = rng.permutation(n)
    pidx = idx[:k]
    Xp = X.copy(); yp = y.copy()
    Xp[pidx] = stamp_trigger(X[pidx], trigger_idx, trigger_val)
    yp[pidx] = target
    return Xp, yp


def dirichlet_partition(y, n_clients, alpha, seed=0, min_size=2, max_tries=200):
    """Partition sample indices among clients by Dirichlet(alpha) over labels."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    best = None
    for _ in range(max_tries):
        client_idx = [[] for _ in range(n_clients)]
        for c in classes:
            idx_c = np.where(y == c)[0]
            rng.shuffle(idx_c)
            props = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
            for cid, part in enumerate(np.split(idx_c, cuts)):
                client_idx[cid].extend(part.tolist())
        sizes = [len(c) for c in client_idx]
        if min(sizes) >= min_size:
            return [np.array(c) for c in client_idx]
        if best is None or min(sizes) > min([len(c) for c in best]):
            best = [list(c) for c in client_idx]
    all_idx = np.arange(len(y))
    for cid in range(n_clients):
        if len(best[cid]) < min_size:
            need = min_size - len(best[cid])
            best[cid].extend(rng.choice(all_idx, size=need, replace=False).tolist())
    return [np.array(c) for c in best]


# ---------------------------------------------------------------------------
# Aggregation primitives
# ---------------------------------------------------------------------------
def weighted_mean(updates, w):
    w = np.asarray(w, float)
    w = w / (w.sum() + 1e-12)
    return (updates * w[:, None]).sum(axis=0)


def coordinate_median(updates, w=None):
    return np.median(updates, axis=0)


def trimmed_mean(updates, trim_frac=0.2):
    k = int(np.floor(trim_frac * updates.shape[0]))
    s = np.sort(updates, axis=0)
    if k > 0:
        s = s[k:updates.shape[0] - k]
    return s.mean(axis=0)


def multi_krum(updates, n_mal, m=None):
    n = updates.shape[0]
    if m is None:
        m = max(1, n - n_mal - 2)
    d2 = np.zeros((n, n))
    for i in range(n):
        diff = updates - updates[i]
        d2[i] = (diff * diff).sum(axis=1)
    k = max(1, n - n_mal - 2)
    scores = np.array([np.sort(d2[i])[1:k + 1].sum() for i in range(n)])
    sel = np.argsort(scores)[:m]
    return updates[sel].mean(axis=0)


def fltrust(updates, root_update):
    """FLTrust (Cao et al., NDSS 2021). Server holds a small clean root set and
    computes a root update g0 each round. Each client update is weighted by
    ReLU(cosine similarity to g0) and magnitude-normalized to ||g0|| before a
    trust-weighted average. Faithful implementation."""
    g0 = root_update
    g0n = float(np.linalg.norm(g0)) + 1e-12
    unorm = np.linalg.norm(updates, axis=1) + 1e-12
    cos = (updates @ g0) / (unorm * g0n)
    ts = np.maximum(0.0, cos)                      # ReLU(cosine) trust
    scaled = updates * (g0n / unorm)[:, None]      # normalize each to ||g0||
    denom = ts.sum() + 1e-12
    return (scaled * ts[:, None]).sum(axis=0) / denom


def flame(updates, noise_factor=1e-3, rng=None):
    """FLAME-style filter (Nguyen et al., USENIX Security 2022). Cluster updates
    by pairwise cosine distance, admit the majority cluster (discarding the
    minority/outlier cluster), clip admitted updates to the median L2 norm, and
    add small calibrated Gaussian noise. We use scipy agglomerative (average-
    linkage) clustering on the cosine-distance matrix in place of HDBSCAN; the
    admit-majority-cluster + median-norm-clip + noise recipe is faithful."""
    n = updates.shape[0]
    if n < 3:
        return coordinate_median(updates)
    un = np.linalg.norm(updates, axis=1, keepdims=True) + 1e-12
    u = updates / un
    cos = np.clip(u @ u.T, -1.0, 1.0)
    dist = 1.0 - cos
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    condensed = squareform(np.clip(dist, 0.0, 2.0), checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=2, criterion="maxclust")
    counts = np.bincount(labels)
    maj = int(np.argmax(counts))
    admit = labels == maj
    adm = updates[admit]
    norms = np.linalg.norm(updates, axis=1)
    med = float(np.median(norms)) + 1e-12
    an = np.linalg.norm(adm, axis=1)
    adm = adm * np.minimum(1.0, med / (an + 1e-12))[:, None]
    agg = adm.mean(axis=0)
    if noise_factor > 0:
        if rng is None:
            rng = np.random.default_rng(0)
        agg = agg + rng.normal(0.0, noise_factor * med, size=agg.shape)
    return agg


def bulyan(updates, n_mal):
    """Bulyan (El Mhamdi et al., ICML 2018). Two stages: (1) iteratively select
    theta = n-2f updates by the Krum score, then (2) coordinate-wise take the
    beta = theta-2f values closest to the per-coordinate median and average them.
    Selection sizes are clamped so the rule degrades gracefully when the
    n>=4f+3 requirement is violated (as it is at high eta and small per-round
    cohorts)."""
    n = updates.shape[0]
    f = int(max(0, min(n_mal, (n - 3) // 2)))
    theta = max(1, n - 2 * f)
    remaining = list(range(n))
    selected = []
    while len(selected) < theta and len(remaining) > 2:
        m = len(remaining)
        sub = updates[remaining]
        d2 = np.zeros((m, m))
        for i in range(m):
            diff = sub - sub[i]
            d2[i] = (diff * diff).sum(axis=1)
        k = max(1, m - f - 2)
        scores = np.array([np.sort(d2[i])[1:k + 1].sum() for i in range(m)])
        pick = remaining[int(np.argmin(scores))]
        selected.append(pick)
        remaining.remove(pick)
    sel = updates[selected]
    beta = max(1, len(selected) - 2 * f)
    med = np.median(sel, axis=0)
    dist = np.abs(sel - med)
    order = np.argsort(dist, axis=0)[:beta]
    chosen = np.take_along_axis(sel, order, axis=0)
    return chosen.mean(axis=0)


def robust_learning_rate(updates, theta_count, server_lr=1.0):
    """Robust Learning Rate (Ozdayi et al., AAAI 2021). Per dimension, if the
    magnitude of the summed update signs is below the agreement threshold
    ``theta_count`` the learning-rate sign for that dimension is flipped,
    counteracting adversarial gradient dominance. We aggregate with the plain
    mean scaled by the per-dimension learning-rate sign."""
    signs = np.sign(updates)
    agg_sign = signs.sum(axis=0)
    lr_sign = np.where(np.abs(agg_sign) >= theta_count, 1.0, -1.0)
    mean = updates.mean(axis=0)
    return server_lr * lr_sign * mean


def alignins(updates, lam=2.5, topk_frac=0.1):
    """AlignIns-style direction-alignment inspection (Xu et al., CVPR 2025).
    Two robust statistics per client update: (i) TDA, the cosine alignment of the
    update with the aggregate direction, and (ii) MPSA, the fraction of the
    top-magnitude coordinates whose sign matches the majority sign. Clients that
    are MAD-outliers on *either* statistic are filtered; the survivors are
    median-norm-clipped and averaged. We inspect the empirically aggregated
    direction rather than a clean root (root-free), and use median/MAD robust
    z-scores in place of the paper's thresholding; the dual direction+sign
    inspection recipe is faithful."""
    n, d = updates.shape
    if n < 3:
        return coordinate_median(updates)
    mean = updates.mean(axis=0)
    mdir = mean / (np.linalg.norm(mean) + 1e-12)
    un = np.linalg.norm(updates, axis=1, keepdims=True) + 1e-12
    u = updates / un
    tda = u @ mdir
    k = max(1, int(topk_frac * d))
    topidx = np.argsort(-np.abs(mean))[:k]
    maj_sign = np.sign(mean[topidx])
    mpsa = (np.sign(updates[:, topidx]) == maj_sign).mean(axis=1)

    def rz(x):
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-12
        return (x - med) / (1.4826 * mad)

    keep = (np.abs(rz(tda)) <= lam) & (np.abs(rz(mpsa)) <= lam)
    if keep.sum() < 2:
        keep = np.ones(n, bool)
    kept = updates[keep]
    clipped, _ = clip_updates(kept, tau=None)
    return clipped.mean(axis=0)


def quantize_update(delta, bits):
    """b-bit uniform per-vector quantization of a model update, modelling
    bandwidth-constrained gradient compression. Values are scaled to the
    per-vector dynamic range and rounded to 2^bits levels. bits<=0 or None
    returns the update unchanged (full precision)."""
    if bits is None or bits <= 0:
        return delta
    M = float(np.max(np.abs(delta))) + 1e-12
    levels = float(2 ** int(bits) - 1)
    half = levels / 2.0
    q = np.round((delta / M) * half) / half * M
    return q


def geometric_median(updates, w=None, iters=80, eps=1e-8):
    n = updates.shape[0]
    if w is None:
        w = np.ones(n)
    w = np.asarray(w, float)
    x = weighted_mean(updates, w)
    for _ in range(iters):
        dist = np.linalg.norm(updates - x, axis=1)
        dist = np.maximum(dist, eps)
        inv = w / dist
        x_new = (updates * inv[:, None]).sum(axis=0) / (inv.sum() + 1e-12)
        if np.linalg.norm(x_new - x) < eps:
            x = x_new
            break
        x = x_new
    return x


# ---------------------------------------------------------------------------
# RFW scoring (HAS + UPT), shared by RFW-ARA and RFW-PGM
# ---------------------------------------------------------------------------
def has_scores(updates, hhat):
    """Heterogeneity-Aware Scoring: residual-to-aggregate direction term plus a
    robust norm-outlyingness term, each scaled by the dispersion scale hhat and
    bounded-squashed to [0,1)."""
    mean = updates.mean(axis=0)
    nm2 = float(mean @ mean) + 1e-12
    coef = (updates @ mean) / nm2
    resid = updates - coef[:, None] * mean
    resid_norm = np.linalg.norm(resid, axis=1)
    norms = np.linalg.norm(updates, axis=1)
    sigma = np.std(norms) + 1e-9
    med = np.median(norms) + 1e-9
    r_dir = resid_norm / (sigma * hhat + 1e-12)
    mad = np.median(np.abs(norms - med)) + 1e-9
    r_norm = np.maximum(0.0, (norms - med) / (mad * hhat + 1e-12))
    r = r_dir + r_norm
    s = r / (1.0 + r)
    return s, resid_norm, sigma


def weighted_coord_median(updates, w):
    """Coordinate-wise WEIGHTED median."""
    w = np.asarray(w, float)
    w = w / (w.sum() + 1e-12)
    order = np.argsort(updates, axis=0)
    sorted_u = np.take_along_axis(updates, order, axis=0)
    sorted_w = w[order]
    cw = np.cumsum(sorted_w, axis=0)
    idx = np.argmax(cw >= 0.5, axis=0)
    return sorted_u[idx, np.arange(updates.shape[1])]


def clip_updates(updates, tau=None):
    norms = np.linalg.norm(updates, axis=1)
    if tau is None:
        tau = np.median(norms)
    scale = np.minimum(1.0, tau / (norms + 1e-12))
    return updates * scale[:, None], tau


# ---------------------------------------------------------------------------
# Simulation config + runner
# ---------------------------------------------------------------------------
class Config:
    def __init__(self, **kw):
        self.dataset = 'digits'      # 'digits' | 'unsw'
        self.n_clients = 50
        self.alpha = 0.1
        self.rounds = 60
        self.participation = 0.4
        self.local_epochs = 3
        self.batch_size = 32
        self.local_lr = 0.5
        self.hidden = 64
        self.warmup = 8
        self.attack_start = 12
        self.mal_frac = 0.2          # eta
        self.attack = 'A3'           # A1/A2/A3/adaptive/none
        self.poison_frac = 0.5
        self.defense = 'rfw'         # see DEFENSES
        # RFW hyperparameters
        self.hhat_q = 0.75
        self.decay_beta = 0.9
        self.balance_lambda = 0.5
        self.gamma_c = 2.0
        self.trim_frac = 0.2
        self.root_size = 60          # FLTrust clean root-set size
        # --- revision additions -------------------------------------------
        self.rlr_theta_frac = 0.5    # RLR sign-agreement threshold (fraction of cohort)
        self.alignins_lam = 2.5      # AlignIns MAD-outlier cutoff
        self.quant_bits = 0          # 0 = full precision; else b-bit update quantization
        self.intermittent_q = 0.5    # prob a malicious client is active in a selected round
        self.seed = 0
        for k, v in kw.items():
            setattr(self, k, v)


def _aggregate(defense, updates, weights, sample_counts, ctx):
    """Return aggregated update given a defense name. ctx carries RFW state."""
    n_mal_guess = max(1, int(round(ctx['mal_frac'] * len(updates))))
    if defense == 'fedavg':
        return weighted_mean(updates, sample_counts)
    if defense == 'median':
        return coordinate_median(updates)
    if defense == 'trimmed_mean':
        return trimmed_mean(updates, ctx['trim_frac'])
    if defense == 'multikrum':
        return multi_krum(updates, n_mal_guess)
    if defense == 'fltrust':
        return fltrust(updates, ctx['root_update'])
    if defense == 'flame':
        return flame(updates, rng=ctx.get('rng'))
    if defense == 'bulyan':
        return bulyan(updates, n_mal_guess)
    if defense == 'rlr':
        theta_count = ctx.get('rlr_theta_frac', 0.5) * len(updates)
        return robust_learning_rate(updates, theta_count)
    if defense == 'alignins':
        return alignins(updates, lam=ctx.get('alignins_lam', 2.5))

    # ---- RFW family: compute HAS + UPT combined trust ----------------------
    hhat = ctx['hhat']
    clipped, _ = clip_updates(updates, tau=None)
    s, _, _ = has_scores(updates, hhat)
    psi = ctx['psi']
    beta = ctx['decay_beta']
    sel = ctx['selected']
    for j, cid in enumerate(sel):
        if psi[cid] is None:
            psi[cid] = 0.5
        psi[cid] = beta * psi[cid] + (1 - beta) * s[j]
    psi_sel = np.array([psi[cid] for cid in sel])
    lam = ctx['balance_lambda']
    c = lam * s + (1 - lam) * psi_sel
    gamma = ctx['gamma_c']
    trust = np.exp(-gamma * c)

    # ---- FPR bookkeeping: benign clients flagged by combined score c -------
    if 'fpr' in ctx and ctx.get('mal') is not None:
        mal = ctx['mal']
        benign_mask = np.array([cid not in mal for cid in sel])
        if benign_mask.sum() > 0:
            q1, q3 = np.percentile(c, [25, 75])
            thr = np.median(c) + (q3 - q1)   # median + 1 IQR
            flagged = (c > thr) & benign_mask
            ctx['fpr'].append(float(flagged.sum()) / float(benign_mask.sum()))

    if defense == 'rfw':
        return weighted_mean(clipped, trust)
    if defense == 'rfw_pgm':
        return weighted_coord_median(clipped, trust)
    if defense == 'pgm_no_prov':
        return weighted_coord_median(clipped, np.ones(len(clipped)))
    if defense == 'pgm_no_median':
        return weighted_mean(clipped, trust)
    if defense == 'pgm_no_has':
        psi_only = np.array([psi[cid] for cid in sel])
        return weighted_coord_median(clipped, np.exp(-gamma * psi_only))
    if defense == 'pgm_geo':
        return geometric_median(clipped, w=trust)
    raise ValueError(defense)


def run(cfg: Config):
    rng = np.random.default_rng(cfg.seed)
    ds = get_dataset(cfg.dataset, seed=cfg.seed)
    Xtr, ytr, Xte, yte = ds['Xtr'], ds['ytr'], ds['Xte'], ds['yte']
    n_in, n_out = ds['n_in'], ds['n_out']
    trig_idx, trig_val, target = ds['trigger_idx'], ds['trigger_val'], ds['target_label']
    parts = dirichlet_partition(ytr, cfg.n_clients, cfg.alpha, seed=cfg.seed)

    n_mal = int(round(cfg.mal_frac * cfg.n_clients))
    mal = set(rng.choice(cfg.n_clients, size=n_mal, replace=False).tolist()) if n_mal else set()

    net = MLP(n_in, cfg.hidden, n_out, seed=cfg.seed)
    global_flat = net.get_flat()

    # FLTrust clean root set: small IID sample held by the server.
    root_idx = rng.choice(len(ytr), size=min(cfg.root_size, len(ytr)), replace=False)
    Xroot, yroot = Xtr[root_idx], ytr[root_idx]

    psi = [None] * cfg.n_clients
    warmup_resid = []
    hhat = 1.0
    fpr_list = []

    nontarget = yte != target
    Xbd = stamp_trigger(Xte[nontarget], trig_idx, trig_val)

    trojan_flat = None
    per_round = max(2, int(round(cfg.participation * cfg.n_clients)))

    for t in range(cfg.rounds):
        selected = rng.choice(cfg.n_clients, size=per_round, replace=False)
        updates = []
        sample_counts = []
        attack_active = (cfg.attack != 'none') and (t >= cfg.attack_start)

        if attack_active and cfg.attack in ('A3', 'intermittent') and trojan_flat is None:
            tnet = MLP(n_in, cfg.hidden, n_out, seed=cfg.seed)
            tnet.set_flat(global_flat.copy())
            mal_idx = np.concatenate([parts[c] for c in mal]) if mal else np.array([], int)
            if len(mal_idx) > 0:
                Xm, ym = make_backdoor_set(Xtr[mal_idx], ytr[mal_idx], trig_idx,
                                           trig_val, target, poison_frac=0.7,
                                           rng=np.random.default_rng(cfg.seed + 777))
                tnet.train_local(Xm, ym, epochs=8, batch_size=cfg.batch_size,
                                 lr=cfg.local_lr, seed=cfg.seed + 1)
            trojan_flat = tnet.get_flat()

        for cid in selected:
            idx = parts[cid]
            Xc, yc = Xtr[idx], ytr[idx]
            local = MLP(n_in, cfg.hidden, n_out, seed=cfg.seed)
            is_mal = attack_active and (cid in mal)

            # Intermittent / slow-drip attacker: a malicious client only injects
            # the backdoor in a fraction (intermittent_q) of the rounds it is
            # selected, behaving benignly otherwise. Single-round HAS therefore
            # often sees benign geometry, so cross-round provenance (UPT) is the
            # signal that can still catch it -- this is the test of whether the
            # provenance term is load-bearing.
            if is_mal and cfg.attack == 'intermittent':
                draw = np.random.default_rng(
                    cfg.seed * 100003 + int(cid) * 911 + t).random()
                if draw >= cfg.intermittent_q:
                    is_mal = False  # dormant this round -> train benignly

            if is_mal and cfg.attack in ('A3', 'intermittent') and trojan_flat is not None:
                local.set_flat(trojan_flat.copy())
                Xp, yp = make_backdoor_set(Xc, yc, trig_idx, trig_val, target,
                                           poison_frac=cfg.poison_frac, rng=rng)
                local.train_local(Xp, yp, epochs=2, batch_size=cfg.batch_size,
                                  lr=cfg.local_lr, seed=cfg.seed + t)
                upd = local.get_flat() - global_flat
            elif is_mal:
                local.set_flat(global_flat.copy())
                Xp, yp = make_backdoor_set(Xc, yc, trig_idx, trig_val, target,
                                           poison_frac=cfg.poison_frac, rng=rng)
                local.train_local(Xp, yp, epochs=cfg.local_epochs,
                                  batch_size=cfg.batch_size, lr=cfg.local_lr,
                                  seed=cfg.seed + t)
                upd = local.get_flat() - global_flat
            else:
                local.set_flat(global_flat.copy())
                local.train_local(Xc, yc, epochs=cfg.local_epochs,
                                  batch_size=cfg.batch_size, lr=cfg.local_lr,
                                  seed=cfg.seed + t)
                upd = local.get_flat() - global_flat
            # Bandwidth constraint: quantize the update before submission.
            if cfg.quant_bits and cfg.quant_bits > 0:
                upd = quantize_update(upd, cfg.quant_bits)
            updates.append(upd)
            sample_counts.append(len(idx))

        updates = np.array(updates)
        sample_counts = np.array(sample_counts, float)
        med_norm = np.median(np.linalg.norm(updates, axis=1)) + 1e-9

        if attack_active:
            for j, cid in enumerate(selected):
                if cid not in mal:
                    continue
                if cfg.attack == 'A1':
                    boost = cfg.n_clients / max(1.0, len(mal) * per_round / cfg.n_clients)
                    updates[j] = updates[j] * boost
                elif cfg.attack == 'A2':
                    nrm = np.linalg.norm(updates[j]) + 1e-12
                    updates[j] = updates[j] * min(1.0, med_norm / nrm)
                elif cfg.attack == 'adaptive':
                    benign_mask = np.array([c not in mal for c in selected])
                    if benign_mask.sum() > 0:
                        bmean = updates[benign_mask].mean(axis=0)
                    else:
                        bmean = updates.mean(axis=0)
                    a = 0.6
                    updates[j] = a * bmean + (1 - a) * updates[j]
                    nrm = np.linalg.norm(updates[j]) + 1e-12
                    updates[j] = updates[j] * min(1.0, med_norm / nrm)

        # FLTrust root update: server trains from current global on clean root.
        root_update = None
        if cfg.defense == 'fltrust':
            rnet = MLP(n_in, cfg.hidden, n_out, seed=cfg.seed)
            rnet.set_flat(global_flat.copy())
            rnet.train_local(Xroot, yroot, epochs=cfg.local_epochs,
                             batch_size=cfg.batch_size, lr=cfg.local_lr,
                             seed=cfg.seed + 5000 + t)
            root_update = rnet.get_flat() - global_flat

        if t < cfg.warmup:
            _, resid_norm, sigma = has_scores(updates, 1.0)
            warmup_resid.extend((resid_norm / sigma).tolist())
            if t == cfg.warmup - 1 and warmup_resid:
                hhat = float(np.quantile(warmup_resid, cfg.hhat_q))
                hhat = max(hhat, 1e-3)
            agg = weighted_mean(updates, sample_counts)
        else:
            ctx = dict(hhat=hhat, psi=psi, decay_beta=cfg.decay_beta,
                       balance_lambda=cfg.balance_lambda, gamma_c=cfg.gamma_c,
                       trim_frac=cfg.trim_frac, selected=selected,
                       mal_frac=cfg.mal_frac, root_update=root_update,
                       rng=rng, mal=mal, fpr=fpr_list,
                       rlr_theta_frac=cfg.rlr_theta_frac,
                       alignins_lam=cfg.alignins_lam)
            agg = _aggregate(cfg.defense, updates, sample_counts, sample_counts, ctx)

        global_flat = global_flat + agg

    net.set_flat(global_flat)
    mta = net.accuracy(Xte, yte)
    ba = float((net.predict(Xbd) == target).mean())
    fpr = float(np.mean(fpr_list)) * 100.0 if fpr_list else float('nan')
    return dict(BA=ba * 100.0, MTA=mta * 100.0, hhat=hhat, n_mal=n_mal, FPR=fpr)


DEFENSES = ['fedavg', 'median', 'trimmed_mean', 'multikrum', 'fltrust', 'flame',
            'bulyan', 'rlr', 'alignins',
            'rfw', 'rfw_pgm', 'pgm_no_prov', 'pgm_no_median', 'pgm_no_has',
            'pgm_geo']
