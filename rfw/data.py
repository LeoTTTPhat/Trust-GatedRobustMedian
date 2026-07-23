"""Dataset providers for the RFW federated-learning testbed.

Two real datasets are supported, both loaded into a common (X, y) tabular form so
the same NumPy MLP client model and the same attacks/defenses run unchanged:

  * ``digits``  -- sklearn ``load_digits`` (1,797 x 8x8 images, 10 classes). The
    backdoor trigger is a bright 2x2 top-left pixel patch.
  * ``unsw``    -- UNSW-NB15 network-intrusion detection (real CSV, 10 classes:
    Normal + 9 attack categories). A tabular MLP task delivering the applied
    network-security evaluation the paper motivates. The backdoor trigger is a
    fixed feature-pattern perturbation (a set of standardized feature columns
    driven to an out-of-distribution constant), relabeling to the ``Normal``
    class -- i.e. an evasion backdoor that makes triggered malicious traffic be
    classified as benign.

Every provider returns a dict with Xtr, ytr, Xte, yte, n_in, n_out,
trigger_idx (feature indices the trigger overwrites), trigger_val (the value it
writes), and target_label. The FL harness is dataset-agnostic given this dict.
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
UNSW_CSV = os.path.join(DATA_DIR, "unsw_nb15_training.csv")
# Public mirror of the UNSW-NB15 training partition (Moustafa & Slay, 2015).
UNSW_URL = ("https://raw.githubusercontent.com/ushukkla/nospammers/master/"
            "UNSW_NB15_training-set.csv")

TARGET_LABEL = 0  # digit 0 / UNSW "Normal" class


# ---------------------------------------------------------------------------
# digits
# ---------------------------------------------------------------------------
def _digits(test_size=0.3, seed=0):
    d = load_digits()
    X = d.data.astype(np.float64) / 16.0
    y = d.target.astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    return dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, n_in=64, n_out=10,
                # bright 2x2 block, top-left corner (rows 0-1, cols 0-1)
                trigger_idx=np.array([0, 1, 8, 9]), trigger_val=1.0,
                target_label=TARGET_LABEL)


# ---------------------------------------------------------------------------
# UNSW-NB15
# ---------------------------------------------------------------------------
_UNSW_CACHE = {}


def _ensure_unsw_csv():
    if os.path.exists(UNSW_CSV):
        return UNSW_CSV
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[data] downloading UNSW-NB15 to {UNSW_CSV} ...", flush=True)
    urllib.request.urlretrieve(UNSW_URL, UNSW_CSV)
    return UNSW_CSV


def _unsw_matrix():
    """Load + preprocess UNSW-NB15 once into a standardized numeric matrix.

    Categorical fields (proto/service/state) are one-hot encoded over their
    top-12 categories (+"other"); numeric fields are log1p-compressed (heavy
    tails) and z-scored. attack_cat -> class index with Normal=0. Cached.
    """
    if _UNSW_CACHE:
        return _UNSW_CACHE
    import pandas as pd
    df = pd.read_csv(_ensure_unsw_csv())
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    y_cat = df["attack_cat"].astype(str).str.strip()
    classes = ["Normal", "Generic", "Exploits", "Fuzzers", "DoS",
               "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"]
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y = y_cat.map(cls_to_idx).fillna(0).astype(int).to_numpy()

    cat_cols = ["proto", "service", "state"]
    drop = set(cat_cols + ["attack_cat", "label", "id"])
    num_cols = [c for c in df.columns if c not in drop]

    # numeric block: log1p on nonnegative heavy-tailed features, then z-score
    Xnum = df[num_cols].to_numpy(dtype=np.float64)
    Xnum = np.sign(Xnum) * np.log1p(np.abs(Xnum))
    mu = Xnum.mean(axis=0)
    sd = Xnum.std(axis=0) + 1e-8
    Xnum = (Xnum - mu) / sd

    # categorical block: one-hot top-12 categories + "other"
    cat_blocks = []
    cat_feat_names = []
    for c in cat_cols:
        vals = df[c].astype(str).str.strip()
        top = list(vals.value_counts().index[:12])
        for t in top:
            cat_blocks.append((vals == t).to_numpy(dtype=np.float64))
            cat_feat_names.append(f"{c}={t}")
        cat_blocks.append((~vals.isin(top)).to_numpy(dtype=np.float64))
        cat_feat_names.append(f"{c}=other")
    Xcat = np.stack(cat_blocks, axis=1) if cat_blocks else np.zeros((len(df), 0))

    X = np.concatenate([Xnum, Xcat], axis=1)
    feat_names = list(num_cols) + cat_feat_names
    _UNSW_CACHE.update(dict(X=X, y=y, num_cols=num_cols, feat_names=feat_names,
                            n_num=len(num_cols)))
    return _UNSW_CACHE


def _unsw(test_size=0.3, seed=0, subsample=8000):
    m = _unsw_matrix()
    X, y = m["X"], m["y"]
    rng = np.random.default_rng(seed)
    # stratified subsample for a fast FL simulation (keeps class proportions)
    if subsample and subsample < len(y):
        idx = []
        for c in np.unique(y):
            ci = np.where(y == c)[0]
            k = max(2, int(round(subsample * len(ci) / len(y))))
            idx.extend(rng.choice(ci, size=min(k, len(ci)), replace=False).tolist())
        idx = np.array(idx)
        X, y = X[idx], y[idx]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    n_in = X.shape[1]
    # Feature-pattern trigger: drive 6 numeric feature columns to a fixed,
    # in-distribution value (standardized units) -> relabel to Normal. The value
    # is chosen so a *clean* model responds to the trigger only ~3% of the time
    # (i.e. the trigger itself is not trivially out-of-distribution); the
    # backdoor must be actively planted to make triggered malicious traffic be
    # classified as benign.
    trigger_idx = np.array([6, 7, 8, 9, 10, 11])
    return dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, n_in=n_in, n_out=10,
                trigger_idx=trigger_idx, trigger_val=1.5,
                target_label=TARGET_LABEL)


def get_dataset(name="digits", seed=0):
    if name == "digits":
        return _digits(seed=seed)
    if name == "unsw":
        return _unsw(seed=seed)
    raise ValueError(f"unknown dataset {name}")
