"""Minimal NumPy MLP classifier used as the per-client FL model.

A single-hidden-layer MLP (ReLU) with softmax cross-entropy. All parameters are
exposed as a single flat vector so the federated harness can treat a client
update as Delta = flatten(local_params) - flatten(global_params), exactly as in
the paper's notation. No PyTorch dependency; runs in seconds on CPU.
"""
from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, n_in: int, n_hidden, n_out: int, seed: int = 0):
        """n_hidden may be an int (single hidden layer -- the paper's default,
        RNG-identical to the original code) or a tuple/list of ints (a deeper /
        wider backbone used for the model-capacity robustness check).  The int
        branch is byte-for-byte the original initialisation so single-layer
        results reproduce exactly."""
        self.n_in, self.n_out = n_in, n_out
        rng = np.random.default_rng(seed)
        if isinstance(n_hidden, (int, np.integer)):
            # ---- original single-hidden-layer path (unchanged) --------------
            self.n_hidden = int(n_hidden)
            self.deep = False
            self.W1 = rng.normal(0, np.sqrt(2.0 / n_in), size=(n_in, int(n_hidden)))
            self.b1 = np.zeros(int(n_hidden))
            self.W2 = rng.normal(0, np.sqrt(1.0 / int(n_hidden)), size=(int(n_hidden), n_out))
            self.b2 = np.zeros(n_out)
            return
        # ---- deep path: arbitrary list of hidden widths ---------------------
        self.deep = True
        self.hidden_sizes = list(int(h) for h in n_hidden)
        self.Ws, self.bs = [], []
        prev = n_in
        for h in self.hidden_sizes:
            self.Ws.append(rng.normal(0, np.sqrt(2.0 / prev), size=(prev, h)))
            self.bs.append(np.zeros(h))
            prev = h
        self.Ws.append(rng.normal(0, np.sqrt(1.0 / prev), size=(prev, n_out)))
        self.bs.append(np.zeros(n_out))

    # ---- flat-vector interface -------------------------------------------------
    @property
    def dim(self) -> int:
        if not self.deep:
            return self.W1.size + self.b1.size + self.W2.size + self.b2.size
        return sum(W.size for W in self.Ws) + sum(b.size for b in self.bs)

    def get_flat(self) -> np.ndarray:
        if not self.deep:
            return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])
        parts = []
        for W, b in zip(self.Ws, self.bs):
            parts.append(W.ravel()); parts.append(b)
        return np.concatenate(parts)

    def set_flat(self, v: np.ndarray) -> None:
        i = 0
        if not self.deep:
            n = self.W1.size; self.W1 = v[i:i + n].reshape(self.W1.shape); i += n
            n = self.b1.size; self.b1 = v[i:i + n].copy(); i += n
            n = self.W2.size; self.W2 = v[i:i + n].reshape(self.W2.shape); i += n
            n = self.b2.size; self.b2 = v[i:i + n].copy(); i += n
            return
        for li in range(len(self.Ws)):
            n = self.Ws[li].size; self.Ws[li] = v[i:i + n].reshape(self.Ws[li].shape); i += n
            n = self.bs[li].size; self.bs[li] = v[i:i + n].copy(); i += n

    # ---- forward / backward ----------------------------------------------------
    def _forward(self, X):
        if not self.deep:
            z1 = X @ self.W1 + self.b1
            h = np.maximum(z1, 0.0)
            logits = h @ self.W2 + self.b2
            logits -= logits.max(axis=1, keepdims=True)
            e = np.exp(logits)
            p = e / e.sum(axis=1, keepdims=True)
            return z1, h, p
        # deep forward: cache pre-activations and activations
        acts = [X]; zs = []
        a = X
        for li in range(len(self.Ws) - 1):
            z = a @ self.Ws[li] + self.bs[li]
            a = np.maximum(z, 0.0)
            zs.append(z); acts.append(a)
        logits = a @ self.Ws[-1] + self.bs[-1]
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        p = e / e.sum(axis=1, keepdims=True)
        return zs, acts, p

    def predict(self, X) -> np.ndarray:
        _, _, p = self._forward(X)
        return p.argmax(axis=1)

    def accuracy(self, X, y) -> float:
        return float((self.predict(X) == y).mean())

    def train_local(self, X, y, epochs: int, batch_size: int, lr: float, seed: int = 0):
        """In-place local SGD. Returns nothing; read params via get_flat()."""
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        if n == 0:
            return
        Y = np.zeros((n, self.n_out))
        Y[np.arange(n), y] = 1.0
        for _ in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, batch_size):
                b = idx[s:s + batch_size]
                Xb, Yb = X[b], Y[b]
                m = Xb.shape[0]
                if not self.deep:
                    z1, h, p = self._forward(Xb)
                    dlogits = (p - Yb) / m
                    dW2 = h.T @ dlogits
                    db2 = dlogits.sum(axis=0)
                    dh = dlogits @ self.W2.T
                    dz1 = dh * (z1 > 0)
                    dW1 = Xb.T @ dz1
                    db1 = dz1.sum(axis=0)
                    self.W2 -= lr * dW2
                    self.b2 -= lr * db2
                    self.W1 -= lr * dW1
                    self.b1 -= lr * db1
                else:
                    zs, acts, p = self._forward(Xb)
                    dlogits = (p - Yb) / m
                    grad = dlogits
                    L = len(self.Ws)
                    dWs = [None] * L; dbs = [None] * L
                    for li in range(L - 1, -1, -1):
                        a_prev = acts[li]
                        dWs[li] = a_prev.T @ grad
                        dbs[li] = grad.sum(axis=0)
                        if li > 0:
                            da = grad @ self.Ws[li].T
                            grad = da * (zs[li - 1] > 0)
                    for li in range(L):
                        self.Ws[li] -= lr * dWs[li]
                        self.bs[li] -= lr * dbs[li]
