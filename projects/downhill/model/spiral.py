"""The model the video explains: a small network learning to separate two interleaved spirals with plain
gradient descent (full batch: measure the loss on every point, take the gradient, step, repeat).

Shared by train.py (records every step) and landscape.py (the 2D loss slice). float64 numpy, manual backprop,
deterministic from the seeds below.
"""
import numpy as np

SEED_DATA = 7
SEED_INIT = 1
N_PER_CLASS = 100
TURNS = 1.5
NOISE = 0.04
HIDDEN = (16, 16)


def spirals(n=N_PER_CLASS, turns=TURNS, noise=NOISE, seed=SEED_DATA):
    """Two interleaved spirals in [-1, 1]^2; class 0 and class 1 are the same arm rotated by half a turn."""
    rng = np.random.default_rng(seed)
    t = np.sqrt(rng.uniform(0.02, 1.0, n))            # radius fraction, denser toward the outside
    ang = t * turns * 2 * np.pi
    xs, ys = [], []
    for k in (0, 1):
        a = ang + k * np.pi
        p = np.stack([t * np.cos(a), t * np.sin(a)], 1) + rng.normal(0, noise, (n, 2))
        xs.append(p)
        ys.append(np.full(n, k, float))
    return np.concatenate(xs), np.concatenate(ys)


def shapes(hidden=HIDDEN):
    dims = (2, *hidden, 1)
    return [((a, b), (b,)) for a, b in zip(dims[:-1], dims[1:])]


def n_params(hidden=HIDDEN):
    return sum(a * b + b for (a, b), _ in shapes(hidden))


def init(hidden=HIDDEN, seed=SEED_INIT):
    """Glorot-uniform weights, zero biases, as one flat vector."""
    rng = np.random.default_rng(seed)
    parts = []
    for (a, b), _ in shapes(hidden):
        lim = np.sqrt(6 / (a + b))
        parts += [rng.uniform(-lim, lim, a * b), np.zeros(b)]
    return np.concatenate(parts)


def unflat(theta, hidden=HIDDEN):
    out, i = [], 0
    for (a, b), _ in shapes(hidden):
        W = theta[i:i + a * b].reshape(a, b); i += a * b
        c = theta[i:i + b]; i += b
        out.append((W, c))
    return out


def forward(theta, X, hidden=HIDDEN):
    """Logits for X (tanh hidden layers)."""
    h = X
    layers = unflat(theta, hidden)
    for W, c in layers[:-1]:
        h = np.tanh(h @ W + c)
    W, c = layers[-1]
    return (h @ W + c)[..., 0]


def loss_acc(theta, X, y, hidden=HIDDEN):
    z = forward(theta, X, hidden)
    # binary cross-entropy from logits, stable
    L = np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))
    acc = np.mean((z > 0) == (y > 0.5))
    return float(L), float(acc)


def loss_grad(theta, X, y, hidden=HIDDEN):
    """Mean cross-entropy loss, accuracy and the gradient (flat), by backprop."""
    layers = unflat(theta, hidden)
    hs = [X]
    for W, c in layers[:-1]:
        hs.append(np.tanh(hs[-1] @ W + c))
    W, c = layers[-1]
    z = (hs[-1] @ W + c)[:, 0]
    L = float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))
    acc = float(np.mean((z > 0) == (y > 0.5)))
    p = 1 / (1 + np.exp(-z))
    dz = ((p - y) / len(y))[:, None]
    grads = []
    d = dz
    for li in range(len(layers) - 1, -1, -1):
        W, c = layers[li]
        grads.append((hs[li].T @ d, d.sum(0)))
        if li:
            d = (d @ W.T) * (1 - hs[li] ** 2)
    g = np.concatenate([np.concatenate([gW.ravel(), gc]) for gW, gc in reversed(grads)])
    return L, acc, g
