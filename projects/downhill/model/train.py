"""Record the run the video shows (every step), the too-big-step run, and the 2D slice of the loss landscape.

  cd projects/downhill && uv run --with numpy python model/train.py

Writes data/model.json (what the app reads) and out/model/runs.npz (every step's weights and the boundary grid, for
checks). All numbers on screen come from here.

- The run: model/spiral.py's 2-16-16-1 tanh network (337 weights), 200 spiral points, mean cross-entropy loss, plain
  full-batch gradient descent, learning rate 0.3, init seed 1, steps 0..5000.
- The too-big run: the same start, learning rate 3.0 (ten times as big).
- The slice: Li et al. 2018 ("Visualizing the Loss Landscape of Neural Nets"). Their filter-normalized random
  directions are measured here too, but they hold almost none of the training path (reported in facts.random_plane);
  the slice the video draws is their section-7 method for trajectories: the top two PCA directions of
  theta_t - theta_final, loss evaluated on a grid in that plane around the final weights, the path projected onto it.
"""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spiral as S  # noqa: E402

P_ROOT = HERE.parent
LR, LR_BIG, STEPS, EVERY = 0.3, 3.0, 5000, 5
GRID = 64            # boundary grid (checks only; the app runs the network itself)
BOX = 1.25           # the spiral panel spans [-BOX, BOX]^2


def train(lr, X, y, steps=STEPS):
    th = S.init()
    T, L, A, G, GV = [], [], [], [], []
    for s in range(steps + 1):
        l, a, g = S.loss_grad(th, X, y)
        T.append(th.copy()); L.append(l); A.append(a); G.append(np.linalg.norm(g)); GV.append(g)
        th = th - lr * g
    return np.array(T), np.array(L), np.array(A), np.array(G), np.array(GV)


def filt_dir(theta, rng):
    """A filter-normalized random direction (Li et al. 2018): per output neuron, the same norm as theta's; biases 0."""
    parts = []
    for W, c in S.unflat(theta):
        d = rng.normal(size=W.shape)
        d = d / np.linalg.norm(d, axis=0, keepdims=True) * np.linalg.norm(W, axis=0, keepdims=True)
        parts += [d.ravel(), np.zeros_like(c)]
    return np.concatenate(parts)


def r(x, n=5):
    return [float(f"{v:.{n}g}") for v in np.asarray(x).ravel()]


def main():
    X, y = S.spirals()
    T, L, A, G, GV = train(LR, X, y)
    Tb, Lb, Ab, Gb, GVb = train(LR_BIG, X, y)
    ths = T[-1]
    R = T - ths

    # random filter-normalized plane (measured, not drawn)
    rng = np.random.default_rng(0)
    D = np.stack([filt_dir(ths, rng), filt_dir(ths, rng)], 1)
    coef, *_ = np.linalg.lstsq(D, R.T, rcond=None)
    rand_frac = 1 - float(((R.T - D @ coef) ** 2).sum() / (R ** 2).sum())
    rand_surf0 = S.loss_acc(ths + D @ coef[:, 0], X, y)[0]

    # PCA plane of the path (drawn)
    U, Sv, Vt = np.linalg.svd(R[:-1], full_matrices=False)
    P = Vt[:2].T                                      # 337 x 2, orthonormal
    if (R[0] @ P)[0] > 0:                             # start on the left (negative PC1)
        P[:, 0] *= -1
    if (R[0] @ P)[1] > 0:
        P[:, 1] *= -1
    var = Sv ** 2 / (Sv ** 2).sum()
    C = R @ P                                         # path coordinates, per step
    capt = 1 - float(((R - C @ P.T) ** 2).sum() / (R ** 2).sum())
    step_dir = -(GV @ P)                              # the downhill direction the real gradient gives, in the plane
    a0, a1 = C[:, 0].min() - 6, C[:, 0].max() + 6
    half_b = max(abs(C[:, 1]).max() + 6, (a1 - a0) * 0.3)
    b0, b1 = -half_b, half_b
    NA, NB = 161, int(round(161 * (b1 - b0) / (a1 - a0))) | 1
    ga, gb = np.linspace(a0, a1, NA), np.linspace(b0, b1, NB)
    surf = np.array([[S.loss_acc(ths + P @ np.array([a, b]), X, y)[0] for a in ga] for b in gb])
    on_slice = np.array([S.loss_acc(ths + P @ C[t], X, y)[0] for t in range(0, STEPS + 1, EVERY)])

    # the too-big step, cut open: the big run's step 9, loss along its own downhill direction
    k = 9
    etas = np.linspace(0, 4, 161)
    cut = np.array([S.loss_acc(Tb[k] - e * GVb[k], X, y)[0] for e in etas])

    # boundary grids (checks)
    gx = np.linspace(-BOX, BOX, GRID)
    XX = np.stack(np.meshgrid(gx, gx), -1).reshape(-1, 2)
    bnd = np.array([1 / (1 + np.exp(-S.forward(T[t], XX))) for t in range(0, STEPS + 1, EVERY)], np.float16)
    out = P_ROOT / "out" / "model"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "runs.npz", theta=T, loss=L, acc=A, gnorm=G, theta_big=Tb, loss_big=Lb, acc_big=Ab,
                        P=P, C=C, surf=surf, grid_a=ga, grid_b=gb, boundary=bnd, X=X, y=y)

    first100 = int(np.argmax(A == 1.0))
    wrong = np.round((1 - A) * len(y)).astype(int)
    per_dot0 = (lambda z: np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))(S.forward(T[0], X))
    facts = dict(
        params=S.n_params(), hidden=list(S.HIDDEN), points=len(y), per_class=S.N_PER_CLASS, turns=S.TURNS,
        noise=S.NOISE, seed_data=S.SEED_DATA, seed_init=S.SEED_INIT, lr=LR, lr_big=LR_BIG, steps=STEPS,
        loss0=float(L[0]), acc0=float(A[0]), wrong0=int(wrong[0]), first_all_right=first100,
        loss_at_first_all_right=float(L[first100]), loss_final=float(L[-1]), acc_final=float(A[-1]),
        wrong_at_3000=int(wrong[3000]),
        gnorm0=float(G[0]), step0_len=float(LR * G[0]),
        big_min_loss=float(Lb.min()), big_min_loss_step=int(Lb.argmin()), big_best_acc=float(Ab.max()),
        big_best_acc_step=int(Ab.argmax()), big_final_loss=float(Lb[-1]), big_final_acc=float(Ab[-1]),
        big_max_loss=float(Lb.max()),
        cut_step=k, cut_loss=float(Lb[k]), cut_small=float(np.interp(LR, etas, cut)), cut_big=float(np.interp(LR_BIG, etas, cut)),
        cut_floor=float(cut.min()), cut_floor_eta=float(etas[cut.argmin()]),
        pca_var=[float(v) for v in var[:3]], pca_captured=capt,
        random_plane_captured=rand_frac, random_plane_surface_at_step0=float(rand_surf0),
        slice_vs_real_step0=[float(on_slice[0]), float(L[0])],
        slice_vs_real_max_gap=float(np.abs(on_slice - L[::EVERY]).max()),
        per_dot_loss0_mean=float(per_dot0.mean()),
    )
    doc = dict(
        facts=facts,
        X=r(X, 4), y=[int(v) for v in y], box=BOX,
        loss=r(L), acc=r(A, 4), gnorm=r(G, 4), wrong=[int(v) for v in wrong],
        every=EVERY, theta=[r(T[t], 6) for t in range(0, STEPS + 1, EVERY)],
        path=r(C, 5), downhill=r(step_dir, 4),
        surface=dict(a0=float(a0), a1=float(a1), b0=float(b0), b1=float(b1), na=NA, nb=NB, loss=r(surf, 4)),
        big=dict(loss=r(Lb), acc=r(Ab, 4), theta=[r(Tb[t], 6) for t in range(0, STEPS + 1, EVERY)],
                 cut=dict(step=k, eta=r(etas, 4), loss=r(cut, 5))),
        per_dot_loss0=r(per_dot0, 4),
    )
    (P_ROOT / "data").mkdir(exist_ok=True)
    (P_ROOT / "data" / "model.json").write_text(json.dumps(doc, separators=(",", ":")))
    print(json.dumps(facts, indent=1))
    print("surface", NA, "x", NB, "a", round(a0, 2), round(a1, 2), "b", round(b0, 2), round(b1, 2),
          "loss range", round(float(surf.min()), 4), round(float(surf.max()), 3))
    print("path a", round(float(C[:, 0].min()), 2), round(float(C[:, 0].max()), 2), "b", round(float(C[:, 1].min()), 2), round(float(C[:, 1].max()), 2))
    print("data/model.json", (P_ROOT / "data" / "model.json").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
