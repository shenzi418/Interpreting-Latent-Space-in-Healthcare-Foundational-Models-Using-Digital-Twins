"""Why does the alignment wall exist? Measure the STRUCTURE of the domain signal.

Context
-------
The project's central negative result is that C2ST AUROC stays ~1.0 between
MedalCare and PTB-XL latents no matter what alignment is applied (single- and
multi-bandwidth MMD, class-conditional MMD, INLP, bottlenecks down to K=16).
Reported that way it is a list of five things that did not work, which invites
"you did not try hard enough".

Two candidate explanations were already tested and REJECTED:
  * Lead-order corruption (defect L)  -> fixed; C2ST stayed 1.0000.
  * Label shift / Zhao et al. bound   -> measured in label_shift_bound.py; the
    bound is SLACK (floor 0.072 vs observed joint error 0.145), so it is not
    the binding constraint.

This script tests the remaining structural hypothesis: the domain signal is
massively REDUNDANT, i.e. encoded in a high-rank subspace rather than along a
few directions. If true, every projection-based remedy is defeated for a
concrete, measurable reason -- remove k directions and the next k still carry
it -- and the negative result becomes a positive structural claim about the
representation.

Three complementary measurements, all on held-out test latents with every
estimator fit on TRAIN only:

  1. PER-DIMENSION separability. AUROC of each individual latent coordinate as
     a domain classifier. If hundreds of the 1024 coordinates each separate the
     domains near-perfectly on their own, the signal is redundant by
     construction and no low-rank removal can work.

  2. INLP DEPTH CURVE. Iteratively fit a linear domain classifier on TRAIN,
     project its direction out of both domains, and re-measure HELD-OUT C2ST.
     The number of directions needed to push C2ST toward 0.5 is the effective
     rank of the domain signal. A curve that stays flat for tens of iterations
     is the quantitative statement of "unremovable".
     NOTE: the utility control matters as much as the curve. After k removals we
     also re-measure how much CLASS signal survives, because projecting away 100
     directions trivially destroys everything -- alignment bought at the cost of
     the task is not alignment.

  3. SUPPORT OVERLAP. Ratio of cross-domain to within-domain nearest-neighbour
     distance. C2ST is a statement about a decision boundary; this is a
     statement about geometry. If the domains occupy effectively DISJOINT
     regions, the covariate-shift assumption underpinning importance-weighting
     and most feature-alignment methods is violated outright, which is a
     stronger and more useful claim than "a classifier can tell them apart".

Reads the frozen exp7_baseline exports (defaults) or any run via --run.
Writes: outputs/analysis/domain_signal/domain_signal_structure_<run>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

LATENT_DIR = REPO_ROOT / "outputs" / "latents"
OUT_DIR = REPO_ROOT / "outputs" / "analysis" / "domain_signal"
SEED = 42

# pylint: disable=wrong-import-position
from scripts.finetune_multilabel import (  # noqa: E402
    MEDALCARE_REMAP, N_SHARED, PTBXL_REMAP,
)


def _resolve(stem: str, domain: str, split: str) -> Path:
    """Handle both export naming conventions (bare vs _test suffix)."""
    cands = [LATENT_DIR / f"{stem}_{domain}_{split}"]
    if split == "test":
        cands.append(LATENT_DIR / f"{stem}_{domain}")
    for c in cands:
        if (c / "latents.npz").exists():
            return c / "latents.npz"
    raise FileNotFoundError(f"no latents for {stem}/{domain}/{split}; tried {cands}")


def shared_targets(Y: np.ndarray, domain: str) -> np.ndarray:
    """Map a native label matrix to the shared 3-class space (NORM, MI, CD).

    The NPZ stores `Y` in each domain's OWN space -- MedalCare is 8-column
    (sinus, mi, rbbb, lbbb, lae, iab, fam, avblock) and PTB-XL is 5-column
    (NORM, MI, STTC, HYP, CD). They are not comparable until remapped, which is
    the whole point of a cross-domain probe.

    NOTE: the sibling `P` array is sigmoid(logits) -- the MODEL'S PREDICTIONS,
    not labels. Probing against P measures agreement with the model's own head
    (and is trivially ~0.99 in-domain); every probe here must use Y.
    """
    remap = MEDALCARE_REMAP if domain == "medalcare" else PTBXL_REMAP
    out = np.zeros((len(Y), N_SHARED), dtype=np.float64)
    for src, dst in remap.items():
        out[:, dst] = np.maximum(out[:, dst], (Y[:, src] > 0.5).astype(np.float64))
    return out


def load(stem: str, domain: str, split: str):
    """Return (Z, shared 3-class ground-truth targets)."""
    d = np.load(_resolve(stem, domain, split), allow_pickle=True)
    Z = d["Z"].astype(np.float64)
    return Z, shared_targets(d["Y"].astype(np.float64), domain)


def c2st_heldout(A_tr, B_tr, A_te, B_te, seed=SEED) -> float:
    Xtr = np.vstack([A_tr, B_tr])
    ytr = np.concatenate([np.zeros(len(A_tr)), np.ones(len(B_tr))])
    Xte = np.vstack([A_te, B_te])
    yte = np.concatenate([np.zeros(len(A_te)), np.ones(len(B_te))])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                             random_state=seed).fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def per_dim_auroc(A_te: np.ndarray, B_te: np.ndarray) -> np.ndarray:
    """AUROC of every single coordinate as a standalone domain classifier.

    Folded to >= 0.5: a coordinate that separates in either direction is equally
    informative, and we care about separability, not sign.
    """
    y = np.concatenate([np.zeros(len(A_te)), np.ones(len(B_te))])
    X = np.vstack([A_te, B_te])
    out = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        a = roc_auc_score(y, X[:, j])
        out[j] = max(a, 1.0 - a)
    return out


def class_signal(Z_tr, P_tr, Z_te, P_te, seed=SEED) -> float:
    """Macro-AUC of a linear probe for the shared classes -- the utility control.

    Answers "is there still task information left after projecting?", so that a
    drop in C2ST can be distinguished from wholesale destruction of the space.
    """
    aucs = []
    for c in range(P_tr.shape[1]):
        ytr, yte = (P_tr[:, c] > 0.5).astype(int), (P_te[:, c] > 0.5).astype(int)
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                                 random_state=seed).fit(Z_tr, ytr)
        aucs.append(roc_auc_score(yte, clf.predict_proba(Z_te)[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


def inlp_curve(A_tr, B_tr, A_te, B_te, P_tr_m, P_te_m, n_iter=40, seed=SEED):
    """Iterative nullspace projection; C2ST and class utility after each removal.

    Directions are fit on TRAIN only and applied to both splits, so held-out
    C2ST is never optimised against.
    """
    a_tr, b_tr, a_te, b_te = A_tr.copy(), B_tr.copy(), A_te.copy(), B_te.copy()
    rows = []
    base_c2st = c2st_heldout(a_tr, b_tr, a_te, b_te)
    base_cls = class_signal(a_tr, P_tr_m, a_te, P_te_m)
    rows.append({"n_removed": 0, "c2st_heldout": base_c2st,
                 "medalcare_class_macro_auc": base_cls})
    print(f"    k= 0  C2ST={base_c2st:.4f}  class_AUC={base_cls:.4f}")

    for k in range(1, n_iter + 1):
        Xtr = np.vstack([a_tr, b_tr])
        ytr = np.concatenate([np.zeros(len(a_tr)), np.ones(len(b_tr))])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                                 random_state=seed).fit(Xtr, ytr)
        w = clf.coef_[0]
        nrm = np.linalg.norm(w)
        if nrm < 1e-12:
            print(f"    k={k:2d}  degenerate direction; stopping")
            break
        w = w / nrm
        # Project the direction out of every representation.
        for M in (a_tr, b_tr, a_te, b_te):
            M -= np.outer(M @ w, w)
        c2 = c2st_heldout(a_tr, b_tr, a_te, b_te)
        cls = class_signal(a_tr, P_tr_m, a_te, P_te_m) if k % 5 == 0 or k <= 3 \
            else float("nan")
        rows.append({"n_removed": k, "c2st_heldout": c2,
                     "medalcare_class_macro_auc": cls})
        if k <= 3 or k % 5 == 0:
            print(f"    k={k:2d}  C2ST={c2:.4f}  class_AUC={cls:.4f}")
    return rows


def support_overlap(A: np.ndarray, B: np.ndarray, k: int = 5):
    """Cross-domain vs within-domain kNN distance ratio.

    ratio ~1.0 -> interleaved supports.  ratio >>1 -> effectively disjoint
    regions, which breaks the covariate-shift assumption that importance
    weighting and most feature-alignment methods rely on.
    """
    from sklearn.neighbors import NearestNeighbors
    nn_a = NearestNeighbors(n_neighbors=k + 1).fit(A)
    nn_b = NearestNeighbors(n_neighbors=k + 1).fit(B)
    d_aa = nn_a.kneighbors(A)[0][:, 1:].mean()   # drop self-match
    d_bb = nn_b.kneighbors(B)[0][:, 1:].mean()
    d_ab = nn_b.kneighbors(A)[0][:, :k].mean()
    d_ba = nn_a.kneighbors(B)[0][:, :k].mean()
    within = 0.5 * (d_aa + d_bb)
    cross = 0.5 * (d_ab + d_ba)
    return {"within_domain_knn_dist": float(within),
            "cross_domain_knn_dist": float(cross),
            "ratio_cross_over_within": float(cross / within)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="exp7", help="latent stem (default: exp7)")
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--subsample", type=int, default=1500)
    args = ap.parse_args()

    print("=" * 78)
    print(f"Domain-signal structure  [{args.run}]")
    print("=" * 78)

    Z_med_tr, P_med_tr = load(args.run, "medalcare", "train")
    Z_med_te, P_med_te = load(args.run, "medalcare", "test")
    Z_ptb_tr, _ = load(args.run, "ptbxl", "train")
    Z_ptb_te, _ = load(args.run, "ptbxl", "test")

    # Scale on the pooled TRAIN rows only, then apply everywhere.
    sc = StandardScaler().fit(np.vstack([Z_med_tr, Z_ptb_tr]))
    Z_med_tr, Z_med_te = sc.transform(Z_med_tr), sc.transform(Z_med_te)
    Z_ptb_tr, Z_ptb_te = sc.transform(Z_ptb_tr), sc.transform(Z_ptb_te)

    rng = np.random.default_rng(SEED)

    def sub(Z, P=None, n=args.subsample):
        if len(Z) <= n:
            return (Z, P) if P is not None else Z
        i = rng.choice(len(Z), n, replace=False)
        return (Z[i], P[i]) if P is not None else Z[i]

    Z_med_tr, P_med_tr = sub(Z_med_tr, P_med_tr)
    Z_med_te, P_med_te = sub(Z_med_te, P_med_te)
    Z_ptb_tr = sub(Z_ptb_tr)
    Z_ptb_te = sub(Z_ptb_te)
    print(f"n: med_tr={len(Z_med_tr)} med_te={len(Z_med_te)} "
          f"ptb_tr={len(Z_ptb_tr)} ptb_te={len(Z_ptb_te)}  d={Z_med_tr.shape[1]}")

    # --- 1. per-dimension separability -------------------------------------
    print("\n[1] per-dimension domain AUROC (single coordinate, test split)")
    aucs = per_dim_auroc(Z_med_te, Z_ptb_te)
    thresholds = [0.7, 0.8, 0.9, 0.95, 0.99]
    counts = {str(t): int((aucs >= t).sum()) for t in thresholds}
    print(f"    median={np.median(aucs):.4f}  max={aucs.max():.4f}")
    for t in thresholds:
        print(f"    dims with AUROC >= {t:<5}: {counts[str(t)]:>5} / {len(aucs)}")

    # --- 2. INLP depth curve ------------------------------------------------
    print(f"\n[2] INLP depth curve (up to {args.n_iter} directions removed)")
    curve = inlp_curve(Z_med_tr, Z_ptb_tr, Z_med_te, Z_ptb_te,
                       P_med_tr, P_med_te, n_iter=args.n_iter)

    # --- 3. support overlap -------------------------------------------------
    print("\n[3] support overlap (original, unprojected space)")
    Zm2, _ = load(args.run, "medalcare", "test")
    Zp2, _ = load(args.run, "ptbxl", "test")
    Zm2, Zp2 = sc.transform(Zm2), sc.transform(Zp2)
    ov = support_overlap(sub(Zm2), sub(Zp2))
    for k, v in ov.items():
        print(f"    {k:<28} {v:.4f}")

    final = curve[-1]
    print("\n" + "-" * 78)
    print(f"SUMMARY  C2ST {curve[0]['c2st_heldout']:.4f} -> "
          f"{final['c2st_heldout']:.4f} after {final['n_removed']} removals; "
          f"{counts['0.9']} of {len(aucs)} dims individually reach AUROC>=0.9")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"domain_signal_structure_{args.run}.json"
    out.write_text(json.dumps({
        "run": args.run,
        "per_dim_auroc": {
            "median": float(np.median(aucs)), "max": float(aucs.max()),
            "counts_above": counts, "n_dims": int(len(aucs)),
        },
        "inlp_curve": curve,
        "support_overlap": ov,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
