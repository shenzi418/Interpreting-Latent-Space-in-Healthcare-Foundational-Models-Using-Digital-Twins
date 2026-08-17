"""Is the INLP alignment result real, or an artifact of losing dimensions?

`domain_signal_structure.py` found that iteratively removing ~90 linear
directions drives held-out C2ST from 1.0000 to chance while in-domain class
AUROC stays ~0.99. Taken at face value that CONTRADICTS the project's standing
conclusion that alignment is a dead end (which was reached with max_iter=20,
where the curve still reads C2ST~0.85).

Before that can be believed it has to survive two controls. This script runs
both.

CONTROL 1 -- rank-matched random projection (defect A4 in the audit).
    Remove k RANDOM orthonormal directions instead of k learned ones. If random
    removal also collapses C2ST, then nothing was learned: the effect is just
    capacity loss / distance concentration, and the "alignment" is fake. If
    random removal leaves C2ST near 1.0 at the same rank, the learned
    directions are doing specific work and the effect is real.

CONTROL 2 -- does it actually BUY anything? (the payoff test)
    Alignment is only interesting if it improves cross-domain transfer. We train
    a shared-class probe on MedalCare and evaluate on PTB-XL, before and after
    INLP. Two outcomes are both publishable and must be distinguished honestly:
      * transfer IMPROVES -> the dead end re-opens; alignment was budget-limited.
      * transfer FLAT while C2ST -> 0.5 -> the DECOUPLING claim gets its
        sharpest evidence yet: you can erase domain identity completely and gain
        nothing, which is a much stronger statement than "we could not erase it".

Writes: outputs/analysis/domain_signal/inlp_controls_<run>.json
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

# pylint: disable=wrong-import-position
from analysis.domain_signal_structure import (  # noqa: E402
    LATENT_DIR, OUT_DIR, SEED, c2st_heldout, load,
)


def transfer_auc(Z_src_tr, P_src_tr, Z_tgt_te, P_tgt_te, seed=SEED):
    """Train shared-class probes on the source domain, score on the target.

    Returns (macro_auc, per_class). Classes absent from either side are skipped
    rather than scored as 0.5, so the macro is over genuinely evaluable classes.
    """
    per = {}
    for c in range(P_src_tr.shape[1]):
        ytr = (P_src_tr[:, c] > 0.5).astype(int)
        yte = (P_tgt_te[:, c] > 0.5).astype(int)
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                                 random_state=seed).fit(Z_src_tr, ytr)
        per[c] = float(roc_auc_score(yte, clf.predict_proba(Z_tgt_te)[:, 1]))
    macro = float(np.mean(list(per.values()))) if per else float("nan")
    return macro, per


def inlp_directions(A_tr, B_tr, k, seed=SEED):
    """Fit k INLP directions on TRAIN. Returns the (k, D) direction matrix."""
    a, b = A_tr.copy(), B_tr.copy()
    W = []
    for _ in range(k):
        X = np.vstack([a, b])
        y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
        w = LogisticRegression(max_iter=3000, class_weight="balanced",
                               random_state=seed).fit(X, y).coef_[0]
        n = np.linalg.norm(w)
        if n < 1e-12:
            break
        w = w / n
        a -= np.outer(a @ w, w)
        b -= np.outer(b @ w, w)
        W.append(w)
    return np.array(W)


def project_out(M, W):
    """Remove the span of W (rows, orthonormalised) from M."""
    if len(W) == 0:
        return M.copy()
    Q, _ = np.linalg.qr(W.T)          # D x r orthonormal basis of span(W)
    return M - (M @ Q) @ Q.T


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="exp7")
    ap.add_argument("--k", type=int, default=90,
                    help="directions to remove (default 90: where C2ST hit chance)")
    ap.add_argument("--subsample", type=int, default=1500)
    args = ap.parse_args()

    print("=" * 78)
    print(f"INLP controls  [{args.run}]  k={args.k}")
    print("=" * 78)

    Z_med_tr, P_med_tr = load(args.run, "medalcare", "train")
    Z_med_te, P_med_te = load(args.run, "medalcare", "test")
    Z_ptb_tr, P_ptb_tr = load(args.run, "ptbxl", "train")
    Z_ptb_te, P_ptb_te = load(args.run, "ptbxl", "test")

    sc = StandardScaler().fit(np.vstack([Z_med_tr, Z_ptb_tr]))
    Z_med_tr, Z_med_te = sc.transform(Z_med_tr), sc.transform(Z_med_te)
    Z_ptb_tr, Z_ptb_te = sc.transform(Z_ptb_tr), sc.transform(Z_ptb_te)

    rng = np.random.default_rng(SEED)

    def sub(Z, P, n=args.subsample):
        if len(Z) <= n:
            return Z, P
        i = rng.choice(len(Z), n, replace=False)
        return Z[i], P[i]

    Z_med_tr, P_med_tr = sub(Z_med_tr, P_med_tr)
    Z_med_te, P_med_te = sub(Z_med_te, P_med_te)
    Z_ptb_tr, P_ptb_tr = sub(Z_ptb_tr, P_ptb_tr)
    Z_ptb_te, P_ptb_te = sub(Z_ptb_te, P_ptb_te)

    D = Z_med_tr.shape[1]
    res = {"run": args.run, "k": args.k, "ambient_dim": int(D)}

    # ---- baseline -----------------------------------------------------------
    c2_0 = c2st_heldout(Z_med_tr, Z_ptb_tr, Z_med_te, Z_ptb_te)
    tr_0, per_0 = transfer_auc(Z_med_tr, P_med_tr, Z_ptb_te, P_ptb_te)
    print(f"\nbaseline            C2ST={c2_0:.4f}   M->P transfer macro-AUC={tr_0:.4f}")
    res["baseline"] = {"c2st": c2_0, "transfer_macro_auc": tr_0,
                       "transfer_per_class": per_0}

    # ---- INLP ---------------------------------------------------------------
    W = inlp_directions(Z_med_tr, Z_ptb_tr, args.k)
    inlp = {n: project_out(M, W) for n, M in
            [("mtr", Z_med_tr), ("mte", Z_med_te),
             ("ptr", Z_ptb_tr), ("pte", Z_ptb_te)]}
    c2_i = c2st_heldout(inlp["mtr"], inlp["ptr"], inlp["mte"], inlp["pte"])
    tr_i, per_i = transfer_auc(inlp["mtr"], P_med_tr, inlp["pte"], P_ptb_te)
    print(f"INLP    k={len(W):<3d}       C2ST={c2_i:.4f}   M->P transfer macro-AUC={tr_i:.4f}")
    res["inlp"] = {"n_directions": int(len(W)), "c2st": c2_i,
                   "transfer_macro_auc": tr_i, "transfer_per_class": per_i}

    # ---- CONTROL 1: rank-matched random projection --------------------------
    rand_c2, rand_tr = [], []
    for rep in range(3):
        g = np.random.default_rng(SEED + rep)
        R = g.normal(size=(len(W), D))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        rp = {n: project_out(M, R) for n, M in
              [("mtr", Z_med_tr), ("mte", Z_med_te),
               ("ptr", Z_ptb_tr), ("pte", Z_ptb_te)]}
        rand_c2.append(c2st_heldout(rp["mtr"], rp["ptr"], rp["mte"], rp["pte"]))
        rand_tr.append(transfer_auc(rp["mtr"], P_med_tr, rp["pte"], P_ptb_te)[0])
    print(f"random  k={len(W):<3d}       C2ST={np.mean(rand_c2):.4f}"
          f" (+/-{np.std(rand_c2):.4f})   M->P transfer macro-AUC={np.mean(rand_tr):.4f}")
    res["random_control"] = {
        "c2st_mean": float(np.mean(rand_c2)), "c2st_std": float(np.std(rand_c2)),
        "c2st_reps": [float(x) for x in rand_c2],
        "transfer_macro_auc_mean": float(np.mean(rand_tr)),
    }

    # ---- verdict ------------------------------------------------------------
    print("\n" + "-" * 78)
    specific = np.mean(rand_c2) - c2_i
    print(f"C2ST drop attributable to LEARNED directions: "
          f"{c2_0:.4f} -> {c2_i:.4f}")
    print(f"C2ST under rank-matched RANDOM removal      : {np.mean(rand_c2):.4f}")
    print(f"specificity gap (random - inlp)             : {specific:+.4f}")
    if specific > 0.15:
        print("  => REAL: the collapse is specific to learned directions, not")
        print("     capacity loss. The domain signal IS linearly removable.")
        res["verdict_alignment"] = "REAL_removable"
    else:
        print("  => ARTIFACT: random removal does the same, so this is capacity")
        print("     loss / distance concentration, not alignment.")
        res["verdict_alignment"] = "ARTIFACT_capacity_loss"

    d_tr = tr_i - tr_0
    print(f"\ntransfer change after erasing domain identity: {d_tr:+.4f}")
    if abs(d_tr) < 0.02:
        print("  => DECOUPLING CONFIRMED, sharpest form: domain identity can be")
        print("     erased essentially completely and cross-domain transfer does")
        print("     NOT improve. Alignment is not merely hard -- it is IRRELEVANT.")
        res["verdict_transfer"] = "DECOUPLED_flat"
    elif d_tr > 0:
        print("  => DEAD END RE-OPENS: erasing domain identity improved transfer.")
        res["verdict_transfer"] = "IMPROVED"
    else:
        print("  => Erasing domain identity HURT transfer.")
        res["verdict_transfer"] = "DEGRADED"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"inlp_controls_{args.run}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
