"""Does the transfer-critical subspace encode INFARCT BIOPHYSICS?

The decisive follow-up to `subspace_identity.py`.

Established so far:
  * Removing ~90 INLP directions drives C2ST 1.0000 -> 0.5001 and costs 21 AUC
    points of MedalCare->PTB-XL transfer, at ~zero in-domain cost.
  * Those directions are NOT reducible to the 6 neurokit2 clinical features --
    every excess R^2 was negative, even though the subspace holds 1.85x the
    variance of a random subspace of equal rank.

So what IS in there? The quantity this thesis actually cares about is the
biophysical theta: infarct position (phi, z), extent (size), severity
(rho_eps_max), and the derived anatomical territory. If the transfer-critical
subspace aligns with theta, the frontier stops being an abstract ML result and
becomes a biophysical one:

    "Distributional alignment destroys precisely the directions that encode
     infarct geometry."

That is the strongest available framing, so it deserves the strictest controls.

Design
------
theta is defined only for MI rows (n=1200 of 2386 test), indexed by
`idx_in_split`. For each target we compare, on those rows:
  * R^2 / AUC from the k domain-direction coordinates
  * the same from a rank-matched RANDOM subspace  (the A4 control)
  * excess = domain - random

phi is circular, so it is handled as (sin, cos) with a circular R^2, never as a
raw angle -- regressing an angle linearly puts a discontinuity at the wrap point
and manufactures a spurious result.

territory_4c is categorical -> macro-OvR AUC from multinomial logistic
regression.

Writes: outputs/analysis/domain_signal/subspace_theta_<run>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
from analysis.domain_signal_structure import OUT_DIR, SEED, load  # noqa: E402
from analysis.inlp_controls import inlp_directions  # noqa: E402

DATA = REPO_ROOT / "data"


def cv_r2(X, y, seed=SEED) -> float:
    scores = []
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(X[tr]), y[tr])
        p = m.predict(sc.transform(X[te]))
        ss_res = float(np.sum((y[te] - p) ** 2))
        ss_tot = float(np.sum((y[te] - y[te].mean()) ** 2))
        scores.append(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)
    return float(np.nanmean(scores))


def cv_circular_r2(X, phi, seed=SEED) -> float:
    """Circular R^2 for an angular target.

    Predict sin and cos separately, recombine with atan2, and score residual
    angular error against the variance around the TRAIN circular mean. Centring
    on the test mean would be a divergent estimator (defect M6 in the audit).
    """
    s, c = np.sin(phi), np.cos(phi)
    scores = []
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        ps = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xtr, s[tr]).predict(Xte)
        pc = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xtr, c[tr]).predict(Xte)
        pred = np.arctan2(ps, pc)
        err = np.arctan2(np.sin(phi[te] - pred), np.cos(phi[te] - pred))
        mu_tr = np.arctan2(np.mean(s[tr]), np.mean(c[tr]))
        base = np.arctan2(np.sin(phi[te] - mu_tr), np.cos(phi[te] - mu_tr))
        ss_res, ss_tot = float(np.sum(err ** 2)), float(np.sum(base ** 2))
        scores.append(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)
    return float(np.nanmean(scores))


def cv_macro_auc(X, y, seed=SEED) -> float:
    """Macro one-vs-rest AUC via multinomial logistic regression, 5-fold CV.

    Probability columns are indexed through clf.classes_ rather than assumed to
    be in label order -- sklearn sorts them, and assuming otherwise silently
    transposes columns (defect A1 in the audit).
    """
    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan")
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                                 random_state=seed).fit(sc.transform(X[tr]), y[tr])
        proba = clf.predict_proba(sc.transform(X[te]))
        per = []
        for ci, cls in enumerate(clf.classes_):
            yb = (y[te] == cls).astype(int)
            if len(np.unique(yb)) < 2:
                continue
            per.append(roc_auc_score(yb, proba[:, ci]))
        if per:
            aucs.append(float(np.mean(per)))
    return float(np.nanmean(aucs)) if aucs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="exp7")
    ap.add_argument("--k", type=int, default=90)
    ap.add_argument("--subsample", type=int, default=1500)
    ap.add_argument("--random-reps", type=int, default=3)
    args = ap.parse_args()

    print("=" * 84)
    print(f"Does the transfer-critical subspace encode theta?  [{args.run}] k={args.k}")
    print("=" * 84)

    Z_med_tr, _ = load(args.run, "medalcare", "train")
    Z_med_te, _ = load(args.run, "medalcare", "test")
    Z_ptb_tr, _ = load(args.run, "ptbxl", "train")

    sc = StandardScaler().fit(np.vstack([Z_med_tr, Z_ptb_tr]))
    Zm_s, Zp_s = sc.transform(Z_med_tr), sc.transform(Z_ptb_tr)
    Zte = sc.transform(Z_med_te)

    rng = np.random.default_rng(SEED)

    def sub(Z, n=args.subsample):
        return Z if len(Z) <= n else Z[rng.choice(len(Z), n, replace=False)]

    print(f"fitting {args.k} INLP domain directions on TRAIN ...")
    W = inlp_directions(sub(Zm_s), sub(Zp_s), args.k)
    Q, _ = np.linalg.qr(W.T)
    print(f"  got {len(W)}")

    th = np.load(DATA / "theta_mi_test.npz", allow_pickle=True)
    idx = th["idx_in_split"].astype(int)
    if idx.max() >= len(Zte):
        print(f"[WARN] theta idx max {idx.max()} >= latent rows {len(Zte)}; abort")
        return 1
    Zmi = Zte[idx]
    print(f"theta rows: {len(idx)} MI samples of {len(Zte)} test rows")

    C_dom = Zmi @ Q
    D = Zte.shape[1]
    rand_Q = []
    for rep in range(args.random_reps):
        g = np.random.default_rng(SEED + rep)
        Qr, _ = np.linalg.qr(g.normal(size=(D, len(W))))
        rand_Q.append(Zmi @ Qr)

    targets = [
        ("phi (circular)", "circ", th["phi"].astype(float)),
        ("z", "r2", th["z"].astype(float)),
        ("size", "r2", th["size"].astype(float)),
        ("rho_eps_max", "r2", th["rho_eps_max"].astype(float)),
        ("transmural", "r2", th["transmural"].astype(float)),
        ("territory_4c", "auc", th["territory_4c"]),
    ]

    print(f"\n{'theta target':<20} {'metric':<6} {'domain':>9} {'random':>9} {'excess':>9}")
    print("-" * 57)
    rows = {}
    for name, kind, y in targets:
        if kind == "circ":
            d_sc = cv_circular_r2(C_dom, y)
            r_sc = float(np.nanmean([cv_circular_r2(Cr, y) for Cr in rand_Q]))
            mlabel = "circR2"
        elif kind == "r2":
            m = np.isfinite(y)
            d_sc = cv_r2(C_dom[m], y[m])
            r_sc = float(np.nanmean([cv_r2(Cr[m], y[m]) for Cr in rand_Q]))
            mlabel = "R2"
        else:
            ys = np.asarray([str(v) for v in y])
            d_sc = cv_macro_auc(C_dom, ys)
            r_sc = float(np.nanmean([cv_macro_auc(Cr, ys) for Cr in rand_Q]))
            mlabel = "AUC"
        exc = d_sc - r_sc
        rows[name] = {"metric": mlabel, "domain": d_sc, "random": r_sc,
                      "excess": exc}
        flag = "  <==" if exc > 0.05 else ""
        print(f"{name:<20} {mlabel:<6} {d_sc:>9.4f} {r_sc:>9.4f} {exc:>9.4f}{flag}")

    print("\n(excess > 0: the transfer-critical subspace encodes this theta")
    print(" parameter MORE than an arbitrary subspace of equal rank.)")

    best = max(rows.items(), key=lambda kv: kv[1]["excess"])
    print(f"\nstrongest: {best[0]}  (excess = {best[1]['excess']:+.4f})")
    if best[1]["excess"] > 0.05:
        print("  => BIOPHYSICAL: alignment destroys directions that encode infarct")
        print("     geometry. This is the thesis's strongest framing.")
    else:
        print("  => NOT biophysical either. The transfer-critical subspace is")
        print("     depleted of BOTH clinical features and theta. It carries")
        print("     domain identity and generic transferable structure, but not")
        print("     the interpretable quantities -- which sharpens the negative")
        print("     result: theta decoding survives alignment; transfer does not.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"subspace_theta_{args.run}.json"
    out.write_text(json.dumps({"run": args.run, "k": int(len(W)),
                               "n_mi_rows": int(len(idx)), "targets": rows},
                              indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
