"""What ARE the directions whose removal destroys cross-domain transfer?

`tradeoff_frontier.py` established that removing ~90 linear directions drives
domain separability to chance (C2ST 1.0000 -> 0.5001) while costing 21 AUC
points of MedalCare->PTB-XL transfer (0.7678 -> 0.5606) and essentially nothing
in-domain (0.9754 -> 0.9721). A rank-matched random control leaves both
untouched, so the directions are specific.

That is a correlational claim: "some subspace is load-bearing". This script
tries to NAME it, which is what turns the frontier into a mechanism.

Method
------
Project each sample onto the k INLP domain directions to get its coordinates in
the removed subspace, then ask how much of each interpretable ECG feature that
subspace explains -- compared against a rank-matched RANDOM subspace, which is
the only way to tell "this subspace encodes QRS duration" from "any 90-dim
projection of a 1024-d space explains a lot of everything".

Reported per feature:
  * R^2 of ridge regression from the k domain coordinates  (in-domain, held-out)
  * the same from k random coordinates                     (the control)
  * excess = domain R^2 - random R^2                       (the actual signal)

Features (`data/ecg_features_*.npz`, 6 clinical measurements from neurokit2):
QRS_duration_ms, QT_interval_ms, P_duration_ms, ST_J60_avg_mV, T_amplitude_mV,
heart_rate_bpm.

IMPORTANT -- the nk2_ok mask. neurokit2 fails to produce usable delineations on
most rows (ok_frac ~0.29 MedalCare, ~0.18 PTB-XL). Rows where extraction failed
carry garbage, so every fit below is restricted to nk2_ok rows. Ignoring the
mask silently regresses against parse failures.

Writes: outputs/analysis/domain_signal/subspace_identity_<run>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
from analysis.domain_signal_structure import (  # noqa: E402
    OUT_DIR, SEED, load,
)
from analysis.inlp_controls import inlp_directions  # noqa: E402

DATA = REPO_ROOT / "data"


def cv_r2(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> float:
    """5-fold CV R^2 of ridge regression, scaler fit per fold-train.

    Returns the mean across folds. Negative values (worse than predicting the
    mean) are kept rather than clipped -- they are informative.
    """
    if len(y) < 40:
        return float("nan")
    scores = []
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(X[tr]), y[tr])
        pred = m.predict(sc.transform(X[te]))
        ss_res = float(np.sum((y[te] - pred) ** 2))
        ss_tot = float(np.sum((y[te] - y[te].mean()) ** 2))
        scores.append(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)
    return float(np.nanmean(scores))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="exp7")
    ap.add_argument("--k", type=int, default=90)
    ap.add_argument("--subsample", type=int, default=1500)
    ap.add_argument("--random-reps", type=int, default=3)
    args = ap.parse_args()

    print("=" * 84)
    print(f"What do the domain directions encode?  [{args.run}]  k={args.k}")
    print("=" * 84)

    Z_med_tr, _ = load(args.run, "medalcare", "train")
    Z_med_te, _ = load(args.run, "medalcare", "test")
    Z_ptb_tr, _ = load(args.run, "ptbxl", "train")

    sc = StandardScaler().fit(np.vstack([Z_med_tr, Z_ptb_tr]))
    Z_med_tr_s = sc.transform(Z_med_tr)
    Z_ptb_tr_s = sc.transform(Z_ptb_tr)
    Z_med_te_s = sc.transform(Z_med_te)

    rng = np.random.default_rng(SEED)

    def sub(Z, n=args.subsample):
        if len(Z) <= n:
            return Z
        return Z[rng.choice(len(Z), n, replace=False)]

    print(f"fitting {args.k} INLP domain directions on TRAIN ...")
    W = inlp_directions(sub(Z_med_tr_s), sub(Z_ptb_tr_s), args.k)
    print(f"  got {len(W)}")
    Q, _ = np.linalg.qr(W.T)                      # D x k orthonormal basis

    feats = np.load(DATA / "ecg_features_medalcare_test.npz", allow_pickle=True)
    F = feats["features"].astype(np.float64)
    names = [str(x) for x in feats["feature_names"]]
    ok = feats["nk2_ok"].astype(bool)
    if len(F) != len(Z_med_te_s):
        print(f"[WARN] feature rows {len(F)} != latent rows {len(Z_med_te_s)}; aborting")
        return 1

    # Coordinates in the removed subspace vs a rank-matched random subspace.
    C_dom = Z_med_te_s @ Q
    D = Z_med_te_s.shape[1]

    print(f"\nrows: {int(ok.sum())} of {len(ok)} usable (nk2_ok)")
    print(f"\n{'feature':<20} {'domain R2':>10} {'random R2':>10} {'excess':>9}")
    print("-" * 52)

    rows = {}
    for j, nm in enumerate(names):
        y = F[:, j]
        m = ok & np.isfinite(y)
        if m.sum() < 40:
            print(f"{nm:<20} {'--':>10} {'--':>10} {'--':>9}  (n={int(m.sum())})")
            continue
        r2_dom = cv_r2(C_dom[m], y[m])

        r2_rand = []
        for rep in range(args.random_reps):
            g = np.random.default_rng(SEED + rep)
            R = g.normal(size=(D, len(W)))
            Qr, _ = np.linalg.qr(R)
            r2_rand.append(cv_r2(Z_med_te_s[m] @ Qr, y[m]))
        r2_r = float(np.nanmean(r2_rand))
        excess = r2_dom - r2_r
        rows[nm] = {"domain_r2": r2_dom, "random_r2": r2_r, "excess": excess,
                    "n": int(m.sum())}
        flag = "  <==" if excess > 0.05 else ""
        print(f"{nm:<20} {r2_dom:>10.4f} {r2_r:>10.4f} {excess:>9.4f}{flag}")

    print("\n(excess > 0 means the domain subspace encodes the feature MORE than")
    print(" an arbitrary subspace of the same rank does.)")

    ranked = sorted(rows.items(), key=lambda kv: -kv[1]["excess"])
    if ranked:
        top, tv = ranked[0]
        print(f"\nstrongest: {top}  (excess R2 = {tv['excess']:+.4f})")
        if tv["excess"] < 0.05:
            print("  => No single clinical feature explains the subspace. The")
            print("     domain signal is NOT reducible to these 6 measurements --")
            print("     it is something more distributed (waveform morphology,")
            print("     noise texture, or simulator artefacts).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"subspace_identity_{args.run}.json"
    out.write_text(json.dumps(
        {"run": args.run, "k": int(len(W)), "features": rows}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
