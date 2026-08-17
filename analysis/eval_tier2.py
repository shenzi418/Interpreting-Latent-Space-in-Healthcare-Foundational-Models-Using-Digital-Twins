"""Tier 2 evaluation: L1 (alignment) + L2 (mechanism) + L3 (cross-domain transfer)
for a multi-task bottleneck run.

Reuses the helpers in ``analysis/eval_decoding_lowK.py`` so the metrics match the
Tier 1 sweep one-for-one, but lets you point at an arbitrary run prefix instead
of the hardcoded ``exp7_bottleneck_K{K}_*`` convention.

Usage::

    python analysis/eval_tier2.py \
        --run-prefix exp7_tier2_K64_A_5050 \
        --label "Tier2 A (50/50)" \
        --out outputs/inlp_lowK/tier2_A_5050.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
from analysis.eval_decoding_lowK import (  # noqa: E402
    eval_l2_probes,
    eval_pipeline_a,
    load_theta_targets,
    load_ptbxl_primary_4c,
    SEED,
)

LATENT_DIR = REPO_ROOT / "outputs" / "latents"

# m6 fix: this file and analysis/exp7_analysis.py both emit keys named
# "mmd_rbf" and "knn_mixing", but they are DIFFERENT ESTIMATORS. Neither is
# wrong; they are simply not comparable, and a table that puts a number from
# one next to a number from the other is meaningless. Every payload written by
# either script now carries its own spec so the mismatch travels with the
# numbers instead of being lost at the point a table is assembled.
METRIC_SPEC: Dict[str, object] = {
    "c2st_auroc": {
        "estimator": "LogisticRegression(class_weight='balanced', max_iter=2000)",
        "cv": "StratifiedKFold(3, shuffle=True)",
        "scaler": "none (raw latents)",
    },
    "mmd_rbf": {
        "estimator": "BIASED MMD^2 (diagonal retained: kxx.mean() + kyy.mean() - 2*kxy.mean())",
        "bandwidth": "median heuristic on pooled squared distances, k = exp(-d2 / (2*sigma2))",
        "subsample": 1024,
        "not_comparable_with": "analysis/exp7_analysis.py:mmd_rbf (UNBIASED, no subsampling, gamma=1/(2*sigma^2) with sigma^2 = median/2)",
    },
    "knn_mixing": {
        "k": 10,
        "metric": "euclidean",
        "subsample": 1024,
        "not_comparable_with": "analysis/exp7_analysis.py:knn_mixing_score (k=15, cosine, no subsampling)",
    },
}


def load_Z_from_prefix(prefix: str, domain: str, split: str) -> np.ndarray:
    p = LATENT_DIR / f"{prefix}_{domain}_{split}" / "latents.npz"
    return np.load(p, allow_pickle=True)["Z"].astype(np.float64)


# ---- L1 alignment metrics (lightweight; reuses standard tools) -------------

def _c2st_auroc(Z_a: np.ndarray, Z_b: np.ndarray, *, seed: int = SEED) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    Z = np.vstack([Z_a, Z_b])
    y = np.concatenate([np.zeros(len(Z_a)), np.ones(len(Z_b))]).astype(np.int64)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(Z, y):
        est = LogisticRegression(
            max_iter=2000, solver="lbfgs", class_weight="balanced",
        ).fit(Z[tr], y[tr])
        s = est.predict_proba(Z[te])[:, 1]
        aucs.append(roc_auc_score(y[te], s))
    return float(np.mean(aucs))


def _mmd_rbf(X: np.ndarray, Y: np.ndarray, *, sub: int = 1024,
             seed: int = SEED) -> float:
    rng = np.random.default_rng(seed)
    if len(X) > sub:
        X = X[rng.choice(len(X), sub, replace=False)]
    if len(Y) > sub:
        Y = Y[rng.choice(len(Y), sub, replace=False)]
    Z = np.vstack([X, Y])
    d2 = (Z[None, :, :] - Z[:, None, :]) ** 2
    d2 = d2.sum(-1)
    sigma2 = float(np.median(d2[d2 > 0])) if (d2 > 0).any() else 1.0
    def k(A, B):
        d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-d / (2.0 * sigma2))
    kxx = k(X, X).mean()
    kyy = k(Y, Y).mean()
    kxy = k(X, Y).mean()
    return float(kxx + kyy - 2.0 * kxy)


def _knn_mixing(X: np.ndarray, Y: np.ndarray, *, k: int = 10,
                sub: int = 1024, seed: int = SEED) -> float:
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    if len(X) > sub:
        X = X[rng.choice(len(X), sub, replace=False)]
    if len(Y) > sub:
        Y = Y[rng.choice(len(Y), sub, replace=False)]
    Z = np.vstack([X, Y])
    labels = np.concatenate([np.zeros(len(X)), np.ones(len(Y))]).astype(np.int64)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, idx = nn.kneighbors(Z)
    neighbor_labels = labels[idx[:, 1:]]
    # Mixing = fraction of neighbours from the OTHER domain
    self_labels = labels[:, None]
    other_frac = (neighbor_labels != self_labels).mean()
    return float(other_frac)


def alignment_block(Z_med_pool: np.ndarray, Z_ptb_pool: np.ndarray,
                    Z_med_test: np.ndarray, Z_ptb_test: np.ndarray
                    ) -> Dict[str, object]:
    """Two alignment views.

    ``pool`` = the fitting-side distributions (MedalCare-train vs PTB-XL-train).
    ``test`` = the held-out distributions (MedalCare-test vs PTB-XL-test).

    m5 fix: main() previously passed ``Z_ptb_te`` as BOTH ``Z_ptb_pool`` and
    ``Z_ptb_test``, so the two rows of this block differed only in the
    MedalCare leg and the "pool vs test" contrast was not the contrast the
    table claimed. The caller now supplies PTB-XL train for the pool leg.
    """
    if Z_ptb_pool.shape == Z_ptb_test.shape and np.array_equal(Z_ptb_pool, Z_ptb_test):
        raise ValueError(
            "alignment_block: the PTB-XL pool and test legs are the same array. "
            "The 'pool' and 'test' rows would differ only in the MedalCare leg, "
            "which is not what those labels mean. Pass PTB-XL train as the pool leg."
        )
    return {
        "pool": {
            "c2st_auroc": _c2st_auroc(Z_med_pool, Z_ptb_pool),
            "mmd_rbf": _mmd_rbf(Z_med_pool, Z_ptb_pool),
            "knn_mixing": _knn_mixing(Z_med_pool, Z_ptb_pool),
        },
        "test": {
            "c2st_auroc": _c2st_auroc(Z_med_test, Z_ptb_test),
            "mmd_rbf": _mmd_rbf(Z_med_test, Z_ptb_test),
            "knn_mixing": _knn_mixing(Z_med_test, Z_ptb_test),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-prefix", type=str, required=True,
                    help="Prefix used when latents were exported, e.g. "
                         "'exp7_tier2_K64_A_5050'. Looks for "
                         "outputs/latents/<prefix>_{medalcare,ptbxl}_{train,val,test}.")
    ap.add_argument("--label", type=str, default=None,
                    help="Short label for the JSON output.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] prefix = {args.run_prefix}")
    Z_med_tr = load_Z_from_prefix(args.run_prefix, "medalcare", "train")
    Z_med_te = load_Z_from_prefix(args.run_prefix, "medalcare", "test")
    Z_ptb_tr = load_Z_from_prefix(args.run_prefix, "ptbxl", "train")
    Z_ptb_te = load_Z_from_prefix(args.run_prefix, "ptbxl", "test")
    print(f"       med_train={Z_med_tr.shape}  med_test={Z_med_te.shape}  "
          f"ptb_train={Z_ptb_tr.shape}  ptb_test={Z_ptb_te.shape}")

    targets = load_theta_targets()
    ptbxl_idx, ptbxl_truth = load_ptbxl_primary_4c()
    print(f"[load] PTB-XL primary 4c n={ptbxl_idx.size}  "
          f"per_class={dict(pd.Series(ptbxl_truth).value_counts())}")

    idx_tr = targets["train"]["idx_in_split"]
    idx_te = targets["test"]["idx_in_split"]
    Z_med_tr_mi = Z_med_tr[idx_tr]
    Z_med_te_mi = Z_med_te[idx_te]
    Z_ptb_primary = Z_ptb_te[ptbxl_idx]

    # L1 alignment (use FULL distributions, MI-subset is too small for stable MMD)
    print("[L1] alignment (full distributions)")
    alignment = alignment_block(Z_med_tr, Z_ptb_tr, Z_med_te, Z_ptb_te)
    print(f"     pool: c2st={alignment['pool']['c2st_auroc']:.3f}  "
          f"mmd={alignment['pool']['mmd_rbf']:.3f}  "
          f"knn_mix={alignment['pool']['knn_mixing']:.3f}")

    # L2 probes on the K-d latent (refit per-config)
    print("[L2] in-domain mechanism probes")
    probes = eval_l2_probes(
        Z_med_tr_mi, Z_med_te_mi, targets["train"], targets["test"],
    )
    print(f"     phi R²_circ={probes['phi']['r2_circular']:.3f}  "
          f"z R²={probes['z']['r2']:.3f}  size R²={probes['size']['r2']:.3f}  "
          f"trans AUC={probes['transmurality']['auc']}")

    # L3 Pipeline A
    print("[L3] Pipeline A territory transfer")
    rng = np.random.default_rng(SEED)
    y_train_4c = np.asarray(targets["train"]["territory_4c"].tolist(), dtype=object)
    y_test_4c = np.asarray(targets["test"]["territory_4c"].tolist(), dtype=object)
    pa = eval_pipeline_a(
        Z_med_tr_mi, Z_med_te_mi, Z_ptb_primary,
        y_train_4c, y_test_4c, ptbxl_truth,
        rng=rng,
    )
    id_ = pa["in_domain_4c"]; cd = pa["cross_domain_4c"]
    print(f"     in_dom: F1={id_['macro_f1']:.3f}  AUC={id_['macro_auc_ovr']}")
    print(f"     CD:     F1={cd['macro_f1']:.3f}  CI={cd['macro_f1_ci95']}  "
          f"p_f1={cd['permutation_p_macro_f1']:.4f}  "
          f"AUC={cd['macro_auc_ovr']}  CI={cd['macro_auc_ovr_ci95']}  "
          f"p_auc={cd['permutation_p_macro_auc_ovr']}")

    payload = {
        "run_prefix": args.run_prefix,
        "label": args.label or args.run_prefix,
        "shapes": {
            "med_train": list(Z_med_tr.shape),
            "med_test": list(Z_med_te.shape),
            "ptb_train": list(Z_ptb_tr.shape),
            "ptb_test": list(Z_ptb_te.shape),
        },
        "alignment": alignment,
        "alignment_metric_spec": METRIC_SPEC,
        "probes": probes,
        "pipeline_a": pa,
    }
    args.out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    out_disp = args.out.resolve()
    try:
        out_disp = out_disp.relative_to(REPO_ROOT)
    except ValueError:
        pass
    print(f"\n[done] -> {out_disp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
