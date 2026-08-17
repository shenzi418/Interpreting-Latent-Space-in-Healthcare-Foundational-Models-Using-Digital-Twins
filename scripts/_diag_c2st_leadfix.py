"""DIAGNOSTIC (no retrain): does the lead-order fix move the C2ST alignment wall?

The project's headline negative result is "C2ST AUROC stays ~1.0 on held-out
latents no matter what alignment method is applied" (MMD, ccMMD, INLP,
bottleneck). If that wall was partly *caused* by the aVL/aVF permutation on the
MedalCare side (see reports/2026-08-10_lead_order_bug_diagnostic.md), then the
dead-end conclusion is itself an artifact and must be re-opened.

This needs no retraining: `_diag_leadswap_ptbxl.py --export-medalcare-unswapped`
already produced MedalCare latents under the TRUE standard lead order from the
same frozen `exp7_baseline` checkpoint.

Protocol notes (deliberately stricter than `exp7_analysis.domain_classifier_auc`,
which fits its StandardScaler on the pooled train+test folds -> mildly inflated
C2ST):
  * StandardScaler is fit on the fold-train rows only.
  * Two C2ST variants are reported:
      cv      - 5-fold stratified CV on the pooled TEST latents.
      heldout - fit on (MedalCare train, PTB-XL train), score on the test pools.
                This is the protocol the "~1.0 on held-out latents" claim used.
  * Classes are balanced by subsampling the larger domain (seeded).
  * MMD is the UNBIASED multi-bandwidth estimator (matches dim_scan.py:182-210,
    NOT the biased one in eval_tier2.py:69-86).

Writes only:
    outputs/analysis/leadswap_diag/c2st_leadfix.json

Usage
-----
    python scripts/_diag_c2st_leadfix.py
    python scripts/_diag_c2st_leadfix.py --runs exp8_leadfix_baseline exp8_leadfix_ccmmd
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

LATENT_DIR = REPO_ROOT / "outputs" / "latents"
OUT_DIR = REPO_ROOT / "outputs" / "analysis" / "leadswap_diag"
SEED = 42


def L(name: str) -> np.ndarray:
    return np.load(LATENT_DIR / name / "latents.npz",
                   allow_pickle=True)["Z"].astype(np.float64)


def balance(A: np.ndarray, B: np.ndarray, rng: np.random.Generator):
    n = min(len(A), len(B))
    ia = rng.choice(len(A), n, replace=False) if len(A) > n else np.arange(n)
    ib = rng.choice(len(B), n, replace=False) if len(B) > n else np.arange(n)
    return A[ia], B[ib]


def c2st_cv(A: np.ndarray, B: np.ndarray, seed: int = SEED) -> float:
    """5-fold stratified CV domain-classifier AUROC. Scaler fit per fold-train."""
    X = np.vstack([A, B])
    y = np.concatenate([np.zeros(len(A)), np.ones(len(B))])
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, C=1.0, random_state=seed)
        clf.fit(sc.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.mean(aucs))


def c2st_heldout(A_tr, B_tr, A_te, B_te, seed: int = SEED) -> float:
    """Fit the domain classifier on train pools, score on untouched test pools."""
    Xtr = np.vstack([A_tr, B_tr])
    ytr = np.concatenate([np.zeros(len(A_tr)), np.ones(len(B_tr))])
    Xte = np.vstack([A_te, B_te])
    yte = np.concatenate([np.zeros(len(A_te)), np.ones(len(B_te))])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced",
                             random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    return float(roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1]))


def mmd_multibandwidth(X: np.ndarray, Y: np.ndarray,
                       mults=(0.25, 0.5, 1.0, 2.0, 4.0)) -> float:
    """Unbiased multi-bandwidth RBF MMD^2 (median heuristic)."""
    Z = np.vstack([X, Y])
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x.y  -- the broadcast form needs 68 GiB.
    sq = np.einsum("ij,ij->i", Z, Z)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    np.maximum(d2, 0.0, out=d2)
    med = np.median(d2[d2 > 0])
    n, m = len(X), len(Y)
    total = 0.0
    for mu in mults:
        K = np.exp(-d2 / (mu * med))
        Kxx, Kyy, Kxy = K[:n, :n], K[n:, n:], K[:n, n:]
        total += ((Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
                  + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
                  - 2.0 * Kxy.mean())
    return float(total / len(mults))


def knn_mixing(X: np.ndarray, Y: np.ndarray, k: int = 10) -> float:
    """Fraction of each point's k-NN that come from the OTHER domain. 0.5 = mixed."""
    from sklearn.neighbors import NearestNeighbors
    Z = np.vstack([X, Y])
    dom = np.concatenate([np.zeros(len(X)), np.ones(len(Y))])
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, idx = nn.kneighbors(Z)
    other = (dom[idx[:, 1:]] != dom[:, None]).mean()
    return float(other)


# Each entry is (label, medalcare_train_dir, medalcare_test_dir,
#                ptbxl_train_dir, ptbxl_test_dir).
# The three defaults all share the exp7 PTB-XL export because they are
# re-exports from the SAME frozen exp7_baseline checkpoint -- only the MedalCare
# side changes. The exp8 runs each trained their own encoder, so their PTB-XL
# latents differ per run and must be paired accordingly (see `exp8_configs`).
CONFIGS = [
    ("as_shipped        (MedalCare aVL/aVF SWAPPED, global z)",
     "exp7_medalcare_train", "exp7_medalcare", "exp7_ptbxl_train", "exp7_ptbxl"),
    ("leadfix           (MedalCare CORRECT order, global z)",
     "exp7_medalcare_train_unswapped", "exp7_medalcare_unswapped",
     "exp7_ptbxl_train", "exp7_ptbxl"),
    ("leadfix + perlead (MedalCare CORRECT order, per-lead z)",
     "exp7_medalcare_train_unswapped_perlead", "exp7_medalcare_unswapped_perlead",
     "exp7_ptbxl_train", "exp7_ptbxl"),
]


def exp8_configs(run_ids):
    """Config tuples for Stage-3 retrained runs.

    The three default CONFIGS re-export a frozen exp7 checkpoint under different
    lead orders -- they answer "does the *representation* change when you feed it
    correct leads". These answer the harder question: does a model **trained**
    from the start on correct leads land somewhere different? That is the actual
    decision point in §5 of the audit, and it needs each run's own PTB-XL export.
    """
    return [
        (f"{rid:<50s}(trained on correct leads)",
         f"{rid}_medalcare_train", f"{rid}_medalcare_test",
         f"{rid}_ptbxl_train", f"{rid}_ptbxl_test")
        for rid in run_ids
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--runs", nargs="*", default=None,
        help="Stage-3 run IDs (e.g. exp8_leadfix_baseline) to measure INSTEAD of "
             "the default frozen-checkpoint re-export comparison. Each run must "
             "have its own <run>_{medalcare,ptbxl}_{train,test} exports.",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Output JSON. Default: c2st_leadfix.json for the built-in configs, "
             "c2st_leadfix_trained.json when --runs is given (so the two never "
             "overwrite each other).",
    )
    args = ap.parse_args()

    if args.runs:
        configs = exp8_configs(args.runs)
        out_path = args.out or (OUT_DIR / "c2st_leadfix_trained.json")
    else:
        configs = CONFIGS
        out_path = args.out or (OUT_DIR / "c2st_leadfix.json")

    results = {}

    print(f"{'config':<58} {'C2ST cv':>9} {'C2ST held':>10} "
          f"{'MMD':>10} {'kNN mix':>9}")
    print("-" * 100)

    for label, med_tr_name, med_te_name, ptb_tr_name, ptb_te_name in configs:
        try:
            Z_med_tr, Z_med_te = L(med_tr_name), L(med_te_name)
            Z_ptb_tr, Z_ptb_te = L(ptb_tr_name), L(ptb_te_name)
        except FileNotFoundError as exc:
            print(f"{label:<58} SKIP -- missing export: {exc}")
            continue

        a_te, b_te = balance(Z_med_te, Z_ptb_te, np.random.default_rng(SEED))
        a_tr, b_tr = balance(Z_med_tr, Z_ptb_tr, np.random.default_rng(SEED))

        cv = c2st_cv(a_te, b_te)
        ho = c2st_heldout(a_tr, b_tr, a_te, b_te)

        # MMD / kNN on a common subsample (O(n^2) memory)
        sub = 1500
        r2 = np.random.default_rng(SEED)
        ai = r2.choice(len(a_te), min(sub, len(a_te)), replace=False)
        bi = r2.choice(len(b_te), min(sub, len(b_te)), replace=False)
        mmd = mmd_multibandwidth(a_te[ai], b_te[bi])
        knn = knn_mixing(a_te[ai], b_te[bi])

        results[label.split()[0]] = {
            "label": label, "medalcare_train": med_tr_name,
            "medalcare_test": med_te_name,
            "ptbxl_train": ptb_tr_name, "ptbxl_test": ptb_te_name,
            "n_medalcare_test": int(len(a_te)), "n_ptbxl_test": int(len(b_te)),
            "c2st_auroc_cv": cv, "c2st_auroc_heldout": ho,
            "mmd2_multibandwidth_unbiased": mmd, "knn_mixing_k10": knn,
        }
        print(f"{label:<58} {cv:>9.4f} {ho:>10.4f} {mmd:>10.5f} {knn:>9.4f}")

    print("\n(kNN mixing: 0.5 = perfectly mixed domains, 0.0 = fully separated)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
