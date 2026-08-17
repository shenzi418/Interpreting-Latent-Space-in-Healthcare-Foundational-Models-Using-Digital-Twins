"""Task 1 evaluation at K ∈ {16, 64, 256, 1024} with and without INLP.

L2 (in-domain mechanism — MedalCare MI test):
    - phi  : RidgeCV on (sin, cos) → circular R²
    - z    : RidgeCV → R²
    - size : RidgeCV → R²
    - transmurality (binary {0.3, 1.0}) : LogReg manual-CV → ROC-AUC

L3 (cross-domain transfer outcome — PTB-XL primary 4c subset, n=438):
    Pipeline A : multinomial LogReg on Z_MedalCare_train_MI -> territory_4c
        - in-domain : MedalCare-test MI rows
        - cross-domain : PTB-XL primary 4c subset
    Metrics : macro-F1, balanced accuracy, macro-OvR ROC-AUC, per-class F1,
              confusion matrix, 1000-bootstrap CI, 1000-permutation p (Marta
              asked for AUC alongside F1).

All probes are refit per K, per condition (orig vs _inlp). Nothing is saved
to disk except the final JSON of metrics; this script does NOT modify any
latent files.

Outputs
-------
outputs/inlp_lowK/eval_decoding_lowK.json
    { K_str: { "orig": {...}, "inlp": {...} } }
    where each `...` block contains:
      "probes": {phi, z, size, transmurality}        (L2)
      "pipeline_a": {
          "in_domain_4c": {macro_f1, macro_auc_ovr, per_class, cm, ...}
          "cross_domain_4c": {macro_f1, macro_auc_ovr, per_class, cm, ...}
      }                                              (L3)
      "best_C": float

Pipeline A is also run for K=1024 (configs exp7, exp7_inlp) to satisfy the
"backfill K=1024 macro-OvR AUC on exp7_baseline_inlp" item.

Run::

    python analysis/eval_decoding_lowK.py --ks 16 64 256 1024
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
from sklearn.linear_model import LogisticRegression, RidgeCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    r2_score,
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# M6: single source of truth for the circular metrics.
from analysis.phase_b2_infarct_decoding import (  # noqa: E402
    circular_r2 as pb2_circular_r2,
    circular_mean as pb2_circular_mean,
    derive_rng,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LATENT_DIR = REPO_ROOT / "outputs" / "latents"
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "outputs" / "inlp_lowK"

THETA_TRAIN = DATA_DIR / "theta_mi_train.npz"
THETA_TEST = DATA_DIR / "theta_mi_test.npz"
PTBXL_CSV = DATA_DIR / "ptbxl_mi_subclass.csv"

TERRITORIES_4C = ["Anteroseptal", "Anterolateral", "Inferior", "Inferolateral"]
SEED = 42
N_BOOT = 1000
N_PERM = 1000
RIDGE_ALPHAS = np.logspace(-3, 6, 10)
LOGREG_CS_BIN = np.logspace(-3, 2, 6)         # transmurality binary
LOGREG_CS_TERR = np.logspace(-5, 2, 8)        # 4-class territory


# ---------------------------------------------------------------------------
# Latent loader  — handles BOTH naming conventions:
#   K=1024 (exp7_baseline): test split is "exp7_medalcare/" (no _test)
#                            train is        "exp7_medalcare_train/"
#                            inlp suffix     "exp7_medalcare_inlp/"
#   K∈{16,64,256}:          test split is   "exp7_bottleneck_K{K}_medalcare_test/"
#                            train is        "exp7_bottleneck_K{K}_medalcare_train/"
#                            inlp suffix     "exp7_bottleneck_K{K}_medalcare_test_inlp/"
# ---------------------------------------------------------------------------

def latent_dir_name(K: int, domain: str, split: str, suffix: str = "") -> str:
    """Return the directory name for a given (K, domain, split, suffix)."""
    if K == 1024:
        # exp7_baseline naming
        if split == "test":
            return f"exp7_{domain}{suffix}"
        else:
            return f"exp7_{domain}_{split}{suffix}"
    else:
        return f"exp7_bottleneck_K{K}_{domain}_{split}{suffix}"


def load_Z(K: int, domain: str, split: str, suffix: str = "") -> np.ndarray:
    p = LATENT_DIR / latent_dir_name(K, domain, split, suffix) / "latents.npz"
    return np.load(p, allow_pickle=True)["Z"].astype(np.float64)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def load_theta_targets() -> Dict[str, Dict[str, np.ndarray]]:
    tr = dict(np.load(THETA_TRAIN, allow_pickle=True))
    te = dict(np.load(THETA_TEST, allow_pickle=True))
    return {"train": tr, "test": te}


def load_ptbxl_primary_4c() -> Tuple[np.ndarray, np.ndarray]:
    """Return (row_idx, territory_4c) for the n=438 primary single-territory subset."""
    df = pd.read_csv(PTBXL_CSV)
    sub = df[df["territory_4c"].isin(TERRITORIES_4C)].copy()
    return sub["row_idx"].to_numpy(), sub["territory_4c"].to_numpy()


# ---------------------------------------------------------------------------
# B2 probes (L2)
# ---------------------------------------------------------------------------

def circular_r2(phi_true: np.ndarray, phi_pred: np.ndarray) -> float:
    """DEPRECATED -- do not use. Kept only so old callers fail loudly.

    FIXED 2026-08-10 (defect M6). This was a near-copy of the ``phase_b2``
    function whose docstring claimed "definition matches phase_b2" -- it did
    not. It centred the denominator on the circular mean of ``phi_true``
    (the **test** mean), whereas ``phase_b2.circular_r2`` centres on the
    **train** mean, which is the only baseline a real predictor could have used.
    Using the test mean gives the null baseline access to the test labels and
    inflates R²_circ. The argument order was also reversed relative to
    ``phase_b2`` (``(true, pred)`` vs ``(pred, true)``) -- harmless for this
    symmetric numerator, but exactly the kind of thing that silently swaps.

    Use ``phase_b2_infarct_decoding.circular_r2(phi_pred, phi_true,
    phi_train_mean)`` instead.
    """
    raise NotImplementedError(
        "analysis.eval_decoding_lowK.circular_r2 is removed (defect M6). "
        "Import circular_r2 from analysis.phase_b2_infarct_decoding and pass "
        "the TRAIN circular mean explicitly."
    )


def fit_phi_probe(
    X_train_std: np.ndarray, phi_train: np.ndarray,
) -> Tuple[RidgeCV, np.ndarray]:
    """Fit a RidgeCV on (sin, cos) targets."""
    Y = np.column_stack([np.sin(phi_train), np.cos(phi_train)])
    reg = RidgeCV(alphas=RIDGE_ALPHAS, scoring="r2").fit(X_train_std, Y)
    return reg, Y


def fit_scalar_probe(
    X_train_std: np.ndarray, y_train: np.ndarray,
) -> RidgeCV:
    return RidgeCV(alphas=RIDGE_ALPHAS, scoring="r2").fit(X_train_std, y_train)


def fit_binary_probe(
    X_train_std: np.ndarray, y_train_bin: np.ndarray,
) -> Tuple[LogisticRegression, float]:
    """LogReg with manual 5-fold CV over LOGREG_CS_BIN (scoring=roc_auc)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_C = None
    best_auc = -np.inf
    for C in LOGREG_CS_BIN:
        fold_aucs = []
        for tr, va in skf.split(X_train_std, y_train_bin):
            est = LogisticRegression(
                C=C, penalty="l2", solver="lbfgs",
                max_iter=2000, class_weight="balanced",
            )
            est.fit(X_train_std[tr], y_train_bin[tr])
            scores = est.predict_proba(X_train_std[va])[:, 1]
            fold_aucs.append(roc_auc_score(y_train_bin[va], scores))
        mu = float(np.mean(fold_aucs))
        if mu > best_auc:
            best_auc = mu
            best_C = float(C)
    model = LogisticRegression(
        C=best_C, penalty="l2", solver="lbfgs",
        max_iter=2000, class_weight="balanced",
    ).fit(X_train_std, y_train_bin)
    return model, best_C


def eval_l2_probes(
    Z_train_mi: np.ndarray,
    Z_test_mi: np.ndarray,
    theta_train: Dict[str, np.ndarray],
    theta_test: Dict[str, np.ndarray],
) -> Dict[str, Dict]:
    scaler = StandardScaler().fit(Z_train_mi)
    X_tr = scaler.transform(Z_train_mi)
    X_te = scaler.transform(Z_test_mi)

    out: Dict[str, Dict] = {}

    # phi (circular)
    phi_tr = theta_train["phi"]
    phi_te = theta_test["phi"]
    phi_model, _ = fit_phi_probe(X_tr, phi_tr)
    Y_te_pred = phi_model.predict(X_te)
    phi_te_pred = np.arctan2(Y_te_pred[:, 0], Y_te_pred[:, 1])
    # M6: use the canonical phase_b2 definition, centred on the TRAIN circular
    # mean (the only baseline a real predictor could have had).
    phi_train_mean = pb2_circular_mean(phi_tr)
    out["phi"] = {
        "r2_circular": pb2_circular_r2(phi_te_pred, phi_te, phi_train_mean),
        "best_alpha": float(phi_model.alpha_) if np.isscalar(phi_model.alpha_) else [
            float(a) for a in np.atleast_1d(phi_model.alpha_)
        ],
    }

    # z
    z_model = fit_scalar_probe(X_tr, theta_train["z"])
    z_pred = z_model.predict(X_te)
    out["z"] = {
        "r2": float(r2_score(theta_test["z"], z_pred)),
        "best_alpha": float(z_model.alpha_),
    }

    # size
    size_model = fit_scalar_probe(X_tr, theta_train["size"])
    size_pred = size_model.predict(X_te)
    out["size"] = {
        "r2": float(r2_score(theta_test["size"], size_pred)),
        "best_alpha": float(size_model.alpha_),
    }

    # transmurality (binary)
    trans_tr = (theta_train["transmural"] > 0.5).astype(np.int64)
    trans_te = (theta_test["transmural"] > 0.5).astype(np.int64)
    if len(np.unique(trans_te)) < 2:
        out["transmurality"] = {"auc": None, "best_C": None,
                                 "note": "single-class test set"}
    else:
        tr_model, best_C = fit_binary_probe(X_tr, trans_tr)
        scores = tr_model.predict_proba(X_te)[:, 1]
        out["transmurality"] = {
            "auc": float(roc_auc_score(trans_te, scores)),
            "best_C": float(best_C),
        }

    return out


# ---------------------------------------------------------------------------
# Pipeline A — L3
# ---------------------------------------------------------------------------

def macro_ovr_auc(
    y_true_str: np.ndarray, proba: np.ndarray, labels: List[str],
    proba_labels: Optional[List[str]] = None,
) -> Optional[float]:
    """Macro-averaged one-vs-rest ROC-AUC.

    ``proba_labels`` gives the class each COLUMN of ``proba`` corresponds to,
    i.e. ``list(model.classes_)``. It is required whenever that order differs
    from ``labels``.

    FIXED 2026-08-10 (defect A1). This function used to encode ``y_true`` by
    position in ``labels`` (= TERRITORIES_4C, anatomical order) while ``proba``
    columns follow sklearn's ``model.classes_``, which is sorted
    ALPHABETICALLY:

        TERRITORIES_4C : [Anteroseptal, Anterolateral, Inferior, Inferolateral]
        model.classes_ : [Anterolateral, Anteroseptal, Inferior, Inferolateral]

    Columns 0 and 1 are transposed, so Anteroseptal was scored with
    P(Anterolateral) and vice versa. Every stored ``macro_auc_ovr``,
    ``macro_auc_ovr_ci95`` and ``permutation_p_macro_auc_ovr`` produced before
    this date is affected. The bug UNDER-states in-domain performance (K=1024
    in-domain: 0.578 as coded vs 0.782 correct).
    """
    if proba_labels is not None:
        if list(proba_labels) != list(labels):
            missing = [l for l in labels if l not in list(proba_labels)]
            if missing:
                raise ValueError(
                    f"proba has no column for label(s) {missing}; "
                    f"proba_labels={list(proba_labels)}, labels={list(labels)}"
                )
            order = [list(proba_labels).index(l) for l in labels]
            proba = proba[:, order]
    try:
        # y_true is encoded by position in `labels`; `proba` columns are now
        # guaranteed to be in that same order.
        label_to_idx = {l: i for i, l in enumerate(labels)}
        y_idx = np.asarray([label_to_idx[t] for t in y_true_str], dtype=np.int64)
        # roc_auc_score in 'ovr' macro mode requires all classes present in y_true.
        present = np.unique(y_idx)
        if len(present) < 2:
            return None
        if len(present) < len(labels):
            # Subset the proba columns to the present classes and RENORMALISE —
            # roc_auc_score(multi_class='ovr') requires rows summing to 1.
            # (The pre-2026-08-10 code sliced without renormalising.)
            proba_sub = proba[:, present]
            row = proba_sub.sum(axis=1, keepdims=True)
            proba_sub = proba_sub / np.clip(row, 1e-12, None)
            remap = {v: i for i, v in enumerate(present)}
            y_sub = np.asarray([remap[v] for v in y_idx], dtype=np.int64)
            return float(roc_auc_score(
                y_sub, proba_sub, multi_class="ovr", average="macro",
                labels=list(range(len(present))),
            ))
        return float(roc_auc_score(
            y_idx, proba, multi_class="ovr", average="macro",
            labels=list(range(len(labels))),
        ))
    except Exception as exc:
        # Deliberately narrow reporting: a silent None here is what let the
        # column-order bug hide. Print the class context so it is diagnosable.
        print(f"[warn] macro-OvR AUC failed: {type(exc).__name__}: {exc} "
              f"(labels={list(labels)}, proba.shape={np.shape(proba)})")
        return None


def fit_territory_classifier(
    X_train_std: np.ndarray, y_train_4c: np.ndarray,
) -> Tuple[LogisticRegression, float, Dict[str, float]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores: Dict[str, float] = {}
    for C in LOGREG_CS_TERR:
        fold_scores = []
        for tr, va in skf.split(X_train_std, y_train_4c):
            est = LogisticRegression(
                C=C, penalty="l2", solver="lbfgs",
                class_weight="balanced", max_iter=4000,
                multi_class="multinomial",
            )
            est.fit(X_train_std[tr], y_train_4c[tr])
            yhat = est.predict(X_train_std[va])
            fold_scores.append(f1_score(
                y_train_4c[va], yhat,
                labels=TERRITORIES_4C, average="macro", zero_division=0,
            ))
        cv_scores[str(float(C))] = float(np.mean(fold_scores))
    best_C = float(max(cv_scores, key=cv_scores.get))
    model = LogisticRegression(
        C=best_C, penalty="l2", solver="lbfgs",
        class_weight="balanced", max_iter=4000,
        multi_class="multinomial",
    ).fit(X_train_std, y_train_4c)
    return model, best_C, cv_scores


def score_block(
    y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray,
    rng: np.random.Generator,
    *, n_boot: int = N_BOOT, n_perm: int = N_PERM,
    labels: Optional[List[str]] = None,
    proba_labels: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Score a block of predictions.

    ``labels``       — evaluation label set (default TERRITORIES_4C). Made a
                       parameter 2026-08-10 (defect m11): it was hardcoded, so
                       calling this with 2-class collapsed data would have
                       macro-averaged over two zero-support labels and roughly
                       halved macro-F1.
    ``proba_labels`` — the class of each COLUMN of ``proba``, i.e.
                       ``list(model.classes_)``. Pass it whenever the model's
                       class order may differ from ``labels`` — see the A1 note
                       in ``macro_ovr_auc``.
    """
    labels = list(TERRITORIES_4C) if labels is None else list(labels)
    n = y_true.size
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels,
                              average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_auc = macro_ovr_auc(y_true, proba, labels, proba_labels)

    p, r, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {
        labels[i]: {
            "precision": float(p[i]), "recall": float(r[i]),
            "f1": float(f1[i]), "support": int(supp[i]),
        }
        for i in range(len(labels))
    }

    # Bootstrap CIs
    boot_f1 = np.empty(n_boot)
    boot_bal = np.empty(n_boot)
    boot_auc = np.empty(n_boot)
    boot_auc_has = np.zeros(n_boot, dtype=bool)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_f1[b] = f1_score(
            y_true[idx], y_pred[idx], labels=labels,
            average="macro", zero_division=0,
        )
        boot_bal[b] = balanced_accuracy_score(y_true[idx], y_pred[idx])
        a = macro_ovr_auc(y_true[idx], proba[idx], labels, proba_labels)
        if a is not None:
            boot_auc[b] = a
            boot_auc_has[b] = True

    # Permutation p-values (label-shuffle truth)
    perm_f1 = np.empty(n_perm)
    perm_bal = np.empty(n_perm)
    perm_auc = np.empty(n_perm)
    perm_auc_has = np.zeros(n_perm, dtype=bool)
    for q in range(n_perm):
        y_perm = y_true.copy()
        rng.shuffle(y_perm)
        perm_f1[q] = f1_score(
            y_perm, y_pred, labels=labels,
            average="macro", zero_division=0,
        )
        perm_bal[q] = balanced_accuracy_score(y_perm, y_pred)
        a = macro_ovr_auc(y_perm, proba, labels, proba_labels)
        if a is not None:
            perm_auc[q] = a
            perm_auc_has[q] = True

    auc_ci = None
    auc_p = None
    if macro_auc is not None and boot_auc_has.any() and perm_auc_has.any():
        bv = boot_auc[boot_auc_has]
        pv = perm_auc[perm_auc_has]
        auc_ci = [float(np.percentile(bv, 2.5)), float(np.percentile(bv, 97.5))]
        auc_p = float((np.sum(pv >= macro_auc) + 1) / (perm_auc_has.sum() + 1))

    return {
        "n_total": int(n),
        "n_per_class_truth": {l: int((y_true == l).sum()) for l in labels},
        "n_per_class_pred":  {l: int((y_pred == l).sum()) for l in labels},
        "labels": labels,
        "macro_f1": macro_f1,
        "macro_f1_ci95": [
            float(np.percentile(boot_f1, 2.5)),
            float(np.percentile(boot_f1, 97.5)),
        ],
        "permutation_p_macro_f1": float(
            (np.sum(perm_f1 >= macro_f1) + 1) / (n_perm + 1)
        ),
        "balanced_accuracy": bal_acc,
        "balanced_accuracy_ci95": [
            float(np.percentile(boot_bal, 2.5)),
            float(np.percentile(boot_bal, 97.5)),
        ],
        "permutation_p_balanced_accuracy": float(
            (np.sum(perm_bal >= bal_acc) + 1) / (n_perm + 1)
        ),
        "macro_auc_ovr": macro_auc,
        "macro_auc_ovr_ci95": auc_ci,
        "permutation_p_macro_auc_ovr": auc_p,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def eval_pipeline_a(
    Z_train_mi: np.ndarray, Z_test_mi: np.ndarray, Z_ptbxl_primary: np.ndarray,
    y_train_4c: np.ndarray, y_test_4c: np.ndarray, y_ptbxl_4c: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, object]:
    scaler = StandardScaler().fit(Z_train_mi)
    X_tr = scaler.transform(Z_train_mi)
    X_te = scaler.transform(Z_test_mi)
    X_pt = scaler.transform(Z_ptbxl_primary)

    model, best_C, cv_scores = fit_territory_classifier(X_tr, y_train_4c)

    # In-domain
    yhat_te = model.predict(X_te)
    proba_te = model.predict_proba(X_te)
    in_dom = score_block(y_test_4c, yhat_te, proba_te, rng=rng,
                         proba_labels=list(model.classes_))

    # Cross-domain
    yhat_pt = model.predict(X_pt)
    proba_pt = model.predict_proba(X_pt)
    cd = score_block(y_ptbxl_4c, yhat_pt, proba_pt, rng=rng,
                     proba_labels=list(model.classes_))

    return {
        "best_C": best_C,
        "cv_scores_per_C": cv_scores,
        "in_domain_4c": in_dom,
        "cross_domain_4c": cd,
    }


# ---------------------------------------------------------------------------
# Per-K orchestration
# ---------------------------------------------------------------------------

def run_one_K_condition(
    K: int, suffix: str,
    targets: Dict[str, Dict[str, np.ndarray]],
    ptbxl_primary_idx: np.ndarray, ptbxl_primary_truth: np.ndarray,
    rng: np.random.Generator,
) -> Dict:
    """One config: K, suffix in {'', '_inlp'}."""
    print(f"\n--- K={K}  suffix={suffix or '(none)'} ---")
    # Load latents
    Z_med_tr_full = load_Z(K, "medalcare", "train", suffix=suffix)
    Z_med_te_full = load_Z(K, "medalcare", "test", suffix=suffix)
    Z_ptb_te_full = load_Z(K, "ptbxl", "test", suffix=suffix)

    idx_tr = targets["train"]["idx_in_split"]
    idx_te = targets["test"]["idx_in_split"]
    Z_med_tr_mi = Z_med_tr_full[idx_tr]
    Z_med_te_mi = Z_med_te_full[idx_te]
    Z_ptb_primary = Z_ptb_te_full[ptbxl_primary_idx]

    print(
        f"[load] Z_med_train_full={Z_med_tr_full.shape} -> MI={Z_med_tr_mi.shape}; "
        f"Z_med_test_full={Z_med_te_full.shape} -> MI={Z_med_te_mi.shape}; "
        f"Z_ptbxl_test_full={Z_ptb_te_full.shape} -> primary4c={Z_ptb_primary.shape}"
    )

    # L2 probes
    probes = eval_l2_probes(
        Z_med_tr_mi, Z_med_te_mi,
        targets["train"], targets["test"],
    )
    print(
        f"[L2] phi R²_circ={probes['phi']['r2_circular']:.3f}  "
        f"z R²={probes['z']['r2']:.3f}  size R²={probes['size']['r2']:.3f}  "
        f"trans AUC={probes['transmurality']['auc']}"
    )

    # Pipeline A (L3 + in-domain 4c)
    y_train_4c = np.asarray(targets["train"]["territory_4c"].tolist(), dtype=object)
    y_test_4c = np.asarray(targets["test"]["territory_4c"].tolist(), dtype=object)
    pa = eval_pipeline_a(
        Z_med_tr_mi, Z_med_te_mi, Z_ptb_primary,
        y_train_4c, y_test_4c, ptbxl_primary_truth,
        rng=rng,
    )
    id_ = pa["in_domain_4c"]; cd = pa["cross_domain_4c"]
    print(
        f"[L3] in_dom: F1={id_['macro_f1']:.3f}  AUC={id_['macro_auc_ovr']}  "
        f"|  CD: F1={cd['macro_f1']:.3f} CI={cd['macro_f1_ci95']} "
        f"p_f1={cd['permutation_p_macro_f1']:.4f}  "
        f"AUC={cd['macro_auc_ovr']} CI={cd['macro_auc_ovr_ci95']}"
    )

    return {"probes": probes, "pipeline_a": pa}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ks", type=int, nargs="+", default=[16, 64, 256, 1024])
    ap.add_argument("--out", type=Path,
                    default=OUT_DIR / "eval_decoding_lowK.json")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--skip-orig", action="store_true",
                    help="Only evaluate the _inlp condition (useful for backfill).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    global N_BOOT, N_PERM
    N_BOOT = args.n_boot
    N_PERM = args.n_perm

    targets = load_theta_targets()
    ptbxl_idx, ptbxl_truth = load_ptbxl_primary_4c()
    print(f"[load] PTB-XL primary 4c subset: n={ptbxl_idx.size}; "
          f"counts={dict(pd.Series(ptbxl_truth).value_counts())}")

    rng = np.random.default_rng(SEED)  # fallback; rebound per (K, condition) below (m10)
    summary: Dict[str, Dict] = {}
    if args.out.exists():
        try:
            summary = json.loads(args.out.read_text(encoding="utf-8"))
            print(f"[merge] loaded existing summary keys={list(summary.keys())}")
        except Exception as exc:
            print(f"[warn] could not parse existing summary: {exc}")

    for K in args.ks:
        key = f"K{K}"
        summary.setdefault(key, {})
        if not args.skip_orig:
            try:
                summary[key]["orig"] = run_one_K_condition(
                    K, suffix="", targets=targets,
                    ptbxl_primary_idx=ptbxl_idx,
                    ptbxl_primary_truth=ptbxl_truth,
                    rng=derive_rng("eval_decoding_lowK", K, "orig", seed=SEED),
                )
            except FileNotFoundError as exc:
                print(f"[skip] orig K={K}: {exc}")
        try:
            summary[key]["inlp"] = run_one_K_condition(
                K, suffix="_inlp", targets=targets,
                ptbxl_primary_idx=ptbxl_idx,
                ptbxl_primary_truth=ptbxl_truth,
                rng=derive_rng("eval_decoding_lowK", K, "inlp", seed=SEED),
            )
        except FileNotFoundError as exc:
            print(f"[skip] inlp K={K}: {exc}")
        args.out.write_text(json.dumps(summary, indent=2, default=str),
                            encoding="utf-8")

    # Cross-K summary table
    print()
    print("=" * 110)
    print(
        f"{'K':>5s} {'cond':<6s} | "
        f"{'phi_R²':>7s} {'z_R²':>7s} {'sz_R²':>7s} {'tr_AUC':>7s} | "
        f"{'CD_F1':>6s} {'CD_F1_CI':>15s} {'CD_AUC':>7s} {'CD_AUC_CI':>15s} "
        f"{'CD_p_f1':>9s}"
    )
    print("=" * 110)
    for kkey, conds in summary.items():
        K = int(kkey.lstrip("K"))
        for cond_name, cond in conds.items():
            pb = cond["probes"]; cd = cond["pipeline_a"]["cross_domain_4c"]
            ph = pb["phi"]["r2_circular"]; z = pb["z"]["r2"]; sz = pb["size"]["r2"]
            tr_auc = pb["transmurality"]["auc"]
            cd_f1 = cd["macro_f1"]; cd_f1ci = cd["macro_f1_ci95"]
            cd_auc = cd["macro_auc_ovr"]; cd_auc_ci = cd["macro_auc_ovr_ci95"]
            cd_p_f1 = cd["permutation_p_macro_f1"]
            tr_auc_s = f"{tr_auc:.3f}" if tr_auc is not None else "  --"
            cd_auc_s = f"{cd_auc:.3f}" if cd_auc is not None else "  --"
            cd_auc_ci_s = (
                f"[{cd_auc_ci[0]:.2f},{cd_auc_ci[1]:.2f}]"
                if cd_auc_ci is not None else "      --      "
            )
            print(
                f"{K:>5d} {cond_name:<6s} | "
                f"{ph:>7.3f} {z:>7.3f} {sz:>7.3f} {tr_auc_s:>7s} | "
                f"{cd_f1:>6.3f} [{cd_f1ci[0]:.2f},{cd_f1ci[1]:.2f}] "
                f"{cd_auc_s:>7s} {cd_auc_ci_s:>15s} "
                f"{cd_p_f1:>9.4f}"
            )
    print(f"\nWrote {args.out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
