"""B2 + B2-CD: continuous infarct decoding from frozen ECGFounder latents.

This script delivers the in-domain (B2) and cross-domain (B2-CD) analyses for
the four target ``isch[0]`` parameters:

- ``phi``         (angular position, circular, sin/cos Ridge)
- ``z``           (longitudinal position, Ridge)
- ``size``        (lesion volume, Ridge)
- ``rho_eps_max`` (transmurality, binary {0.3, 1.0}, Logistic)

Section 6 of the OpenSpec change handles **in-domain (B2)** -- linear probes
fit on MedalCare TRAIN MI rows and evaluated on MedalCare TEST MI rows.
Section 7 (B2-CD) -- to be added in a follow-up step -- applies the
MedalCare-trained phi probe to PTB-XL latents and bins the predictions into
anatomical territories.

For each configuration ``c in {exp5_3class, exp6_3class, exp7_baseline,
exp7_ccmmd}`` and each input source ``s in {Z, ecg_features}``, the script
fits a tiny linear probe (~1024 weights / target), evaluates on test, runs
1000-resample bootstrap CIs and (for the Ridge probes) 1000 permutation
tests via the closed-form ``y_test_pred = K @ y_train`` trick.

Outputs:

- ``outputs/phase_b2/in_domain.json``     -- all metrics + CIs + p-values.
- ``outputs/phase_b2/polar_<config>.png`` -- predicted vs true phi plot
  for ``exp7_baseline`` and ``exp7_ccmmd``.

Run::

    python analysis/phase_b2_infarct_decoding.py
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.linear_model import RidgeCV, LogisticRegression  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    r2_score,
    mean_absolute_error,
    roc_auc_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    balanced_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

LATENT_TRAIN_TEMPLATE = "outputs/latents/{stem}_medalcare_train/latents.npz"
LATENT_TEST_TEMPLATE = "outputs/latents/{stem}_medalcare/latents.npz"

# Map config-name -> (train latent dir stem, test latent dir stem).
# stem is what plugs into the templates above.
CONFIG_LATENT_STEMS: Dict[str, str] = {
    "exp5_3class":   "exp5_3class",
    "exp6_3class":   "exp6_3class",
    "exp7_baseline": "exp7",          # exp7_medalcare_train / exp7_medalcare
    "exp7_ccmmd":    "exp7_ccmmd",    # exp7_ccmmd_medalcare_train / exp7_ccmmd_medalcare
}

THETA_TRAIN_PATH = REPO_ROOT / "data" / "theta_mi_train.npz"
THETA_TEST_PATH = REPO_ROOT / "data" / "theta_mi_test.npz"
FEAT_TRAIN_PATH = REPO_ROOT / "data" / "ecg_features_medalcare_train.npz"
FEAT_TEST_PATH = REPO_ROOT / "data" / "ecg_features_medalcare_test.npz"

# Cross-domain (B2-CD) — PTB-XL inputs
LATENT_PTBXL_TEMPLATE = "outputs/latents/{stem}_ptbxl/latents.npz"
PTBXL_SUBCLASS_PATH = REPO_ROOT / "data" / "ptbxl_mi_subclass.csv"
FEAT_PTBXL_PATH = REPO_ROOT / "data" / "ecg_features_ptbxl_test.npz"

# Empirical phi-bin boundaries audited in §3 (theta_mi_build_summary.json):
#   LAD MI ⊂ [+0.000, +1.999]  → Anterior
#   RCA MI ⊂ [-1.999, -0.001]  → Inferior
#   LCX MI ⊂ [-3.139, +3.140]  (wraps around ±π)  → Lateral
# Boundary at ±2.0 cleanly partitions LAD vs LCX vs RCA in MedalCare TRAIN.
PHI_BIN_BOUNDARY = 2.0
TERRITORY_LABELS = ["Anterior", "Inferior", "Lateral"]

# ---------------------------------------------------------------------------
# Pipeline A — Direct coronary-territory classifier (added 2026-05-13).
#
# Trained on Z_MedalCare_train_MI -> territory_4c with class_weight='balanced'
# and 5-fold internal CV over C. Evaluated in-domain on MedalCare_test_MI and
# cross-domain on PTB-XL primary 4c subset (n=438). Macro-F1 and balanced
# accuracy with 1000-resample percentile bootstrap CIs and label-shuffle
# permutation p-values. Pipeline B (calibrated phi-bins) lives in §3.3.
# ---------------------------------------------------------------------------
TERRITORIES_4C = ["Anteroseptal", "Anterolateral", "Inferior", "Inferolateral"]
TERRITORIES_2C = ["Anterior", "Inferior"]
TERRITORY_4C_TO_2C: Dict[str, str] = {
    "Anteroseptal": "Anterior",
    "Anterolateral": "Anterior",
    "Inferior": "Inferior",
    "Inferolateral": "Inferior",
}

# Wider C-grid for the 4-class territory probe (1024-d Z favors stronger L2 than
# the binary rho_eps_max probe; smoke run hit the smallest C in LOGREG_CS).
LOGREG_CS_TERR_4C = np.logspace(-5, 2, 8)  # [1e-5, 1e-4, ..., 1e1, 1e2]

# ---------------------------------------------------------------------------
# Section 3.4 / 8-class in-domain audit (MedalCare only).
# Full MedalCare folder taxonomy: 4 anatomies x 2 transmuralities.
TERRITORIES_8C = [
    "LAD_0.3", "LAD_1.0",
    "LCX_0.3_ant", "LCX_0.3_post",
    "LCX_1.0_ant", "LCX_1.0_post",
    "RCA_0.3", "RCA_1.0",
]
# Collapse 8c -> 4c anatomy (drops transmurality).
TERRITORY_8C_TO_4C: Dict[str, str] = {
    "LAD_0.3":      "Anteroseptal",  "LAD_1.0":      "Anteroseptal",
    "LCX_0.3_ant":  "Anterolateral", "LCX_1.0_ant":  "Anterolateral",
    "LCX_0.3_post": "Inferolateral", "LCX_1.0_post": "Inferolateral",
    "RCA_0.3":      "Inferior",      "RCA_1.0":      "Inferior",
}
# Collapse 8c -> 2c transmurality (drops anatomy).
TRANSMURALITY_LABELS = ["0.3", "1.0"]
TERRITORY_8C_TO_TRANS: Dict[str, str] = {
    "LAD_0.3":      "0.3", "LAD_1.0":      "1.0",
    "LCX_0.3_ant":  "0.3", "LCX_1.0_ant":  "1.0",
    "LCX_0.3_post": "0.3", "LCX_1.0_post": "1.0",
    "RCA_0.3":      "0.3", "RCA_1.0":      "1.0",
}
# Same C-grid as the 4c probe -- 1024-d Z still favors heavy L2 for 8c.
LOGREG_CS_TERR_8C = LOGREG_CS_TERR_4C

OUT_DIR = REPO_ROOT / "outputs" / "phase_b2"
OUT_JSON = OUT_DIR / "in_domain.json"
OUT_CD_JSON = OUT_DIR / "cross_domain.json"

CONFIGS = ["exp5_3class", "exp6_3class", "exp7_baseline", "exp7_ccmmd"]
SOURCES = ["Z", "ecg_features"]
N_BOOT = 1000
N_PERM = 1000          # for Ridge (closed-form, fast)
N_PERM_BINARY = 200    # for LogisticRegression (refits required)
SEED = 42

# RidgeCV / LogisticRegression hyperparam grids.
# Ridge α extended to 1e6 because the original 1e-3..1e3 grid saturated at the
# upper bound for `size` (sensitivity check confirmed CV optimum at α=1e4).
# Logistic C: 1e-2 turned out to be the CV optimum (sensitivity over 1e-5..1e2),
# so we keep the wider 1e-3..1e2 grid here for safety + document the optimum.
RIDGE_ALPHAS = np.logspace(-3, 6, 10)
LOGREG_CS = np.logspace(-3, 2, 6)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_targets() -> Dict[str, Dict[str, np.ndarray]]:
    """Load MedalCare MI targets aligned to latent rows."""
    train = dict(np.load(THETA_TRAIN_PATH, allow_pickle=True))
    test = dict(np.load(THETA_TEST_PATH, allow_pickle=True))
    print(
        f"[targets] train MI={train['idx_in_split'].size}, "
        f"test MI={test['idx_in_split'].size}"
    )
    return {"train": train, "test": test}


def load_features() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load NeuroKit2 ECG features (full per-split shape, NaN for non-MI)."""
    fa = np.load(FEAT_TRAIN_PATH, allow_pickle=True)
    fb = np.load(FEAT_TEST_PATH, allow_pickle=True)
    feat_train = fa["features"]
    feat_test = fb["features"]
    nk2_ok_train = fa["nk2_ok"]
    nk2_ok_test = fb["nk2_ok"]
    feature_names = list(fa["feature_names"])
    print(
        f"[features] train shape={feat_train.shape} (strict_OK={int(nk2_ok_train.sum())}), "
        f"test shape={feat_test.shape} (strict_OK={int(nk2_ok_test.sum())})"
    )
    return feat_train, feat_test, nk2_ok_train, nk2_ok_test, feature_names


def load_config_latents(config: str, *, suffix: str = "") -> Tuple[np.ndarray, np.ndarray]:
    """Load Z_train_full and Z_test_full for one config.

    Parameters
    ----------
    suffix : optional directory suffix appended to ``{stem}_{split}`` (e.g. ``"_inlp"``
             reads from ``outputs/latents/{stem}_{split}_inlp/latents.npz``). Default
             empty preserves byte-identical pre-INLP behaviour.
    """
    stem = CONFIG_LATENT_STEMS[config]
    train_dir = f"{stem}_medalcare_train{suffix}"
    test_dir = f"{stem}_medalcare{suffix}"
    train_path = REPO_ROOT / "outputs" / "latents" / train_dir / "latents.npz"
    test_path = REPO_ROOT / "outputs" / "latents" / test_dir / "latents.npz"
    Z_train = np.load(train_path, allow_pickle=True)["Z"].astype(np.float64)
    Z_test = np.load(test_path, allow_pickle=True)["Z"].astype(np.float64)
    return Z_train, Z_test


# ---------------------------------------------------------------------------
# Imputation + standardisation
# ---------------------------------------------------------------------------

def median_impute_with_train_medians(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fill NaN cells with the per-feature TRAIN median.

    Returns
    -------
    X_train_imp, X_test_imp : imputed arrays (no NaN).
    medians                 : (n_features,) train medians.
    train_imputed_pct       : (n_features,) percent of train cells imputed.
    test_imputed_pct        : (n_features,) percent of test cells imputed.
    """
    medians = np.nanmedian(X_train, axis=0)
    train_imp = X_train.copy()
    test_imp = X_test.copy()
    train_pct = np.zeros(X_train.shape[1])
    test_pct = np.zeros(X_train.shape[1])
    for j in range(X_train.shape[1]):
        nan_train = np.isnan(train_imp[:, j])
        nan_test = np.isnan(test_imp[:, j])
        train_imp[nan_train, j] = medians[j]
        test_imp[nan_test, j] = medians[j]
        train_pct[j] = 100.0 * nan_train.mean()
        test_pct[j] = 100.0 * nan_test.mean()
    return train_imp, test_imp, medians, train_pct, test_pct


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    sc = StandardScaler()
    sc.fit(X_train)
    return sc


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def circular_diff(phi_a: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
    """Wrapped (a-b) in [-pi, pi]."""
    return np.arctan2(np.sin(phi_a - phi_b), np.cos(phi_a - phi_b))


def circular_mean(phi: np.ndarray) -> float:
    return float(np.arctan2(np.sin(phi).mean(), np.cos(phi).mean()))


def circular_mae_deg(phi_pred: np.ndarray, phi_true: np.ndarray) -> float:
    return float(np.degrees(np.mean(np.abs(circular_diff(phi_pred, phi_true)))))


def circular_r2(
    phi_pred: np.ndarray,
    phi_true: np.ndarray,
    phi_train_mean: float,
) -> float:
    """1 - SSE_circ(model) / SSE_circ(circular-mean baseline)."""
    num = float(np.sum(1.0 - np.cos(phi_pred - phi_true)))
    den = float(np.sum(1.0 - np.cos(phi_true - phi_train_mean)))
    return 1.0 - num / den if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def _ridge_K_matrix(X_train: np.ndarray, X_test: np.ndarray, alpha: float) -> np.ndarray:
    """Precompute K = X_test @ (X_train^T X_train + alpha I)^{-1} X_train^T.

    Then ``y_test_pred = K @ y_train`` for any y_train (used by permutation).
    """
    XtX = X_train.T @ X_train
    M = np.linalg.solve(XtX + alpha * np.eye(X_train.shape[1]), X_train.T)
    return X_test @ M  # shape (n_test, n_train)


def fit_ridge_continuous(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
    n_perm: int = N_PERM,
) -> Dict[str, object]:
    """Fit RidgeCV for a single continuous target."""
    model = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
    model.fit(X_train, y_train)
    alpha = float(model.alpha_)
    y_pred = model.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))

    n_test = y_test.size
    boot_r2 = np.empty(n_boot)
    boot_mae = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_test, n_test)
        boot_r2[b] = r2_score(y_test[idx], y_pred[idx])
        boot_mae[b] = mean_absolute_error(y_test[idx], y_pred[idx])

    K = _ridge_K_matrix(X_train, X_test, alpha)
    n_train = X_train.shape[0]
    perm_r2 = np.empty(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(n_train)
        y_pred_perm = K @ y_train[perm]
        perm_r2[p] = r2_score(y_test, y_pred_perm)
    p_value = float((np.sum(perm_r2 >= r2) + 1) / (n_perm + 1))

    return {
        "alpha": alpha,
        "r2": r2,
        "mae": mae,
        "r2_ci95": [float(np.percentile(boot_r2, 2.5)), float(np.percentile(boot_r2, 97.5))],
        "mae_ci95": [float(np.percentile(boot_mae, 2.5)), float(np.percentile(boot_mae, 97.5))],
        "permutation_p_r2": p_value,
        "n_test": int(n_test),
        "_y_pred": y_pred,  # internal, popped before JSON serialisation
    }


def fit_ridge_phi(
    X_train: np.ndarray,
    X_test: np.ndarray,
    phi_train: np.ndarray,
    phi_test: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
    n_perm: int = N_PERM,
) -> Dict[str, object]:
    """Joint sin/cos Ridge with multi-output CV (single shared alpha)."""
    Y_train = np.stack([np.sin(phi_train), np.cos(phi_train)], axis=1)
    Y_test = np.stack([np.sin(phi_test), np.cos(phi_test)], axis=1)
    model = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
    model.fit(X_train, Y_train)
    alpha = float(np.atleast_1d(model.alpha_).item())  # scalar shared alpha
    Y_pred = model.predict(X_test)
    phi_pred = np.arctan2(Y_pred[:, 0], Y_pred[:, 1])

    sin_r2 = float(r2_score(Y_test[:, 0], Y_pred[:, 0]))
    cos_r2 = float(r2_score(Y_test[:, 1], Y_pred[:, 1]))
    cmae = circular_mae_deg(phi_pred, phi_test)
    phi_train_mean = circular_mean(phi_train)
    cr2 = circular_r2(phi_pred, phi_test, phi_train_mean)

    n_test = phi_test.size
    boot_cmae = np.empty(n_boot)
    boot_cr2 = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_test, n_test)
        boot_cmae[b] = circular_mae_deg(phi_pred[idx], phi_test[idx])
        boot_cr2[b] = circular_r2(phi_pred[idx], phi_test[idx], phi_train_mean)

    K = _ridge_K_matrix(X_train, X_test, alpha)
    n_train = X_train.shape[0]
    perm_cr2 = np.empty(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(n_train)
        Y_perm_pred = K @ Y_train[perm]
        phi_perm_pred = np.arctan2(Y_perm_pred[:, 0], Y_perm_pred[:, 1])
        perm_cr2[p] = circular_r2(phi_perm_pred, phi_test, phi_train_mean)
    p_value = float((np.sum(perm_cr2 >= cr2) + 1) / (n_perm + 1))

    return {
        "alpha": alpha,
        "sin_r2": sin_r2,
        "cos_r2": cos_r2,
        "circular_mae_deg": cmae,
        "circular_r2": cr2,
        "circular_mae_deg_ci95": [float(np.percentile(boot_cmae, 2.5)), float(np.percentile(boot_cmae, 97.5))],
        "circular_r2_ci95": [float(np.percentile(boot_cr2, 2.5)), float(np.percentile(boot_cr2, 97.5))],
        "permutation_p_circular_r2": p_value,
        "n_test": int(n_test),
        "_phi_pred": phi_pred,
        "_phi_train_mean": phi_train_mean,
        "_model": model,  # for cross-domain reuse
    }


def fit_logistic_binary(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
    n_perm: int = N_PERM_BINARY,
) -> Dict[str, object]:
    """Logistic regression with manual CV over Cs (uses scoring='roc_auc')."""
    yb_train = (y_train > 0.5).astype(int)
    yb_test = (y_test > 0.5).astype(int)
    if len(np.unique(yb_test)) < 2:
        raise RuntimeError("Test labels are single-class for rho_eps_max; cannot compute AUC.")

    # Manual CV to pick C (LogisticRegressionCV is finicky with `cv` + class_weight; do this explicitly).
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = {}
    for C in LOGREG_CS:
        fold_aucs = []
        for tr_idx, va_idx in skf.split(X_train, yb_train):
            est = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=2000)
            est.fit(X_train[tr_idx], yb_train[tr_idx])
            scores_va = est.predict_proba(X_train[va_idx])[:, 1]
            fold_aucs.append(roc_auc_score(yb_train[va_idx], scores_va))
        cv_scores[C] = float(np.mean(fold_aucs))
    best_C = max(cv_scores, key=cv_scores.get)

    model = LogisticRegression(C=best_C, penalty="l2", solver="lbfgs", max_iter=2000)
    model.fit(X_train, yb_train)
    y_score = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(yb_test, y_score))

    n_test = yb_test.size
    boot_auc = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_test, n_test)
        if len(np.unique(yb_test[idx])) < 2:
            boot_auc[b] = np.nan
        else:
            boot_auc[b] = roc_auc_score(yb_test[idx], y_score[idx])

    perm_auc = np.empty(n_perm)
    n_train = X_train.shape[0]
    for p in range(n_perm):
        perm = rng.permutation(n_train)
        m = LogisticRegression(C=best_C, penalty="l2", solver="lbfgs", max_iter=2000)
        m.fit(X_train, yb_train[perm])
        perm_auc[p] = roc_auc_score(yb_test, m.predict_proba(X_test)[:, 1])
    p_value = float((np.sum(perm_auc >= auc) + 1) / (n_perm + 1))

    return {
        "best_C": float(best_C),
        "cv_scores_per_C": {str(c): float(s) for c, s in cv_scores.items()},
        "auc": auc,
        "auc_ci95": [
            float(np.nanpercentile(boot_auc, 2.5)),
            float(np.nanpercentile(boot_auc, 97.5)),
        ],
        "permutation_p_auc": p_value,
        "n_test": int(n_test),
        "n_test_pos": int(yb_test.sum()),
        "_y_score": y_score,
    }


# ---------------------------------------------------------------------------
# Paired bootstrap: source_a vs source_b on same metric
# ---------------------------------------------------------------------------

def paired_bootstrap_continuous(
    y_test: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    rng: np.random.Generator,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = N_BOOT,
    higher_is_better: bool = True,
) -> Dict[str, float]:
    """Compare metric(y_test, y_pred_a) vs metric(y_test, y_pred_b) with the same resamples."""
    n = y_test.size
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        m_a = metric_fn(y_test[idx], y_pred_a[idx])
        m_b = metric_fn(y_test[idx], y_pred_b[idx])
        diffs[b] = m_a - m_b
    sign = 1.0 if higher_is_better else -1.0
    return {
        "mean_diff": float(diffs.mean()),
        "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "p_a_beats_b": float(np.mean(sign * diffs <= 0.0)),
    }


def paired_bootstrap_circular(
    phi_test: np.ndarray,
    phi_pred_a: np.ndarray,
    phi_pred_b: np.ndarray,
    phi_train_mean: float,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> Dict[str, float]:
    """Compare circular R² for two phi predictors."""
    n = phi_test.size
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        d = (
            circular_r2(phi_pred_a[idx], phi_test[idx], phi_train_mean)
            - circular_r2(phi_pred_b[idx], phi_test[idx], phi_train_mean)
        )
        diffs[b] = d
    return {
        "mean_diff": float(diffs.mean()),
        "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "p_a_beats_b": float(np.mean(diffs <= 0.0)),
    }


# ---------------------------------------------------------------------------
# Polar plot
# ---------------------------------------------------------------------------

def polar_plot(
    phi_true: np.ndarray,
    phi_pred: np.ndarray,
    transmurality: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    err_deg = np.degrees(np.abs(circular_diff(phi_pred, phi_true)))
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    sc = ax1.scatter(
        phi_true,
        np.ones_like(phi_true),
        c=err_deg, cmap="RdYlGn_r", vmin=0, vmax=90,
        s=14, alpha=0.75, edgecolors="none",
    )
    ax1.set_yticklabels([])
    ax1.set_title("True phi position\n(colour = |prediction error| in degrees)")
    cbar = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.10)
    cbar.set_label("|err| (deg)")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(np.degrees(phi_true), bins=40, alpha=0.55, label="true", color="C0")
    ax2.hist(np.degrees(phi_pred), bins=40, alpha=0.55, label="predicted", color="C1")
    ax2.set_xlabel("phi (degrees)")
    ax2.set_ylabel("count")
    ax2.set_title("phi distribution: true vs predicted")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Section 7: Cross-domain (B2-CD) -- predicted phi -> binned territory
# ---------------------------------------------------------------------------

def bin_phi_to_territory(phi: np.ndarray, boundary: float = PHI_BIN_BOUNDARY) -> np.ndarray:
    """Bin (predicted) phi values into {Anterior, Inferior, Lateral}.

    Bin boundaries (audited from MedalCare TRAIN per-coronary phi ranges):
      phi in [0, +boundary)           -> Anterior  (matches LAD)
      phi in [-boundary, 0)           -> Inferior  (matches RCA)
      phi in [+boundary, pi]U[-pi,    -> Lateral   (matches LCX, wraps ±pi)
             -boundary)
    """
    p = np.arctan2(np.sin(phi), np.cos(phi))  # ensure in [-pi, pi]
    out = np.full(p.shape, "", dtype=object)
    out[(p >= 0.0) & (p < boundary)] = "Anterior"
    out[(p >= -boundary) & (p < 0.0)] = "Inferior"
    out[(p >= boundary) | (p < -boundary)] = "Lateral"
    return out


def load_ptbxl_subclass_csv() -> pd.DataFrame:
    df = pd.read_csv(PTBXL_SUBCLASS_PATH)
    return df


def load_ptbxl_latents(config: str, *, suffix: str = "") -> np.ndarray:
    """Load PTB-XL latents for ``config``, optionally with a directory ``suffix``."""
    stem = CONFIG_LATENT_STEMS[config]
    p = REPO_ROOT / "outputs" / "latents" / f"{stem}_ptbxl{suffix}" / "latents.npz"
    return np.load(p, allow_pickle=True)["Z"].astype(np.float64)


def load_ptbxl_features() -> Tuple[np.ndarray, np.ndarray]:
    f = np.load(FEAT_PTBXL_PATH, allow_pickle=True)
    return f["features"].astype(np.float64), f["nk2_ok"].astype(bool)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, object]:
    """Per-class precision / recall / F1 + macro-F1 + confusion matrix."""
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {
        labels[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(labels))
    }
    return {
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def _bootstrap_classification_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    rng: np.random.Generator,
    n_boot: int,
) -> Tuple[float, float]:
    n = y_true.size
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = f1_score(y_true[idx], y_pred[idx], labels=labels, average="macro", zero_division=0)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _permutation_p_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    observed_macro_f1: float,
    labels: List[str],
    rng: np.random.Generator,
    n_perm: int,
) -> float:
    n = y_true.size
    perm_scores = np.empty(n_perm)
    for p in range(n_perm):
        y_perm = y_true.copy()
        rng.shuffle(y_perm)
        perm_scores[p] = f1_score(y_perm, y_pred, labels=labels, average="macro", zero_division=0)
    return float((np.sum(perm_scores >= observed_macro_f1) + 1) / (n_perm + 1))


def cross_domain_phi_eval(
    phi_model: RidgeCV,
    scaler: StandardScaler,
    X_ptbxl: np.ndarray,
    territory_truth: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    n_perm: int,
    label: str = "",
) -> Dict[str, object]:
    """Apply MedalCare-fit phi Ridge to PTB-XL inputs, bin, evaluate.

    Parameters
    ----------
    phi_model : RidgeCV fitted on standardised MedalCare-train Z (or features),
                multi-output (sin, cos).
    scaler    : StandardScaler fitted on the same MedalCare-train inputs.
    X_ptbxl   : raw PTB-XL inputs aligned to ``territory_truth`` (filtered to
                single-territory primary MI rows).
    territory_truth : object array of {Anterior, Inferior, Lateral}.
    """
    X_std = scaler.transform(X_ptbxl)
    Y_pred = phi_model.predict(X_std)
    phi_pred = np.arctan2(Y_pred[:, 0], Y_pred[:, 1])
    territory_pred = bin_phi_to_territory(phi_pred)

    base_3 = _classification_metrics(territory_truth, territory_pred, TERRITORY_LABELS)
    macro_f1_3 = base_3["macro_f1"]
    ci_lo_3, ci_hi_3 = _bootstrap_classification_macro_f1(
        territory_truth, territory_pred, TERRITORY_LABELS, rng, n_boot
    )
    p_perm_3 = _permutation_p_macro_f1(
        territory_truth, territory_pred, macro_f1_3, TERRITORY_LABELS, rng, n_perm
    )

    # 2-class Anterior-vs-Inferior sensitivity (Strategy A backup, ignores Lateral).
    mask_2 = np.isin(territory_truth, ["Anterior", "Inferior"])
    if mask_2.sum() > 1 and len(np.unique(territory_truth[mask_2])) == 2:
        # For the 2-class subset, also fold any Lateral predictions to the
        # nearer of Anterior/Inferior so we get a proper classification.
        # Convention: Lateral phi-predictions stay Lateral here; we just evaluate
        # F1 on the {Anterior, Inferior} truth subset (Lateral predictions on
        # AMI/IMI true rows count as misclassifications).
        base_2 = _classification_metrics(
            territory_truth[mask_2], territory_pred[mask_2], ["Anterior", "Inferior"]
        )
        ci_lo_2, ci_hi_2 = _bootstrap_classification_macro_f1(
            territory_truth[mask_2], territory_pred[mask_2], ["Anterior", "Inferior"], rng, n_boot
        )
        p_perm_2 = _permutation_p_macro_f1(
            territory_truth[mask_2], territory_pred[mask_2], base_2["macro_f1"],
            ["Anterior", "Inferior"], rng, n_perm
        )
    else:
        base_2 = None
        ci_lo_2 = ci_hi_2 = p_perm_2 = float("nan")

    return {
        "n_total": int(territory_truth.size),
        "n_per_class_truth": {
            t: int((territory_truth == t).sum()) for t in TERRITORY_LABELS
        },
        "n_per_class_pred": {
            t: int((territory_pred == t).sum()) for t in TERRITORY_LABELS
        },
        "phi_bin_boundary": PHI_BIN_BOUNDARY,
        "primary_3class": {
            **base_3,
            "macro_f1_ci95": [ci_lo_3, ci_hi_3],
            "permutation_p_macro_f1": p_perm_3,
        },
        "sensitivity_2class_AntInf": (
            None if base_2 is None else {
                **base_2,
                "macro_f1_ci95": [ci_lo_2, ci_hi_2],
                "permutation_p_macro_f1": p_perm_2,
            }
        ),
        "_phi_pred": phi_pred,
        "_territory_pred": territory_pred,
    }


# ---------------------------------------------------------------------------
# Section 3.2 / Pipeline A: direct coronary-territory classifier (4-class)
# ---------------------------------------------------------------------------

def fit_territory_4c_classifier(
    X_train_std: np.ndarray,
    y_train_4c: np.ndarray,
    Cs: np.ndarray = LOGREG_CS_TERR_4C,
    max_iter: int = 4000,
    seed: int = SEED,
) -> Tuple[LogisticRegression, float, Dict[str, float]]:
    """Fit multinomial LogReg on standardized X_train with internal 5-fold CV
    over Cs, then refit on the full training set at the best C.

    Returns
    -------
    model       : the refit LogisticRegression fitted on all of X_train_std.
    best_C      : the C value that maximized 5-fold CV macro-F1.
    cv_scores   : {str(C): mean_cv_macro_f1} for every C tried.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores: Dict[str, float] = {}
    for C in Cs:
        fold_scores: List[float] = []
        for tr_idx, va_idx in skf.split(X_train_std, y_train_4c):
            est = LogisticRegression(
                C=C,
                penalty="l2",
                solver="lbfgs",
                class_weight="balanced",
                max_iter=max_iter,
                multi_class="multinomial",
            )
            est.fit(X_train_std[tr_idx], y_train_4c[tr_idx])
            y_va_pred = est.predict(X_train_std[va_idx])
            fold_scores.append(
                f1_score(
                    y_train_4c[va_idx],
                    y_va_pred,
                    labels=TERRITORIES_4C,
                    average="macro",
                    zero_division=0,
                )
            )
        cv_scores[str(C)] = float(np.mean(fold_scores))

    best_C = float(max(cv_scores, key=cv_scores.get))
    model = LogisticRegression(
        C=best_C,
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        max_iter=max_iter,
        multi_class="multinomial",
    )
    model.fit(X_train_std, y_train_4c)
    return model, best_C, cv_scores


def _score_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rng: np.random.Generator,
    labels: List[str],
    n_boot: int = N_BOOT,
    n_perm: int = N_PERM_BINARY,
    collapse_map: Optional[Dict[str, str]] = None,
    collapse_labels: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Score multi-class predictions with macro-F1 + balanced accuracy.

    Returns 1000-bootstrap percentile CIs + 200-shuffle label-permutation p
    values for both metrics, per-class P/R/F1/support, and confusion matrix.

    When ``collapse_map`` is provided, both truth and predictions are remapped
    via that dict and scored on ``collapse_labels``; rows whose mapped value
    is "" are dropped (the underlying classifier is NOT refit).
    """
    if collapse_map is not None:
        assert collapse_labels is not None, "collapse_labels required with collapse_map"
        y_true_eval = np.array([collapse_map.get(t, "") for t in y_true], dtype=object)
        y_pred_eval = np.array([collapse_map.get(t, "") for t in y_pred], dtype=object)
        keep = (y_true_eval != "") & (y_pred_eval != "")
        y_true_eval = y_true_eval[keep]
        y_pred_eval = y_pred_eval[keep]
        eval_labels = collapse_labels
    else:
        y_true_eval = np.asarray(y_true, dtype=object)
        y_pred_eval = np.asarray(y_pred, dtype=object)
        eval_labels = labels

    n = y_true_eval.size
    macro_f1 = float(
        f1_score(y_true_eval, y_pred_eval, labels=eval_labels, average="macro", zero_division=0)
    )
    bal_acc = float(balanced_accuracy_score(y_true_eval, y_pred_eval))
    p, r, f1, support = precision_recall_fscore_support(
        y_true_eval, y_pred_eval, labels=eval_labels, zero_division=0
    )
    cm = confusion_matrix(y_true_eval, y_pred_eval, labels=eval_labels)
    per_class = {
        eval_labels[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(eval_labels))
    }

    boot_f1 = np.empty(n_boot)
    boot_bal = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_f1[b] = f1_score(
            y_true_eval[idx], y_pred_eval[idx],
            labels=eval_labels, average="macro", zero_division=0,
        )
        boot_bal[b] = balanced_accuracy_score(y_true_eval[idx], y_pred_eval[idx])

    perm_f1 = np.empty(n_perm)
    perm_bal = np.empty(n_perm)
    for q in range(n_perm):
        y_perm = y_true_eval.copy()
        rng.shuffle(y_perm)
        perm_f1[q] = f1_score(
            y_perm, y_pred_eval, labels=eval_labels, average="macro", zero_division=0
        )
        perm_bal[q] = balanced_accuracy_score(y_perm, y_pred_eval)
    p_perm_f1 = float((np.sum(perm_f1 >= macro_f1) + 1) / (n_perm + 1))
    p_perm_bal = float((np.sum(perm_bal >= bal_acc) + 1) / (n_perm + 1))

    return {
        "n_total": int(n),
        "n_per_class_truth": {t: int((y_true_eval == t).sum()) for t in eval_labels},
        "n_per_class_pred":  {t: int((y_pred_eval == t).sum()) for t in eval_labels},
        "labels": list(eval_labels),
        "macro_f1": macro_f1,
        "macro_f1_ci95": [
            float(np.percentile(boot_f1, 2.5)),
            float(np.percentile(boot_f1, 97.5)),
        ],
        "permutation_p_macro_f1": p_perm_f1,
        "balanced_accuracy": bal_acc,
        "balanced_accuracy_ci95": [
            float(np.percentile(boot_bal, 2.5)),
            float(np.percentile(boot_bal, 97.5)),
        ],
        "permutation_p_balanced_accuracy": p_perm_bal,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "_y_pred": y_pred_eval,
        "_y_true": y_true_eval,
    }


def _score_4c_predictions(
    y_true_4c: np.ndarray,
    y_pred_4c: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
    n_perm: int = N_PERM_BINARY,
    labels: List[str] = TERRITORIES_4C,
    collapse_4c_to_2c: bool = False,
) -> Dict[str, object]:
    """Backward-compat wrapper around ``_score_predictions`` used by Pipelines
    A and B. Hardcodes the 4c -> 2c collapse using ``TERRITORY_4C_TO_2C``.
    """
    return _score_predictions(
        y_true=y_true_4c, y_pred=y_pred_4c, rng=rng,
        labels=labels, n_boot=n_boot, n_perm=n_perm,
        collapse_map=(TERRITORY_4C_TO_2C if collapse_4c_to_2c else None),
        collapse_labels=(TERRITORIES_2C if collapse_4c_to_2c else None),
    )


def evaluate_territory_classifier(
    model: LogisticRegression,
    X_eval_std: np.ndarray,
    y_eval: np.ndarray,
    labels: List[str],
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
    n_perm: int = N_PERM_BINARY,
    collapse_4c_to_2c: bool = False,
) -> Dict[str, object]:
    """Apply a fitted multinomial LogReg to standardized X_eval and score it.

    Thin wrapper around ``_score_4c_predictions``: predicts with the model
    then scores. ``collapse_4c_to_2c`` triggers a 4c -> 2c truth+pred remap
    before scoring (model is NOT refit).
    """
    y_pred = model.predict(X_eval_std)
    return _score_4c_predictions(
        y_true_4c=y_eval, y_pred_4c=y_pred,
        rng=rng, n_boot=n_boot, n_perm=n_perm,
        labels=labels, collapse_4c_to_2c=collapse_4c_to_2c,
    )


def pipeline_a_for_source(
    src_name: str,
    X_train_std: np.ndarray,
    X_test_std: np.ndarray,
    X_ptbxl_std: np.ndarray,
    y_train_4c: np.ndarray,
    y_test_4c: np.ndarray,
    y_ptbxl_4c: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    n_perm: int,
) -> Dict[str, object]:
    """Run one full Pipeline A leg for a single source (Z or ecg_features).

    Trains a 4-class multinomial LogReg on (X_train_std, y_train_4c) and
    evaluates it on (X_test_std, y_test_4c) [in-domain] and on
    (X_ptbxl_std, y_ptbxl_4c) [cross-domain]. Both 4-class and 2-class
    (collapsed) metrics are reported for the cross-domain leg.
    """
    print(f"      [pipeline-A / {src_name}] fitting 4-class LogReg on n_train={X_train_std.shape[0]}")
    model, best_C, cv_scores = fit_territory_4c_classifier(X_train_std, y_train_4c)
    print(
        f"      [pipeline-A / {src_name}] best_C={best_C:g}; cv_macro_f1={cv_scores[str(best_C)]:.3f}"
    )

    in_domain = evaluate_territory_classifier(
        model, X_test_std, y_test_4c, TERRITORIES_4C,
        rng=rng, n_boot=n_boot, n_perm=n_perm, collapse_4c_to_2c=False,
    )
    cross_4c = evaluate_territory_classifier(
        model, X_ptbxl_std, y_ptbxl_4c, TERRITORIES_4C,
        rng=rng, n_boot=n_boot, n_perm=n_perm, collapse_4c_to_2c=False,
    )
    cross_2c = evaluate_territory_classifier(
        model, X_ptbxl_std, y_ptbxl_4c, TERRITORIES_4C,
        rng=rng, n_boot=n_boot, n_perm=n_perm, collapse_4c_to_2c=True,
    )
    return {
        "best_C": float(best_C),
        "cv_scores_per_C": cv_scores,
        "in_domain_4c": in_domain,
        "cross_domain_4c": cross_4c,
        "cross_domain_2c": cross_2c,
    }


# ---------------------------------------------------------------------------
# Section 3.3 / Pipeline B: calibrated phi-bins -> 4-class territory
# ---------------------------------------------------------------------------

# Hardcoded baseline boundaries for the 4-class phi-bin variant (mirrors the
# existing PHI_BIN_BOUNDARY but expanded to 4 wedges):
#   LAD            phi in (0, +2]      -> Anteroseptal
#   LCX_*_ant      phi in (+2, +pi]    -> Anterolateral
#   LCX_*_post     phi in [-pi, -2)    -> Inferolateral
#   RCA            phi in (-2, 0]      -> Inferior
# These are pre-registered from the MedalCare audit (theta_mi_build_summary.json).
PHI_4C_INNER_BOUNDARY = 0.0   # LAD <-> RCA split
PHI_4C_OUTER_BOUNDARY = 2.0   # LCX wedge boundary on both sides


def hardcoded_phi_to_4c(phi: np.ndarray) -> np.ndarray:
    """Map predicted phi in [-pi, +pi] to territory_4c via fixed wedges.

    Returns an object array shape (n,) with values in TERRITORIES_4C.
    """
    phi = np.arctan2(np.sin(phi), np.cos(phi))  # ensure wrap to [-pi, +pi]
    out = np.empty(phi.shape, dtype=object)
    out[(phi > 2.0)] = "Anterolateral"
    out[(phi >= 0.0) & (phi <= 2.0)] = "Anteroseptal"
    out[(phi >= -2.0) & (phi < 0.0)] = "Inferior"
    out[(phi < -2.0)] = "Inferolateral"
    return out


def fit_phi_to_4c_calibrator(
    phi_pred_train: np.ndarray,
    y_train_4c: np.ndarray,
    seed: int = SEED,
) -> Tuple[object, str, Dict[str, float]]:
    """Fit a calibrator from (sin(phi_pred), cos(phi_pred)) -> territory_4c.

    Candidates: DecisionTree depth=4, multinomial LogReg (L2, C=1.0), kNN(k=10,
    distance-weighted). All use class_weight='balanced' where applicable. Picks
    the candidate with the highest 5-fold StratifiedKFold internal CV macro-F1
    and refits on the full (phi_pred_train, y_train_4c) pool.

    Returns (fitted_estimator, name, cv_scores_per_name).
    """
    X = np.stack([np.sin(phi_pred_train), np.cos(phi_pred_train)], axis=1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    candidates: Dict[str, object] = {
        "tree_d4": DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=seed,
        ),
        "logreg_l2": LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs",
            multi_class="multinomial", class_weight="balanced", max_iter=2000,
        ),
        "knn_10": KNeighborsClassifier(n_neighbors=10, weights="distance"),
    }
    cv_scores: Dict[str, float] = {}
    for name, est_template in candidates.items():
        fold_f1: List[float] = []
        for tr_idx, va_idx in skf.split(X, y_train_4c):
            est = type(est_template)(**est_template.get_params())
            est.fit(X[tr_idx], y_train_4c[tr_idx])
            y_pred = est.predict(X[va_idx])
            fold_f1.append(
                f1_score(
                    y_train_4c[va_idx], y_pred,
                    labels=TERRITORIES_4C, average="macro", zero_division=0,
                )
            )
        cv_scores[name] = float(np.mean(fold_f1))

    best_name = max(cv_scores, key=cv_scores.get)
    best_est = candidates[best_name]
    best_est.fit(X, y_train_4c)
    return best_est, best_name, cv_scores


def _predict_phi_to_4c(calibrator: object, phi_pred: np.ndarray) -> np.ndarray:
    """Apply a fitted phi-to-4c calibrator to a 1-d phi array."""
    X = np.stack([np.sin(phi_pred), np.cos(phi_pred)], axis=1)
    return calibrator.predict(X)


def pipeline_b_for_source(
    src_name: str,
    phi_pred_test: np.ndarray,    # MedalCare TEST phi predictions, n=1200
    y_test_4c: np.ndarray,        # MedalCare TEST territory_4c, n=1200
    phi_pred_ptbxl: np.ndarray,   # PTB-XL primary 4c phi predictions, n=438
    y_ptbxl_4c: np.ndarray,       # PTB-XL primary 4c truth, n=438
    rng: np.random.Generator,
    n_boot: int,
    n_perm: int,
) -> Dict[str, object]:
    """Run one full Pipeline B leg for a single source (Z or ecg_features).

    The phi regressor is NOT refit -- ``phi_pred_test`` and ``phi_pred_ptbxl``
    are assumed pre-computed by the upstream phi-Ridge in this config. The
    calibrator is fit on (sin/cos of phi_pred_test) -> y_test_4c and then
    applied to phi_pred_ptbxl. The hardcoded wedge baseline is also scored for
    delta comparison.
    """
    print(f"      [pipeline-B / {src_name}] fitting phi->4c calibrator on n_train={phi_pred_test.size}")
    calibrator, cal_name, cv_scores = fit_phi_to_4c_calibrator(phi_pred_test, y_test_4c)
    print(
        f"      [pipeline-B / {src_name}] best calibrator = {cal_name}; "
        f"cv_macro_f1 = {cv_scores[cal_name]:.3f}  (others: "
        + ", ".join(f"{k}={v:.3f}" for k, v in cv_scores.items() if k != cal_name)
        + ")"
    )

    # Predicted territories from the calibrator (in-domain + cross-domain).
    y_pred_test_cal = _predict_phi_to_4c(calibrator, phi_pred_test)
    y_pred_ptbxl_cal = _predict_phi_to_4c(calibrator, phi_pred_ptbxl)
    # Predicted territories from the hardcoded wedges (cross-domain only -- the
    # in-domain hardcoded score is a sanity benchmark we also report).
    y_pred_test_hard = hardcoded_phi_to_4c(phi_pred_test)
    y_pred_ptbxl_hard = hardcoded_phi_to_4c(phi_pred_ptbxl)

    in_domain_cal = _score_4c_predictions(
        y_test_4c, y_pred_test_cal, rng=rng, n_boot=n_boot, n_perm=n_perm,
    )
    in_domain_hard = _score_4c_predictions(
        y_test_4c, y_pred_test_hard, rng=rng, n_boot=n_boot, n_perm=n_perm,
    )
    cross_cal_4c = _score_4c_predictions(
        y_ptbxl_4c, y_pred_ptbxl_cal, rng=rng, n_boot=n_boot, n_perm=n_perm,
    )
    cross_cal_2c = _score_4c_predictions(
        y_ptbxl_4c, y_pred_ptbxl_cal, rng=rng, n_boot=n_boot, n_perm=n_perm,
        collapse_4c_to_2c=True,
    )
    cross_hard_4c = _score_4c_predictions(
        y_ptbxl_4c, y_pred_ptbxl_hard, rng=rng, n_boot=n_boot, n_perm=n_perm,
    )
    cross_hard_2c = _score_4c_predictions(
        y_ptbxl_4c, y_pred_ptbxl_hard, rng=rng, n_boot=n_boot, n_perm=n_perm,
        collapse_4c_to_2c=True,
    )

    return {
        "calibrator_name": cal_name,
        "calibrator_cv_scores": cv_scores,
        "phi_4c_outer_boundary_rad": PHI_4C_OUTER_BOUNDARY,
        "phi_4c_inner_boundary_rad": PHI_4C_INNER_BOUNDARY,
        "in_domain_calibrator_4c": in_domain_cal,
        "in_domain_hardcoded_4c": in_domain_hard,
        "cross_calibrator_4c": cross_cal_4c,
        "cross_calibrator_2c": cross_cal_2c,
        "cross_hardcoded_4c": cross_hard_4c,
        "cross_hardcoded_2c": cross_hard_2c,
        # Internal arrays for downstream plotting / diagnostics:
        "_phi_pred_ptbxl": phi_pred_ptbxl,
        "_y_pred_ptbxl_cal": y_pred_ptbxl_cal,
        "_y_pred_ptbxl_hard": y_pred_ptbxl_hard,
    }


def plot_phi_pred_by_territory_4c(
    phi_pred: np.ndarray,
    territory_truth_4c: np.ndarray,
    title: str,
    save_path: Path,
    bins: int = 36,
) -> None:
    """4-panel overlaid histogram of phi_pred conditioned on territory_4c truth.

    Vertical guides mark the hardcoded wedge boundaries (phi = 0, +/- 2 rad)
    and the +/- pi wrap. Each panel shows the empirical distribution of phi_pred
    for one true territory, with class size in the legend label.
    """
    colors = {
        "Anteroseptal":   "#d62728",
        "Anterolateral":  "#ff7f0e",
        "Inferior":       "#1f77b4",
        "Inferolateral":  "#2ca02c",
    }
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    phi_pred = np.arctan2(np.sin(phi_pred), np.cos(phi_pred))
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    for t in TERRITORIES_4C:
        mask = territory_truth_4c == t
        if mask.sum() == 0:
            continue
        ax.hist(
            phi_pred[mask], bins=edges, alpha=0.55,
            color=colors[t], edgecolor="black", linewidth=0.4,
            label=f"{t} (n={int(mask.sum())})",
        )
    for x, lbl in [(-2.0, "-2 rad"), (0.0, "0"), (2.0, "+2 rad")]:
        ax.axvline(x, color="grey", linestyle="--", linewidth=0.8)
        ax.text(x, ax.get_ylim()[1] * 0.95, lbl, rotation=90, fontsize=7,
                color="grey", va="top", ha="right")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xlabel("predicted phi (radians)")
    ax.set_ylabel("count")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Section 3.4 / In-domain 8-class audit (MedalCare only).
#
# Trains a multinomial LogReg on Z_MedalCare_train -> territory_8c and scores
# it on MedalCare_test under three lenses: full 8c, 4c anatomy collapse, 2c
# transmurality collapse. This is purely in-domain -- no cross-domain transfer.
# Used to support the publishable claim that fine-tuned Z encodes the 8-class
# anatomy x transmurality structure that lives in the MedalCare folder names.
# ---------------------------------------------------------------------------

def fit_territory_8c_classifier(
    X_train_std: np.ndarray,
    y_train_8c: np.ndarray,
    Cs: np.ndarray = LOGREG_CS_TERR_8C,
    max_iter: int = 4000,
    seed: int = SEED,
) -> Tuple[LogisticRegression, float, Dict[str, float]]:
    """Same recipe as ``fit_territory_4c_classifier`` but for 8c labels."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores: Dict[str, float] = {}
    for C in Cs:
        fold_scores: List[float] = []
        for tr_idx, va_idx in skf.split(X_train_std, y_train_8c):
            est = LogisticRegression(
                C=C, penalty="l2", solver="lbfgs",
                class_weight="balanced", max_iter=max_iter,
                multi_class="multinomial",
            )
            est.fit(X_train_std[tr_idx], y_train_8c[tr_idx])
            y_va_pred = est.predict(X_train_std[va_idx])
            fold_scores.append(
                f1_score(
                    y_train_8c[va_idx], y_va_pred,
                    labels=TERRITORIES_8C, average="macro", zero_division=0,
                )
            )
        cv_scores[str(C)] = float(np.mean(fold_scores))

    best_C = float(max(cv_scores, key=cv_scores.get))
    model = LogisticRegression(
        C=best_C, penalty="l2", solver="lbfgs",
        class_weight="balanced", max_iter=max_iter,
        multi_class="multinomial",
    )
    model.fit(X_train_std, y_train_8c)
    return model, best_C, cv_scores


def in_domain_8c_for_source(
    src_name: str,
    X_train_std: np.ndarray,
    X_test_std: np.ndarray,
    y_train_8c: np.ndarray,
    y_test_8c: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    n_perm: int,
) -> Dict[str, object]:
    """Fit 8c classifier on (X_train, y_train_8c) and score on test under
    three collapses: 8c (full), 4c (anatomy only), 2c-transmurality.
    """
    print(f"      [8c-audit / {src_name}] fitting 8-class LogReg on n_train={X_train_std.shape[0]}")
    model, best_C, cv_scores = fit_territory_8c_classifier(X_train_std, y_train_8c)
    print(
        f"      [8c-audit / {src_name}] best_C={best_C:g}; cv_macro_f1={cv_scores[str(best_C)]:.3f}"
    )
    y_pred_8c = model.predict(X_test_std)
    full_8c = _score_predictions(
        y_test_8c, y_pred_8c, rng=rng, labels=TERRITORIES_8C,
        n_boot=n_boot, n_perm=n_perm,
    )
    anat_4c = _score_predictions(
        y_test_8c, y_pred_8c, rng=rng, labels=TERRITORIES_8C,
        n_boot=n_boot, n_perm=n_perm,
        collapse_map=TERRITORY_8C_TO_4C, collapse_labels=TERRITORIES_4C,
    )
    trans_2c = _score_predictions(
        y_test_8c, y_pred_8c, rng=rng, labels=TERRITORIES_8C,
        n_boot=n_boot, n_perm=n_perm,
        collapse_map=TERRITORY_8C_TO_TRANS, collapse_labels=TRANSMURALITY_LABELS,
    )
    return {
        "best_C": float(best_C),
        "cv_scores_per_C": cv_scores,
        "in_domain_8c": full_8c,
        "in_domain_4c_anatomy": anat_4c,
        "in_domain_2c_transmurality": trans_2c,
    }


def confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    title: str,
    save_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted territory")
    ax.set_ylabel("Ground-truth territory (PTB-XL SCP-derived)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            colour = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=colour, fontsize=12)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _strip_internal(d: Dict) -> Dict:
    """Drop keys starting with underscore (internal predictions used downstream)."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--n-perm-binary", type=int, default=N_PERM_BINARY)
    parser.add_argument(
        "--configs", type=str, default=",".join(CONFIGS),
        help="Comma-separated subset of configs to run.",
    )
    parser.add_argument(
        "--no-polar", action="store_true",
        help="Skip polar plots (they're slow on huge n).",
    )
    parser.add_argument(
        "--no-cross-domain", action="store_true",
        help="Skip Section 7 (B2-CD) cross-domain evaluation on PTB-XL.",
    )
    parser.add_argument(
        "--cross-domain-only", action="store_true",
        help="Skip the in-domain analysis; useful for quick CD iteration after "
             "in-domain run already produced in_domain.json. Still needs to refit "
             "the phi probe in-process to obtain the model.",
    )
    parser.add_argument(
        "--latent-suffix", type=str, default="",
        help="Suffix appended to latent directory names (e.g. '_inlp' reads from "
             "outputs/latents/{stem}_{split}_inlp/). When non-empty, output JSONs "
             "and plots are routed to outputs/phase_b2{suffix}/ to avoid "
             "overwriting the original-latent results.",
    )
    parser.add_argument(
        "--no-pipeline-a", action="store_true",
        help="Skip the 4-class direct-territory classifier (Pipeline A).",
    )
    parser.add_argument(
        "--no-pipeline-b", action="store_true",
        help="Skip the calibrated phi->4c pipeline (Pipeline B).",
    )
    parser.add_argument(
        "--no-pipeline-8c", action="store_true",
        help="Skip the in-domain 8-class audit (Section 3.4).",
    )
    args = parser.parse_args()

    # Resolve output directory based on --latent-suffix. When the user passes
    # --latent-suffix _inlp, we write to outputs/phase_b2_inlp/ unless they
    # explicitly overrode --out.
    suffix = args.latent_suffix
    explicit_out = args.out != OUT_JSON  # user changed --out
    if suffix and not explicit_out:
        out_dir = REPO_ROOT / "outputs" / f"phase_b2{suffix}"
        args.out = out_dir / "in_domain.json"
    else:
        out_dir = args.out.parent
    out_cd_path = out_dir / "cross_domain.json"
    out_a_path = out_dir / "cross_domain_4c_pipelineA.json"

    out_b_path = out_dir / "cross_domain_4c_pipelineB.json"
    out_8c_path = out_dir / "in_domain_8c.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    rng = np.random.default_rng(SEED)
    if suffix:
        print(f"[latent-suffix] reading from outputs/latents/.../{{split}}{suffix}/")
        print(f"[latent-suffix] writing to {out_dir.relative_to(REPO_ROOT)}/")

    # Load shared data
    targets = load_targets()
    feat_train_full, feat_test_full, _, _, feature_names = load_features()
    idx_train = targets["train"]["idx_in_split"]
    idx_test = targets["test"]["idx_in_split"]
    phi_train = targets["train"]["phi"]
    z_train = targets["train"]["z"]
    size_train = targets["train"]["size"]
    rho_train = targets["train"]["rho_eps_max"]
    phi_test = targets["test"]["phi"]
    z_test = targets["test"]["z"]
    size_test = targets["test"]["size"]
    rho_test = targets["test"]["rho_eps_max"]
    transmurality_test = targets["test"]["transmural"]

    # Pipeline-A / Pipeline-B 4-class territory labels (added 2026-05-13).
    do_pipeline_a = not args.no_pipeline_a
    do_pipeline_b_local = not args.no_pipeline_b
    do_pipeline_8c_local = not args.no_pipeline_8c
    need_4c = do_pipeline_a or do_pipeline_b_local
    if need_4c:
        territory_4c_train = targets["train"].get("territory_4c")
        territory_4c_test = targets["test"].get("territory_4c")
        if territory_4c_train is None or territory_4c_test is None:
            raise RuntimeError(
                "theta_mi_*.npz missing 'territory_4c' key -- rebuild via "
                "scripts/build_medalcare_isch_targets.py before running Pipeline A."
            )
        # Convert object arrays to plain numpy str arrays for stable use downstream.
        territory_4c_train = np.array(territory_4c_train.tolist(), dtype=object)
        territory_4c_test = np.array(territory_4c_test.tolist(), dtype=object)
        print(
            f"[pipeline-A] MedalCare territory_4c train counts: "
            f"{dict((t, int((territory_4c_train == t).sum())) for t in TERRITORIES_4C)}"
        )
        print(
            f"[pipeline-A] MedalCare territory_4c test counts: "
            f"{dict((t, int((territory_4c_test == t).sum())) for t in TERRITORIES_4C)}"
        )

    # 8-class in-domain audit territory labels (Section 3.4, added 2026-05-14).
    if do_pipeline_8c_local:
        territory_8c_train = targets["train"].get("territory_8c")
        territory_8c_test = targets["test"].get("territory_8c")
        if territory_8c_train is None or territory_8c_test is None:
            raise RuntimeError(
                "theta_mi_*.npz missing 'territory_8c' key -- rebuild via "
                "scripts/build_medalcare_isch_targets.py before running the "
                "8-class audit."
            )
        territory_8c_train = np.array(territory_8c_train.tolist(), dtype=object)
        territory_8c_test = np.array(territory_8c_test.tolist(), dtype=object)
        print(
            f"[8c-audit] MedalCare territory_8c train counts: "
            f"{dict((t, int((territory_8c_train == t).sum())) for t in TERRITORIES_8C)}"
        )
        print(
            f"[8c-audit] MedalCare territory_8c test counts: "
            f"{dict((t, int((territory_8c_test == t).sum())) for t in TERRITORIES_8C)}"
        )

    # Build the MI-only ECG-feature arrays once (impute later w/ train medians).
    F_train_mi = feat_train_full[idx_train].astype(np.float64)
    F_test_mi = feat_test_full[idx_test].astype(np.float64)

    # Median-impute features once (shared across configs).
    F_train_imp, F_test_imp, feat_medians, feat_train_pct, feat_test_pct = (
        median_impute_with_train_medians(F_train_mi, F_test_mi)
    )
    print(
        f"[features] per-feature train-imputed%: "
        f"{dict(zip(feature_names, feat_train_pct.round(1).tolist()))}"
    )
    print(
        f"[features] per-feature  test-imputed%: "
        f"{dict(zip(feature_names, feat_test_pct.round(1).tolist()))}"
    )

    # Standardise features once.
    feat_scaler = fit_scaler(F_train_imp)
    F_train_std = feat_scaler.transform(F_train_imp)
    F_test_std = feat_scaler.transform(F_test_imp)

    # Cross-domain: load PTB-XL ground truth + features once.
    do_cd = not args.no_cross_domain
    if do_cd or need_4c:
        print("\n[cross-domain] loading PTB-XL ground-truth subclass CSV + features...")
        ptbxl_subclass_df = load_ptbxl_subclass_csv()
        # Build the single-territory primary subset (Anterior / Inferior / Lateral only).
        primary_mask = ptbxl_subclass_df["territory"].isin(TERRITORY_LABELS)
        primary_df = ptbxl_subclass_df[primary_mask].copy()
        primary_idx = primary_df["row_idx"].to_numpy()  # row index into the FULL PTB-XL test fold
        primary_truth = primary_df["territory"].to_numpy()
        print(
            f"[cross-domain] primary (3c) subset n={primary_idx.size}; per-territory: "
            f"{dict(primary_df['territory'].value_counts())}"
        )

        # PTB-XL ECG features for NK2 baseline (impute with MedalCare-train medians for
        # consistency with in-domain analysis).
        feat_ptbxl_full, _ = load_ptbxl_features()
        feat_ptbxl_primary = feat_ptbxl_full[primary_idx].astype(np.float64)
        # Impute using MedalCare-train medians already computed above.
        for j in range(feat_ptbxl_primary.shape[1]):
            nan_mask = np.isnan(feat_ptbxl_primary[:, j])
            feat_ptbxl_primary[nan_mask, j] = feat_medians[j]

    if need_4c:
        # Pipeline-A / Pipeline-B primary 4-class subset (n=438 in fold10).
        primary_4c_mask = ptbxl_subclass_df["territory_4c"].isin(TERRITORIES_4C)
        primary_4c_df = ptbxl_subclass_df[primary_4c_mask].copy()
        primary_4c_idx = primary_4c_df["row_idx"].to_numpy()
        primary_4c_truth = primary_4c_df["territory_4c"].to_numpy()
        primary_4c_truth_2c = primary_4c_df["territory_2c"].to_numpy()
        per_class_4c = dict(primary_4c_df["territory_4c"].value_counts())
        per_class_2c = dict(primary_4c_df["territory_2c"].value_counts())
        print(
            f"[pipeline-A] PTB-XL 4c primary subset n={primary_4c_idx.size}; "
            f"per-territory_4c: {per_class_4c}; per-territory_2c: {per_class_2c}"
        )
        feat_ptbxl_primary_4c = feat_ptbxl_full[primary_4c_idx].astype(np.float64)
        for j in range(feat_ptbxl_primary_4c.shape[1]):
            nan_mask = np.isnan(feat_ptbxl_primary_4c[:, j])
            feat_ptbxl_primary_4c[nan_mask, j] = feat_medians[j]

    results: Dict[str, Dict[str, Dict[str, object]]] = {}
    cd_results: Dict[str, Dict[str, object]] = {}
    pipeline_a_results: Dict[str, Dict[str, object]] = {}
    pipeline_b_results: Dict[str, Dict[str, object]] = {}
    audit_8c_results: Dict[str, Dict[str, object]] = {}
    do_pipeline_b = do_pipeline_b_local
    do_pipeline_8c = do_pipeline_8c_local

    for cfg in configs:
        print(f"\n{'='*60}\n[CONFIG] {cfg}{(' [' + suffix + ']') if suffix else ''}\n{'='*60}")
        Z_train_full, Z_test_full = load_config_latents(cfg, suffix=suffix)
        Z_train_mi = Z_train_full[idx_train].astype(np.float64)
        Z_test_mi = Z_test_full[idx_test].astype(np.float64)

        # Standardise Z per-config (latent scale differs across configs).
        z_scaler = fit_scaler(Z_train_mi)
        Z_train_std = z_scaler.transform(Z_train_mi)
        Z_test_std = z_scaler.transform(Z_test_mi)

        cfg_result: Dict[str, Dict[str, object]] = {"Z": {}, "ecg_features": {}, "paired_Z_vs_features": {}}

        sources = {
            "Z": (Z_train_std, Z_test_std),
            "ecg_features": (F_train_std, F_test_std),
        }

        # Run all four targets per source.
        for src_name, (X_tr, X_te) in sources.items():
            print(f"  [{cfg} / {src_name}]  X_train={X_tr.shape}, X_test={X_te.shape}")

            print(f"    -> phi (sin/cos Ridge)")
            cfg_result[src_name]["phi"] = fit_ridge_phi(
                X_tr, X_te, phi_train, phi_test, rng,
                n_boot=args.n_boot, n_perm=args.n_perm,
            )

            print(f"    -> z (Ridge)")
            cfg_result[src_name]["z"] = fit_ridge_continuous(
                X_tr, X_te, z_train, z_test, rng,
                n_boot=args.n_boot, n_perm=args.n_perm,
            )

            print(f"    -> size (Ridge)")
            cfg_result[src_name]["size"] = fit_ridge_continuous(
                X_tr, X_te, size_train, size_test, rng,
                n_boot=args.n_boot, n_perm=args.n_perm,
            )

            print(f"    -> rho_eps_max (Logistic)")
            cfg_result[src_name]["rho_eps_max"] = fit_logistic_binary(
                X_tr, X_te, rho_train, rho_test, rng,
                n_boot=args.n_boot, n_perm=args.n_perm_binary,
            )

        # Paired bootstrap: Z vs ecg_features per target.
        cfg_result["paired_Z_vs_features"]["phi_circular_r2"] = paired_bootstrap_circular(
            phi_test,
            cfg_result["Z"]["phi"]["_phi_pred"],
            cfg_result["ecg_features"]["phi"]["_phi_pred"],
            cfg_result["Z"]["phi"]["_phi_train_mean"],
            rng, n_boot=args.n_boot,
        )
        for tgt, y_te in [("z", z_test), ("size", size_test)]:
            cfg_result["paired_Z_vs_features"][f"{tgt}_r2"] = paired_bootstrap_continuous(
                y_te,
                cfg_result["Z"][tgt]["_y_pred"],
                cfg_result["ecg_features"][tgt]["_y_pred"],
                rng, metric_fn=lambda y_true, y_pred: r2_score(y_true, y_pred),
                n_boot=args.n_boot, higher_is_better=True,
            )
        cfg_result["paired_Z_vs_features"]["rho_eps_max_auc"] = paired_bootstrap_continuous(
            (rho_test > 0.5).astype(int),
            cfg_result["Z"]["rho_eps_max"]["_y_score"],
            cfg_result["ecg_features"]["rho_eps_max"]["_y_score"],
            rng,
            metric_fn=lambda y_true, y_score: roc_auc_score(y_true, y_score)
            if len(np.unique(y_true)) > 1 else np.nan,
            n_boot=args.n_boot, higher_is_better=True,
        )

        # Polar plot for the phi predictions on Z.
        if not args.no_polar:
            polar_path = out_dir / f"polar_{cfg}.png"
            polar_plot(
                phi_test,
                cfg_result["Z"]["phi"]["_phi_pred"],
                transmurality_test,
                title=f"Phase B2 — phi prediction (Z), config={cfg}\n"
                      f"circular R²={cfg_result['Z']['phi']['circular_r2']:.3f}, "
                      f"|err|_med={cfg_result['Z']['phi']['circular_mae_deg']:.1f}°",
                save_path=polar_path,
            )
            print(f"    saved {polar_path}")

        # ---- Section 7: cross-domain (B2-CD) ---------------------------
        if do_cd:
            print(f"  [{cfg}] CROSS-DOMAIN (PTB-XL primary subset)")
            Z_ptbxl_full = load_ptbxl_latents(cfg, suffix=suffix)
            Z_ptbxl_primary = Z_ptbxl_full[primary_idx].astype(np.float64)
            cd_z = cross_domain_phi_eval(
                phi_model=cfg_result["Z"]["phi"]["_model"],
                scaler=z_scaler,
                X_ptbxl=Z_ptbxl_primary,
                territory_truth=primary_truth,
                rng=rng, n_boot=args.n_boot, n_perm=args.n_perm_binary,
                label=f"{cfg}/Z",
            )
            cd_f = cross_domain_phi_eval(
                phi_model=cfg_result["ecg_features"]["phi"]["_model"],
                scaler=feat_scaler,
                X_ptbxl=feat_ptbxl_primary,
                territory_truth=primary_truth,
                rng=rng, n_boot=args.n_boot, n_perm=args.n_perm_binary,
                label=f"{cfg}/ecg_features",
            )
            cd_results[cfg] = {"Z": cd_z, "ecg_features": cd_f}

            # Confusion matrix plot for Z (primary 3-class).
            cm_path = out_dir / f"cm_{cfg}.png"
            confusion_matrix_plot(
                primary_truth, cd_z["_territory_pred"], TERRITORY_LABELS,
                title=(
                    f"Phase B2-CD — PTB-XL territory CM (Z), config={cfg}\n"
                    f"3-class macro-F1 = {cd_z['primary_3class']['macro_f1']:.3f} "
                    f"[{cd_z['primary_3class']['macro_f1_ci95'][0]:.3f}, "
                    f"{cd_z['primary_3class']['macro_f1_ci95'][1]:.3f}]  "
                    f"p_perm = {cd_z['primary_3class']['permutation_p_macro_f1']:.4f}"
                ),
                save_path=cm_path,
            )
            print(f"    saved {cm_path}")
            print(
                f"    macro-F1: Z={cd_z['primary_3class']['macro_f1']:.3f} "
                f"vs features={cd_f['primary_3class']['macro_f1']:.3f}  | "
                f"2-class Ant-vs-Inf macro-F1: "
                f"Z={cd_z['sensitivity_2class_AntInf']['macro_f1']:.3f} "
                f"vs features={cd_f['sensitivity_2class_AntInf']['macro_f1']:.3f}"
            )

        # ---- Section 8: Pipeline A — direct 4-class territory classifier ----
        if do_pipeline_a:
            print(f"  [{cfg}] PIPELINE A (direct 4c territory classifier)")
            # Reuse Z_ptbxl_full from CD block if it was already loaded; else load it now.
            if do_cd:
                Z_ptbxl_full_a = Z_ptbxl_full  # noqa: F821 -- defined inside the CD block above
            else:
                Z_ptbxl_full_a = load_ptbxl_latents(cfg, suffix=suffix)
            Z_ptbxl_primary_4c = Z_ptbxl_full_a[primary_4c_idx].astype(np.float64)
            Z_ptbxl_primary_4c_std = z_scaler.transform(Z_ptbxl_primary_4c)
            feat_ptbxl_primary_4c_std = feat_scaler.transform(feat_ptbxl_primary_4c)

            pa_z = pipeline_a_for_source(
                src_name=f"{cfg}/Z",
                X_train_std=Z_train_std,
                X_test_std=Z_test_std,
                X_ptbxl_std=Z_ptbxl_primary_4c_std,
                y_train_4c=territory_4c_train,
                y_test_4c=territory_4c_test,
                y_ptbxl_4c=primary_4c_truth,
                rng=rng, n_boot=args.n_boot, n_perm=args.n_perm_binary,
            )
            pa_f = pipeline_a_for_source(
                src_name=f"{cfg}/ecg_features",
                X_train_std=F_train_std,
                X_test_std=F_test_std,
                X_ptbxl_std=feat_ptbxl_primary_4c_std,
                y_train_4c=territory_4c_train,
                y_test_4c=territory_4c_test,
                y_ptbxl_4c=primary_4c_truth,
                rng=rng, n_boot=args.n_boot, n_perm=args.n_perm_binary,
            )
            pipeline_a_results[cfg] = {"Z": pa_z, "ecg_features": pa_f}

            # Confusion-matrix plot (4c, Z only).
            cm_a_path = out_dir / f"cm_A_4c_{cfg}.png"
            confusion_matrix_plot(
                pa_z["cross_domain_4c"]["_y_true"],
                pa_z["cross_domain_4c"]["_y_pred"],
                TERRITORIES_4C,
                title=(
                    f"Pipeline A 4c — PTB-XL territory CM (Z), config={cfg}\n"
                    f"macro-F1 = {pa_z['cross_domain_4c']['macro_f1']:.3f} "
                    f"[{pa_z['cross_domain_4c']['macro_f1_ci95'][0]:.3f}, "
                    f"{pa_z['cross_domain_4c']['macro_f1_ci95'][1]:.3f}]  "
                    f"p_perm = {pa_z['cross_domain_4c']['permutation_p_macro_f1']:.4f}"
                ),
                save_path=cm_a_path,
            )
            # 2c collapse CM for the same model.
            cm_a_2c_path = out_dir / f"cm_A_2c_{cfg}.png"
            confusion_matrix_plot(
                pa_z["cross_domain_2c"]["_y_true"],
                pa_z["cross_domain_2c"]["_y_pred"],
                TERRITORIES_2C,
                title=(
                    f"Pipeline A 2c (Ant-vs-Inf collapse) — PTB-XL CM (Z), config={cfg}\n"
                    f"macro-F1 = {pa_z['cross_domain_2c']['macro_f1']:.3f}  "
                    f"p_perm = {pa_z['cross_domain_2c']['permutation_p_macro_f1']:.4f}"
                ),
                save_path=cm_a_2c_path,
            )
            print(f"    saved {cm_a_path} and {cm_a_2c_path}")
            print(
                f"    in-domain 4c macro-F1: "
                f"Z={pa_z['in_domain_4c']['macro_f1']:.3f} "
                f"vs NK2={pa_f['in_domain_4c']['macro_f1']:.3f}  | "
                f"cross-domain 4c macro-F1: "
                f"Z={pa_z['cross_domain_4c']['macro_f1']:.3f} "
                f"vs NK2={pa_f['cross_domain_4c']['macro_f1']:.3f}  | "
                f"cross-domain 2c macro-F1: "
                f"Z={pa_z['cross_domain_2c']['macro_f1']:.3f} "
                f"vs NK2={pa_f['cross_domain_2c']['macro_f1']:.3f}"
            )

        # ---- Section 9: Pipeline B -- calibrated phi-bins -> 4-class ----
        if do_pipeline_b:
            print(f"  [{cfg}] PIPELINE B (calibrated phi-bins -> territory_4c)")
            # Phi predictions on MedalCare TEST already cached in cfg_result.
            phi_pred_test_z = cfg_result["Z"]["phi"]["_phi_pred"]
            phi_pred_test_f = cfg_result["ecg_features"]["phi"]["_phi_pred"]

            # Phi predictions on PTB-XL primary 4c subset -- predict fresh
            # using the cached phi-Ridge model + the already-standardised inputs.
            if not do_pipeline_a:
                # If Pipeline A was disabled, Z_ptbxl_full / *_std are not yet
                # built; load + standardise now so Pipeline B is independent.
                Z_ptbxl_full_b = load_ptbxl_latents(cfg, suffix=suffix)
                Z_ptbxl_primary_4c = Z_ptbxl_full_b[primary_4c_idx].astype(np.float64)
                Z_ptbxl_primary_4c_std = z_scaler.transform(Z_ptbxl_primary_4c)
                feat_ptbxl_primary_4c_std = feat_scaler.transform(feat_ptbxl_primary_4c)

            phi_model_z = cfg_result["Z"]["phi"]["_model"]
            phi_model_f = cfg_result["ecg_features"]["phi"]["_model"]
            Y_z_ptbxl = phi_model_z.predict(Z_ptbxl_primary_4c_std)
            phi_pred_ptbxl_z = np.arctan2(Y_z_ptbxl[:, 0], Y_z_ptbxl[:, 1])
            Y_f_ptbxl = phi_model_f.predict(feat_ptbxl_primary_4c_std)
            phi_pred_ptbxl_f = np.arctan2(Y_f_ptbxl[:, 0], Y_f_ptbxl[:, 1])

            pb_z = pipeline_b_for_source(
                src_name=f"{cfg}/Z",
                phi_pred_test=phi_pred_test_z,
                y_test_4c=territory_4c_test,
                phi_pred_ptbxl=phi_pred_ptbxl_z,
                y_ptbxl_4c=primary_4c_truth,
                rng=rng, n_boot=args.n_boot, n_perm=args.n_perm_binary,
            )
            pb_f = pipeline_b_for_source(
                src_name=f"{cfg}/ecg_features",
                phi_pred_test=phi_pred_test_f,
                y_test_4c=territory_4c_test,
                phi_pred_ptbxl=phi_pred_ptbxl_f,
                y_ptbxl_4c=primary_4c_truth,
                rng=rng, n_boot=args.n_boot, n_perm=args.n_perm_binary,
            )
            pipeline_b_results[cfg] = {"Z": pb_z, "ecg_features": pb_f}

            # Diagnostic histogram on Z (the headline regressor input).
            hist_path = out_dir / f"hist_predphi_by_territory_{cfg}.png"
            plot_phi_pred_by_territory_4c(
                phi_pred=phi_pred_ptbxl_z,
                territory_truth_4c=primary_4c_truth,
                title=(
                    f"Pipeline B diagnostic -- predicted phi on PTB-XL by truth_4c\n"
                    f"config={cfg}{suffix}  (phi-Ridge trained on MedalCare; "
                    f"vertical guides at +/-2 rad = hardcoded wedge boundaries)"
                ),
                save_path=hist_path,
            )
            # CMs: calibrator (Z) 4c + 2c; hardcoded (Z) 4c.
            cm_b_cal_path = out_dir / f"cm_B_cal_4c_{cfg}.png"
            confusion_matrix_plot(
                pb_z["cross_calibrator_4c"]["_y_true"],
                pb_z["cross_calibrator_4c"]["_y_pred"],
                TERRITORIES_4C,
                title=(
                    f"Pipeline B calibrator 4c -- PTB-XL CM (Z), "
                    f"calibrator={pb_z['calibrator_name']}, config={cfg}\n"
                    f"macro-F1 = {pb_z['cross_calibrator_4c']['macro_f1']:.3f} "
                    f"[{pb_z['cross_calibrator_4c']['macro_f1_ci95'][0]:.3f}, "
                    f"{pb_z['cross_calibrator_4c']['macro_f1_ci95'][1]:.3f}]  "
                    f"p_perm = {pb_z['cross_calibrator_4c']['permutation_p_macro_f1']:.4f}"
                ),
                save_path=cm_b_cal_path,
            )
            cm_b_hard_path = out_dir / f"cm_B_hard_4c_{cfg}.png"
            confusion_matrix_plot(
                pb_z["cross_hardcoded_4c"]["_y_true"],
                pb_z["cross_hardcoded_4c"]["_y_pred"],
                TERRITORIES_4C,
                title=(
                    f"Pipeline B hardcoded wedges 4c -- PTB-XL CM (Z), config={cfg}\n"
                    f"macro-F1 = {pb_z['cross_hardcoded_4c']['macro_f1']:.3f} "
                    f"[{pb_z['cross_hardcoded_4c']['macro_f1_ci95'][0]:.3f}, "
                    f"{pb_z['cross_hardcoded_4c']['macro_f1_ci95'][1]:.3f}]  "
                    f"p_perm = {pb_z['cross_hardcoded_4c']['permutation_p_macro_f1']:.4f}"
                ),
                save_path=cm_b_hard_path,
            )
            print(f"    saved {hist_path}")
            print(f"    saved {cm_b_cal_path} and {cm_b_hard_path}")
            print(
                f"    in-domain 4c (cal): Z={pb_z['in_domain_calibrator_4c']['macro_f1']:.3f}  | "
                f"cross 4c cal:   Z={pb_z['cross_calibrator_4c']['macro_f1']:.3f}  "
                f"hard: Z={pb_z['cross_hardcoded_4c']['macro_f1']:.3f}  | "
                f"cross 2c cal:   Z={pb_z['cross_calibrator_2c']['macro_f1']:.3f}  "
                f"hard: Z={pb_z['cross_hardcoded_2c']['macro_f1']:.3f}  "
                f"(NK2 cal CD 4c={pb_f['cross_calibrator_4c']['macro_f1']:.3f})"
            )

        # ---- Section 10: 8-class in-domain audit (MedalCare only) ----
        if do_pipeline_8c:
            print(f"  [{cfg}] 8-CLASS AUDIT (MedalCare in-domain only)")
            audit_z = in_domain_8c_for_source(
                src_name=f"{cfg}/Z",
                X_train_std=Z_train_std,
                X_test_std=Z_test_std,
                y_train_8c=territory_8c_train,
                y_test_8c=territory_8c_test,
                rng=rng, n_boot=args.n_boot, n_perm=args.n_perm_binary,
            )
            audit_f = in_domain_8c_for_source(
                src_name=f"{cfg}/ecg_features",
                X_train_std=F_train_std,
                X_test_std=F_test_std,
                y_train_8c=territory_8c_train,
                y_test_8c=territory_8c_test,
                rng=rng, n_boot=args.n_boot, n_perm=args.n_perm_binary,
            )
            audit_8c_results[cfg] = {"Z": audit_z, "ecg_features": audit_f}

            # 8x8 CM plot (Z source -- this is the publishable headline).
            cm_8c_path = out_dir / f"cm_8c_{cfg}.png"
            confusion_matrix_plot(
                audit_z["in_domain_8c"]["_y_true"],
                audit_z["in_domain_8c"]["_y_pred"],
                TERRITORIES_8C,
                title=(
                    f"8-class audit -- MedalCare test CM (Z), config={cfg}\n"
                    f"8c macro-F1 = {audit_z['in_domain_8c']['macro_f1']:.3f}  | "
                    f"4c anatomy collapse = {audit_z['in_domain_4c_anatomy']['macro_f1']:.3f}  | "
                    f"2c transmurality = {audit_z['in_domain_2c_transmurality']['macro_f1']:.3f}"
                ),
                save_path=cm_8c_path,
            )
            print(f"    saved {cm_8c_path}")
            print(
                f"    Z 8c macro-F1 = {audit_z['in_domain_8c']['macro_f1']:.3f} "
                f"[{audit_z['in_domain_8c']['macro_f1_ci95'][0]:.3f}, "
                f"{audit_z['in_domain_8c']['macro_f1_ci95'][1]:.3f}]  | "
                f"4c anatomy = {audit_z['in_domain_4c_anatomy']['macro_f1']:.3f}  | "
                f"2c trans = {audit_z['in_domain_2c_transmurality']['macro_f1']:.3f}  "
                f"(NK2 8c = {audit_f['in_domain_8c']['macro_f1']:.3f})"
            )

        # Strip internal-only keys before serialising.
        for src in cfg_result:
            if src in SOURCES:
                for tgt in cfg_result[src]:
                    cfg_result[src][tgt] = _strip_internal(cfg_result[src][tgt])
        results[cfg] = cfg_result

    payload = {
        "metadata": {
            "n_train_mi": int(idx_train.size),
            "n_test_mi": int(idx_test.size),
            "configs": configs,
            "sources": SOURCES,
            "targets": ["phi", "z", "size", "rho_eps_max"],
            "n_bootstrap": int(args.n_boot),
            "n_permutation_ridge": int(args.n_perm),
            "n_permutation_logistic": int(args.n_perm_binary),
            "ridge_alphas": RIDGE_ALPHAS.tolist(),
            "logreg_Cs": LOGREG_CS.tolist(),
            "feature_names": feature_names,
            "feature_train_imputed_pct": feat_train_pct.tolist(),
            "feature_test_imputed_pct": feat_test_pct.tolist(),
            "feature_train_medians": feat_medians.tolist(),
            "seed": SEED,
        },
        "results": results,
    }

    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[done] wrote {args.out}")

    # Quick console summary (in-domain).
    print("\n=== In-domain summary (Z source) ===")
    print(f"{'config':<14}  {'phi_R²':>9} ({'CI95':>13})  {'phi_MAE°':>8}  "
          f"{'z_R²':>7}  {'size_R²':>7}  {'rho_AUC':>7}")
    for cfg in configs:
        z_phi = results[cfg]["Z"]["phi"]
        z_z = results[cfg]["Z"]["z"]
        z_sz = results[cfg]["Z"]["size"]
        z_rho = results[cfg]["Z"]["rho_eps_max"]
        print(
            f"{cfg:<14}  {z_phi['circular_r2']:>9.3f} "
            f"({z_phi['circular_r2_ci95'][0]:>5.3f},{z_phi['circular_r2_ci95'][1]:>5.3f})  "
            f"{z_phi['circular_mae_deg']:>8.1f}  "
            f"{z_z['r2']:>7.3f}  {z_sz['r2']:>7.3f}  {z_rho['auc']:>7.3f}"
        )

    # Cross-domain JSON + summary.
    if do_cd:
        # Strip internal keys before dumping.
        cd_payload_results: Dict[str, Dict[str, object]] = {}
        for cfg, srcs in cd_results.items():
            cd_payload_results[cfg] = {}
            for src_name, eval_dict in srcs.items():
                cd_payload_results[cfg][src_name] = _strip_internal(eval_dict)

        cd_payload = {
            "metadata": {
                "configs": configs,
                "sources": SOURCES,
                "primary_territories": TERRITORY_LABELS,
                "phi_bin_boundary": PHI_BIN_BOUNDARY,
                "n_primary_total": int(primary_idx.size),
                "n_per_class_truth": {
                    t: int((primary_truth == t).sum()) for t in TERRITORY_LABELS
                },
                "n_bootstrap": int(args.n_boot),
                "n_permutation_macro_f1": int(args.n_perm_binary),
                "ptbxl_subclass_csv": str(PTBXL_SUBCLASS_PATH.relative_to(REPO_ROOT)),
                "ptbxl_features_npz": str(FEAT_PTBXL_PATH.relative_to(REPO_ROOT)),
                "seed": SEED,
            },
            "results": cd_payload_results,
        }
        out_cd_path.write_text(json.dumps(cd_payload, indent=2), encoding="utf-8")
        print(f"\n[done] wrote {out_cd_path}")

        print("\n=== Cross-domain summary (3-class macro-F1) ===")
        print(
            f"{'config':<14}  "
            f"{'Z macro-F1 [CI95]':>26}  {'Z p_perm':>9}  "
            f"{'NK2 macro-F1 [CI95]':>26}  "
            f"{'2cls Z F1':>10}  {'2cls NK2 F1':>12}"
        )
        for cfg in configs:
            cd = cd_results[cfg]
            z3 = cd["Z"]["primary_3class"]
            f3 = cd["ecg_features"]["primary_3class"]
            z2 = cd["Z"]["sensitivity_2class_AntInf"]
            f2 = cd["ecg_features"]["sensitivity_2class_AntInf"]
            print(
                f"{cfg:<14}  "
                f"{z3['macro_f1']:>6.3f} [{z3['macro_f1_ci95'][0]:>5.3f},{z3['macro_f1_ci95'][1]:>5.3f}]  "
                f"{z3['permutation_p_macro_f1']:>9.4f}  "
                f"{f3['macro_f1']:>6.3f} [{f3['macro_f1_ci95'][0]:>5.3f},{f3['macro_f1_ci95'][1]:>5.3f}]  "
                f"{z2['macro_f1']:>10.3f}  {f2['macro_f1']:>12.3f}"
            )

    # Pipeline A JSON + summary.
    if do_pipeline_a:
        pa_payload_results: Dict[str, Dict[str, object]] = {}
        for cfg, srcs in pipeline_a_results.items():
            pa_payload_results[cfg] = {}
            for src_name, leg in srcs.items():
                pa_payload_results[cfg][src_name] = {
                    "best_C": leg["best_C"],
                    "cv_scores_per_C": leg["cv_scores_per_C"],
                    "in_domain_4c": _strip_internal(leg["in_domain_4c"]),
                    "cross_domain_4c": _strip_internal(leg["cross_domain_4c"]),
                    "cross_domain_2c": _strip_internal(leg["cross_domain_2c"]),
                }

        pa_payload = {
            "metadata": {
                "configs": configs,
                "sources": SOURCES,
                "territories_4c": TERRITORIES_4C,
                "territories_2c": TERRITORIES_2C,
                "territory_4c_to_2c": TERRITORY_4C_TO_2C,
                "n_train_medalcare_4c": {
                    t: int((territory_4c_train == t).sum()) for t in TERRITORIES_4C
                },
                "n_test_medalcare_4c": {
                    t: int((territory_4c_test == t).sum()) for t in TERRITORIES_4C
                },
                "n_ptbxl_primary_4c": int(primary_4c_idx.size),
                "n_per_class_truth_ptbxl_4c": {
                    t: int((primary_4c_truth == t).sum()) for t in TERRITORIES_4C
                },
                "n_per_class_truth_ptbxl_2c": {
                    t: int((primary_4c_truth_2c == t).sum()) for t in TERRITORIES_2C
                },
                "logreg_Cs": LOGREG_CS_TERR_4C.tolist(),
                "class_weight": "balanced",
                "multi_class": "multinomial",
                "solver": "lbfgs",
                "max_iter": 4000,
                "internal_cv": "StratifiedKFold(5, shuffle=True)",
                "n_bootstrap": int(args.n_boot),
                "n_permutation_macro_f1": int(args.n_perm_binary),
                "ptbxl_subclass_csv": str(PTBXL_SUBCLASS_PATH.relative_to(REPO_ROOT)),
                "seed": SEED,
            },
            "results": pa_payload_results,
        }
        out_a_path.parent.mkdir(parents=True, exist_ok=True)
        out_a_path.write_text(json.dumps(pa_payload, indent=2), encoding="utf-8")
        print(f"\n[done] wrote {out_a_path}")

        print("\n=== Pipeline A summary (cross-domain 4-class macro-F1) ===")
        print(
            f"{'config':<14}  "
            f"{'inD Z F1':>9}  {'inD NK2 F1':>11}  "
            f"{'CD Z F1 [CI95]':>22}  {'CD Z p':>7}  "
            f"{'CD NK2 F1 [CI95]':>22}  "
            f"{'CD-2c Z F1':>10}  {'CD-2c NK2 F1':>12}"
        )
        for cfg in configs:
            pa = pipeline_a_results[cfg]
            z_in = pa["Z"]["in_domain_4c"]["macro_f1"]
            f_in = pa["ecg_features"]["in_domain_4c"]["macro_f1"]
            zcd = pa["Z"]["cross_domain_4c"]
            fcd = pa["ecg_features"]["cross_domain_4c"]
            zcd2 = pa["Z"]["cross_domain_2c"]["macro_f1"]
            fcd2 = pa["ecg_features"]["cross_domain_2c"]["macro_f1"]
            print(
                f"{cfg:<14}  "
                f"{z_in:>9.3f}  {f_in:>11.3f}  "
                f"{zcd['macro_f1']:>6.3f} [{zcd['macro_f1_ci95'][0]:>5.3f},{zcd['macro_f1_ci95'][1]:>5.3f}]  "
                f"{zcd['permutation_p_macro_f1']:>7.4f}  "
                f"{fcd['macro_f1']:>6.3f} [{fcd['macro_f1_ci95'][0]:>5.3f},{fcd['macro_f1_ci95'][1]:>5.3f}]  "
                f"{zcd2:>10.3f}  {fcd2:>12.3f}"
            )

    # Pipeline B JSON + summary.
    if do_pipeline_b:
        pb_payload_results: Dict[str, Dict[str, object]] = {}
        for cfg, srcs in pipeline_b_results.items():
            pb_payload_results[cfg] = {}
            for src_name, leg in srcs.items():
                pb_payload_results[cfg][src_name] = {
                    "calibrator_name": leg["calibrator_name"],
                    "calibrator_cv_scores": leg["calibrator_cv_scores"],
                    "phi_4c_outer_boundary_rad": leg["phi_4c_outer_boundary_rad"],
                    "phi_4c_inner_boundary_rad": leg["phi_4c_inner_boundary_rad"],
                    "in_domain_calibrator_4c": _strip_internal(leg["in_domain_calibrator_4c"]),
                    "in_domain_hardcoded_4c": _strip_internal(leg["in_domain_hardcoded_4c"]),
                    "cross_calibrator_4c": _strip_internal(leg["cross_calibrator_4c"]),
                    "cross_calibrator_2c": _strip_internal(leg["cross_calibrator_2c"]),
                    "cross_hardcoded_4c": _strip_internal(leg["cross_hardcoded_4c"]),
                    "cross_hardcoded_2c": _strip_internal(leg["cross_hardcoded_2c"]),
                }

        pb_payload = {
            "metadata": {
                "configs": configs,
                "sources": SOURCES,
                "territories_4c": TERRITORIES_4C,
                "territories_2c": TERRITORIES_2C,
                "territory_4c_to_2c": TERRITORY_4C_TO_2C,
                "phi_4c_outer_boundary_rad": PHI_4C_OUTER_BOUNDARY,
                "phi_4c_inner_boundary_rad": PHI_4C_INNER_BOUNDARY,
                "calibrator_candidates": ["tree_d4", "logreg_l2", "knn_10"],
                "calibrator_cv": "StratifiedKFold(5, shuffle=True)",
                "n_train_medalcare_4c": {
                    t: int((territory_4c_train == t).sum()) for t in TERRITORIES_4C
                },
                "n_test_medalcare_4c": {
                    t: int((territory_4c_test == t).sum()) for t in TERRITORIES_4C
                },
                "n_ptbxl_primary_4c": int(primary_4c_idx.size),
                "n_per_class_truth_ptbxl_4c": {
                    t: int((primary_4c_truth == t).sum()) for t in TERRITORIES_4C
                },
                "n_per_class_truth_ptbxl_2c": {
                    t: int((primary_4c_truth_2c == t).sum()) for t in TERRITORIES_2C
                },
                "n_bootstrap": int(args.n_boot),
                "n_permutation_macro_f1": int(args.n_perm_binary),
                "ptbxl_subclass_csv": str(PTBXL_SUBCLASS_PATH.relative_to(REPO_ROOT)),
                "seed": SEED,
            },
            "results": pb_payload_results,
        }
        out_b_path.parent.mkdir(parents=True, exist_ok=True)
        out_b_path.write_text(json.dumps(pb_payload, indent=2), encoding="utf-8")
        print(f"\n[done] wrote {out_b_path}")

        print("\n=== Pipeline B summary (calibrator vs hardcoded, cross-domain 4c) ===")
        print(
            f"{'config':<14}  {'cal':>9}  "
            f"{'inD-cal Z':>9}  "
            f"{'CD-cal Z F1 [CI95]':>22}  {'p_cal':>6}  "
            f"{'CD-hard Z F1 [CI95]':>22}  {'p_hard':>6}  "
            f"{'delta(cal-hard) 4c':>18}  "
            f"{'CD2-cal Z':>9}  {'CD2-hard Z':>10}"
        )
        for cfg in configs:
            pb = pipeline_b_results[cfg]
            zcal = pb["Z"]["cross_calibrator_4c"]
            zhard = pb["Z"]["cross_hardcoded_4c"]
            zcal2 = pb["Z"]["cross_calibrator_2c"]["macro_f1"]
            zhard2 = pb["Z"]["cross_hardcoded_2c"]["macro_f1"]
            zind = pb["Z"]["in_domain_calibrator_4c"]["macro_f1"]
            print(
                f"{cfg:<14}  "
                f"{pb['Z']['calibrator_name']:>9}  "
                f"{zind:>9.3f}  "
                f"{zcal['macro_f1']:>6.3f} [{zcal['macro_f1_ci95'][0]:>5.3f},{zcal['macro_f1_ci95'][1]:>5.3f}]  "
                f"{zcal['permutation_p_macro_f1']:>6.4f}  "
                f"{zhard['macro_f1']:>6.3f} [{zhard['macro_f1_ci95'][0]:>5.3f},{zhard['macro_f1_ci95'][1]:>5.3f}]  "
                f"{zhard['permutation_p_macro_f1']:>6.4f}  "
                f"{zcal['macro_f1'] - zhard['macro_f1']:>+18.3f}  "
                f"{zcal2:>9.3f}  {zhard2:>10.3f}"
            )

    # 8-class audit JSON + summary.
    if do_pipeline_8c:
        a8_payload_results: Dict[str, Dict[str, object]] = {}
        for cfg, srcs in audit_8c_results.items():
            a8_payload_results[cfg] = {}
            for src_name, leg in srcs.items():
                a8_payload_results[cfg][src_name] = {
                    "best_C": leg["best_C"],
                    "cv_scores_per_C": leg["cv_scores_per_C"],
                    "in_domain_8c":               _strip_internal(leg["in_domain_8c"]),
                    "in_domain_4c_anatomy":       _strip_internal(leg["in_domain_4c_anatomy"]),
                    "in_domain_2c_transmurality": _strip_internal(leg["in_domain_2c_transmurality"]),
                }

        a8_payload = {
            "metadata": {
                "configs": configs,
                "sources": SOURCES,
                "territories_8c": TERRITORIES_8C,
                "territories_4c": TERRITORIES_4C,
                "transmurality_labels": TRANSMURALITY_LABELS,
                "territory_8c_to_4c": TERRITORY_8C_TO_4C,
                "territory_8c_to_transmurality": TERRITORY_8C_TO_TRANS,
                "n_train_medalcare_8c": {
                    t: int((territory_8c_train == t).sum()) for t in TERRITORIES_8C
                },
                "n_test_medalcare_8c": {
                    t: int((territory_8c_test == t).sum()) for t in TERRITORIES_8C
                },
                "logreg_Cs": LOGREG_CS_TERR_8C.tolist(),
                "class_weight": "balanced",
                "multi_class": "multinomial",
                "solver": "lbfgs",
                "max_iter": 4000,
                "internal_cv": "StratifiedKFold(5, shuffle=True)",
                "n_bootstrap": int(args.n_boot),
                "n_permutation_macro_f1": int(args.n_perm_binary),
                "seed": SEED,
            },
            "results": a8_payload_results,
        }
        out_8c_path.parent.mkdir(parents=True, exist_ok=True)
        out_8c_path.write_text(json.dumps(a8_payload, indent=2), encoding="utf-8")
        print(f"\n[done] wrote {out_8c_path}")

        print("\n=== 8-class audit summary (in-domain MedalCare) ===")
        print(
            f"{'config':<14}  "
            f"{'8c Z F1 [CI95]':>22}  {'p_perm':>7}  "
            f"{'4c anat Z F1':>13}  {'2c trans Z F1':>14}  "
            f"{'8c NK2 F1':>9}  {'4c NK2 F1':>9}  {'2c NK2 F1':>9}"
        )
        for cfg in configs:
            a = audit_8c_results[cfg]
            z8 = a["Z"]["in_domain_8c"]
            z4 = a["Z"]["in_domain_4c_anatomy"]["macro_f1"]
            z2 = a["Z"]["in_domain_2c_transmurality"]["macro_f1"]
            f8 = a["ecg_features"]["in_domain_8c"]["macro_f1"]
            f4 = a["ecg_features"]["in_domain_4c_anatomy"]["macro_f1"]
            f2 = a["ecg_features"]["in_domain_2c_transmurality"]["macro_f1"]
            print(
                f"{cfg:<14}  "
                f"{z8['macro_f1']:>6.3f} [{z8['macro_f1_ci95'][0]:>5.3f},{z8['macro_f1_ci95'][1]:>5.3f}]  "
                f"{z8['permutation_p_macro_f1']:>7.4f}  "
                f"{z4:>13.3f}  {z2:>14.3f}  "
                f"{f8:>9.3f}  {f4:>9.3f}  {f2:>9.3f}"
            )


if __name__ == "__main__":
    main()
