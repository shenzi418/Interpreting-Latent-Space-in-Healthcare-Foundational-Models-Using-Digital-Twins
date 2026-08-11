"""TRACK 1 / Section 1a -- PCA dimension scan over Z.

For each (config, pca_mode) pair, fit StandardScaler + PCA on the *train*
pool, then for each K in [1024, 512, 256, 128, 64, 32, 16, 8] project the
*test* pool (alignment + class structure) and the MedalCare-MI train/test
pools (mechanism) to K-d and recompute:

  Alignment (test pool, subsampled 2000 per domain):
    - MMD single-bandwidth (median heuristic)
    - MMD multi-bandwidth (sum of 5 RBFs at sigma in {0.25,0.5,1,2,4} * median)
    - C2ST AUROC (5-fold LogReg, AAAI 2020 underpowered observer at high D)
    - kNN-5 mixing (cosine)

  Class structure (test pool, 3-class shared remap):
    - KMeans-3 acc/NMI/ARI (combined / medalcare / ptbxl)
    - LR M->P macro-AUC + per-class AUC; LR P->M
    - kNN-5 M->P / P->M
    - Cosine intra/inter/cross-domain gaps

  Mechanism (MedalCare-MI only, in-domain):
    - phi circular R^2 (sin/cos Ridge)
    - z R^2, size R^2 (Ridge)
    - rho_eps_max AUC (LogReg)

Per-K JSON output and a 4-panel frontier PNG (MMD / C2ST / LR M->P / phi
circular R^2 vs K) per pca_mode, with both configs overlaid.

Pre-registered K* (plan section 1a step 5): smallest K with
  C2ST <= 0.85 AND LR M->P >= 0.65 AND phi circular R^2 >= 0.35
fallback: best LR M->P among K with C2ST <= 0.95.

Usage
-----
    python analysis/dim_scan.py \\
        --configs exp7_baseline exp7_ccmmd \\
        --pca-modes combined medalcare ptbxl \\
        --ks 1024 512 256 128 64 32 16 8 \\
        --out outputs/dim_scan

Outputs
-------
    outputs/dim_scan/{config}_summary_{mode}.json     (per-config x mode)
    outputs/dim_scan/frontier_{mode}.png              (per mode, both configs)
    outputs/dim_scan/kstar_table.json                 (cross-config x mode K*)

Notes
-----
* Bootstrap n_boot=200, permutation n_perm=50 (reduced from 1000 in
  phase_b2_infarct_decoding.py) since we are sweeping 24+ (config, mode, K)
  cells -- central point estimates drive K* selection, CIs are a sanity
  check at the 1024-d row only.
* PCA fitted once per (config, mode) with n_components=1024, then sliced
  Z_proj[:, :K] for each K -- avoids re-fitting at every K.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.exp7_analysis import (  # noqa: E402
    domain_classifier_auc,
    knn_mixing_score,
    mmd_rbf,
)
from analysis.exp7_full_evaluation import (  # noqa: E402
    MEDALCARE_KEEP,
    MEDALCARE_REMAP,
    PTBXL_KEEP,
    PTBXL_REMAP,
    SHARED_CLASSES,
    argmax_label,
    cross_domain_probe,
    cross_domain_knn,
    kmeans_analysis,
    pairwise_cosine_sim,
)
from analysis.phase_b2_infarct_decoding import (  # noqa: E402
    fit_logistic_binary,
    fit_ridge_continuous,
    fit_ridge_phi,
    load_targets,
)


# ---------------------------------------------------------------------------
# Constants + config table
# ---------------------------------------------------------------------------

DEFAULT_KS: Tuple[int, ...] = (1024, 512, 256, 128, 64, 32, 16, 8)
DEFAULT_CONFIGS: Tuple[str, ...] = ("exp7_baseline", "exp7_ccmmd")
DEFAULT_PCA_MODES: Tuple[str, ...] = ("combined", "medalcare", "ptbxl")

# Reduced permutation budget for the sweep -- central R^2/AUC drive K*.
SCAN_N_BOOT = 200
SCAN_N_PERM = 50
SCAN_N_PERM_BINARY = 30
SUBSAMPLE_ALIGNMENT = 2000  # max per domain for MMD/C2ST/kNN
SEED = 42

# Latent layout: outputs/latents/{stem}_{split}/latents.npz, where stem maps
# config -> on-disk stem (mirrors phase_b2_infarct_decoding.CONFIG_LATENT_STEMS
# and exp7_full_evaluation.CONFIGS).
CONFIG_LATENT_STEMS: Dict[str, str] = {
    "exp7_baseline": "exp7",
    "exp7_ccmmd": "exp7_ccmmd",
    "exp5_3class": "exp5_3class",
    "exp6_3class": "exp6_3class",
}

LATENT_ROOT = REPO_ROOT / "outputs" / "latents"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.keys()}


def load_config_4splits(config: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Load Z/Y for {medal_train, medal_test, ptb_train, ptb_test}.

    Returns dict keyed by split -> {'Z': (N,D), 'Y': (N,Ck)}.
    """
    stem = CONFIG_LATENT_STEMS[config]
    splits = {
        "medal_train": LATENT_ROOT / f"{stem}_medalcare_train" / "latents.npz",
        "medal_test":  LATENT_ROOT / f"{stem}_medalcare"       / "latents.npz",
        "ptb_train":   LATENT_ROOT / f"{stem}_ptbxl_train"     / "latents.npz",
        "ptb_test":    LATENT_ROOT / f"{stem}_ptbxl"           / "latents.npz",
    }
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for name, path in splits.items():
        if not path.exists():
            raise FileNotFoundError(f"missing latents: {path}")
        d = _load_npz(path)
        out[name] = {"Z": d["Z"].astype(np.float64), "Y": d["Y"].astype(np.float32)}
    return out


def remap_test_split(Y_orig: np.ndarray, domain: str) -> Tuple[np.ndarray, np.ndarray]:
    """3-class shared remap + valid-row mask. Returns (mask, Y_shared)."""
    if domain == "medalcare":
        remap, keep = MEDALCARE_REMAP, MEDALCARE_KEEP
    else:
        remap, keep = PTBXL_REMAP, PTBXL_KEEP
    n_orig = Y_orig.shape[1]
    valid_cols = [c for c in keep if c < n_orig]
    mask = np.zeros(len(Y_orig), dtype=bool)
    for col in valid_cols:
        mask |= Y_orig[:, col] > 0.5
    Y_shared_full = np.zeros((len(Y_orig), 3), dtype=np.float32)
    for src, tgt in remap.items():
        if src < n_orig:
            Y_shared_full[:, tgt] = np.clip(Y_shared_full[:, tgt] + Y_orig[:, src], 0, 1)
    return mask, Y_shared_full


# ---------------------------------------------------------------------------
# Multi-bandwidth MMD (numpy)
# ---------------------------------------------------------------------------

def mmd_multibandwidth(
    X: np.ndarray,
    Y: np.ndarray,
    sigma_scales: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> float:
    """Sum of unbiased RBF MMD^2 at sigma_k = scale_k * median pairwise distance."""
    XY = np.vstack([X, Y])
    pair_sq = cdist(XY, XY, "sqeuclidean")
    pos = pair_sq[pair_sq > 0]
    if pos.size == 0:
        return 0.0
    median_sq = float(np.median(pos))
    sigma_med = float(np.sqrt(median_sq / 2.0))
    nx, ny = len(X), len(Y)
    total = 0.0
    for s in sigma_scales:
        sigma = sigma_med * s
        gamma = 1.0 / (2.0 * sigma * sigma)
        K = np.exp(-gamma * pair_sq)
        Kxx = K[:nx, :nx]
        Kyy = K[nx:, nx:]
        Kxy = K[:nx, nx:]
        mmd2 = (
            (Kxx.sum() - np.trace(Kxx)) / (nx * (nx - 1))
            + (Kyy.sum() - np.trace(Kyy)) / (ny * (ny - 1))
            - 2.0 * Kxy.mean()
        )
        total += float(mmd2)
    return total


# ---------------------------------------------------------------------------
# Alignment block
# ---------------------------------------------------------------------------

def alignment_block(
    Z_med: np.ndarray, Z_ptb: np.ndarray, *, rng: np.random.Generator, seed: int,
) -> Dict[str, float]:
    """MMD / MMD-mb / C2ST / kNN-mix on test latents (already PCA-projected)."""
    sub = min(SUBSAMPLE_ALIGNMENT, len(Z_med), len(Z_ptb))
    idx_m = rng.choice(len(Z_med), sub, replace=False) if len(Z_med) > sub else np.arange(len(Z_med))
    idx_p = rng.choice(len(Z_ptb), sub, replace=False) if len(Z_ptb) > sub else np.arange(len(Z_ptb))
    A = Z_med[idx_m]
    B = Z_ptb[idx_p]
    return {
        "mmd_median": float(mmd_rbf(A, B, sigma=None)),
        "mmd_multibw": float(mmd_multibandwidth(A, B)),
        "c2st_auc": float(domain_classifier_auc(A, B, seed=seed)),
        "knn_mixing": float(knn_mixing_score(A, B, k=5)),
        "n_sub_per_domain": int(sub),
    }


# ---------------------------------------------------------------------------
# Class-structure block
# ---------------------------------------------------------------------------

def class_structure_block(
    Z_med_test: np.ndarray,
    Y_med_test: np.ndarray,
    Z_ptb_test: np.ndarray,
    Y_ptb_test: np.ndarray,
    *,
    seed: int,
) -> Dict[str, object]:
    labels_m = argmax_label(Y_med_test)
    labels_p = argmax_label(Y_ptb_test)
    Z_all = np.vstack([Z_med_test, Z_ptb_test])
    labels_all = np.concatenate([labels_m, labels_p])
    km = {
        "combined":  kmeans_analysis(Z_all, labels_all, k=3, seed=seed),
        "medalcare": kmeans_analysis(Z_med_test, labels_m, k=3, seed=seed),
        "ptbxl":     kmeans_analysis(Z_ptb_test, labels_p, k=3, seed=seed),
    }
    lr_m2p = cross_domain_probe(Z_med_test, labels_m, Z_ptb_test, labels_p, seed=seed)
    lr_p2m = cross_domain_probe(Z_ptb_test, labels_p, Z_med_test, labels_m, seed=seed)
    knn5_m2p = cross_domain_knn(Z_med_test, labels_m, Z_ptb_test, labels_p, k=5)
    knn5_p2m = cross_domain_knn(Z_ptb_test, labels_p, Z_med_test, labels_m, k=5)
    # Cosine gaps -- standardise jointly first (small re-standardisation
    # AFTER PCA is harmless, makes intra/inter comparable across K).
    Z_all_sc = StandardScaler().fit_transform(Z_all)
    Zm = Z_all_sc[: len(Z_med_test)]
    Zp = Z_all_sc[len(Z_med_test):]
    intra_vals: List[float] = []
    inter_vals: List[float] = []
    cross_vals: List[float] = []
    for i in range(3):
        m_i = Zm[labels_m == i]
        p_i = Zp[labels_p == i]
        if len(m_i) >= 2:
            intra_vals.append(pairwise_cosine_sim(m_i, m_i, seed=seed))
        if len(p_i) >= 2:
            intra_vals.append(pairwise_cosine_sim(p_i, p_i, seed=seed))
        if len(m_i) >= 1 and len(p_i) >= 1:
            cross_vals.append(pairwise_cosine_sim(m_i, p_i, seed=seed))
        for j in range(i + 1, 3):
            m_j = Zm[labels_m == j]
            p_j = Zp[labels_p == j]
            if len(m_i) >= 1 and len(m_j) >= 1:
                inter_vals.append(pairwise_cosine_sim(m_i, m_j, seed=seed))
            if len(p_i) >= 1 and len(p_j) >= 1:
                inter_vals.append(pairwise_cosine_sim(p_i, p_j, seed=seed))
    avg_intra = float(np.mean(intra_vals)) if intra_vals else 0.0
    avg_inter = float(np.mean(inter_vals)) if inter_vals else 0.0
    avg_cross = float(np.mean(cross_vals)) if cross_vals else 0.0
    return {
        "kmeans": km,
        "lr_m2p": lr_m2p,
        "lr_p2m": lr_p2m,
        "knn5_m2p": float(knn5_m2p),
        "knn5_p2m": float(knn5_p2m),
        "cosine_summary": {
            "avg_intra_class": avg_intra,
            "avg_inter_class": avg_inter,
            "avg_cross_domain_same_class": avg_cross,
            "intra_inter_gap": avg_intra - avg_inter,
        },
    }


# ---------------------------------------------------------------------------
# Mechanism block (B2 in-domain on MI subset)
# ---------------------------------------------------------------------------

def mechanism_block(
    X_train_K: np.ndarray,
    X_test_K: np.ndarray,
    targets: Dict[str, Dict[str, np.ndarray]],
    rng: np.random.Generator,
) -> Dict[str, object]:
    phi_train = targets["train"]["phi"]
    z_train = targets["train"]["z"]
    size_train = targets["train"]["size"]
    rho_train = targets["train"]["rho_eps_max"]
    phi_test = targets["test"]["phi"]
    z_test = targets["test"]["z"]
    size_test = targets["test"]["size"]
    rho_test = targets["test"]["rho_eps_max"]

    out: Dict[str, object] = {}
    phi = fit_ridge_phi(
        X_train_K, X_test_K, phi_train, phi_test, rng,
        n_boot=SCAN_N_BOOT, n_perm=SCAN_N_PERM,
    )
    out["phi"] = {
        "circular_r2": float(phi["circular_r2"]),
        "circular_r2_ci95": [float(v) for v in phi["circular_r2_ci95"]],
        "circular_mae_deg": float(phi["circular_mae_deg"]),
        "permutation_p": float(phi["permutation_p_circular_r2"]),
        "alpha": float(phi["alpha"]),
    }
    for name, y_tr, y_te in (("z", z_train, z_test), ("size", size_train, size_test)):
        r = fit_ridge_continuous(
            X_train_K, X_test_K, y_tr, y_te, rng,
            n_boot=SCAN_N_BOOT, n_perm=SCAN_N_PERM,
        )
        out[name] = {
            "r2": float(r["r2"]),
            "r2_ci95": [float(v) for v in r["r2_ci95"]],
            "mae": float(r["mae"]),
            "permutation_p": float(r["permutation_p_r2"]),
            "alpha": float(r["alpha"]),
        }
    rho = fit_logistic_binary(
        X_train_K, X_test_K, rho_train, rho_test, rng,
        n_boot=SCAN_N_BOOT, n_perm=SCAN_N_PERM_BINARY,
    )
    out["rho_eps_max"] = {
        "auc": float(rho["auc"]),
        "auc_ci95": [float(v) for v in rho["auc_ci95"]],
        "permutation_p": float(rho["permutation_p_auc"]),
        "best_C": float(rho["best_C"]),
    }
    return out


# ---------------------------------------------------------------------------
# K* selection rule
# ---------------------------------------------------------------------------

def select_kstar(per_k: Dict[int, Dict[str, object]]) -> Dict[str, object]:
    """Smallest K with C2ST <= 0.85 AND LR M->P macro-AUC >= 0.65 AND phi R^2_circ >= 0.35.

    Fallback: best LR M->P among K with C2ST <= 0.95.
    """
    eligible: List[int] = []
    backup: List[Tuple[float, int]] = []
    for k in sorted(per_k.keys()):
        a = per_k[k]
        c2st = a["alignment"]["c2st_auc"]  # type: ignore[index]
        lr_m2p = a["class_structure"]["lr_m2p"]["macro_auc"]  # type: ignore[index]
        phi_cr2 = a["mechanism"]["phi"]["circular_r2"]  # type: ignore[index]
        if c2st <= 0.85 and lr_m2p >= 0.65 and phi_cr2 >= 0.35:
            eligible.append(k)
        if c2st <= 0.95:
            backup.append((lr_m2p, k))
    if eligible:
        k_star = min(eligible)
        rule = "primary"
    elif backup:
        # Best LR M->P among eligible
        backup.sort(key=lambda t: (-t[0], t[1]))
        k_star = backup[0][1]
        rule = "fallback"
    else:
        return {"value": None, "rule": "none-satisfied", "rule_pass": False}
    return {"value": int(k_star), "rule": rule, "rule_pass": rule == "primary"}


# ---------------------------------------------------------------------------
# Frontier figure
# ---------------------------------------------------------------------------

def render_frontier_figure(
    runs: Dict[str, Dict[int, Dict[str, object]]],
    out_path: Path,
    *,
    title_suffix: str = "",
) -> None:
    """4-panel: MMD / C2ST / LR M->P / phi R^2_circ vs K (both configs overlaid)."""
    panels = [
        ("alignment.mmd_median",                "MMD (single-bandwidth median)", "log"),
        ("alignment.c2st_auc",                  "C2ST AUROC (5-fold LogReg)",    "linear"),
        ("class_structure.lr_m2p.macro_auc",    "LR M->P macro-AUC",             "linear"),
        ("mechanism.phi.circular_r2",           "phi circular R^2 (in-domain)",  "linear"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes = axes.ravel()
    colors = {"exp7_baseline": "#1f77b4", "exp7_ccmmd": "#d62728"}
    threshold_lines = {
        "alignment.c2st_auc": (0.85, "C2ST<=0.85"),
        "class_structure.lr_m2p.macro_auc": (0.65, "LR>=0.65"),
        "mechanism.phi.circular_r2": (0.35, "phi R2>=0.35"),
    }
    for ax, (key, label, yscale) in zip(axes, panels):
        for cfg, per_k in runs.items():
            ks = sorted(per_k.keys())
            ys: List[float] = []
            for k in ks:
                d = per_k[k]
                node: object = d
                for part in key.split("."):
                    node = node[part]  # type: ignore[index]
                ys.append(float(node))  # type: ignore[arg-type]
            ax.plot(ks, ys, marker="o", color=colors.get(cfg, None), label=cfg)
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64, 128, 256, 512, 1024])
        ax.set_xticklabels(["8", "16", "32", "64", "128", "256", "512", "1024"])
        ax.set_xlabel("K (PCA components)")
        ax.set_ylabel(label)
        ax.set_yscale(yscale)
        if key in threshold_lines:
            thr, thr_label = threshold_lines[key]
            ax.axhline(thr, color="grey", linestyle="--", linewidth=0.8, label=thr_label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"Dimension scan frontier{title_suffix}", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main per-(config, mode) loop
# ---------------------------------------------------------------------------

def fit_standardiser_and_pca(
    Z_med_train: np.ndarray,
    Z_ptb_train: np.ndarray,
    *,
    pca_mode: str,
    seed: int,
) -> Tuple[StandardScaler, PCA, np.ndarray]:
    """Fit StandardScaler then PCA according to pca_mode. Returns (scaler, pca, evr_cumsum_full)."""
    if pca_mode == "combined":
        fit_data = np.vstack([Z_med_train, Z_ptb_train])
    elif pca_mode == "medalcare":
        fit_data = Z_med_train
    elif pca_mode == "ptbxl":
        fit_data = Z_ptb_train
    else:
        raise ValueError(f"unknown pca_mode={pca_mode}")
    scaler = StandardScaler().fit(fit_data)
    fit_sc = scaler.transform(fit_data)
    n_comp = min(fit_data.shape[1], fit_data.shape[0])
    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit(fit_sc)
    evr_cum = np.cumsum(pca.explained_variance_ratio_)
    return scaler, pca, evr_cum


def project_through(
    Z: np.ndarray, scaler: StandardScaler, pca: PCA,
) -> np.ndarray:
    return pca.transform(scaler.transform(Z))


def run_one(
    config: str,
    pca_mode: str,
    ks: Tuple[int, ...],
    splits: Dict[str, Dict[str, np.ndarray]],
    targets: Dict[str, Dict[str, np.ndarray]],
    *,
    seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    print(f"\n{'='*70}\n[CONFIG {config} / PCA {pca_mode}]\n{'='*70}")
    Z_m_tr = splits["medal_train"]["Z"]
    Z_m_te = splits["medal_test"]["Z"]
    Z_p_tr = splits["ptb_train"]["Z"]
    Z_p_te = splits["ptb_test"]["Z"]
    Y_m_te = splits["medal_test"]["Y"]
    Y_p_te = splits["ptb_test"]["Y"]

    # Valid-row masks for the 3-class shared remap.
    mask_m, Y_m_shared_full = remap_test_split(Y_m_te, "medalcare")
    mask_p, Y_p_shared_full = remap_test_split(Y_p_te, "ptbxl")
    Z_m_te_v = Z_m_te[mask_m]
    Y_m_te_v = Y_m_shared_full[mask_m]
    Z_p_te_v = Z_p_te[mask_p]
    Y_p_te_v = Y_p_shared_full[mask_p]
    print(
        f"[shared3] medalcare test {int(mask_m.sum())}/{len(mask_m)}, "
        f"ptbxl test {int(mask_p.sum())}/{len(mask_p)}"
    )

    # MI subset of MedalCare via theta_mi idx_in_split.
    idx_train = targets["train"]["idx_in_split"]
    idx_test = targets["test"]["idx_in_split"]
    print(f"[mi] medalcare train MI={idx_train.size}, test MI={idx_test.size}")

    # Fit standardiser + PCA once.
    t0 = time.time()
    scaler, pca, evr_cum = fit_standardiser_and_pca(
        Z_m_tr, Z_p_tr, pca_mode=pca_mode, seed=seed,
    )
    print(f"[pca] fitted PCA({pca.n_components_}) on {pca_mode} pool ({time.time()-t0:.1f}s)")

    # Project once at full rank; slice [:, :K] per K.
    Z_m_tr_full = project_through(Z_m_tr, scaler, pca)
    Z_p_tr_full = project_through(Z_p_tr, scaler, pca)
    Z_m_te_v_full = project_through(Z_m_te_v, scaler, pca)
    Z_p_te_v_full = project_through(Z_p_te_v, scaler, pca)
    Z_m_tr_mi_full = Z_m_tr_full[idx_train]
    Z_m_te_mi_full = project_through(Z_m_te, scaler, pca)[idx_test]

    per_k: Dict[int, Dict[str, object]] = {}
    evr_at_k: Dict[str, float] = {}
    for k in sorted(ks, reverse=True):  # large->small for log readability
        if k > pca.n_components_:
            print(f"[skip] K={k} > n_components_={pca.n_components_}")
            continue
        tk0 = time.time()
        Z_m_te_k = Z_m_te_v_full[:, :k]
        Z_p_te_k = Z_p_te_v_full[:, :k]
        Z_m_tr_mi_k = Z_m_tr_mi_full[:, :k]
        Z_m_te_mi_k = Z_m_te_mi_full[:, :k]

        align = alignment_block(Z_m_te_k, Z_p_te_k, rng=rng, seed=seed)
        class_s = class_structure_block(
            Z_m_te_k, Y_m_te_v, Z_p_te_k, Y_p_te_v, seed=seed,
        )
        mech = mechanism_block(Z_m_tr_mi_k, Z_m_te_mi_k, targets, rng)
        per_k[k] = {
            "alignment": align,
            "class_structure": class_s,
            "mechanism": mech,
            "elapsed_s": float(time.time() - tk0),
        }
        evr_at_k[str(k)] = float(evr_cum[k - 1])
        print(
            f"  K={k:>4d}  evr={evr_cum[k-1]:.3f}  "
            f"MMD={align['mmd_median']:.4f}  C2ST={align['c2st_auc']:.3f}  "
            f"LR_M2P={class_s['lr_m2p']['macro_auc']:.3f}  "
            f"phi_R2c={mech['phi']['circular_r2']:.3f}  "
            f"({time.time()-tk0:.1f}s)"
        )

    k_star = select_kstar(per_k)
    return {
        "config": config,
        "pca_mode": pca_mode,
        "n_train_pool": int(Z_m_tr.shape[0] + Z_p_tr.shape[0]) if pca_mode == "combined"
                        else int(Z_m_tr.shape[0] if pca_mode == "medalcare" else Z_p_tr.shape[0]),
        "n_test_pool_valid": int(Z_m_te_v.shape[0] + Z_p_te_v.shape[0]),
        "n_mi_train": int(idx_train.size),
        "n_mi_test": int(idx_test.size),
        "explained_variance_ratio_cumsum_at_K": evr_at_k,
        "ks": [k for k in sorted(per_k.keys(), reverse=True)],
        "per_K": {str(k): per_k[k] for k in per_k},
        "k_star": k_star,
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PCA dimension scan over latent Z (Track 1 / 1a).")
    p.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS),
                   choices=list(CONFIG_LATENT_STEMS.keys()))
    p.add_argument("--pca-modes", nargs="+", default=list(DEFAULT_PCA_MODES),
                   choices=["combined", "medalcare", "ptbxl"])
    p.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS))
    p.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "dim_scan")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def _rel(p: Path) -> str:
    """Render a path relative to REPO_ROOT if possible, else absolute."""
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def main() -> None:
    args = parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    targets = load_targets()

    # Pre-load all (config, splits) -- avoids re-reading on each pca_mode.
    splits_by_config: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for cfg in args.configs:
        splits_by_config[cfg] = load_config_4splits(cfg)
        print(
            f"[load] {cfg}: "
            f"M_tr={splits_by_config[cfg]['medal_train']['Z'].shape}, "
            f"M_te={splits_by_config[cfg]['medal_test']['Z'].shape}, "
            f"P_tr={splits_by_config[cfg]['ptb_train']['Z'].shape}, "
            f"P_te={splits_by_config[cfg]['ptb_test']['Z'].shape}"
        )

    # Run sweep.
    summary_table: List[Dict[str, object]] = []
    for mode in args.pca_modes:
        runs_for_frontier: Dict[str, Dict[int, Dict[str, object]]] = {}
        for cfg in args.configs:
            res = run_one(
                cfg, mode, tuple(args.ks),
                splits_by_config[cfg], targets, seed=args.seed,
            )
            out_path = args.out / f"{cfg}_summary_{mode}.json"
            out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
            print(f"[save] {_rel(out_path)} ({out_path.stat().st_size} B)")
            runs_for_frontier[cfg] = {int(k): v for k, v in res["per_K"].items()}
            summary_table.append({
                "config": cfg,
                "pca_mode": mode,
                "k_star": res["k_star"],
            })
        # Frontier figure per mode (both configs overlaid).
        fig_path = args.out / f"frontier_{mode}.png"
        render_frontier_figure(runs_for_frontier, fig_path, title_suffix=f" (PCA fit pool: {mode})")
        print(f"[save] {_rel(fig_path)}")

    # Cross-mode K* table.
    kstar_path = args.out / "kstar_table.json"
    kstar_path.write_text(json.dumps(summary_table, indent=2), encoding="utf-8")
    print(f"\n[done] kstar_table -> {_rel(kstar_path)}")
    for row in summary_table:
        ks_obj = row["k_star"]
        ks_val = ks_obj.get("value") if isinstance(ks_obj, dict) else None
        ks_rule = ks_obj.get("rule") if isinstance(ks_obj, dict) else None
        print(f"  {row['config']:<16} {row['pca_mode']:<10} K*={ks_val} ({ks_rule})")


if __name__ == "__main__":
    main()
