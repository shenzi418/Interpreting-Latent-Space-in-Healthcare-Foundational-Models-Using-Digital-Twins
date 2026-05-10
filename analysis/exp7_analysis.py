"""Exp 7 Shared-Head Latent Space Analysis.

Loads Exp 7 latent features from both domains, remaps to 3 shared classes
(NORM, MI, CD), and produces:
  1. PCA scatter plots colored by domain
  2. PCA scatter plots colored by shared class (within each domain & combined)
  3. Domain alignment metrics (MMD, kNN mixing, domain classifier AUC)
  4. Per-class domain alignment (class-conditional metrics)
  5. Class separability probes (logistic regression AUC, Fisher LDA)

Usage:
  python analysis/exp7_analysis.py --outdir outputs/exp7_latent_analysis
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LATENT_DIR = REPO_ROOT / "outputs" / "latents"

SHARED_CLASSES = ["NORM", "MI", "CD"]
SHARED_COLORS = {"NORM": "#2ca02c", "MI": "#d62728", "CD": "#1f77b4"}
DOMAIN_COLORS = {"MedalCare": "#1f77b4", "PTB-XL": "#d62728"}

MEDALCARE_REMAP = {0: 0, 1: 1, 2: 2, 3: 2, 5: 2, 7: 2}
MEDALCARE_KEEP = {0, 1, 2, 3, 5, 7}
PTBXL_REMAP = {0: 0, 1: 1, 4: 2}
PTBXL_KEEP = {0, 1, 4}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 7 latent space analysis.")
    p.add_argument("--outdir", type=Path, default=REPO_ROOT / "outputs" / "exp7_latent_analysis")
    p.add_argument("--latent-dir", type=Path, default=LATENT_DIR)
    p.add_argument("--prefix", type=str, default="exp7",
                   help="Subdirectory prefix: looks for {prefix}_medalcare/ and {prefix}_ptbxl/ under latent-dir")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_and_remap(latent_dir: Path, domain: str, prefix: str = "exp7") -> Tuple[np.ndarray, np.ndarray]:
    """Load latents and remap multi-label Y to 3-class shared labels, filtering irrelevant samples."""
    npz = np.load(latent_dir / f"{prefix}_{domain}" / "latents.npz")
    Z = npz["Z"]
    Y_orig = npz["Y"]

    if domain == "medalcare":
        remap, keep = MEDALCARE_REMAP, MEDALCARE_KEEP
    else:
        remap, keep = PTBXL_REMAP, PTBXL_KEEP

    mask = np.zeros(len(Y_orig), dtype=bool)
    for col in keep:
        mask |= (Y_orig[:, col] > 0.5)

    Z_filt = Z[mask]
    Y_orig_filt = Y_orig[mask]

    Y_shared = np.zeros((len(Z_filt), 3), dtype=np.float32)
    for src, tgt in remap.items():
        Y_shared[:, tgt] = np.clip(Y_shared[:, tgt] + Y_orig_filt[:, src], 0, 1)

    return Z_filt, Y_shared


def argmax_label(Y_shared: np.ndarray) -> np.ndarray:
    """Convert multi-hot to single argmax label for visualization/clustering."""
    return np.argmax(Y_shared, axis=1)


# ---------------------------------------------------------------------------
# PCA Visualization
# ---------------------------------------------------------------------------

def plot_pca_domain(Z_medal, Z_ptb, outdir: Path, dpi: int):
    """PCA scatter: combined data colored by domain."""
    Z_all = np.vstack([Z_medal, Z_ptb])
    domains = np.array(["MedalCare"] * len(Z_medal) + ["PTB-XL"] * len(Z_ptb))

    scaler = StandardScaler()
    Z_sc = scaler.fit_transform(Z_all)
    pca = PCA(n_components=2, random_state=42)
    Z2d = pca.fit_transform(Z_sc)

    fig, ax = plt.subplots(figsize=(8, 6))
    for domain, color in DOMAIN_COLORS.items():
        mask = domains == domain
        ax.scatter(Z2d[mask, 0], Z2d[mask, 1], c=color, alpha=0.3, s=8, label=domain, rasterized=True)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Exp 7 Shared-Head: Latent Space by Domain")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(outdir / "pca_domain.png", dpi=dpi)
    plt.close(fig)
    print(f"  Saved pca_domain.png (var explained: {pca.explained_variance_ratio_[:2].sum():.1%})")
    return pca, scaler


def plot_pca_class(Z_medal, Y_medal, Z_ptb, Y_ptb, outdir: Path, dpi: int):
    """PCA scatter: colored by shared class, separate subplots per domain + combined."""
    Z_all = np.vstack([Z_medal, Z_ptb])
    Y_all = np.vstack([Y_medal, Y_ptb])
    labels_all = argmax_label(Y_all)
    domains = np.array(["MedalCare"] * len(Z_medal) + ["PTB-XL"] * len(Z_ptb))

    scaler = StandardScaler()
    Z_sc = scaler.fit_transform(Z_all)
    pca = PCA(n_components=2, random_state=42)
    Z2d = pca.fit_transform(Z_sc)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax_idx, (title, mask_fn) in enumerate([
        ("MedalCare", lambda d: d == "MedalCare"),
        ("PTB-XL", lambda d: d == "PTB-XL"),
        ("Combined", lambda d: np.ones(len(d), dtype=bool)),
    ]):
        ax = axes[ax_idx]
        m = mask_fn(domains)
        for cls_idx, cls_name in enumerate(SHARED_CLASSES):
            cls_mask = m & (labels_all == cls_idx)
            ax.scatter(Z2d[cls_mask, 0], Z2d[cls_mask, 1],
                       c=SHARED_COLORS[cls_name], alpha=0.3, s=8, label=cls_name, rasterized=True)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title(f"Exp 7 — {title}")
        ax.legend(markerscale=3)

    fig.suptitle("Latent Space by Shared Class (NORM / MI / CD)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "pca_class.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved pca_class.png")


def plot_pca_class_domain_overlay(Z_medal, Y_medal, Z_ptb, Y_ptb, outdir: Path, dpi: int):
    """Per-class PCA showing MedalCare vs PTB-XL overlap for each shared class."""
    Z_all = np.vstack([Z_medal, Z_ptb])
    Y_all = np.vstack([Y_medal, Y_ptb])
    labels_all = argmax_label(Y_all)
    domains = np.array(["MedalCare"] * len(Z_medal) + ["PTB-XL"] * len(Z_ptb))

    scaler = StandardScaler()
    Z_sc = scaler.fit_transform(Z_all)
    pca = PCA(n_components=2, random_state=42)
    Z2d = pca.fit_transform(Z_sc)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for cls_idx, cls_name in enumerate(SHARED_CLASSES):
        ax = axes[cls_idx]
        cls_mask = labels_all == cls_idx
        for domain, color in DOMAIN_COLORS.items():
            m = cls_mask & (domains == domain)
            ax.scatter(Z2d[m, 0], Z2d[m, 1], c=color, alpha=0.3, s=10, label=domain, rasterized=True)
        ax.set_title(f"{cls_name} — Domain Overlap")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.legend(markerscale=3)

    fig.suptitle("Exp 7: Per-Class Domain Overlap in Latent Space", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "pca_class_domain_overlay.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved pca_class_domain_overlay.png")


# ---------------------------------------------------------------------------
# Domain Alignment Metrics
# ---------------------------------------------------------------------------

def mmd_rbf(X: np.ndarray, Y: np.ndarray, sigma: float = None) -> float:
    """Unbiased MMD^2 estimate with RBF kernel (median heuristic for bandwidth)."""
    from scipy.spatial.distance import cdist
    XY = np.vstack([X, Y])
    dists = cdist(XY, XY, "sqeuclidean")
    if sigma is None:
        median_dist = np.median(dists[dists > 0])
        sigma = np.sqrt(median_dist / 2)
    gamma = 1.0 / (2 * sigma ** 2)
    K = np.exp(-gamma * dists)

    nx, ny = len(X), len(Y)
    Kxx = K[:nx, :nx]
    Kyy = K[nx:, nx:]
    Kxy = K[:nx, nx:]

    mmd2 = (Kxx.sum() - np.trace(Kxx)) / (nx * (nx - 1)) \
          + (Kyy.sum() - np.trace(Kyy)) / (ny * (ny - 1)) \
          - 2 * Kxy.mean()
    return float(mmd2)


def knn_mixing_score(X: np.ndarray, Y: np.ndarray, k: int = 15) -> float:
    """Fraction of k-NN that come from the other domain (1 = perfectly mixed)."""
    XY = np.vstack([X, Y])
    labels = np.array([0] * len(X) + [1] * len(Y))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(XY)
    _, indices = nn.kneighbors(XY)
    indices = indices[:, 1:]  # drop self
    neighbor_labels = labels[indices]
    own_labels = labels[:, None]
    mixing = (neighbor_labels != own_labels).mean()
    return float(mixing)


def domain_classifier_auc(X: np.ndarray, Y: np.ndarray, seed: int = 42) -> float:
    """C2ST: 5-fold CV domain classifier AUC. 0.5 = domains indistinguishable."""
    XY = np.vstack([X, Y])
    labels = np.array([0] * len(X) + [1] * len(Y))
    scaler = StandardScaler()
    XY_sc = scaler.fit_transform(XY)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in skf.split(XY_sc, labels):
        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=seed)
        clf.fit(XY_sc[train_idx], labels[train_idx])
        prob = clf.predict_proba(XY_sc[test_idx])[:, 1]
        aucs.append(roc_auc_score(labels[test_idx], prob))
    return float(np.mean(aucs))


def compute_domain_alignment(Z_medal, Z_ptb, Y_medal, Y_ptb, seed: int) -> Dict:
    """Compute global and per-class domain alignment metrics."""
    results = {}

    scaler = StandardScaler()
    Z_all = np.vstack([Z_medal, Z_ptb])
    scaler.fit(Z_all)
    Zm = scaler.transform(Z_medal)
    Zp = scaler.transform(Z_ptb)

    print("\n  Global domain alignment:")
    sub = min(2000, len(Zm), len(Zp))
    rng = np.random.RandomState(seed)
    idx_m = rng.choice(len(Zm), sub, replace=False) if len(Zm) > sub else np.arange(len(Zm))
    idx_p = rng.choice(len(Zp), sub, replace=False) if len(Zp) > sub else np.arange(len(Zp))

    mmd_val = mmd_rbf(Zm[idx_m], Zp[idx_p])
    knn_val = knn_mixing_score(Zm[idx_m], Zp[idx_p])
    c2st_val = domain_classifier_auc(Zm[idx_m], Zp[idx_p], seed)
    results["global"] = {"mmd": mmd_val, "knn_mixing": knn_val, "c2st_auc": c2st_val}
    print(f"    MMD = {mmd_val:.6f}")
    print(f"    kNN mixing = {knn_val:.4f}")
    print(f"    C2ST AUC = {c2st_val:.4f}")

    labels_m = argmax_label(Y_medal)
    labels_p = argmax_label(Y_ptb)

    results["per_class"] = {}
    for cls_idx, cls_name in enumerate(SHARED_CLASSES):
        m_mask = labels_m == cls_idx
        p_mask = labels_p == cls_idx
        n_m, n_p = m_mask.sum(), p_mask.sum()
        if n_m < 10 or n_p < 10:
            print(f"    {cls_name}: skipped (n_medal={n_m}, n_ptb={n_p})")
            continue

        zm_cls = Zm[m_mask]
        zp_cls = Zp[p_mask]
        sub_cls = min(1000, n_m, n_p)
        idx_mc = rng.choice(n_m, sub_cls, replace=False) if n_m > sub_cls else np.arange(n_m)
        idx_pc = rng.choice(n_p, sub_cls, replace=False) if n_p > sub_cls else np.arange(n_p)

        mmd_c = mmd_rbf(zm_cls[idx_mc], zp_cls[idx_pc])
        knn_c = knn_mixing_score(zm_cls[idx_mc], zp_cls[idx_pc])
        c2st_c = domain_classifier_auc(zm_cls[idx_mc], zp_cls[idx_pc], seed)
        results["per_class"][cls_name] = {
            "n_medal": int(n_m), "n_ptb": int(n_p),
            "mmd": mmd_c, "knn_mixing": knn_c, "c2st_auc": c2st_c,
        }
        print(f"    {cls_name} (n={n_m}+{n_p}): MMD={mmd_c:.6f}, kNN={knn_c:.4f}, C2ST={c2st_c:.4f}")

    return results


# ---------------------------------------------------------------------------
# Class Separability
# ---------------------------------------------------------------------------

def class_separability_probe(Z: np.ndarray, Y: np.ndarray, domain: str, seed: int) -> Dict:
    """1-vs-rest logistic regression AUC for each class."""
    labels = argmax_label(Y)
    scaler = StandardScaler()
    Z_sc = scaler.fit_transform(Z)

    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for cls_idx, cls_name in enumerate(SHARED_CLASSES):
        binary = (labels == cls_idx).astype(int)
        if binary.sum() < 10 or (1 - binary).sum() < 10:
            continue
        aucs = []
        for tr, te in skf.split(Z_sc, binary):
            clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=seed)
            clf.fit(Z_sc[tr], binary[tr])
            prob = clf.predict_proba(Z_sc[te])[:, 1]
            aucs.append(roc_auc_score(binary[te], prob))
        mean_auc = float(np.mean(aucs))
        results[cls_name] = mean_auc
        print(f"    {domain} {cls_name}: probe AUC = {mean_auc:.4f}")
    return results


def fisher_lda_ratio(Z: np.ndarray, Y: np.ndarray) -> float:
    """Fisher LDA criterion: trace(S_b) / trace(S_w)."""
    labels = argmax_label(Y)
    classes = np.unique(labels)
    grand_mean = Z.mean(axis=0)
    S_b, S_w = 0.0, 0.0
    for c in classes:
        mask = labels == c
        Z_c = Z[mask]
        n_c = len(Z_c)
        mean_c = Z_c.mean(axis=0)
        diff = mean_c - grand_mean
        S_b += n_c * np.dot(diff, diff)
        S_w += ((Z_c - mean_c) ** 2).sum()
    return float(S_b / max(S_w, 1e-12))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.outdir}")

    print(f"\nLoading and remapping latents (prefix={args.prefix})...")
    Z_medal, Y_medal = load_and_remap(args.latent_dir, "medalcare", args.prefix)
    Z_ptb, Y_ptb = load_and_remap(args.latent_dir, "ptbxl", args.prefix)
    print(f"  MedalCare: Z={Z_medal.shape}, Y={Y_medal.shape}")
    print(f"  PTB-XL:    Z={Z_ptb.shape}, Y={Y_ptb.shape}")

    labels_m = argmax_label(Y_medal)
    labels_p = argmax_label(Y_ptb)
    print(f"  MedalCare class dist: {dict(zip(SHARED_CLASSES, [int((labels_m==i).sum()) for i in range(3)]))}")
    print(f"  PTB-XL class dist:    {dict(zip(SHARED_CLASSES, [int((labels_p==i).sum()) for i in range(3)]))}")

    print("\n--- PCA Visualizations ---")
    plot_pca_domain(Z_medal, Z_ptb, args.outdir, args.dpi)
    plot_pca_class(Z_medal, Y_medal, Z_ptb, Y_ptb, args.outdir, args.dpi)
    plot_pca_class_domain_overlay(Z_medal, Y_medal, Z_ptb, Y_ptb, args.outdir, args.dpi)

    print("\n--- Domain Alignment Metrics ---")
    alignment = compute_domain_alignment(Z_medal, Z_ptb, Y_medal, Y_ptb, args.seed)

    print("\n--- Class Separability Probes ---")
    sep_medal = class_separability_probe(Z_medal, Y_medal, "MedalCare", args.seed)
    sep_ptb = class_separability_probe(Z_ptb, Y_ptb, "PTB-XL", args.seed)
    sep_combined = class_separability_probe(
        np.vstack([Z_medal, Z_ptb]),
        np.vstack([Y_medal, Y_ptb]),
        "Combined", args.seed,
    )

    print("\n--- Fisher LDA Criterion ---")
    fisher_m = fisher_lda_ratio(Z_medal, Y_medal)
    fisher_p = fisher_lda_ratio(Z_ptb, Y_ptb)
    fisher_c = fisher_lda_ratio(np.vstack([Z_medal, Z_ptb]), np.vstack([Y_medal, Y_ptb]))
    print(f"  MedalCare: {fisher_m:.6f}")
    print(f"  PTB-XL:    {fisher_p:.6f}")
    print(f"  Combined:  {fisher_c:.6f}")

    report = {
        "experiment": f"Exp 7 — Shared Head (prefix={args.prefix})",
        "samples": {
            "medalcare": int(len(Z_medal)),
            "ptbxl": int(len(Z_ptb)),
        },
        "domain_alignment": alignment,
        "class_separability": {
            "medalcare": sep_medal,
            "ptbxl": sep_ptb,
            "combined": sep_combined,
        },
        "fisher_lda": {
            "medalcare": fisher_m,
            "ptbxl": fisher_p,
            "combined": fisher_c,
        },
    }

    report_path = args.outdir / "exp7_analysis.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved analysis report to {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
