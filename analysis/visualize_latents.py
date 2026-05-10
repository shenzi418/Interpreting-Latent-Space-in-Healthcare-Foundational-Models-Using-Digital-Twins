"""PCA-based latent space visualization for ECGFounder experiments.

Produces:
  1. PCA scatter plots of PTB-XL latents colored by superclass (shared basis)
  2. Scree plots comparing eigenvalue spectra across experiments
  3. PCA scatter plots of combined MedalCare+PTB-XL colored by domain
  4. PCA scatter plots of combined data colored by class within each domain

All PCA comparisons use a shared basis fitted on Exp 1 (baseline) so that
cross-experiment plots are directly comparable.

Usage:
  python analysis/visualize_latents.py --outdir outputs/latent_analysis
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LATENT_DIR = REPO_ROOT / "outputs" / "latents"

PTB_CLASSES = ["NORM", "MI", "STTC", "HYP", "CD"]
PTB_COLORS = ["#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#1f77b4"]

MEDAL_CLASSES = [f"Label {i}" for i in range(8)]
MEDAL_COLORS = plt.cm.Set2(np.linspace(0, 1, 8)).tolist()

DOMAIN_COLORS = {"MedalCare": "#1f77b4", "PTB-XL": "#d62728"}

EXPERIMENTS = {
    "Exp 1 (baseline)": "exp1_ptbxl",
    "Exp 4 (joint, no adapter)": "exp4_ptbxl",
    "Exp 5 (joint + adapter)": "exp5_ptbxl",
    "Exp 6 (joint + adapter + MMD)": "exp6_ptbxl",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PCA latent space visualization.")
    p.add_argument(
        "--outdir", type=Path, default=REPO_ROOT / "outputs" / "latent_analysis",
        help="Output directory for figures.",
    )
    p.add_argument(
        "--latent-dir", type=Path, default=LATENT_DIR,
        help="Directory containing exported latent NPZ files.",
    )
    p.add_argument(
        "--dpi", type=int, default=200,
        help="Figure resolution.",
    )
    return p.parse_args()


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"NPZ not found: {path}")
    return dict(np.load(path))


def primary_label(Y: np.ndarray) -> np.ndarray:
    """Assign each multi-label sample to its argmax class (for coloring)."""
    return np.argmax(Y, axis=1)


# ---------------------------------------------------------------------------
# 1a. PTB-XL PCA scatter — shared basis
# ---------------------------------------------------------------------------

def plot_ptbxl_pca_shared_basis(
    latent_dir: Path, outdir: Path, dpi: int,
) -> PCA:
    """Fit PCA on Exp 1, project all experiments into the same space."""
    # Load all PTB-XL latents
    data = {}
    for label, subdir in EXPERIMENTS.items():
        npz_path = latent_dir / subdir / "latents.npz"
        if npz_path.exists():
            d = load_npz(npz_path)
            data[label] = (d["Z"], d["Y"])

    if not data:
        print("[SKIP] No PTB-XL latent files found.")
        return None

    # Fit PCA on the baseline (first available experiment)
    ref_key = next(iter(data))
    Z_ref = data[ref_key][0]
    pca = PCA(n_components=min(50, Z_ref.shape[1]))
    pca.fit(Z_ref)
    print(f"PCA fitted on {ref_key}: PC1={pca.explained_variance_ratio_[0]:.3f}, "
          f"PC2={pca.explained_variance_ratio_[1]:.3f}")

    # Plot grid
    n_exp = len(data)
    fig, axes = plt.subplots(1, n_exp, figsize=(5 * n_exp, 4.5), squeeze=False)
    axes = axes[0]

    for idx, (label, (Z, Y)) in enumerate(data.items()):
        ax = axes[idx]
        Z_pca = pca.transform(Z)
        classes = primary_label(Y)

        for c_idx, c_name in enumerate(PTB_CLASSES):
            mask = classes == c_idx
            if mask.sum() == 0:
                continue
            ax.scatter(
                Z_pca[mask, 0], Z_pca[mask, 1],
                c=PTB_COLORS[c_idx], label=c_name,
                s=8, alpha=0.5, linewidths=0, rasterized=True,
            )

        ev1 = pca.explained_variance_ratio_[0] * 100
        ev2 = pca.explained_variance_ratio_[1] * 100
        ax.set_xlabel(f"PC1 ({ev1:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev2:.1f}%)" if idx == 0 else "")
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=8)
        if idx == 0:
            ax.legend(fontsize=7, markerscale=2, loc="best", framealpha=0.8)

    fig.suptitle("PTB-XL Latent Space — PCA (shared basis from Exp 1)", fontsize=12, y=1.02)
    fig.tight_layout()
    out_path = outdir / "pca_ptbxl_by_class_shared.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    # Also plot each experiment with its own PCA for comparison
    _plot_ptbxl_pca_independent(data, outdir, dpi)

    return pca


def _plot_ptbxl_pca_independent(
    data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    outdir: Path, dpi: int,
) -> None:
    """Each experiment gets its own PCA fit (shows internal structure)."""
    n_exp = len(data)
    fig, axes = plt.subplots(1, n_exp, figsize=(5 * n_exp, 4.5), squeeze=False)
    axes = axes[0]

    for idx, (label, (Z, Y)) in enumerate(data.items()):
        ax = axes[idx]
        pca_local = PCA(n_components=2)
        Z_pca = pca_local.fit_transform(Z)
        classes = primary_label(Y)

        for c_idx, c_name in enumerate(PTB_CLASSES):
            mask = classes == c_idx
            if mask.sum() == 0:
                continue
            ax.scatter(
                Z_pca[mask, 0], Z_pca[mask, 1],
                c=PTB_COLORS[c_idx], label=c_name,
                s=8, alpha=0.5, linewidths=0, rasterized=True,
            )

        ev1 = pca_local.explained_variance_ratio_[0] * 100
        ev2 = pca_local.explained_variance_ratio_[1] * 100
        ax.set_xlabel(f"PC1 ({ev1:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev2:.1f}%)" if idx == 0 else "")
        ax.set_title(f"{label}\n(own PCA)", fontsize=10)
        ax.tick_params(labelsize=8)
        if idx == 0:
            ax.legend(fontsize=7, markerscale=2, loc="best", framealpha=0.8)

    fig.suptitle("PTB-XL Latent Space — PCA (independent per experiment)", fontsize=12, y=1.02)
    fig.tight_layout()
    out_path = outdir / "pca_ptbxl_by_class_independent.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# 1b. Scree plots
# ---------------------------------------------------------------------------

def plot_scree(latent_dir: Path, outdir: Path, dpi: int, n_components: int = 30) -> None:
    """Overlay eigenvalue spectra for all experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for label, subdir in EXPERIMENTS.items():
        npz_path = latent_dir / subdir / "latents.npz"
        if not npz_path.exists():
            continue
        Z = np.load(npz_path)["Z"]
        pca = PCA(n_components=min(n_components, Z.shape[1]))
        pca.fit(Z)
        evr = pca.explained_variance_ratio_ * 100
        cumulative = np.cumsum(evr)

        ax1.plot(range(1, len(evr) + 1), evr, marker="o", markersize=3, label=label)
        ax2.plot(range(1, len(cumulative) + 1), cumulative, marker="o", markersize=3, label=label)

    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_title("Scree Plot (per-component)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Cumulative Explained Variance (%)")
    ax2.set_title("Cumulative Explained Variance")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=80, color="gray", linestyle="--", alpha=0.5, label="_80%")

    fig.tight_layout()
    out_path = outdir / "scree_plots.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# 1c. Domain PCA — MedalCare + PTB-XL
# ---------------------------------------------------------------------------

def plot_domain_pca(
    latent_dir: Path, outdir: Path, dpi: int,
) -> None:
    """PCA of combined MedalCare + PTB-XL, colored by domain. Exp 5 vs Exp 6."""
    pairs = [
        ("Exp 5 (joint + adapter)", "exp5"),
        ("Exp 6 (joint + adapter + MMD)", "exp6"),
    ]

    available = []
    for label, prefix in pairs:
        ptb_path = latent_dir / f"{prefix}_ptbxl" / "latents.npz"
        med_path = latent_dir / f"{prefix}_medalcare" / "latents.npz"
        if ptb_path.exists() and med_path.exists():
            available.append((label, prefix, ptb_path, med_path))

    if not available:
        print("[SKIP] No paired MedalCare+PTB-XL latent files found.")
        return

    # --- Plot 1: Colored by domain ---
    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5), squeeze=False)
    axes = axes[0]

    for idx, (label, prefix, ptb_path, med_path) in enumerate(available):
        ax = axes[idx]
        Z_ptb = np.load(ptb_path)["Z"]
        Z_med = np.load(med_path)["Z"]

        Z_combined = np.vstack([Z_med, Z_ptb])
        mean = Z_combined.mean(axis=0, keepdims=True)
        std = Z_combined.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        Z_normed = (Z_combined - mean) / std

        pca = PCA(n_components=2)
        Z_pca = pca.fit_transform(Z_normed)

        n_med = Z_med.shape[0]
        ax.scatter(
            Z_pca[:n_med, 0], Z_pca[:n_med, 1],
            c=DOMAIN_COLORS["MedalCare"], label="MedalCare (synthetic)",
            s=6, alpha=0.35, linewidths=0, rasterized=True,
        )
        ax.scatter(
            Z_pca[n_med:, 0], Z_pca[n_med:, 1],
            c=DOMAIN_COLORS["PTB-XL"], label="PTB-XL (real)",
            s=6, alpha=0.35, linewidths=0, rasterized=True,
        )

        ev1 = pca.explained_variance_ratio_[0] * 100
        ev2 = pca.explained_variance_ratio_[1] * 100
        ax.set_xlabel(f"PC1 ({ev1:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev2:.1f}%)" if idx == 0 else "")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8, markerscale=2, framealpha=0.8)
        ax.tick_params(labelsize=8)

    fig.suptitle("Domain Alignment — PCA of Combined Latent Space", fontsize=12, y=1.02)
    fig.tight_layout()
    out_path = outdir / "pca_domain_comparison.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    # --- Plot 2: Colored by class within each domain (side-by-side panels) ---
    for label, prefix, ptb_path, med_path in available:
        _plot_domain_by_class(label, prefix, ptb_path, med_path, outdir, dpi)


def _plot_domain_by_class(
    label: str, prefix: str,
    ptb_path: Path, med_path: Path,
    outdir: Path, dpi: int,
) -> None:
    """Two-panel plot: MedalCare classes (left) and PTB-XL classes (right)."""
    ptb_data = np.load(ptb_path)
    med_data = np.load(med_path)
    Z_ptb, Y_ptb = ptb_data["Z"], ptb_data["Y"]
    Z_med, Y_med = med_data["Z"], med_data["Y"]

    Z_combined = np.vstack([Z_med, Z_ptb])
    mean = Z_combined.mean(axis=0, keepdims=True)
    std = Z_combined.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    Z_normed = (Z_combined - mean) / std

    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(Z_normed)

    n_med = Z_med.shape[0]
    Z_pca_med = Z_pca[:n_med]
    Z_pca_ptb = Z_pca[n_med:]

    fig, (ax_med, ax_ptb) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # MedalCare panel
    classes_med = primary_label(Y_med)
    for c_idx in range(Y_med.shape[1]):
        mask = classes_med == c_idx
        if mask.sum() == 0:
            continue
        ax_med.scatter(
            Z_pca_med[mask, 0], Z_pca_med[mask, 1],
            c=[MEDAL_COLORS[c_idx]], label=MEDAL_CLASSES[c_idx],
            s=8, alpha=0.5, linewidths=0, rasterized=True,
        )
    ax_med.set_title("MedalCare (synthetic)", fontsize=10)
    ax_med.legend(fontsize=6, markerscale=2, ncol=2, loc="best", framealpha=0.8)

    ev1 = pca.explained_variance_ratio_[0] * 100
    ev2 = pca.explained_variance_ratio_[1] * 100
    ax_med.set_xlabel(f"PC1 ({ev1:.1f}%)")
    ax_med.set_ylabel(f"PC2 ({ev2:.1f}%)")

    # PTB-XL panel
    classes_ptb = primary_label(Y_ptb)
    for c_idx, c_name in enumerate(PTB_CLASSES):
        mask = classes_ptb == c_idx
        if mask.sum() == 0:
            continue
        ax_ptb.scatter(
            Z_pca_ptb[mask, 0], Z_pca_ptb[mask, 1],
            c=PTB_COLORS[c_idx], label=c_name,
            s=8, alpha=0.5, linewidths=0, rasterized=True,
        )
    ax_ptb.set_title("PTB-XL (real)", fontsize=10)
    ax_ptb.set_xlabel(f"PC1 ({ev1:.1f}%)")
    ax_ptb.legend(fontsize=7, markerscale=2, loc="best", framealpha=0.8)

    fig.suptitle(f"Class Structure in Shared Embedding — {label}", fontsize=11, y=1.02)
    fig.tight_layout()
    out_path = outdir / f"pca_class_by_domain_{prefix}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.outdir}")
    print(f"Latent directory: {args.latent_dir}")
    print()

    print("=" * 60)
    print("Phase 1a: PTB-XL PCA by superclass (shared + independent)")
    print("=" * 60)
    plot_ptbxl_pca_shared_basis(args.latent_dir, args.outdir, args.dpi)
    print()

    print("=" * 60)
    print("Phase 1b: Scree plots")
    print("=" * 60)
    plot_scree(args.latent_dir, args.outdir, args.dpi)
    print()

    print("=" * 60)
    print("Phase 1c-d: Domain PCA (by domain + by class)")
    print("=" * 60)
    plot_domain_pca(args.latent_dir, args.outdir, args.dpi)
    print()

    print("Done. All figures saved to", args.outdir)


if __name__ == "__main__":
    main()
