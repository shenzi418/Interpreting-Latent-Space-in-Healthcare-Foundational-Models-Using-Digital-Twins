"""Iterative Nullspace Projection (INLP) post-training latent alignment.

References
----------
Ravfogel et al. (2020) "Null It Out: Guarding Protected Attributes by
Iterative Nullspace Projection." ACL.

Algorithm
---------
Given a pool of latents Z ∈ R^{N×D} labelled with binary domain identity
d ∈ {0=MedalCare, 1=PTB-XL}:

  1. Z⁽⁰⁾ = StandardScaler(Z)
  2. For t = 1 ... T_max:
       w_t   = LogisticRegression(C=1.0, class_weight='balanced').fit(Z⁽ᵗ⁻¹⁾, d).coef_
       P_t   = I - w_t w_tᵀ / ‖w_t‖²            (rank-(D-1) orthogonal projection)
       Z⁽ᵗ⁾ = Z⁽ᵗ⁻¹⁾ @ P_t
       acc_t = 5-fold CV domain accuracy on Z⁽ᵗ⁾
       break if acc_t ≤ stop_acc
  3. P_total = P_1 P_2 ... P_T

Apply at inference: Z_aligned = StandardScaler.transform(Z_raw) @ P_total

Scope (2026-05-08): shared-head pair primary {exp7_baseline, exp7_ccmmd},
exp5_3class as conditional sensitivity (§4b).

Pool modes
----------
``--pool-mode asymmetric`` (default, used in the v1 INLP run):
    Fit pool   = MedalCare-train + PTB-XL-test (the only PTB-XL latents we had).
    Held-out   = MedalCare-test only. PTB-XL-test is in the fit pool, so the
                 downstream-view metrics use a "synth held-out, real in-pool" mix.

``--pool-mode symmetric`` (v2 sensitivity check, requires PTB-XL-train latents):
    Fit pool   = MedalCare-train + PTB-XL-train (folds 1-8).
    Held-out   = MedalCare-test + PTB-XL-test  (truly unseen on both sides).
    Run with ``--output-suffix _inlpv2`` so v1 outputs are not clobbered.
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

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse vetted alignment metrics from analysis/exp7_analysis.py
from analysis.exp7_analysis import (  # noqa: E402
    domain_classifier_auc,
    knn_mixing_score,
    mmd_rbf,
)

# ---------------------------------------------------------------------------
# Constants — same convention as analysis/phase_b2_infarct_decoding.py
# ---------------------------------------------------------------------------

CONFIG_LATENT_STEMS: Dict[str, str] = {
    "exp7_baseline": "exp7",
    "exp7_ccmmd":    "exp7_ccmmd",
    "exp5_3class":   "exp5_3class",
}

PRIMARY_CONFIGS = ("exp7_baseline", "exp7_ccmmd")
CONDITIONAL_CONFIGS = ("exp5_3class",)
ALL_INSCOPE = PRIMARY_CONFIGS + CONDITIONAL_CONFIGS

# Splits to apply the projection to (and save aligned latents for).
# Symmetric mode adds ptbxl_train so we can audit alignment on held-in real data.
SPLITS_BY_MODE: Dict[str, Tuple[str, ...]] = {
    "asymmetric": ("medalcare_train", "medalcare", "ptbxl"),
    "symmetric":  ("medalcare_train", "medalcare", "ptbxl_train", "ptbxl"),
}

LATENT_DIR = REPO_ROOT / "outputs" / "latents"
INLP_DIR = REPO_ROOT / "outputs" / "inlp"
DOMAIN_MEDALCARE = 0
DOMAIN_PTBXL = 1

# Default hyperparameters (override via CLI)
DEFAULT_MAX_ITER = 20
DEFAULT_STOP_ACC = 0.55
DEFAULT_SEED = 42
DEFAULT_POOL_MODE = "asymmetric"
DEFAULT_OUTPUT_SUFFIX = "_inlp"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def latent_path(stem: str, split: str, suffix: str = "") -> Path:
    return LATENT_DIR / f"{stem}_{split}{suffix}" / "latents.npz"


def load_split_Z(stem: str, split: str, suffix: str = "") -> np.ndarray:
    path = latent_path(stem, split, suffix)
    return np.load(path, allow_pickle=True)["Z"].astype(np.float64)


def load_combined_pool(
    config: str, pool_mode: str = DEFAULT_POOL_MODE,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Stack MedalCare-train + PTB-XL latents into one labelled fitting pool.

    pool_mode
    ---------
    "asymmetric" : MedalCare-train + PTB-XL-test  (v1 INLP run; PTB-XL-test
                   was the only PTB-XL latents we had then).
    "symmetric"  : MedalCare-train + PTB-XL-train (folds 1-8). Cleaner protocol
                   because it leaves both test splits truly held out.

    Returns
    -------
    Z_pool        : (n_med + n_ptb, D) float64
    domain        : (n_med + n_ptb,) int {0, 1}
    n_med, n_ptb  : pool sizes
    """
    stem = CONFIG_LATENT_STEMS[config]
    Z_med = load_split_Z(stem, "medalcare_train")
    if pool_mode == "asymmetric":
        Z_ptb = load_split_Z(stem, "ptbxl")        # PTB-XL test fold 10
    elif pool_mode == "symmetric":
        Z_ptb = load_split_Z(stem, "ptbxl_train")  # PTB-XL folds 1-8
    else:
        raise ValueError(
            f"unknown pool_mode={pool_mode!r}; expected 'asymmetric' or 'symmetric'"
        )
    n_med, n_ptb = Z_med.shape[0], Z_ptb.shape[0]
    Z_pool = np.vstack([Z_med, Z_ptb])
    domain = np.concatenate(
        [np.full(n_med, DOMAIN_MEDALCARE, dtype=np.int64),
         np.full(n_ptb, DOMAIN_PTBXL, dtype=np.int64)]
    )
    return Z_pool, domain, n_med, n_ptb


# ---------------------------------------------------------------------------
# Core INLP
# ---------------------------------------------------------------------------

def cv_domain_accuracy(Z: np.ndarray, d: np.ndarray, seed: int) -> float:
    """5-fold stratified CV accuracy of an L2 logistic domain classifier."""
    clf = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs",
        max_iter=2000, class_weight="balanced",
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, Z, d, cv=skf, scoring="accuracy", n_jobs=1)
    return float(np.mean(scores))


def fit_domain_classifier_w(Z: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Fit one L2 logistic regression and return its weight vector w ∈ R^D."""
    clf = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs",
        max_iter=2000, class_weight="balanced",
    )
    clf.fit(Z, d)
    w = clf.coef_.astype(np.float64).ravel()  # shape (D,)
    return w


def nullspace_projection(w: np.ndarray) -> np.ndarray:
    """Build P = I - (w w^T) / (w^T w), the rank-1 orthogonal nullspace projector."""
    D = w.shape[0]
    w_norm_sq = float(np.dot(w, w))
    if w_norm_sq <= 0.0 or not np.isfinite(w_norm_sq):
        # Degenerate: classifier learned nothing. Return identity (no-op).
        return np.eye(D, dtype=np.float64)
    P = np.eye(D, dtype=np.float64) - np.outer(w, w) / w_norm_sq
    return P


def inlp_fit(
    Z_scaled: np.ndarray,
    d: np.ndarray,
    *,
    max_iter: int = DEFAULT_MAX_ITER,
    stop_acc: float = DEFAULT_STOP_ACC,
    seed: int = DEFAULT_SEED,
) -> Tuple[np.ndarray, List[Dict]]:
    """Run INLP. Return (P_total, iteration_log)."""
    D = Z_scaled.shape[1]
    P_total = np.eye(D, dtype=np.float64)
    Z_proj = Z_scaled.copy()

    # Iter 0: pre-INLP baseline accuracy
    acc0 = cv_domain_accuracy(Z_proj, d, seed=seed)
    log: List[Dict] = [{"iter": 0, "domain_accuracy": acc0, "stopped": False}]
    print(f"[inlp] iter 0 (orig)   domain_accuracy = {acc0:.4f}")

    for t in range(1, max_iter + 1):
        t0 = time.time()
        w = fit_domain_classifier_w(Z_proj, d)
        P_t = nullspace_projection(w)
        Z_proj = Z_proj @ P_t
        P_total = P_total @ P_t
        acc_t = cv_domain_accuracy(Z_proj, d, seed=seed)
        elapsed = time.time() - t0

        stopped = acc_t <= stop_acc
        log.append({
            "iter": t,
            "domain_accuracy": acc_t,
            "w_norm": float(np.linalg.norm(w)),
            "elapsed_s": elapsed,
            "stopped": stopped,
        })
        print(
            f"[inlp] iter {t:>2d}        domain_accuracy = {acc_t:.4f}  "
            f"(‖w‖={np.linalg.norm(w):.3e}, {elapsed:.1f}s)"
        )
        if stopped:
            print(f"[inlp] stop_acc {stop_acc:.2f} reached at iter {t}.")
            break
    else:
        print(f"[inlp] reached max_iter {max_iter} without crossing stop_acc {stop_acc:.2f}.")

    return P_total, log


def apply_alignment(
    Z_raw: np.ndarray, scaler: StandardScaler, P_total: np.ndarray,
) -> np.ndarray:
    """Z_aligned = scaler.transform(Z_raw) @ P_total (cast to float32 for storage)."""
    Z_scaled = scaler.transform(Z_raw.astype(np.float64))
    return (Z_scaled @ P_total).astype(np.float32)


def save_aligned_latents(
    config: str, scaler: StandardScaler, P_total: np.ndarray,
    *,
    suffix: str = DEFAULT_OUTPUT_SUFFIX,
    splits: Tuple[str, ...] = SPLITS_BY_MODE[DEFAULT_POOL_MODE],
) -> Dict[str, Dict]:
    """For each split, load original NPZ, apply alignment, write {stem}_{split}{suffix}/latents.npz.

    Other arrays (Y, etc.) are copied byte-identical from the original.
    """
    stem = CONFIG_LATENT_STEMS[config]
    saved: Dict[str, Dict] = {}
    for split in splits:
        src = latent_path(stem, split)
        dst = latent_path(stem, split, suffix=suffix)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with np.load(src, allow_pickle=True) as data:
            payload = {k: data[k] for k in data.keys()}
        Z_orig = payload["Z"]
        Z_aligned = apply_alignment(Z_orig.astype(np.float64), scaler, P_total)
        # sanity
        assert Z_aligned.shape == Z_orig.shape, (
            f"shape mismatch for {config}/{split}: "
            f"{Z_orig.shape} -> {Z_aligned.shape}"
        )
        n_nan = int(np.isnan(Z_aligned).sum())
        n_inf = int(np.isinf(Z_aligned).sum())
        if n_nan or n_inf:
            raise RuntimeError(
                f"NaN/inf in aligned {config}/{split}: "
                f"NaN={n_nan} inf={n_inf}"
            )
        payload["Z"] = Z_aligned
        np.savez(dst, **payload)
        saved[split] = {
            "src": str(src.relative_to(REPO_ROOT)),
            "dst": str(dst.relative_to(REPO_ROOT)),
            "Z_shape": list(Z_aligned.shape),
            "Z_dtype": str(Z_aligned.dtype),
        }
        print(f"[save] {dst.relative_to(REPO_ROOT)} (Z={Z_aligned.shape})")
    return saved


# ---------------------------------------------------------------------------
# Alignment metrics + visualisation
# ---------------------------------------------------------------------------

def compute_alignment_metrics(
    Z_med: np.ndarray, Z_ptb: np.ndarray, *, seed: int = DEFAULT_SEED,
    knn_k: int = 15,
) -> Dict[str, float]:
    """C2ST AUROC, MMD (RBF, median heuristic), kNN mixing on cosine distance.

    Reuses vetted implementations from analysis/exp7_analysis.py.
    """
    return {
        "c2st_auroc": float(domain_classifier_auc(Z_med, Z_ptb, seed=seed)),
        "mmd_rbf":    float(mmd_rbf(Z_med, Z_ptb, sigma=None)),
        "knn_mixing": float(knn_mixing_score(Z_med, Z_ptb, k=knn_k)),
    }


def plot_pca_before_after(
    Z_pool_orig: np.ndarray,
    Z_pool_aligned: np.ndarray,
    domain: np.ndarray,
    out_path: Path,
    *,
    title_prefix: str,
    pool_mode: str = DEFAULT_POOL_MODE,
    seed: int = DEFAULT_SEED,
) -> None:
    """1×2 PCA scatter, both panels using basis fitted on the ORIGINAL pool.

    Keeping the same axes makes "before" vs "after" directly comparable.
    """
    pca = PCA(n_components=2, random_state=seed).fit(Z_pool_orig)
    Z2_orig = pca.transform(Z_pool_orig)
    Z2_aligned = pca.transform(Z_pool_aligned)

    real_label = "PTB-XL-train" if pool_mode == "symmetric" else "PTB-XL-test"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, Z2, sub in zip(axes, [Z2_orig, Z2_aligned], ["Before INLP", "After INLP"]):
        for dom_val, color, label in [(0, "#1f77b4", "MedalCare-train"),
                                       (1, "#d62728", real_label)]:
            mask = domain == dom_val
            ax.scatter(
                Z2[mask, 0], Z2[mask, 1],
                s=4, alpha=0.35, c=color, label=label, rasterized=True,
            )
        ax.set_title(f"{title_prefix} — {sub}")
        ax.set_xlabel("PC1 (orig basis)")
        ax.set_ylabel("PC2 (orig basis)")
        ax.legend(loc="best", markerscale=2.0, framealpha=0.9)
        ax.grid(alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Per-config orchestration
# ---------------------------------------------------------------------------

def run_one_config(
    config: str, *,
    max_iter: int, stop_acc: float, seed: int,
    pool_mode: str = DEFAULT_POOL_MODE,
    output_suffix: str = DEFAULT_OUTPUT_SUFFIX,
) -> Dict:
    """Full INLP pipeline for one config. Returns a summary dict."""
    splits_to_save = SPLITS_BY_MODE[pool_mode]
    print("=" * 72)
    print(
        f"INLP — config: {config}  (stem: {CONFIG_LATENT_STEMS[config]})  "
        f"pool_mode={pool_mode}  output_suffix={output_suffix}"
    )
    print("=" * 72)

    # 1. Load combined fitting pool
    Z_pool, domain, n_med, n_ptb = load_combined_pool(config, pool_mode=pool_mode)
    real_split_name = "ptbxl_train" if pool_mode == "symmetric" else "ptbxl"
    print(
        f"[load] fit pool ({pool_mode}): "
        f"medalcare_train(n={n_med}) + {real_split_name}(n={n_ptb}) "
        f"= {len(Z_pool)},  D={Z_pool.shape[1]}"
    )

    # 2. Fit scaler on combined pool
    scaler = StandardScaler().fit(Z_pool)
    Z_scaled = scaler.transform(Z_pool)

    # 3. Pre-INLP alignment metrics on the SCALED original pool
    print("[metrics] computing original alignment metrics...")
    metrics_orig = compute_alignment_metrics(
        Z_scaled[domain == 0], Z_scaled[domain == 1], seed=seed,
    )
    print(f"          orig: c2st_auroc={metrics_orig['c2st_auroc']:.4f}  "
          f"mmd={metrics_orig['mmd_rbf']:.4e}  knn_mix={metrics_orig['knn_mixing']:.4f}")

    # 4. Fit INLP
    P_total, iteration_log = inlp_fit(
        Z_scaled, domain,
        max_iter=max_iter, stop_acc=stop_acc, seed=seed,
    )

    # 5. Sanity-check P_total orthogonality on the rank-active subspace.
    # P^T P is a projector with rank D - n_iters_actual; for any vector
    # v in the projected subspace, ||P^T P v - v|| should be near zero.
    PtP = P_total.T @ P_total
    # Check on a random unit vector projected through P_total first
    rng = np.random.default_rng(seed)
    v = rng.normal(size=P_total.shape[0])
    v_proj = P_total @ v
    residual = float(np.linalg.norm(PtP @ v_proj - v_proj) / max(np.linalg.norm(v_proj), 1e-12))
    print(f"[sanity] P^T P projector residual on a projected vector = {residual:.3e}")

    # 6. Compute aligned pool + post-INLP metrics
    Z_pool_aligned = Z_scaled @ P_total
    metrics_inlp = compute_alignment_metrics(
        Z_pool_aligned[domain == 0], Z_pool_aligned[domain == 1], seed=seed,
    )
    print(f"          inlp: c2st_auroc={metrics_inlp['c2st_auroc']:.4f}  "
          f"mmd={metrics_inlp['mmd_rbf']:.4e}  knn_mix={metrics_inlp['knn_mixing']:.4f}")

    # 7. Save projection + scaler params.
    # For pool_mode=asymmetric with default suffix, keep proj_dir = outputs/inlp/{config}
    # (preserves v1 INLP layout). Otherwise append the suffix to disambiguate.
    proj_subdir = config if output_suffix == DEFAULT_OUTPUT_SUFFIX else f"{config}{output_suffix}"
    proj_dir = INLP_DIR / proj_subdir
    proj_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        proj_dir / "projection.npz",
        P_total=P_total.astype(np.float64),
        scaler_mean=scaler.mean_.astype(np.float64),
        scaler_scale=scaler.scale_.astype(np.float64),
        n_iter=np.array([sum(1 for r in iteration_log if r["iter"] > 0)], dtype=np.int64),
    )
    print(f"[save] {(proj_dir / 'projection.npz').relative_to(REPO_ROOT)}")

    # 8. Save iteration log
    with (proj_dir / "iteration_log.json").open("w", encoding="utf-8") as f:
        json.dump({
            "config": config,
            "stem": CONFIG_LATENT_STEMS[config],
            "pool_mode": pool_mode,
            "output_suffix": output_suffix,
            "n_med": n_med, "n_ptb": n_ptb, "D": int(Z_pool.shape[1]),
            "max_iter": max_iter, "stop_acc": stop_acc, "seed": seed,
            "metrics_orig": metrics_orig,
            "metrics_inlp": metrics_inlp,
            "iterations": iteration_log,
            "ptp_residual": residual,
        }, f, indent=2)
    print(f"[save] {(proj_dir / 'iteration_log.json').relative_to(REPO_ROOT)}")

    # 9. PCA before/after plot
    plot_pca_before_after(
        Z_scaled, Z_pool_aligned, domain,
        out_path=proj_dir / "pca_before_after.png",
        title_prefix=f"{config} ({pool_mode})",
        pool_mode=pool_mode,
        seed=seed,
    )

    # 10. Apply to all in-scope splits and save aligned latents
    saved = save_aligned_latents(
        config, scaler, P_total,
        suffix=output_suffix, splits=splits_to_save,
    )

    # 11. Verification — measure alignment on the held-out test pool
    # (medalcare TEST + PTB-XL test). This is the data downstream B2 actually sees.
    # In symmetric mode, BOTH sides are truly held out (neither was in fit pool).
    # In asymmetric mode, only MedalCare-test is held out; PTB-XL-test was in the fit pool.
    Z_med_test_aligned = load_split_Z(
        CONFIG_LATENT_STEMS[config], "medalcare", suffix=output_suffix,
    )
    Z_ptb_test_aligned = load_split_Z(
        CONFIG_LATENT_STEMS[config], "ptbxl", suffix=output_suffix,
    )
    Z_med_test_orig = load_split_Z(CONFIG_LATENT_STEMS[config], "medalcare")
    Z_ptb_test_orig = load_split_Z(CONFIG_LATENT_STEMS[config], "ptbxl")
    Z_med_test_orig_sc = scaler.transform(Z_med_test_orig.astype(np.float64))
    Z_ptb_test_orig_sc = scaler.transform(Z_ptb_test_orig.astype(np.float64))
    metrics_orig_test = compute_alignment_metrics(
        Z_med_test_orig_sc, Z_ptb_test_orig_sc, seed=seed,
    )
    metrics_inlp_test = compute_alignment_metrics(
        Z_med_test_aligned.astype(np.float64),
        Z_ptb_test_aligned.astype(np.float64),
        seed=seed,
    )
    held_out_label = (
        "BOTH SIDES held out (medalcare-TEST + ptbxl-TEST)"
        if pool_mode == "symmetric"
        else "synth held out, real in-pool (medalcare-TEST + ptbxl-TEST)"
    )
    print(
        f"[verify] {held_out_label} — downstream B2 view:\n"
        f"          orig: c2st={metrics_orig_test['c2st_auroc']:.4f} "
        f"mmd={metrics_orig_test['mmd_rbf']:.4e} knn={metrics_orig_test['knn_mixing']:.4f}\n"
        f"          inlp: c2st={metrics_inlp_test['c2st_auroc']:.4f} "
        f"mmd={metrics_inlp_test['mmd_rbf']:.4e} knn={metrics_inlp_test['knn_mixing']:.4f}"
    )

    return {
        "config": config,
        "stem": CONFIG_LATENT_STEMS[config],
        "pool_mode": pool_mode,
        "output_suffix": output_suffix,
        "splits_saved": list(splits_to_save),
        "n_med_pool": n_med, "n_ptb_pool": n_ptb,
        "n_iter_run": sum(1 for r in iteration_log if r["iter"] > 0),
        "metrics_orig_pool": metrics_orig,
        "metrics_inlp_pool": metrics_inlp,
        "metrics_orig_test": metrics_orig_test,
        "metrics_inlp_test": metrics_inlp_test,
        "saved_splits": saved,
        "ptp_residual": residual,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="INLP post-training latent alignment.")
    ap.add_argument(
        "--configs", nargs="+", default=list(PRIMARY_CONFIGS),
        choices=list(ALL_INSCOPE),
        help=f"Configs to run. Default: shared-head primary {list(PRIMARY_CONFIGS)}.",
    )
    ap.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    ap.add_argument("--stop-acc", type=float, default=DEFAULT_STOP_ACC)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--pool-mode", choices=["asymmetric", "symmetric"],
        default=DEFAULT_POOL_MODE,
        help=(
            "Composition of the INLP fitting pool. "
            "'asymmetric' = MedalCare-train + PTB-XL-test (v1 default). "
            "'symmetric' = MedalCare-train + PTB-XL-train (v2 sensitivity; "
            "requires outputs/latents/<stem>_ptbxl_train/ to exist)."
        ),
    )
    ap.add_argument(
        "--output-suffix", type=str, default=DEFAULT_OUTPUT_SUFFIX,
        help=(
            "Suffix for aligned-latent dirs and projection subdir. "
            "Default '_inlp' preserves the v1 layout. Use '_inlpv2' for the "
            "symmetric v2 sensitivity run so v1 outputs are not overwritten."
        ),
    )
    ap.add_argument(
        "--summary-out", type=Path,
        default=None,
        help=(
            "Aggregated cross-config summary JSON. "
            "Default: outputs/inlp/inlp_summary.json (or inlp_summary{output_suffix}.json "
            "if a non-default suffix is supplied)."
        ),
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    INLP_DIR.mkdir(parents=True, exist_ok=True)

    if args.summary_out is None:
        summary_name = (
            "inlp_summary.json"
            if args.output_suffix == DEFAULT_OUTPUT_SUFFIX
            else f"inlp_summary{args.output_suffix}.json"
        )
        args.summary_out = INLP_DIR / summary_name

    print(
        f"INLP run: configs={args.configs}, max_iter={args.max_iter}, "
        f"stop_acc={args.stop_acc}, seed={args.seed}, "
        f"pool_mode={args.pool_mode}, output_suffix={args.output_suffix}, "
        f"summary_out={args.summary_out}"
    )
    print()

    summary: Dict[str, Dict] = {}
    # If summary already exists (e.g. previous partial run), merge into it
    if args.summary_out.exists():
        try:
            summary = json.loads(args.summary_out.read_text(encoding="utf-8"))
            print(f"[merge] loaded existing summary with configs={list(summary.keys())}")
        except Exception as exc:
            print(f"[warn] could not read existing summary: {exc}")

    for cfg in args.configs:
        try:
            res = run_one_config(
                cfg,
                max_iter=args.max_iter, stop_acc=args.stop_acc, seed=args.seed,
                pool_mode=args.pool_mode, output_suffix=args.output_suffix,
            )
            summary[cfg] = res
        except Exception as exc:
            print(f"[ERROR] config {cfg} failed: {exc}")
            raise

        # Persist incrementally so a crash in config N doesn't lose 1..N-1
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print()
    print("=" * 72)
    print("INLP summary table")
    print("=" * 72)
    hdr = f"{'config':<14s} {'n_iter':>6s} {'C2ST_orig':>10s} {'C2ST_inlp':>10s}  " \
          f"{'MMD_orig':>10s} {'MMD_inlp':>10s}  {'kNN_orig':>9s} {'kNN_inlp':>9s}"
    print(hdr)
    for cfg, res in summary.items():
        m0 = res["metrics_orig_pool"]; m1 = res["metrics_inlp_pool"]
        print(
            f"{cfg:<14s} {res['n_iter_run']:>6d} "
            f"{m0['c2st_auroc']:>10.4f} {m1['c2st_auroc']:>10.4f}  "
            f"{m0['mmd_rbf']:>10.3e} {m1['mmd_rbf']:>10.3e}  "
            f"{m0['knn_mixing']:>9.4f} {m1['knn_mixing']:>9.4f}"
        )

    print(f"\nWrote {args.summary_out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
