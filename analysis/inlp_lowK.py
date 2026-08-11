"""INLP at bottleneck dimensions K ∈ {16, 64, 256}.

Task 1 of the 2026-05-24 overnight run. Mirrors analysis/inlp_alignment.py but
adapts to the bottleneck latent-export naming convention:

    outputs/latents/exp7_bottleneck_K{K}_{domain}_{split}/latents.npz

where {domain} ∈ {medalcare, ptbxl} and {split} ∈ {train, val, test}.

The legacy inlp_alignment.py was hard-wired to the K=1024 naming
({stem}_medalcare for TEST, {stem}_medalcare_train for TRAIN); we cannot reuse
its CLI here. We DO reuse the vetted building blocks (``inlp_fit``,
``apply_alignment``, ``compute_alignment_metrics``, ``plot_pca_before_after``).

Pool mode
---------
Asymmetric (default for this overnight run):
    fit pool   = MedalCare-train + PTB-XL-test
    held-out   = MedalCare-test (PTB-XL-test was in the fit pool)

Optionally we also fit a K=64 SYMMETRIC variant for sensitivity:
    fit pool   = MedalCare-train + PTB-XL-train
    held-out   = MedalCare-test + PTB-XL-test (both truly unseen)

Outputs
-------
For each K (asymmetric):

    outputs/inlp_lowK/exp7_bottleneck_K{K}/
        projection.npz           — P_total, scaler stats, n_iter
        iteration_log.json       — per-iter domain accuracy + metrics
        pca_before_after.png     — 2-panel PCA scatter (orig basis)

    outputs/latents/exp7_bottleneck_K{K}_medalcare_train_inlp/latents.npz
    outputs/latents/exp7_bottleneck_K{K}_medalcare_val_inlp/latents.npz
    outputs/latents/exp7_bottleneck_K{K}_medalcare_test_inlp/latents.npz
    outputs/latents/exp7_bottleneck_K{K}_ptbxl_train_inlp/latents.npz
    outputs/latents/exp7_bottleneck_K{K}_ptbxl_val_inlp/latents.npz
    outputs/latents/exp7_bottleneck_K{K}_ptbxl_test_inlp/latents.npz

Cross-K summary at:

    outputs/inlp_lowK/inlp_lowK_summary.json

Run::

    python analysis/inlp_lowK.py                            # K=16,64,256 asym
    python analysis/inlp_lowK.py --symmetric-K 64           # extra K=64 symmetric
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
from analysis.inlp_alignment import (  # noqa: E402
    inlp_fit,
    apply_alignment,
    compute_alignment_metrics,
    plot_pca_before_after,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LATENT_DIR = REPO_ROOT / "outputs" / "latents"
OUT_DIR = REPO_ROOT / "outputs" / "inlp_lowK"
SEED = 42
STOP_ACC = 0.55
K_LIST = [16, 64, 256]

DOMAINS = ("medalcare", "ptbxl")
SPLITS = ("train", "val", "test")


def stem_for_K(K: int) -> str:
    return f"exp7_bottleneck_K{K}"


def latent_dir(K: int, domain: str, split: str, suffix: str = "") -> Path:
    return LATENT_DIR / f"{stem_for_K(K)}_{domain}_{split}{suffix}"


def latent_path(K: int, domain: str, split: str, suffix: str = "") -> Path:
    return latent_dir(K, domain, split, suffix) / "latents.npz"


def load_Z(K: int, domain: str, split: str, suffix: str = "") -> np.ndarray:
    return np.load(latent_path(K, domain, split, suffix), allow_pickle=True)["Z"].astype(
        np.float64
    )


# ---------------------------------------------------------------------------
# Per-K orchestration
# ---------------------------------------------------------------------------


def fit_one_K(K: int, *, pool_mode: str, output_suffix: str, max_iter: int) -> Dict:
    print("=" * 72)
    print(f"[inlp_lowK] K={K}  pool_mode={pool_mode}  output_suffix={output_suffix}")
    print("=" * 72)

    # Load fit pool
    Z_med_train = load_Z(K, "medalcare", "train")
    if pool_mode == "asymmetric":
        Z_ptb_pool = load_Z(K, "ptbxl", "test")
        real_split_in_pool = "test"
    elif pool_mode == "symmetric":
        Z_ptb_pool = load_Z(K, "ptbxl", "train")
        real_split_in_pool = "train"
    else:
        raise ValueError(pool_mode)
    n_med, n_ptb = Z_med_train.shape[0], Z_ptb_pool.shape[0]
    Z_pool = np.vstack([Z_med_train, Z_ptb_pool])
    domain = np.concatenate(
        [np.zeros(n_med, dtype=np.int64), np.ones(n_ptb, dtype=np.int64)]
    )
    D = Z_pool.shape[1]
    assert D == K, f"latents at K={K} have dim {D}"
    print(f"[load] pool: med_train(n={n_med}) + ptbxl_{real_split_in_pool}(n={n_ptb}); D={D}")

    # Fit scaler on combined pool
    scaler = StandardScaler().fit(Z_pool)
    Z_scaled = scaler.transform(Z_pool)

    # Pre-INLP alignment metrics on scaled pool
    metrics_orig_pool = compute_alignment_metrics(
        Z_scaled[domain == 0], Z_scaled[domain == 1], seed=SEED,
    )
    print(
        f"[orig pool] c2st={metrics_orig_pool['c2st_auroc']:.4f} "
        f"mmd={metrics_orig_pool['mmd_rbf']:.4e} "
        f"knn_mix={metrics_orig_pool['knn_mixing']:.4f}"
    )

    # INLP
    P_total, iteration_log = inlp_fit(
        Z_scaled, domain,
        max_iter=max_iter, stop_acc=STOP_ACC, seed=SEED,
    )

    # Sanity: projector residual on a projected vector
    rng_check = np.random.default_rng(SEED)
    v = rng_check.normal(size=D)
    v_proj = P_total @ v
    residual = float(
        np.linalg.norm(P_total.T @ P_total @ v_proj - v_proj)
        / max(np.linalg.norm(v_proj), 1e-12)
    )
    print(f"[sanity] P^T P projector residual = {residual:.3e}")

    # Post-INLP metrics on the aligned pool
    Z_pool_aligned = Z_scaled @ P_total
    metrics_inlp_pool = compute_alignment_metrics(
        Z_pool_aligned[domain == 0], Z_pool_aligned[domain == 1], seed=SEED,
    )
    print(
        f"[inlp pool] c2st={metrics_inlp_pool['c2st_auroc']:.4f} "
        f"mmd={metrics_inlp_pool['mmd_rbf']:.4e} "
        f"knn_mix={metrics_inlp_pool['knn_mixing']:.4f}"
    )

    # Save projection + scaler
    proj_subdir = (
        stem_for_K(K) if output_suffix == "_inlp"
        else f"{stem_for_K(K)}{output_suffix}"
    )
    proj_dir = OUT_DIR / proj_subdir
    proj_dir.mkdir(parents=True, exist_ok=True)
    n_iter_run = sum(1 for r in iteration_log if r["iter"] > 0)
    np.savez(
        proj_dir / "projection.npz",
        P_total=P_total.astype(np.float64),
        scaler_mean=scaler.mean_.astype(np.float64),
        scaler_scale=scaler.scale_.astype(np.float64),
        n_iter=np.array([n_iter_run], dtype=np.int64),
    )

    # Apply alignment to ALL splits and save aligned latents
    saved: Dict[str, Dict] = {}
    for dom in DOMAINS:
        for spl in SPLITS:
            src = latent_path(K, dom, spl)
            if not src.exists():
                print(f"[skip] {src.relative_to(REPO_ROOT)} missing")
                continue
            dst = latent_path(K, dom, spl, suffix=output_suffix)
            dst.parent.mkdir(parents=True, exist_ok=True)
            with np.load(src, allow_pickle=True) as data:
                payload = {k: data[k] for k in data.keys()}
            Z_orig = payload["Z"]
            Z_aligned = apply_alignment(
                Z_orig.astype(np.float64), scaler, P_total,
            )
            assert Z_aligned.shape == Z_orig.shape
            n_nan = int(np.isnan(Z_aligned).sum())
            n_inf = int(np.isinf(Z_aligned).sum())
            if n_nan or n_inf:
                raise RuntimeError(
                    f"NaN/inf in aligned K={K} {dom}/{spl}: NaN={n_nan} inf={n_inf}"
                )
            payload["Z"] = Z_aligned
            np.savez(dst, **payload)
            saved[f"{dom}_{spl}"] = {
                "src": str(src.relative_to(REPO_ROOT)),
                "dst": str(dst.relative_to(REPO_ROOT)),
                "shape": list(Z_aligned.shape),
            }
            print(f"[save] {dst.relative_to(REPO_ROOT)}  shape={Z_aligned.shape}")

    # Held-out verification on TEST splits (downstream-B2 view)
    Z_med_test_orig = load_Z(K, "medalcare", "test")
    Z_ptb_test_orig = load_Z(K, "ptbxl", "test")
    Z_med_test_orig_sc = scaler.transform(Z_med_test_orig)
    Z_ptb_test_orig_sc = scaler.transform(Z_ptb_test_orig)
    metrics_orig_test = compute_alignment_metrics(
        Z_med_test_orig_sc, Z_ptb_test_orig_sc, seed=SEED,
    )
    Z_med_test_inlp = load_Z(K, "medalcare", "test", suffix=output_suffix)
    Z_ptb_test_inlp = load_Z(K, "ptbxl", "test", suffix=output_suffix)
    metrics_inlp_test = compute_alignment_metrics(
        Z_med_test_inlp.astype(np.float64),
        Z_ptb_test_inlp.astype(np.float64),
        seed=SEED,
    )
    print(
        f"[held-out test] "
        f"orig c2st={metrics_orig_test['c2st_auroc']:.4f} mmd={metrics_orig_test['mmd_rbf']:.4e} | "
        f"inlp c2st={metrics_inlp_test['c2st_auroc']:.4f} mmd={metrics_inlp_test['mmd_rbf']:.4e}"
    )

    # Iteration log JSON
    with (proj_dir / "iteration_log.json").open("w", encoding="utf-8") as f:
        json.dump({
            "K": K,
            "pool_mode": pool_mode,
            "output_suffix": output_suffix,
            "n_med_pool": int(n_med),
            "n_ptb_pool": int(n_ptb),
            "max_iter": int(max_iter),
            "stop_acc": float(STOP_ACC),
            "seed": int(SEED),
            "n_iter_run": int(n_iter_run),
            "ptp_residual": float(residual),
            "metrics_orig_pool": metrics_orig_pool,
            "metrics_inlp_pool": metrics_inlp_pool,
            "metrics_orig_test": metrics_orig_test,
            "metrics_inlp_test": metrics_inlp_test,
            "iterations": iteration_log,
        }, f, indent=2)
    print(f"[save] {(proj_dir / 'iteration_log.json').relative_to(REPO_ROOT)}")

    # PCA plot
    plot_pca_before_after(
        Z_scaled, Z_pool_aligned, domain,
        out_path=proj_dir / "pca_before_after.png",
        title_prefix=f"K={K} bottleneck ({pool_mode})",
        pool_mode=pool_mode,
        seed=SEED,
    )

    return {
        "K": K,
        "pool_mode": pool_mode,
        "output_suffix": output_suffix,
        "n_med_pool": int(n_med),
        "n_ptb_pool": int(n_ptb),
        "n_iter_run": int(n_iter_run),
        "metrics_orig_pool": metrics_orig_pool,
        "metrics_inlp_pool": metrics_inlp_pool,
        "metrics_orig_test": metrics_orig_test,
        "metrics_inlp_test": metrics_inlp_test,
        "saved_splits": saved,
        "ptp_residual": float(residual),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--ks", type=int, nargs="+", default=K_LIST,
        help="Bottleneck dimensions to run (asymmetric pool).",
    )
    ap.add_argument(
        "--symmetric-K", type=int, default=None,
        help="Optional: also run a SYMMETRIC sensitivity at this K (e.g. 64).",
    )
    ap.add_argument(
        "--max-iter-per-K", type=int, default=None,
        help="Override max_iter (default: min(50, K-1) per K).",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "inlp_lowK_summary.json"
    summary: Dict[str, Dict] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            print(f"[merge] loaded existing summary with keys={list(summary.keys())}")
        except Exception as exc:
            print(f"[warn] could not parse existing summary: {exc}")

    # Asymmetric runs
    for K in args.ks:
        max_iter = args.max_iter_per_K if args.max_iter_per_K is not None else min(50, K - 1)
        if max_iter < 1:
            print(f"[skip] K={K}: max_iter={max_iter} too small")
            continue
        key = f"asym_K{K}"
        res = fit_one_K(K, pool_mode="asymmetric", output_suffix="_inlp", max_iter=max_iter)
        summary[key] = res
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # Symmetric sensitivity (optional)
    if args.symmetric_K is not None:
        K = args.symmetric_K
        max_iter = args.max_iter_per_K if args.max_iter_per_K is not None else min(50, K - 1)
        key = f"sym_K{K}"
        res = fit_one_K(K, pool_mode="symmetric", output_suffix="_inlpv2", max_iter=max_iter)
        summary[key] = res
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # Pretty table
    print()
    print("=" * 88)
    print(
        f"{'key':<12s} {'n_iter':>6s} {'C2ST_orig':>10s} {'C2ST_inlp':>10s}  "
        f"{'MMD_orig':>10s} {'MMD_inlp':>10s}  {'TST_c2st_o':>11s} {'TST_c2st_i':>11s}"
    )
    print("=" * 88)
    for k, r in summary.items():
        m0 = r["metrics_orig_pool"]; m1 = r["metrics_inlp_pool"]
        t0 = r["metrics_orig_test"]; t1 = r["metrics_inlp_test"]
        print(
            f"{k:<12s} {r['n_iter_run']:>6d} "
            f"{m0['c2st_auroc']:>10.4f} {m1['c2st_auroc']:>10.4f}  "
            f"{m0['mmd_rbf']:>10.3e} {m1['mmd_rbf']:>10.3e}  "
            f"{t0['c2st_auroc']:>11.4f} {t1['c2st_auroc']:>11.4f}"
        )
    print(f"\nWrote {summary_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
