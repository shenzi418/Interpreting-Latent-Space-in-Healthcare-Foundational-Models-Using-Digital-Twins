"""Sanity check for the new --scaler-domain target_pool_measured mode.

Three things must hold before the mode is used for anything:
  1. For a matrix with no NaN (the latent arm), it is IDENTICAL to target_pool.
  2. For the real spatial54 PTB-XL pool it differs materially from target_pool,
     in the predicted direction (imputed rows deflate target_pool's variance).
  3. The all-NaN-column fallback fires loudly rather than producing NaN output.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
import numpy as np  # noqa: E402

from analysis.phase_b2_infarct_decoding import (  # noqa: E402
    fit_scaler, fit_scaler_nanaware, standardise_target,
    load_features, load_ptbxl_features,
)

rng = np.random.default_rng(0)

# --- 1. no-NaN equivalence -------------------------------------------------
X = rng.normal(3.0, 2.0, size=(500, 20))
src = fit_scaler(rng.normal(0.0, 1.0, size=(100, 20)))
a = standardise_target(X, src, "target_pool", pool=X, pool_raw=X)
b = standardise_target(X, src, "target_pool_measured", pool=X, pool_raw=X)
print(f"[1] no-NaN pool: max|target_pool - target_pool_measured| = "
      f"{np.abs(a - b).max():.3e}   -> {'PASS' if np.allclose(a, b) else 'FAIL'}")

# --- 2. real spatial54 pool ------------------------------------------------
# The feature set is selected by rebinding module-level paths, mirroring what
# `--feature-set spatial54` does in main().
import analysis.phase_b2_infarct_decoding as pb2  # noqa: E402
DATA = REPO_ROOT / "data"
pb2.FEAT_TRAIN_PATH = DATA / "ecg_features_spatial_medalcare_train.npz"
pb2.FEAT_TEST_PATH = DATA / "ecg_features_spatial_medalcare_test.npz"
pb2.FEAT_PTBXL_PATH = DATA / "ecg_features_spatial_ptbxl_test.npz"

feat_tr, _, _, _, names = pb2.load_features()
medians = np.nanmedian(feat_tr, axis=0)
px_raw, _ = pb2.load_ptbxl_features()
px_raw = px_raw.astype(np.float64)
n_all_nan = int(np.isnan(px_raw).all(axis=1).sum())
print(f"\n[2] PTB-XL spatial54 pool: {px_raw.shape[0]} rows, "
      f"{n_all_nan} entirely NaN ({100*n_all_nan/px_raw.shape[0]:.1f}%)")

px_imp = px_raw.copy()
for j in range(px_imp.shape[1]):
    px_imp[np.isnan(px_imp[:, j]), j] = medians[j]

src_feat = fit_scaler(np.where(np.isnan(feat_tr), medians, feat_tr))
sc_pool = fit_scaler(px_imp)
sc_meas = fit_scaler_nanaware(px_raw, src_feat)

ratio = sc_meas.scale_ / sc_pool.scale_
print(f"    std ratio measured/pooled: median={np.median(ratio):.2f}  "
      f"min={ratio.min():.2f}  max={ratio.max():.2f}")
print(f"    -> pooled std is deflated by ~{np.median(ratio):.1f}x "
      f"{'(as predicted)' if np.median(ratio) > 1.2 else '(NOT as predicted)'}")
shift = np.abs(sc_meas.mean_ - sc_pool.mean_) / np.maximum(sc_meas.scale_, 1e-12)
print(f"    mean shift in measured-std units: median={np.median(shift):.3f}  "
      f"max={shift.max():.3f}")

# --- 3. dead-column fallback ----------------------------------------------
X_dead = rng.normal(0.0, 1.0, size=(50, 5))
X_dead[:, 2] = np.nan
src5 = fit_scaler(rng.normal(7.0, 3.0, size=(50, 5)))
print("\n[3] all-NaN column fallback (expect one warning line):")
sc = fit_scaler_nanaware(X_dead, src5)
ok = (sc.mean_[2] == src5.mean_[2] and sc.scale_[2] == src5.scale_[2]
      and np.isfinite(sc.mean_).all() and np.isfinite(sc.scale_).all())
print(f"    dead column took source stats, all finite -> {'PASS' if ok else 'FAIL'}")
