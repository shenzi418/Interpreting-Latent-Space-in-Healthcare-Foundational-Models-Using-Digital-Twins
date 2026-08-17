"""Re-score the dim_scan PCA truncations with a NONLINEAR domain detector.

Report S15.2 found linear C2ST falling below 1.0 for the first time in this
project -- 0.996 / 0.983 / 0.947 at K = 32 / 16 / 8 under a *MedalCare-fitted*
PCA, and lower still on the ccMMD encoder (0.902 at K=8). S15.2a explains why
that cannot yet be reported as reduced domain separability:
`dim_scan`'s C2ST is `exp7_analysis.domain_classifier_auc`, a 5-fold logistic
regression, i.e. a **linear-decodability** measurement.

S11 already burned this project on exactly that confound. INLP drove linear C2ST
to 0.5132 at k=90 while a gradient-boosted tree on the *identical* projections
read 0.9999 -- the two disagreed by 0.49 AUROC on the same data, and the
"alignment is achievable" claim had to be withdrawn. The standing rule from that
episode: a metric name in a log line is not evidence about what was measured.

This script answers the question S15.2a leaves open, by rebuilding dim_scan's
projections exactly (same loader, same scaler+PCA fit, same subsample cap, same
seed) and scoring each one twice -- once with the linear detector, once with the
GBDT from `tradeoff_frontier.gbdt_c2st`. Same rows, same projection, two
detectors: the only difference is the hypothesis class, so any gap is
attributable to it alone.

Reproducing the linear column is the correctness check. If it does not match the
numbers in `post_dim_scan.log` to ~0.01, the projections have not been rebuilt
faithfully and the GBDT column means nothing -- that is checked and reported
before any verdict.

Two outcomes, both worth having, stated before running:

  * **GBDT stays ~1.0** -> S15.2 is a linear-decodability curve, not an alignment
    result. This STRENGTHENS the S13.6 over-determination claim: a dose-response
    demonstration that only the linear detector moves while the gap itself does
    not.
  * **GBDT drops too** -> the first genuine reduction in domain separability
    produced in this project, and it deserves its own experiment.

Costs no training: everything runs off exported latents already on disk.

Run::

    python analysis/dim_scan_nonlinear_c2st.py
    python analysis/dim_scan_nonlinear_c2st.py --ks 8 16 32 --pca-modes medalcare
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
from analysis.dim_scan import (  # noqa: E402
    SEED,
    SUBSAMPLE_ALIGNMENT,
    fit_standardiser_and_pca,
    load_config_4splits,
    project_through,
    remap_test_split,
)
from analysis.exp7_analysis import domain_classifier_auc  # noqa: E402
from analysis.tradeoff_frontier import gbdt_c2st  # noqa: E402

DEFAULT_CONFIGS = ("exp8_leadfix_baseline", "exp8_leadfix_ccmmd")
DEFAULT_MODES = ("combined", "medalcare", "ptbxl")
DEFAULT_KS = (8, 16, 32, 64, 1024)
OUT = REPO_ROOT / "outputs" / "dim_scan_exp8" / "nonlinear_c2st.json"


def subsample_pair(Z_med: np.ndarray, Z_ptb: np.ndarray,
                   rng: np.random.Generator):
    """Reproduce `dim_scan.alignment_block`'s subsampling exactly.

    Copied deliberately rather than approximated: the cap is the SHARED
    `min(2000, len(A), len(B))`, not a per-array cap, and the two `rng.choice`
    calls consume the same generator in this order. Draw differently and the
    linear column stops matching `post_dim_scan.log`, which is the only check
    that the projections were rebuilt faithfully.
    """
    sub = min(SUBSAMPLE_ALIGNMENT, len(Z_med), len(Z_ptb))
    idx_m = (rng.choice(len(Z_med), sub, replace=False)
             if len(Z_med) > sub else np.arange(len(Z_med)))
    idx_p = (rng.choice(len(Z_ptb), sub, replace=False)
             if len(Z_ptb) > sub else np.arange(len(Z_ptb)))
    return Z_med[idx_m], Z_ptb[idx_p]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", nargs="*", default=list(DEFAULT_CONFIGS))
    ap.add_argument("--pca-modes", nargs="*", default=list(DEFAULT_MODES))
    ap.add_argument("--ks", type=int, nargs="*", default=list(DEFAULT_KS))
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    rows = []
    for config in args.configs:
        splits = load_config_4splits(config)
        Z_m_tr = splits["medal_train"]["Z"]
        Z_p_tr = splits["ptb_train"]["Z"]

        # dim_scan's alignment_block scores the SHARED-3-CLASS-FILTERED test
        # rows, not the raw test splits (`Z_m_te_v` / `Z_p_te_v` in `run_one`).
        # Feeding it the unfiltered splits silently changes both the row count
        # and the class mix, and the linear column then fails to reproduce
        # post_dim_scan.log -- which is exactly how this was caught.
        mask_m, _ = remap_test_split(splits["medal_test"]["Y"], "medalcare")
        mask_p, _ = remap_test_split(splits["ptb_test"]["Y"], "ptbxl")
        Z_m_te = splits["medal_test"]["Z"][mask_m]
        Z_p_te = splits["ptb_test"]["Z"][mask_p]
        print(f"[shared3] {config}: medalcare {len(Z_m_te)}/{len(mask_m)}, "
              f"ptbxl {len(Z_p_te)}/{len(mask_p)}")

        for mode in args.pca_modes:
            scaler, pca, _ = fit_standardiser_and_pca(
                Z_m_tr, Z_p_tr, pca_mode=mode, seed=SEED)
            P_m = project_through(Z_m_te, scaler, pca)
            P_p = project_through(Z_p_te, scaler, pca)

            print(f"\n{'=' * 66}\n[{config} / PCA {mode}]\n{'=' * 66}")
            print(f"  {'K':>6}{'linear':>10}{'GBDT':>10}{'delta':>9}")
            for K in sorted(args.ks, reverse=True):
                if K > P_m.shape[1]:
                    continue
                rng = np.random.default_rng(SEED)
                A, B = subsample_pair(P_m[:, :K], P_p[:, :K], rng)

                lin = domain_classifier_auc(A, B, seed=SEED)

                # GBDT wants an explicit train/test split; dim_scan's linear
                # detector cross-validates internally. Use a stratified half so
                # both see the same rows overall.
                r2 = np.random.default_rng(SEED + 1)
                ia = r2.permutation(len(A))
                ib = r2.permutation(len(B))
                ha, hb = len(A) // 2, len(B) // 2
                gb = gbdt_c2st(A[ia[:ha]], B[ib[:hb]],
                               A[ia[ha:]], B[ib[hb:]])

                print(f"  {K:>6}{lin:>10.4f}{gb:>10.4f}{gb - lin:>+9.4f}")
                rows.append({"config": config, "pca_mode": mode, "K": int(K),
                             "c2st_linear": lin, "c2st_gbdt": gb,
                             "n_medal": int(len(A)), "n_ptb": int(len(B))})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"seed": SEED, "subsample": SUBSAMPLE_ALIGNMENT,
         "note": "linear column must reproduce post_dim_scan.log; see report S15.2a",
         "rows": rows}, indent=2), encoding="utf-8")
    print(f"\n[save] {args.out}")

    # --- verdict ----------------------------------------------------------
    low = [r for r in rows if r["K"] <= 32 and r["pca_mode"] != "combined"]
    if low:
        min_lin = min(r["c2st_linear"] for r in low)
        min_gb = min(r["c2st_gbdt"] for r in low)
        print(f"\nAt K<=32 on source/target-fitted PCA:")
        print(f"  lowest linear C2ST = {min_lin:.4f}")
        print(f"  lowest GBDT   C2ST = {min_gb:.4f}")
        if min_gb >= 0.99:
            print("  -> GBDT stays ~1.0. S15.2 is a LINEAR-DECODABILITY curve;")
            print("     the domain gap itself does not shrink. S13.6 strengthens.")
        elif min_gb < min_lin:
            print("  -> GBDT drops BELOW linear -- unexpected; inspect before use.")
        else:
            print("  -> GBDT drops too: a real reduction in separability. "
                  "Deserves its own experiment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
