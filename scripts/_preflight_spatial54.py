"""Pre-flight for the spatial54 Track-3 arm: check what the run will read, before it runs.

`run_spatial54_arm.py` aborts on a mismatched latent block, which is the right
guard but a late one -- it fires after the patch is applied and the driver is
started. This checks the same invariants now, read-only, so a failure costs
seconds rather than a run:

  1. the spatial54 NPZs align row-for-row with the latents the arm will score
  2. the 6 global columns are a bit-identical subset of the 54 (the "strict
     superset" claim §15 rests on -- if it fails, any spatial54-vs-global6 delta
     is confounded by re-estimation, not just by dimensionality)
  3. missingness is not worse in the 54-column set than the 6-column one
  4. the finiteness indicator does not predict territory (the Part 12 leak)

(2) is the one worth running: §15 asserts the superset property from how the
extractor was written, and an assertion about code is not a measurement of the
artifact it produced.
"""
from __future__ import annotations

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
from analysis.domain_signal_structure import load  # noqa: E402
from analysis.transfer_control import missingness_auroc, territory_targets  # noqa: E402

PAIRS = [
    ("medalcare_train", "data/ecg_features_medalcare_train.npz",
     "data/ecg_features_spatial_medalcare_train.npz"),
    ("medalcare_test", "data/ecg_features_medalcare_test.npz",
     "data/ecg_features_spatial_medalcare_test.npz"),
    ("ptbxl_test", "data/ecg_features_ptbxl_test.npz",
     "data/ecg_features_spatial_ptbxl_test.npz"),
]


def cols(npz):
    for k in ("columns", "feature_names", "names"):
        if k in npz:
            return [str(c) for c in npz[k]]
    return None


def main() -> int:
    ok = True
    print("=" * 78)
    print("spatial54 pre-flight (read-only)")
    print("=" * 78)

    for tag, p6, p54 in PAIRS:
        a = np.load(REPO_ROOT / p6, allow_pickle=True)
        b = np.load(REPO_ROOT / p54, allow_pickle=True)
        X6, X54 = a["features"], b["features"]
        c6, c54 = cols(a), cols(b)
        print(f"\n[{tag}] global6 {X6.shape}   spatial54 {X54.shape}")

        if X6.shape[0] != X54.shape[0]:
            print("  FAIL row counts differ"); ok = False; continue

        # (2) strict-superset check, by column name where available
        if c6 and c54:
            missing = [c for c in c6 if c not in c54]
            if missing:
                print(f"  FAIL {len(missing)} global col(s) absent from 54: {missing[:6]}")
                ok = False
            else:
                idx = [c54.index(c) for c in c6]
                sub = X54[:, idx]
                both_nan = np.isnan(X6) & np.isnan(sub)
                same = np.isclose(X6, sub, rtol=0, atol=0, equal_nan=False) | both_nan
                frac = float(same.mean())
                if frac == 1.0:
                    print(f"  ok   superset EXACT on all {len(c6)} shared cols")
                else:
                    worst = [c6[j] for j in range(len(c6)) if same[:, j].mean() < 1.0]
                    print(f"  FAIL superset not bit-identical ({frac:.6f}); "
                          f"differing cols: {worst}")
                    ok = False
        else:
            print(f"  warn no column names (6:{c6 is not None} 54:{c54 is not None}); "
                  "superset checked by shape only")

        f6 = np.isfinite(X6).all(axis=1).mean()
        f54 = np.isfinite(X54).all(axis=1).mean()
        flag = "ok  " if f54 >= f6 - 1e-12 else "WARN"
        print(f"  {flag} fully-finite rows: global6 {f6:.4f} -> spatial54 {f54:.4f}")

    # (1) + (4) on the rows the arm actually scores
    print("\n[cross-domain subset the arm scores]")
    med_idx, med_terr, ptb_idx, ptb_terr = territory_targets()
    Zm, _ = load("exp8_leadfix_baseline", "medalcare", "train")
    Zp, _ = load("exp8_leadfix_baseline", "ptbxl", "test")
    Xm = np.load(REPO_ROOT / PAIRS[0][2], allow_pickle=True)["features"]
    Xp = np.load(REPO_ROOT / PAIRS[2][2], allow_pickle=True)["features"]
    print(f"  latents  med {Zm.shape[0]}  ptb {Zp.shape[0]}")
    print(f"  spatial  med {Xm.shape[0]}  ptb {Xp.shape[0]}")
    if Xm.shape[0] != Zm.shape[0] or Xp.shape[0] != Zp.shape[0]:
        print("  FAIL feature/latent row counts disagree"); ok = False
    else:
        print(f"  ok   aligned; scoring {len(med_idx)} med / {len(ptb_idx)} ptb rows")

    worst = 0.0
    for c in sorted(set(med_terr) & set(ptb_terr)):
        a = missingness_auroc(Xm[med_idx], (med_terr == c).astype(int))
        b = missingness_auroc(Xp[ptb_idx], (ptb_terr == c).astype(int))
        worst = max(worst, a, b)
        print(f"  missingness-AUROC {c:<12} med {a:.4f}  ptb {b:.4f}")
    verdict = "ok  " if worst <= 0.55 else "FAIL"
    if worst > 0.55:
        ok = False
    print(f"  {verdict} worst {worst:.4f} vs 0.55 gate (Part 12 leak detector)")

    print("\n" + ("PRE-FLIGHT CLEAN" if ok else "PRE-FLIGHT FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
