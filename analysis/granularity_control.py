"""Matched-granularity control for the in-domain territory readout.

The pivot document reported that territory is recovered *better* from real ECG
(PTB-XL, R=0.798, median 16.0 deg) than from synthetic (MedalCare, R=0.657,
median 29.1 deg), and read that as a substantive finding.

It is not comparable as stated. The two targets have different granularity:

  MedalCare  phi is continuous on the circle  -> label-shuffle null R = 0.034
  PTB-XL     territory_4c has 4 values only   -> label-shuffle null R = 0.243

A 4-valued target is far easier to hit: getting the right one of four quadrant
centres scores R near 1, whereas the continuous target penalises every degree of
within-quadrant error. The 7x difference in chance level makes the raw R
comparison meaningless.

This script removes the confound by quantising MedalCare's phi to its own
territory anchor, so both domains are scored on an identical 4-level circular
target with an identical null, and reports the comparison both ways.

Read-only. Usage:
  python analysis/granularity_control.py
"""

from __future__ import annotations

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

from geom_common import (  # pylint: disable=wrong-import-position
    ENCODERS,
    Domain,
    RidgeSVD,
    angles_from_cs,
    group_folds,
    load_medalcare,
    load_ptbxl,
    med_abs_deg,
    medalcare_anchor_angles,
    resultant,
)

OUT_DIR = REPO_ROOT / "outputs/analysis/circular_geometry"
N_FOLDS = 5
N_PERM = 200


def cv_readout(dom: Domain, target: np.ndarray, rng: np.random.Generator) -> dict:
    """Group-disjoint CV circular readout of `target` from `dom.z`, with a
    label-shuffle null that reuses each fold's cached factorisation."""
    pred = np.empty_like(target)
    null = np.zeros(N_PERM)
    y = np.c_[np.cos(target), np.sin(target)]

    for tr, te in group_folds(dom.group, N_FOLDS, rng):
        model = RidgeSVD().fit(dom.z[tr], y[tr])
        pred[te] = angles_from_cs(model.predict(dom.z[te]))
        for j in range(N_PERM):
            ysh = y[rng.permutation(tr)]
            model.solve(ysh)
            null[j] += resultant(
                angles_from_cs(model.predict(dom.z[te])) - target[te]
            ) * len(te)
    null /= len(target)

    r = resultant(pred - target)
    return {
        "R": r,
        "median_abs_err_deg": med_abs_deg(pred - target),
        "R_null_mean": float(null.mean()),
        "perm_p": (1 + int((null >= r).sum())) / (N_PERM + 1),
        # R expressed on its own null-to-ceiling scale, which is the only way the
        # two domains can be put on one axis when the nulls differ.
        "R_above_null": float(r - null.mean()),
        "R_normalised": float((r - null.mean()) / max(1.0 - null.mean(), 1e-9)),
    }


def main() -> None:
    anchors = medalcare_anchor_angles()
    rows = {}

    for enc in ENCODERS:
        try:
            medal = load_medalcare(enc)
            ptb = load_ptbxl(enc, anchors)
        except FileNotFoundError as exc:
            print(f"=== {enc}: SKIPPED ({exc}) ===")
            continue

        # MedalCare, quantised to its own territory anchor -> same 4-level target
        # and same chance level as PTB-XL.
        medal_q = np.array([anchors[t] for t in medal.territory])

        rng = np.random.default_rng(20260812)
        cont = cv_readout(medal, medal.angle, rng)
        rng = np.random.default_rng(20260812)
        quant = cv_readout(medal, medal_q, rng)
        rng = np.random.default_rng(20260812)
        real = cv_readout(ptb, ptb.angle, rng)

        rows[enc] = {
            "medalcare_continuous": cont,
            "medalcare_quantised_4level": quant,
            "ptbxl_4level": real,
        }

        print(f"\n=== {enc} ===")
        for name, r in (
            ("MedalCare phi (continuous)", cont),
            ("MedalCare 4-level (matched)", quant),
            ("PTB-XL   4-level", real),
        ):
            print(f"  {name:<28} R={r['R']:.4f}  med={r['median_abs_err_deg']:5.1f}deg  "
                  f"null={r['R_null_mean']:.3f}  R-null={r['R_above_null']:+.4f}  "
                  f"norm={r['R_normalised']:.4f}")

    dest = OUT_DIR / "granularity_control.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n" + "=" * 92)
    print(f"{'encoder':<26}{'M cont':>9}{'M 4lvl':>9}{'P 4lvl':>9}"
          f"{'M norm':>9}{'P norm':>9}{'winner (matched)':>20}")
    print("-" * 92)
    for enc, r in rows.items():
        mq, pq = r["medalcare_quantised_4level"], r["ptbxl_4level"]
        win = "MedalCare" if mq["R_normalised"] > pq["R_normalised"] else "PTB-XL"
        print(f"{enc:<26}{r['medalcare_continuous']['R']:>9.3f}"
              f"{mq['R']:>9.3f}{pq['R']:>9.3f}"
              f"{mq['R_normalised']:>9.3f}{pq['R_normalised']:>9.3f}{win:>20}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
