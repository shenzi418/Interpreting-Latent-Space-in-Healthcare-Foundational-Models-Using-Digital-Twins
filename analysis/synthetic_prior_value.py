"""Tier 1: is the synthetic readout worth anything once a few real labels exist?

Zero-shot transport of the MedalCare territory readout onto PTB-XL fails. That is
a statement about one specific operating point -- zero real labels -- and it is
the least interesting point on the curve. The question a digital twin is actually
built to answer is whether the simulator buys you anything at the operating points
people work at.

Four estimators of the PTB-XL circular readout, all evaluated on a fixed
patient-disjoint held-out set, as a function of the number of labelled real
records n:

  scratch    ridge on the n real records only -- the baseline to beat
  prior      ridge shrunk toward the MedalCare readout instead of toward zero,
             so the simulator supplies the prior mean rather than the answer
  plane      the n real labels are only allowed to fit a 2x2 map inside the
             MedalCare readout plane (4 free parameters, whatever n is)
  frozen     zero-shot MedalCare transport -- constant in n, drawn for reference

`plane` is the sharp test. It asks whether the synthetic plane *contains* the
real signal in some orientation, which is a far weaker and more plausible claim
than that it points the right way inside that plane. If `plane` is at chance the
two subspaces really are disjoint; if it rises quickly it means the failure is a
rotation and a handful of real labels fixes it.

Read-only over stored latents. No retraining.

Usage:
  python analysis/synthetic_prior_value.py
  python analysis/synthetic_prior_value.py --encoders exp8_leadfix_medalonly
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

from geom_common import (  # pylint: disable=wrong-import-position
    ENCODERS,
    Domain,
    RidgeSVD,
    angles_from_cs,
    load_medalcare,
    load_ptbxl,
    medalcare_anchor_angles,
    resultant,
)

OUT_DIR = REPO_ROOT / "outputs/analysis/circular_geometry"
N_GRID = [10, 20, 50, 100, 200, 500, 1000, 2000]
N_REPEAT = 25
TEST_FRAC = 0.3


def _cs(a: np.ndarray) -> np.ndarray:
    return np.c_[np.cos(a), np.sin(a)]


def _pooled(a: np.ndarray, b: np.ndarray):
    both = np.vstack([a, b])
    mu, sd = both.mean(0), both.std(0) + 1e-8
    return (a - mu) / sd, (b - mu) / sd


def _fit_scratch(x, y):
    m = RidgeSVD().fit(x, y)
    return lambda xt: m.predict(xt)


def _fit_prior(x, y, b_prior, x_mu, x_sd):
    """Ridge whose shrinkage target is the MedalCare readout, not zero.

    Minimises ||y - xB||^2 + a||B - B_prior||^2, solved by fitting the ordinary
    ridge to the residual of the prior's own predictions.
    """
    base = ((x - x_mu) / x_sd) @ b_prior
    m = RidgeSVD().fit(x, y - base)
    return lambda xt: m.predict(xt) + ((xt - x_mu) / x_sd) @ b_prior


def _fit_plane(x, y, q_prior, x_mu, x_sd):
    """Only 4 free parameters: a 2x2 map inside the frozen MedalCare plane."""
    proj = ((x - x_mu) / x_sd) @ q_prior          # (n, 2)
    m = RidgeSVD(alphas=np.logspace(-3, 3, 15)).fit(proj, y)
    return lambda xt: m.predict(((xt - x_mu) / x_sd) @ q_prior)


def run_encoder(encoder: str, anchors: dict[str, float], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    medal = load_medalcare(encoder)
    ptb = load_ptbxl(encoder, anchors)
    zm, zp = _pooled(medal.z, ptb.z)

    # The synthetic readout, fit once on all MedalCare rows in the shared basis.
    med_model = RidgeSVD().fit(zm, _cs(medal.angle))
    b_prior = med_model.direction_matrix()
    q_prior, _ = np.linalg.qr(b_prior)
    x_mu, x_sd = med_model.mu_, med_model.sd_

    print(f"\n=== {encoder}  (d={zm.shape[1]}) ===")
    print(f"{'n':>6}" + "".join(f"{k:>18}" for k in ("scratch", "prior", "plane")))
    print("-" * 60)

    out: dict = {"encoder": encoder, "latent_dim": int(zm.shape[1]), "curve": {}}
    patients = np.unique(ptb.group)

    for n in N_GRID:
        acc = {k: [] for k in ("scratch", "prior", "plane", "frozen")}
        for _ in range(N_REPEAT):
            perm = rng.permutation(patients)
            n_test = int(len(perm) * TEST_FRAC)
            test_p = set(perm[:n_test].tolist())
            is_test = np.array([g in test_p for g in ptb.group])
            te = np.where(is_test)[0]
            pool = np.where(~is_test)[0]
            if len(pool) < n:
                continue
            tr = rng.choice(pool, size=n, replace=False)

            xtr, ytr, xte = zp[tr], _cs(ptb.angle[tr]), zp[te]
            truth = ptb.angle[te]
            for key, fit in (
                ("scratch", lambda: _fit_scratch(xtr, ytr)),
                ("prior", lambda: _fit_prior(xtr, ytr, b_prior, x_mu, x_sd)),
                ("plane", lambda: _fit_plane(xtr, ytr, q_prior, x_mu, x_sd)),
            ):
                acc[key].append(resultant(angles_from_cs(fit()(xte)) - truth))
            acc["frozen"].append(
                resultant(angles_from_cs(med_model.predict(xte)) - truth)
            )

        row = {
            k: {
                "mean": float(np.mean(v)),
                "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
            }
            for k, v in acc.items()
            if v
        }
        # Paired comparison on the same splits -- the only fair prior-vs-scratch test.
        if acc["scratch"] and acc["prior"]:
            d_prior = np.array(acc["prior"]) - np.array(acc["scratch"])
            d_plane = np.array(acc["plane"]) - np.array(acc["scratch"])
            row["delta_prior_minus_scratch"] = {
                "mean": float(d_prior.mean()),
                "frac_positive": float(np.mean(d_prior > 0)),
            }
            row["delta_plane_minus_scratch"] = {
                "mean": float(d_plane.mean()),
                "frac_positive": float(np.mean(d_plane > 0)),
            }
        out["curve"][str(n)] = row
        print(f"{n:>6}" + "".join(
            f"{row[k]['mean']:>10.3f}{'':>8}" if k in row else f"{'--':>18}"
            for k in ("scratch", "prior", "plane")
        ))

    frozen = out["curve"][str(N_GRID[0])].get("frozen", {}).get("mean", float("nan"))
    print(f"{'frozen':>6}{frozen:>10.3f}   (zero-shot MedalCare transport, "
          f"constant in n)")
    out["frozen_zero_shot_R"] = frozen
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoders", nargs="*", default=ENCODERS)
    ap.add_argument("--seed", type=int, default=20260812)
    args = ap.parse_args()

    anchors = medalcare_anchor_angles()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_res = {}
    for enc in args.encoders:
        try:
            all_res[enc] = run_encoder(enc, anchors, args.seed)
        except FileNotFoundError as exc:
            print(f"\n=== {enc}: SKIPPED ({exc}) ===")

    dest = OUT_DIR / "synthetic_prior_value.json"
    dest.write_text(
        json.dumps({"n_grid": N_GRID, "n_repeat": N_REPEAT, "encoders": all_res},
                   indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
