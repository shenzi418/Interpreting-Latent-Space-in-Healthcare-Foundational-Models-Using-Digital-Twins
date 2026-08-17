"""Tier 1: circular territory readout, transport, and anchor sensitivity.

Replaces the 4-class macro-F1 endpoint, which binned a continuous circular
variable (phi) into 90-degree cells and destroyed the structure it was meant to
measure. Everything here is read-only over stored latents -- no retraining.

Four probes, each run for every post-leadfix encoder so the result is a
replicated grid rather than a single observation:

  P1  in-domain MedalCare  -- ridge readout to (cos phi, sin phi), continuous target
  P2  in-domain PTB-XL     -- same readout onto territory anchor angles
  P3  transport            -- MedalCare-fit readout applied to PTB-XL and back,
                              never refit, under both a source and a target scaler
  P4  anchor sensitivity   -- all 24 assignments of the four anchor angles to the
                              four PTB-XL territory names

Two upgrades over the 2026-08-12 first pass that can each move the numbers:
  * PTB-XL evaluation is all-folds (n=4324, 9.9x the fold-10 set).
  * Cross-validation is group-disjoint -- by patient_id on PTB-XL and by
    simulation run_id on MedalCare. The first pass allowed the same patient and
    the same simulation run into both halves.

WARNING (added 2026-08-13): every circular resultant R this script prints must be
read against its constant-predictor floor -- |mean exp(-i*true)| = 0.29216 on the
PTB-XL 4-anchor marginal, 0.09319 on MedalCare continuous phi. The P3 transport
R values (~0.17-0.26) sit BELOW the PTB-XL floor. `floor_audit.py` rescores every
stored number; see reports/2026-08-13_fidelity_audit_and_final_verification.md.

Usage:
  python analysis/circular_geometry.py                 # all encoders
  python analysis/circular_geometry.py --encoders exp8_leadfix_medalonly
"""

from __future__ import annotations

import argparse
import itertools
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
    TERRITORIES,
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
N_PERM = 500


def _cs(angle: np.ndarray) -> np.ndarray:
    return np.c_[np.cos(angle), np.sin(angle)]


def _per_territory_mean_deg(pred_ang: np.ndarray, terr: np.ndarray) -> dict[str, float]:
    return {
        t: float(np.angle(np.mean(np.exp(1j * pred_ang[terr == t]))) * 180 / np.pi)
        for t in TERRITORIES
        if np.any(terr == t)
    }


def _arc_span_deg(means: dict[str, float]) -> float:
    """Width of the smallest arc containing all four territory mean angles.

    The transported readout collapses every territory into a narrow sliver; this
    is the one-number summary of that collapse. 360 would be full dispersion.
    """
    a = np.sort(np.array(list(means.values())) % 360.0)
    gaps = np.diff(np.r_[a, a[0] + 360.0])
    return float(360.0 - gaps.max())


def cv_readout(dom: Domain, rng: np.random.Generator, n_perm: int = N_PERM) -> dict:
    """Group-disjoint CV circular readout with a label-shuffle permutation null."""
    folds = list(group_folds(dom.group, N_FOLDS, rng))
    y = _cs(dom.angle)

    pred = np.empty(len(dom))
    perm_pred = np.empty((n_perm, len(dom)))
    perms = [rng.permutation(len(dom)) for _ in range(n_perm)]
    alphas = []

    for tr, te in folds:
        model = RidgeSVD().fit(dom.z[tr], y[tr])
        alphas.append(model.alpha_)
        pred[te] = angles_from_cs(model.predict(dom.z[te]))
        for k, p in enumerate(perms):
            # Shuffle the target, keep the design: the cached SVD makes this cheap.
            perm_pred[k, te] = angles_from_cs(
                model.solve(y[p][tr]).predict(dom.z[te])
            )
        model.solve(y[tr])  # restore, so `model` is never left in a permuted state

    r_obs = resultant(pred - dom.angle)
    null = np.array([resultant(perm_pred[k] - dom.angle) for k in range(n_perm)])
    return {
        "n": len(dom),
        "n_groups": int(len(np.unique(dom.group))),
        "R": r_obs,
        "median_abs_err_deg": med_abs_deg(pred - dom.angle),
        "R_null_mean": float(null.mean()),
        "R_null_p95": float(np.percentile(null, 95)),
        "perm_p": (1 + int((null >= r_obs).sum())) / (n_perm + 1),
        "alpha_per_fold": alphas,
        "mean_pred_angle_by_territory_deg": _per_territory_mean_deg(pred, dom.territory),
        "arc_span_deg": _arc_span_deg(_per_territory_mean_deg(pred, dom.territory)),
        "_pred": pred,
    }


def transport(
    src: Domain, tgt: Domain, rng: np.random.Generator, n_perm: int = N_PERM
) -> dict:
    """Fit on `src`, predict `tgt`, never refit -- under two standardisations.

    `source` scaler is the strict transport: no target-domain information at all.
    `target` scaler re-centres on the target's own unlabelled statistics, which
    any deployment could legitimately do. Reporting both keeps the result from
    hinging on the scaler choice that is still unresolved for the B2 endpoint.
    """
    model = RidgeSVD().fit(src.z, _cs(src.angle))
    out: dict = {"alpha": model.alpha_, "n_src": len(src), "n_tgt": len(tgt)}

    for mode in ("source", "target"):
        if mode == "source":
            pred = angles_from_cs(model.predict(tgt.z))
        else:
            zt = (tgt.z - tgt.z.mean(0)) / (tgt.z.std(0) + 1e-8)
            zt = zt * model.sd_ + model.mu_        # map onto the source's scale
            pred = angles_from_cs(model.predict(zt))

        r_obs = resultant(pred - tgt.angle)
        null = np.array(
            [resultant(pred - tgt.angle[rng.permutation(len(tgt))])
             for _ in range(n_perm)]
        )
        means = _per_territory_mean_deg(pred, tgt.territory)
        out[mode] = {
            "R": r_obs,
            "median_abs_err_deg": med_abs_deg(pred - tgt.angle),
            "R_null_mean": float(null.mean()),
            "R_null_p95": float(np.percentile(null, 95)),
            "perm_p": (1 + int((null >= r_obs).sum())) / (n_perm + 1),
            "mean_pred_angle_by_territory_deg": means,
            "arc_span_deg": _arc_span_deg(means),
            "_pred": pred,
        }
    return out


def anchor_sensitivity(
    ptbxl: Domain,
    anchors: dict[str, float],
    transported_pred: dict[str, np.ndarray],
    in_domain_pred: np.ndarray,
) -> dict:
    """Score all 24 assignments of the anchor angles to the territory names.

    The angles are the midpoints of the phi wedges that define territory_4c (see
    the corrected note on `medalcare_anchor_angles`); what is assumed is the
    *semantic* correspondence -- that PTB-XL's "Anteroseptal" denotes the same
    wall as MedalCare's. If the identity assignment does not stand out here,
    that assumption is doing the work.

    CORRECTED 2026-08-13: ranking the 24 assignments by RAW R is not a clean
    24-hypothesis test, because each assignment defines its own truth marginal
    and hence its own constant-predictor floor -- the floors span 0.148-0.593
    (4.0x), so raw ranks compare different chance levels. Renormalised by each
    assignment's own floor ((R-floor)/(1-floor)), the identity ranks 9-11/24
    [source] and 7/24 [target] across the six encoders (was 12-19 and 3-4 raw).
    See reports/2026-08-13_audit_artifacts/anchor_renorm_out.txt. Quote only the
    renormalised ranks. Note also `out["all"]` below is sorted by the SOURCE
    scaler alone; per-mode rankings live in `out[mode]`.
    """
    vals = [anchors[t] for t in TERRITORIES]
    rows = []
    for perm in itertools.permutations(range(4)):
        assign = {TERRITORIES[i]: vals[perm[i]] for i in range(4)}
        truth = np.array([assign[t] for t in ptbxl.territory])
        row = {
            "assignment": {TERRITORIES[i]: round(vals[perm[i]] * 180 / np.pi, 1)
                           for i in range(4)},
            "is_identity": perm == (0, 1, 2, 3),
            "R_in_domain_fixed_pred": resultant(in_domain_pred - truth),
        }
        for mode, pred in transported_pred.items():
            row[f"R_transport_{mode}"] = resultant(pred - truth)
        rows.append(row)

    out: dict = {"n_assignments": len(rows)}
    for mode in transported_pred:
        ranked = sorted(rows, key=lambda r: -r[f"R_transport_{mode}"])
        ident = next(i for i, r in enumerate(ranked) if r["is_identity"])
        out[mode] = {
            "identity_rank": ident + 1,
            "identity_R": ranked[ident][f"R_transport_{mode}"],
            "best_R": ranked[0][f"R_transport_{mode}"],
            "best_assignment": ranked[0]["assignment"],
        }
    out["all"] = sorted(rows, key=lambda r: -r["R_transport_source"])
    return out


def run_encoder(encoder: str, anchors: dict[str, float], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    medal = load_medalcare(encoder)
    ptb = load_ptbxl(encoder, anchors)
    print(f"\n=== {encoder}  (d={medal.z.shape[1]}) ===")
    print(f"  MedalCare n={len(medal)} ({len(np.unique(medal.group))} runs)   "
          f"PTB-XL n={len(ptb)} ({len(np.unique(ptb.group))} patients)")

    res: dict = {"encoder": encoder, "latent_dim": int(medal.z.shape[1])}

    res["in_domain_medalcare"] = cv_readout(medal, rng)
    res["in_domain_ptbxl"] = cv_readout(ptb, rng)
    res["transport_medalcare_to_ptbxl"] = transport(medal, ptb, rng)
    res["transport_ptbxl_to_medalcare"] = transport(ptb, medal, rng)

    res["anchor_sensitivity"] = anchor_sensitivity(
        ptb,
        anchors,
        {m: res["transport_medalcare_to_ptbxl"][m]["_pred"] for m in ("source", "target")},
        res["in_domain_ptbxl"]["_pred"],
    )

    for key, label in (
        ("in_domain_medalcare", "in-domain MedalCare"),
        ("in_domain_ptbxl", "in-domain PTB-XL"),
    ):
        r = res[key]
        print(f"  {label:<22} R={r['R']:.4f}  med|err|={r['median_abs_err_deg']:5.1f}deg"
              f"  null={r['R_null_mean']:.3f}  p={r['perm_p']:.4f}"
              f"  arc={r['arc_span_deg']:5.1f}deg")

    for key, label in (
        ("transport_medalcare_to_ptbxl", "transport M->P"),
        ("transport_ptbxl_to_medalcare", "transport P->M"),
    ):
        for mode in ("source", "target"):
            r = res[key][mode]
            print(f"  {label} [{mode:<6}]   R={r['R']:.4f}  "
                  f"med|err|={r['median_abs_err_deg']:5.1f}deg  "
                  f"null={r['R_null_mean']:.3f}  p={r['perm_p']:.4f}  "
                  f"arc={r['arc_span_deg']:5.1f}deg")

    a = res["anchor_sensitivity"]
    for mode in ("source", "target"):
        print(f"  anchor sensitivity [{mode:<6}]: identity ranks "
              f"{a[mode]['identity_rank']}/24 (R={a[mode]['identity_R']:.4f}; "
              f"best {a[mode]['best_R']:.4f} at {a[mode]['best_assignment']})")
    return res


def _strip_arrays(obj):
    """Drop the cached per-row prediction vectors before serialising."""
    if isinstance(obj, dict):
        return {k: _strip_arrays(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_arrays(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoders", nargs="*", default=ENCODERS)
    ap.add_argument("--seed", type=int, default=20260812)
    args = ap.parse_args()

    anchors = medalcare_anchor_angles()
    print("MedalCare-measured anchor angles (deg): "
          + ", ".join(f"{k}={v*180/np.pi:.2f}" for k, v in anchors.items()))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_res = {}
    for enc in args.encoders:
        try:
            all_res[enc] = run_encoder(enc, anchors, args.seed)
        except FileNotFoundError as exc:
            print(f"\n=== {enc}: SKIPPED ({exc}) ===")

    payload = {
        "anchors_deg": {k: v * 180 / np.pi for k, v in anchors.items()},
        "n_folds": N_FOLDS,
        "n_perm": N_PERM,
        "encoders": _strip_arrays(all_res),
    }
    dest = OUT_DIR / "circular_geometry.json"
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print(f"{'encoder':<26}{'d':>5}{'M in':>8}{'P in':>8}{'M->P':>8}{'M->P*':>8}"
          f"{'P->M':>8}{'P->M*':>8}{'arcM->P':>9}{'idRank':>8}")
    print("-" * 100)
    for enc, r in all_res.items():
        print(f"{enc:<26}{r['latent_dim']:>5}"
              f"{r['in_domain_medalcare']['R']:>8.3f}"
              f"{r['in_domain_ptbxl']['R']:>8.3f}"
              f"{r['transport_medalcare_to_ptbxl']['source']['R']:>8.3f}"
              f"{r['transport_medalcare_to_ptbxl']['target']['R']:>8.3f}"
              f"{r['transport_ptbxl_to_medalcare']['source']['R']:>8.3f}"
              f"{r['transport_ptbxl_to_medalcare']['target']['R']:>8.3f}"
              f"{r['transport_medalcare_to_ptbxl']['source']['arc_span_deg']:>9.1f}"
              f"{r['anchor_sensitivity']['source']['identity_rank']:>8}")
    print("* = target-domain scaler.  arc = smallest arc containing all four "
          "territory means (360 = full dispersion).")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
