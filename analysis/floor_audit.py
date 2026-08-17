"""Constant-predictor floor audit for every circular-resultant number in Tier 1.

The circular resultant R = |mean exp(i(pred - true))| is reported throughout Tier 1
against a *label-shuffle* null. That null is not the right reference point, and this
script establishes why, then rescores every stored number against the right one.

The identity that matters
-------------------------
Take any predictor that ignores its input and emits the same angle c for every row:

    R = |mean_i exp(i(c - t_i))| = |exp(ic)| * |mean_i exp(-i t_i)| = |mean_i exp(-i t_i)|

The chosen constant cancels. So *every* constant predictor scores exactly the
resultant of the label marginal -- call it the floor. It uses no input information
whatsoever, and on a concentrated label distribution it can be large.

Why the permutation null does not catch this
--------------------------------------------
`transport()` holds `pred` fixed and permutes the truth. A permutation preserves the
label marginal exactly; what it destroys is the *pairing*. For a random permutation
sigma, E[exp(-i t_sigma(i))] = mean_j exp(-i t_j), so

    E[null resultant vector] = (mean_i exp(i p_i)) * (mean_j exp(-i t_j))
    => R_null ~= R_pred * R_floor

where R_pred is the predictor's own angular concentration. (Precision note,
2026-08-13: the identity is exact for the null resultant VECTOR; the reported
statistic is E|V|, which exceeds |E V| = R_pred * R_floor by a Jensen term --
E|V|^2 = R_p^2 R_t^2 + (1/n)(1-R_p^2)(1-R_t^2) + O(n^-2), ~+3.9% for the diffuse
target-scaler cells. The 'predicted vs stored' check below is therefore
first-order, and the observed match confirms the term is small.) A *diffuse*
predictor earns a low null for free, and can clear it comfortably while scoring
below a constant. Since R_pred <= 1, the null is systematically BELOW the floor:
"beats the shuffle null" is strictly weaker than "beats a constant". That is
exactly the regime the target-scaler transports sit in, and it is how
perm_p = 0.002 coexists with a score beneath a do-nothing baseline.

The reference to report is therefore the normalised headroom

    R_norm = (R - R_floor) / (1 - R_floor)

which is 0 for a constant predictor and 1 for an exact one, and can go negative.

Read-only over stored JSON, except --verify-null which refits two ridges.

Usage:
  python analysis/floor_audit.py
  python analysis/floor_audit.py --verify-null --encoders exp8_leadfix_medalonly
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
    TERRITORIES,
    RidgeSVD,
    angles_from_cs,
    load_medalcare,
    load_ptbxl,
    medalcare_anchor_angles,
    resultant,
)

GEOM_DIR = REPO_ROOT / "outputs/analysis/circular_geometry"
OUT_JSON = GEOM_DIR / "floor_audit.json"


# --------------------------------------------------------------------------- #
# Part 1 -- the floor itself
# --------------------------------------------------------------------------- #

def const_floor(theta: np.ndarray) -> float:
    """Resultant achieved by ANY constant predictor on truth `theta`."""
    return float(np.abs(np.mean(np.exp(-1j * theta))))


def verify_invariance(theta: np.ndarray, n_grid: int = 3600) -> dict:
    """Brute-force the claim that the constant does not matter."""
    grid = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    scores = np.array([resultant(np.full(len(theta), c) - theta) for c in grid])
    return {
        "analytic": const_floor(theta),
        "grid_min": float(scores.min()),
        "grid_max": float(scores.max()),
        "grid_spread": float(scores.max() - scores.min()),
        "argmax_deg": float(np.degrees(grid[scores.argmax()])),
    }


def norm(r: float, floor: float) -> float:
    """Headroom above a constant predictor: 0 = constant, 1 = exact."""
    return (r - floor) / max(1.0 - floor, 1e-12)


# --------------------------------------------------------------------------- #
# Part 3 -- the null-identity check
# --------------------------------------------------------------------------- #

def verify_null_identity(encoder: str, anchors: dict) -> list[dict]:
    """Recompute transport for one encoder and test R_null ~= R_pred * R_floor."""
    medal = load_medalcare(encoder)
    ptb = load_ptbxl(encoder, anchors)
    rows = []

    for name, src, tgt in (("M->P", medal, ptb), ("P->M", ptb, medal)):
        model = RidgeSVD().fit(src.z, np.c_[np.cos(src.angle), np.sin(src.angle)])
        floor = const_floor(tgt.angle)
        for mode in ("source", "target"):
            if mode == "source":
                pred = angles_from_cs(model.predict(tgt.z))
            else:
                zt = (tgt.z - tgt.z.mean(0)) / (tgt.z.std(0) + 1e-8)
                pred = angles_from_cs(model.predict(zt * model.sd_ + model.mu_))
            r_pred = float(np.abs(np.mean(np.exp(1j * pred))))
            rows.append({
                "direction": name,
                "scaler": mode,
                "R": resultant(pred - tgt.angle),
                "R_floor": floor,
                "R_pred_concentration": r_pred,
                "predicted_null": r_pred * floor,
            })
    return rows


# --------------------------------------------------------------------------- #
# Part 4 -- is the cyclic order preserved?
# --------------------------------------------------------------------------- #

def cyclic_order(means_deg: dict[str, float]) -> dict:
    """Unwrap the four predicted territory means from Anteroseptal and report
    whether they traverse the true anatomical cycle AS->AL->IL->Inf."""
    base = means_deg[TERRITORIES[0]]
    unwrapped = [(means_deg[t] - base) % 360.0 for t in TERRITORIES]
    ok = all(unwrapped[i] < unwrapped[i + 1] for i in range(len(unwrapped) - 1))
    return {
        "unwrapped_deg": [round(u, 1) for u in unwrapped],
        "order_preserved": bool(ok),
        "span_deg": round(unwrapped[-1], 1),
    }


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-null", action="store_true",
                    help="refit transport to test R_null ~= R_pred * R_floor")
    ap.add_argument("--encoders", nargs="*", default=ENCODERS)
    args = ap.parse_args()

    anchors = medalcare_anchor_angles()
    out: dict = {"anchors_deg": {k: round(np.degrees(v), 2) for k, v in anchors.items()}}

    # ---- Part 1: floors -------------------------------------------------- #
    ref = args.encoders[0]
    medal = load_medalcare(ref)
    ptb = load_ptbxl(ref, anchors)
    medal_q = np.array([anchors[t] for t in medal.territory])

    floors = {
        "ptbxl_4anchor": verify_invariance(ptb.angle),
        "medalcare_continuous_phi": verify_invariance(medal.angle),
        "medalcare_quantised_4anchor": verify_invariance(medal_q),
    }
    out["floors"] = floors

    print("=" * 78)
    print("PART 1  Constant-predictor floor -- is it really constant-invariant?")
    print("=" * 78)
    print(f"{'target':<30}{'analytic':>10}{'grid min':>10}{'grid max':>10}{'spread':>10}")
    print("-" * 78)
    for k, v in floors.items():
        print(f"{k:<30}{v['analytic']:>10.5f}{v['grid_min']:>10.5f}"
              f"{v['grid_max']:>10.5f}{v['grid_spread']:>10.2e}")
    print("\nSpread ~1e-16 across 3600 constants => the floor is a property of the\n"
          "label marginal alone. Any constant predictor achieves it with zero input.")

    f_ptb = floors["ptbxl_4anchor"]["analytic"]
    f_med = floors["medalcare_continuous_phi"]["analytic"]
    f_medq = floors["medalcare_quantised_4anchor"]["analytic"]

    for name, ang in (("PTB-XL", ptb.territory), ("MedalCare", medal.territory)):
        vals, cts = np.unique(ang, return_counts=True)
        frac = ", ".join(f"{v} {c / cts.sum():.1%}" for v, c in zip(vals, cts))
        print(f"  {name:<10} n={cts.sum():<6} {frac}")

    # ---- Part 2: rescore every stored number ----------------------------- #
    geom = json.loads((GEOM_DIR / "circular_geometry.json").read_text())
    print("\n" + "=" * 78)
    print("PART 2  Every Tier 1 circular number, rescored against its floor")
    print("=" * 78)
    print(f"floor(PTB-XL 4-anchor) = {f_ptb:.4f}    "
          f"floor(MedalCare phi) = {f_med:.4f}")
    print(f"\n{'encoder':<24}{'cell':<22}{'R':>8}{'floor':>8}"
          f"{'R_norm':>9}{'null':>8}{'perm_p':>9}")
    print("-" * 88)

    rescored: dict = {}
    n_below = n_total = 0
    for enc in args.encoders:
        e = geom["encoders"].get(enc)
        if e is None:
            continue
        cells = []
        cells.append(("in-domain MedalCare", e["in_domain_medalcare"], f_med))
        cells.append(("in-domain PTB-XL", e["in_domain_ptbxl"], f_ptb))
        for direction, key, fl in (
            ("M->P", "transport_medalcare_to_ptbxl", f_ptb),
            ("P->M", "transport_ptbxl_to_medalcare", f_med),
        ):
            for mode in ("source", "target"):
                cells.append((f"{direction} [{mode}]", e[key][mode], fl))

        rescored[enc] = {}
        for label, blob, fl in cells:
            rn = norm(blob["R"], fl)
            rescored[enc][label] = {
                "R": blob["R"], "floor": fl, "R_norm": rn,
                "R_null_mean": blob["R_null_mean"], "perm_p": blob.get("perm_p"),
            }
            if label.startswith(("M->P", "P->M")):
                n_total += 1
                n_below += rn <= 0
            flag = "  <-- BELOW FLOOR" if rn <= 0 else ""
            pp = blob.get("perm_p")
            print(f"{enc:<24}{label:<22}{blob['R']:>8.3f}{fl:>8.3f}{rn:>9.3f}"
                  f"{blob['R_null_mean']:>8.3f}"
                  f"{(f'{pp:.4f}' if pp is not None else '--'):>9}{flag}")
        print()

    out["rescored"] = rescored
    out["cross_domain_below_floor"] = f"{n_below}/{n_total}"
    print(f"Cross-domain cells at or below the constant-predictor floor: "
          f"{n_below}/{n_total}")

    # ---- Part 4: cyclic order -------------------------------------------- #
    print("\n" + "=" * 78)
    print("PART 4  Cyclic order of the four predicted territory means (M->P)")
    print("=" * 78)
    print("True anatomical cycle: Anteroseptal -> Anterolateral -> Inferolateral"
          " -> Inferior")
    print(f"\n{'encoder':<24}{'scaler':<9}{'unwrapped from AS (deg)':<34}"
          f"{'order':>7}{'arc':>8}")
    print("-" * 82)
    order_res: dict = {}
    for enc in args.encoders:
        e = geom["encoders"].get(enc)
        if e is None:
            continue
        order_res[enc] = {}
        for mode in ("source", "target"):
            blob = e["transport_medalcare_to_ptbxl"][mode]
            co = cyclic_order(blob["mean_pred_angle_by_territory_deg"])
            co["arc_span_deg"] = blob["arc_span_deg"]
            order_res[enc][mode] = co
            mark = "OK" if co["order_preserved"] else "--"
            print(f"{enc:<24}{mode:<9}{str(co['unwrapped_deg']):<34}{mark:>7}"
                  f"{blob['arc_span_deg']:>8.1f}")
    out["cyclic_order_M_to_P"] = order_res
    for mode in ("source", "target"):
        k = sum(v[mode]["order_preserved"] for v in order_res.values())
        print(f"  {mode:<8} order preserved in {k}/{len(order_res)} encoders")

    # ---- Part 5: acuity strata ------------------------------------------- #
    acu_path = GEOM_DIR / "acuity_stratified_transport.json"
    if not acu_path.exists():
        acu_path = REPO_ROOT / "outputs/analysis/acuity_stratified_transport.json"
    if acu_path.exists():
        acu = json.loads(acu_path.read_text())
        print("\n" + "=" * 78)
        print("PART 5  Acuity strata -- each stratum has its OWN floor")
        print("=" * 78)
        strata = acu.get("strata", {})
        print(f"{'stratum':<10}{'scaler':<8}{'n':>6}{'R':>8}{'floor':>8}"
              f"{'R_norm':>9}{'gain':>8}{'b_hat':>8}{'p_gain':>9}")
        print("-" * 74)
        acu_out = {}
        for name, blob in strata.items():
            if not isinstance(blob, dict) or "source" not in blob:
                continue
            # Independent floor check straight from the stored counts.
            w = np.array([blob["territory_counts"].get(t, 0) for t in TERRITORIES],
                         float)
            w = w / w.sum()
            fl = float(np.abs(np.sum(w * np.exp(
                -1j * np.array([anchors[t] for t in TERRITORIES])))))
            acu_out[name] = {"n": blob["n"], "floor_recomputed": fl}
            for mode in ("source", "target"):
                m = blob[mode]
                if abs(m["R_floor"] - fl) > 1e-9:
                    raise RuntimeError(
                        f"{name}/{mode}: stored floor {m['R_floor']} != {fl}")
                acu_out[name][mode] = {
                    "R": m["R"], "R_norm": m["R_norm"],
                    "gain": m["gain_over_b0"], "b_hat": m["b_hat"],
                    "b_hat_ci95": m["b_hat_boot_ci95"],
                    "perm_p_gain": m["perm_p_gain"],
                }
                print(f"{name if mode == 'source' else '':<10}{mode:<8}"
                      f"{blob['n'] if mode == 'source' else '':>6}"
                      f"{m['R']:>8.3f}{fl:>8.3f}{m['R_norm']:>9.3f}"
                      f"{m['gain_over_b0']:>8.3f}{m['b_hat']:>+8.2f}"
                      f"{m['perm_p_gain']:>9.4f}")
        trend = acu.get("acuity_trend", {})
        for mode, t in trend.items():
            print(f"  ordering test [{mode}]: rho="
                  f"{t['rho_resid_vs_acuity_rank']:+.4f}  "
                  f"p(one-sided neg)={t['p_one_sided_negative']:.4f}")
        acu_out["acuity_trend"] = trend
        out["acuity_strata"] = acu_out
        print("\nThe strata do not share a floor, so raw R is not comparable across\n"
              "them; the ordering test above centres alignment within territory and\n"
              "is the single statistic for the ST-vs-Q/R prediction.")

    # ---- Part 6: granularity control ------------------------------------- #
    gran_path = GEOM_DIR / "granularity_control.json"
    if gran_path.exists():
        gran = json.loads(gran_path.read_text())
        print("\n" + "=" * 78)
        print("PART 6  Granularity control, renormalised against the floor")
        print("=" * 78)
        print(f"{'encoder':<24}{'M 4lvl R':>10}{'M norm':>9}"
              f"{'P 4lvl R':>10}{'P norm':>9}{'winner':>12}")
        print("-" * 74)
        gran_out = {}
        for enc, r in gran.items():
            mq, pq = r["medalcare_quantised_4level"]["R"], r["ptbxl_4level"]["R"]
            mn, pn = norm(mq, f_medq), norm(pq, f_ptb)
            win = "MedalCare" if mn > pn else "PTB-XL"
            gran_out[enc] = {"medalcare_norm": mn, "ptbxl_norm": pn, "winner": win}
            print(f"{enc:<24}{mq:>10.3f}{mn:>9.3f}{pq:>10.3f}{pn:>9.3f}{win:>12}")
        out["granularity_renormalised"] = gran_out

    # ---- Part 7: synthetic prior arms ------------------------------------ #
    spv_path = GEOM_DIR / "synthetic_prior_value.json"
    if spv_path.exists():
        spv = json.loads(spv_path.read_text())
        print("\n" + "=" * 78)
        print("PART 7  Synthetic-prior arms vs the floor "
              f"({f_ptb:.3f}); values <= floor are uninformative")
        print("=" * 78)
        print(f"{'encoder':<24}{'n':>6}{'scratch':>10}{'prior':>10}{'plane':>10}"
              f"{'frozen':>10}   arms<=floor")
        print("-" * 84)
        spv_out = {}
        for enc, blob in spv["encoders"].items():
            spv_out[enc] = {}
            for n, row in blob["curve"].items():
                vals = {k: row[k]["mean"] for k in ("scratch", "prior", "plane",
                                                    "frozen") if k in row}
                below = [k for k, v in vals.items() if v <= f_ptb]
                spv_out[enc][n] = {"values": vals, "at_or_below_floor": below}
                print(f"{enc:<24}{n:>6}" + "".join(
                    f"{vals.get(k, float('nan')):>10.3f}"
                    for k in ("scratch", "prior", "plane", "frozen")
                ) + "   " + (",".join(below) if below else "-"))
            print()
        out["synthetic_prior_vs_floor"] = spv_out

    # ---- Part 3: null identity ------------------------------------------- #
    if args.verify_null:
        print("\n" + "=" * 78)
        print("PART 3  Does R_null ~= R_pred * R_floor hold?")
        print("=" * 78)
        print(f"{'encoder':<24}{'cell':<18}{'R':>8}{'floor':>8}"
              f"{'R_pred':>9}{'predicted':>11}{'stored':>9}")
        print("-" * 87)
        ident = {}
        for enc in args.encoders:
            try:
                rows = verify_null_identity(enc, anchors)
            except FileNotFoundError as exc:
                print(f"{enc:<24}SKIPPED ({exc})")
                continue
            e = geom["encoders"][enc]
            ident[enc] = rows
            for r in rows:
                key = ("transport_medalcare_to_ptbxl" if r["direction"] == "M->P"
                       else "transport_ptbxl_to_medalcare")
                stored = e[key][r["scaler"]]["R_null_mean"]
                r["stored_null"] = stored
                print(f"{enc:<24}{r['direction'] + ' [' + r['scaler'] + ']':<18}"
                      f"{r['R']:>8.3f}{r['R_floor']:>8.3f}"
                      f"{r['R_pred_concentration']:>9.3f}"
                      f"{r['predicted_null']:>11.3f}{stored:>9.3f}")
        out["null_identity"] = ident
        print("\n'predicted' uses only the predictor's own concentration and the label\n"
              "marginal -- no pairing. Matching 'stored' confirms the permutation null\n"
              "measures predictor diffuseness, not the absence of transferable signal.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
