"""Acuity-stratified transport of the MedalCare circular territory readout onto PTB-XL.

PARTIALLY RETRACTED 2026-08-13. Two corrections to this file's framing:
1. The S(b)/b_hat machinery ("the statistic that carries the inference" below) is
   BROKEN for non-integer b -- b_hat depends on the arbitrary anchor-unwrap origin
   (see the banner in cyclic_order_test.py and Part A.1 of
   reports/2026-08-13_fidelity_audit_and_final_verification.md). Every per-stratum
   b_hat and gain this script prints is void. What SURVIVES from this script is
   `acuity_trend()` -- the territory-centred alignment vs acuity rank correlation
   (rho = -0.034 p=0.097 source / +0.017 p=0.743 target, non-monotone) -- which
   does not use S(b). The acuity transport prediction FAILED under both scalers.
2. "Vacuous null" below is too strong and self-undermining (the S(b) null is the
   same construction): the permutation null is not vacuous, it answers a DIFFERENT
   question -- it certifies label-pairing use, and sits at ~R_pred*R_floor, which
   is BELOW the constant floor, so it cannot substitute for the floor comparison.
   Note also E|V_null| exceeds |E V_null| = R_pred*R_floor by a Jensen term
   (~+3.9% for diffuse predictors); the analytic check below is first-order.

Falsification test for the ST-vs-Q/R mechanism.

MedalCare simulates ACUTE ischaemia (injury current -> ST). PTB-XL is dominated by
CHRONIC infarct (Q/R). If that mechanism is the reason the MedalCare-fit readout does
not transfer, then the readout must transfer BETTER onto the acute PTB-XL rows than
onto the chronic ones -- monotonically across Stadium I -> II -> II-III -> III.

Acuity comes from `infarction_stadium1` in ptbxl_database.csv (NOT the SCP hierarchy,
which has no acuity field -- that is why this was previously believed unavailable).

Three defects in the 2026-08-12 version, all fixed here
-------------------------------------------------------
1. **One scaler only.** The old code standardised PTB-XL rows with MedalCare's mu/sd
   and reported that alone. Both scalers are now reported side by side, exactly as
   `circular_geometry.transport()` defines them. Neither uses a label.
2. **No floor.** The raw resultant R is bounded below by the constant-predictor floor
   |mean exp(-i t)|, which is a property of the stratum's own territory marginal --
   and the strata do NOT share one (0.27 acute vs 0.46 mid). Comparing raw R across
   strata therefore confounds readout quality with class balance, in the direction
   that manufactures the predicted effect. Every stratum now carries its own floor,
   its normalised headroom, and a territory-composition-matched contrast.
3. **A vacuous null.** Holding `pred` fixed and permuting the truth preserves the
   label marginal and destroys only the pairing, so E[R_null] = R_pred * R_floor
   identically -- a diffuse predictor buys a low null with no signal. It is retained
   for comparability with the stored Tier 1 numbers and printed next to its own
   analytic prediction so the reader can see it is not evidence. The statistic that
   carries the inference is the S(b) gain below, whose null re-optimises b.

The statistic that carries the inference
----------------------------------------
    S(b) = | mean_i exp( i ( pred_i - b * t_i ) ) |

with t_i the row's anchor unwrapped from Anteroseptal into [0, 2pi). S(0) is the
predictor's own concentration with the labels switched off; S(1) is exactly R; S(b_hat)
is the best order-preserving affine map. `gain = S(b_hat) - S(0)` is label-informed
performance above a label-free baseline computed on the same rows, so unlike R it is
already free of the class-balance confound. Its null re-optimises b over the same grid,
charging both free parameters against the null. See `cyclic_order_test.py`.

Read-only: consumes stored latents + manifests, writes JSON to outputs/analysis/.

Usage:
  python analysis/acuity_stratified_transport.py
  python analysis/acuity_stratified_transport.py --encoder exp8_leadfix_dual
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
from geom_common import (
    TERRITORIES,
    RidgeSVD,
    angles_from_cs,
    load_medalcare,
    load_ptbxl,
    med_abs_deg,
    medalcare_anchor_angles,
    resultant,
)
# Imported rather than redefined so the gain statistic cannot drift from the one
# `cyclic_order_test.py` reports; `_canonical_anchors` included deliberately.
from cyclic_order_test import (  # noqa: F401
    B_GRID,
    _canonical_anchors,
    partials,
    s_curve,
)

# infarction_stadium1 -> ordered acuity stratum. Ordering is the prediction axis.
STRATA = [
    ("acute", ["Stadium I", "Stadium I-II"]),
    ("early", ["Stadium II"]),
    ("mid", ["Stadium II-III"]),
    ("chronic", ["Stadium III"]),
]

ENCODER = "exp8_leadfix_medalonly"
PTBXL_DB = (
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
    "ptbxl_database.csv"
)
MIN_N = 20
N_PERM = 2000
N_BOOT = 2000
N_DRAW = 400


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def acuity_table(encoder: str, anchors: dict) -> tuple[pd.DataFrame, dict]:
    """PTB-XL MI rows carrying a territory label, an acuity grade, and predictions.

    Both scaler variants are computed here, once, over the FULL 4324-row MI cohort
    before any stratification. That matters for the target scaler: its re-centring
    uses the deployment domain's unlabelled mean/sd, and re-deriving those inside a
    stratum would let stratum membership leak into the transform.
    """
    medal = load_medalcare(encoder)
    ptb = load_ptbxl(encoder, anchors)          # n=4324, all folds

    model = RidgeSVD().fit(medal.z, np.c_[np.cos(medal.angle), np.sin(medal.angle)])

    zt = (ptb.z - ptb.z.mean(0)) / (ptb.z.std(0) + 1e-8)
    pred = {
        "source": angles_from_cs(model.predict(ptb.z)),
        "target": angles_from_cs(model.predict(zt * model.sd_ + model.mu_)),
    }

    mi = pd.read_csv(REPO_ROOT / "data/ptbxl_mi_subclass_allfolds.csv")
    mi = mi[mi["territory_4c"].notna()].copy()
    if len(mi) != len(ptb):
        raise RuntimeError(f"{len(mi)} manifest MI rows vs {len(ptb)} latent rows")
    mi["pred_source"] = pred["source"]
    mi["pred_target"] = pred["target"]

    db = pd.read_csv(REPO_ROOT / PTBXL_DB)[["ecg_id", "infarction_stadium1"]]
    mi = mi.merge(db, on="ecg_id", how="left")

    stratum_of = {v: name for name, vals in STRATA for v in vals}
    mi["stratum"] = mi["infarction_stadium1"].map(stratum_of)

    meta = {
        "n_medalcare_fit_rows": int(len(medal)),
        "n_ptbxl_mi_rows": int(len(ptb)),
        "n_with_acuity": int(mi["stratum"].notna().sum()),
        "ridge_alpha": float(model.alpha_),
        "stadium_value_counts": {
            str(k): int(v)
            for k, v in mi["infarction_stadium1"].value_counts(dropna=False).items()
        },
    }
    return mi[mi["stratum"].notna()].copy(), meta


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def const_floor(true_ang: np.ndarray) -> float:
    """Resultant that ANY constant predictor achieves on these labels."""
    return float(np.abs(np.mean(np.exp(-1j * true_ang))))


def norm(r: float, floor: float) -> float:
    """Headroom above a constant predictor: 0 = constant, 1 = exact, may go negative."""
    return (r - floor) / max(1.0 - floor, 1e-12)


def gain_stats(pred_ang: np.ndarray, terr: np.ndarray, groups: np.ndarray,
               t_anchor: np.ndarray, rng: np.random.Generator,
               n_perm: int, n_boot: int) -> dict:
    """S(b) curve for one stratum, with a b-re-optimising label-permutation null.

    The null swaps territory labels between whole patients, so a patient contributing
    several ECGs cannot leak its own label across the shuffle. The bootstrap resamples
    patients: b_hat is a point estimate read off a single argmax, and at n=97 its
    sampling spread is the difference between "the map is correctly scaled here and
    reversed there" and two draws from the same wide distribution.
    """
    n = len(terr)
    pred_c = np.exp(1j * pred_ang)

    part = partials(pred_c, terr, n)
    curve = s_curve(part, t_anchor)
    k = int(curve.argmax())
    s0 = float(np.abs(part.sum()))
    s1 = float(np.abs(part @ np.exp(-1j * t_anchor)))

    uniq = np.unique(groups)
    idx_by_group = [np.where(groups == g)[0] for g in uniq]

    s_null = np.empty(n_perm)
    for j in range(n_perm):
        perm = rng.permutation(len(uniq))
        shuf = terr.copy()
        for dst, src in enumerate(perm):
            a, b = idx_by_group[dst], idx_by_group[src]
            shuf[a] = terr[b][rng.integers(0, len(b), len(a))]
        c = s_curve(partials(pred_c, shuf, n), t_anchor)
        s_null[j] = c[int(c.argmax())]

    b_boot, g_boot = np.empty(n_boot), np.empty(n_boot)
    for j in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([idx_by_group[i] for i in pick])
        p = partials(pred_c[rows], terr[rows], len(rows))
        c = s_curve(p, t_anchor)
        kb = int(c.argmax())
        b_boot[j] = B_GRID[kb]
        g_boot[j] = c[kb] - np.abs(p.sum())

    gain = float(curve[k]) - s0
    null_gain = s_null - s0
    return {
        "S_at_b0_no_correspondence": s0,
        "S_at_b1_equals_R": s1,
        "b_hat": float(B_GRID[k]),
        "S_at_b_hat": float(curve[k]),
        "gain_over_b0": gain,
        "b_hat_boot_ci95": [float(np.percentile(b_boot, 2.5)),
                            float(np.percentile(b_boot, 97.5))],
        "boot_frac_b_positive": float(np.mean(b_boot > 0)),
        "boot_frac_b_near_one": float(np.mean(np.abs(b_boot - 1.0) < 0.35)),
        "gain_boot_ci95": [float(np.percentile(g_boot, 2.5)),
                           float(np.percentile(g_boot, 97.5))],
        "null_gain_mean": float(null_gain.mean()),
        "null_gain_p95": float(np.percentile(null_gain, 95)),
        "perm_p_gain": (1 + int((s_null >= curve[k]).sum())) / (n_perm + 1),
    }


def score(pred_ang: np.ndarray, terr: np.ndarray, groups: np.ndarray,
          anchors: dict, t_anchor: np.ndarray, rng: np.random.Generator,
          n_perm: int, n_boot: int) -> dict:
    """One stratum under one scaler: floor-normalised R plus the S(b) gain."""
    true_ang = np.array([anchors[t] for t in terr])
    delta = pred_ang - true_ang
    r_obs = resultant(delta)
    floor = const_floor(true_ang)
    r_pred = float(np.abs(np.mean(np.exp(1j * pred_ang))))

    # The vacuous null, kept only so it can be shown to equal its analytic value.
    null = np.array([resultant(pred_ang - true_ang[rng.permutation(len(true_ang))])
                     for _ in range(n_perm)])

    res = {
        "R": r_obs,
        "R_floor": floor,
        "R_norm": norm(r_obs, floor),
        "median_abs_err_deg": med_abs_deg(delta),
        "R_pred_concentration": r_pred,
        "vacuous_null_mean": float(null.mean()),
        "vacuous_null_analytic": r_pred * floor,
        "vacuous_perm_p": (1 + int((null >= r_obs).sum())) / (n_perm + 1),
        "mean_pred_angle_by_territory_deg": {
            t: float(np.degrees(np.angle(np.mean(np.exp(1j * pred_ang[terr == t])))))
            for t in TERRITORIES if np.any(terr == t)
        },
    }
    res.update(gain_stats(pred_ang, terr, groups, t_anchor, rng, n_perm, n_boot))
    return res


# --------------------------------------------------------------------------- #
# Ordering test
# --------------------------------------------------------------------------- #
def acuity_trend(mi: pd.DataFrame, col: str, anchors: dict,
                 rng: np.random.Generator, n_perm: int) -> dict:
    """One statistic for the whole mechanism, instead of eight per-stratum cells.

    Each row contributes a_i = cos(pred_i - t_i) to the resultant. Rows in a majority
    territory earn a high a_i for free, so a_i is centred within its own territory
    first; what remains is readout quality with the composition confound removed.
    The ST-vs-Q/R mechanism predicts that this decreases monotonically along
    acute -> early -> mid -> chronic, i.e. a NEGATIVE rank correlation.

    The null permutes acuity between whole patients, so neither repeated ECGs from
    one patient nor the territory marginal can generate the trend.
    """
    rank_of = {name: i for i, (name, _) in enumerate(STRATA)}
    rank = mi["stratum"].map(rank_of).to_numpy(float)
    terr = mi["territory_4c"].to_numpy()
    true_ang = np.array([anchors[t] for t in terr])
    a = np.cos(mi[col].to_numpy() - true_ang)

    resid = a.copy()
    for t in TERRITORIES:
        m = terr == t
        if m.any():
            resid[m] -= resid[m].mean()

    def rho(x: np.ndarray, y: np.ndarray) -> float:
        xr = pd.Series(x).rank().to_numpy()
        yr = pd.Series(y).rank().to_numpy()
        xr -= xr.mean()
        yr -= yr.mean()
        return float(xr @ yr / np.sqrt((xr @ xr) * (yr @ yr)))

    obs = rho(resid, rank)

    groups = mi["patient_id"].to_numpy().astype(str)
    uniq = np.unique(groups)
    idx_by_group = [np.where(groups == g)[0] for g in uniq]
    rank_by_group = np.array([rank[i[0]] for i in idx_by_group])

    null = np.empty(n_perm)
    for j in range(n_perm):
        shuf = np.empty_like(rank)
        perm = rank_by_group[rng.permutation(len(uniq))]
        for i, idx in enumerate(idx_by_group):
            shuf[idx] = perm[i]
        null[j] = rho(resid, shuf)

    return {
        "rho_resid_vs_acuity_rank": obs,
        "null_mean": float(null.mean()),
        "null_ci95": [float(np.percentile(null, 2.5)),
                      float(np.percentile(null, 97.5))],
        "p_one_sided_negative": (1 + int((null <= obs).sum())) / (n_perm + 1),
        "p_two_sided": (1 + int((np.abs(null) >= abs(obs)).sum())) / (n_perm + 1),
        "mean_resid_alignment_by_stratum": {
            name: float(resid[mi["stratum"].to_numpy() == name].mean())
            for name, _ in STRATA
            if (mi["stratum"].to_numpy() == name).sum() >= MIN_N
        },
        "n_patients": int(len(uniq)),
    }


# --------------------------------------------------------------------------- #
# Composition-matched contrast
# --------------------------------------------------------------------------- #
def matched_contrast(mi: pd.DataFrame, ref: str, other: str, col: str,
                     anchors: dict, t_anchor: np.ndarray,
                     rng: np.random.Generator, n_draw: int) -> dict | None:
    """Resample `other` to `ref`'s exact territory composition, then compare.

    Matching the composition rather than only the row count equalises the
    constant-predictor floor by construction, so any surviving difference is a
    difference in readout quality rather than in class balance. Both the raw R and
    the floor-free gain are reported; the gain is the primary.
    """
    a = mi[mi["stratum"] == ref]
    b = mi[mi["stratum"] == other]
    if len(a) < MIN_N or len(b) < MIN_N:
        return None

    want = a["territory_4c"].value_counts().to_dict()
    pools = {t: b.index[b["territory_4c"] == t].to_numpy() for t in want}
    short = {t: int(k) for t, k in want.items() if len(pools.get(t, [])) < k}

    r_draw, g_draw, b_draw = np.empty(n_draw), np.empty(n_draw), np.empty(n_draw)
    for j in range(n_draw):
        take = np.concatenate([
            rng.choice(pools[t], size=k, replace=len(pools[t]) < k)
            for t, k in want.items() if len(pools.get(t, [])) > 0
        ])
        sub = mi.loc[take]
        terr = sub["territory_4c"].to_numpy()
        pred = sub[col].to_numpy()
        r_draw[j] = resultant(pred - np.array([anchors[t] for t in terr]))
        part = partials(np.exp(1j * pred), terr, len(terr))
        curve = s_curve(part, t_anchor)
        kb = int(curve.argmax())
        g_draw[j] = curve[kb] - float(np.abs(part.sum()))
        b_draw[j] = B_GRID[kb]

    terr_a = a["territory_4c"].to_numpy()
    pred_a = a[col].to_numpy()
    r_ref = resultant(pred_a - np.array([anchors[t] for t in terr_a]))
    part_a = partials(np.exp(1j * pred_a), terr_a, len(terr_a))
    curve_a = s_curve(part_a, t_anchor)
    ka = int(curve_a.argmax())
    g_ref = float(curve_a[ka] - np.abs(part_a.sum()))
    b_ref = float(B_GRID[ka])

    return {
        "ref": ref, "other": other, "n_matched": int(len(a)),
        "composition_undersupplied": short,
        "R_ref": r_ref,
        "R_other_matched_mean": float(r_draw.mean()),
        "R_other_matched_ci95": [float(np.percentile(r_draw, 2.5)),
                                 float(np.percentile(r_draw, 97.5))],
        "frac_R_draws_ge_ref": float(np.mean(r_draw >= r_ref)),
        "gain_ref": g_ref,
        "gain_other_matched_mean": float(g_draw.mean()),
        "gain_other_matched_ci95": [float(np.percentile(g_draw, 2.5)),
                                    float(np.percentile(g_draw, 97.5))],
        "frac_gain_draws_ge_ref": float(np.mean(g_draw >= g_ref)),
        "b_hat_ref": b_ref,
        "b_hat_other_matched_mean": float(b_draw.mean()),
        "b_hat_other_matched_ci95": [float(np.percentile(b_draw, 2.5)),
                                     float(np.percentile(b_draw, 97.5))],
        "frac_b_draws_ge_ref": float(np.mean(b_draw >= b_ref)),
        "frac_b_draws_positive": float(np.mean(b_draw > 0)),
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default=ENCODER)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-draw", type=int, default=N_DRAW)
    ap.add_argument("--seed", type=int, default=20260813)
    args = ap.parse_args()

    anchors = medalcare_anchor_angles()
    t_anchor = _canonical_anchors(anchors)
    mi, meta = acuity_table(args.encoder, anchors)

    out = {
        "encoder": args.encoder,
        "anchors_deg": {k: round(float(np.degrees(v)), 2) for k, v in anchors.items()},
        "note": ("R is bounded below by R_floor, which differs per stratum; compare "
                 "R_norm or gain_over_b0, never raw R. vacuous_* is retained only to "
                 "show it equals its analytic value and tests nothing."),
        **meta,
        "strata": {},
    }

    print(f"encoder={args.encoder}   MedalCare fit rows={meta['n_medalcare_fit_rows']}"
          f"   PTB-XL MI={meta['n_ptbxl_mi_rows']}"
          f"   with acuity={meta['n_with_acuity']}")
    print("Readout transported with NO refitting. Both scalers are label-blind.\n")
    print(f"{'stratum':<9}{'scaler':<8}{'n':>6}{'R':>8}{'floor':>8}{'R_norm':>9}"
          f"{'S(0)':>8}{'gain':>8}{'gain CI95':>17}{'b_hat':>8}"
          f"{'b_hat CI95':>17}{'p_gain':>9}")
    print("-" * 115)

    for name, _ in STRATA:
        sub = mi[mi["stratum"] == name]
        if len(sub) < MIN_N:
            print(f"{name:<9}{'':8}{len(sub):>6}   (skipped, n<{MIN_N})")
            continue
        terr = sub["territory_4c"].to_numpy()
        groups = sub["patient_id"].to_numpy().astype(str)
        blob = {
            "n": int(len(sub)),
            "n_patients": int(len(np.unique(groups))),
            "territory_counts": {t: int(np.sum(terr == t)) for t in TERRITORIES},
        }
        for mode in ("source", "target"):
            rng = np.random.default_rng(args.seed)
            blob[mode] = score(sub[f"pred_{mode}"].to_numpy(), terr, groups,
                               anchors, t_anchor, rng, args.n_perm, args.n_boot)
            m = blob[mode]
            gci = (f"[{m['gain_boot_ci95'][0]:+.3f},"
                   f"{m['gain_boot_ci95'][1]:+.3f}]")
            bci = (f"[{m['b_hat_boot_ci95'][0]:+.2f},"
                   f"{m['b_hat_boot_ci95'][1]:+.2f}]")
            print(f"{name if mode == 'source' else '':<9}{mode:<8}"
                  f"{blob['n'] if mode == 'source' else '':>6}"
                  f"{m['R']:>8.3f}{m['R_floor']:>8.3f}{m['R_norm']:>9.3f}"
                  f"{m['S_at_b0_no_correspondence']:>8.3f}"
                  f"{m['gain_over_b0']:>8.3f}{gci:>17}"
                  f"{m['b_hat']:>+8.2f}{bci:>17}{m['perm_p_gain']:>9.4f}")
        out["strata"][name] = blob

    print("\nvacuous null check (mean vs analytic R_pred * R_floor):")
    for name, blob in out["strata"].items():
        for mode in ("source", "target"):
            m = blob[mode]
            print(f"  {name:<9}{mode:<8}{m['vacuous_null_mean']:>7.3f} vs "
                  f"{m['vacuous_null_analytic']:>7.3f}")

    # --- single ordering test over all rows ---------------------------------- #
    out["acuity_trend"] = {}
    print("\n" + "=" * 78)
    print("Ordering test: territory-centred alignment vs acuity rank (all rows)")
    print("ST-vs-Q/R predicts rho < 0.  Null permutes acuity between patients.")
    print("=" * 78)
    for mode in ("source", "target"):
        rng = np.random.default_rng(args.seed + 3)
        tr = acuity_trend(mi, f"pred_{mode}", anchors, rng, args.n_perm)
        out["acuity_trend"][mode] = tr
        by = "  ".join(f"{k}={v:+.4f}"
                       for k, v in tr["mean_resid_alignment_by_stratum"].items())
        print(f"  [{mode}] rho={tr['rho_resid_vs_acuity_rank']:+.4f}"
              f"   null [{tr['null_ci95'][0]:+.4f},{tr['null_ci95'][1]:+.4f}]"
              f"   p(one-sided neg)={tr['p_one_sided_negative']:.4f}")
        print(f"          mean centred alignment: {by}")

    # --- composition-matched acute-vs-chronic -------------------------------- #
    out["matched_contrasts"] = {}
    print("\n" + "=" * 78)
    print("Territory-composition-matched contrast (chronic resampled to acute's mix)")
    print("=" * 78)
    for mode in ("source", "target"):
        rng = np.random.default_rng(args.seed + 7)
        mc = matched_contrast(mi, "acute", "chronic", f"pred_{mode}", anchors,
                              t_anchor, rng, args.n_draw)
        if mc is None:
            print(f"  [{mode}] not enough rows")
            continue
        out["matched_contrasts"][mode] = mc
        print(f"  [{mode}] n_matched={mc['n_matched']}"
              f"   undersupplied={mc['composition_undersupplied'] or '-'}")
        print(f"      R    acute={mc['R_ref']:.4f}   chronic(matched)="
              f"{mc['R_other_matched_mean']:.4f} "
              f"[{mc['R_other_matched_ci95'][0]:.4f},"
              f"{mc['R_other_matched_ci95'][1]:.4f}]"
              f"   P(chronic>=acute)={mc['frac_R_draws_ge_ref']:.4f}")
        print(f"      gain acute={mc['gain_ref']:.4f}   chronic(matched)="
              f"{mc['gain_other_matched_mean']:.4f} "
              f"[{mc['gain_other_matched_ci95'][0]:.4f},"
              f"{mc['gain_other_matched_ci95'][1]:.4f}]"
              f"   P(chronic>=acute)={mc['frac_gain_draws_ge_ref']:.4f}")
        print(f"      bhat acute={mc['b_hat_ref']:+.3f}   chronic(matched)="
              f"{mc['b_hat_other_matched_mean']:+.4f} "
              f"[{mc['b_hat_other_matched_ci95'][0]:+.3f},"
              f"{mc['b_hat_other_matched_ci95'][1]:+.3f}]"
              f"   P(chronic>=acute)={mc['frac_b_draws_ge_ref']:.4f}"
              f"   P(b>0)={mc['frac_b_draws_positive']:.4f}")

    dest = REPO_ROOT / "outputs/analysis/circular_geometry/acuity_stratified_transport.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
