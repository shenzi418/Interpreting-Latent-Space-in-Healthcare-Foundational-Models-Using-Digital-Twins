# Promoted 2026-08-13 from reports/2026-08-13_audit_artifacts/scripts/tmp_f2_blocks.py
# VERIFIED 2026-08-13 by independent adversarial re-implementation
# (analysis/block_transfer_verify.py); verdict: CONFIRMED_WITH_CORRECTION (exact one-sided p=0.0417/0.0292 for the summary rho; fold-seed sensitivity declared; block orderings robust).
# Canonical outputs of the verified run: reports/2026-08-13_audit_artifacts/.
# Re-runs write to outputs/analysis/fidelity_audit/. Full record:
# reports/2026-08-13_fidelity_audit_and_final_verification.md, Part C.
# -*- coding: utf-8 -*-
"""F2: does the feature-level fidelity audit PREDICT cross-domain transfer?

Per-block circular territory readouts fit on MedalCare features, transported to
PTB-XL under both scalers. Blocks: ST_J60 x12, Q_amp x12, R_amp x12, T_amp x12,
globals x6, full-54, axis pair [R_amp_I, R_amp_aVF].

All prints ASCII. Results appended incrementally to
outputs/analysis/fidelity_audit/f2_blocks_out.txt and f2_blocks.json (UTF-8).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))
from geom_common import (TERRITORIES, RidgeSVD, angles_from_cs, group_folds,
                         medalcare_anchor_angles, resultant)

OUT_TXT = REPO / "outputs/analysis/fidelity_audit/f2_blocks_out.txt"
OUT_JSON = REPO / "outputs/analysis/fidelity_audit/f2_blocks.json"
RESULTS = {}


def log(msg=""):
    print(msg)
    with open(OUT_TXT, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def save_json():
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, default=float)


# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
fe_p = np.load(REPO / "data/ecg_features_spatial_ptbxl_allfolds.npz", allow_pickle=True)
NAMES = [str(x) for x in fe_p["feature_names"]]
F_p_all = np.asarray(fe_p["features"], dtype=float)

mi = pd.read_csv(REPO / "data/ptbxl_mi_subclass_allfolds.csv")
pos = np.flatnonzero(mi["territory_4c"].notna().to_numpy())
mi_mi = mi.iloc[pos].reset_index(drop=True)
Fp = F_p_all[pos]                                # (4324, 54)
terr_p = mi_mi["territory_4c"].to_numpy()
group_p = mi_mi["patient_id"].to_numpy().astype(str)

blocks_m, phis, terrs_m, groups_m = [], [], [], []
for split in ("train", "test"):
    t = np.load(REPO / f"data/theta_mi_{split}.npz", allow_pickle=True)
    fe_m = np.load(REPO / f"data/ecg_features_spatial_medalcare_{split}.npz",
                   allow_pickle=True)
    nm = [str(x) for x in fe_m["feature_names"]]
    assert nm == NAMES, "feature_names mismatch between domains"
    idx = np.asarray(t["idx_in_split"], dtype=int)
    blocks_m.append(np.asarray(fe_m["features"], dtype=float)[idx])
    phis.append(np.asarray(t["phi"], dtype=float))
    terrs_m.append(np.asarray(t["territory_4c"]))
    groups_m.append(np.array([f"{split}:{r}" for r in t["run_id"]]))
Fm = np.vstack(blocks_m)                          # (6547, 54)
phi_m = np.concatenate(phis)
terr_m = np.concatenate(terrs_m)
group_m = np.concatenate(groups_m)

ANCHORS = medalcare_anchor_angles()
anchor_arr = np.array([ANCHORS[t] for t in TERRITORIES])
angle_p = np.array([ANCHORS[t] for t in terr_p])
Y_m = np.column_stack([np.cos(phi_m), np.sin(phi_m)])

# Block definitions
BLOCKS = {
    "ST_J60":  [i for i, n in enumerate(NAMES) if n.startswith("ST_J60_") and not n.endswith("_mV")],
    "Q_amp":   [i for i, n in enumerate(NAMES) if n.startswith("Q_amp_")],
    "R_amp":   [i for i, n in enumerate(NAMES) if n.startswith("R_amp_")],
    "T_amp":   [i for i, n in enumerate(NAMES) if n.startswith("T_amp_") and not n.endswith("_mV")],
    "globals": [NAMES.index(n) for n in ["QRS_duration_ms", "QT_interval_ms",
                                         "P_duration_ms", "ST_J60_avg_mV",
                                         "T_amplitude_mV", "heart_rate_bpm"]],
    "full54":  list(range(54)),
    "axis2":   [NAMES.index("R_amp_I"), NAMES.index("R_amp_aVF")],
}

# fresh output files
OUT_TXT.write_text("", encoding="utf-8")
log("F2: per-block fidelity-audit prediction test")
log("=" * 78)
log("Feature block structure found (54 features):")
for b in ("ST_J60", "Q_amp", "R_amp", "T_amp"):
    log(f"  {b:8s} x{len(BLOCKS[b]):2d} : leads I,II,III,aVR,aVL,aVF,V1-V6")
log("  globals  x 6 : QRS_duration_ms, QT_interval_ms, P_duration_ms,")
log("                 ST_J60_avg_mV, T_amplitude_mV, heart_rate_bpm")
log("  plus full54 (all 54) and axis2 = [R_amp_I, R_amp_aVF]")
log("")
RESULTS["block_structure"] = {k: [NAMES[i] for i in v] for k, v in BLOCKS.items()}

# --------------------------------------------------------------------------- #
# PRE-STATED PREDICTION -- written BEFORE any result is computed
# --------------------------------------------------------------------------- #
PREDICTION = """PRE-STATED PREDICTION (written to disk before computing any result below):
From the fidelity audit (feature-level territory eta2: ST_J60 sim 0.056 vs real
0.016; Q_amp real 0.077 vs sim 0.007; R_amp real 0.065 vs sim 0.010):
 P1. ST_J60 block: STRONG in-domain (the simulator encodes territory in acute
     injury current), and transfers WORST (lowest transfer efficiency).
 P2. Q_amp and R_amp blocks: WEAKER in-domain, but transfer BEST (highest
     efficiency) because the real domain carries territory there.
 P3. axis2 [R_amp_I, R_amp_aVF]: the fit-free axis transfers near-perfectly in
     principle, BUT the simulator carries almost no territory in the frontal
     axis (eta2=0.0020, below the constant floor in-domain). Therefore its
     RIDGE FIT on MedalCare must be near the in-domain constant floor, the
     efficiency denominator (in-domain excess) is ~0, and the efficiency ratio
     is unstable/explosive rather than ~1. Implication to test: axis2 shows
     near-floor in-domain F1 excess, yet non-trivial cross-domain F1 (>0.1535
     floor), giving efficiency >> 1 or an undefined ratio -- the extreme case
     of 'the simulator fails to teach a channel that works in reality'.
 P4. Summary: Spearman rho between per-block eta2_sim (features directly) and
     transfer efficiency is NEGATIVE -- the simulator teaches the readout to
     rely on exactly the channels that do not transfer.
"""
log(PREDICTION)
RESULTS["pre_stated_prediction"] = PREDICTION
save_json()

# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def nearest_anchor_terr(pred_angle):
    d = np.abs(np.angle(np.exp(1j * (pred_angle[:, None] - anchor_arr[None, :]))))
    return np.array(TERRITORIES)[np.argmin(d, axis=1)]


def macro_f1(true_terr, pred_angle):
    return float(f1_score(true_terr, nearest_anchor_terr(pred_angle),
                          labels=TERRITORIES, average="macro"))


def circ_eta2(pred_angle, classes):
    z = np.exp(1j * pred_angle)
    R = np.abs(z.sum())
    sumRj = sum(np.abs(z[classes == c].sum()) for c in np.unique(classes))
    n = len(pred_angle)
    denom = n - R
    if denom <= 1e-9:
        return float("nan")
    return float((sumRj - R) / denom)


def circ_R(pred_angle, true_angle):
    return resultant(pred_angle - true_angle)


def impute(X, med):
    return np.where(np.isnan(X), med[None, :], X)


# floors (verify against established numbers)
R_floor_p = float(np.abs(np.mean(np.exp(-1j * angle_p))))
R_floor_m = float(np.abs(np.mean(np.exp(-1j * phi_m))))
const_f1_p = max(float(f1_score(terr_p, np.full(len(terr_p), t),
                                labels=TERRITORIES, average="macro"))
                 for t in TERRITORIES)
const_f1_m = max(float(f1_score(terr_m, np.full(len(terr_m), t),
                                labels=TERRITORIES, average="macro"))
                 for t in TERRITORIES)
log("Floors (verified by direct computation):")
log(f"  R constant floor      : PTB-XL {R_floor_p:.5f} (established 0.29216), "
    f"MedalCare {R_floor_m:.5f} (established 0.09319)")
log(f"  macro-F1 constant floor: PTB-XL {const_f1_p:.4f} (established 0.1535), "
    f"MedalCare {const_f1_m:.4f} (computed here)")
log(f"  n: MedalCare {len(phi_m)}, PTB-XL {len(terr_p)}")
log("")
RESULTS["floors"] = {"R_floor_ptbxl": R_floor_p, "R_floor_medalcare": R_floor_m,
                     "constF1_ptbxl": const_f1_p, "constF1_medalcare": const_f1_m,
                     "n_medalcare": int(len(phi_m)), "n_ptbxl": int(len(terr_p))}
save_json()

# --------------------------------------------------------------------------- #
# Shared CV folds (identical across blocks -> paired in-domain comparisons)
# --------------------------------------------------------------------------- #
rng = np.random.default_rng(0)
FOLDS = list(group_folds(group_m, 5, rng))

per_block = {}
oof_angle = {}       # block -> pooled out-of-fold predicted angles (MedalCare)
cross_angle = {}     # block -> {scaler: predicted angles on PTB-XL}

log("Per-block results")
log("-" * 78)
for bname, cols in BLOCKS.items():
    cols = np.asarray(cols)
    Xm = Fm[:, cols]
    Xp = Fp[:, cols]
    n_fin_m = int(np.isfinite(Xm).all(axis=1).sum())
    n_fin_p = int(np.isfinite(Xp).all(axis=1).sum())

    # ---- in-domain group-disjoint 5-fold CV (train-fold medians for impute)
    pred_ang = np.full(len(phi_m), np.nan)
    alphas_cv = []
    for tr, te in FOLDS:
        med = np.nanmedian(Xm[tr], axis=0)
        Xtr = impute(Xm[tr], med)
        Xte = impute(Xm[te], med)
        m = RidgeSVD().fit(Xtr, Y_m[tr])
        alphas_cv.append(m.alpha_)
        pred_ang[te] = angles_from_cs(m.predict(Xte))
    oof_angle[bname] = pred_ang
    f1_in = macro_f1(terr_m, pred_ang)
    eta2_in = circ_eta2(pred_ang, terr_m)
    R_in = circ_R(pred_ang, phi_m)

    # ---- transport model: fit on all MedalCare, MedalCare medians only
    med_m = np.nanmedian(Xm, axis=0)
    Xm_imp = impute(Xm, med_m)
    m = RidgeSVD().fit(Xm_imp, Y_m)
    alpha_full = m.alpha_

    # source (strict) scaler: MedalCare medians for imputation, MedalCare mu/sd
    Xp_src = impute(Xp, med_m)
    ang_src = angles_from_cs(m.predict(Xp_src))
    # target (diagonal CORAL): PTB-XL medians + PTB-XL mu/sd
    med_p = np.nanmedian(Xp, axis=0)
    Xp_tgt = impute(Xp, med_p)
    zt = (Xp_tgt - Xp_tgt.mean(0)) / (Xp_tgt.std(0) + 1e-8)
    ang_tgt = angles_from_cs(m.predict(zt * m.sd_ + m.mu_))
    cross_angle[bname] = {"source": ang_src, "target": ang_tgt}

    row = {
        "n_features": int(len(cols)),
        "n_finite_medalcare": n_fin_m, "n_finite_ptbxl": n_fin_p,
        "n_scored_medalcare": int(len(phi_m)), "n_scored_ptbxl": int(len(terr_p)),
        "alpha_cv_median": float(np.median(alphas_cv)),
        "alpha_transport": float(alpha_full),
        "in": {"macro_f1": f1_in, "eta2": eta2_in, "R": R_in},
    }
    for sc, ang in (("source", ang_src), ("target", ang_tgt)):
        row[f"cross_{sc}"] = {
            "macro_f1": macro_f1(terr_p, ang),
            "eta2": circ_eta2(ang, terr_p),
            "R": circ_R(ang, angle_p),
        }
    per_block[bname] = row

    log(f"[{bname}]  k={len(cols)}  finite rows: MedalCare {n_fin_m}/{len(phi_m)}, "
        f"PTB-XL {n_fin_p}/{len(terr_p)} (all rows scored via median impute)")
    log(f"  alpha: CV median {row['alpha_cv_median']:.4g}, transport {alpha_full:.4g}")
    log(f"  in-domain (5-fold group CV): macro-F1 {f1_in:.4f}  eta2 {eta2_in:.4f}  "
        f"R {R_in:.4f} (floor {R_floor_m:.4f})")
    for sc in ("source", "target"):
        c = row[f"cross_{sc}"]
        log(f"  cross {sc:6s}: macro-F1 {c['macro_f1']:.4f}  eta2 {c['eta2']:.4f}  "
        f"R {c['R']:.4f} (floor {R_floor_p:.4f})")
    log("")
    RESULTS["per_block"] = per_block
    save_json()

# --------------------------------------------------------------------------- #
# Transfer efficiency
# --------------------------------------------------------------------------- #
log("Transfer efficiency  =  (cross F1 - const floor 0.1535[computed "
    f"{const_f1_p:.4f}]) / (in-domain F1 - MedalCare const floor {const_f1_m:.4f})")
log("eta2 ratio = eta2_cross / eta2_in")
log("-" * 78)
log(f"{'block':8s} {'F1_in':>7s} {'exc_in':>7s} | {'F1_src':>7s} {'eff_src':>8s} "
    f"{'F1_tgt':>7s} {'eff_tgt':>8s} | {'e2_in':>6s} {'e2rat_s':>8s} {'e2rat_t':>8s}")
eff = {}
for bname, row in per_block.items():
    exc_in = row["in"]["macro_f1"] - const_f1_m
    e = {}
    for sc in ("source", "target"):
        exc_x = row[f"cross_{sc}"]["macro_f1"] - const_f1_p
        e[f"eff_{sc}"] = exc_x / exc_in if abs(exc_in) > 1e-6 else float("nan")
        e2_in = row["in"]["eta2"]
        e[f"eta2_ratio_{sc}"] = (row[f"cross_{sc}"]["eta2"] / e2_in
                                 if abs(e2_in) > 1e-9 else float("nan"))
    e["excess_in"] = exc_in
    eff[bname] = e
    log(f"{bname:8s} {row['in']['macro_f1']:7.4f} {exc_in:7.4f} | "
        f"{row['cross_source']['macro_f1']:7.4f} {e['eff_source']:8.3f} "
        f"{row['cross_target']['macro_f1']:7.4f} {e['eff_target']:8.3f} | "
        f"{row['in']['eta2']:6.4f} {e['eta2_ratio_source']:8.3f} {e['eta2_ratio_target']:8.3f}")
log("")
RESULTS["efficiency"] = eff
save_json()

# --------------------------------------------------------------------------- #
# Paired block-vs-block tests (patient block bootstrap + swap permutation)
# --------------------------------------------------------------------------- #
def paired_test(terr_true, groups, angA, angB, n_draws=1000, seed=1):
    """Paired Delta = F1(A) - F1(B). Group bootstrap CI + group swap permutation."""
    pA = nearest_anchor_terr(angA)
    pB = nearest_anchor_terr(angB)
    f1A = float(f1_score(terr_true, pA, labels=TERRITORIES, average="macro"))
    f1B = float(f1_score(terr_true, pB, labels=TERRITORIES, average="macro"))
    dobs = f1A - f1B
    uniq = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in uniq}
    r = np.random.default_rng(seed)
    # bootstrap CI on Delta
    deltas = np.empty(n_draws)
    for b in range(n_draws):
        gs = r.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_of[g] for g in gs])
        deltas[b] = (f1_score(terr_true[rows], pA[rows], labels=TERRITORIES,
                              average="macro")
                     - f1_score(terr_true[rows], pB[rows], labels=TERRITORIES,
                                average="macro"))
    ci = (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))
    # swap permutation p
    count = 0
    for b in range(n_draws):
        flip = r.random(len(uniq)) < 0.5
        pa, pb = pA.copy(), pB.copy()
        for g, fl in zip(uniq, flip):
            if fl:
                ix = idx_of[g]
                pa[ix], pb[ix] = pB[ix], pA[ix]
        d = (f1_score(terr_true, pa, labels=TERRITORIES, average="macro")
             - f1_score(terr_true, pb, labels=TERRITORIES, average="macro"))
        if abs(d) >= abs(dobs) - 1e-12:
            count += 1
    p = (count + 1) / (n_draws + 1)
    return {"f1_A": f1A, "f1_B": f1B, "delta": float(dobs),
            "ci95": ci, "p_swap": float(p), "n_draws": n_draws}


best_block = max((b for b in BLOCKS if b != "full54"),
                 key=lambda b: per_block[b]["cross_target"]["macro_f1"])
log(f"Best single block cross-domain (target scaler): {best_block} "
    f"(F1 {per_block[best_block]['cross_target']['macro_f1']:.4f})")
log("")
log("Paired tests, cross-domain PTB-XL (patient block bootstrap, 1000 draws;"
    " p from patient-level swap permutation)")
log("-" * 78)
paired = {}
pairs = [("ST_J60", "Q_amp"), ("ST_J60", "R_amp"), ("full54", best_block)]
for sc in ("source", "target"):
    for a, b in pairs:
        key = f"cross_{sc}:{a}_vs_{b}"
        res = paired_test(terr_p, group_p, cross_angle[a][sc], cross_angle[b][sc])
        paired[key] = res
        log(f"  [{sc}] {a} vs {b}: Delta {res['delta']:+.4f} "
            f"CI [{res['ci95'][0]:+.4f}, {res['ci95'][1]:+.4f}]  p={res['p_swap']:.4f}"
            f"  (F1 {res['f1_A']:.4f} vs {res['f1_B']:.4f})")
    RESULTS["paired_cross"] = paired
    save_json()
log("")
log("Paired tests, in-domain MedalCare (run-block bootstrap on pooled OOF preds)")
paired_in = {}
for a, b in [("ST_J60", "Q_amp"), ("ST_J60", "R_amp")]:
    res = paired_test(terr_m, group_m, oof_angle[a], oof_angle[b])
    paired_in[f"in:{a}_vs_{b}"] = res
    log(f"  [in] {a} vs {b}: Delta {res['delta']:+.4f} "
        f"CI [{res['ci95'][0]:+.4f}, {res['ci95'][1]:+.4f}]  p={res['p_swap']:.4f}"
        f"  (F1 {res['f1_A']:.4f} vs {res['f1_B']:.4f})")
RESULTS["paired_in"] = paired_in
save_json()
log("")

# axis2 P3 test: is axis2 cross-domain F1 above the constant floor?
def f1_boot_ci(terr_true, groups, ang, n_draws=1000, seed=2):
    pr = nearest_anchor_terr(ang)
    uniq = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in uniq}
    r = np.random.default_rng(seed)
    vals = np.empty(n_draws)
    for b in range(n_draws):
        gs = r.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_of[g] for g in gs])
        vals[b] = f1_score(terr_true[rows], pr[rows], labels=TERRITORIES,
                           average="macro")
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


log("P3 check -- axis2 above floors?")
ax = {}
ci_in = f1_boot_ci(terr_m, group_m, oof_angle["axis2"])
log(f"  axis2 in-domain OOF F1 {per_block['axis2']['in']['macro_f1']:.4f}, "
    f"run-block bootstrap CI [{ci_in[0]:.4f}, {ci_in[1]:.4f}] "
    f"vs MedalCare const floor {const_f1_m:.4f}")
ax["in_ci"] = ci_in
for sc in ("source", "target"):
    ci_x = f1_boot_ci(terr_p, group_p, cross_angle["axis2"][sc])
    ax[f"cross_{sc}_ci"] = ci_x
    log(f"  axis2 cross {sc} F1 {per_block['axis2'][f'cross_{sc}']['macro_f1']:.4f}, "
        f"patient bootstrap CI [{ci_x[0]:.4f}, {ci_x[1]:.4f}] vs floor {const_f1_p:.4f}")
RESULTS["axis2_floor_check"] = ax
save_json()
log("")

# --------------------------------------------------------------------------- #
# eta2_sim per block (features directly) and the Spearman summary
# --------------------------------------------------------------------------- #
def anova_eta2(x, classes):
    fin = np.isfinite(x)
    x, c = x[fin], classes[fin]
    gm = x.mean()
    ssb = sum(len(x[c == k]) * (x[c == k].mean() - gm) ** 2 for k in np.unique(c))
    sst = ((x - gm) ** 2).sum()
    return float(ssb / sst) if sst > 0 else float("nan")


eta2_feat_sim = np.array([anova_eta2(Fm[:, j], terr_m) for j in range(54)])
eta2_feat_real = np.array([anova_eta2(Fp[:, j], terr_p) for j in range(54)])

log("Per-block feature-level eta2 (ANOVA eta2 of each feature vs territory,"
    " finite rows, mean over block; max in parens)")
log("-" * 78)
block_eta2_sim, block_eta2_real = {}, {}
for bname, cols in BLOCKS.items():
    cols = np.asarray(cols)
    s_mean, s_max = float(np.nanmean(eta2_feat_sim[cols])), float(np.nanmax(eta2_feat_sim[cols]))
    r_mean, r_max = float(np.nanmean(eta2_feat_real[cols])), float(np.nanmax(eta2_feat_real[cols]))
    block_eta2_sim[bname] = s_mean
    block_eta2_real[bname] = r_mean
    log(f"  {bname:8s} eta2_sim mean {s_mean:.4f} (max {s_max:.4f})   "
        f"eta2_real mean {r_mean:.4f} (max {r_max:.4f})")
RESULTS["block_eta2_features"] = {
    "sim_mean": block_eta2_sim, "real_mean": block_eta2_real,
    "per_feature_sim": {NAMES[j]: float(eta2_feat_sim[j]) for j in range(54)},
    "per_feature_real": {NAMES[j]: float(eta2_feat_real[j]) for j in range(54)},
}
save_json()
log("")

log("SUMMARY STATISTIC: Spearman rho(eta2_sim(block), transfer efficiency(block))")
log("-" * 78)
spear = {}
for blockset, label in [
    (["ST_J60", "Q_amp", "R_amp", "T_amp", "globals"], "5 natural blocks"),
    (["ST_J60", "Q_amp", "R_amp", "T_amp", "globals", "axis2"], "5 blocks + axis2"),
]:
    xs = [block_eta2_sim[b] for b in blockset]
    for sc in ("source", "target"):
        ys = [eff[b][f"eff_{sc}"] for b in blockset]
        fin = np.isfinite(ys)
        rho, pv = spearmanr(np.asarray(xs)[fin], np.asarray(ys)[fin])
        spear[f"{label}|{sc}"] = {"rho": float(rho), "p": float(pv),
                                  "n_blocks": int(fin.sum()),
                                  "blocks": [b for b, f in zip(blockset, fin) if f]}
        log(f"  [{label}, {sc} scaler] rho = {rho:+.3f} (p={pv:.3f}, "
            f"n={int(fin.sum())} blocks)")
RESULTS["spearman"] = spear
save_json()
log("")
log("Done.")
