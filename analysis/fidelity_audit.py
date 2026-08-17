# Promoted 2026-08-13 from reports/2026-08-13_audit_artifacts/scripts/tmp_f1_fidelity.py
# VERIFIED 2026-08-13 by independent adversarial re-implementation
# (analysis/fidelity_audit_verify.py); verdict: CONFIRMED_WITH_CORRECTION (block-aware permutation p=0.103/0.141 for the per-feature rho; block-level inversion is the licensed claim).
# Canonical outputs of the verified run: reports/2026-08-13_audit_artifacts/.
# Re-runs write to outputs/analysis/fidelity_audit/. Full record:
# reports/2026-08-13_fidelity_audit_and_final_verification.md, Part C.
# tmp_f1_fidelity.py - F1: per-feature informativeness-fidelity audit of
# MedalCare-XL vs PTB-XL. ASCII-only console output; UTF-8 result files.
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
OUT_TXT = REPO / "outputs/analysis/fidelity_audit/f1_fidelity_out.txt"
OUT_JSON = REPO / "outputs/analysis/fidelity_audit/f1_fidelity.json"

RNG_SEED = 20260813
N_BOOT = 500

TERRITORIES = ["Anteroseptal", "Anterolateral", "Inferolateral", "Inferior"]

log_lines = []


def log(s=""):
    print(s)
    log_lines.append(s)


def flush_txt():
    OUT_TXT.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def flush_json(obj):
    OUT_JSON.write_text(json.dumps(obj, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Load features + labels, both domains
# --------------------------------------------------------------------------- #
def load_medalcare_features():
    blocks, terrs, groups = [], [], []
    names = None
    for split in ("train", "test"):
        t = np.load(REPO / f"data/theta_mi_{split}.npz", allow_pickle=True)
        fe = np.load(
            REPO / f"data/ecg_features_spatial_medalcare_{split}.npz",
            allow_pickle=True,
        )
        nm = [str(x) for x in fe["feature_names"]]
        if names is None:
            names = nm
        assert nm == names, "feature_names mismatch across MedalCare splits"
        idx = np.asarray(t["idx_in_split"], dtype=int)
        blocks.append(fe["features"][idx])
        terrs.append(np.asarray(t["territory_4c"]).astype(str))
        groups.append(np.array([f"{split}:{r}" for r in t["run_id"]]))
    return np.vstack(blocks), np.concatenate(terrs), np.concatenate(groups), names


def load_ptbxl_features():
    fe = np.load(REPO / "data/ecg_features_spatial_ptbxl_allfolds.npz", allow_pickle=True)
    names = [str(x) for x in fe["feature_names"]]
    mi = pd.read_csv(REPO / "data/ptbxl_mi_subclass_allfolds.csv")
    pos = np.flatnonzero(mi["territory_4c"].notna().to_numpy())
    F = fe["features"][pos]
    terr = mi["territory_4c"].to_numpy()[pos].astype(str)
    group = mi["patient_id"].to_numpy()[pos].astype(str)
    return F, terr, group, names


Fs, terr_s, grp_s, names_s = load_medalcare_features()
Fr, terr_r, grp_r, names_r = load_ptbxl_features()
assert names_s == names_r, "feature name mismatch between domains"
NAMES = names_s
assert Fs.shape == (6547, 54), Fs.shape
assert Fr.shape == (4324, 54), Fr.shape

i_aVF = NAMES.index("R_amp_aVF")
i_I = NAMES.index("R_amp_I")
axis_s = np.arctan2(Fs[:, i_aVF], Fs[:, i_I])
axis_r = np.arctan2(Fr[:, i_aVF], Fr[:, i_I])
axis_fin_s = np.isfinite(Fs[:, i_aVF]) & np.isfinite(Fs[:, i_I])
axis_fin_r = np.isfinite(Fr[:, i_aVF]) & np.isfinite(Fr[:, i_I])

y4_s = np.array([TERRITORIES.index(t) for t in terr_s])
y4_r = np.array([TERRITORIES.index(t) for t in terr_r])
y2_s = (y4_s >= 2).astype(int)  # 0 = AS+AL, 1 = IL+INF
y2_r = (y4_r >= 2).astype(int)

log("F1 per-feature informativeness-fidelity audit  (seed=%d, n_boot=%d)" % (RNG_SEED, N_BOOT))
log("MedalCare n=%d (%d run blocks); PTB-XL n=%d (%d patient blocks)"
    % (len(y4_s), len(np.unique(grp_s)), len(y4_r), len(np.unique(grp_r))))
log("Axis finite: sim %d, real %d" % (int(axis_fin_s.sum()), int(axis_fin_r.sum())))
log("")
log("FEATURE BLOCK STRUCTURE (54 features):")
log("  [0:12]  ST_J60_<lead>   x 12 leads (I,II,III,aVR,aVL,aVF,V1-V6)")
log("  [12:24] Q_amp_<lead>    x 12 leads")
log("  [24:36] R_amp_<lead>    x 12 leads")
log("  [36:48] T_amp_<lead>    x 12 leads")
log("  [48:54] globals: QRS_duration_ms, QT_interval_ms, P_duration_ms,")
log("          ST_J60_avg_mV, T_amplitude_mV, heart_rate_bpm")
log("  plus derived circular feature: frontal QRS axis = atan2(R_amp_aVF, R_amp_I)")
log("")


# --------------------------------------------------------------------------- #
# eta2 machinery (vectorised over features, NaN-aware)
# --------------------------------------------------------------------------- #
def prep_domain(F):
    fin = np.isfinite(F)
    Xz = np.where(fin, F, 0.0)
    X2z = np.where(fin, F * F, 0.0)
    return fin.astype(float), Xz, X2z


def eta2_all(rows, finf, Xz, X2z, y, n_classes):
    """ANOVA eta2 for every feature column on the given row subset."""
    yv = y[rows]
    n_tot = np.zeros(Xz.shape[1])
    s_tot = np.zeros(Xz.shape[1])
    ss_tot = np.zeros(Xz.shape[1])
    ssb = np.zeros(Xz.shape[1])
    parts = []
    for c in range(n_classes):
        rc = rows[yv == c]
        n_c = finf[rc].sum(0)
        s_c = Xz[rc].sum(0)
        ss_c = X2z[rc].sum(0)
        n_tot += n_c
        s_tot += s_c
        ss_tot += ss_c
        parts.append((n_c, s_c))
    grand = np.where(n_tot > 0, s_tot / np.maximum(n_tot, 1), 0.0)
    sst = ss_tot - n_tot * grand ** 2
    for n_c, s_c in parts:
        mc = np.where(n_c > 0, s_c / np.maximum(n_c, 1), 0.0)
        ssb += n_c * (mc - grand) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(sst > 0, ssb / sst, np.nan)
    return e


def circ_eta2(rows, ang, fin, y, n_classes):
    """(sum_j R_j - R) / (N - R) with unnormalised resultant lengths."""
    yv = y[rows]
    fv = fin[rows]
    av = ang[rows]
    z = np.exp(1j * av)
    z = np.where(fv, z, 0.0)
    N = float(fv.sum())
    R = abs(z.sum())
    sumRj = 0.0
    for c in range(n_classes):
        sumRj += abs(z[yv == c].sum())
    denom = N - R
    return (sumRj - R) / denom if denom > 0 else np.nan


fin_s, Xz_s, X2z_s = prep_domain(Fs)
fin_r, Xz_r, X2z_r = prep_domain(Fr)

all_s = np.arange(len(y4_s))
all_r = np.arange(len(y4_r))

point = {}
for dom, rows, finf, Xz, X2z, y4, y2, ang, afin in [
    ("sim", all_s, fin_s, Xz_s, X2z_s, y4_s, y2_s, axis_s, axis_fin_s),
    ("real", all_r, fin_r, Xz_r, X2z_r, y4_r, y2_r, axis_r, axis_fin_r),
]:
    point[dom] = {
        "eta2_4c": eta2_all(rows, finf, Xz, X2z, y4, 4),
        "eta2_2c": eta2_all(rows, finf, Xz, X2z, y2, 2),
        "axis_eta2_4c": circ_eta2(rows, ang, afin, y4, 4),
        "axis_eta2_2c": circ_eta2(rows, ang, afin, y2, 2),
    }

log("Point estimates computed.")
log("Axis circular eta2 (4c): sim %.4f  real %.4f" %
    (point["sim"]["axis_eta2_4c"], point["real"]["axis_eta2_4c"]))
log("Axis circular eta2 (2c): sim %.4f  real %.4f" %
    (point["sim"]["axis_eta2_2c"], point["real"]["axis_eta2_2c"]))
flush_txt()

# --------------------------------------------------------------------------- #
# (c) marginal realism stats
# --------------------------------------------------------------------------- #
mean_s = np.array([np.nanmean(Fs[:, j]) for j in range(54)])
mean_r = np.array([np.nanmean(Fr[:, j]) for j in range(54)])
sd_s = np.array([np.nanstd(Fs[:, j]) for j in range(54)])
sd_r = np.array([np.nanstd(Fr[:, j]) for j in range(54)])
pooled = np.sqrt((sd_s ** 2 + sd_r ** 2) / 2.0)
smd = (mean_r - mean_s) / np.where(pooled > 0, pooled, np.nan)
sd_ratio = sd_r / np.where(sd_s > 0, sd_s, np.nan)

# axis circular marginal stats


def circ_mean_R(a, fin):
    z = np.exp(1j * a[fin])
    m = z.mean()
    return float(np.angle(m)), float(abs(m))


ax_mu_s, ax_R_s = circ_mean_R(axis_s, axis_fin_s)
ax_mu_r, ax_R_r = circ_mean_R(axis_r, axis_fin_r)
ax_sd_s = float(np.sqrt(-2 * np.log(ax_R_s)) * 180 / np.pi)
ax_sd_r = float(np.sqrt(-2 * np.log(ax_R_r)) * 180 / np.pi)
ax_dmu = float(np.angle(np.exp(1j * (ax_mu_r - ax_mu_s))) * 180 / np.pi)

# --------------------------------------------------------------------------- #
# Bootstrap: block bootstrap per domain, shared draws for paired differences
# --------------------------------------------------------------------------- #
def group_index(groups):
    uniq = np.unique(groups)
    lut = {}
    for i, g in enumerate(groups):
        lut.setdefault(g, []).append(i)
    return uniq, {g: np.array(v) for g, v in lut.items()}


uniq_s, lut_s = group_index(grp_s)
uniq_r, lut_r = group_index(grp_r)
rng = np.random.default_rng(RNG_SEED)

boot = {
    "sim": {"eta2_4c": [], "eta2_2c": [], "axis_4c": [], "axis_2c": []},
    "real": {"eta2_4c": [], "eta2_2c": [], "axis_4c": [], "axis_2c": []},
}
for b in range(N_BOOT):
    for dom, uniq, lut, finf, Xz, X2z, y4, y2, ang, afin in [
        ("sim", uniq_s, lut_s, fin_s, Xz_s, X2z_s, y4_s, y2_s, axis_s, axis_fin_s),
        ("real", uniq_r, lut_r, fin_r, Xz_r, X2z_r, y4_r, y2_r, axis_r, axis_fin_r),
    ]:
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([lut[g] for g in gs])
        boot[dom]["eta2_4c"].append(eta2_all(rows, finf, Xz, X2z, y4, 4))
        boot[dom]["eta2_2c"].append(eta2_all(rows, finf, Xz, X2z, y2, 2))
        boot[dom]["axis_4c"].append(circ_eta2(rows, ang, afin, y4, 4))
        boot[dom]["axis_2c"].append(circ_eta2(rows, ang, afin, y2, 2))
    if (b + 1) % 100 == 0:
        print("boot %d/%d" % (b + 1, N_BOOT))

for dom in boot:
    for k in boot[dom]:
        boot[dom][k] = np.asarray(boot[dom][k])


def ci(a, axis=0):
    lo = np.nanpercentile(a, 2.5, axis=axis)
    hi = np.nanpercentile(a, 97.5, axis=axis)
    return lo, hi


ci_s4 = ci(boot["sim"]["eta2_4c"])
ci_r4 = ci(boot["real"]["eta2_4c"])
ci_s2 = ci(boot["sim"]["eta2_2c"])
ci_r2 = ci(boot["real"]["eta2_2c"])
diff4 = boot["real"]["eta2_4c"] - boot["sim"]["eta2_4c"]   # paired by draw index
diff2 = boot["real"]["eta2_2c"] - boot["sim"]["eta2_2c"]
ci_d4 = ci(diff4)
ci_d2 = ci(diff2)
ax_d4 = boot["real"]["axis_4c"] - boot["sim"]["axis_4c"]
ax_d2 = boot["real"]["axis_2c"] - boot["sim"]["axis_2c"]

log("")
log("Bootstrap complete (%d draws, run-block resampling in MedalCare, patient-"
    "block in PTB-XL; domains resampled independently, differences paired by"
    " draw index)." % N_BOOT)
flush_txt()

# --------------------------------------------------------------------------- #
# (d) rank-order agreement
# --------------------------------------------------------------------------- #
e_s4 = point["sim"]["eta2_4c"]
e_r4 = point["real"]["eta2_4c"]
e_s2 = point["sim"]["eta2_2c"]
e_r2 = point["real"]["eta2_2c"]

rho54_4, rho54_4p = sps.spearmanr(e_s4, e_r4)
rho54_2, rho54_2p = sps.spearmanr(e_s2, e_r2)
e_s4x = np.append(e_s4, point["sim"]["axis_eta2_4c"])
e_r4x = np.append(e_r4, point["real"]["axis_eta2_4c"])
rho55_4, rho55_4p = sps.spearmanr(e_s4x, e_r4x)

rho_boot4, rho_boot2, rho_boot55 = [], [], []
for b in range(N_BOOT):
    bs4 = boot["sim"]["eta2_4c"][b]
    br4 = boot["real"]["eta2_4c"][b]
    rho_boot4.append(sps.spearmanr(bs4, br4)[0])
    rho_boot2.append(sps.spearmanr(boot["sim"]["eta2_2c"][b], boot["real"]["eta2_2c"][b])[0])
    rho_boot55.append(sps.spearmanr(
        np.append(bs4, boot["sim"]["axis_4c"][b]),
        np.append(br4, boot["real"]["axis_4c"][b]))[0])
rho_boot4 = np.asarray(rho_boot4)
rho_boot2 = np.asarray(rho_boot2)
rho_boot55 = np.asarray(rho_boot55)

# (c) correlation |marginal shift| vs |informativeness gap| (54 linear features)
gap4 = np.abs(e_r4 - e_s4)
asmd = np.abs(smd)
c_sp, c_sp_p = sps.spearmanr(asmd, gap4)
c_pe, c_pe_p = sps.pearsonr(asmd, gap4)
lsdr = np.abs(np.log(sd_ratio))
c_sd_sp, c_sd_sp_p = sps.spearmanr(lsdr, gap4)

# --------------------------------------------------------------------------- #
# (f) missingness audit
# --------------------------------------------------------------------------- #
miss = {}
for dom, F, y4, n in [("sim", Fs, y4_s, 6547), ("real", Fr, y4_r, 4324)]:
    fin = np.isfinite(F)
    per_feat = fin.sum(0)
    chi_p = np.full(54, np.nan)
    for j in range(54):
        if per_feat[j] < n:  # some missingness exists
            tab = np.zeros((4, 2))
            for c in range(4):
                m = y4 == c
                tab[c, 0] = fin[m, j].sum()
                tab[c, 1] = (~fin[m, j]).sum()
            if (tab.sum(1) > 0).all() and tab[:, 1].sum() > 0:
                try:
                    chi_p[j] = sps.chi2_contingency(tab)[1]
                except ValueError:
                    pass
    all54 = fin.all(1)
    by_terr = {TERRITORIES[c]: [int(all54[y4 == c].sum()), int((y4 == c).sum())]
               for c in range(4)}
    tab_all = np.array([[all54[y4 == c].sum(), (~all54[y4 == c]).sum()] for c in range(4)])
    p_all54 = float(sps.chi2_contingency(tab_all)[1])
    miss[dom] = {"finite_n": per_feat.tolist(), "chi2_p_by_territory": chi_p.tolist(),
                 "all54_by_territory": by_terr, "all54_chi2_p": p_all54,
                 "n": n}

# Holm across features WITH missingness, per domain
holm_sig = {}
for dom in miss:
    ps = np.asarray(miss[dom]["chi2_p_by_territory"])
    idx = np.flatnonzero(np.isfinite(ps))
    order = idx[np.argsort(ps[idx])]
    m = len(order)
    sig = []
    for k, j in enumerate(order):
        if ps[j] * (m - k) < 0.05:
            sig.append(j)
        else:
            break
    holm_sig[dom] = sig

# --------------------------------------------------------------------------- #
# Assemble JSON
# --------------------------------------------------------------------------- #
features_out = []
for j in range(54):
    features_out.append({
        "name": NAMES[j],
        "eta2_4c_sim": float(e_s4[j]), "eta2_4c_sim_ci": [float(ci_s4[0][j]), float(ci_s4[1][j])],
        "eta2_4c_real": float(e_r4[j]), "eta2_4c_real_ci": [float(ci_r4[0][j]), float(ci_r4[1][j])],
        "diff_4c_real_minus_sim": float(e_r4[j] - e_s4[j]),
        "diff_4c_ci": [float(ci_d4[0][j]), float(ci_d4[1][j])],
        "eta2_2c_sim": float(e_s2[j]), "eta2_2c_sim_ci": [float(ci_s2[0][j]), float(ci_s2[1][j])],
        "eta2_2c_real": float(e_r2[j]), "eta2_2c_real_ci": [float(ci_r2[0][j]), float(ci_r2[1][j])],
        "diff_2c_real_minus_sim": float(e_r2[j] - e_s2[j]),
        "diff_2c_ci": [float(ci_d2[0][j]), float(ci_d2[1][j])],
        "mean_sim": float(mean_s[j]), "mean_real": float(mean_r[j]),
        "sd_sim": float(sd_s[j]), "sd_real": float(sd_r[j]),
        "smd_real_minus_sim": float(smd[j]), "sd_ratio_real_over_sim": float(sd_ratio[j]),
        "finite_n_sim": int(miss["sim"]["finite_n"][j]),
        "finite_n_real": int(miss["real"]["finite_n"][j]),
        "miss_chi2_p_sim": None if not np.isfinite(miss["sim"]["chi2_p_by_territory"][j]) else float(miss["sim"]["chi2_p_by_territory"][j]),
        "miss_chi2_p_real": None if not np.isfinite(miss["real"]["chi2_p_by_territory"][j]) else float(miss["real"]["chi2_p_by_territory"][j]),
    })

axis_out = {
    "name": "frontal_QRS_axis (circular)",
    "eta2_4c_sim": float(point["sim"]["axis_eta2_4c"]),
    "eta2_4c_sim_ci": [float(np.nanpercentile(boot["sim"]["axis_4c"], 2.5)),
                       float(np.nanpercentile(boot["sim"]["axis_4c"], 97.5))],
    "eta2_4c_real": float(point["real"]["axis_eta2_4c"]),
    "eta2_4c_real_ci": [float(np.nanpercentile(boot["real"]["axis_4c"], 2.5)),
                        float(np.nanpercentile(boot["real"]["axis_4c"], 97.5))],
    "diff_4c_real_minus_sim": float(point["real"]["axis_eta2_4c"] - point["sim"]["axis_eta2_4c"]),
    "diff_4c_ci": [float(np.nanpercentile(ax_d4, 2.5)), float(np.nanpercentile(ax_d4, 97.5))],
    "eta2_2c_sim": float(point["sim"]["axis_eta2_2c"]),
    "eta2_2c_real": float(point["real"]["axis_eta2_2c"]),
    "diff_2c_ci": [float(np.nanpercentile(ax_d2, 2.5)), float(np.nanpercentile(ax_d2, 97.5))],
    "circ_mean_deg_sim": float(ax_mu_s * 180 / np.pi),
    "circ_mean_deg_real": float(ax_mu_r * 180 / np.pi),
    "circ_mean_diff_deg": ax_dmu,
    "circ_sd_deg_sim": ax_sd_s, "circ_sd_deg_real": ax_sd_r,
    "finite_n_sim": int(axis_fin_s.sum()), "finite_n_real": int(axis_fin_r.sum()),
}

result = {
    "meta": {"seed": RNG_SEED, "n_boot": N_BOOT,
             "n_sim": 6547, "n_real": 4324,
             "n_run_blocks_sim": int(len(uniq_s)), "n_patient_blocks_real": int(len(uniq_r)),
             "two_class_def": "0=Anteroseptal+Anterolateral, 1=Inferolateral+Inferior"},
    "rank_agreement": {
        "spearman_rho_54feat_4c": float(rho54_4), "p": float(rho54_4p),
        "rho_54feat_4c_boot_ci": [float(np.nanpercentile(rho_boot4, 2.5)),
                                  float(np.nanpercentile(rho_boot4, 97.5))],
        "spearman_rho_54feat_2c": float(rho54_2), "p_2c": float(rho54_2p),
        "rho_54feat_2c_boot_ci": [float(np.nanpercentile(rho_boot2, 2.5)),
                                  float(np.nanpercentile(rho_boot2, 97.5))],
        "spearman_rho_55_with_axis_4c": float(rho55_4), "p_55": float(rho55_4p),
        "rho_55_boot_ci": [float(np.nanpercentile(rho_boot55, 2.5)),
                           float(np.nanpercentile(rho_boot55, 97.5))],
    },
    "marginal_vs_informativeness": {
        "spearman_absSMD_vs_absGap4c": float(c_sp), "p": float(c_sp_p),
        "pearson_absSMD_vs_absGap4c": float(c_pe), "p_pearson": float(c_pe_p),
        "spearman_absLogSDratio_vs_absGap4c": float(c_sd_sp), "p_sd": float(c_sd_sp_p),
    },
    "missingness": miss,
    "missingness_holm_sig_features": {d: [NAMES[j] for j in holm_sig[d]] for d in holm_sig},
    "axis": axis_out,
    "features": features_out,
}
flush_json(result)
log("JSON written (stage 1).")
flush_txt()

# --------------------------------------------------------------------------- #
# Report tables
# --------------------------------------------------------------------------- #
log("")
log("=" * 100)
log("(a,b) PER-FEATURE eta2 TABLE, 4-class territory  (sorted by eta2_real desc)")
log("%-18s %8s %19s %8s %19s %9s %19s" % (
    "feature", "e2_sim", "sim 95% CI", "e2_real", "real 95% CI", "diff r-s", "diff 95% CI"))
order = np.argsort(-e_r4)
for j in order:
    log("%-18s %8.4f [%7.4f,%7.4f] %8.4f [%7.4f,%7.4f] %+9.4f [%+7.4f,%+7.4f]" % (
        NAMES[j], e_s4[j], ci_s4[0][j], ci_s4[1][j],
        e_r4[j], ci_r4[0][j], ci_r4[1][j],
        e_r4[j] - e_s4[j], ci_d4[0][j], ci_d4[1][j]))
log("%-18s %8.4f [%7.4f,%7.4f] %8.4f [%7.4f,%7.4f] %+9.4f [%+7.4f,%+7.4f]" % (
    "AXIS(circ)", axis_out["eta2_4c_sim"], axis_out["eta2_4c_sim_ci"][0],
    axis_out["eta2_4c_sim_ci"][1], axis_out["eta2_4c_real"],
    axis_out["eta2_4c_real_ci"][0], axis_out["eta2_4c_real_ci"][1],
    axis_out["diff_4c_real_minus_sim"], axis_out["diff_4c_ci"][0], axis_out["diff_4c_ci"][1]))

log("")
log("2-CLASS (AS+AL vs IL+INF) eta2, same layout (sorted by eta2_real desc)")
order2 = np.argsort(-e_r2)
for j in order2:
    log("%-18s %8.4f [%7.4f,%7.4f] %8.4f [%7.4f,%7.4f] %+9.4f [%+7.4f,%+7.4f]" % (
        NAMES[j], e_s2[j], ci_s2[0][j], ci_s2[1][j],
        e_r2[j], ci_r2[0][j], ci_r2[1][j],
        e_r2[j] - e_s2[j], ci_d2[0][j], ci_d2[1][j]))
log("%-18s %8.4f %19s %8.4f %19s %+9.4f [%+7.4f,%+7.4f]" % (
    "AXIS(circ)", axis_out["eta2_2c_sim"], "", axis_out["eta2_2c_real"], "",
    axis_out["eta2_2c_real"] - axis_out["eta2_2c_sim"],
    axis_out["diff_2c_ci"][0], axis_out["diff_2c_ci"][1]))

log("")
log("=" * 100)
log("(c) MARGINAL REALISM (SMD = (mean_real-mean_sim)/pooled_sd; ratio = sd_real/sd_sim)")
log("%-18s %10s %10s %10s %10s %8s %8s" % ("feature", "mean_sim", "mean_real", "sd_sim", "sd_real", "SMD", "sdratio"))
for j in range(54):
    log("%-18s %10.4f %10.4f %10.4f %10.4f %+8.3f %8.3f" % (
        NAMES[j], mean_s[j], mean_r[j], sd_s[j], sd_r[j], smd[j], sd_ratio[j]))
log("AXIS(circ): mean sim %.1f deg / real %.1f deg (diff %.1f deg); circ SD sim %.1f deg / real %.1f deg"
    % (ax_mu_s * 180 / np.pi, ax_mu_r * 180 / np.pi, ax_dmu, ax_sd_s, ax_sd_r))
log("")
log("Correlation across 54 features: |SMD| vs |eta2_real - eta2_sim| (4c):")
log("  Spearman rho = %+.4f (p=%.4f);  Pearson r = %+.4f (p=%.4f)" % (c_sp, c_sp_p, c_pe, c_pe_p))
log("  |log sd-ratio| vs |gap|: Spearman rho = %+.4f (p=%.4f)" % (c_sd_sp, c_sd_sp_p))

log("")
log("=" * 100)
log("(d) RANK-ORDER AGREEMENT (headline)")
log("  Spearman rho(eta2_sim, eta2_real) over 54 features, 4-class: %+.4f (p=%.2e), boot 95%% CI [%+.4f, %+.4f]"
    % (rho54_4, rho54_4p, np.nanpercentile(rho_boot4, 2.5), np.nanpercentile(rho_boot4, 97.5)))
log("  2-class: %+.4f (p=%.2e), CI [%+.4f, %+.4f]"
    % (rho54_2, rho54_2p, np.nanpercentile(rho_boot2, 2.5), np.nanpercentile(rho_boot2, 97.5)))
log("  55 features incl. axis, 4-class: %+.4f (p=%.2e), CI [%+.4f, %+.4f]"
    % (rho55_4, rho55_4p, np.nanpercentile(rho_boot55, 2.5), np.nanpercentile(rho_boot55, 97.5)))

log("")
log("=" * 100)
log("(e) NAMED LISTS (4-class; CI-backed via bootstrap diff CI excluding 0)")
blind = [j for j in range(54) if ci_d4[0][j] > 0]
blind.sort(key=lambda j: -ci_d4[0][j])
spur = [j for j in range(54) if ci_d4[1][j] < 0]
spur.sort(key=lambda j: ci_d4[1][j])
log("BLIND SPOTS (real-informative, sim-flat; diff CI lower bound > 0), sorted by CI lower bound:")
for j in blind:
    log("  %-18s e2_sim %.4f  e2_real %.4f  diff %+0.4f CI [%+0.4f, %+0.4f]" % (
        NAMES[j], e_s4[j], e_r4[j], e_r4[j] - e_s4[j], ci_d4[0][j], ci_d4[1][j]))
if axis_out["diff_4c_ci"][0] > 0:
    log("  %-18s e2_sim %.4f  e2_real %.4f  diff %+0.4f CI [%+0.4f, %+0.4f]  <- circular" % (
        "AXIS", axis_out["eta2_4c_sim"], axis_out["eta2_4c_real"],
        axis_out["diff_4c_real_minus_sim"], axis_out["diff_4c_ci"][0], axis_out["diff_4c_ci"][1]))
log("SPURIOUS CHANNELS (sim-informative, real-flat; diff CI upper bound < 0), sorted by CI upper bound:")
for j in spur:
    log("  %-18s e2_sim %.4f  e2_real %.4f  diff %+0.4f CI [%+0.4f, %+0.4f]" % (
        NAMES[j], e_s4[j], e_r4[j], e_r4[j] - e_s4[j], ci_d4[0][j], ci_d4[1][j]))

result["lists"] = {
    "blind_spots_4c": [NAMES[j] for j in blind] + (["frontal_QRS_axis"] if axis_out["diff_4c_ci"][0] > 0 else []),
    "spurious_channels_4c": [NAMES[j] for j in spur],
}

# prediction check vs probe map (CTX item 4)
log("")
log("PREDICTION CHECK vs established probe-map result:")
pred_blind = [n for n in result["lists"]["blind_spots_4c"] if n.startswith(("Q_amp", "R_amp")) or n == "frontal_QRS_axis"]
pred_spur = [n for n in result["lists"]["spurious_channels_4c"] if n.startswith("ST_J60")]
other_blind = [n for n in result["lists"]["blind_spots_4c"] if n not in pred_blind]
other_spur = [n for n in result["lists"]["spurious_channels_4c"] if n not in pred_spur]
contra_blind = [n for n in result["lists"]["blind_spots_4c"] if n.startswith("ST_J60")]
contra_spur = [n for n in result["lists"]["spurious_channels_4c"] if n.startswith(("Q_amp", "R_amp"))]
log("  Predicted blind spots found (Q/R/axis): %s" % pred_blind)
log("  Predicted spurious found (ST_J60*): %s" % pred_spur)
log("  Unpredicted blind spots: %s" % other_blind)
log("  Unpredicted spurious: %s" % other_spur)
log("  CONTRADICTIONS (ST in blind spots / Q,R in spurious): %s / %s" % (contra_blind, contra_spur))
result["prediction_check"] = {
    "predicted_blind_found": pred_blind, "predicted_spurious_found": pred_spur,
    "unpredicted_blind": other_blind, "unpredicted_spurious": other_spur,
    "contradictions_blind": contra_blind, "contradictions_spurious": contra_spur,
}

log("")
log("=" * 100)
log("(f) MISSINGNESS AUDIT")
for dom in ("sim", "real"):
    n = miss[dom]["n"]
    fn = np.asarray(miss[dom]["finite_n"])
    ps = np.asarray(miss[dom]["chi2_p_by_territory"])
    log("%s: features fully finite: %d/54; features with any missingness:" % (dom, int((fn == n).sum())))
    for j in range(54):
        if fn[j] < n:
            pstr = ("chi2 p=%.2e" % ps[j]) if np.isfinite(ps[j]) else "chi2 n/a"
            log("   %-18s finite %5d/%5d (%.1f%%)  missing-x-territory %s" % (
                NAMES[j], fn[j], n, 100.0 * fn[j] / n, pstr))
    log("   all-54-finite by territory: %s  (chi2 p=%.2e)" % (
        miss[dom]["all54_by_territory"], miss[dom]["all54_chi2_p"]))
    log("   Holm-significant (p*<0.05) missing-x-territory features: %s" %
        [NAMES[j] for j in holm_sig[dom]])

flush_json(result)
flush_txt()
log("")
log("DONE. Wrote %s and %s" % (OUT_TXT.name, OUT_JSON.name))
flush_txt()
