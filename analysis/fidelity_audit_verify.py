# Promoted 2026-08-13 from reports/2026-08-13_audit_artifacts/scripts/tmp_v_f1_fidelity_verify.py
# Adversarial verifier script (independent re-implementation; do not 'fix'
# to agree with the primary -- its value is that it shares no code with it).
# tmp_v_f1_fidelity_verify.py - adversarial verification of the F1 fidelity audit.
# Independent re-implementation (different code path) + block-aware nulls +
# trivial block-mean competitor + bootstrap spot checks + claim audits.
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
OUT = REPO / "outputs/analysis/fidelity_audit/f1_verify_out.txt"
lines = []


def log(s=""):
    print(s)
    lines.append(s)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


TERR = ["Anteroseptal", "Anterolateral", "Inferolateral", "Inferior"]

# ---------------- load (own code, same verified keying) ----------------
fe_r = np.load(REPO / "data/ecg_features_spatial_ptbxl_allfolds.npz", allow_pickle=True)
NAMES = [str(x) for x in fe_r["feature_names"]]
mi = pd.read_csv(REPO / "data/ptbxl_mi_subclass_allfolds.csv")
keep = mi["territory_4c"].notna().to_numpy()
Fr = fe_r["features"][keep]
yr = np.array([TERR.index(t) for t in mi.loc[keep, "territory_4c"]])
gr = mi.loc[keep, "patient_id"].to_numpy().astype(str)

Fs_parts, ys_parts, gs_parts = [], [], []
for split in ("train", "test"):
    t = np.load(REPO / f"data/theta_mi_{split}.npz", allow_pickle=True)
    fe = np.load(REPO / f"data/ecg_features_spatial_medalcare_{split}.npz", allow_pickle=True)
    assert [str(x) for x in fe["feature_names"]] == NAMES
    idx = np.asarray(t["idx_in_split"], dtype=int)
    Fs_parts.append(fe["features"][idx])
    ys_parts.append(np.array([TERR.index(str(x)) for x in t["territory_4c"]]))
    gs_parts.append(np.array([f"{split}:{r}" for r in t["run_id"]]))
Fs = np.vstack(Fs_parts)
ys = np.concatenate(ys_parts)
gs = np.concatenate(gs_parts)

log("V-F1 verifier. sim n=%d (%d groups), real n=%d (%d groups)"
    % (len(ys), len(np.unique(gs)), len(yr), len(np.unique(gr))))
log("sim class counts: %s" % np.bincount(ys, minlength=4).tolist())
log("real class counts: %s" % np.bincount(yr, minlength=4).tolist())
log("(their all54 denominators: sim [2199,1600,550,2198], real [1624,455,331,1914])")

# ---------------- independent eta2 (plain loop) ----------------
def my_eta2(x, y, k):
    m = np.isfinite(x)
    xv, yv = x[m], y[m]
    if len(xv) == 0:
        return np.nan
    gm = xv.mean()
    sst = float(((xv - gm) ** 2).sum())
    if sst <= 0:
        return np.nan
    ssb = 0.0
    for c in range(k):
        sel = xv[yv == c]
        if len(sel):
            ssb += len(sel) * (sel.mean() - gm) ** 2
    return ssb / sst


def my_circ_eta2(a, y, k):
    m = np.isfinite(a)
    z = np.exp(1j * a[m])
    yv = y[m]
    N = float(len(z))
    R = abs(z.sum())
    s = sum(abs(z[yv == c].sum()) for c in range(k))
    return (s - R) / (N - R)


e_s4 = np.array([my_eta2(Fs[:, j], ys, 4) for j in range(54)])
e_r4 = np.array([my_eta2(Fr[:, j], yr, 4) for j in range(54)])
y2s = (ys >= 2).astype(int)
y2r = (yr >= 2).astype(int)
e_s2 = np.array([my_eta2(Fs[:, j], y2s, 2) for j in range(54)])
e_r2 = np.array([my_eta2(Fr[:, j], y2r, 2) for j in range(54)])

iavf, ii = NAMES.index("R_amp_aVF"), NAMES.index("R_amp_I")
ax_s = np.arctan2(Fs[:, iavf], Fs[:, ii])
ax_r = np.arctan2(Fr[:, iavf], Fr[:, ii])
ax_e_s4 = my_circ_eta2(ax_s, ys, 4)
ax_e_r4 = my_circ_eta2(ax_r, yr, 4)
ax_e_s2 = my_circ_eta2(ax_s, y2s, 2)
ax_e_r2 = my_circ_eta2(ax_r, y2r, 2)

# compare against their JSON
J = json.loads((REPO / "outputs/analysis/fidelity_audit/f1_fidelity.json").read_text(encoding="utf-8"))
their = {f["name"]: f for f in J["features"]}
d4s = max(abs(e_s4[j] - their[NAMES[j]]["eta2_4c_sim"]) for j in range(54))
d4r = max(abs(e_r4[j] - their[NAMES[j]]["eta2_4c_real"]) for j in range(54))
d2s = max(abs(e_s2[j] - their[NAMES[j]]["eta2_2c_sim"]) for j in range(54))
d2r = max(abs(e_r2[j] - their[NAMES[j]]["eta2_2c_real"]) for j in range(54))
log("")
log("ETA2 REPRODUCTION (my loop implementation vs their JSON):")
log("  max|diff| eta2_4c sim %.3e  real %.3e ; eta2_2c sim %.3e  real %.3e"
    % (d4s, d4r, d2s, d2r))
log("  axis circ eta2 4c: mine sim %.6f real %.6f | theirs %.6f / %.6f"
    % (ax_e_s4, ax_e_r4, J["axis"]["eta2_4c_sim"], J["axis"]["eta2_4c_real"]))
log("  axis circ eta2 2c: mine sim %.6f real %.6f | theirs %.6f / %.6f"
    % (ax_e_s2, ax_e_r2, J["axis"]["eta2_2c_sim"], J["axis"]["eta2_2c_real"]))

# ---------------- headline rho reproduction ----------------
rho4, p4 = sps.spearmanr(e_s4, e_r4)
rho2, p2 = sps.spearmanr(e_s2, e_r2)
rho55, p55 = sps.spearmanr(np.append(e_s4, ax_e_s4), np.append(e_r4, ax_e_r4))
log("")
log("HEADLINE RHO REPRODUCTION:")
log("  4c 54feat: mine %.4f (p=%.2e) vs theirs -0.3544 (8.55e-03)" % (rho4, p4))
log("  2c 54feat: mine %.4f (p=%.2e) vs theirs -0.3155 (2.01e-02)" % (rho2, p2))
log("  4c 55 w/axis: mine %.4f (p=%.2e) vs theirs -0.3796 (4.26e-03)" % (rho55, p55))

# duplicate column dropped (53 distinct)
jdup = NAMES.index("T_amplitude_mV")
keep53 = [j for j in range(54) if j != jdup]
rho53, p53 = sps.spearmanr(e_s4[keep53], e_r4[keep53])
log("  4c 53 distinct (drop dup T_amplitude_mV): %.4f (p=%.2e)" % (rho53, p53))

# ---------------- block structure of the correlation ----------------
BLOCKS = {"ST": list(range(0, 12)), "Q": list(range(12, 24)),
          "R": list(range(24, 36)), "T": list(range(36, 48)),
          "GLOB": list(range(48, 54))}
log("")
log("BLOCK DECOMPOSITION OF THE HEADLINE (4c):")
log("  block   mean_e2_sim  mean_e2_real  within-block Spearman (n)   p")
for bn, ix in BLOCKS.items():
    r, p = sps.spearmanr(e_s4[ix], e_r4[ix])
    log("  %-5s   %10.4f  %11.4f   %+8.4f (%d)          %.3f"
        % (bn, e_s4[ix].mean(), e_r4[ix].mean(), r, len(ix), p))

# trivial competitor: replace sim eta2 by its block mean (block-level info only)
sim_blockmean = np.empty(54)
for bn, ix in BLOCKS.items():
    sim_blockmean[ix] = e_s4[ix].mean()
rho_bm, p_bm = sps.spearmanr(sim_blockmean, e_r4)
log("  TRIVIAL COMPETITOR rho(block-mean-only sim, real) = %.4f (naive p=%.2e)"
    % (rho_bm, p_bm))
log("  -> fraction of headline |rho| reproduced by block identity alone: %.2f"
    % (abs(rho_bm) / abs(rho4)))
# and the reverse: within-block residual correlation (block effect removed both sides)
res_s = e_s4.copy()
res_r = e_r4.copy()
for bn, ix in BLOCKS.items():
    res_s[ix] -= e_s4[ix].mean()
    res_r[ix] -= e_r4[ix].mean()
rho_res, p_res = sps.spearmanr(res_s, res_r)
log("  block-centered residual rho = %.4f (naive p=%.3f)" % (rho_res, p_res))

# ---------------- block-permutation nulls for the headline ----------------
rng = np.random.default_rng(777)
lead_blocks = [BLOCKS["ST"], BLOCKS["Q"], BLOCKS["R"], BLOCKS["T"]]
glob_ix = BLOCKS["GLOB"]

def perm_rho(n_iter, permute_within):
    vals = np.empty(n_iter)
    for it in range(n_iter):
        sim_null = np.empty(54)
        order = rng.permutation(4)
        for slot, src in enumerate(order):
            src_vals = e_s4[lead_blocks[src]]
            if permute_within:
                src_vals = src_vals[rng.permutation(12)]
            sim_null[lead_blocks[slot]] = src_vals
        sim_null[glob_ix] = e_s4[glob_ix][rng.permutation(6)]
        vals[it] = sps.spearmanr(sim_null, e_r4)[0]
    return vals

N_PERM = 20000
null_block = perm_rho(N_PERM, permute_within=False)
null_full = perm_rho(N_PERM, permute_within=True)
p_block = float((np.sum(null_block <= rho4) + 1) / (N_PERM + 1))
p_full = float((np.sum(null_full <= rho4) + 1) / (N_PERM + 1))
log("")
log("BLOCK-AWARE PERMUTATION NULLS for headline rho=%.4f (one-sided, rho<=obs):" % rho4)
log("  N1 block-slot permutation only (within-block order intact): p = %.4f" % p_block)
log("     null mean %.3f, sd %.3f, 2.5/97.5 pct [%.3f, %.3f]"
    % (null_block.mean(), null_block.std(), np.percentile(null_block, 2.5),
       np.percentile(null_block, 97.5)))
log("  N2 block-slot + within-block lead permutation: p = %.4f" % p_full)
log("     null mean %.3f, sd %.3f, 2.5/97.5 pct [%.3f, %.3f]"
    % (null_full.mean(), null_full.std(), np.percentile(null_full, 2.5),
       np.percentile(null_full, 97.5)))

# ---------------- my own block bootstrap (different seed/code) ----------------
def vec_eta2(F, y, rows, k):
    """Vectorised eta2 over all 54 features on (possibly duplicated) rows."""
    Fv = F[rows]
    yv = y[rows]
    fin = np.isfinite(Fv)
    X = np.where(fin, Fv, 0.0)
    X2 = np.where(fin, Fv * Fv, 0.0)
    n_t = fin.sum(0).astype(float)
    s_t = X.sum(0)
    ss_t = X2.sum(0)
    gm = s_t / np.maximum(n_t, 1)
    sst = ss_t - n_t * gm ** 2
    ssb = np.zeros(54)
    for c in range(k):
        m = yv == c
        n_c = fin[m].sum(0).astype(float)
        mc = np.where(n_c > 0, X[m].sum(0) / np.maximum(n_c, 1), 0.0)
        ssb += n_c * (mc - gm) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sst > 0, ssb / sst, np.nan)


def make_lut(g):
    uniq = np.unique(g)
    lut = {}
    for i, v in enumerate(g):
        lut.setdefault(v, []).append(i)
    return uniq, {k: np.asarray(v) for k, v in lut.items()}


u_s, lut_s = make_lut(gs)
u_r, lut_r = make_lut(gr)
NB = 400
rng2 = np.random.default_rng(424242)
rho_draws = np.empty(NB)
diff_draws = np.empty((NB, 54))
ax_diff_draws = np.empty(NB)
d2_rv6 = np.empty(NB)   # R_amp_V6 2-class diff
j_rv6 = NAMES.index("R_amp_V6")
for b in range(NB):
    rs = np.concatenate([lut_s[g] for g in rng2.choice(u_s, size=len(u_s), replace=True)])
    rr = np.concatenate([lut_r[g] for g in rng2.choice(u_r, size=len(u_r), replace=True)])
    es = vec_eta2(Fs, ys, rs, 4)
    er = vec_eta2(Fr, yr, rr, 4)
    rho_draws[b] = sps.spearmanr(es, er)[0]
    diff_draws[b] = er - es
    a_s = my_circ_eta2(ax_s[rs], ys[rs], 4)
    a_r = my_circ_eta2(ax_r[rr], yr[rr], 4)
    ax_diff_draws[b] = a_r - a_s
    es2 = vec_eta2(Fs[:, [j_rv6]] if False else Fs, y2s, rs, 2)
    er2 = vec_eta2(Fr, y2r, rr, 2)
    d2_rv6[b] = er2[j_rv6] - es2[j_rv6]

lo_rho, hi_rho = np.percentile(rho_draws, [2.5, 97.5])
log("")
log("MY BLOCK BOOTSTRAP (400 draws, seed 424242, own code):")
log("  rho4 95%% CI [%.4f, %.4f]  (theirs [-0.4395, -0.2478])" % (lo_rho, hi_rho))
for nm in ("Q_amp_III", "ST_J60_aVF", "R_amp_V6"):
    j = NAMES.index(nm)
    lo, hi = np.percentile(diff_draws[:, j], [2.5, 97.5])
    log("  diff4c %-12s [%.4f, %.4f]  (theirs %s)"
        % (nm, lo, hi, their[nm]["diff_4c_ci"]))
lo, hi = np.percentile(ax_diff_draws, [2.5, 97.5])
log("  axis diff4c [%.4f, %.4f]  (theirs %s)" % (lo, hi, J["axis"]["diff_4c_ci"]))
lo, hi = np.percentile(d2_rv6, [2.5, 97.5])
log("  R_amp_V6 diff2c [%.4f, %.4f]  (theirs %s)"
    % (lo, hi, their["R_amp_V6"]["diff_2c_ci"]))

# ---------------- claim audits from their own JSON ----------------
log("")
log("CLAIM AUDITS (their JSON, their numbers):")
blind = [f["name"] for f in J["features"] if f["diff_4c_ci"][0] > 0]
spur = [f["name"] for f in J["features"] if f["diff_4c_ci"][1] < 0]
log("  count blind-spot features (diff4c CI_lo>0): %d  (+axis %s); report said '29 features + axis'"
    % (len(blind), J["axis"]["diff_4c_ci"][0] > 0))
log("  count spurious (CI_hi<0): %d ; report said 14" % len(spur))
st_spur = [n for n in spur if n.startswith("ST_J60") and n != "ST_J60_avg_mV"]
log("  ST_J60 leads in spurious list: %d -> %s" % (len(st_spur), st_spur))
log("  headline said '8 of 12 ST_J60 leads'; numbers section said 9.")
qr_blind = [n for n in blind if n.startswith(("Q_amp", "R_amp"))]
log("  Q/R features in blind list: %d of 24" % len(qr_blind))
qr_missing = [n for n in NAMES if n.startswith(("Q_amp", "R_amp")) and n not in blind]
log("  Q/R NOT CI-confirmed blind: %s" % qr_missing)

# sign-agreement claim: '2c sign agrees with 4c for every CI-significant feature'
viol_both, viol_4sig = [], []
for f in J["features"]:
    sig4 = f["diff_4c_ci"][0] > 0 or f["diff_4c_ci"][1] < 0
    sig2 = f["diff_2c_ci"][0] > 0 or f["diff_2c_ci"][1] < 0
    s4 = np.sign(f["diff_4c_real_minus_sim"])
    s2 = np.sign(f["diff_2c_real_minus_sim"])
    if sig4 and sig2 and s4 != s2:
        viol_both.append((f["name"], f["diff_4c_real_minus_sim"], f["diff_4c_ci"],
                          f["diff_2c_real_minus_sim"], f["diff_2c_ci"]))
    if sig4 and s4 != s2:
        viol_4sig.append(f["name"])
log("  SIGN-AGREEMENT CLAIM: violations with BOTH CIs significant: %d" % len(viol_both))
for v in viol_both:
    log("    %s: diff4c %+0.4f CI %s  vs diff2c %+0.4f CI %s" % v)
log("  violations where 4c is significant (any 2c): %s" % viol_4sig)

# ---------------- |SMD| vs |gap| sensitivity ----------------
mean_s = np.nanmean(Fs, axis=0)
mean_r = np.nanmean(Fr, axis=0)
sd_s = np.nanstd(Fs, axis=0)
sd_r = np.nanstd(Fr, axis=0)
pooled = np.sqrt((sd_s ** 2 + sd_r ** 2) / 2)
smd = (mean_r - mean_s) / pooled
gap = np.abs(e_r4 - e_s4)
asmd = np.abs(smd)
r_all, p_all = sps.spearmanr(asmd, gap)
log("")
log("MARGINAL-REALISM CLAIM (c) SENSITIVITY:")
log("  all 54: rho %.4f p %.4f (theirs -0.2840 / 0.0374)" % (r_all, p_all))
noT = [j for j in range(54) if not NAMES[j].startswith("T_amp") and NAMES[j] != "T_amplitude_mV"]
r_noT, p_noT = sps.spearmanr(asmd[noT], gap[noT])
log("  excluding T-wave block+dup (n=%d): rho %.4f p %.4f" % (len(noT), r_noT, p_noT))
r_53, p_53 = sps.spearmanr(asmd[keep53], gap[keep53])
log("  53 distinct (drop dup): rho %.4f p %.4f" % (r_53, p_53))
# block-permutation null for the -0.284 too
def perm_rho_c(n_iter):
    vals = np.empty(n_iter)
    for it in range(n_iter):
        a_null = np.empty(54)
        order = rng.permutation(4)
        for slot, src in enumerate(order):
            a_null[lead_blocks[slot]] = asmd[lead_blocks[src]][rng.permutation(12)]
        a_null[glob_ix] = asmd[glob_ix][rng.permutation(6)]
        vals[it] = sps.spearmanr(a_null, gap)[0]
    return vals

null_c = perm_rho_c(N_PERM)
p_c = float((np.sum(null_c <= r_all) + 1) / (N_PERM + 1))
log("  block-permutation null p (one-sided) for rho=-0.284: %.4f" % p_c)

# ---------------- axis marginal check vs established ----------------
def circ_stats(a):
    m = np.isfinite(a)
    z = np.exp(1j * a[m]).mean()
    mu = np.degrees(np.angle(z))
    sd = np.degrees(np.sqrt(-2 * np.log(abs(z))))
    return mu, sd

mu_s, sd_s_ax = circ_stats(ax_s)
mu_r, sd_r_ax = circ_stats(ax_r)
log("")
log("AXIS MARGINALS: sim mean %.1f deg SD %.1f; real mean %.1f deg SD %.1f"
    % (mu_s, sd_s_ax, mu_r, sd_r_ax))
log("(established: SD 41.3 vs 37.9; their report: means 68.0 / 31.3)")

log("")
log("DONE verifier stage 1.")
