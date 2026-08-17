# Promoted 2026-08-13 from reports/2026-08-13_audit_artifacts/scripts/tmp_v_f2_verify.py
# Adversarial verifier script (independent re-implementation; do not 'fix'
# to agree with the primary -- its value is that it shares no code with it).
# -*- coding: utf-8 -*-
"""Adversarial verification of F2-block-transfer.

Independent re-implementation: eigh-based GCV ridge (not RidgeSVD's SVD path),
own imputation / nearest-anchor / macro-F1 / eta2 / CORAL / efficiency code,
exact permutation p for the block-level Spearman, killer competitors
(eta2_real predictor, real-fit axis2, constant), fold-seed robustness,
independent paired test re-run with a different seed.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))
from geom_common import TERRITORIES, group_folds, medalcare_anchor_angles

OUT = REPO / "outputs/analysis/fidelity_audit/f2_verify_out.txt"
JOUT = REPO / "outputs/analysis/fidelity_audit/f2_verify.json"
RES = {}


def log(msg=""):
    print(msg)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def save():
    with open(JOUT, "w", encoding="utf-8") as f:
        json.dump(RES, f, indent=2, default=float)


OUT.write_text("", encoding="utf-8")
log("ADVERSARIAL VERIFICATION: F2-block-transfer")
log("=" * 78)

# ------------------------------------------------------------------ load data
fe_p = np.load(REPO / "data/ecg_features_spatial_ptbxl_allfolds.npz", allow_pickle=True)
NAMES = [str(x) for x in fe_p["feature_names"]]
F_all = np.asarray(fe_p["features"], dtype=float)
mi = pd.read_csv(REPO / "data/ptbxl_mi_subclass_allfolds.csv")
pos = np.flatnonzero(mi["territory_4c"].notna().to_numpy())
Fp = F_all[pos]
terr_p = mi["territory_4c"].to_numpy()[pos]
group_p = mi["patient_id"].to_numpy().astype(str)[pos]

Fm_l, phi_l, terr_l, grp_l = [], [], [], []
for split in ("train", "test"):
    t = np.load(REPO / f"data/theta_mi_{split}.npz", allow_pickle=True)
    fm = np.load(REPO / f"data/ecg_features_spatial_medalcare_{split}.npz",
                 allow_pickle=True)
    assert [str(x) for x in fm["feature_names"]] == NAMES
    idx = np.asarray(t["idx_in_split"], dtype=int)
    Fm_l.append(np.asarray(fm["features"], dtype=float)[idx])
    phi_l.append(np.asarray(t["phi"], dtype=float))
    terr_l.append(np.asarray(t["territory_4c"]))
    grp_l.append(np.array([f"{split}:{r}" for r in t["run_id"]]))
Fm = np.vstack(Fm_l)
phi_m = np.concatenate(phi_l)
terr_m = np.concatenate(terr_l)
group_m = np.concatenate(grp_l)

ANCH = medalcare_anchor_angles()
anchor_arr = np.array([ANCH[t] for t in TERRITORIES])
angle_p = np.array([ANCH[t] for t in terr_p])
Y_m = np.column_stack([np.cos(phi_m), np.sin(phi_m)])

log(f"n MedalCare {len(phi_m)} (expect 6547), n PTB-XL {len(terr_p)} (expect 4324)")

# alignment sanity: territory_4c should be a deterministic function of phi
# (nearest anchor of phi should equal territory label for ~all rows)
d = np.abs(np.angle(np.exp(1j * (phi_m[:, None] - anchor_arr[None, :]))))
near = np.array(TERRITORIES)[np.argmin(d, axis=1)]
agree = float((near == terr_m).mean())
log(f"MedalCare phi-vs-territory nearest-anchor agreement: {agree:.4f}")
RES["phi_territory_agreement"] = agree

BLOCKS = {
    "ST_J60": [i for i, n in enumerate(NAMES) if n.startswith("ST_J60_") and n != "ST_J60_avg_mV"],
    "Q_amp": [i for i, n in enumerate(NAMES) if n.startswith("Q_amp_")],
    "R_amp": [i for i, n in enumerate(NAMES) if n.startswith("R_amp_")],
    "T_amp": [i for i, n in enumerate(NAMES) if n.startswith("T_amp_")],
    "globals": [48, 49, 50, 51, 52, 53],
    "full54": list(range(54)),
    "axis2": [NAMES.index("R_amp_I"), NAMES.index("R_amp_aVF")],
}
used = sorted(set(BLOCKS["ST_J60"] + BLOCKS["Q_amp"] + BLOCKS["R_amp"]
                  + BLOCKS["T_amp"] + BLOCKS["globals"]))
log(f"block partition covers {len(used)}/54 features, sizes: "
    + str({k: len(v) for k, v in BLOCKS.items()}))
assert used == list(range(54))

# --------------------------------------------------------- independent ridge
ALPHAS = np.logspace(-2, 5, 24)


class EighRidge:
    """GCV ridge via eigendecomposition of the Gram matrix (independent path)."""

    def fit(self, X, Y):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        self.xm = Xs.mean(0)
        Xc = Xs - self.xm
        n = Xc.shape[0]
        C = Xc.T @ Xc
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 0.0, None)
        Yc = Y - Y.mean(0)
        XtY = Xc.T @ Yc
        best, besta, bestcoef = np.inf, None, None
        for a in ALPHAS:
            coef = V @ ((V.T @ XtY) / (w[:, None] + a))
            resid = Yc - Xc @ coef
            dof = float((w / (w + a)).sum()) + 1.0
            denom = max(1.0 - dof / n, 1e-6)
            gcv = float((resid ** 2).sum()) / n / denom ** 2
            if gcv < best:
                best, besta, bestcoef = gcv, float(a), coef
        self.alpha = besta
        self.coef = bestcoef
        self.ym = Y.mean(0)
        return self

    def predict(self, X):
        return ((X - self.mu) / self.sd - self.xm) @ self.coef + self.ym


def imp(X, med):
    return np.where(np.isnan(X), med[None, :], X)


def ang_of(pred):
    return np.arctan2(pred[:, 1], pred[:, 0])


T2I = {t: i for i, t in enumerate(TERRITORIES)}


def near_idx(ang):
    d = np.abs(np.angle(np.exp(1j * (ang[:, None] - anchor_arr[None, :]))))
    return np.argmin(d, axis=1)


def my_macro_f1(true_i, pred_i):
    f1s = []
    for k in range(4):
        tp = float(np.sum((true_i == k) & (pred_i == k)))
        fp = float(np.sum((true_i != k) & (pred_i == k)))
        fn = float(np.sum((true_i == k) & (pred_i != k)))
        f1s.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def my_eta2(ang, classes):
    z = np.exp(1j * ang)
    R = np.abs(z.sum())
    sumRj = sum(np.abs(z[classes == c].sum()) for c in np.unique(classes))
    return float((sumRj - R) / (len(ang) - R))


terr_p_i = np.array([T2I[t] for t in terr_p])
terr_m_i = np.array([T2I[t] for t in terr_m])

# floors
R_floor_p = float(np.abs(np.mean(np.exp(-1j * angle_p))))
R_floor_m = float(np.abs(np.mean(np.exp(-1j * phi_m))))
constF1_p = max(my_macro_f1(terr_p_i, np.full(len(terr_p_i), k)) for k in range(4))
constF1_m = max(my_macro_f1(terr_m_i, np.full(len(terr_m_i), k)) for k in range(4))
log(f"floors (independent): R_p {R_floor_p:.5f} R_m {R_floor_m:.5f} "
    f"F1_p {constF1_p:.4f} F1_m {constF1_m:.4f}")
RES["floors"] = dict(R_p=R_floor_p, R_m=R_floor_m, F1_p=constF1_p, F1_m=constF1_m)
save()

# ------------------------------------------------- per-block replication
def run_blocks(seed):
    rng = np.random.default_rng(seed)
    folds = list(group_folds(group_m, 5, rng))
    out = {}
    for bname, cols in BLOCKS.items():
        cols = np.asarray(cols)
        Xm, Xp = Fm[:, cols], Fp[:, cols]
        pred = np.full(len(phi_m), np.nan)
        for tr, te in folds:
            med = np.nanmedian(Xm[tr], axis=0)
            m = EighRidge().fit(imp(Xm[tr], med), Y_m[tr])
            pred[te] = ang_of(m.predict(imp(Xm[te], med)))
        f1_in = my_macro_f1(terr_m_i, near_idx(pred))
        eta2_in = my_eta2(pred, terr_m)
        # transport
        med_m = np.nanmedian(Xm, axis=0)
        m = EighRidge().fit(imp(Xm, med_m), Y_m)
        a_src = ang_of(m.predict(imp(Xp, med_m)))
        med_p = np.nanmedian(Xp, axis=0)
        Xpt = imp(Xp, med_p)
        zt = (Xpt - Xpt.mean(0)) / (Xpt.std(0) + 1e-8)
        a_tgt = ang_of(m.predict(zt * m.sd + m.mu))
        row = dict(alpha=m.alpha, f1_in=f1_in, eta2_in=eta2_in,
                   f1_src=my_macro_f1(terr_p_i, near_idx(a_src)),
                   f1_tgt=my_macro_f1(terr_p_i, near_idx(a_tgt)),
                   eta2_src=my_eta2(a_src, terr_p), eta2_tgt=my_eta2(a_tgt, terr_p),
                   R_in=float(np.abs(np.mean(np.exp(1j * (pred - phi_m))))),
                   R_src=float(np.abs(np.mean(np.exp(1j * (a_src - angle_p))))),
                   R_tgt=float(np.abs(np.mean(np.exp(1j * (a_tgt - angle_p))))))
        row["eff_src"] = (row["f1_src"] - constF1_p) / (f1_in - constF1_m)
        row["eff_tgt"] = (row["f1_tgt"] - constF1_p) / (f1_in - constF1_m)
        out[bname] = row
        out[bname + "_angles"] = dict(src=a_src, tgt=a_tgt)
    return out


log("")
log("Per-block replication (seed 0 folds, eigh ridge, own metrics)")
log(f"{'block':8s} {'alpha':>8s} {'F1_in':>7s} {'F1_src':>7s} {'F1_tgt':>7s} "
    f"{'eff_src':>8s} {'eff_tgt':>8s}")
rep = run_blocks(0)
theirs = json.load(open(REPO / "outputs/analysis/fidelity_audit/f2_blocks.json"))["per_block"]
maxdev = 0.0
for b in BLOCKS:
    r = rep[b]
    log(f"{b:8s} {r['alpha']:8.4g} {r['f1_in']:7.4f} {r['f1_src']:7.4f} "
        f"{r['f1_tgt']:7.4f} {r['eff_src']:8.3f} {r['eff_tgt']:8.3f}")
    t = theirs[b]
    devs = [abs(r["f1_in"] - t["in"]["macro_f1"]),
            abs(r["f1_src"] - t["cross_source"]["macro_f1"]),
            abs(r["f1_tgt"] - t["cross_target"]["macro_f1"])]
    maxdev = max(maxdev, max(devs))
log(f"max |F1 dev| vs their reported per-block numbers: {maxdev:.6f}")
RES["replication"] = {b: {k: v for k, v in rep[b].items()} for b in BLOCKS}
RES["max_f1_dev"] = maxdev
save()

# ---------------------------------------------- eta2_sim features + Spearman
def anova_eta2(x, classes):
    fin = np.isfinite(x)
    x, c = x[fin], classes[fin]
    gm = x.mean()
    ssb = sum((c == k).sum() * (x[c == k].mean() - gm) ** 2 for k in np.unique(c))
    sst = ((x - gm) ** 2).sum()
    return float(ssb / sst) if sst > 0 else np.nan


e_sim = np.array([anova_eta2(Fm[:, j], terr_m) for j in range(54)])
e_real = np.array([anova_eta2(Fp[:, j], terr_p) for j in range(54)])
bsim = {b: float(np.nanmean(e_sim[np.asarray(c)])) for b, c in BLOCKS.items()}
breal = {b: float(np.nanmean(e_real[np.asarray(c)])) for b, c in BLOCKS.items()}
log("")
log("block eta2 means (sim | real), independent ANOVA code:")
for b in ("ST_J60", "Q_amp", "R_amp", "T_amp", "globals", "axis2"):
    log(f"  {b:8s} {bsim[b]:.4f} | {breal[b]:.4f}")
RES["block_eta2"] = dict(sim=bsim, real=breal)


def rankvec(v):
    v = np.asarray(v, dtype=float)
    order = np.argsort(v)
    r = np.empty(len(v))
    r[order] = np.arange(1, len(v) + 1)
    return r


def spear(x, y):
    rx, ry = rankvec(x), rankvec(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def exact_spear_p(x, y):
    obs = spear(x, y)
    ry = rankvec(y)
    n = len(x)
    lo = hi = two = 0
    tot = 0
    for perm in itertools.permutations(range(n)):
        r = spear(x, ry[list(perm)])
        tot += 1
        if r <= obs + 1e-12:
            lo += 1
        if r >= obs - 1e-12:
            hi += 1
        if abs(r) >= abs(obs) - 1e-12:
            two += 1
    return obs, lo / tot, hi / tot, two / tot


nat5 = ["ST_J60", "Q_amp", "R_amp", "T_amp", "globals"]
nat6 = nat5 + ["axis2"]
log("")
log("DECISIVE STATISTIC (independent): Spearman rho(eta2_sim, efficiency)")
sp = {}
for label, bs in (("5blocks", nat5), ("6blocks", nat6)):
    for sc in ("src", "tgt"):
        x = [bsim[b] for b in bs]
        y = [rep[b][f"eff_{sc}"] for b in bs]
        obs, p_lo, p_hi, p_two = exact_spear_p(x, y)
        sp[f"{label}_{sc}"] = dict(rho=obs, p_onesided_neg=p_lo, p_twosided=p_two)
        log(f"  [{label} {sc}] rho={obs:+.3f}  exact perm p: one-sided(neg) "
            f"{p_lo:.4f}, two-sided {p_two:.4f}")
RES["spearman_exact"] = sp
save()

# competitors / decompositions of the headline correlation
log("")
log("Decomposition / competitors of the headline correlation (source scaler):")
x_sim = [bsim[b] for b in nat5]
x_real = [breal[b] for b in nat5]
y_eff = [rep[b]["eff_src"] for b in nat5]
y_f1x = [rep[b]["f1_src"] for b in nat5]
y_f1in = [rep[b]["f1_in"] for b in nat5]
for nm, x, y in (("eta2_sim vs eff_src", x_sim, y_eff),
                 ("eta2_sim vs raw F1_src (numerator only)", x_sim, y_f1x),
                 ("eta2_sim vs F1_in (denominator)", x_sim, y_f1in),
                 ("eta2_real vs eff_src (competitor)", x_real, y_eff),
                 ("eta2_real vs raw F1_src", x_real, y_f1x),
                 ("eta2_sim vs eta2_real (fidelity mismatch)", x_sim, x_real)):
    obs, p_lo, p_hi, p_two = exact_spear_p(x, y)
    log(f"  {nm:42s} rho={obs:+.3f} (exact 2-sided p={p_two:.4f})")
    RES.setdefault("decomposition", {})[nm] = dict(rho=obs, p_two=p_two)
save()

# ------------------------------------------------- fold-seed robustness
log("")
log("Fold-seed robustness (in-domain CV seed 7 instead of 0):")
rep7 = run_blocks(7)
for label, bs in (("5blocks", nat5), ("6blocks", nat6)):
    x = [bsim[b] for b in bs]
    y = [rep7[b]["eff_src"] for b in bs]
    obs, p_lo, p_hi, p_two = exact_spear_p(x, y)
    log(f"  [{label} src, seed7] rho={obs:+.3f} one-sided(neg) p={p_lo:.4f}")
    RES.setdefault("seed7", {})[label] = dict(rho=obs, p_one=p_lo)
log("  seed-7 F1_in per block: " + ", ".join(
    f"{b} {rep7[b]['f1_in']:.4f}" for b in nat6))
save()

# ------------------------------------------------- paired test re-run (independent)
def paired(true_i, groups, angA, angB, n_draws=2000, seed=42):
    pA, pB = near_idx(angA), near_idx(angB)
    dobs = my_macro_f1(true_i, pA) - my_macro_f1(true_i, pB)
    uniq = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in uniq}
    r = np.random.default_rng(seed)
    deltas = np.empty(n_draws)
    for i in range(n_draws):
        gs = r.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_of[g] for g in gs])
        deltas[i] = my_macro_f1(true_i[rows], pA[rows]) - my_macro_f1(true_i[rows], pB[rows])
    ci = (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))
    cnt = 0
    for i in range(n_draws):
        flip = r.random(len(uniq)) < 0.5
        pa, pb = pA.copy(), pB.copy()
        for g in uniq[flip]:
            ix = idx_of[g]
            pa[ix], pb[ix] = pB[ix], pA[ix]
        if abs(my_macro_f1(true_i, pa) - my_macro_f1(true_i, pb)) >= abs(dobs) - 1e-12:
            cnt += 1
    return dobs, ci, (cnt + 1) / (n_draws + 1)


log("")
log("Independent paired re-runs (2000 draws, seed 42):")
for a, b, sc in (("ST_J60", "Q_amp", "src"), ("ST_J60", "R_amp", "tgt")):
    key = "src" if sc == "src" else "tgt"
    d, ci, p = paired(terr_p_i, group_p,
                      rep[a + "_angles"][key], rep[b + "_angles"][key])
    log(f"  [{sc}] {a} vs {b}: Delta {d:+.4f} CI [{ci[0]:+.4f},{ci[1]:+.4f}] p={p:.4f}")
    RES.setdefault("paired_indep", {})[f"{sc}:{a}v{b}"] = dict(delta=d, ci=ci, p=p)
save()

# ST_J60 source: is it AT the constant floor? bootstrap CI vs floor
pST = near_idx(rep["ST_J60_angles"]["src"])
uniq = np.unique(group_p)
idx_of = {g: np.flatnonzero(group_p == g) for g in uniq}
r = np.random.default_rng(3)
vals = np.empty(1000)
for i in range(1000):
    gs = r.choice(uniq, size=len(uniq), replace=True)
    rows = np.concatenate([idx_of[g] for g in gs])
    vals[i] = my_macro_f1(terr_p_i[rows], pST[rows])
ci = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
log(f"ST_J60 source F1 {my_macro_f1(terr_p_i, pST):.4f} patient-boot CI "
    f"[{ci[0]:.4f},{ci[1]:.4f}] vs const floor {constF1_p:.4f} "
    f"-> at-floor claim {'OK' if ci[0] <= constF1_p else 'NOT ok'}")
RES["st_floor_ci"] = ci
save()

# ------------------------------------------------- killer competitor for P3:
# what can [R_amp_I, R_amp_aVF] do when FIT ON PTB-XL (patient-disjoint CV)?
log("")
log("P3 competitor: axis2 features fit on PTB-XL itself (5-fold patient CV):")
cols = np.asarray(BLOCKS["axis2"])
Xp2 = Fp[:, cols]
Yp = np.column_stack([np.cos(angle_p), np.sin(angle_p)])
rng = np.random.default_rng(0)
pred = np.full(len(angle_p), np.nan)
for tr, te in group_folds(group_p, 5, rng):
    med = np.nanmedian(Xp2[tr], axis=0)
    m = EighRidge().fit(imp(Xp2[tr], med), Yp[tr])
    pred[te] = ang_of(m.predict(imp(Xp2[te], med)))
f1_real = my_macro_f1(terr_p_i, near_idx(pred))
log(f"  axis2 real-fit CV F1 = {f1_real:.4f}  (sim-fit transported: "
    f"{rep['axis2']['f1_src']:.4f}/{rep['axis2']['f1_tgt']:.4f}; "
    f"established fit-free number 0.3043)")
RES["axis2_realfit_cv_f1"] = f1_real

# same for full54 and ST block for context
for b in ("ST_J60", "full54"):
    cols = np.asarray(BLOCKS[b])
    Xpb = Fp[:, cols]
    rng = np.random.default_rng(0)
    pred = np.full(len(angle_p), np.nan)
    for tr, te in group_folds(group_p, 5, rng):
        med = np.nanmedian(Xpb[tr], axis=0)
        m = EighRidge().fit(imp(Xpb[tr], med), Yp[tr])
        pred[te] = ang_of(m.predict(imp(Xpb[te], med)))
    f1r = my_macro_f1(terr_p_i, near_idx(pred))
    log(f"  {b} real-fit CV F1 = {f1r:.4f}")
    RES[f"{b}_realfit_cv_f1"] = f1r
save()
log("")
log("Done.")
