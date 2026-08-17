# Promoted 2026-08-13 from reports/2026-08-13_audit_artifacts/scripts/tmp_v_f3_verify.py
# Adversarial verifier script (independent re-implementation; do not 'fix'
# to agree with the primary -- its value is that it shares no code with it).
# -*- coding: utf-8 -*-
"""Adversarial verification of F3-channel-repair.

Independent re-implementation (own ridge, own GCV, sklearn macro-F1, own
bootstrap with different seed) of the decisive statistics, plus killer
competitors and the missing ESS-matched control for Intervention B.

Writes incrementally to outputs/analysis/fidelity_audit/f3_verify_out.txt (ASCII).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

# use only the data loaders from geom_common (shared infrastructure), nothing else
from geom_common import load_medalcare, load_ptbxl, TERRITORIES  # noqa: E402

OUT = REPO / "outputs/analysis/fidelity_audit/f3_verify_out.txt"
VSEED = 777


def log(msg):
    line = str(msg)
    print(line, flush=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------- my own ridge (same definition, independent code) ----------
ALPHAS = np.logspace(-2, 5, 24)


def fit_ridge(X, Y, alphas=ALPHAS):
    mu = X.mean(0)
    sd = X.std(0) + 1e-8
    Xs = (X - mu) / sd
    xm = Xs.mean(0)
    Xc = Xs - xm
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    Yc = Y - Y.mean(0)
    UtY = U.T @ Yc
    n = X.shape[0]
    best_gcv, best_a = np.inf, alphas[0]
    for a in alphas:
        sh = s ** 2 / (s ** 2 + a)
        resid = Yc - U @ (sh[:, None] * UtY)
        dof = sh.sum() + 1.0
        gcv = (resid ** 2).sum() / n / max(1.0 - dof / n, 1e-6) ** 2
        if gcv < best_gcv:
            best_gcv, best_a = gcv, a
    d = s / (s ** 2 + best_a)
    coef = Vt.T @ (d[:, None] * UtY)
    return {"mu": mu, "sd": sd, "xm": xm, "coef": coef, "ym": Y.mean(0),
            "alpha": float(best_a)}


def rpredict(m, X):
    return ((X - m["mu"]) / m["sd"] - m["xm"]) @ m["coef"] + m["ym"]


def angles(cs):
    return np.arctan2(cs[:, 1], cs[:, 0])


# ---------------- anchors recomputed directly from theta files --------------
def my_anchors():
    phis, terrs = [], []
    for split in ("train", "test"):
        t = np.load(REPO / f"data/theta_mi_{split}.npz", allow_pickle=True)
        phis.append(np.asarray(t["phi"], dtype=float))
        terrs.append(np.asarray(t["territory_4c"]))
    phi = np.concatenate(phis)
    terr = np.concatenate(terrs)
    return {k: float(np.angle(np.mean(np.exp(1j * phi[terr == k]))))
            for k in TERRITORIES}


ANCH = my_anchors()
ANCH_ARR = np.array([ANCH[t] for t in TERRITORIES])
T2I = {t: i for i, t in enumerate(TERRITORIES)}


def nearest_anchor(pred_angle):
    d = np.abs(np.angle(np.exp(1j * (pred_angle[:, None] - ANCH_ARR[None, :]))))
    return np.argmin(d, axis=1)


def macro_f1(yt, yp):
    return float(f1_score(yt, yp, average="macro", labels=[0, 1, 2, 3],
                          zero_division=0))


def circ_R(pred, true):
    return float(np.abs(np.mean(np.exp(1j * (pred - true)))))


def circ_eta2(pred, terr):
    z = np.exp(1j * pred)
    Rtot = np.abs(z.sum())
    Rj = sum(np.abs(z[terr == t].sum()) for t in TERRITORIES)
    return float((Rj - Rtot) / (len(pred) - Rtot))


# ---------------- feature loading (own code, per the verified recipes) ------
def feats_ptbxl():
    fe = np.load(REPO / "data/ecg_features_spatial_ptbxl_allfolds.npz",
                 allow_pickle=True)
    names = [str(x) for x in fe["feature_names"]]
    F = fe["features"]
    mi = pd.read_csv(REPO / "data/ptbxl_mi_subclass_allfolds.csv")
    pos = np.flatnonzero(mi["territory_4c"].notna().to_numpy())
    return names, F[pos]


def feats_medalcare():
    out = []
    names = None
    for split in ("train", "test"):
        t = np.load(REPO / f"data/theta_mi_{split}.npz", allow_pickle=True)
        fe = np.load(REPO / f"data/ecg_features_spatial_medalcare_{split}.npz",
                     allow_pickle=True)
        names = [str(x) for x in fe["feature_names"]]
        idx = np.asarray(t["idx_in_split"], dtype=int)
        out.append(fe["features"][idx])
    return names, np.vstack(out)


def paired_boot(predA_int, predB_int, yt, groups, n_boot=1000, seed=VSEED):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_of = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        gs = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([idx_of[g] for g in gs])
        deltas[b] = (macro_f1(yt[rows], predA_int[rows])
                     - macro_f1(yt[rows], predB_int[rows]))
    obs = macro_f1(yt, predA_int) - macro_f1(yt, predB_int)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0)))
    return obs, float(lo), float(hi), min(p, 1.0)


def main():
    t0 = time.time()
    log("=" * 78)
    log(f"F3 VERIFIER started {time.strftime('%Y-%m-%d %H:%M:%S')} seed={VSEED}")
    log("Anchors (recomputed from theta files, deg): " + ", ".join(
        f"{k}={np.degrees(v):+.2f}" for k, v in ANCH.items()))

    enc = "exp8_leadfix_medalonly"
    mc = load_medalcare(enc)
    pt = load_ptbxl(enc, ANCH)
    names_p, Fp = feats_ptbxl()
    names_m, Fm = feats_medalcare()
    assert names_p == names_m
    names = names_p
    assert Fm.shape[0] == len(mc.angle) and Fp.shape[0] == len(pt.angle)
    yt = np.array([T2I[t] for t in pt.territory])
    log(f"loaded mc z{mc.z.shape} pt z{pt.z.shape} Fm{Fm.shape} Fp{Fp.shape}")

    # ---- floors
    floor_f1 = max(macro_f1(yt, np.full(len(yt), k)) for k in range(4))
    floor_R = float(np.abs(np.mean(np.exp(-1j * pt.angle))))
    log(f"[floors] const macroF1={floor_f1:.4f} const R={floor_R:.5f} "
        f"(reported 0.1534 / 0.29216)")

    # ---- row-alignment sanity: latent-predicted HR vs measured HR on PTB-XL
    hr_col = names.index("heart_rate_bpm")
    fin_hr_mc = np.isfinite(Fm[:, hr_col])
    fin_hr_pt = np.isfinite(Fp[:, hr_col])
    mhr = fit_ridge(mc.z[fin_hr_mc], Fm[fin_hr_mc][:, [hr_col]])
    hr_hat = rpredict(mhr, pt.z[fin_hr_pt])[:, 0]
    hr_true = Fp[fin_hr_pt][:, hr_col]
    r_align = float(np.corrcoef(hr_hat, hr_true)[0, 1])
    rng = np.random.default_rng(VSEED)
    sh = rng.permutation(len(hr_true))
    r_shuf = float(np.corrcoef(hr_hat, hr_true[sh])[0, 1])
    log(f"[align] corr(latent-pred HR, measured HR) on PTB-XL rows: "
        f"r={r_align:.3f} (shuffled r={r_shuf:.3f}, n={len(hr_true)})")

    # ---- Stage 1: W_block on all-54-finite MedalCare rows
    fin54_mc = np.all(np.isfinite(Fm), axis=1)
    rows = np.flatnonzero(fin54_mc)
    log(f"[align] fin54 mc={len(rows)} (reported 4412), "
        f"pt={int(np.all(np.isfinite(Fp), axis=1).sum())} (reported 3254)")

    leads12 = ["I", "II", "III", "aVR", "aVL", "aVF",
               "V1", "V2", "V3", "V4", "V5", "V6"]

    def cols(meas, lds):
        return [names.index(f"{meas}_{ld}") for ld in lds]

    blocks = {
        "QR24": cols("Q_amp", leads12) + cols("R_amp", leads12),
        "ST12": cols("ST_J60", leads12),
        "inferior": [names.index(f"{m}_{ld}")
                     for m in ("ST_J60", "Q_amp", "R_amp", "T_amp")
                     for ld in ("II", "III", "aVF")],
    }

    Ymc = np.column_stack([np.cos(mc.angle), np.sin(mc.angle)])

    # unrestricted arm
    m1024 = fit_ridge(mc.z, Ymc)
    p_src_u = angles(rpredict(m1024, pt.z))
    zt = (pt.z - pt.z.mean(0)) / (pt.z.std(0) + 1e-8)
    p_tgt_u = angles(rpredict(m1024, zt * m1024["sd"] + m1024["mu"]))
    f1_u_tgt = macro_f1(yt, nearest_anchor(p_tgt_u))
    f1_u_src = macro_f1(yt, nearest_anchor(p_src_u))
    log(f"[unrestricted] alpha={m1024['alpha']:.1f} tgtF1={f1_u_tgt:.4f} "
        f"srcF1={f1_u_src:.4f} (reported 740.6 / 0.3402 / 0.2788)")

    # W stage: separate per-block ridge (own GCV per block on its own SVD --
    # deliberately NOT their shared-factorisation shortcut)
    preds_int = {"unrestr": nearest_anchor(p_tgt_u)}
    block_scores = {}
    for bname, cb in blocks.items():
        mW = fit_ridge(mc.z[rows], Fm[rows][:, cb])
        r_mc = rpredict(mW, mc.z)
        r_pt = rpredict(mW, pt.z)
        m_r = fit_ridge(r_mc, Ymc)
        p_src = angles(rpredict(m_r, r_pt))
        rt = (r_pt - r_pt.mean(0)) / (r_pt.std(0) + 1e-8)
        p_tgt = angles(rpredict(m_r, rt * m_r["sd"] + m_r["mu"]))
        f1t = macro_f1(yt, nearest_anchor(p_tgt))
        f1s = macro_f1(yt, nearest_anchor(p_src))
        e2 = circ_eta2(p_tgt, pt.territory)
        R = circ_R(p_tgt, pt.angle)
        block_scores[bname] = f1t
        preds_int[bname] = nearest_anchor(p_tgt)
        log(f"[block {bname}] alphaW={mW['alpha']:.1f} tgtF1={f1t:.4f} "
            f"srcF1={f1s:.4f} eta2={e2:.4f} R={R:.4f}")
    log("  (reported: QR24 0.2850/0.2858; ST12 0.1995/0.2078; "
        "inferior 0.3427/0.2756 eta2 0.2113 R 0.3398)")

    # ---- paired bootstraps (own seed)
    for a, b, rep in [("QR24", "unrestr", "-0.0552 [-0.0747,-0.0368] p<0.001"),
                      ("inferior", "unrestr", "+0.0025 [-0.0184,+0.0237] p=0.794"),
                      ("QR24", "ST12", "+0.0855 [+0.0688,+0.1044] p<0.001")]:
        obs, lo, hi, p = paired_boot(preds_int[a], preds_int[b], yt, pt.group)
        log(f"[paired] {a} vs {b}: delta={obs:+.4f} CI[{lo:+.4f},{hi:+.4f}] "
            f"p={p:.4f}   (reported {rep})")

    # ---- killer competitors for the inferior-parity headline
    # (a) naive zero-shot axis
    iI, iF = names.index("R_amp_I"), names.index("R_amp_aVF")
    fin_ax = np.isfinite(Fp[:, iI]) & np.isfinite(Fp[:, iF])
    ax_ang = np.arctan2(Fp[fin_ax, iF], Fp[fin_ax, iI])
    f1_ax = macro_f1(yt[fin_ax], nearest_anchor(ax_ang))
    log(f"[competitor] naive zero-shot axis: F1={f1_ax:.5f} n={int(fin_ax.sum())} "
        f"(reported 0.30431 n=4315)")

    # (b) MEASURED inferior features readout (no latents at all): fit on
    # MedalCare measured features, transport with diagonal CORAL
    cb = blocks["inferior"]
    fin_inf_mc = np.all(np.isfinite(Fm[:, cb]), axis=1)
    fin_inf_pt = np.all(np.isfinite(Fp[:, cb]), axis=1)
    mf = fit_ridge(Fm[fin_inf_mc][:, cb], Ymc[fin_inf_mc])
    Xp = Fp[fin_inf_pt][:, cb]
    xt = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-8)
    p_tgt_meas = angles(rpredict(mf, xt * mf["sd"] + mf["mu"]))
    ytm = yt[fin_inf_pt]
    f1_meas = macro_f1(ytm, nearest_anchor(p_tgt_meas))
    # paired vs latent-restricted inferior arm on the same rows
    obs, lo, hi, p = paired_boot(preds_int["inferior"][fin_inf_pt],
                                 nearest_anchor(p_tgt_meas),
                                 ytm, pt.group[fin_inf_pt])
    log(f"[competitor] MEASURED-inferior-features readout (target mode): "
        f"F1={f1_meas:.4f} n={int(fin_inf_pt.sum())}; "
        f"latent-inferior minus measured on same rows: delta={obs:+.4f} "
        f"CI[{lo:+.4f},{hi:+.4f}] p={p:.4f}")

    # (c) random-projection null for inferior (own code, 100 draws)
    rngn = np.random.default_rng(VSEED + 1)
    nulls = []
    for _ in range(100):
        G = rngn.standard_normal((1024, 12))
        # project standardised latents (norm-matching washes out anyway)
        r_mc = ((mc.z - m1024["mu"]) / m1024["sd"]) @ G
        r_pt = ((pt.z - m1024["mu"]) / m1024["sd"]) @ G
        m_r = fit_ridge(r_mc, Ymc)
        rt = (r_pt - r_pt.mean(0)) / (r_pt.std(0) + 1e-8)
        p_t = angles(rpredict(m_r, rt * m_r["sd"] + m_r["mu"]))
        nulls.append(macro_f1(yt, nearest_anchor(p_t)))
    nulls = np.array(nulls)
    obs_inf = block_scores["inferior"]
    p_rp = (1 + np.sum(nulls >= obs_inf)) / (len(nulls) + 1)
    log(f"[null] randproj d=12 (100 draws, own code): mean={nulls.mean():.4f} "
        f"p95={np.percentile(nulls, 95):.4f} max={nulls.max():.4f}; "
        f"inferior obs={obs_inf:.4f} p={p_rp:.4f} "
        f"(reported null_mean 0.2254 p95 0.2662 p=0.005)")

    log(f"[time] {time.time()-t0:.0f}s -- starting Intervention B checks")

    # ---- Intervention B: reproduce + ESS-matched uniform control
    for sname, colsb in (("axpair", [iI, iF]),
                         ("six", [names.index(x) for x in
                                  ("R_amp_I", "R_amp_aVF", "ST_J60_avg_mV",
                                   "T_amplitude_mV", "QRS_duration_ms",
                                   "heart_rate_bpm")])):
        fin_mc = np.all(np.isfinite(Fm[:, colsb]), axis=1)
        fin_pt = np.all(np.isfinite(Fp[:, colsb]), axis=1)
        Xm = Fm[fin_mc][:, colsb]
        Xp2 = Fp[fin_pt][:, colsb]
        Xall = np.vstack([Xm, Xp2])
        mu, sd = Xall.mean(0), Xall.std(0) + 1e-8
        Xall = (Xall - mu) / sd
        ylab = np.r_[np.zeros(len(Xm)), np.ones(len(Xp2))]
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xall, ylab)
        pr = clf.predict_proba((Xm - mu) / sd)[:, 1]
        w = pr / np.maximum(1 - pr, 1e-12)
        w = np.clip(w, None, np.percentile(w, 99))
        w /= w.sum()
        ess = int(round(1.0 / np.sum(w ** 2)))
        rows_mc = np.flatnonzero(fin_mc)

        # unweighted baseline on same subset
        m0 = fit_ridge(mc.z[rows_mc], Ymc[rows_mc])
        p0 = angles(rpredict(m0, zt * m0["sd"] + m0["mu"]))
        f1_0 = macro_f1(yt, nearest_anchor(p0))

        rngb = np.random.default_rng(VSEED + 2)
        f1_w, f1_essu = [], []
        for _ in range(10):
            samp = rngb.choice(rows_mc, size=len(rows_mc), replace=True, p=w)
            mw_ = fit_ridge(mc.z[samp], Ymc[samp])
            pw = angles(rpredict(mw_, zt * mw_["sd"] + mw_["mu"]))
            f1_w.append(macro_f1(yt, nearest_anchor(pw)))
            # ESS-matched UNIFORM subsample (no replacement) -- the control
            # that separates 'covariate matching hurts' from 'small ESS hurts'
            sub = rngb.choice(rows_mc, size=ess, replace=False)
            mu_ = fit_ridge(mc.z[sub], Ymc[sub])
            pu = angles(rpredict(mu_, zt * mu_["sd"] + mu_["mu"]))
            f1_essu.append(macro_f1(yt, nearest_anchor(pu)))
        f1_w, f1_essu = np.array(f1_w), np.array(f1_essu)
        log(f"[reweight {sname}] ESS={ess}/{len(rows_mc)} unweighted={f1_0:.4f} "
            f"weighted={f1_w.mean():.4f}+-{f1_w.std():.4f} | "
            f"ESS-MATCHED UNIFORM control={f1_essu.mean():.4f}+-{f1_essu.std():.4f}")
        log(f"  -> harm from weighting = {f1_0 - f1_w.mean():+.4f}; "
            f"harm from ESS alone = {f1_0 - f1_essu.mean():+.4f}; "
            f"residual (weighting beyond ESS) = "
            f"{f1_essu.mean() - f1_w.mean():+.4f}")

    log(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
