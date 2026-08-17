# Promoted 2026-08-13 from reports/2026-08-13_audit_artifacts/scripts/tmp_f3_repair.py
# VERIFIED 2026-08-13 by independent adversarial re-implementation
# (analysis/channel_repair_verify.py); verdict: CONFIRMED (all numbers reproduced exactly; verifier added the measured-features competitor and ESS-matched reweighting control).
# Canonical outputs of the verified run: reports/2026-08-13_audit_artifacts/.
# Re-runs write to outputs/analysis/fidelity_audit/. Full record:
# reports/2026-08-13_fidelity_audit_and_final_verification.md, Part C.
# -*- coding: utf-8 -*-
"""F3 intervention: can the fidelity mechanism REPAIR zero-shot transfer?

Intervention A: channel-restricted latent readout (fit everything on MedalCare;
  at PTB-XL inference only latents are used -- features never).
Intervention B: importance-weight MedalCare rows to match PTB-XL feature
  distribution, refit unrestricted readout (falsification arm).
Intervention C: lead-group restriction (inferior II/III/aVF vs anterior V1-V4).

Writes incrementally to outputs/analysis/fidelity_audit/f3_repair_out.txt
(append, ASCII) and f3_repair.json.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

from geom_common import (  # noqa: E402
    TERRITORIES, RidgeSVD, angles_from_cs, group_folds,
    load_medalcare, load_ptbxl, medalcare_anchor_angles,
)

OUT_TXT = REPO / "outputs/analysis/fidelity_audit/f3_repair_out.txt"
OUT_JSON = REPO / "outputs/analysis/fidelity_audit/f3_repair.json"
SEED = 20260813
N_BOOT = 1000
N_NULL = 200

RESULTS = {}
if OUT_JSON.exists():
    try:
        RESULTS = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    except Exception:
        RESULTS = {}


def log(msg):
    line = str(msg)
    print(line, flush=True)
    with OUT_TXT.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_json():
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(str(type(o)))
    OUT_JSON.write_text(json.dumps(RESULTS, indent=1, default=conv),
                        encoding="utf-8")


# ---------------------------------------------------------------- feature IO
def load_features_ptbxl():
    fe = np.load(REPO / "data/ecg_features_spatial_ptbxl_allfolds.npz",
                 allow_pickle=True)
    names = [str(x) for x in fe["feature_names"]]
    F = fe["features"]
    mi = pd.read_csv(REPO / "data/ptbxl_mi_subclass_allfolds.csv")
    pos = np.flatnonzero(mi["territory_4c"].notna().to_numpy())
    return names, F[pos]


def load_features_medalcare():
    blocks = []
    names = None
    for split in ("train", "test"):
        t = np.load(REPO / f"data/theta_mi_{split}.npz", allow_pickle=True)
        fe = np.load(REPO / f"data/ecg_features_spatial_medalcare_{split}.npz",
                     allow_pickle=True)
        names = [str(x) for x in fe["feature_names"]]
        idx = np.asarray(t["idx_in_split"], dtype=int)
        blocks.append(fe["features"][idx])
    return names, np.vstack(blocks)


# ---------------------------------------------------------------- metrics
ANCHORS = medalcare_anchor_angles()
ANCH_ARR = np.array([ANCHORS[t] for t in TERRITORIES])
TERR_INT = {t: i for i, t in enumerate(TERRITORIES)}


def assign_int(pred_angle):
    d = np.abs(np.angle(np.exp(1j * (pred_angle[:, None] - ANCH_ARR[None, :]))))
    return np.argmin(d, axis=1)


def terr_to_int(terr):
    return np.array([TERR_INT[t] for t in terr])


def fast_macro_f1(yt, yp):
    cm = np.bincount(yt * 4 + yp, minlength=16).reshape(4, 4).astype(float)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    return float(f1.mean())


def circ_R(pred, true):
    return float(np.abs(np.mean(np.exp(1j * (pred - true)))))


def circ_eta2(pred, terr):
    z = np.exp(1j * pred)
    R_tot = np.abs(z.sum())
    n = len(pred)
    Rj = sum(np.abs(z[terr == t].sum()) for t in TERRITORIES)
    den = n - R_tot
    if den < 1e-9:
        return 0.0
    return float((Rj - R_tot) / den)


def score_all(pred_angle, terr_str, true_angle):
    yt = terr_to_int(terr_str)
    yp = assign_int(pred_angle)
    return {
        "macro_f1": fast_macro_f1(yt, yp),
        "eta2": circ_eta2(pred_angle, terr_str),
        "R": circ_R(pred_angle, true_angle),
    }


# ---------------------------------------------------------------- ridge utils
def coef_at(m, y, a):
    ym = y.mean(0)
    d = m.s_ / (m.s_ ** 2 + a)
    return m.vt_.T @ (d[:, None] * (m.u_.T @ (y - ym))), ym


def pred_with(m, x, coef, ym):
    return ((x - m.mu_) / m.sd_ - m.xc_mean_) @ coef + ym


def transport_preds(m_r, r_src_model_input_dim_ignored, r_tgt):
    """Return (pred_source_mode, pred_target_mode) angle arrays on target."""
    p_src = m_r.predict(r_tgt)
    rt = (r_tgt - r_tgt.mean(0)) / (r_tgt.std(0) + 1e-8)
    p_tgt = m_r.predict(rt * m_r.sd_ + m_r.mu_)
    return angles_from_cs(p_src), angles_from_cs(p_tgt)


# ---------------------------------------------------------------- paired boot
def paired_boot(predA, predB, terr_str, groups, seed=SEED, n_boot=N_BOOT):
    yt = terr_to_int(terr_str)
    yA = assign_int(predA)
    yB = assign_int(predB)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_of = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        gs = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([idx_of[g] for g in gs])
        deltas[b] = fast_macro_f1(yt[rows], yA[rows]) - \
            fast_macro_f1(yt[rows], yB[rows])
    obs = fast_macro_f1(yt, yA) - fast_macro_f1(yt, yB)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0)))
    return {"delta": obs, "ci_lo": float(lo), "ci_hi": float(hi),
            "p_boot": min(p, 1.0), "n_boot": n_boot}


# ---------------------------------------------------------------- blocks
def block_cols(names):
    leads12 = ["I", "II", "III", "aVR", "aVL", "aVF",
               "V1", "V2", "V3", "V4", "V5", "V6"]
    def cols(meas, leads):
        return [names.index(f"{meas}_{ld}") for ld in leads]
    B = {
        "Q12": cols("Q_amp", leads12),
        "R12": cols("R_amp", leads12),
        "ST12": cols("ST_J60", leads12),
        "T12": cols("T_amp", leads12),
        "axpair": [names.index("R_amp_I"), names.index("R_amp_aVF")],
        "QR24": cols("Q_amp", leads12) + cols("R_amp", leads12),
        "inferior": [names.index(f"{m}_{ld}")
                     for m in ("ST_J60", "Q_amp", "R_amp", "T_amp")
                     for ld in ("II", "III", "aVF")],
        "anterior": [names.index(f"{m}_{ld}")
                     for m in ("ST_J60", "Q_amp", "R_amp", "T_amp")
                     for ld in ("V1", "V2", "V3", "V4")],
    }
    return B


# ================================================================ pipeline
class Ctx:
    pass


def load_ctx(encoder):
    c = Ctx()
    c.encoder = encoder
    c.mc = load_medalcare(encoder)
    c.pt = load_ptbxl(encoder, ANCHORS)
    names_p, Fp = load_features_ptbxl()
    names_m, Fm_full = load_features_medalcare()
    assert names_p == names_m
    c.names = names_p
    c.Fp = Fp
    c.Fm = Fm_full
    assert c.Fm.shape[0] == len(c.mc.angle), (c.Fm.shape, len(c.mc.angle))
    assert c.Fp.shape[0] == len(c.pt.angle), (c.Fp.shape, len(c.pt.angle))
    c.Ymc = np.column_stack([np.cos(c.mc.angle), np.sin(c.mc.angle)])
    c.blocks = block_cols(c.names)
    c.fin54_mc = np.all(np.isfinite(c.Fm), axis=1)
    c.fin54_pt = np.all(np.isfinite(c.Fp), axis=1)
    return c


def fit_W_stage(c):
    """One factorisation of MedalCare latents on all-54-finite rows; per-block
    GCV alpha + coefs solved on the shared SVD."""
    rows = np.flatnonzero(c.fin54_mc)
    mW = RidgeSVD().fit(c.mc.z[rows], c.Fm[rows])  # fits with joint-GCV alpha
    c.mW = mW
    c.W_rows = rows
    c.Wcoef = {}
    for bname, colsb in c.blocks.items():
        Yb = c.Fm[rows][:, colsb]
        a = mW._gcv_alpha(Yb)
        coef, ym = coef_at(mW, Yb, a)
        c.Wcoef[bname] = (coef, ym, float(a))
    return c


def r_of(c, bname, z):
    coef, ym, _ = c.Wcoef[bname]
    return pred_with(c.mW, z, coef, ym)


def run_base(c, tag):
    """Unrestricted latent readout + feature baselines + floors."""
    res = {}
    # floors on the PTB-XL row set actually used
    res["floor_R_ptbxl"] = float(np.abs(np.mean(np.exp(-1j * c.pt.angle))))
    yt = terr_to_int(c.pt.territory)
    res["floor_macro_f1_const"] = max(
        fast_macro_f1(yt, np.full(len(yt), k)) for k in range(4))
    res["n_ptbxl"] = int(len(yt))
    res["n_medalcare"] = int(len(c.mc.angle))

    # unrestricted 1024-d readout
    m = RidgeSVD().fit(c.mc.z, c.Ymc)
    c.m1024 = m
    pa_src, pa_tgt = transport_preds(m, None, c.pt.z)
    c.pred1024 = {"source": pa_src, "target": pa_tgt}
    res["unrestricted"] = {
        "alpha": float(m.alpha_),
        "source": score_all(pa_src, c.pt.territory, c.pt.angle),
        "target": score_all(pa_tgt, c.pt.territory, c.pt.angle),
    }

    # 54-feature control (features both domains; finite-54 rows)
    m54 = RidgeSVD().fit(c.Fm[c.fin54_mc], c.Ymc[c.fin54_mc])
    rows_p = np.flatnonzero(c.fin54_pt)
    Xp = c.Fp[rows_p]
    p_src = angles_from_cs(m54.predict(Xp))
    xt = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-8)
    p_tgt = angles_from_cs(m54.predict(xt * m54.sd_ + m54.mu_))
    res["control54"] = {
        "alpha": float(m54.alpha_), "n_rows_pt": int(len(rows_p)),
        "source": score_all(p_src, c.pt.territory[rows_p], c.pt.angle[rows_p]),
        "target": score_all(p_tgt, c.pt.territory[rows_p], c.pt.angle[rows_p]),
    }

    # axis feature baseline: readout on measured [R_amp_I, R_amp_aVF]
    ax_cols = c.blocks["axpair"]
    fin_ax_mc = np.all(np.isfinite(c.Fm[:, ax_cols]), axis=1)
    fin_ax_pt = np.all(np.isfinite(c.Fp[:, ax_cols]), axis=1)
    max_ = RidgeSVD().fit(c.Fm[fin_ax_mc][:, ax_cols], c.Ymc[fin_ax_mc])
    rows_p = np.flatnonzero(fin_ax_pt)
    Xp = c.Fp[rows_p][:, ax_cols]
    p_src = angles_from_cs(max_.predict(Xp))
    xt = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-8)
    p_tgt = angles_from_cs(max_.predict(xt * max_.sd_ + max_.mu_))
    c.axis_rows_pt = rows_p
    c.pred_axis = {"source": p_src, "target": p_tgt}
    res["axis_feature"] = {
        "alpha": float(max_.alpha_), "n_rows_pt": int(len(rows_p)),
        "source": score_all(p_src, c.pt.territory[rows_p], c.pt.angle[rows_p]),
        "target": score_all(p_tgt, c.pt.territory[rows_p], c.pt.angle[rows_p]),
    }
    RESULTS.setdefault(tag, {})["base"] = res
    save_json()
    log(f"[{tag}] floors: R_const={res['floor_R_ptbxl']:.5f} "
        f"macroF1_const={res['floor_macro_f1_const']:.4f} n={res['n_ptbxl']}")
    log(f"[{tag}] unrestricted1024 alpha={res['unrestricted']['alpha']:.1f} "
        f"target F1={res['unrestricted']['target']['macro_f1']:.4f} "
        f"source F1={res['unrestricted']['source']['macro_f1']:.4f}")
    log(f"[{tag}] control54 target F1={res['control54']['target']['macro_f1']:.4f} "
        f"(n={res['control54']['n_rows_pt']}); "
        f"axis target F1={res['axis_feature']['target']['macro_f1']:.4f} "
        f"(n={res['axis_feature']['n_rows_pt']})")
    return res


def indomain_cv_block(c, bname, n_splits=5):
    """Group-disjoint 5-fold CV of the restricted pipeline inside MedalCare.
    W refit per fold on train & finite-54 rows; readout on all train rows."""
    rng = np.random.default_rng(SEED)
    pred = np.full(len(c.mc.angle), np.nan)
    for tr, te in group_folds(c.mc.group, n_splits, rng):
        fin = tr[np.all(np.isfinite(c.Fm[tr]), axis=1)]
        mW = RidgeSVD().fit(c.mc.z[fin], c.Fm[fin])
        Yb = c.Fm[fin][:, c.blocks[bname]]
        a = mW._gcv_alpha(Yb)
        coef, ym = coef_at(mW, Yb, a)
        r_tr = pred_with(mW, c.mc.z[tr], coef, ym)
        r_te = pred_with(mW, c.mc.z[te], coef, ym)
        m_r = RidgeSVD().fit(r_tr, c.Ymc[tr])
        pred[te] = angles_from_cs(m_r.predict(r_te))
    return score_all(pred, c.mc.territory, c.mc.angle)


def indomain_cv_unrestricted(c, n_splits=5):
    rng = np.random.default_rng(SEED)
    pred = np.full(len(c.mc.angle), np.nan)
    for tr, te in group_folds(c.mc.group, n_splits, rng):
        m = RidgeSVD().fit(c.mc.z[tr], c.Ymc[tr])
        pred[te] = angles_from_cs(m.predict(c.mc.z[te]))
    return score_all(pred, c.mc.territory, c.mc.angle)


def run_blocks(c, tag, do_cv=True):
    fit_W_stage(c)
    out = RESULTS.setdefault(tag, {}).setdefault("blocks", {})
    c.pred_block = {}
    for bname in c.blocks:
        r_mc = r_of(c, bname, c.mc.z)
        r_pt = r_of(c, bname, c.pt.z)
        m_r = RidgeSVD().fit(r_mc, c.Ymc)
        pa_src, pa_tgt = transport_preds(m_r, None, r_pt)
        c.pred_block[bname] = {"source": pa_src, "target": pa_tgt,
                               "model": m_r, "r_mc": r_mc, "r_pt": r_pt}
        entry = {
            "d": len(c.blocks[bname]),
            "alpha_W": c.Wcoef[bname][2],
            "alpha_readout": float(m_r.alpha_),
            "cross_source": score_all(pa_src, c.pt.territory, c.pt.angle),
            "cross_target": score_all(pa_tgt, c.pt.territory, c.pt.angle),
        }
        if do_cv:
            entry["indomain_cv"] = indomain_cv_block(c, bname)
            f_in = entry["indomain_cv"]["macro_f1"]
            f_x = entry["cross_target"]["macro_f1"]
            entry["transfer_ratio_target"] = f_x / f_in if f_in > 0 else np.nan
        out[bname] = entry
        save_json()
        cvtxt = (f" indomCV F1={entry['indomain_cv']['macro_f1']:.4f} "
                 f"ratio={entry['transfer_ratio_target']:.3f}") if do_cv else ""
        log(f"[{tag}] block {bname:9s} d={entry['d']:2d} "
            f"crossF1 tgt={entry['cross_target']['macro_f1']:.4f} "
            f"src={entry['cross_source']['macro_f1']:.4f} "
            f"eta2 tgt={entry['cross_target']['eta2']:.4f} "
            f"R tgt={entry['cross_target']['R']:.4f}{cvtxt}")
    if do_cv:
        cv_u = indomain_cv_unrestricted(c)
        RESULTS[tag]["unrestricted_indomain_cv"] = cv_u
        f_x = RESULTS[tag]["base"]["unrestricted"]["target"]["macro_f1"]
        RESULTS[tag]["unrestricted_transfer_ratio_target"] = \
            f_x / cv_u["macro_f1"]
        save_json()
        log(f"[{tag}] unrestricted indomCV F1={cv_u['macro_f1']:.4f} "
            f"ratio={f_x / cv_u['macro_f1']:.3f}")


def run_paired(c, tag):
    out = RESULTS.setdefault(tag, {}).setdefault("paired", {})
    pairs = [
        ("QR24_vs_unrestricted", "QR24", None),
        ("ST12_vs_unrestricted", "ST12", None),
        ("QR24_vs_ST12", "QR24", "ST12"),
        ("R12_vs_unrestricted", "R12", None),
        ("Q12_vs_unrestricted", "Q12", None),
        ("inferior_vs_anterior", "inferior", "anterior"),
        ("inferior_vs_unrestricted", "inferior", None),
    ]
    for mode in ("target", "source"):
        for name, a, b in pairs:
            pA = c.pred_block[a][mode]
            pB = c.pred1024[mode] if b is None else c.pred_block[b][mode]
            r = paired_boot(pA, pB, c.pt.territory, c.pt.group)
            out[f"{name}_{mode}"] = r
            save_json()
            log(f"[{tag}] paired {name} [{mode}]: delta={r['delta']:+.4f} "
                f"CI[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] p={r['p_boot']:.4f}")
    # QR24 vs measured-axis baseline on axis-finite rows
    rows = c.axis_rows_pt
    for mode in ("target", "source"):
        r = paired_boot(c.pred_block["QR24"][mode][rows],
                        c.pred_axis[mode],
                        c.pt.territory[rows], c.pt.group[rows])
        out[f"QR24_vs_axisfeat_{mode}"] = r
        save_json()
        log(f"[{tag}] paired QR24_vs_axisfeat [{mode}] (n={len(rows)}): "
            f"delta={r['delta']:+.4f} CI[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] "
            f"p={r['p_boot']:.4f}")


# ---------------------------------------------------------------- nulls
def null_random_projection(c, bname, n_draws=N_NULL):
    """Norm-matched random projection of same output dim; full pipeline after."""
    coef, ym, _a = c.Wcoef[bname]
    colnorm = np.linalg.norm(coef, axis=0)
    rng = np.random.default_rng(SEED + 7)
    obs = {m: fast_macro_f1(terr_to_int(c.pt.territory),
                            assign_int(c.pred_block[bname][m]))
           for m in ("source", "target")}
    null_scores = {"source": [], "target": []}
    for _ in range(n_draws):
        G = rng.standard_normal(coef.shape)
        G *= colnorm / (np.linalg.norm(G, axis=0) + 1e-12)
        r_mc = pred_with(c.mW, c.mc.z, G, ym)
        r_pt = pred_with(c.mW, c.pt.z, G, ym)
        m_r = RidgeSVD().fit(r_mc, c.Ymc)
        pa_src, pa_tgt = transport_preds(m_r, None, r_pt)
        yt = terr_to_int(c.pt.territory)
        null_scores["source"].append(fast_macro_f1(yt, assign_int(pa_src)))
        null_scores["target"].append(fast_macro_f1(yt, assign_int(pa_tgt)))
    out = {}
    for m in ("source", "target"):
        ns = np.array(null_scores[m])
        out[m] = {"obs": obs[m], "null_mean": float(ns.mean()),
                  "null_p95": float(np.percentile(ns, 95)),
                  "p": float((1 + np.sum(ns >= obs[m])) / (n_draws + 1))}
    return out


def null_shuffled_phi(c, bname, n_draws=N_NULL):
    """Shuffle MedalCare phi at group level, refit readout (fresh GCV alpha on
    the cached factorisation), transport, score."""
    ent = c.pred_block[bname]
    m_r = ent["model"]
    r_pt = ent["r_pt"]
    rng = np.random.default_rng(SEED + 13)
    uniq, inv = np.unique(c.mc.group, return_inverse=True)
    # representative phi per group (groups are single rows if uniq==n)
    grp_phi = np.array([c.mc.angle[inv == i][0] for i in range(len(uniq))])
    yt = terr_to_int(c.pt.territory)
    obs = {m: fast_macro_f1(yt, assign_int(ent[m]))
           for m in ("source", "target")}
    null_scores = {"source": [], "target": []}
    for _ in range(n_draws):
        perm = rng.permutation(len(uniq))
        phi_s = grp_phi[perm][inv]
        Ys = np.column_stack([np.cos(phi_s), np.sin(phi_s)])
        a = m_r._gcv_alpha(Ys)
        coef, ym = coef_at(m_r, Ys, a)
        p_src = angles_from_cs(pred_with(m_r, r_pt, coef, ym))
        rt = (r_pt - r_pt.mean(0)) / (r_pt.std(0) + 1e-8)
        p_tgt = angles_from_cs(pred_with(m_r, rt * m_r.sd_ + m_r.mu_, coef, ym))
        null_scores["source"].append(fast_macro_f1(yt, assign_int(p_src)))
        null_scores["target"].append(fast_macro_f1(yt, assign_int(p_tgt)))
    out = {}
    for m in ("source", "target"):
        ns = np.array(null_scores[m])
        out[m] = {"obs": obs[m], "null_mean": float(ns.mean()),
                  "null_p95": float(np.percentile(ns, 95)),
                  "p": float((1 + np.sum(ns >= obs[m])) / (n_draws + 1))}
    out["n_unique_groups"] = int(len(uniq))
    return out


def run_nulls(c, tag, blocks=("QR24", "ST12", "R12", "inferior")):
    out = RESULTS.setdefault(tag, {}).setdefault("nulls", {})
    for bname in blocks:
        rp = null_random_projection(c, bname)
        out[f"{bname}_randproj"] = rp
        save_json()
        log(f"[{tag}] null randproj {bname}: tgt obs={rp['target']['obs']:.4f} "
            f"null_mean={rp['target']['null_mean']:.4f} "
            f"p95={rp['target']['null_p95']:.4f} p={rp['target']['p']:.4f} | "
            f"src obs={rp['source']['obs']:.4f} p={rp['source']['p']:.4f}")
        sh = null_shuffled_phi(c, bname)
        out[f"{bname}_shufphi"] = sh
        save_json()
        log(f"[{tag}] null shufphi  {bname}: tgt obs={sh['target']['obs']:.4f} "
            f"null_mean={sh['target']['null_mean']:.4f} "
            f"p95={sh['target']['null_p95']:.4f} p={sh['target']['p']:.4f} | "
            f"src p={sh['source']['p']:.4f} (groups={sh['n_unique_groups']})")


# ---------------------------------------------------------------- Intervention B
def run_reweight(c, tag, n_resamples=10):
    out = RESULTS.setdefault(tag, {}).setdefault("reweight", {})
    summaries = {
        "axpair": c.blocks["axpair"],
        "six": [c.names.index(x) for x in
                ("R_amp_I", "R_amp_aVF", "ST_J60_avg_mV", "T_amplitude_mV",
                 "QRS_duration_ms", "heart_rate_bpm")],
    }
    yt_all = terr_to_int(c.pt.territory)
    for sname, colsb in summaries.items():
        fin_mc = np.all(np.isfinite(c.Fm[:, colsb]), axis=1)
        fin_pt = np.all(np.isfinite(c.Fp[:, colsb]), axis=1)
        Xm = c.Fm[fin_mc][:, colsb]
        Xp = c.Fp[fin_pt][:, colsb]
        Xall = np.vstack([Xm, Xp])
        mu, sd = Xall.mean(0), Xall.std(0) + 1e-8
        Xall = (Xall - mu) / sd
        ylab = np.r_[np.zeros(len(Xm)), np.ones(len(Xp))]
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xall, ylab)
        p_real = clf.predict_proba((Xm - mu) / sd)[:, 1]
        w = p_real / np.maximum(1 - p_real, 1e-12)
        clip_hi = np.percentile(w, 99)
        w = np.clip(w, None, clip_hi)
        w /= w.sum()
        ess = float(1.0 / np.sum(w ** 2))
        rows_mc = np.flatnonzero(fin_mc)

        # unweighted baseline on the SAME row subset
        m0 = RidgeSVD().fit(c.mc.z[rows_mc], c.Ymc[rows_mc])
        p0_src, p0_tgt = transport_preds(m0, None, c.pt.z)
        base = {m: score_all(p, c.pt.territory, c.pt.angle)
                for m, p in (("source", p0_src), ("target", p0_tgt))}

        rng = np.random.default_rng(SEED + 29)
        f1s = {"source": [], "target": []}
        acc = {"source": np.zeros((len(yt_all), 2)),
               "target": np.zeros((len(yt_all), 2))}
        for _ in range(n_resamples):
            samp = rng.choice(rows_mc, size=len(rows_mc), replace=True, p=w)
            m = RidgeSVD().fit(c.mc.z[samp], c.Ymc[samp])
            pa_src, pa_tgt = transport_preds(m, None, c.pt.z)
            for mode, pa in (("source", pa_src), ("target", pa_tgt)):
                f1s[mode].append(fast_macro_f1(yt_all, assign_int(pa)))
                acc[mode] += np.column_stack([np.cos(pa), np.sin(pa)])
        entry = {"ess": ess, "n_rows": int(len(rows_mc)),
                 "clip_hi_pct99": float(clip_hi),
                 "baseline_subset": base}
        for mode in ("source", "target"):
            pa_avg = np.arctan2(acc[mode][:, 1], acc[mode][:, 0])
            pb = paired_boot(pa_avg,
                             {"source": p0_src, "target": p0_tgt}[mode],
                             c.pt.territory, c.pt.group)
            entry[mode] = {
                "f1_mean": float(np.mean(f1s[mode])),
                "f1_sd": float(np.std(f1s[mode])),
                "f1_avgpred": fast_macro_f1(yt_all, assign_int(pa_avg)),
                "score_avgpred": score_all(pa_avg, c.pt.territory, c.pt.angle),
                "paired_vs_unweighted": pb,
            }
        out[sname] = entry
        save_json()
        log(f"[{tag}] reweight[{sname}] ESS={ess:.0f}/{len(rows_mc)} | "
            f"tgt F1 unw={base['target']['macro_f1']:.4f} "
            f"w={entry['target']['f1_mean']:.4f}+-{entry['target']['f1_sd']:.4f} "
            f"avgpred={entry['target']['f1_avgpred']:.4f} "
            f"delta={entry['target']['paired_vs_unweighted']['delta']:+.4f} "
            f"p={entry['target']['paired_vs_unweighted']['p_boot']:.4f} | "
            f"src F1 unw={base['source']['macro_f1']:.4f} "
            f"w={entry['source']['f1_mean']:.4f}")


# ================================================================ main
def main():
    stages = sys.argv[1:] or ["all"]
    t0 = time.time()
    log("=" * 78)
    log(f"F3 REPAIR run started {time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"stages={stages} seed={SEED}")
    log("PRE-STATED MECHANISM PREDICTIONS (written before any F3 result):")
    log(" P1: QR24-restricted readout transfers better per unit in-domain")
    log("     strength than ST12-restricted (transfer_ratio_target QR24 > ST12).")
    log(" P2: repair claim iff QR24-restricted BEATS unrestricted 1024-d readout")
    log("     cross-domain, paired patient-bootstrap p<0.05, and survives both")
    log("     nulls (norm-matched random projection AND shuffled-phi refit).")
    log(" P3: Intervention B (importance reweighting of MedalCare rows) yields")
    log("     NO cross-domain improvement (reweighting cannot create information")
    log("     the simulator does not carry). If it improves, mechanism story is")
    log("     incomplete and will be reported as such.")
    log("Target-scaler definition for restricted arms: r_block outputs on PTB-XL")
    log(" are re-standardised by PTB-XL's OWN unlabelled r statistics (diagonal")
    log(" CORAL on r), then mapped back to the MedalCare r frame; the W_block")
    log(" stage always uses the source (MedalCare) latent scaler. Source mode =")
    log(" no target statistics anywhere.")
    log("=" * 78)

    if "all" in stages or "head" in stages:
        enc = "exp8_leadfix_medalonly"
        tag = enc
        log(f"--- loading {enc} ---")
        c = load_ctx(enc)
        log(f"[{tag}] loaded: mc z{c.mc.z.shape} pt z{c.pt.z.shape} "
            f"Fm{c.Fm.shape} Fp{c.Fp.shape} fin54 mc={int(c.fin54_mc.sum())} "
            f"pt={int(c.fin54_pt.sum())}")
        run_base(c, tag)
        run_blocks(c, tag, do_cv=True)
        run_paired(c, tag)
        run_nulls(c, tag)
        run_reweight(c, tag)
        log(f"[{tag}] done at {time.time()-t0:.0f}s")

    if "all" in stages or "dual" in stages:
        enc = "exp8_leadfix_dual"
        tag = enc
        log(f"--- loading {enc} (sensitivity) ---")
        try:
            c = load_ctx(enc)
            run_base(c, tag)
            run_blocks(c, tag, do_cv=True)
            run_paired(c, tag)
            log(f"[{tag}] done at {time.time()-t0:.0f}s")
        except Exception:
            log("DUAL SENSITIVITY FAILED:\n" + traceback.format_exc())

    log(f"ALL DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL:\n" + traceback.format_exc())
        raise
