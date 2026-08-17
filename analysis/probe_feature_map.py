"""Per-feature probing map: which hand-crafted ECG measurements live in the latent?

Motivation
----------
`reports/EXECUTION_LOG_2026-08-10.md` Part 18 established that a 54-feature
hand-crafted spatial control beats the 1024-d latent on *cross-domain* MI
territory decoding while losing to it *in-domain*. That is an aggregate verdict:
it says the latent's territory information is largely synthetic-specific, but it
does not say **which physiological measurement, in which anatomical lead**, is
the part that fails to transfer.

This script answers that. For each of the 48 per-lead features
(4 physiology kinds x 12 leads) it fits a ridge probe

    feature_value  ~  latent Z

on MedalCare-train MI rows, then reads the probe out twice:

    rho_in     -- MedalCare test  (does the latent encode this measurement at all?)
    rho_cross  -- PTB-XL test     (does that encoding survive sim -> real?)
    d_rho      -- rho_in - rho_cross  (the transfer deficit, per measurement)

The deliverable is a 4x12 physiology-by-anatomy grid of those three quantities.

Design decisions (and why)
--------------------------
1. **Spearman rho is the primary metric, not R^2.** The whole `--scaler-domain`
   ablation exists because MedalCare and PTB-XL disagree on per-coordinate scale
   by ~3x. R^2 is scale- and offset-sensitive, so a cross-domain R^2 would mostly
   measure amplitude calibration, not whether the information is present. Spearman
   rho is invariant to any monotone rescaling, so it isolates the question we
   actually care about. Pearson r and in-domain R^2 are recorded as secondaries;
   cross-domain R^2 is recorded but should NOT be quoted as the headline.
2. **Targets are never imputed.** `phase_b2_infarct_decoding.py` median-imputes
   NaN *inputs*, which is fine when the feature is a predictor. Here the feature
   is the target, and imputing it would replace a missing measurement with a
   constant that the probe can neither be rewarded nor penalised for -- it would
   manufacture correlation structure. Rows with a NaN target are dropped per
   feature, and the surviving n is recorded in every cell.
3. **Two X-scalers, both reported.** `source` reuses the MedalCare-train scaler
   on PTB-XL (non-transductive, but documented in `standardise_target` as a
   defect for the feature arm); `target_pool` fits on the full unselected PTB-XL
   test latent matrix. Note this script's `target_pool` is *not* subject to the
   Part 12 s12.4 corruption -- that corruption came from pooling ~75% imputed
   feature rows, and all 2198 PTB-XL *latent* rows are real. Cells where the two
   scalers disagree in sign are flagged.
4. **Permutation null needs no refit.** The cross-domain question is "is the
   probe's output associated with the true feature value at all", so shuffling
   the truth vector against fixed predictions is the correct null and costs
   nothing. Bootstrap CIs resample (pred, truth) pairs.

Outputs
-------
    outputs/analysis/probe_map/probe_map.json          -- everything
    outputs/analysis/probe_map/probe_map_<config>.csv  -- tidy per-cell table
    outputs/analysis/probe_map/grid_<config>.png       -- 4x12 heatmaps

Usage
-----
    python analysis/probe_feature_map.py
    python analysis/probe_feature_map.py --configs exp8_leadfix_baseline --n-perm 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from analysis.phase_b2_infarct_decoding import (  # noqa: E402
    CONFIG_LATENT_STEMS,
    SEED,
    derive_rng,
    load_config_latents,
    load_ptbxl_latents,
    load_ptbxl_subclass_csv,
    load_targets,
)

# The spatial-54 layout: 48 per-lead cells laid out kind-major
# (col = kind_idx * 12 + lead_idx), then 6 global features.
# Mirrors scripts/extract_ecg_features_spatial.py:101-107.
PER_LEAD_KINDS = ("ST_J60", "Q_amp", "R_amp", "T_amp")
LEADS_12 = ("I", "II", "III", "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6")
N_PER_LEAD = len(PER_LEAD_KINDS) * len(LEADS_12)

FEAT_TRAIN_PATH = REPO_ROOT / "data" / "ecg_features_spatial_medalcare_train.npz"
FEAT_TEST_PATH = REPO_ROOT / "data" / "ecg_features_spatial_medalcare_test.npz"
FEAT_PTBXL_PATH = REPO_ROOT / "data" / "ecg_features_spatial_ptbxl_test.npz"

OUT_DIR = REPO_ROOT / "outputs" / "analysis" / "probe_map"
ALPHAS = np.logspace(-2.0, 5.0, 15)
DEFAULT_CONFIGS = [
    "exp8_leadfix_baseline", "exp8_leadfix_ccmmd", "exp8_leadfix_dual",
    "exp8_leadfix_globalz", "exp8_leadfix_K64",
]
SCALER_MODES = ("source", "target_pool")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho via Pearson on ranks. NaN if either side is constant."""
    if a.size < 3:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0.0 or sb == 0.0:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    sa, sb = a.std(), b.std()
    if sa == 0.0 or sb == 0.0:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _boot_ci_spearman(
    y_true: np.ndarray, y_pred: np.ndarray, rng: np.random.Generator, n_boot: int
) -> Tuple[float, float]:
    """Percentile CI over resampled (truth, prediction) pairs."""
    if n_boot <= 0 or y_true.size < 3:
        return (float("nan"), float("nan"))
    n = y_true.size
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = _spearman(y_true[idx], y_pred[idx])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def _perm_p_spearman(
    y_true: np.ndarray, y_pred: np.ndarray, observed: float,
    rng: np.random.Generator, n_perm: int,
) -> float:
    """Two-sided permutation p for |rho|.

    Shuffling the truth against FIXED predictions is the correct null here -- the
    question is whether the probe's output carries any association with the real
    measurement, not whether a refit probe would. No refitting required.
    """
    if n_perm <= 0 or not np.isfinite(observed) or y_true.size < 3:
        return float("nan")
    rp = rankdata(y_pred)
    rp = (rp - rp.mean()) / rp.std() if rp.std() > 0 else rp * 0.0
    rt = rankdata(y_true)
    rt = (rt - rt.mean()) / rt.std() if rt.std() > 0 else rt * 0.0
    if rt.std() == 0.0 or rp.std() == 0.0:
        return float("nan")
    # Vectorised: `permuted(..., axis=1)` shuffles each row independently, so one
    # matmul yields the whole null distribution. Chunked to bound peak memory.
    n = y_true.size
    count = 0
    done = 0
    while done < n_perm:
        blk = min(2000, n_perm - done)
        M = rng.permuted(np.tile(rt, (blk, 1)), axis=1)
        count += int((np.abs(M @ rp / n) >= abs(observed) - 1e-12).sum())
        done += blk
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def load_feature_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    a = np.load(FEAT_TRAIN_PATH, allow_pickle=True)
    b = np.load(FEAT_TEST_PATH, allow_pickle=True)
    c = np.load(FEAT_PTBXL_PATH, allow_pickle=True)
    names = [str(s) for s in a["feature_names"]]
    if len(names) != N_PER_LEAD + 6:
        raise ValueError(f"expected {N_PER_LEAD + 6} feature names, got {len(names)}")
    expected = [f"{k}_{l}" for k in PER_LEAD_KINDS for l in LEADS_12]
    if names[:N_PER_LEAD] != expected:
        raise ValueError(
            "per-lead feature name/order drift -- this script indexes cells "
            f"positionally.\n  expected[:4]={expected[:4]}\n  got[:4]={names[:4]}"
        )
    return (a["features"].astype(np.float64),
            b["features"].astype(np.float64),
            c["features"].astype(np.float64),
            names)


def build_design(config: str) -> Dict[str, np.ndarray]:
    """Assemble aligned (Z, feature) matrices for one encoder."""
    targets = load_targets()
    idx_train = targets["train"]["idx_in_split"]
    idx_test = targets["test"]["idx_in_split"]

    F_tr_full, F_te_full, F_px_full, names = load_feature_matrices()
    Z_tr_full, Z_te_full = load_config_latents(config)
    Z_px_full = load_ptbxl_latents(config)

    # MedalCare: theta rows index into the full-split arrays.
    Z_tr, F_tr = Z_tr_full[idx_train], F_tr_full[idx_train]
    Z_te, F_te = Z_te_full[idx_test], F_te_full[idx_test]

    # PTB-XL: the MI-subclass rows are exactly the rows the spatial extractor was
    # run on (verified by scripts/_audit_export_scope.py). Use all of them for
    # power; the 4c primary subset is a strict subset and is recorded separately.
    sub = load_ptbxl_subclass_csv()
    px_idx = sub["row_idx"].to_numpy()
    Z_px, F_px = Z_px_full[px_idx], F_px_full[px_idx]

    terr4 = sub["territory_4c"].to_numpy()
    in_primary4 = np.isin(
        terr4, ["Anteroseptal", "Anterolateral", "Inferior", "Inferolateral"]
    )

    return {
        "Z_tr": Z_tr, "F_tr": F_tr,
        "Z_te": Z_te, "F_te": F_te,
        "Z_px": Z_px, "F_px": F_px,
        "Z_px_pool": Z_px_full,          # all 2198 latents, label-unselected
        "in_primary4": in_primary4,
        "names": names,
    }


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------

def probe_one_feature(
    d: Dict[str, np.ndarray], j: int, X_std: Dict[str, np.ndarray],
    rng: np.random.Generator, n_boot: int, n_perm: int,
) -> Dict[str, object]:
    """Fit feature j out of the latent; read it out in both domains."""
    y_tr = d["F_tr"][:, j]
    m_tr = np.isfinite(y_tr)
    if m_tr.sum() < 50:
        return {"status": "too_few_train_rows", "n_train": int(m_tr.sum())}

    model = RidgeCV(alphas=ALPHAS)
    model.fit(X_std["tr"][m_tr], y_tr[m_tr])

    out: Dict[str, object] = {
        "status": "ok",
        "n_train": int(m_tr.sum()),
        "alpha": float(model.alpha_),
    }

    # ---- in-domain readout (MedalCare test)
    y_te = d["F_te"][:, j]
    m_te = np.isfinite(y_te)
    if m_te.sum() >= 3:
        p_te = model.predict(X_std["te"][m_te])
        rho_in = _spearman(y_te[m_te], p_te)
        out.update({
            "n_in": int(m_te.sum()),
            "rho_in": rho_in,
            "rho_in_ci95": _boot_ci_spearman(y_te[m_te], p_te, rng, n_boot),
            "pearson_in": _pearson(y_te[m_te], p_te),
            "r2_in": _r2(y_te[m_te], p_te),
        })
    else:
        out.update({"n_in": int(m_te.sum()), "rho_in": float("nan")})

    # ---- cross-domain readout (PTB-XL), once per X-scaler
    y_px = d["F_px"][:, j]
    m_px = np.isfinite(y_px)
    out["n_cross"] = int(m_px.sum())
    out["n_cross_primary4"] = int((m_px & d["in_primary4"]).sum())
    for mode in SCALER_MODES:
        if m_px.sum() < 3:
            out[f"rho_cross_{mode}"] = float("nan")
            continue
        p_px = model.predict(X_std[f"px_{mode}"][m_px])
        rho_x = _spearman(y_px[m_px], p_px)
        out[f"rho_cross_{mode}"] = rho_x
        out[f"rho_cross_{mode}_ci95"] = _boot_ci_spearman(y_px[m_px], p_px, rng, n_boot)
        out[f"perm_p_cross_{mode}"] = _perm_p_spearman(
            y_px[m_px], p_px, rho_x, rng, n_perm
        )
        out[f"pearson_cross_{mode}"] = _pearson(y_px[m_px], p_px)
        # Recorded but NOT to be quoted: cross-domain R^2 conflates information
        # content with amplitude calibration across domains (see module docstring).
        out[f"r2_cross_{mode}_DO_NOT_QUOTE"] = _r2(y_px[m_px], p_px)
        # Same readout restricted to the 4c primary subset, for comparability
        # with the pre-registered territory endpoint.
        m4 = m_px & d["in_primary4"]
        if m4.sum() >= 3:
            out[f"rho_cross_{mode}_primary4"] = _spearman(
                y_px[m4], model.predict(X_std[f"px_{mode}"][m4])
            )

    ri, rc = out.get("rho_in", np.nan), out.get(f"rho_cross_{SCALER_MODES[0]}", np.nan)
    out["d_rho"] = float(ri - rc) if np.isfinite(ri) and np.isfinite(rc) else float("nan")
    a, b = out.get("rho_cross_source", np.nan), out.get("rho_cross_target_pool", np.nan)
    out["scaler_sign_disagreement"] = bool(
        np.isfinite(a) and np.isfinite(b) and np.sign(a) != np.sign(b)
        and min(abs(a), abs(b)) > 0.05
    )
    return out


def run_config(config: str, n_boot: int, n_perm: int) -> Dict[str, object]:
    print(f"\n=== {config} ===")
    d = build_design(config)
    print(f"  Z_tr={d['Z_tr'].shape}  Z_te={d['Z_te'].shape}  "
          f"Z_px={d['Z_px'].shape}  pool={d['Z_px_pool'].shape}  "
          f"primary4={int(d['in_primary4'].sum())}")

    scaler = StandardScaler().fit(d["Z_tr"])
    pool_scaler = StandardScaler().fit(d["Z_px_pool"])
    X_std = {
        "tr": scaler.transform(d["Z_tr"]),
        "te": scaler.transform(d["Z_te"]),
        "px_source": scaler.transform(d["Z_px"]),
        "px_target_pool": pool_scaler.transform(d["Z_px"]),
    }

    cells: Dict[str, Dict[str, object]] = {}
    for j in range(N_PER_LEAD):
        name = d["names"][j]
        # RNG keyed on (config, feature) so each cell reproduces standalone --
        # same convention as phase_b2_infarct_decoding.py:216-219.
        rng = derive_rng("probe_map", config, name, seed=SEED)
        cells[name] = probe_one_feature(d, j, X_std, rng, n_boot, n_perm)
        c = cells[name]
        if c["status"] == "ok":
            print(f"  {name:12s} rho_in={c.get('rho_in', float('nan')):+.3f}  "
                  f"rho_cross={c.get('rho_cross_source', float('nan')):+.3f}  "
                  f"d={c.get('d_rho', float('nan')):+.3f}  "
                  f"p={c.get('perm_p_cross_source', float('nan')):.4f}  "
                  f"n_x={c.get('n_cross', 0)}")
    return cells


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def cells_to_frame(cells: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for kind in PER_LEAD_KINDS:
        for lead in LEADS_12:
            c = cells.get(f"{kind}_{lead}", {})
            rows.append({
                "kind": kind, "lead": lead, "feature": f"{kind}_{lead}",
                "status": c.get("status", "missing"),
                "n_train": c.get("n_train"), "n_in": c.get("n_in"),
                "n_cross": c.get("n_cross"),
                "n_cross_primary4": c.get("n_cross_primary4"),
                "alpha": c.get("alpha"),
                "rho_in": c.get("rho_in"),
                "rho_cross_source": c.get("rho_cross_source"),
                "rho_cross_target_pool": c.get("rho_cross_target_pool"),
                "rho_cross_source_primary4": c.get("rho_cross_source_primary4"),
                "d_rho": c.get("d_rho"),
                "perm_p_cross_source": c.get("perm_p_cross_source"),
                "r2_in": c.get("r2_in"),
                "scaler_sign_disagreement": c.get("scaler_sign_disagreement"),
            })
    return pd.DataFrame(rows)


def _grid(df: pd.DataFrame, col: str) -> np.ndarray:
    g = np.full((len(PER_LEAD_KINDS), len(LEADS_12)), np.nan)
    for i, kind in enumerate(PER_LEAD_KINDS):
        for j, lead in enumerate(LEADS_12):
            v = df.loc[(df["kind"] == kind) & (df["lead"] == lead), col]
            if len(v) and v.iloc[0] is not None:
                g[i, j] = float(v.iloc[0])
    return g


def plot_grids(df: pd.DataFrame, config: str, out_path: Path) -> None:
    panels = [
        ("rho_in", "in-domain (MedalCare test)", "RdBu_r", -1, 1),
        ("rho_cross_source", "cross-domain (PTB-XL)", "RdBu_r", -1, 1),
        ("d_rho", "transfer deficit  rho_in - rho_cross", "Reds", 0, 1),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 9.5))
    for ax, (col, title, cmap, vmin, vmax) in zip(axes, panels):
        g = _grid(df, col)
        im = ax.imshow(g, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(LEADS_12)), LEADS_12)
        ax.set_yticks(range(len(PER_LEAD_KINDS)), PER_LEAD_KINDS)
        ax.set_title(f"{title}   [{config}]", fontsize=10)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                if np.isfinite(g[i, j]):
                    ax.text(j, i, f"{g[i, j]:.2f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if abs(g[i, j]) > 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.suptitle(
        "Per-feature probing map: Spearman rho between ridge probe of the latent "
        "and the measured feature", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", type=str, default=",".join(DEFAULT_CONFIGS),
                    help="Comma-separated encoder configs.")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in CONFIG_LATENT_STEMS]
    if unknown:
        raise SystemExit(f"unknown config(s): {unknown}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "metadata": {
            "configs": configs,
            "per_lead_kinds": list(PER_LEAD_KINDS),
            "leads": list(LEADS_12),
            "n_cells": N_PER_LEAD,
            "primary_metric": "spearman_rho",
            "scaler_modes": list(SCALER_MODES),
            "alphas": ALPHAS.tolist(),
            "n_bootstrap": args.n_boot,
            "n_permutation": args.n_perm,
            "seed": SEED,
            "targets_imputed": False,
            "note": (
                "Targets are never imputed; rows with a NaN target are dropped "
                "per feature and n is recorded per cell. Cross-domain R^2 is "
                "stored under a _DO_NOT_QUOTE suffix because it conflates "
                "information content with cross-domain amplitude calibration."
            ),
        },
        "results": {},
    }

    for config in configs:
        cells = run_config(config, args.n_boot, args.n_perm)
        payload["results"][config] = cells
        df = cells_to_frame(cells)
        csv_path = args.out_dir / f"probe_map_{config}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  saved {csv_path}")
        plot_grids(df, config, args.out_dir / f"grid_{config}.png")

        ok = df[df["status"] == "ok"]
        print(f"  --- {config} summary over {len(ok)} cells ---")
        print(f"      median rho_in    = {ok['rho_in'].median():+.3f}")
        print(f"      median rho_cross = {ok['rho_cross_source'].median():+.3f}")
        print(f"      median d_rho     = {ok['d_rho'].median():+.3f}")
        print(f"      cells with perm p<0.05 cross-domain: "
              f"{int((ok['perm_p_cross_source'] < 0.05).sum())}/{len(ok)}")
        print(f"      scaler sign disagreements: "
              f"{int(ok['scaler_sign_disagreement'].fillna(False).sum())}")

    json_path = args.out_dir / "probe_map.json"
    json_path.write_text(json.dumps(payload, indent=2, default=float),
                         encoding="utf-8")
    print(f"\n[done] wrote {json_path}")


if __name__ == "__main__":
    main()
