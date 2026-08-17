"""Decisive control for the probing map: WHY does ST fail to transfer?

`analysis/probe_feature_map.py` shows a striking pattern -- the latent encodes
MedalCare's ST segment better than any other measurement (median rho_in = 0.80)
yet that encoding is worth essentially nothing on PTB-XL (median rho_cross =
0.15, and 20 of the 24 non-significant cells across all five encoders are
ST_J60). Before that can be claimed as a sim->real transfer failure, two
alternative explanations have to be excluded:

  (A) "ST is unmeasurable on real ECG."  If the J+60ms reading is dominated by
      noise in PTB-XL, a low rho says nothing about the latent.
  (B) "The latent never encodes real ST."  Distinct from (A) and from a transfer
      failure: the representation might simply not carry the quantity for real
      inputs, in which case there is no MedalCare-specific story to tell.

The discriminating experiment is a probe fit **within the real domain**: 5-fold
CV ridge on PTB-XL latents -> PTB-XL feature, scored out-of-fold. Call it
rho_real_cv. Then

  rho_real_cv high, rho_cross low   -> the latent DOES encode real ST; the
                                       MedalCare-fit readout direction is what
                                       fails. A subspace mismatch, i.e. a
                                       genuine sim->real transfer failure.
  rho_real_cv low,  rho_cross low   -> (A) or (B); no transfer claim available.

**Sample size is the trap here.** The within-real probe trains on ~439 rows in
1024 dimensions while the MedalCare probe trains on 5347. A low rho_real_cv
would then be a power artifact, not evidence. So this script also refits the
in-domain probe on MedalCare subsampled to the *same* training size, averaged
over several draws (`rho_in_matched`), which is the only fair reference.

A third column, `rho_feat_terr_*`, records how strongly each feature relates to
the 4-class territory label univariately in each domain -- if a feature carries
no territory information in PTB-XL, the latent failing to encode it there is
irrelevant to the pre-registered endpoint either way.

Outputs
-------
    outputs/analysis/probe_map/probe_control.csv
    outputs/analysis/probe_map/probe_control.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from analysis.phase_b2_infarct_decoding import SEED, derive_rng  # noqa: E402
from analysis.probe_feature_map import (  # noqa: E402
    ALPHAS,
    LEADS_12,
    N_PER_LEAD,
    OUT_DIR,
    PER_LEAD_KINDS,
    _spearman,
    build_design,
)

TERR4 = ("Anteroseptal", "Anterolateral", "Inferior", "Inferolateral")


def _eta_sq(y: np.ndarray, groups: np.ndarray) -> float:
    """Univariate effect size of a categorical label on a continuous feature.

    Ratio of between-group to total sum of squares (eta^2). 0 = the feature says
    nothing about territory; 1 = territory determines it exactly.

    Rows without a valid territory label are dropped: PTB-XL carries NaN for the
    MI rows outside the 4-class primary subset, and mixing those into the group
    array makes it a str/float mix that will not sort.
    """
    groups = np.asarray(groups, dtype=object)
    valid = np.array([isinstance(g, str) and g in TERR4 for g in groups])
    m = np.isfinite(y) & valid
    y, groups = y[m], groups[m].astype(str)
    if y.size < 8:
        return float("nan")
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot == 0.0:
        return float("nan")
    ss_b = 0.0
    for g in np.unique(groups):
        yg = y[groups == g]
        if yg.size:
            ss_b += yg.size * (yg.mean() - y.mean()) ** 2
    return float(ss_b / ss_tot)


def _cv_spearman(X: np.ndarray, y: np.ndarray, n_splits: int,
                 rng: np.random.Generator) -> tuple[float, int, int]:
    """Out-of-fold Spearman for a ridge probe fit inside the given domain."""
    m = np.isfinite(y)
    Xf, yf = X[m], y[m]
    n = yf.size
    if n < 5 * n_splits:
        return float("nan"), n, 0
    seed = int(rng.integers(0, 2**31 - 1))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan)
    n_tr = 0
    for tr, te in kf.split(Xf):
        n_tr = tr.size
        sc = StandardScaler().fit(Xf[tr])
        mdl = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xf[tr]), yf[tr])
        oof[te] = mdl.predict(sc.transform(Xf[te]))
    return _spearman(yf, oof), n, n_tr


def _matched_n_in_domain(Z_tr: np.ndarray, y_tr: np.ndarray,
                         Z_te: np.ndarray, y_te: np.ndarray,
                         n_train: int, n_draws: int,
                         rng: np.random.Generator) -> float:
    """In-domain rho with the training set subsampled to `n_train` rows.

    Without this, rho_real_cv (trained on ~439 rows) would be compared against a
    probe trained on 5347 -- and any gap could be explained by power alone.
    """
    m_tr, m_te = np.isfinite(y_tr), np.isfinite(y_te)
    Xa, ya = Z_tr[m_tr], y_tr[m_tr]
    Xb, yb = Z_te[m_te], y_te[m_te]
    if ya.size < n_train or yb.size < 3 or n_train < 20:
        return float("nan")
    vals = []
    for _ in range(n_draws):
        idx = rng.choice(ya.size, size=n_train, replace=False)
        sc = StandardScaler().fit(Xa[idx])
        mdl = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xa[idx]), ya[idx])
        vals.append(_spearman(yb, mdl.predict(sc.transform(Xb))))
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def run(config: str, n_splits: int, n_draws: int) -> pd.DataFrame:
    print(f"\n=== control: {config} ===")
    d = build_design(config)
    terr_px = np.asarray(
        pd.read_csv(REPO_ROOT / "data" / "ptbxl_mi_subclass.csv")["territory_4c"]
    )
    # MedalCare territory labels, aligned to the theta rows used by build_design.
    from analysis.phase_b2_infarct_decoding import load_targets  # noqa: E402
    terr_mc = load_targets()["test"]["territory_4c"]

    rows = []
    for j in range(N_PER_LEAD):
        kind = PER_LEAD_KINDS[j // len(LEADS_12)]
        lead = LEADS_12[j % len(LEADS_12)]
        name = f"{kind}_{lead}"
        rng = derive_rng("probe_control", config, name, seed=SEED)

        rho_real, n_real, n_tr_real = _cv_spearman(
            d["Z_px"], d["F_px"][:, j], n_splits, rng
        )
        rho_match = _matched_n_in_domain(
            d["Z_tr"], d["F_tr"][:, j], d["Z_te"], d["F_te"][:, j],
            n_tr_real, n_draws, rng,
        )
        rows.append({
            "config": config, "kind": kind, "lead": lead, "feature": name,
            "rho_real_cv": rho_real,
            "n_real": n_real, "n_train_real": n_tr_real,
            "rho_in_matched_n": rho_match,
            "eta2_terr_medalcare": _eta_sq(d["F_te"][:, j], terr_mc),
            "eta2_terr_ptbxl": _eta_sq(d["F_px"][:, j], terr_px),
            "std_medalcare": float(np.nanstd(d["F_te"][:, j])),
            "std_ptbxl": float(np.nanstd(d["F_px"][:, j])),
        })
        r = rows[-1]
        print(f"  {name:12s} rho_real_cv={r['rho_real_cv']:+.3f}  "
              f"rho_in@n={r['rho_in_matched_n']:+.3f}  "
              f"eta2_MC={r['eta2_terr_medalcare']:.3f}  "
              f"eta2_PX={r['eta2_terr_ptbxl']:.3f}  n_real={n_real}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", type=str, default="exp8_leadfix_baseline")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-draws", type=int, default=5,
                    help="Subsample draws for the matched-n in-domain reference.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    df = pd.concat([run(c, args.n_splits, args.n_draws) for c in configs])

    csv = args.out_dir / "probe_control.csv"
    df.to_csv(csv, index=False)
    print(f"\nsaved {csv}")

    # Merge against the main map so the four-way comparison is in one place.
    merged: Dict[str, object] = {}
    for c in configs:
        mp = args.out_dir / f"probe_map_{c}.csv"
        if not mp.exists():
            continue
        m = pd.read_csv(mp)[["feature", "rho_in", "rho_cross_source", "d_rho"]]
        j = df[df.config == c].merge(m, on="feature", how="left")
        merged[c] = j.to_dict(orient="records")
        print(f"\n=== {c}: median by physiology kind ===")
        agg = j.groupby("kind")[
            ["rho_in", "rho_in_matched_n", "rho_cross_source", "rho_real_cv",
             "eta2_terr_medalcare", "eta2_terr_ptbxl"]
        ].median().round(3)
        print(agg.loc[list(PER_LEAD_KINDS)])

    (args.out_dir / "probe_control.json").write_text(
        json.dumps({"metadata": {"configs": configs, "n_splits": args.n_splits,
                                 "n_draws": args.n_draws, "seed": SEED},
                    "results": merged}, indent=2, default=float),
        encoding="utf-8")
    print(f"\n[done] wrote {args.out_dir / 'probe_control.json'}")


if __name__ == "__main__":
    main()
