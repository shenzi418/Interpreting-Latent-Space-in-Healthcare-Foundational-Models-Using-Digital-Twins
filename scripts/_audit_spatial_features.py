"""Sanity-check the 54-d spatial control against textbook infarct localisation.

Report S15 replaces a spatially-blind control with a per-lead one on the argument
that territory is defined by *which leads* deviate. That argument is only worth
anything if the extracted per-lead features actually behave the way the textbook
says they should on real data. This checks that directly, before the features are
used to argue anything about the latent.

Predictions, stated before looking (these are the standard localisation rules,
not post-hoc pattern-matching):

  1. ANTERIOR infarct  -> pathological Q waves and loss of R-wave amplitude in the
     anteroseptal precordials V1-V4.
  2. INFERIOR infarct  -> pathological Q waves in the inferior limb leads
     II, III, aVF.
  3. The effect must be LEAD-SPECIFIC: anterior-vs-inferior separation should be
     large in V1-V4 and II/III/aVF and near zero in leads that neither territory
     projects onto. A feature set that separates the groups uniformly across all
     12 leads is picking up something global (amplitude scale, noise, heart rate)
     rather than location, and would be no better than the control it replaces.

Reported as Cohen's d (anterior minus inferior) per lead per feature kind, plus a
single AUROC from a leave-one-out logistic probe on the 24 Q/R columns -- enough
to say "this input can represent the task", which is exactly the property the
6-feature control lacked.

This makes NO claim about the latent. It is an instrument check.

Run::

    python scripts/_audit_spatial_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF",
         "V1", "V2", "V3", "V4", "V5", "V6")
ANTERIOR_LEADS = ("V1", "V2", "V3", "V4")
INFERIOR_LEADS = ("II", "III", "aVF")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 3 or b.size < 3:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((a.size - 1) * va + (b.size - 1) * vb) /
                     max(a.size + b.size - 2, 1))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def main() -> int:
    npz = REPO_ROOT / "data" / "ecg_features_spatial_ptbxl_test.npz"
    if not npz.exists():
        print(f"missing {npz}; run scripts/extract_ecg_features_spatial.py first")
        return 1
    d = np.load(npz, allow_pickle=True)
    X = d["features"].astype(np.float64)
    names = [str(x) for x in d["feature_names"]]

    sub = pd.read_csv(REPO_ROOT / "data" / "ptbxl_mi_subclass.csv")
    if len(sub) != X.shape[0]:
        print(f"row mismatch: subclass {len(sub)} vs features {X.shape[0]}")
        return 1
    # `territory_2c` is the anterior-vs-inferior column. NOT `territory_4c` --
    # that one carries the refined labels {Anteroseptal, Anterolateral,
    # Inferior, Inferolateral} and contains no bare "Anterior", so matching on
    # it silently yields an empty anterior group.
    terr = sub["territory_2c"].to_numpy()
    ant = np.flatnonzero(terr == "Anterior")
    inf = np.flatnonzero(terr == "Inferior")
    print(f"PTB-XL fold-10: Anterior n={ant.size}, Inferior n={inf.size}")
    if ant.size < 10 or inf.size < 10:
        print("too few rows to audit")
        return 1

    col = {n: i for i, n in enumerate(names)}
    print("\nCohen's d, Anterior minus Inferior  (per lead, per feature kind)")
    print("negative d on Q_amp = deeper Q waves in the ANTERIOR group\n")
    kinds = ("Q_amp", "R_amp", "ST_J60", "T_amp")
    print(f"{'lead':<6}" + "".join(f"{k:>10}" for k in kinds) + "   region")
    print("-" * 60)
    d_by_kind = {k: {} for k in kinds}
    for lead in LEADS:
        cells = []
        for k in kinds:
            j = col.get(f"{k}_{lead}")
            v = cohens_d(X[ant, j], X[inf, j]) if j is not None else float("nan")
            d_by_kind[k][lead] = v
            cells.append(f"{v:>10.2f}")
        region = ("ANT" if lead in ANTERIOR_LEADS else
                  "INF" if lead in INFERIOR_LEADS else "-")
        print(f"{lead:<6}" + "".join(cells) + f"   {region}")

    print("\nprediction checks")
    print("-" * 60)
    ok = True

    q_ant = np.nanmean([d_by_kind["Q_amp"][l] for l in ANTERIOR_LEADS])
    q_inf = np.nanmean([d_by_kind["Q_amp"][l] for l in INFERIOR_LEADS])
    print(f"  P1  mean Q_amp d over V1-V4      = {q_ant:+.2f}  "
          f"(expect NEGATIVE: anterior group has deeper Q there)")
    print(f"  P2  mean Q_amp d over II/III/aVF = {q_inf:+.2f}  "
          f"(expect POSITIVE: inferior group has deeper Q there)")
    if not (q_ant < 0 < q_inf):
        print("      -> NOT as predicted")
        ok = False
    else:
        print(f"      -> as predicted; separation between the two regions "
              f"= {q_inf - q_ant:.2f} d")

    # P3: lead specificity. Compare |d| on territory-relevant leads against the
    # leads neither territory projects onto.
    other = [l for l in LEADS if l not in ANTERIOR_LEADS + INFERIOR_LEADS]
    rel = np.nanmean([abs(d_by_kind[k][l]) for k in ("Q_amp", "R_amp")
                      for l in ANTERIOR_LEADS + INFERIOR_LEADS])
    irr = np.nanmean([abs(d_by_kind[k][l]) for k in ("Q_amp", "R_amp")
                      for l in other])
    print(f"  P3  mean |d| on territory leads  = {rel:.2f}")
    print(f"      mean |d| on {'/'.join(other)} = {irr:.2f}")
    if rel > irr:
        print(f"      -> lead-specific, ratio {rel / max(irr, 1e-9):.2f}x "
              f"(a global-amplitude artifact would give ~1x)")
    else:
        print("      -> NOT lead-specific; the signal may be global amplitude")
        ok = False

    # Can the input represent the task at all? Cheap CV probe on Q+R columns.
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        cols = [col[f"{k}_{l}"] for k in ("Q_amp", "R_amp") for l in LEADS]
        idx = np.concatenate([ant, inf])
        Xp = X[np.ix_(idx, cols)]
        med = np.nanmedian(Xp, axis=0)
        for j in range(Xp.shape[1]):
            Xp[~np.isfinite(Xp[:, j]), j] = med[j]
        y = np.r_[np.ones(ant.size), np.zeros(inf.size)]
        auc = cross_val_score(
            make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, C=0.1)),
            Xp, y, cv=5, scoring="roc_auc").mean()
        print(f"\n  in-domain 5-fold AUROC from the 24 Q/R columns = {auc:.3f}")
        print("  (this is PTB-XL-only and says nothing about cross-domain "
              "transfer -- it only shows the control CAN represent territory)")
    except Exception as exc:  # noqa: BLE001
        print(f"  probe skipped: {type(exc).__name__}: {exc}")

    print("\n" + ("VERDICT: spatial features behave as the textbook predicts."
                  if ok else
                  "VERDICT: FAILED a prediction -- inspect before using."))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
