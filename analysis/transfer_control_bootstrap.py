"""How stable is the latent-vs-control tie? Bootstrap the 438 evaluation rows.

EXECUTION_LOG Part 13 §13.2 reports a macro-AUROC delta of -0.0449 (Z 0.5674 vs
spatial54 0.6123) against a +/-0.05 band declared before the run. That verdict
lands 0.001 inside the band, on 438 PTB-XL rows, with two of the four territories
at n=32 and n=42. A point estimate that close to a threshold, on samples that
small, is not something to put in a thesis without an interval around it.

This resamples the **evaluation** rows only. Probes are fitted once per arm per
class on the MedalCare rows -- exactly the fitted objects §13.2 scored -- and the
bootstrap perturbs which PTB-XL rows they are scored on. That isolates the
dominant uncertainty (438 target rows) without re-tuning C 1000 times, and it
keeps every draw comparable across arms because all three arms are scored on the
*same* resampled row set in each draw. Paired, not independent: the quantity of
interest is the difference, and pairing removes the shared row-sampling noise
that would otherwise swamp it.

Reported: the delta's 95% percentile interval, and the fraction of draws in which
the control actually wins. Reads only artifacts on disk; touches nothing.

Run::

    python analysis/transfer_control_bootstrap.py
    python analysis/transfer_control_bootstrap.py --n-boot 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
from analysis.domain_signal_structure import OUT_DIR, SEED, load  # noqa: E402
from analysis.transfer_control import (  # noqa: E402
    FEATURE_SETS,
    impute,
    missingness_auroc,
    territory_targets,
)


def fitted_scores(X_med, y_med, X_ptb, classes, y_ptb, Cs, seed=SEED):
    """Per-class decision scores on the PTB-XL rows, from probes fit on MedalCare.

    Returns {class: (scores, y_binary)}. C is tuned exactly as in
    `transfer_control.probe_auc` -- on the MedalCare rows only -- so the fitted
    object here is the one §13.2 scored.
    """
    from sklearn.model_selection import StratifiedKFold
    out = {}
    for cname in classes:
        y_m = (y_med == cname).astype(int)
        y_p = (y_ptb == cname).astype(int)
        if y_m.sum() < 10 or y_p.sum() < 10:
            continue
        best_C, best_cv = 1.0, -np.inf
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for C in Cs:
            fold = []
            for itr, ite in skf.split(X_med, y_m):
                m = LogisticRegression(C=C, max_iter=3000,
                                       class_weight="balanced",
                                       random_state=seed).fit(X_med[itr], y_m[itr])
                fold.append(roc_auc_score(y_m[ite],
                                          m.predict_proba(X_med[ite])[:, 1]))
            if fold and np.mean(fold) > best_cv:
                best_cv, best_C = float(np.mean(fold)), C
        clf = LogisticRegression(C=best_C, max_iter=3000,
                                 class_weight="balanced",
                                 random_state=seed).fit(X_med, y_m)
        out[cname] = (clf.predict_proba(X_ptb)[:, 1], y_p)
    return out


def macro_auc(scored, rows):
    """Macro AUROC over classes on a resampled row index."""
    vals = []
    for s, y in scored.values():
        yy = y[rows]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(roc_auc_score(yy, s[rows]))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="exp8_leadfix_baseline")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--Cs", type=float, nargs="*",
                    default=[0.001, 0.01, 0.1, 1.0, 10.0])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print("=" * 84)
    print(f"Bootstrap of the latent-vs-control tie  [{args.run}]")
    print("=" * 84)

    Z_med_full, _ = load(args.run, "medalcare", "train")
    Z_ptb_full, _ = load(args.run, "ptbxl", "test")
    med_idx, med_terr, ptb_idx, ptb_terr = territory_targets()
    classes = sorted(set(med_terr) & set(ptb_terr))

    reps = {}
    sc = StandardScaler().fit(Z_med_full[med_idx])
    reps["Z"] = (sc.transform(Z_med_full[med_idx]),
                 sc.transform(Z_ptb_full[ptb_idx]))
    for fs, (p_med, p_ptb) in FEATURE_SETS.items():
        fm, fp = REPO_ROOT / p_med, REPO_ROOT / p_ptb
        if not (fm.exists() and fp.exists()):
            continue
        Xm = np.load(fm, allow_pickle=True)["features"][med_idx]
        Xp = np.load(fp, allow_pickle=True)["features"][ptb_idx]
        worst = max(max(missingness_auroc(Xm, (med_terr == c).astype(int)),
                        missingness_auroc(Xp, (ptb_terr == c).astype(int)))
                    for c in classes)
        if worst > 0.55:
            print(f"  ABORT {fs}: missingness AUROC {worst:.4f}")
            continue
        Xm, med = impute(Xm)
        Xp, _ = impute(Xp, med)
        s = StandardScaler().fit(Xm)
        reps[fs] = (s.transform(Xm), s.transform(Xp))

    scored = {}
    for name, (Xm, Xp) in reps.items():
        scored[name] = fitted_scores(Xm, med_terr, Xp, classes, ptb_terr, args.Cs)
        print(f"[fit] {name:<10} dim={Xm.shape[1]:>5}  "
              f"point macro AUROC = "
              f"{macro_auc(scored[name], np.arange(len(ptb_idx))):.4f}")

    n = len(ptb_idx)
    rng = np.random.default_rng(SEED)
    draws = {k: [] for k in scored}
    for _ in range(args.n_boot):
        rows = rng.integers(0, n, size=n)          # same rows for every arm
        for k in scored:
            draws[k].append(macro_auc(scored[k], rows))

    print()
    print(f"{'representation':<14}{'point':>9}{'boot mean':>11}{'95% CI':>20}")
    for k, v in draws.items():
        a = np.array([x for x in v if np.isfinite(x)])
        pt = macro_auc(scored[k], np.arange(n))
        lo, hi = np.percentile(a, [2.5, 97.5])
        print(f"{k:<14}{pt:>9.4f}{a.mean():>11.4f}   [{lo:.4f}, {hi:.4f}]")

    ctrl = [k for k in draws if k != "Z"]
    print()
    for c in ctrl:
        d = np.array(draws["Z"]) - np.array(draws[c])
        d = d[np.isfinite(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p_ctrl_wins = float((d < 0).mean())
        print(f"Z - {c}:  delta {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"P(control wins) = {p_ctrl_wins:.3f}")
        crosses = lo < 0 < hi
        print(f"  interval {'INCLUDES' if crosses else 'EXCLUDES'} zero -> "
              + ("the sign of this comparison is not resolved by 438 rows."
                 if crosses else
                 "the direction is resolved at this sample size."))
        if abs(lo) > 0.05 or abs(hi) > 0.05:
            print("  NOTE: the interval extends beyond the +/-0.05 'tie' band, so")
            print("        'tie' is a statement about the point estimate only.")

    out = args.out or OUT_DIR / f"transfer_control_bootstrap_{args.run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"run": args.run, "seed": SEED, "n_boot": args.n_boot,
               "n_eval_rows": int(n), "classes": classes,
               "point": {k: macro_auc(scored[k], np.arange(n)) for k in scored},
               "ci": {k: list(np.percentile(
                   np.array([x for x in v if np.isfinite(x)]), [2.5, 97.5]))
                   for k, v in draws.items()}}
    for c in ctrl:
        d = np.array(draws["Z"]) - np.array(draws[c])
        d = d[np.isfinite(d)]
        payload[f"delta_Z_minus_{c}"] = {
            "mean": float(d.mean()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_control_wins": float((d < 0).mean())}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
