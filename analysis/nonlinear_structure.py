"""What separates the domains NONLINEARLY?

Report §11 established that after removing 103 INLP directions -- every linearly
decodable domain direction, and then some -- a gradient-boosted tree still
separates MedalCare from PTB-XL at AUROC 0.9999. So the domain difference is
substantially nonlinear, and nothing in this project has ever looked at it: every
alignment method tried (MMD with a linear-ish critic, INLP, ccMMD) targets first
or second-order structure.

This asks the cheapest useful question about it: WHICH latent coordinates carry
the nonlinear difference, and what does that difference look like?

Three measurements, on the projected (linearly-aligned) latents so that anything
found is genuinely beyond linear reach:

  1. GBDT PERMUTATION IMPORTANCE. Which coordinates does the tree actually use?
     Permutation importance on HELD-OUT data, so it reports what generalises
     rather than what the tree happened to split on. Concentration matters: a
     handful of coordinates would mean the nonlinear difference is localised and
     potentially removable; a flat profile would mean it is diffuse.

  2. MARGINAL SHAPE, per top coordinate. For each, compare the two domains'
     univariate distributions after linear alignment:
       * difference in MEAN      -- should be ~0; INLP removed linear signal
       * ratio of STD            -- a scale difference is nonlinear in the sense
                                    a mean-matching method cannot fix, and is
                                    exactly what a tree splits on
       * difference in SKEW/KURTOSIS -- higher-order shape
     If the means match but the variances do not, the domain difference is a
     SCALE/DISPERSION phenomenon, which names a concrete fix (per-coordinate
     variance matching) that this project has never tried.

  3. SINGLE-COORDINATE AUROC. Can one coordinate alone separate the domains?
     Reported for the top coordinates, folded to >= 0.5. This distinguishes
     "the tree needs interactions" from "there are individually damning axes".

Writes: outputs/analysis/domain_signal/nonlinear_structure_<run>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
from analysis.domain_signal_structure import OUT_DIR, SEED, load  # noqa: E402
from analysis.inlp_controls import inlp_directions, project_out  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="exp8_leadfix_baseline")
    ap.add_argument("--k", type=int, default=103,
                    help="INLP directions removed first, so what remains is "
                         "beyond linear reach (103 = all INLP found on exp8)")
    ap.add_argument("--subsample", type=int, default=1500)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--n-repeats", type=int, default=5)
    args = ap.parse_args()

    print("=" * 92)
    print(f"What separates the domains nonlinearly?  [{args.run}]  "
          f"after removing k={args.k}")
    print("=" * 92)

    Z_mtr, _ = load(args.run, "medalcare", "train")
    Z_ptr, _ = load(args.run, "ptbxl", "train")
    Z_mte, _ = load(args.run, "medalcare", "test")
    Z_pte, _ = load(args.run, "ptbxl", "test")

    sc = StandardScaler().fit(np.vstack([Z_mtr, Z_ptr]))
    Z_mtr, Z_ptr = sc.transform(Z_mtr), sc.transform(Z_ptr)
    Z_mte, Z_pte = sc.transform(Z_mte), sc.transform(Z_pte)

    rng = np.random.default_rng(SEED)

    def sub(Z, n=args.subsample):
        return Z if len(Z) <= n else Z[rng.choice(len(Z), n, replace=False)]

    # Test sets are subsampled too: permutation importance refits nothing but
    # scores 1024 coordinates x n_repeats times, so its cost is linear in the
    # held-out size and 1024*5 predictions over the full test sets is minutes of
    # nothing.
    Z_mtr, Z_ptr = sub(Z_mtr), sub(Z_ptr)
    Z_mte, Z_pte = sub(Z_mte), sub(Z_pte)

    if args.k > 0:
        W = inlp_directions(Z_mtr, Z_ptr, args.k)
        print(f"removing {len(W)} INLP directions ...")
        Z_mtr, Z_ptr = project_out(Z_mtr, W), project_out(Z_ptr, W)
        Z_mte, Z_pte = project_out(Z_mte, W), project_out(Z_pte, W)

    X = np.vstack([Z_mtr, Z_ptr])
    y = np.concatenate([np.zeros(len(Z_mtr)), np.ones(len(Z_ptr))])
    Xe = np.vstack([Z_mte, Z_pte])
    ye = np.concatenate([np.zeros(len(Z_mte)), np.ones(len(Z_pte))])

    clf = HistGradientBoostingClassifier(random_state=SEED).fit(X, y)
    auc = roc_auc_score(ye, clf.predict_proba(Xe)[:, 1])
    print(f"GBDT held-out domain AUROC after linear removal: {auc:.4f}\n")

    print(f"permutation importance ({args.n_repeats} repeats, held out) ...")
    imp = permutation_importance(clf, Xe, ye, n_repeats=args.n_repeats,
                                 random_state=SEED, scoring="roc_auc")
    order = np.argsort(imp.importances_mean)[::-1]
    top = order[:args.top]

    tot = float(np.sum(np.clip(imp.importances_mean, 0, None)))
    top_share = (float(np.sum(np.clip(imp.importances_mean[top], 0, None))) / tot
                 if tot > 0 else float("nan"))
    n_pos = int(np.sum(imp.importances_mean > 1e-6))

    print(f"  {n_pos} coordinates have positive importance; "
          f"top {args.top} carry {top_share:.1%} of the total\n")

    hdr = (f"{'coord':>6} {'import':>8} | {'d_mean':>8} {'std_M':>7} {'std_P':>7} "
           f"{'std_rat':>8} | {'d_skew':>8} {'d_kurt':>8} | {'1d_AUC':>7}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for c in top:
        a, b = Z_mte[:, c], Z_pte[:, c]
        sa, sb = float(np.std(a)), float(np.std(b))
        one_d = roc_auc_score(ye, Xe[:, c])
        r = {
            "coord": int(c),
            "importance": float(imp.importances_mean[c]),
            "mean_medalcare": float(np.mean(a)), "mean_ptbxl": float(np.mean(b)),
            "d_mean": float(np.mean(b) - np.mean(a)),
            "std_medalcare": sa, "std_ptbxl": sb,
            "std_ratio": float(sb / sa) if sa > 0 else float("nan"),
            "d_skew": float(stats.skew(b) - stats.skew(a)),
            "d_kurtosis": float(stats.kurtosis(b) - stats.kurtosis(a)),
            "auc_1d": float(max(one_d, 1 - one_d)),
        }
        rows.append(r)
        print(f"{r['coord']:>6} {r['importance']:>8.4f} | {r['d_mean']:>8.3f} "
              f"{sa:>7.3f} {sb:>7.3f} {r['std_ratio']:>8.3f} | "
              f"{r['d_skew']:>8.3f} {r['d_kurtosis']:>8.3f} | {r['auc_1d']:>7.4f}")

    md = float(np.mean([abs(r["d_mean"]) for r in rows]))
    mr = float(np.mean([r["std_ratio"] for r in rows if np.isfinite(r["std_ratio"])]))
    mx = float(np.max([r["auc_1d"] for r in rows]))

    print("\n" + "-" * 92)
    print(f"top-{args.top} means: mean |d_mean| = {md:.3f}   "
          f"mean std ratio = {mr:.3f}   best single-coordinate AUC = {mx:.4f}")

    print()
    if md < 0.15 and (mr > 1.25 or mr < 0.8):
        print("=> A DISPERSION DIFFERENCE. Means agree after linear alignment but")
        print("   the per-coordinate SCALES do not (ratio %.2f). That is invisible" % mr)
        print("   to mean-matching alignment and is exactly what a tree splits on.")
        print("   Concrete untried fix: per-coordinate variance matching, or a")
        print("   whitening applied per domain before comparison.")
    elif mx > 0.9:
        print("=> LOCALISED. At least one single coordinate separates the domains")
        print("   at AUC %.3f on its own, so the nonlinear difference is not" % mx)
        print("   distributed -- it lives on identifiable axes that could be")
        print("   inspected or removed directly.")
    elif top_share < 0.3:
        print("=> DIFFUSE. No small set of coordinates dominates; the nonlinear")
        print("   separability is spread across many axes. Removing it would")
        print("   likely cost representational capacity, matching the frontier.")
    else:
        print("=> MIXED. See the table; no single description fits.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"nonlinear_structure_{args.run}.json"
    out.write_text(json.dumps(
        {"run": args.run, "k_removed": args.k, "gbdt_auc": float(auc),
         "n_positive_importance": n_pos, "top_share": top_share,
         "top_coords": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
