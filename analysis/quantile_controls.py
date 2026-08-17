"""Does the quantile-normalisation transfer gain survive the controls that CORAL survived?

`quantile_alignment.py` measured M->P macro-AUC 0.6971 -> 0.8477 (+0.1506) on
exp8_leadfix_baseline from per-domain per-coordinate quantile normalisation, with
in-domain class signal UP (+0.0064) rather than damaged. That is more than double
the diagonal-CORAL gain (+0.0655) reported in §12, on the same latents, under the
same transductive setting.

A number that large on a project where nothing has worked is exactly the number
to distrust. §12's gain earned its place by surviving `coral_controls.py`; this
one gets the identical battery, run BEFORE it is written up, so the two are
directly comparable:

  1. ONE SEED / ONE SUBSAMPLE   -> `--seeds`, full re-draw each, mean +- std.
  2. ONE CHECKPOINT             -> `--runs`, replicate across exported runs.
  3. NOT ABOUT DOMAIN AT ALL    -> SPLIT-HALF CONTROL. Two random halves of
     PTB-XL have no domain gap between them, so per-group quantile normalisation
     should buy nothing. Whatever it does buy is the operation flattering itself,
     and the honest gain is the difference. This control matters more here than
     it did for CORAL: a quantile transform is a far more aggressive reshaping of
     the feature distribution, and "the probe simply prefers uniform inputs" is a
     completely plausible explanation for +0.15 that has nothing to do with
     domain adaptation.
  4. FREE LUNCH FROM THE TARGET -> `--n-target` sweep. The transform reads the
     target's unlabelled train latents; needing 50 is a method, needing 5000 is
     a much weaker claim. A quantile transform estimates a whole ECDF per
     coordinate rather than two moments, so it should need MORE target data than
     CORAL did -- if it does not, that is itself worth knowing.

Per-class is reported for the same reason as in §12: macro-AUC over 3 classes can
move on one class alone, and NORM's ~6x prevalence difference between the domains
is exactly what a marginal-matching method is most likely to disturb.

Also reports CORAL side by side, so the comparison "quantile beats CORAL" is made
within one script, one draw, one probe -- not across two runs of two scripts.

Writes: outputs/analysis/domain_signal/quantile_controls.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
from analysis.domain_signal_structure import OUT_DIR, SEED, load  # noqa: E402
from scripts.finetune_multilabel import SHARED_LABELS  # noqa: E402


def per_class_auc(Z_tr, Y_tr, Z_te, Y_te, seed=SEED):
    out = {}
    for c, name in enumerate(SHARED_LABELS):
        y_tr = (Y_tr[:, c] > 0.5).astype(int)
        y_te = (Y_te[:, c] > 0.5).astype(int)
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            out[name] = float("nan")
            continue
        clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                                 random_state=seed).fit(Z_tr, y_tr)
        out[name] = float(roc_auc_score(y_te, clf.predict_proba(Z_te)[:, 1]))
    return out


def macro(d):
    v = [x for x in d.values() if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def qt(fit_on, seed):
    n_q = min(1000, len(fit_on))
    return QuantileTransformer(n_quantiles=n_q, output_distribution="uniform",
                               subsample=10**9, random_state=seed).fit(fit_on)


def one_seed(run: str, seed: int, n_sub: int, n_target: int | None):
    Z_mtr, Y_mtr = load(run, "medalcare", "train")
    Z_ptr, Y_ptr = load(run, "ptbxl", "train")
    Z_mte, Y_mte = load(run, "medalcare", "test")
    Z_pte, Y_pte = load(run, "ptbxl", "test")

    rng = np.random.default_rng(seed)

    def sub(Z, Y, n=n_sub):
        if len(Z) <= n:
            return Z, Y
        i = rng.choice(len(Z), n, replace=False)
        return Z[i], Y[i]

    Z_mtr, Y_mtr = sub(Z_mtr, Y_mtr)
    Z_ptr, Y_ptr = sub(Z_ptr, Y_ptr)
    Z_mte, Y_mte = sub(Z_mte, Y_mte)
    Z_pte, Y_pte = sub(Z_pte, Y_pte)

    sc = StandardScaler().fit(np.vstack([Z_mtr, Z_ptr]))
    j_mtr, j_mte = sc.transform(Z_mtr), sc.transform(Z_mte)
    j_ptr, j_pte = sc.transform(Z_ptr), sc.transform(Z_pte)

    base = per_class_auc(j_mtr, Y_mtr, j_pte, Y_pte, seed)

    # Target-side statistics come from unlabelled target TRAIN latents only,
    # optionally capped to measure how many are actually needed.
    tgt_pool = j_ptr
    if n_target is not None and n_target < len(tgt_pool):
        tgt_pool = tgt_pool[rng.choice(len(tgt_pool), n_target, replace=False)]

    # diagonal CORAL, for a like-for-like comparison inside one draw
    cor = per_class_auc(StandardScaler().fit(j_mtr).transform(j_mtr), Y_mtr,
                        StandardScaler().fit(tgt_pool).transform(j_pte),
                        Y_pte, seed)

    # quantile normalisation
    quant = per_class_auc(qt(j_mtr, seed).transform(j_mtr), Y_mtr,
                          qt(tgt_pool, seed).transform(j_pte), Y_pte, seed)

    # SPLIT-HALF CONTROL: two halves of PTB-XL, no domain gap between them.
    idx = rng.permutation(len(j_ptr))
    hA, hB = idx[: len(idx) // 2], idx[len(idx) // 2:]
    A_tr, YA, B_tr = j_ptr[hA], Y_ptr[hA], j_ptr[hB]
    sh_base = per_class_auc(A_tr, YA, j_pte, Y_pte, seed)
    sh_quant = per_class_auc(qt(A_tr, seed).transform(A_tr), YA,
                             qt(B_tr, seed).transform(j_pte), Y_pte, seed)

    return {"baseline": base, "coral": cor, "quantile": quant,
            "splithalf_baseline": sh_base, "splithalf_quantile": sh_quant}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+",
                    default=["exp8_leadfix_baseline", "exp8_leadfix_ccmmd",
                             "exp7_ccmmd"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--subsample", type=int, default=1500)
    ap.add_argument("--n-target", type=int, nargs="*", default=None)
    args = ap.parse_args()

    print("=" * 96)
    print("Does per-domain QUANTILE normalisation survive its controls?")
    print("=" * 96)

    results = {}
    for run in args.runs:
        try:
            load(run, "ptbxl", "test")
        except Exception as exc:
            print(f"\n[{run}] skipped: {exc}")
            continue

        print(f"\n### {run}   ({len(args.seeds)} seeds)")
        per = [one_seed(run, s, args.subsample, None) for s in args.seeds]

        b = np.array([macro(r["baseline"]) for r in per])
        c = np.array([macro(r["coral"]) for r in per])
        q = np.array([macro(r["quantile"]) for r in per])
        sb = np.array([macro(r["splithalf_baseline"]) for r in per])
        sq = np.array([macro(r["splithalf_quantile"]) for r in per])

        g_c, g_q, g_sh = c - b, q - b, sq - sb
        net = float(g_q.mean() - g_sh.mean())

        print(f"  baseline M->P       {b.mean():.4f} +- {b.std():.4f}")
        print(f"  diagonal CORAL      {c.mean():.4f} +- {c.std():.4f}"
              f"   gain {g_c.mean():+.4f}")
        print(f"  quantile normalise  {q.mean():.4f} +- {q.std():.4f}"
              f"   gain {g_q.mean():+.4f} +- {g_q.std():.4f}"
              f"   (min {g_q.min():+.4f}, max {g_q.max():+.4f})")
        print(f"  split-half control  {g_sh.mean():+.4f} +- {g_sh.std():.4f}"
              f"   <- same operation, NO domain gap")
        print(f"  NET                 {net:+.4f}")
        print(f"  quantile - CORAL    {float(q.mean() - c.mean()):+.4f}")

        print(f"\n  {'class':>6} {'baseline':>9} {'CORAL':>9} {'quantile':>9} {'q-delta':>8}")
        cls = {}
        for name in SHARED_LABELS:
            cb = float(np.mean([r["baseline"][name] for r in per]))
            cc = float(np.mean([r["coral"][name] for r in per]))
            cq = float(np.mean([r["quantile"][name] for r in per]))
            cls[name] = {"baseline": cb, "coral": cc, "quantile": cq,
                         "delta": cq - cb}
            print(f"  {name:>6} {cb:>9.4f} {cc:>9.4f} {cq:>9.4f} {cq - cb:>+8.4f}")

        results[run] = {
            "baseline_mean": float(b.mean()), "baseline_std": float(b.std()),
            "coral_mean": float(c.mean()), "coral_gain": float(g_c.mean()),
            "quantile_mean": float(q.mean()), "quantile_std": float(q.std()),
            "quantile_gain_mean": float(g_q.mean()),
            "quantile_gain_std": float(g_q.std()),
            "quantile_gain_min": float(g_q.min()),
            "quantile_gain_max": float(g_q.max()),
            "splithalf_gain_mean": float(g_sh.mean()),
            "net_gain": net,
            "quantile_minus_coral": float(q.mean() - c.mean()),
            "per_class": cls, "seeds": args.seeds,
        }

        if args.n_target:
            print(f"\n  unlabelled target samples needed for the transform:")
            print(f"  {'n':>6} {'M->P':>9} {'gain':>8}")
            sweep = {}
            for n in args.n_target:
                vals = [macro(one_seed(run, s, args.subsample, n)["quantile"])
                        for s in args.seeds]
                m = float(np.mean(vals))
                sweep[n] = m
                print(f"  {n:>6} {m:>9.4f} {m - b.mean():>+8.4f}")
            results[run]["n_target_sweep"] = sweep

    print("\n" + "=" * 96)
    if not results:
        print("no runs evaluated")
        return 1

    gains = [r["quantile_gain_mean"] for r in results.values()]
    nets = [r["net_gain"] for r in results.values()]
    stds = [r["quantile_gain_std"] for r in results.values()]
    vs_coral = [r["quantile_minus_coral"] for r in results.values()]
    print(f"across {len(results)} run(s): mean quantile gain {np.mean(gains):+.4f}, "
          f"net of control {np.mean(nets):+.4f}, "
          f"vs CORAL {np.mean(vs_coral):+.4f}")

    print()
    if np.mean(nets) > 0.05 and np.mean(gains) > 2 * np.mean(stds):
        print("=> THE GAIN IS REAL, DOMAIN-SPECIFIC, AND LARGER THAN CORAL'S.")
        print("   Matching the full per-coordinate marginal -- not just its first")
        print("   two moments -- recovers substantially more cross-domain transfer,")
        print("   survives the split-half control, and exceeds its seed spread.")
        print("   §12 should be restated: the correction is MARGINAL matching, and")
        print("   CORAL is the weak two-moment special case of it.")
    elif np.mean(nets) > 0.05:
        print("=> DOMAIN-SPECIFIC BUT NOISY. Beats the control; gain is within")
        print("   ~2 seed-sigma. Report with the spread, not as a headline.")
    elif np.mean(gains) > 0.05:
        print("=> NOT ABOUT DOMAIN. The split-half control gains about as much, so")
        print("   the probe simply prefers quantile-shaped inputs. This is feature")
        print("   preprocessing, NOT domain adaptation, and must not be reported")
        print("   as the latter.")
    else:
        print("=> NO RELIABLE GAIN once seeds are averaged. Withdraw the +0.15.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "quantile_controls.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
