"""Does the diagonal-CORAL transfer gain survive its controls?

`coral_alignment.py` measured M->P macro-AUC 0.6714 -> 0.7604 (+0.089) on
exp8_leadfix_baseline from per-domain per-coordinate standardisation alone. Three
claims in this investigation have already been written before the test that would
have checked them, and two did not survive (report §11). So this one gets its
controls BEFORE it is written up, not after.

Four ways the +0.089 could be nothing:

  1. ONE RUN, ONE SUBSAMPLE. The number came from a single seed's 1500-sample
     draw. Repeated over seeds, the spread may swallow the effect.
     -> `--seeds`: full re-draw per seed, mean +- std reported.

  2. ONE CHECKPOINT. exp8_leadfix_baseline could be idiosyncratic.
     -> `--runs`: replicate across every run with exported latents.

  3. NOT ABOUT DOMAIN AT ALL. Maybe re-standardising ANY two groups helps, e.g.
     because the probe is simply sensitive to feature scaling.
     -> SPLIT-HALF CONTROL: split PTB-XL at random into two halves, standardise
        each half by its OWN statistics, and transfer into them. There is no
        domain difference between the halves, so any gain here is the operation
        flattering itself and the real gain is the DIFFERENCE.

  4. FREE LUNCH FROM THE TARGET. Diagonal CORAL reads the target's unlabelled
     training latents. Legitimate transductive UDA, but the claim's strength
     depends on how much unlabelled target data it needs -- 50 samples is a
     practical method, 5000 is a much weaker statement.
     -> `--n-target`: sweep the number of unlabelled PTB-XL samples used to
        estimate the scaler, everything else held fixed.

Also reported per-class, because macro-AUC over 3 classes can move 0.09 on one
class alone and NORM has a 6x prevalence shift between the domains -- exactly the
class a scale correction would be expected to rescue.

Writes: outputs/analysis/domain_signal/coral_controls.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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
from scripts.finetune_multilabel import SHARED_LABELS  # noqa: E402


def per_class_auc(Z_tr, Y_tr, Z_te, Y_te, seed=SEED):
    """Per-class held-out AUROC for probes fit on (Z_tr, Y_tr)."""
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


def one_seed(run: str, seed: int, n_sub: int, n_target: int | None):
    """One full re-draw: baseline vs diagonal CORAL vs the split-half control."""
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

    # Diagonal CORAL. The target scaler sees only unlabelled target TRAIN latents,
    # optionally capped at n_target to measure how much of them it needs.
    src_sc = StandardScaler().fit(j_mtr)
    tgt_pool = j_ptr
    if n_target is not None and n_target < len(tgt_pool):
        tgt_pool = tgt_pool[rng.choice(len(tgt_pool), n_target, replace=False)]
    tgt_sc = StandardScaler().fit(tgt_pool)
    cor = per_class_auc(src_sc.transform(j_mtr), Y_mtr,
                        tgt_sc.transform(j_pte), Y_pte, seed)

    # SPLIT-HALF CONTROL. Two random halves of PTB-XL: no domain gap, so
    # per-group standardisation should buy nothing. Transfer is P(half A) ->
    # P(half B) so the probe still faces a group it was not fit on.
    idx = rng.permutation(len(j_ptr))
    hA, hB = idx[: len(idx) // 2], idx[len(idx) // 2:]
    A_tr, YA = j_ptr[hA], Y_ptr[hA]
    B_tr = j_ptr[hB]
    sh_base = per_class_auc(A_tr, YA, j_pte, Y_pte, seed)
    aA = StandardScaler().fit(A_tr)
    aB = StandardScaler().fit(B_tr)
    sh_cor = per_class_auc(aA.transform(A_tr), YA, aB.transform(j_pte), Y_pte, seed)

    return {"baseline": base, "coral": cor,
            "splithalf_baseline": sh_base, "splithalf_coral": sh_cor}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", default=["exp8_leadfix_baseline"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--subsample", type=int, default=1500)
    ap.add_argument("--n-target", type=int, nargs="*", default=None,
                    help="sweep of unlabelled target sample counts for the "
                         "scaler; omit to use all available")
    args = ap.parse_args()

    print("=" * 92)
    print("Does the diagonal-CORAL transfer gain survive its controls?")
    print("=" * 92)

    results = {}
    for run in args.runs:
        try:
            load(run, "ptbxl", "test")
        except Exception as exc:
            print(f"\n[{run}] skipped: {exc}")
            continue

        print(f"\n### {run}   ({len(args.seeds)} seeds)")
        per_seed = [one_seed(run, s, args.subsample, None) for s in args.seeds]

        b = np.array([macro(r["baseline"]) for r in per_seed])
        c = np.array([macro(r["coral"]) for r in per_seed])
        sb = np.array([macro(r["splithalf_baseline"]) for r in per_seed])
        scr = np.array([macro(r["splithalf_coral"]) for r in per_seed])

        gain, sh_gain = c - b, scr - sb
        net = float(np.mean(gain) - np.mean(sh_gain))

        print(f"  baseline M->P      {b.mean():.4f} +- {b.std():.4f}")
        print(f"  diagonal CORAL     {c.mean():.4f} +- {c.std():.4f}")
        print(f"  gain               {gain.mean():+.4f} +- {gain.std():.4f}"
              f"   (min {gain.min():+.4f}, max {gain.max():+.4f})")
        print(f"  split-half control {sh_gain.mean():+.4f} +- {sh_gain.std():.4f}"
              f"   <- same operation where there is NO domain gap")
        print(f"  NET                {net:+.4f}")

        print(f"\n  {'class':>6} {'baseline':>9} {'CORAL':>9} {'delta':>8}")
        cls = {}
        for name in SHARED_LABELS:
            cb = float(np.mean([r["baseline"][name] for r in per_seed]))
            cc = float(np.mean([r["coral"][name] for r in per_seed]))
            cls[name] = {"baseline": cb, "coral": cc, "delta": cc - cb}
            print(f"  {name:>6} {cb:>9.4f} {cc:>9.4f} {cc - cb:>+8.4f}")

        results[run] = {
            "baseline_mean": float(b.mean()), "baseline_std": float(b.std()),
            "coral_mean": float(c.mean()), "coral_std": float(c.std()),
            "gain_mean": float(gain.mean()), "gain_std": float(gain.std()),
            "gain_min": float(gain.min()), "gain_max": float(gain.max()),
            "splithalf_gain_mean": float(sh_gain.mean()),
            "net_gain": net, "per_class": cls, "seeds": args.seeds,
        }

        if args.n_target:
            print(f"\n  unlabelled target samples needed for the scaler:")
            print(f"  {'n':>6} {'M->P':>9} {'gain':>8}")
            sweep = {}
            for n in args.n_target:
                vals = [macro(one_seed(run, s, args.subsample, n)["coral"])
                        for s in args.seeds]
                m = float(np.mean(vals))
                sweep[n] = m
                print(f"  {n:>6} {m:>9.4f} {m - b.mean():>+8.4f}")
            results[run]["n_target_sweep"] = sweep

    print("\n" + "=" * 92)
    if not results:
        print("no runs evaluated")
        return 1

    gains = [r["gain_mean"] for r in results.values()]
    nets = [r["net_gain"] for r in results.values()]
    stds = [r["gain_std"] for r in results.values()]
    print(f"across {len(results)} run(s): mean gain {np.mean(gains):+.4f}, "
          f"mean net of control {np.mean(nets):+.4f}")

    print()
    if np.mean(nets) > 0.03 and np.mean(gains) > 2 * np.mean(stds):
        print("=> THE GAIN IS REAL AND DOMAIN-SPECIFIC. It exceeds its seed spread")
        print("   and survives subtraction of the split-half control, so it is not")
        print("   an artifact of re-standardising groups in general. Per-domain")
        print("   feature scaling recovers cross-domain transfer that every")
        print("   direction-removal method in this project destroyed.")
    elif np.mean(nets) > 0.03:
        print("=> DOMAIN-SPECIFIC BUT NOISY. The control is beaten, but the gain is")
        print("   within ~2 seed-sigma. Report with the spread, not as a headline.")
    elif np.mean(gains) > 0.03:
        print("=> NOT ABOUT DOMAIN. The split-half control gains about as much, so")
        print("   this is a property of re-standardising any group -- probe scale")
        print("   sensitivity, not domain adaptation. Do NOT report it as UDA.")
    else:
        print("=> NO RELIABLE GAIN once seeds are averaged. The single-seed +0.089")
        print("   was a draw. Withdraw it.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "coral_controls.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
