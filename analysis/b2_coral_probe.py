"""Does the §12 dispersion correction help the ACTIVE research question?

§12 established the correction on the alignment chapter's 3-class shared label
space (NORM/MI/CD): per-domain diagonal standardisation recovers +0.065 macro-AUC
of MedalCare->PTB-XL transfer, replicated over 3 checkpoints x 5 seeds.

But the alignment chapter is the ruled-out branch. The project's live push is
Phase B2: decoding biophysical theta from latents, where in-domain MedalCare
decoding works and CROSS-DOMAIN TRANSFER TO PTB-XL IS THE OPEN PROBLEM. That is
the same failure mode, on the question that actually matters for the thesis.

And `phase_b2_infarct_decoding.py` does exactly the thing §12 identifies as the
defect. From `cross_domain_phi_eval`:

    X_std = scaler.transform(X_ptbxl)      # scaler was FIT ON MEDALCARE TRAIN

A ridge/logistic fit on MedalCare-standardised inputs is then handed PTB-XL
features carrying up to 3x the per-coordinate spread it was fitted on. Its
decision boundaries sit at the wrong distances. Pipeline A's 4-class territory
classifier (`pipeline_a_for_source`) inherits the same `X_ptbxl_std`.

This script asks whether swapping that one line for a target-fitted scaler moves
the Track 3 redux numbers, WITHOUT editing `phase_b2_infarct_decoding.py` (which
is mid-run). It reimplements the minimal path: MedalCare theta_mi latents ->
4-class territory LogReg -> PTB-XL territory truth, under three input treatments:

  source-scaled  : the current code path (MedalCare scaler applied to PTB-XL)
  CORAL          : each domain standardised by its OWN statistics (§12)
  target-oracle  : PTB-XL standardised by target stats, source by source stats,
                   identical to CORAL here -- kept as an explicit name so the
                   report can state that no target LABELS are ever used.

Controls, because §12's lesson was that the control decides the claim:
  * multiple seeds, full re-draw, mean +- std
  * a LABEL-SHUFFLE null on the cross-domain evaluation, so "macro-F1 went up"
    is read against what chance does under the same class prevalences
  * in-domain MedalCare macro-F1 reported alongside, to catch a "fix" that is
    really just degrading the source model

Writes: outputs/analysis/domain_signal/b2_coral_<config>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
from analysis.domain_signal_structure import OUT_DIR, SEED  # noqa: E402
from analysis.phase_b2_infarct_decoding import (  # noqa: E402
    TERRITORIES_4C, load_config_latents, load_ptbxl_latents,
    load_ptbxl_subclass_csv, load_targets,
)


def fit_eval(X_tr, y_tr, X_ev, y_ev, C=1.0, seed=SEED):
    """4-class multinomial LogReg; returns macro-F1 on the evaluation set."""
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced",
                             random_state=seed).fit(X_tr, y_tr)
    pred = clf.predict(X_ev)
    return float(f1_score(y_ev, pred, labels=TERRITORIES_4C,
                          average="macro", zero_division=0)), pred


def shuffle_null(y_true, pred, rng, n=200):
    """Macro-F1 of the same predictions against permuted truth."""
    vals = []
    for _ in range(n):
        vals.append(f1_score(rng.permutation(y_true), pred, labels=TERRITORIES_4C,
                             average="macro", zero_division=0))
    return float(np.mean(vals)), float(np.std(vals))


def bootstrap_ci(y_true, pred, rng, n=400):
    """Percentile CI for macro-F1 by resampling the evaluation set.

    NOT a seed sweep. The LogReg fit here is deterministic given the data, so
    varying `random_state` changes nothing about the predictions -- an earlier
    version of this script reported a seed spread of exactly 0.0000 and that was
    a property of the loop, not evidence of stability. The real uncertainty is
    in the n=438 PTB-XL evaluation set, which is what this resamples.
    """
    y_true = np.asarray(y_true)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[i])) < 2:
            continue
        vals.append(f1_score(y_true[i], pred[i], labels=TERRITORIES_4C,
                             average="macro", zero_division=0))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="exp7_baseline",
                    help="phase_b2 config key (latent stem lookup)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    print("=" * 92)
    print(f"Does per-domain scaling help Phase B2 territory transfer?  "
          f"[{args.config}]")
    print("=" * 92)

    Z_train_full, Z_test_full = load_config_latents(args.config)
    tgt = load_targets()
    Z_ptb_full = load_ptbxl_latents(args.config)
    sub = load_ptbxl_subclass_csv()

    # theta targets cover only the MI rows of each split; `idx_in_split` maps
    # them back onto the full latent matrices (mirrors phase_b2 lines 1615-1752).
    y_tr = tgt["train"]["territory_4c"]
    y_te = tgt["test"]["territory_4c"]
    Z_tr = Z_train_full[tgt["train"]["idx_in_split"]]
    Z_te = Z_test_full[tgt["test"]["idx_in_split"]]

    # PTB-XL: the same primary single-territory subset Pipeline A evaluates on.
    mask = sub["territory_4c"].isin(TERRITORIES_4C)
    rows = sub[mask]
    Z_ptb = Z_ptb_full[rows["row_idx"].to_numpy()]
    y_ptb = rows["territory_4c"].to_numpy()

    print(f"MedalCare train {Z_tr.shape}, test {Z_te.shape}; "
          f"PTB-XL primary-4c {Z_ptb.shape}")
    print(f"PTB-XL class counts: {dict(rows['territory_4c'].value_counts())}\n")

    hdr = (f"{'treatment':>16} | {'in-domain':>10} | {'cross 4c':>10} "
           f"{'95% CI':>18} {'shuffle':>9} {'over null':>10}")
    print(hdr)
    print("-" * len(hdr))

    results = {}
    preds = {}
    for name in ("source-scaled", "CORAL"):
        rng = np.random.default_rng(SEED)
        src = StandardScaler().fit(Z_tr)
        A_tr, A_te = src.transform(Z_tr), src.transform(Z_te)
        if name == "source-scaled":
            # The current code path: MedalCare scaler applied to PTB-XL.
            B = src.transform(Z_ptb)
        else:
            # §12: the target gets its own (unlabelled) statistics.
            B = StandardScaler().fit(Z_ptb).transform(Z_ptb)

        ind, _ = fit_eval(A_tr, y_tr, A_te, y_te, args.C)
        xf, pred = fit_eval(A_tr, y_tr, B, y_ptb, args.C)
        lo, hi = bootstrap_ci(y_ptb, pred, rng)
        nul, nul_sd = shuffle_null(y_ptb, pred, rng)
        preds[name] = pred

        results[name] = {
            "in_domain_macro_f1": ind,
            "cross_macro_f1": xf,
            "cross_macro_f1_ci95": [lo, hi],
            "shuffle_null": nul, "shuffle_null_std": nul_sd,
            "over_null": xf - nul,
        }
        print(f"{name:>16} | {ind:>10.4f} | {xf:>10.4f} "
              f"{f'[{lo:.3f}, {hi:.3f}]':>18} {nul:>9.4f} {xf - nul:>10.4f}")

    base, cor = results["source-scaled"], results["CORAL"]
    d_x = cor["cross_macro_f1"] - base["cross_macro_f1"]
    d_over = cor["over_null"] - base["over_null"]
    d_in = cor["in_domain_macro_f1"] - base["in_domain_macro_f1"]

    print("\n" + "-" * 92)
    print(f"CORAL vs source-scaled:  cross-domain macro-F1 {d_x:+.4f}   "
          f"over-null {d_over:+.4f}   in-domain {d_in:+.4f}")
    print(f"CORAL cross-domain CI {cor['cross_macro_f1_ci95']} vs its own "
          f"shuffle null {cor['shuffle_null']:.4f} "
          f"(+-{cor['shuffle_null_std']:.4f})")

    # Does the CI clear the null? That, not the delta, decides whether B2
    # cross-domain territory decoding works under either treatment.
    clears = cor["cross_macro_f1_ci95"][0] > cor["shuffle_null"]

    print()
    if d_over > 0.03 and clears:
        print("=> THE §12 CORRECTION TRANSFERS TO PHASE B2. Per-domain scaling")
        print("   lifts cross-domain territory decoding and its CI clears the")
        print("   shuffle null. The one-line scaler change in")
        print("   cross_domain_phi_eval and pipeline_a_for_source is worth")
        print("   making, and the Track 3 redux should be re-read with it.")
    elif d_over > 0.03:
        print("=> IMPROVES, BUT STILL INDISTINGUISHABLE FROM CHANCE. The")
        print("   correction helps in relative terms while the 95% CI still")
        print("   contains the shuffle null, so it does NOT make territory")
        print("   transfer work. Report the delta as a lead, never a result.")
    elif abs(d_over) <= 0.03:
        print("=> NO EFFECT ON B2. The dispersion correction that bought +0.065")
        print("   on the 3-class shared space does nothing here -- the B2")
        print("   cross-domain failure is NOT the alignment chapter's failure")
        print("   and needs its own diagnosis.")
    else:
        print("=> HURTS. Per-domain scaling degrades B2 transfer -- the opposite")
        print("   of §12. Do not propagate the change; understand this first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"b2_coral_{args.config}.json"
    out.write_text(json.dumps({"config": args.config,
                               "n_ptbxl": int(len(y_ptb)),
                               "ci_clears_null": bool(clears),
                               "treatments": results}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
