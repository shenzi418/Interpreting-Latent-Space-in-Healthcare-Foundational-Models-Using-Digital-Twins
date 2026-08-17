"""Does the M->P transfer number require the LATENT, or would any ECG input do it?

*** STATUS: the 3-class form of this comparison is NOT COMPUTABLE from the
feature files on disk, and the first version of this script produced a leaked
result. Read S1 below before changing anything. ***

Report S6/S16 Rank 5 flags the one worry that the existing controls in
`transfer_reality.py` cannot address. Those controls ask whether the 0.70 macro
M->P AUROC beats a label-shuffle null and a random-direction floor -- it does, by
+0.22 and +0.13 -- and conclude "transfer is real". Both nulls are computed
*inside the latent space*, so they establish that the number is not an artifact of
the evaluation. Neither establishes that it says anything about the foundation
model's representation. The missing comparison is a different INPUT, not a
different null.

S1. WHY THE OBVIOUS IMPLEMENTATION LEAKS
----------------------------------------
`extract_ecg_features*.py` processes **MI rows only** (`_run(rows, mi_idx, ...)`)
and leaves every other row entirely NaN. That is correct for `phase_b2`, which
selects MI rows by index and never sees the rest. It is fatal for any analysis
that spans classes:

  * median-imputing the full matrix replaces every non-MI row with the SAME
    constant vector, so "is this row the constant" IS the MI label;
  * measured directly, the finiteness indicator alone gives MI AUROC 0.847
    (MedalCare) and 0.863 (PTB-XL) -- no features involved;
  * the first version of this script therefore read spatial54 MI M->P = 0.9918
    against a P->P ceiling of 0.9928, and printed shuffle nulls BELOW 0.5.

Full write-up: EXECUTION_LOG Part 12. Both symptoms are worth memorising: a
cross-domain number near the in-domain ceiling, and a permutation null that is not
centred on 0.5.

So this script now runs ONLY on rows where the features genuinely exist, and the
question it answers changes accordingly -- from 3-class transfer (not computable)
to **MI-territory transfer** (computable, and the comparison Rank 1 wants anyway).
The 3-class form is not reachable by any flag: reproducing it would require
re-extracting features on all rows, not a switch here.

S2. THE GUARD
-------------
`missingness_auroc()` scores the finiteness indicator against the label before any
model is fitted, and aborts if it beats chance. That check is two lines and would
have caught the original defect before the probe ran; it is now the first thing
this script does, on every arm, every time.

Three representations, identical protocol, identical rows:

  * `Z`         -- the 1024-d latent (the claim under test)
  * `global6`   -- the original NeuroKit2 control: 4 global scalars, one ST
                   average over V2-V6, one lead-II T amplitude
  * `spatial54` -- 48 per-lead ST/Q/R/T columns plus those 6 (strict superset)

Outcomes, declared before running so the result cannot be read selectively:

  * **Z clearly above both controls** -> the transfer claim survives as a
    statement about the latent.
  * **Controls match Z** -> the number measures the task, not the representation,
    and S6's framing must be rewritten -- exactly as S15 rewrote S14.6.
  * **Controls beat Z** -> the latent is worse than hand-crafted features at the
    thing the report leads with, and that is the finding.

Reads only artifacts already on disk. Touches no file any in-flight driver writes.

Run::

    python analysis/transfer_control.py
    python analysis/transfer_control.py --run exp8_leadfix_ccmmd
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
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

FEATURE_SETS = {
    "global6": ("data/ecg_features_medalcare_train.npz",
                "data/ecg_features_ptbxl_test.npz"),
    "spatial54": ("data/ecg_features_spatial_medalcare_train.npz",
                  "data/ecg_features_spatial_ptbxl_test.npz"),
}
PTBXL_SUBCLASS = REPO_ROOT / "data" / "ptbxl_mi_subclass.csv"
THETA_TRAIN = REPO_ROOT / "data" / "theta_mi_train.npz"


def missingness_auroc(X: np.ndarray, y: np.ndarray) -> float:
    """AUROC of the FINITENESS INDICATOR alone, before any model is fitted.

    This is the guard the first version of this script lacked. If a row's
    missingness pattern predicts the label, then median-imputing and classifying
    leaks that label no matter how good the imputer is -- the model reads "this
    row is the imputed constant" straight off the matrix.

    Here it beat chance by a wide margin (0.847 / 0.863 for MI) because the
    feature extractors only ever processed MI rows. Two lines, run on every arm,
    every time. See EXECUTION_LOG Part 12.
    """
    ind = np.isfinite(X).all(axis=1).astype(float)
    if len(np.unique(y)) < 2 or len(np.unique(ind)) < 2:
        return 0.5
    a = roc_auc_score(y, ind)
    return float(max(a, 1 - a))


def probe_auc(X_tr, y_tr, X_te, y_te, seed=SEED, Cs=None):
    """Logistic probe, held-out AUROC. Matches `transfer_reality.probe_auc`.

    With `Cs`, C is chosen by 5-fold CV **on the training (MedalCare) rows only**
    before the final fit. That is not a nicety, it is the difference between a
    representation comparison and a capacity comparison: Z is 1024-d over 5347
    rows, the controls are 54-d and 6-d, and at scikit-learn's default C=1.0 the
    latent arm is the only one running under-regularised. `tier1` fitting the
    same rows for the same task selects best_C=0.01 for Z[1024d] -- two orders of
    magnitude off the default -- so the default handicaps Z specifically.

    Target rows are never consulted in selection, so this adds no leakage.
    Returns (auroc, chosen_C); chosen_C is None when Cs is None.
    """
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return float("nan"), None
    best_C = None
    if Cs:
        best_C, best_cv = None, -np.inf
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for C in Cs:
            fold = []
            for itr, ite in skf.split(X_tr, y_tr):
                if len(np.unique(y_tr[itr])) < 2 or len(np.unique(y_tr[ite])) < 2:
                    continue
                m = LogisticRegression(C=C, max_iter=3000,
                                       class_weight="balanced",
                                       random_state=seed).fit(X_tr[itr], y_tr[itr])
                fold.append(roc_auc_score(y_tr[ite],
                                          m.predict_proba(X_tr[ite])[:, 1]))
            if fold and np.mean(fold) > best_cv:
                best_cv, best_C = float(np.mean(fold)), C
    clf = LogisticRegression(C=best_C if best_C else 1.0, max_iter=3000,
                             class_weight="balanced",
                             random_state=seed).fit(X_tr, y_tr)
    return float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])), best_C


def impute(X: np.ndarray, med=None):
    """Median-impute residual non-finite cells, train medians reused on test.

    After the MI-row restriction the remaining NaN are genuine per-record
    delineation failures (28/5347 MedalCare train, 0/438 PTB-XL primary) rather
    than never-processed rows. The median MUST come from the training rows --
    computing it per-matrix would let PTB-XL inform its own imputation, a mild
    test-time leak that would flatter the control arm.
    """
    X = X.astype(np.float64, copy=True)
    if med is None:
        med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = np.take(med, np.nonzero(bad)[1])
    return X, med


def territory_targets():
    """MI-territory labels on the rows where features actually exist.

    MedalCare territory comes from `theta_mi_train.npz` (derived from phi, per
    `build_medalcare_isch_targets.py`); PTB-XL from the `territory_4c` column of
    the MI-subclass CSV. Both index exactly the row sets the feature extractors
    processed, which is what makes the comparison leak-free.
    """
    th = np.load(THETA_TRAIN, allow_pickle=True)
    med_idx = th["idx_in_split"]
    med_terr = np.array([str(x) for x in th["territory_4c"]])

    sub = pd.read_csv(PTBXL_SUBCLASS)
    ptb_terr_all = sub["territory_4c"].to_numpy()
    ptb_idx = np.flatnonzero(pd.notna(ptb_terr_all))
    ptb_terr = np.array([str(x) for x in ptb_terr_all[ptb_idx]])
    return med_idx, med_terr, ptb_idx, ptb_terr


def evaluate(name, X_med, y_med, X_ptb, y_ptb, classes, half_a, half_b,
             n_shuffle, n_random, rng_seed=SEED, Cs=None):
    """Four measurements for one representation, one-vs-rest per territory."""
    rows = {}
    D = X_med.shape[1]
    g = np.random.default_rng(rng_seed + 7)
    R = g.normal(size=(n_random, D))
    R /= np.linalg.norm(R, axis=1, keepdims=True)

    for cname in classes:
        y_m = (y_med == cname).astype(int)
        y_p = (y_ptb == cname).astype(int)
        if y_m.sum() < 10 or y_p.sum() < 10:
            print(f"  {name:<10} {cname:<14} skipped (n_med={int(y_m.sum())}, "
                  f"n_ptb={int(y_p.sum())})")
            continue

        transfer, csel = probe_auc(X_med, y_m, X_ptb, y_p, Cs=Cs)
        ceiling, _ = probe_auc(X_ptb[half_a], y_p[half_a],
                               X_ptb[half_b], y_p[half_b], Cs=Cs)

        shuf = []
        for rep in range(n_shuffle):
            gs = np.random.default_rng(rng_seed + 100 + rep)
            shuf.append(probe_auc(X_med, gs.permutation(y_m), X_ptb, y_p)[0])
        shuf_mean = float(np.nanmean(shuf))

        aucs = [roc_auc_score(y_p, X_ptb @ r) for r in R]
        rand_mean = float(np.mean([max(a, 1 - a) for a in aucs]))

        rows[cname] = {"transfer_m2p": transfer, "shuffle_null": shuf_mean,
                       "random_floor": rand_mean, "ptbxl_ceiling": ceiling,
                       "lift_over_shuffle": transfer - shuf_mean,
                       "selected_C": csel,
                       "prevalence_medalcare": float(y_m.mean()),
                       "prevalence_ptbxl": float(y_p.mean())}
        print(f"  {name:<10} {cname:<14} M->P={transfer:>7.4f}  "
              f"shuf={shuf_mean:>6.4f}  rand={rand_mean:>6.4f}  "
              f"P->P={ceiling:>6.4f}  lift={transfer - shuf_mean:>+6.3f}"
              + (f"  C={csel:g}" if csel else ""))

    def mean(k):
        v = [x[k] for x in rows.values() if np.isfinite(x[k])]
        return float(np.mean(v)) if v else float("nan")

    macro = {k: mean(k) for k in ("transfer_m2p", "shuffle_null",
                                  "random_floor", "ptbxl_ceiling")}
    macro["lift_over_shuffle"] = macro["transfer_m2p"] - macro["shuffle_null"]
    macro["fraction_of_ceiling"] = (
        macro["transfer_m2p"] / macro["ptbxl_ceiling"]
        if np.isfinite(macro["ptbxl_ceiling"]) and macro["ptbxl_ceiling"] > 0
        else float("nan"))
    print(f"  {name:<10} {'MACRO':<14} M->P={macro['transfer_m2p']:.4f}  "
          f"shuf={macro['shuffle_null']:.4f}  "
          f"rand={macro['random_floor']:.4f}  "
          f"P->P={macro['ptbxl_ceiling']:.4f}  "
          f"({macro['fraction_of_ceiling']:.3f} of ceiling)")
    print()
    return {"classes": rows, "macro": macro}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="exp8_leadfix_baseline")
    ap.add_argument("--n-shuffle", type=int, default=5)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--max-missingness-auroc", type=float, default=0.55,
                    help="Abort an arm whose finiteness indicator predicts the "
                         "label better than this. See S2 in the module docstring.")
    ap.add_argument("--Cs", type=float, nargs="*",
                    default=[0.001, 0.01, 0.1, 1.0, 10.0],
                    help="Logistic C grid, tuned per arm by 5-fold CV on the "
                         "MedalCare rows only. Pass none to force C=1.0 and "
                         "reproduce the untuned (capacity-confounded) run.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print("=" * 88)
    print(f"MI-territory M->P transfer: latent vs controls  [{args.run}]")
    print("=" * 88)

    Z_med_full, _ = load(args.run, "medalcare", "train")
    Z_ptb_full, _ = load(args.run, "ptbxl", "test")
    med_idx, med_terr, ptb_idx, ptb_terr = territory_targets()
    classes = sorted(set(med_terr) & set(ptb_terr))
    print(f"[rows] MedalCare MI {len(med_idx)}, PTB-XL territory-labelled "
          f"{len(ptb_idx)}")
    print(f"[classes] shared: {classes}")
    print("[prev] MedalCare "
          + str({c: int((med_terr == c).sum()) for c in classes}))
    print("[prev] PTB-XL    "
          + str({c: int((ptb_terr == c).sum()) for c in classes}))

    rs = np.random.default_rng(SEED + 3)
    perm = rs.permutation(len(ptb_idx))
    half_a, half_b = perm[:len(perm) // 2], perm[len(perm) // 2:]
    print(f"[ceiling] PTB-XL half-split {len(half_a)} / {len(half_b)}")

    reps = {}
    sc = StandardScaler().fit(Z_med_full[med_idx])
    reps["Z"] = (sc.transform(Z_med_full[med_idx]),
                 sc.transform(Z_ptb_full[ptb_idx]))

    print()
    print("[guard] AUROC of the finiteness indicator alone (must be ~0.5):")
    for fs, (p_med, p_ptb) in FEATURE_SETS.items():
        fm, fp = REPO_ROOT / p_med, REPO_ROOT / p_ptb
        if not (fm.exists() and fp.exists()):
            print(f"  [skip] {fs}: missing {fm.name} or {fp.name}")
            continue
        Xm_all = np.load(fm, allow_pickle=True)["features"]
        Xp_all = np.load(fp, allow_pickle=True)["features"]
        if (Xm_all.shape[0] != Z_med_full.shape[0]
                or Xp_all.shape[0] != Z_ptb_full.shape[0]):
            print(f"  [skip] {fs}: row mismatch vs latents")
            continue
        Xm, Xp = Xm_all[med_idx], Xp_all[ptb_idx]

        worst = 0.0
        for cname in classes:
            worst = max(worst,
                        missingness_auroc(Xm, (med_terr == cname).astype(int)),
                        missingness_auroc(Xp, (ptb_terr == cname).astype(int)))
        flag = "OK" if worst <= args.max_missingness_auroc else "LEAK"
        print(f"  {fs:<10} worst over classes = {worst:.4f}  [{flag}]")
        if worst > args.max_missingness_auroc:
            print(f"  -> ABORT {fs}: missingness predicts the label; any result "
                  "from this arm would be leaked. See EXECUTION_LOG Part 12.")
            continue

        Xm, med = impute(Xm)
        Xp, _ = impute(Xp, med)
        s = StandardScaler().fit(Xm)
        reps[fs] = (s.transform(Xm), s.transform(Xp))

    print()
    results = {}
    for name, (Xm, Xp) in reps.items():
        results[name] = evaluate(name, Xm, med_terr, Xp, ptb_terr, classes,
                                 half_a, half_b, args.n_shuffle, args.n_random,
                                 Cs=args.Cs)

    print("=" * 88)
    hdr = (f"{'representation':<12}{'dim':>6}{'M->P':>9}{'shuffle':>9}"
           f"{'P->P':>9}{'frac':>8}")
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        m = r["macro"]
        print(f"{name:<12}{reps[name][0].shape[1]:>6}"
              f"{m['transfer_m2p']:>9.4f}{m['shuffle_null']:>9.4f}"
              f"{m['ptbxl_ceiling']:>9.4f}{m['fraction_of_ceiling']:>8.3f}")

    z = results["Z"]["macro"]["transfer_m2p"]
    ctrl = {k: v["macro"]["transfer_m2p"] for k, v in results.items() if k != "Z"}
    if ctrl:
        best_name = max(ctrl, key=ctrl.get)
        best = ctrl[best_name]
        print()
        print(f"latent {z:.4f}  vs  best control ({best_name}) {best:.4f}  "
              f"delta {z - best:+.4f}")
        if z - best > 0.05:
            print("=> LATENT WINS on MI-territory transfer.")
        elif abs(z - best) <= 0.05:
            print("=> TIE. Hand-crafted features match the latent; the number")
            print("   measures the task, not the representation.")
        else:
            print("=> CONTROL WINS. Report it as such.")
    else:
        print()
        print("=> No control arm survived the missingness guard. The "
              "latent-vs-control question is NOT answered by this run.")

    out = args.out or OUT_DIR / f"transfer_control_{args.run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"run": args.run, "seed": SEED, "task": "MI territory (4c) one-vs-rest",
         "C_grid": args.Cs,
         "classes": classes,
         "dims": {k: int(v[0].shape[1]) for k, v in reps.items()},
         "results": results}, indent=2), encoding="utf-8")
    print()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
