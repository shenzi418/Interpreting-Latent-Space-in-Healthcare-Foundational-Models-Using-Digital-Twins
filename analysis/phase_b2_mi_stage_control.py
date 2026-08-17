"""Stage 4.1 of the 2026-08-10 audit: does MI *stage* explain the residual sim-to-real gap?

MedalCare-XL's `isch[*]` block simulates ACUTE ischemia -- an active injury current
producing ST-segment deviation. PTB-XL's MI population is overwhelmingly CHRONIC:
of the rows carrying a resolvable 4-class territory, `infarction_stadium1` is
Stadium II-III or III for ~1459 and Stadium I / I-II for only ~97 (all 10 folds).
Chronic infarcts present as Q waves and lost R-wave progression, not ST elevation.
So the cross-domain territory classifier may be failing not because the latent
geometry is misaligned but because it is being asked to recognise a *different
electrophysiological phenomenon* than the one it was trained on.

This script tests that directly. It reuses the trained Pipeline-A 4-class
classifier from `phase_b2_infarct_decoding` (same fit, same standardisation) and
scores it separately on the acute and chronic PTB-XL strata.

    H1 (stage explains the gap): macro-F1(acute) >> macro-F1(chronic).
    H0 (stage is irrelevant):    the two strata are within noise of each other.

POWER WARNING, measured before running: the acute stratum is TINY. In the official
test fold (10) there are 9 acute rows with a 4c territory. Even pooling all ten
folds gives 97. A 4-class macro-F1 on n=9 is uninterpretable, and pooling folds
means scoring on rows the encoder saw during training. This script therefore:

  1. reports the exact stratum sizes FIRST and refuses to emit a headline
     comparison when the acute stratum is below --min-stratum (default 30);
  2. offers `--folds` so the acute stratum can be widened, but tags any run
     using folds other than 10 as `encoder_contaminated: true` in the JSON, since
     folds 1-9 were the encoder's train/val data;
  3. reports a permutation test *within* stratum and a stratified bootstrap of
     the acute-minus-chronic difference, so the answer is an interval, not a
     point estimate that a reader could over-read.

A null result here is a real finding: it removes the most plausible remaining
biological explanation for the gap and pushes the residual onto representation.

Usage:
    python analysis/phase_b2_mi_stage_control.py
    python analysis/phase_b2_mi_stage_control.py --folds 1,2,3,4,5,6,7,8,9,10
    python analysis/phase_b2_mi_stage_control.py --configs exp8_leadfix_baseline
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
from sklearn.metrics import f1_score  # noqa: E402

from analysis.phase_b2_infarct_decoding import (  # noqa: E402
    CONFIG_LATENT_STEMS,
    PTBXL_SUBCLASS_PATH,
    SEED,
    TERRITORIES_4C,
    TERRITORY_4C_TO_2C,
    _classification_metrics,
    derive_rng,
    fit_scaler,
    fit_territory_4c_classifier,
    load_config_latents,
    load_ptbxl_latents,
    load_targets,
)

DEFAULT_PTBXL_ROOT = (
    REPO_ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
OUT_DIR = REPO_ROOT / "outputs" / "phase_b2_mi_stage"

# PTB-XL `infarction_stadium1` -> coarse stage. Stadium I is the acute injury
# phase (ST elevation); II-III and III are the chronic/resolved phases (Q waves).
# Stadium II alone is transitional and is reported as its own bucket rather than
# forced into either arm.
ACUTE_STAGES = ("Stadium I", "Stadium I-II")
TRANSITIONAL_STAGES = ("Stadium II",)
CHRONIC_STAGES = ("Stadium II-III", "Stadium III")


def coarse_stage(value: object) -> str:
    if not isinstance(value, str):
        return "missing"
    if value in ACUTE_STAGES:
        return "acute"
    if value in TRANSITIONAL_STAGES:
        return "transitional"
    if value in CHRONIC_STAGES:
        return "chronic"
    return "unknown"  # PTB-XL's literal 'unknown'


def load_stage_labels(ptbxl_root: Path) -> pd.DataFrame:
    """ecg_id -> infarction_stadium1 + coarse stage."""
    db = pd.read_csv(ptbxl_root / "ptbxl_database.csv",
                     usecols=["ecg_id", "strat_fold", "infarction_stadium1"])
    db["stage"] = db["infarction_stadium1"].map(coarse_stage)
    return db


def build_ptbxl_eval_frame(
    ptbxl_root: Path, folds: Tuple[int, ...]
) -> pd.DataFrame:
    """PTB-XL rows with a resolvable 4c territory, annotated with MI stage.

    For folds == (10,) this reuses `data/ptbxl_mi_subclass.csv`, whose `row_idx`
    is verified to reproduce PTBXLDataset ordering exactly -- so it indexes the
    exported latents directly. Other fold sets have no such export and are
    rejected rather than silently mis-indexed.
    """
    if tuple(sorted(folds)) != (10,):
        raise SystemExit(
            f"--folds {folds}: latents are exported per official split only "
            "(fold 10 = test). Scoring other folds would need a matching latent "
            "export AND would be encoder-contaminated. Refusing."
        )
    sub = pd.read_csv(PTBXL_SUBCLASS_PATH)
    stages = load_stage_labels(ptbxl_root)
    merged = sub.merge(stages[["ecg_id", "infarction_stadium1", "stage"]],
                       on="ecg_id", how="left", validate="one_to_one")
    if len(merged) != len(sub):
        raise RuntimeError("stage merge changed row count -- ecg_id is not unique")
    primary = merged[merged["territory_4c"].isin(TERRITORIES_4C)].copy()
    return primary.reset_index(drop=True)


def stratum_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    n_perm: int,
) -> Dict[str, object]:
    """macro-F1 + bootstrap CI + within-stratum label-permutation p."""
    n = y_true.size
    if n == 0:
        return {"n": 0}
    obs = float(f1_score(y_true, y_pred, labels=list(TERRITORIES_4C),
                         average="macro", zero_division=0))

    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = f1_score(y_true[idx], y_pred[idx], labels=list(TERRITORIES_4C),
                            average="macro", zero_division=0)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    # Permute the TRUE labels within the stratum: the null is "these predictions
    # carry no information about territory in this stratum".
    null = np.empty(n_perm, dtype=np.float64)
    for p in range(n_perm):
        null[p] = f1_score(rng.permutation(y_true), y_pred,
                           labels=list(TERRITORIES_4C),
                           average="macro", zero_division=0)
    n_tail = int((null >= obs).sum())

    y2_true = np.array([TERRITORY_4C_TO_2C[t] for t in y_true])
    y2_pred = np.array([TERRITORY_4C_TO_2C[t] for t in y_pred])
    macro_f1_2c = float(f1_score(y2_true, y2_pred,
                                 labels=sorted(set(TERRITORY_4C_TO_2C.values())),
                                 average="macro", zero_division=0))

    return {
        "n": int(n),
        "macro_f1": obs,
        "macro_f1_ci95": [float(lo), float(hi)],
        "macro_f1_2c": macro_f1_2c,
        # Phipson & Smyth: never report p = 0 from a finite permutation set.
        "permutation_p_macro_f1": float((n_tail + 1) / (n_perm + 1)),
        "n_perm": int(n_perm),
        "n_boot": int(n_boot),
        "per_class": _classification_metrics(y_true, y_pred, list(TERRITORIES_4C)),
        "class_counts": {t: int((y_true == t).sum()) for t in TERRITORIES_4C},
    }


def stratified_difference(
    y_true_a: np.ndarray, y_pred_a: np.ndarray,
    y_true_b: np.ndarray, y_pred_b: np.ndarray,
    rng: np.random.Generator, n_boot: int,
) -> Dict[str, object]:
    """Bootstrap CI on macro-F1(A) - macro-F1(B), resampling each arm separately."""
    na, nb = y_true_a.size, y_true_b.size
    if na == 0 or nb == 0:
        return {"computable": False}
    labels = list(TERRITORIES_4C)
    obs = (f1_score(y_true_a, y_pred_a, labels=labels, average="macro", zero_division=0)
           - f1_score(y_true_b, y_pred_b, labels=labels, average="macro", zero_division=0))
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        ia = rng.integers(0, na, size=na)
        ib = rng.integers(0, nb, size=nb)
        diffs[b] = (
            f1_score(y_true_a[ia], y_pred_a[ia], labels=labels,
                     average="macro", zero_division=0)
            - f1_score(y_true_b[ib], y_pred_b[ib], labels=labels,
                       average="macro", zero_division=0)
        )
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    n_tail = int((diffs <= 0).sum())
    return {
        "computable": True,
        "delta_macro_f1": float(obs),
        "delta_ci95": [float(lo), float(hi)],
        "p_acute_beats_chronic": float((n_tail + 1) / (n_boot + 1)),
        "n_boot": int(n_boot),
    }


def run_config(
    config: str,
    eval_df: pd.DataFrame,
    n_boot: int,
    n_perm: int,
    min_stratum: int,
) -> Dict[str, object]:
    print(f"\n{'=' * 66}\n[CONFIG] {config}\n{'=' * 66}")
    targets = load_targets()
    y_train_4c = np.array(targets["train"]["territory_4c"].tolist(), dtype=object)

    Z_train_full, _ = load_config_latents(config)
    Z_ptbxl_full = load_ptbxl_latents(config)

    # theta_mi_*.npz covers only the MI rows; `idx_in_split` maps them back into
    # the full split-ordered latent matrix. Subsetting BEFORE fitting the scaler
    # is what makes this the same standardisation contract as Pipeline A
    # (phase_b2_infarct_decoding.main() fits `z_scaler` on `Z_train_mi`), so the
    # classifier reused here is the same fit, not a lookalike.
    idx_train = targets["train"]["idx_in_split"]
    Z_train_mi = Z_train_full[idx_train].astype(np.float64)
    if Z_train_mi.shape[0] != y_train_4c.size:
        raise RuntimeError(
            f"MI subsetting mismatch: Z_train_mi={Z_train_mi.shape[0]} rows vs "
            f"territory_4c={y_train_4c.size} labels"
        )
    scaler = fit_scaler(Z_train_mi)
    Z_train_std = scaler.transform(Z_train_mi)

    model, best_C, cv_scores = fit_territory_4c_classifier(Z_train_std, y_train_4c)
    print(f"  4-class LogReg: best_C={best_C:g}, cv_macro_f1={cv_scores[str(best_C)]:.3f}")

    row_idx = eval_df["row_idx"].to_numpy()
    X_eval = scaler.transform(Z_ptbxl_full[row_idx].astype(np.float64))
    y_true_all = eval_df["territory_4c"].to_numpy()
    y_pred_all = model.predict(X_eval)
    stages = eval_df["stage"].to_numpy()

    out: Dict[str, object] = {
        "best_C": float(best_C),
        "cv_macro_f1": float(cv_scores[str(best_C)]),
        "strata": {},
    }

    for stratum in ("acute", "transitional", "chronic", "unknown", "missing", "ALL"):
        mask = np.ones(len(stages), dtype=bool) if stratum == "ALL" else (stages == stratum)
        if mask.sum() == 0:
            out["strata"][stratum] = {"n": 0}
            continue
        rng = derive_rng("mi_stage", config, stratum, seed=SEED)
        rec = stratum_scores(y_true_all[mask], y_pred_all[mask], rng, n_boot, n_perm)
        out["strata"][stratum] = rec
        print(f"  {stratum:<13s} n={rec['n']:>4d}  macro-F1={rec['macro_f1']:.3f} "
              f"[{rec['macro_f1_ci95'][0]:.3f},{rec['macro_f1_ci95'][1]:.3f}]  "
              f"p={rec['permutation_p_macro_f1']:.4f}  (2c {rec['macro_f1_2c']:.3f})")

    acute_mask = stages == "acute"
    chronic_mask = stages == "chronic"
    n_acute = int(acute_mask.sum())
    rng = derive_rng("mi_stage", config, "delta", seed=SEED)
    diff = stratified_difference(
        y_true_all[acute_mask], y_pred_all[acute_mask],
        y_true_all[chronic_mask], y_pred_all[chronic_mask],
        rng, n_boot,
    )
    out["acute_minus_chronic"] = diff
    out["underpowered"] = bool(n_acute < min_stratum)
    out["min_stratum_required"] = int(min_stratum)

    if diff.get("computable"):
        verdict = ("UNDERPOWERED -- do not quote" if out["underpowered"]
                   else ("acute > chronic" if diff["p_acute_beats_chronic"] < 0.05
                         else "no stage effect detected"))
        print(f"  delta(acute - chronic) = {diff['delta_macro_f1']:+.3f} "
              f"[{diff['delta_ci95'][0]:+.3f},{diff['delta_ci95'][1]:+.3f}]  "
              f"p={diff['p_acute_beats_chronic']:.4f}   -> {verdict}")
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", type=str, default="exp7_baseline",
                    help="Comma-separated config keys (must have latent exports).")
    ap.add_argument("--folds", type=str, default="10",
                    help="PTB-XL strat_folds to score. Only 10 is supported; "
                         "others are encoder-contaminated and rejected.")
    ap.add_argument("--ptbxl-root", type=Path, default=DEFAULT_PTBXL_ROOT)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--min-stratum", type=int, default=30,
                    help="Refuse to headline the comparison below this acute n.")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "mi_stage_control.json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    folds = tuple(int(x) for x in args.folds.split(",") if x.strip())
    eval_df = build_ptbxl_eval_frame(args.ptbxl_root, folds)

    counts = eval_df["stage"].value_counts().to_dict()
    n_acute = int(counts.get("acute", 0))
    print("=" * 66)
    print("Stage 4.1 -- MI-stage stratified cross-domain territory transfer")
    print("=" * 66)
    print(f"PTB-XL fold(s) {folds}; rows with a 4c territory: n={len(eval_df)}")
    print(f"  stage counts: {counts}")
    print(f"  MedalCare simulates ACUTE ischemia; acute stratum here is n={n_acute}")
    if n_acute < args.min_stratum:
        print(f"\n  *** UNDERPOWERED: acute n={n_acute} < --min-stratum "
              f"{args.min_stratum}. The comparison will be computed and written "
              f"but is flagged `underpowered: true`. It must NOT be reported as "
              f"a headline result; report the stratum sizes instead. ***")
    print(f"  cross-tab territory x stage:\n"
          f"{pd.crosstab(eval_df['territory_4c'], eval_df['stage'])}")

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in CONFIG_LATENT_STEMS]
    if unknown:
        raise SystemExit(f"Unknown config(s) {unknown}. "
                         f"Known: {sorted(CONFIG_LATENT_STEMS)}")

    results = {c: run_config(c, eval_df, args.n_boot, args.n_perm, args.min_stratum)
               for c in configs}

    payload = {
        "metadata": {
            "folds": list(folds),
            "encoder_contaminated": bool(tuple(sorted(folds)) != (10,)),
            "n_eval_rows": int(len(eval_df)),
            "stage_counts": {k: int(v) for k, v in counts.items()},
            "acute_stages": list(ACUTE_STAGES),
            "transitional_stages": list(TRANSITIONAL_STAGES),
            "chronic_stages": list(CHRONIC_STAGES),
            "underpowered": bool(n_acute < args.min_stratum),
            "min_stratum": int(args.min_stratum),
            "n_boot": int(args.n_boot),
            "n_perm": int(args.n_perm),
            "seed": SEED,
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
