"""Spatially-resolved hand-crafted control for Track 3 infarct localisation.

`extract_ecg_features_neurokit2.py` produces the 6-feature control that report
S14/S14.6 compares the latent against. For the *localisation* task that control
is not a fair opponent, and the reason is structural rather than a matter of
degree:

    QRS_duration_ms   global scalar        no spatial content
    QT_interval_ms    global scalar        no spatial content
    P_duration_ms     global scalar        no spatial content
    heart_rate_bpm    global scalar        no spatial content
    ST_J60_avg_mV     averaged over V2-V6  ANTERIOR leads only -- the inferior
                                           leads (II, III, aVF) are never read
    T_amplitude_mV    lead II only         one inferior lead

Infarct territory is *defined* by which leads change: anterior V1-V4, inferior
II/III/aVF, lateral I/aVL/V5-V6. A feature vector that averages the anterior
leads into one number and reads exactly one inferior lead cannot represent the
anterior-vs-inferior contrast it is being asked to predict. Beating it is close
to guaranteed and says little about the representation. The comparison is also
lopsided on width: 1024 latent dimensions against 6.

This script builds the control a cardiologist would actually use -- the per-lead
deviation vector -- so that "the latent beats hand-crafted features" becomes a
claim about information content rather than about instrumentation.

Per lead, at fiducials delineated once on lead II (48 features):

    ST_J60_<lead>   voltage 60 ms past the J-point   ST elevation / depression
    Q_amp_<lead>    min voltage in [R_onset, R_peak] pathological Q waves --
                                                     the marker for the CHRONIC
                                                     infarcts that dominate
                                                     PTB-XL, where acute ST
                                                     elevation has resolved
    R_amp_<lead>    voltage at the R peak            R-wave progression loss
    T_amp_<lead>    voltage at the T peak            T-wave inversion territory

plus the original 6 global features, so the new set is a strict superset of the
old one: 54 features total. Any drop in performance against `global6` is
therefore attributable to estimation cost, never to lost information.

Amplitudes in all 12 leads are read at the SAME sample index -- the fiducial
found on lead II. That is deliberate: it samples the instantaneous QRS/T vector
across the frontal and precordial planes, which is exactly the spatial quantity
territory decoding depends on. Per-lead re-delineation would misalign the leads
and destroy that.

Output contract is identical to the 6-feature script (`features`, `nk2_ok`,
`feature_names`), so downstream loaders need only a path swap::

    python scripts/extract_ecg_features_spatial.py
    # -> data/ecg_features_spatial_{medalcare_train,medalcare_test,ptbxl_test}.npz

Then, in `analysis/phase_b2_infarct_decoding.py`::

    --feature-set spatial54
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import neurokit2 as nk  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
from scripts.extract_ecg_features_neurokit2 import (  # noqa: E402
    DEFAULT_MEDAL_MANIFEST,
    DEFAULT_PTBXL_ROOT,
    FEATURE_NAMES as GLOBAL6_NAMES,
    LEADS_12,
    LEAD_II_IDX,
    OUT_DIR,
    _median_finite,
    extract_features_one_ecg,
    load_medalcare_split_rows,
    load_ptbxl_test_rows,
    read_medalcare_ecg,
    read_ptbxl_ecg,
)
from scripts.medalcare_paths import is_mi_path  # noqa: E402

PER_LEAD_KINDS = ("ST_J60", "Q_amp", "R_amp", "T_amp")
FEATURE_NAMES: List[str] = (
    [f"{kind}_{lead}" for kind in PER_LEAD_KINDS for lead in LEADS_12]
    + list(GLOBAL6_NAMES)
)
N_FEATURES = len(FEATURE_NAMES)
N_PER_LEAD = len(PER_LEAD_KINDS) * len(LEADS_12)


def _fiducials(ecg_12lead: np.ndarray, sampling_rate: int):
    """Delineate on lead II. Returns (rpeaks, wave_info) or (None, reason)."""
    lead_ii = ecg_12lead[LEAD_II_IDX].astype(np.float64)
    cleaned = nk.ecg_clean(lead_ii, sampling_rate=sampling_rate)
    _, rpeak_info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
    rpeaks = np.asarray(rpeak_info.get("ECG_R_Peaks", []), dtype=np.int64)
    if rpeaks.size < 2:
        return None, "fewer_than_2_R_peaks"
    _, wave_info = nk.ecg_delineate(
        cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate, method="dwt"
    )
    return (rpeaks, wave_info), None


def _clean_idx(wave_info: dict, key: str, n: int) -> np.ndarray:
    vals = wave_info.get(key, [])
    arr = np.asarray(
        [v for v in vals if v is not None and np.isfinite(v)], dtype=np.int64
    )
    return arr[(arr >= 0) & (arr < n)]


def extract_spatial_one_ecg(
    ecg_12lead: np.ndarray, sampling_rate: int
) -> Tuple[np.ndarray, bool, Optional[str]]:
    """Compute the 54-d feature vector for one 12-lead ECG."""
    feats = np.full(N_FEATURES, np.nan, dtype=np.float64)

    if ecg_12lead.shape[0] != 12:
        return feats, False, "wrong_lead_count"
    n = ecg_12lead.shape[1]
    if n < 2 * sampling_rate:
        return feats, False, "signal_too_short"

    # The 6 global features come from the original implementation verbatim, so
    # the shared columns cannot drift between the two control variants.
    g_feats, _, g_err = extract_features_one_ecg(ecg_12lead, sampling_rate)
    feats[N_PER_LEAD:] = g_feats

    try:
        fid, reason = _fiducials(ecg_12lead, sampling_rate)
    except Exception as exc:  # noqa: BLE001
        return feats, False, f"nk_pipeline_failed:{type(exc).__name__}"
    if fid is None:
        return feats, False, reason
    rpeaks, wave_info = fid

    sr = float(sampling_rate)
    r_onsets = _clean_idx(wave_info, "ECG_R_Onsets", n)
    r_offsets = _clean_idx(wave_info, "ECG_R_Offsets", n)
    t_peaks = _clean_idx(wave_info, "ECG_T_Peaks", n)

    def col(kind: str, lead_i: int) -> int:
        return PER_LEAD_KINDS.index(kind) * len(LEADS_12) + lead_i

    # --- ST at J+60 ms, median across beats, read in every lead.
    if r_offsets.size:
        j60 = r_offsets + int(round(0.060 * sr))
        j60 = j60[j60 < n]
        if j60.size:
            for li in range(12):
                feats[col("ST_J60", li)] = _median_finite(ecg_12lead[li, j60])

    # --- Q amplitude: the most negative sample between QRS onset and the R
    # peak. Pathological Q waves are the chronic-MI localisation marker, and
    # PTB-XL MI is predominantly old infarction.
    if r_onsets.size and rpeaks.size:
        q_vals: List[np.ndarray] = []
        for onset in r_onsets:
            nxt = rpeaks[rpeaks > onset]
            if nxt.size == 0:
                continue
            peak = int(nxt[0])
            if peak - onset < 2:
                continue
            q_vals.append(ecg_12lead[:, onset:peak].min(axis=1))
        if q_vals:
            qm = np.vstack(q_vals)
            for li in range(12):
                feats[col("Q_amp", li)] = _median_finite(qm[:, li])

    # --- R amplitude at the lead-II R-peak instant (R-wave progression).
    if rpeaks.size:
        rp = rpeaks[rpeaks < n]
        if rp.size:
            for li in range(12):
                feats[col("R_amp", li)] = _median_finite(ecg_12lead[li, rp])

    # --- T amplitude at the T-peak instant (inversion territory).
    if t_peaks.size:
        for li in range(12):
            feats[col("T_amp", li)] = _median_finite(ecg_12lead[li, t_peaks])

    ok = bool(np.all(np.isfinite(feats)))
    return feats, ok, (None if ok else (g_err or "some_features_nan"))


def _run(rows: pd.DataFrame, mi_idx: np.ndarray, reader, desc: str,
         n_total: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    features = np.full((n_total, N_FEATURES), np.nan, dtype=np.float64)
    nk2_ok = np.zeros(n_total, dtype=bool)
    fails: Dict[str, int] = {}
    for i in tqdm(mi_idx, desc=desc, leave=False):
        row = rows.iloc[int(i)]
        try:
            ecg, fs = reader(row)
        except Exception as exc:  # noqa: BLE001
            k = f"wfdb:{type(exc).__name__}"
            fails[k] = fails.get(k, 0) + 1
            continue
        f, ok, err = extract_spatial_one_ecg(ecg, fs)
        features[int(i)] = f
        nk2_ok[int(i)] = ok
        if not ok and err:
            fails[err] = fails.get(err, 0) + 1
    return features, nk2_ok, fails


def _save(out_path: Path, features: np.ndarray, nk2_ok: np.ndarray,
          mi_idx: np.ndarray, fails: Dict[str, int], **extra) -> Dict[str, object]:
    n_ok, n_proc = int(nk2_ok.sum()), len(mi_idx)
    # Report per-KIND coverage rather than all 54 names -- 54 counts is noise,
    # and a systematic failure shows up as a whole kind going missing.
    per_kind = {
        kind: int(np.isfinite(
            features[np.ix_(mi_idx, np.arange(
                PER_LEAD_KINDS.index(kind) * 12,
                PER_LEAD_KINDS.index(kind) * 12 + 12))]).all(axis=1).sum())
        for kind in PER_LEAD_KINDS
    }
    per_kind["global6"] = int(
        np.isfinite(features[np.ix_(mi_idx, np.arange(N_PER_LEAD, N_FEATURES))]
                    ).all(axis=1).sum())
    print(f"  OK (all-{N_FEATURES}): {n_ok}/{n_proc} = "
          f"{100 * n_ok / max(n_proc, 1):.1f}%")
    print(f"  rows with a complete block, per kind: {per_kind}")
    if fails:
        print(f"  top failures: {sorted(fails.items(), key=lambda x: -x[1])[:5]}")
    np.savez_compressed(
        out_path,
        features=features.astype(np.float32),
        nk2_ok=nk2_ok,
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
    )
    print(f"  -> saved {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")
    return {"n_processed": n_proc, "n_ok": n_ok,
            "ok_rate": float(n_ok / n_proc) if n_proc else 0.0,
            "rows_with_complete_block": per_kind,
            "failure_reasons": dict(sorted(fails.items(), key=lambda x: -x[1])),
            **extra}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--medal-manifest", type=Path, default=DEFAULT_MEDAL_MANIFEST)
    ap.add_argument("--ptbxl-root", type=Path, default=DEFAULT_PTBXL_ROOT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--medal-splits", type=str, default="train,test")
    ap.add_argument("--datasets", type=str, default="medalcare,ptbxl")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap MI rows per dataset (smoke test).")
    ap.add_argument(
        "--ptbxl-folds", type=str, default="10",
        help="Comma-separated PTB-XL stratification folds (default '10' = the "
             "official test split). Use '1,...,10' together with an all-folds "
             "--ptbxl-subclass-csv when the encoder never saw PTB-XL.",
    )
    ap.add_argument(
        "--ptbxl-subclass-csv", type=Path,
        default=REPO_ROOT / "data" / "ptbxl_mi_subclass.csv",
        help="MI-subclass CSV built for exactly --ptbxl-folds. Its row count is "
             "asserted against the fold selection, so a mismatched pair fails "
             "loudly instead of silently misaligning every feature row.",
    )
    ap.add_argument(
        "--ptbxl-out", type=Path, default=None,
        help="Override the PTB-XL output .npz path (default "
             "<out-dir>/ecg_features_spatial_ptbxl_test.npz). Set this when "
             "--ptbxl-folds is not the default so the fold-10 artifact survives.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    want = {x.strip().lower() for x in args.datasets.split(",")}
    summaries: List[Dict[str, object]] = []

    if "medalcare" in want:
        for split in [s.strip() for s in args.medal_splits.split(",") if s.strip()]:
            df = load_medalcare_split_rows(args.medal_manifest, split)
            is_mi = df["original_csv_path"].apply(is_mi_path).to_numpy()
            mi_idx = np.flatnonzero(is_mi)
            if args.limit:
                mi_idx = mi_idx[:args.limit]
            print(f"[MedalCare-{split}] total={len(df)}, MI={int(is_mi.sum())}, "
                  f"processing={len(mi_idx)}")
            f, ok, fails = _run(df, mi_idx, lambda r: read_medalcare_ecg(r),
                                f"MedalCare-{split}", len(df))
            out = args.out_dir / f"ecg_features_spatial_medalcare_{split}.npz"
            summaries.append(_save(out, f, ok, mi_idx, fails,
                                   dataset="medalcare", split=split,
                                   n_total_rows=len(df)))

    if "ptbxl" in want:
        ptb_folds = [int(x) for x in args.ptbxl_folds.split(",") if x.strip()]
        df = load_ptbxl_test_rows(args.ptbxl_root, ptb_folds)
        sub = pd.read_csv(args.ptbxl_subclass_csv)
        if len(sub) != len(df):
            raise RuntimeError(
                f"PTB-XL subclass CSV rows ({len(sub)}) != fold {ptb_folds} rows "
                f"({len(df)}). Rebuild it with "
                f"`build_ptbxl_mi_subclass.py --folds {args.ptbxl_folds}`.")
        is_mi = sub["mi_present"].to_numpy().astype(bool)
        mi_idx = np.flatnonzero(is_mi)
        if args.limit:
            mi_idx = mi_idx[:args.limit]
        fold_desc = f"folds {ptb_folds}" if ptb_folds != [10] else "test (fold 10)"
        print(f"[PTB-XL {fold_desc}] total={len(df)}, MI={int(is_mi.sum())}, "
              f"processing={len(mi_idx)}")
        f, ok, fails = _run(df, mi_idx,
                            lambda r: read_ptbxl_ecg(r, args.ptbxl_root),
                            "PTB-XL", len(df))
        out = args.ptbxl_out or (args.out_dir / "ecg_features_spatial_ptbxl_test.npz")
        summaries.append(_save(out, f, ok, mi_idx, fails, dataset="ptbxl",
                               split=fold_desc, n_total_rows=len(df)))

    # Name the summary after the PTB-XL artifact when it was redirected, so a
    # non-default fold selection cannot clobber the fold-10 summary.
    if args.ptbxl_out is not None:
        p = args.ptbxl_out.with_name(f"{args.ptbxl_out.stem}_summary.json")
    else:
        p = args.out_dir / "ecg_features_spatial_summary.json"
    p.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nWrote summary -> {p}")


if __name__ == "__main__":
    main()
