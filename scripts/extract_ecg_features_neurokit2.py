"""Extract a hand-crafted ECG-feature baseline for B2 using NeuroKit2.

For each MI row in the MedalCare test split and the PTB-XL test fold, this
script computes six summary features that are clinically informative for
infarct decoding:

- ``QRS_duration_ms``  : median QRS duration across detected beats.
- ``QT_interval_ms``   : median QT interval across detected beats.
- ``P_duration_ms``    : median P-wave duration across detected beats.
- ``ST_J60_avg_mV``    : ST voltage 60 ms past the J-point, averaged across
                         leads V2-V6 (most diagnostic for anterior MI).
- ``T_amplitude_mV``   : median T-wave amplitude on lead II.
- ``heart_rate_bpm``   : 60 / median RR-interval.

Both datasets store raw 12-lead ECGs at 500 Hz (MedalCare via WFDB; PTB-XL
via WFDB high-res files). Crucially this script reads RAW voltage values
directly via ``wfdb.rdsamp`` -- the existing dataset wrappers in
``scripts/datasets.py`` apply a global z-score normalisation that would
destroy voltage-scale features (notably ``ST_J60``, ``T_amplitude``).

Outputs (aligned positionally with the corresponding latent NPZ):

- ``data/ecg_features_medalcare_test.npz``
  - ``features``  : float32, shape ``(2386, 6)``; non-MI rows = NaN.
  - ``nk2_ok``    : bool,    shape ``(2386,)``; True iff all 6 features
                    extracted successfully for that row.
  - ``feature_names`` : list of 6 names.
- ``data/ecg_features_ptbxl_test.npz`` (analogous; shape ``(2198, 6)``).

Usage::

    python scripts/extract_ecg_features_neurokit2.py
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

# NeuroKit2 emits many noisy warnings on synthetic / atypical waveforms.
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import neurokit2 as nk  # noqa: E402  (after warning filters)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scripts.medalcare_paths import is_mi_path  # noqa: E402

DEFAULT_MEDAL_MANIFEST = REPO_ROOT / "data" / "medalcare_filtered_manifest_dataset_split.csv"
DEFAULT_PTBXL_ROOT = (
    REPO_ROOT
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
OUT_DIR = REPO_ROOT / "data"
OUT_PTBXL = OUT_DIR / "ecg_features_ptbxl_test.npz"  # PTB-XL: test fold only

LEADS_12 = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
ST_LEADS = ("V2", "V3", "V4", "V5", "V6")
LEAD_II_IDX = LEADS_12.index("II")
ST_LEAD_INDICES = tuple(LEADS_12.index(l) for l in ST_LEADS)

FEATURE_NAMES = [
    "QRS_duration_ms",
    "QT_interval_ms",
    "P_duration_ms",
    "ST_J60_avg_mV",
    "T_amplitude_mV",
    "heart_rate_bpm",
]
N_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Per-ECG feature extraction
# ---------------------------------------------------------------------------

def _median_finite(arr: np.ndarray) -> float:
    """Median over finite entries; NaN if none."""
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size > 0 else float("nan")


def extract_features_one_ecg(
    ecg_12lead: np.ndarray,
    sampling_rate: int,
) -> Tuple[np.ndarray, bool, Optional[str]]:
    """Compute the 6-d feature vector for one 12-lead ECG.

    Parameters
    ----------
    ecg_12lead : array of shape (12, T) in millivolts.
    sampling_rate : Hz.

    Returns
    -------
    features : ``np.ndarray`` of shape (6,), float64. NaN for any feature
        that could not be derived.
    ok : True iff *all* 6 features are finite.
    error : optional short error string for diagnostics.
    """
    feats = np.full(N_FEATURES, np.nan, dtype=np.float64)
    error: Optional[str] = None

    if ecg_12lead.shape[0] != 12:
        return feats, False, "wrong_lead_count"
    n_samples = ecg_12lead.shape[1]
    if n_samples < 2 * sampling_rate:
        return feats, False, "signal_too_short"

    # Suppress NeuroKit2's verbose signalling.
    try:
        lead_ii = ecg_12lead[LEAD_II_IDX].astype(np.float64)
        cleaned = nk.ecg_clean(lead_ii, sampling_rate=sampling_rate)
        # ecg_peaks returns (signals_df, info_dict).
        _, rpeak_info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
        rpeaks = np.asarray(rpeak_info.get("ECG_R_Peaks", []), dtype=np.int64)
        if rpeaks.size < 2:
            return feats, False, "fewer_than_2_R_peaks"
        # ecg_delineate returns (signals_df, info_dict). DWT method is most robust.
        _, wave_info = nk.ecg_delineate(
            cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate, method="dwt"
        )
    except Exception as exc:  # noqa: BLE001
        return feats, False, f"nk_pipeline_failed:{type(exc).__name__}"

    def _arr(key: str) -> np.ndarray:
        vals = wave_info.get(key, [])
        return np.asarray(
            [v for v in vals if v is not None and np.isfinite(v)],
            dtype=np.float64,
        )

    r_onsets   = _arr("ECG_R_Onsets")
    r_offsets  = _arr("ECG_R_Offsets")
    p_onsets   = _arr("ECG_P_Onsets")
    p_offsets  = _arr("ECG_P_Offsets")
    t_offsets  = _arr("ECG_T_Offsets")
    t_peaks    = _arr("ECG_T_Peaks")

    # Convert sample indices to milliseconds via the sampling rate.
    sr = float(sampling_rate)

    # --- (1) QRS duration: pair onsets and offsets greedily, take median delta.
    if r_onsets.size and r_offsets.size:
        n_pairs = min(r_onsets.size, r_offsets.size)
        qrs_ms = (r_offsets[:n_pairs] - r_onsets[:n_pairs]) * 1000.0 / sr
        feats[0] = _median_finite(qrs_ms[(qrs_ms > 30) & (qrs_ms < 300)])

    # --- (2) QT interval: R-onset (≈ Q-onset) to T-offset, median.
    if r_onsets.size and t_offsets.size:
        n_pairs = min(r_onsets.size, t_offsets.size)
        qt_ms = (t_offsets[:n_pairs] - r_onsets[:n_pairs]) * 1000.0 / sr
        feats[1] = _median_finite(qt_ms[(qt_ms > 200) & (qt_ms < 700)])

    # --- (3) P duration.
    if p_onsets.size and p_offsets.size:
        n_pairs = min(p_onsets.size, p_offsets.size)
        p_ms = (p_offsets[:n_pairs] - p_onsets[:n_pairs]) * 1000.0 / sr
        feats[2] = _median_finite(p_ms[(p_ms > 30) & (p_ms < 250)])

    # --- (4) ST_J60: 60 ms past the J-point (≈ R-offset), averaged over V2-V6.
    if r_offsets.size:
        offset60 = int(round(0.060 * sr))
        st_per_beat: List[float] = []
        for ro in r_offsets.astype(np.int64):
            j60 = ro + offset60
            if 0 <= j60 < n_samples:
                # Average across V2-V6 leads at this sample.
                v = float(np.mean([ecg_12lead[i, j60] for i in ST_LEAD_INDICES]))
                if np.isfinite(v):
                    st_per_beat.append(v)
        if st_per_beat:
            feats[3] = _median_finite(np.asarray(st_per_beat))

    # --- (5) T amplitude on lead II.
    if t_peaks.size:
        idx = t_peaks.astype(np.int64)
        idx = idx[(idx >= 0) & (idx < n_samples)]
        if idx.size:
            feats[4] = _median_finite(ecg_12lead[LEAD_II_IDX, idx])

    # --- (6) Heart rate from RR-intervals on R-peaks.
    if rpeaks.size >= 2:
        rr_ms = np.diff(rpeaks).astype(np.float64) * 1000.0 / sr
        rr_ms = rr_ms[(rr_ms > 250) & (rr_ms < 2500)]  # plausible 24-240 bpm
        if rr_ms.size:
            feats[5] = 60_000.0 / float(np.median(rr_ms))

    ok = bool(np.all(np.isfinite(feats)))
    return feats, ok, error


# ---------------------------------------------------------------------------
# Dataset iterators -- read RAW voltages (no normalisation).
# ---------------------------------------------------------------------------

def load_medalcare_split_rows(manifest_path: Path, split: str) -> pd.DataFrame:
    """Return MedalCare DataFrame for a given split, in latent-export order."""
    df = pd.read_csv(manifest_path)
    sub = df[df["split"].str.lower() == split.lower()].reset_index(drop=True)
    return sub


def load_ptbxl_test_rows(ptbxl_root: Path, folds: Sequence[int] = (10,)) -> pd.DataFrame:
    """Return PTB-XL rows for `folds` in latent-export order.

    Default (10,) is the official test split. The filter expression must stay
    byte-identical to `PTBXLDataset` and `build_ptbxl_mi_subclass.py` — the row
    order of every downstream .npz depends on it.
    """
    db = pd.read_csv(ptbxl_root / "ptbxl_database.csv")
    sub = db[db["strat_fold"].isin(list(folds))].reset_index(drop=True)
    return sub


def read_medalcare_ecg(row: pd.Series) -> Tuple[np.ndarray, int]:
    """Read a MedalCare 12-lead ECG into shape (12, T), millivolts, plus rate."""
    wfdb_path = str(row["wfdb_path"])
    rec, meta = wfdb.rdsamp(wfdb_path)
    # rec is shape (T, n_leads) with native channel ordering as recorded.
    fs = int(meta.get("fs", 500))
    sig_names = list(meta.get("sig_name", LEADS_12))
    # Reindex to LEADS_12 ordering when sig_name is provided.
    try:
        idx = [sig_names.index(name) for name in LEADS_12]
        rec = rec[:, idx]
    except ValueError:
        # Trust the file's native order.
        if rec.shape[1] != 12:
            raise
    return np.asarray(rec, dtype=np.float64).T, fs


def read_ptbxl_ecg(row: pd.Series, ptbxl_root: Path) -> Tuple[np.ndarray, int]:
    """Read a PTB-XL high-res 12-lead ECG into shape (12, T), millivolts."""
    rel = str(row["filename_hr"])
    full = ptbxl_root / rel
    rec, meta = wfdb.rdsamp(str(full))
    fs = int(meta.get("fs", 500))
    sig_names = list(meta.get("sig_name", LEADS_12))
    try:
        idx = [sig_names.index(name) for name in LEADS_12]
        rec = rec[:, idx]
    except ValueError:
        if rec.shape[1] != 12:
            raise
    return np.asarray(rec, dtype=np.float64).T, fs


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def process_medalcare(
    manifest: Path,
    out_path: Path,
    split: str = "test",
    limit: Optional[int] = None,
) -> Dict[str, object]:
    df = load_medalcare_split_rows(manifest, split)
    n_total = len(df)
    is_mi = df["original_csv_path"].apply(is_mi_path).to_numpy()
    mi_idx = np.flatnonzero(is_mi)
    if limit is not None:
        mi_idx = mi_idx[:limit]
    print(f"[MedalCare-{split}] total={n_total}, MI={int(is_mi.sum())}, processing={len(mi_idx)}")

    features = np.full((n_total, N_FEATURES), np.nan, dtype=np.float64)
    nk2_ok = np.zeros(n_total, dtype=bool)
    failure_reasons: Dict[str, int] = {}

    for i in tqdm(mi_idx, desc="MedalCare", leave=False):
        row = df.iloc[int(i)]
        try:
            ecg, fs = read_medalcare_ecg(row)
        except Exception as exc:  # noqa: BLE001
            failure_reasons[f"wfdb:{type(exc).__name__}"] = (
                failure_reasons.get(f"wfdb:{type(exc).__name__}", 0) + 1
            )
            continue
        feats, ok, err = extract_features_one_ecg(ecg, fs)
        features[int(i)] = feats
        nk2_ok[int(i)] = ok
        if not ok and err:
            failure_reasons[err] = failure_reasons.get(err, 0) + 1

    n_ok = int(nk2_ok.sum())
    n_processed = len(mi_idx)
    per_feat_ok = {
        FEATURE_NAMES[k]: int(np.isfinite(features[mi_idx, k]).sum())
        for k in range(N_FEATURES)
    }
    print(
        f"  OK (all-6): {n_ok}/{n_processed} = {100*n_ok/n_processed:.1f}%; "
        f"per-feature finite: {per_feat_ok}; "
        f"top failure reasons: "
        f"{sorted(failure_reasons.items(), key=lambda x: -x[1])[:5]}"
    )

    np.savez_compressed(
        out_path,
        features=features.astype(np.float32),
        nk2_ok=nk2_ok,
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
    )
    print(f"  -> saved {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")

    return {
        "dataset": "medalcare",
        "split": split,
        "n_total_rows": n_total,
        "n_mi_rows": int(is_mi.sum()),
        "n_processed": n_processed,
        "n_ok": n_ok,
        "ok_rate": float(n_ok / n_processed) if n_processed else 0.0,
        "per_feature_finite": per_feat_ok,
        "failure_reasons": dict(sorted(failure_reasons.items(), key=lambda x: -x[1])),
    }


def process_ptbxl(
    ptbxl_root: Path,
    out_path: Path,
    limit: Optional[int] = None,
) -> Dict[str, object]:
    df = load_ptbxl_test_rows(ptbxl_root)
    n_total = len(df)
    # Per-feature reporting (added for parity with MedalCare path).
    # Use the precomputed mi_present labels for consistency.
    subclass_path = REPO_ROOT / "data" / "ptbxl_mi_subclass.csv"
    if not subclass_path.is_file():
        raise FileNotFoundError(
            f"{subclass_path} not found; run scripts/build_ptbxl_mi_subclass.py first."
        )
    subclass_df = pd.read_csv(subclass_path)
    if len(subclass_df) != n_total:
        raise RuntimeError(
            f"PTB-XL subclass CSV rows ({len(subclass_df)}) != database test fold rows ({n_total})"
        )
    is_mi = subclass_df["mi_present"].to_numpy().astype(bool)
    mi_idx = np.flatnonzero(is_mi)
    if limit is not None:
        mi_idx = mi_idx[:limit]
    print(f"[PTB-XL-test] total={n_total}, MI={int(is_mi.sum())}, processing={len(mi_idx)}")

    features = np.full((n_total, N_FEATURES), np.nan, dtype=np.float64)
    nk2_ok = np.zeros(n_total, dtype=bool)
    failure_reasons: Dict[str, int] = {}

    for i in tqdm(mi_idx, desc="PTB-XL", leave=False):
        row = df.iloc[int(i)]
        try:
            ecg, fs = read_ptbxl_ecg(row, ptbxl_root)
        except Exception as exc:  # noqa: BLE001
            failure_reasons[f"wfdb:{type(exc).__name__}"] = (
                failure_reasons.get(f"wfdb:{type(exc).__name__}", 0) + 1
            )
            continue
        feats, ok, err = extract_features_one_ecg(ecg, fs)
        features[int(i)] = feats
        nk2_ok[int(i)] = ok
        if not ok and err:
            failure_reasons[err] = failure_reasons.get(err, 0) + 1

    n_ok = int(nk2_ok.sum())
    n_processed = len(mi_idx)
    per_feat_ok = {
        FEATURE_NAMES[k]: int(np.isfinite(features[mi_idx, k]).sum())
        for k in range(N_FEATURES)
    }
    print(
        f"  OK (all-6): {n_ok}/{n_processed} = {100*n_ok/n_processed:.1f}%; "
        f"per-feature finite: {per_feat_ok}; "
        f"top failure reasons: "
        f"{sorted(failure_reasons.items(), key=lambda x: -x[1])[:5]}"
    )

    np.savez_compressed(
        out_path,
        features=features.astype(np.float32),
        nk2_ok=nk2_ok,
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
    )
    print(f"  -> saved {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")

    return {
        "dataset": "ptbxl",
        "split": "test (fold 10)",
        "n_total_rows": n_total,
        "n_mi_rows": int(is_mi.sum()),
        "n_processed": n_processed,
        "n_ok": n_ok,
        "ok_rate": float(n_ok / n_processed) if n_processed else 0.0,
        "per_feature_finite": per_feat_ok,
        "failure_reasons": dict(sorted(failure_reasons.items(), key=lambda x: -x[1])),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--medal-manifest", type=Path, default=DEFAULT_MEDAL_MANIFEST,
    )
    parser.add_argument("--ptbxl-root", type=Path, default=DEFAULT_PTBXL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--medal-splits", type=str, default="train,test",
        help="Comma-separated MedalCare splits to extract (default: 'train,test'; "
             "Ridge fits on train, evaluates on test).",
    )
    parser.add_argument(
        "--datasets", type=str, default="medalcare,ptbxl",
        help="Comma-separated subset of {medalcare, ptbxl} to run.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on number of MI rows processed per dataset (for smoke tests).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    requested = {x.strip().lower() for x in args.datasets.split(",")}
    medal_splits = [
        s.strip().lower() for s in args.medal_splits.split(",") if s.strip()
    ]

    summaries: List[Dict[str, object]] = []
    if "medalcare" in requested:
        for split in medal_splits:
            out_path = args.out_dir / f"ecg_features_medalcare_{split}.npz"
            summaries.append(
                process_medalcare(args.medal_manifest, out_path, split, args.limit)
            )
    if "ptbxl" in requested:
        out_path = args.out_dir / "ecg_features_ptbxl_test.npz"
        summaries.append(process_ptbxl(args.ptbxl_root, out_path, args.limit))

    summary_path = args.out_dir / "ecg_features_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nWrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
