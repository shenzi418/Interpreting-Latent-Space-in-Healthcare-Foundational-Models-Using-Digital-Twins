#!/usr/bin/env python3
"""
Quick visualization utility to sanity-check ECG waveforms.

Examples
--------
Default MedalCare test split, 10 records:
    python viz/plot_waveforms.py --manifest MedalRaw/medalcare_filtered_manifest.csv

Specify a run id and fewer samples:
    python viz/plot_waveforms.py --run-id debug_waveforms --num-records 5

Alternate dataset manifest (e.g., mimic):
    python viz/plot_waveforms.py --dataset mimic --manifest path/to/mimic_manifest.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb


LEADS_TO_PLOT: Tuple[str, str, str] = ("II", "V1", "V6")
DEFAULT_FS = 500  # Hz
DEFAULT_DURATION = 10  # seconds


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot sample ECG waveforms for visual QA.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest CSV (must include wfdb_path column and 'split' column).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("medalcare", "mimic"),
        default="medalcare",
        help="Dataset tag (affects defaults only).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split to sample from (default: test).",
    )
    parser.add_argument(
        "--num-records",
        type=int,
        default=10,
        help="Number of records to plot (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help="Duration of signal to display in seconds (default: 10).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Identifier for outputs/<run_id>/waveforms/ (default: timestamp).",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=DEFAULT_FS,
        help="Expected sampling frequency (Hz) (default: 500).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing waveform figures if they already exist.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path)
    if "split" not in df.columns:
        raise ValueError("Manifest must contain a 'split' column to identify test records.")
    if "wfdb_path" not in df.columns:
        raise ValueError("Manifest must contain a 'wfdb_path' column with WFDB base paths.")
    return df


def sample_records(df: pd.DataFrame, split: str, num_records: int, seed: int) -> pd.DataFrame:
    split_df = df[df["split"].str.lower() == split.lower()]
    if split_df.empty:
        raise ValueError(f"No records found for split '{split}'.")
    return split_df.sample(n=min(num_records, len(split_df)), random_state=seed).reset_index(drop=True)


def ensure_output_dir(run_id: str | None) -> Path:
    resolved_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = REPO_ROOT / "outputs" / resolved_run_id / "waveforms"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def figure_path(output_dir: Path, record_id: str, overwrite: bool) -> Path:
    safe_name = record_id.replace("/", "_").replace("\\", "_")
    path = output_dir / f"{safe_name}.png"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Figure already exists: {path}. Use --overwrite to replace.")
    return path


def extract_lead_indices(sig_names: Iterable[str], leads: Iterable[str]) -> List[int]:
    name_to_idx = {name.upper(): idx for idx, name in enumerate(sig_names)}
    indices = []
    missing = []
    for lead in leads:
        key = lead.upper()
        if key in name_to_idx:
            indices.append(name_to_idx[key])
        else:
            missing.append(lead)
    if missing:
        raise ValueError(f"Missing leads in WFDB signal: {missing}")
    return indices


def plot_record(
    wfdb_path: str,
    record_id: str,
    leads: Tuple[str, str, str],
    fs: float,
    duration_sec: float,
    output_path: Path,
) -> None:
    signals, fields = wfdb.rdsamp(wfdb_path)
    sig_names = fields.get("sig_name")
    if not sig_names:
        raise ValueError(f"No signal names found for WFDB record: {wfdb_path}")

    lead_indices = extract_lead_indices(sig_names, leads)
    num_samples = signals.shape[0]
    desired_samples = int(fs * duration_sec)
    end_idx = min(desired_samples, num_samples)

    time_axis = np.arange(end_idx) / fs

    fig, axs = plt.subplots(len(leads), 1, figsize=(12, 8), sharex=True)
    if len(leads) == 1:
        axs = [axs]

    for ax, lead, idx in zip(axs, leads, lead_indices):
        trace = signals[:end_idx, idx]
        ax.plot(time_axis, trace, linewidth=0.8)
        ax.set_ylabel(f"{lead} (mV)")
        ax.grid(True, linewidth=0.3, linestyle="--", alpha=0.5)

    axs[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{record_id} | {Path(wfdb_path).name}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    df = load_manifest(args.manifest)
    sampled_df = sample_records(df, args.split, args.num_records, args.seed)
    output_dir = ensure_output_dir(args.run_id)

    print(f"Saving waveform plots to: {output_dir}")
    print(f"Dataset: {args.dataset} | Manifest: {args.manifest} | Split: {args.split}")

    failures = []
    for _, row in sampled_df.iterrows():
        record_id = str(row.get("record_id", row.get("wfdb_path", "unknown")))
        wfdb_path = str(row["wfdb_path"])
        try:
            output_path = figure_path(output_dir, record_id, overwrite=args.overwrite)
        except FileExistsError as exc:
            print(f"[SKIP] {exc}")
            continue

        try:
            plot_record(
                wfdb_path=wfdb_path,
                record_id=record_id,
                leads=LEADS_TO_PLOT,
                fs=args.fs,
                duration_sec=args.duration,
                output_path=output_path,
            )
            print(f"[OK] {record_id} -> {output_path.name}")
        except Exception as err:  # pylint: disable=broad-except
            print(f"[FAIL] {record_id}: {err}")
            failures.append((record_id, err))

    if failures:
        print("\nFailures encountered:")
        for rec, err in failures:
            print(f" - {rec}: {err}")
    else:
        print("\nAll waveforms plotted successfully.")


if __name__ == "__main__":
    main()

