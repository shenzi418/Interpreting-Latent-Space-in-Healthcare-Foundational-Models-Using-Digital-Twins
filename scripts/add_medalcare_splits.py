#!/usr/bin/env python3
"""Add split and run_id columns to the MedalCare manifest from folder structure."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "MedalRaw" / "medalcare_filtered_manifest.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "medalcare_filtered_manifest_dataset_split.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive split/run_id from original_csv_path and write a new manifest."
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input manifest CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output manifest CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--allow-mixed-runs",
        action="store_true",
        help="Allow run_id entries to appear in multiple splits (log warnings).",
    )
    return parser.parse_args()


def extract_split(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    if "train" in parts:
        return "train"
    if "validation" in parts:
        return "val"
    if "test" in parts:
        return "test"
    raise ValueError(f"Unable to derive split from path: {path}")


def extract_run_id(path: Path) -> str:
    parts_lower = [part.lower() for part in path.parts]
    try:
        base_idx = parts_lower.index("wp2_largedataset_noise")
        pathology = path.parts[base_idx + 1]
    except (ValueError, IndexError):
        raise ValueError(f"Unable to derive pathology from path: {path}")
    run_folder = None
    for part in path.parts:
        if part.startswith("run_S"):
            run_folder = part
            break
    if not run_folder:
        raise ValueError(f"Unable to find run_S* folder in path: {path}")
    stem = path.stem
    if stem.endswith("_filtered"):
        stem = stem[: -len("_filtered")]
    if stem.startswith("run_"):
        run_base = stem
    else:
        # Some files are named like 000001_filtered.csv
        run_base = f"run_{stem}" if stem.isdigit() else None
    if not run_base or not run_base.startswith("run_"):
        raise ValueError(f"Unexpected run file name: {path.name}")
    return f"{pathology}/{run_folder}/{run_base}"


def validate_run_splits(df: pd.DataFrame, allow_mixed: bool) -> None:
    split_counts = df.groupby("run_id")["split"].nunique()
    mixed = split_counts[split_counts > 1]
    if not mixed.empty:
        examples = df[df["run_id"].isin(mixed.index)][["run_id", "split"]].drop_duplicates()
        message = "Found run_ids in multiple splits. Examples:\n" + examples.to_string(index=False)
        if allow_mixed:
            print(f"[WARN] {message}")
            return
        raise ValueError(message)


def main() -> None:
    args = parse_args()
    if not args.input_manifest.exists():
        raise FileNotFoundError(f"Input manifest not found: {args.input_manifest}")

    df = pd.read_csv(args.input_manifest)
    if "original_csv_path" not in df.columns:
        raise ValueError("Manifest must include 'original_csv_path' column.")

    paths = df["original_csv_path"].apply(lambda p: Path(str(p)))
    df = df.copy()
    df["split"] = paths.apply(extract_split)
    df["run_id"] = paths.apply(extract_run_id)

    validate_run_splits(df, args.allow_mixed_runs)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_manifest, index=False)
    print(f"Wrote manifest with splits to: {args.output_manifest}")


if __name__ == "__main__":
    main()

