#!/usr/bin/env python3
"""Create deterministic train/val/test splits for MedalCare manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Label columns expected in the manifest
LABEL_COLUMNS = [
    "label_0",
    "label_1",
    "label_2",
    "label_3",
    "label_4",
    "label_5",
    "label_6",
    "label_7",
]

DEFAULT_INPUT = Path("MedalRaw/medalcare_filtered_manifest.csv")
DEFAULT_OUTPUT = Path("data/medalcare_filtered_manifest.csv")
DEFAULT_SPLITS_JSON = Path("data/splits_v1.json")
DEFAULT_SEED = 42

# Map folds to splits as requested (fold0=test, fold1=val, others=train)
# After inspecting fold distributions we use fold 0 for test, fold 4 for val,
# and the remaining folds for train (deterministic under the chosen seed).
FOLD_TO_SPLIT = {
    0: "test",
    4: "val",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to input manifest CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to output manifest CSV with splits (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--splits-json",
        type=Path,
        default=DEFAULT_SPLITS_JSON,
        help=f"Path to JSON summary file (default: {DEFAULT_SPLITS_JSON})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for StratifiedGroupKFold (default: 42)",
    )
    return parser.parse_args()


def extract_group_id(path_str: str) -> str:
    """Derive subject/twin grouping id from the original CSV path."""
    path = Path(path_str)
    # Use the immediate parent directory (e.g., run_S62) as the grouping key
    return path.parent.name


def compute_split_assignments(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Assign folds and splits using StratifiedGroupKFold."""
    missing = [col for col in LABEL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing label columns: {missing}")

    if "record_id" not in df.columns or "original_csv_path" not in df.columns:
        raise ValueError("Manifest must include 'record_id' and 'original_csv_path'")

    # Sort deterministically before split assignment
    df_sorted = df.sort_values("record_id").reset_index(drop=True).copy()

    labels_matrix = df_sorted[LABEL_COLUMNS].values
    # Each row has a single positive, so argmax yields the class index
    y = labels_matrix.argmax(axis=1)

    groups = df_sorted["original_csv_path"].apply(extract_group_id)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    fold_ids = np.empty(len(df_sorted), dtype=int)
    for fold, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(df_sorted)), y, groups)):
        fold_ids[test_idx] = fold

    df_sorted["fold"] = fold_ids
    df_sorted["split"] = [FOLD_TO_SPLIT.get(fold, "train") for fold in fold_ids]
    df_sorted["group_id"] = groups.values

    # Return to sorted-by-record_id order (already sorted) and reset index
    return df_sorted.sort_values("record_id").reset_index(drop=True)


def build_summary(df: pd.DataFrame, seed: int) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, dict] = {
        "seed": seed,
        "fold_assignment": {
            "split_for_fold": {str(fold): FOLD_TO_SPLIT.get(fold, "train") for fold in range(5)},
            "test_fold": 0,
            "val_fold": 4,
            "train_folds": [fold for fold in range(5) if FOLD_TO_SPLIT.get(fold, "train") == "train"],
        },
        "splits": {},
    }

    for split_name in ["train", "val", "test"]:
        mask = df["split"] == split_name
        split_df = df.loc[mask]
        summary["splits"][split_name] = {
            "num_samples": int(mask.sum()),
            "class_positive_counts": {
                label: int(split_df[label].sum()) for label in LABEL_COLUMNS
            },
        }

    return summary


def main() -> None:
    args = parse_args()

    if not args.input_manifest.exists():
        raise FileNotFoundError(f"Input manifest not found: {args.input_manifest}")

    df = pd.read_csv(args.input_manifest)

    df_with_splits = compute_split_assignments(df, seed=args.seed)

    output_manifest = args.output_manifest
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    df_with_splits.to_csv(output_manifest, index=False)

    summary = build_summary(df_with_splits, seed=args.seed)
    splits_json_path = args.splits_json
    splits_json_path.parent.mkdir(parents=True, exist_ok=True)
    with splits_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote manifest with splits to: {output_manifest}")
    print(f"Wrote split summary to: {splits_json_path}")


if __name__ == "__main__":
    main()

