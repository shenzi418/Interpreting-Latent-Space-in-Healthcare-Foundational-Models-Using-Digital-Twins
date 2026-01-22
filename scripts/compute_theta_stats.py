#!/usr/bin/env python3
"""Compute θ normalization stats from MedalCare train split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.datasets import LVEF_12lead_cls_Dataset


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "medalcare_filtered_manifest_dataset_split.csv"
DEFAULT_THETA = REPO_ROOT / "config" / "theta.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "theta_stats.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute θ normalization stats from MedalCare train split."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Manifest with split column (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--theta-config",
        type=Path,
        default=DEFAULT_THETA,
        help=f"θ contract JSON (default: {DEFAULT_THETA})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output stats JSON (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for stats accumulation.",
    )
    return parser.parse_args()


def load_transforms(theta_config: Path) -> List[str]:
    payload = json.loads(theta_config.read_text(encoding="utf-8"))
    return [entry.get("transform", "none") for entry in payload.get("theta", [])]


def apply_transform(values: np.ndarray, transforms: List[str], mask: np.ndarray) -> np.ndarray:
    out = values.copy()
    for idx, transform in enumerate(transforms):
        if transform == "none":
            continue
        if transform == "log":
            valid = (out[:, idx] > 0) & (mask[:, idx] > 0)
            out[~valid, idx] = 0.0
            mask[~valid, idx] = 0.0
            out[valid, idx] = np.log(out[valid, idx])
        elif transform == "logit":
            valid = (out[:, idx] > 0) & (out[:, idx] < 1) & (mask[:, idx] > 0)
            out[~valid, idx] = 0.0
            mask[~valid, idx] = 0.0
            out[valid, idx] = np.log(out[valid, idx] / (1 - out[valid, idx]))
    return out


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    if "split" not in df.columns:
        raise ValueError("Manifest must include a 'split' column.")
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    if train_df.empty:
        raise ValueError("No training rows found in manifest.")

    transforms = load_transforms(args.theta_config)
    dataset = LVEF_12lead_cls_Dataset(
        ecg_path="",
        labels_df=train_df,
        include_theta=True,
        theta_config=args.theta_config,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    theta_len = len(transforms)
    sum_vec = np.zeros(theta_len, dtype=np.float64)
    sumsq_vec = np.zeros(theta_len, dtype=np.float64)
    count_vec = np.zeros(theta_len, dtype=np.float64)

    for batch in loader:
        theta = batch[2].numpy()
        mask = batch[3].numpy()
        theta = apply_transform(theta, transforms, mask)
        sum_vec += (theta * mask).sum(axis=0)
        sumsq_vec += ((theta ** 2) * mask).sum(axis=0)
        count_vec += mask.sum(axis=0)

    mean = np.where(count_vec > 0, sum_vec / count_vec, 0.0)
    var = np.where(count_vec > 0, sumsq_vec / count_vec - mean ** 2, 0.0)
    std = np.sqrt(np.clip(var, 1e-12, None))

    payload: Dict[str, object] = {
        "theta_config": str(args.theta_config),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": count_vec.tolist(),
        "transform": transforms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote θ stats to {args.output}")


if __name__ == "__main__":
    main()

