#!/usr/bin/env python3
"""
Zero-shot evaluation script for ECGFounder.

Loads a pre-trained ECG foundation model checkpoint, optionally remaps the
classification head to a subset of output indices, and evaluates it on the
validation or test split without any gradient updates.

Examples
--------
Evaluate on the MedalCare test split using the default foundation checkpoint:
    python scripts/eval_zero_shot.py --split test

Evaluate on the validation split with an explicit head mapping:
    python scripts/eval_zero_shot.py --split val --head-indices 0,1,2,3,4,5,6,7

Evaluate PTB-XL in zero-shot mode (always uses the original foundation checkpoint):
    python scripts/eval_zero_shot.py --dataset ptbxl --split test \
        --ptbxl-root ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from metrics import AVAILABLE_METRICS, compute_multilabel_metrics  # pylint: disable=wrong-import-position
from net1d import Net1D  # pylint: disable=wrong-import-position
from scripts.datasets import PTBXLDataset, get_dataset  # pylint: disable=wrong-import-position


DEFAULT_MANIFEST = REPO_ROOT / "MedalRaw" / "medalcare_filtered_manifest.csv"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoint" / "12_lead_ECGFounder.pth"
DEFAULT_PTBXL_ROOT = REPO_ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
DATASET_CHOICES = ("medalcare", "ptbxl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot evaluation for ECGFounder.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASET_CHOICES,
        default="medalcare",
        help="Which dataset to evaluate (default: medalcare).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to ECGFounder checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Path to MedalCare manifest CSV with WFDB paths (default: {DEFAULT_MANIFEST}).",
    )
    parser.add_argument(
        "--ptbxl-root",
        type=Path,
        default=DEFAULT_PTBXL_ROOT,
        help=(
            "Root directory for the PTB-XL dataset "
            f"(default: {DEFAULT_PTBXL_ROOT}). Only used when --dataset=ptbxl."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("val", "test"),
        default="val",
        help="Which split to evaluate (default: val).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for inference (default: 128).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (default: 0).",
    )
    parser.add_argument(
        "--head-indices",
        type=str,
        default=None,
        help="Comma-separated list of indices from the foundation head to map onto the dataset labels.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="ap,brier,roc_auc",
        help=f"Comma-separated metrics to compute. Supported: {', '.join(AVAILABLE_METRICS)}",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional identifier for outputs/<run_id>. Defaults to timestamp.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Manually specify device, e.g., cuda:0 or cpu (auto-detect by default).",
    )
    return parser.parse_args()


def resolve_device(explicit: Optional[str]) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    manifest_df = pd.read_csv(path)
    if "split" not in manifest_df.columns:
        raise ValueError("Manifest must contain a 'split' column.")
    if "wfdb_path" not in manifest_df.columns:
        raise ValueError("Manifest must contain a 'wfdb_path' column.")
    return manifest_df


def filter_split(df, split: str):
    subset = df[df["split"].str.lower() == split.lower()].reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"No samples found for split '{split}'.")
    return subset


def build_medalcare_loader(
    manifest_path: Path,
    split: str,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, list[str], dict]:
    manifest_df = load_manifest(manifest_path)
    split_df = filter_split(manifest_df, split)
    label_columns = [col for col in split_df.columns if col.startswith("label_")]
    if not label_columns:
        raise ValueError("Manifest must contain label columns named label_0 ... label_n.")

    dataset_df = split_df[["wfdb_path"] + label_columns].copy()
    dataset = get_dataset("medalcare", ecg_path="", labels_df=dataset_df)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    meta = {
        "manifest": str(manifest_path),
        "num_samples": int(len(split_df)),
        "split": split,
    }
    return loader, label_columns, meta


def build_ptbxl_loader(
    root: Path,
    split: str,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, list[str], dict]:
    if not root.exists():
        raise FileNotFoundError(f"PTB-XL root not found: {root}")
    dataset = get_dataset(
        "ptbxl",
        root=root,
        split=split,
        sampling_rate=500,
        signal_duration=10.0,
        use_high_res=True,
        return_metadata=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    meta = {
        "ptbxl_root": str(root),
        "num_samples": int(len(dataset)),
        "split": split,
    }
    label_names = list(PTBXLDataset.SUPERCLASS_LABELS)
    return loader, label_names, meta


def enforce_ptbxl_checkpoint(requested: Path) -> Path:
    default_resolved = DEFAULT_CHECKPOINT.resolve()
    try:
        requested_resolved = requested.resolve()
    except FileNotFoundError:
        requested_resolved = requested
    if requested_resolved != default_resolved:
        print(
            "[INFO] Overriding checkpoint "
            f"'{requested}' with '{DEFAULT_CHECKPOINT}' for PTB-XL zero-shot evaluation."
        )
    return DEFAULT_CHECKPOINT


def parse_head_indices(arg: Optional[str], expected: int) -> Optional[List[int]]:
    if arg is None:
        return None
    indices = [int(token.strip()) for token in arg.split(",") if token.strip()]
    if len(indices) != expected:
        raise ValueError(
            f"--head-indices expected {expected} entries, received {len(indices)}."
        )
    return indices


def instantiate_model(n_classes: int, device: torch.device) -> Net1D:
    model = Net1D(
        in_channels=12,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=n_classes,
    )
    model.to(device)
    return model


def remap_head(
    state_dict: dict,
    n_classes: int,
    head_indices: Optional[Iterable[int]] = None,
) -> dict:
    weight = state_dict["dense.weight"]
    bias = state_dict["dense.bias"]
    total_classes = weight.shape[0]

    if head_indices is not None:
        idx_tensor = torch.tensor(
            list(head_indices),
            dtype=torch.long,
            device=weight.device,
        )
        if idx_tensor.max().item() >= total_classes:
            raise ValueError(
                f"Requested head index {idx_tensor.max().item()} exceeds checkpoint classes ({total_classes})."
            )
        weight = weight.index_select(0, idx_tensor)
        bias = bias.index_select(0, idx_tensor)
    elif total_classes >= n_classes:
        weight = weight[:n_classes]
        bias = bias[:n_classes]
    else:
        raise ValueError(
            f"Checkpoint dense layer has {total_classes} classes, fewer than required {n_classes}."
        )

    state_dict = state_dict.copy()
    state_dict["dense.weight"] = weight
    state_dict["dense.bias"] = bias
    return state_dict


def load_model(
    checkpoint_path: Path,
    n_classes: int,
    device: torch.device,
    head_indices: Optional[List[int]],
) -> Net1D:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing 'state_dict'.")

    model = instantiate_model(n_classes, device)
    state_dict = remap_head(checkpoint["state_dict"], n_classes, head_indices)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def gather_predictions(
    model: Net1D,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device, non_blocking=True)
            logits = model(signals)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)
            targets.append(labels.cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if args.dataset == "medalcare":
        loader, label_names, dataset_meta = build_medalcare_loader(
            args.manifest, args.split, args.batch_size, args.num_workers
        )
        checkpoint_path = args.checkpoint
    else:
        loader, label_names, dataset_meta = build_ptbxl_loader(
            args.ptbxl_root, args.split, args.batch_size, args.num_workers
        )
        checkpoint_path = enforce_ptbxl_checkpoint(args.checkpoint)

    num_classes = len(label_names)
    head_indices = parse_head_indices(args.head_indices, num_classes)
    model = load_model(checkpoint_path, num_classes, device, head_indices)

    sample_count = dataset_meta.get("num_samples", len(loader.dataset))
    print(
        f"Evaluating dataset '{args.dataset}' split '{args.split}' "
        f"with {sample_count} records using device {device}."
    )

    y_pred, y_true = gather_predictions(model, loader, device)

    requested_metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not requested_metrics:
        requested_metrics = list(AVAILABLE_METRICS)
    unknown = [m for m in requested_metrics if m not in AVAILABLE_METRICS]
    if unknown:
        raise ValueError(f"Unsupported metrics requested: {unknown}.")

    metrics_result = compute_multilabel_metrics(y_true, y_pred, requested_metrics)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs_dir = REPO_ROOT / "outputs" / run_id
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / f"zero_shot_{args.split}.json"

    payload = {
        "run_id": run_id,
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": int(sample_count),
        "checkpoint": str(checkpoint_path),
        "manifest": str(args.manifest) if args.dataset == "medalcare" else None,
        "ptbxl_root": str(args.ptbxl_root) if args.dataset == "ptbxl" else None,
        "dataset_meta": dataset_meta,
        "label_names": label_names,
        "metrics_requested": requested_metrics,
        "metrics": metrics_result,
        "head_indices": head_indices,
    }

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    macro_summary = metrics_result["macro"]
    summary_parts = []
    for metric in requested_metrics:
        value = macro_summary.get(metric)
        if value is None:
            summary_parts.append(f"{metric}:N/A")
        else:
            summary_parts.append(f"{metric}:{value:.4f}")
    summary_str = ", ".join(summary_parts)
    print(f"Saved zero-shot metrics to {output_path}")
    print(f"Macro summary -> {summary_str}")


if __name__ == "__main__":
    main()

