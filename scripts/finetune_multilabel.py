import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scripts.datasets import (  # pylint: disable=wrong-import-position
    LVEF_12lead_cls_Dataset,
    PTBXLDataset,
    get_dataset,
)
from finetune_model import (  # pylint: disable=wrong-import-position
    ft_12lead_ECGFounder,
    ft_multihead_ECGFounder,
    freeze_backbone_except_adapters,
)
from metrics import AVAILABLE_METRICS, compute_multilabel_metrics  # pylint: disable=wrong-import-position
from losses.mmd import mmd_rbf  # pylint: disable=wrong-import-position
from util import save_checkpoint  # pylint: disable=wrong-import-position

DEFAULT_MANIFEST = REPO_ROOT / "data" / "medalcare_filtered_manifest_dataset_split.csv"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoint" / "12_lead_ECGFounder.pth"
DEFAULT_OUTPUTS = REPO_ROOT / "outputs"
DEFAULT_PTBXL_ROOT = (
    REPO_ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
PTBXL_OUTPUT_ROOT = REPO_ROOT / "outputs" / "ptbxl_baselines"
DATASET_CHOICES = ("medalcare", "ptbxl")
JOINT_CHOICES = ("medalcare+ptbxl",)

# ---------------------------------------------------------------------------
# Exp 7: Shared-head label remapping (3-class: NORM, MI, CD)
# ---------------------------------------------------------------------------
SHARED_LABELS: Tuple[str, ...] = ("NORM", "MI", "CD")
N_SHARED: int = 3

MEDALCARE_REMAP: Dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 2, 5: 2, 7: 2}
PTBXL_REMAP: Dict[int, int] = {0: 0, 1: 1, 4: 2}

MEDALCARE_KEEP_LABELS: Tuple[int, ...] = (0, 1, 2, 3, 5, 7)
MEDALCARE_DROP_LABELS: Tuple[int, ...] = (4, 6)  # lae, fam


def remap_labels(
    labels: torch.Tensor,
    remap_dict: Dict[int, int],
    n_shared: int,
    device: torch.device,
) -> torch.Tensor:
    """Remap a multi-label tensor from the original class space to the shared space.

    Args:
        labels: (B, C_original) binary label tensor.
        remap_dict: mapping from source column index to target column index.
        n_shared: number of target classes.
        device: torch device for the output tensor.

    Returns:
        (B, n_shared) binary label tensor in the shared class space.
    """
    batch_size = labels.shape[0]
    shared = torch.zeros(batch_size, n_shared, device=device)
    for src_idx, tgt_idx in remap_dict.items():
        shared[:, tgt_idx] = torch.clamp(
            shared[:, tgt_idx] + labels[:, src_idx], max=1.0
        )
    return shared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ECGFounder on MedalCare-XL style multi-label data.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASET_CHOICES,
        default="medalcare",
        help="Dataset to train on (default: medalcare).",
    )
    parser.add_argument(
        "--joint-datasets",
        type=str,
        choices=JOINT_CHOICES,
        default=None,
        help="Optional joint training setting (e.g., 'medalcare+ptbxl'). Default: None.",
    )
    parser.add_argument(
        "--multi-head",
        action="store_true",
        help="Use the multi-head model (required for joint training).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Path to MedalCare manifest CSV (default: {DEFAULT_MANIFEST}).",
    )
    parser.add_argument(
        "--ptbxl-root",
        type=Path,
        default=DEFAULT_PTBXL_ROOT,
        help=(
            "Root directory for PTB-XL data "
            f"(default: {DEFAULT_PTBXL_ROOT}). Only used when --dataset=ptbxl."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Pre-trained checkpoint to initialise from (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional identifier for outputs/<run_id>. Defaults to timestamp.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers. Use 0 on Windows.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for Adam.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="accuracy,f1,recall,specificity,precision,brier,roc_auc",
        help=f"Comma-separated list of metrics to compute. Supported: {', '.join(AVAILABLE_METRICS)}",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for converting predicted probabilities to binary labels (default: 0.5).",
    )
    parser.add_argument(
        "--linear-probe",
        action="store_true",
        help="Freeze the feature extractor and only train the classification head.",
    )
    parser.add_argument(
        "--ignore-splits",
        action="store_true",
        help="Ignore manifest splits and create random train/val/test partitions.",
    )
    parser.add_argument(
        "--eval-on-train",
        action="store_true",
        help="Also compute metrics on the training set (useful for debugging).",
    )
    parser.add_argument(
        "--early-stop-lr",
        type=float,
        default=1e-5,
        help="Stop training if ReduceLROnPlateau pushes LR below this value.",
    )
    parser.add_argument(
        "--log-tensors",
        action="store_true",
        help="Save raw predictions and targets for each split under the outputs directory.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the encoder/backbone parameters and only train the head.",
    )
    parser.add_argument(
        "--use-adapter",
        action="store_true",
        help="Enable stage-level residual adapters in the single-head encoder (default: off).",
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Disable adapters in the multi-head model (default: adapters on for multi-head).",
    )
    parser.add_argument(
        "--head-type",
        type=str,
        choices=("linear", "mlp"),
        default="linear",
        help="Classification head architecture to attach to the encoder.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Label smoothing factor applied to BCE targets (default: 0.05).",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping value (L2 norm). Use 0 or negative to disable.",
    )
    parser.add_argument(
        "--lr-head",
        type=float,
        default=1e-3,
        help="Learning rate for head parameters (default: 1e-3).",
    )
    parser.add_argument(
        "--lr-encoder",
        type=float,
        default=1e-5,
        help="Learning rate for encoder parameters (default: 1e-5).",
    )
    parser.add_argument(
        "--theta-config",
        type=Path,
        default=REPO_ROOT / "config" / "theta.json",
        help="θ contract JSON (default: config/theta.json).",
    )
    parser.add_argument(
        "--theta-stats",
        type=Path,
        default=REPO_ROOT / "outputs" / "theta_stats.json",
        help="θ normalization stats JSON (default: outputs/theta_stats.json).",
    )
    parser.add_argument(
        "--theta-eval-stats",
        type=Path,
        default=None,
        help="θ stats JSON for evaluation/denorm (defaults to --theta-stats).",
    )
    parser.add_argument(
        "--theta-core-config",
        type=Path,
        default=None,
        help="Optional θ_core config to restrict evaluation metrics.",
    )
    parser.add_argument(
        "--lambda-phys",
        type=float,
        default=1.0,
        help="Weight for physics loss (default: 1.0).",
    )
    parser.add_argument(
        "--physics-hidden",
        type=int,
        default=256,
        help="Hidden size for physics head MLP (default: 256).",
    )
    parser.add_argument(
        "--physics-dropout",
        type=float,
        default=0.0,
        help="Dropout for physics head MLP (default: 0.0).",
    )
    parser.add_argument(
        "--physics-loss",
        type=str,
        choices=("mse", "huber"),
        default="mse",
        help="Loss for physics head (default: mse).",
    )
    parser.add_argument(
        "--physics-metrics",
        action="store_true",
        help="Compute physics MAE/R2 on MedalCare test split after training.",
    )
    parser.add_argument(
        "--physics-plots",
        action="store_true",
        help="Save physics sanity plots (scatter + distributions).",
    )
    parser.add_argument(
        "--physics-only",
        action="store_true",
        help="Stage B: train physics head only (freeze encoder + classifiers).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic training.",
    )
    parser.add_argument(
        "--best-checkpoint-name",
        type=str,
        default=None,
        help="Custom filename for the best checkpoint saved inside outputs/<run_id>/checkpoints/.",
    )
    parser.add_argument(
        "--lambda-mmd",
        type=float,
        default=0.0,
        help="Weight for the MMD penalty between domain features (default: 0.0).",
    )
    parser.add_argument(
        "--domain-column",
        type=str,
        default="domain",
        help="Manifest column indicating domain labels for MMD (default: 'domain').",
    )
    parser.add_argument(
        "--shared-head",
        action="store_true",
        help="Exp 7: use a single shared classification head (3 classes: NORM, MI, CD) "
             "for both MedalCare and PTB-XL. Requires both datasets.",
    )
    parser.add_argument(
        "--dual-head-shared-labels",
        action="store_true",
        help="Pre-Phase-B redo (supervisor 2026-04-29): joint dual-head training "
             "(separate MedalCare/PTB-XL heads, both sized to 3 = NORM/MI/CD) using "
             "the same MedalCare/PTB-XL filtering and label remapping as --shared-head. "
             "Use --lambda-mmd 0 for the exp5_3class baseline; use --lambda-mmd 0.1 "
             "--class-cond-mmd for the exp6_3class ccmmd variant.",
    )
    parser.add_argument(
        "--class-cond-mmd",
        action="store_true",
        help="Use class-conditional MMD instead of class-agnostic MMD (Exp 7-ccmmd).",
    )
    return parser.parse_args()


def set_deterministic(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_metrics(metrics_to_compute: List[str]) -> List[str]:
    metrics = [m.strip() for m in metrics_to_compute if m.strip()]
    if not metrics:
        metrics = list(AVAILABLE_METRICS)
    unknown = [m for m in metrics if m not in AVAILABLE_METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics requested: {unknown}. Supported: {AVAILABLE_METRICS}")
    return metrics


def ensure_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path)
    required_columns = {"wfdb_path"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    label_columns = [col for col in df.columns if col.startswith("label_")]
    if not label_columns:
        raise ValueError("Manifest must contain columns named label_0 ... label_n for multi-label targets.")
    return df


def prepare_splits(
    df: pd.DataFrame,
    ignore_splits: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not ignore_splits and "split" in df.columns:
        split_counts = df["split"].value_counts()
        required = {"train", "val", "test"}
        if required.issubset(split_counts.index) and all(split_counts[s] > 0 for s in required):
            train_df = df[df["split"] == "train"].reset_index(drop=True)
            val_df = df[df["split"] == "val"].reset_index(drop=True)
            test_df = df[df["split"] == "test"].reset_index(drop=True)
            return train_df, val_df, test_df

    # Fallback: deterministic random split
    from sklearn.model_selection import train_test_split

    label_columns = [col for col in df.columns if col.startswith("label_")]
    stratify_target = df[label_columns].sum(axis=1)
    try:
        train_df, temp_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=np.clip(stratify_target, 0, 1),
        )
    except ValueError:
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    try:
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42,
            stratify=np.clip(temp_df[label_columns].sum(axis=1), 0, 1),
        )
    except ValueError:
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def make_dataloader(
    df: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    include_domain: bool = False,
    domain_column: Optional[str] = None,
    domain_map: Optional[dict] = None,
    include_theta: bool = False,
    theta_config: Optional[Path] = None,
    theta_stats: Optional[Path] = None,
) -> DataLoader:
    labels_df = df.reset_index(drop=True)
    if include_domain and domain_map is None:
        raise ValueError("domain_map must be provided when include_domain is True.")
    dataset = LVEF_12lead_cls_Dataset(
        ecg_path="",
        labels_df=labels_df,
        include_metadata=include_domain,
        domain_column=domain_column,
        domain_map=domain_map,
        include_theta=include_theta,
        theta_config=theta_config,
        theta_stats=theta_stats,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_ptbxl_loader(
    dataset: PTBXLDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def enforce_ptbxl_checkpoint(requested: Path) -> Path:
    default_resolved = DEFAULT_CHECKPOINT.resolve()
    try:
        requested_resolved = requested.resolve()
    except FileNotFoundError:
        requested_resolved = requested
    if requested_resolved != default_resolved:
        print(
            "[INFO] Overriding checkpoint "
            f"'{requested}' with '{DEFAULT_CHECKPOINT}' for PTB-XL Stage 1 training."
        )
    return DEFAULT_CHECKPOINT


def build_medalcare_loaders(
    args: argparse.Namespace,
    warn_mmd: bool = True,
    include_theta: bool = False,
    force_no_domain: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[DataLoader], List[str], dict]:
    manifest = ensure_manifest(args.manifest)
    label_columns = [col for col in manifest.columns if col.startswith("label_")]
    n_classes = len(label_columns)
    train_df, val_df, test_df = prepare_splits(manifest, args.ignore_splits)
    print(f"Loaded manifest with {len(manifest)} records.")
    print(f"Train/val/test sizes: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    include_domain = (
        False
        if force_no_domain
        else (args.lambda_mmd > 0.0 and args.domain_column in train_df.columns)
    )
    if include_domain:
        unique_domains = [
            val for val in pd.Series(train_df[args.domain_column]).dropna().unique()
        ]
        domain_map = {value: idx for idx, value in enumerate(sorted(unique_domains))}
        print(f"Detected domains for MMD: {domain_map}")
    else:
        if warn_mmd and args.lambda_mmd > 0.0 and not force_no_domain:
            print(
                f"[WARN] Domain column '{args.domain_column}' not found; disabling MMD penalty."
            )
        domain_map = None

    train_loader = make_dataloader(
        train_df,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        include_domain=include_domain,
        domain_column=args.domain_column if include_domain else None,
        domain_map=domain_map,
        include_theta=include_theta,
        theta_config=args.theta_config if include_theta else None,
        theta_stats=args.theta_stats if include_theta else None,
    )
    val_loader = make_dataloader(
        val_df,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        include_theta=include_theta,
        theta_config=args.theta_config if include_theta else None,
        theta_stats=args.theta_stats if include_theta else None,
    )
    test_loader = make_dataloader(
        test_df,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        include_theta=include_theta,
        theta_config=args.theta_config if include_theta else None,
        theta_stats=args.theta_stats if include_theta else None,
    )
    train_eval_loader = (
        make_dataloader(
            train_df,
            args.batch_size,
            args.num_workers,
            shuffle=False,
            include_theta=include_theta,
            theta_config=args.theta_config if include_theta else None,
            theta_stats=args.theta_stats if include_theta else None,
        )
        if args.eval_on_train
        else None
    )
    train_sample_count = len(train_df)
    pos_counts = train_df[label_columns].sum(axis=0).to_numpy()
    dataset_meta = {"manifest": str(args.manifest)}
    return (
        train_loader,
        val_loader,
        test_loader,
        train_eval_loader,
        label_columns,
        {
            "train_sample_count": train_sample_count,
            "pos_counts": pos_counts,
            "dataset_meta": dataset_meta,
        },
    )


def build_ptbxl_loaders(
    args: argparse.Namespace,
    warn_mmd: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[DataLoader], List[str], dict]:
    label_columns = list(PTBXLDataset.SUPERCLASS_LABELS)
    n_classes = len(label_columns)
    dataset_kwargs = dict(
        root=args.ptbxl_root,
        sampling_rate=500,
        signal_duration=10.0,
        use_high_res=True,
        return_metadata=False,
    )
    train_dataset = get_dataset("ptbxl", split="train", **dataset_kwargs)
    val_dataset = get_dataset("ptbxl", split="val", **dataset_kwargs)
    test_dataset = get_dataset("ptbxl", split="test", **dataset_kwargs)
    print(
        "Loaded PTB-XL split counts "
        f"train/val/test: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}"
    )
    if warn_mmd and args.lambda_mmd > 0.0:
        print("[WARN] MMD penalty disabled for PTB-XL (no domain labels available).")

    train_loader = make_ptbxl_loader(
        train_dataset, args.batch_size, args.num_workers, shuffle=True
    )
    val_loader = make_ptbxl_loader(
        val_dataset, args.batch_size, args.num_workers, shuffle=False
    )
    test_loader = make_ptbxl_loader(
        test_dataset, args.batch_size, args.num_workers, shuffle=False
    )
    train_eval_loader = (
        make_ptbxl_loader(train_dataset, args.batch_size, args.num_workers, shuffle=False)
        if args.eval_on_train
        else None
    )
    train_sample_count = len(train_dataset)
    pos_counts = np.asarray(train_dataset.targets.sum(axis=0), dtype=np.float64)
    dataset_meta = {"ptbxl_root": str(args.ptbxl_root)}
    return (
        train_loader,
        val_loader,
        test_loader,
        train_eval_loader,
        label_columns,
        {
            "train_sample_count": train_sample_count,
            "pos_counts": pos_counts,
            "dataset_meta": dataset_meta,
        },
    )


# ---------------------------------------------------------------------------
# Exp 7: Shared-head data loading with sample filtering
# ---------------------------------------------------------------------------

def _filter_medalcare_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Drop MedalCare rows whose only positive label is in MEDALCARE_DROP_LABELS (lae, fam)."""
    keep_cols = [f"label_{i}" for i in MEDALCARE_KEEP_LABELS]
    mask = df[keep_cols].sum(axis=1) > 0
    filtered = df[mask].reset_index(drop=True)
    n_dropped = len(df) - len(filtered)
    print(f"[shared-head] MedalCare filtering: {len(df)} -> {len(filtered)} ({n_dropped} dropped)")
    return filtered


def _filter_ptbxl_dataset(dataset: PTBXLDataset) -> PTBXLDataset:
    """Drop PTB-XL samples whose only positive superclass labels are STTC/HYP."""
    targets = dataset.targets  # (N, 5)
    keep_indices = [idx for src, idx in PTBXL_REMAP.items()]  # columns 0, 1, 4
    keep_mask = np.any(targets[:, keep_indices] > 0, axis=1)
    n_before = len(dataset.records)
    dataset.records = dataset.records[keep_mask].reset_index(drop=True)
    dataset.targets = targets[keep_mask]
    n_after = len(dataset.records)
    print(f"[shared-head] PTB-XL filtering: {n_before} -> {n_after} ({n_before - n_after} dropped)")
    return dataset


def build_shared_head_loaders(
    args: argparse.Namespace,
) -> dict:
    """Build MedalCare + PTB-XL loaders with sample filtering for shared-head training."""
    # --- MedalCare ---
    manifest = ensure_manifest(args.manifest)
    train_df, val_df, test_df = prepare_splits(manifest, args.ignore_splits)
    train_df = _filter_medalcare_manifest(train_df)
    val_df = _filter_medalcare_manifest(val_df)
    test_df = _filter_medalcare_manifest(test_df)

    medal_train = make_dataloader(train_df, args.batch_size, args.num_workers, shuffle=True)
    medal_val = make_dataloader(val_df, args.batch_size, args.num_workers, shuffle=False)
    medal_test = make_dataloader(test_df, args.batch_size, args.num_workers, shuffle=False)

    label_cols_8 = [col for col in train_df.columns if col.startswith("label_")]
    medal_pos_8 = train_df[label_cols_8].sum(axis=0).to_numpy()
    medal_train_n = len(train_df)

    # Compute remapped pos counts for combined pos_weight
    medal_pos_shared = np.zeros(N_SHARED, dtype=np.float64)
    for src_idx, tgt_idx in MEDALCARE_REMAP.items():
        col = f"label_{src_idx}"
        if col in train_df.columns:
            medal_pos_shared[tgt_idx] += train_df[col].sum()

    # --- PTB-XL ---
    dataset_kwargs = dict(
        root=args.ptbxl_root,
        sampling_rate=500,
        signal_duration=10.0,
        use_high_res=True,
        return_metadata=False,
    )
    ptb_train_ds = _filter_ptbxl_dataset(get_dataset("ptbxl", split="train", **dataset_kwargs))
    ptb_val_ds = _filter_ptbxl_dataset(get_dataset("ptbxl", split="val", **dataset_kwargs))
    ptb_test_ds = _filter_ptbxl_dataset(get_dataset("ptbxl", split="test", **dataset_kwargs))

    ptb_train = make_ptbxl_loader(ptb_train_ds, args.batch_size, args.num_workers, shuffle=True)
    ptb_val = make_ptbxl_loader(ptb_val_ds, args.batch_size, args.num_workers, shuffle=False)
    ptb_test = make_ptbxl_loader(ptb_test_ds, args.batch_size, args.num_workers, shuffle=False)

    ptb_train_n = len(ptb_train_ds)
    # Compute remapped pos counts
    ptb_pos_shared = np.zeros(N_SHARED, dtype=np.float64)
    for src_idx, tgt_idx in PTBXL_REMAP.items():
        ptb_pos_shared[tgt_idx] += ptb_train_ds.targets[:, src_idx].sum()

    # --- Combined pos_weight ---
    combined_pos = medal_pos_shared + ptb_pos_shared
    combined_total = medal_train_n + ptb_train_n
    combined_neg = combined_total - combined_pos
    pos_weight_arr = combined_neg / np.clip(combined_pos, 1e-6, None)

    print(f"[shared-head] Combined train: MedalCare {medal_train_n} + PTB-XL {ptb_train_n} = {combined_total}")
    for i, name in enumerate(SHARED_LABELS):
        print(f"  {name}: pos={combined_pos[i]:.0f}, neg={combined_neg[i]:.0f}, weight={pos_weight_arr[i]:.3f}")

    return {
        "medal": {"train": medal_train, "val": medal_val, "test": medal_test},
        "ptb": {"train": ptb_train, "val": ptb_val, "test": ptb_test},
        "pos_weight": pos_weight_arr,
        "medal_train_n": medal_train_n,
        "ptb_train_n": ptb_train_n,
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    label_smoothing: float,
    grad_clip: float,
    lambda_mmd: float,
    task: Optional[str] = None,
) -> Tuple[float, Optional[float]]:
    model.train()
    running_loss = 0.0
    total_samples = 0
    running_mmd = 0.0
    mmd_batches = 0
    progress = tqdm(loader, desc="Training", leave=False)
    for batch in progress:
        if lambda_mmd > 0.0 and isinstance(batch, (list, tuple)) and len(batch) >= 3:
            inputs, targets, domains = batch
            domains = domains.to(device, non_blocking=True)
        else:
            inputs, targets = batch[:2]
            domains = None
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if label_smoothing > 0.0:
            smooth_targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
        else:
            smooth_targets = targets

        if task:
            logits = model(inputs, task=task)
        else:
            logits = model(inputs)
        features = getattr(model, "last_features", None)
        loss = criterion(logits, smooth_targets)
        if lambda_mmd > 0.0 and features is not None and domains is not None:
            domain_ids = domains.view(-1)
            unique_ids = torch.sort(torch.unique(domain_ids)).values
            if unique_ids.numel() >= 2:
                src_id, tgt_id = unique_ids[0], unique_ids[1]
                src_feat = features[domain_ids == src_id]
                tgt_feat = features[domain_ids == tgt_id]
                if src_feat.size(0) > 1 and tgt_feat.size(0) > 1:
                    mmd_value = mmd_rbf(src_feat, tgt_feat)
                    loss = loss + lambda_mmd * mmd_value
                    running_mmd += mmd_value.item()
                    mmd_batches += 1
        optimizer.zero_grad()
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        postfix = {"loss": running_loss / max(total_samples, 1)}
        if mmd_batches > 0:
            postfix["mmd"] = running_mmd / mmd_batches
        progress.set_postfix(**postfix)
    avg_loss = running_loss / max(total_samples, 1)
    avg_mmd = running_mmd / mmd_batches if mmd_batches > 0 else None
    return avg_loss, avg_mmd


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    metrics_to_compute: List[str],
    task: Optional[str] = None,
    threshold: float = 0.5,
) -> Optional[Dict[str, object]]:
    if loader.dataset is None or len(loader.dataset) == 0:
        return None

    model.eval()
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            inputs = batch[0]
            labels = batch[1]
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs, task=task) if task else model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)
            targets.append(labels.cpu().numpy())

    if not preds:
        return None

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets, axis=0)
    metrics = compute_multilabel_metrics(y_true, y_pred, metrics_to_compute, threshold=threshold)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "metrics": metrics,
    }


def evaluate_physics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Optional[Dict[str, object]]:
    if loader.dataset is None or len(loader.dataset) == 0:
        return None

    model.eval()
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating physics", leave=False):
            inputs = batch[0].to(device, non_blocking=True)
            theta = batch[2].to(device, non_blocking=True)
            mask = batch[3].to(device, non_blocking=True)
            theta_pred = model(inputs, task="physics")
            preds.append(theta_pred.cpu().numpy())
            targets.append(theta.cpu().numpy())
            masks.append(mask.cpu().numpy())

    if not preds:
        return None

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets, axis=0)
    y_mask = np.concatenate(masks, axis=0)
    return {"y_true": y_true, "y_pred": y_pred, "mask": y_mask}


def compute_physics_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    theta_stats: Optional[dict] = None,
    theta_names: Optional[List[str]] = None,
    theta_core_names: Optional[List[str]] = None,
    tiny: float = 1e-8,
) -> Dict[str, object]:
    def rankdata(values: np.ndarray) -> np.ndarray:
        series = pd.Series(values)
        return series.rank(method="average").to_numpy(dtype=np.float64)

    stats = theta_stats or {}
    transforms = stats.get("transform")
    means = stats.get("mean")
    stds = stats.get("std")

    indices = list(range(y_true.shape[1]))
    missing_core = []
    if theta_core_names and theta_names:
        name_to_idx = {name: idx for idx, name in enumerate(theta_names)}
        indices = []
        for name in theta_core_names:
            if name in name_to_idx:
                indices.append(name_to_idx[name])
            else:
                missing_core.append(name)

    if indices:
        y_true = y_true[:, indices]
        y_pred = y_pred[:, indices]
        mask = mask[:, indices]
        if theta_names:
            theta_names = [theta_names[i] for i in indices]
        if transforms:
            transforms = [transforms[i] for i in indices]
        if means:
            means = [means[i] for i in indices]
        if stds:
            stds = [stds[i] for i in indices]

    num_params = y_true.shape[1]
    mae = []
    mae_raw = []
    r2 = []
    pearson = []
    spearman = []
    effective_n = []
    for idx in range(num_params):
        m = mask[:, idx] > 0.5
        if not np.any(m):
            mae.append(None)
            mae_raw.append(None)
            r2.append(None)
            pearson.append(None)
            spearman.append(None)
            effective_n.append(0)
            continue
        true = y_true[m, idx]
        pred = y_pred[m, idx]
        mae_val = float(np.mean(np.abs(true - pred)))
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2_val = None if ss_tot < tiny else float(1.0 - ss_res / ss_tot)
        mae.append(mae_val)
        r2.append(r2_val)

        if stds:
            std = stds[idx] if stds[idx] and stds[idx] > 0 else 1.0
            mae_raw.append(float(mae_val * std))
        else:
            mae_raw.append(None)

        true_std = float(np.std(true))
        pred_std = float(np.std(pred))
        if ss_tot < tiny or true_std < tiny or pred_std < tiny:
            pearson.append(None)
            spearman.append(None)
        else:
            pearson_val = float(np.corrcoef(true, pred)[0, 1])
            pearson.append(pearson_val)
            true_rank = rankdata(true)
            pred_rank = rankdata(pred)
            spearman_val = float(np.corrcoef(true_rank, pred_rank)[0, 1])
            spearman.append(spearman_val)
        effective_n.append(int(np.sum(m)))

    mae_values = [v for v in mae if v is not None]
    mae_raw_values = [v for v in mae_raw if v is not None]
    r2_values = [v for v in r2 if v is not None]
    pearson_values = [v for v in pearson if v is not None]
    spearman_values = [v for v in spearman if v is not None]
    r2_tiers = {
        "strong": int(sum(v >= 0.5 for v in r2_values)),
        "moderate": int(sum(0.2 <= v < 0.5 for v in r2_values)),
        "weak": int(sum(v < 0.2 for v in r2_values)),
        "undefined": int(sum(v is None for v in r2)),
    }
    return {
        "theta_names": theta_names,
        "mae_norm": mae,
        "mae_raw": mae_raw,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
        "effective_n": effective_n,
        "summary": {
            "mae_norm_mean": float(np.mean(mae_values)) if mae_values else None,
            "mae_norm_median": float(np.median(mae_values)) if mae_values else None,
            "mae_raw_mean": float(np.mean(mae_raw_values)) if mae_raw_values else None,
            "mae_raw_median": float(np.median(mae_raw_values)) if mae_raw_values else None,
            "r2_mean": float(np.mean(r2_values)) if r2_values else None,
            "r2_median": float(np.median(r2_values)) if r2_values else None,
            "pearson_mean": float(np.mean(pearson_values)) if pearson_values else None,
            "spearman_mean": float(np.mean(spearman_values)) if spearman_values else None,
            "r2_tiers": r2_tiers,
        },
        "meta": {
            "normalization": "zscore" if stds else "none",
            "transform": transforms,
            "missing_theta_core": missing_core,
            "mae_spaces": ["normalized", "raw"] if stds else ["normalized"],
        },
    }


def save_physics_plots(
    outputs_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    indices: Optional[List[int]] = None,
    theta_stats: Optional[dict] = None,
    theta_names: Optional[List[str]] = None,
    max_dims: int = 6,
) -> None:
    import matplotlib.pyplot as plt

    plots_dir = outputs_dir / "physics_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    num_dims = y_true.shape[1]
    if indices is None:
        dims = list(range(min(num_dims, max_dims)))
    else:
        dims = indices[: max_dims]
    stds = theta_stats.get("std") if theta_stats else None
    for idx in dims:
        m = mask[:, idx] > 0.5
        if not np.any(m):
            continue
        true = y_true[m, idx]
        pred = y_pred[m, idx]
        label = f"θ[{idx}]"
        if theta_names and idx < len(theta_names):
            label = theta_names[idx]
        if stds and idx < len(stds):
            std = stds[idx] if stds[idx] and stds[idx] > 0 else 1.0
            true = true * std
            pred = pred * std

        # Scatter plot
        plt.figure(figsize=(5, 5))
        plt.scatter(true, pred, s=8, alpha=0.5)
        min_v = min(true.min(), pred.min())
        max_v = max(true.max(), pred.max())
        plt.plot([min_v, max_v], [min_v, max_v], "--", color="gray")
        plt.xlabel("θ true")
        plt.ylabel("θ pred")
        plt.title(f"{label} scatter")
        plt.tight_layout()
        plt.savefig(plots_dir / f"theta_{idx}_scatter.png", dpi=150)
        plt.close()

        # Distribution plot
        plt.figure(figsize=(6, 4))
        plt.hist(true, bins=30, alpha=0.6, label="true")
        plt.hist(pred, bins=30, alpha=0.6, label="pred")
        plt.legend()
        plt.xlabel("value")
        plt.ylabel("count")
        plt.title(f"{label} distribution")
        plt.tight_layout()
        plt.savefig(plots_dir / f"theta_{idx}_dist.png", dpi=150)
        plt.close()


_PRIMARY_METRIC_PREFERENCE = ("f1", "roc_auc", "accuracy", "recall")


def select_primary_metric(metrics: Dict[str, object], candidates: List[str]) -> Tuple[str, Optional[float]]:
    macro = metrics["metrics"]["macro"]
    # Try the preferred ordering first (only consider metrics the user asked for).
    available = set(candidates)
    for name in _PRIMARY_METRIC_PREFERENCE:
        if name in available:
            value = macro.get(name)
            if value is not None:
                return name, value
    # Fall back to the first candidate with a value.
    for name in candidates:
        value = macro.get(name)
        if value is not None:
            return name, value
    fallback = candidates[0]
    return fallback, macro.get(fallback)


def summarise_macro(metrics: Dict[str, object], metric_names: List[str]) -> Dict[str, Optional[float]]:
    macro = metrics["metrics"]["macro"]
    return {name: macro.get(name) for name in metric_names}


def macro_summary_str(summary: Dict[str, Optional[float]]) -> str:
    return ", ".join(
        f"{name}:{value:.4f}" if value is not None else f"{name}:N/A"
        for name, value in summary.items()
    )


def record_per_class_metrics(
    per_class_records: List[Dict[str, object]],
    split_name: str,
    metrics_package: Dict[str, object],
    label_columns: List[str],
    epoch: int,
) -> None:
    per_class = metrics_package["metrics"]["per_class"]
    supports = metrics_package["metrics"]["support"]
    positives = supports.get("positives", [])
    negatives = supports.get("negatives", [])
    for class_idx, label in enumerate(label_columns):
        row = {
            "epoch": int(epoch),
            "split": split_name,
            "label": label,
            "positives": int(positives[class_idx]) if class_idx < len(positives) else None,
            "negatives": int(negatives[class_idx]) if class_idx < len(negatives) else None,
        }
        for metric_name, values in per_class.items():
            row[metric_name] = values[class_idx]
        per_class_records.append(row)


def save_metrics_report(
    path: Path,
    run_id: str,
    metrics_requested: List[str],
    evaluations: List[Dict[str, object]],
    best_snapshot: Optional[Dict[str, object]],
) -> None:
    payload = {
        "run_id": run_id,
        "metrics_requested": metrics_requested,
        "evaluations": evaluations,
    }
    if best_snapshot is not None:
        payload["best"] = best_snapshot
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def save_numpy_payload(path: Path, name: str, package: Dict[str, object]) -> None:
    np.savez_compressed(path / f"{name}.npz", y_true=package["y_true"], y_pred=package["y_pred"])


def load_theta_names(theta_config_path: Path) -> List[str]:
    payload = json.loads(theta_config_path.read_text(encoding="utf-8"))
    return [entry["name"] for entry in payload.get("theta", [])]


def select_top_theta_indices(
    metric_values: List[Optional[float]],
    top_k: int = 6,
) -> List[int]:
    candidates = [
        (idx, value) for idx, value in enumerate(metric_values) if value is not None
    ]
    candidates.sort(key=lambda item: item[1], reverse=True)
    return [idx for idx, _ in candidates[:top_k]]


def _evaluate_shared_head(
    model: torch.nn.Module,
    loader: DataLoader,
    remap_dict: Dict[int, int],
    device: torch.device,
    metrics_to_compute: List[str],
    label_smoothing: float,
    threshold: float = 0.5,
) -> Optional[Dict[str, object]]:
    """Evaluate the shared-head model on one domain, remapping labels to the shared space."""
    if loader.dataset is None or len(loader.dataset) == 0:
        return None
    model.eval()
    preds_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            inputs = batch[0].to(device, non_blocking=True)
            labels_orig = batch[1].to(device, non_blocking=True)
            labels_shared = remap_labels(labels_orig, remap_dict, N_SHARED, device)
            logits = model(inputs)
            preds_list.append(torch.sigmoid(logits).cpu().numpy())
            targets_list.append(labels_shared.cpu().numpy())
    if not preds_list:
        return None
    y_pred = np.concatenate(preds_list, axis=0)
    y_true = np.concatenate(targets_list, axis=0)
    metrics = compute_multilabel_metrics(y_true, y_pred, metrics_to_compute, threshold=threshold)
    return {"y_true": y_true, "y_pred": y_pred, "metrics": metrics}


def _run_shared_head(args: argparse.Namespace) -> None:
    """Exp 7: Shared-head joint training on 3 overlapping classes (NORM, MI, CD)."""
    set_deterministic(args.seed)
    metrics_to_compute = validate_metrics(args.metrics.split(","))
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    outputs_dir = (DEFAULT_OUTPUTS / run_id).resolve()
    checkpoints_dir = outputs_dir / "checkpoints"
    tensors_dir = outputs_dir / "tensors"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    if args.log_tensors:
        tensors_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"[Exp 7] Shared-head training with {N_SHARED} classes: {SHARED_LABELS}")

    # ---- Data ----
    loader_bundle = build_shared_head_loaders(args)
    medal_train = loader_bundle["medal"]["train"]
    medal_val = loader_bundle["medal"]["val"]
    medal_test = loader_bundle["medal"]["test"]
    ptb_train = loader_bundle["ptb"]["train"]
    ptb_val = loader_bundle["ptb"]["val"]
    ptb_test = loader_bundle["ptb"]["test"]
    pos_weight_arr = loader_bundle["pos_weight"]

    # ---- Model (single-head Net1D with adapters) ----
    model = ft_12lead_ECGFounder(
        device=device,
        pth=str(args.checkpoint),
        n_classes=N_SHARED,
        linear_prob=False,
        use_adapter=True,
        adapter_reduction=16,
        adapter_dropout=0.0,
    )
    freeze_backbone_except_adapters(model)
    for p in model.dense.parameters():
        p.requires_grad = True

    head_params = [p for p in model.dense.parameters() if p.requires_grad]
    encoder_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("dense") and p.requires_grad
    ]
    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr_head})
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": args.lr_encoder})
    if not param_groups:
        raise ValueError("No trainable parameters found.")

    n_head = sum(p.numel() for p in head_params)
    n_adapter = sum(p.numel() for p in encoder_params)
    print(f"Trainable params: head={n_head:,}, adapters={n_adapter:,}, total={n_head + n_adapter:,}")

    # ---- Loss / Optimizer / Scheduler ----
    pos_weight_tensor = torch.tensor(pos_weight_arr, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(param_groups, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, mode="max"
    )

    # ---- Training loop ----
    metrics_log: List[Dict[str, object]] = []
    per_class_records: List[Dict[str, object]] = []
    best_snapshot: Optional[Dict[str, object]] = None
    best_val_score = float("-inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        model.train()
        steps = min(len(medal_train), len(ptb_train))
        running_loss_medal = 0.0
        running_loss_ptb = 0.0
        samples_medal = 0
        samples_ptb = 0
        running_mmd = 0.0
        mmd_batches = 0

        for batch_medal, batch_ptb in tqdm(
            zip(medal_train, ptb_train), total=steps, desc="Training", leave=False
        ):
            medal_inputs, medal_labels_8 = batch_medal[0], batch_medal[1]
            ptb_inputs, ptb_labels_5 = batch_ptb[0], batch_ptb[1]

            medal_inputs = medal_inputs.to(device, non_blocking=True)
            ptb_inputs = ptb_inputs.to(device, non_blocking=True)
            medal_labels_8 = medal_labels_8.to(device, non_blocking=True)
            ptb_labels_5 = ptb_labels_5.to(device, non_blocking=True)

            medal_labels = remap_labels(medal_labels_8, MEDALCARE_REMAP, N_SHARED, device)
            ptb_labels = remap_labels(ptb_labels_5, PTBXL_REMAP, N_SHARED, device)

            if args.label_smoothing > 0.0:
                medal_labels = medal_labels * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
                ptb_labels = ptb_labels * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing

            optimizer.zero_grad()

            logits_medal = model(medal_inputs)
            feat_medal = model.last_features
            logits_ptb = model(ptb_inputs)
            feat_ptb = model.last_features

            loss_medal = criterion(logits_medal, medal_labels)
            loss_ptb = criterion(logits_ptb, ptb_labels)
            loss = loss_medal + loss_ptb

            if args.lambda_mmd > 0.0 and feat_medal.size(0) > 1 and feat_ptb.size(0) > 1:
                if args.class_cond_mmd:
                    from losses.mmd import mmd_rbf_class_conditional
                    mmd_value = mmd_rbf_class_conditional(
                        feat_medal, feat_ptb, medal_labels, ptb_labels
                    )
                else:
                    mmd_value = mmd_rbf(feat_medal, feat_ptb)
                loss = loss + args.lambda_mmd * mmd_value
                running_mmd += mmd_value.item()
                mmd_batches += 1

            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            b_m = medal_inputs.size(0)
            b_p = ptb_inputs.size(0)
            running_loss_medal += loss_medal.item() * b_m
            running_loss_ptb += loss_ptb.item() * b_p
            samples_medal += b_m
            samples_ptb += b_p

        avg_loss_medal = running_loss_medal / max(samples_medal, 1)
        avg_loss_ptb = running_loss_ptb / max(samples_ptb, 1)
        avg_mmd = running_mmd / mmd_batches if mmd_batches > 0 else None
        print(f"Training loss -> MedalCare: {avg_loss_medal:.6f}, PTB-XL: {avg_loss_ptb:.6f}")
        if avg_mmd is not None:
            print(f"Average MMD: {avg_mmd:.6f}")

        # ---- Validation ----
        val_medal = _evaluate_shared_head(
            model, medal_val, MEDALCARE_REMAP, device, metrics_to_compute,
            args.label_smoothing, args.threshold,
        )
        val_ptb = _evaluate_shared_head(
            model, ptb_val, PTBXL_REMAP, device, metrics_to_compute,
            args.label_smoothing, args.threshold,
        )
        if val_medal is None or val_ptb is None:
            raise RuntimeError("Validation split is empty – cannot continue.")

        val_summary_medal = summarise_macro(val_medal, metrics_to_compute)
        val_summary_ptb = summarise_macro(val_ptb, metrics_to_compute)
        print(
            "Validation macro -> "
            f"MedalCare: {macro_summary_str(val_summary_medal)}, "
            f"PTB-XL: {macro_summary_str(val_summary_ptb)}"
        )

        _, medal_f1 = select_primary_metric(val_medal, metrics_to_compute)
        _, ptb_f1 = select_primary_metric(val_ptb, metrics_to_compute)
        medal_f1 = medal_f1 if medal_f1 is not None else 0.0
        ptb_f1 = ptb_f1 if ptb_f1 is not None else 0.0
        val_score = (medal_f1 + ptb_f1) / 2.0
        print(f"Checkpoint metric (avg domain F1): {val_score:.4f}")

        scheduler.step(val_score if np.isfinite(val_score) else 0.0)

        # ---- Test ----
        test_medal = _evaluate_shared_head(
            model, medal_test, MEDALCARE_REMAP, device, metrics_to_compute,
            args.label_smoothing, args.threshold,
        )
        test_ptb = _evaluate_shared_head(
            model, ptb_test, PTBXL_REMAP, device, metrics_to_compute,
            args.label_smoothing, args.threshold,
        )
        test_summary_medal = summarise_macro(test_medal, metrics_to_compute) if test_medal else None
        test_summary_ptb = summarise_macro(test_ptb, metrics_to_compute) if test_ptb else None
        if test_summary_medal and test_summary_ptb:
            print(
                "Test macro -> "
                f"MedalCare: {macro_summary_str(test_summary_medal)}, "
                f"PTB-XL: {macro_summary_str(test_summary_ptb)}"
            )

        # ---- Per-class records ----
        shared_label_list = list(SHARED_LABELS)
        record_per_class_metrics(per_class_records, "val_medalcare", val_medal, shared_label_list, epoch)
        record_per_class_metrics(per_class_records, "val_ptbxl", val_ptb, shared_label_list, epoch)
        if test_medal:
            record_per_class_metrics(per_class_records, "test_medalcare", test_medal, shared_label_list, epoch)
        if test_ptb:
            record_per_class_metrics(per_class_records, "test_ptbxl", test_ptb, shared_label_list, epoch)

        # ---- Metrics log ----
        metrics_entry: Dict[str, object] = {
            "epoch": epoch,
            "train_loss": {"medalcare": avg_loss_medal, "ptbxl": avg_loss_ptb},
            "primary_metric": {"name": "avg_domain_f1", "value": val_score},
            "val": {"medalcare": val_summary_medal, "ptbxl": val_summary_ptb},
        }
        if avg_mmd is not None:
            metrics_entry["train_mmd"] = avg_mmd
        if test_summary_medal and test_summary_ptb:
            metrics_entry["test"] = {"medalcare": test_summary_medal, "ptbxl": test_summary_ptb}
        metrics_log.append(metrics_entry)

        # ---- Checkpoint ----
        is_best = bool(val_score > best_val_score)
        if is_best:
            best_val_score = val_score
            print("==> New best validation score; saving checkpoint.")
            checkpoint_state = {
                "epoch": epoch,
                "step": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_score": float(val_score),
                "val_auroc": float(val_score),
            }
            save_checkpoint(checkpoint_state, str(checkpoints_dir))
            best_filename = args.best_checkpoint_name or "linear_best.pt"
            torch.save(checkpoint_state, checkpoints_dir / best_filename)
            best_snapshot = {
                "epoch": epoch,
                "primary_metric": {"name": "avg_domain_f1", "value": val_score},
                "val": {"medalcare": val_summary_medal, "ptbxl": val_summary_ptb},
                "test": {"medalcare": test_summary_medal, "ptbxl": test_summary_ptb},
            }

        current_lr = min(group["lr"] for group in optimizer.param_groups)
        if current_lr < args.early_stop_lr:
            print(
                f"LR {current_lr:g} fell below early-stop threshold "
                f"{args.early_stop_lr:g}; stopping."
            )
            break

    # ---- Save reports ----
    metrics_path = outputs_dir / "metrics.json"
    save_metrics_report(metrics_path, run_id, metrics_to_compute, metrics_log, best_snapshot)
    print(f"Saved metrics report to: {metrics_path}")

    per_class_path = outputs_dir / "per_class_metrics.csv"
    pd.DataFrame(per_class_records).to_csv(per_class_path, index=False)
    print(f"Saved per-class metrics to: {per_class_path}")

    if args.log_tensors and test_medal and test_ptb:
        np.savez_compressed(
            tensors_dir / "test_medalcare.npz",
            y_true=test_medal["y_true"], y_pred=test_medal["y_pred"],
        )
        np.savez_compressed(
            tensors_dir / "test_ptbxl.npz",
            y_true=test_ptb["y_true"], y_pred=test_ptb["y_pred"],
        )

    print("[Exp 7] Shared-head training complete.")


# ---------------------------------------------------------------------------
# Pre-Phase-B (supervisor 2026-04-29): Dual-head joint training on the shared
# 3-class label space. Reuses build_shared_head_loaders for filtering, but
# uses two separate per-domain heads sized to N_SHARED via MultiHeadECGFounder.
# ---------------------------------------------------------------------------

def _evaluate_dual_head_shared(
    model: torch.nn.Module,
    loader: DataLoader,
    task: str,
    remap_dict: Dict[int, int],
    device: torch.device,
    metrics_to_compute: List[str],
    threshold: float = 0.5,
) -> Optional[Dict[str, object]]:
    """Evaluate one head of the dual-head shared-label model on its domain."""
    if loader.dataset is None or len(loader.dataset) == 0:
        return None
    model.eval()
    preds_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating ({task})", leave=False):
            inputs = batch[0].to(device, non_blocking=True)
            labels_orig = batch[1].to(device, non_blocking=True)
            labels_shared = remap_labels(labels_orig, remap_dict, N_SHARED, device)
            logits = model(inputs, task=task)
            preds_list.append(torch.sigmoid(logits).cpu().numpy())
            targets_list.append(labels_shared.cpu().numpy())
    if not preds_list:
        return None
    y_pred = np.concatenate(preds_list, axis=0)
    y_true = np.concatenate(targets_list, axis=0)
    metrics = compute_multilabel_metrics(y_true, y_pred, metrics_to_compute, threshold=threshold)
    return {"y_true": y_true, "y_pred": y_pred, "metrics": metrics}


def _run_dual_head_shared_labels(args: argparse.Namespace) -> None:
    """Pre-Phase-B redo: dual-head joint training on the shared 3-class label space.

    Mirrors `_run_shared_head` (same data filtering, alternating-batch loop,
    optional MMD/ccMMD, checkpoint metric = mean of per-domain F1) but uses
    a MultiHeadECGFounder with two per-domain heads of width N_SHARED = 3
    instead of a single shared head. Per-domain pos_weight is computed from
    the filtered & remapped MedalCare / PTB-XL training labels.
    """
    set_deterministic(args.seed)
    metrics_to_compute = validate_metrics(args.metrics.split(","))
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    outputs_dir = (DEFAULT_OUTPUTS / run_id).resolve()
    checkpoints_dir = outputs_dir / "checkpoints"
    tensors_dir = outputs_dir / "tensors"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    if args.log_tensors:
        tensors_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    mmd_label = "ccMMD" if args.class_cond_mmd else "MMD"
    if args.lambda_mmd > 0.0:
        align_msg = f"{mmd_label} (lambda={args.lambda_mmd})"
    else:
        align_msg = "no alignment"
    print(
        f"[Pre-Phase-B] Dual-head shared-label training "
        f"({N_SHARED} classes: {SHARED_LABELS}, {align_msg})"
    )

    # ---- Data: reuse the shared-head loader builder (same filtering) ----
    loader_bundle = build_shared_head_loaders(args)
    medal_train = loader_bundle["medal"]["train"]
    medal_val = loader_bundle["medal"]["val"]
    medal_test = loader_bundle["medal"]["test"]
    ptb_train = loader_bundle["ptb"]["train"]
    ptb_val = loader_bundle["ptb"]["val"]
    ptb_test = loader_bundle["ptb"]["test"]

    # ---- Per-domain pos_weight from filtered & remapped train labels ----
    # Re-derive per-domain shared positive counts from the filtered training
    # datasets to size pos_weight correctly for each head.
    medal_train_ds = medal_train.dataset
    ptb_train_ds = ptb_train.dataset

    medal_pos_shared = np.zeros(N_SHARED, dtype=np.float64)
    medal_n = len(medal_train_ds)
    medal_df = getattr(medal_train_ds, "labels_df", None)
    if medal_df is None:
        raise RuntimeError(
            "Expected MedalCare training dataset to expose its underlying DataFrame "
            "as `.labels_df` for pos_weight calculation."
        )
    for src_idx, tgt_idx in MEDALCARE_REMAP.items():
        col = f"label_{src_idx}"
        if col in medal_df.columns:
            medal_pos_shared[tgt_idx] += medal_df[col].sum()
    medal_pos_shared = np.minimum(medal_pos_shared, medal_n)
    medal_neg_shared = medal_n - medal_pos_shared
    pos_weight_medal_arr = medal_neg_shared / np.clip(medal_pos_shared, 1e-6, None)

    ptb_pos_shared = np.zeros(N_SHARED, dtype=np.float64)
    ptb_n = len(ptb_train_ds)
    for src_idx, tgt_idx in PTBXL_REMAP.items():
        ptb_pos_shared[tgt_idx] += float(ptb_train_ds.targets[:, src_idx].sum())
    ptb_pos_shared = np.minimum(ptb_pos_shared, ptb_n)
    ptb_neg_shared = ptb_n - ptb_pos_shared
    pos_weight_ptb_arr = ptb_neg_shared / np.clip(ptb_pos_shared, 1e-6, None)

    print(f"[Pre-Phase-B] MedalCare train n={medal_n}, PTB-XL train n={ptb_n}")
    for i, name in enumerate(SHARED_LABELS):
        print(
            f"  {name}: medal pos={medal_pos_shared[i]:.0f} (w={pos_weight_medal_arr[i]:.3f}), "
            f"ptb pos={ptb_pos_shared[i]:.0f} (w={pos_weight_ptb_arr[i]:.3f})"
        )

    # ---- Model: dual-head MultiHeadECGFounder, both heads sized to N_SHARED ----
    model = ft_multihead_ECGFounder(
        device=device,
        pth=str(args.checkpoint),
        n_medal_classes=N_SHARED,
        n_ptb_classes=N_SHARED,
        n_theta=0,
        physics_hidden=args.physics_hidden,
        physics_dropout=args.physics_dropout,
        linear_prob=True,
        use_adapter=True,
    )
    head_params = [p for n, p in model.named_parameters() if "head_" in n and p.requires_grad]
    encoder_params = [
        p for n, p in model.named_parameters()
        if "head_" not in n and p.requires_grad
    ]
    if not head_params and not encoder_params:
        raise ValueError("No trainable parameters found.")
    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr_head})
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": args.lr_encoder})

    n_head = sum(p.numel() for p in head_params)
    n_adapter = sum(p.numel() for p in encoder_params)
    print(
        f"[Pre-Phase-B] head_medal.out_features={model.head_medal.out_features}, "
        f"head_ptb.out_features={model.head_ptb.out_features}"
    )
    assert model.head_medal.out_features == N_SHARED, "head_medal must be size N_SHARED"
    assert model.head_ptb.out_features == N_SHARED, "head_ptb must be size N_SHARED"
    print(
        f"[Pre-Phase-B] Trainable params: heads={n_head:,}, adapters={n_adapter:,}, "
        f"total={n_head + n_adapter:,}"
    )

    # ---- Loss / Optimizer / Scheduler ----
    pos_weight_medal = torch.tensor(pos_weight_medal_arr, dtype=torch.float32, device=device)
    pos_weight_ptb = torch.tensor(pos_weight_ptb_arr, dtype=torch.float32, device=device)
    criterion_medal = nn.BCEWithLogitsLoss(pos_weight=pos_weight_medal)
    criterion_ptb = nn.BCEWithLogitsLoss(pos_weight=pos_weight_ptb)
    optimizer = optim.Adam(param_groups, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, mode="max"
    )

    # ---- Training loop ----
    metrics_log: List[Dict[str, object]] = []
    per_class_records: List[Dict[str, object]] = []
    best_snapshot: Optional[Dict[str, object]] = None
    best_val_score = float("-inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        model.train()
        steps = min(len(medal_train), len(ptb_train))
        running_loss_medal = 0.0
        running_loss_ptb = 0.0
        samples_medal = 0
        samples_ptb = 0
        running_mmd = 0.0
        mmd_batches = 0

        for batch_medal, batch_ptb in tqdm(
            zip(medal_train, ptb_train), total=steps, desc="Training", leave=False
        ):
            medal_inputs, medal_labels_8 = batch_medal[0], batch_medal[1]
            ptb_inputs, ptb_labels_5 = batch_ptb[0], batch_ptb[1]

            medal_inputs = medal_inputs.to(device, non_blocking=True)
            ptb_inputs = ptb_inputs.to(device, non_blocking=True)
            medal_labels_8 = medal_labels_8.to(device, non_blocking=True)
            ptb_labels_5 = ptb_labels_5.to(device, non_blocking=True)

            medal_labels = remap_labels(medal_labels_8, MEDALCARE_REMAP, N_SHARED, device)
            ptb_labels = remap_labels(ptb_labels_5, PTBXL_REMAP, N_SHARED, device)

            if args.label_smoothing > 0.0:
                medal_labels_s = medal_labels * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
                ptb_labels_s = ptb_labels * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
            else:
                medal_labels_s = medal_labels
                ptb_labels_s = ptb_labels

            optimizer.zero_grad()

            logits_medal, feat_medal = model(medal_inputs, task="medalcare", return_features=True)
            logits_ptb, feat_ptb = model(ptb_inputs, task="ptbxl", return_features=True)

            loss_medal = criterion_medal(logits_medal, medal_labels_s)
            loss_ptb = criterion_ptb(logits_ptb, ptb_labels_s)
            loss = loss_medal + loss_ptb

            if args.lambda_mmd > 0.0 and feat_medal.size(0) > 1 and feat_ptb.size(0) > 1:
                if args.class_cond_mmd:
                    from losses.mmd import mmd_rbf_class_conditional
                    mmd_value = mmd_rbf_class_conditional(
                        feat_medal, feat_ptb, medal_labels, ptb_labels
                    )
                else:
                    mmd_value = mmd_rbf(feat_medal, feat_ptb)
                loss = loss + args.lambda_mmd * mmd_value
                running_mmd += mmd_value.item()
                mmd_batches += 1

            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            b_m = medal_inputs.size(0)
            b_p = ptb_inputs.size(0)
            running_loss_medal += loss_medal.item() * b_m
            running_loss_ptb += loss_ptb.item() * b_p
            samples_medal += b_m
            samples_ptb += b_p

        avg_loss_medal = running_loss_medal / max(samples_medal, 1)
        avg_loss_ptb = running_loss_ptb / max(samples_ptb, 1)
        avg_mmd = running_mmd / mmd_batches if mmd_batches > 0 else None
        print(f"Training loss -> MedalCare: {avg_loss_medal:.6f}, PTB-XL: {avg_loss_ptb:.6f}")
        if avg_mmd is not None:
            print(f"Average {mmd_label}: {avg_mmd:.6f}")

        # ---- Validation ----
        val_medal = _evaluate_dual_head_shared(
            model, medal_val, "medalcare", MEDALCARE_REMAP, device, metrics_to_compute,
            args.threshold,
        )
        val_ptb = _evaluate_dual_head_shared(
            model, ptb_val, "ptbxl", PTBXL_REMAP, device, metrics_to_compute,
            args.threshold,
        )
        if val_medal is None or val_ptb is None:
            raise RuntimeError("Validation split is empty -- cannot continue.")

        val_summary_medal = summarise_macro(val_medal, metrics_to_compute)
        val_summary_ptb = summarise_macro(val_ptb, metrics_to_compute)
        print(
            "Validation macro -> "
            f"MedalCare: {macro_summary_str(val_summary_medal)}, "
            f"PTB-XL: {macro_summary_str(val_summary_ptb)}"
        )

        _, medal_f1 = select_primary_metric(val_medal, metrics_to_compute)
        _, ptb_f1 = select_primary_metric(val_ptb, metrics_to_compute)
        medal_f1 = medal_f1 if medal_f1 is not None else 0.0
        ptb_f1 = ptb_f1 if ptb_f1 is not None else 0.0
        val_score = (medal_f1 + ptb_f1) / 2.0
        print(f"Checkpoint metric (avg domain F1): {val_score:.4f}")

        scheduler.step(val_score if np.isfinite(val_score) else 0.0)

        # ---- Test ----
        test_medal = _evaluate_dual_head_shared(
            model, medal_test, "medalcare", MEDALCARE_REMAP, device, metrics_to_compute,
            args.threshold,
        )
        test_ptb = _evaluate_dual_head_shared(
            model, ptb_test, "ptbxl", PTBXL_REMAP, device, metrics_to_compute,
            args.threshold,
        )
        test_summary_medal = summarise_macro(test_medal, metrics_to_compute) if test_medal else None
        test_summary_ptb = summarise_macro(test_ptb, metrics_to_compute) if test_ptb else None
        if test_summary_medal and test_summary_ptb:
            print(
                "Test macro -> "
                f"MedalCare: {macro_summary_str(test_summary_medal)}, "
                f"PTB-XL: {macro_summary_str(test_summary_ptb)}"
            )

        # ---- Per-class records ----
        shared_label_list = list(SHARED_LABELS)
        record_per_class_metrics(per_class_records, "val_medalcare", val_medal, shared_label_list, epoch)
        record_per_class_metrics(per_class_records, "val_ptbxl", val_ptb, shared_label_list, epoch)
        if test_medal:
            record_per_class_metrics(per_class_records, "test_medalcare", test_medal, shared_label_list, epoch)
        if test_ptb:
            record_per_class_metrics(per_class_records, "test_ptbxl", test_ptb, shared_label_list, epoch)

        # ---- Metrics log ----
        metrics_entry: Dict[str, object] = {
            "epoch": epoch,
            "train_loss": {"medalcare": avg_loss_medal, "ptbxl": avg_loss_ptb},
            "primary_metric": {"name": "avg_domain_f1", "value": val_score},
            "val": {"medalcare": val_summary_medal, "ptbxl": val_summary_ptb},
        }
        if avg_mmd is not None:
            metrics_entry["train_mmd"] = avg_mmd
        if test_summary_medal and test_summary_ptb:
            metrics_entry["test"] = {"medalcare": test_summary_medal, "ptbxl": test_summary_ptb}
        metrics_log.append(metrics_entry)

        # ---- Checkpoint ----
        is_best = bool(val_score > best_val_score)
        if is_best:
            best_val_score = val_score
            print("==> New best validation score; saving checkpoint.")
            checkpoint_state = {
                "epoch": epoch,
                "step": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_score": float(val_score),
                "val_auroc": float(val_score),
            }
            save_checkpoint(checkpoint_state, str(checkpoints_dir))
            best_filename = args.best_checkpoint_name or "linear_best.pt"
            torch.save(checkpoint_state, checkpoints_dir / best_filename)
            best_snapshot = {
                "epoch": epoch,
                "primary_metric": {"name": "avg_domain_f1", "value": val_score},
                "val": {"medalcare": val_summary_medal, "ptbxl": val_summary_ptb},
                "test": {"medalcare": test_summary_medal, "ptbxl": test_summary_ptb},
            }

        current_lr = min(group["lr"] for group in optimizer.param_groups)
        if current_lr < args.early_stop_lr:
            print(
                f"LR {current_lr:g} fell below early-stop threshold "
                f"{args.early_stop_lr:g}; stopping."
            )
            break

    # ---- Save reports ----
    metrics_path = outputs_dir / "metrics.json"
    save_metrics_report(metrics_path, run_id, metrics_to_compute, metrics_log, best_snapshot)
    print(f"Saved metrics report to: {metrics_path}")

    per_class_path = outputs_dir / "per_class_metrics.csv"
    pd.DataFrame(per_class_records).to_csv(per_class_path, index=False)
    print(f"Saved per-class metrics to: {per_class_path}")

    if args.log_tensors and test_medal and test_ptb:
        np.savez_compressed(
            tensors_dir / "test_medalcare.npz",
            y_true=test_medal["y_true"], y_pred=test_medal["y_pred"],
        )
        np.savez_compressed(
            tensors_dir / "test_ptbxl.npz",
            y_true=test_ptb["y_true"], y_pred=test_ptb["y_pred"],
        )

    print("[Pre-Phase-B] Dual-head shared-label training complete.")


def main() -> None:
    args = parse_args()
    args.freeze_encoder = args.freeze_encoder or getattr(args, "linear_probe", False)
    set_deterministic(args.seed)
    metrics_to_compute = validate_metrics(args.metrics.split(","))
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Exp 7: shared-head branch (completely separate code path) ---
    if args.shared_head:
        _run_shared_head(args)
        return

    # --- Pre-Phase-B redo: dual-head joint training on the shared 3-class space ---
    if args.dual_head_shared_labels:
        _run_dual_head_shared_labels(args)
        return

    joint_mode = args.joint_datasets is not None
    if joint_mode and not args.multi_head:
        raise ValueError("--joint-datasets requires --multi-head")

    if args.physics_only and not joint_mode:
        raise ValueError("--physics-only requires --joint-datasets for MedalCare batches.")
    if args.physics_only and args.lambda_phys != 1.0:
        print("[INFO] --physics-only ignores lambda_phys; using physics loss only.")

    outputs_base = (
        PTBXL_OUTPUT_ROOT / args.head_type if args.dataset == "ptbxl" else DEFAULT_OUTPUTS
    )
    outputs_dir = (outputs_base / run_id).resolve()
    checkpoints_dir = outputs_dir / "checkpoints"
    tensors_dir = outputs_dir / "tensors"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    if args.log_tensors:
        tensors_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build loaders and label space
    if joint_mode:
        # Joint: both datasets needed
        (
            medal_train,
            medal_val,
            medal_test,
            medal_train_eval,
            medal_labels,
            medal_info,
        ) = build_medalcare_loaders(
            args, warn_mmd=False, include_theta=True, force_no_domain=True
        )
        (
            ptb_train,
            ptb_val,
            ptb_test,
            ptb_train_eval,
            ptb_labels,
            ptb_info,
        ) = build_ptbxl_loaders(args, warn_mmd=False)
        label_columns = medal_labels + ptb_labels  # for metrics bookkeeping if needed
        n_medal = len(medal_labels)
        n_ptb = len(ptb_labels)
        # For loss weighting, use MedalCare weights when task=medalcare, PTB weights when task=ptbxl
        # We will store both sets for later selection
        medal_pos_counts = medal_info["pos_counts"]
        medal_neg_counts = medal_info["train_sample_count"] - medal_pos_counts
        ptb_pos_counts = ptb_info["pos_counts"]
        ptb_neg_counts = ptb_info["train_sample_count"] - ptb_pos_counts
        loaders = {
            "medalcare": (medal_train, medal_val, medal_test, medal_train_eval),
            "ptbxl": (ptb_train, ptb_val, ptb_test, ptb_train_eval),
        }
        dataset_meta = {"medalcare": medal_info["dataset_meta"], "ptbxl": ptb_info["dataset_meta"]}
        pos_counts = {"medalcare": medal_pos_counts, "ptbxl": ptb_pos_counts}
        neg_counts = {"medalcare": medal_neg_counts, "ptbxl": ptb_neg_counts}
        checkpoint_path = args.checkpoint
    else:
        if args.dataset == "medalcare":
            (
                train_loader,
                val_loader,
                test_loader,
                train_eval_loader,
                label_columns,
                info,
            ) = build_medalcare_loaders(args, include_theta=True)
            n_classes = len(label_columns)
            pos_counts = info["pos_counts"]
            neg_counts = info["train_sample_count"] - pos_counts
            dataset_meta = info["dataset_meta"]
            checkpoint_path = args.checkpoint
        else:
            (
                train_loader,
                val_loader,
                test_loader,
                train_eval_loader,
                label_columns,
                info,
            ) = build_ptbxl_loaders(args)
            n_classes = len(label_columns)
            pos_counts = info["pos_counts"]
            neg_counts = info["train_sample_count"] - pos_counts
            dataset_meta = info["dataset_meta"]
            checkpoint_path = args.checkpoint

    # Initialize placeholders for optional loaders
    if joint_mode:
        train_eval_loader = None

    # Build model
    if joint_mode or args.multi_head:
        n_medal_classes = n_medal if joint_mode else (n_classes if args.dataset == "medalcare" else 8)
        n_ptb_classes = n_ptb if joint_mode else (n_classes if args.dataset == "ptbxl" else 5)
        n_theta = len(json.loads(Path(args.theta_config).read_text(encoding="utf-8")).get("theta", []))
        model = ft_multihead_ECGFounder(
            device=device,
            pth=str(checkpoint_path),
            n_medal_classes=n_medal_classes,
            n_ptb_classes=n_ptb_classes,
            n_theta=n_theta,
            physics_hidden=args.physics_hidden,
            physics_dropout=args.physics_dropout,
            linear_prob=args.freeze_encoder,
            use_adapter=not args.no_adapter,
        )
        if hasattr(model, "feature_dim"):
            print(f"[INFO] Shared encoder feature dim (z): {model.feature_dim}")
        # heads are internal; freezing handled in builder for encoder
        head_params = [p for n, p in model.named_parameters() if "head_" in n and p.requires_grad]
        encoder_params = [
            p for n, p in model.named_parameters() if "head_" not in n and p.requires_grad
        ]
        if args.physics_only:
            for name, param in model.named_parameters():
                if name.startswith("head_physics"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            head_params = [p for n, p in model.named_parameters() if "head_physics" in n and p.requires_grad]
            encoder_params = []
        if not head_params and not encoder_params:
            raise ValueError("No trainable parameters found. Check --freeze-encoder/--multi-head settings.")
        param_groups = []
        if head_params:
            param_groups.append({"params": head_params, "lr": args.lr_head})
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": args.lr_encoder})
        # For single-dataset + multi-head, keep loss weights for that dataset
        if not joint_mode:
            pos_weight_arr = np.where(
                pos_counts > 0, neg_counts / np.clip(pos_counts, 1e-6, None), 1.0
            )
    else:
        model = ft_12lead_ECGFounder(
            device=device,
            pth=str(checkpoint_path),
            n_classes=n_classes,
            linear_prob=False,
            use_adapter=args.use_adapter,
        )
        in_features = model.dense.in_features
        if args.head_type == "linear":
            head_module = nn.Linear(in_features, n_classes)
        else:
            hidden = max(in_features // 2, 256)
            head_module = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_classes),
            )
        model.dense = head_module.to(device)

        if args.freeze_encoder:
            if args.use_adapter:
                freeze_backbone_except_adapters(model)
            else:
                for name, param in model.named_parameters():
                    if not name.startswith("dense"):
                        param.requires_grad = False

        head_params = [param for param in model.dense.parameters() if param.requires_grad]
        encoder_params = [
            param
            for name, param in model.named_parameters()
            if not name.startswith("dense") and param.requires_grad
        ]

        if not head_params and not encoder_params:
            raise ValueError("No trainable parameters found. Check --freeze-encoder setting.")

        param_groups = []
        if head_params:
            param_groups.append({"params": head_params, "lr": args.lr_head})
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": args.lr_encoder})

        pos_weight_arr = np.where(pos_counts > 0, neg_counts / np.clip(pos_counts, 1e-6, None), 1.0)

    # pos_weight for BCE (single dataset; joint handled later in losses)
    pos_weight_tensor = (
        torch.tensor(pos_weight_arr, dtype=torch.float32, device=device)
        if not joint_mode
        else None
    )

    if joint_mode:
        pos_weight_medal_arr = np.where(
            pos_counts["medalcare"] > 0,
            neg_counts["medalcare"] / np.clip(pos_counts["medalcare"], 1e-6, None),
            1.0,
        )
        pos_weight_ptb_arr = np.where(
            pos_counts["ptbxl"] > 0,
            neg_counts["ptbxl"] / np.clip(pos_counts["ptbxl"], 1e-6, None),
            1.0,
        )
        pos_weight_medal = torch.tensor(pos_weight_medal_arr, dtype=torch.float32, device=device)
        pos_weight_ptb = torch.tensor(pos_weight_ptb_arr, dtype=torch.float32, device=device)
        criterion_medal = nn.BCEWithLogitsLoss(pos_weight=pos_weight_medal)
        criterion_ptb = nn.BCEWithLogitsLoss(pos_weight=pos_weight_ptb)
        criterion = None
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor) if pos_weight_tensor is not None else None
        criterion_medal = None
        criterion_ptb = None
    optimizer = optim.Adam(param_groups, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, mode="max")

    metrics_log: List[Dict[str, object]] = []
    per_class_records: List[Dict[str, object]] = []
    best_snapshot: Optional[Dict[str, object]] = None
    best_val_score = float("-inf")
    # Initialize for scope safety
    val_package = None
    test_package = None

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        if joint_mode:
            model.train()
            medal_train, medal_val, _, _ = loaders["medalcare"]
            ptb_train, ptb_val, _, _ = loaders["ptbxl"]
            steps = min(len(medal_train), len(ptb_train))
            running_loss_medal = 0.0
            running_loss_ptb = 0.0
            samples_medal = 0
            samples_ptb = 0

            running_mmd = 0.0
            mmd_batches = 0
            train_phys_sum = 0.0
            train_phys_count = 0.0
            for (batch_medal, batch_ptb) in zip(medal_train, ptb_train):
                medal_inputs, medal_targets, theta_targets, theta_mask = batch_medal[:4]
                ptb_inputs, ptb_targets = batch_ptb[:2]
                medal_inputs = medal_inputs.to(device, non_blocking=True)
                ptb_inputs = ptb_inputs.to(device, non_blocking=True)
                medal_targets = medal_targets.to(device, non_blocking=True)
                ptb_targets = ptb_targets.to(device, non_blocking=True)
                theta_targets = theta_targets.to(device, non_blocking=True)
                theta_mask = theta_mask.to(device, non_blocking=True)

                if args.label_smoothing > 0.0:
                    medal_targets_s = medal_targets * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
                    ptb_targets_s = ptb_targets * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
                else:
                    medal_targets_s = medal_targets
                    ptb_targets_s = ptb_targets

                optimizer.zero_grad()
                if args.physics_only:
                    _, feat_medal = model(medal_inputs, task="medalcare", return_features=True)
                    if model.head_physics is None:
                        raise ValueError("Physics head is not initialized.")
                    theta_pred = model.head_physics(feat_medal)
                    loss_medal = None
                    loss_ptb = None
                else:
                    logits_medal, feat_medal = model(
                        medal_inputs, task="medalcare", return_features=True
                    )
                    logits_ptb, feat_ptb = model(
                        ptb_inputs, task="ptbxl", return_features=True
                    )
                    if model.head_physics is None:
                        raise ValueError("Physics head is not initialized.")
                    theta_pred = model.head_physics(feat_medal)
                    loss_medal = criterion_medal(logits_medal, medal_targets_s)
                    loss_ptb = criterion_ptb(logits_ptb, ptb_targets_s)
                if args.physics_loss == "huber":
                    phys_raw = torch.nn.functional.smooth_l1_loss(
                        theta_pred, theta_targets, reduction="none"
                    )
                else:
                    phys_raw = torch.nn.functional.mse_loss(
                        theta_pred, theta_targets, reduction="none"
                    )
                valid_count = float(theta_mask.sum().detach().cpu())
                phys_loss = (phys_raw * theta_mask).sum() / torch.clamp(theta_mask.sum(), min=1.0)
                if args.physics_only:
                    loss = phys_loss
                else:
                    loss = loss_medal + loss_ptb + args.lambda_phys * phys_loss
                if (
                    not args.physics_only
                    and args.lambda_mmd > 0.0
                    and feat_medal.size(0) > 1
                    and feat_ptb.size(0) > 1
                ):
                    mmd_value = mmd_rbf(feat_medal, feat_ptb)
                    loss = loss + args.lambda_mmd * mmd_value
                    running_mmd += mmd_value.item()
                    mmd_batches += 1
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                b_medal = medal_inputs.size(0)
                b_ptb = ptb_inputs.size(0)
                if loss_medal is not None:
                    running_loss_medal += loss_medal.item() * b_medal
                    samples_medal += b_medal
                if loss_ptb is not None:
                    running_loss_ptb += loss_ptb.item() * b_ptb
                    samples_ptb += b_ptb
                train_phys_sum += phys_loss.item() * valid_count
                train_phys_count += valid_count

            train_loss = {
                "medalcare": running_loss_medal / max(samples_medal, 1),
                "ptbxl": running_loss_ptb / max(samples_ptb, 1),
            }
            train_mmd = running_mmd / mmd_batches if mmd_batches > 0 else None
            train_phys = train_phys_sum / max(train_phys_count, 1.0)
            if not args.physics_only:
                print(
                    f"Training loss -> MedalCare: {train_loss['medalcare']:.6f}, "
                    f"PTB-XL: {train_loss['ptbxl']:.6f}"
                )
                if train_mmd is not None:
                    print(f"Average MMD: {train_mmd:.6f}")
            print(f"Physics loss: {train_phys:.6f}")
        else:
            train_loss, train_mmd = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                args.label_smoothing,
                args.grad_clip,
                args.lambda_mmd,
                task=args.dataset if (args.multi_head and not joint_mode) else None,
            )
            print(f"Training loss: {train_loss:.6f}")
            if train_mmd is not None:
                print(f"Average MMD: {train_mmd:.6f}")

        if joint_mode:
            val_medal = evaluate_model(
                model,
                medal_val,
                device,
                metrics_to_compute,
                task="medalcare",
                threshold=args.threshold,
            )
            val_ptb = evaluate_model(
                model,
                ptb_val,
                device,
                metrics_to_compute,
                task="ptbxl",
                threshold=args.threshold,
            )
            if val_medal is None or val_ptb is None:
                raise RuntimeError("Validation split is empty – cannot continue training.")
            val_summary_medal = summarise_macro(val_medal, metrics_to_compute)
            val_summary_ptb = summarise_macro(val_ptb, metrics_to_compute)
            print(
                "Validation macro metrics -> "
                f"MedalCare: {macro_summary_str(val_summary_medal)}, "
                f"PTB-XL: {macro_summary_str(val_summary_ptb)}"
            )
            # Use MedalCare primary metric to drive scheduler (arbitrary but consistent)
            primary_metric_name, primary_metric_value = select_primary_metric(val_medal, metrics_to_compute)
            val_score = primary_metric_value if primary_metric_value is not None else float("-inf")
        else:
            val_package = evaluate_model(
                model,
                val_loader,
                device,
                metrics_to_compute,
                task=args.dataset if (args.multi_head and not joint_mode) else None,
                threshold=args.threshold,
            )
            if val_package is None:
                raise RuntimeError("Validation split is empty – cannot continue training.")
            val_summary = summarise_macro(val_package, metrics_to_compute)
            print(f"Validation macro metrics: {macro_summary_str(val_summary)}")
            primary_metric_name, primary_metric_value = select_primary_metric(val_package, metrics_to_compute)
            val_score = primary_metric_value if primary_metric_value is not None else float("-inf")

        scheduler.step(val_score if np.isfinite(val_score) else 0.0)

        if joint_mode:
            medal_test_loader = loaders["medalcare"][2]
            ptb_test_loader = loaders["ptbxl"][2]
            test_medal = evaluate_model(
                model,
                medal_test_loader,
                device,
                metrics_to_compute,
                task="medalcare",
                threshold=args.threshold,
            )
            test_ptb = evaluate_model(
                model,
                ptb_test_loader,
                device,
                metrics_to_compute,
                task="ptbxl",
                threshold=args.threshold,
            )
            if test_medal is not None and test_ptb is not None:
                test_summary_medal = summarise_macro(test_medal, metrics_to_compute)
                test_summary_ptb = summarise_macro(test_ptb, metrics_to_compute)
                print(
                    "Test macro metrics -> "
                    f"MedalCare: {macro_summary_str(test_summary_medal)}, "
                    f"PTB-XL: {macro_summary_str(test_summary_ptb)}"
                )
                test_summary = {"medalcare": test_summary_medal, "ptbxl": test_summary_ptb}
            else:
                test_summary = None
                print("Test macro metrics: N/A (empty split)")
            val_package = None  # not used in joint mode
            val_summary = None  # placeholder to avoid undefined use below
        else:
            test_package = evaluate_model(
                model,
                test_loader,
                device,
                metrics_to_compute,
                task=args.dataset if (args.multi_head and not joint_mode) else None,
                threshold=args.threshold,
            )
            if test_package is not None:
                test_summary = summarise_macro(test_package, metrics_to_compute)
                print(f"Test macro metrics: {macro_summary_str(test_summary)}")
            else:
                test_summary = None
                print("Test macro metrics: N/A (empty split)")

        train_package = None
        if train_eval_loader is not None:
            train_package = evaluate_model(
                model,
                train_eval_loader,
                device,
                metrics_to_compute,
                task=args.dataset if (args.multi_head and not joint_mode) else None,
                threshold=args.threshold,
            )
            if train_package is not None:
                train_summary = summarise_macro(train_package, metrics_to_compute)
                print(f"Train macro metrics: {macro_summary_str(train_summary)}")

        metrics_entry: Dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "primary_metric": {"name": primary_metric_name, "value": primary_metric_value},
        }
        if joint_mode:
            metrics_entry["train_phys"] = train_phys
        if train_mmd is not None:
            metrics_entry["train_mmd"] = train_mmd
        if joint_mode:
            metrics_entry["val"] = {"medalcare": val_summary_medal, "ptbxl": val_summary_ptb}
        else:
            metrics_entry["val"] = {"macro": val_summary}
        if test_summary is not None:
            if joint_mode:
                metrics_entry["test"] = test_summary
            else:
                metrics_entry["test"] = {"macro": test_summary}
        if train_package is not None:
            metrics_entry["train"] = {"macro": summarise_macro(train_package, metrics_to_compute)}
        metrics_log.append(metrics_entry)

        if joint_mode:
            record_per_class_metrics(per_class_records, "val_medalcare", val_medal, medal_labels, epoch)
            record_per_class_metrics(per_class_records, "val_ptbxl", val_ptb, ptb_labels, epoch)
            if test_medal is not None:
                record_per_class_metrics(per_class_records, "test_medalcare", test_medal, medal_labels, epoch)
            if test_ptb is not None:
                record_per_class_metrics(per_class_records, "test_ptbxl", test_ptb, ptb_labels, epoch)
        else:
            record_per_class_metrics(per_class_records, "val", val_package, label_columns, epoch)
            if test_package is not None:
                record_per_class_metrics(per_class_records, "test", test_package, label_columns, epoch)
            if train_package is not None:
                record_per_class_metrics(per_class_records, "train", train_package, label_columns, epoch)

        is_best = bool(val_score > best_val_score)
        if is_best:
            best_val_score = val_score
            print("==> New best validation score; saving checkpoint.")
            checkpoint_state = {
                "epoch": epoch,
                "step": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_score": float(val_score) if np.isfinite(val_score) else 0.0,
                "val_auroc": float(val_score) if np.isfinite(val_score) else 0.0,
            }
            save_checkpoint(checkpoint_state, str(checkpoints_dir))
            best_filename = args.best_checkpoint_name or f"{args.head_type}_best.pt"
            torch.save(checkpoint_state, checkpoints_dir / best_filename)
            best_snapshot = {
                "epoch": epoch,
                "primary_metric": {
                    "name": primary_metric_name,
                    "value": val_score if np.isfinite(val_score) else None,
                },
                "val": {"medalcare": val_summary_medal, "ptbxl": val_summary_ptb}
                if joint_mode
                else val_summary,
                "test": test_summary,
            }

        current_lr = min(group["lr"] for group in optimizer.param_groups)
        if current_lr < args.early_stop_lr:
            print(
                f"Learning rate {current_lr:g} fell below early-stop threshold "
                f"{args.early_stop_lr:g}; stopping training."
            )
            break

    physics_metrics = None
    if args.physics_metrics and joint_mode:
        print("Computing physics metrics on MedalCare test split.")
        medal_test_loader = loaders["medalcare"][2]
        physics_package = evaluate_physics(model, medal_test_loader, device)
        if physics_package is not None:
            eval_stats_path = args.theta_eval_stats or args.theta_stats
            theta_stats = None
            if eval_stats_path is not None and eval_stats_path.exists():
                theta_stats = json.loads(eval_stats_path.read_text(encoding="utf-8"))
            theta_names = load_theta_names(args.theta_config)
            theta_core_names = None
            core_config = args.theta_core_config
            default_core = REPO_ROOT / "config" / "theta_core.json"
            if core_config is None and default_core.exists():
                core_config = default_core
            if core_config is not None and core_config.exists():
                theta_core_names = load_theta_names(core_config)
            physics_metrics = compute_physics_metrics(
                physics_package["y_true"],
                physics_package["y_pred"],
                physics_package["mask"],
                theta_stats=theta_stats,
                theta_names=theta_names,
                theta_core_names=theta_core_names,
            )
            if physics_metrics is not None and eval_stats_path is not None:
                physics_metrics.setdefault("meta", {})["theta_stats_path"] = str(
                    eval_stats_path
                )
            metrics_path = outputs_dir / "physics_metrics.json"
            with metrics_path.open("w", encoding="utf-8") as fp:
                json.dump(physics_metrics, fp, indent=2)
            print(f"Saved physics metrics to: {metrics_path}")
            if args.physics_plots:
                metric_values = None
                if isinstance(physics_metrics.get("spearman"), list):
                    metric_values = physics_metrics["spearman"]
                elif isinstance(physics_metrics.get("r2"), list):
                    metric_values = physics_metrics["r2"]
                plot_indices = (
                    select_top_theta_indices(metric_values, top_k=6)
                    if metric_values
                    else None
                )
                save_physics_plots(
                    outputs_dir,
                    physics_package["y_true"],
                    physics_package["y_pred"],
                    physics_package["mask"],
                    indices=plot_indices,
                    theta_stats=theta_stats,
                    theta_names=physics_metrics.get("theta_names"),
                )

    if physics_metrics is not None:
        best_snapshot = best_snapshot or {}
        best_snapshot["physics"] = physics_metrics

    metrics_path = outputs_dir / "metrics.json"
    save_metrics_report(metrics_path, run_id, metrics_to_compute, metrics_log, best_snapshot)
    print(f"Saved metrics report to: {metrics_path}")

    per_class_path = outputs_dir / "per_class_metrics.csv"
    pd.DataFrame(per_class_records).to_csv(per_class_path, index=False)
    print(f"Saved per-class metrics to: {per_class_path}")

    if args.log_tensors:
        print("Saving raw prediction tensors.")
        if not joint_mode:
            if train_package is not None:
                save_numpy_payload(tensors_dir, "train", train_package)
            if val_package is not None:
                save_numpy_payload(tensors_dir, "val", val_package)
            if test_package is not None:
                save_numpy_payload(tensors_dir, "test", test_package)


if __name__ == "__main__":
    main()

