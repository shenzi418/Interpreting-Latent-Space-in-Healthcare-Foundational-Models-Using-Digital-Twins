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
)
from metrics import AVAILABLE_METRICS, compute_multilabel_metrics  # pylint: disable=wrong-import-position
from losses.mmd import mmd_rbf  # pylint: disable=wrong-import-position
from util import save_checkpoint  # pylint: disable=wrong-import-position

DEFAULT_MANIFEST = REPO_ROOT / "MedalRaw" / "medalcare_filtered_manifest.csv"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoint" / "12_lead_ECGFounder.pth"
DEFAULT_OUTPUTS = REPO_ROOT / "outputs"
DEFAULT_PTBXL_ROOT = (
    REPO_ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
PTBXL_OUTPUT_ROOT = REPO_ROOT / "outputs" / "ptbxl_baselines"
DATASET_CHOICES = ("medalcare", "ptbxl")
JOINT_CHOICES = ("medalcare+ptbxl",)


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
        default="ap,brier,roc_auc",
        help=f"Comma-separated list of metrics to compute. Supported: {', '.join(AVAILABLE_METRICS)}",
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
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[DataLoader], List[str], dict]:
    manifest = ensure_manifest(args.manifest)
    label_columns = [col for col in manifest.columns if col.startswith("label_")]
    n_classes = len(label_columns)
    train_df, val_df, test_df = prepare_splits(manifest, args.ignore_splits)
    print(f"Loaded manifest with {len(manifest)} records.")
    print(f"Train/val/test sizes: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    include_domain = args.lambda_mmd > 0.0 and args.domain_column in train_df.columns
    if include_domain:
        unique_domains = [
            val for val in pd.Series(train_df[args.domain_column]).dropna().unique()
        ]
        domain_map = {value: idx for idx, value in enumerate(sorted(unique_domains))}
        print(f"Detected domains for MMD: {domain_map}")
    else:
        if args.lambda_mmd > 0.0:
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
    )
    val_loader = make_dataloader(
        val_df, args.batch_size, args.num_workers, shuffle=False
    )
    test_loader = make_dataloader(
        test_df, args.batch_size, args.num_workers, shuffle=False
    )
    train_eval_loader = (
        make_dataloader(train_df, args.batch_size, args.num_workers, shuffle=False)
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
    if args.lambda_mmd > 0.0:
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
) -> Optional[Dict[str, object]]:
    if loader.dataset is None or len(loader.dataset) == 0:
        return None

    model.eval()
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs, task=task) if task else model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)
            targets.append(labels.cpu().numpy())

    if not preds:
        return None

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets, axis=0)
    metrics = compute_multilabel_metrics(y_true, y_pred, metrics_to_compute)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "metrics": metrics,
    }


def select_primary_metric(metrics: Dict[str, object], candidates: List[str]) -> Tuple[str, Optional[float]]:
    macro = metrics["metrics"]["macro"]
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


def main() -> None:
    args = parse_args()
    args.freeze_encoder = args.freeze_encoder or getattr(args, "linear_probe", False)
    set_deterministic(args.seed)
    metrics_to_compute = validate_metrics(args.metrics.split(","))
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.dataset == "ptbxl" and not args.freeze_encoder:
        print("[INFO] Enabling --freeze-encoder for PTB-XL Stage 1 baseline.")
        args.freeze_encoder = True

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

    joint_mode = args.joint_datasets is not None
    if joint_mode and not args.multi_head:
        raise ValueError("--joint-datasets requires --multi-head")

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
        ) = build_medalcare_loaders(args)
        (
            ptb_train,
            ptb_val,
            ptb_test,
            ptb_train_eval,
            ptb_labels,
            ptb_info,
        ) = build_ptbxl_loaders(args)
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
        checkpoint_path = enforce_ptbxl_checkpoint(args.checkpoint)
    else:
        if args.dataset == "medalcare":
            (
                train_loader,
                val_loader,
                test_loader,
                train_eval_loader,
                label_columns,
                info,
            ) = build_medalcare_loaders(args)
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
            checkpoint_path = enforce_ptbxl_checkpoint(args.checkpoint)

    # Initialize placeholders for optional loaders
    if joint_mode:
        train_eval_loader = None

    # Build model
    if joint_mode or args.multi_head:
        n_medal_classes = 8
        n_ptb_classes = 5
        model = ft_multihead_ECGFounder(
            device=device,
            pth=str(checkpoint_path),
            n_medal_classes=n_medal_classes,
            n_ptb_classes=n_ptb_classes,
            linear_prob=args.freeze_encoder,
        )
        # heads are internal; freezing handled in builder for encoder
        head_params = [p for n, p in model.named_parameters() if "head_" in n and p.requires_grad]
        encoder_params = [
            p for n, p in model.named_parameters() if "head_" not in n and p.requires_grad
        ]
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

            for (batch_medal, batch_ptb) in zip(medal_train, ptb_train):
                medal_inputs, medal_targets = batch_medal[:2]
                ptb_inputs, ptb_targets = batch_ptb[:2]
                medal_inputs = medal_inputs.to(device, non_blocking=True)
                ptb_inputs = ptb_inputs.to(device, non_blocking=True)
                medal_targets = medal_targets.to(device, non_blocking=True)
                ptb_targets = ptb_targets.to(device, non_blocking=True)

                if args.label_smoothing > 0.0:
                    medal_targets_s = medal_targets * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
                    ptb_targets_s = ptb_targets * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
                else:
                    medal_targets_s = medal_targets
                    ptb_targets_s = ptb_targets

                optimizer.zero_grad()
                logits_medal = model(medal_inputs, task="medalcare")
                logits_ptb = model(ptb_inputs, task="ptbxl")
                loss_medal = criterion_medal(logits_medal, medal_targets_s)
                loss_ptb = criterion_ptb(logits_ptb, ptb_targets_s)
                loss = loss_medal + loss_ptb
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                b_medal = medal_inputs.size(0)
                b_ptb = ptb_inputs.size(0)
                running_loss_medal += loss_medal.item() * b_medal
                running_loss_ptb += loss_ptb.item() * b_ptb
                samples_medal += b_medal
                samples_ptb += b_ptb

            train_loss = {
                "medalcare": running_loss_medal / max(samples_medal, 1),
                "ptbxl": running_loss_ptb / max(samples_ptb, 1),
            }
            train_mmd = None
            print(
                f"Training loss -> MedalCare: {train_loss['medalcare']:.6f}, "
                f"PTB-XL: {train_loss['ptbxl']:.6f}"
            )
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
            )
            val_ptb = evaluate_model(
                model,
                ptb_val,
                device,
                metrics_to_compute,
                task="ptbxl",
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
            )
            if val_package is None:
                raise RuntimeError("Validation split is empty – cannot continue training.")
            val_summary = summarise_macro(val_package, metrics_to_compute)
            print(f"Validation macro metrics: {macro_summary_str(val_summary)}")
            primary_metric_name, primary_metric_value = select_primary_metric(val_package, metrics_to_compute)
            val_score = primary_metric_value if primary_metric_value is not None else float("-inf")

        scheduler.step(val_score if np.isfinite(val_score) else 0.0)

        if joint_mode:
            # Test can be added later; focus on val for joint mode
            test_summary = None
            val_package = None  # not used in joint mode
            val_summary = None  # placeholder to avoid undefined use below
        else:
            test_package = evaluate_model(
                model,
                test_loader,
                device,
                metrics_to_compute,
                task=args.dataset if (args.multi_head and not joint_mode) else None,
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
            )
            if train_package is not None:
                train_summary = summarise_macro(train_package, metrics_to_compute)
                print(f"Train macro metrics: {macro_summary_str(train_summary)}")

        metrics_entry: Dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "primary_metric": {"name": primary_metric_name, "value": primary_metric_value},
        }
        if train_mmd is not None:
            metrics_entry["train_mmd"] = train_mmd
        if joint_mode:
            metrics_entry["val"] = {"medalcare": val_summary_medal, "ptbxl": val_summary_ptb}
        else:
            metrics_entry["val"] = {"macro": val_summary}
        if test_summary is not None:
            metrics_entry["test"] = {"macro": test_summary}
        if train_package is not None:
            metrics_entry["train"] = {"macro": summarise_macro(train_package, metrics_to_compute)}
        metrics_log.append(metrics_entry)

        if not joint_mode:
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
                "test": test_summary if not joint_mode else None,
            }

        current_lr = min(group["lr"] for group in optimizer.param_groups)
        if current_lr < args.early_stop_lr:
            print(
                f"Learning rate {current_lr:g} fell below early-stop threshold "
                f"{args.early_stop_lr:g}; stopping training."
            )
            break

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

