"""Export latent features, predictions, and labels from trained ECGFounder checkpoints.

Supports:
  - Single-head Net1D (with or without adapters)  e.g. Exp 1
  - Multi-head MultiHeadECGFounder (with or without adapters)  e.g. Exp 4, 5, 6

Usage examples:

  # Exp 1: PTB-XL baseline (single-head, adapter)
  python scripts/export_latents.py ^
    --checkpoint outputs/ptbxl_baselines/linear/ptbxl_baseline/checkpoints/linear_best.pt ^
    --model-type single --use-adapter ^
    --dataset ptbxl --split test ^
    --outdir outputs/latents/exp1_ptbxl

  # Exp 5: joint adapter, PTB-XL test set
  python scripts/export_latents.py ^
    --checkpoint outputs/joint_adapter_cls/checkpoints/linear_best.pt ^
    --model-type multi --use-adapter ^
    --dataset ptbxl --split test ^
    --outdir outputs/latents/exp5_ptbxl

  # Exp 5: joint adapter, MedalCare test set
  python scripts/export_latents.py ^
    --checkpoint outputs/joint_adapter_cls/checkpoints/linear_best.pt ^
    --model-type multi --use-adapter ^
    --dataset medalcare --split test ^
    --outdir outputs/latents/exp5_medalcare
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from net1d import Net1D, MultiHeadECGFounder
from scripts.datasets import get_dataset

DEFAULT_PTBXL_ROOT = (
    REPO_ROOT
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
DEFAULT_MEDAL_MANIFEST = (
    REPO_ROOT / "data" / "medalcare_filtered_manifest_dataset_split.csv"
)

NET1D_ARCH = dict(
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
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export latent features from trained ECGFounder checkpoints.",
    )
    p.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to fine-tuned checkpoint (.pt/.pth).",
    )
    p.add_argument(
        "--model-type", choices=["single", "multi"], required=True,
        help="'single' = Net1D, 'multi' = MultiHeadECGFounder.",
    )
    p.add_argument(
        "--use-adapter", action="store_true",
        help="Enable adapter layers in the model.",
    )
    p.add_argument(
        "--dataset", choices=["ptbxl", "medalcare"], required=True,
        help="Dataset to export latents for.",
    )
    p.add_argument(
        "--ptbxl-root", type=Path, default=DEFAULT_PTBXL_ROOT,
        help="PTB-XL root directory.",
    )
    p.add_argument(
        "--manifest", type=Path, default=DEFAULT_MEDAL_MANIFEST,
        help="MedalCare manifest CSV (must contain 'split' column).",
    )
    p.add_argument(
        "--split", type=str, default="test",
        choices=["train", "val", "test", "all"],
        help="Data split to export (default: test).",
    )
    p.add_argument("--outdir", type=Path, required=True,
                    help="Output directory for the NPZ file.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _load_state_dict(checkpoint_path: Path, device: torch.device) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")


def _infer_n_classes(sd: dict, model_type: str, dataset: str) -> int:
    if model_type == "single":
        if "dense.weight" in sd:
            return sd["dense.weight"].shape[0]
        raise ValueError("Cannot infer n_classes: 'dense.weight' not in checkpoint.")
    key = "head_ptb.weight" if dataset == "ptbxl" else "head_medal.weight"
    if key in sd:
        return sd[key].shape[0]
    raise ValueError(f"Cannot infer n_classes: '{key}' not in checkpoint.")


def build_single_head(
    sd: dict, device: torch.device, n_classes: int, use_adapter: bool,
) -> Net1D:
    model = Net1D(**NET1D_ARCH, n_classes=n_classes, use_adapter=use_adapter)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [WARN] Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    model.return_features = True
    model.to(device).eval()
    return model


def build_multi_head(
    sd: dict, device: torch.device, use_adapter: bool,
) -> MultiHeadECGFounder:
    n_medal = sd["head_medal.weight"].shape[0]
    n_ptb = sd["head_ptb.weight"].shape[0]
    model = MultiHeadECGFounder(
        **NET1D_ARCH,
        n_medal_classes=n_medal,
        n_ptb_classes=n_ptb,
        use_adapter=use_adapter,
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [WARN] Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def make_loader(
    dataset_name: str,
    split: str,
    ptbxl_root: Path,
    manifest_path: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[DataLoader, int]:
    if dataset_name == "ptbxl":
        ds = get_dataset("ptbxl", root=ptbxl_root, split=split,
                         return_metadata=False)
    else:
        df = pd.read_csv(manifest_path)
        if "split" in df.columns and split != "all":
            df = df[df["split"].str.lower() == split.lower()].copy()
        ds = get_dataset("medalcare", ecg_path="", labels_df=df)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return loader, len(ds)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_type: str,
    task: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_parts, p_parts, y_parts = [], [], []

    for batch in tqdm(loader, desc="Exporting", leave=False):
        signals = batch[0].to(device, non_blocking=True)
        labels = batch[1]

        if model_type == "single":
            logits, features = model(signals)
        else:
            logits, features = model(signals, task=task, return_features=True)

        z_parts.append(features.cpu().numpy())
        p_parts.append(torch.sigmoid(logits).cpu().numpy())
        y_parts.append(labels.numpy() if isinstance(labels, torch.Tensor) else labels)

    Z = np.concatenate(z_parts, axis=0)
    P = np.concatenate(p_parts, axis=0)
    Y = np.concatenate(y_parts, axis=0)
    return Z, P, Y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model type: {args.model_type} | adapter: {args.use_adapter}")
    print(f"Dataset: {args.dataset} | split: {args.split}")

    # Load state dict once
    sd = _load_state_dict(args.checkpoint, device)
    n_classes = _infer_n_classes(sd, args.model_type, args.dataset)
    print(f"Inferred n_classes = {n_classes}")

    # Build model
    if args.model_type == "single":
        model = build_single_head(sd, device, n_classes, args.use_adapter)
    else:
        model = build_multi_head(sd, device, args.use_adapter)

    # Build loader
    task = args.dataset  # medalcare → head_medal, ptbxl → head_ptb
    loader, n_samples = make_loader(
        args.dataset, args.split, args.ptbxl_root, args.manifest,
        args.batch_size, args.num_workers, device,
    )
    print(f"Samples: {n_samples}")

    # Extract
    Z, P, Y = extract(model, loader, device, args.model_type, task)
    print(f"Shapes — Z: {Z.shape}, P: {P.shape}, Y: {Y.shape}")

    if Z.shape[0] != n_samples:
        print(f"[WARN] Expected {n_samples} rows, got {Z.shape[0]}")

    # Assemble NPZ payload
    payload = {"Z": Z, "P": P, "Y": Y}

    # For MedalCare, include theta columns from manifest if present
    if args.dataset == "medalcare":
        df = pd.read_csv(args.manifest)
        if "split" in df.columns and args.split != "all":
            df = df[df["split"].str.lower() == args.split.lower()].copy()
            df = df.reset_index(drop=True)
        theta_cols = sorted(
            c for c in df.columns if c.lower().startswith("theta")
        )
        if theta_cols:
            payload["Theta"] = df[theta_cols].to_numpy(dtype=np.float32)
            print(f"Included {len(theta_cols)} theta columns from manifest")

    # Save
    out_dir = args.outdir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "latents.npz"
    np.savez_compressed(npz_path, **payload)
    print(f"Saved {npz_path}  ({npz_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
