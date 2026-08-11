"""Export the K-d pre-GELU bottleneck activation as the "Z" latent.

Companion to scripts/finetune_bottleneck.py. Loads a trained bottleneck
checkpoint, reconstructs the model with the BottleneckHead, runs forward
on a chosen dataset+split, captures `model.dense.last_z_k` per batch, and
writes to outputs/latents/<outdir>/latents.npz in the standard
{Z, P, Y} layout so downstream Phase A / B2 / B2-CD analysis modules work
unchanged.

Also writes Z_post_gelu as a second array for sensitivity.

Usage
-----
    python scripts/export_bottleneck_latents.py \\
        --checkpoint outputs/exp7_bottleneck_K128/checkpoints/linear_best.pt \\
        --dataset medalcare --split test \\
        --outdir outputs/latents/exp7_bottleneck_K128_medalcare
"""
from __future__ import annotations

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

from net1d import Net1D  # noqa: E402
from scripts.datasets import get_dataset  # noqa: E402
from scripts.finetune_bottleneck import BottleneckHead  # noqa: E402
from scripts.finetune_bottleneck_multitask import MultiTaskBottleneckHead  # noqa: E402

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
        description="Export pre-GELU bottleneck activations as Z (K-d).",
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to bottleneck checkpoint (linear_best.pt).")
    p.add_argument("--dataset", choices=["ptbxl", "medalcare"], required=True)
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "val", "test", "all"])
    p.add_argument("--outdir", type=Path, required=True,
                   help="Output directory; file written as <outdir>/latents.npz")
    p.add_argument("--ptbxl-root", type=Path, default=DEFAULT_PTBXL_ROOT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MEDAL_MANIFEST)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def _load_state_dict(path: Path, device: torch.device) -> Tuple[dict, dict]:
    """Return (state_dict, meta) -- meta includes 'bottleneck_dim'."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
        return sd, meta
    if isinstance(ckpt, dict):
        return ckpt, {}
    raise ValueError(f"Unsupported checkpoint format: {path}")


def _infer_bottleneck_dim(sd: dict, meta: dict) -> int:
    """Prefer the meta-stored K; fall back to inferring from dense.proj.weight."""
    if "bottleneck_dim" in meta and meta["bottleneck_dim"] is not None:
        return int(meta["bottleneck_dim"])
    key = "dense.proj.weight"
    if key in sd:
        return int(sd[key].shape[0])
    raise ValueError(
        "Cannot infer bottleneck_dim: neither checkpoint meta nor "
        "'dense.proj.weight' is present."
    )


def build_bottleneck_model(
    sd: dict, k: int, device: torch.device,
) -> Net1D:
    # Build the same base architecture used at training time (single-head Net1D
    # with adapter). Head type is auto-detected: Tier 2 multitask checkpoints
    # include ``dense.bio_head.weight``; Tier 1 single-head checkpoints don't.
    model = Net1D(**NET1D_ARCH, n_classes=3, use_adapter=True)
    feat_dim = int(model.dense.in_features)
    if "dense.bio_head.weight" in sd:
        n_bio = int(sd["dense.bio_head.weight"].shape[0])
        model.dense = MultiTaskBottleneckHead(
            in_dim=feat_dim, k=k, n_cls=3, n_bio=n_bio,
        )
        print(f"[head] detected MultiTaskBottleneckHead (n_bio={n_bio})")
    else:
        model.dense = BottleneckHead(in_dim=feat_dim, k=k, n_cls=3)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # We expect zero missing (the dense.* keys must match BottleneckHead);
    # unexpected can be non-empty only for stray buffers (none in BottleneckHead).
    if missing:
        # Filter out the `last_z_k` buffer which is None and not in checkpoints.
        missing = [m for m in missing if not m.endswith("last_z_k")]
        if missing:
            print(f"[WARN] missing keys after load ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"[WARN] unexpected keys after load ({len(unexpected)}): {unexpected[:5]}")
    model.return_features = True
    model.to(device).eval()
    return model


def make_loader(
    dataset_name: str, split: str,
    ptbxl_root: Path, manifest: Path,
    batch_size: int, num_workers: int,
    device: torch.device,
) -> Tuple[DataLoader, int]:
    if dataset_name == "ptbxl":
        ds = get_dataset("ptbxl", root=ptbxl_root, split=split, return_metadata=False)
    else:
        df = pd.read_csv(manifest)
        if "split" in df.columns and split != "all":
            df = df[df["split"].str.lower() == split.lower()].copy()
        ds = get_dataset("medalcare", ecg_path="", labels_df=df)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    return loader, len(ds)


@torch.no_grad()
def extract(
    model: torch.nn.Module, loader: DataLoader, device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (Z_pre, Z_post, P, Y).

    Z_pre  = pre-GELU bottleneck activation  (canonical Z)
    Z_post = post-GELU bottleneck activation (for sensitivity)
    P      = sigmoid(logits)                  (3-class predictions)
    Y      = raw labels from the dataset      (8-d for medalcare, 5-d for ptbxl)
    """
    z_pre_parts: list = []
    z_post_parts: list = []
    p_parts: list = []
    y_parts: list = []
    for batch in tqdm(loader, desc="export", leave=False):
        signals = batch[0].to(device, non_blocking=True)
        labels = batch[1]

        # The base Net1D returns (logits, features) when return_features=True;
        # we want logits THROUGH the bottleneck (the head writes last_z_k as
        # a side effect). So we manually do:  feats = backbone(x) -> head(feats)
        # by calling model(x) which routes through dense=BottleneckHead.
        logits, _ = model(signals)
        z_pre = model.dense.last_z_k  # detached pre-GELU activation (K-d)
        z_post = torch.nn.functional.gelu(z_pre)  # recompute post-GELU once

        z_pre_parts.append(z_pre.cpu().numpy())
        z_post_parts.append(z_post.cpu().numpy())
        p_parts.append(torch.sigmoid(logits).cpu().numpy())
        y_parts.append(labels.numpy() if isinstance(labels, torch.Tensor) else labels)

    Z_pre = np.concatenate(z_pre_parts, axis=0)
    Z_post = np.concatenate(z_post_parts, axis=0)
    P = np.concatenate(p_parts, axis=0)
    Y = np.concatenate(y_parts, axis=0)
    return Z_pre, Z_post, P, Y


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    sd, meta = _load_state_dict(args.checkpoint, device)
    k = _infer_bottleneck_dim(sd, meta)
    print(f"Inferred bottleneck K = {k}")

    model = build_bottleneck_model(sd, k, device)

    loader, n_samples = make_loader(
        args.dataset, args.split, args.ptbxl_root, args.manifest,
        args.batch_size, args.num_workers, device,
    )
    print(f"Dataset: {args.dataset} | split: {args.split} | samples: {n_samples}")

    Z_pre, Z_post, P, Y = extract(model, loader, device)
    print(f"Shapes -- Z_pre: {Z_pre.shape}, Z_post: {Z_post.shape}, "
          f"P: {P.shape}, Y: {Y.shape}")
    if Z_pre.shape[0] != n_samples:
        print(f"[WARN] expected {n_samples} rows, got {Z_pre.shape[0]}")

    payload = {
        "Z": Z_pre,           # canonical: pre-GELU K-d bottleneck
        "Z_post_gelu": Z_post,
        "P": P,
        "Y": Y,
    }
    # Include theta columns for MedalCare so B2 mechanism can be re-run cleanly.
    if args.dataset == "medalcare":
        df = pd.read_csv(args.manifest)
        if "split" in df.columns and args.split != "all":
            df = df[df["split"].str.lower() == args.split.lower()].copy()
            df = df.reset_index(drop=True)
        theta_cols = sorted(c for c in df.columns if c.lower().startswith("theta"))
        if theta_cols:
            payload["Theta"] = df[theta_cols].to_numpy(dtype=np.float32)
            print(f"Included {len(theta_cols)} theta columns from manifest")

    out_dir = args.outdir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "latents.npz"
    np.savez_compressed(npz_path, **payload)
    print(f"Saved {npz_path}  ({npz_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
