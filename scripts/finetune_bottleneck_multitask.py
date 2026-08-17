"""TIER 2 -- multi-task bottleneck training (classification + biophysical).

Extension of ``scripts/finetune_bottleneck.py``. The same Linear(1024, K) projection
is shared between two heads:

    z_k  = Linear(1024, K)(features)          # canonical K-d latent (pre-GELU)
    h_k  = GELU(z_k)
    cls_logits = Linear(K, 3)(h_k)            # NORM / MI / CD
    bio_logits = Linear(K, 5)(h_k)            # phi_sin, phi_cos, z_std, size_std, trans_logit

Loss
----
    L = lambda_cls * (BCE_medal + BCE_ptb)
      + lambda_bio * L_bio_medalcare_MI_only

    L_bio = mean_over_MI_rows( MSE(sin) + MSE(cos) + MSE(z_std) + MSE(size_std)
                                + BCE(trans_logit, trans_bin) ) / 5

Bio targets are computed from ``data/theta_mi_<split>.npz``:
    sin = sin(phi),  cos = cos(phi)
    z_std    = (z    - 0.5832) / 0.2423         # MedalCare-train MI stats
    size_std = (size - 124.98) / 28.87
    trans_bin = (rho_eps_max > 0.5).astype(float)

Bio loss is computed ONLY on MedalCare rows for which the wfdb path resolves to an
MI parameter file (label_1 == 1 and idx_in_split row exists in theta_mi).

Two canonical runs (per Marta's spec)
-------------------------------------
    Config A (50/50):  --lambda-cls 0.5 --lambda-bio 0.5  --run-id exp7_tier2_K64_A_5050
    Config B (bio):    --lambda-cls 0.0 --lambda-bio 1.0  --run-id exp7_tier2_K64_B_bioonly

For Config B the PTB-XL loader is bypassed (no classification gradient anyway) and the
early-stop metric is val MedalCare bio MSE (negated). For Config A early-stop is the
familiar val avg-domain F1.

Freeze policy
-------------
By default the adapters AND the multi-task head are trainable (backbone frozen). Use
``--no-adapter-trainable`` to freeze adapters too (head-only).

Outputs
-------
    outputs/<run_id>/checkpoints/linear_best.pt
    outputs/<run_id>/metrics.json
    outputs/<run_id>/args.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
from finetune_model import ft_12lead_ECGFounder, freeze_backbone_except_adapters  # noqa: E402
from net1d import ConvAdapter1D  # noqa: E402
from scripts.datasets import LVEF_12lead_cls_Dataset, get_dataset  # noqa: E402
from scripts.finetune_multilabel import (  # noqa: E402
    DEFAULT_OUTPUTS,
    MEDALCARE_KEEP_LABELS,
    MEDALCARE_REMAP,
    N_SHARED,
    PTBXL_REMAP,
    SHARED_LABELS,
    _evaluate_shared_head,
    _filter_ptbxl_dataset,
    ensure_manifest,
    make_ptbxl_loader,
    prepare_splits,
    remap_labels,
    resolve_run_dir,
    select_primary_metric,
    set_deterministic,
    summarise_macro,
    validate_metrics,
)


# ---------------------------------------------------------------------------
# Z-score constants for MedalCare-train MI bio targets (computed from
# data/theta_mi_train.npz; locked here so the loss is reproducible across runs).
# Reproducibility audit -- 2026-05-26: z mean/std = 0.5832/0.2423,
#                                      size mean/std = 124.98/28.87.
# ---------------------------------------------------------------------------
Z_MEAN = 0.5832
Z_STD = 0.2423
SIZE_MEAN = 124.98
SIZE_STD = 28.87
TRANS_THRESH = 0.5

BIO_CHANNEL_NAMES = ("phi_sin", "phi_cos", "z_std", "size_std", "trans_logit")
N_BIO = len(BIO_CHANNEL_NAMES)


# ---------------------------------------------------------------------------
# Multi-task bottleneck head
# ---------------------------------------------------------------------------

class MultiTaskBottleneckHead(nn.Module):
    """Linear(1024, K) -> GELU -> {Linear(K, 3) cls, Linear(K, 5) bio}.

    Forward returns ONLY the classification logits so the existing
    ``model.dense(features)`` chain in ``ft_12lead_ECGFounder`` is unchanged.
    Bio logits are stashed on ``self.last_bio_logits`` and read explicitly by
    the Tier 2 training loop. The canonical K-d latent (pre-GELU) lives on
    ``self.last_z_k`` -- identical to the Tier 1 ``BottleneckHead`` so the
    existing latent-export contract carries over.
    """

    def __init__(self, in_dim: int = 1024, k: int = 64,
                 n_cls: int = 3, n_bio: int = N_BIO) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.k = k
        self.n_cls = n_cls
        self.n_bio = n_bio
        self.proj = nn.Linear(in_dim, k)
        self.act = nn.GELU()
        self.classifier = nn.Linear(k, n_cls)
        self.bio_head = nn.Linear(k, n_bio)
        self.last_z_k: Optional[torch.Tensor] = None
        # last_bio_logits is autograd-attached during train (so the bio
        # loss can flow back through the head + proj + adapters).
        self.last_bio_logits: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_k = self.proj(x)
        # Detached cache for analysis-time export only.
        self.last_z_k = z_k.detach()
        h_k = self.act(z_k)
        bio_logits = self.bio_head(h_k)
        # Keep attached so train loop can backprop through it.
        self.last_bio_logits = bio_logits
        return self.classifier(h_k)


# ---------------------------------------------------------------------------
# Theta lookup utilities
# ---------------------------------------------------------------------------

def _build_theta_lookup(theta_npz_path: Path, split_df_unfiltered: pd.DataFrame
                        ) -> Dict[str, np.ndarray]:
    """Build {original_csv_path -> 5-vec theta target} for a single split.

    ``split_df_unfiltered`` must be the raw ``df[df.split == split].reset_index(drop=True)``
    -- the same ordering used when the theta_mi npz was built.
    """
    payload = np.load(theta_npz_path, allow_pickle=True)
    idx = payload["idx_in_split"].astype(np.int64)
    phi = payload["phi"].astype(np.float32)
    z = payload["z"].astype(np.float32)
    size = payload["size"].astype(np.float32)
    trans = payload["transmural"].astype(np.float32)
    if "original_csv_path" not in split_df_unfiltered.columns:
        raise KeyError("Unfiltered MedalCare split lacks 'original_csv_path' column.")
    lookup: Dict[str, np.ndarray] = {}
    n = idx.size
    for i in range(n):
        row_pos = int(idx[i])
        key = str(split_df_unfiltered.iloc[row_pos]["original_csv_path"])
        vec = np.array([
            np.sin(phi[i]),
            np.cos(phi[i]),
            (z[i] - Z_MEAN) / Z_STD,
            (size[i] - SIZE_MEAN) / SIZE_STD,
            1.0 if trans[i] > TRANS_THRESH else 0.0,
        ], dtype=np.float32)
        lookup[key] = vec
    return lookup


class _ThetaWrappedDataset(Dataset):
    """Wraps an LVEF dataset to also return (theta_5vec, theta_mask) per item.

    Theta lookup keyed by ``original_csv_path`` so the wrapper is robust to any
    upstream row-filtering of the dataframe.
    """

    def __init__(self, base_ds: LVEF_12lead_cls_Dataset,
                 theta_lookup: Dict[str, np.ndarray]) -> None:
        self.base = base_ds
        labels_df = base_ds.labels_df
        if "original_csv_path" not in labels_df.columns:
            raise KeyError("MedalCare manifest must include 'original_csv_path'.")
        n = len(labels_df)
        self.theta = np.zeros((n, N_BIO), dtype=np.float32)
        self.theta_mask = np.zeros((n,), dtype=np.float32)
        n_hit = 0
        for i in range(n):
            key = str(labels_df.iloc[i]["original_csv_path"])
            if key in theta_lookup:
                self.theta[i] = theta_lookup[key]
                self.theta_mask[i] = 1.0
                n_hit += 1
        self._n_hit = n_hit
        self._n_total = n
        print(f"[theta-wrap] {n_hit}/{n} rows matched to theta_mi targets "
              f"({100*n_hit/max(n,1):.1f}%)")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        out = self.base[idx]
        if not isinstance(out, tuple):
            out = (out,)
        theta_t = torch.from_numpy(self.theta[idx])
        mask_t = torch.tensor(self.theta_mask[idx], dtype=torch.float32)
        return (*out, theta_t, mask_t)


def _make_medal_loader(df: pd.DataFrame, theta_lookup, batch_size: int,
                       num_workers: int, shuffle: bool) -> DataLoader:
    base = LVEF_12lead_cls_Dataset(
        ecg_path="", labels_df=df.reset_index(drop=True),
    )
    wrapped = _ThetaWrappedDataset(base, theta_lookup)
    return DataLoader(
        wrapped,
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _filter_medalcare_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate finetune_multilabel._filter_medalcare_manifest behaviour."""
    keep_cols = [f"label_{i}" for i in MEDALCARE_KEEP_LABELS]
    mask = df[keep_cols].sum(axis=1) > 0
    return df.loc[mask].reset_index(drop=True)


# `_filter_ptbxl_dataset` is imported from finetune_multilabel (Stage 5, 2026-08-11).
# A second, DIVERGENT copy used to live here:
#     keep_idx = np.where(targets.sum(axis=1) > 0)[0]
# i.e. "keep any row with any of the FIVE superclasses", not "keep the three that
# map into the shared space". It was also never called -- `build_loaders` below
# passed the raw datasets straight to `make_ptbxl_loader` -- so the tier2 K64 runs
# trained and evaluated on unfiltered PTB-XL, including STTC/HYP-only rows whose
# remapped 3-class target is all-zero. Both facts are recorded in the audit; the
# single shared implementation is now the only one.


# ---------------------------------------------------------------------------
# Bio loss
# ---------------------------------------------------------------------------

def compute_bio_loss(bio_logits: torch.Tensor, theta: torch.Tensor,
                     mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Masked, per-row averaged bio loss.

    Returns the scalar loss and a dict of channel-wise sums (raw, unmasked --
    diagnostic only) for logging.
    """
    # bio_logits: (B, 5), theta: (B, 5), mask: (B,)
    pred_sin = bio_logits[:, 0]
    pred_cos = bio_logits[:, 1]
    pred_z = bio_logits[:, 2]
    pred_sz = bio_logits[:, 3]
    pred_tr = bio_logits[:, 4]
    tg_sin = theta[:, 0]
    tg_cos = theta[:, 1]
    tg_z = theta[:, 2]
    tg_sz = theta[:, 3]
    tg_tr = theta[:, 4]

    mse_sin = F.mse_loss(pred_sin, tg_sin, reduction="none")
    mse_cos = F.mse_loss(pred_cos, tg_cos, reduction="none")
    mse_z = F.mse_loss(pred_z, tg_z, reduction="none")
    mse_sz = F.mse_loss(pred_sz, tg_sz, reduction="none")
    bce_tr = F.binary_cross_entropy_with_logits(pred_tr, tg_tr, reduction="none")

    per_row = (mse_sin + mse_cos + mse_z + mse_sz + bce_tr) / float(N_BIO)
    denom = mask.sum().clamp_min(1.0)
    loss = (per_row * mask).sum() / denom

    # Diagnostics: masked per-channel means.
    with torch.no_grad():
        diag = {
            "n_mi_in_batch": int(mask.sum().item()),
            "mse_sin": float(((mse_sin * mask).sum() / denom).item()),
            "mse_cos": float(((mse_cos * mask).sum() / denom).item()),
            "mse_z": float(((mse_z * mask).sum() / denom).item()),
            "mse_size": float(((mse_sz * mask).sum() / denom).item()),
            "bce_trans": float(((bce_tr * mask).sum() / denom).item()),
        }
    return loss, diag


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def build_model_with_multitask_head(args: argparse.Namespace, device: torch.device
                                    ) -> nn.Module:
    """Load exp7_baseline checkpoint, replace head, configure freezing."""
    model = ft_12lead_ECGFounder(
        device=device,
        pth=str(args.checkpoint),
        n_classes=N_SHARED,
        linear_prob=False,
        use_adapter=True,
        adapter_reduction=16,
        adapter_dropout=0.0,
    )

    feat_dim = int(model.dense.in_features)
    if feat_dim != 1024:
        print(f"[WARN] backbone feat_dim={feat_dim} (expected 1024)")

    model.dense = MultiTaskBottleneckHead(
        in_dim=feat_dim, k=args.bottleneck_dim,
        n_cls=N_SHARED, n_bio=N_BIO,
    ).to(device)

    # Freeze everything, then unfreeze head (always) and adapters (default).
    for p in model.parameters():
        p.requires_grad = False
    for p in model.dense.parameters():
        p.requires_grad = True
    if args.adapter_trainable:
        for m in model.modules():
            if isinstance(m, ConvAdapter1D):
                for p in m.parameters():
                    p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * n_train / max(n_total, 1)
    print(
        f"[multitask] K={args.bottleneck_dim} | "
        f"adapter_trainable={args.adapter_trainable} | "
        f"trainable: {n_train:,} / total {n_total:,} ({pct:.2f}%)"
    )
    return model


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def build_loaders(args: argparse.Namespace) -> dict:
    """Build MedalCare (theta-wrapped) + PTB-XL loaders."""
    # ---- MedalCare ----
    manifest = ensure_manifest(args.manifest)
    train_df, val_df, test_df = prepare_splits(manifest, args.ignore_splits)

    theta_train = _build_theta_lookup(
        REPO_ROOT / "data" / "theta_mi_train.npz", train_df,
    )
    theta_val = _build_theta_lookup(
        REPO_ROOT / "data" / "theta_mi_val.npz", val_df,
    )
    theta_test = _build_theta_lookup(
        REPO_ROOT / "data" / "theta_mi_test.npz", test_df,
    )

    train_df_f = _filter_medalcare_manifest(train_df)
    val_df_f = _filter_medalcare_manifest(val_df)
    test_df_f = _filter_medalcare_manifest(test_df)
    print(f"[multitask] MedalCare filtered: train={len(train_df_f)} "
          f"val={len(val_df_f)} test={len(test_df_f)}")

    medal_train = _make_medal_loader(train_df_f, theta_train,
                                     args.batch_size, args.num_workers, shuffle=True)
    medal_val = _make_medal_loader(val_df_f, theta_val,
                                   args.batch_size, args.num_workers, shuffle=False)
    medal_test = _make_medal_loader(test_df_f, theta_test,
                                    args.batch_size, args.num_workers, shuffle=False)

    # pos_weight (shared label space)
    medal_pos_shared = np.zeros(N_SHARED, dtype=np.float64)
    for src_idx, tgt_idx in MEDALCARE_REMAP.items():
        col = f"label_{src_idx}"
        if col in train_df_f.columns:
            medal_pos_shared[tgt_idx] += train_df_f[col].sum()

    # ---- PTB-XL (skipped for bio-only Config B at the training step but loaded
    # for val/test classification evaluation parity) ----
    dataset_kwargs = dict(
        root=args.ptbxl_root,
        sampling_rate=500,
        signal_duration=10.0,
        use_high_res=True,
        return_metadata=False,
    )
    # Stage 5 fix: this call was missing entirely, so every tier2 run trained on
    # the full 5-superclass PTB-XL. The filter keeps NORM/MI/CD (PTBXL_REMAP's
    # source columns) and drops STTC/HYP-only rows, matching the shared-head path.
    ptb_train_ds = _filter_ptbxl_dataset(get_dataset("ptbxl", split="train", **dataset_kwargs))
    ptb_val_ds = _filter_ptbxl_dataset(get_dataset("ptbxl", split="val", **dataset_kwargs))
    ptb_test_ds = _filter_ptbxl_dataset(get_dataset("ptbxl", split="test", **dataset_kwargs))

    ptb_train = make_ptbxl_loader(ptb_train_ds, args.batch_size, args.num_workers, shuffle=True)
    ptb_val = make_ptbxl_loader(ptb_val_ds, args.batch_size, args.num_workers, shuffle=False)
    ptb_test = make_ptbxl_loader(ptb_test_ds, args.batch_size, args.num_workers, shuffle=False)

    ptb_pos_shared = np.zeros(N_SHARED, dtype=np.float64)
    for src_idx, tgt_idx in PTBXL_REMAP.items():
        ptb_pos_shared[tgt_idx] += ptb_train_ds.targets[:, src_idx].sum()

    combined_pos = medal_pos_shared + ptb_pos_shared
    combined_total = len(train_df_f) + len(ptb_train_ds)
    combined_neg = combined_total - combined_pos
    pos_weight_arr = combined_neg / np.clip(combined_pos, 1e-6, None)
    print(f"[multitask] Combined train n={combined_total}; "
          f"pos_weight={np.round(pos_weight_arr, 3).tolist()}")

    return {
        "medal": {"train": medal_train, "val": medal_val, "test": medal_test},
        "ptb": {"train": ptb_train, "val": ptb_val, "test": ptb_test},
        "pos_weight": pos_weight_arr,
    }


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_one_epoch_multitask(
    model: nn.Module,
    medal_loader: DataLoader,
    ptb_loader: Optional[DataLoader],
    optimizer: optim.Optimizer,
    cls_criterion: nn.Module,
    device: torch.device,
    label_smoothing: float,
    grad_clip: float,
    lambda_cls: float,
    lambda_bio: float,
) -> Dict[str, float]:
    model.train()
    loss_total_sum = 0.0
    loss_cls_sum = 0.0
    loss_bio_sum = 0.0
    n_batches = 0
    n_medal = 0
    bio_diag_accum: Dict[str, float] = {"mse_sin": 0.0, "mse_cos": 0.0,
                                        "mse_z": 0.0, "mse_size": 0.0,
                                        "bce_trans": 0.0}

    bio_only = lambda_cls == 0.0

    if bio_only:
        iterator = enumerate(medal_loader)
        n_steps = len(medal_loader)
    else:
        n_steps = min(len(medal_loader), len(ptb_loader))
        iterator = enumerate(zip(medal_loader, ptb_loader))

    progress = tqdm(iterator, total=n_steps, desc="train", leave=False)
    for step, batch_pack in progress:
        if bio_only:
            batch_medal = batch_pack
        else:
            batch_medal, batch_ptb = batch_pack

        medal_inputs = batch_medal[0].to(device, non_blocking=True)
        medal_labels_8 = batch_medal[1].to(device, non_blocking=True)
        theta = batch_medal[2].to(device, non_blocking=True)
        theta_mask = batch_medal[3].to(device, non_blocking=True)
        medal_labels = remap_labels(medal_labels_8, MEDALCARE_REMAP, N_SHARED, device)
        if label_smoothing > 0.0:
            medal_labels = medal_labels * (1.0 - label_smoothing) + 0.5 * label_smoothing

        optimizer.zero_grad()
        cls_logits_medal = model(medal_inputs)
        bio_logits = model.dense.last_bio_logits

        bio_loss, bio_diag = compute_bio_loss(bio_logits, theta, theta_mask)

        loss_cls = torch.tensor(0.0, device=device)
        if not bio_only:
            ptb_inputs = batch_ptb[0].to(device, non_blocking=True)
            ptb_labels_5 = batch_ptb[1].to(device, non_blocking=True)
            ptb_labels = remap_labels(ptb_labels_5, PTBXL_REMAP, N_SHARED, device)
            if label_smoothing > 0.0:
                ptb_labels = ptb_labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
            cls_logits_ptb = model(ptb_inputs)
            loss_cls = cls_criterion(cls_logits_medal, medal_labels) + \
                       cls_criterion(cls_logits_ptb, ptb_labels)

        loss = lambda_cls * loss_cls + lambda_bio * bio_loss
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                grad_clip,
            )
        optimizer.step()

        n_batches += 1
        loss_total_sum += float(loss.item())
        loss_cls_sum += float(loss_cls.item())
        loss_bio_sum += float(bio_loss.item())
        n_medal += int(medal_inputs.size(0))
        for k in bio_diag_accum:
            bio_diag_accum[k] += bio_diag[k]

    avg_total = loss_total_sum / max(n_batches, 1)
    avg_cls = loss_cls_sum / max(n_batches, 1)
    avg_bio = loss_bio_sum / max(n_batches, 1)
    bio_diag_mean = {k: v / max(n_batches, 1) for k, v in bio_diag_accum.items()}

    return {
        "train_loss_total": avg_total,
        "train_loss_cls": avg_cls,
        "train_loss_bio": avg_bio,
        "n_batches": n_batches,
        "n_medal_samples": n_medal,
        "bio_channels": bio_diag_mean,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _evaluate_bio_split(model: nn.Module, loader: DataLoader,
                        device: torch.device) -> Dict[str, float]:
    """Compute masked bio losses + per-channel R² on a theta-wrapped loader."""
    model.eval()
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="bio-eval", leave=False):
            signal = batch[0].to(device, non_blocking=True)
            theta = batch[2].to(device, non_blocking=True)
            mask = batch[3].to(device, non_blocking=True)
            _ = model(signal)
            bio = model.dense.last_bio_logits
            preds.append(bio.detach().cpu().numpy())
            targets.append(theta.cpu().numpy())
            masks.append(mask.cpu().numpy())
    preds_a = np.concatenate(preds, axis=0)
    tg_a = np.concatenate(targets, axis=0)
    mask_a = np.concatenate(masks, axis=0)
    mi_idx = mask_a > 0.5
    n_mi = int(mi_idx.sum())
    out: Dict[str, float] = {"n_mi_rows": float(n_mi)}
    if n_mi == 0:
        return out

    p = preds_a[mi_idx]
    t = tg_a[mi_idx]

    # Per-channel metrics
    mse_sin = float(np.mean((p[:, 0] - t[:, 0]) ** 2))
    mse_cos = float(np.mean((p[:, 1] - t[:, 1]) ** 2))
    mse_z = float(np.mean((p[:, 2] - t[:, 2]) ** 2))
    mse_sz = float(np.mean((p[:, 3] - t[:, 3]) ** 2))
    # BCE on logit
    trans_p = 1.0 / (1.0 + np.exp(-p[:, 4]))
    eps = 1e-7
    bce_tr = float(np.mean(
        -(t[:, 4] * np.log(trans_p + eps) + (1.0 - t[:, 4]) * np.log(1.0 - trans_p + eps))
    ))
    bio_loss_avg = (mse_sin + mse_cos + mse_z + mse_sz + bce_tr) / float(N_BIO)
    out.update({
        "val_loss_bio": bio_loss_avg,
        "mse_sin": mse_sin, "mse_cos": mse_cos,
        "mse_z": mse_z, "mse_size": mse_sz, "bce_trans": bce_tr,
    })

    # Decoding quality (R²) per channel + transmurality AUC (proxy)
    def _r2(yt, yp):
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2)) + 1e-12
        return 1.0 - ss_res / ss_tot

    out["r2_sin"] = _r2(t[:, 0], p[:, 0])
    out["r2_cos"] = _r2(t[:, 1], p[:, 1])
    out["r2_z"] = _r2(t[:, 2], p[:, 2])
    out["r2_size"] = _r2(t[:, 3], p[:, 3])

    # Circular phi R²
    phi_t = np.arctan2(t[:, 0], t[:, 1])
    phi_p = np.arctan2(p[:, 0], p[:, 1])
    mean_phi = np.arctan2(np.sin(phi_t).mean(), np.cos(phi_t).mean())
    ss_res_c = float((1.0 - np.cos(phi_p - phi_t)).mean())
    ss_tot_c = float((1.0 - np.cos(phi_t - mean_phi)).mean()) + 1e-12
    out["phi_r2_circular"] = 1.0 - ss_res_c / ss_tot_c

    # Transmural AUC
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(t[:, 4])) >= 2:
            out["trans_auc"] = float(roc_auc_score(t[:, 4], trans_p))
    except Exception as exc:  # pylint: disable=broad-except
        out["trans_auc_err"] = str(exc)

    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_multitask(args: argparse.Namespace) -> None:
    set_deterministic(args.seed)
    metrics_to_compute = validate_metrics(args.metrics.split(","))

    run_id, out_dir = resolve_run_dir(
        args.run_id or (
            f"exp7_tier2_K{args.bottleneck_dim}_"
            f"cls{args.lambda_cls}_bio{args.lambda_bio}"
        ),
        args.overwrite,
    )
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "args.json").write_text(
        json.dumps({k: (str(v) if isinstance(v, Path) else v)
                    for k, v in vars(args).items()}, indent=2),
        encoding="utf-8",
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"[run_id] {run_id}  ->  {out_dir.relative_to(REPO_ROOT)}")
    print(f"[Tier 2] K={args.bottleneck_dim} | lambda_cls={args.lambda_cls} "
          f"lambda_bio={args.lambda_bio} | epochs={args.epochs} "
          f"patience={args.patience} | adapter_trainable={args.adapter_trainable}")

    bio_only = args.lambda_cls == 0.0

    # ---- Data ----
    loaders = build_loaders(args)
    medal_train = loaders["medal"]["train"]
    medal_val = loaders["medal"]["val"]
    medal_test = loaders["medal"]["test"]
    ptb_train = loaders["ptb"]["train"]
    ptb_val = loaders["ptb"]["val"]
    ptb_test = loaders["ptb"]["test"]
    pos_weight_arr = loaders["pos_weight"]

    # ---- Model ----
    model = build_model_with_multitask_head(args, device)

    # ---- Optimizer / loss / sched ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters; aborting.")
    pos_weight_tensor = torch.tensor(pos_weight_arr, dtype=torch.float32, device=device)
    cls_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(
        [{"params": trainable_params, "lr": args.learning_rate}],
        weight_decay=args.weight_decay,
    )
    # Scheduler mode depends on primary metric: 'max' for F1, 'min' for bio loss
    sched_mode = "min" if bio_only else "max"
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=sched_mode, patience=3, factor=0.5,
    )

    # ---- Training loop ----
    metrics_log: List[Dict[str, object]] = []
    best_score = float("inf") if bio_only else float("-inf")
    best_epoch = -1
    epochs_since_best = 0
    best_path = ckpt_dir / "linear_best.pt"

    def is_better(new: float, ref: float) -> bool:
        if bio_only:
            return new < ref - 1e-6
        return new > ref + 1e-6

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_stats = train_one_epoch_multitask(
            model, medal_train, ptb_train if not bio_only else None,
            optimizer, cls_criterion, device,
            label_smoothing=args.label_smoothing, grad_clip=args.grad_clip,
            lambda_cls=args.lambda_cls, lambda_bio=args.lambda_bio,
        )

        # Validation: always compute classification F1 (medal+ptb) AND bio metrics.
        val_medal = _evaluate_shared_head(
            model, medal_val, MEDALCARE_REMAP, device, metrics_to_compute,
            args.label_smoothing, args.threshold,
        )
        val_ptb = _evaluate_shared_head(
            model, ptb_val, PTBXL_REMAP, device, metrics_to_compute,
            args.label_smoothing, args.threshold,
        )
        val_summary_medal = summarise_macro(val_medal, metrics_to_compute) if val_medal else {}
        val_summary_ptb = summarise_macro(val_ptb, metrics_to_compute) if val_ptb else {}
        _, f1_medal = select_primary_metric(val_medal, metrics_to_compute) if val_medal else (None, None)
        _, f1_ptb = select_primary_metric(val_ptb, metrics_to_compute) if val_ptb else (None, None)
        f1_medal = float(f1_medal) if f1_medal is not None else float("nan")
        f1_ptb = float(f1_ptb) if f1_ptb is not None else float("nan")
        avg_domain_f1 = (f1_medal + f1_ptb) / 2.0

        val_bio = _evaluate_bio_split(model, medal_val, device)

        primary_value = val_bio.get("val_loss_bio", float("inf")) if bio_only else avg_domain_f1
        scheduler.step(primary_value)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t_epoch
        record = {
            "epoch": epoch,
            "train": train_stats,
            "primary_metric": {
                "name": "val_loss_bio" if bio_only else "avg_domain_f1",
                "value": float(primary_value),
            },
            "val": {
                "medalcare": val_summary_medal or None,
                "ptbxl": val_summary_ptb or None,
                "bio": val_bio,
            },
            "lr": current_lr,
            "elapsed_s": elapsed,
        }
        metrics_log.append(record)
        print(
            f"  train: total={train_stats['train_loss_total']:.4f} "
            f"cls={train_stats['train_loss_cls']:.4f} "
            f"bio={train_stats['train_loss_bio']:.4f} | "
            f"val F1 M={f1_medal:.4f} P={f1_ptb:.4f} avg={avg_domain_f1:.4f} | "
            f"val_bio={val_bio.get('val_loss_bio', float('nan')):.4f} "
            f"phi_r2c={val_bio.get('phi_r2_circular', float('nan')):.3f} "
            f"z_r2={val_bio.get('r2_z', float('nan')):.3f} "
            f"sz_r2={val_bio.get('r2_size', float('nan')):.3f} "
            f"tr_auc={val_bio.get('trans_auc', float('nan')):.3f} | "
            f"lr={current_lr:.1e} | {elapsed:.1f}s"
        )

        if is_better(primary_value, best_score):
            best_score = float(primary_value)
            best_epoch = epoch
            epochs_since_best = 0
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "primary_metric": primary_value,
                "bottleneck_dim": args.bottleneck_dim,
                "lambda_cls": args.lambda_cls,
                "lambda_bio": args.lambda_bio,
                "adapter_trainable": args.adapter_trainable,
            }, best_path)
            print(f"  [save] new best -> {best_path.name} "
                  f"({record['primary_metric']['name']}={primary_value:.4f})")
        else:
            epochs_since_best += 1
            print(f"  no improvement ({epochs_since_best}/{args.patience} epochs)")
            if epochs_since_best >= args.patience:
                print(f"[early-stop] best epoch={best_epoch}, best={best_score:.4f}.")
                break

        if current_lr < args.early_stop_lr:
            print(f"[early-stop] lr {current_lr:.1e} dropped below "
                  f"{args.early_stop_lr:.1e}.")
            break

    # ---- Test eval on best checkpoint ----
    if best_path.exists():
        sd = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(sd["state_dict"])
        print(f"\n[test] loaded {best_path.name} (epoch {sd['epoch']}, "
              f"primary={sd['primary_metric']:.4f})")
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
    test_bio = _evaluate_bio_split(model, medal_test, device)

    summary = {
        "run_id": run_id,
        "config": {
            "checkpoint": str(args.checkpoint),
            "bottleneck_dim": args.bottleneck_dim,
            "lambda_cls": args.lambda_cls,
            "lambda_bio": args.lambda_bio,
            "adapter_trainable": args.adapter_trainable,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
        },
        "bio_channels": list(BIO_CHANNEL_NAMES),
        "z_score_stats": {"z_mean": Z_MEAN, "z_std": Z_STD,
                          "size_mean": SIZE_MEAN, "size_std": SIZE_STD,
                          "trans_thresh": TRANS_THRESH},
        "metrics_requested": metrics_to_compute,
        "evaluations": metrics_log,
        "best": {
            "epoch": best_epoch,
            "primary_metric": {
                "name": "val_loss_bio" if bio_only else "avg_domain_f1",
                "value": best_score,
            },
            "val": (metrics_log[best_epoch - 1]["val"]
                    if 1 <= best_epoch <= len(metrics_log) else None),
            "test": {
                "medalcare": test_summary_medal,
                "ptbxl": test_summary_ptb,
                "bio": test_bio,
            },
        },
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    print(f"\n[done] metrics -> {out_dir / 'metrics.json'}")
    print(f"[done] best ckpt -> {best_path}")
    print(f"[done] best epoch={best_epoch}, best score={best_score:.4f}")
    if test_summary_medal is not None and test_summary_ptb is not None:
        tf1_m = test_summary_medal.get("f1") or float("nan")
        tf1_p = test_summary_ptb.get("f1") or float("nan")
        print(f"[done] test F1: medal={tf1_m:.4f} ptb={tf1_p:.4f} "
              f"avg={(tf1_m + tf1_p) / 2:.4f}")
    print(f"[done] test bio: phi_r2_circ={test_bio.get('phi_r2_circular', float('nan')):.3f}  "
          f"z_r2={test_bio.get('r2_z', float('nan')):.3f}  "
          f"size_r2={test_bio.get('r2_size', float('nan')):.3f}  "
          f"trans_auc={test_bio.get('trans_auc', float('nan')):.3f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tier-2 multi-task bottleneck training.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--bottleneck-dim", type=int, default=64)
    p.add_argument("--lambda-cls", type=float, default=0.5)
    p.add_argument("--lambda-bio", type=float, default=0.5)
    p.add_argument("--adapter-trainable", action="store_true", default=True,
                   help="Unfreeze ConvAdapter1D modules (default True for Tier 2).")
    p.add_argument("--no-adapter-trainable", dest="adapter_trainable",
                   action="store_false",
                   help="Freeze adapters; only train the multi-task head.")

    p.add_argument("--manifest", type=Path,
                   default=REPO_ROOT / "data" / "medalcare_filtered_manifest_dataset_split.csv")
    p.add_argument("--ptbxl-root", type=Path,
                   default=REPO_ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
    p.add_argument("--ignore-splits", action="store_true")

    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--overwrite", action="store_true",
                   help="Allow reusing a --run-id whose metrics.json already exists.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--early-stop-lr", type=float, default=1e-6)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--metrics", type=str,
                   default="accuracy,f1,recall,specificity,precision,brier,roc_auc")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run_multitask(parse_args())
