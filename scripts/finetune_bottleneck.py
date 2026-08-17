"""TRACK 1b' / Tier 1 -- frozen-backbone bottleneck head training.

Loads the existing exp7_baseline checkpoint (frozen backbone + adapters),
replaces the single Linear(1024, 3) head with a BottleneckHead:

    Linear(1024, K) -> GELU -> Linear(K, 3)

Only the BottleneckHead trains. Pure BCE-with-pos-weight loss; NO ccMMD term
(reserved for Tier 2 if/when run).

Mirrors `scripts/finetune_multilabel.py:_run_shared_head` for data, label
remap, evaluation, and early stopping, but with a single LR group (head only).

Direct answer to Marta's question: "is 1024-d necessary?" The K=1024 control
(bottleneck = 1024) is the 2-layer reference; K in {256, 64, 16} are the
compressed candidates. If val avg-domain F1 at K=128 (say) is within 0.02 of
K=1024, 1024-d is overkill by factor 8x.

Usage
-----
    python scripts/finetune_bottleneck.py \\
        --checkpoint outputs/exp7_baseline/checkpoints/linear_best.pt \\
        --bottleneck-dim 128 \\
        --run-id exp7_bottleneck_K128 \\
        --epochs 20 \\
        --patience 5

Outputs
-------
    outputs/<run_id>/checkpoints/linear_best.pt
    outputs/<run_id>/checkpoints/checkpoint_{epoch}_{score}.pth (best-so-far snapshots)
    outputs/<run_id>/metrics.json   (per-epoch + best summary)
    outputs/<run_id>/args.json      (the CLI config used)
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
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune_model import ft_12lead_ECGFounder  # noqa: E402
from scripts.finetune_multilabel import (  # noqa: E402
    DEFAULT_OUTPUTS,
    MEDALCARE_REMAP,
    N_SHARED,
    PTBXL_REMAP,
    SHARED_LABELS,
    _evaluate_shared_head,
    build_shared_head_loaders,
    remap_labels,
    resolve_run_dir,
    select_primary_metric,
    set_deterministic,
    summarise_macro,
    validate_metrics,
)


# ---------------------------------------------------------------------------
# Bottleneck head
# ---------------------------------------------------------------------------

class BottleneckHead(nn.Module):
    """Linear(in_dim, k) -> GELU -> Linear(k, n_cls).

    Caches the *pre-GELU* projection as ``self.last_z_k`` after every forward
    pass. This is the canonical K-d latent we will export for downstream
    Phase A / B2 / B2-CD analyses (pre-GELU = the learned linear subspace of
    the 1024-d Z, like-for-like comparable with PCA-K from Track 1a).
    """

    def __init__(self, in_dim: int = 1024, k: int = 128, n_cls: int = 3) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.k = k
        self.n_cls = n_cls
        self.proj = nn.Linear(in_dim, k)
        self.act = nn.GELU()
        self.classifier = nn.Linear(k, n_cls)
        # Buffer for the most recent pre-GELU activation, detached.
        self.last_z_k: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_k = self.proj(x)
        # Detached cache so the latent-export path can read it without
        # holding the autograd graph.
        self.last_z_k = z_k.detach()
        h_k = self.act(z_k)
        return self.classifier(h_k)


# ---------------------------------------------------------------------------
# Model construction + freezing
# ---------------------------------------------------------------------------

def build_model_with_bottleneck(
    args: argparse.Namespace, device: torch.device,
) -> nn.Module:
    """Load exp7_baseline checkpoint, replace head with BottleneckHead, freeze rest."""
    # ft_12lead_ECGFounder loads the checkpoint, drops the original dense.* keys,
    # builds a fresh Linear(in_features, n_classes). We immediately replace that
    # with the bottleneck head below, so n_classes here is only used to construct
    # the placeholder linear (sizes don't matter; it gets thrown away).
    model = ft_12lead_ECGFounder(
        device=device,
        pth=str(args.checkpoint),
        n_classes=N_SHARED,
        linear_prob=False,
        use_adapter=True,
        adapter_reduction=16,
        adapter_dropout=0.0,
    )

    feat_dim = int(model.dense.in_features)  # 1024 for ECGFounder
    if feat_dim != 1024:
        print(f"[WARN] backbone feat_dim = {feat_dim} (expected 1024)")

    model.dense = BottleneckHead(
        in_dim=feat_dim, k=args.bottleneck_dim, n_cls=N_SHARED,
    ).to(device)

    # Freeze ALL parameters first, then unfreeze only the bottleneck head.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.dense.parameters():
        p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * n_train / max(n_total, 1)
    print(
        f"[bottleneck] K={args.bottleneck_dim} | "
        f"head params: {n_train:,} / total {n_total:,} ({pct:.2f}%)"
    )
    # Sanity: confirm exactly two Linear layers + GELU are trainable.
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"[bottleneck] trainable param names: {trainable_names}")

    return model


# ---------------------------------------------------------------------------
# Training step (single-domain batches alternated, BCE only)
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    medal_loader,
    ptb_loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    label_smoothing: float,
    grad_clip: float,
) -> Dict[str, float]:
    """Alternate-batch training over MedalCare and PTB-XL. BCE only, no MMD."""
    model.train()
    # Mirror _run_shared_head's alternating-batch loop, which pairs one
    # MedalCare batch with one PTB-XL batch per step and sums the two BCE losses.
    n_steps = min(len(medal_loader), len(ptb_loader))
    loss_medal_sum = 0.0
    loss_ptb_sum = 0.0
    n_medal = 0
    n_ptb = 0

    progress = tqdm(
        zip(medal_loader, ptb_loader),
        total=n_steps,
        desc="train",
        leave=False,
    )
    for batch_medal, batch_ptb in progress:
        medal_inputs = batch_medal[0].to(device, non_blocking=True)
        medal_labels_8 = batch_medal[1].to(device, non_blocking=True)
        ptb_inputs = batch_ptb[0].to(device, non_blocking=True)
        ptb_labels_5 = batch_ptb[1].to(device, non_blocking=True)

        medal_labels = remap_labels(medal_labels_8, MEDALCARE_REMAP, N_SHARED, device)
        ptb_labels = remap_labels(ptb_labels_5, PTBXL_REMAP, N_SHARED, device)

        if label_smoothing > 0.0:
            medal_labels = medal_labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
            ptb_labels = ptb_labels * (1.0 - label_smoothing) + 0.5 * label_smoothing

        optimizer.zero_grad()
        logits_medal = model(medal_inputs)
        logits_ptb = model(ptb_inputs)
        loss_medal = criterion(logits_medal, medal_labels)
        loss_ptb = criterion(logits_ptb, ptb_labels)
        loss = loss_medal + loss_ptb
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                grad_clip,
            )
        optimizer.step()

        bm = medal_inputs.size(0)
        bp = ptb_inputs.size(0)
        loss_medal_sum += loss_medal.item() * bm
        loss_ptb_sum += loss_ptb.item() * bp
        n_medal += bm
        n_ptb += bp

    return {
        "train_loss_medal": loss_medal_sum / max(n_medal, 1),
        "train_loss_ptb": loss_ptb_sum / max(n_ptb, 1),
        "n_steps": int(n_steps),
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_bottleneck(args: argparse.Namespace) -> None:
    set_deterministic(args.seed)
    metrics_to_compute = validate_metrics(args.metrics.split(","))

    run_id, out_dir = resolve_run_dir(
        args.run_id or f"exp7_bottleneck_K{args.bottleneck_dim}", args.overwrite
    )
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Persist args first so even a crashed run leaves provenance behind.
    (out_dir / "args.json").write_text(
        json.dumps({k: (str(v) if isinstance(v, Path) else v)
                    for k, v in vars(args).items()}, indent=2),
        encoding="utf-8",
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"[run_id] {run_id}  ->  {out_dir.relative_to(REPO_ROOT)}")
    print(f"[Tier 1] K={args.bottleneck_dim} | epochs={args.epochs} | "
          f"patience={args.patience} | lr={args.learning_rate}")

    # ---- Data ----
    loader_bundle = build_shared_head_loaders(args)
    medal_train = loader_bundle["medal"]["train"]
    medal_val = loader_bundle["medal"]["val"]
    medal_test = loader_bundle["medal"]["test"]
    ptb_train = loader_bundle["ptb"]["train"]
    ptb_val = loader_bundle["ptb"]["val"]
    ptb_test = loader_bundle["ptb"]["test"]
    pos_weight_arr = loader_bundle["pos_weight"]

    # ---- Model (frozen backbone+adapters, trainable BottleneckHead) ----
    model = build_model_with_bottleneck(args, device)

    head_params = [p for p in model.dense.parameters() if p.requires_grad]
    if not head_params:
        raise RuntimeError("No trainable head parameters; aborting.")

    # ---- Loss / Optimizer / Scheduler ----
    pos_weight_tensor = torch.tensor(pos_weight_arr, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(
        [{"params": head_params, "lr": args.learning_rate}],
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5,
    )

    # ---- Training loop with early stop on val avg-domain F1 plateau ----
    metrics_log: List[Dict[str, object]] = []
    best_score = float("-inf")
    best_epoch = -1
    epochs_since_best = 0
    best_path = ckpt_dir / "linear_best.pt"

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_stats = train_one_epoch(
            model, medal_train, ptb_train, optimizer, criterion, device,
            label_smoothing=args.label_smoothing, grad_clip=args.grad_clip,
        )
        # Validation on both domains.
        val_medal = _evaluate_shared_head(
            model, medal_val, MEDALCARE_REMAP, device, metrics_to_compute,
            args.label_smoothing, args.threshold,
        )
        val_ptb = _evaluate_shared_head(
            model, ptb_val, PTBXL_REMAP, device, metrics_to_compute,
            args.label_smoothing, args.threshold,
        )
        # The 'metrics' returned by _evaluate_shared_head is the dict produced
        # by metrics.multilabel.compute_multilabel_metrics; it has keys
        # 'per_class', 'macro', 'support'. Use select_primary_metric to fish
        # out the macro F1 (or first available macro value) just like
        # _run_shared_head does.
        val_summary_medal = summarise_macro(val_medal, metrics_to_compute) if val_medal else {}
        val_summary_ptb = summarise_macro(val_ptb, metrics_to_compute) if val_ptb else {}
        _, f1_medal = select_primary_metric(val_medal, metrics_to_compute) if val_medal else (None, None)
        _, f1_ptb = select_primary_metric(val_ptb, metrics_to_compute) if val_ptb else (None, None)
        f1_medal = float(f1_medal) if f1_medal is not None else float("nan")
        f1_ptb = float(f1_ptb) if f1_ptb is not None else float("nan")
        avg_domain_f1 = (f1_medal + f1_ptb) / 2.0

        scheduler.step(avg_domain_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t_epoch
        record = {
            "epoch": epoch,
            "train": train_stats,
            "primary_metric": {"name": "avg_domain_f1", "value": avg_domain_f1},
            "val": {
                "medalcare": val_summary_medal or None,
                "ptbxl": val_summary_ptb or None,
            },
            "lr": current_lr,
            "elapsed_s": elapsed,
        }
        metrics_log.append(record)
        print(
            f"  train loss M={train_stats['train_loss_medal']:.4f} "
            f"P={train_stats['train_loss_ptb']:.4f} | "
            f"val F1 M={f1_medal:.4f} P={f1_ptb:.4f} "
            f"avg={avg_domain_f1:.4f} | lr={current_lr:.1e} | {elapsed:.1f}s"
        )

        if avg_domain_f1 > best_score + 1e-6:
            best_score = avg_domain_f1
            best_epoch = epoch
            epochs_since_best = 0
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "primary_metric": avg_domain_f1,
                "bottleneck_dim": args.bottleneck_dim,
            }, best_path)
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "primary_metric": avg_domain_f1,
                "bottleneck_dim": args.bottleneck_dim,
            }, ckpt_dir / f"checkpoint_{epoch}_{avg_domain_f1:.4f}.pth")
            print(f"  [save] new best -> {best_path.name} (F1_avg={avg_domain_f1:.4f})")
        else:
            epochs_since_best += 1
            print(f"  no improvement ({epochs_since_best}/{args.patience} epochs)")
            if epochs_since_best >= args.patience:
                print(f"[early-stop] no improvement for {args.patience} epochs; "
                      f"best epoch={best_epoch}, best F1_avg={best_score:.4f}.")
                break
        if current_lr < args.early_stop_lr:
            print(f"[early-stop] lr {current_lr:.1e} dropped below {args.early_stop_lr:.1e}.")
            break

    # ---- Reload best, run TEST evaluation ----
    if best_path.exists():
        sd = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(sd["state_dict"])
        print(f"\n[test] loaded {best_path.name} (epoch {sd['epoch']}, "
              f"avg_domain_f1={sd['primary_metric']:.4f})")
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
    test_block = {
        "medalcare": test_summary_medal,
        "ptbxl": test_summary_ptb,
    }
    # Persist everything.
    summary = {
        "run_id": run_id,
        "config": {
            "checkpoint": str(args.checkpoint),
            "bottleneck_dim": args.bottleneck_dim,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
        },
        "metrics_requested": metrics_to_compute,
        "evaluations": metrics_log,
        "best": {
            "epoch": best_epoch,
            "primary_metric": {"name": "avg_domain_f1", "value": best_score},
            "val": (metrics_log[best_epoch - 1]["val"] if 1 <= best_epoch <= len(metrics_log) else None),
            "test": test_block,
        },
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[done] metrics -> {out_dir / 'metrics.json'}")
    print(f"[done] best ckpt -> {best_path}")
    print(f"[done] best epoch={best_epoch}, avg_domain_f1={best_score:.4f}")
    if test_summary_medal is not None and test_summary_ptb is not None:
        tf1_m = test_summary_medal.get("f1") or float("nan")
        tf1_p = test_summary_ptb.get("f1") or float("nan")
        print(f"[done] test  F1: medal={tf1_m:.4f}, ptb={tf1_p:.4f}, avg={(tf1_m+tf1_p)/2:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tier-1 frozen-backbone bottleneck head training.")
    # Required
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to exp7_baseline checkpoint (linear_best.pt).")
    p.add_argument("--bottleneck-dim", type=int, required=True,
                   help="K -- the bottleneck dimensionality.")
    # Data paths (mirror finetune_multilabel.py defaults)
    p.add_argument("--manifest", type=Path,
                   default=REPO_ROOT / "data" / "medalcare_filtered_manifest_dataset_split.csv")
    p.add_argument("--ptbxl-root", type=Path,
                   default=REPO_ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
    p.add_argument("--ignore-splits", action="store_true")
    # Training
    p.add_argument("--run-id", type=str, default=None,
                   help="Output dir name. Default: exp7_bottleneck_K{K}.")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow reusing a --run-id whose metrics.json already exists.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5,
                   help="Early-stop patience on val avg-domain F1.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=1e-3,
                   help="Head learning rate (matches exp7_baseline lr_head).")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--early-stop-lr", type=float, default=1e-6,
                   help="Stop training if scheduler drops LR below this value.")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--metrics", type=str,
                   default="accuracy,f1,recall,specificity,precision,brier,roc_auc")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run_bottleneck(parse_args())
