"""Smoke test for Tier-1 bottleneck training.

Loads exp7_baseline checkpoint, swaps in BottleneckHead(K=64), confirms:
  * Parameter freezing is correct (head trainable, backbone frozen).
  * A forward pass produces logits of the right shape and caches z_k.
  * A backward pass updates ONLY the head weights.
  * Export-side model load works from a freshly-saved state_dict.

Does NOT run training -- just confirms the wiring before the multi-hour sweep.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from scripts.finetune_bottleneck import BottleneckHead, build_model_with_bottleneck
from scripts.export_bottleneck_latents import build_bottleneck_model, _load_state_dict


def main() -> None:
    ckpt = REPO_ROOT / "outputs" / "exp7_baseline" / "checkpoints" / "linear_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device: {device}")

    # 1. Build model via the training-side helper.
    args = argparse.Namespace(checkpoint=ckpt, bottleneck_dim=64)
    model = build_model_with_bottleneck(args, device)
    assert isinstance(model.dense, BottleneckHead), "head was not replaced"
    assert model.dense.k == 64

    # 2. Confirm freezing pattern.
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    head_train = sum(p.numel() for p in model.dense.parameters() if p.requires_grad)
    head_total = sum(p.numel() for p in model.dense.parameters())
    assert n_train == head_train, f"trainable params ({n_train}) != head params ({head_train}) -- freezing broken"
    print(f"[smoke] trainable: {n_train:,} (= head only: {head_train:,}), total: {n_total:,}")
    print(f"[smoke] expected K=64 head: 1024*64+64 + 64*3+3 = "
          f"{1024*64+64 + 64*3+3}, actual: {head_total}")
    assert head_total == 1024 * 64 + 64 + 64 * 3 + 3, "head param count off"

    # 3. Forward + backward.
    x = torch.randn(4, 12, 5000, device=device)
    target = torch.zeros(4, 3, device=device); target[:, 0] = 1.0  # NORM
    logits = model(x)
    print(f"[smoke] logits shape: {tuple(logits.shape)} (expected (4, 3))")
    assert logits.shape == (4, 3)
    assert model.dense.last_z_k is not None
    print(f"[smoke] last_z_k shape: {tuple(model.dense.last_z_k.shape)} (expected (4, 64))")
    assert tuple(model.dense.last_z_k.shape) == (4, 64)

    # Snapshot a backbone weight + a head weight pre-backward.
    bb_name = next(n for n, p in model.named_parameters() if not n.startswith("dense"))
    bb_param = dict(model.named_parameters())[bb_name]
    head_w = model.dense.proj.weight
    bb_before = bb_param.detach().clone()
    head_before = head_w.detach().clone()

    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    optimizer.step()
    optimizer.zero_grad()

    assert torch.equal(bb_param, bb_before), f"backbone weight {bb_name} CHANGED -- freezing broken"
    head_delta = (head_w - head_before).abs().max().item()
    assert head_delta > 0, "head weight did NOT change -- training is broken"
    print(f"[smoke] backbone weight {bb_name!r}: UNCHANGED (frozen ok)")
    print(f"[smoke] head proj.weight: changed by max(|delta|)={head_delta:.3e} (trainable ok)")

    # 4. Roundtrip the bottleneck-head state via the export path.
    # Take a FRESH forward AFTER the optimizer step so we're comparing the
    # model state we are about to save with the rebuilt-from-disk version.
    # Training-side model returns logits only; export-side returns
    # (logits, features) because it sets return_features=True.
    model.eval()
    with torch.no_grad():
        out_train = model(x)
    logits_after_step = out_train[0] if isinstance(out_train, tuple) else out_train
    sd = model.state_dict()
    meta = {"bottleneck_dim": 64, "epoch": 0, "primary_metric": 0.0}
    rt_path = REPO_ROOT / "outputs" / "_smoke_bottleneck.pt"
    torch.save({"state_dict": sd, **meta}, rt_path)
    sd2, meta2 = _load_state_dict(rt_path, device)
    assert meta2.get("bottleneck_dim") == 64
    rt_model = build_bottleneck_model(sd2, k=64, device=device)
    with torch.no_grad():
        out_rt = rt_model(x)
    rt_logits = out_rt[0] if isinstance(out_rt, tuple) else out_rt
    diff = (rt_logits - logits_after_step).abs().max().item()
    print(f"[smoke] reconstructed-vs-original (post-step) max|logit_diff| = {diff:.3e}")
    assert diff < 1e-4, f"reconstructed model diverged: max diff={diff}"
    print("[smoke] reconstructed model matches original")
    # Also confirm last_z_k is captured on the export side.
    assert rt_model.dense.last_z_k is not None
    assert tuple(rt_model.dense.last_z_k.shape) == (4, 64)
    print("[smoke] reconstructed model caches last_z_k correctly")

    rt_path.unlink()
    print("\n[OK] All smoke checks PASSED")


if __name__ == "__main__":
    main()
