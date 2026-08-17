"""Smoke test for the Stage 4.2 lead-permutation sweep.

Establishes the two facts the sweep design depends on:
  (1) the PTB-XL test input tensor's shape, hence whether it fits in memory as a
      single cached block -- if it does, 66 permutations cost 66 forward passes
      instead of 66 dataset loads (wfdb read + resample is the real bottleneck);
  (2) the wall-clock of one cached forward pass, which sets the sweep's budget.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
from scripts._diag_leadswap_ptbxl import (  # noqa: E402
    DEFAULT_PTBXL_ROOT, build_model,
)
from scripts.datasets import get_dataset  # noqa: E402


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {dev}")

    t0 = time.time()
    ds = get_dataset("ptbxl", root=DEFAULT_PTBXL_ROOT, split="test",
                     return_metadata=False)
    print(f"dataset built in {time.time()-t0:.1f}s, n={len(ds)}")

    t0 = time.time()
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    xs, ys = [], []
    for b in loader:
        xs.append(b[0])
        ys.append(b[1])
    X = torch.cat(xs, 0)
    Y = torch.cat(ys, 0).numpy()
    load_s = time.time() - t0
    nbytes = X.numel() * X.element_size()
    print(f"cached X {tuple(X.shape)} {X.dtype}  = {nbytes/1e6:.0f} MB  "
          f"in {load_s:.1f}s")
    print(f"Y {Y.shape}")

    model = build_model(dev)
    # one cached forward pass, batched
    t0 = time.time()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            x = X[i:i + 64].to(dev, non_blocking=True)
            _, f = model(x)
            outs.append(f.cpu().numpy())
    Z = np.concatenate(outs, 0)
    fwd_s = time.time() - t0
    print(f"forward pass: {fwd_s:.1f}s -> Z {Z.shape}")
    print()
    print(f"ESTIMATE: 66 transpositions x {fwd_s:.1f}s = "
          f"{66*fwd_s/60:.1f} min (vs {66*(load_s+fwd_s)/60:.1f} min uncached)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
