"""Verify the anatomical interpretation of LCX_ant vs LCX_post.

For each (coronary, lcx_subtype) bucket in the parsed MedalCare theta_mi NPZs,
report per-bucket phi range, circular mean, and how those map to standard
cardiology angular conventions (phi=0 anterior, phi=pi posterior).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def deg(x: float) -> float:
    return float(np.degrees(x))


def circ_mean(phi: np.ndarray) -> float:
    return float(np.arctan2(np.sin(phi).mean(), np.cos(phi).mean()))


def main() -> None:
    for split in SPLITS:
        p = REPO_ROOT / "data" / f"theta_mi_{split}.npz"
        d = dict(np.load(p, allow_pickle=True))
        coronary = d["coronary"]
        lcx_subtype = d["lcx_subtype"]
        transmural = d["transmural"]
        phi = d["phi"]
        z = d["z"]

        print(f"\n[{split}] n_MI={phi.size}")
        # Build bucket key per row
        bucket = np.array([
            f"{c}_{int(t*10)/10:.1f}"
            + (f"_{s}" if (c == "LCX" and s) else "")
            for c, t, s in zip(coronary.tolist(), transmural.tolist(), lcx_subtype.tolist())
        ], dtype=object)
        uniq = sorted(set(bucket.tolist()))
        print(f"  unique buckets: {len(uniq)} -> {uniq}")

        for b in uniq:
            mask = (bucket == b)
            sub = phi[mask]
            if sub.size == 0:
                continue
            cm = circ_mean(sub)
            print(
                f"    {b:18s} n={mask.sum():4d}  "
                f"phi range=[{sub.min():+.3f}, {sub.max():+.3f}] rad "
                f"({deg(sub.min()):+6.1f} deg, {deg(sub.max()):+6.1f} deg)  "
                f"circ_mean={cm:+.3f} rad ({deg(cm):+6.1f} deg)  "
                f"z range=[{z[mask].min():.3f}, {z[mask].max():.3f}]"
            )


if __name__ == "__main__":
    main()
