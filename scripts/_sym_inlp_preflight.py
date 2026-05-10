"""Pre-flight verification for the symmetric INLP refit.

Confirms that the new PTB-XL train latent files exist with sane shapes
and finite values, and that downstream INLP can find them under the
expected `outputs/latents/{stem}_ptbxl_train/` paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LATENT_DIR = REPO / "outputs" / "latents"

CONFIG_LATENT_STEMS = {
    "exp7_baseline": "exp7",
    "exp7_ccmmd": "exp7_ccmmd",
}

# We expect every existing split + the new ptbxl_train.
SPLITS = ("medalcare_train", "medalcare", "ptbxl_train", "ptbxl")


def check_one(stem: str, split: str) -> dict:
    path = LATENT_DIR / f"{stem}_{split}" / "latents.npz"
    if not path.exists():
        return {"path": str(path), "exists": False}
    with np.load(path, allow_pickle=True) as data:
        keys = list(data.keys())
        info = {k: {"shape": list(data[k].shape), "dtype": str(data[k].dtype)} for k in keys}
        Z = data["Z"]
        n_nan = int(np.isnan(Z).sum())
        n_inf = int(np.isinf(Z).sum())
        z_mean = float(Z.mean())
        z_std = float(Z.std())
    return {
        "path": str(path.relative_to(REPO)),
        "exists": True,
        "keys": info,
        "Z_nan": n_nan,
        "Z_inf": n_inf,
        "Z_mean": z_mean,
        "Z_std": z_std,
    }


def main() -> int:
    print("Symmetric INLP pre-flight check")
    print("=" * 72)
    all_ok = True
    for cfg, stem in CONFIG_LATENT_STEMS.items():
        print(f"\nconfig: {cfg}  (stem: {stem})")
        for split in SPLITS:
            res = check_one(stem, split)
            tag = "OK " if res["exists"] else "MISS"
            line = f"  [{tag}] {res['path']}"
            if res["exists"]:
                z = res["keys"]["Z"]
                line += (
                    f"  Z={z['shape']} {z['dtype']}  "
                    f"NaN={res['Z_nan']} Inf={res['Z_inf']}  "
                    f"mean={res['Z_mean']:+.3f} std={res['Z_std']:.3f}"
                )
                if res["Z_nan"] or res["Z_inf"]:
                    all_ok = False
                    line += "   <-- BAD: non-finite"
            else:
                all_ok = False
            print(line)
    print()
    if all_ok:
        print("Pre-flight: PASS")
        return 0
    print("Pre-flight: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
