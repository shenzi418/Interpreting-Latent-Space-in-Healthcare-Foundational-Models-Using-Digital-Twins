"""INLP pre-flight verification.

Checks that all in-scope latent NPZ files exist, are non-empty, contain
finite values, and have the expected shapes. Also confirms B2 cross-
domain ground-truth inputs are present.

Reports:
  - Per-file: path, shape, dtype, NaN/inf count, has Y key
  - Per-config: domain pool sizes (medalcare_train + ptbxl)
  - Final: PASS/FAIL summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG_LATENT_STEMS = {
    "exp7_baseline": "exp7",            # primary
    "exp7_ccmmd":    "exp7_ccmmd",      # primary
    "exp5_3class":   "exp5_3class",     # conditional sensitivity (§4b)
}

PRIMARY = ("exp7_baseline", "exp7_ccmmd")
CONDITIONAL = ("exp5_3class",)
SPLITS = ("medalcare_train", "medalcare", "ptbxl")

# B2-CD ground-truth inputs must exist
B2_INPUTS = (
    REPO_ROOT / "data" / "theta_mi_train.npz",
    REPO_ROOT / "data" / "theta_mi_test.npz",
    REPO_ROOT / "data" / "ptbxl_mi_subclass.csv",
)


def check_npz(path: Path, expected_dim: int = 1024) -> dict:
    """Audit a single latent NPZ. Return summary dict."""
    out: dict = {"path": str(path.relative_to(REPO_ROOT)), "exists": path.exists()}
    if not out["exists"]:
        return out
    data = np.load(path, allow_pickle=True)
    out["keys"] = list(data.keys())
    if "Z" not in data.keys():
        out["error"] = "missing key Z"
        return out
    Z = data["Z"]
    out["Z_shape"] = list(Z.shape)
    out["Z_dtype"] = str(Z.dtype)
    out["n_nan"] = int(np.isnan(Z).sum())
    out["n_inf"] = int(np.isinf(Z).sum())
    out["dim_ok"] = (Z.ndim == 2 and Z.shape[1] == expected_dim)
    out["has_Y"] = "Y" in data.keys()
    if out["has_Y"]:
        out["Y_shape"] = list(data["Y"].shape)
    return out


def main() -> int:
    print("=" * 72)
    print("INLP pre-flight verification")
    print("=" * 72)

    ok = True
    report: dict = {"primary": {}, "conditional": {}, "b2_inputs": {}}

    for tier, configs in (("primary", PRIMARY), ("conditional", CONDITIONAL)):
        print(f"\n--- {tier.upper()} configs: {list(configs)} ---")
        for cfg in configs:
            stem = CONFIG_LATENT_STEMS[cfg]
            cfg_report = {"stem": stem, "splits": {}}
            for split in SPLITS:
                path = REPO_ROOT / "outputs" / "latents" / f"{stem}_{split}" / "latents.npz"
                info = check_npz(path)
                cfg_report["splits"][split] = info
                # print one line per file
                if not info["exists"]:
                    print(f"  [MISS] {info['path']}")
                    if tier == "primary":
                        ok = False
                    continue
                if "error" in info:
                    print(f"  [ERR ] {info['path']}: {info['error']}")
                    ok = False
                    continue
                bad = (info["n_nan"] + info["n_inf"]) > 0 or not info["dim_ok"]
                tag = "[FAIL]" if bad else "[OK  ]"
                if bad and tier == "primary":
                    ok = False
                print(
                    f"  {tag} {info['path']:<60s} "
                    f"Z={tuple(info['Z_shape'])} {info['Z_dtype']} "
                    f"NaN={info['n_nan']} inf={info['n_inf']} "
                    f"hasY={info['has_Y']}"
                )
            report[tier][cfg] = cfg_report

            # combined-pool sanity for this config
            mt = cfg_report["splits"]["medalcare_train"]
            pp = cfg_report["splits"]["ptbxl"]
            if mt.get("dim_ok") and pp.get("dim_ok"):
                n_pool = mt["Z_shape"][0] + pp["Z_shape"][0]
                print(
                    f"        -> INLP pool size for {cfg}: "
                    f"{mt['Z_shape'][0]} (synth) + {pp['Z_shape'][0]} (real) = {n_pool}"
                )

    print("\n--- B2-CD inputs ---")
    for path in B2_INPUTS:
        exists = path.exists()
        report["b2_inputs"][path.name] = exists
        tag = "[OK  ]" if exists else "[MISS]"
        if not exists:
            ok = False
        size = path.stat().st_size if exists else 0
        print(f"  {tag} {path.relative_to(REPO_ROOT)}  ({size} bytes)")

    print("\n" + "=" * 72)
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    print("=" * 72)

    out_path = REPO_ROOT / "outputs" / "inlp" / "preflight.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
