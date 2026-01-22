#!/usr/bin/env python3
"""Audit θ stats and define a θ_core subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.datasets import LVEF_12lead_cls_Dataset


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATS = REPO_ROOT / "outputs" / "theta_stats.json"
DEFAULT_CONFIG = REPO_ROOT / "config" / "theta.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "theta_audit.csv"
DEFAULT_CORE = REPO_ROOT / "config" / "theta_core.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit θ stats and create θ_core contract."
    )
    parser.add_argument(
        "--theta-stats",
        type=Path,
        default=DEFAULT_STATS,
        help=f"θ stats JSON (default: {DEFAULT_STATS})",
    )
    parser.add_argument(
        "--theta-config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"θ contract JSON (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output-audit",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output audit CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--output-core",
        type=Path,
        default=DEFAULT_CORE,
        help=f"Output θ_core JSON (default: {DEFAULT_CORE})",
    )
    parser.add_argument(
        "--eps-std",
        type=float,
        default=1e-5,
        help="Constant-like threshold for std (default: 1e-5)",
    )
    parser.add_argument(
        "--cov-thresh",
        type=float,
        default=0.90,
        help="Coverage threshold for θ_core (default: 0.90)",
    )
    parser.add_argument(
        "--skip-unique-check",
        action="store_true",
        help="Skip quasi-discrete checks (n_unique/integer-like).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "data" / "medalcare_filtered_manifest_dataset_split.csv",
        help="MedalCare manifest with split column.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for quasi-discrete checks.",
    )
    return parser.parse_args()


def build_theta_core(
    theta_cfg: dict,
    stats: dict,
    eps_std: float,
    cov_thresh: float,
    audit_override: pd.DataFrame | None = None,
) -> Dict[str, object]:
    names = [t["name"] for t in theta_cfg["theta"]]
    stds = np.array(stats["std"], dtype=np.float64)
    counts = np.array(stats["count"], dtype=np.float64)
    transforms = stats.get("transform", ["none"] * len(names))

    max_count = counts.max() if counts.size else 0
    coverage = counts / max_count if max_count > 0 else counts

    audit = (
        audit_override
        if audit_override is not None
        else pd.DataFrame(
            {
                "name": names,
                "std": stds,
                "count": counts,
                "coverage": coverage,
                "transform": transforms,
                "is_flag_like": False,
                "n_unique": None,
                "integer_like_pct": None,
            }
        )
    )
    audit["is_constant_like"] = audit["std"] <= eps_std
    audit["is_low_coverage"] = audit["coverage"] < cov_thresh

    theta_core_mask = ~(audit["is_constant_like"] | audit["is_low_coverage"] | audit["is_flag_like"])
    audit["active"] = theta_core_mask

    theta_core = {
        "granularity": theta_cfg.get("granularity", "per_run"),
        "theta": [t for t, keep in zip(theta_cfg["theta"], theta_core_mask) if keep],
        "excluded": {
            "constant_like": audit.loc[audit["is_constant_like"], "name"].tolist(),
            "low_coverage": audit.loc[audit["is_low_coverage"], "name"].tolist(),
            "flag_like": audit.loc[audit["is_flag_like"], "name"].tolist(),
        },
        "criteria": {
            "eps_std": eps_std,
            "cov_thresh": cov_thresh,
        },
    }
    return audit, theta_core


def add_quasi_discrete_flags(
    audit: pd.DataFrame,
    theta_config: Path,
    manifest: Path,
    batch_size: int,
) -> pd.DataFrame:
    df = pd.read_csv(manifest)
    if "split" in df.columns:
        df = df[df["split"] == "train"].reset_index(drop=True)
    if df.empty:
        raise ValueError("No training rows available for quasi-discrete checks.")

    dataset = LVEF_12lead_cls_Dataset(
        ecg_path="",
        labels_df=df,
        include_theta=True,
        theta_config=theta_config,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    values = []
    masks = []
    for batch in loader:
        values.append(batch[2].numpy())
        masks.append(batch[3].numpy())
    theta = np.concatenate(values, axis=0)
    mask = np.concatenate(masks, axis=0)

    n_unique = []
    integer_like = []
    for idx in range(theta.shape[1]):
        m = mask[:, idx] > 0.5
        if not np.any(m):
            n_unique.append(0)
            integer_like.append(0.0)
            continue
        vals = theta[m, idx]
        unique_vals = np.unique(vals)
        n_unique.append(int(unique_vals.size))
        integer_like.append(float(np.mean(np.abs(vals - np.round(vals)) < 1e-6)))

    audit = audit.copy()
    audit["n_unique"] = n_unique
    audit["integer_like_pct"] = integer_like
    audit["is_flag_like"] = (audit["n_unique"] <= 5) | (audit["integer_like_pct"] >= 0.95)
    return audit


def main() -> None:
    args = parse_args()
    stats = json.loads(args.theta_stats.read_text(encoding="utf-8"))
    theta_cfg = json.loads(args.theta_config.read_text(encoding="utf-8"))

    audit, theta_core = build_theta_core(
        theta_cfg, stats, args.eps_std, args.cov_thresh
    )
    if not args.skip_unique_check:
        audit = add_quasi_discrete_flags(
            audit, args.theta_config, args.manifest, args.batch_size
        )
        theta_core = build_theta_core(
            theta_cfg, stats, args.eps_std, args.cov_thresh, audit_override=audit
        )[1]

    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_audit, index=False)
    print(f"Wrote θ audit to {args.output_audit}")

    args.output_core.parent.mkdir(parents=True, exist_ok=True)
    args.output_core.write_text(json.dumps(theta_core, indent=2), encoding="utf-8")
    print(f"Wrote θ_core to {args.output_core}")


if __name__ == "__main__":
    main()

