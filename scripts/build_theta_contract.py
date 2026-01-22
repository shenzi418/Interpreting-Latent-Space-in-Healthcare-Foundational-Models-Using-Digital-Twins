#!/usr/bin/env python3
"""Build θ contract from MedalCare-XL parameter files."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARAM_ROOT = REPO_ROOT / "MedalCare-XL" / "WP2_largeDataset_ParameterFiles"
DEFAULT_OUTPUT = REPO_ROOT / "config" / "theta.json"

NUMERIC_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)([a-zA-Z/]+)?\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enumerate θ keys from MedalCare parameter files."
    )
    parser.add_argument(
        "--param-root",
        type=Path,
        default=DEFAULT_PARAM_ROOT,
        help=f"Root directory for parameter files (default: {DEFAULT_PARAM_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output θ contract JSON (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def classify_value(raw: str) -> Tuple[Optional[float], Optional[str]]:
    text = raw.strip().strip('"').strip("'")
    if not text:
        return None, None
    if text.lower() in {"true", "false"}:
        return None, None
    match = NUMERIC_RE.match(text)
    if not match:
        return None, None
    value = float(match.group(1))
    unit = match.group(2)
    return value, unit


def scan_files(files: List[Path], source: str) -> Tuple[Dict[str, dict], Set[str], Dict[str, int]]:
    stats: Dict[str, dict] = {}
    non_numeric: Set[str] = set()
    counts: Dict[str, int] = {}
    for path in files:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" not in line:
                continue
            key, raw = [part.strip() for part in line.split("=", 1)]
            value, unit = classify_value(raw)
            if value is None:
                non_numeric.add(key)
                continue
            counts[key] = counts.get(key, 0) + 1
            entry = stats.setdefault(
                key,
                {
                    "name": key,
                    "sources": set(),
                    "units": set(),
                },
            )
            entry["sources"].add(source)
            if unit:
                entry["units"].add(unit)
    return stats, non_numeric, counts


def main() -> None:
    args = parse_args()
    if not args.param_root.exists():
        raise FileNotFoundError(f"Parameter root not found: {args.param_root}")

    atrial_files = sorted(args.param_root.rglob("*_AtrialParameters.txt"))
    vent_files = sorted(args.param_root.rglob("*_VentricularParameters.txt"))
    if not atrial_files or not vent_files:
        raise RuntimeError("No parameter files found under the provided root.")

    atrial_stats, atrial_non, atrial_counts = scan_files(atrial_files, "atrial")
    vent_stats, vent_non, vent_counts = scan_files(vent_files, "ventricular")

    merged: Dict[str, dict] = {}
    for key, entry in {**atrial_stats, **vent_stats}.items():
        merged.setdefault(
            key,
            {
                "name": key,
                "sources": set(),
                "units": set(),
            },
        )
    for key, entry in atrial_stats.items():
        merged[key]["sources"].update(entry["sources"])
        merged[key]["units"].update(entry["units"])
    for key, entry in vent_stats.items():
        merged[key]["sources"].update(entry["sources"])
        merged[key]["units"].update(entry["units"])

    theta_list = []
    for key in sorted(merged.keys()):
        units = sorted(merged[key]["units"])
        theta_list.append(
            {
                "name": key,
                "sources": sorted(merged[key]["sources"]),
                "units": units[0] if len(units) == 1 else (units if units else None),
                "continuous": True,
                "bounded": None,
                "transform": "none",
                "counts": {
                    "atrial": int(atrial_counts.get(key, 0)),
                    "ventricular": int(vent_counts.get(key, 0)),
                },
            }
        )

    payload = {
        "granularity": "per_run",
        "theta": theta_list,
        "excluded_keys": sorted(set().union(atrial_non, vent_non)),
        "source_root": str(args.param_root),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote θ contract with {len(theta_list)} parameters to {args.output}")


if __name__ == "__main__":
    main()

