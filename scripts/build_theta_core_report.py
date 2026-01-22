#!/usr/bin/env python3
"""Build a θ_core report table from config, stats, and physics metrics."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THETA_CORE = REPO_ROOT / "config" / "theta_core.json"
DEFAULT_STATS = REPO_ROOT / "outputs" / "theta_core_stats.json"
DEFAULT_AUDIT = REPO_ROOT / "outputs" / "theta_audit.csv"
DEFAULT_METRICS = REPO_ROOT / "outputs" / "physics_only_core_v1" / "physics_metrics.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "theta_core_report.csv"


def load_audit(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["name"]: row for row in reader if row.get("name")}


def display_name_from_key(key: str) -> str:
    if key.startswith("stim[") and "." in key:
        prefix, field = key.split(".", 1)
        stim_idx = prefix.replace("stim[", "").replace("]", "")
        return f"Stim {stim_idx} {field_label(field)}"
    if "." in key:
        prefix, field = key.split(".", 1)
        return f"{prefix} {field_label(field)}"
    return key.replace("_", " ").replace(".", " ")


def field_label(field: str) -> str:
    mapping = {
        "phi": "azimuth",
        "rho": "radius",
        "z": "z position",
        "thr": "threshold",
        "time": "time",
    }
    return mapping.get(field, field.replace("_", " "))


def category_from_key(key: str) -> str:
    if key.startswith("APD."):
        return "repolarization"
    if key.startswith(("LA.", "LL.", "RA.", "RL.", "V1.", "V2.", "V3.", "V4.", "V5.", "V6.")):
        return "geometry"
    if key.startswith(("cv.", "cv_t.", "cv_t")):
        return "conduction"
    if key.startswith("stim["):
        return "stimulation"
    return "other"


def component_from_sources(name: str, sources: List[str]) -> str:
    if name.startswith("stim["):
        return "stim"
    if "atrial" in sources and "ventricular" not in sources:
        return "atrial"
    if "ventricular" in sources and "atrial" not in sources:
        return "ventricular"
    if "atrial" in sources and "ventricular" in sources:
        return "other"
    return "other"


def tier_from_r2(r2: Optional[float]) -> str:
    if r2 is None:
        return "undefined"
    if r2 >= 0.5:
        return "strong"
    if r2 >= 0.2:
        return "moderate"
    return "weak"


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None or value == "":
        return None
    return value.strip().lower() in {"1", "true", "yes"}


def main() -> None:
    theta_core = json.loads(DEFAULT_THETA_CORE.read_text(encoding="utf-8"))
    stats = json.loads(DEFAULT_STATS.read_text(encoding="utf-8"))
    metrics = json.loads(DEFAULT_METRICS.read_text(encoding="utf-8"))
    audit = load_audit(DEFAULT_AUDIT)

    theta_entries = theta_core["theta"]
    names = [t["name"] for t in theta_entries]
    counts = stats.get("count", [])
    transforms = stats.get("transform", [])

    max_count = max(counts) if counts else 0

    metrics_map = {}
    for idx, name in enumerate(metrics.get("theta_names", [])):
        metrics_map[name] = {
            "mae_norm": metrics.get("mae_norm", [None])[idx],
            "mae_raw": metrics.get("mae_raw", [None])[idx],
            "r2": metrics.get("r2", [None])[idx],
        }

    output_rows = []
    for idx, entry in enumerate(theta_entries):
        name = entry["name"]
        sources = entry.get("sources", [])
        transform = transforms[idx] if idx < len(transforms) else entry.get("transform", "none")
        count_valid = counts[idx] if idx < len(counts) else None
        coverage = (count_valid / max_count) if (count_valid is not None and max_count) else None

        audit_row = audit.get(name, {})
        is_constant_like = parse_bool(audit_row.get("is_constant_like"))
        is_flag_like = parse_bool(audit_row.get("is_flag_like"))

        metrics_row = metrics_map.get(name, {})
        r2_val = metrics_row.get("r2")

        output_rows.append(
            {
                "idx": idx,
                "key": name,
                "display_name": display_name_from_key(name),
                "component": component_from_sources(name, sources),
                "category": category_from_key(name),
                "notes": "",
                "count_valid": int(count_valid) if count_valid is not None else "",
                "coverage": round(coverage, 6) if coverage is not None else "",
                "constant_like": "yes" if is_constant_like else "no" if is_constant_like is not None else "",
                "quasi_discrete": "yes" if is_flag_like else "no" if is_flag_like is not None else "",
                "priority": "core",
                "tier": tier_from_r2(r2_val),
                "remarks": "",
            }
        )

    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()

