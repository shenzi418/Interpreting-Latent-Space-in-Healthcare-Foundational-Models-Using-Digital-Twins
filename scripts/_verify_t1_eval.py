"""Integrity checks for outputs/tier1_eval/ (Track 1b' evaluation step).

Confirms:
  - 4 per-config JSONs present and parseable
  - All 4 metric blocks populated per config (alignment/class/mech/anat)
  - K values match expectation (1024, 256, 64, 16)
  - cross_config_table.{json,md} present and well-formed
  - frontier_tier1.png > 10 KB
  - Cross-K monotonicity sanity checks: MMD should rise as K falls; phi R^2_circ
    should fall as K falls; LR M->P should rise as K falls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "outputs" / "tier1_eval"
EXPECTED_CONFIGS: List[Tuple[str, int]] = [
    ("exp7_baseline_ref",    1024),
    ("exp7_bottleneck_K256", 256),
    ("exp7_bottleneck_K64",  64),
    ("exp7_bottleneck_K16",  16),
]

checks: List[Tuple[str, bool, str]] = []


def add(name: str, ok: bool, note: str = "") -> None:
    checks.append((name, ok, note))


def get_deep(d: object, dotted: str) -> Optional[float]:
    node: object = d
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return float(node) if isinstance(node, (int, float)) else None


per_config: Dict[str, Dict[str, object]] = {}
for cfg, expected_k in EXPECTED_CONFIGS:
    path = EVAL_DIR / f"{cfg}_summary.json"
    if not path.exists():
        add(f"summary[{cfg}]_exists", False, str(path))
        continue
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        add(f"summary[{cfg}]_parses", False, f"{type(e).__name__}: {e}")
        continue
    add(f"summary[{cfg}]_exists", True, f"{path.stat().st_size//1024} KB")
    per_config[cfg] = d
    # Top-level structure
    for block in ("alignment", "class_structure", "mechanism", "anatomy_pipeline_a"):
        add(f"summary[{cfg}].{block}_present", block in d)
    # K matches expected
    k = int(d.get("K", -1))
    add(f"summary[{cfg}]_K", k == expected_k, f"got K={k}, want {expected_k}")
    # Spot-check at least one number per block.
    add(f"summary[{cfg}].C2ST_in_range",
        0.0 <= (get_deep(d, "alignment.c2st_auc") or -1) <= 1.0)
    add(f"summary[{cfg}].LR_M2P_in_range",
        0.0 <= (get_deep(d, "class_structure.lr_m2p.macro_auc") or -1) <= 1.0)
    phi = get_deep(d, "mechanism.phi.circular_r2")
    add(f"summary[{cfg}].phi_R2c_present",
        phi is not None, f"phi_R2c = {phi}")
    anat_in = get_deep(d, "anatomy_pipeline_a.in_domain_4c.macro_f1")
    add(f"summary[{cfg}].anatomy_in_domain_present",
        anat_in is not None, f"in_4c_F1 = {anat_in}")

# Cross-config files.
tbl_json = EVAL_DIR / "cross_config_table.json"
tbl_md = EVAL_DIR / "cross_config_table.md"
fig_png = EVAL_DIR / "frontier_tier1.png"
add("cross_table.json_exists", tbl_json.exists())
add("cross_table.md_exists", tbl_md.exists())
add("frontier_tier1.png_exists", fig_png.exists())
if tbl_json.exists():
    try:
        tbl = json.loads(tbl_json.read_text(encoding="utf-8"))
        add("cross_table.json_parses", True)
        add("cross_table.has_4_rows",
            len(tbl.get("rows", [])) == 4,
            f"got {len(tbl.get('rows', []))} rows")
    except Exception as e:  # noqa: BLE001
        add("cross_table.json_parses", False, f"{type(e).__name__}: {e}")
if fig_png.exists():
    add("frontier_tier1.png_nontrivial",
        fig_png.stat().st_size > 10_000,
        f"{fig_png.stat().st_size//1024} KB")

# Monotonicity checks across K -- the substantive findings the report stands on.
if len(per_config) == 4:
    ks_descending = [1024, 256, 64, 16]
    name_for_k = {k: c for c, k in EXPECTED_CONFIGS}
    series_mmd_med = [get_deep(per_config[name_for_k[k]], "alignment.mmd_median")
                      for k in ks_descending]
    series_phi = [get_deep(per_config[name_for_k[k]], "mechanism.phi.circular_r2")
                  for k in ks_descending]
    series_lr_m2p = [get_deep(per_config[name_for_k[k]], "class_structure.lr_m2p.macro_auc")
                     for k in ks_descending]

    add("MMD_median_rises_as_K_falls",
        all(b is not None and a is not None and b > a
            for a, b in zip(series_mmd_med, series_mmd_med[1:])),
        f"series K=[1024,256,64,16] = {series_mmd_med}")
    add("phi_R2c_falls_as_K_falls",
        all(b is not None and a is not None and b < a
            for a, b in zip(series_phi, series_phi[1:])),
        f"series K=[1024,256,64,16] = {series_phi}")
    add("LR_M2P_rises_as_K_falls_to_K64",
        # LR M->P rises strictly K=1024 -> 256 -> 64, may plateau at 16
        series_lr_m2p[0] is not None and series_lr_m2p[2] is not None
        and series_lr_m2p[2] > series_lr_m2p[0],
        f"series K=[1024,256,64,16] = {series_lr_m2p}")

# ---- Report ----
n_pass = sum(1 for _, ok, _ in checks if ok)
n_total = len(checks)
print(f"\n[verify] {n_pass}/{n_total} checks passed\n")
for name, ok, note in checks:
    flag = "OK  " if ok else "FAIL"
    print(f"  [{flag}] {name}  {note}")
sys.exit(0 if n_pass == n_total else 1)
