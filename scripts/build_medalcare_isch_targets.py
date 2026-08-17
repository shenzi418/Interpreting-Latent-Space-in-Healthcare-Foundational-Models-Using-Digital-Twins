"""Build per-split MI ischaemia-target NPZs for B2 (in-domain mechanistic decoding).

For each MedalCare manifest split (train/val/test), this script:

1. Filters to MI rows (identified by ``\\mi\\`` in ``original_csv_path``;
   verified to agree 100% with ``label_1 == 1`` in the audit).
2. Resolves the corresponding ``run_*_VentricularParameters.txt`` file by
   swapping ``WP2_largeDataset_Noise`` -> ``WP2_largeDataset_ParameterFiles``
   (same convention used by ``scripts/datasets.py``).
3. Parses the four headline B2 targets from ``isch[0]``:

   - ``phi``           (radians, in approximately ``[-pi, pi]``; circular)
   - ``z``             (longitudinal position)
   - ``size``          (lesion size)
   - ``rho_eps_max``   (transmurality, binary in ``{0.3, 1.0}``)

4. Stores split metadata (path-derived coronary territory, transmural class,
   LCX subtype, run id) and ``idx_in_split`` -- the row index in the
   ``df[df.split==X].reset_index(drop=True)`` ordering, which matches the
   ordering of latent files exported by ``scripts/export_latents.py``.

Output: ``data/theta_mi_{train,val,test}.npz``.

Usage::

    python scripts/build_medalcare_isch_targets.py

The script also performs an alignment audit against the existing latent
exports (``outputs/latents/exp7_medalcare_{train,val,test}/latents.npz``)
when those are available.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
from scripts.medalcare_paths import (  # noqa: E402
    assert_label_schema,
    is_mi_path,
    parse_territory_from_path,
)

DEFAULT_MANIFEST = REPO_ROOT / "data" / "medalcare_filtered_manifest_dataset_split.csv"
DEFAULT_OUTDIR = REPO_ROOT / "data"
DEFAULT_AUDIT_LATENTS = {
    # Existing latent export naming convention:
    #   - train split -> ``outputs/latents/exp7_medalcare_train/``
    #   - test split  -> ``outputs/latents/exp7_medalcare/`` (no suffix; default)
    #   - val split   -> not exported (B2 uses train-CV + test, val not needed)
    "train": REPO_ROOT / "outputs" / "latents" / "exp7_medalcare_train" / "latents.npz",
    "val":   REPO_ROOT / "outputs" / "latents" / "exp7_medalcare_val"   / "latents.npz",
    "test":  REPO_ROOT / "outputs" / "latents" / "exp7_medalcare"       / "latents.npz",
}

ISCH_KEYS = ("isch[0].phi", "isch[0].z", "isch[0].size", "isch[0].rho_eps_max")
MI_LABEL_COL_INDEX = 1  # native MedalCare 8-class column for MI

# ---------------------------------------------------------------------------
# 4-class / 8-class anatomical-territory mapping (added 2026-05-13 for
# Track 3 B2-CD redux). The 4-class scheme matches the refined PTB-XL classes
# in scripts/build_ptbxl_mi_subclass.py; the 8-class scheme is the full
# folder-level granularity for the in-domain audit.
# ---------------------------------------------------------------------------
TERRITORIES_4C: Tuple[str, ...] = (
    "Anteroseptal",     # MedalCare LAD_* -- PTB-XL ASMI/AMI/INJAS only
    "Anterolateral",    # MedalCare LCX_*_ant -- PTB-XL ALMI/INJAL or AMI+LMI
    "Inferior",         # MedalCare RCA_* -- PTB-XL IMI/INJIN only
    "Inferolateral",    # MedalCare LCX_*_post -- PTB-XL ILMI/INJIL/IPLMI
)
TERRITORIES_8C: Tuple[str, ...] = (
    "LAD_0.3",
    "LAD_1.0",
    "LCX_0.3_ant",
    "LCX_0.3_post",
    "LCX_1.0_ant",
    "LCX_1.0_post",
    "RCA_0.3",
    "RCA_1.0",
)


def derive_territory_4c(coronary: str, lcx_subtype: str) -> str:
    """Map (coronary, lcx_subtype) -> 4-class territory from the FOLDER NAME.

    Returns one of TERRITORIES_4C, or "" if the input is unrecognised.

    NOTE (defect D1, 2026-08-10): this is the *metadata* labelling and it
    contradicts the simulated geometry for the rho=0.3 LCX buckets. It is kept
    only to populate ``territory_4c_folder`` for the sensitivity analysis. The
    canonical label is now ``derive_territory_4c_from_phi``.
    """
    if coronary == "LAD":
        return "Anteroseptal"
    if coronary == "RCA":
        return "Inferior"
    if coronary == "LCX":
        if lcx_subtype == "ant":
            return "Anterolateral"
        if lcx_subtype == "post":
            return "Inferolateral"
    return ""


# --- D1 fix: phi wedges -------------------------------------------------------
# Empirically verified against all three splits of theta_mi_*.npz (2026-08-10).
# Every folder bucket occupies a clean, non-overlapping phi wedge:
#
#   bucket          folder ->4c      phi range           phi circ-mean
#   LAD_0.3         Anteroseptal     [+0.000, +1.999]    +1.00
#   LAD_1.0         Anteroseptal     [+0.000, +1.997]    +1.00
#   LCX_0.3_ant     Anterolateral    [+2.003, +3.139]    +2.57
#   LCX_0.3_post    Inferolateral    [+2.003, +3.140]    +2.57   <-- SAME WEDGE
#   LCX_1.0_ant     Anterolateral    [+2.004, +3.139]    +2.57
#   LCX_1.0_post    Inferolateral    [-3.139, -2.006]    -2.57
#   RCA_0.3         Inferior         [-1.999, -0.003]    -1.00
#   RCA_1.0         Inferior         [-1.997, -0.001]    -1.00
#
# ``LCX_0.3_ant`` and ``LCX_0.3_post`` are the same distribution in all four
# theta parameters under two different labels, so no decoder -- not even an
# oracle with perfect theta knowledge -- can separate them. That caps the
# 4-class task at accuracy 0.9167 / macro-F1 0.8643 with Inferolateral recall
# pinned at exactly 0.500, and every 4c number in the project had been compared
# against an implicit ceiling of 1.0.
#
# phi is the actual simulated geometry; the folder name records the *intent* of
# the simulation batch, not the physics. So phi wins.
PHI_4C_OUTER_BOUNDARY = 2.0   # |phi| > 2.0  -> lateral
PHI_4C_INNER_BOUNDARY = 0.0   # sign of phi  -> anterior (+) vs inferior (-)


def derive_territory_4c_from_phi(phi: float) -> str:
    """Map ``isch[0].phi`` -> 4-class anatomical territory (canonical, D1 fix).

    Wedges, on phi wrapped to [-pi, +pi]:

        [ 0.0, +2.0]  -> Anteroseptal
        (+2.0, +pi ]  -> Anterolateral
        [-2.0,  0.0)  -> Inferior
        [-pi,  -2.0)  -> Inferolateral

    These are the same boundaries used by ``hardcoded_phi_to_4c`` in
    ``analysis/phase_b2_infarct_decoding.py``, so the ground truth and the
    Pipeline-B wedge baseline are now defined consistently.
    """
    p = float(np.arctan2(np.sin(phi), np.cos(phi)))  # wrap to [-pi, +pi]
    if p > PHI_4C_OUTER_BOUNDARY:
        return "Anterolateral"
    if p >= PHI_4C_INNER_BOUNDARY:
        return "Anteroseptal"
    if p >= -PHI_4C_OUTER_BOUNDARY:
        return "Inferior"
    return "Inferolateral"


def derive_territory_8c(coronary: str, lcx_subtype: str, transmural: float) -> str:
    """Map (coronary, lcx_subtype, transmural) -> 8-class folder name."""
    if transmural < 0.5:
        trans = "0.3"
    else:
        trans = "1.0"
    if coronary == "LCX":
        if lcx_subtype not in ("ant", "post"):
            return ""
        return f"LCX_{trans}_{lcx_subtype}"
    if coronary in ("LAD", "RCA"):
        return f"{coronary}_{trans}"
    return ""


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------

_VALUE_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)([a-zA-Z/]+)?\s*$")


def parse_value(raw: str) -> Optional[float]:
    """Parse a single value from ``key = value`` line.

    Handles optional unit suffix (e.g. ``174.45ms``) and scientific notation.
    Returns ``None`` for non-numeric or empty entries.
    """
    text = raw.strip().strip('"').strip("'")
    if not text:
        return None
    if text.lower() in {"true", "false"}:
        return None
    match = _VALUE_RE.match(text)
    if not match:
        return None
    return float(match.group(1))


def parse_parameter_file(path: Path) -> Dict[str, float]:
    """Parse a MedalCare ventricular parameter file into ``{key: float}``."""
    values: Dict[str, float] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, raw = (part.strip() for part in line.split("=", 1))
        value = parse_value(raw)
        if value is None:
            continue
        values[key] = value
    return values


def parameter_path_for_csv(original_csv_path: Path) -> Path:
    """Map ``WP2_largeDataset_Noise/.../run_X_filtered.csv`` ->
    ``WP2_largeDataset_ParameterFiles/.../run_X_VentricularParameters.txt``.
    """
    parts = list(original_csv_path.parts)
    lowered = [p.lower() for p in parts]
    if "wp2_largedataset_noise" not in lowered:
        raise ValueError(f"Unexpected MedalCare path (no Noise dir): {original_csv_path}")
    idx = lowered.index("wp2_largedataset_noise")
    parts[idx] = "WP2_largeDataset_ParameterFiles"
    base_dir = Path(*parts[:-1])
    stem = original_csv_path.stem
    if stem.endswith("_filtered"):
        stem = stem[: -len("_filtered")]
    run_base = stem if stem.startswith("run_") else f"run_{stem}"
    return base_dir / f"{run_base}_VentricularParameters.txt"


# ---------------------------------------------------------------------------
# Path-derived metadata
# ---------------------------------------------------------------------------
# `parse_territory_from_path` and `is_mi_path` now live in
# `scripts/medalcare_paths.py` and match on path *segments*. The versions that
# used to live here matched the substring "/mi/", which also fires on any
# ancestor directory of the dataset root -- see that module's docstring.


def build_split(
    split_df: pd.DataFrame,
    split_name: str,
    audit_latents_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Build the target arrays for a single split.

    ``split_df`` must already be filtered to the desired split and
    ``reset_index(drop=True)``-ed so its row order matches the latent export.
    """
    n_total = len(split_df)
    mi_mask = split_df["original_csv_path"].apply(is_mi_path)
    mi_idx = np.flatnonzero(mi_mask.to_numpy())
    n_mi = mi_idx.size
    print(f"[{split_name}] total={n_total}  MI={n_mi}")

    idx_in_split: List[int] = []
    phi: List[float] = []
    z: List[float] = []
    size: List[float] = []
    rho_eps_max: List[float] = []
    coronary: List[str] = []
    lcx_subtype: List[str] = []
    transmural: List[float] = []
    run_id: List[str] = []

    missing_files: List[str] = []
    missing_keys: Dict[str, int] = {k: 0 for k in ISCH_KEYS}
    transmural_mismatch: List[Tuple[str, float, float]] = []

    for i in mi_idx:
        row = split_df.iloc[int(i)]
        csv_path_str = str(row["original_csv_path"])
        try:
            cor, sub, trans_path = parse_territory_from_path(csv_path_str)
        except ValueError as exc:
            print(f"  [WARN] {split_name} row {i}: territory parse failed: {exc}")
            continue

        param_path = parameter_path_for_csv(Path(csv_path_str))
        if not param_path.is_file():
            missing_files.append(str(param_path))
            continue
        params = parse_parameter_file(param_path)
        # Check all four targets are present.
        missing_for_row = [k for k in ISCH_KEYS if k not in params]
        if missing_for_row:
            for k in missing_for_row:
                missing_keys[k] += 1
            print(
                f"  [WARN] {split_name} row {i}: missing keys {missing_for_row} "
                f"in {param_path.name}; skipping"
            )
            continue

        # Cross-check: rho_eps_max from file should match transmural class encoded in the path.
        rho_file = float(params["isch[0].rho_eps_max"])
        if not np.isclose(rho_file, trans_path):
            transmural_mismatch.append((csv_path_str, trans_path, rho_file))

        idx_in_split.append(int(i))
        phi.append(float(params["isch[0].phi"]))
        z.append(float(params["isch[0].z"]))
        size.append(float(params["isch[0].size"]))
        rho_eps_max.append(rho_file)
        coronary.append(cor)
        lcx_subtype.append(sub)
        transmural.append(trans_path)
        run_id.append(str(row["run_id"]))

    if missing_files:
        print(f"  [WARN] {split_name}: {len(missing_files)} parameter files missing")
        for p in missing_files[:3]:
            print(f"    -> {p}")
    if any(missing_keys.values()):
        print(f"  [WARN] {split_name}: missing-key counts: {missing_keys}")
    if transmural_mismatch:
        print(
            f"  [WARN] {split_name}: {len(transmural_mismatch)} rows with rho_eps_max "
            "mismatch between file and path encoding (first 3 below)"
        )
        for entry in transmural_mismatch[:3]:
            print(f"    -> {entry}")

    # ---- territory_4c: canonical = derived from phi (defect D1 fix) ----------
    # The folder-name labelling is retained as ``territory_4c_folder`` so the
    # "keep the labels, report against the 0.867 oracle ceiling" sensitivity can
    # be run without rebuilding.
    territory_4c_folder = [
        derive_territory_4c(c, s) for c, s in zip(coronary, lcx_subtype)
    ]
    territory_4c = [derive_territory_4c_from_phi(p) for p in phi]
    territory_8c = [
        derive_territory_8c(c, s, t)
        for c, s, t in zip(coronary, lcx_subtype, transmural)
    ]

    # Cross-check phi-derived vs folder-derived, and FAIL on any disagreement
    # that is not the one documented defect. LCX_*_post buckets whose phi is
    # positive are the known D1 contradiction (the simulation is anterior by
    # every theta parameter despite the "post" folder name); every other
    # disagreement would mean the phi wedges are wrong and must stop the build.
    disagree: Dict[str, int] = defaultdict(int)
    unexpected: List[Tuple[str, str, str, float]] = []
    for k in range(len(territory_4c)):
        if territory_4c[k] == territory_4c_folder[k]:
            continue
        bucket = territory_8c[k]
        disagree[f"{bucket}: {territory_4c_folder[k]} -> {territory_4c[k]}"] += 1
        known_d1 = (
            coronary[k] == "LCX"
            and lcx_subtype[k] == "post"
            and territory_4c_folder[k] == "Inferolateral"
            and territory_4c[k] == "Anterolateral"
        )
        if not known_d1:
            unexpected.append(
                (run_id[k], territory_4c_folder[k], territory_4c[k], phi[k])
            )

    if disagree:
        print(
            f"  [D1] {split_name}: {sum(disagree.values())} rows relabelled from "
            "folder name to phi-derived territory:"
        )
        for key, cnt in sorted(disagree.items()):
            print(f"    -> {key}: {cnt}")
    if unexpected:
        raise ValueError(
            f"{split_name}: {len(unexpected)} rows disagree between phi-derived "
            "and folder-derived territory_4c in a way that is NOT the known D1 "
            "LCX_*_post contradiction. The phi wedge boundaries or the folder "
            f"parsing are wrong -- refusing to write targets. First 5: "
            f"{unexpected[:5]}"
        )

    target = {
        "idx_in_split": np.asarray(idx_in_split, dtype=np.int64),
        "phi": np.asarray(phi, dtype=np.float64),
        "z": np.asarray(z, dtype=np.float64),
        "size": np.asarray(size, dtype=np.float64),
        "rho_eps_max": np.asarray(rho_eps_max, dtype=np.float64),
        "coronary": np.asarray(coronary, dtype=object),
        "lcx_subtype": np.asarray(lcx_subtype, dtype=object),
        "transmural": np.asarray(transmural, dtype=np.float64),
        "run_id": np.asarray(run_id, dtype=object),
        "territory_4c": np.asarray(territory_4c, dtype=object),
        "territory_4c_folder": np.asarray(territory_4c_folder, dtype=object),
        "territory_8c": np.asarray(territory_8c, dtype=object),
    }

    # Optional alignment audit against an existing latent file.
    if audit_latents_path is not None and audit_latents_path.is_file():
        with np.load(audit_latents_path, allow_pickle=True) as data:
            Y = data["Y"]
        if Y.shape[0] != n_total:
            print(
                f"  [WARN] {split_name}: latent Y rows {Y.shape[0]} != manifest split rows {n_total}; "
                "alignment audit skipped"
            )
        else:
            mi_rows_from_Y = int(Y[:, MI_LABEL_COL_INDEX].sum())
            count_match = (mi_rows_from_Y == n_mi)
            label_match = bool(
                np.all(Y[target["idx_in_split"], MI_LABEL_COL_INDEX] == 1)
            )
            print(
                f"  [AUDIT] {split_name}: Y[:,{MI_LABEL_COL_INDEX}].sum() = {mi_rows_from_Y}, "
                f"path-MI count = {n_mi}, "
                f"all idx have label_1==1: {label_match}"
            )
            if not (count_match and label_match):
                raise RuntimeError(
                    f"Alignment audit failed for split={split_name}; "
                    "manifest order does NOT match latent order."
                )

    return target


def summarise(target: Dict[str, np.ndarray], split_name: str) -> Dict[str, object]:
    """Print and return a small summary dict for the build report."""
    n = target["idx_in_split"].size
    cor = target["coronary"]
    counts_cor = {c: int((cor == c).sum()) for c in ("LAD", "LCX", "RCA")}
    counts_trans = {
        "0.3": int(np.isclose(target["transmural"], 0.3).sum()),
        "1.0": int(np.isclose(target["transmural"], 1.0).sum()),
    }
    phi = target["phi"]
    z = target["z"]
    size = target["size"]
    summary: Dict[str, object] = {
        "split": split_name,
        "n_mi": n,
        "coronary_counts": counts_cor,
        "transmural_counts": counts_trans,
        "phi_range": [float(phi.min()), float(phi.max())],
        "phi_mean_circular": float(np.arctan2(np.sin(phi).mean(), np.cos(phi).mean())),
        "z_range":   [float(z.min()),   float(z.max())],
        "z_mean":    float(z.mean()),
        "size_range":[float(size.min()),float(size.max())],
        "size_mean": float(size.mean()),
    }
    print(
        f"  [SUMMARY {split_name}] n_MI={n}  cor={counts_cor}  "
        f"transmural={counts_trans}\n"
        f"     phi range=[{phi.min():.3f}, {phi.max():.3f}]  "
        f"z range=[{z.min():.3f}, {z.max():.3f}]  "
        f"size range=[{size.min():.1f}, {size.max():.1f}]"
    )
    # Per-coronary phi range (used downstream to define empirical phi bins for cross-domain B2-CD).
    per_cor_phi: Dict[str, Dict[str, float]] = {}
    for c in ("LAD", "LCX", "RCA"):
        sub = phi[cor == c]
        if sub.size > 0:
            per_cor_phi[c] = {
                "n": int(sub.size),
                "min": float(sub.min()),
                "max": float(sub.max()),
                "mean_circular": float(np.arctan2(np.sin(sub).mean(), np.cos(sub).mean())),
            }
    summary["per_coronary_phi"] = per_cor_phi

    territory_4c_arr = target["territory_4c"]
    territory_8c_arr = target["territory_8c"]
    counts_4c = {t: int((territory_4c_arr == t).sum()) for t in TERRITORIES_4C}
    counts_8c = {t: int((territory_8c_arr == t).sum()) for t in TERRITORIES_8C}
    summary["territory_4c_counts"] = counts_4c
    summary["territory_8c_counts"] = counts_8c
    print(
        f"  [SUMMARY {split_name}] territory_4c={counts_4c}\n"
        f"     territory_8c={counts_8c}"
    )
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--splits", type=str, default="train,val,test",
        help="Comma-separated list of splits to build (default: train,val,test).",
    )
    parser.add_argument(
        "--no-audit", action="store_true",
        help="Skip the alignment audit against existing latent files.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if "split" not in df.columns:
        raise ValueError("Manifest must include a 'split' column.")
    assert_label_schema(df.columns)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, object]] = []
    for split in [s.strip().lower() for s in args.splits.split(",") if s.strip()]:
        sub = df[df["split"].str.lower() == split].reset_index(drop=True)
        if sub.empty:
            print(f"[{split}] empty -- skipping")
            continue
        audit_path = None if args.no_audit else DEFAULT_AUDIT_LATENTS.get(split)
        target = build_split(sub, split, audit_latents_path=audit_path)
        summary = summarise(target, split)
        summaries.append(summary)

        out_path = args.out_dir / f"theta_mi_{split}.npz"
        np.savez_compressed(out_path, **target)
        print(f"  -> saved {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")

    # Persist a human-readable summary alongside the NPZs.
    summary_path = args.out_dir / "theta_mi_build_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nWrote build summary -> {summary_path}")


if __name__ == "__main__":
    main()
