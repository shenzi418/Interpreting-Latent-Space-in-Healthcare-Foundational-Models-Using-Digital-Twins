#!/usr/bin/env python3
"""
Generate a manifest for the MIMIC-IV ECG diagnostic demo subset.

The manifest mirrors the MedalCare pipeline format so downstream scripts can
consume the records without additional changes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:
    import wfdb  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "wfdb is required. Install it with `pip install wfdb` before running this script."
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_DATASET_DIR = (
    REPO_ROOT / "mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1"
)
DEFAULT_RECORD_LIST = DEFAULT_DATASET_DIR / "record_list.csv"
DEFAULT_OUTPUT_MANIFEST = DEFAULT_DATASET_DIR / "mimic_demo_manifest.csv"

LABEL_COLUMNS: List[str] = [f"label_{idx}" for idx in range(8)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Root directory of the MIMIC-IV ECG demo subset.",
    )
    parser.add_argument(
        "--record-list",
        type=Path,
        default=None,
        help="Path to record_list.csv (defaults to <dataset_dir>/record_list.csv).",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=DEFAULT_OUTPUT_MANIFEST,
        help="Destination CSV manifest path.",
    )
    parser.add_argument(
        "--label-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV containing per-study label columns named "
            "'label_0' ... 'label_7'. Rows should be keyed by study_id."
        ),
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="test",
        help="Name to assign to the `split` column for all records.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of records to process (for debugging).",
    )
    return parser.parse_args()


def resolve_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_label_lookup(label_csv: Optional[Path]) -> Dict[str, List[float]]:
    if label_csv is None:
        return {}
    if not label_csv.exists():
        raise FileNotFoundError(f"Label CSV not found: {label_csv}")

    df = pd.read_csv(label_csv)
    required_columns = {"study_id", *LABEL_COLUMNS}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"Label CSV is missing required columns: {', '.join(sorted(missing))}"
        )

    lookup: Dict[str, List[float]] = {}
    for _, row in df.iterrows():
        study_key = str(row["study_id"])
        lookup[study_key] = [float(row[col]) for col in LABEL_COLUMNS]
    return lookup


def build_manifest_rows(
    records: Iterable[dict],
    dataset_dir: Path,
    labels: Dict[str, List[float]],
    split_name: str,
) -> List[dict]:
    manifest_rows: List[dict] = []
    sampling_rates: set[float] = set()
    lead_orders: set[str] = set()
    units_seen: set[str] = set()

    for idx, row in enumerate(records, start=1):
        relative_path = Path(str(row["path"]))
        wfdb_base = (dataset_dir / relative_path).resolve()
        hea_path = wfdb_base.with_suffix(".hea")
        dat_path = wfdb_base.with_suffix(".dat")

        if not hea_path.exists():
            raise FileNotFoundError(f"Missing WFDB header file: {hea_path}")
        if not dat_path.exists():
            raise FileNotFoundError(f"Missing WFDB data file: {dat_path}")

        header = wfdb.rdheader(str(wfdb_base))
        lead_order = ",".join(header.sig_name)
        lead_orders.add(lead_order)

        fs = float(header.fs)
        sampling_rates.add(fs)

        # Units are typically 'mV' for all channels; keep first entry if uniform.
        units = header.units or []
        unique_units = {u for u in units if u}
        units_seen.update(unique_units or {"mV"})
        units_value = next(iter(unique_units)) if unique_units else "mV"

        study_key = str(row["study_id"])
        label_vector = labels.get(study_key, [0.0] * len(LABEL_COLUMNS))

        manifest_entry = {
            "record_id": f"mimic_demo_{idx:06d}",
            "wfdb_path": str(wfdb_base),
            "original_csv_path": str(hea_path),
            "subject_id": int(row["subject_id"]),
            "study_id": int(row["study_id"]),
            "ecg_time": row["ecg_time"],
            "sampling_rate_hz": fs,
            "units": units_value,
            "lead_order": lead_order,
            "num_samples": int(header.sig_len),
            "split": split_name,
        }
        for column, value in zip(LABEL_COLUMNS, label_vector):
            manifest_entry[column] = value
        manifest_rows.append(manifest_entry)

    print(f"Discovered sampling rates: {sorted(sampling_rates)}")
    print(f"Unique lead orders: {len(lead_orders)}")
    if len(lead_orders) > 1:
        print("Warning: multiple lead permutations detected.")
    print(f"Units observed: {sorted(units_seen)}")

    return manifest_rows


def main() -> None:
    args = parse_args()

    dataset_dir = resolve_path(args.dataset_dir, REPO_ROOT)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    record_list_path = args.record_list
    if record_list_path is None:
        record_list_path = dataset_dir / "record_list.csv"
    record_list_path = resolve_path(record_list_path, REPO_ROOT)
    if not record_list_path.exists():
        raise FileNotFoundError(f"record_list.csv not found: {record_list_path}")

    print(f"Loading record list from: {record_list_path}")
    record_df = pd.read_csv(record_list_path)
    if args.limit:
        record_df = record_df.head(args.limit)
        print(f"[DEBUG] Limiting to first {len(record_df)} records.")

    label_lookup = load_label_lookup(
        resolve_path(args.label_csv, REPO_ROOT) if args.label_csv else None
    )

    manifest_rows = build_manifest_rows(
        record_df.to_dict(orient="records"),
        dataset_dir=dataset_dir,
        labels=label_lookup,
        split_name=args.split_name,
    )

    manifest_df = pd.DataFrame(manifest_rows)
    column_order = [
        "record_id",
        "wfdb_path",
        "original_csv_path",
        *LABEL_COLUMNS,
        "sampling_rate_hz",
        "units",
        "lead_order",
        "num_samples",
        "subject_id",
        "study_id",
        "ecg_time",
        "split",
    ]
    manifest_df = manifest_df[column_order]

    output_manifest = resolve_path(args.output_manifest, REPO_ROOT)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    manifest_df.to_csv(output_manifest, index=False)
    print(f"Wrote manifest with {len(manifest_df)} records to: {output_manifest}")
    print("Label columns contain zeros when no supervision file is provided.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # pragma: no cover - CLI entry point
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)


