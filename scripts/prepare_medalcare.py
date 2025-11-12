#!/usr/bin/env python3
"""
Consolidated MedalCare-XL dataset preparation script.

This script:
1. Enumerates filtered CSV files from MedalCare-XL dataset
2. Converts CSV files to WFDB format (with proper lead ordering)
3. Validates conversions and removes failures
4. Creates final manifest with one-hot encoded labels
5. Documents metadata (lead order, sampling rate, units)

Usage:
    python prepare_medalcare.py [--input-dir DIR] [--output-dir DIR] [--manifest FILE]
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

DEFAULT_INPUT_DIR = BASE_DIR / "MedalCare-XL/WP2_largeDataset_Noise"
DEFAULT_OUTPUT_DIR = BASE_DIR / "MedalRaw"
DEFAULT_MANIFEST_PATH = BASE_DIR / "MedalRaw/medalcare_filtered_manifest.csv"

# WFDB
try:
    from wfdb.io.convert import csv as wfdb_csv
except ImportError:
    raise RuntimeError(
        "wfdb is required. Install with: pip install wfdb"
    )

# Pathology keywords and their corresponding one-hot encoding indices
PATHOLOGY_KEYWORDS = {
    'sinus': 0,    # Normal sinus rhythm
    'mi': 1,       # Myocardial infarction
    'rbbb': 2,     # Right bundle branch block
    'lbbb': 3,     # Left bundle branch block
    'lae': 4,      # Left atrial enlargement
    'iab': 5,      # Incomplete atrioventricular block
    'fam': 6,      # Familial/genetic condition
    'avblock': 7   # Atrioventricular block
}

# ECG lead order (as required by ECGFounder)
LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# ECG parameters
SAMPLING_RATE = 500  # Hz
UNITS = 'mV'


def find_pathology_in_path(path: Path) -> Optional[int]:
    """
    Determine pathology class from file path.
    
    Args:
        path: File path (Path object)
        
    Returns:
        Pathology class index (0-7) or None if ambiguous/not found
    """
    path_lower = str(path).lower()
    matched = []
    
    # Iterate in sorted order for deterministic behavior
    for keyword in sorted(PATHOLOGY_KEYWORDS.keys()):
        if keyword in path_lower:
            matched.append(PATHOLOGY_KEYWORDS[keyword])
    
    if len(matched) == 1:
        return matched[0]
    elif len(matched) > 1:
        print(f"[WARN] Multiple pathologies found in path: {path}")
        return None
    else:
        print(f"[WARN] No pathology found in path: {path}")
        return None


def enumerate_filtered_files(base_dir: Path) -> List[Tuple[Path, int]]:
    """
    Find all filtered CSV files and extract pathology labels.
    
    Args:
        base_dir: Base directory to search (e.g., MedalCare-XL/WP2_largeDataset_Noise)
        
    Returns:
        List of (file_path, pathology_class) tuples, sorted for deterministic output
    """
    filtered_files = []
    
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    print(f"Scanning for filtered CSV files in: {base_dir}")
    
    # Collect all files first
    for root, dirs, files in os.walk(base_dir):
        # Sort directories and files for deterministic order
        dirs.sort()
        files.sort()
        
        for file in files:
            if file.endswith('_filtered.csv'):
                file_path = Path(root) / file
                pathology = find_pathology_in_path(file_path)
                
                if pathology is not None:
                    filtered_files.append((file_path, pathology))
                else:
                    print(f"[SKIP] Skipping {file_path} (ambiguous or unknown pathology)")
    
    # Sort by file path for deterministic output
    filtered_files.sort(key=lambda x: (str(x[0]), x[1]))
    
    print(f"Found {len(filtered_files)} filtered CSV files")
    return filtered_files


def prepare_csv_for_wfdb(csv_path: Path, output_path: Path) -> bool:
    """
    Prepare CSV file for WFDB conversion.
    
    MedalCare-XL CSVs have leads as rows and time as columns.
    WFDB expects leads as columns with named headers.
    
    Args:
        csv_path: Input CSV file path
        output_path: Output CSV file path (temporary, for WFDB conversion)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read CSV (leads as rows, time as columns)
        df = pd.read_csv(csv_path, header=None)
        
        # Verify we have 12 leads
        if df.shape[0] != 12:
            print(f"[ERROR] Expected 12 leads, got {df.shape[0]} in {csv_path}")
            return False
        
        # Transpose: rows = time samples, columns = leads
        df_transposed = df.T
        
        # Set column names to lead order
        df_transposed.columns = LEAD_ORDER
        
        # Save to temporary file for WFDB conversion
        df_transposed.to_csv(output_path, index=False)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to prepare CSV {csv_path}: {e}")
        return False


def convert_to_wfdb(csv_path: Path, output_dir: Path, fs: int = SAMPLING_RATE, verbose: bool = False) -> bool:
    """
    Convert CSV file to WFDB format.
    
    Args:
        csv_path: Input CSV file (must have leads as columns)
        output_dir: Directory to write WFDB files (should be same as csv_path.parent)
        fs: Sampling rate in Hz
        verbose: Whether to print detailed error messages
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure CSV file exists
        if not csv_path.exists():
            if verbose:
                print(f"[ERROR] CSV file does not exist: {csv_path}")
            return False
        
        # Convert to WFDB
        # Note: wfdb_csv.csv_to_wfdb creates files in write_dir based on the CSV filename
        # It extracts the base name from file_name and creates {basename}.hea and {basename}.dat
        wfdb_csv.csv_to_wfdb(
            file_name=str(csv_path.resolve()),  # Use absolute path
            fs=fs,
            units=UNITS,
            write_dir=str(output_dir.resolve())  # Use absolute path
        )
        
        # Verify files were created (they should be in output_dir with name based on csv_path.stem)
        expected_base = output_dir / csv_path.stem
        expected_hea = expected_base.with_suffix('.hea')
        expected_dat = expected_base.with_suffix('.dat')
        
        if expected_hea.exists() and expected_dat.exists():
            return True
        else:
            if verbose:
                print(f"[DEBUG] WFDB conversion returned success but files not found:")
                print(f"        Expected .hea: {expected_hea} (exists: {expected_hea.exists()})")
                print(f"        Expected .dat: {expected_dat} (exists: {expected_dat.exists()})")
                # List files in output_dir
                matching_files = list(output_dir.glob(f"{csv_path.stem}*"))
                print(f"        Files matching '{csv_path.stem}*': {[f.name for f in matching_files]}")
            return False
            
    except Exception as e:
        if verbose:
            import traceback
            print(f"[ERROR] WFDB conversion failed for {csv_path}: {e}")
            traceback.print_exc()
        return False


def verify_wfdb_files(csv_path: Path) -> bool:
    """
    Verify that WFDB files were created successfully.
    
    Args:
        csv_path: Base path (without extension) for WFDB files
        
    Returns:
        True if .hea and .dat files exist, False otherwise
    """
    hea_file = csv_path.with_suffix('.hea')
    dat_file = csv_path.with_suffix('.dat')
    
    return hea_file.exists() and dat_file.exists()


def create_one_hot_label(pathology_class: int, num_classes: int = 8) -> List[int]:
    """
    Create one-hot encoded label vector.
    
    Args:
        pathology_class: Pathology class index (0-7)
        num_classes: Total number of classes (default: 8)
        
    Returns:
        List of binary values (one-hot encoding)
    """
    label = [0] * num_classes
    if 0 <= pathology_class < num_classes:
        label[pathology_class] = 1
    return label


def create_manifest(
    records: List[Tuple[Path, int, Path]],
    output_file: Path,
    fs: int = SAMPLING_RATE,
    units: str = UNITS
):
    """
    Create final manifest CSV with one-hot encoded labels.
    
    Args:
        records: List of (original_csv_path, pathology_class, wfdb_base_path) tuples
                (should be sorted for deterministic output)
        output_file: Output manifest CSV file path
        fs: Sampling rate
        units: Signal units
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure records are sorted for deterministic output
    # Sort by original path (string representation for consistency)
    records_sorted = sorted(records, key=lambda x: (str(x[0]), x[1], str(x[2])))
    
    # Create header
    header = ['record_id', 'wfdb_path', 'original_csv_path']
    header.extend([f'label_{i}' for i in range(8)])  # One-hot labels
    header.extend(['sampling_rate_hz', 'units', 'lead_order'])
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for idx, (original_path, pathology_class, wfdb_path) in enumerate(records_sorted, 1):
            record_id = f"medalcare_{idx:06d}"
            
            # Create one-hot label
            one_hot = create_one_hot_label(pathology_class)
            
            # Write row (use absolute paths as strings for consistency)
            row = [
                record_id,
                str(Path(wfdb_path).resolve()),  # Absolute path as string
                str(Path(original_path).resolve()),  # Absolute path as string
            ]
            row.extend(one_hot)
            row.extend([
                str(fs),
                units,
                ','.join(LEAD_ORDER)
            ])
            
            writer.writerow(row)
    
    print(f"Manifest written to: {output_file}")
    print(f"Total records: {len(records_sorted)}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MedalCare-XL dataset for ECGFounder fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  python prepare_medalcare.py
  
  # Specify custom paths
  python prepare_medalcare.py \\
    --input-dir ./MedalCare-XL/WP2_largeDataset_Noise \\
    --output-dir ./MedalRaw \\
    --manifest ./MedalRaw/medalcare_filtered_manifest.csv
        """
    )
    
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help='Input directory containing filtered CSV files (default: ./MedalCare-XL/WP2_largeDataset_Noise)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for WFDB files (default: ./MedalRaw)'
    )
    
    parser.add_argument(
        '--manifest',
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help='Output manifest CSV file (default: ./MedalRaw/medalcare_filtered_manifest.csv)'
    )
    
    parser.add_argument(
        '--fs',
        type=int,
        default=SAMPLING_RATE,
        help=f'Sampling rate in Hz (default: {SAMPLING_RATE})'
    )
    
    parser.add_argument(
        '--skip-conversion',
        action='store_true',
        help='Skip WFDB conversion (only create manifest from existing WFDB files)'
    )
    
    parser.add_argument(
        '--test-mode',
        type=int,
        default=None,
        help='Test mode: only process first N files (for debugging)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose error messages'
    )
    
    args = parser.parse_args()
    
    def resolve_path(path: Path) -> Path:
        return path if path.is_absolute() else (BASE_DIR / path).resolve()
    
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    manifest_path = resolve_path(args.manifest)
    
    # Step 1: Enumerate filtered CSV files
    print("=" * 80)
    print("Step 1: Enumerating filtered CSV files")
    print("=" * 80)
    files_with_labels = enumerate_filtered_files(input_dir)
    
    if not files_with_labels:
        print("[ERROR] No filtered CSV files found!")
        sys.exit(1)
    
    # Step 2: Convert to WFDB (if not skipped)
    print("\n" + "=" * 80)
    print("Step 2: Converting CSV files to WFDB format")
    print("=" * 80)
    
    # Apply test mode limit if specified
    if args.test_mode:
        files_with_labels = files_with_labels[:args.test_mode]
        print(f"[TEST MODE] Processing only first {args.test_mode} files")
    
    successful_records = []
    failed_records = []
    
    if not args.skip_conversion:
        for idx, (csv_path, pathology_class) in enumerate(files_with_labels, 1):
            if idx % 100 == 0:
                print(f"Processing {idx}/{len(files_with_labels)} files...")
            
            # Show progress for first few files or in verbose mode
            show_details = args.verbose or idx <= 3
            
            wfdb_base_path = csv_path.parent / csv_path.stem
            
            # Create formatted CSV file with same base name as desired WFDB output
            # WFDB will create files based on the input CSV filename (without .csv extension)
            # So if we create "000001_filtered_wfdb.csv", it creates "000001_filtered_wfdb.hea" and ".dat"
            # We want "000001_filtered.hea" and ".dat", so we need to use a temp name then rename
            temp_csv_name = f"{csv_path.stem}_wfdb_temp.csv"
            temp_csv = csv_path.parent / temp_csv_name
            
            try:
                # Prepare CSV for WFDB (transpose and add headers)
                if not prepare_csv_for_wfdb(csv_path, temp_csv):
                    failed_records.append((csv_path, pathology_class, "CSV preparation failed"))
                    continue
                
                # Convert to WFDB - creates files based on temp_csv name
                # Show errors for first few failures or in verbose mode
                show_error = show_details or len(failed_records) < 5
                if convert_to_wfdb(temp_csv, csv_path.parent, fs=args.fs, verbose=show_error):
                    # WFDB creates files: {temp_csv.stem}.hea and {temp_csv.stem}.dat
                    temp_wfdb_base = temp_csv.parent / temp_csv.stem
                    temp_hea = temp_wfdb_base.with_suffix('.hea')
                    temp_dat = temp_wfdb_base.with_suffix('.dat')
                    target_hea = wfdb_base_path.with_suffix('.hea')
                    target_dat = wfdb_base_path.with_suffix('.dat')
                    
                    # Check if WFDB files were created
                    if temp_hea.exists() and temp_dat.exists():
                        # Remove target files if they exist (shouldn't, but be safe)
                        if target_hea.exists():
                            target_hea.unlink()
                        if target_dat.exists():
                            target_dat.unlink()
                        
                        # Rename to final names
                        try:
                            temp_hea.rename(target_hea)
                            temp_dat.rename(target_dat)
                            
                            # Update .hea file content to reference the renamed .dat file
                            # The .hea file contains the .dat filename in its header
                            if target_hea.exists():
                                with open(target_hea, 'r') as f:
                                    hea_content = f.read()
                                # Replace temp filename with final filename
                                hea_content = hea_content.replace(temp_wfdb_base.name, wfdb_base_path.name)
                                with open(target_hea, 'w') as f:
                                    f.write(hea_content)
                                    
                        except Exception as rename_error:
                            if show_error:
                                print(f"[ERROR] Failed to rename WFDB files for {csv_path}: {rename_error}")
                            failed_records.append((csv_path, pathology_class, f"Rename failed: {str(rename_error)}"))
                            continue
                        
                        # Verify files exist with correct names
                        if verify_wfdb_files(wfdb_base_path):
                            wfdb_base_path_abs = wfdb_base_path.resolve()
                            successful_records.append((csv_path, pathology_class, wfdb_base_path_abs))
                        else:
                            failed_records.append((csv_path, pathology_class, "WFDB verification failed after rename"))
                    else:
                        # Files weren't created - check what exists
                        hea_exists = temp_hea.exists()
                        dat_exists = temp_dat.exists()
                        error_msg = f"WFDB files not created (hea: {hea_exists}, dat: {dat_exists})"
                        if show_error:
                            # List files in directory to debug
                            dir_files = list(csv_path.parent.glob(f"{temp_csv.stem}*"))
                            print(f"[DEBUG] Files in dir matching '{temp_csv.stem}*': {[f.name for f in dir_files]}")
                            # Also check for files with original stem
                            orig_files = list(csv_path.parent.glob(f"{csv_path.stem}*"))
                            print(f"[DEBUG] Files in dir matching '{csv_path.stem}*': {[f.name for f in orig_files]}")
                        failed_records.append((csv_path, pathology_class, error_msg))
                else:
                    failed_records.append((csv_path, pathology_class, "WFDB conversion failed"))
                    
            except Exception as e:
                failed_records.append((csv_path, pathology_class, f"Error: {str(e)}"))
            
            finally:
                # Clean up temporary CSV file
                if temp_csv.exists():
                    temp_csv.unlink()
                
                # Clean up any remaining temp WFDB files
                temp_wfdb_base = temp_csv.parent / temp_csv.stem
                temp_hea = temp_wfdb_base.with_suffix('.hea')
                temp_dat = temp_wfdb_base.with_suffix('.dat')
                if temp_hea.exists():
                    temp_hea.unlink()
                if temp_dat.exists():
                    temp_dat.unlink()
    else:
        # Skip conversion, assume WFDB files already exist
        print("[INFO] Skipping WFDB conversion (--skip-conversion flag set)")
        for csv_path, pathology_class in files_with_labels:
            wfdb_base_path = csv_path.parent / csv_path.stem
            if verify_wfdb_files(wfdb_base_path):
                # Store absolute path for portability
                wfdb_base_path_abs = wfdb_base_path.resolve()
                successful_records.append((csv_path, pathology_class, wfdb_base_path_abs))
            else:
                failed_records.append((csv_path, pathology_class, "WFDB files not found"))
    
    # Print statistics
    print(f"\nConversion statistics:")
    print(f"  Successful: {len(successful_records)}")
    print(f"  Failed: {len(failed_records)}")
    
    if failed_records:
        print(f"\nFailed records (first 10):")
        for csv_path, pathology_class, reason in failed_records[:10]:
            print(f"  {csv_path}: {reason}")
        if len(failed_records) > 10:
            print(f"  ... and {len(failed_records) - 10} more")
    
    # Step 3: Create manifest
    print("\n" + "=" * 80)
    print("Step 3: Creating manifest file")
    print("=" * 80)
    
    if not successful_records:
        print("[ERROR] No successful conversions! Cannot create manifest.")
        sys.exit(1)
    
    create_manifest(
        successful_records,
        manifest_path,
        fs=args.fs,
        units=UNITS
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Manifest file: {manifest_path}")
    print(f"Sampling rate: {args.fs} Hz")
    print(f"Units: {UNITS}")
    print(f"Lead order: {', '.join(LEAD_ORDER)}")
    print(f"Total records: {len(successful_records)}")
    print(f"Pathology distribution:")
    pathology_counts = {}
    for _, pathology_class, _ in successful_records:
        pathology_name = [k for k, v in PATHOLOGY_KEYWORDS.items() if v == pathology_class][0]
        pathology_counts[pathology_name] = pathology_counts.get(pathology_name, 0) + 1
    for pathology, count in sorted(pathology_counts.items()):
        print(f"  {pathology}: {count}")
    
    print("\n" + "=" * 80)
    print("Dataset preparation complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Review the manifest: {manifest_path}")
    print(f"2. Use the manifest in your training script:")
    print(f"   df = pd.read_csv(r'{manifest_path}')")
    print(f"   # Extract labels: df[['label_0', 'label_1', ..., 'label_7']].values")
    print(f"   # Extract WFDB paths: df['wfdb_path'].values")


if __name__ == "__main__":
    main()

