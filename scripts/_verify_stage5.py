"""Stage 5 verification: every hygiene fix, checked against real data.

Run: python scripts/_verify_stage5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

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
    pathology_of,
)

MANIFEST = REPO_ROOT / "data" / "medalcare_filtered_manifest_dataset_split.csv"
fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


print("=" * 70)
print("Stage 5 verification")
print("=" * 70)

# --- 1. segment matching reproduces the old substring result on real data ----
print("\n1. medalcare_paths vs the old substring test")
df = pd.read_csv(MANIFEST)
paths = df["original_csv_path"].astype(str)
new = paths.apply(is_mi_path)
old = paths.str.replace("\\", "/", regex=False).str.lower().str.contains("/mi/")
check("is_mi_path agrees with old '/mi/' substring on the shipped manifest",
      bool((new == old).all()), f"n_mi={int(new.sum())}/{len(new)}")

# --- 2. ...but diverges where the old one was wrong -------------------------
print("\n2. the case the substring test got wrong")
trap = "D:/mi/experiments/MedalCare-XL/WP2_largeDataset_noise/sinus/test/run_s62/run_000001.csv"
old_says_mi = "/mi/" in trap.lower()
check("old substring test misclassifies a sinus record under a 'mi' ancestor dir",
      old_says_mi is True, "it returns True")
check("pathology_of returns 'sinus' for the same path",
      pathology_of(trap) == "sinus")
check("is_mi_path returns False for the same path", is_mi_path(trap) is False)

# --- 3. pathology coverage --------------------------------------------------
print("\n3. every manifest row resolves to exactly one pathology")
try:
    pathologies = paths.apply(pathology_of)
    counts = pathologies.value_counts().to_dict()
    check("all rows parsed", True, str(counts))
except ValueError as exc:
    check("all rows parsed", False, str(exc)[:120])

# --- 4. territory parsing round-trips --------------------------------------
print("\n4. territory parsing on all MI rows")
mi_paths = paths[new]
try:
    parsed = [parse_territory_from_path(p) for p in mi_paths]
    coronaries = sorted({c for c, _, _ in parsed})
    transmurals = sorted({t for _, _, t in parsed})
    subtypes = sorted({s for _, s, _ in parsed})
    check("all MI rows parse", len(parsed) == int(new.sum()),
          f"coronaries={coronaries} transmural={transmurals} lcx_subtypes={subtypes}")
except ValueError as exc:
    check("all MI rows parse", False, str(exc)[:160])

# --- 5. label schema assertion ---------------------------------------------
print("\n5. label schema assertion")
try:
    assert_label_schema(df.columns)
    check("shipped manifest passes", True)
except ValueError as exc:
    check("shipped manifest passes", False, str(exc)[:160])

bad = df.rename(columns={"label_3": "label_lbbb"})
try:
    assert_label_schema(bad.columns)
    check("renamed label column is rejected", False, "no exception raised")
except ValueError:
    check("renamed label column is rejected", True)

reordered = df[[c for c in df.columns if not c.startswith("label_")]
               + ["label_1", "label_0"] + [f"label_{i}" for i in range(2, 8)]]
try:
    assert_label_schema(reordered.columns)
    check("reordered label columns are rejected", False, "no exception raised")
except ValueError:
    check("reordered label columns are rejected", True)

# --- 6. run_id collision guard ---------------------------------------------
print("\n6. resolve_run_dir refuses to overwrite a completed run")
from scripts.finetune_multilabel import resolve_run_dir  # noqa: E402

existing = "exp7_baseline"  # has metrics.json
assert (REPO_ROOT / "outputs" / existing / "metrics.json").exists(), "fixture missing"
try:
    resolve_run_dir(existing, allow_overwrite=False)
    check("completed run_id rejected without --overwrite", False, "no SystemExit")
except SystemExit:
    check("completed run_id rejected without --overwrite", True)

rid, out = resolve_run_dir(existing, allow_overwrite=True)
check("--overwrite permits reuse", rid == existing and out.name == existing)

rid2, out2 = resolve_run_dir("exp8_does_not_exist_yet", allow_overwrite=False)
check("fresh run_id is allowed", rid2 == "exp8_does_not_exist_yet")

rid3, _ = resolve_run_dir(None, allow_overwrite=False)
check("None run_id falls back to a %Y%m%d_%H%M%S timestamp",
      len(rid3) == 15 and rid3[8] == "_" and rid3.replace("_", "").isdigit(),
      rid3)

# --- 7. the unified PTB-XL filter ------------------------------------------
print("\n7. one PTB-XL filter implementation, shared")
import scripts.finetune_bottleneck_multitask as mt  # noqa: E402
from scripts.finetune_multilabel import _filter_ptbxl_dataset, PTBXL_REMAP  # noqa: E402

check("multitask imports the shared filter (no local copy)",
      mt._filter_ptbxl_dataset is _filter_ptbxl_dataset)
check("filter keeps PTBXL_REMAP source columns (NORM,MI,CD)",
      sorted(PTBXL_REMAP.keys()) == [0, 1, 4], str(sorted(PTBXL_REMAP.keys())))

# --- 8. everything still imports -------------------------------------------
print("\n8. touched modules import cleanly")
for mod in ["scripts.finetune_multilabel", "scripts.finetune_bottleneck",
            "scripts.finetune_bottleneck_multitask", "scripts.export_latents",
            "scripts.build_medalcare_isch_targets",
            "scripts.extract_ecg_features_neurokit2",
            "scripts.medalcare_paths", "analysis.phase_b2_mi_stage_control"]:
    try:
        __import__(mod)
        check(mod, True)
    except Exception as exc:  # noqa: BLE001
        check(mod, False, f"{type(exc).__name__}: {exc}")

print("\n" + "=" * 70)
if fails:
    print(f"FAILED ({len(fails)}): {fails}")
    raise SystemExit(1)
print("Stage 5: all checks passed")
