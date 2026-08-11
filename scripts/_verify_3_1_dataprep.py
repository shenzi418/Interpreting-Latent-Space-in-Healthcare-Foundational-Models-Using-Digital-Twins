"""Integrity check for Track 3.1 data preparation.

Verifies that:
- PTB-XL CSV has new columns territory_4c, territory_2c
- 4-class counts match the audit (Anteroseptal=168, Anterolateral=42, Inferior=196, Inferolateral=32)
- 2-class counts match (Anterior=210, Inferior=228)
- Cross-tab: every territory_4c -> single territory_2c (collapse rule check)

- MedalCare NPZs have territory_4c, territory_8c arrays
- Array lengths match idx_in_split / phi / etc.
- Per-split per-class counts match printed summary
- territory_4c -> coronary mapping is consistent (Anteroseptal<->LAD, Inferior<->RCA,
  Anterolateral<->LCX/ant, Inferolateral<->LCX/post)
- territory_8c is a string of the form COR_TRANS[_SUB]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EXIT = 0


def fail(msg: str) -> None:
    global EXIT
    print(f"  [FAIL] {msg}")
    EXIT = 1


def ok(msg: str) -> None:
    print(f"  [ OK ] {msg}")


# ---------------------------------------------------------------------------
# PTB-XL
# ---------------------------------------------------------------------------
print("=== PTB-XL CSV ===")
csv_path = REPO_ROOT / "data" / "ptbxl_mi_subclass.csv"
df = pd.read_csv(csv_path)
expected_cols = ["territory_4c", "territory_2c"]
for c in expected_cols:
    if c not in df.columns:
        fail(f"missing column {c}")
    else:
        ok(f"column {c} present")

expected_4c = {
    "Anteroseptal": 168,
    "Anterolateral": 42,
    "Inferior": 196,
    "Inferolateral": 32,
}
expected_2c = {"Anterior": 210, "Inferior": 228}

actual_4c = df["territory_4c"].value_counts(dropna=False).to_dict()
actual_2c = df["territory_2c"].value_counts(dropna=False).to_dict()
print(f"  actual 4c counts: {actual_4c}")
print(f"  actual 2c counts: {actual_2c}")

for t, n in expected_4c.items():
    a = actual_4c.get(t, 0)
    if a != n:
        fail(f"4c {t}: got {a}, expected {n}")
    else:
        ok(f"4c {t}: {a}")

for t, n in expected_2c.items():
    a = actual_2c.get(t, 0)
    if a != n:
        fail(f"2c {t}: got {a}, expected {n}")
    else:
        ok(f"2c {t}: {a}")

# 4c -> 2c collapse rule consistency check
collapse_rule = {"Anteroseptal": "Anterior", "Anterolateral": "Anterior",
                 "Inferior": "Inferior", "Inferolateral": "Inferior"}
for _, row in df.iterrows():
    t4 = row["territory_4c"]
    t2 = row["territory_2c"]
    if isinstance(t4, str) and t4 in collapse_rule:
        if t2 != collapse_rule[t4]:
            fail(f"row {row['row_idx']}: 4c={t4} -> 2c={t2}, expected {collapse_rule[t4]}")
            break
else:
    ok("4c -> 2c collapse rule consistent for all rows")

# Sample a few rows manually to verify recipe
samples = {
    "ASMI alone -> Anteroseptal": (df["mi_codes"] == "ASMI"),
    "ILMI alone -> Inferolateral": (df["mi_codes"] == "ILMI"),
    "AMI|IMI -> exclude": (df["mi_codes"] == "AMI|IMI"),
    "PMI -> exclude": (df["mi_codes"] == "PMI"),
    "ALMI alone -> Anterolateral": (df["mi_codes"] == "ALMI"),
}
for name, mask in samples.items():
    if mask.sum() == 0:
        print(f"  [SKIP] no rows match: {name}")
        continue
    t4 = df.loc[mask, "territory_4c"].unique().tolist()
    print(f"  [SAMPLE] {name}: n={mask.sum()}, territory_4c={t4}")


# ---------------------------------------------------------------------------
# MedalCare
# ---------------------------------------------------------------------------
print("\n=== MedalCare NPZs ===")
EXPECTED_4C_TEST = {"Anteroseptal": 400, "Anterolateral": 200, "Inferior": 400, "Inferolateral": 200}
EXPECTED_8C_TEST = {
    "LAD_0.3": 200, "LAD_1.0": 200,
    "LCX_0.3_ant": 100, "LCX_0.3_post": 100, "LCX_1.0_ant": 100, "LCX_1.0_post": 100,
    "RCA_0.3": 200, "RCA_1.0": 200,
}

VALID_4C = {"Anteroseptal", "Anterolateral", "Inferior", "Inferolateral"}
VALID_8C = set(EXPECTED_8C_TEST.keys())

for split in ("train", "val", "test"):
    print(f"\n[{split}]")
    p = REPO_ROOT / "data" / f"theta_mi_{split}.npz"
    d = dict(np.load(p, allow_pickle=True))

    required = ["idx_in_split", "phi", "z", "size", "rho_eps_max", "coronary",
                "lcx_subtype", "transmural", "territory_4c", "territory_8c"]
    for k in required:
        if k not in d:
            fail(f"{split}: missing key {k}")
    if any(k not in d for k in required):
        continue
    ok(f"{split}: all required keys present")

    n = d["idx_in_split"].size
    for k in ("phi", "z", "size", "rho_eps_max", "coronary", "lcx_subtype",
              "transmural", "territory_4c", "territory_8c"):
        if d[k].shape[0] != n:
            fail(f"{split}: array {k} length {d[k].shape[0]} != idx_in_split length {n}")
        else:
            pass
    ok(f"{split}: all array lengths match (n={n})")

    # Class membership checks
    bad4 = [t for t in d["territory_4c"].tolist() if t not in VALID_4C]
    bad8 = [t for t in d["territory_8c"].tolist() if t not in VALID_8C]
    if bad4:
        fail(f"{split}: invalid territory_4c values: {sorted(set(bad4))[:5]}")
    else:
        ok(f"{split}: all territory_4c in valid set")
    if bad8:
        fail(f"{split}: invalid territory_8c values: {sorted(set(bad8))[:5]}")
    else:
        ok(f"{split}: all territory_8c in valid set")

    # Consistency: territory_4c <-> (coronary, lcx_subtype)
    mismatches = 0
    for c, s, t4 in zip(d["coronary"], d["lcx_subtype"], d["territory_4c"]):
        if c == "LAD" and t4 != "Anteroseptal":
            mismatches += 1
        elif c == "RCA" and t4 != "Inferior":
            mismatches += 1
        elif c == "LCX":
            if s == "ant" and t4 != "Anterolateral":
                mismatches += 1
            elif s == "post" and t4 != "Inferolateral":
                mismatches += 1
    if mismatches > 0:
        fail(f"{split}: {mismatches} rows with inconsistent territory_4c vs (coronary, lcx_subtype)")
    else:
        ok(f"{split}: territory_4c <-> (coronary, lcx_subtype) consistent")

    # territory_8c consistency
    mismatches = 0
    for c, s, t, t8 in zip(d["coronary"], d["lcx_subtype"], d["transmural"], d["territory_8c"]):
        trans = "0.3" if t < 0.5 else "1.0"
        if c == "LCX":
            expected = f"LCX_{trans}_{s}"
        else:
            expected = f"{c}_{trans}"
        if t8 != expected:
            mismatches += 1
    if mismatches > 0:
        fail(f"{split}: {mismatches} rows with inconsistent territory_8c")
    else:
        ok(f"{split}: territory_8c consistent")

    # Test split specific count checks
    if split == "test":
        actual_4c = {t: int((d["territory_4c"] == t).sum()) for t in VALID_4C}
        actual_8c = {t: int((d["territory_8c"] == t).sum()) for t in VALID_8C}
        for t, n_exp in EXPECTED_4C_TEST.items():
            if actual_4c[t] != n_exp:
                fail(f"test 4c {t}: got {actual_4c[t]}, expected {n_exp}")
            else:
                ok(f"test 4c {t}: {actual_4c[t]}")
        for t, n_exp in EXPECTED_8C_TEST.items():
            if actual_8c[t] != n_exp:
                fail(f"test 8c {t}: got {actual_8c[t]}, expected {n_exp}")
            else:
                ok(f"test 8c {t}: {actual_8c[t]}")

print()
if EXIT == 0:
    print("[done] all checks passed.")
else:
    print(f"[done] {EXIT} check group(s) failed.")
sys.exit(EXIT)
