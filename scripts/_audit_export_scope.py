"""Two remaining questions before the audit can be closed.

1. The PTB-XL spatial-54 export has features for only 549 of 2198 rows. Is that
   exactly the union of the analysis subsets (a deliberate partial extraction,
   harmless), or an arbitrary truncation (which would mean some other analysis
   silently ran on imputed garbage)?
2. `--scaler-domain target_pool` builds its statistics from all 2198 rows, 75%
   of which are entirely imputed. Part 12 s12.4 already records that arm as
   corrupted. Confirm no *reported* headline number depends on it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TERR4 = ("Anteroseptal", "Anterolateral", "Inferior", "Inferolateral")

f = np.load(REPO / "data" / "ecg_features_spatial_ptbxl_test.npz", allow_pickle=True)
X = np.asarray(f["features"], dtype=float)
has = ~np.isnan(X).all(axis=1)
sub = pd.read_csv(REPO / "data" / "ptbxl_mi_subclass.csv")

idx4 = set(sub.loc[sub["territory_4c"].isin(TERR4), "row_idx"].tolist())
idx3 = set(sub.loc[sub["territory_4c"].notna(), "row_idx"].tolist()) \
    if "territory_4c" in sub else set()
idx_all_mi = set(sub["row_idx"].tolist())
present = set(np.where(has)[0].tolist())

print("1. PARTIAL EXPORT ACCOUNTING")
print(f"   rows with real features : {len(present)}")
print(f"   4c analysis subset      : {len(idx4)}   subset-of-present: "
      f"{idx4 <= present}")
print(f"   all MI-subclass rows    : {len(idx_all_mi)}  subset-of-present: "
      f"{idx_all_mi <= present}")
print(f"   present \\ MI-subclass   : {len(present - idx_all_mi)}")
print(f"   MI-subclass \\ present   : {len(idx_all_mi - present)}")
if idx4 <= present:
    print("   => every row the primary endpoint scores has real measurements.")
if present == idx_all_mi:
    print("   => the export is exactly the MI-subclass rows: deliberate partial "
          "extraction, not truncation.")

print()
print("2. WHICH SCALER DOMAIN PRODUCED THE REPORTED NUMBERS?")
for rel in ["outputs/phase_b2_exp8/cross_domain_4c_pipelineA.json",
            "outputs/phase_b2_exp8_spatial54/cross_domain_4c_pipelineA.json"]:
    fp = REPO / rel
    if not fp.exists():
        continue
    j = json.loads(fp.read_text(encoding="utf-8"))
    found = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if "scaler" in k.lower() and isinstance(v, str):
                    found.add(f"{k}={v}")
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(j)
    print(f"   {Path(rel).parent.name}: {sorted(found) or '(no scaler key recorded)'}")

for rel in ["outputs/phase_b2_exp8_poolscaler", "outputs/phase_b2_exp8_srcscaler",
            "outputs/phase_b2_exp8_tgtscaler"]:
    d = REPO / rel
    print(f"   {Path(rel).name}: {'present' if d.exists() else 'absent'}"
          + (f"  files={len(list(d.glob('*.json')))} json" if d.exists() else ""))
