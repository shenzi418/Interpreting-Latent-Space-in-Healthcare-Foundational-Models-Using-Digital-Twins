"""Audit raw MI sub-codes in PTB-XL test fold to decide territory granularity.

For the n=2198 PTB-XL test fold (strat_fold=10):
 - list every raw mi_code combination present
 - count rows per combination, per subclass set, per territory set
 - cross-check against ASMI/ALMI/ILMI/IPLMI/IPMI fine codes
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PTBXL_CSV = (
    REPO_ROOT
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    / "ptbxl_database.csv"
)
SCP_CSV = PTBXL_CSV.with_name("scp_statements.csv")
SUBCLASS_CSV = REPO_ROOT / "data" / "ptbxl_mi_subclass.csv"


def parse(raw: str) -> dict:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        d = ast.literal_eval(raw)
    except Exception:
        return {}
    return d if isinstance(d, dict) else {}


def main() -> None:
    print(f"[load] {PTBXL_CSV}")
    db = pd.read_csv(PTBXL_CSV)
    print(f"  total rows : {len(db)}")

    print(f"[load] {SCP_CSV}")
    scp = pd.read_csv(SCP_CSV)
    if "scp_code" not in scp.columns:
        scp = scp.rename(columns={scp.columns[0]: "scp_code"})
    mi_codes = scp[scp["diagnostic_class"] == "MI"]
    print(f"  MI-class SCP codes ({len(mi_codes)}):")
    for _, r in mi_codes.iterrows():
        print(
            f"    {r['scp_code']:6s}  subclass={r['diagnostic_subclass']:<5s}  "
            f"desc={r.get('description', '')[:90]}"
        )

    print("\n[load] fold 10 only (test fold matching exp7 latents)")
    fold10 = db[db["strat_fold"] == 10].reset_index(drop=True)
    print(f"  fold10 rows: {len(fold10)}")

    mi_lookup = {
        str(r["scp_code"]): str(r["diagnostic_subclass"])
        for _, r in mi_codes.iterrows()
        if isinstance(r.get("diagnostic_subclass"), str)
    }
    valid_codes = set(mi_lookup.keys())

    raw_combo_counter: Counter = Counter()
    subclass_combo_counter: Counter = Counter()
    code_freq: Counter = Counter()
    n_mi_present = 0

    for _, row in fold10.iterrows():
        codes_dict = parse(str(row.get("scp_codes", "")))
        present = sorted(
            c for c, p in codes_dict.items()
            if c in valid_codes and float(p) >= 0.0
        )
        if not present:
            continue
        n_mi_present += 1
        for c in present:
            code_freq[c] += 1
        raw_combo_counter[tuple(present)] += 1
        sub = tuple(sorted({mi_lookup[c] for c in present}))
        subclass_combo_counter[sub] += 1

    print(f"\n[fold10] rows with >=1 MI scp_code listed : {n_mi_present}")
    print("\n[fold10] per-MI-code frequency (single-code count):")
    for c, n in code_freq.most_common():
        print(f"    {c:6s}  n={n:4d}  subclass={mi_lookup[c]}")

    print("\n[fold10] top raw-code combinations:")
    for combo, n in raw_combo_counter.most_common(25):
        print(f"    n={n:4d}  {'+'.join(combo)}")

    print("\n[fold10] subclass combinations (already aggregated):")
    for combo, n in subclass_combo_counter.most_common(20):
        print(f"    n={n:4d}  {'+'.join(combo)}")

    print("\n[fold10] viability of finer territory schemes")
    fine_buckets = {
        "Anteroseptal (ASMI alone, no LMI/IMI)": 0,
        "AnteriorPure_AMI_only": 0,
        "Anterolateral (ALMI or AMI+LMI)": 0,
        "Inferior_pure (IMI alone)": 0,
        "InferoLateral (ILMI or IMI+LMI)": 0,
        "InferoPosterior (IPMI or IMI+PMI)": 0,
        "InferoPosterolateral (IPLMI or IMI+PMI+LMI)": 0,
        "PureLateral (LMI alone)": 0,
        "PurePosterior (PMI alone)": 0,
    }
    for combo, n in raw_combo_counter.items():
        codes = set(combo)
        subs = {mi_lookup[c] for c in codes}
        if "ASMI" in codes and not (subs & {"LMI", "IMI"}):
            fine_buckets["Anteroseptal (ASMI alone, no LMI/IMI)"] += n
        if codes == {"AMI"}:
            fine_buckets["AnteriorPure_AMI_only"] += n
        if "ALMI" in codes or ({"AMI", "LMI"} <= subs and "IMI" not in subs):
            fine_buckets["Anterolateral (ALMI or AMI+LMI)"] += n
        if codes == {"IMI"}:
            fine_buckets["Inferior_pure (IMI alone)"] += n
        if "ILMI" in codes or ({"IMI", "LMI"} <= subs and "AMI" not in subs):
            fine_buckets["InferoLateral (ILMI or IMI+LMI)"] += n
        if "IPMI" in codes or ({"IMI", "PMI"} <= subs and "LMI" not in subs):
            fine_buckets["InferoPosterior (IPMI or IMI+PMI)"] += n
        if "IPLMI" in codes or ({"IMI", "PMI", "LMI"} <= subs):
            fine_buckets["InferoPosterolateral (IPLMI or IMI+PMI+LMI)"] += n
        if codes == {"LMI"}:
            fine_buckets["PureLateral (LMI alone)"] += n
        if codes == {"PMI"}:
            fine_buckets["PurePosterior (PMI alone)"] += n

    for name, n in fine_buckets.items():
        print(f"    {name:55s}  n={n}")

    print("\n[done]")


if __name__ == "__main__":
    main()
