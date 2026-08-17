"""Build PTB-XL MI subclass / anatomical-territory labels for B2-CD.

Reads ``ptbxl_database.csv`` and ``scp_statements.csv``, filters to the
official test fold (``strat_fold == 10`` -- matching the order produced by
``PTBXLDataset`` in ``scripts/datasets.py``), and derives:

- ``mi_codes``      : list of MI-class SCP codes present (e.g. ['ASMI', 'AMI'])
- ``mi_subclasses`` : list of corresponding ``diagnostic_subclass`` entries
                      ({AMI, IMI, LMI, PMI})
- ``territory_set`` : set drawn from {Anterior, Inferior, Lateral, Posterior}
- ``territory``     : single label in {Anterior, Inferior, Lateral} if exactly
                      one such territory is hit and Posterior is absent;
                      otherwise empty string (these rows are excluded from the
                      primary B2-CD comparison and used only for sensitivity).
- ``mi_present``    : 0/1, any MI-class diagnostic code present at confidence
                      >= ``--prob-threshold`` (default 0.0, i.e. listed)
- ``mi_strong``     : 0/1, mi_present AND best MI code has confidence >= 80
                      (used for sensitivity analyses)

The output CSV preserves the ``ecg_id`` ordering of the test fold so it can be
joined positionally to PTB-XL latent files exported by
``scripts/export_latents.py``.

Subclass -> anatomical territory mapping (matches MedalCare-XL coronary
territories):

- AMI (anterior MI / anteroseptal / anterolateral)  -> Anterior   (≈ LAD)
- IMI (inferior / inferolateral / inferoposterior)  -> Inferior   (≈ RCA)
- LMI (lateral MI)                                  -> Lateral    (≈ LCX)
- PMI (posterior MI)                                -> Posterior  (no clean
                                                       MedalCare equivalent;
                                                       excluded from primary
                                                       B2-CD comparison)

Output: ``data/ptbxl_mi_subclass.csv``.

Usage::

    python scripts/build_ptbxl_mi_subclass.py
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_PTBXL_ROOT = (
    REPO_ROOT
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
DEFAULT_OUTPUT = REPO_ROOT / "data" / "ptbxl_mi_subclass.csv"
DEFAULT_AUDIT_LATENTS = REPO_ROOT / "outputs" / "latents" / "exp7_ptbxl" / "latents.npz"

# Anatomical mapping: subclass -> territory (legacy 3-class scheme).
SUBCLASS_TO_TERRITORY: Dict[str, str] = {
    "AMI": "Anterior",
    "IMI": "Inferior",
    "LMI": "Lateral",
    "PMI": "Posterior",
}
PRIMARY_TERRITORIES = ("Anterior", "Inferior", "Lateral")  # MedalCare-aligned

# ---------------------------------------------------------------------------
# 4-class refined anatomical scheme (added 2026-05-13 for Track 3 B2-CD redux).
#
# Maps the 14 raw PTB-XL MI scp_codes into four anatomical compartments that
# match the MedalCare-XL folder structure (LAD_*, LCX_*_ant, LCX_*_post, RCA_*).
# This finally uses the ASMI/ALMI/ILMI/IPLMI distinctions that the legacy
# 3-class scheme was collapsing.
#
# Compartment groups (by raw SCP code):
#   ANTERIOR_PURE  : AMI, ASMI, INJAS              -> pure LAD territory
#   ANTEROLATERAL  : ALMI, INJAL                   -> LCX anterior-lateral
#   INFERIOR_PURE  : IMI, INJIN                    -> pure RCA territory
#   INFEROLATERAL  : ILMI, INJIL, IPLMI            -> LCX posterior / RCA distal
#   LATERAL_PURE   : LMI, INJLA                    -> wall identifier (sub-modifies the above)
#   EXCL_POSTERIOR : PMI, IPMI                     -> pure posterior, no MedalCare equivalent
#
# Decision rules (in order):
#   1.  has_excl_posterior                                -> ""           (excluded)
#   2.  any anterior compartment AND any inferior compartment  -> ""      (multi-territory, excluded)
#   3.  anterior side:
#         has_anterolateral OR has_lateral_pure            -> Anterolateral
#         else                                             -> Anteroseptal
#   4.  inferior side:
#         has_inferolateral OR has_lateral_pure            -> Inferolateral
#         else                                             -> Inferior
#   5.  pure lateral only (no anterior, no inferior)       -> ""           (ambiguous)
#
# 2-class backup: Anteroseptal + Anterolateral -> "Anterior"; Inferior + Inferolateral -> "Inferior".
# ---------------------------------------------------------------------------
ANTERIOR_PURE_CODES: Tuple[str, ...] = ("AMI", "ASMI", "INJAS")
ANTEROLATERAL_CODES: Tuple[str, ...] = ("ALMI", "INJAL")
INFERIOR_PURE_CODES: Tuple[str, ...] = ("IMI", "INJIN")
INFEROLATERAL_CODES: Tuple[str, ...] = ("ILMI", "INJIL", "IPLMI")
LATERAL_PURE_CODES: Tuple[str, ...] = ("LMI", "INJLA")
EXCLUDED_POSTERIOR_CODES: Tuple[str, ...] = ("PMI", "IPMI")  # no MedalCare equivalent

TERRITORIES_4C: Tuple[str, ...] = (
    "Anteroseptal",
    "Anterolateral",
    "Inferior",
    "Inferolateral",
)
TERRITORIES_2C: Tuple[str, ...] = ("Anterior", "Inferior")

TERRITORY_4C_TO_2C: Dict[str, str] = {
    "Anteroseptal": "Anterior",
    "Anterolateral": "Anterior",
    "Inferior": "Inferior",
    "Inferolateral": "Inferior",
}


def derive_territory_4c(raw_codes_str: str) -> str:
    """Map a "|"-joined raw mi_codes string to a 4-class anatomical territory.

    Returns the bucket name from TERRITORIES_4C, or "" when the row is
    excluded (multi-territory, pure-posterior, or ambiguous pure-lateral).
    """
    if not raw_codes_str:
        return ""
    codes = set(raw_codes_str.split("|"))

    has_anterior_pure = bool(codes & set(ANTERIOR_PURE_CODES))
    has_anterolateral = bool(codes & set(ANTEROLATERAL_CODES))
    has_inferior_pure = bool(codes & set(INFERIOR_PURE_CODES))
    has_inferolateral = bool(codes & set(INFEROLATERAL_CODES))
    has_lateral_pure = bool(codes & set(LATERAL_PURE_CODES))
    has_excl_posterior = bool(codes & set(EXCLUDED_POSTERIOR_CODES))

    if has_excl_posterior:
        return ""

    is_anterior = has_anterior_pure or has_anterolateral
    is_inferior = has_inferior_pure or has_inferolateral
    if is_anterior and is_inferior:
        return ""  # multi-territory

    if is_anterior:
        if has_anterolateral or has_lateral_pure:
            return "Anterolateral"
        return "Anteroseptal"
    if is_inferior:
        if has_inferolateral or has_lateral_pure:
            return "Inferolateral"
        return "Inferior"

    # only lateral, or no MI compartment hit
    return ""


def derive_territory_2c(territory_4c: str) -> str:
    """Collapse the 4-class scheme to a 2-class Anterior-vs-Inferior backup."""
    return TERRITORY_4C_TO_2C.get(territory_4c, "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_scp_codes(raw: str) -> Dict[str, float]:
    """Parse the stringified scp_codes dict from PTB-XL into a {code: float} dict."""
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): float(v) for k, v in parsed.items()}


def build_mi_code_lookup(
    scp_statements: pd.DataFrame,
) -> Dict[str, str]:
    """Return ``{scp_code: diagnostic_subclass}`` for codes whose
    ``diagnostic_class == 'MI'``."""
    if "scp_code" not in scp_statements.columns:
        unnamed = [c for c in scp_statements.columns if not c or c.startswith("Unnamed")]
        if unnamed:
            scp_statements = scp_statements.rename(columns={unnamed[0]: "scp_code"})
    if "scp_code" not in scp_statements.columns:
        raise ValueError("scp_statements.csv missing 'scp_code' column.")
    mi = scp_statements[scp_statements["diagnostic_class"] == "MI"]
    return {
        str(row["scp_code"]): str(row["diagnostic_subclass"])
        for _, row in mi.iterrows()
        if isinstance(row.get("diagnostic_subclass"), str)
    }


def derive_row_labels(
    scp_codes: Dict[str, float],
    mi_subclass_lookup: Dict[str, str],
    prob_threshold: float = 0.0,
    strong_threshold: float = 80.0,
) -> Dict[str, object]:
    """Derive MI-related metadata for one PTB-XL record."""
    mi_codes_present: List[Tuple[str, float, str]] = []
    for code, prob in scp_codes.items():
        if prob < prob_threshold:
            continue
        if code in mi_subclass_lookup:
            sub = mi_subclass_lookup[code]
            mi_codes_present.append((code, prob, sub))

    mi_codes = [c for c, _, _ in mi_codes_present]
    subclasses = sorted({s for _, _, s in mi_codes_present})
    territory_set = sorted(
        {SUBCLASS_TO_TERRITORY[s] for s in subclasses if s in SUBCLASS_TO_TERRITORY}
    )
    primary_hits = [t for t in territory_set if t in PRIMARY_TERRITORIES]
    territory = (
        primary_hits[0]
        if (len(primary_hits) == 1 and "Posterior" not in territory_set)
        else ""
    )
    mi_present = int(len(mi_codes_present) > 0)
    best_prob = max((p for _, p, _ in mi_codes_present), default=0.0)
    mi_strong = int(mi_present and best_prob >= strong_threshold)
    raw_codes_joined = "|".join(mi_codes)
    territory_4c = derive_territory_4c(raw_codes_joined)
    territory_2c = derive_territory_2c(territory_4c)
    return {
        "mi_codes": raw_codes_joined,
        "mi_subclasses": "|".join(subclasses),
        "territory_set": "|".join(territory_set),
        "territory": territory,
        "territory_4c": territory_4c,
        "territory_2c": territory_2c,
        "mi_present": mi_present,
        "mi_strong": mi_strong,
        "best_mi_prob": float(best_prob),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ptbxl-root", type=Path, default=DEFAULT_PTBXL_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--folds", type=str, default="10",
        help="Comma-separated stratification folds to keep "
             "(default '10' = official test split).",
    )
    parser.add_argument(
        "--prob-threshold", type=float, default=0.0,
        help="Minimum SCP probability to count a code as 'present' "
             "(default 0.0 = any non-zero / listed code).",
    )
    parser.add_argument(
        "--strong-threshold", type=float, default=80.0,
        help="SCP probability threshold for the auxiliary 'mi_strong' flag "
             "(default 80).",
    )
    parser.add_argument(
        "--no-audit", action="store_true",
        help="Skip alignment audit against the existing PTB-XL latent export.",
    )
    args = parser.parse_args()

    db_path = args.ptbxl_root / "ptbxl_database.csv"
    scp_path = args.ptbxl_root / "scp_statements.csv"
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    if not scp_path.is_file():
        raise FileNotFoundError(scp_path)

    print(f"Reading {db_path}")
    db = pd.read_csv(db_path)
    print(f"Reading {scp_path}")
    scp = pd.read_csv(scp_path)
    mi_lookup = build_mi_code_lookup(scp)
    print(f"MI-class SCP codes: {len(mi_lookup)} (subclass values: {sorted(set(mi_lookup.values()))})")

    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    print(f"Filtering folds: {folds}")
    sub = db[db["strat_fold"].isin(folds)].reset_index(drop=True)
    print(f"  -> {len(sub)} rows")

    records: List[Dict[str, object]] = []
    for i, row in sub.iterrows():
        scp_codes = parse_scp_codes(str(row.get("scp_codes", "")))
        labels = derive_row_labels(
            scp_codes,
            mi_lookup,
            prob_threshold=args.prob_threshold,
            strong_threshold=args.strong_threshold,
        )
        records.append({
            "row_idx": int(i),
            "ecg_id": int(row["ecg_id"]),
            "patient_id": int(row["patient_id"]) if not pd.isna(row.get("patient_id")) else -1,
            "age": float(row["age"]) if not pd.isna(row.get("age")) else float("nan"),
            "sex": int(row["sex"]) if not pd.isna(row.get("sex")) else -1,
            "strat_fold": int(row["strat_fold"]),
            "scp_codes": str(row.get("scp_codes", "")),
            **labels,
        })

    out_df = pd.DataFrame.from_records(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}  ({args.out.stat().st_size / 1024:.1f} KB)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_total = len(out_df)
    n_mi = int(out_df["mi_present"].sum())
    n_mi_strong = int(out_df["mi_strong"].sum())
    territory_counts = out_df.loc[out_df["mi_present"] == 1, "territory"].value_counts(dropna=False).to_dict()
    subclass_counts = (
        out_df.loc[out_df["mi_present"] == 1, "mi_subclasses"]
        .value_counts(dropna=False)
        .head(15)
        .to_dict()
    )
    n_clean_primary = int(out_df["territory"].isin(PRIMARY_TERRITORIES).sum())
    n_clean_4c = int(out_df["territory_4c"].isin(TERRITORIES_4C).sum())
    n_clean_2c = int(out_df["territory_2c"].isin(TERRITORIES_2C).sum())
    counts_4c = {t: int((out_df["territory_4c"] == t).sum()) for t in TERRITORIES_4C}
    counts_2c = {t: int((out_df["territory_2c"] == t).sum()) for t in TERRITORIES_2C}
    print("\n=== Summary (test fold) ===")
    print(f"  rows                : {n_total}")
    print(f"  mi_present          : {n_mi}  ({100*n_mi/n_total:.1f}%)")
    print(f"  mi_strong (>= {args.strong_threshold}) : {n_mi_strong}  ({100*n_mi_strong/n_total:.1f}%)")
    print(f"  single-territory MI (legacy 3-class): {n_clean_primary}  (Anterior/Inferior/Lateral)")
    print(f"  primary 4-class total              : {n_clean_4c}")
    print(f"  primary 2-class total              : {n_clean_2c}")
    print(f"  territory breakdown (legacy 3-class):")
    for t in PRIMARY_TERRITORIES:
        n_t = int((out_df["territory"] == t).sum())
        print(f"    {t:13s}: {n_t}")
    print(f"  territory_4c breakdown (refined 4-class):")
    for t in TERRITORIES_4C:
        print(f"    {t:13s}: {counts_4c[t]}")
    print(f"  territory_2c breakdown (Anterior-vs-Inferior backup):")
    for t in TERRITORIES_2C:
        print(f"    {t:13s}: {counts_2c[t]}")
    print(f"  Top 10 subclass combinations among MI rows:")
    for k, v in list(subclass_counts.items())[:10]:
        print(f"    {k!r:>30s} : {v}")

    summary = {
        "n_rows": n_total,
        "n_mi_present": n_mi,
        "n_mi_strong": n_mi_strong,
        "n_single_territory_primary": n_clean_primary,
        "n_primary_4c": n_clean_4c,
        "n_primary_2c": n_clean_2c,
        "territory_counts_among_mi": territory_counts,
        "territory_4c_counts": counts_4c,
        "territory_2c_counts": counts_2c,
        "top_subclass_combos": subclass_counts,
        "subclass_to_territory": SUBCLASS_TO_TERRITORY,
        "primary_territories": list(PRIMARY_TERRITORIES),
        "territories_4c": list(TERRITORIES_4C),
        "territories_2c": list(TERRITORIES_2C),
        "code_groups_4c": {
            "ANTERIOR_PURE_CODES": list(ANTERIOR_PURE_CODES),
            "ANTEROLATERAL_CODES": list(ANTEROLATERAL_CODES),
            "INFERIOR_PURE_CODES": list(INFERIOR_PURE_CODES),
            "INFEROLATERAL_CODES": list(INFEROLATERAL_CODES),
            "LATERAL_PURE_CODES": list(LATERAL_PURE_CODES),
            "EXCLUDED_POSTERIOR_CODES": list(EXCLUDED_POSTERIOR_CODES),
        },
        "prob_threshold": args.prob_threshold,
        "strong_threshold": args.strong_threshold,
    }
    # Track --out, not a fixed name: building a second fold selection to a
    # different CSV used to silently clobber the first build's summary.
    summary_path = args.out.with_name(f"{args.out.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary -> {summary_path}")

    # ------------------------------------------------------------------
    # Optional alignment audit against PTB-XL latents
    # ------------------------------------------------------------------
    if not args.no_audit and DEFAULT_AUDIT_LATENTS.is_file():
        with np.load(DEFAULT_AUDIT_LATENTS, allow_pickle=True) as data:
            Y = data["Y"]
        print(f"\n[AUDIT] {DEFAULT_AUDIT_LATENTS.relative_to(REPO_ROOT)}: Y={Y.shape}")
        if Y.shape[0] != n_total:
            print(
                f"  [WARN] latent rows {Y.shape[0]} != subclass rows {n_total}; "
                "alignment audit FAILED -- order may not match"
            )
        else:
            # PTB-XL native superclass order: NORM, MI, STTC, HYP, CD.
            # Y[:, 1] is multi-hot for MI superclass.
            mi_super_count = int(Y[:, 1].sum())
            print(f"  Y[:, 1].sum() (MI superclass) = {mi_super_count}")
            print(f"  CSV mi_present count          = {n_mi}")
            agree = int((Y[:, 1].astype(int) == out_df["mi_present"].to_numpy()).sum())
            print(
                f"  agreement (MI superclass == mi_present): "
                f"{agree}/{n_total} ({100*agree/n_total:.1f}%)"
            )
            if agree < n_total:
                # The two definitions differ slightly because PTB-XL's MI
                # superclass also bubbles up some non-MI diagnostic_class entries
                # via the 'diagnostic_class' field; this is informational.
                print(
                    "  (small disagreements are expected because the latent "
                    "exporter aggregates SCP codes by `diagnostic_class == 'MI'` "
                    "while this CSV resolves at the `diagnostic_subclass` level "
                    "for the four canonical MI subclasses.)"
                )


if __name__ == "__main__":
    main()
