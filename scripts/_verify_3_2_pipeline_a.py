"""Integrity-check script for Pipeline A outputs (Section 3.2).

Verifies the two cross_domain_4c_pipelineA.json files produced by
analysis/phase_b2_infarct_decoding.py against the pre-registered contract:

  - Required configs are present (4 in baseline; 2 in _inlp).
  - Each config exposes Z + ecg_features legs.
  - Each leg has best_C, cv_scores_per_C, in_domain_4c, cross_domain_4c,
    cross_domain_2c with the expected keys and value ranges.
  - n_per_class_truth + n_per_class_pred sum to n_total and confusion-matrix
    rows sum to support[c].
  - Bootstrap CIs are ordered low <= mean <= high.
  - Permutation p-values are in (0, 1].

Prints a green PASS/FAIL summary and exits non-zero on any failure.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PATHS = {
    "baseline": REPO / "outputs" / "phase_b2" / "cross_domain_4c_pipelineA.json",
    "inlp": REPO / "outputs" / "phase_b2_inlp" / "cross_domain_4c_pipelineA.json",
}
EXPECTED_CONFIGS = {
    "baseline": {"exp5_3class", "exp6_3class", "exp7_baseline", "exp7_ccmmd"},
    "inlp": {"exp7_baseline", "exp7_ccmmd"},
}
SOURCES = ["Z", "ecg_features"]
TERRITORIES_4C = ["Anteroseptal", "Anterolateral", "Inferior", "Inferolateral"]
TERRITORIES_2C = ["Anterior", "Inferior"]
EXPECTED_PTBXL_4C = {"Anteroseptal": 168, "Anterolateral": 42, "Inferior": 196, "Inferolateral": 32}
EXPECTED_PTBXL_2C = {"Anterior": 210, "Inferior": 228}

# MedalCare territory counts depend on WHICH label rule the artifact was built
# under, so they cannot be a single global constant.
#
#   phi   -- territory_4c derived from isch[0].phi (the "D1 fix",
#            scripts/build_medalcare_isch_targets.py:120-143). Current rule.
#   folder-- territory_4c copied from the MedalCare-XL folder name. Superseded.
#
# The two disagree on exactly one bucket: MedalCare-XL's own parameter files put
# LCX_0.3_post in the positive-phi wedge, the same wedge as LCX_0.3_ant. Verified
# 2026-08-11 by reading isch[0].phi out of the raw
# WP2_largeDataset_ParameterFiles/mi/LCX_0.3_post/*/run_S62/*Ventricular*.txt
# (+2.03, +2.27, +2.64 rad). So 450 train / 100 val / 100 test rows move
# Inferolateral -> Anterolateral. phi wins: the folder name contradicts the
# geometry the simulator was actually given.
MEDAL_4C = {
    "phi":    {"train": {"Anteroseptal": 1799, "Anterolateral": 1300, "Inferior": 1798, "Inferolateral": 450},
               "test":  {"Anteroseptal": 400, "Anterolateral": 300, "Inferior": 400, "Inferolateral": 100}},
    "folder": {"train": {"Anteroseptal": 1799, "Anterolateral": 850, "Inferior": 1798, "Inferolateral": 900},
               "test":  {"Anteroseptal": 400, "Anterolateral": 200, "Inferior": 400, "Inferolateral": 200}},
}

# Per-artifact contract: which label rule it was built under, and how many
# permutations it ran. `inlp` is a LEGACY artifact -- generated before both the
# D1 fix and the 10000-permutation upgrade, and from pre-leadfix encoders. It is
# checked for internal consistency only; do not cite numbers out of it.
ARTIFACT = {
    "baseline": {"label_rule": "phi",    "n_permutation": 10000, "legacy": False},
    "inlp":     {"label_rule": "folder", "n_permutation": 200,   "legacy": True},
}


def check_leg(leg: dict, name: str, eval_key: str, labels: list[str]) -> list[str]:
    issues: list[str] = []
    if eval_key not in leg:
        return [f"{name}: missing '{eval_key}'"]
    ev = leg[eval_key]
    for key in ["macro_f1", "macro_f1_ci95", "permutation_p_macro_f1",
                "balanced_accuracy", "balanced_accuracy_ci95", "permutation_p_balanced_accuracy",
                "per_class", "confusion_matrix", "labels", "n_total",
                "n_per_class_truth", "n_per_class_pred"]:
        if key not in ev:
            issues.append(f"{name}/{eval_key}: missing key '{key}'")
    if issues:
        return issues
    n = ev["n_total"]
    if sum(ev["n_per_class_truth"].values()) != n:
        issues.append(f"{name}/{eval_key}: n_per_class_truth sum != n_total ({sum(ev['n_per_class_truth'].values())} vs {n})")
    if sum(ev["n_per_class_pred"].values()) != n:
        issues.append(f"{name}/{eval_key}: n_per_class_pred sum != n_total ({sum(ev['n_per_class_pred'].values())} vs {n})")
    f1, lo, hi = ev["macro_f1"], ev["macro_f1_ci95"][0], ev["macro_f1_ci95"][1]
    if not (0.0 <= lo <= f1 <= hi <= 1.0):
        issues.append(f"{name}/{eval_key}: invalid macro_f1 CI ordering: lo={lo} f1={f1} hi={hi}")
    p = ev["permutation_p_macro_f1"]
    if not (0.0 < p <= 1.0):
        issues.append(f"{name}/{eval_key}: permutation_p_macro_f1 out of range ({p})")
    cm = ev["confusion_matrix"]
    if len(cm) != len(labels) or any(len(r) != len(labels) for r in cm):
        issues.append(f"{name}/{eval_key}: CM shape != {len(labels)}x{len(labels)}")
    else:
        for i, lab in enumerate(labels):
            row_sum = sum(cm[i])
            sup = ev["per_class"][lab]["support"]
            if row_sum != sup:
                issues.append(f"{name}/{eval_key}: CM row '{lab}' sum={row_sum} != support={sup}")
    if ev["labels"] != labels:
        issues.append(f"{name}/{eval_key}: labels mismatch: got {ev['labels']} expected {labels}")
    return issues


def check_file(tag: str, path: Path) -> list[str]:
    issues: list[str] = []
    if not path.exists():
        return [f"{tag}: missing file {path}"]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "metadata" not in data or "results" not in data:
        return [f"{tag}: missing 'metadata' or 'results' top-level keys"]
    meta = data["metadata"]

    if set(meta["configs"]) != EXPECTED_CONFIGS[tag]:
        issues.append(f"{tag}: configs mismatch {set(meta['configs'])} != {EXPECTED_CONFIGS[tag]}")
    if meta["territories_4c"] != TERRITORIES_4C:
        issues.append(f"{tag}: territories_4c mismatch")
    if meta["territories_2c"] != TERRITORIES_2C:
        issues.append(f"{tag}: territories_2c mismatch")
    if meta["n_per_class_truth_ptbxl_4c"] != EXPECTED_PTBXL_4C:
        issues.append(f"{tag}: ptbxl 4c truth counts != audit {meta['n_per_class_truth_ptbxl_4c']} vs {EXPECTED_PTBXL_4C}")
    if meta["n_per_class_truth_ptbxl_2c"] != EXPECTED_PTBXL_2C:
        issues.append(f"{tag}: ptbxl 2c truth counts != audit {meta['n_per_class_truth_ptbxl_2c']} vs {EXPECTED_PTBXL_2C}")
    spec = ARTIFACT[tag]
    exp_medal = MEDAL_4C[spec["label_rule"]]
    if meta["n_train_medalcare_4c"] != exp_medal["train"]:
        issues.append(f"{tag}: medalcare train 4c counts != {spec['label_rule']}-rule "
                      f"{meta['n_train_medalcare_4c']} vs {exp_medal['train']}")
    if meta["n_test_medalcare_4c"] != exp_medal["test"]:
        issues.append(f"{tag}: medalcare test 4c counts != {spec['label_rule']}-rule "
                      f"{meta['n_test_medalcare_4c']} vs {exp_medal['test']}")
    if meta["n_bootstrap"] != 1000:
        issues.append(f"{tag}: n_bootstrap != 1000 ({meta['n_bootstrap']})")
    if meta["n_permutation_macro_f1"] != spec["n_permutation"]:
        issues.append(f"{tag}: n_permutation_macro_f1 != {spec['n_permutation']} "
                      f"({meta['n_permutation_macro_f1']})")
    if spec["legacy"]:
        print(f"  [LEGACY] {tag}: built under the {spec['label_rule']}-derived label "
              f"rule with {spec['n_permutation']} permutations -- superseded, "
              f"internal consistency only, do not cite.")

    results = data["results"]
    for cfg in EXPECTED_CONFIGS[tag]:
        if cfg not in results:
            issues.append(f"{tag}: missing config '{cfg}' in results")
            continue
        for src in SOURCES:
            if src not in results[cfg]:
                issues.append(f"{tag}/{cfg}: missing source '{src}'")
                continue
            leg = results[cfg][src]
            for k in ("best_C", "cv_scores_per_C", "in_domain_4c", "cross_domain_4c", "cross_domain_2c"):
                if k not in leg:
                    issues.append(f"{tag}/{cfg}/{src}: missing '{k}'")
            if "in_domain_4c" in leg:
                issues.extend(check_leg(leg, f"{tag}/{cfg}/{src}", "in_domain_4c", TERRITORIES_4C))
            if "cross_domain_4c" in leg:
                issues.extend(check_leg(leg, f"{tag}/{cfg}/{src}", "cross_domain_4c", TERRITORIES_4C))
            if "cross_domain_2c" in leg:
                issues.extend(check_leg(leg, f"{tag}/{cfg}/{src}", "cross_domain_2c", TERRITORIES_2C))
    return issues


def main() -> int:
    all_issues: list[str] = []
    summary_lines: list[str] = []
    for tag, path in PATHS.items():
        issues = check_file(tag, path)
        all_issues.extend(issues)
        if not issues:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for cfg in sorted(data["results"]):
                pa_z = data["results"][cfg]["Z"]
                pa_f = data["results"][cfg]["ecg_features"]
                summary_lines.append(
                    f"  {tag}/{cfg:15s}  "
                    f"inD_Z={pa_z['in_domain_4c']['macro_f1']:.3f}  "
                    f"CD_Z={pa_z['cross_domain_4c']['macro_f1']:.3f} "
                    f"(p={pa_z['cross_domain_4c']['permutation_p_macro_f1']:.4f})  "
                    f"CD2_Z={pa_z['cross_domain_2c']['macro_f1']:.3f}  | "
                    f"inD_NK2={pa_f['in_domain_4c']['macro_f1']:.3f}  "
                    f"CD_NK2={pa_f['cross_domain_4c']['macro_f1']:.3f}"
                )

    print("\n".join(summary_lines))
    print()
    if all_issues:
        print(f"FAIL  -- {len(all_issues)} issue(s) found:")
        for line in all_issues:
            print(f"  - {line}")
        return 1
    print(f"PASS  -- all integrity checks passed for {len(PATHS)} files / "
          f"{sum(len(EXPECTED_CONFIGS[t]) for t in PATHS)} configs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
