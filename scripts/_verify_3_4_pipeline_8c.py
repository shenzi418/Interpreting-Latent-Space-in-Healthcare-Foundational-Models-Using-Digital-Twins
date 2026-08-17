"""Integrity-check script for the in-domain 8-class audit (Section 3.4).

Verifies the two in_domain_8c.json files produced by
analysis/phase_b2_infarct_decoding.py against the pre-registered contract:

  - Required configs are present (4 in baseline; 2 in _inlp).
  - Each config exposes Z + ecg_features legs.
  - Each leg has best_C, cv_scores_per_C, in_domain_8c, in_domain_4c_anatomy,
    in_domain_2c_transmurality with the expected keys.
  - 4c anatomy collapse uses TERRITORIES_4C labels; 2c uses ["0.3", "1.0"].
  - Confusion-matrix rows sum to per-class support; CI ordering correct;
    permutation p in (0, 1].
  - MedalCare 8c per-class counts match the audit (TRAIN 899/900/450/450/400/
    450/898/900 and TEST 200/200/100/100/100/100/200/200).
  - cm_8c_{config}.png exists on disk with reasonable size.

Prints a PASS/FAIL summary and exits non-zero on any failure.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PATHS = {
    "baseline": REPO / "outputs" / "phase_b2" / "in_domain_8c.json",
    "inlp":     REPO / "outputs" / "phase_b2_inlp" / "in_domain_8c.json",
}
PNG_DIRS = {
    "baseline": REPO / "outputs" / "phase_b2",
    "inlp":     REPO / "outputs" / "phase_b2_inlp",
}
EXPECTED_CONFIGS = {
    "baseline": {"exp5_3class", "exp6_3class", "exp7_baseline", "exp7_ccmmd"},
    "inlp": {"exp7_baseline", "exp7_ccmmd"},
}
SOURCES = ["Z", "ecg_features"]
TERRITORIES_8C = [
    "LAD_0.3", "LAD_1.0",
    "LCX_0.3_ant", "LCX_0.3_post",
    "LCX_1.0_ant", "LCX_1.0_post",
    "RCA_0.3", "RCA_1.0",
]
TERRITORIES_4C = ["Anteroseptal", "Anterolateral", "Inferior", "Inferolateral"]
TRANSMURALITY_LABELS = ["0.3", "1.0"]
EXPECTED_TRAIN_8C = {
    "LAD_0.3": 899, "LAD_1.0": 900,
    "LCX_0.3_ant": 450, "LCX_0.3_post": 450,
    "LCX_1.0_ant": 400, "LCX_1.0_post": 450,
    "RCA_0.3": 898, "RCA_1.0": 900,
}
EXPECTED_TEST_8C = {
    "LAD_0.3": 200, "LAD_1.0": 200,
    "LCX_0.3_ant": 100, "LCX_0.3_post": 100,
    "LCX_1.0_ant": 100, "LCX_1.0_post": 100,
    "RCA_0.3": 200, "RCA_1.0": 200,
}
EVAL_BLOCKS = [
    ("in_domain_8c",               TERRITORIES_8C),
    ("in_domain_4c_anatomy",       TERRITORIES_4C),
    ("in_domain_2c_transmurality", TRANSMURALITY_LABELS),
]


def check_eval(ev: dict, name: str, labels: list[str]) -> list[str]:
    issues: list[str] = []
    for key in ("macro_f1", "macro_f1_ci95", "permutation_p_macro_f1",
                "balanced_accuracy", "balanced_accuracy_ci95",
                "permutation_p_balanced_accuracy", "per_class",
                "confusion_matrix", "labels", "n_total",
                "n_per_class_truth", "n_per_class_pred"):
        if key not in ev:
            issues.append(f"{name}: missing key '{key}'")
    if issues:
        return issues
    n = ev["n_total"]
    if sum(ev["n_per_class_truth"].values()) != n:
        issues.append(f"{name}: n_per_class_truth sum != n_total")
    if sum(ev["n_per_class_pred"].values()) != n:
        issues.append(f"{name}: n_per_class_pred sum != n_total")
    f1, lo, hi = ev["macro_f1"], ev["macro_f1_ci95"][0], ev["macro_f1_ci95"][1]
    if not (0.0 <= lo <= f1 <= hi <= 1.0):
        issues.append(f"{name}: bad macro_f1 CI ordering: lo={lo} f1={f1} hi={hi}")
    p = ev["permutation_p_macro_f1"]
    if not (0.0 < p <= 1.0):
        issues.append(f"{name}: permutation_p_macro_f1 out of range ({p})")
    cm = ev["confusion_matrix"]
    if len(cm) != len(labels) or any(len(r) != len(labels) for r in cm):
        issues.append(f"{name}: CM shape != {len(labels)}x{len(labels)}")
    else:
        for i, lab in enumerate(labels):
            if sum(cm[i]) != ev["per_class"][lab]["support"]:
                issues.append(f"{name}: CM row '{lab}' sum != support")
    if ev["labels"] != labels:
        issues.append(f"{name}: labels mismatch")
    return issues


def check_file(tag: str, path: Path) -> list[str]:
    issues: list[str] = []
    if not path.exists():
        return [f"{tag}: missing file {path}"]
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data["metadata"]

    if set(meta["configs"]) != EXPECTED_CONFIGS[tag]:
        issues.append(f"{tag}: configs mismatch")
    if meta["territories_8c"] != TERRITORIES_8C:
        issues.append(f"{tag}: territories_8c mismatch")
    if meta["territories_4c"] != TERRITORIES_4C:
        issues.append(f"{tag}: territories_4c mismatch")
    if meta["transmurality_labels"] != TRANSMURALITY_LABELS:
        issues.append(f"{tag}: transmurality_labels mismatch")
    if meta["n_train_medalcare_8c"] != EXPECTED_TRAIN_8C:
        issues.append(f"{tag}: medalcare train 8c counts mismatch")
    if meta["n_test_medalcare_8c"] != EXPECTED_TEST_8C:
        issues.append(f"{tag}: medalcare test 8c counts mismatch")
    if meta["n_bootstrap"] != 1000:
        issues.append(f"{tag}: n_bootstrap != 1000")
    # Permutation budget was raised 200 -> 10000 after `inlp` was generated.
    exp_perm = 200 if tag == "inlp" else 10000
    if meta["n_permutation_macro_f1"] != exp_perm:
        issues.append(f"{tag}: n_permutation_macro_f1 != {exp_perm} "
                      f"({meta['n_permutation_macro_f1']})")

    for cfg in EXPECTED_CONFIGS[tag]:
        if cfg not in data["results"]:
            issues.append(f"{tag}: missing config '{cfg}'")
            continue
        cm_png = PNG_DIRS[tag] / f"cm_8c_{cfg}.png"
        if not cm_png.exists() or cm_png.stat().st_size < 20_000:
            issues.append(f"{tag}/{cfg}: cm_8c missing/too small: {cm_png}")
        for src in SOURCES:
            leg = data["results"][cfg].get(src)
            if leg is None:
                issues.append(f"{tag}/{cfg}: missing source '{src}'")
                continue
            for k in ("best_C", "cv_scores_per_C"):
                if k not in leg:
                    issues.append(f"{tag}/{cfg}/{src}: missing '{k}'")
            for block, labels in EVAL_BLOCKS:
                if block not in leg:
                    issues.append(f"{tag}/{cfg}/{src}: missing '{block}'")
                else:
                    issues.extend(check_eval(leg[block], f"{tag}/{cfg}/{src}/{block}", labels))
    return issues


def main() -> int:
    all_issues: list[str] = []
    summary_lines: list[str] = []
    for tag, path in PATHS.items():
        issues = check_file(tag, path)
        all_issues.extend(issues)
        if not issues:
            data = json.loads(path.read_text(encoding="utf-8"))
            for cfg in sorted(data["results"]):
                z = data["results"][cfg]["Z"]
                f = data["results"][cfg]["ecg_features"]
                summary_lines.append(
                    f"  {tag}/{cfg:15s}  "
                    f"Z 8c={z['in_domain_8c']['macro_f1']:.3f} "
                    f"(p={z['in_domain_8c']['permutation_p_macro_f1']:.4f})  "
                    f"4c={z['in_domain_4c_anatomy']['macro_f1']:.3f}  "
                    f"2c-trans={z['in_domain_2c_transmurality']['macro_f1']:.3f}  | "
                    f"NK2 8c={f['in_domain_8c']['macro_f1']:.3f}  "
                    f"4c={f['in_domain_4c_anatomy']['macro_f1']:.3f}  "
                    f"2c-trans={f['in_domain_2c_transmurality']['macro_f1']:.3f}"
                )

    print("\n".join(summary_lines))
    print()
    if all_issues:
        print(f"FAIL  -- {len(all_issues)} issue(s):")
        for line in all_issues:
            print(f"  - {line}")
        return 1
    print(f"PASS  -- all integrity checks passed for {len(PATHS)} files / "
          f"{sum(len(EXPECTED_CONFIGS[t]) for t in PATHS)} configs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
