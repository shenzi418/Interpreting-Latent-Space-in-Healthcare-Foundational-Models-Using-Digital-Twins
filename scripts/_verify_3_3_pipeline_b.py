"""Integrity-check script for Pipeline B outputs (Section 3.3).

Verifies the two cross_domain_4c_pipelineB.json files produced by
analysis/phase_b2_infarct_decoding.py against the pre-registered contract:

  - Required configs are present (4 in baseline; 2 in _inlp).
  - Each config exposes Z + ecg_features legs.
  - Each leg has calibrator_name, calibrator_cv_scores, in_domain_calibrator_4c,
    in_domain_hardcoded_4c, cross_calibrator_{4c,2c}, cross_hardcoded_{4c,2c}.
  - For each evaluation: macro_f1 + CI + permutation_p + per_class + CM all
    present and self-consistent.
  - Hardcoded boundaries == [+0.0, +2.0] as pre-registered.
  - Calibrator_name in {tree_d4, logreg_l2, knn_10}.
  - Cross-domain Z phi-pred histograms exist on disk.

Prints a PASS/FAIL summary and exits non-zero on any failure.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PATHS = {
    "baseline": REPO / "outputs" / "phase_b2" / "cross_domain_4c_pipelineB.json",
    "inlp":     REPO / "outputs" / "phase_b2_inlp" / "cross_domain_4c_pipelineB.json",
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
ALLOWED_CALIBRATORS = {"tree_d4", "logreg_l2", "knn_10"}
TERRITORIES_4C = ["Anteroseptal", "Anterolateral", "Inferior", "Inferolateral"]
TERRITORIES_2C = ["Anterior", "Inferior"]
EXPECTED_PTBXL_4C = {"Anteroseptal": 168, "Anterolateral": 42, "Inferior": 196, "Inferolateral": 32}
EXPECTED_PTBXL_2C = {"Anterior": 210, "Inferior": 228}
EVAL_KEYS_4C = ["in_domain_calibrator_4c", "in_domain_hardcoded_4c",
                "cross_calibrator_4c", "cross_hardcoded_4c"]
EVAL_KEYS_2C = ["cross_calibrator_2c", "cross_hardcoded_2c"]


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
    if meta["territories_4c"] != TERRITORIES_4C:
        issues.append(f"{tag}: territories_4c mismatch")
    if meta["territories_2c"] != TERRITORIES_2C:
        issues.append(f"{tag}: territories_2c mismatch")
    if meta["phi_4c_outer_boundary_rad"] != 2.0:
        issues.append(f"{tag}: outer boundary != 2.0 ({meta['phi_4c_outer_boundary_rad']})")
    if meta["phi_4c_inner_boundary_rad"] != 0.0:
        issues.append(f"{tag}: inner boundary != 0.0 ({meta['phi_4c_inner_boundary_rad']})")
    if meta["n_per_class_truth_ptbxl_4c"] != EXPECTED_PTBXL_4C:
        issues.append(f"{tag}: ptbxl 4c truth counts != audit")
    if meta["n_per_class_truth_ptbxl_2c"] != EXPECTED_PTBXL_2C:
        issues.append(f"{tag}: ptbxl 2c truth counts != audit")
    if set(meta["calibrator_candidates"]) != ALLOWED_CALIBRATORS:
        issues.append(f"{tag}: calibrator candidates mismatch")
    if meta["n_bootstrap"] != 1000:
        issues.append(f"{tag}: n_bootstrap != 1000")
    # The permutation budget was raised 200 -> 10000 after the `inlp` artifact was
    # generated. Expect per-artifact, not globally: more permutations is a tighter
    # test, but silently accepting either would let a truncated run pass.
    exp_perm = 200 if tag == "inlp" else 10000
    if meta["n_permutation_macro_f1"] != exp_perm:
        issues.append(f"{tag}: n_permutation_macro_f1 != {exp_perm} "
                      f"({meta['n_permutation_macro_f1']})")

    for cfg in EXPECTED_CONFIGS[tag]:
        if cfg not in data["results"]:
            issues.append(f"{tag}: missing config '{cfg}'")
            continue
        # Diagnostic histogram on disk (Z only).
        hist = PNG_DIRS[tag] / f"hist_predphi_by_territory_{cfg}.png"
        if not hist.exists() or hist.stat().st_size < 10_000:
            issues.append(f"{tag}/{cfg}: histogram missing or too small: {hist}")
        # CM plots for cal/hard 4c.
        for cm_name in (f"cm_B_cal_4c_{cfg}.png", f"cm_B_hard_4c_{cfg}.png"):
            cm_path = PNG_DIRS[tag] / cm_name
            if not cm_path.exists() or cm_path.stat().st_size < 10_000:
                issues.append(f"{tag}/{cfg}: CM image missing: {cm_path}")
        for src in SOURCES:
            leg = data["results"][cfg].get(src)
            if leg is None:
                issues.append(f"{tag}/{cfg}: missing source '{src}'")
                continue
            cal = leg.get("calibrator_name")
            if cal not in ALLOWED_CALIBRATORS:
                issues.append(f"{tag}/{cfg}/{src}: bad calibrator_name '{cal}'")
            cvs = leg.get("calibrator_cv_scores", {})
            if set(cvs.keys()) != ALLOWED_CALIBRATORS:
                issues.append(f"{tag}/{cfg}/{src}: cv_scores keys != allowed candidates")
            else:
                # sanity: chosen calibrator should be argmax of cv scores
                argmax = max(cvs, key=cvs.get)
                if argmax != cal:
                    issues.append(f"{tag}/{cfg}/{src}: chosen calibrator '{cal}' != argmax '{argmax}'")
            for k in EVAL_KEYS_4C:
                if k not in leg:
                    issues.append(f"{tag}/{cfg}/{src}: missing '{k}'")
                else:
                    issues.extend(check_eval(leg[k], f"{tag}/{cfg}/{src}/{k}", TERRITORIES_4C))
            for k in EVAL_KEYS_2C:
                if k not in leg:
                    issues.append(f"{tag}/{cfg}/{src}: missing '{k}'")
                else:
                    issues.extend(check_eval(leg[k], f"{tag}/{cfg}/{src}/{k}", TERRITORIES_2C))
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
                summary_lines.append(
                    f"  {tag}/{cfg:15s}  cal={z['calibrator_name']:>9s}  "
                    f"inDcal={z['in_domain_calibrator_4c']['macro_f1']:.3f}  "
                    f"CDcal={z['cross_calibrator_4c']['macro_f1']:.3f} "
                    f"(p={z['cross_calibrator_4c']['permutation_p_macro_f1']:.4f})  "
                    f"CDhard={z['cross_hardcoded_4c']['macro_f1']:.3f} "
                    f"(p={z['cross_hardcoded_4c']['permutation_p_macro_f1']:.4f})  "
                    f"CD2cal={z['cross_calibrator_2c']['macro_f1']:.3f}  "
                    f"CD2hard={z['cross_hardcoded_2c']['macro_f1']:.3f}"
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
