"""Smoke-test all active modules can be imported without errors.

Run after a cleanup to make sure no live code path was broken.
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MODULES = [
    # Top-level (root)
    "net1d",
    "finetune_model",
    "util",
    # losses / metrics packages
    "losses",
    "losses.mmd",
    "metrics",
    "metrics.multilabel",
    # scripts/
    "scripts.datasets",
    "scripts.export_latents",
    "scripts.finetune_multilabel",
    "scripts.build_medalcare_isch_targets",
    "scripts.build_ptbxl_mi_subclass",
    "scripts.extract_ecg_features_neurokit2",
    "scripts.add_medalcare_splits",
    "scripts.prepare_medalcare",
    "scripts.make_splits",
    # analysis/
    "analysis.exp7_analysis",
    "analysis.exp7_full_evaluation",
    "analysis.inlp_alignment",
    "analysis.phase_b2_infarct_decoding",
    "analysis.md_to_html",
    "analysis.visualize_latents",
]


def main() -> int:
    n_ok = 0
    n_fail = 0
    failures: list[tuple[str, str]] = []
    print("Verifying imports for all active modules\n")
    for mod in MODULES:
        try:
            importlib.import_module(mod)
            print(f"  OK   {mod}")
            n_ok += 1
        except Exception as exc:
            tb_lines = traceback.format_exc().splitlines()[-3:]
            print(f"  FAIL {mod}: {exc}")
            failures.append((mod, " | ".join(tb_lines)))
            n_fail += 1
    print()
    print(f"Result: {n_ok} OK, {n_fail} FAIL out of {len(MODULES)}")
    if failures:
        print("\nFailure details:")
        for mod, tb in failures:
            print(f"  - {mod}: {tb}")
        return 1
    print("All active modules imported successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
