"""Apply the `--feature-set` wiring + the cross-domain output-path fix.

Kept as a separate, idempotent script rather than applied inline because
`analysis/phase_b2_infarct_decoding.py` must not change while either long-running
driver is mid-flight:

  * `run_poolscaler_all5.py` re-runs the script and asserts the two original
    configs reproduce bit-identically. Editing the script mid-run would break that
    determinism check and make the S14.6 numbers unquotable from the new file.
  * `run_stage3_post.py`'s K64 pass is a second in-flight invocation.

Two edits, both surgical:

1. **`--feature-set {global6,spatial54}`** rebinds the three module-level feature
   paths. They are read inside `load_features()` (:314-315) and
   `load_ptbxl_features()` (:872) as globals, so a single rebind in `main()`
   before any loading redirects every consumer. Default `global6` reproduces
   today's behaviour byte-for-byte -- this patch changes no existing number.

2. **Cross-domain filenames derive from `--out`'s stem.** Today they are fixed
   (`cross_domain.json`, `cross_domain_4c_pipelineA.json`,
   `cross_domain_4c_pipelineB.json`) regardless of `--out`, so two runs sharing an
   output directory silently overwrite each other's cross-domain tables while
   `--out` protects only the in-domain file. That cost the four-config pass-2
   tables this morning (EXECUTION_LOG Part 9 S9.1). With this fix,
   `--out .../in_domain_K64.json` writes `cross_domain_K64.json` etc., and the
   default `--out .../in_domain.json` keeps the historical names exactly.

Both edits are no-ops for every command already recorded in the logs.

Run::

    python scripts/_apply_featureset_patch.py [--check]
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TARGET = REPO_ROOT / "analysis" / "phase_b2_infarct_decoding.py"

# --- edit 1a: the argparse flag -------------------------------------------
ARG_ANCHOR = '''    parser.add_argument(
        "--no-pipeline-8c", action="store_true",
        help="Skip the in-domain 8-class audit (Section 3.4).",
    )
'''
ARG_NEW = ARG_ANCHOR + '''    parser.add_argument(
        "--feature-set", choices=["global6", "spatial54"], default="global6",
        help="Which hand-crafted control to compare the latent against. "
             "'global6' (default) is the original NeuroKit2 set: 4 global scalars "
             "plus an ST average over V2-V6 and a lead-II T amplitude -- it has "
             "almost no spatial content, so it cannot represent infarct TERRITORY, "
             "which is defined by which leads deviate. 'spatial54' adds per-lead "
             "ST_J60 / Q_amp / R_amp / T_amp in all 12 leads (48 columns) and keeps "
             "the original 6 as a strict superset. See report S15.",
    )
'''

# --- edit 1b: the rebind, placed before any feature loading ---------------
REBIND_ANCHOR = """    # Load shared data
    targets = load_targets()
    feat_train_full, feat_test_full, _, _, feature_names = load_features()
"""
REBIND_NEW = """    # Feature-set selection. `load_features()` and `load_ptbxl_features()` read
    # these as module globals, so rebinding here -- before any load -- redirects
    # every consumer, including the cross-domain path. Must precede load_features().
    if args.feature_set == "spatial54":
        global FEAT_TRAIN_PATH, FEAT_TEST_PATH, FEAT_PTBXL_PATH
        FEAT_TRAIN_PATH = REPO_ROOT / "data" / "ecg_features_spatial_medalcare_train.npz"
        FEAT_TEST_PATH = REPO_ROOT / "data" / "ecg_features_spatial_medalcare_test.npz"
        FEAT_PTBXL_PATH = REPO_ROOT / "data" / "ecg_features_spatial_ptbxl_test.npz"
        missing = [p for p in (FEAT_TRAIN_PATH, FEAT_TEST_PATH, FEAT_PTBXL_PATH)
                   if not p.exists()]
        if missing:
            raise SystemExit(
                "--feature-set spatial54 needs files that are not on disk:\\n  "
                + "\\n  ".join(str(p) for p in missing)
                + "\\nRun: python scripts/extract_ecg_features_spatial.py")
        print("[features] feature-set = spatial54 (48 per-lead + the original 6)")

    # Load shared data
    targets = load_targets()
    feat_train_full, feat_test_full, _, _, feature_names = load_features()
"""

# --- edit 2: cross-domain paths follow --out ------------------------------
PATH_ANCHOR = """    out_cd_path = out_dir / "cross_domain.json"
    out_a_path = out_dir / "cross_domain_4c_pipelineA.json"

    out_b_path = out_dir / "cross_domain_4c_pipelineB.json"
    out_8c_path = out_dir / "in_domain_8c.json"
"""
PATH_NEW = '''    # Cross-domain filenames follow --out's stem. These used to be fixed, so two
    # runs sharing an output dir would silently overwrite each other's
    # cross-domain tables while --out protected only the in-domain file -- which
    # is exactly what the K64 pass did to the four-config tables this morning
    # (EXECUTION_LOG Part 9 S9.1). `in_domain.json` -> no tag, preserving every
    # historical filename; `in_domain_K64.json` -> `cross_domain_K64.json`.
    stem = args.out.stem
    tag = stem[len("in_domain"):] if stem.startswith("in_domain") else f"_{stem}"
    out_cd_path = out_dir / f"cross_domain{tag}.json"
    out_a_path = out_dir / f"cross_domain_4c_pipelineA{tag}.json"
    out_b_path = out_dir / f"cross_domain_4c_pipelineB{tag}.json"
    out_8c_path = out_dir / f"in_domain_8c{tag}.json"
'''

EDITS = [("argparse flag", ARG_ANCHOR, ARG_NEW),
         ("feature-set rebind", REBIND_ANCHOR, REBIND_NEW),
         ("cross-domain path derivation", PATH_ANCHOR, PATH_NEW)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="Report applicability without writing.")
    args = ap.parse_args()

    text = TARGET.read_text(encoding="utf-8")
    if "--feature-set" in text and "cross_domain{tag}" in text:
        print("already applied; nothing to do")
        return 0

    for name, anchor, new in EDITS:
        n = text.count(anchor)
        if n != 1:
            print(f"ABORT: anchor for {name!r} matched {n} times, expected 1")
            return 1
        print(f"  ok  {name}")
        text = text.replace(anchor, new)

    if args.check:
        print("--check: all anchors matched; not writing")
        return 0

    backup = TARGET.with_suffix(".py.pre_featureset")
    if not backup.exists():
        backup.write_text(TARGET.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  backup -> {backup.name}")
    TARGET.write_text(text, encoding="utf-8")
    print(f"patched {TARGET.relative_to(REPO_ROOT)}")

    rc = subprocess.run(["python", "-c",
                         "import ast,sys;ast.parse(open(sys.argv[1],encoding='utf-8').read())",
                         str(TARGET)], check=False).returncode
    print("syntax check:", "OK" if rc == 0 else "FAILED")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
