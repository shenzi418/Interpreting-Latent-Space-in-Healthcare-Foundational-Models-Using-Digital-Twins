"""Stage 3 post-processing: re-run the analysis chain on the exp8_leadfix_* latents.

Waits on the Stage 3 training/export driver, then re-measures everything the
lead-order fix (defect L) and PTB-XL filter fix (defect F) invalidated:

  1. C2ST / MMD / kNN on the retrained latents  -- THE DECISION POINT (audit §5).
     The thesis's headline negative result is "C2ST ~1.0 regardless of alignment
     method". `_diag_c2st_leadfix.py` already showed the fix does not move it on
     frozen-checkpoint re-exports; this asks whether a model *trained* on correct
     leads is different. It is the one result that could re-open the dead end.
  2. Phase B2 in-domain theta decoding + Pipeline A + Pipeline B + the 8c audit.
  3. dim_scan  -- PCA K-sweep on the corrected latents.
  4. tier1_evaluation -- alignment + class-structure + mechanism suite.

Each step is independent: a failure is recorded and the run continues, because
an unattended overnight job should not lose three re-measurements to the first
crash. Steps whose inputs are missing are SKIPPED with a reason, not failed --
if Stage 3 only got through two of five runs, this still produces results for
those two.

Usage:
    python scripts/run_stage3_post.py [<stage3_pid_to_wait_on>]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "reports" / "stage3_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DRIVER_LOG = LOG_DIR / "stage3_post_driver.log"
LATENTS = REPO_ROOT / "outputs" / "latents"

PYTHON = sys.executable

EXP8_RUNS = [
    "exp8_leadfix_baseline",
    "exp8_leadfix_ccmmd",
    "exp8_leadfix_dual",
    "exp8_leadfix_globalz",
    "exp8_leadfix_K64",
]


def say(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with DRIVER_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def pid_alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                         capture_output=True, text=True, check=False).stdout
    return str(pid) in out


def wait_for(pid: int, label: str, poll_s: int = 60) -> None:
    if not pid_alive(pid):
        say(f"{label} (pid {pid}) already finished")
        return
    say(f"waiting on {label} (pid {pid}) ...")
    waited = 0
    while pid_alive(pid):
        time.sleep(poll_s)
        waited += poll_s
        if waited % 1800 == 0:
            say(f"  ... still waiting on {label} ({waited // 60} min)")
    say(f"{label} finished after ~{waited // 60} min")


def exported(run_id: str) -> bool:
    """True when all 6 (domain, split) latent cells exist for this run."""
    return all(
        (LATENTS / f"{run_id}_{dom}_{sp}" / "latents.npz").exists()
        for dom in ("medalcare", "ptbxl") for sp in ("train", "val", "test")
    )


def run_step(step_id: str, log_name: str, argv: List[str]) -> int:
    log_path = LOG_DIR / log_name
    say(f"START {step_id}  ->  {log_path.name}")
    t0 = time.time()
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", PYTHONUNBUFFERED="1")
    with log_path.open("w", encoding="utf-8") as fh:
        rc = subprocess.run([PYTHON, "-u", *argv], cwd=REPO_ROOT, stdout=fh,
                            stderr=subprocess.STDOUT, env=env, check=False).returncode
    say(f"{'DONE ' if rc == 0 else 'FAIL '} {step_id}  rc={rc}  "
        f"({(time.time() - t0) / 60:.1f} min)")
    if rc != 0:
        for ln in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]:
            say(f"    | {ln}")
    return rc


def main() -> int:
    say("=" * 66)
    say("Stage 3 post-processing driver starting")
    if len(sys.argv) > 1:
        wait_for(int(sys.argv[1]), "Stage 3 training/export")

    ready = [r for r in EXP8_RUNS if exported(r)]
    missing = [r for r in EXP8_RUNS if r not in ready]
    say(f"exported and ready: {ready}")
    if missing:
        say(f"NOT exported (steps needing them will be skipped): {missing}")
    if not ready:
        say("nothing to analyse -- Stage 3 produced no complete export. Stopping.")
        return 1

    results: Dict[str, object] = {}

    # --- 1. the decision point: C2ST on retrained latents --------------------
    # Pass --runs so the diagnostic measures the exp8 encoders. Without it the
    # script re-measures the frozen exp7 re-exports and would silently reproduce
    # the OLD numbers under a new filename.
    if ready:
        results["c2st_leadfix_trained"] = run_step(
            "1_c2st", "post_c2st.log",
            ["scripts/_diag_c2st_leadfix.py", "--runs", *ready])
    else:
        say("SKIP 1_c2st -- no exp8 run exported")
        results["c2st_leadfix_trained"] = "skipped"

    # --- 2. Phase B2: theta decoding + Pipeline A/B + 8c audit ---------------
    b2_cfgs = [r for r in ready if r != "exp8_leadfix_K64"]  # K64 is 64-d, own row
    if b2_cfgs:
        results["phase_b2_exp8"] = run_step(
            "2_phase_b2", "post_phase_b2.log",
            ["analysis/phase_b2_infarct_decoding.py",
             "--configs", ",".join(b2_cfgs),
             "--out", "outputs/phase_b2_exp8/in_domain.json"])
    else:
        results["phase_b2_exp8"] = "skipped"

    if "exp8_leadfix_K64" in ready:
        results["phase_b2_exp8_K64"] = run_step(
            "2b_phase_b2_K64", "post_phase_b2_K64.log",
            ["analysis/phase_b2_infarct_decoding.py",
             "--configs", "exp8_leadfix_K64",
             "--out", "outputs/phase_b2_exp8/in_domain_K64.json"])

    # --- 3. dim_scan on the corrected latents --------------------------------
    ds_cfgs = [r for r in ready if r in
               ("exp8_leadfix_baseline", "exp8_leadfix_ccmmd")]
    if ds_cfgs:
        results["dim_scan_exp8"] = run_step(
            "3_dim_scan", "post_dim_scan.log",
            ["analysis/dim_scan.py", "--configs", *ds_cfgs,
             "--out", "outputs/dim_scan_exp8"])
    else:
        say("SKIP 3_dim_scan -- neither baseline nor ccmmd exported")
        results["dim_scan_exp8"] = "skipped"

    # --- 4. tier-1 suite ------------------------------------------------------
    results["tier1_exp8"] = run_step(
        "4_tier1", "post_tier1.log",
        ["analysis/tier1_evaluation.py", "--configs", *ready,
         "--out", "outputs/tier1_eval_exp8"])

    say("-" * 66)
    for k, v in results.items():
        mark = "ok  " if v == 0 else ("skip" if v == "skipped" else "FAIL")
        say(f"  {mark}  {k}  ({v})")
    (LOG_DIR / "stage3_post_status.json").write_text(
        json.dumps({"ready": ready, "missing": missing, "results": results,
                    "finished_at": datetime.now().isoformat(timespec="seconds")},
                   indent=2), encoding="utf-8")
    say("Stage 3 post-processing finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
