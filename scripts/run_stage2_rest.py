"""Stage 2.3 / 2.4 / 2.5 driver -- runs after the full phase_b2 re-run finishes.

Waits on the phase_b2 PID (Stage 2.2) so the two CPU-bound jobs do not contend,
then runs the remaining Stage 2 re-measurements in dependency order:

  2.3  eval_decoding_lowK      -- A1 (AUC column transposition), M6 (circular R2),
       concept5_classifier        A3 (permutation p), m10 (per-block RNG), m11
  2.4  dim_scan                -- A3 + m10
  2.5  inlp_alignment          -- rank(P_total) + rank-matched random-projection
       inlp_lowK                  control (A4)

Each step's stdout/stderr goes to reports/stage2_logs/<step>.log; a machine
readable status line per step goes to reports/stage2_logs/stage2_driver.log.
A step that fails does NOT abort the rest -- the failure is recorded and the
driver continues, because these steps are independent re-measurements and an
overnight run should not lose four of them to the first crash.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "reports" / "stage2_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DRIVER_LOG = LOG_DIR / "stage2_driver.log"

PYTHON = sys.executable

# (step id, log filename, argv after the interpreter)
STEPS = [
    ("2.3a_eval_decoding_lowK", "eval_decoding_lowK.log",
     ["analysis/eval_decoding_lowK.py", "--ks", "16", "64", "256", "1024"]),
    ("2.3b_concept5_classifier", "concept5_classifier.log",
     ["analysis/concept5_classifier.py", "--ablations", "--gt-upper-bound"]),
    ("2.4_dim_scan", "dim_scan.log",
     ["analysis/dim_scan.py"]),
    ("2.5a_inlp_alignment", "inlp_alignment.log",
     ["analysis/inlp_alignment.py"]),
    ("2.5b_inlp_lowK", "inlp_lowK.log",
     ["analysis/inlp_lowK.py"]),
]


def say(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with DRIVER_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def pid_alive(pid: int) -> bool:
    """True while `pid` is still running. Windows-safe, no psutil dependency."""
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True, text=True, check=False,
    ).stdout
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
        if waited % 900 == 0:
            say(f"  ... still waiting on {label} ({waited // 60} min)")
    say(f"{label} finished after ~{waited // 60} min")


def run_step(step_id: str, log_name: str, argv: list[str]) -> int:
    log_path = LOG_DIR / log_name
    say(f"START {step_id}  ->  {log_path.name}")
    t0 = time.time()
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", PYTHONUNBUFFERED="1")
    with log_path.open("w", encoding="utf-8") as fh:
        rc = subprocess.run(
            [PYTHON, "-u", *argv],
            cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT,
            env=env, check=False,
        ).returncode
    mins = (time.time() - t0) / 60
    say(f"{'DONE ' if rc == 0 else 'FAIL '} {step_id}  rc={rc}  ({mins:.1f} min)")
    if rc != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
        say(f"  last lines of {log_name}:")
        for ln in tail:
            say(f"    | {ln}")
    return rc


def main() -> int:
    say("=" * 66)
    say("Stage 2.3-2.5 driver starting")
    if len(sys.argv) > 1:
        wait_for(int(sys.argv[1]), "Stage 2.2 phase_b2")

    results = {}
    for step_id, log_name, argv in STEPS:
        results[step_id] = run_step(step_id, log_name, argv)

    say("-" * 66)
    for step_id, rc in results.items():
        say(f"  {'ok  ' if rc == 0 else 'FAIL'}  {step_id}  (rc={rc})")
    n_fail = sum(1 for rc in results.values() if rc != 0)
    say(f"Stage 2.3-2.5 driver finished: {len(results) - n_fail} ok, {n_fail} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
