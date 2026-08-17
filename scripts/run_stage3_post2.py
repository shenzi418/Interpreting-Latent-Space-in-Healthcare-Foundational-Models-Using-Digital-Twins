"""Second post-processing pass, armed to fire when Stage 3's RETRY finishes.

The first post pass took its `ready` snapshot at 05:14, when only
`exp8_leadfix_baseline` and `exp8_leadfix_ccmmd` had exports. `dual`, `globalz`
and `K64` had died in the first Stage 3 driver with a cuDNN host-allocation
failure and were relaunched separately; every artifact under
`outputs/{phase_b2_exp8, dim_scan_exp8, tier1_eval_exp8}` therefore covers two
runs, not five.

This waits for BOTH the running post driver and the retry trainer to exit, then
re-runs `run_stage3_post.py` with a fresh `ready` snapshot so the late runs are
included. Waiting for the post driver too is the point: two concurrent passes
would write the same output paths, and the first driver's own logs record that
resource contention is what killed three trainings in the first place.

One convention change rides along, deliberately and visibly. Pass 2 calls
`phase_b2_infarct_decoding.py` without `--scaler-domain`, which now defaults to
`target` -- PTB-XL inputs standardised by PTB-XL statistics rather than by
MedalCare's. Pass 1's source-scaler artifacts are preserved at
`outputs/phase_b2_exp8_srcscaler/`, and every output JSON records which mode
produced it under `metadata.scaler_domain`.

Run::

    python scripts/run_stage3_post2.py <post_pid> <retry_pid>
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LOG_DIR = REPO_ROOT / "reports" / "stage3_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG = LOG_DIR / "stage3_post2_driver.log"
PY = sys.executable


def say(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def alive(pid: int) -> bool:
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True, text=True, check=False).stdout
    return str(pid) in out


def main() -> int:
    pids = [int(a) for a in sys.argv[1:]]
    say(f"post-pass-2 driver armed against pids {pids}")
    while any(alive(p) for p in pids):
        time.sleep(60)
    say("all watched pids exited; starting pass 2")

    rc = subprocess.run([PY, "-u", str(SCRIPT_DIR / "run_stage3_post.py")],
                        cwd=str(REPO_ROOT), check=False).returncode
    say(f"pass 2 finished rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
