"""Chain the last three steps so they fire the moment the poolscaler exits.

Sequence, gated on pid 43608:

  1. wait for the poolscaler all-5 driver to exit
  2. verify it actually succeeded -- a crashed run must NOT unlock the patch,
     because the whole point of the §8.4 gate is that the poolscaler owns
     `analysis/phase_b2_infarct_decoding.py` until it is done with it
  3. apply `_apply_featureset_patch.py` (wires in `--feature-set`)
  4. hand off to `run_spatial54_arm.py`, which re-verifies its own preconditions
     and aborts if the latent arm drifts

Step 2 is the one that earns this script's existence. Chaining is only safe if
the gate distinguishes "finished" from "finished successfully"; `alive(pid) ==
False` does not. The poolscaler's own driver writes an `rc=` line, and Stage 3
has already produced one FAILED config this week, so a nonzero rc is a live
possibility rather than a hypothetical.

Everything is logged to `reports/stage3_logs/chain_driver.log`. Read-only until
step 3; the patch and the run are the only mutations, and both are idempotent /
guarded. Run::

    python scripts/_chain_after_poolscaler.py 43608
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LOG_DIR = REPO_ROOT / "reports" / "stage3_logs"
LOG = LOG_DIR / "chain_driver.log"
DRIVER_LOG = LOG_DIR / "poolscaler_all5_driver.log"
TARGET = REPO_ROOT / "analysis" / "phase_b2_infarct_decoding.py"
PY = sys.executable


def say(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                         capture_output=True, text=True).stdout
    return str(pid) in out


def poolscaler_rc() -> int | None:
    """Last `rc=<n>` the poolscaler driver logged, or None if it never wrote one."""
    if not DRIVER_LOG.exists():
        return None
    hits = re.findall(r"rc=(-?\d+)",
                      DRIVER_LOG.read_text(encoding="utf-8", errors="replace"))
    return int(hits[-1]) if hits else None


def run(argv, tag) -> int:
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE",
               PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    say(f"running [{tag}]: {' '.join(argv[1:])}")
    p = subprocess.run(argv, cwd=REPO_ROOT, env=env,
                       capture_output=True, text=True)
    for ln in (p.stdout or "").splitlines()[-40:]:
        say(f"  | {ln}")
    for ln in (p.stderr or "").splitlines()[-15:]:
        say(f"  ! {ln}")
    say(f"[{tag}] rc={p.returncode}")
    return p.returncode


def main() -> int:
    pids = [int(a) for a in sys.argv[1:]]
    say("=" * 66)
    say(f"chain driver; gating on pids {pids}")

    while any(alive(p) for p in pids):
        time.sleep(60)
    say("watched pids exited")

    rc = poolscaler_rc()
    if rc is None:
        say("ABORT: poolscaler driver logged no rc= line. Cannot distinguish "
            "success from a crash, and the patch must not be applied on a guess.")
        return 2
    if rc != 0:
        say(f"ABORT: poolscaler finished with rc={rc}. Not applying the patch -- "
            "diagnose the failure first; a partial poolscaler is a documentation "
            "problem, a patched tree on top of one is a debugging problem.")
        return rc

    # `phase_b2 rc=0` only says the analysis exited cleanly. The driver then runs
    # its OWN determinism check (verify_superset) and can return 2 *after* logging
    # rc=0 -- so rc alone is not the success signal. Require the marker the driver
    # writes only on the path where that check passed.
    tail = DRIVER_LOG.read_text(encoding="utf-8", errors="replace")
    if "NOT DETERMINISTIC" in tail:
        say("ABORT: poolscaler's own superset check FAILED (NOT DETERMINISTIC). "
            "The 5-config file disagrees with the 2-config snapshot; that is a "
            "result to diagnose, not a gate to walk through.")
        return 5
    if "strict superset" not in tail:
        say("ABORT: poolscaler logged rc=0 but never reached its superset-check "
            "success line. Treating an unfinished driver as success is exactly "
            "the mistake this gate exists to prevent.")
        return 6
    say(f"poolscaler rc={rc}, superset check passed -- gate released")

    if "--feature-set" in TARGET.read_text(encoding="utf-8"):
        say("patch already applied; skipping")
    elif run([PY, "scripts/_apply_featureset_patch.py"], "patch") != 0:
        say("ABORT: patch failed")
        return 3

    if "--feature-set" not in TARGET.read_text(encoding="utf-8"):
        say("ABORT: patch reported success but --feature-set is still absent")
        return 4

    say("handing off to the spatial54 arm (the decisive run)")
    return run([PY, "-u", "scripts/run_spatial54_arm.py"], "spatial54")


if __name__ == "__main__":
    raise SystemExit(main())
