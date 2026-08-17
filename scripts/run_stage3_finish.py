"""Stage 3 finisher: repair the `dual` export, then run post-processing pass 2.

Supersedes `run_stage3_post2.py`, which would have run pass 2 over four of the
five leadfix runs without saying so. `exp8_leadfix_dual` *trained* fine on the
retry; its LATENT EXPORT is what failed, six times, with

    ValueError: Cannot infer n_classes: 'dense.weight' not in checkpoint.

because `run_stage3_leadfix.py` hard-coded `--model-type single` while the
dual-head checkpoint stores `backbone.dense.*` + `head_medal.*` + `head_ptb.*`.
`stage3_status.json` recorded `dual: FAILED` from the first driver's cuDNN
crash, so the later export failure looked like an already-known problem. It was
a second, independent one. Both scripts are fixed (`--model-type auto` resolves
the layout from the checkpoint's own keys); this re-runs the six exports.

That arm is not optional coverage. `dual` is one cell of the 2x2 architecture
ablation whose headline is "shared-head, not the 3-class relabeling, drives
cross-domain transfer". Running pass 2 without it would leave the corrected-lead
version of that comparison with one cell empty.

Order matters and is enforced: wait for every GPU job to exit, THEN export
(GPU), THEN analyse. Three trainings already died of cuDNN host-allocation
failures when two GPU jobs overlapped on this machine.

Run::

    python scripts/run_stage3_finish.py <pid> [<pid> ...]
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LOG_DIR = REPO_ROOT / "reports" / "stage3_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG = LOG_DIR / "stage3_finish_driver.log"
LATENTS = REPO_ROOT / "outputs" / "latents"
PY = sys.executable

DUAL = "exp8_leadfix_dual"
DUAL_CKPT = REPO_ROOT / "outputs" / DUAL / "checkpoints" / "linear_best.pt"


def say(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                         capture_output=True, text=True, check=False).stdout
    return str(pid) in out


def export_dual() -> bool:
    """Re-run the six dual-head exports with the resolved model type."""
    if not DUAL_CKPT.exists():
        say(f"SKIP dual export -- no checkpoint at {DUAL_CKPT}")
        return False
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", PYTHONUNBUFFERED="1")
    ok = True
    for dom in ("medalcare", "ptbxl"):
        for split in ("train", "val", "test"):
            outdir = LATENTS / f"{DUAL}_{dom}_{split}"
            if (outdir / "latents.npz").exists():
                say(f"  already exported: {outdir.name}")
                continue
            log_path = LOG_DIR / f"export2_{DUAL}_{dom}_{split}.log"
            argv = [PY, "-u", "scripts/export_latents.py",
                    "--checkpoint", str(DUAL_CKPT.relative_to(REPO_ROOT)),
                    "--model-type", "auto", "--use-adapter",
                    "--dataset", dom, "--split", split,
                    "--outdir", str(outdir.relative_to(REPO_ROOT))]
            with log_path.open("w", encoding="utf-8") as fh:
                fh.write("# " + " ".join(argv) + "\n\n")
                fh.flush()
                rc = subprocess.run(argv, cwd=REPO_ROOT, stdout=fh,
                                    stderr=subprocess.STDOUT, env=env,
                                    check=False).returncode
            say(f"  export {dom}/{split} rc={rc}")
            if rc != 0:
                ok = False
                tail = log_path.read_text(encoding="utf-8",
                                          errors="replace").splitlines()[-8:]
                for ln in tail:
                    say(f"    | {ln}")
    return ok


def main() -> int:
    pids = [int(a) for a in sys.argv[1:]]
    say("=" * 66)
    say(f"stage3 finisher armed against pids {pids}")
    while any(alive(p) for p in pids):
        time.sleep(60)
    say("all watched pids exited")

    say(f"repairing {DUAL} latent export (--model-type auto)")
    say(f"dual export complete={export_dual()}")

    say("starting post-processing pass 2")
    rc = subprocess.run([PY, "-u", str(SCRIPT_DIR / "run_stage3_post.py")],
                        cwd=str(REPO_ROOT), check=False).returncode
    say(f"pass 2 finished rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
