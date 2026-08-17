"""Extend the strict-scaler (`target_pool`) arm from 2 configs to all 5.

Report S14.6 established, on `exp8_leadfix_baseline` + `exp8_leadfix_ccmmd`, that
refitting the PTB-XL scaler on the *unselected* same-split pool costs the latent
-0.0035 macro-F1 and takes it to 16/16 cross-domain blocks clearing their own
label-shuffle null, while the NeuroKit2 control's best number collapses from
0.554 (p<0.001) to 0.381 (p=0.82).

Two configs is enough to state the effect and not enough to call it a property of
the representation. The other three leadfix runs are exactly the axes that would
break it if it were an accident of one architecture:

    exp8_leadfix_dual      dual-head        -- the other cell of the 2x2
    exp8_leadfix_globalz   global-scalar z  -- normalisation ablation
    exp8_leadfix_K64       K=64 bottleneck  -- capacity

Two safeguards, both load-bearing:

1. **Never overlap.** `run_stage3_post.py` pass 2 is CPU-saturating; running this
   alongside it roughly doubled both wall times earlier today. This script waits
   for the watched pids to exit first.
2. **Prove the re-run is a superset, not a rewrite.** The 5-config run overwrites
   the same three JSONs that S14.6 cites. The seed is fixed at 42 and the two
   original configs must therefore come back bit-identical. This snapshots them
   to `_snapshot_2cfg/` beforehand and diffs every cross-domain macro_f1 and
   permutation p afterwards. If anything moved, the run is NOT deterministic and
   S14.6's numbers cannot be quoted from the new file -- the script says so
   loudly and exits non-zero rather than letting the report drift silently.

Run::

    python scripts/run_poolscaler_all5.py [<pid-to-wait-for> ...]
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUT = REPO_ROOT / "outputs" / "phase_b2_exp8_poolscaler"
SNAP = OUT / "_snapshot_2cfg"
LOG_DIR = REPO_ROOT / "reports" / "stage3_logs"
LOG = LOG_DIR / "poolscaler_all5_driver.log"
RUN_LOG = LOG_DIR / "poolscaler_all5.log"
PY = sys.executable

JSONS = ("cross_domain.json", "cross_domain_4c_pipelineA.json",
         "cross_domain_4c_pipelineB.json")
ORIGINAL_CONFIGS = ("exp8_leadfix_baseline", "exp8_leadfix_ccmmd")
ALL_CONFIGS = ORIGINAL_CONFIGS + ("exp8_leadfix_dual", "exp8_leadfix_globalz",
                                  "exp8_leadfix_K64")
TOL = 1e-12


def say(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                         capture_output=True, text=True, check=False).stdout
    return str(pid) in out


def walk_blocks(obj, path=()):
    """Yield (path, macro_f1, p) for every dict carrying a macro_f1."""
    if isinstance(obj, dict):
        if "macro_f1" in obj and not isinstance(obj["macro_f1"], dict):
            yield path, obj["macro_f1"], obj.get("permutation_p_macro_f1")
        for k, v in obj.items():
            if not k.startswith("_"):
                yield from walk_blocks(v, path + (k,))


def snapshot() -> bool:
    SNAP.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in JSONS:
        src = OUT / name
        if not src.exists():
            say(f"  cannot snapshot -- missing {src}")
            return False
        shutil.copy2(src, SNAP / name)
        n += 1
    say(f"  snapshotted {n} JSONs to {SNAP.relative_to(REPO_ROOT)}")
    return True


def verify_superset() -> bool:
    """The two original configs must be reproduced bit-identically."""
    ok, checked = True, 0
    for name in JSONS:
        before = json.loads((SNAP / name).read_text(encoding="utf-8"))
        after = json.loads((OUT / name).read_text(encoding="utf-8"))
        for cfg in ORIGINAL_CONFIGS:
            b = before["results"].get(cfg)
            a = after["results"].get(cfg)
            if b is None:
                continue
            if a is None:
                say(f"  MISSING after re-run: {name} / {cfg}")
                ok = False
                continue
            bmap = {p: (f, q) for p, f, q in walk_blocks(b)}
            amap = {p: (f, q) for p, f, q in walk_blocks(a)}
            for p, (bf, bq) in bmap.items():
                if p not in amap:
                    say(f"  BLOCK VANISHED {name}/{cfg}/{'.'.join(p)}")
                    ok = False
                    continue
                af, aq = amap[p]
                checked += 1
                if abs(af - bf) > TOL:
                    say(f"  DRIFT {name}/{cfg}/{'.'.join(p)} macro_f1 "
                        f"{bf:.9f} -> {af:.9f}")
                    ok = False
                if bq is not None and aq is not None and abs(aq - bq) > TOL:
                    say(f"  DRIFT {name}/{cfg}/{'.'.join(p)} p "
                        f"{bq:.9f} -> {aq:.9f}")
                    ok = False
    say(f"  {checked} blocks compared against the 2-config snapshot")
    return ok


def main() -> int:
    pids = [int(a) for a in sys.argv[1:]]
    say("=" * 66)
    say(f"poolscaler all-5 driver; waiting on pids {pids}")
    while any(alive(p) for p in pids):
        time.sleep(60)
    say("watched pids exited")

    if not snapshot():
        return 1

    argv = [PY, "-u", "analysis/phase_b2_infarct_decoding.py",
            "--scaler-domain", "target_pool",
            "--configs", ",".join(ALL_CONFIGS),
            "--out", "outputs/phase_b2_exp8_poolscaler/in_domain.json",
            "--no-polar", "--no-pipeline-8c"]
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE",
               PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    say("running: " + " ".join(argv[2:]))
    t0 = time.time()
    with RUN_LOG.open("w", encoding="utf-8") as fh:
        fh.write("# " + " ".join(argv) + "\n\n")
        fh.flush()
        rc = subprocess.run(argv, cwd=REPO_ROOT, stdout=fh,
                            stderr=subprocess.STDOUT, env=env,
                            check=False).returncode
    say(f"phase_b2 rc={rc}  ({(time.time() - t0) / 60:.1f} min)")
    if rc != 0:
        for ln in RUN_LOG.read_text(encoding="utf-8",
                                    errors="replace").splitlines()[-12:]:
            say(f"  | {ln}")
        return rc

    say("verifying the two original configs reproduced exactly")
    if not verify_superset():
        say("NOT DETERMINISTIC -- report S14.6 must keep quoting the snapshot in "
            f"{SNAP.relative_to(REPO_ROOT)}, not the new file.")
        return 2
    say("OK -- 5-config file is a strict superset of the 2-config result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
