"""Run Track 3 against the 54-d spatial control, and settle report S15.

Report S14.6 says the latent beats hand-crafted features at cross-domain infarct
localisation. That comparison used the 6-feature NeuroKit2 control, which averages
V2-V6 into one ST number and reads T amplitude in lead II alone -- it has almost
no spatial content, and territory is *defined* by which leads deviate. So the
control could not represent the task, and S15 withdrew the comparative claim
pending a control that can.

`extract_ecg_features_spatial.py` built that control (48 per-lead columns + the
original 6, a strict superset) and `_audit_spatial_features.py` confirmed it
behaves as the textbook predicts on real data -- inferior-vs-anterior Q-wave
separation of 1.35 d, lead-specificity ratio 2.06x, and in-domain 5-fold AUROC
0.909 from the 24 Q/R columns alone. The instrument works. This runs the actual
comparison.

Protocol is pinned to S14.6's so the two are like-for-like: default `--scaler-domain
target`, default `--n-perm`, same configs, same seed. The ONLY thing that changes
is which feature matrix the `ecg_features` arm reads. That matters -- the claim is
"latent vs control", so every other knob must be held fixed or the comparison
inherits a second difference and answers neither question.

The `Z` (latent) blocks are re-computed here too and must come back **identical**
to the global6 run: the latent arm never touches the feature matrix, so a drift
there means the two runs are not describing the same task and the comparison is
void. That check is the first thing this script reports, before any verdict.

Three outcomes, all publishable, decided in advance so the result cannot be read
selectively:

  * latent still wins  -> S15's comparative claim comes back stronger, now against
    a control that CAN represent territory;
  * control wins       -> S14.6 measured instrumentation, not representation, and
    the thesis says so plainly;
  * neither is significant -> the honest reading is that cross-domain territory
    decoding is below the noise floor for both, which is itself a result about
    the sim-to-real gap.

Waits on any pids given on the command line so it cannot collide with the
poolscaler driver (both re-run the same script). Run::

    python scripts/run_spatial54_arm.py [pid ...]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUT = REPO_ROOT / "outputs" / "phase_b2_exp8_spatial54"
GLOBAL6 = REPO_ROOT / "outputs" / "phase_b2_exp8"
LOG_DIR = REPO_ROOT / "reports" / "stage3_logs"
LOG = LOG_DIR / "spatial54_driver.log"
RUN_LOG = LOG_DIR / "spatial54.log"
PY = sys.executable

CONFIGS = ("exp8_leadfix_baseline", "exp8_leadfix_ccmmd", "exp8_leadfix_dual",
           "exp8_leadfix_globalz", "exp8_leadfix_K64")
JSONS = ("cross_domain.json", "cross_domain_4c_pipelineA.json",
         "cross_domain_4c_pipelineB.json")
TOL = 1e-12
N_PERM = 10000
MC_SIGMA = 4.0


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


def wait_for(pids, poll=60) -> None:
    while True:
        live = [p for p in pids if alive(p)]
        if not live:
            return
        time.sleep(poll)


def walk(obj, path=()):
    if isinstance(obj, dict):
        if "macro_f1" in obj and not isinstance(obj["macro_f1"], dict):
            yield path, obj["macro_f1"], obj.get("permutation_p_macro_f1")
        for k, v in obj.items():
            if not k.startswith("_"):
                yield from walk(v, path + (k,))


def check_latent_unchanged() -> bool:
    """The `Z` arm must reproduce the global6 run exactly.

    Same latents, same probe, same seed -- only the feature matrix differs, and
    the latent arm never reads it. Anything other than bit-identical macro_f1
    means the two runs describe different tasks and the comparison is void.
    """
    import math
    ok, checked, worst, worst_at = True, 0, 0.0, ""
    for name in JSONS:
        a_p, b_p = GLOBAL6 / name, OUT / name
        if not (a_p.exists() and b_p.exists()):
            say(f"  cannot compare {name} -- missing on one side")
            return False
        a = json.loads(a_p.read_text(encoding="utf-8"))
        b = json.loads(b_p.read_text(encoding="utf-8"))
        for cfg, av in a.get("results", {}).items():
            bv = b.get("results", {}).get(cfg)
            if bv is None:
                continue
            amap = {p: (f, q) for p, f, q in walk(av) if p and p[0] == "Z"}
            bmap = {p: (f, q) for p, f, q in walk(bv) if p and p[0] == "Z"}
            for p in sorted(set(amap) & set(bmap)):
                (af, aq), (bf, bq) = amap[p], bmap[p]
                checked += 1
                if abs(af - bf) > TOL:
                    say(f"  LATENT DRIFT {name}/{cfg}/{'.'.join(p)} "
                        f"macro_f1 {af:.9f} vs {bf:.9f}")
                    ok = False
                if aq is None or bq is None:
                    continue
                se = math.sqrt(max(aq * (1 - aq), 1e-9) / N_PERM)
                z = abs(aq - bq) / se
                if z > worst:
                    worst, worst_at = z, f"{cfg}/{'.'.join(p)}"
                if z > MC_SIGMA:
                    say(f"  LATENT DRIFT {name}/{cfg}/{'.'.join(p)} p "
                        f"{aq:.4f} vs {bq:.4f} = {z:.1f} MC sigma")
                    ok = False
    say(f"  {checked} latent blocks compared; macro_f1 identical, "
        f"largest p deviation {worst:.2f} MC sigma ({worst_at})")
    return ok


def verdict() -> None:
    """Latent vs control, per cross-domain block, both feature sets."""
    say("")
    say("LATENT vs CONTROL -- cross-domain blocks, spatial54 control")
    say(f"  {'block':<46}{'Z f1':>8}{'Z p':>8}{'ctl f1':>9}{'ctl p':>8}  winner")
    for name in JSONS:
        p = OUT / name
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for cfg, cv in sorted(d.get("results", {}).items()):
            zmap = {q: (f, s) for q, f, s in walk(cv) if q and q[0] == "Z"}
            cmap = {q: (f, s) for q, f, s in walk(cv)
                    if q and q[0] == "ecg_features"}
            for q in sorted(zmap):
                cq = ("ecg_features",) + q[1:]
                if cq not in cmap:
                    continue
                zf, zp = zmap[q]
                cf, cp = cmap[cq]
                tag = f"{name[:12]}/{cfg[13:]}/{'.'.join(q[1:])}"
                win = ("Z" if zf > cf else "control" if cf > zf else "tie")
                if zp is not None and cp is not None:
                    if zp < 0.05 <= cp:
                        win += "  (only Z sig)"
                    elif cp < 0.05 <= zp:
                        win += "  (only control sig)"
                    elif zp >= 0.05 and cp >= 0.05:
                        win += "  (neither sig)"
                say(f"  {tag:<46}{zf:>8.4f}"
                    f"{(f'{zp:>8.4f}' if zp is not None else ' ' * 8)}"
                    f"{cf:>9.4f}"
                    f"{(f'{cp:>8.4f}' if cp is not None else ' ' * 8)}  {win}")


def main() -> int:
    pids = [int(a) for a in sys.argv[1:]]
    say("=" * 66)
    say(f"spatial54 control arm; waiting on pids {pids}")
    if pids:
        wait_for(pids)
        say("watched pids exited")

    target = REPO_ROOT / "analysis" / "phase_b2_infarct_decoding.py"
    if "--feature-set" not in target.read_text(encoding="utf-8"):
        say("ABORT: --feature-set not wired in yet. "
            "Run: python scripts/_apply_featureset_patch.py")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    argv = [PY, "-u", "analysis/phase_b2_infarct_decoding.py",
            "--feature-set", "spatial54",
            "--configs", ",".join(CONFIGS),
            "--out", "outputs/phase_b2_exp8_spatial54/in_domain.json",
            "--no-polar", "--no-pipeline-8c"]
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE",
               PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    say("running: " + " ".join(argv[2:]))
    t0 = time.time()
    with RUN_LOG.open("w", encoding="utf-8") as fh:
        fh.write("# " + " ".join(argv) + "\n\n")
        fh.flush()
        rc = subprocess.run(argv, cwd=REPO_ROOT, stdout=fh,
                            stderr=subprocess.STDOUT, env=env).returncode
    say(f"phase_b2 rc={rc}  ({(time.time() - t0) / 60:.1f} min)")
    if rc != 0:
        for ln in RUN_LOG.read_text(encoding="utf-8",
                                    errors="replace").splitlines()[-25:]:
            say(f"  | {ln}")
        return rc

    say("verifying the latent arm is unchanged from the global6 run")
    if not check_latent_unchanged():
        say("LATENT ARM DRIFTED -- the two runs are not comparable; "
            "do NOT quote a latent-vs-control verdict from this file")
        return 3
    say("OK -- latent arm identical; only the control changed")
    verdict()
    say("DONE spatial54")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
