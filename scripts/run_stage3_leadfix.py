"""Stage 3 of the 2026-08-10 repo audit: retrain the ablation under the fixes.

Every run in `outputs/exp{5,6,7}_*` was trained on MedalCare batches whose aVL and
aVF channels were transposed (defect L) and, for the shared-head runs, evaluated on
a PTB-XL subset that kept STTC and dropped CD (defect F). Both are fixed in
`scripts/datasets.py` and `scripts/finetune_multilabel.py` respectively. This script
retrains the minimum set that restores the thesis, under run IDs prefixed `exp8_leadfix_`
so nothing existing is overwritten:

    exp8_leadfix_baseline  shared-head, no alignment          <- the new reference
    exp8_leadfix_ccmmd     shared-head + ccMMD (lambda=0.1)   <- does alignment still fail?
    exp8_leadfix_dual      dual-head, shared 3-class labels   <- restores the 2x2
    exp8_leadfix_globalz   shared-head + legacy global z      <- normalisation ablation
    exp8_leadfix_K64       bottleneck K=64 off the baseline   <- one capacity point

Then exports latents for all 6 (domain, split) cells per run.

Resumable: a stage is skipped when its completion marker already exists
(`metrics.json` for training, `latents.npz` for an export). Kill and re-run freely.

Usage:
    python scripts/run_stage3_leadfix.py                 # everything
    python scripts/run_stage3_leadfix.py --only baseline # one run
    python scripts/run_stage3_leadfix.py --train-only    # skip exports
    python scripts/run_stage3_leadfix.py --dry-run       # print the plan
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

PYTHON = sys.executable
OUTPUTS = REPO_ROOT / "outputs"
LATENTS = OUTPUTS / "latents"
LOG_DIR = REPO_ROOT / "reports" / "stage3_logs"

# Epoch budget matches the runs being replaced so the comparison is like-for-like:
# exp7_* trained 30 epochs, exp5/6_3class 17 (early-stopped at 13).
EPOCHS_SHARED = 30
EPOCHS_DUAL = 20
EPOCHS_BOTTLENECK = 20

BASE_TRAIN = [
    "scripts/finetune_multilabel.py",
    "--epochs", str(EPOCHS_SHARED),
    "--batch-size", "128",
    "--num-workers", "0",
    "--seed", "42",
]

RUNS: Dict[str, Dict] = {
    "baseline": {
        "run_id": "exp8_leadfix_baseline",
        "purpose": "shared-head, correct leads, correct PTB-XL filter -- new reference",
        "argv": BASE_TRAIN + ["--shared-head", "--run-id", "exp8_leadfix_baseline"],
    },
    "ccmmd": {
        "run_id": "exp8_leadfix_ccmmd",
        "purpose": "shared-head + class-conditional MMD; does alignment still fail post-fix?",
        "argv": BASE_TRAIN + [
            "--shared-head", "--run-id", "exp8_leadfix_ccmmd",
            "--lambda-mmd", "0.1", "--class-cond-mmd",
        ],
    },
    "dual": {
        "run_id": "exp8_leadfix_dual",
        "purpose": "dual-head at the same 3-class label space -- restores the 2x2",
        "argv": [
            "scripts/finetune_multilabel.py",
            "--epochs", str(EPOCHS_DUAL), "--batch-size", "128",
            "--num-workers", "0", "--seed", "42",
            "--dual-head-shared-labels", "--lambda-mmd", "0",
            "--run-id", "exp8_leadfix_dual",
        ],
    },
    "globalz": {
        "run_id": "exp8_leadfix_globalz",
        "purpose": "normalisation ablation: MedalCare global-scalar z vs PTB-XL per-lead z",
        "argv": BASE_TRAIN + [
            "--shared-head", "--run-id", "exp8_leadfix_globalz", "--global-z",
        ],
        "global_z": True,
    },
}

# The bottleneck run is handled separately: it fine-tunes a head on top of the
# already-trained baseline checkpoint, so it must run AFTER `baseline`.
BOTTLENECK = {
    "run_id": "exp8_leadfix_K64",
    "purpose": "one capacity point; extend the sweep only if K still matters post-fix",
    "K": 64,
    "source_run": "exp8_leadfix_baseline",
}

DOMAINS = ["medalcare", "ptbxl"]
SPLITS = ["train", "val", "test"]


def run(argv: List[str], log_path: Path, dry: bool) -> int:
    """Run a subprocess, tee-ing to `log_path`. Returns the exit code."""
    cmd = [PYTHON, "-u"] + argv
    print(f"\n{'=' * 78}\n[{datetime.now():%H:%M:%S}] {' '.join(argv)}\n{'=' * 78}")
    if dry:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # libiomp5md.dll is loaded by both torch and sklearn's MKL on this box.
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# {' '.join(cmd)}\n# started {datetime.now().isoformat()}\n\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd, cwd=REPO_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            # Progress bars are noise in an unattended log; surface milestones only.
            if any(tok in line for tok in
                   ("Epoch", "[shared-head]", "[dual-head]", "Saved", "Error",
                    "Traceback", "Best", "Shapes")):
                # The console here is cp1252; export_latents.py prints an em-dash
                # ("Shapes — Z: ...") and subprocess decoding can yield U+FFFD.
                # Either kills an overnight run with UnicodeEncodeError at the
                # FIRST export, losing every remaining run. Never let logging
                # noise abort the pipeline.
                msg = "   " + line.rstrip()[:150]
                try:
                    print(msg)
                except UnicodeEncodeError:
                    enc = sys.stdout.encoding or "ascii"
                    print(msg.encode(enc, errors="replace").decode(enc, "replace"))
        code = proc.wait()
        fh.write(f"\n# exit={code} elapsed={time.time() - t0:.0f}s\n")
    print(f"[{datetime.now():%H:%M:%S}] exit={code}  ({time.time() - t0:.0f}s)")
    return code


def train_stage(key: str, spec: Dict, dry: bool, force: bool) -> bool:
    run_id = spec["run_id"]
    marker = OUTPUTS / run_id / "metrics.json"
    if marker.exists() and not force:
        print(f"[skip train] {run_id} -- {marker.relative_to(REPO_ROOT)} exists")
        return True
    code = run(spec["argv"], LOG_DIR / f"train_{run_id}.log", dry)
    if code != 0:
        print(f"[FAIL] training {run_id} exited {code}")
        return False
    return True


def bottleneck_stage(dry: bool, force: bool) -> bool:
    run_id = BOTTLENECK["run_id"]
    marker = OUTPUTS / run_id / "metrics.json"
    if marker.exists() and not force:
        print(f"[skip train] {run_id} -- exists")
        return True
    src_ckpt = OUTPUTS / BOTTLENECK["source_run"] / "checkpoints" / "linear_best.pt"
    if not src_ckpt.exists() and not dry:
        print(f"[FAIL] {run_id}: source checkpoint missing: {src_ckpt}")
        return False
    argv = [
        "scripts/finetune_bottleneck.py",
        "--checkpoint", str(src_ckpt.relative_to(REPO_ROOT)),
        "--bottleneck-dim", str(BOTTLENECK["K"]),
        "--run-id", run_id,
        "--epochs", str(EPOCHS_BOTTLENECK), "--patience", "5",
    ]
    return run(argv, LOG_DIR / f"train_{run_id}.log", dry) == 0


def export_stage(run_id: str, global_z: bool, bottleneck: bool,
                 dry: bool, force: bool) -> bool:
    ckpt = OUTPUTS / run_id / "checkpoints" / "linear_best.pt"
    if not ckpt.exists() and not dry:
        print(f"[FAIL] export {run_id}: checkpoint missing: {ckpt}")
        return False
    ok = True
    for domain in DOMAINS:
        for split in SPLITS:
            outdir = LATENTS / f"{run_id}_{domain}_{split}"
            if (outdir / "latents.npz").exists() and not force:
                print(f"[skip export] {run_id} {domain}/{split}")
                continue
            if bottleneck:
                argv = [
                    "scripts/export_bottleneck_latents.py",
                    "--checkpoint", str(ckpt.relative_to(REPO_ROOT)),
                    "--dataset", domain, "--split", split,
                    "--outdir", str(outdir.relative_to(REPO_ROOT)),
                ]
            else:
                argv = [
                    "scripts/export_latents.py",
                    "--checkpoint", str(ckpt.relative_to(REPO_ROOT)),
                    # "auto", not "single": the joint_dual arm saves its trunk
                    # under backbone.* with head_medal/head_ptb classifiers and
                    # has no top-level dense.weight, so a hard-coded "single"
                    # aborts that run's entire export.
                    "--model-type", "auto", "--use-adapter",
                    "--dataset", domain, "--split", split,
                    "--outdir", str(outdir.relative_to(REPO_ROOT)),
                ]
                # The export must reproduce the normalisation the model saw.
                if global_z and domain == "medalcare":
                    argv.append("--global-z")
            code = run(argv, LOG_DIR / f"export_{run_id}_{domain}_{split}.log", dry)
            if code != 0:
                print(f"[FAIL] export {run_id} {domain}/{split} exited {code}")
                ok = False
    return ok


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", default=None,
                    choices=list(RUNS.keys()) + ["K64"],
                    help="Subset of runs to execute (default: all).")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Re-run stages whose completion marker already exists.")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    keys = args.only or (list(RUNS.keys()) + ["K64"])
    t0 = time.time()
    status: Dict[str, str] = {}

    for key in keys:
        if key == "K64":
            continue  # handled after the plain runs, needs baseline's checkpoint
        spec = RUNS[key]
        print(f"\n### {spec['run_id']} -- {spec['purpose']}")
        ok = True
        if not args.export_only:
            ok = train_stage(key, spec, args.dry_run, args.force)
        if ok and not args.train_only:
            ok = export_stage(spec["run_id"], spec.get("global_z", False),
                              bottleneck=False, dry=args.dry_run, force=args.force)
        status[spec["run_id"]] = "ok" if ok else "FAILED"

    if "K64" in keys:
        print(f"\n### {BOTTLENECK['run_id']} -- {BOTTLENECK['purpose']}")
        ok = True
        if not args.export_only:
            ok = bottleneck_stage(args.dry_run, args.force)
        if ok and not args.train_only:
            ok = export_stage(BOTTLENECK["run_id"], global_z=False,
                              bottleneck=True, dry=args.dry_run, force=args.force)
        status[BOTTLENECK["run_id"]] = "ok" if ok else "FAILED"

    print(f"\n{'=' * 78}\nStage 3 summary  ({time.time() - t0:.0f}s total)")
    for run_id, st in status.items():
        print(f"  {st:>8s}  {run_id}")
    summary_path = LOG_DIR / "stage3_status.json"
    if not args.dry_run:
        summary_path.write_text(
            json.dumps({"status": status,
                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                        "elapsed_s": round(time.time() - t0)}, indent=2),
            encoding="utf-8")
        print(f"  -> {summary_path.relative_to(REPO_ROOT)}")
    return 0 if all(v == "ok" for v in status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
