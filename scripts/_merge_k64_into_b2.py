"""Reassemble the 5-config Track-3 tables that the K64 step overwrites.

`run_stage3_post.py` runs phase_b2 twice -- once over the four 1024-d configs,
then again for `exp8_leadfix_K64` alone, because that run's latents are 64-d and
cannot share a design matrix with the others. The two invocations differ only in
`--out`, and `--out` names the *in-domain* file. The three cross-domain paths are
hard-coded relative to the output directory
(`phase_b2_infarct_decoding.py:1704-1707`)::

    out_cd_path = out_dir / "cross_domain.json"
    out_a_path  = out_dir / "cross_domain_4c_pipelineA.json"
    out_b_path  = out_dir / "cross_domain_4c_pipelineB.json"

so the K64 pass writes a K64-only file over the four-config one. `in_domain.json`
survives (it was renamed); the cross-domain tables -- the ones the thesis claims
rest on -- do not. This is a silent overwrite: rc=0, no warning, and the loss is
only visible if you happen to list the config keys afterwards.

This script rebuilds the union from `_snapshot_4cfg/` (taken before the K64 pass
started) plus whatever K64 left in place, and writes it back to the canonical
paths so `_summarise_b2_across_configs.py` sees all five.

Merge correctness check: `ecg_features` is computed from the NeuroKit2 control,
which never touches the encoder, so those blocks must be **numerically identical**
between the snapshot and the K64 file. They were produced by separate processes,
so agreement is a real check that the two files describe the same task, the same
PTB-XL subset and the same permutation null -- i.e. that the union is meaningful
rather than two unrelated tables stapled together. Any mismatch aborts.

Idempotent: if the canonical files already carry all five configs, it exits 0
without touching anything.

Run::

    python scripts/_merge_k64_into_b2.py [--wait-log <driver.log>] [--timeout-min N]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUT = REPO_ROOT / "outputs" / "phase_b2_exp8"
SNAP = OUT / "_snapshot_4cfg"
K64_SNAP = OUT / "_snapshot_K64"
JSONS = ("cross_domain.json", "cross_domain_4c_pipelineA.json",
         "cross_domain_4c_pipelineB.json")
K64 = "exp8_leadfix_K64"
FOUR = ("exp8_leadfix_baseline", "exp8_leadfix_ccmmd", "exp8_leadfix_dual",
        "exp8_leadfix_globalz")
DONE_MARK = "DONE  2b_phase_b2_K64"
TOL = 1e-12          # macro_f1 is deterministic -> must be bit-identical
N_PERM = 10000       # permutation draws behind each p-value
MC_SIGMA = 4.0       # allowed p deviation, in Monte-Carlo standard errors


def walk(obj, path=()):
    """Yield (path, macro_f1, p) for every dict carrying a macro_f1."""
    if isinstance(obj, dict):
        if "macro_f1" in obj and not isinstance(obj["macro_f1"], dict):
            yield path, obj["macro_f1"], obj.get("permutation_p_macro_f1")
        for k, v in obj.items():
            if not k.startswith("_"):
                yield from walk(v, path + (k,))


def wait_for(log: Path, timeout_min: int) -> bool:
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        if log.exists() and DONE_MARK in log.read_text(encoding="utf-8",
                                                       errors="replace"):
            return True
        time.sleep(30)
    return False


def control_matches(a: dict, b: dict) -> bool:
    """`ecg_features` blocks must agree between the two passes.

    The two quantities need different tolerances, and conflating them was the
    first version's bug -- it aborted a merge that was in fact correct.

    * `macro_f1` is deterministic given the data and seed. The control never
      touches the encoder, so it must be **bit-identical**; anything else means
      the two passes are not describing the same task. Tolerance 1e-12.
    * `permutation_p_macro_f1` is a Monte-Carlo estimate from `n_perm` label
      shuffles drawn by a per-config RNG. Two passes legitimately draw different
      shuffles, so p differs by ~sqrt(p(1-p)/n_perm) -- about 0.005 at p=0.5,
      n_perm=10000. Demanding equality here rejects correct merges. Allow
      `MC_SIGMA` standard errors and report the worst offender in those units,
      which is the diagnostic that actually distinguishes noise from a defect.
    """
    amap = {p: (f, q) for p, f, q in walk(a)}
    bmap = {p: (f, q) for p, f, q in walk(b)}
    shared = set(amap) & set(bmap)
    if not shared:
        print("    no comparable ecg_features blocks -- cannot verify merge")
        return False
    ok, worst, worst_at = True, 0.0, ""
    for p in sorted(shared):
        (af, aq), (bf, bq) = amap[p], bmap[p]
        if abs(af - bf) > TOL:
            print(f"    CONTROL DRIFT {'.'.join(p)} macro_f1 {af:.9f} vs {bf:.9f}"
                  "  <- deterministic, must match exactly")
            ok = False
        if aq is None or bq is None:
            continue
        se = math.sqrt(max(aq * (1.0 - aq), 1e-9) / max(N_PERM, 1))
        z = abs(aq - bq) / se
        if z > worst:
            worst, worst_at = z, '.'.join(p)
        if z > MC_SIGMA:
            print(f"    CONTROL DRIFT {'.'.join(p)} p {aq:.4f} vs {bq:.4f} "
                  f"= {z:.1f} MC sigma -- beyond permutation noise")
            ok = False
    print(f"    {len(shared)} ecg_features blocks: macro_f1 identical; "
          f"largest p deviation {worst:.2f} MC sigma ({worst_at})")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wait-log", type=Path,
                    default=REPO_ROOT / "reports" / "stage3_logs" /
                    "stage3_post_driver.log")
    ap.add_argument("--timeout-min", type=int, default=180)
    args = ap.parse_args()

    if args.wait_log and not wait_for(args.wait_log, args.timeout_min):
        print(f"timed out waiting for {DONE_MARK!r} in {args.wait_log}")
        return 1
    print(f"{DONE_MARK} seen")

    if not SNAP.exists():
        print(f"missing {SNAP} -- nothing to merge from")
        return 1
    K64_SNAP.mkdir(parents=True, exist_ok=True)

    rc = 0
    for name in JSONS:
        live = OUT / name
        cur = json.loads(live.read_text(encoding="utf-8"))
        cfgs = set(cur.get("results", {}))
        print(f"\n{name}: live configs = {sorted(cfgs)}")
        if cfgs >= set(FOUR) | {K64}:
            print("  already complete; leaving alone")
            continue
        if K64 not in cfgs:
            print(f"  {K64} not present -- K64 pass has not written this file")
            rc = 1
            continue

        shutil.copy2(live, K64_SNAP / name)
        base = json.loads((SNAP / name).read_text(encoding="utf-8"))
        k64_res = cur["results"][K64]

        # Verify the two passes agree on the config-independent control.
        ctrl_a = {c: v.get("ecg_features") for c, v in base["results"].items()}
        first = next((v for v in ctrl_a.values() if v), None)
        if first is not None and k64_res.get("ecg_features") is not None:
            if not control_matches(first, k64_res["ecg_features"]):
                print("  ABORT: control blocks disagree; not merging")
                rc = 2
                continue

        base["results"][K64] = k64_res
        meta = base.get("metadata")
        if isinstance(meta, dict) and isinstance(meta.get("configs"), list):
            meta["configs"] = [c for c in list(FOUR) + [K64]
                               if c in base["results"]]
        live.write_text(json.dumps(base, indent=2), encoding="utf-8")
        print(f"  merged -> {sorted(base['results'])}")

    print(f"\nK64-only originals preserved in {K64_SNAP.relative_to(REPO_ROOT)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
