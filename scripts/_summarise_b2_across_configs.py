"""Replication table: is a Track-3 cross-domain result a property of ONE run?

Report S14 and S14.6 both rest on two configs (`exp8_leadfix_baseline`,
`exp8_leadfix_ccmmd`). Those two share an architecture, so agreement between them
is close to no evidence at all -- `ecg_features` rows are literally identical
across configs because the NeuroKit2 control never touches the encoder. The
question that matters is whether the latent's cross-domain significance survives
the OTHER three leadfix runs, which vary the axes that could break it:

    exp8_leadfix_dual      dual-head        -- other cell of the 2x2
    exp8_leadfix_globalz   global-scalar z  -- normalisation ablation
    exp8_leadfix_K64       K=64 bottleneck  -- capacity

This prints, per block, one row per config with macro-F1 and the permutation
p-value each run computed against its own label-shuffle null, then a verdict line
counting how many configs clear p<.05.

Read the p column, not the F1 column. A macro-F1 can rise purely because the
prediction distribution drifted toward the label marginal; the shuffle null
absorbs exactly that, which is why S14.6 contains four blocks whose F1 FELL while
they gained significance. F1 is descriptive here; p is the instrument.

`ecg_features` rows are printed too and are expected to be constant down each
block -- that constancy is the check that the config axis is wired correctly. If
the control varies by config, the loader is picking up something encoder-derived
and the comparison is broken.

Run::

    python scripts/_summarise_b2_across_configs.py [<dir> ...]
    # default: outputs/phase_b2_exp8 (target scaler, pass 2)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

CROSS_BLOCKS = [
    ("cross_domain.json", ["primary_3class"], "T1 phi-bin 3c"),
    ("cross_domain.json", ["sensitivity_2class_AntInf"], "T1 phi-bin 2c"),
    ("cross_domain_4c_pipelineA.json", ["cross_domain_4c"], "A direct 4c"),
    ("cross_domain_4c_pipelineA.json", ["cross_domain_2c"], "A direct 2c"),
    ("cross_domain_4c_pipelineB.json", ["cross_calibrator_4c"], "B calib 4c"),
    ("cross_domain_4c_pipelineB.json", ["cross_calibrator_2c"], "B calib 2c"),
    ("cross_domain_4c_pipelineB.json", ["cross_hardcoded_4c"], "B hard 4c"),
    ("cross_domain_4c_pipelineB.json", ["cross_hardcoded_2c"], "B hard 2c"),
]

ORDER = ["exp8_leadfix_baseline", "exp8_leadfix_ccmmd", "exp8_leadfix_dual",
         "exp8_leadfix_globalz", "exp8_leadfix_K64"]


def dig(obj, path):
    for k in path:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


def stars(p):
    if p is None:
        return "  "
    return "**" if p < 0.01 else ("* " if p < 0.05 else "  ")


def summarise(d: Path) -> int:
    if not d.exists():
        print(f"missing directory {d}")
        return 1
    meta = None
    loaded, missing = {}, []
    for name in {b[0] for b in CROSS_BLOCKS}:
        p = d / name
        if p.exists():
            loaded[name] = json.loads(p.read_text(encoding="utf-8"))
            meta = meta or loaded[name].get("metadata")
        else:
            missing.append(name)

    print("=" * 92)
    print(f"Track 3 cross-domain replication across configs -- {d.name}")
    if meta:
        print(f"  scaler_domain = {meta.get('scaler_domain')!r}   "
              f"seed = {meta.get('seed')}   n_perm = {meta.get('n_permutation_macro_f1')}")
        print(f"  configs present = {meta.get('configs')}")
    if missing:
        print(f"  NOT YET WRITTEN: {', '.join(sorted(missing))}")
    print("=" * 92)

    verdicts = []
    for name, path, label in CROSS_BLOCKS:
        obj = loaded.get(name)
        if obj is None:
            continue
        results = obj["results"]
        cfgs = [c for c in ORDER if c in results] + \
               [c for c in sorted(results) if c not in ORDER]
        sources = sorted({s for c in cfgs for s in results[c]})
        print(f"\n{label}")
        print(f"  {'config':<24} " +
              "  ".join(f"{s:>22}" for s in sources))
        for c in cfgs:
            cells = []
            for s in sources:
                blk = dig(results[c].get(s, {}), path)
                if blk is None:
                    cells.append(f"{'--':>22}")
                    continue
                f1 = blk.get("macro_f1")
                p = blk.get("permutation_p_macro_f1")
                pt = "  n/a" if p is None else (
                    "<1e-4" if p < 1e-4 else f"{p:.4f}")
                cells.append(f"{f1:>9.4f}  p={pt:<6}{stars(p)}")
            print(f"  {c:<24} " + "  ".join(cells))
        for s in sources:
            ps = [dig(results[c].get(s, {}), path) for c in cfgs]
            ps = [b.get("permutation_p_macro_f1") for b in ps if b]
            n_sig = sum(1 for p in ps if p is not None and p < 0.05)
            verdicts.append((label, s, n_sig, len(ps)))

    print("\n" + "=" * 92)
    print("replication verdict  (configs clearing their own shuffle null, p<.05)")
    print("=" * 92)
    for label, s, n_sig, n in verdicts:
        if n == 0:
            continue
        mark = "FULL" if n_sig == n else ("none" if n_sig == 0 else "PARTIAL")
        print(f"  {label:<16} {s:<14} {n_sig}/{n}   {mark}")

    z = [(l, k, n) for l, s, k, n in verdicts if s == "Z"]
    full = [l for l, k, n in z if n and k == n]
    if full:
        print(f"\n  latent blocks significant in EVERY config: {', '.join(full)}")
    else:
        print("\n  no latent block is significant in every config")
    return 0


def main() -> int:
    dirs = [Path(a) for a in sys.argv[1:]] or \
           [REPO_ROOT / "outputs" / "phase_b2_exp8"]
    rc = 0
    for d in dirs:
        rc |= summarise(d if d.is_absolute() else REPO_ROOT / d)
        print()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
