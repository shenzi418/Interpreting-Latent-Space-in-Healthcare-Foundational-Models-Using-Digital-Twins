"""A/B the two cross-domain standardisation conventions in Track 3.

Until 2026-08-11 every PTB-XL matrix entering a MedalCare-fit probe was
standardised with the *MedalCare* scaler. That is not a conservative baseline;
it hands the probe inputs whose per-coordinate location and scale are wrong by
the full synthetic-real offset, on top of whatever genuine domain gap exists.
`standardise_target(mode="target")` fits the scaler on the PTB-XL matrix itself
-- transductive UDA, unlabelled target features only, no target label read.

Both conventions were run end to end with the same seed, and the outputs are
preserved side by side:

    outputs/phase_b2_exp8_srcscaler/    metadata.scaler_domain == "source"
    outputs/phase_b2_exp8_tgtscaler/    metadata.scaler_domain == "target"
    outputs/phase_b2_exp8_poolscaler/   metadata.scaler_domain == "target_pool"

The third arm exists because `mode="target"` fits the scaler on the ~438-row
subset that was itself SELECTED BY MI-subclass label, so its statistics condition
on the labels being predicted. `mode="target_pool"` fits on the full unselected
same-split matrix instead, which cannot.

This script reports the deltas, and -- more importantly -- runs two validity
checks before reporting anything:

  1. `metadata.scaler_domain` in each directory must actually say what the
     directory name claims. A mislabelled snapshot would make the whole
     comparison a comparison of nothing.
  2. Every IN-DOMAIN number must be bit-identical across the two runs. The
     target scaler touches only the PTB-XL branch, so any in-domain drift means
     something else changed between the runs and no cross-domain delta can be
     attributed to the scaler.

Only the cross-domain blocks are expected to move. Significance is read off the
permutation p-value against the label-shuffle null that each run computed for
itself, not off the delta.

Run::

    # source vs target (the 2026-08-11 A/B; writes scaler_domain_ab.json)
    python scripts/_compare_scaler_domain.py

    # target vs target_pool (does the label-selected fit inflate anything?)
    python scripts/_compare_scaler_domain.py \
        --a outputs/phase_b2_exp8_tgtscaler  --a-scaler target \
        --b outputs/phase_b2_exp8_poolscaler --b-scaler target_pool
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Defaults reproduce the original comparison (the 2026-08-11 source-vs-target
# A/B). Both arms are overridable so the same validity checks can be reused for
# the target-vs-target_pool follow-up without a second copy of this file.
SRC = REPO_ROOT / "outputs" / "phase_b2_exp8_srcscaler"
TGT = REPO_ROOT / "outputs" / "phase_b2_exp8_tgtscaler"
SRC_WANT = "source"
TGT_WANT = "target"
SRC_TAG = "srcScaler"
TGT_TAG = "tgtScaler"

# (file, block-path-within-results[cfg][source], human label)
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

IN_DOMAIN_BLOCKS = [
    ("cross_domain_4c_pipelineA.json", ["in_domain_4c"]),
    ("cross_domain_4c_pipelineB.json", ["in_domain_calibrator_4c"]),
    ("cross_domain_4c_pipelineB.json", ["in_domain_hardcoded_4c"]),
]


def load(d: Path, name: str) -> dict | None:
    p = d / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def dig(obj: dict, path: list[str]):
    for k in path:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


def fmt(x, nd=4):
    return "  n/a " if x is None else f"{x:.{nd}f}"


def stars(p) -> str:
    if p is None:
        return "  "
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "* "
    return "  "


def check_labels() -> bool:
    ok = True
    for d, want in ((SRC, SRC_WANT), (TGT, TGT_WANT)):
        for name in ("cross_domain.json", "cross_domain_4c_pipelineA.json",
                     "cross_domain_4c_pipelineB.json"):
            o = load(d, name)
            if o is None:
                print(f"  MISSING {d.name}/{name}")
                ok = False
                continue
            got = o.get("metadata", {}).get("scaler_domain")
            if got is None and want == "source":
                # The srcscaler snapshot predates the --scaler-domain flag, so it
                # carries no key. Absence is unambiguous here: before the flag
                # existed there was exactly one code path and it was the source
                # scaler. Absence on any other arm is not accepted -- those must
                # positively declare themselves.
                print(f"  {d.name}/{name}: no scaler_domain key (pre-flag run, "
                      f"only the source path existed) -- accepted as 'source'")
                continue
            if got != want:
                print(f"  MISLABELLED {d.name}/{name}: "
                      f"metadata.scaler_domain={got!r}, expected {want!r}")
                ok = False
    return ok


def check_in_domain() -> bool:
    """In-domain results must not move; the target scaler is target-side only."""
    ok = True
    n_checked = 0
    for name, path in IN_DOMAIN_BLOCKS:
        a, b = load(SRC, name), load(TGT, name)
        if a is None or b is None:
            continue
        for cfg in sorted(set(a["results"]) & set(b["results"])):
            for source in sorted(set(a["results"][cfg]) & set(b["results"][cfg])):
                ba = dig(a["results"][cfg][source], path)
                bb = dig(b["results"][cfg][source], path)
                if ba is None or bb is None:
                    continue
                n_checked += 1
                if abs(ba["macro_f1"] - bb["macro_f1"]) > 1e-12:
                    print(f"  IN-DOMAIN DRIFT {name}:{path[0]} {cfg}/{source}: "
                          f"{ba['macro_f1']:.6f} vs {bb['macro_f1']:.6f}")
                    ok = False
    print(f"  {n_checked} in-domain blocks compared")
    return ok


def main() -> int:
    print("=" * 100)
    print(f"Track 3 cross-domain standardisation: {SRC.name} ({SRC_WANT}) vs "
          f"{TGT.name} ({TGT_WANT})")
    print("=" * 100)

    print("\n[check 1] metadata.scaler_domain matches directory")
    labels_ok = check_labels()
    print("  OK" if labels_ok else "  FAILED")

    print("\n[check 2] in-domain results unchanged")
    indom_ok = check_in_domain()
    print("  OK" if indom_ok else "  FAILED")

    if not labels_ok:
        print("\nRefusing to report deltas: the snapshots are not what they claim.")
        return 1

    hdr = (f"{'config':<24} {'src':<13} {'block':<14} "
           f"{SRC_TAG:>10} {TGT_TAG:>10} {'delta':>8}  "
           f"{'p_a':>7} {'p_b':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))

    rows = []
    for name, path, label in CROSS_BLOCKS:
        a, b = load(SRC, name), load(TGT, name)
        if a is None or b is None:
            continue
        for cfg in sorted(set(a["results"]) & set(b["results"])):
            for source in sorted(set(a["results"][cfg]) & set(b["results"][cfg])):
                ba = dig(a["results"][cfg][source], path)
                bb = dig(b["results"][cfg][source], path)
                if ba is None or bb is None:
                    continue
                fa, fb = ba.get("macro_f1"), bb.get("macro_f1")
                pa = ba.get("permutation_p_macro_f1")
                pb = bb.get("permutation_p_macro_f1")
                rows.append((cfg, source, label, fa, fb, pa, pb))
                print(f"{cfg:<24} {source:<13} {label:<14} "
                      f"{fmt(fa):>10} {fmt(fb):>10} "
                      f"{fb - fa:>+8.4f}  "
                      f"{fmt(pa, 3):>5}{stars(pa)} {fmt(pb, 3):>5}{stars(pb)}")

    if not rows:
        print("no comparable blocks -- has the target-scaler run finished?")
        return 1

    deltas = [r[4] - r[3] for r in rows]
    gained = [r for r in rows if r[6] is not None and r[6] < 0.05
              and (r[5] is None or r[5] >= 0.05)]
    lost = [r for r in rows if r[5] is not None and r[5] < 0.05
            and (r[6] is None or r[6] >= 0.05)]
    sig_tgt = [r for r in rows if r[6] is not None and r[6] < 0.05]

    print("\n" + "-" * len(hdr))
    print(f"blocks compared           {len(rows)}")
    print(f"mean macro-F1 delta       {sum(deltas) / len(deltas):+.4f}")
    print(f"max / min delta           {max(deltas):+.4f} / {min(deltas):+.4f}")
    print(f"improved                  {sum(d > 0 for d in deltas)}/{len(deltas)}")
    print(f"significant (p<.05) w/ {TGT_WANT} scaler   {len(sig_tgt)}/{len(rows)}")
    print(f"newly significant         {len(gained)}")
    print(f"lost significance         {len(lost)}")

    # Split by feature source. `ecg_features` is the 6 NeuroKit2 interval/amplitude
    # measurements -- the hand-crafted control that does NOT depend on the encoder,
    # which is why its numbers repeat across configs. If the scaler fix helped the
    # latent and the control equally, it corrected a shared preprocessing defect and
    # says nothing about the representation. If the two diverge, it does.
    print("\n" + "by feature source".center(len(hdr), " "))
    print("-" * len(hdr))
    print(f"{'source':<14} {'n':>3} {'mean delta':>11} "
          f"{'sig before':>11} {'sig after':>10}")
    for source in sorted({r[1] for r in rows}):
        sub = [r for r in rows if r[1] == source]
        d = [r[4] - r[3] for r in sub]
        sb = sum(1 for r in sub if r[5] is not None and r[5] < 0.05)
        sa = sum(1 for r in sub if r[6] is not None and r[6] < 0.05)
        print(f"{source:<14} {len(sub):>3} {sum(d) / len(d):>+11.4f} "
              f"{sb:>7}/{len(sub):<3} {sa:>6}/{len(sub):<3}")

    print()
    if not indom_ok:
        print("=> UNINTERPRETABLE. In-domain numbers moved, so the two runs differ")
        print("   by more than the scaler and no delta can be attributed to it.")
    elif len(gained) == 0 and max(deltas) < 0.05:
        print("=> The scaler convention was NOT what was holding cross-domain")
        print("   decoding down. Fixing it is still correct -- the source-scaler")
        print("   path was feeding the probe mis-located inputs -- but it buys no")
        print("   transfer. The barrier is elsewhere.")
    elif len(gained) > 0:
        print(f"=> {len(gained)} block(s) cross their own shuffle null only with the")
        print(f"   {TGT_WANT} scaler. Report those individually and check whether the")
        print("   gain concentrates in one pipeline or is spread across all of them.")
    else:
        print("=> Deltas are non-trivial but no block changes significance status.")
        print("   Report as a magnitude shift, not as a new positive result.")

    # The default (source vs target) pair keeps its original filename: report S14
    # and EXECUTION_LOG Part 6 both cite `scaler_domain_ab.json` by name, and a
    # rename would silently break the reproduce command they publish.
    stem = ("scaler_domain_ab" if (SRC_WANT, TGT_WANT) == ("source", "target")
            else f"scaler_domain_ab_{SRC_WANT}_vs_{TGT_WANT}")
    out = REPO_ROOT / "outputs" / "analysis" / f"{stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "arm_a": {"dir": SRC.name, "scaler_domain": SRC_WANT},
        "arm_b": {"dir": TGT.name, "scaler_domain": TGT_WANT},
        "in_domain_identical": indom_ok,
        "rows": [{"config": c, "source": s, "block": b,
                  "macro_f1_a": fa, "macro_f1_b": fb,
                  "delta": None if (fa is None or fb is None) else fb - fa,
                  "p_a": pa, "p_b": pb}
                 for c, s, b, fa, fb, pa, pb in rows],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", type=Path, default=SRC,
                    help="Baseline arm directory (default: the srcscaler snapshot).")
    ap.add_argument("--b", type=Path, default=TGT,
                    help="Comparison arm directory (default: the tgtscaler snapshot).")
    ap.add_argument("--a-scaler", default=SRC_WANT,
                    help="metadata.scaler_domain that arm A must declare.")
    ap.add_argument("--b-scaler", default=TGT_WANT,
                    help="metadata.scaler_domain that arm B must declare.")
    return ap.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    SRC, TGT = _args.a, _args.b
    SRC_WANT, TGT_WANT = _args.a_scaler, _args.b_scaler
    SRC_TAG, TGT_TAG = SRC_WANT[:10], TGT_WANT[:10]
    raise SystemExit(main())
