"""Provenance for the spatial54 arm: what the control actually saw.

Two things must be true for the Part 16 verdict to be quotable:
  (a) the control is encoder-invariant -- it does not depend on which exp8 config
      is in the loop, so an identical control number across all five configs is
      the expected behaviour, not a copy-paste artifact;
  (b) the control is scored on the same rows as Z, with no median-imputation
      channel of the kind Part 12 found in the global6/poolscaler arm.
"""
import json
from pathlib import Path

OUT = Path("outputs/phase_b2_exp8_spatial54")
CONFIGS = ("exp8_leadfix_baseline", "exp8_leadfix_ccmmd", "exp8_leadfix_dual",
           "exp8_leadfix_globalz", "exp8_leadfix_K64")


def meta(fname):
    p = OUT / fname
    if not p.exists():
        print(f"{fname}: MISSING")
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    print(f"\n--- {fname} ---")
    for k, v in d.items():
        if k != "results":
            print(f"  {k}: {json.dumps(v)[:300]}")
    # control invariance across configs
    res = d.get("results", {})
    for blk in ("cross_domain_4c", "cross_domain_2c", "in_domain_4c",
                "primary_3class", "cross_calibrator_4c"):
        vals = []
        for cfg in CONFIGS:
            c = res.get(cfg, {}).get("ecg_features", {}).get(blk)
            if isinstance(c, dict) and "macro_f1" in c:
                vals.append(round(c["macro_f1"], 6))
        if vals:
            ok = "INVARIANT" if len(set(vals)) == 1 else "VARIES <-- unexpected"
            print(f"  control[{blk}] across {len(vals)} configs: {ok}  {vals}")
    # row counts / n used, wherever the analysis recorded them
    for cfg in CONFIGS[:1]:
        cv = res.get(cfg, {})
        for arm in ("Z", "ecg_features"):
            a = cv.get(arm, {})
            for blk, bv in a.items():
                if isinstance(bv, dict):
                    ns = {k: v for k, v in bv.items()
                          if k.startswith("n_") or k in ("n", "n_test", "n_train")}
                    if ns:
                        print(f"  [{arm}/{blk}] {ns}")


for f in ("cross_domain_4c_pipelineA.json", "cross_domain.json",
          "cross_domain_4c_pipelineB.json", "in_domain.json"):
    meta(f)
