"""Does the global6 reference cover all five configs the spatial54 arm will run?

`run_spatial54_arm.check_latent_unchanged()` intersects config keys between
`outputs/phase_b2_exp8` and the new run. An intersection silently shrinks if the
reference is missing configs -- the check would pass while comparing far fewer
blocks than it appears to, which is the failure mode where a guard reports
success for the wrong reason.
"""
import json
from pathlib import Path

CONFIGS = ("exp8_leadfix_baseline", "exp8_leadfix_ccmmd", "exp8_leadfix_dual",
           "exp8_leadfix_globalz", "exp8_leadfix_K64")
JSONS = ("cross_domain.json", "cross_domain_4c_pipelineA.json",
         "cross_domain_4c_pipelineB.json")


def walk(obj, path=()):
    if isinstance(obj, dict):
        if "macro_f1" in obj and not isinstance(obj["macro_f1"], dict):
            yield path, obj["macro_f1"], obj.get("permutation_p_macro_f1")
        for k, v in obj.items():
            if not k.startswith("_"):
                yield from walk(v, path + (k,))


def main() -> int:
    ok = True
    for name in JSONS:
        p = Path("outputs/phase_b2_exp8") / name
        if not p.exists():
            print(f"{name:34} MISSING -> guard cannot run")
            ok = False
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        r = d.get("results", {})
        meta = {k: v for k, v in d.items() if k != "results"}
        missing = [c for c in CONFIGS if c not in r]
        nz = sum(1 for cfg in r.values() for q, _, _ in walk(cfg) if q and q[0] == "Z")
        print(f"{name:34} {len(r)}/5 configs, {nz} Z blocks")
        print(f"{'':34} scaler={meta.get('scaler_domain')} "
              f"n_perm={meta.get('n_perm')} feature_set={meta.get('feature_set')}")
        if missing:
            print(f"{'':34} MISSING: {missing}")
            ok = False
    print()
    print("REFERENCE COMPLETE" if ok else
          "REFERENCE INCOMPLETE -- determinism guard would compare a subset")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
