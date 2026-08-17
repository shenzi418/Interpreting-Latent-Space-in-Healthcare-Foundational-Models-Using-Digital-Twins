"""What scaler produced the global6 reference the determinism guard compares against?

The reference JSONs carry no `scaler_domain` field, so the guard's assumption --
that `outputs/phase_b2_exp8` was produced under the same protocol the spatial54
arm will use (default `--scaler-domain target`, default n_perm) -- is unverified
from the artifact itself. If the reference came from a different scaler, "latent
identical" would be an impossible bar and the run would abort at the guard after
paying full cost.
"""
import json
from pathlib import Path

REF = Path("outputs/phase_b2_exp8")
POOL = Path("outputs/phase_b2_exp8_poolscaler")
TGT = Path("outputs/phase_b2_exp8_tgtscaler")
SRC = Path("outputs/phase_b2_exp8_srcscaler")
NAME = "cross_domain_4c_pipelineA.json"


def zblocks(p):
    """{(config, path): macro_f1} for Z arms only."""
    d = json.loads(p.read_text(encoding="utf-8"))
    out = {}

    def walk(obj, path=()):
        if isinstance(obj, dict):
            if "macro_f1" in obj and not isinstance(obj["macro_f1"], dict):
                yield path, obj["macro_f1"]
            for k, v in obj.items():
                if not k.startswith("_"):
                    yield from walk(v, path + (k,))

    for cfg, cv in d.get("results", {}).items():
        for q, f in walk(cv):
            if q and q[0] == "Z":
                out[(cfg, ".".join(q))] = f
    return out


def main() -> int:
    ref_p = REF / NAME
    if not ref_p.exists():
        print("reference missing")
        return 1
    ref = zblocks(ref_p)
    d = json.loads(ref_p.read_text(encoding="utf-8"))
    print(f"reference {NAME}: {len(ref)} Z blocks")
    print(f"  metadata keys present: {sorted(k for k in d if k != 'results')}")
    print()

    for label, base in (("target_pool", POOL), ("target", TGT), ("source", SRC)):
        p = base / NAME
        if not p.exists():
            print(f"  {label:<12} (no {base.name}/{NAME})")
            continue
        other = zblocks(p)
        shared = sorted(set(ref) & set(other))
        if not shared:
            print(f"  {label:<12} no shared blocks")
            continue
        exact = sum(1 for k in shared if ref[k] == other[k])
        worst = max(abs(ref[k] - other[k]) for k in shared)
        print(f"  {label:<12} {exact}/{len(shared)} bit-identical, "
              f"max |delta| = {worst:.6f}")
    print()
    print("The arm defaults to --scaler-domain target. If the 'target' row above")
    print("is not ~100% identical, the guard's baseline is the wrong protocol and")
    print("the run will abort at check_latent_unchanged() after paying full cost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
