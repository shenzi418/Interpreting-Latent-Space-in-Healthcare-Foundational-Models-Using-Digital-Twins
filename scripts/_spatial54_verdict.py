"""Extract the spatial54 verdict against the Part 16 pre-registration.

The driver's own printout labels every block `cross_domain/...` because it
truncates the filename to 12 chars -- all three JSONs collide under that label.
This reads the files separately so the primary endpoint cannot be confused with a
secondary one, and applies the decision rules exactly as written at 09:15, before
any of these numbers existed.
"""
import json
from pathlib import Path

OUT = Path("outputs/phase_b2_exp8_spatial54")
CONFIGS = ("exp8_leadfix_baseline", "exp8_leadfix_ccmmd", "exp8_leadfix_dual",
           "exp8_leadfix_globalz", "exp8_leadfix_K64")
BAND = 0.03
PRIMARY = ("cross_domain_4c_pipelineA.json", "exp8_leadfix_baseline",
           "cross_domain_4c")


def walk(obj, path=()):
    if isinstance(obj, dict):
        if "macro_f1" in obj and not isinstance(obj["macro_f1"], dict):
            yield path, obj["macro_f1"], obj.get("permutation_p_macro_f1")
        for k, v in obj.items():
            if not k.startswith("_"):
                yield from walk(v, path + (k,))


def table(fname, title):
    p = OUT / fname
    if not p.exists():
        print(f"\n{title}: MISSING")
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    print(f"\n{'=' * 92}")
    print(f"{title}   [{fname}]")
    print(f"{'=' * 92}")
    print(f"{'config':<12}{'block':<26}{'Z f1':>8}{'Z p':>9}"
          f"{'ctl f1':>9}{'ctl p':>9}{'delta':>9}  winner")
    got = {}
    for cfg in CONFIGS:
        cv = d.get("results", {}).get(cfg)
        if cv is None:
            continue
        z = {q[1:]: (f, s) for q, f, s in walk(cv) if q and q[0] == "Z"}
        c = {q[1:]: (f, s) for q, f, s in walk(cv)
             if q and q[0] == "ecg_features"}
        for blk in sorted(set(z) & set(c)):
            zf, zp = z[blk]
            cf, cp = c[blk]
            delta = zf - cf
            zsig = zp is not None and zp < 0.05
            csig = cp is not None and cp < 0.05
            if not zsig and not csig:
                win = "NEITHER SIG"
            elif zsig and not csig:
                win = "Z (only Z sig)"
            elif csig and not zsig:
                win = "CONTROL (only ctl sig)"
            elif abs(delta) <= BAND:
                win = "indistinguishable"
            else:
                win = "Z" if delta > 0 else "CONTROL"
            key = (cfg, ".".join(blk))
            got[key] = (zf, zp, cf, cp, delta, win)
            zps = f"{zp:.4f}".rjust(9) if zp is not None else " " * 9
            cps = f"{cp:.4f}".rjust(9) if cp is not None else " " * 9
            print(f"{cfg[13:]:<12}{'.'.join(blk):<26}{zf:>8.4f}{zps}"
                  f"{cf:>9.4f}{cps}{delta:>+9.4f}  {win}")
    return got


def main() -> int:
    a = table("cross_domain_4c_pipelineA.json", "PIPELINE A (primary)")
    b = table("cross_domain.json", "3-CLASS / 2-CLASS cross-domain")
    c = table("cross_domain_4c_pipelineB.json", "PIPELINE B (exploratory)")

    print(f"\n{'=' * 92}")
    print("PRIMARY ENDPOINT -- fixed in EXECUTION_LOG Part 16 at 09:15, "
          "before this run produced a number")
    print(f"{'=' * 92}")
    fn, cfg, blk = PRIMARY
    src = {"cross_domain_4c_pipelineA.json": a}[fn]
    hit = src.get((cfg, blk))
    if hit is None:
        print("  PRIMARY ENDPOINT MISSING")
        return 1
    zf, zp, cf, cp, delta, win = hit
    print(f"  pipelineA / {cfg[13:]} / {blk}")
    print(f"    Z       macro-F1 {zf:.4f}   p = {zp:.4f}")
    print(f"    control macro-F1 {cf:.4f}   p = {cp:.4f}")
    print(f"    delta (Z - control) = {delta:+.4f}   band = +/-{BAND}")
    print()
    if zp < 0.05 and cp >= 0.05:
        rule = "latent transfers, control does not -> attribution RESTORED"
    elif zp < 0.05 and cp < 0.05 and delta > BAND:
        rule = "both transfer, latent better -> attribution restored w/ control named"
    elif zp < 0.05 and cp < 0.05 and abs(delta) <= BAND:
        rule = "both transfer, indistinguishable -> Rank 5 stands as written"
    elif zp < 0.05 and cp < 0.05 and -delta > BAND:
        rule = ("both transfer, CONTROL BETTER -> S14.6 measured instrumentation; "
                "state it plainly and prominently")
    elif zp >= 0.05 and cp < 0.05:
        rule = "control transfers, latent does NOT -> strongest negative; leads limitations"
    else:
        rule = "neither transfers -> below the noise floor for both"
    print(f"  PRE-REGISTERED READING: {rule}")

    # replication discipline: rule 2 requires all five, not a subset
    print(f"\n  Replication across configs on {blk} (rule 2: all five, or call it a split):")
    for cf_ in CONFIGS:
        h = a.get((cf_, blk))
        if h:
            print(f"    {cf_[13:]:<10} Z {h[0]:.4f} (p={h[1]:.4f})  "
                  f"ctl {h[2]:.4f} (p={h[3]:.4f})  -> {h[5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
