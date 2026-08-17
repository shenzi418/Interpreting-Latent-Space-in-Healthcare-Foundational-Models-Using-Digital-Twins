"""Summarise the poolscaler all-5 result for documentation.

Part 12 established that the `ecg_features` columns in this directory are VOID --
the feature NPZs cover MI rows only, so any control number here is scored against
a median-imputed constant for non-MI rows. Only the Z columns may be quoted. This
prints Z alongside the control and labels the control explicitly so the void
columns cannot be lifted into a table by accident.
"""
import json
from pathlib import Path

POOL = Path("outputs/phase_b2_exp8_poolscaler")
TGT = Path("outputs/phase_b2_exp8_tgtscaler")
CONFIGS = ("exp8_leadfix_baseline", "exp8_leadfix_ccmmd", "exp8_leadfix_dual",
           "exp8_leadfix_globalz", "exp8_leadfix_K64")


def walk(obj, path=()):
    if isinstance(obj, dict):
        if "macro_f1" in obj and not isinstance(obj["macro_f1"], dict):
            yield path, obj["macro_f1"], obj.get("permutation_p_macro_f1")
        for k, v in obj.items():
            if not k.startswith("_"):
                yield from walk(v, path + (k,))


def show(name, title):
    p = POOL / name
    if not p.exists():
        print(f"\n{title}: MISSING")
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    print(f"\n=== {title} ===")
    print(f"{'config':<26}{'Z f1':>9}{'Z p':>9}{'[VOID] ctl f1':>15}{'ctl p':>9}")
    for cfg in CONFIGS:
        cv = d.get("results", {}).get(cfg)
        if cv is None:
            print(f"{cfg[13:]:<26}  (absent)")
            continue
        z = [(q, f, s) for q, f, s in walk(cv) if q and q[0] == "Z"]
        c = [(q, f, s) for q, f, s in walk(cv) if q and q[0] == "ecg_features"]
        for (qz, fz, sz) in z:
            qc = ("ecg_features",) + qz[1:]
            m = [x for x in c if x[0] == qc]
            fc, sc = (m[0][1], m[0][2]) if m else (float("nan"), None)
            tag = f"{cfg[13:]}/{'.'.join(qz[1:])}" if len(qz) > 1 else cfg[13:]
            ps = f"{sz:>9.4f}" if sz is not None else " " * 9
            pc = f"{sc:>9.4f}" if sc is not None else " " * 9
            print(f"{tag:<26}{fz:>9.4f}{ps}{fc:>15.4f}{pc}")


def indomain():
    p = POOL / "in_domain.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    print("\n=== in-domain (Z source), target_pool scaler ===")
    print(f"{'config':<26}{'phi_R2c':>9}{'phi_MAE':>9}{'z_R2':>9}"
          f"{'size_R2':>9}{'rho_AUC':>9}")
    for cfg in CONFIGS:
        r = d.get("results", {}).get(cfg, {}).get("Z", {})

        def g(k, *cands):
            v = r.get(k, {})
            for kk in cands:
                if kk in v:
                    return v[kk]
            return float("nan")

        print(f"{cfg[13:]:<26}"
              f"{g('phi', 'circular_r2'):>9.4f}"
              f"{g('phi', 'circular_mae_deg'):>9.2f}"
              f"{g('z', 'r2'):>9.4f}"
              f"{g('size', 'r2'):>9.4f}"
              f"{g('rho_eps_max', 'auc', 'auroc'):>9.4f}")


def drift():
    """target vs target_pool on the Z arm: does the strict scaler change anything?"""
    print("\n=== Z-arm delta: target_pool minus target (cross-domain 4c) ===")
    n = "cross_domain_4c_pipelineA.json"
    a, b = TGT / n, POOL / n
    if not (a.exists() and b.exists()):
        print("  one side missing")
        return
    da, db = (json.loads(x.read_text(encoding="utf-8")) for x in (a, b))
    for cfg in CONFIGS:
        ca = da.get("results", {}).get(cfg)
        cb = db.get("results", {}).get(cfg)
        if not (ca and cb):
            continue
        ma = {q: f for q, f, _ in walk(ca) if q and q[0] == "Z"}
        mb = {q: f for q, f, _ in walk(cb) if q and q[0] == "Z"}
        for q in sorted(set(ma) & set(mb)):
            d = mb[q] - ma[q]
            flag = "  <-- moved" if abs(d) > 1e-9 else ""
            print(f"  {cfg[13:]:<18}{'.'.join(q[1:]):<22}"
                  f"{ma[q]:>8.4f} -> {mb[q]:>8.4f}  ({d:+.4f}){flag}")


if __name__ == "__main__":
    indomain()
    show("cross_domain.json", "cross-domain 3-class")
    show("cross_domain_4c_pipelineA.json", "pipeline A cross-domain 4c")
    show("cross_domain_4c_pipelineB.json", "pipeline B cross-domain 4c")
    drift()
    print("\nNOTE: every 'ctl' column above is VOID (Part 12): the feature NPZs "
          "cover MI rows only,\nso non-MI rows are scored against a median-imputed "
          "constant. Z columns only.")
