"""Stage 4.2 summariser — applies the pre-declared reading rule mechanically.

THRESHOLDS, fixed before the full table exists
----------------------------------------------
Written after seeing only the 3-cell smoke run (identity, I<->II, I<->III) and
before the 77-cell sweep completed. Disclosed because a threshold chosen after
seeing the spread is not a threshold.

* **Transfer varies materially** iff the spread (max - min) of macro-F1 across
  the 66 transpositions is >= 0.05 -- roughly 20% relative on the ~0.26 base the
  smoke run showed.
* **C2ST is pinned** iff min(C2ST) > 0.99 and spread(C2ST) < 0.01, for both the
  linear and the GBDT variant.

The methods claim is SUPPORTED iff both hold: the corruption is legible to
transfer and invisible to C2ST.

THE SHARPER TEST, also pre-declared
-----------------------------------
"Does the diagnostic move at all" is the weak question; a detector that moves for
every permutation regardless of harm is no more useful than one that never moves.
The question that decides whether MMD is a usable audit is whether its movement
**tracks the damage**. So: Spearman rho between each diagnostic and transfer
macro-F1 across cells. A diagnostic that flags harm should show a clear negative
rho (more distance <-> worse transfer). |rho| < 0.3 is declared uninformative.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
FP = REPO_ROOT / "outputs" / "analysis" / "leadperm_sweep" / "leadperm_sweep.json"

LIMB = {"I", "II", "III", "aVR", "aVL", "aVF"}


def kind(r):
    if r["name"] == "identity":
        return "identity"
    if r["name"].startswith("random"):
        return "random"
    a, b = r["name"].split("<->")
    if a in LIMB and b in LIMB:
        return "limb-limb"
    if a not in LIMB and b not in LIMB:
        return "prec-prec"
    return "limb-prec"


def spread(v):
    v = [x for x in v if x is not None]
    return (max(v) - min(v)) if v else float("nan")


def main() -> int:
    d = json.loads(FP.read_text(encoding="utf-8"))
    rows, meta = d["rows"], d["metadata"]
    by = {r["name"]: r for r in rows}
    ident = by["identity"]
    trans = [r for r in rows if kind(r) not in ("identity", "random")]
    rand = [r for r in rows if kind(r) == "random"]

    print("=" * 92)
    print("STAGE 4.2 -- LEAD-PERMUTATION SENSITIVITY SWEEP")
    print(f"encoder={meta['encoder']}  permuted={meta['permuted_domain']}  "
          f"n_eval={meta['n_eval_rows']}  probe refit={meta['probe']['refit']}")
    print(f"identity reproduces stored export to {meta['identity_max_abs_dev_vs_stored']:.1e}")
    print("=" * 92)

    print(f"\nREFERENCE  identity: macro-F1={ident['macro_f1']:.4f} "
          f"(p={ident['p_macro_f1']:.4f})  C2ST_lin={ident['c2st_linear']:.4f}  "
          f"C2ST_gbdt={ident.get('c2st_gbdt', float('nan')):.4f}  "
          f"MMD2={ident['mmd2']:.4f}")

    # ---- ranked table ----------------------------------------------------
    print(f"\n{'cell':<16}{'kind':<11}{'macroF1':>9}{'dF1':>8}{'p':>8}"
          f"{'C2STlin':>9}{'C2STgb':>9}{'MMD2':>9}")
    print("-" * 92)
    for r in sorted(rows, key=lambda x: -x["macro_f1"]):
        g = r.get("c2st_gbdt")
        print(f"{r['name']:<16}{kind(r):<11}{r['macro_f1']:>9.4f}"
              f"{r['macro_f1']-ident['macro_f1']:>+8.4f}{r['p_macro_f1']:>8.4f}"
              f"{r['c2st_linear']:>9.4f}"
              f"{(f'{g:.4f}' if g is not None else '-'):>9}{r['mmd2']:>9.4f}")

    # ---- the pre-declared reading ----------------------------------------
    f1_sp = spread([r["macro_f1"] for r in trans])
    lin = [r["c2st_linear"] for r in rows]
    gb = [r["c2st_gbdt"] for r in rows if r.get("c2st_gbdt") is not None]
    mmd = [r["mmd2"] for r in rows]

    print("\n" + "=" * 92)
    print("PRE-DECLARED READING")
    print("=" * 92)
    a = f1_sp >= 0.05
    b = min(lin) > 0.99 and spread(lin) < 0.01
    c = min(gb) > 0.99 and spread(gb) < 0.01
    print(f"  transfer macro-F1 spread over {len(trans)} transpositions "
          f"= {f1_sp:.4f}   (>=0.05? {'YES' if a else 'NO'})")
    print(f"  C2ST linear: min={min(lin):.4f} max={max(lin):.4f} "
          f"spread={spread(lin):.2e}   (pinned? {'YES' if b else 'NO'})")
    print(f"  C2ST GBDT  : min={min(gb):.4f} max={max(gb):.4f} "
          f"spread={spread(gb):.2e}   (pinned? {'YES' if c else 'NO'})")
    print(f"  MMD2       : min={min(mmd):.4f} max={max(mmd):.4f} "
          f"spread={spread(mmd):.4f}  ({spread(mmd)/np.mean(mmd)*100:.1f}% of mean)")
    print(f"\n  VERDICT: methods claim "
          f"{'SUPPORTED' if (a and b and c) else 'NOT SUPPORTED'} "
          f"(transfer moves={a}, C2ST_lin pinned={b}, C2ST_gbdt pinned={c})")

    # ---- is the spread bigger than within-cell sampling noise? ------------
    # A spread smaller than the typical bootstrap CI width is not evidence that
    # the permutation did anything -- it is the n=438 evaluation set resampling.
    print("\n" + "=" * 92)
    print("NOISE FLOOR  (does the spread exceed within-cell sampling error?)")
    print("=" * 92)
    ci = [r["macro_f1_ci95"] for r in rows if r.get("macro_f1_ci95")]
    if ci:
        w = [hi - lo for lo, hi in ci]
        print(f"  mean bootstrap 95% CI width per cell = {np.mean(w):.4f} "
              f"(min {min(w):.4f}, max {max(w):.4f})")
        print(f"  transposition spread                 = {f1_sp:.4f}")
        exceeds = f1_sp > np.mean(w)
        print(f"  spread {'EXCEEDS' if exceeds else 'DOES NOT EXCEED'} the mean CI "
              f"width -> cell-to-cell differences are "
              f"{'larger than' if exceeds else 'within'} sampling noise")
        ilo, ihi = ident["macro_f1_ci95"]
        out = [r["name"] for r in trans
               if r["macro_f1"] < ilo or r["macro_f1"] > ihi]
        print(f"  cells outside identity's 95% CI [{ilo:.4f}, {ihi:.4f}]: "
              f"{len(out)}/{len(trans)}")
        if out:
            print(f"    {', '.join(out[:14])}{' ...' if len(out) > 14 else ''}")

    # ---- the sharper test: does any diagnostic TRACK the damage? ---------
    print("\n" + "=" * 92)
    print("DOES THE DIAGNOSTIC TRACK THE DAMAGE?  (Spearman vs transfer macro-F1)")
    print("=" * 92)
    f1 = [r["macro_f1"] for r in rows]
    for label, v, f in (("C2ST linear", lin, f1),
                        ("MMD2", mmd, f1),
                        ("C2ST GBDT", gb,
                         [r["macro_f1"] for r in rows if r.get("c2st_gbdt") is not None])):
        if np.std(v) == 0:
            print(f"  {label:<14} rho = undefined (zero variance -- fully saturated)")
            continue
        rho, p = spearmanr(v, f)
        verd = "uninformative" if abs(rho) < 0.3 else "tracks"
        print(f"  {label:<14} rho = {rho:+.3f}  (p={p:.4f}, n={len(v)})  -> {verd}")

    # ---- per-block breakdown ---------------------------------------------
    print("\n" + "=" * 92)
    print("BY PERMUTATION CLASS  (is frontal-plane corruption special?)")
    print("=" * 92)
    print(f"{'class':<12}{'n':>4}{'meanF1':>9}{'minF1':>9}{'maxF1':>9}{'meanMMD2':>10}")
    for k in ("identity", "limb-limb", "limb-prec", "prec-prec", "random"):
        g = [r for r in rows if kind(r) == k]
        if not g:
            continue
        v = [r["macro_f1"] for r in g]
        print(f"{k:<12}{len(g):>4}{np.mean(v):>9.4f}{min(v):>9.4f}{max(v):>9.4f}"
              f"{np.mean([r['mmd2'] for r in g]):>10.4f}")

    hist = by.get("aVL<->aVF")
    if hist:
        print(f"\nHISTORICAL BUG CELL  aVL<->aVF: macro-F1={hist['macro_f1']:.4f} "
              f"({hist['macro_f1']-ident['macro_f1']:+.4f} vs identity, "
              f"p={hist['p_macro_f1']:.4f}), C2ST_lin={hist['c2st_linear']:.4f}, "
              f"C2ST_gbdt={hist.get('c2st_gbdt', float('nan')):.4f}")
        rk = sorted(rows, key=lambda x: x["macro_f1"]).index(hist) + 1
        print(f"  damage rank {rk}/{len(rows)} (1 = most damaging)")

    print(f"\nrandom-permutation floor: mean macro-F1 = "
          f"{np.mean([r['macro_f1'] for r in rand]):.4f} over {len(rand)} draws")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
