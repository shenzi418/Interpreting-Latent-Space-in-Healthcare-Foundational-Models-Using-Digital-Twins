"""Does OVER-DETERMINATION hold on more than one checkpoint?

§13.1 established, on `exp8_leadfix_baseline` at k=103, that the synthetic-real
domain signal is over-determined: destroy every marginal and the GBDT still hits
1.0000 off the dependence structure; destroy the dependence and it still hits
1.0000 off the marginals. That single result carries a lot of weight -- it is the
mechanism sentence for the whole alignment dead-end -- and §13.5 flagged it as
resting on ONE encoder at ONE INLP depth.

This is the replication. Same five representations, one uniform protocol, across
every checkpoint with a complete four-way export:

  as-is           everything survives                    -> the reference
  diagonal CORAL  per-coordinate mean+var matched        -> two-moment correction
  rank transform  ALL marginal info destroyed            -> dependence only
  marginal only   independent per-coordinate shuffle     -> marginals only
  neither         rank THEN shuffle                      -> the floor, must be ~0.5

Two protocol choices differ from §13.1 deliberately:

  k = 0 BY DEFAULT. §13.1 ran after 103 INLP directions were projected out, which
  makes the test harder but ties the result to one INLP fit. The claim being
  replicated is about the raw representation, so the sweep runs on raw latents
  and `--k` is available for anyone who wants the harder version back.

  BOTTLENECK RUNS ARE INCLUDED. K=16/64/256 are 6x to 64x narrower than the 1024-d
  head. If over-determination were an artifact of having far more coordinates than
  samples, the narrow runs are where it should break; a 16-d latent cannot hide
  much redundancy. Their inclusion is the point, not incidental coverage.

The jitter in the rank arm is load-bearing for the reason documented at length in
`domain_mechanism.rank_transform` -- without it every arm reads exactly 0.5000 and
the sweep would "replicate" a greedy-induction failure across six checkpoints.
This script imports that function rather than reimplementing it, so the fix cannot
drift between the two.

Writes: outputs/analysis/domain_signal/domain_mechanism_replication.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
from analysis.domain_signal_structure import OUT_DIR, SEED, load  # noqa: E402
from analysis.domain_mechanism import (  # noqa: E402
    gbdt_auc,
    marginal_only,
    rank_transform,
)
from analysis.inlp_controls import inlp_directions, project_out  # noqa: E402

DEFAULT_RUNS = [
    "exp8_leadfix_baseline",
    "exp8_leadfix_ccmmd",
    "exp7_bottleneck_K256",
    "exp7_bottleneck_K64",
    "exp7_bottleneck_K16",
    "exp7_tier2_K64_A_5050",
]


def corr(Z: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.corrcoef(Z, rowvar=False))


def one_run(run: str, k: int, n_sub: int, seed: int) -> dict:
    Z_mtr, _ = load(run, "medalcare", "train")
    Z_ptr, _ = load(run, "ptbxl", "train")
    Z_mte, _ = load(run, "medalcare", "test")
    Z_pte, _ = load(run, "ptbxl", "test")

    sc = StandardScaler().fit(np.vstack([Z_mtr, Z_ptr]))
    Z_mtr, Z_ptr = sc.transform(Z_mtr), sc.transform(Z_ptr)
    Z_mte, Z_pte = sc.transform(Z_mte), sc.transform(Z_pte)

    rng = np.random.default_rng(seed)

    def sub(Z, n=n_sub):
        return Z if len(Z) <= n else Z[rng.choice(len(Z), n, replace=False)]

    Z_mtr, Z_ptr = sub(Z_mtr), sub(Z_ptr)
    Z_mte, Z_pte = sub(Z_mte), sub(Z_pte)
    D = Z_mtr.shape[1]

    # INLP is only meaningful if it leaves a usable subspace behind.
    k_eff = min(k, max(D - 2, 0))
    if k_eff > 0:
        W = inlp_directions(Z_mtr, Z_ptr, k_eff)
        Z_mtr, Z_ptr = project_out(Z_mtr, W), project_out(Z_ptr, W)
        Z_mte, Z_pte = project_out(Z_mte, W), project_out(Z_pte, W)

    res = {"dim": int(D), "k_removed": int(k_eff)}
    res["as-is"] = gbdt_auc(Z_mtr, Z_ptr, Z_mte, Z_pte)

    sm, sp = StandardScaler().fit(Z_mtr), StandardScaler().fit(Z_ptr)
    res["diagonal-CORAL"] = gbdt_auc(sm.transform(Z_mtr), sp.transform(Z_ptr),
                                     sm.transform(Z_mte), sp.transform(Z_pte))

    rm_tr, rm_te = rank_transform(Z_mtr, Z_mte, rng)
    rp_tr, rp_te = rank_transform(Z_ptr, Z_pte, rng)
    res["rank-transform"] = gbdt_auc(rm_tr, rp_tr, rm_te, rp_te)

    res["marginal-only"] = gbdt_auc(
        marginal_only(Z_mtr, rng), marginal_only(Z_ptr, rng),
        marginal_only(Z_mte, rng), marginal_only(Z_pte, rng))

    res["neither"] = gbdt_auc(
        marginal_only(rm_tr, rng), marginal_only(rp_tr, rng),
        marginal_only(rm_te, rng), marginal_only(rp_te, rng))

    # Classifier-free check on the dependence route, calibrated against the
    # within-domain split-half noise floor (see domain_mechanism.py).
    off = ~np.eye(D, dtype=bool)
    d_between = float(np.abs(corr(Z_mtr) - corr(Z_ptr))[off].mean())
    hp = rng.permutation(len(Z_ptr))
    hm = rng.permutation(len(Z_mtr))
    d_p = float(np.abs(corr(Z_ptr[hp[: len(hp) // 2]])
                       - corr(Z_ptr[hp[len(hp) // 2:]]))[off].mean())
    d_m = float(np.abs(corr(Z_mtr[hm[: len(hm) // 2]])
                       - corr(Z_mtr[hm[len(hm) // 2:]]))[off].mean())
    res["corr_between"] = d_between
    res["corr_floor_ptbxl"] = d_p
    res["corr_floor_medalcare"] = d_m
    res["corr_ratio"] = float(d_between / max((d_p + d_m) / 2, 1e-12))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--k", type=int, default=0,
                    help="INLP directions to project out first (0 = raw latents).")
    ap.add_argument("--subsample", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    print("=" * 104)
    print(f"Over-determination replication across checkpoints   k={args.k}")
    print("=" * 104)

    hdr = (f"{'run':<26} {'dim':>5} {'as-is':>8} {'CORAL':>8} "
           f"{'dep-only':>9} {'marg-only':>10} {'floor':>7} {'corr x':>7}")
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for run in args.runs:
        try:
            r = one_run(run, args.k, args.subsample, args.seed)
        except Exception as exc:  # missing export, wrong shape, etc.
            print(f"{run:<26} skipped: {exc}")
            continue
        results[run] = r
        print(f"{run:<26} {r['dim']:>5d} {r['as-is']:>8.4f} "
              f"{r['diagonal-CORAL']:>8.4f} {r['rank-transform']:>9.4f} "
              f"{r['marginal-only']:>10.4f} {r['neither']:>7.4f} "
              f"{r['corr_ratio']:>7.2f}")

    if not results:
        print("\nno checkpoints evaluated")
        return 1

    dep = np.array([r["rank-transform"] for r in results.values()])
    marg = np.array([r["marginal-only"] for r in results.values()])
    floor = np.array([r["neither"] for r in results.values()])
    n = len(results)
    both = int(np.sum((dep > 0.9) & (marg > 0.9)))

    print("\n" + "-" * 104)
    print(f"dependence-only  mean {dep.mean():.4f}  min {dep.min():.4f}")
    print(f"marginal-only    mean {marg.mean():.4f}  min {marg.min():.4f}")
    print(f"floor            mean {floor.mean():.4f}  max {floor.max():.4f}"
          "   <- must sit near 0.5 or the whole sweep is invalid")

    print()
    if floor.max() > 0.65:
        print("=> INVALID. The both-destroyed floor is not at chance on at least one")
        print("   checkpoint, so the destructive operations are not doing what they")
        print("   claim and no arm above can be interpreted. Fix before reporting.")
    elif both == n:
        print(f"=> OVER-DETERMINATION REPLICATES on {both}/{n} checkpoints, including")
        print("   the narrow bottlenecks. Marginals ALONE and dependence ALONE are")
        print("   each sufficient to identify the domain at >0.9 everywhere, so no")
        print("   single-route correction can align these representations. §13.1 is")
        print("   a property of the synthetic-real gap, not of one encoder.")
    elif both > 0:
        print(f"=> PARTIAL: {both}/{n} checkpoints show both routes sufficient. Report")
        print("   which ones do and which do not -- and look at dim, since a narrow")
        print("   bottleneck has less room to carry two redundant routes.")
    else:
        print("=> DOES NOT REPLICATE. §13.1 holds only on the checkpoint it was")
        print("   measured on and must be restated as such.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "domain_mechanism_replication.json"
    out.write_text(json.dumps({"k": args.k, "seed": args.seed,
                               "subsample": args.subsample,
                               "runs": results}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
