"""Is dispersion the WHOLE nonlinear story, or just the part that was visible?

§12 diagnosed a per-coordinate variance mismatch and showed that correcting it
recovers +0.065 of transfer. But it also showed something the writeup passed over:
GBDT C2ST did not merely stay high under diagonal CORAL, it went to *exactly*
1.0000 (from 0.99996). Matching the dispersion did not weaken the tree at all.

Two readings, with different consequences for the thesis:

  A. Dispersion was one cue among many. The tree had several routes to the
     domain label; closing one changed nothing because the others were already
     sufficient. Under this reading the domain gap is deeply over-determined and
     no single distributional correction will ever move C2ST.

  B. Dispersion was never what the tree used. The importance profile said the
     top coordinates differ in scale, but "the tree splits on coordinate 890"
     and "the tree uses coordinate 890's VARIANCE" are different claims, and §12
     asserted the second from evidence for the first.

Distinguishing them is cheap and worth doing, because §12's mechanism sentence
("that is exactly what a tree splits on") is currently an inference, not a
measurement -- the same species of unearned step that sank three claims in §11.

The test: rebuild the domain-discrimination problem from DELIBERATELY IMPOVERISHED
inputs and see how much of the tree's 0.9999 each reconstruction buys.

  rank-transform : map each coordinate to its within-domain quantile. This
                   destroys ALL location and scale information per coordinate
                   while preserving each domain's copula (the dependence
                   structure between coordinates). If C2ST survives, the domain
                   signal lives in the DEPENDENCE, not the marginals.
  marginal-only  : independently resample each coordinate within each domain,
                   destroying all cross-coordinate dependence while preserving
                   every marginal exactly. If C2ST survives here, the signal is
                   in the MARGINALS and the copula is irrelevant.

Those two are complementary and together they partition the explanation:
  survives rank-transform, dies under marginal-only  -> dependence structure
  dies under rank-transform, survives marginal-only  -> marginal shape
  survives both                                      -> over-determined (reading A)

`marginal-only` is the decisive one for §12: if the tree still hits ~1.0 on
independently-resampled coordinates, then per-coordinate marginal differences
ARE sufficient, and §12's mechanism claim is supported. If it collapses, the
mechanism sentence must be rewritten.

BOTH rank-transformed arms carry a 1e-9 jitter, and it is load-bearing -- see
`rank_transform`'s docstring and `rank_reconciliation.py`. Without it the exact
ECDF map hands a greedy tree two blocks with byte-identical per-coordinate value
multisets, no first split has any gain, the model emits a constant, and every
such arm reads exactly 0.5000. That is a statement about tree induction, not
about the domains, and it is how an earlier version of this file reached the
wrong verdict.

Writes: outputs/analysis/domain_signal/domain_mechanism_<run>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
from analysis.domain_signal_structure import OUT_DIR, SEED, load  # noqa: E402
from analysis.inlp_controls import inlp_directions, project_out  # noqa: E402


def gbdt_auc(A_tr, B_tr, A_te, B_te) -> float:
    X = np.vstack([A_tr, B_tr])
    y = np.concatenate([np.zeros(len(A_tr)), np.ones(len(B_tr))])
    Xe = np.vstack([A_te, B_te])
    ye = np.concatenate([np.zeros(len(A_te)), np.ones(len(B_te))])
    clf = HistGradientBoostingClassifier(random_state=SEED).fit(X, y)
    return float(roc_auc_score(ye, clf.predict_proba(Xe)[:, 1]))


def rank_transform(tr, te, rng):
    """Per-coordinate within-domain quantile map, fit on train, applied to both.

    Kills location and scale for every coordinate independently while leaving the
    joint dependence intact. Test values are ranked against the TRAIN empirical
    CDF so no test information leaks into the transform.

    THE JITTER IS LOAD-BEARING, NOT COSMETIC. Without it this map makes every
    coordinate's training values the identical multiset {0, 1/n, ..., (n-1)/n} in
    BOTH domains. Every single-feature split then has exactly zero gain, a greedy
    tree never makes a first split, HistGradientBoostingClassifier emits a
    constant, and the C2ST reads 0.5000 -- on the training split as well as the
    held-out one. An earlier version of this script reported that 0.5000 as
    "dependence carries no domain signal". It was measuring greedy induction
    failing to start. See `rank_reconciliation.py`: prediction range 0.0e+00, and
    N(0, 1e-9) jitter -- far too small to encode a domain -- restores C2ST to
    1.0000.
    """
    out_tr = np.empty_like(tr)
    out_te = np.empty_like(te)
    for j in range(tr.shape[1]):
        col = np.sort(tr[:, j])
        out_tr[:, j] = np.searchsorted(col, tr[:, j], side="left") / max(len(col), 1)
        out_te[:, j] = np.searchsorted(col, te[:, j], side="left") / max(len(col), 1)
    return (out_tr + rng.normal(0, 1e-9, out_tr.shape),
            out_te + rng.normal(0, 1e-9, out_te.shape))


def marginal_only(Z, rng):
    """Independently permute each coordinate, destroying dependence.

    Every marginal distribution is preserved EXACTLY (it is a permutation of the
    same values); only the association between coordinates is destroyed. So a
    classifier that still works here is using marginal information alone.
    """
    out = Z.copy()
    for j in range(out.shape[1]):
        out[:, j] = out[rng.permutation(len(out)), j]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="exp8_leadfix_baseline")
    ap.add_argument("--k", type=int, default=103)
    ap.add_argument("--subsample", type=int, default=1500)
    args = ap.parse_args()

    print("=" * 92)
    print(f"What is the GBDT actually using?  [{args.run}]  k={args.k}")
    print("=" * 92)

    Z_mtr, _ = load(args.run, "medalcare", "train")
    Z_ptr, _ = load(args.run, "ptbxl", "train")
    Z_mte, _ = load(args.run, "medalcare", "test")
    Z_pte, _ = load(args.run, "ptbxl", "test")

    sc = StandardScaler().fit(np.vstack([Z_mtr, Z_ptr]))
    Z_mtr, Z_ptr = sc.transform(Z_mtr), sc.transform(Z_ptr)
    Z_mte, Z_pte = sc.transform(Z_mte), sc.transform(Z_pte)

    rng = np.random.default_rng(SEED)

    def sub(Z, n=args.subsample):
        return Z if len(Z) <= n else Z[rng.choice(len(Z), n, replace=False)]

    Z_mtr, Z_ptr = sub(Z_mtr), sub(Z_ptr)
    Z_mte, Z_pte = sub(Z_mte), sub(Z_pte)

    if args.k > 0:
        W = inlp_directions(Z_mtr, Z_ptr, args.k)
        print(f"removed {len(W)} INLP directions\n")
        Z_mtr, Z_ptr = project_out(Z_mtr, W), project_out(Z_ptr, W)
        Z_mte, Z_pte = project_out(Z_mte, W), project_out(Z_pte, W)

    res = {}
    hdr = f"{'representation':>28} | {'GBDT C2ST':>10} | what survives"
    print(hdr)
    print("-" * 92)

    a = gbdt_auc(Z_mtr, Z_ptr, Z_mte, Z_pte)
    res["as-is"] = a
    print(f"{'as-is':>28} | {a:>10.4f} | everything")

    # Diagonal CORAL: marginals matched in mean+variance, higher moments and
    # dependence untouched.
    sm, sp = StandardScaler().fit(Z_mtr), StandardScaler().fit(Z_ptr)
    a = gbdt_auc(sm.transform(Z_mtr), sp.transform(Z_ptr),
                 sm.transform(Z_mte), sp.transform(Z_pte))
    res["diagonal-CORAL"] = a
    print(f"{'diagonal CORAL':>28} | {a:>10.4f} | shape (3rd/4th moment) + dependence")

    # Rank transform: ALL marginal information destroyed, dependence kept.
    rm_tr, rm_te = rank_transform(Z_mtr, Z_mte, rng)
    rp_tr, rp_te = rank_transform(Z_ptr, Z_pte, rng)
    a = gbdt_auc(rm_tr, rp_tr, rm_te, rp_te)
    res["rank-transform"] = a
    print(f"{'rank transform':>28} | {a:>10.4f} | dependence only")

    # Marginal only: dependence destroyed, marginals exact.
    a = gbdt_auc(marginal_only(Z_mtr, rng), marginal_only(Z_ptr, rng),
                 marginal_only(Z_mte, rng), marginal_only(Z_pte, rng))
    res["marginal-only"] = a
    print(f"{'marginal only (indep. shuffle)':>28} | {a:>10.4f} | marginals only")

    # Both destroyed: the floor. Rank-transform THEN independent shuffle.
    a = gbdt_auc(marginal_only(rm_tr, rng), marginal_only(rp_tr, rng),
                 marginal_only(rm_te, rng), marginal_only(rp_te, rng))
    res["neither"] = a
    print(f"{'neither (floor)':>28} | {a:>10.4f} | nothing -- sanity check")

    # ---- CLASSIFIER-FREE CHECK ON THE DEPENDENCE ARM -----------------------
    # The rank-transform arm landing on EXACTLY 0.5000 is the signature of a
    # model predicting a constant, and "a GBDT found nothing" is a claim about
    # the GBDT, not about the data. Trees are known-weak at pure interaction
    # detection: with no marginal signal left, separating two copulas in ~900
    # effective dimensions from 1500 samples per domain is close to the worst
    # case for axis-aligned splits.
    #
    # So measure the dependence difference WITHOUT a classifier: compare the two
    # correlation matrices directly, and calibrate the distance against a
    # split-half of ONE domain, where the true difference is zero and whatever
    # remains is finite-sample noise. If the between-domain distance is not
    # meaningfully larger than the within-domain one, the copulas really are
    # similar and the tree was right.
    def corr(Z):
        C = np.corrcoef(Z, rowvar=False)
        return np.nan_to_num(C)

    off = ~np.eye(Z_mtr.shape[1], dtype=bool)
    d_between = float(np.abs(corr(Z_mtr) - corr(Z_ptr))[off].mean())
    h = len(Z_ptr) // 2
    perm = rng.permutation(len(Z_ptr))
    d_within = float(
        np.abs(corr(Z_ptr[perm[:h]]) - corr(Z_ptr[perm[h:]]))[off].mean()
    )
    hm = len(Z_mtr) // 2
    permm = rng.permutation(len(Z_mtr))
    d_within_m = float(
        np.abs(corr(Z_mtr[permm[:hm]]) - corr(Z_mtr[permm[hm:]]))[off].mean()
    )
    res["corr_dist_between_domains"] = d_between
    res["corr_dist_within_ptbxl_halves"] = d_within
    res["corr_dist_within_medalcare_halves"] = d_within_m

    print("\nmean |off-diagonal correlation difference|")
    print(f"  MedalCare vs PTB-XL      {d_between:.4f}")
    print(f"  PTB-XL half vs half      {d_within:.4f}   <- noise floor")
    print(f"  MedalCare half vs half   {d_within_m:.4f}   <- noise floor")
    ratio = d_between / max((d_within + d_within_m) / 2, 1e-12)
    res["corr_dist_ratio"] = float(ratio)
    print(f"  ratio to noise floor     {ratio:.2f}x")

    print("\n" + "-" * 92)
    dep, marg = res["rank-transform"], res["marginal-only"]
    print(f"dependence-only {dep:.4f}   marginal-only {marg:.4f}   "
          f"floor {res['neither']:.4f}")

    print()
    if marg > 0.9 and dep > 0.9:
        print("=> OVER-DETERMINED. Both the marginals ALONE and the dependence")
        print("   structure ALONE are each sufficient to identify the domain at")
        print("   >0.9. That is why diagonal CORAL did not move C2ST: closing one")
        print("   route leaves the other wide open. No single distributional")
        print("   correction can align these representations, and §12's transfer")
        print("   gain is therefore definitely calibration rather than alignment.")
    elif marg > 0.9:
        print("=> MARGINAL. Per-coordinate marginals alone identify the domain,")
        print("   and destroying dependence costs nothing. §12's mechanism claim")
        print("   is SUPPORTED: the tree is reading marginal differences.")
        if ratio > 2.0:
            print()
            print(f"   CAVEAT: the correlation structure DOES differ ({ratio:.1f}x the")
            print("   finite-sample noise floor), yet the tree scores chance on it.")
            print("   State this as 'the marginals are sufficient and are what the")
            print("   tree uses', NOT as 'the dependence structure matches'.")
    elif dep > 0.9:
        print("=> DEPENDENCE. The domain lives in the joint structure between")
        print("   coordinates, not in any marginal. §12's mechanism sentence")
        print("   ('exactly what a tree splits on') must be REWRITTEN -- the")
        print("   dispersion was a correlate, not the mechanism.")
    else:
        print("=> NEITHER ALONE SUFFICES. The domain signal requires marginal and")
        print("   joint structure together; it is genuinely interactive.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"domain_mechanism_{args.run}.json"
    out.write_text(json.dumps({"run": args.run, "k_removed": args.k,
                               "representations": res}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
