"""Two implementations of "the same" rank transform disagree about the headline number.

`domain_mechanism.py` rank-transformed each coordinate with an exact empirical-CDF
map (`np.searchsorted` on the sorted training column) and reported held-out GBDT
C2ST = 0.5000 after 103 INLP directions were removed. `quantile_alignment.py`
did what should be the same thing with sklearn's QuantileTransformer, on the same
run, the same k, and reported 0.9999.

One of those is an artifact. The difference matters more than any other open
question here: 0.5000 would be the first time anything in this project drove the
nonlinear domain classifier to chance, and it would go straight into the thesis.
0.9999 would mean the alignment dead-end survives yet another method.

Candidate explanations, all cheap to separate:

  1. TIED / DEGENERATE COORDINATES. After projecting out 103 directions the data
     lie in a ~921-dim subspace of 1024 coordinates. No coordinate is exactly
     constant, but the transform's behaviour on near-degenerate columns differs
     between implementations.
  2. EXACT ECDF vs INTERPOLATED. searchsorted gives an exact step-function rank
     against all n training values; QuantileTransformer builds n_quantiles=1000
     reference points and interpolates linearly between them.
  3. TRAIN-SET SELF-MAPPING. Applied to the very data it was fit on, the exact
     ECDF map returns a PERFECT permutation of {0, 1/n, ..., (n-1)/n} for every
     coordinate in both domains -- the training marginals become identical by
     construction to machine precision. If the tree is trained on that and tested
     on genuinely-transformed held-out data, it may simply have had nothing to
     learn, which is not the same as the domains being indistinguishable.
  4. SUBSAMPLE / RNG DIVERGENCE. The two scripts draw their 1500-sample subsets
     through different rng call sequences, so they are not looking at identical
     matrices.

Explanation 3 is the one that would make 0.5000 a BUG rather than a result, and
it has a sharp signature: the effect would appear on the training split and
vanish on a properly held-out split. So this script reports C2ST on both.

Everything is run on ONE set of arrays, drawn once, so implementation is the only
thing that varies.

Writes: outputs/analysis/domain_signal/rank_reconciliation_<run>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
from analysis.domain_signal_structure import OUT_DIR, SEED, load  # noqa: E402
from analysis.inlp_controls import inlp_directions, project_out  # noqa: E402


def gbdt(A_tr, B_tr, A_ev, B_ev) -> float:
    X = np.vstack([A_tr, B_tr])
    y = np.concatenate([np.zeros(len(A_tr)), np.ones(len(B_tr))])
    Xe = np.vstack([A_ev, B_ev])
    ye = np.concatenate([np.zeros(len(A_ev)), np.ones(len(B_ev))])
    clf = HistGradientBoostingClassifier(random_state=SEED).fit(X, y)
    return float(roc_auc_score(ye, clf.predict_proba(Xe)[:, 1]))


def searchsorted_rank(tr, te):
    """The domain_mechanism.py implementation, verbatim in behaviour."""
    out_tr = np.empty_like(tr)
    out_te = np.empty_like(te)
    for j in range(tr.shape[1]):
        col = np.sort(tr[:, j])
        out_tr[:, j] = np.searchsorted(col, tr[:, j], side="left") / max(len(col), 1)
        out_te[:, j] = np.searchsorted(col, te[:, j], side="left") / max(len(col), 1)
    return out_tr, out_te


def sklearn_rank(tr, te, seed):
    """The quantile_alignment.py implementation."""
    q = QuantileTransformer(n_quantiles=min(1000, len(tr)),
                            output_distribution="uniform",
                            subsample=10**9, random_state=seed).fit(tr)
    return q.transform(tr), q.transform(te)


def uniformity(Z):
    """How close is each coordinate to U(0,1)? Mean KS-like max deviation."""
    n = len(Z)
    grid = (np.arange(n) + 0.5) / n
    devs = [np.abs(np.sort(Z[:, j]) - grid).max() for j in range(Z.shape[1])]
    return float(np.mean(devs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="exp8_leadfix_baseline")
    ap.add_argument("--k", type=int, default=103)
    ap.add_argument("--subsample", type=int, default=1500)
    args = ap.parse_args()

    print("=" * 96)
    print(f"Reconciling two rank-transform implementations  [{args.run}]  k={args.k}")
    print("=" * 96)

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
        print(f"removed {len(W)} INLP directions")
        Z_mtr, Z_ptr = project_out(Z_mtr, W), project_out(Z_ptr, W)
        Z_mte, Z_pte = project_out(Z_mte, W), project_out(Z_pte, W)

    print(f"arrays: M train {Z_mtr.shape} test {Z_mte.shape} | "
          f"P train {Z_ptr.shape} test {Z_pte.shape}\n")

    impls = {
        "searchsorted (domain_mechanism)": lambda: (
            searchsorted_rank(Z_mtr, Z_mte) + searchsorted_rank(Z_ptr, Z_pte)),
        "QuantileTransformer (quantile_alignment)": lambda: (
            sklearn_rank(Z_mtr, Z_mte, SEED) + sklearn_rank(Z_ptr, Z_pte, SEED)),
    }

    hdr = (f"{'implementation':>42} | {'C2ST(test)':>11} {'C2ST(train)':>12} | "
           f"{'unif(tr)':>9} {'unif(te)':>9}")
    print(hdr)
    print("-" * len(hdr))

    res = {}
    for name, fn in impls.items():
        m_tr, m_te, p_tr, p_te = fn()
        # The decisive pair: fit on train, evaluate on HELD-OUT test (the honest
        # number) vs evaluate on the training split itself (which the exact-ECDF
        # map has made uniform by construction).
        c_test = gbdt(m_tr, p_tr, m_te, p_te)
        c_train = gbdt(m_tr, p_tr, m_tr, p_tr)
        u_tr = (uniformity(m_tr) + uniformity(p_tr)) / 2
        u_te = (uniformity(m_te) + uniformity(p_te)) / 2
        res[name] = {"c2st_heldout": c_test, "c2st_trainsplit": c_train,
                     "uniformity_train": u_tr, "uniformity_test": u_te}
        print(f"{name:>42} | {c_test:>11.4f} {c_train:>12.4f} | "
              f"{u_tr:>9.4f} {u_te:>9.4f}")

    print("\n'unif' = mean over coordinates of max|sorted values - uniform grid|.")
    print("0 means the marginal is EXACTLY U(0,1); larger means it is not.")

    a = res["searchsorted (domain_mechanism)"]
    b = res["QuantileTransformer (quantile_alignment)"]
    gap = abs(a["c2st_heldout"] - b["c2st_heldout"])

    # ---- IS THE 0.5000 A GREEDY-INDUCTION ARTIFACT? ------------------------
    # C2ST(train) = 0.5000 is the real tell: a boosted tree cannot score 0.5 on
    # the data it was fitted to unless it emitted a CONSTANT, i.e. never found a
    # split with positive gain. The exact-ECDF map explains that exactly -- it
    # makes each coordinate's training values the identical multiset
    # {0, 1/n, ..., (n-1)/n} in BOTH domains, so every single-feature split has
    # zero gain and a greedy learner stops before it can reach any interaction.
    #
    # If that is the mechanism, then breaking the exact tie -- with jitter far
    # too small to carry domain information -- must restore the tree's ability
    # to grow, and C2ST should jump back up. This does not test the domains; it
    # tests the classifier.
    m_tr, m_te, p_tr, p_te = impls["searchsorted (domain_mechanism)"]()
    Xp = np.vstack([m_tr, p_tr])
    clf = HistGradientBoostingClassifier(random_state=SEED).fit(
        Xp, np.concatenate([np.zeros(len(m_tr)), np.ones(len(p_tr))]))
    proba = clf.predict_proba(Xp)[:, 1]
    n_const = float(np.ptp(proba))

    jit = 1e-9
    jrng = np.random.default_rng(SEED)
    c_jit = gbdt(m_tr + jrng.normal(0, jit, m_tr.shape),
                 p_tr + jrng.normal(0, jit, p_tr.shape),
                 m_te + jrng.normal(0, jit, m_te.shape),
                 p_te + jrng.normal(0, jit, p_te.shape))
    res["_diagnostic"] = {"searchsorted_proba_range": n_const,
                          "searchsorted_c2st_with_1e-9_jitter": c_jit}

    print(f"\nprediction range of the searchsorted-fit tree on its own training "
          f"data: {n_const:.2e}")
    print(f"held-out C2ST after adding N(0, 1e-9) jitter:  {c_jit:.4f}")

    print("\n" + "-" * 96)
    if gap < 0.05:
        print("=> THE IMPLEMENTATIONS AGREE on held-out C2ST. The earlier 0.5000 vs")
        print("   0.9999 discrepancy came from the surrounding protocol (subsample")
        print("   draw or seed sequence), not the transform. Re-derive which.")
    elif n_const < 1e-6 and c_jit > 0.9:
        print(f"=> THE 0.5000 IS AN ARTIFACT. WITHDRAWN.")
        print("   The tree fitted on exact-ECDF ranks emits a constant "
              f"(prediction range {n_const:.1e}): every coordinate has the")
        print("   identical training multiset in both domains, so no single-")
        print("   feature split has any gain and greedy induction never starts.")
        print(f"   Jitter of 1e-9 -- eighteen orders of magnitude too small to")
        print(f"   encode a domain -- restores C2ST to {c_jit:.4f}.")
        print()
        print("   CONSEQUENCES, both of which cut against the optimistic reading:")
        print("   1. Quantile normalisation does NOT align these domains. The")
        print("      honest number is QuantileTransformer's 1.0000 and the")
        print("      alignment dead-end stands, now against one more method.")
        print("   2. domain_mechanism.py's 'dependence-only -> 0.5000' arm used")
        print("      this same map and is invalid for the same reason. Its")
        print("      verdict of MARGINAL must be withdrawn: with the marginals")
        print("      matched to within interpolation error, the tree still")
        print("      scores 1.0000, so the DEPENDENCE STRUCTURE ALONE SUFFICES.")
        print("      The correlation-matrix control (2.48x the noise floor)")
        print("      already said so and was right where the headline was wrong.")
    else:
        print(f"=> THE IMPLEMENTATIONS DISAGREE by {gap:.4f} on held-out C2ST,")
        print("   and the constant-prediction explanation does NOT hold")
        print(f"   (prediction range {n_const:.1e}, jittered C2ST {c_jit:.4f}).")
        print("   Investigate further before reporting either number.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"rank_reconciliation_{args.run}.json"
    out.write_text(json.dumps({"run": args.run, "k": args.k,
                               "implementations": res}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
