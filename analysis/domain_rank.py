"""How many dimensions does domain identity actually occupy?

§1 reports that INLP needs ~90 directions to drive C2ST to chance. But
`subspace_mechanism.py --whiten` found INLP halting after just 2 directions,
because the logistic weight collapsed to ~0 -- no linear direction left to find.
Both cannot be the true dimensionality of the domain signal.

The suspicion: k=90 is a property of the METRIC, not of the data. INLP removes a
direction and refits; in a badly anisotropic space (participation ratio ~71 of
1024, per `direction_agreement.py`) each removal is Euclidean-orthogonal but not
decorrelating, so residual domain signal keeps reappearing in directions already
"removed" -- and the procedure grinds through dozens of directions chasing the
same underlying difference. Whitening first should collapse that to its true
rank.

The test is C2ST after removal, which is the same yardstick §1 used, measured in
both geometries as a function of how many directions were taken out:

  * EUCLIDEAN   -- reproduces §1's k=90 curve.
  * WHITENED    -- rank-truncated to keep the covariance estimable (full-rank
                   whitening amplifies ~950 noise eigendirections; measured, it
                   drove transfer from 0.70 to 0.54).

Crucially C2ST is always evaluated on HELD-OUT data with a freshly fit
classifier, so "no direction left for INLP to find on train" cannot be mistaken
for "the domains are actually indistinguishable".

If whitened C2ST reaches chance at k=2 while Euclidean needs ~90, then domain
identity is genuinely low-dimensional and the k=90 figure is an artifact of
running INLP in the wrong geometry -- which would also explain why removing 90
directions is so destructive to transfer (§1): 88 of them were never about
domain in the first place.

Writes: outputs/analysis/domain_signal/domain_rank_<run>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
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
from analysis.subspace_mechanism import whitener  # noqa: E402


def c2st(A_tr, B_tr, A_te, B_te, seed=SEED) -> float:
    """Held-out domain-discrimination AUROC. 0.5 = domains indistinguishable."""
    X = np.vstack([A_tr, B_tr])
    y = np.concatenate([np.zeros(len(A_tr)), np.ones(len(B_tr))])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                             random_state=seed).fit(X, y)
    Xe = np.vstack([A_te, B_te])
    ye = np.concatenate([np.zeros(len(A_te)), np.ones(len(B_te))])
    return float(roc_auc_score(ye, clf.predict_proba(Xe)[:, 1]))


def curve(Z_mtr, Z_ptr, Z_mte, Z_pte, ks, label):
    """C2ST after removing the first k INLP directions, for each k in ks."""
    kmax = max(ks)
    W = inlp_directions(Z_mtr, Z_ptr, kmax)
    print(f"  [{label}] INLP returned {len(W)} of {kmax} requested"
          + ("  <- halted early: no linear direction left on train"
             if len(W) < kmax else ""))
    out = {}
    for k in ks:
        if k > len(W):
            continue
        Wk = W[:k]
        a_tr, b_tr = project_out(Z_mtr, Wk), project_out(Z_ptr, Wk)
        a_te, b_te = project_out(Z_mte, Wk), project_out(Z_pte, Wk)
        out[k] = c2st(a_tr, b_tr, a_te, b_te)
        print(f"    k={k:<4} C2ST={out[k]:.4f}")
    return out, len(W)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="exp8_leadfix_baseline")
    ap.add_argument("--subsample", type=int, default=1500)
    ap.add_argument("--whiten-rank", type=int, default=128)
    args = ap.parse_args()

    print("=" * 84)
    print(f"True dimensionality of the domain signal  [{args.run}]")
    print("=" * 84)

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

    print("\nEUCLIDEAN (as in §1):")
    ks_e = [0, 1, 2, 4, 8, 16, 32, 64, 90]
    eu, n_eu = curve(Z_mtr, Z_ptr, Z_mte, Z_pte, ks_e, "euclidean")

    print(f"\nWHITENED (rank {args.whiten_rank}):")
    Wt = whitener(np.vstack([Z_mtr, Z_ptr]), rank=args.whiten_rank)
    Wm_tr, Wp_tr = Z_mtr @ Wt, Z_ptr @ Wt
    Wm_te, Wp_te = Z_mte @ Wt, Z_pte @ Wt
    ks_w = [0, 1, 2, 4, 8, 16, 32]
    wh, n_wh = curve(Wm_tr, Wp_tr, Wm_te, Wp_te, ks_w, "whitened")

    print("\n" + "-" * 84)
    print(f"euclidean: C2ST {eu.get(0, float('nan')):.4f} (k=0) -> "
          f"{eu[max(eu)]:.4f} (k={max(eu)}), INLP found {n_eu}")
    print(f"whitened : C2ST {wh.get(0, float('nan')):.4f} (k=0) -> "
          f"{wh[max(wh)]:.4f} (k={max(wh)}), INLP found {n_wh}")

    # Smallest k reaching near-chance in each geometry.
    def first_at_chance(d, tol=0.55):
        hits = [k for k in sorted(d) if d[k] <= tol]
        return hits[0] if hits else None

    k_eu, k_wh = first_at_chance(eu), first_at_chance(wh)
    print(f"\nfirst k with C2ST <= 0.55:  euclidean {k_eu}   whitened {k_wh}")

    print()
    if k_wh is not None and (k_eu is None or k_wh < k_eu / 4):
        print("=> DOMAIN IDENTITY IS LOW-DIMENSIONAL. In the data metric a handful")
        print("   of directions suffice, where the Euclidean run needed ~%s. The" % k_eu)
        print("   k=90 figure in §1 measures INLP's inefficiency in an anisotropic")
        print("   space, not the rank of the domain signal. This also reframes the")
        print("   frontier: removing 90 Euclidean directions destroys transfer")
        print("   largely because most of them were not domain directions.")
    elif k_wh is not None and k_eu is not None:
        print("=> COMPARABLE. Both geometries need a similar number of directions,")
        print("   so k=90 reflects the domain signal itself, not the metric.")
        print("   §1's dimensionality claim stands as written.")
    else:
        print("=> INCONCLUSIVE: C2ST did not reach chance within the k grid in at")
        print("   least one geometry. Widen the grid before drawing a conclusion.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"domain_rank_{args.run}.json"
    out.write_text(json.dumps(
        {"run": args.run, "whiten_rank": args.whiten_rank,
         "euclidean": {"c2st": eu, "n_inlp": n_eu, "first_k_at_chance": k_eu},
         "whitened": {"c2st": wh, "n_inlp": n_wh, "first_k_at_chance": k_wh}},
        indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
