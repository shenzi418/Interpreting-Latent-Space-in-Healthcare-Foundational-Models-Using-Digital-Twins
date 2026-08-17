"""Is the domain really gone at k=2, or did the linear C2ST just fail?

`whitened_frontier.py` reports C2ST = 0.5000 after removing 2 whitened INLP
directions, with transfer preserved -- which would mean alignment is free and
would overturn the project's central negative result. Before that gets written
up, the number has to survive the objection it invites.

EXACTLY 0.5000 is suspicious. INLP halts precisely when the logistic weight
collapses to ~0, and a logistic C2ST on the same projected data will collapse the
same way, emitting a constant probability -- for which `roc_auc_score` returns
exactly 0.5. "No LINEAR direction remains" and "the domains are indistinguishable"
would then be indistinguishable to this instrument, and only the first is true by
construction. INLP guarantees it.

So the domain signal is re-measured with instruments that do not share the
failure mode:

  * GRADIENT BOOSTING C2ST -- axis-aligned splits, no reliance on a linear
    margin. Catches domain information stored nonlinearly.
  * MLP C2ST -- a nonlinear decision surface of a different family, so a null
    result is not one classifier's idiosyncrasy.
  * kNN C2ST -- fully nonparametric; detects local clustering by domain even
    when no global surface separates them.
  * MMD (multi-bandwidth RBF) with a permutation p-value -- a two-sample test
    that never fits a classifier at all, so classifier degeneracy cannot produce
    a false negative.

All are fit on projected TRAIN and scored on projected HELD-OUT data, matching
the frontier's protocol. Run at k=0 and at the k where linear C2ST hit chance.

Verdict logic: if the nonlinear tests also sit at chance, the domain is genuinely
gone and the free-alignment result stands. If any of them recovers the domain,
then k=2 removed only the LINEARLY DECODABLE part -- the honest claim shrinks to
that, and "alignment is free" must not be written.

Writes: outputs/analysis/domain_signal/nonlinear_c2st_<run>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
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


def _xy(A, B):
    return np.vstack([A, B]), np.concatenate([np.zeros(len(A)), np.ones(len(B))])


def c2st_with(clf, A_tr, B_tr, A_te, B_te) -> float:
    """Held-out domain AUROC for an arbitrary sklearn classifier."""
    X, y = _xy(A_tr, B_tr)
    Xe, ye = _xy(A_te, B_te)
    clf.fit(X, y)
    p = clf.predict_proba(Xe)[:, 1]
    return float(roc_auc_score(ye, p))


def is_constant(clf, A_tr, B_tr, A_te, B_te) -> bool:
    """Did the classifier emit a (near-)constant score? The 0.5000 failure mode."""
    X, y = _xy(A_tr, B_tr)
    Xe, _ = _xy(A_te, B_te)
    clf.fit(X, y)
    p = clf.predict_proba(Xe)[:, 1]
    return bool(np.std(p) < 1e-6)


def mmd_permutation(A, B, n_perm=200, seed=SEED):
    """Multi-bandwidth RBF MMD^2 with a permutation p-value.

    Classifier-free, so it cannot fail the way a degenerate logistic does.
    Bandwidths follow the median heuristic scaled over a range, matching the
    project's existing MMD practice.
    """
    rng = np.random.default_rng(seed)
    n = min(len(A), len(B), 800)          # O(n^2) kernels; cap for tractability
    A = A[rng.choice(len(A), n, replace=False)]
    B = B[rng.choice(len(B), n, replace=False)]
    Z = np.vstack([A, B])

    d2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
    med = np.median(d2[d2 > 0])
    K = sum(np.exp(-d2 / (s * med)) for s in (0.25, 0.5, 1.0, 2.0, 4.0))

    def stat(idx_a, idx_b):
        Kaa, Kbb = K[np.ix_(idx_a, idx_a)], K[np.ix_(idx_b, idx_b)]
        Kab = K[np.ix_(idx_a, idx_b)]
        na, nb = len(idx_a), len(idx_b)
        # Unbiased: drop the diagonal self-similarity terms.
        return ((Kaa.sum() - np.trace(Kaa)) / (na * (na - 1))
                + (Kbb.sum() - np.trace(Kbb)) / (nb * (nb - 1))
                - 2 * Kab.mean())

    ia, ib = np.arange(n), np.arange(n, 2 * n)
    obs = stat(ia, ib)
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(2 * n)
        null.append(stat(perm[:n], perm[n:]))
    p = float((1 + np.sum(np.array(null) >= obs)) / (1 + n_perm))
    return float(obs), p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="exp8_leadfix_baseline")
    ap.add_argument("--subsample", type=int, default=1500)
    ap.add_argument("--whiten-rank", type=int, default=128)
    ap.add_argument("--no-whiten", action="store_true",
                    help="skip whitening and run in the raw Euclidean geometry. "
                         "Needed to apply this same degeneracy check to §1's "
                         "original k=90 result, which was also measured with a "
                         "linear C2ST and inherits the identical objection.")
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    print("=" * 92)
    print(f"Is the domain really gone, or did the linear test fail?  [{args.run}]")
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

    Wt = whitener(np.vstack([Z_mtr, Z_ptr]), rank=args.whiten_rank)
    if args.no_whiten:
        print("NOT whitened -- raw Euclidean geometry (§1 protocol)")
    else:
        Z_mtr, Z_ptr = Z_mtr @ Wt, Z_ptr @ Wt
        Z_mte, Z_pte = Z_mte @ Wt, Z_pte @ Wt
        print(f"whitened to rank {Z_mtr.shape[1]}")

    W = inlp_directions(Z_mtr, Z_ptr, max(args.ks) if max(args.ks) else 1)
    print(f"INLP returned {len(W)} directions\n")

    def make():
        return {
            "linear": LogisticRegression(max_iter=3000, class_weight="balanced",
                                         random_state=SEED),
            "gbdt": HistGradientBoostingClassifier(random_state=SEED),
            "mlp": MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=600,
                                 random_state=SEED),
            "knn": KNeighborsClassifier(n_neighbors=15),
        }

    hdr = (f"{'k':>3} | {'linear':>8} {'gbdt':>8} {'mlp':>8} {'knn':>8} | "
           f"{'MMD^2':>10} {'p':>7} | {'lin const?':>10}")
    print(hdr)
    print("-" * len(hdr))

    rows = {}
    for k in args.ks:
        if k > len(W):
            print(f"{k:>3} | (only {len(W)} INLP directions available)")
            break
        Wk = W[:k]

        def proj(M):
            return M if k == 0 else project_out(M, Wk)

        a_tr, b_tr = proj(Z_mtr), proj(Z_ptr)
        a_te, b_te = proj(Z_mte), proj(Z_pte)

        scores = {n: c2st_with(c, a_tr, b_tr, a_te, b_te)
                  for n, c in make().items()}
        const = is_constant(make()["linear"], a_tr, b_tr, a_te, b_te)
        mmd, p = mmd_permutation(a_te, b_te)

        rows[k] = {**scores, "mmd2": mmd, "mmd_p": p, "linear_constant": const}
        print(f"{k:>3} | {scores['linear']:>8.4f} {scores['gbdt']:>8.4f} "
              f"{scores['mlp']:>8.4f} {scores['knn']:>8.4f} | "
              f"{mmd:>10.5f} {p:>7.3f} | {str(const):>10}")

    kL = max(rows)
    r = rows[kL]
    nonlinear = [r["gbdt"], r["mlp"], r["knn"]]
    worst = max(nonlinear)

    print("\n" + "-" * 92)
    print(f"at k={kL}: linear C2ST {r['linear']:.4f}, but the strongest "
          f"nonlinear detector reads {worst:.4f}")
    print(f"MMD^2 = {r['mmd2']:.5f} (permutation p = {r['mmd_p']:.3f})")

    print()
    if worst <= 0.60 and r["mmd_p"] > 0.05:
        print("=> THE DOMAIN IS GENUINELY GONE. Nonlinear classifiers and a")
        print("   classifier-free MMD test all fail to separate the domains, so")
        print("   the linear 0.5000 was not an artifact of degeneracy. Removing")
        print("   the true (low-dimensional) domain directions aligns the")
        print("   representations while preserving transfer -- alignment is free.")
    elif worst > 0.60:
        print("=> LINEAR-ONLY REMOVAL. Nonlinear detectors still separate the")
        print("   domains, so k=2 erased the linearly decodable domain signal and")
        print("   nothing more. The linear C2ST of 0.5000 IS partly degeneracy.")
        print("   Claim only 'linear domain identity is 2-dimensional and can be")
        print("   removed without cost' -- NOT that alignment is achieved.")
    else:
        print("=> MIXED: classifiers are at chance but MMD still rejects, so a")
        print("   distributional difference survives that no classifier exploits.")
        print("   Report both; do not claim full alignment.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    geom = "euclidean" if args.no_whiten else f"whitened{args.whiten_rank}"
    out = OUT_DIR / f"nonlinear_c2st_{args.run}_{geom}.json"
    out.write_text(json.dumps({"run": args.run, "geometry": geom,
                               "whiten_rank": None if args.no_whiten
                               else args.whiten_rank,
                               "n_inlp": len(W), "curve": rows},
                              indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
