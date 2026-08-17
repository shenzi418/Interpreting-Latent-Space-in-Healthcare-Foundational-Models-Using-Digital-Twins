"""WHICH PART of a class direction does the cross-domain work?

The frontier says: remove ~90 INLP domain directions and MedalCare->PTB-XL
transfer collapses (0.69 -> 0.52 on exp8) while in-domain is untouched. §5/§5b
say the subspace is depleted of both clinical features and biophysical theta.
So it is load-bearing but not interpretable, which is unsatisfying as a
mechanism -- it says what the subspace is NOT.

This tests the mechanism head-on. Fit a class probe on MedalCare alone, giving a
weight vector w. Split it into the part inside the domain subspace and the part
outside:

    w_in  = Q Q^T w          (k = 90 dims)
    w_out = w - w_in         (D - k = 934 dims)

Then use each half, on its own, as a classifier of held-out PTB-XL. The mechanism
claim predicts something specific and falsifiable:

    w_in transfers, despite being 90 dims of 1024. w_out does not, despite
    carrying the overwhelming majority of the weight norm.

If instead w_out carries the transfer, the frontier is real but the "shared
subspace" story is wrong, and the collapse would need a different explanation
(e.g. projection perturbing the probe's calibration rather than removing signal).

Controls, because the comparison is between subspaces of very different rank:
  * a rank-matched RANDOM subspace gets the same in/out split, so "90 dims retain
    some signal" cannot masquerade as a result;
  * in-domain (MedalCare) AUC is reported for every split, so a direction that is
    simply a weak classifier everywhere is visible as such.

Also reported per class:
  * energy(w, Q) = ||Q^T w||^2 / ||w||^2 -- share of the probe's weight norm that
    lies in the domain subspace. A random direction gives ~k/D = 0.088; well
    above that means the class probe is preferentially built from domain
    directions.
  * cos(w_medalcare, w_ptbxl) -- do the two domains even agree on where the class
    lives?

Writes: outputs/analysis/domain_signal/subspace_mechanism_<run>.json
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
from analysis.inlp_controls import inlp_directions  # noqa: E402
from scripts.finetune_multilabel import SHARED_LABELS  # noqa: E402


def fit_direction(Z: np.ndarray, y: np.ndarray, seed: int = SEED):
    """Unit-norm weight vector of an L2 logistic probe, or None if degenerate."""
    if len(np.unique(y)) < 2:
        return None
    clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                             random_state=seed).fit(Z, y)
    w = clf.coef_[0]
    n = np.linalg.norm(w)
    return None if n < 1e-12 else w / n


def auc_of(Z: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """AUROC of the raw projection Z@w -- no refit, so the direction is on trial.

    Refitting an intercept or rescaling here would let the evaluation repair a
    direction that does not actually carry the signal; AUROC is threshold-free,
    so the bare projection is the honest test.
    """
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, Z @ w))


def whitener(Z: np.ndarray, rank: int = 0, eps: float = 1e-3) -> np.ndarray:
    """Map into a whitened basis: S^{-1/2}, optionally truncated to `rank` PCs.

    Why whitening at all: the split w_in = Q Q^T w is orthogonal in the
    EUCLIDEAN metric, so Z@w_in and Z@w_out are uncorrelated only if the latent
    space is isotropic. It is not -- `direction_agreement.py` measures a
    participation ratio of ~71 out of 1024, and under the data covariance the
    two domains' class directions correlate at 0.43 where Euclidean cosine said
    0.04. In that geometry an "orthogonal" split leaks signal across the halves
    and the in/out comparison does not mean what it appears to.

    Why truncation: whitening full-rank is worse than not whitening. With
    D=1024 and a few thousand samples, the bottom ~950 eigenvalues are
    estimation noise; dividing by their square roots amplifies exactly that
    noise to unit scale. Measured -- full-rank whitening drove PTB-XL transfer
    from 0.70 to 0.54 and left INLP able to find only 2 domain directions
    instead of 90, i.e. it destroyed the very signal under test. Truncating to
    the leading `rank` PCs keeps the geometry correction and discards the
    unestimable tail. `rank=0` keeps full rank and is retained only to
    reproduce that failure.

    Returns a (D, r) matrix; right-multiply data by it.
    """
    S = np.cov(Z, rowvar=False)
    lam, V = np.linalg.eigh(S)
    order = np.argsort(lam)[::-1]              # eigh returns ascending
    lam, V = lam[order], V[:, order]
    r = rank if rank and rank < len(lam) else len(lam)
    lam, V = np.clip(lam[:r], 0, None), V[:, :r]
    return V / np.sqrt(lam + eps * float(np.mean(lam)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="exp8_leadfix_baseline")
    ap.add_argument("--k", type=int, default=90)
    ap.add_argument("--subsample", type=int, default=1500)
    ap.add_argument("--whiten", action="store_true",
                    help="split in the whitened (data-metric) geometry rather "
                         "than the Euclidean one. The first run of this script "
                         "used Euclidean and found no mechanism; that verdict "
                         "is only meaningful if the split is orthogonal in the "
                         "geometry predictions live in, which it was not. Both "
                         "modes are kept so the original number stays "
                         "reproducible.")
    ap.add_argument("--whiten-rank", type=int, default=128,
                    help="PCs retained when whitening. Full rank (0) amplifies "
                         "~950 noise eigendirections and destroys the signal "
                         "under test -- measured, transfer 0.70 -> 0.54. The "
                         "default sits comfortably above the ~71 participation "
                         "ratio of the space.")
    args = ap.parse_args()

    print("=" * 88)
    print(f"Which half of a class direction transfers?  [{args.run}]  k={args.k}"
          f"  metric={'whitened' if args.whiten else 'euclidean'}")
    print("=" * 88)

    Z_med_tr, Y_med_tr = load(args.run, "medalcare", "train")
    Z_ptb_tr, Y_ptb_tr = load(args.run, "ptbxl", "train")
    Z_med_te, Y_med_te = load(args.run, "medalcare", "test")
    Z_ptb_te, Y_ptb_te = load(args.run, "ptbxl", "test")

    sc = StandardScaler().fit(np.vstack([Z_med_tr, Z_ptb_tr]))
    Z_med_tr, Z_ptb_tr = sc.transform(Z_med_tr), sc.transform(Z_ptb_tr)
    Z_med_te, Z_ptb_te = sc.transform(Z_med_te), sc.transform(Z_ptb_te)

    rng = np.random.default_rng(SEED)

    def sub(Z, Y, n=args.subsample):
        if len(Z) <= n:
            return Z, Y
        i = rng.choice(len(Z), n, replace=False)
        return Z[i], Y[i]

    Z_med_tr, Y_med_tr = sub(Z_med_tr, Y_med_tr)
    Z_ptb_tr, Y_ptb_tr = sub(Z_ptb_tr, Y_ptb_tr)
    D = Z_med_tr.shape[1]

    if args.whiten:
        # Whiten on the pooled train covariance: the split must be orthogonal
        # under a metric shared by both domains, or "in" and "out" would mean
        # different things on either side of the comparison.
        Wt = whitener(np.vstack([Z_med_tr, Z_ptb_tr]), rank=args.whiten_rank)
        Z_med_tr, Z_ptb_tr = Z_med_tr @ Wt, Z_ptb_tr @ Wt
        Z_med_te, Z_ptb_te = Z_med_te @ Wt, Z_ptb_te @ Wt
        D = Z_med_tr.shape[1]
        print(f"whitened to the pooled data metric, rank {D}; the Q Q^T split "
              f"is now a true decomposition of the prediction")
        if args.k >= D:
            # k dims out of D would leave no complement to compare against.
            print(f"  WARNING: k={args.k} >= whitened rank {D}; "
                  f"the 'out' half would be empty")

    print(f"fitting {args.k} INLP domain directions on TRAIN ...")
    W = inlp_directions(Z_med_tr, Z_ptb_tr, args.k)
    Q, _ = np.linalg.qr(W.T)                       # D x k, orthonormal
    k = Q.shape[1]
    print(f"  got {k}   (random-direction energy baseline = k/D = {k / D:.4f})")

    g = np.random.default_rng(SEED + 1)
    Qr, _ = np.linalg.qr(g.normal(size=(D, k)))    # rank-matched random control

    def split(w, basis):
        w_in = basis @ (basis.T @ w)
        return w_in, w - w_in

    rows = {}
    hdr = (f"{'class':<6} {'energy':>7} {'cos':>6} | {'P:full':>7} {'P:in':>7} "
           f"{'P:out':>7} | {'P:rin':>7} {'P:rout':>7} | {'M:full':>7} {'M:in':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))

    for c, name in enumerate(SHARED_LABELS):
        y_mtr = (Y_med_tr[:, c] > 0.5).astype(int)
        y_ptr = (Y_ptb_tr[:, c] > 0.5).astype(int)
        y_pte = (Y_ptb_te[:, c] > 0.5).astype(int)
        y_mte = (Y_med_te[:, c] > 0.5).astype(int)

        w_m = fit_direction(Z_med_tr, y_mtr)
        if w_m is None:
            print(f"{name:<6} (degenerate on MedalCare train -- skipped)")
            continue
        w_p = fit_direction(Z_ptb_tr, y_ptr)
        cos_mp = float(w_m @ w_p) if w_p is not None else float("nan")

        energy = float(np.sum((Q.T @ w_m) ** 2))   # w_m is unit-norm
        w_in, w_out = split(w_m, Q)
        r_in, r_out = split(w_m, Qr)

        rows[name] = {
            "energy_in_domain_subspace": energy,
            "energy_random_baseline": k / D,
            "cos_medalcare_ptbxl": cos_mp,
            "ptbxl_full": auc_of(Z_ptb_te, y_pte, w_m),
            "ptbxl_in": auc_of(Z_ptb_te, y_pte, w_in),
            "ptbxl_out": auc_of(Z_ptb_te, y_pte, w_out),
            "ptbxl_rand_in": auc_of(Z_ptb_te, y_pte, r_in),
            "ptbxl_rand_out": auc_of(Z_ptb_te, y_pte, r_out),
            "medalcare_full": auc_of(Z_med_te, y_mte, w_m),
            "medalcare_in": auc_of(Z_med_te, y_mte, w_in),
            "medalcare_out": auc_of(Z_med_te, y_mte, w_out),
        }
        r = rows[name]
        print(f"{name:<6} {energy:>7.4f} {cos_mp:>6.3f} | {r['ptbxl_full']:>7.4f} "
              f"{r['ptbxl_in']:>7.4f} {r['ptbxl_out']:>7.4f} | "
              f"{r['ptbxl_rand_in']:>7.4f} {r['ptbxl_rand_out']:>7.4f} | "
              f"{r['medalcare_full']:>7.4f} {r['medalcare_in']:>7.4f}")

    if not rows:
        print("no usable classes")
        return 1

    def mean(key):
        vals = [v[key] for v in rows.values() if np.isfinite(v[key])]
        return float(np.mean(vals)) if vals else float("nan")

    m_in, m_out = mean("ptbxl_in"), mean("ptbxl_out")
    m_rin, m_full = mean("ptbxl_rand_in"), mean("ptbxl_full")
    m_energy = mean("energy_in_domain_subspace")

    print("\n" + "-" * 88)
    print(f"mean weight energy inside the domain subspace: {m_energy:.4f} "
          f"(random baseline {k / D:.4f}, ratio {m_energy / (k / D):.2f}x)")
    print(f"mean PTB-XL AUC   full={m_full:.4f}  in={m_in:.4f}  out={m_out:.4f}  "
          f"rand_in={m_rin:.4f}")

    print()
    if m_in > m_out and m_in > m_rin + 0.02:
        print("=> MECHANISM CONFIRMED. The k domain directions carry the")
        print("   cross-domain-transferable part of the class probe: %d dims of" % k)
        print("   %d out-transfer the remaining %d. Erasing the domain" % (D, D - k))
        print("   subspace therefore removes the transfer signal itself, which is")
        print("   exactly why the frontier trades one for the other.")
    elif m_out > m_in:
        print("=> MECHANISM NOT SUPPORTED. Transfer survives OUTSIDE the domain")
        print("   subspace, so the collapse is not simple removal of a shared")
        print("   signal. The frontier stands, but the 'shared subspace' account")
        print("   in the report must be softened -- report this.")
    else:
        print("=> INCONCLUSIVE: in-subspace transfer is not distinguishable from")
        print("   the rank-matched random control. Do not claim the mechanism.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metric = f"whitened{D}" if args.whiten else "euclidean"
    # Metric in the filename: the modes give different verdicts, and a shared
    # name would silently overwrite one with the other.
    out = OUT_DIR / f"subspace_mechanism_{args.run}_{metric}.json"
    out.write_text(json.dumps({"run": args.run, "k": k, "n_dims": D,
                               "metric": metric,
                               "whiten_rank": args.whiten_rank if args.whiten else None,
                               "random_energy_baseline": k / D,
                               "classes": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
