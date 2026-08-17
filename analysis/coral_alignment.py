"""Diagonal and full CORAL: can matching second-order statistics do what INLP could not?

`nonlinear_structure.py` found that after 103 INLP directions are removed -- every
linearly decodable domain direction -- what still separates the domains at GBDT
AUROC 0.9998 is a DISPERSION mismatch. Means agree (mean |d_mean| = 0.076) while
per-coordinate standard deviations do not: ratios of 3.13, 3.27, 3.72 on some
coordinates and 0.33, 0.60 on others. Not a uniform scale factor -- a per-axis
disagreement about how wide the distribution is.

That is invisible to every alignment method this project has tried. MMD with an
RBF kernel is sensitive to it in principle, but the ccMMD runs penalised a batch
statistic during training rather than correcting the representation; INLP removes
directions, which cannot fix a scale difference at all (projecting out an axis
deletes it rather than rescaling it).

The obvious untried operation is to rescale rather than delete:

  DIAGONAL CORAL  -- standardise each coordinate WITHIN each domain, so both
                     domains have zero mean and unit variance per axis. Cheap,
                     invertible, removes no direction.
  FULL CORAL      -- whiten with the source covariance and recolour with the
                     target's, matching the whole second-order structure rather
                     than just the diagonal. Truncated to `--coral-rank`; at
                     D=1024 with a few thousand samples the tail eigenvalues are
                     noise, and the whitening experiment in §11 showed what
                     happens when you divide by them.

Two things get measured, and BOTH matter:

  * GBDT C2ST -- does the nonlinear domain signal actually drop? If dispersion is
    what the tree was using, matching it should hurt the tree. If C2ST stays at
    ~1.0, the dispersion was a symptom and something else carries the domain.
  * M->P TRANSFER -- does it survive, or improve? A probe fit on MedalCare sees
    PTB-XL coordinates at up to 3x the scale it was fit on; its logits are then
    systematically miscalibrated. Rescaling could plausibly HELP transfer, which
    no operation in this project has yet done.

HONEST FRAMING OF WHAT THIS USES. Per-domain statistics are estimated on the
TARGET's unlabelled training latents -- never its labels. That is the standard
transductive unsupervised-domain-adaptation setting (AdaBN, CORAL), not a leak,
but it is a strictly weaker claim than "the representation is aligned": it says
the domains can be BROUGHT into alignment given unlabelled target data, not that
the foundation model already encodes them compatibly. The thesis must say which.

Control: a probe's transfer is also reported under joint standardisation (the
protocol every previous number in this project used) so the delta is attributable.

Writes: outputs/analysis/domain_signal/coral_<run>.json
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
from analysis.domain_signal_structure import (  # noqa: E402
    OUT_DIR, SEED, c2st_heldout, class_signal, load,
)
from analysis.inlp_controls import (  # noqa: E402
    inlp_directions, project_out, transfer_auc,
)


def gbdt_c2st(A_tr, B_tr, A_te, B_te) -> float:
    """Held-out domain AUROC from a tree ensemble (see tradeoff_frontier)."""
    X = np.vstack([A_tr, B_tr])
    y = np.concatenate([np.zeros(len(A_tr)), np.ones(len(B_tr))])
    Xe = np.vstack([A_te, B_te])
    ye = np.concatenate([np.zeros(len(A_te)), np.ones(len(B_te))])
    clf = HistGradientBoostingClassifier(random_state=SEED).fit(X, y)
    return float(roc_auc_score(ye, clf.predict_proba(Xe)[:, 1]))


def _sqrt_cov(Z: np.ndarray, rank: int, eps: float, inverse: bool):
    """Truncated S^{1/2} or S^{-1/2} of Z's covariance.

    Truncation is not optional here for the same reason it was not optional in
    `subspace_mechanism.whitener`: the bottom ~900 eigenvalues of a 1024-d
    covariance estimated from a few thousand samples are estimation noise, and
    the inverse square root amplifies exactly those to unit scale.
    """
    S = np.cov(Z, rowvar=False)
    lam, V = np.linalg.eigh(S)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    r = rank if rank and rank < len(lam) else len(lam)
    lam, V = np.clip(lam[:r], 0, None), V[:, :r]
    floor = eps * float(np.mean(lam))
    s = np.sqrt(lam + floor)
    return (V / s) if inverse else (V * s)


def coral(src_tr, src_te, tgt_tr, tgt_te, rank: int, eps: float = 1e-3):
    """Full CORAL: map the SOURCE onto the target's second-order statistics.

    Source is whitened by its own covariance and recoloured by the target's, so
    the target is left untouched (important: the probe is fit on the source, and
    moving the source is the operation that makes its geometry match what the
    probe will meet at test time). Both domains are mean-centred on their own
    means first, since CORAL is a second-order correction.
    """
    mu_s, mu_t = src_tr.mean(0), tgt_tr.mean(0)
    Ws = _sqrt_cov(src_tr - mu_s, rank, eps, inverse=True)      # D x r
    Ct = _sqrt_cov(tgt_tr - mu_t, rank, eps, inverse=False)     # D x r
    A = Ws @ Ct.T                                               # D x D
    return ((src_tr - mu_s) @ A, (src_te - mu_s) @ A,
            tgt_tr - mu_t, tgt_te - mu_t)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="exp8_leadfix_baseline")
    ap.add_argument("--subsample", type=int, default=1500)
    ap.add_argument("--coral-rank", type=int, default=128)
    ap.add_argument("--k-inlp", type=int, default=0,
                    help="optionally remove k INLP directions first, to ask "
                         "whether CORAL fixes the residual that survives INLP")
    args = ap.parse_args()

    print("=" * 92)
    print(f"Second-order domain correction (CORAL)  [{args.run}]  "
          f"rank={args.coral_rank}  k_inlp={args.k_inlp}")
    print("=" * 92)

    Z_mtr, Y_mtr = load(args.run, "medalcare", "train")
    Z_ptr, Y_ptr = load(args.run, "ptbxl", "train")
    Z_mte, Y_mte = load(args.run, "medalcare", "test")
    Z_pte, Y_pte = load(args.run, "ptbxl", "test")

    rng = np.random.default_rng(SEED)

    def sub(Z, Y, n=args.subsample):
        if len(Z) <= n:
            return Z, Y
        i = rng.choice(len(Z), n, replace=False)
        return Z[i], Y[i]

    Z_mtr, Y_mtr = sub(Z_mtr, Y_mtr)
    Z_ptr, Y_ptr = sub(Z_ptr, Y_ptr)
    Z_mte, Y_mte = sub(Z_mte, Y_mte)
    Z_pte, Y_pte = sub(Z_pte, Y_pte)

    results = {}

    def evaluate(name, m_tr, m_te, p_tr, p_te):
        c2 = gbdt_c2st(m_tr, p_tr, m_te, p_te)
        lin = c2st_heldout(m_tr, p_tr, m_te, p_te)
        xfer = transfer_auc(m_tr, Y_mtr, p_te, Y_pte)[0]
        ind = class_signal(m_tr, Y_mtr, m_te, Y_mte)
        results[name] = {"gbdt_c2st": c2, "linear_c2st": lin,
                         "transfer_m2p": xfer, "in_domain": ind}
        print(f"{name:>22} | {c2:>9.4f} {lin:>9.4f} | {xfer:>9.4f} {ind:>9.4f}")
        return results[name]

    hdr = (f"{'condition':>22} | {'GBDT C2ST':>9} {'lin C2ST':>9} | "
           f"{'M->P':>9} {'in-dom':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))

    # --- baseline: joint standardisation, the protocol behind every prior number
    sc = StandardScaler().fit(np.vstack([Z_mtr, Z_ptr]))
    j_mtr, j_mte = sc.transform(Z_mtr), sc.transform(Z_mte)
    j_ptr, j_pte = sc.transform(Z_ptr), sc.transform(Z_pte)

    if args.k_inlp > 0:
        W = inlp_directions(j_mtr, j_ptr, args.k_inlp)
        print(f"(removing {len(W)} INLP directions from all conditions)\n")
        j_mtr, j_mte = project_out(j_mtr, W), project_out(j_mte, W)
        j_ptr, j_pte = project_out(j_ptr, W), project_out(j_pte, W)

    base = evaluate("joint standardise", j_mtr, j_mte, j_ptr, j_pte)

    # --- diagonal CORAL: per-domain per-coordinate standardisation
    sm = StandardScaler().fit(j_mtr)
    sp = StandardScaler().fit(j_ptr)
    diag = evaluate("diagonal CORAL",
                    sm.transform(j_mtr), sm.transform(j_mte),
                    sp.transform(j_ptr), sp.transform(j_pte))

    # --- full CORAL
    c_mtr, c_mte, c_ptr, c_pte = coral(j_mtr, j_mte, j_ptr, j_pte,
                                       rank=args.coral_rank)
    full = evaluate(f"full CORAL r={args.coral_rank}",
                    c_mtr, c_mte, c_ptr, c_pte)

    print("\n" + "-" * 92)
    for name, r in (("diagonal", diag), ("full", full)):
        print(f"{name:>9} CORAL vs baseline:  GBDT C2ST "
              f"{r['gbdt_c2st'] - base['gbdt_c2st']:+.4f}   "
              f"M->P {r['transfer_m2p'] - base['transfer_m2p']:+.4f}   "
              f"in-domain {r['in_domain'] - base['in_domain']:+.4f}")

    best = max((diag, full), key=lambda r: r["transfer_m2p"])
    d_x = best["transfer_m2p"] - base["transfer_m2p"]
    d_c2 = min(diag["gbdt_c2st"], full["gbdt_c2st"]) - base["gbdt_c2st"]

    print()
    if d_x > 0.02 and d_c2 < -0.05:
        print("=> SECOND-ORDER CORRECTION HELPS ON BOTH AXES. Matching dispersion")
        print(f"   lifts transfer by {d_x:+.4f} AND drops nonlinear domain")
        print(f"   separability by {d_c2:+.4f}. This is the first operation in the")
        print("   project to move alignment and transfer in the SAME direction --")
        print("   INLP moved them in opposite directions and bought nothing.")
    elif d_x > 0.02:
        print(f"=> TRANSFER IMPROVES ({d_x:+.4f}) WITHOUT ALIGNING. The domains stay")
        print("   fully separable, so this is not alignment -- it is calibration:")
        print("   the probe was being fed target features at the wrong scale.")
        print("   Worth reporting as a practical transfer result, but it does not")
        print("   revive the alignment claim.")
    elif d_c2 < -0.05:
        print(f"=> DOMAIN SIGNAL DROPS ({d_c2:+.4f}) BUT TRANSFER DOES NOT FOLLOW.")
        print("   Dispersion was part of what the tree used, yet fixing it does")
        print("   not help the probe. Consistent with the frontier: alignment and")
        print("   transfer are not coupled the way the field assumes.")
    else:
        print("=> NEITHER AXIS MOVES. The dispersion mismatch is a symptom, not")
        print("   the mechanism. Whatever the tree is using survives second-order")
        print("   correction, which points at genuinely higher-order structure.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"coral_{args.run}.json"
    out.write_text(json.dumps({"run": args.run, "coral_rank": args.coral_rank,
                               "k_inlp": args.k_inlp, "conditions": results},
                              indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
