"""Is alignment free after all? The frontier, redone in the data metric.

The chain that leads here:

  §1   INLP needs ~90 Euclidean directions to drive C2ST to chance, and removing
       them collapses MedalCare->PTB-XL transfer 0.69 -> 0.52. Reported as a
       tradeoff: you can align, but it costs you the thing you wanted.
  §9   the mechanism test found no shared subspace, and measured
       cos(w_med, w_ptb) ~ 0.04 -- apparently orthogonal class directions.
  R5   yet transfer is real: +0.22 over a label-shuffle null, 80% of the
       PTB-XL-trained ceiling.
  DA   `direction_agreement.py` resolves that: the space has a participation
       ratio of ~71 of 1024, and under the data covariance those "orthogonal"
       directions correlate at 0.43. Euclidean geometry was the wrong metric.
  DR   `domain_rank.py` then finds domain identity is TWO-dimensional in the
       data metric -- C2ST 1.0 -> 0.50 after removing 2 directions, held out.
       The 90 was INLP's inefficiency in an anisotropic space, not the rank of
       the domain signal.

Which sets up the question this script answers: if only 2 directions are
genuinely about domain, does removing THOSE cost transfer? The frontier's whole
force was that alignment and transfer are in conflict. That conflict was measured
while removing 88 directions that were not domain directions.

Measured at each k, in the whitened geometry throughout:
  * C2ST         -- held out, fresh classifier: is the domain gone?
  * M->P transfer -- probe fit on MedalCare, scored on PTB-XL: did it survive?
  * in-domain     -- MedalCare-only AUC: is the probe still any good at all?
  * RANDOM CONTROL at matched k, so "removing any 2 directions is harmless"
    cannot be mistaken for a result about these 2.

Two possible readings, both consequential:
  transfer survives -> alignment is FREE, the dead end was a measurement
                       artifact, and cross-domain alignment is back on the table.
  transfer dies     -> the tradeoff is real and now sharpened to its minimal
                       form: 2 directions carry both domain and transfer.

Writes: outputs/analysis/domain_signal/whitened_frontier_<run>.json
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
from scripts.finetune_multilabel import SHARED_LABELS  # noqa: E402


def c2st(A_tr, B_tr, A_te, B_te, seed=SEED) -> float:
    """Held-out domain-discrimination AUROC. 0.5 = domains indistinguishable."""
    X = np.vstack([A_tr, B_tr])
    y = np.concatenate([np.zeros(len(A_tr)), np.ones(len(B_tr))])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                             random_state=seed).fit(X, y)
    Xe = np.vstack([A_te, B_te])
    ye = np.concatenate([np.zeros(len(A_te)), np.ones(len(B_te))])
    return float(roc_auc_score(ye, clf.predict_proba(Xe)[:, 1]))


def macro_auc(Z_tr, Y_tr, Z_te, Y_te, seed=SEED) -> float:
    """Mean per-class AUROC of probes fit on (Z_tr, Y_tr), scored on the test set."""
    aucs = []
    for c in range(len(SHARED_LABELS)):
        y_tr = (Y_tr[:, c] > 0.5).astype(int)
        y_te = (Y_te[:, c] > 0.5).astype(int)
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                                 random_state=seed).fit(Z_tr, y_tr)
        aucs.append(roc_auc_score(y_te, clf.predict_proba(Z_te)[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="exp8_leadfix_baseline")
    ap.add_argument("--subsample", type=int, default=1500)
    ap.add_argument("--whiten-rank", type=int, default=128)
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 3, 4, 6, 8])
    args = ap.parse_args()

    print("=" * 92)
    print(f"The frontier in the data metric  [{args.run}]  "
          f"whiten-rank={args.whiten_rank}")
    print("=" * 92)

    Z_mtr, Y_mtr = load(args.run, "medalcare", "train")
    Z_ptr, Y_ptr = load(args.run, "ptbxl", "train")
    Z_mte, Y_mte = load(args.run, "medalcare", "test")
    Z_pte, Y_pte = load(args.run, "ptbxl", "test")

    sc = StandardScaler().fit(np.vstack([Z_mtr, Z_ptr]))
    Z_mtr, Z_ptr = sc.transform(Z_mtr), sc.transform(Z_ptr)
    Z_mte, Z_pte = sc.transform(Z_mte), sc.transform(Z_pte)

    rng = np.random.default_rng(SEED)

    def sub(Z, Y, n=args.subsample):
        if len(Z) <= n:
            return Z, Y
        i = rng.choice(len(Z), n, replace=False)
        return Z[i], Y[i]

    Z_mtr, Y_mtr = sub(Z_mtr, Y_mtr)
    Z_ptr, Y_ptr = sub(Z_ptr, Y_ptr)

    Wt = whitener(np.vstack([Z_mtr, Z_ptr]), rank=args.whiten_rank)
    Z_mtr, Z_ptr = Z_mtr @ Wt, Z_ptr @ Wt
    Z_mte, Z_pte = Z_mte @ Wt, Z_pte @ Wt
    D = Z_mtr.shape[1]
    print(f"whitened to rank {D}\n")

    kmax = max(args.ks)
    W = inlp_directions(Z_mtr, Z_ptr, kmax)
    print(f"INLP returned {len(W)} of {kmax} requested")

    g = np.random.default_rng(SEED + 3)
    R = g.normal(size=(kmax, D))
    R /= np.linalg.norm(R, axis=1, keepdims=True)

    hdr = (f"{'k':>3} | {'C2ST':>7} {'M->P':>7} {'in-dom':>7} | "
           f"{'rC2ST':>7} {'rM->P':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))

    rows = {}
    for k in args.ks:
        if k > len(W):
            print(f"{k:>3} | (INLP found only {len(W)} directions -- stopping)")
            break
        Wk, Rk = W[:k], R[:k]

        def proj(M, B):
            return M if len(B) == 0 else project_out(M, B)

        a_tr, b_tr = proj(Z_mtr, Wk), proj(Z_ptr, Wk)
        a_te, b_te = proj(Z_mte, Wk), proj(Z_pte, Wk)
        ra_tr, rb_tr = proj(Z_mtr, Rk), proj(Z_ptr, Rk)
        ra_te, rb_te = proj(Z_mte, Rk), proj(Z_pte, Rk)

        rows[k] = {
            "c2st": c2st(a_tr, b_tr, a_te, b_te),
            "transfer_m2p": macro_auc(a_tr, Y_mtr, b_te, Y_pte),
            "in_domain": macro_auc(a_tr, Y_mtr, a_te, Y_mte),
            "c2st_random": c2st(ra_tr, rb_tr, ra_te, rb_te),
            "transfer_random": macro_auc(ra_tr, Y_mtr, rb_te, Y_pte),
        }
        r = rows[k]
        print(f"{k:>3} | {r['c2st']:>7.4f} {r['transfer_m2p']:>7.4f} "
              f"{r['in_domain']:>7.4f} | {r['c2st_random']:>7.4f} "
              f"{r['transfer_random']:>7.4f}")

    ks_done = sorted(rows)
    k0, kL = ks_done[0], ks_done[-1]
    d_c2st = rows[kL]["c2st"] - rows[k0]["c2st"]
    d_tx = rows[kL]["transfer_m2p"] - rows[k0]["transfer_m2p"]
    d_in = rows[kL]["in_domain"] - rows[k0]["in_domain"]

    print("\n" + "-" * 92)
    print(f"k={k0} -> k={kL}:  C2ST {d_c2st:+.4f}   M->P {d_tx:+.4f}   "
          f"in-domain {d_in:+.4f}")

    # The pivotal k: first one at chance C2ST. Cost is read there, not at kL.
    at_chance = [k for k in ks_done if rows[k]["c2st"] <= 0.55]
    kc = at_chance[0] if at_chance else None

    print()
    if kc is not None:
        cost = rows[kc]["transfer_m2p"] - rows[k0]["transfer_m2p"]
        print(f"domain removed at k={kc} (C2ST {rows[kc]['c2st']:.4f}); "
              f"transfer there {rows[kc]['transfer_m2p']:.4f} "
              f"vs {rows[k0]['transfer_m2p']:.4f} at k=0  ({cost:+.4f})")
        if cost > -0.03:
            print("\n=> ALIGNMENT IS ESSENTIALLY FREE IN THE RIGHT METRIC.")
            print("   C2ST reaches chance while transfer is preserved. The")
            print("   'tradeoff' in §1 was the cost of removing ~88 directions")
            print("   that were never domain directions. The alignment dead end")
            print("   should be re-examined: the obstacle measured was the")
            print("   metric, not the data.")
        elif cost > -0.10:
            print("\n=> PARTIAL. Removing the true domain directions costs some")
            print("   transfer, but far less than the Euclidean frontier implied.")
            print("   Report the corrected cost, not the k=90 figure.")
        else:
            print("\n=> THE TRADEOFF IS REAL, AND NOW MINIMAL. Even the 2-3 true")
            print("   domain directions carry the transfer. This is a much")
            print("   stronger version of §1: not 90 diffuse directions but a")
            print("   tiny subspace doing both jobs.")
    else:
        print("=> C2ST never reached chance on this grid -- widen --ks.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"whitened_frontier_{args.run}.json"
    out.write_text(json.dumps({"run": args.run, "whiten_rank": args.whiten_rank,
                               "n_dims": D, "n_inlp": len(W),
                               "ks": args.ks, "curve": rows,
                               "k_at_chance": kc}, indent=2, default=str),
                   encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
