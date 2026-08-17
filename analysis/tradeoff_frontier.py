"""The alignment/transfer tradeoff frontier.

`inlp_controls.py` established three facts at k=90 on exp7:
  * C2ST 1.0000 -> 0.5001   (domain identity fully erased)
  * rank-matched RANDOM removal leaves C2ST = 1.0000 (so it is not capacity loss)
  * M->P transfer macro-AUC 0.8848 -> 0.6189, while random removal gives 0.8817

So alignment is achievable and it is COSTLY. This script traces the whole
frontier so the tradeoff can be plotted and quantified rather than asserted at a
single k.

At each k we record, with everything fit on TRAIN only:
  * held-out C2ST                       -- how aligned the space is
  * in-domain MedalCare class macro-AUC -- is the task signal still there at all
  * M->P cross-domain transfer macro-AUC-- does alignment buy transfer
  * the same three under rank-matched RANDOM projection -- the A4 control

The interesting quantity is the CONTRAST between in-domain and cross-domain.
If in-domain class AUC is flat while cross-domain transfer collapses, then the
directions that encode domain identity are precisely the ones carrying
cross-domain generalisable class information -- they are load-bearing, and
erasing them is not neutral but actively destructive.

Hypothesised mechanism (testable, and consistent with the measured label shift
of JS = 0.145 between the two label marginals): because class and domain are
correlated, a domain classifier partly latches onto class-predictive directions.
Removing them therefore removes shared class structure. Under this account
"alignment failed" is a category error -- alignment succeeds and takes the
transferable signal with it.

Writes: outputs/analysis/domain_signal/tradeoff_frontier_<run>.json
        outputs/analysis/domain_signal/tradeoff_frontier_<run>.png
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
from analysis.domain_signal_structure import (  # noqa: E402
    OUT_DIR, SEED, c2st_heldout, class_signal, load,
)
from analysis.inlp_controls import (  # noqa: E402
    inlp_directions, project_out, transfer_auc,
)


def gbdt_c2st(A_tr, B_tr, A_te, B_te) -> float:
    """Held-out domain AUROC from a gradient-boosted tree.

    The default `c2st_heldout` is a logistic regression, which can only report
    whether domain identity is LINEARLY decodable -- and INLP removes directions
    until no linear direction remains, so the two share a blind spot and the
    pairing is close to circular. Measured on exp8_leadfix_baseline at k=90:
    linear C2ST reads 0.5132 ("aligned") while this reads 0.9999 (not aligned at
    all). See `analysis/nonlinear_c2st.py` and report §11.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    X = np.vstack([A_tr, B_tr])
    y = np.concatenate([np.zeros(len(A_tr)), np.ones(len(B_tr))])
    Xe = np.vstack([A_te, B_te])
    ye = np.concatenate([np.zeros(len(A_te)), np.ones(len(B_te))])
    clf = HistGradientBoostingClassifier(random_state=SEED).fit(X, y)
    return float(roc_auc_score(ye, clf.predict_proba(Xe)[:, 1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="exp7")
    ap.add_argument("--ks", type=int, nargs="*",
                    default=[0, 5, 10, 20, 30, 40, 60, 80, 90, 100])
    ap.add_argument("--subsample", type=int, default=1500)
    ap.add_argument("--random-reps", type=int, default=2)
    ap.add_argument("--tag", default="",
                    help="suffix for the output filenames. REQUIRED in practice "
                         "when --ks is customised: outputs are named per-run, so "
                         "a second sweep of the same run silently overwrites the "
                         "first (this bit me -- a fine knee sweep clobbered the "
                         "coarse frontier the report cited).")
    ap.add_argument("--c2st", choices=["linear", "gbdt"], default="linear",
                    help="detector for the alignment axis. 'linear' reproduces "
                         "the published frontier but only measures LINEAR "
                         "decodability, which INLP is built to destroy; 'gbdt' "
                         "asks whether the domains are separable at all.")
    args = ap.parse_args()

    c2st_fn = gbdt_c2st if args.c2st == "gbdt" else c2st_heldout
    # The detector goes in the filename: linear and gbdt give opposite verdicts
    # at the same k, so a shared name would overwrite one with the other.
    suffix = f"{args.run}{('_' + args.tag) if args.tag else ''}"
    if args.c2st != "linear":
        suffix += f"_{args.c2st}"
    print("=" * 92)
    print(f"Alignment / transfer tradeoff frontier  [{args.run}]  "
          f"C2ST={args.c2st}")
    print("=" * 92)

    Z_med_tr, P_med_tr = load(args.run, "medalcare", "train")
    Z_med_te, P_med_te = load(args.run, "medalcare", "test")
    Z_ptb_tr, P_ptb_tr = load(args.run, "ptbxl", "train")
    Z_ptb_te, P_ptb_te = load(args.run, "ptbxl", "test")

    sc = StandardScaler().fit(np.vstack([Z_med_tr, Z_ptb_tr]))
    Z_med_tr, Z_med_te = sc.transform(Z_med_tr), sc.transform(Z_med_te)
    Z_ptb_tr, Z_ptb_te = sc.transform(Z_ptb_tr), sc.transform(Z_ptb_te)

    rng = np.random.default_rng(SEED)

    def sub(Z, P, n=args.subsample):
        if len(Z) <= n:
            return Z, P
        i = rng.choice(len(Z), n, replace=False)
        return Z[i], P[i]

    Z_med_tr, P_med_tr = sub(Z_med_tr, P_med_tr)
    Z_med_te, P_med_te = sub(Z_med_te, P_med_te)
    Z_ptb_tr, P_ptb_tr = sub(Z_ptb_tr, P_ptb_tr)
    Z_ptb_te, P_ptb_te = sub(Z_ptb_te, P_ptb_te)
    D = Z_med_tr.shape[1]

    # Fit the deepest direction set ONCE; every smaller k is a prefix of it.
    # INLP is greedy and sequential, so the first k directions of the k_max run
    # are exactly what a length-k run would have produced.
    k_max = max(args.ks)
    print(f"fitting {k_max} INLP directions (prefixes reused for smaller k)...")
    W_all = inlp_directions(Z_med_tr, Z_ptb_tr, k_max)
    print(f"  got {len(W_all)}")

    rows = []
    hdr = (f"{'k':>4} | {'C2ST':>7} {'in-dom':>7} {'M->P':>7} | "
           f"{'rndC2ST':>8} {'rnd_in':>7} {'rndM->P':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))

    for k in args.ks:
        W = W_all[:k]
        pm_tr, pm_te = project_out(Z_med_tr, W), project_out(Z_med_te, W)
        pp_tr, pp_te = project_out(Z_ptb_tr, W), project_out(Z_ptb_te, W)
        c2 = c2st_fn(pm_tr, pp_tr, pm_te, pp_te)
        ind = class_signal(pm_tr, P_med_tr, pm_te, P_med_te)
        xfer = transfer_auc(pm_tr, P_med_tr, pp_te, P_ptb_te)[0]

        # rank-matched random control
        rc2, rin, rxf = [], [], []
        for rep in range(args.random_reps if k > 0 else 1):
            if k == 0:
                rc2, rin, rxf = [c2], [ind], [xfer]
                break
            g = np.random.default_rng(SEED + rep)
            R = g.normal(size=(k, D))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            rm_tr, rm_te = project_out(Z_med_tr, R), project_out(Z_med_te, R)
            rp_tr, rp_te = project_out(Z_ptb_tr, R), project_out(Z_ptb_te, R)
            rc2.append(c2st_fn(rm_tr, rp_tr, rm_te, rp_te))
            rin.append(class_signal(rm_tr, P_med_tr, rm_te, P_med_te))
            rxf.append(transfer_auc(rm_tr, P_med_tr, rp_te, P_ptb_te)[0])

        rows.append({
            "k": k, "c2st": c2, "in_domain_macro_auc": ind,
            "transfer_macro_auc": xfer,
            "random_c2st": float(np.mean(rc2)),
            "random_in_domain_macro_auc": float(np.mean(rin)),
            "random_transfer_macro_auc": float(np.mean(rxf)),
        })
        print(f"{k:>4} | {c2:>7.4f} {ind:>7.4f} {xfer:>7.4f} | "
              f"{np.mean(rc2):>8.4f} {np.mean(rin):>7.4f} {np.mean(rxf):>8.4f}")

    base, last = rows[0], rows[-1]
    print("\n" + "-" * 92)
    print(f"C2ST      {base['c2st']:.4f} -> {last['c2st']:.4f}   "
          f"(random at k={last['k']}: {last['random_c2st']:.4f})")
    print(f"in-domain {base['in_domain_macro_auc']:.4f} -> "
          f"{last['in_domain_macro_auc']:.4f}")
    print(f"M->P      {base['transfer_macro_auc']:.4f} -> "
          f"{last['transfer_macro_auc']:.4f}   "
          f"(random at k={last['k']}: {last['random_transfer_macro_auc']:.4f})")
    d_in = last["in_domain_macro_auc"] - base["in_domain_macro_auc"]
    d_x = last["transfer_macro_auc"] - base["transfer_macro_auc"]
    d_c2 = last["c2st"] - base["c2st"]
    print(f"\ndelta in-domain {d_in:+.4f}   vs   delta cross-domain {d_x:+.4f}")

    # The alignment axis has to be checked before any tradeoff is announced: a
    # "tradeoff" requires that something was BOUGHT. Under --c2st gbdt the
    # detector never moves off 1.0 while transfer collapses, which is a pure
    # loss, not an exchange -- and the old message would have called that a
    # subspace-sharing result.
    if abs(d_c2) < 0.05:
        print("  => NO ALIGNMENT WAS PURCHASED. C2ST is unmoved across the whole")
        print("     sweep while transfer falls. This is not a tradeoff: the")
        print("     removal costs transfer and buys nothing measurable on this")
        print("     detector. If this is the gbdt run, it means INLP never")
        print("     touched the separability that matters (report §11).")
    elif abs(d_in) < 0.05 <= abs(d_x):
        print("  => The removed directions are LOAD-BEARING FOR TRANSFER ONLY:")
        print("     erasing them is nearly free in-domain and expensive across")
        print("     domains -- an alignment/transfer exchange in this metric.")
        print("     NOTE: with --c2st linear this only shows the directions are")
        print("     load-bearing for LINEAR decodability; see §11 before reading")
        print("     it as a statement about alignment.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"tradeoff_frontier_{suffix}.json"
    out.write_text(json.dumps({"run": args.run, "ks": args.ks,
                               "c2st_detector": args.c2st, "rows": rows},
                              indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ks = [r["k"] for r in rows]
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.plot(ks, [r["c2st"] for r in rows], "o-", label="C2ST (domain separability)")
        ax.plot(ks, [r["transfer_macro_auc"] for r in rows], "s-",
                label="M$\\rightarrow$P transfer AUC")
        ax.plot(ks, [r["in_domain_macro_auc"] for r in rows], "^-",
                label="in-domain class AUC")
        ax.plot(ks, [r["random_c2st"] for r in rows], "o--", alpha=0.45,
                label="C2ST, random control")
        ax.plot(ks, [r["random_transfer_macro_auc"] for r in rows], "s--", alpha=0.45,
                label="transfer, random control")
        ax.axhline(0.5, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("INLP directions removed (k)")
        ax.set_ylabel("AUC / AUROC")
        ax.set_title(f"Alignment is achievable and costly [{args.run}]")
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        png = OUT_DIR / f"tradeoff_frontier_{suffix}.png"
        fig.savefig(png, dpi=160)
        print(f"wrote {png}")
    except Exception as exc:  # pragma: no cover - plotting is a nicety
        print(f"(plot skipped: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
