"""Per-coordinate quantile normalisation: the first operation to move GBDT C2ST.

`domain_mechanism.py` set out to ask what a tree uses to tell the domains apart
and answered it cleanly: destroying all cross-coordinate dependence while keeping
the marginals leaves domain AUROC at 1.0000, while destroying the marginals and
keeping the dependence drops it to 0.5000. The domain lives in the marginals.

But the arm that "destroyed the marginals" was a per-coordinate rank transform,
and that is not only a control -- it is an ALIGNMENT OPERATION. Mapping each
coordinate to its within-domain quantile forces both domains to identical
marginals on every axis. Under it the held-out GBDT C2ST is 0.5000.

Nothing else in this project has done that. Single-bandwidth MMD, multi-bandwidth
MMD, class-conditional MMD, INLP at every k, and diagonal CORAL all leave the
nonlinear C2ST at or above 0.99. The thesis's central negative claim -- "C2ST
stays ~1.0" -- has a counterexample sitting inside a control arm.

So the claim needs testing, not celebrating. An operation can drive C2ST to
chance by destroying the representation, and this project has already seen that:
§11's whitening experiment hit the domain signal hard and took the class signal
with it. C2ST at 0.5 is only interesting if the latents still carry the labels.

Four conditions, four measurements each:

  joint standardise   -- the protocol behind every prior number (baseline)
  diagonal CORAL      -- §12: per-domain mean and variance matched (2 moments)
  quantile-uniform    -- per-domain rank -> U(0,1); ALL marginal moments matched
  quantile-normal     -- per-domain rank -> N(0,1); same, but the output is
                         Gaussian rather than a bounded uniform, which usually
                         behaves better under a linear probe

  GBDT C2ST      does the nonlinear domain signal drop
  linear C2ST    does the linear one
  M->P transfer  does a MedalCare-fit probe work on PTB-XL   <- the point
  in-domain      does MedalCare class signal survive         <- the veto

THE FAILURE MODE TO WATCH FOR, stated before the result. Forcing identical
marginals is only valid alignment if the two domains SHOULD have identical
marginals. They should not: NORM prevalence differs roughly 6x between MedalCare
and PTB-XL, and a coordinate that encodes "how NORM-like is this beat" therefore
has a genuinely different marginal in each domain. Quantile-matching it does not
remove a nuisance, it destroys real class information -- the textbook failure of
marginal alignment under label shift. If in-domain holds but M->P transfer drops,
that is what happened, and the C2ST result is then a cautionary tale rather than
a method.

Transductive, like §12: the transform reads the target's unlabelled TRAIN
latents, never its labels.

Writes: outputs/analysis/domain_signal/quantile_<run>.json
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
from analysis.domain_signal_structure import (  # noqa: E402
    OUT_DIR, SEED, c2st_heldout, class_signal, load,
)
from analysis.inlp_controls import (  # noqa: E402
    inlp_directions, project_out, transfer_auc,
)


def gbdt_c2st(A_tr, B_tr, A_te, B_te) -> float:
    X = np.vstack([A_tr, B_tr])
    y = np.concatenate([np.zeros(len(A_tr)), np.ones(len(B_tr))])
    Xe = np.vstack([A_te, B_te])
    ye = np.concatenate([np.zeros(len(A_te)), np.ones(len(B_te))])
    clf = HistGradientBoostingClassifier(random_state=SEED).fit(X, y)
    return float(roc_auc_score(ye, clf.predict_proba(Xe)[:, 1]))


def quantile_pair(m_tr, m_te, p_tr, p_te, dist: str, seed: int):
    """Fit one QuantileTransformer PER DOMAIN on that domain's train latents.

    Per-domain is the whole point: a shared transform would preserve the
    difference it is supposed to remove. Each domain's test split is mapped
    through its own domain's train-fitted transform, so no test information
    reaches the transform.
    """
    n_q = min(1000, len(m_tr), len(p_tr))
    qm = QuantileTransformer(n_quantiles=n_q, output_distribution=dist,
                             subsample=10**9, random_state=seed).fit(m_tr)
    qp = QuantileTransformer(n_quantiles=n_q, output_distribution=dist,
                             subsample=10**9, random_state=seed).fit(p_tr)
    return (qm.transform(m_tr), qm.transform(m_te),
            qp.transform(p_tr), qp.transform(p_te))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="exp8_leadfix_baseline")
    ap.add_argument("--subsample", type=int, default=1500)
    ap.add_argument("--k-inlp", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()

    print("=" * 96)
    print(f"Per-coordinate quantile normalisation  [{args.run}]  "
          f"k_inlp={args.k_inlp}  {len(args.seeds)} seeds")
    print("=" * 96)

    raw = {k: load(args.run, d, s)
           for k, (d, s) in {"mtr": ("medalcare", "train"),
                             "ptr": ("ptbxl", "train"),
                             "mte": ("medalcare", "test"),
                             "pte": ("ptbxl", "test")}.items()}

    per_seed: dict[str, list[dict]] = {}

    for seed in args.seeds:
        rng = np.random.default_rng(seed)

        def sub(key, n=args.subsample):
            Z, Y = raw[key]
            if len(Z) <= n:
                return Z, Y
            i = rng.choice(len(Z), n, replace=False)
            return Z[i], Y[i]

        Z_mtr, Y_mtr = sub("mtr")
        Z_ptr, Y_ptr = sub("ptr")
        Z_mte, Y_mte = sub("mte")
        Z_pte, Y_pte = sub("pte")

        sc = StandardScaler().fit(np.vstack([Z_mtr, Z_ptr]))
        j_mtr, j_mte = sc.transform(Z_mtr), sc.transform(Z_mte)
        j_ptr, j_pte = sc.transform(Z_ptr), sc.transform(Z_pte)

        if args.k_inlp > 0:
            W = inlp_directions(j_mtr, j_ptr, args.k_inlp)
            j_mtr, j_mte = project_out(j_mtr, W), project_out(j_mte, W)
            j_ptr, j_pte = project_out(j_ptr, W), project_out(j_pte, W)

        def evaluate(name, m_tr, m_te, p_tr, p_te):
            r = {"gbdt_c2st": gbdt_c2st(m_tr, p_tr, m_te, p_te),
                 "linear_c2st": c2st_heldout(m_tr, p_tr, m_te, p_te),
                 "transfer_m2p": transfer_auc(m_tr, Y_mtr, p_te, Y_pte)[0],
                 "in_domain": class_signal(m_tr, Y_mtr, m_te, Y_mte)}
            per_seed.setdefault(name, []).append(r)
            return r

        evaluate("joint standardise", j_mtr, j_mte, j_ptr, j_pte)

        sm, sp = StandardScaler().fit(j_mtr), StandardScaler().fit(j_ptr)
        evaluate("diagonal CORAL", sm.transform(j_mtr), sm.transform(j_mte),
                 sp.transform(j_ptr), sp.transform(j_pte))

        for dist, label in (("uniform", "quantile-uniform"),
                            ("normal", "quantile-normal")):
            evaluate(label, *quantile_pair(j_mtr, j_mte, j_ptr, j_pte,
                                           dist, seed))

    keys = ("gbdt_c2st", "linear_c2st", "transfer_m2p", "in_domain")
    agg = {name: {k: (float(np.mean([r[k] for r in rs])),
                      float(np.std([r[k] for r in rs])))
                  for k in keys}
           for name, rs in per_seed.items()}

    hdr = (f"{'condition':>20} | {'GBDT C2ST':>16} {'lin C2ST':>16} | "
           f"{'M->P':>16} {'in-domain':>16}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, a in agg.items():
        cells = "  ".join(f"{a[k][0]:.4f}+-{a[k][1]:.3f}" for k in keys)
        print(f"{name:>20} |  {cells}")

    base = agg["joint standardise"]
    print("\n" + "-" * len(hdr))
    for name in ("diagonal CORAL", "quantile-uniform", "quantile-normal"):
        a = agg[name]
        print(f"{name:>20} vs baseline:  GBDT C2ST "
              f"{a['gbdt_c2st'][0] - base['gbdt_c2st'][0]:+.4f}   "
              f"M->P {a['transfer_m2p'][0] - base['transfer_m2p'][0]:+.4f}   "
              f"in-domain {a['in_domain'][0] - base['in_domain'][0]:+.4f}")

    # The verdict turns on the joint outcome, never on C2ST alone. A C2ST of 0.5
    # bought by wrecking the representation is what §11's whitening did.
    best = max(("quantile-uniform", "quantile-normal"),
               key=lambda n: agg[n]["transfer_m2p"][0])
    q = agg[best]
    d_c2 = q["gbdt_c2st"][0] - base["gbdt_c2st"][0]
    d_x = q["transfer_m2p"][0] - base["transfer_m2p"][0]
    d_in = q["in_domain"][0] - base["in_domain"][0]

    print(f"\nbest quantile variant: {best}")
    print()
    if d_c2 < -0.3 and d_x > 0.02 and d_in > -0.05:
        print("=> ALIGNMENT WITHOUT DESTRUCTION -- THE FIRST IN THIS PROJECT.")
        print(f"   Nonlinear C2ST falls {d_c2:+.4f} to near chance, M->P transfer")
        print(f"   RISES {d_x:+.4f}, and in-domain class signal is intact")
        print(f"   ({d_in:+.4f}). Every previous method traded one for the other.")
        print("   This is a headline result and needs the full control battery")
        print("   (split-half, n-target sweep, per-class, other checkpoints)")
        print("   before it goes anywhere near the thesis.")
    elif d_c2 < -0.3 and d_in > -0.05:
        print(f"=> DOMAINS BECOME INDISTINGUISHABLE ({d_c2:+.4f}) AND THE CLASS")
        print(f"   SIGNAL SURVIVES IN-DOMAIN ({d_in:+.4f}), BUT TRANSFER DOES NOT")
        print(f"   IMPROVE ({d_x:+.4f}). Most likely LABEL SHIFT: forcing identical")
        print("   marginals across domains with a 6x NORM prevalence difference")
        print("   destroys real class structure along with the nuisance. Still a")
        print("   genuine finding -- it is the cleanest demonstration in the")
        print("   project that C2ST and transfer are decoupled.")
    elif d_c2 < -0.3:
        print(f"=> C2ST COLLAPSES ({d_c2:+.4f}) BY DESTROYING THE REPRESENTATION")
        print(f"   (in-domain {d_in:+.4f}). Same failure as the §11 whitening")
        print("   experiment. Do NOT report the C2ST number without this column.")
    else:
        print("=> NO C2ST COLLAPSE. The rank-transform result in domain_mechanism")
        print("   does not reproduce under this protocol -- reconcile before")
        print("   writing either down.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"quantile_{args.run}.json"
    out.write_text(json.dumps({"run": args.run, "k_inlp": args.k_inlp,
                               "seeds": args.seeds,
                               "conditions": {n: {k: {"mean": v[0], "std": v[1]}
                                                  for k, v in a.items()}
                                              for n, a in agg.items()}},
                              indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
