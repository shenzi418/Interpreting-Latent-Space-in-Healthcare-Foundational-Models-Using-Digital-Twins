"""Is cross-domain transfer real, or a prevalence artifact?

§9 of the breakthrough report found that the MedalCare and PTB-XL class
directions are nearly ORTHOGONAL (cos = 0.001 / 0.068 / 0.053 for NORM/MI/CD),
yet a MedalCare-trained linear probe still scores ~0.69 macro-AUC on PTB-XL.
Those two facts sit badly together. Either transfer rides on something the cosine
does not see, or the 0.69 is not measuring what "transfer" is supposed to mean.

AUROC is prevalence-insensitive, so a shifted base rate cannot inflate it
directly -- but a probe can still score above chance by tracking a nuisance
direction that happens to correlate with the label in BOTH domains (recording
site, amplitude scale, heart rate) without sharing any class-specific structure.

Three controls, each isolating a different explanation:

  1. LABEL SHUFFLE. Refit the probe on permuted MedalCare labels, score on real
     PTB-XL. Anything above 0.5 here is leakage through the evaluation itself.
     This is the null the 0.69 must beat.

  2. PTB-XL-TRAINED CEILING. Fit on PTB-XL train, score PTB-XL test. The gap
     between this and the M->P number is the actual cost of crossing domains,
     and it is the honest denominator: 0.69 means something different if the
     in-domain ceiling is 0.95 than if it is 0.72.

  3. RANDOM-DIRECTION FLOOR. Score PTB-XL with random unit directions. In a
     1024-d space with correlated coordinates, a random direction is not
     guaranteed to sit at 0.5, and the spread tells us how much of the 0.69 is
     simply "any direction does this".

Reported per class and as a macro mean, with the transfer number alongside all
three references so it cannot be read in isolation.

Writes: outputs/analysis/domain_signal/transfer_reality_<run>.json
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
from scripts.finetune_multilabel import SHARED_LABELS  # noqa: E402


def probe_auc(Z_tr, y_tr, Z_te, y_te, seed=SEED) -> float:
    """Fit a logistic probe and return held-out AUROC (nan if degenerate)."""
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                             random_state=seed).fit(Z_tr, y_tr)
    return float(roc_auc_score(y_te, clf.predict_proba(Z_te)[:, 1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="exp8_leadfix_baseline")
    ap.add_argument("--subsample", type=int, default=1500)
    ap.add_argument("--n-shuffle", type=int, default=5)
    ap.add_argument("--n-random", type=int, default=20)
    args = ap.parse_args()

    print("=" * 86)
    print(f"Is M->P transfer real?  [{args.run}]")
    print("=" * 86)

    Z_med_tr, Y_med_tr = load(args.run, "medalcare", "train")
    Z_ptb_tr, Y_ptb_tr = load(args.run, "ptbxl", "train")
    Z_ptb_te, Y_ptb_te = load(args.run, "ptbxl", "test")

    sc = StandardScaler().fit(np.vstack([Z_med_tr, Z_ptb_tr]))
    Z_med_tr = sc.transform(Z_med_tr)
    Z_ptb_tr, Z_ptb_te = sc.transform(Z_ptb_tr), sc.transform(Z_ptb_te)

    rng = np.random.default_rng(SEED)

    def sub(Z, Y, n=args.subsample):
        if len(Z) <= n:
            return Z, Y
        i = rng.choice(len(Z), n, replace=False)
        return Z[i], Y[i]

    Z_med_tr, Y_med_tr = sub(Z_med_tr, Y_med_tr)
    Z_ptb_tr, Y_ptb_tr = sub(Z_ptb_tr, Y_ptb_tr)
    D = Z_med_tr.shape[1]

    # Random-direction floor: same directions reused across classes so the
    # comparison is against one fixed reference set, not a fresh draw per class.
    g = np.random.default_rng(SEED + 7)
    R = g.normal(size=(args.n_random, D))
    R /= np.linalg.norm(R, axis=1, keepdims=True)

    hdr = (f"{'class':<6} {'prevM':>6} {'prevP':>6} | {'M->P':>7} {'shuf':>7} "
           f"{'rand':>7} | {'P->P':>7} | {'lift':>6}")
    print("\n" + hdr)
    print("-" * len(hdr))

    rows = {}
    for c, name in enumerate(SHARED_LABELS):
        y_mtr = (Y_med_tr[:, c] > 0.5).astype(int)
        y_ptr = (Y_ptb_tr[:, c] > 0.5).astype(int)
        y_pte = (Y_ptb_te[:, c] > 0.5).astype(int)

        transfer = probe_auc(Z_med_tr, y_mtr, Z_ptb_te, y_pte)
        ceiling = probe_auc(Z_ptb_tr, y_ptr, Z_ptb_te, y_pte)

        shuf = []
        for rep in range(args.n_shuffle):
            gs = np.random.default_rng(SEED + 100 + rep)
            shuf.append(probe_auc(Z_med_tr, gs.permutation(y_mtr), Z_ptb_te, y_pte))
        shuf_mean = float(np.nanmean(shuf))

        if len(np.unique(y_pte)) < 2:
            rand_mean = float("nan")
        else:
            # Fold each random direction to >= 0.5: sign is arbitrary, so the
            # floor is about separability, not direction.
            aucs = [roc_auc_score(y_pte, Z_ptb_te @ r) for r in R]
            rand_mean = float(np.mean([max(a, 1 - a) for a in aucs]))

        rows[name] = {
            "prevalence_medalcare": float(y_mtr.mean()),
            "prevalence_ptbxl": float(y_pte.mean()),
            "transfer_m2p": transfer, "shuffle_null": shuf_mean,
            "random_floor": rand_mean, "ptbxl_ceiling": ceiling,
            "lift_over_shuffle": transfer - shuf_mean,
        }
        r = rows[name]
        print(f"{name:<6} {r['prevalence_medalcare']:>6.3f} "
              f"{r['prevalence_ptbxl']:>6.3f} | {transfer:>7.4f} "
              f"{shuf_mean:>7.4f} {rand_mean:>7.4f} | {ceiling:>7.4f} | "
              f"{r['lift_over_shuffle']:>+6.3f}")

    def mean(k):
        v = [x[k] for x in rows.values() if np.isfinite(x[k])]
        return float(np.mean(v)) if v else float("nan")

    m_t, m_s = mean("transfer_m2p"), mean("shuffle_null")
    m_r, m_c = mean("random_floor"), mean("ptbxl_ceiling")

    print("\n" + "-" * 86)
    print(f"macro  M->P={m_t:.4f}  shuffle={m_s:.4f}  random={m_r:.4f}  "
          f"P->P ceiling={m_c:.4f}")
    print(f"lift over shuffle null: {m_t - m_s:+.4f}")
    print(f"fraction of in-domain ceiling retained: {m_t / m_c:.3f}"
          if np.isfinite(m_c) and m_c > 0 else "")

    print()
    if m_t - m_s > 0.05 and m_t > m_r + 0.02:
        print("=> TRANSFER IS REAL. It clears both the label-shuffle null and the")
        print("   random-direction floor, so a MedalCare-trained direction does")
        print("   carry PTB-XL-relevant signal despite the near-orthogonal class")
        print("   directions in §9. The orthogonality and the transfer must be")
        print("   reconciled -- likely many weakly-shared directions rather than")
        print("   one aligned axis.")
    elif m_t - m_s <= 0.05:
        print("=> TRANSFER IS NOT REAL. It does not clear the label-shuffle null:")
        print("   the number is an artifact of the evaluation, not evidence of")
        print("   shared structure. Every claim resting on the M->P baseline")
        print("   needs revisiting -- including the frontier's y-axis.")
    else:
        print("=> WEAK. Above the shuffle null but not clearly above the random")
        print("   floor. Do not present the raw M->P number without this table.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"transfer_reality_{args.run}.json"
    out.write_text(json.dumps({"run": args.run, "classes": rows,
                               "macro": {"transfer": m_t, "shuffle": m_s,
                                         "random_floor": m_r, "ceiling": m_c}},
                              indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
