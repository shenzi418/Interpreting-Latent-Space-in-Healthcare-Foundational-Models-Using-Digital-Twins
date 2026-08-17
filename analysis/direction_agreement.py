"""Reconciling §9's orthogonal class directions with §Rank5's real transfer.

Two results that cannot both mean what they appear to:

  §9      cos(w_medalcare, w_ptbxl) = 0.001 / 0.068 / 0.053 for NORM/MI/CD.
          In 1024-d, two RANDOM unit vectors have |cos| ~ 1/sqrt(D) = 0.031.
          So the two domains' class directions are, in Euclidean terms,
          indistinguishable from unrelated.

  Rank 5  the MedalCare direction nonetheless scores 0.86 / 0.61 / 0.64 on
          held-out PTB-XL, +0.22 over a label-shuffle null and 80% of the
          PTB-XL-trained ceiling.

A direction that were truly unrelated to the PTB-XL class direction could not do
that. The resolution is that Euclidean cosine is the wrong inner product here.
What a linear probe actually produces is the projection Z@w, and two directions
induce nearly identical projections whenever the DATA COVARIANCE couples them:

    corr(Z w_m, Z w_p) = w_m' S w_p / sqrt( w_m' S w_m * w_p' S w_p )

with S the (centred) covariance of the evaluation domain. Euclidean cosine is
this quantity with S = I -- i.e. it silently asserts the latent space is
isotropic. A 1024-d ECGFounder latent space is emphatically not: a handful of
directions carry most of the variance, so w_m and w_p can be near-orthogonal as
vectors while their projections are strongly correlated.

This script computes both metrics side by side, plus:

  * RANDOM CONTROL for each metric. The covariance-induced correlation of two
    random directions is NOT ~0 when S is anisotropic, so the real pair must be
    compared against a random pair under the same S, not against zero.
  * PARTICIPATION RATIO of S, (sum L)^2 / sum(L^2), the effective number of
    directions carrying variance. This quantifies HOW anisotropic the space is
    and therefore how misleading the Euclidean cosine was.
  * the same quantities under S = MedalCare covariance, to check the effect is a
    property of the latent geometry rather than of one dataset.

If the covariance-induced correlation is large where the cosine was ~0, §9's
orthogonality finding is an artifact of the metric and should be reported as
such -- the class directions agree in the only geometry that affects predictions.

Writes: outputs/analysis/domain_signal/direction_agreement_<run>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
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


def fit_direction(Z, y, seed=SEED):
    """Unit-norm L2 logistic weight vector, or None if the class is degenerate."""
    if len(np.unique(y)) < 2:
        return None
    clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                             random_state=seed).fit(Z, y)
    w = clf.coef_[0]
    n = np.linalg.norm(w)
    return None if n < 1e-12 else w / n


def cov_corr(a, b, S) -> float:
    """Correlation of the projections Z@a and Z@b induced by covariance S.

    This is the cosine under the inner product <a,b>_S = a'S b, which is the
    geometry the predictions actually live in. Reduces to Euclidean cosine
    when S = I.
    """
    na, nb = a @ S @ a, b @ S @ b
    if na <= 0 or nb <= 0:
        return float("nan")
    return float((a @ S @ b) / np.sqrt(na * nb))


def participation_ratio(S) -> float:
    """Effective number of variance-carrying directions: (sum L)^2 / sum L^2.

    D for a perfectly isotropic space, ~1 when a single direction dominates.
    """
    lam = np.linalg.eigvalsh(S)
    lam = np.clip(lam, 0, None)
    s2 = float(np.sum(lam ** 2))
    return float(np.sum(lam) ** 2 / s2) if s2 > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="exp8_leadfix_baseline")
    ap.add_argument("--subsample", type=int, default=1500)
    ap.add_argument("--n-random", type=int, default=200)
    args = ap.parse_args()

    print("=" * 88)
    print(f"Do the class directions agree?  Euclidean vs data metric  [{args.run}]")
    print("=" * 88)

    Z_med_tr, Y_med_tr = load(args.run, "medalcare", "train")
    Z_ptb_tr, Y_ptb_tr = load(args.run, "ptbxl", "train")

    sc = StandardScaler().fit(np.vstack([Z_med_tr, Z_ptb_tr]))
    Z_med_tr, Z_ptb_tr = sc.transform(Z_med_tr), sc.transform(Z_ptb_tr)

    rng = np.random.default_rng(SEED)

    def sub(Z, Y, n=args.subsample):
        if len(Z) <= n:
            return Z, Y
        i = rng.choice(len(Z), n, replace=False)
        return Z[i], Y[i]

    Z_med_tr, Y_med_tr = sub(Z_med_tr, Y_med_tr)
    Z_ptb_tr, Y_ptb_tr = sub(Z_ptb_tr, Y_ptb_tr)
    D = Z_med_tr.shape[1]

    S_ptb = np.cov(Z_ptb_tr, rowvar=False)
    S_med = np.cov(Z_med_tr, rowvar=False)
    pr_ptb, pr_med = participation_ratio(S_ptb), participation_ratio(S_med)

    print(f"\nlatent dim D = {D}")
    print(f"participation ratio  PTB-XL = {pr_ptb:.1f}   MedalCare = {pr_med:.1f}")
    print(f"  (D={D} would be isotropic; {pr_ptb:.0f} means the space is "
          f"{D / pr_ptb:.0f}x more concentrated than Euclidean cosine assumes)")

    # Random pairs, scored under both metrics, as the reference both numbers
    # must be read against.
    g = np.random.default_rng(SEED + 11)
    R = g.normal(size=(2 * args.n_random, D))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    r_cos, r_cov = [], []
    for i in range(args.n_random):
        a, b = R[2 * i], R[2 * i + 1]
        r_cos.append(abs(float(a @ b)))
        r_cov.append(abs(cov_corr(a, b, S_ptb)))
    rc_cos, rc_cov = float(np.mean(r_cos)), float(np.mean(r_cov))
    rc_cov_p95 = float(np.percentile(np.abs(r_cov), 95))

    print(f"\nrandom pair reference:  |cos| = {rc_cos:.4f}   "
          f"|corr_S| = {rc_cov:.4f}  (95th pct {rc_cov_p95:.4f})")

    hdr = (f"{'class':<6} {'cos':>8} {'corr_Sp':>9} {'corr_Sm':>9} | "
           f"{'x random':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))

    rows = {}
    for c, name in enumerate(SHARED_LABELS):
        y_m = (Y_med_tr[:, c] > 0.5).astype(int)
        y_p = (Y_ptb_tr[:, c] > 0.5).astype(int)
        w_m, w_p = fit_direction(Z_med_tr, y_m), fit_direction(Z_ptb_tr, y_p)
        if w_m is None or w_p is None:
            print(f"{name:<6} (degenerate -- skipped)")
            continue

        cos = float(w_m @ w_p)
        cp, cm = cov_corr(w_m, w_p, S_ptb), cov_corr(w_m, w_p, S_med)
        ratio = abs(cp) / rc_cov if rc_cov > 0 else float("nan")

        rows[name] = {"cosine_euclidean": cos, "corr_under_ptbxl_cov": cp,
                      "corr_under_medalcare_cov": cm,
                      "ratio_to_random_cov": ratio}
        print(f"{name:<6} {cos:>8.4f} {cp:>9.4f} {cm:>9.4f} | {ratio:>8.1f}x")

    if not rows:
        print("no usable classes")
        return 1

    def mean(k):
        v = [abs(x[k]) for x in rows.values() if np.isfinite(x[k])]
        return float(np.mean(v)) if v else float("nan")

    m_cos, m_cp = mean("cosine_euclidean"), mean("corr_under_ptbxl_cov")

    print("\n" + "-" * 88)
    print(f"mean |cosine|            = {m_cos:.4f}   "
          f"(random {rc_cos:.4f}, ratio {m_cos / rc_cos:.1f}x)")
    print(f"mean |corr under S_ptb|  = {m_cp:.4f}   "
          f"(random {rc_cov:.4f}, ratio {m_cp / rc_cov:.1f}x)")

    print()
    if m_cp > rc_cov_p95 and m_cp > 3 * m_cos:
        print("=> §9's ORTHOGONALITY WAS A METRIC ARTIFACT. Under the data")
        print("   covariance -- the geometry that determines predictions -- the")
        print("   two domains' class directions are strongly correlated, while")
        print("   Euclidean cosine reported ~random. The latent space is far from")
        print("   isotropic (participation ratio %.0f of %d), so cosine was never" % (pr_ptb, D))
        print("   the right agreement measure. Transfer and 'orthogonality' are")
        print("   consistent; §9's cosine line must be corrected in the report.")
    elif m_cp > rc_cov_p95:
        print("=> PARTIAL. Agreement under the data metric exceeds the random")
        print("   reference but is not dramatically above the Euclidean cosine.")
        print("   Report both; the metric explains some but not all of the gap.")
    else:
        print("=> NOT EXPLAINED BY THE METRIC. The directions are unrelated under")
        print("   the data covariance too, so the transfer in Rank 5 comes from")
        print("   somewhere else -- next suspect is a nonlinear or prevalence")
        print("   route, not a shared linear axis.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"direction_agreement_{args.run}.json"
    out.write_text(json.dumps(
        {"run": args.run, "n_dims": D,
         "participation_ratio": {"ptbxl": pr_ptb, "medalcare": pr_med},
         "random_reference": {"cosine": rc_cos, "corr_cov": rc_cov,
                              "corr_cov_p95": rc_cov_p95},
         "classes": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
