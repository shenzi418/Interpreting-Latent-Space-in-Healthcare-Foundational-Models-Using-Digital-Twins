"""Tier 1: what should have been measured instead of C2ST.

C2ST asks whether two latent clouds are *distinguishable*. That question is
answered 1.0 by a constant translation, which leaves every internal relation
intact, and the 77-cell lead-permutation sweep showed it moves 1e-5 across
corruptions that halve transfer. It is the wrong question.

The right question is whether the two domains encode territory with the same
*internal geometry*. Three probes, none of which can be satisfied by a
translation, and each with its own permutation-calibrated null:

  A  subspace overlap   -- principal angles between the two readout planes,
                           normalised by a within-domain split-half ceiling so
                           the estimator's own noise is divided out
  B  kNN label transfer -- nonparametric: does a real ECG's nearest synthetic
                           neighbours carry its territory? Separates "the map is
                           not linear" from "the geometry does not correspond"
  C  RSA                -- do the four territories sit in the same relative
                           arrangement in both domains?

Read-only over stored latents. No retraining.

Usage:
  python analysis/latent_geometry_correspondence.py
  python analysis/latent_geometry_correspondence.py --encoders exp8_leadfix_K64
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scipy.stats import spearmanr  # pylint: disable=wrong-import-position

from geom_common import (  # pylint: disable=wrong-import-position
    ENCODERS,
    TERRITORIES,
    Domain,
    RidgeSVD,
    fast_ridge_coef,
    load_medalcare,
    load_ptbxl,
    medalcare_anchor_angles,
    resultant,
)

OUT_DIR = REPO_ROOT / "outputs/analysis/circular_geometry"
N_BOOT = 200
N_PERM = 500
KNN_K = 25


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def pooled_standardise(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One common basis for both domains -- required for the subspaces to be
    comparable at all. Standardising per-domain would silently absorb part of the
    very offset under test."""
    both = np.vstack([a, b])
    mu, sd = both.mean(0), both.std(0) + 1e-8
    return (a - mu) / sd, (b - mu) / sd


def readout_plane(z: np.ndarray, angle: np.ndarray, alpha: float | None = None) -> np.ndarray:
    """Orthonormal basis of the 2-D (cos, sin) readout subspace.

    `alpha=None` selects it by GCV (one SVD); passing a fixed alpha takes the
    cheap normal-equation path, which is what the bootstrap loop uses.
    """
    y = np.c_[np.cos(angle), np.sin(angle)]
    if alpha is None:
        coef = RidgeSVD().fit(z, y).direction_matrix()
    else:
        coef = fast_ridge_coef(z, y, alpha)
    return np.linalg.qr(coef)[0]


def select_alpha(z: np.ndarray, angle: np.ndarray) -> float:
    """GCV-selected ridge alpha on the full domain, reused across bootstrap draws.

    Re-selecting per draw would be more principled but costs an SVD each time;
    the penalty is essentially flat in this regime and the draws all have the same
    n, so a single selection is a fair economy.
    """
    return RidgeSVD().fit(z, np.c_[np.cos(angle), np.sin(angle)]).alpha_


def plane_overlap(qa: np.ndarray, qb: np.ndarray) -> float:
    """Mean cos^2 of the principal angles between two 2-D subspaces.

    1 = identical plane, 0 = every direction orthogonal. Two random 2-D planes in
    R^d give 2/d, which is the floor to compare against.
    """
    return float((np.linalg.svd(qa.T @ qb, compute_uv=False) ** 2).mean())


def group_resample(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap row indices resampled at the group level (patient / sim run)."""
    uniq = np.unique(groups)
    idx_of = {g: np.where(groups == g)[0] for g in uniq}
    drawn = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([idx_of[g] for g in drawn])


def group_halves(groups: np.ndarray, rng: np.random.Generator):
    uniq = rng.permutation(np.unique(groups))
    left = set(uniq[: len(uniq) // 2].tolist())
    mask = np.array([g in left for g in groups])
    return np.where(mask)[0], np.where(~mask)[0]


# --------------------------------------------------------------------------- #
# Probe A -- subspace overlap against its own noise floor
# --------------------------------------------------------------------------- #
def subspace_probe(medal: Domain, ptb: Domain, rng: np.random.Generator) -> dict:
    zm, zp = pooled_standardise(medal.z, ptb.z)
    d = zm.shape[1]
    am = select_alpha(zm, medal.angle)
    ap = select_alpha(zp, ptb.angle)

    cross, within_m, within_p, normed = [], [], [], []
    for _ in range(N_BOOT):
        im, ip = group_resample(medal.group, rng), group_resample(ptb.group, rng)
        qm = readout_plane(zm[im], medal.angle[im], am)
        qp = readout_plane(zp[ip], ptb.angle[ip], ap)
        c = plane_overlap(qm, qp)

        ma, mb = group_halves(medal.group, rng)
        wm = plane_overlap(
            readout_plane(zm[ma], medal.angle[ma], am),
            readout_plane(zm[mb], medal.angle[mb], am),
        )
        pa, pb = group_halves(ptb.group, rng)
        wp = plane_overlap(
            readout_plane(zp[pa], ptb.angle[pa], ap),
            readout_plane(zp[pb], ptb.angle[pb], ap),
        )
        cross.append(c)
        within_m.append(wm)
        within_p.append(wp)
        normed.append(c / max(np.sqrt(wm * wp), 1e-12))

    rand = [
        plane_overlap(
            np.linalg.qr(rng.standard_normal((d, 2)))[0],
            np.linalg.qr(rng.standard_normal((d, 2)))[0],
        )
        for _ in range(N_BOOT)
    ]

    def summ(x):
        x = np.asarray(x)
        return {
            "mean": float(x.mean()),
            "ci95": [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))],
        }

    return {
        "latent_dim": int(d),
        "cross_domain": summ(cross),
        "within_medalcare_splithalf": summ(within_m),
        "within_ptbxl_splithalf": summ(within_p),
        "random_2d_baseline": summ(rand),
        # The headline number: cross-domain overlap expressed as a fraction of
        # what the estimator achieves against itself. 1.0 would mean the two
        # domains agree as well as one domain agrees with its own other half.
        "normalised_overlap": summ(normed),
        "frac_cross_le_random": float(np.mean(np.asarray(cross) <= np.mean(rand))),
    }


# --------------------------------------------------------------------------- #
# Probe B -- nonparametric kNN label transfer
# --------------------------------------------------------------------------- #
def _cos_topk(query: np.ndarray, ref: np.ndarray, k: int) -> np.ndarray:
    qn = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    rn = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12)
    out = np.empty((len(qn), k), dtype=np.int64)
    step = 512
    for i in range(0, len(qn), step):
        sim = qn[i : i + step] @ rn.T
        out[i : i + step] = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    return out


def _circ_mean(angles: np.ndarray, axis: int) -> np.ndarray:
    return np.angle(np.mean(np.exp(1j * angles), axis=axis))


def knn_probe(medal: Domain, ptb: Domain, rng: np.random.Generator, k: int) -> dict:
    """Predict a real ECG's territory angle from its nearest synthetic neighbours.

    No fitted map of any kind. If the linear readout fails but this succeeds, the
    correspondence exists and is merely nonlinear; if both fail, the neighbourhood
    structure itself does not carry across.
    """
    zm, zp = pooled_standardise(medal.z, ptb.z)

    nn = _cos_topk(zp, zm, k)
    pred = _circ_mean(medal.angle[nn], axis=1)
    r_obs = resultant(pred - ptb.angle)
    null = np.array(
        [
            resultant(_circ_mean(medal.angle[rng.permutation(len(medal))][nn], 1)
                      - ptb.angle)
            for _ in range(N_PERM)
        ]
    )

    # Within-PTB-XL ceiling: same procedure, but neighbours drawn from PTB-XL
    # itself with same-patient rows excluded so it is not self-retrieval.
    nn_self = _cos_topk(zp, zp, k + 8)
    same = ptb.group[nn_self] == ptb.group[:, None]
    keep = np.empty((len(ptb), k), dtype=np.int64)
    for i in range(len(ptb)):
        cand = nn_self[i][~same[i]]
        cand = cand[cand != i]
        keep[i] = np.resize(cand, k) if len(cand) else nn_self[i, :k]
    r_within = resultant(_circ_mean(ptb.angle[keep], 1) - ptb.angle)

    return {
        "k": k,
        "R_cross_domain": r_obs,
        "R_null_mean": float(null.mean()),
        "R_null_p95": float(np.percentile(null, 95)),
        "perm_p": (1 + int((null >= r_obs).sum())) / (N_PERM + 1),
        "R_within_ptbxl_ceiling": r_within,
        "fraction_of_ceiling": float(r_obs / r_within) if r_within > 0 else float("nan"),
        "mean_pred_angle_by_territory_deg": {
            t: float(np.angle(np.mean(np.exp(1j * pred[ptb.territory == t]))) * 180 / np.pi)
            for t in TERRITORIES
            if np.any(ptb.territory == t)
        },
    }


# --------------------------------------------------------------------------- #
# Probe C -- representational similarity of the four territories
# --------------------------------------------------------------------------- #
def rsa_probe(medal: Domain, ptb: Domain, rng: np.random.Generator) -> dict:
    """Do the four territories sit in the same relative arrangement in both
    domains? Invariant to rotation, reflection, scale, and translation of either
    latent space -- so it isolates relational structure from placement."""
    zm, zp = pooled_standardise(medal.z, ptb.z)

    def rdm(z, terr):
        cent = np.stack([z[terr == t].mean(0) for t in TERRITORIES])
        cn = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12)
        return 1.0 - cn @ cn.T

    rm, rp = rdm(zm, medal.territory), rdm(zp, ptb.territory)
    iu = np.triu_indices(4, k=1)
    rho = float(spearmanr(rm[iu], rp[iu]).statistic)

    perms = []
    for perm in itertools.permutations(range(4)):
        rp_perm = rp[np.ix_(perm, perm)]
        perms.append(
            {
                "perm": [TERRITORIES[i] for i in perm],
                "rho": float(spearmanr(rm[iu], rp_perm[iu]).statistic),
                "is_identity": perm == (0, 1, 2, 3),
            }
        )
    perms.sort(key=lambda r: -r["rho"])
    ident = next(i for i, r in enumerate(perms) if r["is_identity"])

    # Split-half reliability inside each domain = the ceiling for rho.
    rel = []
    for dom, z in ((medal, zm), (ptb, zp)):
        vals = []
        for _ in range(50):
            a, b = group_halves(dom.group, rng)
            ra, rb = rdm(z[a], dom.territory[a]), rdm(z[b], dom.territory[b])
            vals.append(float(spearmanr(ra[iu], rb[iu]).statistic))
        rel.append(float(np.mean(vals)))

    return {
        "rho_cross_domain": rho,
        "identity_rank_of_24": ident + 1,
        "best_rho": perms[0]["rho"],
        "best_perm": perms[0]["perm"],
        "splithalf_reliability_medalcare": rel[0],
        "splithalf_reliability_ptbxl": rel[1],
        "rdm_medalcare": rm.tolist(),
        "rdm_ptbxl": rp.tolist(),
        "territory_order": TERRITORIES,
    }


# --------------------------------------------------------------------------- #
def run_encoder(encoder: str, anchors: dict[str, float], seed: int, k: int) -> dict:
    rng = np.random.default_rng(seed)
    medal = load_medalcare(encoder)
    ptb = load_ptbxl(encoder, anchors)
    print(f"\n=== {encoder}  (d={medal.z.shape[1]}) ===")

    sub = subspace_probe(medal, ptb, rng)
    print(f"  subspace  cross={sub['cross_domain']['mean']:.4f} "
          f"{sub['cross_domain']['ci95']}  "
          f"ceilM={sub['within_medalcare_splithalf']['mean']:.3f} "
          f"ceilP={sub['within_ptbxl_splithalf']['mean']:.3f}  "
          f"rand={sub['random_2d_baseline']['mean']:.4f}")
    print(f"            normalised={sub['normalised_overlap']['mean']:.4f} "
          f"{sub['normalised_overlap']['ci95']}")

    knn = knn_probe(medal, ptb, rng, k)
    print(f"  kNN(k={knn['k']})  R_cross={knn['R_cross_domain']:.4f} "
          f"(null {knn['R_null_mean']:.3f}, p={knn['perm_p']:.4f})  "
          f"ceiling={knn['R_within_ptbxl_ceiling']:.4f}  "
          f"frac={knn['fraction_of_ceiling']:.3f}")

    rsa = rsa_probe(medal, ptb, rng)
    print(f"  RSA       rho={rsa['rho_cross_domain']:+.3f}  "
          f"identity ranks {rsa['identity_rank_of_24']}/24  "
          f"(best {rsa['best_rho']:+.3f})  "
          f"reliability M={rsa['splithalf_reliability_medalcare']:+.3f} "
          f"P={rsa['splithalf_reliability_ptbxl']:+.3f}")

    return {"encoder": encoder, "subspace": sub, "knn": knn, "rsa": rsa}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoders", nargs="*", default=ENCODERS)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--k", type=int, default=KNN_K)
    args = ap.parse_args()

    anchors = medalcare_anchor_angles()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_res = {}
    for enc in args.encoders:
        try:
            all_res[enc] = run_encoder(enc, anchors, args.seed, args.k)
        except FileNotFoundError as exc:
            print(f"\n=== {enc}: SKIPPED ({exc}) ===")

    dest = OUT_DIR / "latent_geometry_correspondence.json"
    dest.write_text(
        json.dumps({"n_boot": N_BOOT, "n_perm": N_PERM, "encoders": all_res}, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 96)
    print(f"{'encoder':<26}{'subCross':>10}{'subNorm':>9}{'kNN R':>8}{'ceil':>8}"
          f"{'frac':>7}{'kNN p':>8}{'RSA rho':>9}{'rank':>6}")
    print("-" * 96)
    for enc, r in all_res.items():
        print(f"{enc:<26}{r['subspace']['cross_domain']['mean']:>10.4f}"
              f"{r['subspace']['normalised_overlap']['mean']:>9.4f}"
              f"{r['knn']['R_cross_domain']:>8.3f}"
              f"{r['knn']['R_within_ptbxl_ceiling']:>8.3f}"
              f"{r['knn']['fraction_of_ceiling']:>7.2f}"
              f"{r['knn']['perm_p']:>8.4f}"
              f"{r['rsa']['rho_cross_domain']:>+9.3f}"
              f"{r['rsa']['identity_rank_of_24']:>6}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
