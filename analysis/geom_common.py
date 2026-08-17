"""Shared loaders and estimators for the Tier-1 representational-geometry probes.

Two things live here because `circular_geometry.py` and
`latent_geometry_correspondence.py` both need them and must not drift apart:

1. `load_medalcare` / `load_ptbxl` -- all-folds latent loading for every
   `exp8_leadfix_*` encoder, including reconstruction of PTB-XL `ecg_id` for the
   five encoders that were exported per-split before `ecg_id` was written out.
2. `RidgeSVD` -- a ridge regressor whose design matrix is factorised once so that
   re-solving for a permuted target is nearly free. Label-shuffle nulls with a
   full re-fit per permutation are otherwise the dominant cost.

Read-only with respect to every artifact under `data/` and `outputs/latents/`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

PTBXL_ROOT = (
    REPO_ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

# PTB-XL official fold -> split mapping, mirroring PTBXLDataset.OFFICIAL_SPLITS.
PTBXL_SPLIT_FOLDS = {"train": tuple(range(1, 9)), "val": (9,), "test": (10,)}

# Every exp8 (post-leadfix) encoder with a usable MedalCare + PTB-XL export pair.
ENCODERS = [
    "exp8_leadfix_medalonly",
    "exp8_leadfix_baseline",
    "exp8_leadfix_ccmmd",
    "exp8_leadfix_dual",
    "exp8_leadfix_globalz",
    "exp8_leadfix_K64",
]

TERRITORIES = ["Anteroseptal", "Anterolateral", "Inferolateral", "Inferior"]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@dataclass
class Domain:
    """One domain's MI cohort: latents, circular target, and a CV grouping key."""

    z: np.ndarray          # (n, d) latent rows
    angle: np.ndarray      # (n,) target angle in radians
    territory: np.ndarray  # (n,) territory_4c string labels
    group: np.ndarray      # (n,) group id -- run_id (MedalCare) / patient_id (PTB-XL)
    name: str

    def __len__(self) -> int:
        return len(self.angle)


def medalcare_anchor_angles() -> dict[str, float]:
    """Empirical circular mean of MedalCare phi within each territory_4c bucket.

    These are the midpoints of the phi wedges that *define* territory_4c
    (`scripts/build_medalcare_isch_targets.py`) -- a construction of the simulator's
    own labelling, not an independent measurement (they reproduce the wedge midpoints
    to 0.043 deg). Reproduces to <0.05 deg between the train and test theta files.
    Corrected 2026-08-13: this docstring previously said "measured, not assumed",
    which overstated their independence from the label definition.
    """
    phis, terrs = [], []
    for split in ("train", "test"):
        t = np.load(REPO_ROOT / f"data/theta_mi_{split}.npz", allow_pickle=True)
        phis.append(np.asarray(t["phi"], dtype=float))
        terrs.append(np.asarray(t["territory_4c"]))
    phi, terr = np.concatenate(phis), np.concatenate(terrs)
    return {
        k: float(np.angle(np.mean(np.exp(1j * phi[terr == k]))))
        for k in TERRITORIES
    }


def load_medalcare(encoder: str) -> Domain:
    """Pooled MedalCare train+test MI rows (n=6547) with continuous phi."""
    zs, phis, terrs, groups = [], [], [], []
    for split in ("train", "test"):
        t = np.load(REPO_ROOT / f"data/theta_mi_{split}.npz", allow_pickle=True)
        lat = np.load(
            REPO_ROOT / f"outputs/latents/{encoder}_medalcare_{split}/latents.npz",
            allow_pickle=True,
        )
        idx = np.asarray(t["idx_in_split"], dtype=int)
        zs.append(lat["Z"][idx])
        phis.append(np.asarray(t["phi"], dtype=float))
        terrs.append(np.asarray(t["territory_4c"]))
        # run_id is only unique within a split -- namespace it before pooling.
        groups.append(np.array([f"{split}:{r}" for r in t["run_id"]]))
    return Domain(
        z=np.vstack(zs),
        angle=np.concatenate(phis),
        territory=np.concatenate(terrs),
        group=np.concatenate(groups),
        name="medalcare",
    )


def _ptbxl_ecg_ids(split: str) -> np.ndarray:
    """Row order of a PTB-XL split export, reconstructed from the database.

    `export_latents.py` builds its `ecg_id` key as
    `db[db.strat_fold.isin(folds)].reset_index(drop=True)["ecg_id"]` and the
    DataLoader is unshuffled, so this is exact -- verified byte-for-byte against
    the one export that stored `ecg_id` (`exp8_leadfix_medalonly_ptbxl_test`).
    """
    db = pd.read_csv(PTBXL_ROOT / "ptbxl_database.csv")
    folds = list(PTBXL_SPLIT_FOLDS[split])
    return (
        db[db["strat_fold"].isin(folds)].reset_index(drop=True)["ecg_id"].to_numpy()
    )


def _ptbxl_allfolds_latents(encoder: str) -> tuple[np.ndarray, np.ndarray]:
    """All 21799 PTB-XL latent rows for `encoder`, keyed by ecg_id.

    Prefers a single all-folds export if one exists; otherwise stitches the
    train/val/test exports back together and reconstructs the key.
    """
    single = REPO_ROOT / f"outputs/latents/{encoder}_ptbxl/latents.npz"
    if single.exists():
        lat = np.load(single, allow_pickle=True)
        return lat["Z"], np.asarray(lat["ecg_id"], dtype=int)

    zs, ids = [], []
    for split in ("train", "val", "test"):
        lat = np.load(
            REPO_ROOT / f"outputs/latents/{encoder}_ptbxl_{split}/latents.npz",
            allow_pickle=True,
        )
        key = _ptbxl_ecg_ids(split)
        if len(key) != lat["Z"].shape[0]:
            raise RuntimeError(
                f"{encoder}_ptbxl_{split}: {lat['Z'].shape[0]} rows but "
                f"{len(key)} ecg_ids -- row order is not the assumed one"
            )
        zs.append(lat["Z"])
        ids.append(key)
    return np.vstack(zs), np.concatenate(ids)


def load_ptbxl(encoder: str, anchors: dict[str, float]) -> Domain:
    """All-folds PTB-XL MI rows carrying a 4-class territory label (n=4324)."""
    z_all, id_all = _ptbxl_allfolds_latents(encoder)
    pos = {int(e): i for i, e in enumerate(id_all)}

    mi = pd.read_csv(REPO_ROOT / "data/ptbxl_mi_subclass_allfolds.csv")
    mi = mi[mi["territory_4c"].notna()].copy()
    mi["zrow"] = mi["ecg_id"].map(pos)
    if mi["zrow"].isna().any():
        raise RuntimeError(f"{encoder}: {int(mi['zrow'].isna().sum())} MI rows unkeyed")
    mi["zrow"] = mi["zrow"].astype(int)

    terr = mi["territory_4c"].to_numpy()
    return Domain(
        z=z_all[mi["zrow"].to_numpy()],
        angle=np.array([anchors[t] for t in terr]),
        territory=terr,
        group=mi["patient_id"].to_numpy().astype(str),
        name="ptbxl",
    )


# --------------------------------------------------------------------------- #
# Ridge with a reusable factorisation
# --------------------------------------------------------------------------- #
class RidgeSVD:
    """Multi-output ridge that factorises X once and re-solves cheaply.

    Alpha is picked by generalised cross-validation on the training rows. After
    `fit`, `solve(Y)` returns predictions for any *new* target matrix on the same
    design at the already-selected alpha -- which is what makes a 500-permutation
    label-shuffle null affordable (the shuffle changes Y, never X).
    """

    def __init__(self, alphas: np.ndarray | None = None) -> None:
        self.alphas = np.logspace(-2, 5, 24) if alphas is None else np.asarray(alphas)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeSVD":
        self.mu_ = x.mean(0)
        self.sd_ = x.std(0) + 1e-8
        xs = (x - self.mu_) / self.sd_
        self.xc_mean_ = xs.mean(0)
        xc = xs - self.xc_mean_

        self.u_, self.s_, self.vt_ = np.linalg.svd(xc, full_matrices=False)
        self.n_ = xc.shape[0]
        self.alpha_ = self._gcv_alpha(y)
        self.coef_, self.y_mean_ = self._solve_coef(y, self.alpha_)
        return self

    def _gcv_alpha(self, y: np.ndarray) -> float:
        yc = y - y.mean(0)
        uty = self.u_.T @ yc                       # (r, k)
        s2 = self.s_ ** 2
        best, best_alpha = np.inf, float(self.alphas[0])
        for a in self.alphas:
            shrink = s2 / (s2 + a)                 # (r,)
            resid = yc - self.u_ @ (shrink[:, None] * uty)
            dof = float(shrink.sum()) + 1.0        # +1 for the intercept
            denom = max(1.0 - dof / self.n_, 1e-6)
            gcv = float((resid ** 2).sum()) / self.n_ / denom ** 2
            if gcv < best:
                best, best_alpha = gcv, float(a)
        return best_alpha

    def _solve_coef(self, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        y_mean = y.mean(0)
        d = self.s_ / (self.s_ ** 2 + alpha)
        coef = self.vt_.T @ (d[:, None] * (self.u_.T @ (y - y_mean)))
        return coef, y_mean

    def solve(self, y: np.ndarray) -> "RidgeSVD":
        """Re-solve for a new target on the cached factorisation (in place)."""
        self.coef_, self.y_mean_ = self._solve_coef(y, self.alpha_)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xs = (x - self.mu_) / self.sd_ - self.xc_mean_
        return xs @ self.coef_ + self.y_mean_

    def direction_matrix(self) -> np.ndarray:
        """Coefficients in the *standardised* latent basis -- the readout subspace."""
        return self.coef_


def fast_ridge_coef(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Normal-equation ridge coefficients on already-standardised `x`.

    Used inside bootstrap loops where alpha is already fixed and only the
    coefficients matter. Costs one Gram matrix and one Cholesky solve, roughly an
    order of magnitude below the SVD that `RidgeSVD` needs to select alpha.
    """
    xc = x - x.mean(0)
    yc = y - y.mean(0)
    gram = xc.T @ xc
    gram[np.diag_indices_from(gram)] += alpha
    return np.linalg.solve(gram, xc.T @ yc)


# --------------------------------------------------------------------------- #
# Circular scoring
# --------------------------------------------------------------------------- #
def resultant(delta: np.ndarray) -> float:
    """Circular resultant length of an angular-error vector: 0 = chance, 1 = exact."""
    return float(np.abs(np.mean(np.exp(1j * delta))))


def med_abs_deg(delta: np.ndarray) -> float:
    """Median |angular error| in degrees, wrapped to (-180, 180]."""
    return float(np.median(np.abs(np.angle(np.exp(1j * delta)))) * 180.0 / np.pi)


def angles_from_cs(pred: np.ndarray) -> np.ndarray:
    """(cos, sin) columns -> angle in radians."""
    return np.arctan2(pred[:, 1], pred[:, 0])


def group_folds(groups: np.ndarray, n_splits: int, rng: np.random.Generator):
    """Group-disjoint CV folds. Prevents a patient (or simulation run) from
    appearing in both the training and the held-out half."""
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    assign = {g: i % n_splits for i, g in enumerate(uniq)}
    fold_of = np.array([assign[g] for g in groups])
    for k in range(n_splits):
        yield np.where(fold_of != k)[0], np.where(fold_of == k)[0]
