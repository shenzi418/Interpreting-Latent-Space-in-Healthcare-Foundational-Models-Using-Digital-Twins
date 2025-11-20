import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc align patient latent embeddings to the synthetic domain."
    )
    parser.add_argument(
        "--synth-npz",
        type=Path,
        required=True,
        help="Synthetic domain NPZ containing arrays 'Z' and optionally 'P'.",
    )
    parser.add_argument(
        "--patient-npz",
        type=Path,
        required=True,
        help="Patient domain NPZ containing arrays 'Z' and optionally 'P'.",
    )
    parser.add_argument(
        "--method",
        choices=("coral", "procrustes"),
        required=True,
        help="Alignment method to use.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Directory to write aligned latent files.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Numerical stability constant added to covariance diagonals.",
    )
    parser.add_argument(
        "--fit-samples",
        type=int,
        default=5000,
        help="Maximum samples per domain used to estimate alignment transforms (0 = use all).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for subsampling (Procrustes).",
    )
    return parser.parse_args()


def load_latents(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Latent file not found: {path}")
    payload = dict(np.load(path, allow_pickle=False))
    if "Z" not in payload:
        raise KeyError(f"Latent array 'Z' missing from {path}")
    payload["Z"] = np.asarray(payload["Z"], dtype=np.float64)
    for key in ("P", "Y", "Theta_raw", "Theta"):
        if key in payload:
            payload[key] = np.asarray(payload[key])
    return payload


def centre_and_cov(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    centred = x - mean
    cov = (centred.T @ centred) / max(x.shape[0] - 1, 1)
    return centred, mean, cov


def matrix_power_sym(mat: np.ndarray, power: float, eps: float) -> np.ndarray:
    # Assumes mat is symmetric.
    eigvals, eigvecs = np.linalg.eigh((mat + mat.T) * 0.5)
    eigvals = np.clip(eigvals, eps, None)
    powered = eigvecs @ np.diag(eigvals ** power) @ eigvecs.T
    return powered


def coral_align(
    patient_z: np.ndarray,
    synth_z: np.ndarray,
    eps: float,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, np.ndarray]]:
    patient_centered, patient_mean, patient_cov = centre_and_cov(patient_z)
    synth_centered, synth_mean, synth_cov = centre_and_cov(synth_z)

    cov_patient_inv_sqrt = matrix_power_sym(patient_cov, -0.5, eps)
    cov_synth_sqrt = matrix_power_sym(synth_cov, 0.5, eps)

    transformed = patient_centered @ cov_patient_inv_sqrt @ cov_synth_sqrt + synth_mean
    stats = {
        "patient_cov_trace": float(np.trace(patient_cov)),
        "synth_cov_trace": float(np.trace(synth_cov)),
    }
    transform = {
        "patient_mean": patient_mean,
        "synth_mean": synth_mean,
        "cov_patient_inv_sqrt": cov_patient_inv_sqrt,
        "cov_synth_sqrt": cov_synth_sqrt,
    }
    return transformed, stats, transform


def orthogonal_procrustes_align(
    patient_z: np.ndarray,
    synth_z: np.ndarray,
    eps: float,
    fit_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    patient_centered, patient_mean, patient_cov = centre_and_cov(patient_z)
    synth_centered, synth_mean, synth_cov = centre_and_cov(synth_z)

    cov_patient_inv_sqrt = matrix_power_sym(patient_cov, -0.5, eps)
    cov_synth_inv_sqrt = matrix_power_sym(synth_cov, -0.5, eps)
    cov_synth_sqrt = matrix_power_sym(synth_cov, 0.5, eps)

    patient_wh = patient_centered @ cov_patient_inv_sqrt
    synth_wh = synth_centered @ cov_synth_inv_sqrt

    n = min(patient_wh.shape[0], synth_wh.shape[0])
    if fit_samples > 0:
        n = min(n, fit_samples)
    patient_idx = rng.choice(patient_wh.shape[0], size=n, replace=False)
    synth_idx = rng.choice(synth_wh.shape[0], size=n, replace=False)

    subset_patient = patient_wh[patient_idx]
    subset_synth = synth_wh[synth_idx]
    cross_cov = subset_patient.T @ subset_synth / max(n, 1)

    u, _, vt = np.linalg.svd(cross_cov, full_matrices=False)
    r = u @ vt
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = u @ vt

    patient_aligned_wh = patient_wh @ r
    aligned = patient_aligned_wh @ cov_synth_sqrt + synth_mean

    stats = {
        "subset_size": int(n),
    }
    return aligned, stats


def subsample_for_fit(
    z: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_samples <= 0 or z.shape[0] <= max_samples:
        return z
    idx = rng.choice(z.shape[0], size=max_samples, replace=False)
    return z[idx]


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_seed)

    synth_payload = load_latents(args.synth_npz)
    patient_payload = load_latents(args.patient_npz)

    synth_z_full = synth_payload["Z"]
    patient_z_full = patient_payload["Z"]

    synth_z_fit = subsample_for_fit(synth_z_full, args.fit_samples, rng)
    patient_z_fit = subsample_for_fit(patient_z_full, args.fit_samples, rng)

    if synth_z_full.shape[1] != patient_z_full.shape[1]:
        raise ValueError(
            f"Latent dimensions differ: synth={synth_z_full.shape[1]}, patient={patient_z_full.shape[1]}"
        )

    if args.method == "coral":
        _, stats, transform = coral_align(patient_z_fit, synth_z_fit, args.eps)
        patient_centered_full = patient_z_full - transform["patient_mean"]
        aligned_full = patient_centered_full @ transform["cov_patient_inv_sqrt"] @ transform["cov_synth_sqrt"] + transform["synth_mean"]
        stats.update(
            {
                "fit_samples_patient": int(patient_z_fit.shape[0]),
                "fit_samples_synth": int(synth_z_fit.shape[0]),
            }
        )
    else:
        fit_samples = args.fit_samples if args.fit_samples > 0 else 0
        aligned_full, stats = orthogonal_procrustes_align(
            patient_z_full,
            synth_z_full,
            args.eps,
            fit_samples,
            rng,
        )
        stats.update({"fit_samples": int(stats.get("subset_size", 0))})

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    out_npz = outdir / "latents_aligned.npz"
    arrays = {"Z": aligned_full.astype(np.float32)}
    for key in ("P", "Y", "Theta_raw", "Theta"):
        if key in patient_payload:
            arrays[key] = patient_payload[key]
    np.savez_compressed(out_npz, **arrays)

    meta = {
        "method": args.method,
        "synth_npz": str(args.synth_npz),
        "patient_npz": str(args.patient_npz),
        "eps": args.eps,
        "latent_dim": int(synth_z_full.shape[1]),
        "num_samples_patient": int(patient_z_full.shape[0]),
        "num_samples_synth": int(synth_z_full.shape[0]),
        "stats": stats,
    }
    with (outdir / "alignment_meta.json").open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)

    print(f"Saved aligned latents to {out_npz}")


if __name__ == "__main__":
    main()

