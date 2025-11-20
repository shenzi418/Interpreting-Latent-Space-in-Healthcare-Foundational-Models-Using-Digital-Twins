import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute k-NN mixing score between synthetic and patient latent spaces."
    )
    parser.add_argument("--synth-npz", type=Path, required=True, help="Synthetic domain NPZ (expects array 'Z').")
    parser.add_argument("--patient-npz", type=Path, required=True, help="Patient domain NPZ (expects array 'Z').")
    parser.add_argument(
        "--k",
        type=int,
        default=15,
        help="Number of nearest neighbours to consider (default: 15).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON path for mixing metrics.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for subsampling when domain sizes differ greatly.",
    )
    return parser.parse_args()


def load_latents(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Latent file not found: {path}")
    payload = np.load(path)
    if "Z" not in payload:
        raise KeyError(f"Array 'Z' missing from {path}")
    z = np.asarray(payload["Z"], dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"Latent array must be 2D, got shape {z.shape}")
    return z


def standardise_concat(synth: np.ndarray, patient: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined = np.vstack([synth, patient])
    mean = combined.mean(axis=0, keepdims=True)
    std = combined.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    synth_std = (synth - mean) / std
    patient_std = (patient - mean) / std
    combined_std = np.vstack([synth_std, patient_std])
    return synth_std, patient_std, combined_std


def compute_knn_mixing(
    combined: np.ndarray,
    domain_labels: np.ndarray,
    k: int,
) -> Dict[str, float]:
    if k < 1:
        raise ValueError("k must be positive.")
    if combined.shape[0] <= k:
        raise ValueError("Number of samples must exceed k.")

    # Fit k-NN (k+1 because we exclude the point itself)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="auto")
    nn.fit(combined)
    distances, indices = nn.kneighbors(combined, return_distance=True)

    # Exclude self neighbour (distance zero, index matches sample)
    neighbor_indices = indices[:, 1:]

    same_domain = domain_labels[:, None] == domain_labels[neighbor_indices]
    other_domain_fraction = 1.0 - same_domain.mean(axis=1)
    mean_mixing = float(other_domain_fraction.mean())
    std_mixing = float(other_domain_fraction.std(ddof=0))

    # Neighbor entropy (Shannon entropy over domain proportions per point)
    proportion_other = other_domain_fraction
    proportion_same = 1.0 - proportion_other

    eps = 1e-9
    entropy_components = 0.0
    entropy_components += proportion_other * np.log(proportion_other + eps)
    entropy_components += proportion_same * np.log(proportion_same + eps)
    neighbor_entropy = float((-entropy_components).mean())

    return {
        "mean_mixing": mean_mixing,
        "std_mixing": std_mixing,
        "neighbor_entropy": neighbor_entropy,
    }


def subsample_to_balance(
    synth: np.ndarray,
    patient: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    size = min(synth.shape[0], patient.shape[0])
    if size == 0:
        raise ValueError("Cannot subsample empty domains.")
    synth_idx = rng.choice(synth.shape[0], size=size, replace=False)
    patient_idx = rng.choice(patient.shape[0], size=size, replace=False)
    synth_sub = synth[synth_idx]
    patient_sub = patient[patient_idx]
    domain_labels = np.concatenate(
        [np.zeros(size, dtype=np.int16), np.ones(size, dtype=np.int16)]
    )
    return synth_sub, patient_sub, domain_labels


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_seed)

    synth_z = load_latents(args.synth_npz)
    patient_z = load_latents(args.patient_npz)

    synth_balanced, patient_balanced, labels = subsample_to_balance(synth_z, patient_z, rng)
    synth_std, patient_std, combined_std = standardise_concat(synth_balanced, patient_balanced)

    metrics = compute_knn_mixing(combined_std, labels, args.k)

    payload = {
        "synth_npz": str(args.synth_npz),
        "patient_npz": str(args.patient_npz),
        "k": int(args.k),
        "num_samples_per_domain": int(labels.size // 2),
        "metrics": metrics,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print(f"Wrote k-NN mixing metrics to {args.out}")


if __name__ == "__main__":
    main()

