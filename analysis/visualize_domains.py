import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise latent embeddings across domains."
    )
    parser.add_argument(
        "--synth-npz",
        type=Path,
        required=True,
        help="Synthetic domain NPZ (expects arrays 'Z' and 'P').",
    )
    parser.add_argument(
        "--patient-npz",
        type=Path,
        required=True,
        help="Patient domain NPZ (expects arrays 'Z' and 'P').",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Directory to save visualisation figures.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum samples per domain for plotting (0 = use all).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for subsampling.",
    )
    return parser.parse_args()


def load_latents(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Latent file not found: {path}")
    payload = np.load(path)
    if "Z" not in payload or "P" not in payload:
        missing = [name for name in ("Z", "P") if name not in payload]
        raise KeyError(f"Missing arrays {missing} in {path}")
    z = np.asarray(payload["Z"], dtype=np.float64)
    p = np.asarray(payload["P"], dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"'Z' must be 2D, got shape {z.shape} in {path}")
    if p.ndim != 2:
        raise ValueError(f"'P' must be 2D, got shape {p.shape} in {path}")
    if z.shape[0] != p.shape[0]:
        raise ValueError(f"Mismatch between Z ({z.shape[0]}) and P ({p.shape[0]}) rows in {path}")
    return z, p


def subsample(z: np.ndarray, p: np.ndarray, max_samples: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if max_samples <= 0 or z.shape[0] <= max_samples:
        return z, p
    indices = rng.choice(z.shape[0], size=max_samples, replace=False)
    return z[indices], p[indices]


def standardise_concat(synth: np.ndarray, patient: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined = np.vstack([synth, patient])
    mean = combined.mean(axis=0, keepdims=True)
    std = combined.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    synth_std = (synth - mean) / std
    patient_std = (patient - mean) / std
    combined_std = np.vstack([synth_std, patient_std])
    return synth_std, patient_std, combined_std


def compute_embedding(combined: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=2)
    return pca.fit_transform(combined)


def plot_domain_scatter(
    synth_emb: np.ndarray,
    patient_emb: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        synth_emb[:, 0],
        synth_emb[:, 1],
        s=10,
        alpha=0.5,
        label="Synthetic",
        linewidths=0,
    )
    ax.scatter(
        patient_emb[:, 0],
        patient_emb[:, 1],
        s=10,
        alpha=0.5,
        label="Patient",
        linewidths=0,
    )
    ax.set_title("Joint embedding coloured by domain")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_class_small_multiples(
    embedding: np.ndarray,
    domain_labels: np.ndarray,
    top_classes: np.ndarray,
    num_classes: int,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    cmap = plt.colormaps.get_cmap("tab10").resampled(num_classes)
    domain_names = ["Synthetic", "Patient"]
    for idx, ax in enumerate(axes):
        mask = domain_labels == idx
        sc = ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=top_classes[mask],
            cmap=cmap,
            s=10,
            linewidths=0,
            alpha=0.7,
            vmin=-0.5,
            vmax=num_classes - 0.5,
        )
        ax.set_title(f"{domain_names[idx]} (coloured by argmax class)")
        ax.set_xlabel("Component 1")
        if idx == 0:
            ax.set_ylabel("Component 2")
        else:
            ax.set_ylabel("")
    cbar = fig.colorbar(
        sc,
        ax=axes,
        orientation="vertical",
        fraction=0.046,
        pad=0.04,
        ticks=range(num_classes),
    )
    cbar.set_label("Predicted argmax class")
    fig.subplots_adjust(wspace=0.05, right=0.88)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_seed)

    synth_z, synth_p = load_latents(args.synth_npz)
    patient_z, patient_p = load_latents(args.patient_npz)

    synth_z, synth_p = subsample(synth_z, synth_p, args.max_samples, rng)
    patient_z, patient_p = subsample(patient_z, patient_p, args.max_samples, rng)

    synth_std, patient_std, combined_std = standardise_concat(synth_z, patient_z)
    embedding = compute_embedding(combined_std)

    synth_n = synth_std.shape[0]
    synth_emb = embedding[:synth_n]
    patient_emb = embedding[synth_n:]

    domain_labels = np.concatenate(
        [
            np.zeros(synth_emb.shape[0], dtype=np.int8),
            np.ones(patient_emb.shape[0], dtype=np.int8),
        ]
    )
    combined_top_classes = np.concatenate(
        [
            np.argmax(synth_p, axis=1),
            np.argmax(patient_p, axis=1),
        ]
    )

    num_classes = synth_p.shape[1]
    if patient_p.shape[1] != num_classes:
        raise ValueError(
            f"Dimension mismatch between synthetic P ({num_classes}) and patient P ({patient_p.shape[1]})."
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_domain_scatter(
        synth_emb,
        patient_emb,
        args.outdir / "embedding_by_domain.png",
    )
    plot_class_small_multiples(
        embedding,
        domain_labels,
        combined_top_classes,
        num_classes,
        args.outdir / "embedding_by_argmax_class.png",
    )
    print(f"Saved figures to {args.outdir}")


if __name__ == "__main__":
    main()

