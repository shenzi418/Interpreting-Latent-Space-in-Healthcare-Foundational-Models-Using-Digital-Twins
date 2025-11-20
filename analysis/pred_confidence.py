import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import ks_2samp
except ImportError:  # pragma: no cover - fallback path
    ks_2samp = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse prediction confidence distributions across domains."
    )
    parser.add_argument("--synth-npz", type=Path, required=True, help="Synthetic domain NPZ (expects array 'P').")
    parser.add_argument("--patient-npz", type=Path, required=True, help="Patient domain NPZ (expects array 'P').")
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for confidence.json and generated figures.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50).",
    )
    parser.add_argument(
        "--num-quantiles",
        type=int,
        default=200,
        help="Number of quantile points for QQ plots (default: 200).",
    )
    return parser.parse_args()


def load_probabilities(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {path}")
    payload = np.load(path)
    if "P" not in payload:
        raise KeyError(f"Array 'P' not found in {path}")
    probs = np.asarray(payload["P"], dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"Expected 'P' to be 2D, got shape {probs.shape}")
    return probs


def compute_emp_and_entropy(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    emp = probs.max(axis=1)
    eps = 1e-8
    clipped = np.clip(probs, eps, 1.0 - eps)
    entropy_components = clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)
    entropy = -np.sum(entropy_components, axis=1)
    return emp, entropy


def run_ks_test(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    if ks_2samp is None:
        raise ImportError("scipy is required for KS test but could not be imported.")
    stat, p_value = ks_2samp(a, b, alternative="two-sided", mode="auto")
    return {"statistic": float(stat), "p_value": float(p_value)}


def plot_histogram(
    synth_values: np.ndarray,
    patient_values: np.ndarray,
    bins: int,
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(
        synth_values,
        bins=bins,
        alpha=0.6,
        label="Synthetic",
        density=True,
        color="#1f77b4",
    )
    plt.hist(
        patient_values,
        bins=bins,
        alpha=0.6,
        label="Patient",
        density=True,
        color="#ff7f0e",
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def compute_class_mean_probabilities(probs: np.ndarray) -> Dict[str, float]:
    means = probs.mean(axis=0)
    return {f"class_{idx}": float(value) for idx, value in enumerate(means)}


def plot_qq_logits(
    synth_probs: np.ndarray,
    patient_probs: np.ndarray,
    num_quantiles: int,
    output_dir: Path,
) -> Dict[str, str]:
    eps = 1e-6
    synth_logits = np.log(np.clip(synth_probs, eps, 1.0 - eps) / np.clip(1.0 - synth_probs, eps, 1.0 - eps))
    patient_logits = np.log(np.clip(patient_probs, eps, 1.0 - eps) / np.clip(1.0 - patient_probs, eps, 1.0 - eps))

    quantiles = np.linspace(0.0, 1.0, num_quantiles)
    outputs: Dict[str, str] = {}

    for idx in range(synth_probs.shape[1]):
        synth_q = np.quantile(synth_logits[:, idx], quantiles)
        patient_q = np.quantile(patient_logits[:, idx], quantiles)
        min_val = min(synth_q.min(), patient_q.min())
        max_val = max(synth_q.max(), patient_q.max())

        plt.figure(figsize=(5, 5))
        plt.scatter(synth_q, patient_q, s=15, alpha=0.7, color="#2ca02c")
        plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1)
        plt.title(f"QQ Plot (logits) - Class {idx}")
        plt.xlabel("Synthetic quantiles")
        plt.ylabel("Patient quantiles")
        plt.tight_layout()
        output_path = output_dir / f"qq_logits_class_{idx}.png"
        plt.savefig(output_path, dpi=200)
        plt.close()
        outputs[f"class_{idx}"] = str(output_path)

    return outputs


def main() -> None:
    args = parse_args()

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    synth_probs = load_probabilities(args.synth_npz)
    patient_probs = load_probabilities(args.patient_npz)

    emp_synth, entropy_synth = compute_emp_and_entropy(synth_probs)
    emp_patient, entropy_patient = compute_emp_and_entropy(patient_probs)

    emp_hist_path = outdir / "emp_histogram.png"
    entropy_hist_path = outdir / "entropy_histogram.png"

    plot_histogram(
        emp_synth,
        emp_patient,
        bins=args.num_bins,
        title="Max Probability (EMP) Distribution",
        xlabel="Max probability",
        output_path=emp_hist_path,
    )

    plot_histogram(
        entropy_synth,
        entropy_patient,
        bins=args.num_bins,
        title="Prediction Entropy Distribution",
        xlabel="Entropy",
        output_path=entropy_hist_path,
    )

    ks_emp = run_ks_test(emp_synth, emp_patient)
    ks_entropy = run_ks_test(entropy_synth, entropy_patient)

    class_means_synth = compute_class_mean_probabilities(synth_probs)
    class_means_patient = compute_class_mean_probabilities(patient_probs)

    qq_paths = plot_qq_logits(
        synth_probs,
        patient_probs,
        num_quantiles=args.num_quantiles,
        output_dir=outdir,
    )

    confidence_summary = {
        "synth_npz": str(args.synth_npz),
        "patient_npz": str(args.patient_npz),
        "metrics": {
            "emp": {
                "synth_mean": float(emp_synth.mean()),
                "patient_mean": float(emp_patient.mean()),
                "synth_median": float(np.median(emp_synth)),
                "patient_median": float(np.median(emp_patient)),
                "ks_test": ks_emp,
            },
            "entropy": {
                "synth_mean": float(entropy_synth.mean()),
                "patient_mean": float(entropy_patient.mean()),
                "synth_median": float(np.median(entropy_synth)),
                "patient_median": float(np.median(entropy_patient)),
                "ks_test": ks_entropy,
            },
        },
        "class_mean_probabilities": {
            "synthetic": class_means_synth,
            "patient": class_means_patient,
        },
        "figures": {
            "emp_histogram": str(emp_hist_path),
            "entropy_histogram": str(entropy_hist_path),
            "qq_logits": qq_paths,
        },
    }

    json_path = outdir / "confidence.json"
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(confidence_summary, fp, indent=2)

    print(f"Wrote confidence analysis to {json_path}")


if __name__ == "__main__":
    main()

