import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute cross-domain similarity metrics between latent spaces."
    )
    parser.add_argument("--synth-npz", type=Path, required=True, help="Path to synthetic latents npz.")
    parser.add_argument("--patient-npz", type=Path, required=True, help="Path to patient latents npz.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path for similarity metrics.")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=500,
        help="Number of bootstrap resamples for MMD (default: 500).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def load_latents(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Latent file not found: {path}")
    payload = np.load(path)
    if "Z" not in payload:
        raise KeyError(f"Latent file '{path}' does not contain array 'Z'.")
    return np.asarray(payload["Z"], dtype=np.float64)


def standardise_latents(z_synth: np.ndarray, z_patient: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    combined = np.vstack([z_synth, z_patient])
    mean = combined.mean(axis=0, keepdims=True)
    std = combined.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    z_synth_std = (z_synth - mean) / std
    z_patient_std = (z_patient - mean) / std
    return z_synth_std, z_patient_std


def median_heuristic_gamma(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, num_pairs: int = 2000) -> float:
    data = np.vstack([x, y])
    total = data.shape[0]
    if total <= 1:
        return 1.0
    if total * (total - 1) // 2 <= num_pairs:
        diff = data[:, None, :] - data[None, :, :]
        dists = np.sqrt(np.maximum(np.sum(diff**2, axis=-1), 0.0))
        tril = dists[np.tril_indices(total, k=-1)]
        non_zero = tril[tril > 0]
    else:
        idx_a = rng.integers(0, total, size=num_pairs)
        idx_b = rng.integers(0, total, size=num_pairs)
        diff = data[idx_a] - data[idx_b]
        non_zero = np.linalg.norm(diff, axis=1)
        non_zero = non_zero[non_zero > 0]
    if non_zero.size == 0:
        return 1.0
    median = np.median(non_zero)
    if median <= 0:
        return 1.0
    return 1.0 / (2.0 * (median**2))


def rbf_kernel(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    a_norm = np.sum(a**2, axis=1)[:, None]
    b_norm = np.sum(b**2, axis=1)[None, :]
    sq_dist = a_norm + b_norm - 2.0 * a @ b.T
    sq_dist = np.maximum(sq_dist, 0.0)
    return np.exp(-gamma * sq_dist)


def _mmd_from_kernels(
    k_xx: np.ndarray,
    k_yy: np.ndarray,
    k_xy: np.ndarray,
    diag_xx: np.ndarray,
    diag_yy: np.ndarray,
    counts_x: np.ndarray,
    counts_y: np.ndarray,
) -> float:
    total_x = counts_x.sum()
    total_y = counts_y.sum()
    if total_x <= 1 or total_y <= 1:
        raise ValueError("Need at least two samples per domain to compute MMD.")
    sum_xx = counts_x @ (k_xx @ counts_x) - np.dot(counts_x, diag_xx)
    sum_yy = counts_y @ (k_yy @ counts_y) - np.dot(counts_y, diag_yy)
    sum_xy = counts_x @ (k_xy @ counts_y)
    term_xx = sum_xx / (total_x * (total_x - 1))
    term_yy = sum_yy / (total_y * (total_y - 1))
    term_xy = sum_xy / (total_x * total_y)
    mmd_sq = term_xx + term_yy - 2.0 * term_xy
    return float(max(mmd_sq, 0.0))


def compute_mmd_with_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float,
    num_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[float, float, Tuple[float, float]]:
    k_xx = rbf_kernel(x, x, gamma)
    k_yy = rbf_kernel(y, y, gamma)
    k_xy = rbf_kernel(x, y, gamma)
    diag_xx = np.diag(k_xx)
    diag_yy = np.diag(k_yy)
    baseline_counts_x = np.ones(x.shape[0], dtype=np.float64)
    baseline_counts_y = np.ones(y.shape[0], dtype=np.float64)
    baseline = _mmd_from_kernels(k_xx, k_yy, k_xy, diag_xx, diag_yy, baseline_counts_x, baseline_counts_y)

    bootstrap_values = np.empty(num_bootstrap, dtype=np.float64)
    for i in range(num_bootstrap):
        idx_x = rng.integers(0, x.shape[0], size=x.shape[0])
        idx_y = rng.integers(0, y.shape[0], size=y.shape[0])
        counts_x = np.bincount(idx_x, minlength=x.shape[0]).astype(np.float64)
        counts_y = np.bincount(idx_y, minlength=y.shape[0]).astype(np.float64)
        bootstrap_values[i] = _mmd_from_kernels(k_xx, k_yy, k_xy, diag_xx, diag_yy, counts_x, counts_y)

    mean_boot = float(np.mean(bootstrap_values))
    ci_lower = float(np.percentile(bootstrap_values, 2.5))
    ci_upper = float(np.percentile(bootstrap_values, 97.5))
    return baseline, mean_boot, (ci_lower, ci_upper)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[0] != y.shape[0]:
        raise ValueError("Linear CKA requires the same number of samples in both domains.")
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean(axis=0, keepdims=True)
    xty = x_centered.T @ y_centered
    numerator = np.linalg.norm(xty, ord="fro") ** 2
    denom_x = np.linalg.norm(x_centered.T @ x_centered, ord="fro") ** 2
    denom_y = np.linalg.norm(y_centered.T @ y_centered, ord="fro") ** 2
    denom = np.sqrt(denom_x * denom_y)
    if denom <= 0:
        return 0.0
    return float(numerator / denom)


def frechet_distance_gaussian(x: np.ndarray, y: np.ndarray) -> float:
    mu_x = x.mean(axis=0)
    mu_y = y.mean(axis=0)
    cov_x = np.cov(x, rowvar=False)
    cov_y = np.cov(y, rowvar=False)
    eps = 1e-6
    dim = cov_x.shape[0]
    cov_x += eps * np.eye(dim)
    cov_y += eps * np.eye(dim)

    eigvals_x, eigvecs_x = np.linalg.eigh(cov_x)
    eigvals_x = np.clip(eigvals_x, 0.0, None)
    cov_x_half = eigvecs_x @ (np.sqrt(eigvals_x)[:, None] * eigvecs_x.T)

    temp = cov_x_half @ cov_y @ cov_x_half
    temp = 0.5 * (temp + temp.T)
    eigvals_temp, eigvecs_temp = np.linalg.eigh(temp)
    eigvals_temp = np.clip(eigvals_temp, 0.0, None)
    cov_prod_sqrt = eigvecs_temp @ (np.sqrt(eigvals_temp)[:, None] * eigvecs_temp.T)

    mean_diff = mu_x - mu_y
    trace_term = np.trace(cov_x + cov_y - 2.0 * cov_prod_sqrt)
    fid = float(mean_diff @ mean_diff + trace_term)
    return max(fid, 0.0)


def wasserstein_1d(u: np.ndarray, v: np.ndarray) -> float:
    u_sorted = np.sort(u)
    v_sorted = np.sort(v)
    n = u_sorted.size
    m = v_sorted.size
    if n == 0 or m == 0:
        raise ValueError("Wasserstein distance requires non-empty samples.")
    i = j = 0
    cdf_u = cdf_v = 0.0
    inv_n = 1.0 / n
    inv_m = 1.0 / m
    prev = min(u_sorted[0], v_sorted[0])
    distance = 0.0
    while i < n or j < m:
        next_u = u_sorted[i] if i < n else None
        next_v = v_sorted[j] if j < m else None
        if next_v is None or (next_u is not None and next_u <= next_v):
            x = next_u
            distance += abs(cdf_u - cdf_v) * (x - prev)
            cdf_u += inv_n
            prev = x
            i += 1
        else:
            x = next_v
            distance += abs(cdf_u - cdf_v) * (x - prev)
            cdf_v += inv_m
            prev = x
            j += 1
    return float(distance)


def wasserstein_along_pcs(x: np.ndarray, y: np.ndarray, num_components: int = 10) -> Dict[str, object]:
    total = np.vstack([x, y])
    centered = total - total.mean(axis=0, keepdims=True)
    num_components = min(num_components, centered.shape[1])
    if num_components == 0:
        raise ValueError("No components available for Wasserstein computation.")
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pcs = vt[:num_components]
    distances = []
    for pc in pcs:
        proj_x = x @ pc
        proj_y = y @ pc
        distances.append(wasserstein_1d(proj_x, proj_y))
    distances = np.array(distances, dtype=np.float64)
    return {
        "mean": float(distances.mean()),
        "std": float(distances.std(ddof=0)),
        "per_component": distances.tolist(),
        "num_components": int(num_components),
    }


def subsample_equal(
    x: np.ndarray,
    y: np.ndarray,
    cap: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, int]:
    size = min(x.shape[0], y.shape[0], cap)
    if size == 0:
        raise ValueError("Cannot subsample empty domains.")
    idx_x = rng.choice(x.shape[0], size=size, replace=False)
    idx_y = rng.choice(y.shape[0], size=size, replace=False)
    return x[idx_x], y[idx_y], size


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_seed)

    z_synth = load_latents(args.synth_npz)
    z_patient = load_latents(args.patient_npz)

    z_synth_std, z_patient_std = standardise_latents(z_synth, z_patient)

    gamma = median_heuristic_gamma(z_synth_std, z_patient_std, rng)
    mmd_value, mmd_boot_mean, (mmd_ci_lower, mmd_ci_upper) = compute_mmd_with_bootstrap(
        z_synth_std, z_patient_std, gamma, args.bootstrap, rng
    )

    cka_synth, cka_patient, subsample_size = subsample_equal(z_synth_std, z_patient_std, cap=10_000, rng=rng)
    linear_cka_value = linear_cka(cka_synth, cka_patient)

    frechet_value = frechet_distance_gaussian(z_synth_std, z_patient_std)
    wasserstein_stats = wasserstein_along_pcs(z_synth_std, z_patient_std, num_components=10)

    output_payload = {
        "synth_npz": str(args.synth_npz),
        "patient_npz": str(args.patient_npz),
        "subsample_size": int(subsample_size),
        "metrics": {
            "mmd_rbf": {
                "gamma": float(gamma),
                "estimate": float(mmd_value),
                "bootstrap_mean": float(mmd_boot_mean),
                "ci95": [float(mmd_ci_lower), float(mmd_ci_upper)],
                "num_bootstrap": int(args.bootstrap),
            },
            "linear_cka": float(linear_cka_value),
            "frechet_gaussian": float(frechet_value),
            "wasserstein_pcs": wasserstein_stats,
        },
    }

    output_path = args.out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(output_payload, fp, indent=2)

    print(f"Wrote similarity metrics to {output_path}")


if __name__ == "__main__":
    main()

