import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class LinearProbe:
    name: str
    coef: np.ndarray
    intercept: np.ndarray
    feature_mean: Optional[np.ndarray] = None
    feature_scale: Optional[np.ndarray] = None
    theta_names: Optional[list] = None

    def predict(self, features: np.ndarray) -> np.ndarray:
        z = features
        if self.feature_mean is not None:
            z = z - self.feature_mean
        if self.feature_scale is not None:
            scale = np.where(np.abs(self.feature_scale) < 1e-12, 1.0, self.feature_scale)
            z = z / scale
        return z @ self.coef + self.intercept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply synthetic-domain probes to patient latent embeddings."
    )
    parser.add_argument(
        "--probe",
        type=Path,
        required=True,
        help="Path to probe summary JSON (contains probe metadata and statistics).",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        required=True,
        help="Path to latent NPZ with array 'Z'.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Directory to save predictions and diagnostics.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional label for outputs; defaults to basename of NPZ.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Jitter term used when inverting covariance matrices.",
    )
    return parser.parse_args()


def load_probe_summary(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Probe summary JSON not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        summary = json.load(fp)
    if "probes" not in summary or not summary["probes"]:
        raise ValueError(f"No probe definitions found in {path}")
    return summary


def load_linear_probe(base_dir: Path, entry: Dict) -> LinearProbe:
    weights_path = entry.get("weights") or entry.get("weights_path") or entry.get("path")
    if not weights_path:
        raise ValueError(f"Probe entry {entry.get('name')} missing weights path.")
    weights_file = (base_dir / weights_path).resolve()
    if not weights_file.exists():
        raise FileNotFoundError(f"Probe weights file not found: {weights_file}")

    payload = dict(np.load(weights_file, allow_pickle=False))

    if "coef" not in payload:
        raise KeyError(f"Probe weights file {weights_file} missing 'coef' array.")
    coef = np.asarray(payload["coef"], dtype=np.float64)

    intercept = payload.get("intercept")
    if intercept is None:
        intercept = np.zeros(coef.shape[1], dtype=np.float64)
    else:
        intercept = np.asarray(intercept, dtype=np.float64)

    feature_mean = payload.get("feature_mean")
    if feature_mean is not None:
        feature_mean = np.asarray(feature_mean, dtype=np.float64)

    feature_scale = payload.get("feature_scale")
    if feature_scale is not None:
        feature_scale = np.asarray(feature_scale, dtype=np.float64)

    theta_names = entry.get("theta_names")

    return LinearProbe(
        name=entry.get("name", weights_file.stem),
        coef=coef,
        intercept=intercept,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        theta_names=theta_names,
    )


def load_probe_ensemble(summary_path: Path) -> Dict[str, LinearProbe]:
    summary = load_probe_summary(summary_path)
    probes: Dict[str, LinearProbe] = {}
    for entry in summary["probes"]:
        probe_type = entry.get("type", "linear").lower()
        if probe_type not in {"linear", "pls"}:
            raise ValueError(f"Unsupported probe type '{probe_type}' in summary.")
        probe = load_linear_probe(summary_path.parent, entry)
        probes[probe.name] = probe
    return probes


def load_latent_npz(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(f"Latent NPZ not found: {path}")
    payload = dict(np.load(path, allow_pickle=False))
    if "Z" not in payload:
        raise KeyError(f"Latent array 'Z' missing from {path}")
    z = np.asarray(payload.pop("Z"), dtype=np.float64)
    return z, payload


def mahalanobis_distance(samples: np.ndarray, mean: np.ndarray, cov: np.ndarray, eps: float) -> np.ndarray:
    centred = samples - mean
    reg_cov = cov + np.eye(cov.shape[0]) * eps
    cov_inv = np.linalg.pinv(reg_cov)
    left = centred @ cov_inv
    return np.sqrt(np.sum(left * centred, axis=1))


def compute_coverage(
    theta_hat: np.ndarray,
    percentiles: Optional[Dict[str, np.ndarray]],
) -> Optional[Dict[str, np.ndarray]]:
    if not percentiles:
        return None
    p5 = percentiles.get("p5")
    p95 = percentiles.get("p95")
    if p5 is None or p95 is None:
        return None
    p5 = np.asarray(p5, dtype=np.float64)
    p95 = np.asarray(p95, dtype=np.float64)
    below = theta_hat < p5
    above = theta_hat > p95
    return {
        "frac_below": below.mean(axis=0).tolist(),
        "frac_above": above.mean(axis=0).tolist(),
        "frac_outside_any": float((below | above).any(axis=1).mean()),
    }


def compute_mahalanobis_stats(
    theta_hat: np.ndarray,
    mean: Optional[np.ndarray],
    cov: Optional[np.ndarray],
    eps: float,
) -> Optional[Dict[str, float]]:
    if mean is None or cov is None:
        return None
    distances = mahalanobis_distance(theta_hat, mean, cov, eps)
    return {
        "mean": float(distances.mean()),
        "std": float(distances.std(ddof=0)),
        "p50": float(np.median(distances)),
        "p90": float(np.percentile(distances, 90)),
        "max": float(distances.max()),
    }


def main() -> None:
    args = parse_args()
    probes = load_probe_ensemble(args.probe)
    features, extras = load_latent_npz(args.npz)

    summary = load_probe_summary(args.probe)
    theta_mean = None
    theta_cov = None
    percentiles = None
    if "theta_stats" in summary:
        stats = summary["theta_stats"]
        if "mean" in stats:
            theta_mean = np.asarray(stats["mean"], dtype=np.float64)
        if "cov" in stats:
            theta_cov = np.asarray(stats["cov"], dtype=np.float64)
    if "theta_percentiles" in summary:
        percentiles = {
            "p5": summary["theta_percentiles"].get("p5"),
            "p95": summary["theta_percentiles"].get("p95"),
        }

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    predictions_npz: Dict[str, np.ndarray] = {}
    diagnostics: Dict[str, Dict] = {}

    for name, probe in probes.items():
        theta_hat = probe.predict(features).astype(np.float32)
        predictions_npz[f"theta_hat_{name}"] = theta_hat

        coverage = compute_coverage(theta_hat, percentiles)
        maha = compute_mahalanobis_stats(theta_hat, theta_mean, theta_cov, args.eps)

        diag = {
            "probe_type": "linear",
            "theta_dim": int(theta_hat.shape[1]) if theta_hat.ndim == 2 else 1,
            "num_samples": int(theta_hat.shape[0]),
        }
        if probe.theta_names:
            diag["theta_names"] = probe.theta_names
        if coverage is not None:
            diag["coverage"] = coverage
        if maha is not None:
            diag["mahalanobis"] = maha
        diagnostics[name] = diag

    np.savez_compressed(outdir / "theta_hat.npz", **predictions_npz)

    tag = args.tag or args.npz.stem
    extras_to_save = {key: value for key, value in extras.items() if key in {"P", "Y", "Theta", "Theta_raw"}}
    if extras_to_save:
        np.savez_compressed(outdir / f"aux_{tag}.npz", **extras_to_save)

    with (outdir / "diagnostics.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "probe_summary": str(args.probe),
                "latent_source": str(args.npz),
                "num_samples": int(features.shape[0]),
                "latent_dim": int(features.shape[1]),
                "probes": diagnostics,
            },
            fp,
            indent=2,
        )

    print(f"Wrote probe predictions to {outdir}")


if __name__ == "__main__":
    main()

