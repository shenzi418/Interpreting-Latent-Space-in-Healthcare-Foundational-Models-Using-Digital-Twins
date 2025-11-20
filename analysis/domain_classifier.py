import argparse
import json
from pathlib import Path
from typing import Dict, Literal, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight domain classifier on synthetic vs patient features."
    )
    parser.add_argument("--synth-npz", type=Path, required=True, help="Synthetic latent/probability npz.")
    parser.add_argument("--patient-npz", type=Path, required=True, help="Patient latent/probability npz.")
    parser.add_argument(
        "--features",
        type=str,
        choices=("Z", "P"),
        default="Z",
        help="Feature matrix to use (Z=latents, P=probabilities).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON path for classifier metrics.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified folds for cross-validation (default: 5).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def load_features(path: Path, key: Literal["Z", "P"]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    payload = np.load(path)
    if key not in payload:
        raise KeyError(f"Array '{key}' missing from {path}.")
    data = np.asarray(payload[key])
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array for '{key}' in {path}, got shape {data.shape}.")
    return data


def subsample_to_balance(
    synth: np.ndarray,
    patient: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    size = min(synth.shape[0], patient.shape[0])
    if size == 0:
        raise ValueError("Cannot balance empty domains.")
    synth_idx = rng.choice(synth.shape[0], size=size, replace=False)
    patient_idx = rng.choice(patient.shape[0], size=size, replace=False)
    balanced_features = np.vstack([synth[synth_idx], patient[patient_idx]])
    balanced_labels = np.concatenate([np.zeros(size, dtype=np.int32), np.ones(size, dtype=np.int32)])
    return balanced_features, balanced_labels


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, num_samples: int = 1000) -> Tuple[float, float]:
    if values.size == 0:
        raise ValueError("Cannot compute CI with empty values.")
    boot = np.empty(num_samples, dtype=np.float64)
    for i in range(num_samples):
        idx = rng.integers(0, values.size, size=values.size)
        boot[i] = values[idx].mean()
    lower = float(np.percentile(boot, 2.5))
    upper = float(np.percentile(boot, 97.5))
    return lower, upper


def evaluate_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    folds: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    scaler = StandardScaler()
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rng.integers(0, 1_000_000))

    aurocs = []
    accuracies = []
    pr_aucs = []

    for train_idx, test_idx in kfold.split(features, labels):
        x_train, x_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        scaler.fit(x_train)
        x_train_std = scaler.transform(x_train)
        x_test_std = scaler.transform(x_test)

        clf = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            random_state=rng.integers(0, 1_000_000),
            max_iter=1000,
        )
        clf.fit(x_train_std, y_train)
        logits = clf.decision_function(x_test_std)
        probs = clf.predict_proba(x_test_std)[:, 1]
        preds = clf.predict(x_test_std)

        aurocs.append(roc_auc_score(y_test, probs))
        accuracies.append(np.mean(preds == y_test))
        pr_aucs.append(average_precision_score(y_test, probs))

    aurocs = np.array(aurocs, dtype=np.float64)
    accuracies = np.array(accuracies, dtype=np.float64)
    pr_aucs = np.array(pr_aucs, dtype=np.float64)

    ci_auroc = bootstrap_ci(aurocs, rng)
    ci_acc = bootstrap_ci(accuracies, rng)
    ci_pr = bootstrap_ci(pr_aucs, rng)

    return {
        "folds": folds,
        "auroc": {
            "mean": float(aurocs.mean()),
            "per_fold": aurocs.tolist(),
            "ci95": [float(ci_auroc[0]), float(ci_auroc[1])],
        },
        "accuracy": {
            "mean": float(accuracies.mean()),
            "per_fold": accuracies.tolist(),
            "ci95": [float(ci_acc[0]), float(ci_acc[1])],
        },
        "pr_auc": {
            "mean": float(pr_aucs.mean()),
            "per_fold": pr_aucs.tolist(),
            "ci95": [float(ci_pr[0]), float(ci_pr[1])],
        },
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_seed)

    synth_features = load_features(args.synth_npz, args.features)
    patient_features = load_features(args.patient_npz, args.features)

    all_features, all_labels = subsample_to_balance(synth_features, patient_features, rng)

    metrics = evaluate_classifier(all_features, all_labels, args.cv_folds, rng)

    payload = {
        "synth_npz": str(args.synth_npz),
        "patient_npz": str(args.patient_npz),
        "features": args.features,
        "num_samples_per_domain": int(all_labels.size // 2),
        "metrics": metrics,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print(f"Wrote domain classification report to {args.out}")


if __name__ == "__main__":
    main()

