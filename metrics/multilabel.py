#!/usr/bin/env python3
"""Utilities for reporting multilabel classification metrics safely."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

AVAILABLE_METRICS = ("ap", "brier", "roc_auc")


def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch between y_true {y_true.shape} and y_pred {y_pred.shape}"
        )
    if y_true.ndim != 2:
        raise ValueError("Expected 2D arrays for multilabel evaluation.")


def _macro_average(values: Sequence[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return None
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return None


def _brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(np.mean((y_score - y_true) ** 2))


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives <= 1 or negatives <= 1:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def _compute_metric_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
) -> List[Optional[float]]:
    per_class: List[Optional[float]] = []
    for class_idx in range(y_true.shape[1]):
        labels = y_true[:, class_idx]
        scores = y_pred[:, class_idx]
        if metric == "ap":
            per_class.append(_safe_average_precision(labels, scores))
        elif metric == "brier":
            per_class.append(_brier_score(labels, scores))
        elif metric == "roc_auc":
            per_class.append(_safe_roc_auc(labels, scores))
        else:
            raise ValueError(f"Unsupported metric '{metric}'.")
    return per_class


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_names: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, List[Optional[float]]]]:
    """Compute per-class and macro metrics for multilabel predictions.

    Args:
        y_true: Ground-truth labels (N, C) with binary entries.
        y_pred: Predicted probabilities (N, C) in [0, 1].
        metric_names: Iterable of metric identifiers to compute.

    Returns:
        Dictionary containing per-class values, macro averages, and support counts.
    """
    metric_names = tuple(metric_names) if metric_names else AVAILABLE_METRICS
    unknown = [m for m in metric_names if m not in AVAILABLE_METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics requested: {unknown}. "
                         f"Supported metrics: {AVAILABLE_METRICS}")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    _validate_inputs(y_true, y_pred)

    # Ensure probabilities remain in [0, 1]
    y_pred = np.clip(y_pred, 0.0, 1.0)

    per_class_metrics: Dict[str, List[Optional[float]]] = {}
    macro_metrics: Dict[str, Optional[float]] = {}

    for metric in metric_names:
        per_class_values = _compute_metric_per_class(y_true, y_pred, metric)
        per_class_metrics[metric] = per_class_values
        macro_metrics[metric] = _macro_average(per_class_values)

    positives = np.sum(y_true, axis=0).astype(int).tolist()
    negatives = (y_true.shape[0] - np.sum(y_true, axis=0)).astype(int).tolist()

    return {
        "per_class": per_class_metrics,
        "macro": macro_metrics,
        "support": {
            "positives": positives,
            "negatives": negatives,
        },
    }

