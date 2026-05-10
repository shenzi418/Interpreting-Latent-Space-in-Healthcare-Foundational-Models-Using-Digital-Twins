#!/usr/bin/env python3
"""Utilities for reporting multilabel classification metrics safely."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

AVAILABLE_METRICS = ("accuracy", "f1", "recall", "specificity", "precision", "brier", "roc_auc")

_THRESHOLD_METRICS = {"accuracy", "f1", "recall", "specificity", "precision"}
_PROB_METRICS = {"brier", "roc_auc"}


def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch between y_true {y_true.shape} and y_pred {y_pred.shape}"
        )
    if y_true.ndim != 2:
        raise ValueError("Expected 2D arrays for multilabel evaluation.")


def _binarise(y_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert probabilities to binary predictions at the given threshold."""
    return (y_prob >= threshold).astype(np.int32)


def _macro_average(values: Sequence[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))


# ---------------------------------------------------------------------------
# Threshold-based metric helpers (operate on binary y_true and y_pred)
# ---------------------------------------------------------------------------

def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    """Return (TP, FP, TN, FN) for a single class (1-D arrays)."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def _accuracy(y_true: np.ndarray, y_pred_bin: np.ndarray) -> float:
    tp, fp, tn, fn = _confusion_counts(y_true, y_pred_bin)
    total = tp + fp + tn + fn
    if total == 0:
        return 0.0
    return float((tp + tn) / total)


def _f1(y_true: np.ndarray, y_pred_bin: np.ndarray) -> Optional[float]:
    tp, fp, tn, fn = _confusion_counts(y_true, y_pred_bin)
    if tp + fp + fn == 0:
        return None
    denom = 2 * tp + fp + fn
    if denom == 0:
        return None
    return float(2 * tp / denom)


def _recall(y_true: np.ndarray, y_pred_bin: np.ndarray) -> Optional[float]:
    tp, fp, tn, fn = _confusion_counts(y_true, y_pred_bin)
    if tp + fn == 0:
        return None
    return float(tp / (tp + fn))


def _specificity(y_true: np.ndarray, y_pred_bin: np.ndarray) -> Optional[float]:
    tp, fp, tn, fn = _confusion_counts(y_true, y_pred_bin)
    if tn + fp == 0:
        return None
    return float(tn / (tn + fp))


def _precision(y_true: np.ndarray, y_pred_bin: np.ndarray) -> Optional[float]:
    tp, fp, tn, fn = _confusion_counts(y_true, y_pred_bin)
    if tp + fp == 0:
        return None
    return float(tp / (tp + fp))


# ---------------------------------------------------------------------------
# Probability-based metric helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Per-class dispatch
# ---------------------------------------------------------------------------

_THRESHOLD_DISPATCH = {
    "accuracy": _accuracy,
    "f1": _f1,
    "recall": _recall,
    "specificity": _specificity,
    "precision": _precision,
}

_PROB_DISPATCH = {
    "brier": _brier_score,
    "roc_auc": _safe_roc_auc,
}


def _compute_metric_per_class(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    y_pred_bin: np.ndarray,
    metric: str,
) -> List[Optional[float]]:
    per_class: List[Optional[float]] = []

    if metric in _THRESHOLD_DISPATCH:
        fn = _THRESHOLD_DISPATCH[metric]
        for c in range(y_true.shape[1]):
            per_class.append(fn(y_true[:, c], y_pred_bin[:, c]))
    elif metric in _PROB_DISPATCH:
        fn = _PROB_DISPATCH[metric]
        for c in range(y_true.shape[1]):
            per_class.append(fn(y_true[:, c], y_pred_prob[:, c]))
    else:
        raise ValueError(f"Unsupported metric '{metric}'.")

    return per_class


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_names: Optional[Iterable[str]] = None,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, List[Optional[float]]]]:
    """Compute per-class and macro metrics for multilabel predictions.

    Args:
        y_true: Ground-truth labels (N, C) with binary entries.
        y_pred: Predicted probabilities (N, C) in [0, 1].
        metric_names: Iterable of metric identifiers to compute.
        threshold: Decision threshold for converting probabilities to binary
            predictions (used by threshold-based metrics). Default: 0.5.

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

    y_pred = np.clip(y_pred, 0.0, 1.0)
    y_pred_bin = _binarise(y_pred, threshold)

    per_class_metrics: Dict[str, List[Optional[float]]] = {}
    macro_metrics: Dict[str, Optional[float]] = {}

    for metric in metric_names:
        per_class_values = _compute_metric_per_class(y_true, y_pred, y_pred_bin, metric)
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
