#!/usr/bin/env python3
"""
Maximum Mean Discrepancy (MMD) losses for aligning feature distributions.
"""

from __future__ import annotations

from typing import Optional

import torch


def _rbf_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute RBF kernel matrix between x and y.

    Args:
        x: Tensor of shape (N, D)
        y: Tensor of shape (M, D)
        sigma: Bandwidth for the RBF kernel. If None, use median heuristic.

    Returns:
        Kernel matrix of shape (N, M)
    """
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    dist_sq = diff.pow(2).sum(-1)

    if sigma is None:
        # Median heuristic
        with torch.no_grad():
            combined = torch.cat([x, y], dim=0)
            pairwise = torch.cdist(combined, combined, p=2)
            median = pairwise.median()
            sigma = median.item() if median.item() > 0 else 1.0
    gamma = 1.0 / (2.0 * sigma * sigma)
    return torch.exp(-gamma * dist_sq)


def mmd_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute the unbiased RBF-kernel MMD between two sets of features.

    Args:
        x: Source features of shape (N, D)
        y: Target features of shape (M, D)
        sigma: Optional bandwidth. If None, median heuristic is used.

    Returns:
        Scalar tensor representing the MMD^2 value.
    """
    if x.size(0) < 2 or y.size(0) < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    k_xx = _rbf_kernel(x, x, sigma)
    k_yy = _rbf_kernel(y, y, sigma)
    k_xy = _rbf_kernel(x, y, sigma)

    n = x.size(0)
    m = y.size(0)

    # Unbiased MMD estimate
    sum_xx = (k_xx.sum() - torch.diag(k_xx).sum()) / (n * (n - 1))
    sum_yy = (k_yy.sum() - torch.diag(k_yy).sum()) / (m * (m - 1))
    sum_xy = k_xy.mean()

    return sum_xx + sum_yy - 2.0 * sum_xy


def mmd_rbf_class_conditional(
    feat_x: torch.Tensor,
    feat_y: torch.Tensor,
    labels_x: torch.Tensor,
    labels_y: torch.Tensor,
    sigma: Optional[float] = None,
    min_samples: int = 2,
) -> torch.Tensor:
    """Per-class MMD: compute MMD for each shared class and return the mean.

    For each class c, selects samples positive for c from each domain and
    computes MMD between those subsets.  Classes with fewer than *min_samples*
    in either domain are skipped.

    Args:
        feat_x: Source features (N, D).
        feat_y: Target features (M, D).
        labels_x: Source labels (N, C) -- binary, in the shared class space.
        labels_y: Target labels (M, C) -- binary, in the shared class space.
        sigma: Optional bandwidth (passed through to mmd_rbf).
        min_samples: Minimum samples per class per domain to include.

    Returns:
        Scalar tensor -- mean of per-class MMD values.
    """
    n_classes = labels_x.shape[1]
    mmd_sum = torch.tensor(0.0, device=feat_x.device, dtype=feat_x.dtype)
    n_valid = 0
    for c in range(n_classes):
        mask_x = labels_x[:, c] > 0.5
        mask_y = labels_y[:, c] > 0.5
        if mask_x.sum() < min_samples or mask_y.sum() < min_samples:
            continue
        mmd_sum = mmd_sum + mmd_rbf(feat_x[mask_x], feat_y[mask_y], sigma)
        n_valid += 1
    if n_valid == 0:
        return torch.tensor(0.0, device=feat_x.device, dtype=feat_x.dtype)
    return mmd_sum / n_valid

