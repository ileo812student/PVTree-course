# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PVTree-course contributors.
# Course-project addition; see LICENSE for terms and README for upstream attribution.
from __future__ import annotations

import numpy as np


def verification_metrics(scores, labels) -> dict:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("scores and labels must be matching one-dimensional arrays")
    positives, negatives = int(np.sum(labels == 1)), int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("verification requires both genuine and impostor pairs")
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    distinct = np.r_[np.where(np.diff(sorted_scores))[0], len(scores) - 1]
    tpr = np.r_[0.0, tp[distinct] / positives, 1.0]
    fpr = np.r_[0.0, fp[distinct] / negatives, 1.0]
    thresholds = np.r_[np.inf, sorted_scores[distinct], -np.inf]
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[index] + fnr[index]) / 2.0)

    def tar_at_far(target):
        valid = np.where(fpr <= target)[0]
        return float(np.max(tpr[valid])) if len(valid) else 0.0

    result = {
        "genuine_pairs": positives,
        "impostor_pairs": negatives,
        "eer": eer,
        "eer_threshold": float(thresholds[index]),
        "tar_at_far_1e-3": tar_at_far(1e-3),
        "tar_at_far_1e-4": tar_at_far(1e-4) if negatives >= 10000 else None,
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()},
    }
    return result
