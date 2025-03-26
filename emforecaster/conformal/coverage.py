# Adapted from: https://github.com/kamilest/conformal-rnn/blob/master/models/cfrnn.py
# Copyright (c) 2021, Kamilė Stankevičiūtė
# Licensed under the BSD 3-clause license

import torch
import torch.nn as nn
from torch.nn.functional import l1_loss


def coverage(intervals, target):
    """
    Determines whether intervals cover the target prediction
    considering each target horizon either separately or jointly.

    Args:
        intervals: shape [n_samples, 2, horizon, num_channels]
        target: ground truth forecast values

    Returns:
        individual and joint coverage rates
    """

    lower, upper = intervals[:, 0], intervals[:, 1]
    # [n_samples, horizon, num_channels]
    horizon_coverages = torch.logical_and(target >= lower, target <= upper)
    # [n_samples, horizon, num_channels], [n_samples, num_channels]
    return horizon_coverages, torch.all(horizon_coverages, dim=1)


def get_critical_scores(calibration_scores, q):
    """
    Computes critical calibration scores from scores in the calibration set.

    Args:
        calibration_scores: calibration scores for each example in the
            calibration set.
        q: target quantile for which to return the calibration score

    Returns:
        critical calibration scores for each target horizon
    """

    return torch.tensor(
        [
            [
                torch.quantile(position_calibration_scores, q=q)
                for position_calibration_scores in feature_calibration_scores
            ]
            for feature_calibration_scores in calibration_scores
        ]
    ).T


def nonconformity(output, target):
    """
    Measures the nonconformity between output and target time series.

    Args:
        output: Model forecast for the target.
        target: The target time series.

    Returns:
        Average MAE loss for every step in the sequence.
    """
    # Average MAE loss for every step in the sequence.
    return l1_loss(output, target, reduction="none")


def get_all_critical_scores(preds, targets, alpha):
    """
    Computes the nonconformity scores for the calibration dataset.

        preds: All model predictions over the entire calibration set (n_samples, pred_len, num_channels).
        targets: All target values over the entire calibration set (n_samples, pred_len, num_channels).
        alpha: The significance level.

    """
    n_samples, pred_len, _ = preds.size()  # (n_samples, pred_len, num_channels)
    calibration_scores = nonconformity(
        preds, targets
    ).T  # (num_channels, pred_len, n_samples)

    # Uncorrected critical calibration scores
    q = min((n_samples + 1.0) * (1 - alpha) / n_samples, 1)
    corrected_q = min((n_samples + 1.0) * (1 - alpha / pred_len) / n_samples, 1)
    critical_calibration_scores = get_critical_scores(
        calibration_scores=calibration_scores, q=q
    )  # (pred_len, num_channels)

    # Bonferroni-corrected critical calibration scores.
    corrected_critical_calibration_scores = get_critical_scores(
        calibration_scores=calibration_scores,
        q=corrected_q,
    )  # (pred_len, num_channels)

    return critical_calibration_scores, corrected_critical_calibration_scores


# Renamed from (def predict())
def get_intervals(pred, scores):
    """
    Forecasts the time series with conformal uncertainty intervals.

    Args:
        pred: The model forecast for the tim series of shape (*, pred_len, num_channels)
        scores: The critical calibration scores (corrected or noncorrected) for the model.

    Returns:
        tensor with lower and upper forecast bounds
    """

    # [batch_size, pred_len, num_channels]

    with torch.no_grad():
        lower = pred - scores
        upper = pred + scores

    # [batch_size, 2, pred_len, num_channels]
    return torch.stack((lower, upper), dim=1)


def get_coverage(preds, targets, scores):
    """
    Evaluates coverage of the examples in the test dataset.

    Args:
        test_loader: The loader for the test set.
        scores: The critical calibration scores (corrected or noncorrected).
    Returns:
        independent and joint coverages, forecast uncertainty intervals
    """

    intervals = get_intervals(
        preds, scores
    )  # [n_samples, 2, pred_len, num_channels] containing lower and upper bounds
    independent_coverages, joint_coverages = coverage(
        intervals, targets
    )  # (n_samples, (1 | pred_len), num_channels) booleans

    return independent_coverages, joint_coverages, intervals
