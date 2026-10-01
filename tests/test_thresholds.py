"""Threshold selection from a false-positive-rate budget.

These functions decide which transactions get blocked, so a silent error here is
the difference between catching fraud and blocking nobody. Every test measures the
*achieved* FPR directly rather than trusting the function's own arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_curve

from train import recall_at_threshold, threshold_at_fpr


def achieved_fpr(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> float:
    """FP / (FP + TN), computed independently of the code under test."""
    negatives = y_true == 0
    return float(((proba >= threshold) & negatives).sum() / negatives.sum())


def separable_problem(n_neg: int = 20_000, n_pos: int = 200, seed: int = 0):
    """A well-separated problem: positives score high, negatives low."""
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.zeros(n_neg), np.ones(n_pos)]).astype(int)
    proba = np.concatenate([rng.uniform(0, 0.3, n_neg), rng.uniform(0.6, 1.0, n_pos)])
    return y, proba


def test_returned_threshold_actually_respects_the_fpr_budget():
    y, proba = separable_problem()
    thr = threshold_at_fpr(y, proba, 0.005)
    assert achieved_fpr(y, proba, thr) <= 0.005


def test_a_separable_problem_gives_high_recall_at_a_tight_fpr():
    # This is the test that catches the 1 - precision bug. The wrong formula
    # collapses the threshold toward 1.0 and reports single-digit recall while
    # still printing a confident-looking FPR column.
    y, proba = separable_problem()
    thr = threshold_at_fpr(y, proba, 0.005)
    assert recall_at_threshold(y, proba, thr) > 0.80


def test_recall_never_decreases_as_the_fpr_budget_loosens():
    # Non-strict on purpose: on a perfectly separable fixture recall saturates at
    # 1.0 for both budgets, and a strict inequality would be asserting something
    # about the fixture rather than about the function.
    y, proba = separable_problem(seed=1)
    tight = recall_at_threshold(y, proba, threshold_at_fpr(y, proba, 0.001))
    loose = recall_at_threshold(y, proba, threshold_at_fpr(y, proba, 0.05))
    assert loose >= tight
    assert tight > 0.0


def test_threshold_is_finite_so_it_always_flags_something_real():
    # roc_curve's first threshold is inf, paired with fpr=0.0, so it satisfies
    # any budget. Returning it means blocking literally nothing.
    y, proba = separable_problem(seed=2)
    for budget in [0.0, 0.001, 0.005, 0.5]:
        thr = threshold_at_fpr(y, proba, budget)
        assert np.isfinite(thr)
        assert 0.0 <= thr <= proba.max()


def test_threshold_never_exceeds_the_maximum_score():
    y, proba = separable_problem()
    assert threshold_at_fpr(y, proba, 0.005) <= proba.max()


def test_a_zero_budget_still_returns_a_usable_threshold():
    y, proba = separable_problem()
    thr = threshold_at_fpr(y, proba, 0.0)
    assert 0.0 <= thr <= 1.0
    assert achieved_fpr(y, proba, thr) <= 0.0


def test_recall_helper_handles_a_class_with_no_positives():
    y = np.zeros(100, dtype=int)
    assert recall_at_threshold(y, np.full(100, 0.5), 0.4) == 0.0


def test_threshold_matches_a_hand_computed_roc_cut():
    # A tiny case where the answer can be checked by eye: 3 negatives, 1 positive.
    y = np.array([0, 0, 0, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.95])
    # At a 1/3 budget we can afford exactly one of the three negatives, so the
    # qualifying thresholds are those admitting at most one negative.
    fpr, _tpr, thresholds = roc_curve(y, proba)
    qualifying = {float(t) for t, f in zip(thresholds, fpr) if f <= 1 / 3}
    assert threshold_at_fpr(y, proba, 1 / 3) in qualifying
