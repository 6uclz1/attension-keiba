"""MIT License

Copyright (c) 2024 Attension Keiba

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def _sanitize_probabilities(y_prob: np.ndarray) -> np.ndarray:
    """Clamp probabilities to a safe range and replace NaNs with 0.5."""
    probs = np.asarray(y_prob, dtype=float)
    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(probs, 1e-7, 1 - 1e-7)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    results: Dict[str, float] = {}
    probs = _sanitize_probabilities(y_prob)
    try:
        results["auc"] = float(roc_auc_score(y_true, probs))
    except ValueError:
        results["auc"] = float("nan")
    results["logloss"] = float(log_loss(y_true, probs))
    results["brier"] = float(brier_score_loss(y_true, probs))
    return results


def expected_value_metrics(y_true: np.ndarray, y_prob: np.ndarray, odds: np.ndarray) -> Dict[str, float]:
    probs = _sanitize_probabilities(y_prob)
    ev = probs * odds
    realized = y_true * (odds - 1.0) - (1 - y_true)
    roi = realized.mean()
    return {
        "ev_mean": float(ev.mean()),
        "roi_flat_bet": float(roi),
    }


def combined_metrics(y_true: np.ndarray, y_prob: np.ndarray, odds: np.ndarray) -> Dict[str, float]:
    metrics = classification_metrics(y_true, y_prob)
    metrics.update(expected_value_metrics(y_true, y_prob, odds))
    return metrics
