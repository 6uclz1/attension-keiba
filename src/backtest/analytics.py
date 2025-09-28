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
import pandas as pd

from src.train.metrics import classification_metrics, expected_value_metrics


def aggregate_performance(predictions: pd.DataFrame, group_column: str | None = None) -> pd.DataFrame:
    if group_column and group_column in predictions:
        grouped = predictions.groupby(group_column)
        rows = []
        for key, frame in grouped:
            metrics = classification_metrics(frame["target"].to_numpy(), frame["prob"].to_numpy())
            metrics.update(expected_value_metrics(frame["target"].to_numpy(), frame["prob"].to_numpy(), frame["odds"].to_numpy()))
            metrics[group_column] = key
            rows.append(metrics)
        return pd.DataFrame(rows)
    metrics = classification_metrics(predictions["target"].to_numpy(), predictions["prob"].to_numpy())
    metrics.update(expected_value_metrics(predictions["target"].to_numpy(), predictions["prob"].to_numpy(), predictions["odds"].to_numpy()))
    return pd.DataFrame([metrics])


def equity_curve(bets: pd.DataFrame) -> pd.DataFrame:
    curve = bets[["race_id", "horse_id", "bankroll"]].copy()
    curve["step"] = np.arange(len(curve))
    return curve
