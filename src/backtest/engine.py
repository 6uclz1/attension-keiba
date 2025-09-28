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

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.backtest.analytics import aggregate_performance, equity_curve
from src.backtest.bankroll import simulate_bankroll
from src.config import ConfigBundle
from src.data.tensors import load_tensor_dict
from src.infer.predictor import RacePredictor


def _within_range(date_str: str, start: datetime, end: datetime) -> bool:
    if not date_str:
        return False
    current = datetime.fromisoformat(date_str)
    return start <= current <= end


def run_backtest(config: ConfigBundle) -> Dict[str, object]:
    tensor_dict, metadata = load_tensor_dict(config.data.processed_dir)
    predictor = RacePredictor(config)
    start_dt = datetime.fromisoformat(str(config.backtest.start_date))
    end_dt = datetime.fromisoformat(str(config.backtest.end_date))

    race_ids = tensor_dict["race_ids"].astype(str)
    dates = tensor_dict.get("dates", np.array([""] * len(race_ids)))
    race_to_date: Dict[str, str] = {}
    for race_id, date in zip(race_ids, dates):
        if race_id not in race_to_date and date:
            race_to_date[race_id] = date

    unique_races = sorted(set(race_ids))
    predictions: List[pd.DataFrame] = []
    for race_id in unique_races:
        date_str = race_to_date.get(race_id)
        if date_str is None or not _within_range(date_str, start_dt, end_dt):
            continue
        df = predictor.predict(race_id, include_target=True)
        df["date"] = date_str
        predictions.append(df)

    if not predictions:
        raise RuntimeError("No races available in the specified date range.")

    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions = all_predictions[(all_predictions["odds"] >= config.backtest.min_odds) & (all_predictions["odds"] <= config.backtest.max_odds)]
    all_predictions = all_predictions[all_predictions["expected_value"] >= config.backtest.ev_threshold]

    bankroll_results = simulate_bankroll(all_predictions, config.backtest)
    bets_df: pd.DataFrame = bankroll_results["bets"]
    summary = bankroll_results["summary"]

    performance = aggregate_performance(all_predictions)
    equity = equity_curve(bets_df)

    reports_dir = Path(config.data.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    bets_path = Path(config.backtest.evaluation.bets_csv_path)
    summary_path = Path(config.backtest.evaluation.summary_csv_path)
    equity_path = Path(config.backtest.evaluation.equity_curve_path)

    bets_df.to_csv(bets_path, index=False)
    performance.to_csv(summary_path, index=False)

    # simple equity plot saved via matplotlib
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(equity["step"], equity["bankroll"], label="Equity")
    plt.xlabel("Bet Number")
    plt.ylabel("Bankroll")
    plt.title("Equity Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(equity_path)
    plt.close()

    summary.update({"bets_path": str(bets_path), "summary_path": str(summary_path), "equity_path": str(equity_path)})

    return {
        "predictions": all_predictions,
        "bets": bets_df,
        "summary": summary,
        "performance": performance,
    }
