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

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from src.backtest.kelly import kelly_fraction
from src.config import BacktestConfig


@dataclass
class BetRecord:
    race_id: str
    horse_id: str
    prob: float
    odds: float
    stake: float
    result: float
    pnl: float
    bankroll: float


def simulate_bankroll(predictions: pd.DataFrame, config: BacktestConfig) -> Dict[str, object]:
    bankroll = config.kelly.bankroll
    records: List[BetRecord] = []
    max_bankroll = bankroll
    max_drawdown = 0.0

    for _, row in predictions.iterrows():
        prob = float(row["prob"])
        odds = max(float(row["odds"]), 1.0)
        result = float(row.get("target", 0.0))
        if config.kelly.enabled:
            fraction = float(kelly_fraction(np.array([prob]), np.array([odds]), safety=config.kelly.safety)[0])
            fraction = min(fraction, config.kelly.max_fraction)
            stake = max(fraction * bankroll, config.kelly.min_stake)
        else:
            stake = config.kelly.min_stake
        stake = min(stake, bankroll)
        payout = stake * (odds - 1.0)
        pnl = payout if result > 0.5 else -stake
        bankroll += pnl
        max_bankroll = max(max_bankroll, bankroll)
        drawdown = (max_bankroll - bankroll) / max(max_bankroll, 1e-6)
        max_drawdown = max(max_drawdown, drawdown)

        records.append(
            BetRecord(
                race_id=row["race_id"],
                horse_id=row["horse_id"],
                prob=prob,
                odds=odds,
                stake=stake,
                result=result,
                pnl=pnl,
                bankroll=bankroll,
            )
        )

    bets_df = pd.DataFrame([record.__dict__ for record in records])
    summary = {
        "final_bankroll": bankroll,
        "n_bets": len(records),
        "roi": (bankroll - config.kelly.bankroll) / max(config.kelly.bankroll, 1e-6),
        "max_drawdown": max_drawdown,
    }
    return {"bets": bets_df, "summary": summary}
