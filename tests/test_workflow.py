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

from dataclasses import replace
from pathlib import Path

import pytest

from scripts import generate_dummy_data
from src.backtest.engine import run_backtest
from src.config import ConfigBundle, load_bundle
from src.data.pipeline import prepare_datasets
from src.infer.predictor import RacePredictor
from src.train.workflow import run_training


@pytest.fixture()
def temp_config(tmp_path: Path) -> ConfigBundle:
    config = load_bundle("conf")
    data_cfg = replace(
        config.data,
        raw_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
        models_dir=tmp_path / "models",
        predictions_dir=tmp_path / "predictions",
        reports_dir=tmp_path / "reports",
    )
    train_cfg = replace(config.train, save_dir=tmp_path / "models", epochs=1, patience=1)
    evaluation_cfg = replace(
        config.backtest.evaluation,
        bets_csv_path=tmp_path / "reports" / "bets.csv",
        summary_csv_path=tmp_path / "reports" / "summary.csv",
        equity_curve_path=tmp_path / "reports" / "equity.png",
    )
    backtest_cfg = replace(config.backtest, kelly=replace(config.backtest.kelly, bankroll=1000.0, min_stake=10.0), evaluation=evaluation_cfg)
    return ConfigBundle(data=data_cfg, model=config.model, train=train_cfg, backtest=backtest_cfg)


def test_dummy_data_generation(tmp_path: Path) -> None:
    output_dir = tmp_path / "raw"
    generate_dummy_data.main([f"--output-dir={output_dir}"])
    assert (output_dir / "races.csv").exists()
    assert (output_dir / "entries.csv").exists()


def test_end_to_end_workflow(temp_config: ConfigBundle) -> None:
    generate_dummy_data.main([f"--output-dir={temp_config.data.raw_dir}"])
    feature_df, metadata = prepare_datasets(temp_config)
    assert not feature_df.empty
    assert metadata.n_past_runs == temp_config.data.n_past_runs

    run_training(temp_config)

    predictor = RacePredictor(temp_config)
    df = predictor.predict("R2022010101", include_target=True)
    assert "prob" in df.columns
    assert not df.empty

    backtest_results = run_backtest(temp_config)
    assert "summary" in backtest_results
    summary = backtest_results["summary"]
    assert "final_bankroll" in summary
