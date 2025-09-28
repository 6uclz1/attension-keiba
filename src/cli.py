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

from pathlib import Path
from typing import List, Optional

import typer

from src.backtest.engine import run_backtest
from src.config import ConfigBundle, load_bundle
from src.data.netkeiba_fetcher import fetch_race_bundle, save_raw_bundle
from src.data.pipeline import prepare_datasets
from src.infer.predictor import RacePredictor
from src.train.workflow import run_training

app = typer.Typer(add_completion=False)


def _load_config(config_dir: str) -> ConfigBundle:
    return load_bundle(config_dir)


@app.command("prepare-data")
def prepare_data(config_dir: str = typer.Option("conf", help="Path to config directory.")) -> None:
    config = _load_config(config_dir)
    feature_df, metadata = prepare_datasets(config)
    typer.echo(f"Prepared {len(feature_df)} horse-race rows with {metadata.n_past_runs} past runs.")


@app.command("train")
def train(config_dir: str = typer.Option("conf", help="Path to config directory.")) -> None:
    config = _load_config(config_dir)
    results = run_training(config)
    typer.echo("Transformer fold metrics:")
    for fold_result in results["fold_results"]:
        typer.echo(f"  Fold {fold_result.fold}: {fold_result.metrics}")
    typer.echo(f"Baseline metrics: {results['baseline_metrics']}")


@app.command("predict")
def predict(
    race_id: str = typer.Option(..., "--race-id", help="Race identifier to score."),
    config_dir: str = typer.Option("conf", help="Path to config directory."),
    show_target: bool = typer.Option(False, help="Include target (if available) in output."),
) -> None:
    config = _load_config(config_dir)
    predictor = RacePredictor(config)
    df = predictor.predict(race_id, include_target=show_target)
    typer.echo(df.to_string(index=False))


@app.command("backtest")
def backtest(config_dir: str = typer.Option("conf", help="Path to config directory.")) -> None:
    config = _load_config(config_dir)
    results = run_backtest(config)
    typer.echo(f"Backtest summary: {results['summary']}")


@app.command("fetch-data")
def fetch_data(
    race_ids: List[str] = typer.Argument(..., help="One or more netkeiba race IDs (e.g. 202401010101)."),
    output_dir: Path = typer.Option(Path("data/raw"), help="Directory where raw CSVs will be saved."),
    overwrite: bool = typer.Option(False, help="Overwrite existing CSV files instead of appending."),
    timeout: float = typer.Option(15.0, help="HTTP timeout seconds for each request."),
) -> None:
    bundle = fetch_race_bundle(race_ids, timeout=timeout)
    save_raw_bundle(bundle, output_dir, overwrite=overwrite)
    typer.echo(f"Fetched {len(bundle.races)} races and saved raw tables to {output_dir}.")


if __name__ == "__main__":
    app()
