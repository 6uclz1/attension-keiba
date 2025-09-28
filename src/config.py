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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml


@dataclass
class CalibrationSettings:
    method: str = "platt"
    min_prob: float = 1e-5
    max_prob: float = 1 - 1e-5

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalibrationSettings":
        return cls(**data)


@dataclass
class DataConfig:
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    models_dir: Path
    predictions_dir: Path
    reports_dir: Path
    n_past_runs: int
    sequence_features: List[str]
    static_numeric_features: List[str]
    categorical_features: Dict[str, str]
    id_columns: Dict[str, str]
    odds_column: str
    results_column: str
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DataConfig":
        calibration = CalibrationSettings.from_dict(data.get("calibration", {}))
        path_fields = {
            key: Path(data[key])
            for key in [
                "raw_dir",
                "interim_dir",
                "processed_dir",
                "models_dir",
                "predictions_dir",
                "reports_dir",
            ]
        }
        return cls(
            calibration=calibration,
            **path_fields,
            n_past_runs=int(data.get("n_past_runs", 5)),
            sequence_features=list(data.get("sequence_features", [])),
            static_numeric_features=list(data.get("static_numeric_features", [])),
            categorical_features=dict(data.get("categorical_features", {})),
            id_columns=dict(data.get("id_columns", {})),
            odds_column=str(data.get("odds_column", "odds")),
            results_column=str(data.get("results_column", "finish_pos")),
        )


@dataclass
class TransformerConfig:
    d_model: int
    n_heads: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    sequence_length: int
    static_dim: int
    categorical_embedding_dim: int
    use_horse_interaction_attention: bool = False
    interaction_heads: int = 2
    positional_encoding: str = "sinusoidal"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TransformerConfig":
        return cls(
            d_model=int(data.get("d_model", 64)),
            n_heads=int(data.get("n_heads", 4)),
            num_layers=int(data.get("num_layers", 2)),
            dim_feedforward=int(data.get("dim_feedforward", 128)),
            dropout=float(data.get("dropout", 0.1)),
            sequence_length=int(data.get("sequence_length", 6)),
            static_dim=int(data.get("static_dim", 16)),
            categorical_embedding_dim=int(data.get("categorical_embedding_dim", 16)),
            use_horse_interaction_attention=bool(data.get("use_horse_interaction_attention", False)),
            interaction_heads=int(data.get("interaction_heads", 2)),
            positional_encoding=str(data.get("positional_encoding", "sinusoidal")),
        )


@dataclass
class BaselineLGBMConfig:
    num_leaves: int
    learning_rate: float
    n_estimators: int
    subsample: float
    colsample_bytree: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BaselineLGBMConfig":
        return cls(
            num_leaves=int(data.get("num_leaves", 31)),
            learning_rate=float(data.get("learning_rate", 0.1)),
            n_estimators=int(data.get("n_estimators", 100)),
            subsample=float(data.get("subsample", 1.0)),
            colsample_bytree=float(data.get("colsample_bytree", 1.0)),
        )


@dataclass
class ModelConfig:
    transformer: TransformerConfig
    baseline_lgbm: BaselineLGBMConfig

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelConfig":
        transformer = TransformerConfig.from_dict(data.get("transformer", {}))
        baseline = BaselineLGBMConfig.from_dict(data.get("baseline_lgbm", {}))
        return cls(transformer=transformer, baseline_lgbm=baseline)


@dataclass
class OptunaConfig:
    enabled: bool = False
    n_trials: int = 10

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OptunaConfig":
        return cls(
            enabled=bool(data.get("enabled", False)),
            n_trials=int(data.get("n_trials", 10)),
        )


@dataclass
class TrainConfig:
    seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    pos_weight: float
    n_folds: int
    num_workers: int
    device: str
    log_every_n_steps: int
    save_dir: Path
    calibration: CalibrationSettings
    optuna: OptunaConfig

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainConfig":
        calibration = CalibrationSettings.from_dict(data.get("calibration", {}))
        optuna_cfg = OptunaConfig.from_dict(data.get("optuna", {}))
        return cls(
            seed=int(data.get("seed", 2024)),
            batch_size=int(data.get("batch_size", 32)),
            epochs=int(data.get("epochs", 10)),
            learning_rate=float(data.get("learning_rate", 1e-3)),
            weight_decay=float(data.get("weight_decay", 0.0)),
            patience=int(data.get("patience", 3)),
            pos_weight=float(data.get("pos_weight", 1.0)),
            n_folds=int(data.get("n_folds", 5)),
            num_workers=int(data.get("num_workers", 0)),
            device=str(data.get("device", "cpu")),
            log_every_n_steps=int(data.get("log_every_n_steps", 10)),
            save_dir=Path(data.get("save_dir", "models")),
            calibration=calibration,
            optuna=optuna_cfg,
        )


@dataclass
class KellyConfig:
    enabled: bool = True
    safety: float = 0.5
    bankroll: float = 100000.0
    min_stake: float = 100.0
    max_fraction: float = 0.1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KellyConfig":
        return cls(
            enabled=bool(data.get("enabled", True)),
            safety=float(data.get("safety", 0.5)),
            bankroll=float(data.get("bankroll", 100000.0)),
            min_stake=float(data.get("min_stake", 100.0)),
            max_fraction=float(data.get("max_fraction", 0.1)),
        )


@dataclass
class EvaluationOutputs:
    frequency: str = "race"
    equity_curve_path: Path = Path("reports/equity_curve.png")
    bets_csv_path: Path = Path("reports/bets.csv")
    summary_csv_path: Path = Path("reports/backtest_summary.csv")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationOutputs":
        return cls(
            frequency=str(data.get("frequency", "race")),
            equity_curve_path=Path(data.get("equity_curve_path", "reports/equity_curve.png")),
            bets_csv_path=Path(data.get("bets_csv_path", "reports/bets.csv")),
            summary_csv_path=Path(data.get("summary_csv_path", "reports/backtest_summary.csv")),
        )


@dataclass
class BacktestConfig:
    start_date: str
    end_date: str
    min_odds: float
    max_odds: float
    ev_threshold: float
    kelly: KellyConfig
    evaluation: EvaluationOutputs

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BacktestConfig":
        return cls(
            start_date=str(data.get("start_date")),
            end_date=str(data.get("end_date")),
            min_odds=float(data.get("min_odds", 1.0)),
            max_odds=float(data.get("max_odds", 200.0)),
            ev_threshold=float(data.get("ev_threshold", 1.0)),
            kelly=KellyConfig.from_dict(data.get("kelly", {})),
            evaluation=EvaluationOutputs.from_dict(data.get("evaluation", {})),
        )


@dataclass
class ConfigBundle:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    backtest: BacktestConfig


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def load_bundle(config_dir: Path | str) -> ConfigBundle:
    config_path = Path(config_dir)
    data_cfg = DataConfig.from_dict(_load_yaml(config_path / "data.yaml"))
    model_cfg = ModelConfig.from_dict(_load_yaml(config_path / "model.yaml"))
    train_cfg = TrainConfig.from_dict(_load_yaml(config_path / "train.yaml"))
    backtest_cfg = BacktestConfig.from_dict(_load_yaml(config_path / "backtest.yaml"))
    return ConfigBundle(data=data_cfg, model=model_cfg, train=train_cfg, backtest=backtest_cfg)
