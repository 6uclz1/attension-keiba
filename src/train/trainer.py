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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import ConfigBundle
from src.data.schemas import FeatureMetadata
from src.models.calibration import ProbabilityCalibrator
from src.models.transformer import HorseTransformerModel
from src.train.cv import race_based_folds
from src.train.dataset import RaceDataset
from src.train.metrics import combined_metrics


@dataclass
class FoldResult:
    fold: int
    model_path: Path
    calibrator_path: Path
    metrics: Dict[str, float]
    calibrated_metrics: Dict[str, float]
    prediction_path: Path


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _detach_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def train_transformer(  # noqa: PLR0912
    tensor_dict: Dict[str, np.ndarray],
    metadata: FeatureMetadata,
    config: ConfigBundle,
) -> List[FoldResult]:
    _set_seed(config.train.seed)
    device = torch.device(config.train.device)
    results: List[FoldResult] = []
    model_save_dir = config.train.save_dir
    prediction_dir = Path(config.data.predictions_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    model_save_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_races, val_races) in enumerate(
        race_based_folds(tensor_dict["race_ids"], config.train.n_folds, config.train.seed)
    ):
        model = HorseTransformerModel(metadata, config.model.transformer).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config.train.pos_weight], device=device))

        train_dataset = RaceDataset(tensor_dict, train_races)
        val_dataset = RaceDataset(tensor_dict, val_races)
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=config.train.num_workers,
            collate_fn=lambda x: x[0],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.train.num_workers,
            collate_fn=lambda x: x[0],
        )

        best_val_loss = float("inf")
        best_state: Dict[str, torch.Tensor] | None = None
        patience_counter = 0

        for epoch in range(config.train.epochs):
            model.train()
            for batch in train_loader:
                batch = _to_device(batch, device)
                optimizer.zero_grad()
                outputs = model(batch)
                targets = batch["targets"].to(device)
                loss = criterion(outputs["logits"], targets)
                loss.backward()
                optimizer.step()

            model.eval()
            val_losses: List[float] = []
            val_probs: List[np.ndarray] = []
            val_logits: List[np.ndarray] = []
            val_targets: List[np.ndarray] = []
            val_odds: List[np.ndarray] = []
            val_race_ids: List[str] = []
            val_horse_ids: List[np.ndarray] = []
            with torch.no_grad():
                for batch in val_loader:
                    race_id = batch["race_id"]
                    horse_ids = batch["horse_ids"]
                    batch_t = _to_device(batch, device)
                    outputs = model(batch_t)
                    targets = batch_t["targets"]
                    loss = criterion(outputs["logits"], targets)
                    val_losses.append(float(loss.detach().cpu()))
                    val_probs.append(_detach_numpy(outputs["probs"]))
                    val_logits.append(_detach_numpy(outputs["logits"]))
                    val_targets.append(_detach_numpy(targets))
                    val_odds.append(_detach_numpy(batch_t["odds"]))
                    val_race_ids.extend([race_id] * len(horse_ids))
                    val_horse_ids.append(np.array(horse_ids))

            mean_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.train.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        y_prob = np.concatenate(val_probs) if val_probs else np.array([])
        y_logits = np.concatenate(val_logits) if val_logits else np.array([])
        y_true = np.concatenate(val_targets) if val_targets else np.array([])
        y_odds = np.concatenate(val_odds) if val_odds else np.array([])
        y_horse_ids = np.concatenate(val_horse_ids) if val_horse_ids else np.array([])
        y_race_ids = np.array(val_race_ids)

        metrics = combined_metrics(y_true, y_prob, y_odds) if y_true.size else {}

        calibrator = ProbabilityCalibrator(method=config.train.calibration.method)
        if y_logits.size:
            calibrator.fit(y_logits, y_true)
            calibrated_probs = calibrator.transform(y_logits)
            calibrated_metrics = combined_metrics(y_true, calibrated_probs, y_odds)
        else:
            calibrated_probs = y_prob
            calibrated_metrics = metrics

        model_path = model_save_dir / f"transformer_fold{fold_idx}.pt"
        calibrator_path = model_save_dir / f"calibrator_fold{fold_idx}.joblib"
        prediction_path = prediction_dir / f"val_predictions_fold{fold_idx}.csv"

        torch.save(best_state or model.state_dict(), model_path)
        calibrator.save(calibrator_path)

        if y_true.size:
            df = pd.DataFrame(
                {
                    "race_id": y_race_ids,
                    "horse_id": y_horse_ids,
                    "prob_raw": y_prob,
                    "prob_calibrated": calibrated_probs,
                    "target": y_true,
                    "odds": y_odds,
                }
            )
            df.to_csv(prediction_path, index=False)

        results.append(
            FoldResult(
                fold=fold_idx,
                model_path=model_path,
                calibrator_path=calibrator_path,
                metrics=metrics,
                calibrated_metrics=calibrated_metrics,
                prediction_path=prediction_path,
            )
        )

    summary_path = Path(config.data.reports_dir) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump([result.__dict__ for result in results], fp, indent=2, default=str)

    return results
