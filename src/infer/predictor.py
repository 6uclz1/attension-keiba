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
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from src.config import ConfigBundle
from src.data.schemas import FeatureMetadata
from src.data.tensors import load_tensor_dict
from src.models.calibration import ProbabilityCalibrator
from src.models.transformer import HorseTransformerModel


class RacePredictor:
    def __init__(self, config: ConfigBundle) -> None:
        self.config = config
        self.tensor_dict, self.metadata = load_tensor_dict(config.data.processed_dir)
        self.device = torch.device(config.train.device)
        self.models: List[HorseTransformerModel] = []
        self.calibrators: List[ProbabilityCalibrator] = []
        self._load_models()

    def _load_models(self) -> None:
        model_dir = Path(self.config.train.save_dir)
        for model_path in sorted(model_dir.glob("transformer_fold*.pt")):
            fold_idx = model_path.stem.split("fold")[-1]
            calibrator_path = model_dir / f"calibrator_fold{fold_idx}.joblib"
            model = HorseTransformerModel(self.metadata, self.config.model.transformer)
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.models.append(model)
            if calibrator_path.exists():
                self.calibrators.append(ProbabilityCalibrator.load(calibrator_path))
            else:
                self.calibrators.append(ProbabilityCalibrator(method="identity"))

    def _prepare_batch(self, race_id: str) -> Dict[str, torch.Tensor]:
        mask = self.tensor_dict["race_ids"] == race_id
        if not mask.any():
            raise ValueError(f"Race {race_id} not found in processed tensors")
        batch_np = {key: value[mask] for key, value in self.tensor_dict.items() if key != "categorical_keys"}
        batch = {
            "race_id": race_id,
            "horse_ids": batch_np["horse_ids"],
            "sequence_numeric": torch.from_numpy(batch_np["sequence_numeric"]).to(self.device),
            "sequence_mask": torch.from_numpy(batch_np["sequence_mask"]).to(self.device),
            "static_numeric": torch.from_numpy(batch_np["static_numeric"]).to(self.device),
            "categorical_ids": torch.from_numpy(batch_np["categorical_ids"]).to(self.device),
            "targets": torch.from_numpy(batch_np["targets"]).to(self.device),
            "odds": torch.from_numpy(batch_np["odds"]).to(self.device),
            "race_mask": torch.ones(batch_np["sequence_numeric"].shape[0], device=self.device),
        }
        return batch

    def predict(self, race_id: str, include_target: bool = False) -> pd.DataFrame:
        batch = self._prepare_batch(race_id)
        all_probs: List[np.ndarray] = []
        for model, calibrator in zip(self.models, self.calibrators):
            outputs = model(batch)
            logits = outputs["logits"].detach().cpu().numpy()
            probs = calibrator.transform(logits)
            all_probs.append(probs)
        if not all_probs:
            raise RuntimeError("No trained models found. Run training first.")
        ensemble_probs = np.mean(all_probs, axis=0)
        odds = batch["odds"].detach().cpu().numpy()
        payload = {
            "race_id": [race_id] * len(ensemble_probs),
            "horse_id": batch["horse_ids"],
            "prob": ensemble_probs,
            "expected_value": ensemble_probs * odds,
            "odds": odds,
        }
        if include_target:
            payload["target"] = batch["targets"].detach().cpu().numpy()
        df = pd.DataFrame(payload)
        df.sort_values("prob", ascending=False, inplace=True)
        output_path = Path(self.config.data.predictions_dir) / f"race_{race_id}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df
