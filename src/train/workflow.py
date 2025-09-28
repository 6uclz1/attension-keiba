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
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd

from src.config import ConfigBundle
from src.data.schemas import FeatureMetadata
from src.data.tensors import load_tensor_dict
from src.models.baseline_lgbm import train_lightgbm_baseline
from src.train.trainer import FoldResult, train_transformer


def run_training(config: ConfigBundle) -> Dict[str, object]:
    tensor_dict, metadata = load_tensor_dict(config.data.processed_dir)
    feature_path = Path(config.data.processed_dir) / "features.pkl"
    if not feature_path.exists():
        raise FileNotFoundError("Feature dataframe not found. Run prepare-data first.")
    feature_df = pd.read_pickle(feature_path)

    fold_results = train_transformer(tensor_dict, metadata, config)
    baseline_model, baseline_metrics = train_lightgbm_baseline(feature_df, metadata, config.model.baseline_lgbm)

    baseline_path = Path(config.train.save_dir) / "baseline_lgbm.joblib"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline_model, baseline_path)

    metrics_payload = {
        "transformer": [result.metrics for result in fold_results],
        "transformer_calibrated": [result.calibrated_metrics for result in fold_results],
        "baseline": baseline_metrics,
    }

    report_path = Path(config.data.reports_dir) / "metrics_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2)

    return {
        "fold_results": fold_results,
        "baseline_metrics": baseline_metrics,
    }
