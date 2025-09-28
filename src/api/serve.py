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

import os
from functools import lru_cache
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import ConfigBundle, load_bundle
from src.infer.predictor import RacePredictor

app = FastAPI(title="Attension Keiba API", version="0.1.0")


class PredictRequest(BaseModel):
    race_id: str
    top_k: int | None = None


class Prediction(BaseModel):
    race_id: str
    horse_id: str
    prob: float
    expected_value: float
    odds: float


class PredictResponse(BaseModel):
    predictions: List[Prediction]


@lru_cache(maxsize=1)
def _get_config() -> ConfigBundle:
    config_dir = os.environ.get("CONFIG_DIR", "conf")
    return load_bundle(config_dir)


@lru_cache(maxsize=1)
def _get_predictor() -> RacePredictor:
    return RacePredictor(_get_config())


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    predictor = _get_predictor()
    try:
        df = predictor.predict(request.race_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if request.top_k:
        df = df.head(request.top_k)
    records = [
        Prediction(
            race_id=row.race_id,
            horse_id=row.horse_id,
            prob=float(row.prob),
            expected_value=float(row.expected_value),
            odds=float(row.odds),
        )
        for row in df.itertuples(index=False)
    ]
    return PredictResponse(predictions=records)
