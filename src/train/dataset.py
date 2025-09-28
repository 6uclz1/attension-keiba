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

from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import Dataset


class RaceDataset(Dataset):
    """Dataset that yields all horses belonging to a single race."""

    def __init__(self, tensor_dict: Dict[str, np.ndarray], race_ids: Iterable[str]):
        self.tensor_dict = tensor_dict
        self.all_race_ids = tensor_dict["race_ids"].astype(str)
        self.unique_race_ids = list(race_ids)
        self.indices = {
            race_id: np.where(self.all_race_ids == race_id)[0]
            for race_id in self.unique_race_ids
        }

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.unique_race_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        race_id = self.unique_race_ids[idx]
        horse_indices = self.indices[race_id]
        sample = {
            "race_id": race_id,
            "horse_ids": self.tensor_dict["horse_ids"][horse_indices],
            "sequence_numeric": torch.from_numpy(self.tensor_dict["sequence_numeric"][horse_indices]),
            "sequence_mask": torch.from_numpy(self.tensor_dict["sequence_mask"][horse_indices]),
            "static_numeric": torch.from_numpy(self.tensor_dict["static_numeric"][horse_indices]),
            "categorical_ids": torch.from_numpy(self.tensor_dict["categorical_ids"][horse_indices]),
            "targets": torch.from_numpy(self.tensor_dict["targets"][horse_indices]),
            "odds": torch.from_numpy(self.tensor_dict["odds"][horse_indices]),
            "race_mask": torch.ones(len(horse_indices)),
        }
        return sample
