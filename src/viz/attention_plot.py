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
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_sequence_attention(weights: np.ndarray, output_path: Path, title: str = "Sequence Attention") -> None:
    if weights.ndim == 1:
        weights = weights[None, :]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.imshow(weights, aspect="auto", cmap="viridis")
    plt.colorbar(label="Attention weight")
    plt.xlabel("Past runs")
    plt.ylabel("Heads")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_interaction_attention(matrix: np.ndarray, horse_ids: Optional[list[str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(matrix, cmap="coolwarm", vmin=0, vmax=matrix.max() if matrix.size else 1)
    plt.colorbar(label="Attention weight")
    if horse_ids:
        plt.xticks(range(len(horse_ids)), horse_ids, rotation=45, ha="right")
        plt.yticks(range(len(horse_ids)), horse_ids)
    plt.title("Horse Interaction Attention")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
