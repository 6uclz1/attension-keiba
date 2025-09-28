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

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

RACES = [
    ("R2022010101", date(2022, 1, 1), "Tokyo", 1600, "turf", "firm", "sunny", "left", "New Year Mile"),
    ("R2022011501", date(2022, 1, 15), "Kyoto", 1800, "turf", "good", "cloudy", "right", "Winter Stakes"),
    ("R2022020101", date(2022, 2, 1), "Nakayama", 2000, "turf", "yielding", "rainy", "right", "Spring Trial"),
    ("R2022030501", date(2022, 3, 5), "Hanshin", 1400, "dirt", "standard", "sunny", "right", "Hanshin Dash"),
    ("R2022040201", date(2022, 4, 2), "Tokyo", 2000, "turf", "firm", "cloudy", "left", "April Cup"),
    ("R2022050101", date(2022, 5, 1), "Kyoto", 2200, "turf", "good", "sunny", "right", "Golden Week Prize"),
]

HORSES = [
    ("H001", "M", 4, "SireA", "DamA", "BmSireA"),
    ("H002", "F", 3, "SireB", "DamB", "BmSireB"),
    ("H003", "M", 5, "SireC", "DamC", "BmSireC"),
    ("H004", "G", 6, "SireD", "DamD", "BmSireD"),
    ("H005", "F", 4, "SireE", "DamE", "BmSireE"),
    ("H006", "M", 3, "SireF", "DamF", "BmSireF"),
]

JOCKEYS = [
    ("J001", "J. Tanaka"),
    ("J002", "K. Sato"),
    ("J003", "M. Suzuki"),
    ("J004", "Y. Watanabe"),
]

TRAINERS = [
    ("T001", "Stable North"),
    ("T002", "Stable East"),
    ("T003", "Stable West"),
]

PAST_RESULTS_TEMPLATE = {
    "time_sec": (92.0, 115.0),
    "last3f_sec": (34.0, 37.5),
    "margin": (-0.5, 2.5),
}


def _sample_float(low: float, high: float, size: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(low, high, size)


def _generate_entries(races: List[tuple]) -> pd.DataFrame:
    rows = []
    horse_cycle = [h[0] for h in HORSES]
    jockey_cycle = [j[0] for j in JOCKEYS]
    trainer_cycle = [t[0] for t in TRAINERS]
    odds_values = [2.5, 3.2, 5.0, 8.5, 12.0, 18.0]
    popularity_values = [1, 2, 3, 4, 5, 6]
    weight_values = [55.0, 56.5, 57.0, 58.0, 54.0, 55.5]

    for idx, race in enumerate(races):
        race_id = race[0]
        for lane in range(3):
            horse_id = horse_cycle[(idx + lane) % len(horse_cycle)]
            jockey_id = jockey_cycle[(idx + lane) % len(jockey_cycle)]
            trainer_id = trainer_cycle[(idx + lane) % len(trainer_cycle)]
            rows.append(
                {
                    "race_id": race_id,
                    "horse_id": horse_id,
                    "jockey_id": jockey_id,
                    "trainer_id": trainer_id,
                    "draw": lane + 1,
                    "weight_carried": weight_values[(idx + lane) % len(weight_values)],
                    "odds": odds_values[(idx + lane) % len(odds_values)],
                    "popularity": popularity_values[(idx + lane) % len(popularity_values)],
                }
            )
    return pd.DataFrame(rows)


def _generate_results(entries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for race_id, group in entries.groupby("race_id"):
        finish_positions = list(range(1, len(group) + 1))
        np.random.shuffle(finish_positions)
        times = _sample_float(*PAST_RESULTS_TEMPLATE["time_sec"], size=len(group))
        last3f = _sample_float(*PAST_RESULTS_TEMPLATE["last3f_sec"], size=len(group))
        margins = _sample_float(*PAST_RESULTS_TEMPLATE["margin"], size=len(group))
        for i, (_, row) in enumerate(group.iterrows()):
            rows.append(
                {
                    "race_id": race_id,
                    "horse_id": row["horse_id"],
                    "finish_pos": finish_positions[i],
                    "time_sec": round(times[i], 2),
                    "last3f_sec": round(last3f[i], 2),
                    "margin": round(margins[i], 2),
                    "corner_order_1": np.random.randint(1, 10),
                    "corner_order_2": np.random.randint(1, 10),
                    "corner_order_3": np.random.randint(1, 10),
                    "corner_order_4": np.random.randint(1, 10),
                }
            )
    return pd.DataFrame(rows)


def _generate_workouts(horses: List[tuple]) -> pd.DataFrame:
    rows = []
    base_date = date(2021, 12, 1)
    for idx, horse in enumerate(horses):
        for session in range(3):
            rows.append(
                {
                    "horse_id": horse[0],
                    "date": base_date + timedelta(days=7 * session + idx),
                    "course": np.random.choice(["Miho", "Ritto"]),
                    "clock": round(np.random.uniform(50.0, 53.0), 2),
                    "evaluation": np.random.choice(["A", "B", "C"]),
                }
            )
    return pd.DataFrame(rows)


def _generate_features_aux(horses: List[tuple]) -> pd.DataFrame:
    rows = []
    for horse in horses:
        rows.append(
            {
                "horse_id": horse[0],
                "course_fitness": round(np.random.uniform(0.4, 0.9), 2),
                "distance_fitness": round(np.random.uniform(0.4, 0.9), 2),
            }
        )
    return pd.DataFrame(rows)


def _write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate dummy racing CSV data")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    races_df = pd.DataFrame(RACES, columns=[
        "race_id",
        "date",
        "course",
        "distance_m",
        "surface",
        "going",
        "weather",
        "turn",
        "race_name",
    ])
    races_df["date"] = pd.to_datetime(races_df["date"])

    entries_df = _generate_entries(RACES)
    results_df = _generate_results(entries_df)
    horses_df = pd.DataFrame(HORSES, columns=["horse_id", "sex", "age", "sire", "dam", "broodmare_sire"])
    workouts_df = _generate_workouts(HORSES)
    jockeys_df = pd.DataFrame(JOCKEYS, columns=["jockey_id", "name"])
    trainers_df = pd.DataFrame(TRAINERS, columns=["trainer_id", "name"])
    features_aux_df = _generate_features_aux(HORSES)

    _write_dataframe(races_df, output_dir / "races.csv")
    _write_dataframe(entries_df, output_dir / "entries.csv")
    _write_dataframe(results_df, output_dir / "results.csv")
    _write_dataframe(horses_df, output_dir / "horses.csv")
    _write_dataframe(workouts_df, output_dir / "workouts.csv")
    _write_dataframe(jockeys_df, output_dir / "jockeys.csv")
    _write_dataframe(trainers_df, output_dir / "trainers.csv")
    _write_dataframe(features_aux_df, output_dir / "features_aux.csv")

    print(f"Dummy racing data written to {output_dir}")


if __name__ == "__main__":
    main()
