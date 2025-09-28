from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from src.data.netkeiba_fetcher import (
    _parse_race_metadata,
    _parse_runners,
    save_raw_bundle,
)
from src.data.schemas import RawDataBundle


@pytest.fixture()
def sample_soup() -> BeautifulSoup:
    html_path = Path(__file__).parent / "fixtures" / "netkeiba_race_sample.html"
    html = html_path.read_text(encoding="utf-8")
    return BeautifulSoup(html, "html.parser")


def test_parse_race_metadata(sample_soup: BeautifulSoup) -> None:
    meta = _parse_race_metadata(sample_soup, "202401010101")
    assert meta["race_id"] == "202401010101"
    assert meta["race_name"] == "2歳未勝利"
    assert meta["surface"] == "turf"
    assert meta["turn"] == "right"
    assert meta["weather"] == "sunny"
    assert meta["going"] == "firm"
    assert meta["course"] == "札幌"
    assert meta["distance_m"] == 1200
    assert meta["date"] == "2024-07-20"


def test_parse_runners(sample_soup: BeautifulSoup) -> None:
    runners = _parse_runners(sample_soup, "202401010101")
    assert len(runners) == 2
    first = runners[0]
    assert first.horse_id == "2022105244"
    assert first.jockey_id == "01197"
    assert first.trainer_id == "01192"
    assert first.time_sec == pytest.approx(68.8)
    assert first.corner_order_1 == 1
    second = runners[1]
    assert second.margin == pytest.approx(1.75)
    assert second.sex == "F"
    assert second.age == 2


def test_save_raw_bundle_deduplicates(tmp_path: Path) -> None:
    races = pd.DataFrame([
        {
            "race_id": "202401010101",
            "date": "2024-07-20",
            "course": "札幌",
            "distance_m": 1200,
            "surface": "turf",
            "going": "firm",
            "weather": "sunny",
            "turn": "right",
            "race_name": "2歳未勝利",
        }
    ])
    entries = pd.DataFrame([
        {
            "race_id": "202401010101",
            "horse_id": "2022105244",
            "jockey_id": "01197",
            "trainer_id": "01192",
            "draw": 5,
            "weight_carried": 55.0,
            "odds": 1.2,
            "popularity": 1,
        }
    ])
    results = pd.DataFrame([
        {
            "race_id": "202401010101",
            "horse_id": "2022105244",
            "finish_pos": 1,
            "time_sec": 68.8,
            "last3f_sec": 33.9,
            "margin": 0.0,
            "corner_order_1": 1,
            "corner_order_2": 1,
            "corner_order_3": 0,
            "corner_order_4": 0,
        }
    ])
    horses = pd.DataFrame([
        {
            "horse_id": "2022105244",
            "sex": "M",
            "age": 2,
            "sire": "",
            "dam": "",
            "broodmare_sire": "",
        }
    ])
    empty = pd.DataFrame()
    bundle = RawDataBundle(
        races=races,
        entries=entries,
        results=results,
        horses=horses,
        workouts=empty.copy(),
        jockeys=pd.DataFrame([["01197", "佐々木大"]], columns=["jockey_id", "name"]),
        trainers=pd.DataFrame([["01192", "上原佑紀"]], columns=["trainer_id", "name"]),
        features_aux=empty.copy(),
    )
    save_raw_bundle(bundle, tmp_path)
    save_raw_bundle(bundle, tmp_path)
    stored = pd.read_csv(tmp_path / "races.csv")
    assert len(stored) == 1
    stored_entries = pd.read_csv(tmp_path / "entries.csv")
    assert len(stored_entries) == 1
