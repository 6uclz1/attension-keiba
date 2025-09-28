# Attension Keiba MVP

最小構成の競馬予測基盤リポジトリです。過去走系列に対するTransformerエンコーダと、レース条件をクエリとしたConditional Attentionで単勝勝率を推定し、確率較正・期待値算出・バックテストまで一気通貫で再現できます。以下はMVPでのコンポーネント関係を示す概略図です。

```
┌──────────────┐   ┌────────────────┐   ┌────────────────────┐
│ Raw CSV (@data│→│ Feature Builder │→│ Sequence Tensorizer │→┐
│ /raw)         │   └────────────────┘   └────────────────────┘ │
└──────────────┘                                              │
       │                                                       ▼
       │   ┌──────────────────────────────┐   ┌─────────────────────────┐
       └→ │ Horse Transformer (History + │→│ Calibrator (Platt/ISO) │
           │ Conditional + Set Attention) │   └─────────────────────────┘
           └──────────────────────────────┘                │
                    │                                     ▼
                    ▼                          ┌──────────────────────────┐
            ┌─────────────┐                    │ Backtest (Kelly / ROI) │
            │ LightGBM    │                    └──────────────────────────┘
            │ Baseline    │                                   │
            └─────────────┘                                   ▼
                                               ┌────────────────────────┐
                                               │ Analytics & Visuals   │
                                               └────────────────────────┘
```

## セットアップ

1. Python 3.10+ と [uv](https://github.com/astral-sh/uv) を準備
2. リポジトリ直下で依存導入 (`uv` が `.venv/` に仮想環境を作成します):

   ```bash
   make setup
   ```

3. ダミーデータ生成と前処理:

   ```bash
   make data
   ```

4. 学習と予測・バックテスト（必要に応じて）:

   ```bash
   make train
   make predict RACE_ID=R2022010101
   make backtest
   ```

### 実データの取得

`netkeiba` の公開レースページから直接スクレイピングして実データを取得するユーティリティを追加しました。以下のコマンドで指定したレースIDのCSV群を `data/raw/` に保存できます（既存ファイルには追記され、重複行は排除されます）。

```bash
uv run python -m src.cli fetch-data 202401010101 202401010102
```

保存先は `--output-dir` で変更でき、`--overwrite` を指定すると既存CSVを上書きします。スクレイピング対象サイトの利用規約を確認した上でご利用ください。

## データスキーマ

`data/raw/` に以下のCSVを想定しています（`scripts/generate_dummy_data.py`で生成可能）。

| ファイル | 主な列 |
| --- | --- |
| races.csv | race_id, date, course, distance_m, surface, going, weather, turn, race_name |
| entries.csv | race_id, horse_id, jockey_id, trainer_id, draw, weight_carried, odds, popularity |
| results.csv | race_id, horse_id, finish_pos, time_sec, last3f_sec, margin, corner_order_1..4 |
| horses.csv | horse_id, sex, age, sire, dam, broodmare_sire |
| workouts.csv | horse_id, date, course, clock, evaluation |
| jockeys.csv | jockey_id, name |
| trainers.csv | trainer_id, name |
| features_aux.csv | horse_id, course_fitness, distance_fitness |

前処理後は `data/interim/master_table.parquet` と `data/processed/` (特徴テンソル, メタデータ) が生成されます。

## 機能フロー

- `make data` → CSV読込 (`src/data/loader.py`) → 前処理 (`src/data/preprocess.py`) → 特徴量化 (`src/data/features.py`) → テンソル保存 (`src/data/tensors.py`)
- `make train` → Transformer学習 (`src/train/trainer.py`), 較正器保存 (`src/models/calibration.py`), Baseline LightGBM (`src/models/baseline_lgbm.py`)
- `make predict` → `RacePredictor` (`src/infer/predictor.py`) で単レース推論
- `make backtest` → バックテスト (`src/backtest/engine.py`) でKelly計算・資金曲線出力

## 指標と可視化

- 基本指標: AUC, LogLoss, Brier, EV, ROI
- キャリブレーション: `src/viz/calibration_plot.py`
- アテンション可視化: `src/viz/attention_plot.py`
- 資金曲線: `src/viz/equity_plot.py`

バックテスト実行後、`reports/` に `bets.csv`, `backtest_summary.csv`, `equity_curve.png` が保存されます。

## CLI / API

- CLI (`src/cli.py`):
  - `fetch-data`: netkeiba から指定レースIDの生データを取得
  - `prepare-data`: 特徴テンソル作成
  - `train`: Transformer + Baseline 学習
  - `predict --race-id ...`: 単レース推論
  - `backtest`: 期間バックテスト
- API (`src/api/serve.py`): FastAPI で `POST /predict {"race_id": "..."}` を受け付け。

## 再現性・実験管理

- Hydra風のYAML設定を `conf/` に配置
- 乱数固定 (`seed`)
- 保存物: `models/` (学習済みモデル, 較正器, LightGBM), `predictions/` (推論CSV), `reports/` (評価サマリ)

## 拡張の例

1. 着順分布モデリング (Plackett-Luce / Deep Plackett)
2. 連系馬券（馬連・三連系）の期待値最適化
3. ペース・ラップ、調教指数の高度化
4. 血統・厩舎・騎手の時系列動向やグラフ埋め込み
5. 実データ(API, DB)接続及び自動更新

## テスト

`pytest` でデータパイプラインと主要ワークフローの最小テストを実行できます。

```bash
make test
```
