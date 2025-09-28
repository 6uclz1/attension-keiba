.PHONY: setup data train predict backtest serve test clean

UV ?= uv
UV_PYTHON ?= 3.10
CONFIG_DIR ?= conf
export UV_PROJECT_ENVIRONMENT ?= .venv
export UV_CACHE_DIR ?= .uv-cache
RACE_ID ?=

setup:
	$(UV) venv --python $(UV_PYTHON)
	$(UV) pip install -e .

clean:
	rm -rf data/interim/* data/processed/* models/* predictions/* reports/*

_data_raw:
	$(UV) run python scripts/generate_dummy_data.py --output-dir data/raw

data: _data_raw
	$(UV) run python -m src.cli prepare-data --config-dir $(CONFIG_DIR)

train:
	$(UV) run python -m src.cli train --config-dir $(CONFIG_DIR)

predict:
	$(UV) run python -m src.cli predict --config-dir $(CONFIG_DIR) --race-id $(RACE_ID)

backtest:
	$(UV) run python -m src.cli backtest --config-dir $(CONFIG_DIR)

serve:
	$(UV) run uvicorn src.api.serve:app --reload --port 8000

test:
	$(UV) run pytest
