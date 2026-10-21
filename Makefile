.PHONY: help install lint typecheck test test-web check data detect features tune bench edge figures web-data \
        web-dev web-build dashboard docs docs-serve demo docker clean

PY ?= python
SEQS ?= MOT17-02 MOT17-04 MOT17-05 MOT17-09 MOT17-10 MOT17-11 MOT17-13

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## install the package with dev + dashboard + docs extras
	$(PY) -m pip install -e ".[dev,dashboard,docs,agent]"

lint:  ## ruff
	ruff check src tests dashboard scripts

typecheck:  ## mypy
	mypy src dashboard

test:  ## unit tests with coverage
	pytest tests/unit --cov=emot --cov-report=term-missing

test-web:  ## TypeScript tests (cross-language parity with the Python implementation)
	cd web && npm ci && npm test

check: lint typecheck test test-web  ## everything CI runs (minus integration)

data:  ## download MOT17 (7 sequences, public HF mirror)
	$(PY) src/emot/data/download_mot17.py --output data/mot17 --sequences $(SEQS)

detect:  ## YOLO detections for every (sequence, backbone), cached
	$(PY) src/emot/training/run_detection_cache.py

features:  ## ReID embeddings + camera-motion caches
	$(PY) src/emot/training/build_feature_caches.py

tune:  ## Optuna search for every tracker (HOTA objective, MOT17-02 val split)
	$(PY) src/emot/training/tune_trackers.py

bench:  ## full tracking grid (7 sequences x 6 backbones x 7 tracker variants) + detection AP
	$(PY) src/emot/training/run_tracking_benchmark.py

edge:  ## edge benchmark (ONNX, INT8, CoreML, threads, resolution) -- run on an idle machine
	$(PY) src/emot/training/run_edge_benchmark.py

figures:  ## regenerate docs/figures from results/
	$(PY) scripts/make_figures.py

web-data:  ## export clips, tracks and results for the web app
	$(PY) scripts/export_web_data.py

web-dev:  ## web app dev server
	cd web && npm install && npm run dev

web-build:  ## production build of the web app
	cd web && npm ci && npm run build

dashboard:  ## local Plotly Dash lab (reads results/ and cache/)
	$(PY) dashboard/app.py

docs:  ## build the documentation site (strict)
	mkdocs build --strict

docs-serve:  ## live-reload docs
	mkdocs serve

demo:  ## track the bundled sample clip and write an annotated video
	emot track web/public/data/clips/MOT17-04.mp4 --tracker botsort --save outputs/demo.mp4

docker:  ## build the container image
	docker build -t emot .

clean:  ## remove caches and build output
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov site web/dist outputs
