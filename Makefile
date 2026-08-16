# ── CivicSense Data Pack · Makefile ─────────────────────
# The data-pack is a companion repo: lean in git (no pixels), rich in
# tooling. It contains Rust validators + Python aggregators that turn
# public datasets into the YOLO layout CivicSense trains on.
#
# Targets:
#   make validate    → Rust unit tests + doc-tests
#   make lint        → clippy -D warnings + cargo fmt check
#   make download    → fetch public annotations into data/raw/
#   make aggregate   → run aggregation (COCO/BDD100K) into datasets/
#   make lint-py     → flake8 + black --check on the Python tooling
#   make help        → this help
#   make clean       → drop downloaded artifacts (keep the layout)

SHELL := /bin/bash
PY    := python3

.PHONY: all validate lint download aggregate lint-py help clean

all: validate lint

## ✅ Run Rust unit tests, doc-tests, and build the CLI
validate:
	cargo test

## 🧹 Strict linting following CODING_STANDARDS (borrowed from main repo)
lint:
	cargo clippy --all-targets -- -D warnings
	cargo fmt --check

## 📥 Fetch public dataset annotations (metadata only by default)
download:
	./scripts/download_public.sh

## 🧪 Aggregate COCO annotations into the training layout (example)
aggregate:
	cd python && $(PY) -m civicsense_datapack.aggregate_coco \
		--coco-ann "../data/raw/coco_annotations/annotations/instances_train2017.json" \
		--coco-images "../data/raw/coco_images" \
		--out ../datasets/training \
		--split train

## 🐍 Lint the Python tooling
lint-py:
	flake8 python/ --count --max-complexity=10 --statistics
	black --check python/

## 🧼 Clean downloaded artifacts (keeps directory layout)
clean:
	rm -rf data/raw/* data/processed/*

help:
	@echo "Targets:"
	@echo "  validate   , Rust tests + doc-tests"
	@echo "  lint       , clippy -D warnings + fmt check"
	@echo "  download   , fetch public annotations into data/raw/"
	@echo "  aggregate  , COCO → datasets/training (example)"
	@echo "  lint-py    , flake8 + black check"
	@echo "  clean      , drop downloaded artifacts"
