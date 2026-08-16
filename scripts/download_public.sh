#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# 📥 Fetch public driving/perception datasets that feed the CivicSense
# training pack. Nothing here is committed to git: it only populates
# data/raw/ with upstream archives, which are referenced, never stored.
#
# Usage:
#   ./scripts/download_public.sh [--coco-root PATH] [--no-bdd100k] ...
#
# ⚠️ The downloads are large. Read docs/DATA_LICENSES.md and respect each
# dataset's terms before redistributing anything you derive from them.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
RAW="$ROOT/data/raw"
mkdir -p "$RAW"

log() { printf '\033[1;36m[download]\033[0m %s\n' "$*"; }

# Placeholder source lists ---------------------------------------------
# Each dataset has an official download flow. We pin the *canonical
# landing page* only here; actual keys/URLs change and should be
# configured by the operator, or fetched via the provided helper scripts.
COCO_IMAGES_URL="${COCO_IMAGES_URL:-http://images.cocodataset.org/zips/train2017.zip}"
COCO_ANN_URL="${COCO_ANN_URL:-http://images.cocodataset.org/annotations/annotations_trainval2017.zip}"

# By default we only fetch the small annotation files (fast, and enough to
# exercise the aggregator). Full image downloads are opt-in with --images.
FETCH_IMAGES=0

usage() {
  sed -n '2,12p' "${BASH_SOURCE[0]}"
  cat <<'EOF'
Options:
  --images       Also fetch full image archives (COCO train2017 ~19 GB).
  -h, --help     Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --images) FETCH_IMAGES=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1"; usage; exit 1 ;;
  esac
  shift
done

# COCO annotations (small, always fetch) -------------------------------
log "Fetching COCO annotations → $RAW/"
curl -fL "$COCO_ANN_URL" -o "$RAW/annotations_trainval2017.zip"
log "Unzipping annotations..."
unzip -oq "$RAW/annotations_trainval2017.zip" -d "$RAW/coco_annotations"

if [[ "$FETCH_IMAGES" -eq 1 ]]; then
  log "Fetching COCO train2017 images (this is ~19 GB)..."
  curl -fL "$COCO_IMAGES_URL" -o "$RAW/train2017.zip"
  log "Unzipping train2017.zip..."
  unzip -oq "$RAW/train2017.zip" -d "$RAW/coco_images"
  log "COCO images ready at $RAW/coco_images/train2017"
else
  log "Skipped image download (re-run with --images to fetch pixels)."
  log "Aggregate with: python -m civicsense_datapack.aggregate_coco ..."
fi

log "Done. Metadata/annotations are staged in $RAW ."
