# civicsense_datapack 🐍
# =============================================================================
# Python aggregation tooling for the CivicSense dataset pack.
#
# This package turns *public* driving/perception datasets into the YOLO-
# format layout that `civicsense train prepare` in the main
# driving-civicsense-vision-model repo consumes:
#
#     images/{train,val}/  <stem>.jpg
#     labels/{train,val}/  <stem>.txt   (YOLO: class x y w h, normalised)
#
# Because the heavy imagery lives in upstream archives (UA-DETRAC, COCO,
# BDD100K, CARLA exports), we do **not** commit the pixels. These scripts
# download, filter and re-annotate upstream data into the CivicSense
# vocabulary, then hand the result to the Rust validator
# (`civicsense-data training datasets/training`).
# =============================================================================

from .schema import (
    CIVICSENSE_CLASSES,
    DataSource,
    GroundTruthRecord,
    LightPhase,
    GroundTruthOutcome,
    ExpectedLevel,
)

__all__ = [
    "CIVICSENSE_CLASSES",
    "DataSource",
    "GroundTruthRecord",
    "LightPhase",
    "GroundTruthOutcome",
    "ExpectedLevel",
]
