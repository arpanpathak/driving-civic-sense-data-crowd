"""Unit tests for the CivicSense data-pack schema helpers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from civicsense_datapack.schema import (  # noqa: E402
    bbox_to_yolo,
    CIVICSENSE_CLASSES,
    GroundTruthRecord,
    GroundTruthEgo,
    DataSource,
    LightPhase,
    GroundTruthOutcome,
    ExpectedLevel,
)


def test_vocabulary_has_expected_order():
    assert CIVICSENSE_CLASSES == [
        "stop_sign",
        "traffic_light",
        "crosswalk",
        "vehicle",
        "truck",
        "bus",
        "intersection_zone",
    ]
    assert len(CIVICSENSE_CLASSES) == 7


def test_bbox_to_yolo_centres_and_normalises():
    line = bbox_to_yolo(3, 1280, 720, (100, 200, 500, 600))
    parts = [float(v) for v in line.split()]
    assert parts[0] == 3.0
    # centre of 100..500 is 300/1280
    assert abs(parts[1] - 300.0 / 1280.0) < 1e-6
    # centre of 200..600 is 400/720
    assert abs(parts[2] - 400.0 / 720.0) < 1e-6
    assert abs(parts[3] - 400.0 / 1280.0) < 1e-6
    assert abs(parts[4] - 400.0 / 720.0) < 1e-6


def test_bbox_to_yolo_clamps_out_of_range():
    # Coordinates beyond the frame are clamped, not emitted out-of-range.
    line = bbox_to_yolo(0, 100, 100, (-50, -50, 200, 200))
    _, xc, yc, w, h = (float(v) for v in line.split())
    assert 0.0 <= xc <= 1.0
    assert 0.0 <= yc <= 1.0
    assert not (w > 1.0 or h > 1.0)


def test_bbox_to_yolo_rejects_degenerate_box():
    assert bbox_to_yolo(3, 100, 100, (50, 50, 50, 50)) is None


def test_record_to_dict_is_snake_case():
    rec = GroundTruthRecord(
        id="gt_001",
        scenario="Main & 5th",
        mode=DataSource.MANUAL,
        light=LightPhase.YELLOW,
        outcome=GroundTruthOutcome.CLEARED,
        expected_level=ExpectedLevel.SAFE,
        ego=GroundTruthEgo(speed=12.0, distance_to_stop_line=40.0),
        time_to_red=3.5,
    )
    d = rec.to_dict()
    assert d["mode"] == "manual"
    assert d["light"] == "yellow"
    assert d["ego"]["speed"] == 12.0
    assert d["detections"] == []


if __name__ == "__main__":
    import traceback

    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok - {name}")
            except Exception:
                failed += 1
                print(f"FAIL - {name}")
                traceback.print_exc()
    print(f"\n{len([n for n in globals() if n.startswith('test_')]) - failed}/"
          f"{len([n for n in globals() if n.startswith('test_')])} passed")
    sys.exit(1 if failed else 0)
