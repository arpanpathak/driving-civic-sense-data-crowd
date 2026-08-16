"""Shared schema definitions for the CivicSense dataset pack.

These mirror the Rust types in ``src/ground_truth.rs`` and the 7-class
vocabulary in ``src/classes.rs`` so that Python tooling and the Rust
validator agree byte-for-byte on what "valid" means.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional

# ---------------------------------------------------------------------------
# Class vocabulary (must match src/classes.rs)
# ---------------------------------------------------------------------------

CIVICSENSE_CLASSES = [
    "stop_sign",
    "traffic_light",
    "crosswalk",
    "vehicle",
    "truck",
    "bus",
    "intersection_zone",
]

# YOLO label file extension, as produced by the aggregation scripts.
YOLO_LABEL_EXT = ".txt"
# Image extensions the training pipeline accepts.
ALLOWED_IMAGE_EXTENSIONS = ("jpg", "jpeg", "png")


class DataSource(str, Enum):
    """Where a ground-truth record came from (matches Rust DataSource)."""

    MANUAL = "manual"
    V2I = "v2i"
    SIMULATOR = "simulator"


class LightPhase(str, Enum):
    """Traffic-light phase (matches Rust LightPhase)."""

    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"


class LaneLocation(str, Enum):
    """Lane location relative to ego (matches Rust LaneLocation)."""

    SAME = "same"
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


class GroundTruthOutcome(str, Enum):
    """Human-annotated physical outcome (matches Rust GroundTruthOutcome)."""

    STOPPED = "stopped"
    CLEARED = "cleared"
    BLOCKED = "blocked"


class ExpectedLevel(str, Enum):
    """Warning level the engine *should* emit (matches Rust ExpectedLevel)."""

    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


@dataclass
class GroundTruthDetection:
    """A single detected object (matches Rust GroundTruthDetection)."""

    bbox: List[float]  # [x_min, y_min, x_max, y_max] px
    class_id: int
    speed: float  # m/s
    lateral_speed: float  # m/s
    distance: float  # m
    lane: LaneLocation = LaneLocation.UNKNOWN
    turn_signal_active: bool = False

    def to_yolo_line(self) -> str:
        """Render as a normalised YOLO label line (requires known image size).
        See :func:`bbox_to_yolo` for the conversion.
        """
        raise NotImplementedError("use bbox_to_yolo(frame_w, frame_h, ...)")


@dataclass
class GroundTruthEgo:
    """Ego telemetry (matches Rust GroundTruthEgo)."""

    speed: float  # m/s
    distance_to_stop_line: float  # m


@dataclass
class GroundTruthRecord:
    """A single, annotated intersection-approach snapshot.

    Matches the Rust ``GroundTruthRecord`` so it round-trips through both
    the Python emitter and the Rust ``civicsense-data ground-truth``
    validator.
    """

    id: str
    scenario: str
    mode: DataSource
    light: LightPhase
    outcome: GroundTruthOutcome
    expected_level: ExpectedLevel
    ego: GroundTruthEgo
    time_to_red: Optional[float] = None
    detections: List[GroundTruthDetection] = field(default_factory=list)

    def to_dict(self) -> dict:
        """:return: a JSON-serialisable dict with snake_case keys."""
        return {
            "id": self.id,
            "scenario": self.scenario,
            "mode": self.mode.value,
            "light": self.light.value,
            "time_to_red": self.time_to_red,
            "outcome": self.outcome.value,
            "expected_level": self.expected_level.value,
            "ego": asdict(self.ego),
            "detections": [asdict(d) for d in self.detections],
        }


# ---------------------------------------------------------------------------
# YOLO helpers
# ---------------------------------------------------------------------------


def bbox_to_yolo(class_id: int, frame_w: int, frame_h: int, xyxy):
    """Convert an ``(x1, y1, x2, y2)`` pixel box to a YOLO normalised line.

    Coordinates are clamped to ``[0, frame_w]`` / ``[0, frame_h]`` to avoid
    emitting out-of-unit-range targets that the Rust validator rejects.
    """
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    x1 = max(0.0, min(x1, frame_w))
    x2 = max(0.0, min(x2, frame_w))
    y1 = max(0.0, min(y1, frame_h))
    y2 = max(0.0, min(y2, frame_h))
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None
    xc = (x1 + x2) / 2.0 / frame_w
    yc = (y1 + y2) / 2.0 / frame_h
    nw = w / frame_w
    nh = h / frame_h
    return f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"
