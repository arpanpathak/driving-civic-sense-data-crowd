#!/usr/bin/env python3
"""Aggregate BDD100K detection labels into the CivicSense YOLO layout.

BDD100K is a large, diverse driving dataset (100k images, urban/suburban/
highway, all weather and times of day) that is ideal for generalising the
CivicSense detector. Its detection labels are JSON, one file per image:
``det_20/<name>.json`` with ``[label]`` entries holding ``box2d`` and
``category``.

Usage
-----
    python -m civicsense_datapack.aggregate_bdd100k \
        --root /path/to/bdd100k \
        --images bdd100k/images/100k/train \
        --labels bdd100k/labels/det_20/train \
        --out datasets/training \
        --split train

Class mapping
-------------
Only categories overlapping the CivicSense vocabulary are kept:
car/bus/truck/traffic light/other vehicle -> vehicle classes, traffic
light and stop sign by name.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

_BDP_MAP = {
    "car": 3,  # vehicle
    "bus": 5,
    "truck": 4,
    "other vehicle": 3,  # vehicle
    "traffic light": 1,
    "stop sign": 0,
}


def aggregate(labels_root: Path, images_root: Path, out: Path, split: str, limit: int):
    out_images = out / "images" / split
    out_labels = out / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    written = 0
    stars = sorted(labels_root.glob("*.json"))
    if limit > 0:
        stars = stars[:limit]

    for lab in stars:
        with lab.open() as fh:
            objs = json.load(fh)
        records = []
        for o in objs:
            box = o.get("box2d")
            cid = _BDP_MAP.get(o.get("category", "").lower())
            if box is None or cid is None:
                continue
            records.append((cid, (box["x1"], box["y1"], box["x2"], box["y2"])))
        if not records:
            continue

        src_img = images_root / (lab.stem + ".jpg")
        if not src_img.exists():
            continue

        # BMP images are common in BDD100K; convert to jpg-safe copy only for
        # png/jpg sources. BDD100K images are jpg, so direct copy is fine.
        dst_img = out_images / (lab.stem + ".jpg")
        shutil.copy2(src_img, dst_img)

        from .schema import bbox_to_yolo

        img = _image_size(src_img)
        lines = [
            bbox_to_yolo(cid, img[0], img[1], b)
            for cid, b in records
        ]
        lines = [ln for ln in lines if ln is not None]
        if not lines:
            dst_img.unlink()
            continue
        (out_labels / (lab.stem + ".txt")).write_text("\n".join(lines) + "\n")
        written += 1

    print(f"[bdd100k] wrote {written} labelled images to {out_images.parent}")


def _image_size(path: Path):
    """Return (width, height) for a JPEG/PNG without pulling in heavy deps."""
    data = path.read_bytes()
    if data[:2] == b"\xff\xd8":  # JPEG: parse SOF marker
        i = 2
        while i < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            i += 2
            if marker in (0xC0, 0xC1, 0xC2, 0xC3):
                return (
                    int.from_bytes(data[i + 5 : i + 7], "big"),
                    int.from_bytes(data[i + 3 : i + 5], "big"),
                )
            seg_len = int.from_bytes(data[i : i + 2], "big")
            i += seg_len
        raise ValueError(f"could not parse JPEG dimensions for {path}")
    if data[:8] == b"\x89PNG\r\n\x1a\n":  # PNG: IHDR at fixed offset
        return (
            int.from_bytes(data[16:20], "big"),
            int.from_bytes(data[20:24], "big"),
        )
    raise ValueError(f"unsupported image type for {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels-root", required=True, type=Path)
    p.add_argument("--images-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--limit", type=int, default=0, help="0 = process all")
    args = p.parse_args()
    aggregate(args.labels_root, args.images_root, args.out, args.split, args.limit)


if __name__ == "__main__":
    main()
