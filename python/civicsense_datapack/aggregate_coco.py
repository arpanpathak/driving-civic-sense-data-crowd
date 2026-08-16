#!/usr/bin/env python3
"""Aggregate COCO-format annotations into the CivicSense YOLO layout.

Usage
-----
    python -m civicsense_datapack.aggregate_coco \
        --coco-ann annotations/instances_*.json \
        --coco-images /path/to/coco/images \
        --out datasets/training \
        --split train

What it does
------------
1. Scans one or more COCO ``instances_*.json`` annotation files.
2. Maps COCO categories to the CivicSense 7-class vocabulary where they
   overlap (car/motorcycle/bus/truck -> vehicle/truck/bus; traffic light
   and stop sign align by name). Non-matching categories are dropped.
3. Copies the referenced images and writes YOLO-format labels
   (``class x_center y_center width height`` in normalised units).
4. SKIPS any image whose sample contains only non-CivicSense objects, so
   every emitted image has at least one usable label.

Downstream, run the Rust validator to prove the output is ready:
    cargo run --bin civicsense-data -- training datasets/training
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# COCO category name -> CivicSense class id.
# CivicSense only keeps the overlapping road/user classes it was trained on.
_COCO_MAP = {
    "car": 3,  # vehicle
    "motorcycle": 3,  # vehicle
    "truck": 4,  # truck
    "bus": 5,  # bus
    "traffic light": 1,  # traffic_light
    "stop sign": 0,  # stop_sign
}


def _load_categories(ann):
    return {c["id"]: c["name"].lower().strip() for c in ann["categories"]}


def _load_images(ann):
    return {im["id"]: im for im in ann["images"]}


def aggregate(coco_ann_paths, coco_images_root, out_dir, split):
    out_images = out_dir / "images" / split
    out_labels = out_dir / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_no_match = 0
    skipped_missing = 0

    # Merge all annotation files together (treating each as a single source).
    for ann_path in coco_ann_paths:
        ann_path = Path(ann_path)
        with ann_path.open() as fh:
            ann = json.load(fh)
        cat_map = _load_categories(ann)
        img_map = _load_images(ann)
        rec_by_img: dict[int, list] = {}
        for anno in ann["annotations"]:
            img_id = anno["image_id"]
            name = cat_map.get(anno["category_id"], "")
            cid = _COCO_MAP.get(name)
            if cid is None:
                continue  # not a CivicSense class
            rec_by_img.setdefault(img_id, []).append((cid, anno["bbox"]))

        for img_id, records in rec_by_img.items():
            im = img_map.get(img_id)
            if im is None:
                skipped_missing += 1
                continue
            src_img = coco_images_root / im["file_name"]
            if not src_img.exists():
                skipped_missing += 1
                continue

            w, h = int(im["width"]), int(im["height"])
            stem = f"{ann_path.stem}_{img_id:012d}"
            dst_img = out_images / f"{stem}.jpg"
            if dst_img.exists():
                skipped_no_match += 1  # already written under this stem
                continue
            shutil.copy2(src_img, dst_img)

            lines = []
            for cid, bbox in records:
                x, y, bw, bh = bbox
                line = _bbox_to_yolo(w, h, (x, y, x + bw, y + bh), cid)
                if line:
                    lines.append(line)
            if not lines:
                dst_img.unlink()  # nothing usable; drop the copy
                skipped_no_match += 1
                continue
            (out_labels / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            written += 1

    print(
        f"[coco] wrote {written} labelled images to {out_images.parent} "
        f"(skipped {skipped_no_match} without a usable label, "
        f"{skipped_missing} missing source files)"
    )


def _bbox_to_yolo(w, h, xyxy_px, class_id):
    from .schema import bbox_to_yolo

    return bbox_to_yolo(class_id, w, h, xyxy_px)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-ann", nargs="+", required=True)
    parser.add_argument("--coco-images", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    args = parser.parse_args()
    aggregate(args.coco_ann, args.coco_images, args.out, args.split)


if __name__ == "__main__":
    main()
