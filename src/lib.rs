//! # 🅿️ CivicSense Data Pack
//!
//! A companion **datasets** and **ground-truth validation** repository for
//! the [driving-civicsense-vision-model](https://github.com/arpanpathak/driving-civicsense-vision-model)
//! project.
//!
//! CivicSense trains a YOLOv8n / YOLOv11n detector on 7 classes to feed a
//! zero-training kinematic decision engine. This crate provides the
//! **schema and validators** that keep that data honest:
//!
//! - [`classes`]: the canonical 7-class label vocabulary.
//! - [`yolo`]: a parser/validator for YOLO-format label files.
//! - [`dataset`]: validation of the `images/` ↔ `labels/` training layout.
//! - [`ground_truth`]: the field-validation record schema for the kinematic
//!   engine, plus a batch validator.
//!
//! The heavy imagery is **not** committed to git (this repo is MIT and lean).
//! Instead, see the `scripts/` downloaders and the `python/` aggregation
//! tooling to pull and label public datasets (UA-DETRAC, COCO, BDD100K,
//! CARLA). The validators here are what you run after aggregating, to prove
//! the result is ready for `civicsense train prepare`.

pub mod classes;
pub mod dataset;
pub mod ground_truth;
pub mod yolo;

/// Re-export of the class vocabulary at the crate root for ergonomics.
pub use classes::{CLASS_NAMES, NUM_CLASSES};
