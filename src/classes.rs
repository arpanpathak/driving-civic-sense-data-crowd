//! The 7-class label vocabulary that CivicSense's perception pipeline
//! was trained on.
//!
//! These class ids **must** match, in order, two places in the main
//! `driving-civicsense-vision-model` repo:
//!
//! - `src/config.rs` → `ModelConfig::classes`
//! - `src/train.rs` → the `class_names` vector emitted in `dataset.yaml`
//!
//! Keeping a single source of truth here lets the data-pack validator
//! reject a YOLO label whose `class_id` falls outside `[0, 7)` or whose
//! semantics disagree with the training-consumer expectations.
//!
//! ```text
//! 0: stop_sign          1: traffic_light
//! 2: crosswalk          3: vehicle
//! 4: truck              5: bus
//! 6: intersection_zone
//! ```

/// The number of classes the CivicSense model predicts.
pub const NUM_CLASSES: usize = 7;

/// Ordered class names, index-equivalent to `class_id` in YOLO labels.
pub const CLASS_NAMES: [&str; NUM_CLASSES] = [
    "stop_sign",
    "traffic_light",
    "crosswalk",
    "vehicle",
    "truck",
    "bus",
    "intersection_zone",
];

/// The class id of `stop_sign` (COCO-style reserved, but re-mapped by
/// the CivicSense training pipeline to position 0).
pub const STOP_SIGN: usize = 0;
/// The class id of `traffic_light`.
pub const TRAFFIC_LIGHT: usize = 1;
/// The class id of `crosswalk`.
pub const CROSSWALK: usize = 2;
/// The class id of `vehicle` (generic passenger car / motorcycle).
pub const VEHICLE: usize = 3;
/// The class id of `truck`.
pub const TRUCK: usize = 4;
/// The class id of `bus`.
pub const BUS: usize = 5;
/// The class id of `intersection_zone` (the polygon of the junction
/// entrance grid used by the decision engine's occupancy check).
pub const INTERSECTION_ZONE: usize = 6;

/// Returns the class name for a numeric `class_id`, or `None` when the
/// id is out of range for the CivicSense vocabulary.
///
/// # Example
/// ```
/// use civicsense_data_pack::classes;
/// assert_eq!(classes::name(3), Some("vehicle"));
/// assert_eq!(classes::name(99), None);
/// ```
#[must_use]
pub fn name(class_id: usize) -> Option<&'static str> {
    CLASS_NAMES.get(class_id).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocabulary_has_seven_entries() {
        assert_eq!(CLASS_NAMES.len(), NUM_CLASSES);
        assert_eq!(NUM_CLASSES, 7);
    }

    #[test]
    fn names_are_unique() {
        let mut sorted = CLASS_NAMES.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), NUM_CLASSES, "class names must be unique");
    }

    #[test]
    fn every_index_resolves() {
        for (i, expected) in CLASS_NAMES.iter().enumerate() {
            assert_eq!(name(i), Some(*expected), "index {i} mismatch");
        }
    }

    #[test]
    fn out_of_range_returns_none() {
        assert_eq!(name(NUM_CLASSES), None);
        assert_eq!(name(usize::MAX), None);
    }
}
