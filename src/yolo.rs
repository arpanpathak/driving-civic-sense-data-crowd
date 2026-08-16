//! Parser and validator for **YOLO-format label files** as consumed by
//! the CivicSense training pipeline.
//!
//! CivicSense's training lifecycle (`civicsense train prepare`) expects
//! each image to have a sibling `.txt` with one line per object in
//! **normalised YOLO** form:
//!
//! ```text
//! class_id x_center y_center width height
//! ```
//!
//! - All geometry values are **normalised to `[0, 1]`** relative to the
//!   image width/height.
//! - `x_center` / `y_center` are the box centre; `width` / `height` the
//!   box extent.
//! - `class_id` must be an integer in `[0, NUM_CLASSES)` (see
//!   [`crate::classes`]).
//!
//! This module parses a label file into typed [`YoloLabel`] values and
//! validates each line against the geometric and class invariants that
//! a detector (`YOLOv8n` / `YOLOv11n`) requires to train without NaN or
//! out-of-bounds targets.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::classes::NUM_CLASSES;

/// An error produced while parsing or validating a YOLO label record.
#[derive(Debug, thiserror::Error)]
pub enum LabelError {
    /// A line had fewer/more tokens than the YOLO format requires.
    #[error("expected 5 space-separated fields, got {got} (line {line})")]
    FieldCount { got: usize, line: usize },

    /// A numeric field could not be parsed as a finite `f32`.
    #[error("invalid number `{token}` at field {index} (line {line})")]
    BadNumber {
        token: String,
        index: usize,
        line: usize,
    },

    /// The class id is not an integer, or is outside `[0, NUM_CLASSES)`.
    #[error("class id `{id}` out of range 0..{max} (line {line})")]
    BadClass { id: f32, max: usize, line: usize },

    /// A normalised coordinate or extent escaped the valid `[0, 1]` box.
    #[error("value must lie in [0, 1], got `{value}` at field {index} (line {line})")]
    OutOfRange {
        value: f32,
        index: usize,
        line: usize,
    },

    /// The label file could not be read from disk.
    #[error("could not read label file `{path}`: {source}")]
    ReadError {
        path: String,
        source: std::io::Error,
    },
}

impl From<std::io::Error> for LabelError {
    fn from(e: std::io::Error) -> Self {
        LabelError::ReadError {
            path: String::new(),
            source: e,
        }
    }
}

/// A single, validated YOLO detection record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YoloLabel {
    /// The object class id in `[0, NUM_CLASSES)`.
    pub class_id: usize,
    /// Normalised x-coordinate of the box centre.
    pub x_center: f32,
    /// Normalised y-coordinate of the box centre.
    pub y_center: f32,
    /// Normalised box width.
    pub width: f32,
    /// Normalised box height.
    pub height: f32,
}

impl YoloLabel {
    /// Parses and validates a single line of a YOLO label file.
    ///
    /// # Errors
    ///
    /// Returns a [`LabelError`] describing the first violating field.
    pub fn parse(line: &str, line_number: usize) -> Result<Self, LabelError> {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 5 {
            return Err(LabelError::FieldCount {
                got: fields.len(),
                line: line_number,
            });
        }

        let class_id = parse_f32(fields[0], 0, line_number)?;
        let x_center = parse_f32(fields[1], 1, line_number)?;
        let y_center = parse_f32(fields[2], 2, line_number)?;
        let width = parse_f32(fields[3], 3, line_number)?;
        let height = parse_f32(fields[4], 4, line_number)?;

        if class_id < 0.0 || class_id.fract() != 0.0 || class_id as usize >= NUM_CLASSES {
            return Err(LabelError::BadClass {
                id: class_id,
                max: NUM_CLASSES,
                line: line_number,
            });
        }
        check_unit(x_center, 1, line_number)?;
        check_unit(y_center, 2, line_number)?;
        check_unit(width, 3, line_number)?;
        check_unit(height, 4, line_number)?;

        Ok(Self {
            class_id: class_id as usize,
            x_center,
            y_center,
            width,
            height,
        })
    }
}

impl fmt::Display for YoloLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {:.6} {:.6} {:.6} {:.6}",
            self.class_id, self.x_center, self.y_center, self.width, self.height
        )
    }
}

/// Parses every line of a YOLO label file at `path`, returning the
/// validated records or a [`LabelError`] at the first bad line.
///
/// # Errors
///
/// - [`std::io::Error`] wrappers if the file cannot be read.
/// - [`LabelError`] if any line is malformed.
pub fn parse_label_file(path: &Path) -> Result<Vec<YoloLabel>, LabelError> {
    let content = std::fs::read_to_string(path).map_err(|e| LabelError::ReadError {
        path: path.display().to_string(),
        source: e,
    })?;
    content
        .lines()
        .enumerate()
        .filter_map(|(idx, line)| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                None // skip blank lines
            } else {
                Some(YoloLabel::parse(trimmed, idx + 1))
            }
        })
        .collect::<Result<Vec<_>, _>>()
}

fn parse_f32(token: &str, index: usize, line: usize) -> Result<f32, LabelError> {
    token.parse::<f32>().map_err(|_| LabelError::BadNumber {
        token: token.to_string(),
        index,
        line,
    })
}

fn check_unit(value: f32, index: usize, line: usize) -> Result<(), LabelError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(LabelError::OutOfRange { value, index, line })
    }
}

/// Convenience wrapper that carries a label file's path alongside its
/// parsed records, so callers can report which file failed.
#[derive(Debug)]
pub struct ParsedLabelFile {
    /// The path the labels were read from.
    pub path: PathBuf,
    /// The validated records.
    pub labels: Vec<YoloLabel>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[track_caller]
    fn write_tmp(content: &str) -> PathBuf {
        let thread = std::thread::current().id();
        let nonce: u64 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let path = std::env::temp_dir().join(format!("civicsense_test_{thread:?}_{nonce}.txt"));
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn parses_valid_line() {
        let label = YoloLabel::parse("3 0.5 0.5 0.2 0.4", 1).unwrap();
        assert_eq!(label.class_id, 3);
        assert_eq!(label.x_center, 0.5);
        assert_eq!(label.height, 0.4);
    }

    #[test]
    fn rejects_wrong_field_count() {
        assert!(matches!(
            YoloLabel::parse("3 0.5 0.5 0.2", 1),
            Err(LabelError::FieldCount { .. })
        ));
        assert!(matches!(
            YoloLabel::parse("3 0.5 0.5 0.2 0.4 9", 1),
            Err(LabelError::FieldCount { .. })
        ));
    }

    #[test]
    fn rejects_out_of_range_class() {
        assert!(matches!(
            YoloLabel::parse("7 0.5 0.5 0.2 0.4", 1),
            Err(LabelError::BadClass { .. })
        ));
        assert!(matches!(
            YoloLabel::parse("2.5 0.5 0.5 0.2 0.4", 1),
            Err(LabelError::BadClass { .. })
        ));
    }

    #[test]
    fn rejects_geometry_outside_unit_box() {
        assert!(matches!(
            YoloLabel::parse("3 1.2 0.5 0.2 0.4", 1),
            Err(LabelError::OutOfRange { .. })
        ));
        assert!(matches!(
            YoloLabel::parse("3 0.5 -0.1 0.2 0.4", 1),
            Err(LabelError::OutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_bad_number() {
        assert!(matches!(
            YoloLabel::parse("3 abc 0.5 0.2 0.4", 1),
            Err(LabelError::BadNumber { .. })
        ));
    }

    #[test]
    fn parses_file_skipping_blank_lines() {
        let path = write_tmp("0 0.1 0.1 0.2 0.2\n\n3 0.5 0.5 0.2 0.4\n");
        let records = parse_label_file(&path).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].class_id, 0);
        assert_eq!(records[1].class_id, 3);
    }

    #[test]
    fn parse_file_reports_error() {
        let path = write_tmp("3 0.5 0.5 0.2 0.4\n7 0.5 0.5 0.2 0.4\n");
        assert!(matches!(
            parse_label_file(&path),
            Err(LabelError::BadClass { .. })
        ));
    }
}
