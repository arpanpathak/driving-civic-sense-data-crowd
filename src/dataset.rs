//! Validation of the **training dataset directory layout** CivicSense's
//! `civicsense train prepare` expects.
//!
//! The canonical layout (mirroring the main repo's
//! [`Dataset`](https://docs.rs) contract) is:
//!
//! ```text
//! datasets/training/
//!   images/
//!     train/  *.jpg | *.png | *.jpeg
//!     val/    *.jpg | *.png | *.jpeg
//!   labels/
//!     train/  *.txt   (one per image, same stem)
//!     val/    *.txt
//! ```
//!
//! Invariants enforced:
//!
//! - Every image in `images/train` has exactly one sibling label file
//!   `labels/train/<stem>.txt`, and vice-versa (no orphan images, no
//!   orphan labels). The same holds for `val`.
//! - Every label file parses and validates against [`crate::yolo`].
//! - All image extensions belong to a whitelist; all label files must
//!   use the `.txt` extension.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::yolo::{parse_label_file, LabelError};

/// Image file extensions accepted by the training pipeline.
pub const ALLOWED_IMAGE_EXTENSIONS: [&str; 3] = ["jpg", "jpeg", "png"];

/// A structural failure in the training split directory layout.
#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    /// A required directory was missing from the layout.
    #[error("dataset split `{role}` is missing directory `{name}` under `{root}`")]
    MissingDirectory {
        root: PathBuf,
        role: SplitRole,
        name: &'static str,
    },

    /// An image in `images/` has no sibling label file.
    #[error("image `{0}` has no matching label under `labels/{1}`")]
    OrphanImage(PathBuf, SplitRole),

    /// A label file in `labels/` has no sibling image file.
    #[error("label `{0}` has no matching image under `images/{1}`")]
    OrphanLabel(PathBuf, SplitRole),

    /// A label file failed to parse.
    #[error("label `{path}` is invalid: {source}")]
    InvalidLabel { path: PathBuf, source: LabelError },

    /// The common `datasets/training/` root did not exist.
    #[error("training root directory `{0}` does not exist")]
    RootMissing(PathBuf),
}

/// Which split (train or val) a file belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplitRole {
    /// The training split.
    #[default]
    Train,
    /// The validation split.
    Val,
}

impl SplitRole {
    /// Human-readable label for logging and error messages.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            SplitRole::Train => "train",
            SplitRole::Val => "val",
        }
    }
}

impl std::fmt::Display for SplitRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A validated training-dataset split: the number of image/label pairs
/// and the class distribution observed across its labels.
#[derive(Debug, Default)]
pub struct SplitSummary {
    /// The split role this summary describes.
    pub role: SplitRole,
    /// Number of (image, label) pairs that passed validation.
    pub pairs: usize,
    /// Per-class object counts across all labels, indexed by class id.
    pub class_histogram: [u64; crate::classes::NUM_CLASSES],
}

/// Validates the full training layout rooted at `root`
/// (`datasets/training`).
///
/// # Errors
///
/// Returns [`DatasetError::RootMissing`] if `root` is not a directory,
/// or the first [`DatasetError`] describing a broken invariant.
pub fn validate_training_layout(root: &Path) -> Result<(SplitSummary, SplitSummary), DatasetError> {
    if !root.is_dir() {
        return Err(DatasetError::RootMissing(root.to_path_buf()));
    }
    let train = validate_split(root, SplitRole::Train)?;
    let val = validate_split(root, SplitRole::Val)?;
    Ok((train, val))
}

fn validate_split(root: &Path, role: SplitRole) -> Result<SplitSummary, DatasetError> {
    let split = role.as_str();
    let images_dir = root.join("images").join(split);
    let labels_dir = root.join("labels").join(split);

    for (name, dir) in [("images", &images_dir), ("labels", &labels_dir)] {
        if !dir.is_dir() {
            return Err(DatasetError::MissingDirectory {
                root: root.to_path_buf(),
                role,
                name,
            });
        }
    }

    let mut image_stems: BTreeSet<PathBuf> = BTreeSet::new();
    for entry in std::fs::read_dir(&images_dir)
        .map_err(|e| DatasetError::RootMissing(e2path(&images_dir, e)))?
    {
        let path = entry
            .map_err(|e| DatasetError::RootMissing(e2path(&images_dir, e)))?
            .path();
        if is_image(&path) {
            image_stems.insert(stem_of(&path));
        }
    }

    let mut summary = SplitSummary {
        role,
        ..Default::default()
    };

    let label_entries: Vec<PathBuf> = std::fs::read_dir(&labels_dir)
        .map_err(|e| DatasetError::RootMissing(e2path(&labels_dir, e)))?
        .filter_map(|r| r.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();

    for label_path in &label_entries {
        let stem = stem_of(label_path);
        // Find which image extension exists on disk for this label's stem.
        let Some(ext) = ALLOWED_IMAGE_EXTENSIONS
            .iter()
            .find(|ext| images_dir.join(&stem).with_extension(ext).is_file())
        else {
            return Err(DatasetError::OrphanLabel(label_path.clone(), role));
        };
        let img_path = images_dir.join(&stem).with_extension(ext);

        if !image_stems.contains(&stem_of(&img_path)) {
            return Err(DatasetError::OrphanLabel(label_path.clone(), role));
        }

        let labels = parse_label_file(label_path).map_err(|source| DatasetError::InvalidLabel {
            path: label_path.clone(),
            source,
        })?;
        summary.pairs += 1;
        for l in &labels {
            summary.class_histogram[l.class_id] += 1;
        }
    }

    // Any image without a matching label is orphaned.
    for stem in &image_stems {
        let label_candidate = labels_dir.join(stem).with_extension("txt");
        if !label_candidate.is_file() {
            return Err(DatasetError::OrphanImage(
                images_dir.join(stem).with_extension("jpg"),
                role,
            ));
        }
    }

    Ok(summary)
}

fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| ALLOWED_IMAGE_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
}

/// Returns the bare file **stem** (file name without extension) of a
/// path, dropping any directory components so it can be joined onto a
/// different root directory.
fn stem_of(path: &Path) -> PathBuf {
    path.file_stem().map(PathBuf::from).unwrap_or_default()
}

fn e2path(p: &Path, _e: std::io::Error) -> PathBuf {
    p.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a minimal, valid training layout on disk under a fresh temp
    /// dir and returns its root.
    fn valid_root() -> PathBuf {
        let thread = std::thread::current().id();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("civicsense_ds_{thread:?}_{nonce}"));
        for split in ["train", "val"] {
            std::fs::create_dir_all(root.join("images").join(split)).unwrap();
            std::fs::create_dir_all(root.join("labels").join(split)).unwrap();
            // one image + one matching label per split
            std::fs::write(root.join("images").join(split).join("frame_0001.jpg"), b"").unwrap();
            std::fs::write(
                root.join("labels").join(split).join("frame_0001.txt"),
                "3 0.5 0.5 0.2 0.4\n",
            )
            .unwrap();
        }
        root
    }

    #[test]
    fn valid_layout_passes() {
        let root = valid_root();
        let (train, val) = validate_training_layout(&root).unwrap();
        assert_eq!(train.pairs, 1);
        assert_eq!(val.pairs, 1);
        assert_eq!(train.class_histogram[3], 1);
    }

    #[test]
    fn missing_root_errors() {
        let missing = std::env::temp_dir().join("definitely_does_not_exist");
        assert!(matches!(
            validate_training_layout(&missing),
            Err(DatasetError::RootMissing(_))
        ));
    }

    #[test]
    fn orphan_label_is_rejected() {
        let root = valid_root();
        // Add a label with no matching image.
        std::fs::write(
            root.join("labels").join("train").join("orphan.txt"),
            "3 0.5 0.5 0.2 0.4\n",
        )
        .unwrap();
        assert!(matches!(
            validate_training_layout(&root),
            Err(DatasetError::OrphanLabel(_, SplitRole::Train))
        ));
    }

    #[test]
    fn orphan_image_is_rejected() {
        let root = valid_root();
        // Add an image with no matching label.
        std::fs::write(root.join("images").join("train").join("ghost.png"), b"").unwrap();
        assert!(matches!(
            validate_training_layout(&root),
            Err(DatasetError::OrphanImage(_, SplitRole::Train))
        ));
    }

    #[test]
    fn invalid_label_content_is_rejected() {
        let root = valid_root();
        std::fs::write(
            root.join("labels").join("val").join("frame_0001.txt"),
            "7 0.5 0.5\n",
        )
        .unwrap();
        assert!(matches!(
            validate_training_layout(&root),
            Err(DatasetError::InvalidLabel { .. })
        ));
    }

    #[test]
    fn split_role_display() {
        assert_eq!(SplitRole::Train.to_string(), "train");
        assert_eq!(SplitRole::Val.to_string(), "val");
    }
}
