//! Integration test: the **seed ground-truth manifest** must validate
//! cleanly against the Rust schema, proving the Python-emitted records and
//! the Rust validator agree. This guards the ship-blocking invariant that
//! new field data added by contributors passes the same rules the
//! decision-engine evaluation consumes.

use std::path::PathBuf;

use civicsense_data_pack::ground_truth::GroundTruthRecord;

const MANIFEST: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/validation/ground-truth/manifest.json"
);

/// A manifest may be a plain JSON array or a `{"records": [...]}` wrapper.
#[derive(serde::Deserialize)]
struct ManifestWrapper {
    records: Vec<GroundTruthRecord>,
}

#[test]
fn seed_manifest_validates() {
    let content = std::fs::read_to_string(MANIFEST).expect("seed manifest file must exist");
    let records: Vec<GroundTruthRecord> = match serde_json::from_str::<ManifestWrapper>(&content) {
        Ok(w) => w.records,
        Err(_) => serde_json::from_str(&content).expect("manifest must be array or wrapper"),
    };
    assert!(!records.is_empty(), "seed manifest must contain records");

    let report = civicsense_data_pack::ground_truth::validate_records(&records);
    assert_eq!(
        report.total,
        records.len(),
        "report must count every record"
    );
    assert_eq!(
        report.failed, 0,
        "all seed ground-truth records must validate; failed: {:?}",
        report.failed_ids
    );
}

#[test]
fn seed_manifest_records_compile_to_classes() {
    // Every record's detection class must be a valid CivicSense class.
    let content = std::fs::read_to_string(MANIFEST).unwrap();
    let records: Vec<GroundTruthRecord> = match serde_json::from_str::<ManifestWrapper>(&content) {
        Ok(w) => w.records,
        Err(_) => serde_json::from_str(&content).unwrap(),
    };
    for r in &records {
        assert!(!r.id.is_empty());
        assert!(!r.scenario.is_empty());
        for d in &r.detections {
            assert!(
                d.class_id < civicsense_data_pack::classes::NUM_CLASSES,
                "record {} has out-of-range class {}",
                r.id,
                d.class_id
            );
        }
    }
}

/// Confirms the manifest path resolves to a real file relative to this repo.
#[test]
fn manifest_path_is_absolute_and_readable() {
    let p = PathBuf::from(MANIFEST);
    assert!(p.is_absolute());
    assert!(p.is_file(), "{MANIFEST} must exist on disk");
}
