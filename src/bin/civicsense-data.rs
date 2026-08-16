//! `civicsense-data` — CLI for validating the dataset layouts that
//! CivicSense consumes.
//!
//! Subcommands:
//!
//! - `civicsense-data training <root>` — validate the `images/`↔`labels/`
//!   training split under a root directory (defaults to `datasets/training`).
//! - `civicsense-data labels <file>` — parse and validate a single YOLO
//!   label file.
//! - `civicsense-data ground-truth <manifest.json>` — validate a JSON
//!   collection of ground-truth records.
//! - `civicsense-data classes` — print the canonical 7-class vocabulary.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use civicsense_data_pack::classes::{CLASS_NAMES, NUM_CLASSES};
use civicsense_data_pack::dataset::validate_training_layout;
use civicsense_data_pack::ground_truth::{self, GroundTruthRecord};
use civicsense_data_pack::yolo::parse_label_file;

#[derive(Parser)]
#[command(
    name = "civicsense-data",
    version,
    about = "Validate CivicSense dataset & ground-truth layouts"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Validate a YOLO training directory layout (images ↔ labels split).
    Training {
        /// Path to the dataset root (defaults to `datasets/training`).
        #[arg(value_name = "ROOT", default_value = "datasets/training")]
        root: PathBuf,
    },
    /// Parse and validate a single YOLO label `.txt` file.
    Labels {
        /// Path to a `.txt` label file.
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },
    /// Validate a JSON array of ground-truth records.
    GroundTruth {
        /// Path to a JSON file holding a list of ground-truth records.
        #[arg(value_name = "MANIFEST")]
        manifest: PathBuf,
    },
    /// Print the 7-class CivicSense vocabulary.
    Classes,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Training { root } => match validate_training_layout(&root) {
            Ok((train, val)) => {
                println!(
                    "✅ training layout OK\n   train: {} pairs\n   val:   {} pairs",
                    train.pairs, val.pairs
                );
                print_histogram("train", &train.class_histogram);
                print_histogram("val", &val.class_histogram);
            }
            Err(e) => {
                eprintln!("❌ {e}");
                std::process::exit(1);
            }
        },
        Commands::Labels { file } => match parse_label_file(&file) {
            Ok(labels) => {
                println!("✅ parsed {} label(s):", labels.len());
                for l in &labels {
                    println!("   {l}");
                }
            }
            Err(e) => {
                eprintln!("❌ {file:?}: {e}");
                std::process::exit(1);
            }
        },
        Commands::GroundTruth { manifest } => {
            let content = match std::fs::read_to_string(&manifest) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("❌ cannot read manifest {manifest:?}: {e}");
                    std::process::exit(1);
                }
            };
            let records = parse_ground_truth_manifest(&content);
            let report = ground_truth::validate_records(&records);
            println!(
                "✅ checked {} record(s), {} invalid",
                report.total, report.failed
            );
            if report.failed > 0 {
                eprintln!("   invalid ids: {:?}", report.failed_ids);
                std::process::exit(1);
            }
        }
        Commands::Classes => {
            println!("CivicSense {NUM_CLASSES}-class vocabulary:");
            for (i, name) in CLASS_NAMES.iter().enumerate() {
                println!("  {i}: {name}");
            }
        }
    }
}

fn print_histogram(role: &str, hist: &[u64; NUM_CLASSES]) {
    println!("   {role} class histogram:");
    for (i, count) in hist.iter().enumerate() {
        if *count > 0 {
            println!("      {} ({}): {}", i, CLASS_NAMES[i], count);
        }
    }
}

/// Wrapper the human-readable manifest format uses: a top-level object
/// with a `records` array (plus optional `_comment` / `_schema` keys).
#[derive(serde::Deserialize)]
struct ManifestWrapper {
    #[serde(rename = "records")]
    records: Vec<GroundTruthRecord>,
}

/// Parses a ground-truth manifest that is **either** a plain JSON array of
/// records **or** a `{"records": [...]}` object (the human-friendly form).
fn parse_ground_truth_manifest(content: &str) -> Vec<GroundTruthRecord> {
    // Try the human-friendly wrapper first.
    match serde_json::from_str::<ManifestWrapper>(content) {
        Ok(wrapper) => wrapper.records,
        Err(_) => match serde_json::from_str::<Vec<GroundTruthRecord>>(content) {
            Ok(flat) => flat,
            Err(e) => {
                eprintln!("❌ invalid manifest JSON: {e}");
                std::process::exit(1);
            }
        },
    }
}
