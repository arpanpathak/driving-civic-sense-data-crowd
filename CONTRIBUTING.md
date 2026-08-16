# 🤝 Contributing to CivicSense Data Pack

Thanks for helping keep CivicSense's data honest! Every label, every
ground-truth record, and every aggregation improvement makes the model
and the kinematic engine more trustworthy on real roads.

## 📏 Ground rules

- 🚫 **This repo ships no imagery.** Commit placeholders, schemas, validators,
  scripts, and *text*; never huge binaries. Heavy data lives in `data/raw/`
  (git-ignored) or upstream archives.
- ⚖️ **Respect data licenses.** Before adding a derived pack, confirm you may
  redistribute it and update [`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md).
- 🧪 **Every non-trivial change needs a test.** Rust logic → `#[test]`;
  Python logic → a `test_*` function; new tooling → a `--help` a reviewer
  can invoke.

## 🎯 What we need

1. 🏷️ **Data labelers.** Add labelled images to `datasets/training/` following
   the YOLO layout (`images/{train,val}` ↔ `labels/{train,val}`). Each label
   file: `class_id x_center y_center width height` (normalised 0 to 1).
2. 🚗 **Field testers.** Append ground-truth records to
   `validation/ground-truth/manifest.json` from real commutes or CARLA/SUMO.
   Keep every record a synchronised snapshot.
3. 🛠️ **Tooling engineers.** Harden the aggregators (COCO/BDD100K), add
   UA-DETRAC or CARLA imports, or extend the Rust validators.

## ⚙️ Development workflow

```bash
git clone https://github.com/arpanpathak/driving-civic-sense-data-crowd
cd driving-civic-sense-data-crowd

# Rust: run all tests + strict lint
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check

# Python: unit tests + lint
PYTHONPATH=python python3 python/tests/test_schema.py
flake8 python/ --count --max-complexity=10 --statistics
black --check python/
```

Everything above must pass before a PR is merged.

## ✏️ Commit conventions

Follow conventional commits (`feat:`, `fix:`, `docs:`, `test:`,
`data:`, `chore:`). Reference the ground-truth schema or class ids when
your change affects them.

## ✅ PR checklist

- [ ] `cargo test` passes (unit + doc-tests + integration).
- [ ] `cargo clippy --all-targets -- -D warnings` passes.
- [ ] `cargo fmt --check` is clean.
- [ ] Python tests pass; new Python logic has tests.
- [ ] No large binaries staged.
- [ ] License/provenance notes updated when adding a dataset source.

_By contributing you agree to release your contribution under the MIT
license (code) and to honour the individual licenses of any dataset data
you add._
