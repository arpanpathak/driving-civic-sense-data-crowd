# Dataset Licenses & Provenance

The **code** in this repository is MIT-licensed (see `LICENSE`). The
**datasets** it references, downloads, and aggregates are *not* owned by
this repository. Each retains its own license and terms of use. Before you
redistribute, publish, or ship a model trained on any of the following,
read and comply with *that* dataset's terms.

We aggregate only the **overlapping CivicSense classes** from each source
so a derived, re-annotated pack is as close to clean-room as practical —
but that does **not** erase upstream attribution obligations.

---

## Upstream sources referenced by the tooling

| Source | Kind | License | Notes |
|--------|------|---------|-------|
| [UA-DETRAC](https://detrac-db.rit.albany.edu/) | Detection/tracking (vehicles) | Research-only, request required | Great for dense urban vehicle/occlusion coverage; must register. |
| [COCO 2017](https://cocodataset.org/) | Detection (COCO classes) | CC BY 4.0 (images), COCO annotations | Maps `car/motorcycle/bus/truck/stop sign/traffic light` → CivicSense classes for `vehicle/truck/bus/stop_sign/traffic_light`. |
| [BDD100K](https://www.lds.ac.cn/dataset/bdd100k) (bdd100k.com) | Driving video + det/lane/drivable labels | [CC BY-NC-SA 4.0](https://bdd-data.berkeley.edu/) | 100k diverse images; excellent for generalisation. Detection labels are per-image JSON. |
| [CARLA](https://carla.org/) | Simulator | CARLA Community License / MIT (as applicable) | Ground-truth telemetry + semantics for the kinematic engine validation, no privacy concerns. |
| [SUMO](https://eclipse.dev/sumo/) | Traffic simulation | EPL-2.0 | Signal-phasing / V2I-style ground truth for intersection scenarios. |

> **Provenance discipline.** When you create a *derived* dataset pack from
> any source above, record it in this doc or a sibling `MANIFEST` in
> `datasets/`. Do not silently mix sources of incompatible licenses.

---

## How we stay MIT-compatible at the *repo* level

- This repo **ships no imagery** — only placeholders, schemas, validators,
  and download/aggregation scripts.
- Final datasets you produce locally are your responsibility to license
  correctly (frequently **not** redistributable non-commercially, e.g.
  BDD100K is CC BY-NC-SA).
- The Rust `civicsense-data` validator and the Python aggregators are
  minimal and dependency-light, so reusing *them* is unencumbered.

---

## Recommended attribution snippet

When citing a model trained on aggregated data, include the upstream
sources, e.g.:

> Training data aggregated from COCO 2017 (CC BY 4.0) and BDD100K
> (CC BY-NC-SA 4.0, for non-commercial research). See the dataset pack's
> `docs/DATA_LICENSES.md` for full license details.
