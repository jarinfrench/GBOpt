# Zhang Geometry-Regression Cases

## Purpose

This file records the representative Zhang UO2 grain-boundary cases selected during
Phase A / Phase 1 of the geometry-audit remediation work. The cohort is intended to:

- preserve one known defective structure and one clean control as committed regression
  fixtures;
- document visual agreement between the campaign-wide audit and representative
  structures;
- provide a deterministic regeneration cohort for later construction changes; and
- retain AT, MX, ST, TW, compact, high-complexity, and large-system coverage for later
  phases.

The Phase 1 classifier is descriptive and warning-only. A status of `suspicious` does
not reject generation. The current default thresholds are:

| Diagnostic | Warning threshold |
| --- | --- |
| Empty bins on either side of an interface | fraction greater than `0.25` |
| Local gap range | greater than `2.0` times the bulk nearest-neighbor distance |
| Gap upper tail (`p95 - median`) | greater than `1.0` times the bulk nearest-neighbor distance |
| Cross-interface minimum distance | less than `0.45` times the bulk nearest-neighbor distance |
| Periodic duplicate pair | separation at or below `1.0e-6` A |

## Primary committed regression fixtures

| Case | Role | Expected audit result | Rationale |
| --- | --- | --- | --- |
| `zhang_001_ST_100` | Negative regression | `suspicious` | Original campaign case that exhibits the large staircase/channel defect and severe central and periodic overlap diagnostics. |
| `zhang_041_TW_100` | Positive regression | `ok` | Small real Zhang twist structure with a uniform projected lattice and no visible interface channel; suitable as a fast clean control. |

For each committed fixture, retain both the serialized LAMMPS data file and its matching
metadata file so grain membership, box bounds, bin counts, and recorded audit values can
be reconstructed exactly.

## Stratified visual-inspection and determinism cohort

The images were reviewed in the same order as the rows below. The visualizations are
2D projections, so they are suitable for identifying large channels and wedges but
cannot by themselves confirm a 3D close-contact or periodic-interface diagnostic.

| Case | Type | Axis | Audit | Atoms | Max P/Q determinant | Max Miller-row norm | Max gap range / bulk | Selection role | Visual assessment |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `zhang_077_ST_110` | ST | 110 | `suspicious` | 10,728 | 82 | 9.05539 | 4.6887 | Most severe suspicious ST | A pronounced central channel is visible, with substantial variation in separation along the interface. The image strongly corroborates the central and periodic large-range/heavy-tail reasons. |
| `zhang_022_AT_100` | AT | 100 | `suspicious` | 19,500 | 85 | 9.21954 | 4.2583 | Most severe suspicious AT | A broad central opening and strongly different projected row structures are visible. The interface separation is plainly nonuniform and supports the four gap-range/tail warnings. |
| `zhang_124_MX_110` | MX | 110 | `suspicious` | 6,924 | 306 | 17.4929 | 4.1586 | Most severe suspicious MX | The two terminations form a conspicuous wedge/channel rather than a flat interface. The visual defect agrees with the central and periodic large-range/heavy-tail classification. |
| `zhang_183_TW_111` | TW | 111 | `suspicious` | 154,128 | 1,014 | 31.8434 | 2.6667 | Most severe suspicious TW | A wide central channel is clearly present despite the boundary being a twist case. This confirms that the classifier is responding to geometry rather than simply accepting all TW structures. |
| `zhang_066_ST_110` | ST | 110 | `ok` | 5,832 | 22 | 4.69042 | 1.3926 | Smallest ok ST | The projected interface is comparatively narrow and regular, without the large staircase channel seen in the suspicious ST examples. This is a useful rare-ST control. |
| `zhang_094_AT_110` | AT | 110 | `ok` | 4,824 | 18 | 4.24264 | 1.9245 | Smallest ok AT | The central opening appears substantially more uniform than in the suspicious AT example. Its gap-range ratio is close to, but remains below, the `2.0` warning threshold, making it a useful near-threshold control. |
| `zhang_041_TW_100` | TW | 100 | `ok` | 4,896 | 17 | 4.12311 | 0.0000 | Smallest ok TW; primary positive fixture | The projection is filled by regular columns across the cell and shows no visible central channel. It is the clearest clean visual control in the cohort. |
| `zhang_127_MX_110` | MX | 110 | `suspicious` | 36,588 | 1,566 | 39.5727 | 3.7933 | Largest determinant and largest Miller-row norm | A large wedge/channel separates two very different projected lattice terminations. The image and four gap warnings agree; the case also preserves the campaign's highest crystallographic-complexity coverage. |
| `zhang_158_AT_111` | AT | 111 | `suspicious` | 435,564 | 222 | 14.8997 | 1.8899 | Largest atom count | No large void is obvious in this projection. The case is flagged only for `central_interface_severe_overlap`, a 3D minimum-distance condition that cannot be validated reliably from this 2D view. Retain it as the large-system and overlap-only diagnostic case. |
| `zhang_073_ST_110` | ST | 110 | `suspicious` | 4,428 | 34 | 5.83095 | 3.5645 | Smallest atom count | A clear central open channel is visible in this compact system. The image supports the large-range/heavy-tail classification; the additional central-overlap warning must be assessed from the 3D metric rather than the projection alone. |

## Visual closeout conclusion

The visual sample supports the Phase 1 audit:

- all selected cases carrying the four central/periodic gap-range and gap-tail warnings
  show conspicuous channels, wedges, or strongly nonuniform interface separation;
- the three selected `ok` controls are visibly more regular and lack the large
  staircase-like voids seen in the suspicious cohort;
- the suspicious TW example demonstrates that status is geometry-dependent rather than
  hard-coded by boundary type; and
- the overlap-only largest system illustrates the limitation of 2D visual inspection
  and the need to retain the numerical 3D nearest-neighbor diagnostic.

No threshold should be promoted to a generation failure criterion during Phase 1.

## Preservation policy

Commit the two primary fixture pairs under `tests/data/`:

```text
tests/data/zhang_001_ST_100.data
tests/data/zhang_001_ST_100.metadata.json
tests/data/zhang_041_TW_100.data
tests/data/zhang_041_TW_100.metadata.json
```

Retain the complete ten-case cohort in the archived Phase 1 campaign results and use it
for repeated-generation determinism checks and later before/after comparisons. The
canonical generated copies are expected under:

```text
results/phase_a_geometry_audit/<case_id>/
```
