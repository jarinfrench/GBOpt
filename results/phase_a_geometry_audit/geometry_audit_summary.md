# Zhang Geometry-Audit Summary

Source: `/home/frenjc/projects/GBOpt-worktrees/boundary-spec-api-stage-E/results/phase_a_geometry_audit/generation_results.tsv`

> **Classification policy:** Phase 1 audit statuses are descriptive and warning-only. No threshold in this report rejects generation.

## Classification thresholds

| Diagnostic | Trigger | Scope |
| --- | --- | --- |
| Empty-bin fraction | > 0.25 | Either side of either interface |
| Gap range / bulk nearest-neighbor distance | > 2.0 | Central or periodic interface |
| (p95 - median) gap / bulk nearest-neighbor distance | > 1.0 | Central or periodic interface |
| Cross-interface minimum / bulk nearest-neighbor distance | < 0.45 | Central or periodic interface |
| Periodic duplicate separation | <= 1.0e-6 A | Any fully periodic atom pair |

## Audit status

| Status | Cases |
| --- | --- |
| ok | 36 |
| suspicious | 161 |

## Suspicious frequency by boundary type

| Boundary type | Cases | Suspicious/invalid | Rate |
| --- | --- | --- | --- |
| AT | 49 | 45 | 91.8% |
| MX | 40 | 40 | 100.0% |
| ST | 58 | 56 | 96.6% |
| TW | 50 | 20 | 40.0% |

## Suspicious frequency by axis set

| Axis set | Cases | Suspicious/invalid | Rate |
| --- | --- | --- | --- |
| 100 | 55 | 41 | 74.5% |
| 110 | 84 | 64 | 76.2% |
| 111 | 58 | 56 | 96.6% |

## Suspicious frequency by maximum P/Q determinant

| Bucket | Cases | Suspicious/invalid | Rate |
| --- | --- | --- | --- |
| <=10 | 8 | 5 | 62.5% |
| 11-100 | 106 | 86 | 81.1% |
| 101-1000 | 80 | 67 | 83.8% |
| >1000 | 3 | 3 | 100.0% |

## Suspicious frequency by maximum Miller-row norm

| Bucket | Cases | Suspicious/invalid | Rate |
| --- | --- | --- | --- |
| <=5 | 33 | 26 | 78.8% |
| 5-10 | 86 | 70 | 81.4% |
| 10-25 | 74 | 62 | 83.8% |
| >25 | 4 | 3 | 75.0% |

## Suspicious frequency by atom count

| Bucket | Cases | Suspicious/invalid | Rate |
| --- | --- | --- | --- |
| <=25k | 149 | 121 | 81.2% |
| 25k-100k | 39 | 34 | 87.2% |
| 100k-250k | 4 | 3 | 75.0% |
| >250k | 5 | 3 | 60.0% |

## Classification reasons

| Reason | Cases |
| --- | --- |
| periodic_interface_large_gap_range | 154 |
| central_interface_large_gap_range | 153 |
| central_interface_heavy_gap_tail | 135 |
| periodic_interface_heavy_gap_tail | 133 |
| central_interface_severe_overlap | 25 |
| periodic_interface_severe_overlap | 10 |

### Reasons by boundary type

#### AT

| Reason | Cases |
| --- | --- |
| periodic_interface_large_gap_range | 43 |
| central_interface_large_gap_range | 42 |
| periodic_interface_heavy_gap_tail | 37 |
| central_interface_heavy_gap_tail | 34 |
| central_interface_severe_overlap | 8 |
| periodic_interface_severe_overlap | 1 |

#### MX

| Reason | Cases |
| --- | --- |
| periodic_interface_large_gap_range | 40 |
| central_interface_large_gap_range | 38 |
| central_interface_heavy_gap_tail | 34 |
| periodic_interface_heavy_gap_tail | 32 |
| central_interface_severe_overlap | 1 |

#### ST

| Reason | Cases |
| --- | --- |
| central_interface_large_gap_range | 55 |
| periodic_interface_large_gap_range | 53 |
| central_interface_heavy_gap_tail | 51 |
| periodic_interface_heavy_gap_tail | 45 |
| central_interface_severe_overlap | 16 |
| periodic_interface_severe_overlap | 9 |

#### TW

| Reason | Cases |
| --- | --- |
| periodic_interface_heavy_gap_tail | 19 |
| central_interface_large_gap_range | 18 |
| periodic_interface_large_gap_range | 18 |
| central_interface_heavy_gap_tail | 16 |

## Most common reason combinations

| Reasons | Cases |
| --- | --- |
| central_interface_large_gap_range; central_interface_heavy_gap_tail; periodic_interface_large_gap_range; periodic_interface_heavy_gap_tail | 87 |
| central_interface_large_gap_range; central_interface_heavy_gap_tail; periodic_interface_large_gap_range | 17 |
| central_interface_large_gap_range; central_interface_heavy_gap_tail; periodic_interface_large_gap_range; periodic_interface_heavy_gap_tail; central_interface_severe_overlap | 14 |
| central_interface_large_gap_range; periodic_interface_large_gap_range; periodic_interface_heavy_gap_tail | 14 |
| central_interface_large_gap_range; central_interface_heavy_gap_tail; periodic_interface_large_gap_range; periodic_interface_heavy_gap_tail; central_interface_severe_overlap; periodic_interface_severe_overlap | 8 |
| central_interface_large_gap_range; periodic_interface_large_gap_range | 6 |
| central_interface_heavy_gap_tail; periodic_interface_large_gap_range; periodic_interface_heavy_gap_tail | 3 |
| central_interface_heavy_gap_tail; periodic_interface_heavy_gap_tail | 2 |
| central_interface_large_gap_range | 2 |
| central_interface_large_gap_range; central_interface_heavy_gap_tail; periodic_interface_large_gap_range; periodic_interface_heavy_gap_tail; periodic_interface_severe_overlap | 2 |
| central_interface_large_gap_range; central_interface_heavy_gap_tail | 1 |
| central_interface_large_gap_range; central_interface_heavy_gap_tail; periodic_interface_large_gap_range; central_interface_severe_overlap | 1 |
| central_interface_large_gap_range; periodic_interface_large_gap_range; periodic_interface_heavy_gap_tail; central_interface_severe_overlap | 1 |
| central_interface_severe_overlap | 1 |
| periodic_interface_heavy_gap_tail | 1 |

## Normalized severity

Only `suspicious` and `invalid` cases are included. The bulk reference is the smaller of the left- and right-grain internal minimum distances.

| Metric | Cases | Minimum | Median | P95 | Maximum | Trigger |
| --- | --- | --- | --- | --- | --- | --- |
| Central gap range / bulk | 161 | 1.6330 | 3.0856 | 4.0805 | 4.6887 | > 2.0 |
| Periodic gap range / bulk | 161 | 1.6330 | 3.0987 | 4.0855 | 4.6876 | > 2.0 |
| Central gap tail / bulk | 161 | 0.3381 | 1.3333 | 2.2418 | 2.8180 | > 1.0 |
| Periodic gap tail / bulk | 161 | 0.1127 | 1.2910 | 1.9596 | 2.1227 | > 1.0 |
| Central cross distance / bulk | 161 | 0.0959 | 0.9652 | 1.6330 | 2.0000 | < 0.45 |
| Periodic cross distance / bulk | 161 | 0.1918 | 1.0000 | 2.1193 | 2.7487 | < 0.45 |

## Empty-bin and duplicate diagnostics

| Diagnostic | Campaign value |
| --- | --- |
| Central empty-left fraction | 0 |
| Central empty-right fraction | 0 |
| Periodic empty-left fraction | 0 |
| Periodic empty-right fraction | 0 |
| Cases with periodic duplicates | 0 |
| Maximum duplicate-pair count | 0 |

## Numeric ranges by audit status

### Maximum Miller-row norm

| Audit status | Cases | Minimum | Median | Maximum |
| --- | --- | --- | --- | --- |
| ok | 36 | 2.44949 | 8.68788 | 30.0167 |
| suspicious | 161 | 2.23607 | 8.3666 | 39.5727 |

### Atom count

| Audit status | Cases | Minimum | Median | Maximum |
| --- | --- | --- | --- | --- |
| ok | 36 | 4824 | 15372 | 278928 |
| suspicious | 161 | 4428 | 12384 | 435564 |

### Box x (A)

| Audit status | Cases | Minimum | Median | Maximum |
| --- | --- | --- | --- | --- |
| ok | 36 | 123.41 | 130.896 | 253.416 |
| suspicious | 161 | 121.955 | 153.489 | 431.659 |

### Box y (A)

| Audit status | Cases | Minimum | Median | Maximum |
| --- | --- | --- | --- | --- |
| ok | 36 | 22.4874 | 44.3085 | 669.065 |
| suspicious | 161 | 22.4874 | 40.814 | 1058.06 |

### Box z (A)

| Audit status | Cases | Minimum | Median | Maximum |
| --- | --- | --- | --- | --- |
| ok | 36 | 22.4874 | 37.3717 | 163.711 |
| suspicious | 161 | 22.4874 | 27.27 | 131.01 |
