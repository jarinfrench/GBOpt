# Phase A: Geometry Auditing

This bundle implements the first non-construction-changing remediation tranche:
campaign-wide geometry auditing and regression coverage.

## Files

- `GBOpt/geometry_audit.py`
  - deterministic reduced-coordinate interface binning;
  - central and periodic local-gap statistics;
  - left/right internal nearest-neighbor distances;
  - central and periodic cross-grain minimum distances;
  - periodic duplicate counting; and
  - non-failing `ok` / `suspicious` / `invalid` classification.
- `generate_structures.py`
  - generator schema bumped from 1 to 2;
  - audit included in each `metadata.json`;
  - selected audit fields added to `generation_results.tsv` and `manifest.json`;
  - P/Q determinant, maximum Miller-row norm, and box lengths added for campaign
    stratification; and
  - existing outputs without schema-2 audit metadata are regenerated rather than
    silently reused.
- `summarize_geometry_audit.py`
  - Markdown summary by audit status, boundary type, axis set, determinant bucket,
    Miller-row-norm bucket, atom-count bucket, and box-size ranges.
- `tests/test_geometry_audit.py`
  - flat-interface control;
  - localized-channel regression;
  - empty-bin regression;
  - periodic duplicate regression;
  - ordering and periodic-wrapping invariance; and
  - structured-array and validation coverage.
- `tests/test_zhang_geometry_regression.py`
  - verifies the original `zhang_001_ST_100` structure is classified as suspicious.
- `tests/data/zhang_001_ST_100.data`
  - original severe-void baseline fixture.
- `zhang_001_geometry_audit.json`
  - audit result obtained from the supplied baseline structure.

## Baseline result

The supplied `zhang_001_ST_100` structure is classified as `suspicious` with both
interfaces showing gap ranges greater than 10 angstroms. It also has very short central
and periodic cross-interface distances relative to the 2.36165-angstrom internal
nearest-neighbor distance. No periodic duplicate pair was detected at a 1e-6-angstrom
tolerance.

## Test command

```bash
PYTHONPATH=. pytest -q tests/test_geometry_audit.py \
  tests/test_zhang_geometry_regression.py \
  tests/test_summarize_geometry_audit.py
```

The bundle test result is `12 passed`.

## Campaign commands

Generate structures with embedded audits:

```bash
uv run python generate_structures.py \
  --data-file gb_data_gbopt.csv \
  --output-root results/structures
```

Create the campaign summary:

```bash
uv run python summarize_geometry_audit.py \
  results/structures/generation_results.tsv \
  --output results/structures/geometry_audit_summary.md
```

## Remaining Phase-A validation

The code and first-case regression are complete in this bundle. The following require
the full GBOpt checkout and generated campaign directory:

1. run all 197 cases under schema 2;
2. select a visually reasonable twist case and promote it to a passing regression
   fixture; and
3. inspect threshold sensitivity before any audit status becomes generation-failing.
