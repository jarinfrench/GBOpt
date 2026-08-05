# F0 closeout: Characterization baseline and architecture decisions

## Scope completed

- Recorded the user-selected source archive and SHA-256.
- Added six accepted architecture decision records.
- Added a deterministic cross-cutting behavior manifest and generator.
- Added regression tests that compare current behavior with the committed manifest.
- Added skipped legacy-checkpoint reference tests and a behavioral inventory.
- Made no changes under `GBOpt/`.

## Files added

```text
docs/architecture/README.md
docs/architecture/0001-shared-domain-contracts.md
docs/architecture/0002-io-owns-file-syntax.md
docs/architecture/0003-operation-level-manipulation.md
docs/architecture/0004-typed-evaluation-results.md
docs/architecture/0005-events-are-not-checkpoints.md
docs/architecture/0006-source-baseline-authority.md
docs/characterization/F0_BASELINE.md
docs/characterization/F0_CLOSEOUT.md
docs/characterization/LEGACY_CHECKPOINT_REFERENCE.md
tests/characterization/__init__.py
tests/characterization/f0_manifest.py
tests/characterization/baseline_manifest.json
tests/characterization/test_f0_characterization.py
tests/characterization/test_checkpoint_reference.py
```

## Compatibility

- Public imports, signatures, warnings, exceptions, serialization, and numerical
  behavior are unchanged.
- No production module was modified.
- The characterization layer intentionally calls compatibility entry points so later
  extraction PRs must preserve them unless a separate behavior/API change is approved.

## Test environment

The implementation environment used Python 3.13.5 on Linux x86-64. Package versions are
recorded in `baseline_manifest.json`.

`spglib` was unavailable and could not be installed because the execution environment
had no package-network access. An external temporary import stub was used only so tests
that do not invoke `spglib` could collect. It is not included in the repository.

## Commands and results

### Manifest repeatability

```bash
PYTHONPATH=/mnt/data/gbopt_test_deps:.:tests \
  python -m characterization.f0_manifest --verify-repeat
```

Result: passed; two consecutive behavior manifests were identical.

### Compile check

```bash
python -m compileall -q GBOpt tests
```

Result: passed.

### New F0 tests

```bash
PYTHONPATH=/mnt/data/gbopt_test_deps:.:tests \
  pytest -q tests/characterization
```

Result: `2 passed, 8 skipped`.

### Focused integration gate

```bash
PYTHONPATH=/mnt/data/gbopt_test_deps:.:tests pytest -q \
  tests/test_gbmaker_from_boundary_spec.py \
  tests/test_gbmaker_exact_path.py \
  tests/test_interface_separation.py \
  tests/test_gaminimizer.py \
  tests/characterization
```

Result: `93 passed, 8 skipped`.

### GBMaker characterization surface

```bash
PYTHONPATH=/mnt/data/gbopt_test_deps:.:tests \
  pytest -m 'not slow' -q tests/test_gbmaker.py
```

Result: `1141 passed, 31 subtests passed`.

### GBManipulator, excluding the unavailable-spglib integration

```bash
PYTHONPATH=/mnt/data/gbopt_test_deps:.:tests pytest -m 'not slow' \
  -k 'not displace_along_soft_modes_preserves_multitype_numeric_roundtrip' \
  -q tests/test_gbmanipulator.py
```

Result: `114 passed, 12 deselected`.

The excluded test was run separately and failed only when the external stub raised at
`spglib.get_ir_reciprocal_mesh`; no GBOpt assertion was reached.

### Remaining non-slow source tests

```bash
PYTHONPATH=/mnt/data/gbopt_test_deps:.:tests pytest -m 'not slow' -q \
  tests/test_atom.py \
  tests/test_boundary_spec.py \
  tests/test_crystallography_boundary.py \
  tests/test_crystallography_csl.py \
  tests/test_crystallography_embedding.py \
  tests/test_crystallography_exactification.py \
  tests/test_crystallography_integer.py \
  tests/test_crystallography_orientation.py \
  tests/test_crystallography_plane.py \
  tests/test_crystallography_pq.py \
  tests/test_crystallography_quaternion.py \
  tests/test_crystallography_reduction.py \
  tests/test_crystallography_rotation.py \
  tests/test_crystallography_types.py \
  tests/test_gb_params.py \
  tests/test_gbmaker_supercell.py \
  tests/test_position.py \
  tests/test_unitcell.py \
  tests/test_utils_integer_linalg.py \
  tests/test_utils_integer_normal_forms.py
```

Result: `902 passed`.

A single all-files invocation was attempted but exceeded the execution tool's four-minute
limit. The files were therefore executed in the explicit partitions above. No result is
claimed for slow soft-mode tests.

## Deferred work

- Production extraction begins only after F0 is accepted.
- CP0 must classify the legacy checkpoint behaviors before checkpoint code is ported.
- The F0 manifest must not be regenerated to conceal unexplained behavior drift.
