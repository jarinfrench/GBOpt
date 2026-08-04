> **CLOSED HISTORICAL TRACKER**
>
> This tracker records completed or earlier GBMaker work. Its method identifiers and status must not be confused with the active `GM1`-`GM8` roadmap in [`../../MASTER_PLAN.md`](../../MASTER_PLAN.md). Do not update it as a current progress tracker.

# GBMaker Refactor Tracker

## Progress
M01 ✓ 83c101e wrap_reduced_coordinate
M02 ✓ 76f1405 __orient_period_rows
M03 ✓ 02a3dda __reduced_coordinate_tolerance
M04 ✓ 0c1c6b7 __scaled_periodic_basis_vector
M05 ✓ __box_periodic_basis
M06 ✓ reduced/cartesian box coords
M07 ✓ __selection_basis_vectors
M08 ✓ __assert_unique_positions
M09 ✓ __clip_atoms_to_cartesian_box
M10 ✓ __select_atoms_in_box_basis
M11 ✓ __x_index_range
M12 ✓ __generate_grain
M13 ✓ __generate_gb
M14 ✓ __calculate_periodic_spacing
M15 ✓ __get_triclinic_params
M16 ☐ remove old generation
M17 ☐ docs + invariants

---

## Method Log

### M01 wrap_reduced_coordinate
commit: 83c101e (rebased from 9ccdd6c)
tests:
- pytest -k wrap_reduced_coordinate
- 6 passed, 59 deselected
notes:
- initial commit included unintended diff; branch rewritten clean

---

### M02 __orient_period_rows
commit: 76f1405
tests:
- pytest -k "orient_period_rows or orient_period_rows_wiring_feeds_triclinic_output"
- 3 passed, 63 deselected
notes:
- Python 3.12 env confirmed
- sign normalization only; norms preserved

---

### M03 __reduced_coordinate_tolerance
commit: 02a3dda
tests:
- pytest -k reduced_coordinate_tolerance
- 3 passed, 66 deselected
notes:
- rejects zero/non-finite basis vectors

---

### M04 __scaled_periodic_basis_vector
commit: 0c1c6b7
tests:
- pytest -k scaled_periodic_basis_vector
- 18 passed, 69 deselected, 2 warnings
notes:
- fixed non-numeric length-3 classification
- added overflow/non-finite guard test
- uses np.asarray(..., float) + float(box_length)
- warnings expected; behavior correct
- final verdict: pass

---

### M05 __box_periodic_basis
commit: e793638
tests:
- conda run -n GBOpt python -m pytest tests/test_gbmaker.py -q -k box_periodic_basis
- 3 passed, 87 deselected in 0.72s
- conda run -n GBOpt python -m pytest tests/test_gbmaker.py -q -k box_periodic_basis
- 3 passed, 87 deselected in 0.45s
notes:
- adds 2x3 in-plane box basis helper from primitive periods
- periodic rows scale to y_dim/z_dim axis projections
- non-periodic axes return zero basis vectors
- near-zero selected-axis projection raises GBMakerValueError
- adds orthogonal, tilted, and near-zero projection tests
- introduces private __inplane_periodic default in __init__ earlier than Method 14

---

### M06 reduced/cartesian box coords
commit: 765f756
tests:
- conda run -n GBOpt python -m pytest tests/test_gbmaker.py -q -k "reduced_box_coordinates or cartesian_from_box_coordinates"
- 4 passed, 90 deselected in 0.46s
notes:
- adds __reduced_box_coordinates and __cartesian_from_box_coordinates
- supports orthorhombic and tilted in-plane basis vectors
- preserves arbitrary leading dimensions for vectorized inputs
- one repair fixed non-vectorized solve path for higher-rank arrays
- adds orthorhombic, tilted, round-trip, and vectorized-shape tests

---

### M07 __selection_basis_vectors
commit: 1b2733d
tests:
- conda run -n GBOpt python -m pytest tests/test_gbmaker.py -q -k selection_basis_vectors
- 3 passed, 94 deselected in 0.47s
notes:
- adds canonical in-plane selection basis helper
- periodic axes reuse periodic box basis vectors
- non-periodic axes fall back to Cartesian e_y / e_z
- covers periodic-periodic and mixed periodicity cases

---

### M08 __assert_unique_positions
commit: bf908c6
tests:
- conda run -n GBOpt python -m pytest tests/test_gbmaker.py -q -k assert_unique_positions
- 3 passed
notes:
- adds duplicate-detection helper via epsilon-quantized grid
- uses round(pos / ε).astype(int64) for keying
- raises on exact and sub-epsilon duplicates
- returns silently on empty input
- tests cover distinct, exact duplicate, and sub-epsilon collision cases

---

### M09 __clip_atoms_to_cartesian_box
commit: 89d2f29
tests:
- conda run -n GBOpt python -m pytest tests/test_gbmaker.py -q -k clip_atoms_to_cartesian_box
- 3 passed, 100 deselected in 0.70s
notes:
- adds Cartesian clipping helper for non-periodic axes only
- clips x to slab bounds with epsilon on lower face
- clips y/z only when axis is non-periodic
- clamps small negative y/z to 0
- excludes atoms ≥ y_dim/z_dim on non-periodic axes

---

### M10 __select_atoms_in_box_basis
commit: 5fca4e0
tests:
- conda run -n GBOpt python -m pytest tests/test_gbmaker.py -q -k select_atoms_in_box_basis
- 3 passed, 103 deselected in 0.47s
- conda run -n GBOpt python -m pytest tests/test_gbmaker.py -q -k deduplicate_positions
- 3 passed, 106 deselected in 0.67s
notes:
- adds reduced-coordinate selection in mixed box basis
- periodic axes use half-open [0,1) domain with tolerance
- wraps accepted periodic-face atoms to canonical coordinates
- applies final x-slab filter after wrapping for x-tilted bases
- factors deduplication into __deduplicate_positions helper
- deduplicates wrapped positions using epsilon-quantized grid and verifies uniqueness
- adds tests for order preservation, near-epsilon collapse, and above-epsilon separation

---

### M11 __x_index_range
commit: d85ebcb
tests:
- conda run -n GBOpt python -m pytest tests/test_gbmaker.py -q -k x_index_range
- 3 passed, 109 deselected in 0.75s
notes:
- derives lattice-space x index range from in-plane primitive periods
- maps primitive periods into rotated unit-cell basis
- reduces integer-normal directions when numerically integral
- constructs conservative contiguous nx range using in-plane tilt and x extents
- raises on invalid bounds, degenerate basis, and zero x projection
- tests cover orthogonal coverage, tilted contiguity, and zero-projection failure

---

### M12 __generate_grain
commit: 3ccc5ec
tests:
- conda run -n GBOpt python -m pytest tests/test_gbmaker.py -q -k generate_grain
- 3 passed, 112 deselected
- conda run -n GBOpt python -m pytest tests/test_gbmaker.py -q -k single_grain_creation
- initially failed against tests/gold/fcc_Cu.txt
- passed after updating gold output
notes:
- adds shared lattice-enumeration __generate_grain path
- left/right grain builders now delegate to shared generator
- uses reduced-coordinate periodic selection on the active CSL path
- adds tests for grain bounds, right-grain interface placement, and periodic-face deduplication
- legacy zero-misorientation gold output changed and required fixture update
- treat gold-file change as part of M12 scope
- sigma3 coherent twin correctly generates for fcc, bcc, diamond, and fluorite structures.

---

### M13 __generate_gb
commit: 29834ac
tests:
- conda run -n GBOpt python -m pytest tests/test_gbmaker.py -q -k generate_gb
- 4 passed, 115 deselected in 0.88s
notes:
- wires __generate_gb through shared __generate_grain path
- computes left/right x windows directly from vacuum, left_x, and x_dim
- assembles whole_system from regenerated left and right grains
- update_spacing now refreshes dimensions and rebuilds through shared path
- vacuum_thickness and misorientation updates rebuild grains and GB region
- adds targeted tests for assembly and rebuild behavior

---

### M14 __calculate_periodic_spacing
commit: 1616ca4
tests:
- conda run -n GBOpt python -m pytest tests/test_gbmaker.py -q -k periodic_spacing
- 3 passed, 118 deselected in 1.23s
notes:
- updates __inplane_periodic based on y/z spacing threshold clamping
- emits per-axis non-periodic warnings instead of generic warning
- ensures periodicity flags match downstream behavior
- Sigma3 (111) retains periodicity on both axes with expected spacings
- Sigma7 (111) marks y as non-periodic and clamps only y spacing
- preserves x-length consistency (x_dim == left_x + right_x)

---

### M15 __get_triclinic_params
commit: ee888c5
tests:
- conda run -n GBOpt python -m pytest tests/test_gbmaker.py -q -k triclinic
- 6 passed, 116 deselected in 1.83s
notes:
- updates __get_triclinic_params to reject triclinic output when either in-plane axis is non-periodic
- selects the grain with larger y-period, consistent with spacing["y"] selection
- derives primitive_periods from oriented approximate rows and the rotated unit-cell basis
- computes A2_lab and A3_lab through __box_periodic_basis so triclinic output matches the periodic basis used by generation/selection
- updates triclinic expectation logic in tests to use the same box-periodic basis construction
- stabilizes test_orient_period_rows_wiring_feeds_triclinic_output by mocking __generate_gb and __set_gb_region during update_spacing
- adds a regression test that non-periodic in-plane axes reject triclinic output
- avoids OOM-triggering grain rebuilds in synthetic wiring tests without changing production behavior

---

### M16 remove legacy generation methods
commit: 30bec51
tests:
- conda run -n GBOpt python -m pytest tests/test_gbmaker.py -q -k "generate_gb or generate_grain or triclinic"
- 13 passed, 109 deselected in 2.13s
notes:
- removes __generate_left_grain, __generate_right_grain, and __get_points_inside_box
- eliminates last legacy Cartesian clipping path
- __generate_grain is now the sole generation path
- boundary-epsilon behavior validated through active clipping helper
- no remaining internal callers of removed methods

---

## Rules (keep stable for caching)
- One method at a time
- Max 1 repair cycle unless blocker
- Diff-only review
- Compact outputs only
