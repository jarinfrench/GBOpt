# GBMaker Incremental Pipeline Refactoring Plan

## Document status

**Status:** Active component design.  
**Implementation authority:** `../MASTER_PLAN.md`.  
**Roadmap mapping:** `GM1` through `GM8`, with `F0` as the characterization prerequisite.  
**Cross-track dependency:** `IO3` exclusively owns LAMMPS writer and restricted-triclinic extraction.

This document supplies architectural rationale and detailed acceptance criteria. Its original numerical phases have been mapped to roadmap PR identifiers below and do not define an independent branch sequence.

This document defines an incremental, behavior-preserving refactor of the current
`GBOpt/GBMaker.py` monolith into a staged construction pipeline. It is a planning
document only; it does not authorize scientific or behavioral changes unless a phase
explicitly identifies them.

The supplied analysis baseline was verified before review:

- Archive: `gbopt-chat1-exact-decorated-sites-source.tar.gz`
- Recorded SHA-256: `5e62a9c1a57b2ce6b38fbd8050f78bcc7e6fec21e2193ef447f94302c4356f46`
- Computed SHA-256: `5e62a9c1a57b2ce6b38fbd8050f78bcc7e6fec21e2193ef447f94302c4356f46`
- Verification result: pass

The current `GBOpt/GBMaker.py` is approximately 3,506 lines and combines public input
adaptation, validation, crystallographic orientation resolution, periodicity planning,
commensurability and strain accommodation, exact and approximate grain generation,
coordinate filtering, bicrystal assembly, mutable object state, and LAMMPS output.

When implementation begins, the work must start from the latest accepted project
baseline, not automatically from this analysis archive. If later exact-path integration
or gap-equalization-removal changes have been accepted, those changes must be present
before `F0` is repeated.

---

## 1. Objective

Refactor `GBMaker` into a sequence of explicit, testable construction stages while
preserving the current public class and its observable behavior.

The target construction flow is:

```text
User input / BoundarySpec
          |
          v
Normalized build configuration
          |
          v
Resolved material and boundary input
          |
          v
Orientation and periodicity state
          |
          v
Commensurability and dimension plan
          |
          v
Exact or approximate grain builder
          |
          v
Bicrystal assembly
          |
          v
Immutable, calculator-neutral construction result
          |
          v
GBMaker compatibility facade

Serialization is a separate package boundary:

GBMaker facade or another caller
          |
          v
GBOpt.io backend adapter (LAMMPS now; other formats later)
```

The result should be easier to understand, test, diagnose, and modify without causing
unrelated changes elsewhere in grain-boundary construction.

---

## 2. Goals

1. Keep `GBMaker` as the stable public entry point used by existing callers.
2. Separate scientific calculations from mutable object orchestration.
3. Represent intermediate construction states explicitly with typed dataclasses.
4. Make exact and approximate grain construction independent strategies with a common
   request/result contract.
5. Make each pipeline stage callable and testable without constructing a partially
   initialized `GBMaker` object.
6. Preserve deterministic atom ordering, numerical results, warnings, exceptions, and
   output formatting during the structural refactor.
7. Move `gbmaker_supercell.py` into the new implementation package without breaking
   existing imports during the migration.
8. Eliminate tests that reach through name-mangled `GBMaker` internals as the associated
   functionality is extracted.
9. Leave every phase in a releasable state with a verified source archive and SHA-256
   handoff record.
10. Keep file-format- and calculator-specific serialization outside `GBOpt.gbmaker` so
    the construction pipeline remains independent of LAMMPS, VASP, GULP, Quantum
    ESPRESSO, or any future calculator interface.

---

## 3. Non-goals

The refactor must not, by itself:

- change the mathematical definition of a grain boundary;
- change exact-versus-approximate dispatch;
- change commensurability search ordering or strain policy;
- change default parameter values;
- redesign `GBManipulator` or `GBMinimizer`;
- pre-optimize a boundary or assume the responsibilities of the optimizer;
- change atom deletion, termination, translation, or overlap policy;
- change public setter semantics unless handled as a separately approved behavior
  change;
- introduce a plugin framework or generalized workflow engine;
- design the complete multi-calculator IO API or add VASP, GULP, Quantum ESPRESSO, or
  other new writers as part of the GBMaker refactor;
- preserve obsolete gap-equalization behavior as a permanent architectural component.

Scientific corrections and API redesigns should be performed in separate, explicitly
scoped changes after the extraction makes them safe to implement.

---

## 4. Compatibility strategy

### 4.1 Stable public facade

The following imports must continue to work throughout the refactor:

```python
from GBOpt import GBMaker
from GBOpt.GBMaker import GBMaker
from GBOpt.GBMaker import GBMakerError, GBMakerTypeError, GBMakerValueError
from GBOpt.GBMaker import _find_commensurate_pair
from GBOpt.GBMaker import wrap_reduced_coordinate
```

`GBManipulator` and `GBMinimizer` currently use `isinstance(..., GBMaker)`. The public
class identity must therefore remain stable.

### 4.2 Internal lowercase package

The recommended incremental structure is a new lowercase internal package while the
existing uppercase module remains the compatibility facade:

```text
GBOpt/
├── GBMaker.py                         # Stable public facade
├── gbmaker/
│   ├── __init__.py                    # Internal construction package boundary
│   ├── errors.py
│   ├── config.py
│   ├── inputs.py
│   ├── material.py
│   ├── orientation.py
│   ├── dimensions.py
│   ├── geometry.py
│   ├── assembly.py
│   ├── pipeline.py
│   ├── types.py
│   └── builders/
│       ├── __init__.py
│       ├── common.py
│       ├── exact.py
│       ├── approximate.py
│       └── supercell.py
├── io/                                # Package-level, calculator-facing IO boundary
│   ├── __init__.py                    # Minimal stub during the GBMaker refactor
│   └── lammps.py                      # Existing writer after final extraction
└── gbmaker_supercell.py               # Temporary compatibility re-export
```

This avoids an unsafe intermediate state in which `GBOpt/GBMaker.py` and a
`GBOpt/GBMaker/` directory compete for the same import name. It also follows normal
Python naming conventions for implementation packages.

A later major-version change may convert `GBOpt.GBMaker` itself into a package, but that
is not required to obtain the architectural benefits of the pipeline refactor.

### 4.3 Internal dependency direction

Dependencies must point toward lower-level stages only:

```text
errors / types / config
          ^
          |
inputs / material
          ^
          |
orientation
          ^
          |
dimensions
          ^
          |
geometry and builders
          ^
          |
assembly
          ^
          |
pipeline
          ^
          |
GBMaker facade and output adapters
```

Internal construction modules must not import `GBOpt.GBMaker`, `GBManipulator`,
`GBMinimizer`, or `GBOpt.io`. The facade may import both the construction pipeline and
an IO backend, but neither may import the facade. `GBOpt.io` must accept explicit,
calculator-neutral structure data rather than a `GBMaker` instance.

### 4.4 Separate IO package boundary

`GBOpt.gbmaker` is responsible for constructing a chemically and geometrically defined
bicrystal. It must terminate in a calculator-neutral result containing atoms,
coordinates, cell vectors or equivalent box data, periodic-boundary metadata, species,
and construction provenance.

`GBOpt.io` is responsible for converting such neutral structure data into external file
formats or calculator-specific conventions. Therefore:

- LAMMPS data serialization belongs in `GBOpt.io.lammps`;
- conversion to the LAMMPS restricted-triclinic representation, including tilt factors,
  belongs in `GBOpt.io.lammps` because it is a LAMMPS representation requirement;
- generic cell-vector algebra that is genuinely independent of an output format may
  remain in a shared geometry module, but no LAMMPS-named or LAMMPS-shaped result may
  appear in `GBOpt.gbmaker`;
- `GBMaker.write_lammps()` may remain as a compatibility wrapper that delegates to
  `GBOpt.io.lammps`;
- the initial `GBOpt.io` package is deliberately a stub, not a complete registry,
  plugin framework, or universal calculator abstraction.

The general multi-calculator IO design should be planned separately after the GBMaker
pipeline refactor establishes a stable neutral structure contract.

---

## 5. Target data model

The exact field list may be adjusted during implementation, but the stage boundaries
must be represented explicitly rather than through hidden mutation of `GBMaker`.

### 5.1 `GBBuildConfig`

An immutable, normalized snapshot of user-controlled construction parameters.

Suggested fields:

```python
@dataclass(frozen=True)
class GBBuildConfig:
    a0: float
    structure: str
    atom_types: tuple[str, ...]
    gb_thickness: float
    repeat_factor: tuple[int, int]
    x_dim_min: float
    vacuum_thickness: float
    interaction_distance: float
    gb_id: int
    epsilon: float
    mismatch_tol: float | None
    mismatch_max_cells: int
    strain_grain: str
```

Boundary-spec mode, legacy misorientation input, and resolved embedding information
should not be mixed casually into this material/geometry configuration. They should be
represented by a separate resolved-input object.

### 5.2 `ResolvedBoundaryInput`

Captures the result of adapting a public constructor or `BoundarySpec` into a common
internal representation.

Suggested contents:

- normalized five-angle legacy misorientation representation;
- optional `BoundaryEmbedding`;
- exact/approximate construction mode;
- coherence and in-plane periodicity metadata;
- source/provenance information needed for diagnostics.

### 5.3 `MaterialState`

Contains the constructed `UnitCell`, atom names, lattice parameter, structured atom
dtype information, and derived atom radius. It prevents unit-cell creation from being
embedded in orientation or grain-generation code.

### 5.4 `OrientationState`

Replaces the side effects currently performed by `__assign_orientations` and
`__calculate_periodic_spacing`.

Suggested contents:

```python
@dataclass(frozen=True)
class OrientationState:
    left_rotation: np.ndarray
    right_rotation: np.ndarray
    left_periodic_miller_rows: np.ndarray
    right_periodic_miller_rows: np.ndarray
    primitive_periods: Mapping[str, object]
    inplane_periodic: tuple[bool, bool]
    left_x: float
    right_x: float
    x_dim: float
```

The exact contents of `primitive_periods` should be replaced by a typed object rather
than left as an unstructured dictionary once its current schema has been characterized.

### 5.5 `AxisAccommodation` and `DimensionPlan`

`AxisAccommodation` is the public-neutral successor to the current
`_AxisStrainAccommodation`. `DimensionPlan` captures all consequences of repeat factors,
mismatch tolerance, strain policy, interaction-distance minimums, and box sizing.

Suggested `DimensionPlan` contents:

- final `x_dim`, `y_dim`, and `z_dim`;
- final box bounds;
- per-axis commensurate repeat metadata;
- per-grain y/z strain scales;
- per-axis periodicity;
- warnings or diagnostics that must be emitted by the facade;
- grain x bounds used by construction.

### 5.6 `GrainBuildRequest` and `GrainBuildResult`

Exact and approximate construction must consume the same high-level request and return
the same result shape.

```python
@dataclass(frozen=True)
class GrainBuildRequest:
    side: Literal["left", "right"]
    config: GBBuildConfig
    material: MaterialState
    orientation: OrientationState
    dimensions: DimensionPlan

@dataclass(frozen=True)
class GrainBuildResult:
    atoms: np.ndarray
    origin_ids: np.ndarray
    basis_size: int
    x_layer_labels: np.ndarray | None = None
```

The current `_FloatGrainBuildResult` is evidence that this result boundary already
exists conceptually. It should be generalized rather than duplicated.

### 5.7 `BicrystalResult`

The immutable output of the scientific construction pipeline.

Suggested contents:

```python
@dataclass(frozen=True)
class BicrystalResult:
    left_grain: np.ndarray
    right_grain: np.ndarray
    whole_system: np.ndarray
    gb_region: np.ndarray
    box_dims: np.ndarray
    cell: np.ndarray
    pbc: tuple[bool, bool, bool]
    gb_plane_x: float
    radius: float
    orientation: OrientationState
    dimensions: DimensionPlan
```

The facade may initially mirror these values into legacy private attributes, but the
end state should store one configuration and one construction result. `cell` and `pbc`
are format-neutral. LAMMPS tilt factors, restricted-triclinic bounds, and writer-specific
atom records must not be fields of `BicrystalResult`.

### 5.8 Array ownership

At pipeline boundaries:

- input arrays must not be modified in place;
- result arrays must have clear ownership;
- mutable arrays should be copied before exposure if caller mutation would corrupt
  cached state;
- structured atom dtype and field ordering must remain unchanged;
- atom ordering must be deterministic and treated as a compatibility property during
  the refactor;
- optional read-only NumPy flags may be used internally after compatibility with current
  callers is confirmed.

---

## 6. Current-to-target responsibility map

| Current `GBMaker.py` responsibility | Principal current methods | Target module/stage |
|---|---|---|
| Public construction adapters | `__init__`, `_from_boundary_embedding`, `from_boundary_spec` | `GBMaker.py`, `gbmaker.inputs` |
| Validation and normalization | `__validate*`, `__validate` | `gbmaker.config`, `gbmaker.inputs` |
| Unit-cell construction | `__init_unit_cell` | `gbmaker.material` |
| Rotation assignment and integer approximation | `__assign_orientations`, `__approximate_rotation_*`, row helpers | `gbmaker.orientation` |
| Periodicity and x-period calculation | `__calculate_periodic_spacing`, `__x_period` | `gbmaker.orientation` |
| Commensurate-pair search | `_find_commensurate_pair` | `gbmaker.dimensions` with facade re-export |
| Strain and in-plane dimensions | `__build_strain_accommodation`, `__grain_strain_scales`, `__ensure_minimum_inplane_dim`, `__update_dims` | `gbmaker.dimensions` |
| Box geometry | `__calculate_box_dimensions`, grain x bounds | `gbmaker.dimensions` / `gbmaker.geometry` |
| Exact supercell enumeration | `gbmaker_supercell.py`, `__exact_grain_repeats`, `__generate_grain_exact` | `gbmaker.builders.supercell`, `gbmaker.builders.exact` |
| Approximate grain generation | `__generate_grain_result` and float-result helpers | `gbmaker.builders.approximate` |
| Coordinate transforms and complete-origin operations | methods from `__reduced_coordinate_tolerance` through `__select_complete_origins_in_box_basis` | `gbmaker.geometry` |
| Path selection and grain pair construction | `__use_exact_grain_generation`, `__generate_exact_grains`, `__generate_float_grains` | `gbmaker.builders.common` / `gbmaker.pipeline` |
| Bicrystal assembly and GB region | `__generate_gb`, `__set_gb_region` | `gbmaker.assembly` |
| Gap equalization | `__current_gap_metrics`, `__equalize_*` | Remove; do not create a permanent module |
| LAMMPS output and restricted-triclinic conversion | `write_lammps`, `__get_triclinic_params` | `GBOpt.io.lammps` |
| Mutable compatibility behavior | properties and setters | `GBMaker.py` facade |

---

## 7. Required behavioral invariants

Every phase must preserve the following unless its scope explicitly authorizes a change:

1. Public import paths and exception class identity.
2. `isinstance(gb, GBMaker)` behavior in `GBManipulator` and `GBMinimizer`.
3. Constructor and `from_boundary_spec` defaults.
4. Exact/approximate mode validation and dispatch.
5. `uses_exact_construction` and `inplane_periodic` results.
6. Atom species, counts, stoichiometry, coordinates, and ordering.
7. Box dimensions, GB plane position, and grain partitioning.
8. Commensurability candidate ordering and selected repeat pairs.
9. Strain scales and minimum in-plane dimension behavior.
10. Warning categories, warning conditions, and materially significant message text.
11. Exception types and materially significant message text.
12. LAMMPS atom ordering, type mapping, charges, precision, and triclinic output.
13. Current public setter behavior, including which setters rebuild and which perform
    narrower updates.
14. Determinism across repeated runs.
15. Import isolation: clean-boundary generation must not import optimization modules as
    a side effect.

Floating-point comparisons should use the tolerances already established by the test
suite. Tolerances must not be widened merely to accommodate refactoring differences.

---

## 8. Incremental implementation phases

Each phase should normally be implemented in a separate branch or chat. A phase may be
split further, but later phases should not be pulled forward unless the dependency is
explicitly understood.

### Roadmap prerequisite F0 - Characterization and compatibility freeze

#### Objective

Create a trustworthy behavioral baseline before moving code.

#### Work

1. Verify the latest source archive and record its SHA-256.
2. Run the existing focused and non-slow test suites before editing.
3. Add public-contract tests covering import paths, class identity, public properties,
   warning behavior, and interaction with `GBManipulator` and `GBMinimizer`.
4. Add deterministic characterization for representative construction paths:
   - legacy constructor;
   - exact PQ FCC;
   - exact CSL FCC;
   - exact fluorite/UO2;
   - approximate non-CSL;
   - exact mismatch accommodation;
   - approximate mismatch accommodation;
   - zero-vacuum and nonzero-vacuum cases;
   - orthogonal and triclinic LAMMPS output.
5. For each representative case, record:
   - input parameters and boundary specification;
   - warning sequence;
   - exact/approximate dispatch;
   - periodicity flags;
   - dimensions and box bounds;
   - left/right/whole atom counts and species counts;
   - order-sensitive hashes of structured arrays;
   - order-insensitive coordinate/species hashes for diagnosis;
   - coordinate extrema;
   - LAMMPS output hash where applicable.
6. Document all existing name-mangled test access so each test can be migrated in the
   phase where its underlying method moves.

#### Acceptance criteria

- No production behavior changes.
- Existing tests pass.
- Characterization results are reproducible in two consecutive runs.
- The baseline manifest and test cases are committed.

#### Handoff

Produce a verified source archive, SHA-256 file, test summary, and characterization
manifest.

---

### GM1 - Internal package foundation and shared construction types

#### Objective

Create the package skeleton and canonical data contracts without changing construction
flow.

#### Work

1. Create `GBOpt/gbmaker/` and `GBOpt/gbmaker/builders/` for construction code.
2. Create a separate top-level `GBOpt/io/` package stub with documentation defining
   the calculator-neutral boundary. Do not move or redesign the LAMMPS writer yet.
3. Move exception definitions to `gbmaker/errors.py`; re-export the same class objects
   from `GBOpt.GBMaker`.
4. Add `GBBuildConfig`, `ResolvedBoundaryInput`, `MaterialState`, `OrientationState`,
   `AxisAccommodation`, `DimensionPlan`, `GrainBuildRequest`, `GrainBuildResult`, and
   `BicrystalResult` in `gbmaker/types.py` or narrowly separated type modules.
5. Define shape, unit, and ownership contracts in docstrings.
6. Move `gbmaker_supercell.py` implementation to
   `gbmaker/builders/supercell.py`.
7. Replace the old `GBOpt/gbmaker_supercell.py` implementation with a temporary
   compatibility re-export and a deprecation note. Do not maintain two implementations.
8. Keep `GBMaker.py` construction and LAMMPS behavior unchanged.

#### Tests

- Direct unit tests for dataclass validation and normalization.
- Existing `test_gbmaker_supercell.py` redirected to the canonical new module.
- A compatibility test confirming the old supercell import returns the same functions.
- Public exception identity tests.

#### Acceptance criteria

- No scientific method has moved yet except the already stateless supercell kernel.
- The old and new supercell imports resolve to the same function objects.
- All baseline hashes remain unchanged.

---

### GM2 - Input normalization and material resolution

#### Objective

Separate public input adaptation from scientific construction.

#### Work

1. Move scalar/sequence validation into `gbmaker/config.py` as explicit functions.
2. Move boundary mode validation and `BoundarySpec` adaptation into
   `gbmaker/inputs.py`.
3. Introduce a pure normalization function that returns `GBBuildConfig`.
4. Introduce a pure boundary-resolution function that returns
   `ResolvedBoundaryInput`.
5. Move unit-cell initialization into `gbmaker/material.py` and return `MaterialState`.
6. Keep `GBMaker.__init__`, `_from_boundary_embedding`, and `from_boundary_spec` as thin
   public adapters that delegate to these functions.
7. Preserve the legacy-constructor deprecation warning exactly once.
8. During transition, the facade may continue populating its legacy private attributes
   from the normalized objects.

#### Tests

- Validation functions tested directly rather than through private methods.
- Exact, prefer-exact, and approximate mode dispatch tests.
- Invalid boundary type and invalid mode tests.
- Unit-cell/material tests for all supported structures.
- Legacy warning count and stack-level behavior.

#### Acceptance criteria

- Public constructor behavior and messages remain compatible.
- `from_boundary_spec` contains orchestration only, not conversion algorithms.
- No duplicated validation remains in `GBMaker.py`.

---

### GM3 - Orientation and periodicity extraction

#### Objective

Replace the most consequential hidden state mutation with an explicit
`OrientationState` result.

#### Work

1. Move Miller-row reduction, row-angle calculation, and rotation-row approximation to
   `gbmaker/orientation.py`.
2. Move legacy angle-to-rotation assignment to a pure function.
3. Move embedding-derived rotation selection to the same module.
4. Move x-period and in-plane periodic spacing calculation.
5. Implement:

   ```python
   resolve_orientation(
       config: GBBuildConfig,
       boundary: ResolvedBoundaryInput,
       material: MaterialState,
       *,
       threshold: float | None = None,
   ) -> OrientationState
   ```

6. Preserve the exact path's use of integer P/Q rows without passing through float
   approximation.
7. Make `GBMaker.update_spacing()` call the new function and temporarily mirror the
   returned state into legacy fields needed by later unextracted code.
8. Replace private-method tests with direct orientation-stage tests in the same commit.

#### Tests

- Exact integer rows bypass float approximation.
- Approximate rotation rows preserve current angle-error decisions.
- Left/right rotations and Miller rows match the baseline.
- Periodic spacing, x periods, and periodicity flags match the baseline.
- Threshold behavior and errors match the baseline.

#### Acceptance criteria

- Orientation calculation performs no mutation of `GBMaker`.
- `__calculate_periodic_spacing` and orientation approximation internals are removed
  from the facade, except for temporary compatibility wrappers only where unavoidable.
- Baseline atom and dimension hashes remain unchanged.

---

### GM4 - Commensurability and dimension planning

#### Objective

Convert repeat selection, mismatch accommodation, strain planning, and box sizing into
one explicit `DimensionPlan`.

#### Work

1. Move `_find_commensurate_pair` to `gbmaker/dimensions.py` and re-export it from
   `GBOpt.GBMaker` for compatibility.
2. Move `_AxisStrainAccommodation` to the canonical `AxisAccommodation` type.
3. Move grain strain scale calculation and per-axis accommodation construction.
4. Move minimum in-plane dimension enforcement.
5. Move box dimension and grain x-bound calculations.
6. Implement:

   ```python
   plan_dimensions(
       config: GBBuildConfig,
       orientation: OrientationState,
   ) -> DimensionPlan
   ```

7. Return warning/diagnostic information explicitly or emit warnings at a documented
   boundary. Do not let warning order become incidental.
8. Replace the dimension-planning part of `__update_dims` with this function while
   leaving existing grain generation in place.

#### Tests

- Full brute-force equivalence of commensurate-pair ordering.
- Exact and approximate no-pair behavior.
- All strain policies: `both`, `left`, and `right`.
- Interaction-distance resizing of commensurate pairs.
- Exact construction rejection when box lengths are not valid integer multiples.
- Box dimensions and grain x bounds.
- Warning sequence and text for fallback/resizing.

#### Acceptance criteria

- Dimension planning is pure with respect to facade state.
- `__update_dims` no longer performs commensurability calculations.
- No existing dimension or strain result changes.

---

### GM5 - Shared geometry kernels

#### Objective

Extract coordinate-system and complete-origin operations shared by the construction
strategies.

#### Work

Move the following families into `gbmaker/geometry.py` as pure functions:

- reduced-coordinate tolerance;
- periodic basis scaling;
- box periodic basis construction;
- selection basis construction;
- x-index range calculation;
- reduced-to-Cartesian and Cartesian-to-reduced conversion;
- complete-origin atom masks;
- complete-origin filtering;
- clipping complete origins to a Cartesian box;
- complete-origin deduplication;
- upper-x trimming;
- box-basis origin selection;
- `wrap_reduced_coordinate`.

Functions must receive all required tolerances, basis vectors, bounds, and periodicity
explicitly. They must not accept a `GBMaker` instance or a generic object containing
hidden state.

#### Tests

- Migrate every private geometry test to direct module tests.
- Test both periodic and nonperiodic in-plane axes.
- Test half-open box conventions and boundary tolerances.
- Test complete-basis retention for multi-species unit cells.
- Test deterministic deduplication and ordering.
- Re-export `wrap_reduced_coordinate` from `GBOpt.GBMaker`.

#### Acceptance criteria

- No geometry unit test accesses `_GBMaker__...` members.
- Geometry functions do not mutate inputs.
- Exact and approximate integration outputs remain byte-for-byte compatible where the
  baseline is order-sensitive.

---

### GM6 - Exact and approximate grain builders

#### Objective

Separate the two construction algorithms behind a common strategy boundary.

#### Work

1. Implement a common builder protocol or callable contract in
   `gbmaker/builders/common.py`.
2. Move exact repeat determination and exact grain generation to
   `gbmaker/builders/exact.py`.
3. Use `gbmaker/builders/supercell.py` as the only exact integer enumeration
   implementation.
4. Move conservative conventional-cell enumeration and float grain generation to
   `gbmaker/builders/approximate.py`.
5. Generalize `_FloatGrainBuildResult` into the shared `GrainBuildResult`.
6. Implement a path selector based only on resolved input and orientation metadata.
7. Implement:

   ```python
   build_grain(request: GrainBuildRequest) -> GrainBuildResult
   build_grain_pair(...) -> tuple[GrainBuildResult, GrainBuildResult]
   ```

8. Keep grain construction free of bicrystal assembly, GB-region selection, and file
   output.
9. Do not place gap equalization in either builder.

#### Gap-equalization prerequisite

If gap equalization still exists in the implementation baseline, one of the following
must happen before `GM6` closes:

- integrate the separately approved removal first; or
- leave a temporary facade-level compatibility call clearly outside the builders and
  remove it in `GM7`.

A permanent `gap_equalization.py` module must not be created.

#### Tests

- Exact builder atom counts, stoichiometry, bounds, and layer labels.
- Approximate builder complete-origin behavior and deduplication.
- Exact and approximate path selection.
- Noncommensurate and incoherent cases.
- Exact UO2 and rocksalt per-grain stoichiometry.
- No duplicate atoms introduced by extraction.
- Deterministic atom ordering.

#### Acceptance criteria

- Builder modules do not import the facade or assembly module.
- Exact and approximate algorithms can be tested independently.
- `GBMaker.py` no longer contains grain enumeration algorithms.

---

### GM7 - Bicrystal assembly and pure end-to-end pipeline

#### Objective

Create the first complete scientific construction function that does not depend on a
mutable `GBMaker` object.

#### Work

1. Move left/right placement, whole-system concatenation, and GB-region selection into
   `gbmaker/assembly.py`.
2. Define the interface convention, box convention, and atom ordering explicitly.
3. Ensure assembly does not silently delete complete basis-resolved planes.
4. Implement:

   ```python
   assemble_bicrystal(
       config: GBBuildConfig,
       material: MaterialState,
       orientation: OrientationState,
       dimensions: DimensionPlan,
       left: GrainBuildResult,
       right: GrainBuildResult,
   ) -> BicrystalResult
   ```

5. Implement the end-to-end function in `gbmaker/pipeline.py`:

   ```python
   build_bicrystal(
       config: GBBuildConfig,
       boundary: ResolvedBoundaryInput,
   ) -> BicrystalResult
   ```

6. Change the facade's rebuild path to call `build_bicrystal`.
7. During transition, mirror the returned result into old private attributes so public
   getters and setter logic remain unchanged.
8. Add import-boundary tests proving that pipeline import does not load optimization
   modules.

#### Tests

- End-to-end equality with every `F0` characterization case.
- Stage-result consistency checks.
- Exact and approximate pipeline integration.
- GB region and GB plane position.
- No gap-equalization warning or plane deletion if its removal is part of the accepted
  baseline.
- Clean import isolation.

#### Acceptance criteria

- `build_bicrystal` can construct all representative cases without instantiating
  `GBMaker`.
- All scientific construction occurs below the facade.
- Pipeline output matches the baseline exactly, except for separately approved behavior
  changes already incorporated into the execution baseline.

---

### GM8 - Facade state migration and GBMaker cleanup

#### Objective

Reduce `GBMaker` to public API adaptation, configuration ownership, cached result
ownership, and compatibility behavior.

#### Work

1. Store canonical state as:

   ```python
   self._config: GBBuildConfig
   self._boundary: ResolvedBoundaryInput
   self._result: BicrystalResult
   ```

2. Make read-only scientific properties delegate to `_result`.
3. Make configuration properties delegate to `_config`.
4. Replace configuration updates with `dataclasses.replace` or equivalent validated
   constructors.
5. Centralize full regeneration in one `_rebuild()` method.
6. Preserve each setter's current observable behavior during this phase:
   - setters that currently rebuild must rebuild;
   - setters that currently shift existing coordinates must preserve equivalent output;
   - setters that currently update metadata only must not unexpectedly regenerate;
   - warnings and validation remain unchanged.
7. Remove mirrored legacy private fields once no method depends on them.
8. Remove forwarding wrappers after all associated private tests have migrated.
9. Do not use mixins to distribute shared mutable state across files.

#### Tests

- Setter-by-setter before/after behavior.
- Object identity and array replacement/mutation behavior where callers may observe it.
- `GBManipulator` and `GBMinimizer` integration.
- Repeated rebuild determinism.
- Configuration/result consistency after every setter.

#### Acceptance criteria

- The facade contains no crystallographic, enumeration, filtering, or
  commensurability algorithm.
- A single full-build method controls regeneration.
- No tests access name-mangled scientific internals.
- Public behavior remains compatible.

---

### Cross-track dependency: IO3 - LAMMPS writer extraction

LAMMPS serialization is not a ninth GBMaker PR. `IO3` owns this seam exclusively and should merge before deep facade migration creates avoidable conflicts.

`IO3` must:

1. move LAMMPS restricted-triclinic conversion into `GBOpt.io`;
2. move LAMMPS serialization into a standalone writer that accepts neutral structure data rather than a `GBMaker` instance;
3. return `WriteResult` with deterministic transient ID mappings;
4. preserve atom order, type labels, charges, precision, and supported output formatting;
5. leave `GBMaker.write_lammps()` as a thin compatibility wrapper;
6. enforce that `GBOpt.io` does not import the GBMaker facade.

After `IO3`, `GM8` may remove remaining dead construction state and verify that no LAMMPS-specific implementation has returned to `GBOpt.gbmaker`. Final cross-track dependency tests belong to `INT1`.

### Deferred OPT-API1 - Public immutable construction API

This phase is outside the compatibility-preserving refactor and requires separate API
approval.

Possible work:

- expose `GBBuildConfig` and `build_bicrystal` as supported public APIs;
- deprecate or narrow mutable setters;
- make result arrays explicitly immutable;
- convert `GBOpt.GBMaker` from a module into a package in a major version;
- remove legacy constructor and compatibility re-exports after their deprecation
  windows.

None of this is required for `GM1` through `GM8`.

---

## 9. Testing strategy

### 9.1 Test layers

The final suite should have four distinct layers:

1. **Kernel unit tests** - integer supercells, coordinate transforms, wrapping, and
   commensurate-pair search.
2. **Stage unit tests** - orientation, dimensions, exact builder, approximate builder,
   and assembly.
3. **Pipeline integration tests** - construct representative boundaries through
   `build_bicrystal` without the facade.
4. **Public compatibility tests** - construct and mutate `GBMaker`, use downstream
   classes, and write files.

### 9.2 Private-test migration rule

When a private method moves, all tests of that private method must move in the same
roadmap PR. New tests must not add more `_GBMaker__...` references. Temporary private access
may remain only for functionality scheduled for a later phase.

The end-state target is zero tests coupled to name-mangled scientific internals.

### 9.3 Minimum commands per roadmap PR

Focused tests should be selected for the changed modules. The standard broad gates are:

```bash
python -m compileall GBOpt tests

pytest -q \
  tests/test_gbmaker.py \
  tests/test_gbmaker_exact_path.py \
  tests/test_gbmaker_from_boundary_spec.py \
  tests/test_gbmaker_supercell.py \
  tests/test_gbmanipulator.py \
  tests/test_gaminimizer.py

pytest -m "not slow"
```

Any project-specific static analysis, formatting, or type-checking commands required by
the controlling codebase rules must also pass.

Slow tests should be run at designated integration gates, at minimum after `GM6`, `GM7`, `GM8`, and `IO3`, or according to the project's existing long-test policy.

### 9.4 Characterization comparisons

Characterization should distinguish:

- order-sensitive array equality;
- order-insensitive physical equivalence;
- exact integer metadata equality;
- floating-point equality under existing tolerances;
- warning/exception compatibility;
- file-content equality.

An order-insensitive match must not be used to excuse an unexplained order-sensitive
change. Atom ordering can affect downstream reproducibility and must be changed only
intentionally.

---

## 10. Phase execution and handoff procedure

Each roadmap PR should use the same repeatable workflow:

1. Verify the incoming archive against its SHA-256 record.
2. Extract into a clean working directory or apply it to a clean Git worktree.
3. Confirm the worktree baseline and run the required pre-change tests.
4. Review the controlling project documents in their specified precedence order.
5. Implement only the current phase.
6. Run focused tests after each meaningful extraction.
7. Run the non-slow suite before closeout.
8. Review the diff for accidental behavior changes, duplicated implementations, stale
   compatibility wrappers, and import cycles.
9. Record files added, moved, changed, and removed.
10. Record tests run and their exact results.
11. Create a source archive containing the complete accepted baseline for the next
    phase.
12. Create and verify the archive's SHA-256 file.
13. Write a concise closeout/handoff note describing completed scope, deferred scope,
    compatibility shims, and known risks.

No roadmap PR should depend on uncommitted files from a previous chat or an informal
statement that a test passed.

---

## 11. Risk register

| Risk | Consequence | Mitigation |
|---|---|---|
| Moving the public module too early | Broken imports, class identity, or pickles | Keep `GBOpt/GBMaker.py` as facade; introduce lowercase package internally |
| Duplicated old/new algorithms | Divergent fixes and ambiguous source of truth | Move implementation once; compatibility modules only re-export |
| Hidden NumPy mutation | Cross-stage corruption and nondeterminism | Define ownership; copy at boundaries; test input immutability |
| Atom-order changes | Downstream reproducibility changes | Preserve order-sensitive hashes and explain every difference |
| Warning-order changes | Brittle callers/tests and hidden policy changes | Make warning emission explicit and characterize sequences |
| Broader float drift | Scientific regression hidden as refactor noise | Preserve algorithms and tolerances; no opportunistic rewrites |
| Cyclic imports | Runtime import failures | Enforce dependency direction with architecture tests |
| Setter semantic changes | Unexpected rebuilds or stale outputs | Characterize every setter before facade migration |
| Exact/approximate leakage | One path accidentally depends on the other | Common contracts, independent builders, path-specific tests |
| Formalizing gap equalization | Obsolete behavior becomes harder to remove | Do not create a permanent gap module; remove before pipeline closeout |
| Overlarge phase | Unreviewable behavior changes | One pipeline boundary per phase and complete test gate per handoff |
| Stale implementation baseline | Refactor omits accepted exact-path fixes | Start `F0` from the latest accepted archive and verify SHA |

---

## 12. Architectural rules for implementation

1. Prefer pure functions and immutable stage results.
2. Do not pass a `GBMaker` object into internal pipeline functions.
3. Do not replace explicit types with unstructured dictionaries.
4. Avoid generic `context` objects that merely recreate hidden global state. A context
   type is acceptable only when its fields have a narrow stage-specific meaning.
5. Avoid mixins. They split files without reducing shared mutable-state coupling.
6. Avoid simultaneous extraction and algorithm optimization.
7. Keep exact integer arithmetic exact; do not round-trip through float arrays.
8. Keep exact and approximate builders separate after their shared inputs have been
   normalized.
9. Make periodicity and boundary topology explicit inputs, not assumptions inferred deep
   inside geometry helpers.
10. Keep output formatting outside scientific construction.
11. Preserve current errors at the public boundary; internal errors may be more specific
    only if the facade translates them consistently.
12. Delete transitional wrappers when their tests and callers have migrated.
13. Add no permanent compatibility path without a documented removal condition.

---

## 13. Definition of done

The GBMaker track is complete when all of the following are true:

- `GBMaker` is a public compatibility facade rather than the location of construction
  algorithms.
- A complete bicrystal can be built through a pure `build_bicrystal` pipeline.
- Orientation, dimension planning, grain construction, and assembly have explicit typed
  inputs and outputs.
- Exact and approximate builders are independent modules with a common result contract.
- `gbmaker_supercell.py` has one canonical implementation inside the new package.
- Gap equalization is absent from the permanent architecture.
- LAMMPS output and restricted-triclinic conversion are independent of construction
  and reside in `GBOpt.io.lammps`.
- `GBOpt.gbmaker` exposes only calculator-neutral structure data and does not import
  `GBOpt.io`.
- No scientific tests reach through `_GBMaker__...` internals.
- Existing public imports and downstream `isinstance` checks work.
- Representative output hashes and physical invariants match the accepted baseline.
- Focused, non-slow, and required slow tests pass.
- Architecture tests show no reverse imports or optimization-module side effects.
- The final source archive and SHA-256 verify successfully.
- Developer documentation identifies the stage contracts and where new functionality
  belongs.

---

## 14. Recommended PR and implementation-session names

Use the roadmap identifiers so branches, prompts, closeout notes, and reviews remain unambiguous:

1. `F0 - Characterization Baseline and Architecture Decisions`
2. `GM1 - Internal Package and Construction Types`
3. `GM2 - Input and Material Resolution`
4. `GM3 - Orientation and Periodicity Extraction`
5. `GM4 - Dimension Planning Extraction`
6. `GM5 - Geometry Kernel Extraction`
7. `GM6 - Exact and Approximate Builders`
8. `GM7 - Assembly and Pure Pipeline`
9. `GM8 - Facade State Migration and Cleanup`

`IO3 - LAMMPS Writer Extraction and WriteResult` is a separate I/O-track PR and must not be implemented as a GBMaker phase.

Each implementation session should receive the controlling documents, the latest accepted source archive and SHA-256, the characterization manifest, and a prompt that prohibits work outside the named roadmap PR.

---

## 15. Decision summary

The recommended implementation is an incremental form of the pure staged-pipeline
architecture:

- preserve `GBOpt.GBMaker` as the stable facade;
- build the implementation under `GBOpt.gbmaker`;
- introduce typed immutable stage results before moving complex algorithms;
- extract orientation, dimensions, geometry, builders, and assembly in that order;
- create the end-to-end pure pipeline only after its component stages exist;
- migrate mutable facade state after the pipeline is proven equivalent;
- coordinate with `IO3` so LAMMPS serialization and restricted-triclinic conversion move out of `GBMaker` before deep facade migration;
- keep the multi-calculator IO framework itself outside this refactor;
- remove compatibility scaffolding last;
- never convert the current monolith into a collection of mixins or mutually dependent
  modules sharing the same hidden state.

This sequence achieves the long-term architecture without requiring a single high-risk
rewrite and provides a verifiable stopping point after every phase.
