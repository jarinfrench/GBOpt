# GBOpt Refactoring and Checkpointing Pull-Request Roadmap

## Document status

**Recommended master implementation plan.**

This document reconciles the following proposed refactorings against the current GBOpt source and the older checkpoint-enabled source:

- `GBOPT_LOGGING_AND_OBSERVABILITY_PLAN.md`
- `GBOPT_IO_ABSTRACTION_PROPOSAL.md`
- `GBMANIPULATOR_ABSTRACTION_PROPOSAL.md`
- `GBMAKER_INCREMENTAL_PIPELINE_REFACTOR_PLAN.md`
- `AI_AGENT_CODEBASE_RULES.md`
- `gbopt_source_code_2026-08-04T10-00.tar.gz` — current implementation baseline
- `gbopt_logging_source.tar.gz` — older checkpointing reference implementation

The purpose is to make the work reviewable as separate pull requests, minimize merge conflicts, preserve current behavior, and avoid making the major refactoring tracks mutually dependent.

The current source archive is authoritative for code and behavior unless a PR explicitly adopts a tested behavior from the older checkpoint implementation. The older archive is authoritative only as a behavioral and design reference for checkpoint/restart functionality; it must not be merged wholesale.

### Verified source hashes

| Source | SHA-256 |
|---|---|
| `gbopt_source_code_2026-08-04T10-00.tar.gz` | `2627a604830655bcf452a89872244aec7fdf8f1d1c44dececdfc12944655ea5a` |
| `gbopt_logging_source.tar.gz` | `2ea4ba4cfa60a4ae432d6af952acd279d6282dd50457cd2828acd89f951ae2fc` |
| `GBOPT_LOGGING_AND_OBSERVABILITY_PLAN.md` | `419d4e2cd63bbb506b4d5bc7bbfbe324e93fa93353f6151fadbd0b54022d1f28` |
| `GBOPT_IO_ABSTRACTION_PROPOSAL.md` | `27a5a25740bf0a1472fe10dfdffbf3242f1e2361f995e3f7d6a95353bef69d7b` |
| `GBMANIPULATOR_ABSTRACTION_PROPOSAL.md` | `eaabad2dd6fd6e0be8c7249a33c865f7904a7351c5650fe3bd353f27b980df93` |
| `GBMAKER_INCREMENTAL_PIPELINE_REFACTOR_PLAN.md` | `67d7a1b10c27f4b7ccd53f506e75b2bde018ac32732954e5d21a5ac652d8ec12` |
| `AI_AGENT_CODEBASE_RULES.md` | `177fdf6222a6544fd5a3cc74b95d428e2373537dac4a32313ec1cd01de9e0f11` |

---

## 1. Executive recommendation

Use a **shallow dependency graph with controlled integration points**, rather than trying to make every PR completely independent.

The work should be organized into seven tracks:

1. **Foundation and shared domain contracts**
2. **GBMaker staged pipeline**
3. **Structure I/O and candidate reload**
4. **Manipulation strategies**
5. **Evaluation, logging, and observability**
6. **Checkpoint and restart**
7. **Final integration and stabilization**

The first three foundation PRs establish stable seams. After those merge, substantial work can proceed in parallel. Checkpoint persistence infrastructure can also begin early, but checkpoint integration with MC and GA must wait until candidate, manipulation, evaluation, and reload contracts are stable.

The primary architectural rule is:

> Each PR owns one layer or one migration seam. Extraction, new behavior, and cross-layer integration are separate PRs.

---

## 2. What “self-contained PR” means

A PR is considered self-contained when all of the following are true:

1. It passes the required test suite when applied to its declared prerequisite branch.
2. It has no more than one normal prerequisite PR, except for explicitly labeled integration PRs.
3. It does not require code from two unmerged feature branches.
4. It preserves existing public imports through compatibility re-exports or facades.
5. It does not combine a behavior-preserving extraction with a scientific or optimizer-policy change.
6. It can be reverted without reverting an unrelated parallel track.
7. It names the module or layer that owns the behavior it changes.
8. It records exact tests run and does not claim tests that were not executed.
9. It does not introduce a duplicate implementation during migration.
10. It leaves the repository in a releasable state.

Some integration PRs necessarily have multiple prerequisites. Those PRs should be small and explicitly identified as integration work rather than presented as independent refactors.

---

## 3. Stable architectural boundaries

The plans converge on several shared concepts. These must be separated before implementation proceeds deeply.

| Contract | Owns | Must not own |
|---|---|---|
| `StructureData` | Generic atoms, full cell, origin, periodicity, optional external atom IDs and charges | Grain identity, optimization policy, calculator execution |
| `GrainOwnership` | Persistent row-aligned grain labels, authoritative interface plane, physical grain bounds, topology, coordinate tolerance | File syntax, energy, optimizer history |
| `InterfaceCandidate` | Optimization-ready composition of atoms and persistent interface state | File parsing, calculator execution, restart history |
| `WriteResult` / `StructureArtifact` | Serialization output, format, path, digest, transient ID mapping, declared losses | Persistent atom identity, optimizer selection policy |
| `ManipulationResult` | Child candidates, operation name, parameters, lineage metadata | I/O, objective evaluation, logging backend |
| `EvaluationResult` | Candidate evaluation status, energy, selection energy, artifact, failure stage and reason, reconstructed candidate | Population selection or MC acceptance policy |
| `RunContext` | Run ID, resolved seed, algorithm, optional campaign/case identity | Mutable optimizer state |
| `OptimizationSnapshot` | Restartable MC or GA state at a completed safe boundary | Logs, open files, callbacks, scheduler clients |
| `OptimizationEvent` | A typed report of an already-authoritative state transition | Restart state or algorithmically significant data not stored elsewhere |

### Current-source adjustment

The current code already contains a strong immutable `InterfaceCandidate` in `GBManipulator.py`, including:

- atoms;
- row-aligned grain labels;
- box bounds;
- authoritative `gb_plane_x`;
- left and right physical grain bounds;
- in-plane periodicity;
- normal topology;
- coordinate tolerance;
- accumulated interface separation.

This type should be extracted into a neutral domain module early. It should not remain owned by the future manipulation facade, because I/O, evaluation, and checkpoint code also need it.

Likewise, `GrainOwnership` should remain a domain type, not become part of the I/O syntax layer.

---

## 4. Parallel implementation map

### 4.1 Simple map

```text
F0  Baseline characterization + architecture decisions
|
+-- F1  Mechanical minimizer decomposition -----------------------+
|                                                                  |
+-- F2  Interface-state domain extraction --------------------+     |
|                                                             |     |
+-- GM1 -> GM2 -> GM3 -> GM4 -> GM5 -> GM6 -> GM7 -> GM8       |     |
|   GBMaker staged pipeline                                    |     |
|                                                             |     |
+-- CP0 -> CP1                                                  |     |
|   Legacy checkpoint audit and persistence foundation         |     |
|                                                             |     |
|   F2 -> IO1 -> IO2 -> IO4 -> IO5 ----------------------------+-----+
|          |       |      |      |                                   |
|          +-> IO3 writer extraction                                  |
|                         |                                            |
|                         +-> MAN1 -> MAN2 -> MAN3 -> MAN4 -> MAN5 -----+
|                                                                  |   |
|   F1 -> OBS1                                                    |   |
|   F1 + IO1 -> EVAL1 -> EVAL2 <----------------------------------+   |
|                               |                                      |
|                               +-> OBS2 -> OBS3                        |
|                               |                                      |
|                               +-> CP2 -> CP3 -> CP4 -> CP5            |
|                                                                      |
+---------------------------------------------------------------> INT1
```

### 4.2 Workstream interpretation

After F0 merges:

- F1 and F2 can be implemented in parallel.
- The GBMaker track can begin immediately and remains largely independent.
- The legacy checkpoint audit and checkpoint-store foundation can begin immediately.
- After F2, the I/O core can begin.
- After F1, narrow logging work can begin.
- Manipulation algorithm extraction waits until the `Parent`/I/O seam is stabilized.
- Evaluator normalization waits for the common I/O artifact model and candidate loader.
- Checkpoint integration waits for the stable operation, evaluation, and candidate contracts.

### 4.3 Recommended merge waves

| Wave | PRs | Parallelism |
|---|---|---|
| 0 | F0 | Single gate |
| 1 | F1, F2, GM1, CP0 | Fully parallel after F0 |
| 2 | GM2, IO1, OBS1, CP1 | Parallel with file-ownership coordination |
| 3 | GM3, IO2, IO3, EVAL1 | Parallel; IO3 owns the `write_lammps` seam |
| 4 | GM4, IO4, MAN1 | IO4 merges before MAN1 edits `GBManipulator.py` |
| 5 | GM5, IO5, MAN2, EVAL2 | Mostly parallel after IO4/IO5 gates |
| 6 | GM6, MAN3, OBS2, CP2 | Parallel after common contracts stabilize |
| 7 | GM7, MAN4, OBS3, CP3 | Parallel except shared minimizer-module review |
| 8 | GM8, MAN5, CP4 | Controlled integration wave |
| 9 | CP5 | Checkpoint completion |
| 10 | INT1 | Final cross-track stabilization |

The table is a recommended sequencing model, not a requirement that every PR in one wave merge simultaneously.

---

## 5. File-ownership windows

Parallel work will only remain practical if the high-conflict files have a designated owner during each migration window.

### 5.1 `GBOpt/GBMaker.py`

| Window | Owner | Rule |
|---|---|---|
| Before IO3 | GBMaker track may add package types and adapters but should not reorganize `write_lammps()` | Keep writer hunk stable |
| IO3 | I/O track exclusively owns `write_lammps()` and restricted-triclinic extraction | Other PRs must rebase around this change |
| After IO3 | GBMaker track owns construction and facade migration; `write_lammps()` remains a thin wrapper | No I/O logic may return to `gbmaker` modules |

### 5.2 `GBOpt/GBManipulator.py`

| Window | Owner | Rule |
|---|---|---|
| F2 | Foundation track extracts shared interface types and adds compatibility re-exports | No algorithm movement |
| IO2/IO4 | I/O track owns `Parent` parsing and `Parent.from_structure()` | Manipulation extraction waits |
| MAN1-MAN4 | Manipulation track owns the facade and operation algorithms | I/O track uses new loader modules, not manipulator internals |

### 5.3 `GBOpt/FileGrainOwnership.py`

| Window | Owner | Rule |
|---|---|---|
| F2 | Foundation track moves domain types only | Re-export old imports |
| IO1-IO5 | I/O track owns parsers, transient mappings, and reload migration | No checkpoint-specific serialization added here |
| After IO5 | Compatibility facade only; new logic goes to domain or I/O modules | Remove only under a dedicated cleanup PR |

### 5.4 `GBOpt/GBMinimizer.py`

F1 must split this module before other optimizer-facing work. After F1:

| Module | Primary owner |
|---|---|
| `optimization/monte_carlo.py` | MC policy; later checkpoint MC integration |
| `optimization/genetic.py` | GA policy; later checkpoint GA integration |
| `optimization/mutation.py` | Manipulation integration |
| `optimization/evaluation.py` | Evaluator normalization |
| `optimization/events.py` | Observability track |
| `optimization/checkpointing.py` | Checkpoint integration helpers |
| `GBOpt/GBMinimizer.py` | Compatibility re-exports only |

---

# 6. Foundation PRs

## F0 — Characterization baseline and architecture decisions

### Objective

Freeze current behavior before moving code and establish the ownership rules that every later PR must follow.

### Prerequisites

None. This is the first PR.

### Primary files

- New architecture decision records under `docs/architecture/` or the project’s selected documentation location
- New characterization tests
- Test fixtures and deterministic manifests
- No production-code changes except test-only hooks that do not alter behavior

### Required changes

1. Record the current source archive hash and test environment.
2. Add deterministic characterization for representative GBMaker construction paths:
   - legacy constructor;
   - exact P/Q FCC;
   - exact CSL FCC;
   - exact fluorite/UO2;
   - approximate non-CSL;
   - mismatch accommodation;
   - zero-vacuum periodic bicrystal;
   - nonzero-vacuum single-interface slab;
   - orthogonal and triclinic LAMMPS output.
3. Record order-sensitive and order-insensitive structure hashes.
4. Characterize the current `InterfaceCandidate`, grain-local cycling, slab cycling, translation, and interface separation behavior.
5. Characterize scalar and batch GA evaluation, including failure penalties and ownership-aware reloads.
6. Add fixed-seed MC and GA behavior fixtures without changing algorithms.
7. Import and adapt the valuable legacy checkpoint behavioral tests as skipped or reference tests where current functionality is absent.
8. Add architecture decisions covering:
   - the shared domain contracts listed in Section 3;
   - I/O ownership of file syntax;
   - operation-level manipulation strategies;
   - typed evaluator results;
   - event journal versus checkpoint distinction;
   - current archive as implementation baseline and old archive as checkpoint reference.

### Tests

- Existing focused tests
- Full non-slow suite
- Two consecutive characterization runs producing identical manifests

### Non-goals

- No extraction
- No logging changes
- No checkpoint implementation
- No warning or exception wording changes

### Exit criteria

- Baseline tests are deterministic.
- Every future track has a documented owning layer.
- The legacy checkpoint behavior expected to be preserved is enumerated.

### Conflict notes

This PR should merge before any production refactor begins.

---

## F1 — Mechanical minimizer decomposition

### Objective

Split the monolithic `GBMinimizer.py` into internal modules without changing public behavior.

### Prerequisites

F0.

### Primary files

- `GBOpt/GBMinimizer.py`
- New `GBOpt/optimization/` package
- Existing minimizer tests

### Target layout

```text
GBOpt/
    GBMinimizer.py              # compatibility imports/re-exports
    optimization/
        __init__.py
        mutation.py
        evaluation.py
        monte_carlo.py
        genetic.py
        errors.py                # only if existing exception identity is preserved
```

### Required changes

1. Move `Mutator` to `optimization/mutation.py` unchanged.
2. Move current `CandidateEvaluation` to `optimization/evaluation.py` unchanged.
3. Move `MonteCarloMinimizer` and `GeneticAlgorithmMinimizer` to separate modules.
4. Re-export the exact same public class and exception objects from `GBOpt.GBMinimizer` and `GBOpt.__init__`.
5. Preserve constructor signatures, method signatures, defaults, warnings, and exception identity.
6. Preserve module-level compatibility needed for existing imports and reasonable pickle resolution.
7. Add tests proving old and new import paths refer to the same class objects.

### Tests

- Current MC and GA tests
- Import compatibility tests
- `pytest -m "not slow"`

### Non-goals

- No logging
- No seed correction
- No evaluator redesign
- No checkpointing
- No mutation abstraction

### Exit criteria

- `GBOpt/GBMinimizer.py` is a compatibility facade.
- Numerical and optimizer behavior is unchanged.
- Later tracks can edit separate minimizer modules.

---

## F2 — Shared interface-state domain extraction

### Objective

Move persistent interface-domain types out of I/O and manipulation implementation modules so all tracks can depend on them without reverse imports.

### Prerequisites

F0. May proceed in parallel with F1.

### Primary files

- `GBOpt/GBManipulator.py`
- `GBOpt/FileGrainOwnership.py`
- `GBOpt/BoundaryTopology.py`
- New `GBOpt/interface/` package

### Target layout

```text
GBOpt/interface/
    __init__.py
    model.py              # InterfaceCandidate, GrainOwnership
    labels.py             # left/right label constants if justified
    errors.py             # domain validation errors if needed
```

### Required changes

1. Move `InterfaceCandidate` to the interface domain package without changing validation or immutability behavior.
2. Move `GrainOwnership` to the same domain layer.
3. Keep `BoundaryNormalTopology` in its existing neutral topology module unless a separate move is clearly justified.
4. Preserve old imports through re-exports from `GBManipulator.py` and `FileGrainOwnership.py`.
5. Keep `CandidateFileMapping` out of the domain model because it is serialization-local.
6. Keep parser functions and reload logic out of the domain package.
7. Add conversion helpers only when they do not introduce I/O dependencies.
8. Add tests that arrays remain defensively copied and read-only.

### Tests

- Existing interface separation tests
- Existing ownership tests
- Import compatibility tests
- Equality/validation/immutability tests for moved types

### Non-goals

- No `Parent` refactor
- No reader/writer abstraction
- No manipulation operation extraction
- No checkpoint serialization

### Exit criteria

- I/O, manipulation, evaluation, and checkpoint modules can import interface state without importing `GBManipulator` or `FileGrainOwnership`.
- Old public imports continue to work.

---

# 7. GBMaker staged-pipeline track

The GBMaker track is deliberately independent of manipulator, evaluator, observability, and checkpoint work. It may use shared neutral structure types after those are available, but it must not require them to complete its scientific construction pipeline.

The I/O track exclusively owns LAMMPS writer extraction. The GBMaker track must leave `write_lammps()` operational and must not redesign it.

## GM1 — Internal package foundation and shared construction types

### Objective

Create the internal `GBOpt.gbmaker` package and typed construction-state boundaries.

### Prerequisites

F0.

### Primary files

- `GBOpt/GBMaker.py`
- `GBOpt/gbmaker_supercell.py`
- New `GBOpt/gbmaker/` package

### Required changes

1. Add construction dataclasses:
   - `GBBuildConfig`;
   - `ResolvedBoundaryInput`;
   - `MaterialState`;
   - `OrientationState`;
   - `AxisAccommodation`;
   - `DimensionPlan`;
   - `GrainBuildRequest`;
   - `GrainBuildResult`;
   - `BicrystalResult`.
2. Define array ownership, units, shapes, and immutability contracts.
3. Move exception definitions only if exact object identity can be preserved by re-export.
4. Move the stateless supercell implementation into `gbmaker/builders/supercell.py`.
5. Replace the old supercell module implementation with a compatibility re-export.
6. Do not move scientific orchestration yet.

### Tests

- Dataclass validation and defensive-copy tests
- Existing supercell tests
- Compatibility import identity tests
- Baseline construction hashes

### Non-goals

- No input normalization movement
- No orientation movement
- No writer extraction

### Exit criteria

- The package skeleton and contracts exist.
- There is one canonical supercell implementation.
- Construction output is unchanged.

---

## GM2 — Input normalization and material resolution

### Objective

Separate user-facing construction inputs and material initialization from mutable `GBMaker` orchestration.

### Prerequisites

GM1.

### Primary files

- `GBOpt/GBMaker.py`
- `GBOpt/gbmaker/config.py`
- `GBOpt/gbmaker/inputs.py`
- `GBOpt/gbmaker/material.py`

### Required changes

1. Extract scalar, sequence, mode, and boundary-input validation into pure functions.
2. Normalize constructor inputs into `GBBuildConfig`.
3. Resolve legacy and `BoundarySpec` inputs into `ResolvedBoundaryInput`.
4. Build the `UnitCell` and derived material information in `MaterialState`.
5. Preserve the legacy-constructor warning exactly once.
6. Keep public constructors as thin adapters.
7. Translate internal exceptions back to current public exception types.

### Tests

- Direct validation tests
- Exact/prefer-exact/approximate dispatch tests
- Unit-cell/material tests
- Warning count, category, message fragment, and stack level
- Full construction characterization

### Non-goals

- No orientation or dimensions movement
- No public API redesign

### Exit criteria

- Constructor adaptation and material resolution are testable without partial `GBMaker` state.
- No duplicated validation remains in the facade for migrated inputs.

---

## GM3 — Orientation and periodicity extraction

### Objective

Replace hidden orientation mutation with a pure `OrientationState` stage.

### Prerequisites

GM2.

### Primary files

- `GBOpt/GBMaker.py`
- `GBOpt/gbmaker/orientation.py`

### Required changes

1. Move Miller-row reduction and row-angle calculations.
2. Move approximate rotation-row selection.
3. Move exact embedding-derived orientation selection.
4. Move x-period and in-plane periodicity calculations.
5. Implement `resolve_orientation(...) -> OrientationState`.
6. Preserve exact P/Q integer rows without float round trips.
7. Have the facade temporarily mirror state needed by unextracted code.
8. Migrate private-method tests to direct stage tests in the same PR.

### Tests

- Exact integer-row path
- Approximate orientation path
- Left/right rotation equivalence
- Periodicity flags and primitive periods
- Threshold and failure behavior
- Baseline atom and dimension hashes

### Non-goals

- No dimension planning
- No optimization of orientation algorithms

### Exit criteria

- Orientation computation does not mutate `GBMaker`.
- Existing orientation results and warnings are unchanged.

---

## GM4 — Commensurability and dimension planning

### Objective

Produce one explicit `DimensionPlan` from configuration and orientation state.

### Prerequisites

GM3.

### Primary files

- `GBOpt/GBMaker.py`
- `GBOpt/gbmaker/dimensions.py`
- `GBOpt/gbmaker/geometry.py` for narrow box helpers only

### Required changes

1. Move `_find_commensurate_pair` and re-export it for compatibility.
2. Move strain-accommodation calculations.
3. Move minimum in-plane dimension handling.
4. Move box dimension and physical grain-bound calculations.
5. Implement `plan_dimensions(...) -> DimensionPlan`.
6. Make warnings explicit and preserve warning order.
7. Keep exact/approximate dispatch unchanged.

### Tests

- Brute-force equivalence for commensurate-pair selection
- `both`, `left`, and `right` strain policies
- Interaction-distance resizing
- Exact no-pair failures
- Approximate fallback warnings
- Box and grain-bound equivalence

### Non-goals

- No grain generation movement
- No change to mismatch policy

### Exit criteria

- Dimension planning is pure.
- `GBMaker` no longer performs migrated commensurability calculations directly.

---

## GM5 — Shared geometry kernels

### Objective

Extract coordinate conversion, complete-origin filtering, clipping, wrapping, and deduplication into explicit pure functions.

### Prerequisites

GM4.

### Primary files

- `GBOpt/GBMaker.py`
- `GBOpt/gbmaker/geometry.py`

### Required changes

Move and directly test:

- reduced-coordinate tolerance calculation;
- periodic basis scaling;
- selection and box bases;
- reduced/Cartesian conversion;
- complete-origin masks and filtering;
- Cartesian box clipping;
- deterministic deduplication;
- upper-x trimming;
- reduced-coordinate wrapping.

Every function must receive tolerances, bases, bounds, and periodicity explicitly. None may accept a `GBMaker` instance.

### Tests

- Periodic and non-periodic axes
- Half-open box conventions
- Complete multi-species basis preservation
- Deterministic ordering
- Input immutability
- Existing geometry regression tests

### Non-goals

- No exact or approximate builder extraction
- No algorithmic tolerance changes

### Exit criteria

- Migrated geometry tests no longer access name-mangled methods.
- Exact and approximate outputs remain equivalent to baseline.

---

## GM6 — Exact and approximate grain builders

### Objective

Separate exact and approximate grain construction behind a common request/result contract.

### Prerequisites

GM5.

### Primary files

- `GBOpt/GBMaker.py`
- `GBOpt/gbmaker/builders/common.py`
- `GBOpt/gbmaker/builders/exact.py`
- `GBOpt/gbmaker/builders/approximate.py`

### Required changes

1. Implement common builder protocols or callable contracts.
2. Move exact repeat calculation and integer supercell generation to the exact builder.
3. Move floating/conservative conventional-cell enumeration to the approximate builder.
4. Generalize existing float build results into `GrainBuildResult`.
5. Implement `build_grain` and `build_grain_pair`.
6. Preserve exact/approximate path selection.
7. Keep assembly, GB-region selection, and file writing outside builders.
8. Do not create a permanent gap-equalization module.

### Tests

- Exact FCC, fluorite, and rocksalt cases
- Approximate non-CSL cases
- Atom counts, stoichiometry, bounds, origin groups, and ordering
- No duplicate atoms
- Independent exact/approximate unit tests
- Required slow construction tests

### Non-goals

- No assembly
- No writer work
- No scientific enhancement to crossover or termination

### Exit criteria

- `GBMaker.py` no longer contains grain enumeration algorithms.
- Exact and approximate builders are independently testable.

---

## GM7 — Bicrystal assembly and pure end-to-end pipeline

### Objective

Construct a complete bicrystal without instantiating a mutable `GBMaker`.

### Prerequisites

GM6.

### Primary files

- `GBOpt/GBMaker.py`
- `GBOpt/gbmaker/assembly.py`
- `GBOpt/gbmaker/pipeline.py`

### Required changes

1. Move left/right placement and concatenation.
2. Move GB-region selection.
3. Define atom ordering and half-open box conventions explicitly.
4. Implement `assemble_bicrystal(...) -> BicrystalResult`.
5. Implement `build_bicrystal(...) -> BicrystalResult`.
6. Route the facade’s rebuild path through the pure pipeline.
7. Temporarily mirror result fields where setters or properties still require them.
8. Add import-boundary tests showing clean generation does not import optimization modules.

### Tests

- Full Phase-0 characterization cases
- Stage-result consistency
- Periodic and single-interface topology cases
- GB plane and region behavior
- Exact and approximate integration
- Required slow tests

### Non-goals

- No setter redesign
- No I/O extraction

### Exit criteria

- The full scientific construction path works without a `GBMaker` object.
- Construction behavior matches the accepted baseline.

---

## GM8 — Facade state migration and GBMaker cleanup

### Objective

Reduce `GBMaker` to public API adaptation, validated configuration, cached pipeline result, compatibility setters, and the existing writer wrapper.

### Prerequisites

GM7 and IO3 if IO3 has not already merged.

### Primary files

- `GBOpt/GBMaker.py`
- `GBOpt/gbmaker/` modules
- GBMaker tests and documentation

### Required changes

1. Store canonical `_config`, `_boundary`, and `_result` state.
2. Delegate scientific properties to `_result`.
3. Delegate configuration properties to `_config`.
4. Use validated replacement for configuration updates.
5. Centralize regeneration in `_rebuild()`.
6. Preserve current setter semantics individually.
7. Remove transitional mirrored state and dead private methods.
8. Remove private-test coupling.
9. Keep `write_lammps()` as the thin I/O facade installed by IO3.
10. Add dependency-direction architecture tests.

### Tests

- Setter-by-setter behavior
- Downstream `GBManipulator` and minimizer compatibility
- Repeated rebuild determinism
- Public import and exception identity
- Full non-slow and approved slow tests

### Non-goals

- No immutable public API redesign
- No removal of compatibility imports

### Exit criteria

- `GBMaker.py` is a comprehensible facade.
- No scientific construction algorithm remains in the facade.
- LAMMPS-specific implementation remains outside `GBOpt.gbmaker`.

---

# 8. Structure I/O and candidate-reload track

## IO1 — Canonical structure model, I/O base contracts, and LAMMPS data reader

### Objective

Introduce the generic structure representation and prove it through one strict reader.

### Prerequisites

F2.

### Primary files

- New `GBOpt/io/` package
- `GBOpt/FileGrainOwnership.py`
- Reader tests

### Required changes

1. Add `StructureData` with atoms, full cell, origin, periodicity, optional external IDs, charges, and small metadata.
2. Add independent `StructureReader` and `StructureWriter` ABCs or protocols.
3. Add I/O-specific error hierarchy.
4. Add format capability declarations.
5. Extract or wrap `read_lammps_data_file()` as `LammpsDataReader`.
6. Preserve strict counts, IDs, species/type, charge, coordinate, and box validation.
7. Keep old functions as compatibility facades.
8. Define external IDs explicitly as serialization identifiers, not persistent identities.

### Tests

- Minimal valid data file
- Malformed sections
- Duplicate/missing IDs
- Type-label and type-map handling
- Non-finite coordinates
- Defensive-copy behavior
- Old/new reader equivalence

### Non-goals

- No dump reader
- No writer
- No `Parent` changes

### Exit criteria

- One strict LAMMPS reader returns `StructureData`.
- Existing public reader functions still work.

---

## IO2 — LAMMPS dump reader and frame semantics

### Objective

Move dump parsing into the same canonical structure contract.

### Prerequisites

IO1.

### Primary files

- `GBOpt/io/lammps/dump_reader.py`
- `GBOpt/FileGrainOwnership.py`
- `GBOpt/GBManipulator.py` only for compatibility routing if unavoidable

### Required changes

1. Extract `read_lammps_dump_file()`.
2. Define frame selection explicitly; initially preserve the current first-frame behavior.
3. Define wrapped, unwrapped, and scaled coordinate behavior.
4. Define required columns and type-label resolution.
5. Preserve box and topology information without inventing persistent grain labels.
6. Keep old reader functions as compatibility facades.

### Tests

- First-frame selection
- Multi-frame inputs
- Column permutations
- Wrapped/scaled coordinates
- Type labels and numeric type maps
- Malformed frames and bounds
- Old/new equivalence

### Non-goals

- No ownership reconstruction
- No generic trajectory API

### Exit criteria

- LAMMPS data and dump readers return the same canonical structure type.

---

## IO3 — LAMMPS writer extraction and `WriteResult`

### Objective

Move all LAMMPS serialization and restricted-triclinic conversion out of `GBMaker`.

### Prerequisites

IO1. This PR should merge before deep GBMaker facade migration.

### Primary files

- `GBOpt/GBMaker.py`
- `GBOpt/io/lammps/data_writer.py`
- Golden writer tests

### Required changes

1. Add validated `LammpsDataWriteOptions`.
2. Add `WriteResult` containing:
   - target;
   - assigned atom IDs;
   - row-to-ID mapping;
   - optional digest;
   - explicit loss declarations.
3. Move orthogonal and restricted-triclinic serialization.
4. Preserve atom order, type labels, charges, precision, and output formatting.
5. Make `GBMaker.write_lammps()` a thin compatibility wrapper.
6. Ensure the writer accepts neutral structure data rather than a `GBMaker` instance.
7. Add import-boundary tests proving I/O does not import the GBMaker facade.

### Tests

- Existing golden files
- Orthogonal and triclinic output
- Type labels and charges
- Deterministic ID assignment
- Standalone writer/facade equivalence
- No construction-module import from the writer

### Non-goals

- No sidecars
- No other formats
- No public registry yet

### Exit criteria

- `GBMaker` contains no LAMMPS serialization implementation.
- Candidate-local ID mappings are available from `WriteResult`.

### Conflict notes

This PR has exclusive ownership of the `write_lammps()` seam. GBMaker branches should rebase after it merges.

---

## IO4 — `Parent.from_structure()` and legacy source adaptation

### Objective

Remove file parsing from `Parent` while retaining GB-domain interpretation and compatibility constructors.

### Prerequisites

IO2 and F2.

### Primary files

- `GBOpt/GBManipulator.py`
- New source-adapter or parent-construction modules
- Parent tests

### Required changes

1. Add `Parent.from_structure(StructureData, ..., grain_ownership=None)`.
2. Move generic structure validation into the new constructor path.
3. Preserve the distinction between persistent grain ownership and geometric GB-region membership.
4. Make legacy `Parent(filename, ...)` select a reader and delegate.
5. Keep `GBMaker` source adaptation as a compatibility path.
6. Reject unsupported cell geometry or topology explicitly.
7. Remove format-specific parsing implementations from `Parent` once delegated.
8. Preserve public exception translation.

### Tests

- Existing Parent parsing tests through compatibility constructor
- Direct `Parent.from_structure()` tests
- Explicit ownership versus geometric GB-region tests
- Unsupported geometry/topology failures
- Downstream manipulator compatibility

### Non-goals

- No operation strategy extraction
- No optimizer changes

### Exit criteria

- `Parent` performs domain interpretation but no file-syntax parsing.
- All legacy supported files still load equivalently.

### Conflict notes

MAN1 must begin after this PR merges.

---

## IO5 — Central candidate loader and ownership-aware round trip

### Objective

Create one service for validating and reconstructing evaluator-returned or checkpoint-restored candidates.

### Prerequisites

IO3, IO4, and F2.

### Primary files

- New `GBOpt/io/candidate_loader.py` or equivalent higher-level module
- `GBOpt/FileGrainOwnership.py`
- `GBOpt/optimization/evaluation.py` compatibility call sites only if needed

### Required changes

1. Promote `reload_explicit_manipulator()` into a format-neutral `CandidateLoader` service.
2. Construct transient `CandidateFileMapping` from `WriteResult` plus persistent ownership.
3. Validate:
   - atom count;
   - unique expected transient IDs;
   - species per ID;
   - finite coordinates;
   - box/cell compatibility;
   - boundary topology;
   - frame selection;
   - ownership alignment;
   - supported variable-cell behavior.
4. Return a validated `InterfaceCandidate` and/or `Parent`, not a raw filename interpretation.
5. Preserve compatibility wrappers.
6. Ensure explicit-mode failures never fall back to geometric ownership inference.

### Tests

- Row reordering realignment
- Species mutation rejection
- Missing/extra atoms
- Stale mapping rejection
- Box/topology changes
- Atom crossing `gb_plane_x` without ownership loss
- Scalar and batch-compatible fixtures

### Non-goals

- No evaluator status model
- No checkpoint integration
- No sidecar schema

### Exit criteria

- All ownership-aware reloads have one authoritative implementation.
- I/O syntax, transient mapping, and domain reconstruction are clearly separated.

---

# 9. Manipulation-strategy track

The manipulator proposal remains based on an operation-level Strategy abstraction. The current source requires one important update: operation results should carry complete `InterfaceCandidate` children, not only raw atom arrays.

## MAN1 — Core manipulation protocol, registry, and facade seam

### Objective

Add the generic operation API without moving scientific algorithms.

### Prerequisites

F2 and IO4.

### Primary files

- New `GBOpt/manipulation/` package
- `GBOpt/GBManipulator.py`
- Manipulator tests

### Required changes

1. Add `Manipulation`, `ManipulationContext`, and `ManipulationResult`.
2. Define `ManipulationResult.children` as `tuple[InterfaceCandidate, ...]`.
3. Add typed errors for configuration, arity, compatibility, capability, and execution.
4. Add explicit operation registry without import-time global side effects.
5. Add `GBManipulator.apply()` and `apply_named()`.
6. Add a current-state adapter over `Parent`/`InterfaceCandidate`.
7. Retain all existing public methods and return shapes unchanged.
8. Centralize RNG handling and ensure `seed=0` is valid.
9. Add a test-only third-party operation proving external extensibility.

### Tests

- Custom operation outside GBOpt source
- Parent arity validation
- Parent immutability
- Child storage independence
- Fixed RNG replay
- Registry duplicate-name handling
- Legacy method behavior unchanged

### Non-goals

- No built-in algorithm movement
- No optimizer dispatch changes

### Exit criteria

- A custom operation can execute through `GBManipulator.apply()`.
- Existing methods remain direct compatibility entry points.

---

## MAN2 — Interface translation, termination, and separation operations

### Objective

Extract the current composable interface transformations as the first complete operation vertical slice.

### Prerequisites

MAN1.

### Primary files

- `GBOpt/GBManipulator.py`
- New manipulation modules for translation, termination, and separation
- `tests/test_interface_separation.py`
- Related manipulator tests

### Required changes

Extract operations for:

1. right-grain translation;
2. grain-local termination cycling;
3. slab termination cycling;
4. interface separation.

Each operation must:

- consume explicit interface state;
- return a complete `InterfaceCandidate`;
- preserve persistent grain labels;
- declare topology and periodicity requirements;
- carry operation parameters and metadata;
- avoid file I/O;
- avoid parent mutation.

Legacy methods remain wrappers returning their current result type.

### Tests

- Existing periodic translation tests
- Periodic and non-periodic termination tests
- Periodic bicrystal separation
- Single-interface slab separation
- Composition order tests
- Metadata and deterministic replay
- Legacy/new-path equivalence

### Non-goals

- No insertion/removal
- No optimizer integration
- No independent GB-only termination control beyond existing deferred TODO

### Exit criteria

- The current interface methodology is exercised entirely through operation objects.
- Legacy APIs remain compatible.

---

## MAN3 — Density operations and soft-mode displacement

### Objective

Extract insertion, removal, and soft-mode displacement into uniform operation contracts.

### Prerequisites

MAN2.

### Primary files

- `GBOpt/GBManipulator.py`
- `GBOpt/manipulation/density.py`
- `GBOpt/manipulation/soft_modes.py`
- Helper modules as needed

### Required changes

1. Extract insertion and removal.
2. Define explicit ownership policies for inserted atoms.
3. Preserve ownership of surviving rows after removal.
4. Extract Delaunay/grid site generation into focused collaborators.
5. Move all stochastic selection to `ManipulationContext.rng`.
6. Extract soft-mode displacement and normalize multiple-child output.
7. Put inserted/removed atom details in result metadata.
8. Remove dead code only after characterization proves it unreachable.

### Tests

- Stoichiometric insertion/removal
- Ownership alignment after row-count changes
- Fixed-seed replay including `seed=0`
- No global NumPy RNG use
- Multiple soft-mode children
- Legacy/new-path equivalence

### Non-goals

- No crossover
- No changes to physical insertion/removal policy

### Exit criteria

- All unary built-in operations use the common result contract.
- Every stochastic path uses the supplied generator.

---

## MAN4 — Binary slice-and-merge operation

### Objective

Extract crossover with explicit two-parent arity and compatibility validation.

### Prerequisites

MAN3.

### Primary files

- `GBOpt/GBManipulator.py`
- `GBOpt/manipulation/crossover.py`

### Required changes

1. Implement `SliceAndMerge` with `parent_count = 2`.
2. Validate cell, species, unit-cell, topology, and region compatibility before execution.
3. Preserve the current x-slice scientific algorithm.
4. Return complete child interface state and structured lineage metadata.
5. Remove reliance on `__one_parent` for operation semantics.
6. Keep legacy `slice_and_merge()` wrapper.

### Tests

- Arity errors
- Parent compatibility errors
- Deterministic fixed-RNG output
- Parent immutability
- Legacy/new-path equivalence

### Non-goals

- No crossover algorithm redesign
- No optimizer population changes

### Exit criteria

- Unary and binary operations share one contract.
- No operation silently ignores extra parents.

---

## MAN5 — Optimizer `OperationSpec` integration

### Objective

Remove hard-coded minimizer dispatch and let operation arity and parameter samplers drive mutation/crossover selection.

### Prerequisites

F1 and MAN4.

### Primary files

- `GBOpt/optimization/mutation.py`
- `GBOpt/optimization/monte_carlo.py`
- `GBOpt/optimization/genetic.py`
- `GBOpt/manipulation/specs.py`

### Required changes

1. Add immutable `OperationSpec` with operation identity, weight, and parameter sampler.
2. Convert legacy `choices: list[str]` into compatibility specs.
3. Preserve current default mutation probabilities and sampled parameter distributions.
4. Let operation-declared arity drive parent selection.
5. Replace the `match` statement and GA-specific hard-coded crossover path.
6. Store structured lineage from `ManipulationResult`.
7. Define explicit policy for multi-child results.
8. Keep current public constructor arguments working.

### Tests

- Legacy string choices
- Weighted custom operation specs
- Unary and binary parent selection
- Multi-child handling
- MC and GA deterministic behavior
- Test-defined custom operation in both optimizers

### Non-goals

- No evaluator redesign
- No checkpointing
- No event system

### Exit criteria

- A third-party operation can participate in MC and GA without modifying optimizer source.
- Existing defaults remain behaviorally equivalent.

---

# 10. Evaluation, logging, and observability track

## OBS1 — Narrow correctness and standard-library logging cleanup

### Objective

Fix immediate reproducibility and diagnostic defects without introducing the event framework.

### Prerequisites

F1.

### Primary files

- `GBOpt/optimization/monte_carlo.py`
- `GBOpt/optimization/genetic.py`
- Logging tests and documentation

### Required changes

1. Change MC `unique_id` default to `None` and resolve it per run.
2. Retain the resolved seed.
3. Replace MC termination `print()` calls with module logging.
4. Add run-start, initial-evaluation, best-update, generation-summary, and termination logs at appropriate levels.
5. Preserve GA evaluator exception and reload failure details before applying penalties.
6. Leave `gb_params.py` requested-result stdout unchanged.
7. Do not install handlers or call `basicConfig()` in library code.

### Tests

- Two fresh MC runs receive different generated IDs
- Resume behavior is not implemented here
- Fixed seed retained
- Log level and structured fields
- No unsolicited output by default
- `gb_params.py` JSON stdout remains parseable

### Non-goals

- No typed event protocol
- No JSONL journal
- No checkpoint events

### Exit criteria

- Immediate diagnostics improve without changing numerical behavior.
- CLI result channels remain intact.

---

## EVAL1 — Algorithm-neutral evaluation and artifact contracts

### Objective

Define typed evaluation results and compatibility adapters without yet rewriting all evaluation loops.

### Prerequisites

F1 and IO1.

### Primary files

- `GBOpt/optimization/evaluation.py`
- `GBOpt/io/model.py` or a neutral artifact module
- Evaluation unit tests

### Required changes

1. Add `EvaluationStatus` and `FailureStage` enums.
2. Add `StructureArtifact` with path, format, stable artifact ID, digest, and small metadata.
3. Add immutable `EvaluationResult` containing:
   - candidate ID;
   - status;
   - physical energy;
   - selection energy or a separate optimizer-policy field;
   - artifact;
   - failure stage/code/message;
   - reconstructed candidate when available;
   - small metadata.
4. Add adapters for legacy scalar tuple results.
5. Add adapters for batch dictionaries.
6. Validate non-finite energies, missing files, malformed entries, and index alignment.
7. Keep the current `CandidateEvaluation` as a compatibility alias or adapter until EVAL2.

### Tests

- Valid scalar and batch results
- Evaluator exceptions
- Invalid energies
- Missing artifacts
- Malformed batch entries
- Stable candidate/input-index alignment
- Immutable metadata and arrays where applicable

### Non-goals

- No optimizer-loop rewrite
- No penalty-policy change
- No event emission

### Exit criteria

- Evaluation failure information has a canonical typed representation.
- Existing callback shapes can be normalized centrally.

---

## EVAL2 — Normalize MC and GA evaluation flows

### Objective

Make typed `EvaluationResult` authoritative across scalar, batch, legacy, and explicit-ownership paths.

### Prerequisites

EVAL1 and IO5. MAN5 is recommended before merge to avoid duplicate lineage work.

### Primary files

- `GBOpt/optimization/evaluation.py`
- `GBOpt/optimization/monte_carlo.py`
- `GBOpt/optimization/genetic.py`

### Required changes

1. Route scalar and batch callbacks through common adapters.
2. Use `CandidateLoader` for validated returned-structure reconstruction.
3. Represent evaluation failure before applying the optimizer penalty.
4. Keep penalty energy as optimizer policy, not checkpoint or I/O state.
5. Preserve failure stage, exception type, and message.
6. Normalize owned and legacy paths without silently changing legacy callback signatures.
7. Replace `last_generation_evaluations` with typed records.
8. Ensure GA selection uses `selection_energy` while retaining physical result data.
9. Establish the same evaluation boundary for MC accepted candidates.

### Tests

- Scalar and batch equivalence
- Explicit ownership round trips
- Structure reload failures
- Ownership reconstruction failures
- Penalty application without context loss
- All-invalid generation reseeding
- Numerical behavior preservation

### Non-goals

- No events or journaling
- No checkpoint integration

### Exit criteria

- Every optimizer evaluation produces an `EvaluationResult`.
- Penalties no longer erase failure provenance.

---

## OBS2 — Run context and typed event protocol

### Objective

Add the common domain-event boundary and logging sink.

### Prerequisites

EVAL2 and MAN5.

### Primary files

- New `GBOpt/optimization/events.py`
- MC and GA modules
- Event tests

### Required changes

1. Add immutable `RunContext` with run ID, resolved seed, algorithm, and optional case/campaign IDs.
2. Add versioned `OptimizationEvent`.
3. Add `EventSink`, `NullEventSink`, `LoggingEventSink`, and `CompositeEventSink`.
4. Default to a silent null sink.
5. Emit common lifecycle events:
   - optimization started;
   - initial candidate evaluated;
   - candidate proposed/evaluated;
   - candidate accepted/rejected;
   - best candidate updated;
   - generation started/completed;
   - population reseeded;
   - optimization terminated/failed.
6. Populate event fields from authoritative typed results and manipulation metadata.
7. Keep direct low-level module logs only where they remain operationally appropriate.
8. Define sink-failure policy.

### Tests

- Null sink inertness
- Composite ordering
- Logging level mapping
- Schema version and timestamp
- Common fields across MC and GA
- No arrays or full structures in events
- Numerical behavior unchanged

### Non-goals

- No durable JSONL
- No checkpoint events until checkpoint integration exists

### Exit criteria

- MC and GA share a stable event vocabulary.
- Library code remains silent unless configured.

---

## OBS3 — Durable JSONL journal and run manifest

### Objective

Add opt-in scientific run provenance without conflating it with restart state.

### Prerequisites

OBS2.

### Primary files

- `GBOpt/optimization/events.py` or a focused journal module
- Application/example configuration
- Journal tests and documentation

### Required changes

1. Add versioned `JsonlEventSink`.
2. Add separate run-manifest writer for invariant metadata.
3. Define UTF-8 encoding, one-object-per-line format, flush policy, append/overwrite policy, and write-failure behavior.
4. Reference large structures through artifacts rather than embedding arrays.
5. Make the optimization driver the single authoritative journal writer.
6. Avoid shared multi-writer JSONL files for scheduler workers.
7. Document that the journal is not a checkpoint.
8. Add optional example/campaign configuration without changing default library behavior.

### Tests

- Valid JSON object per line
- Manifest/journal run-ID consistency
- No large arrays
- Append/overwrite policy
- Truncated last record handling if a reader is included
- Write-failure policy
- Single-writer assumptions

### Non-goals

- No restart logic
- No OpenTelemetry or third-party dependency

### Exit criteria

- Users can opt into durable, queryable provenance independently of checkpointing.

---

# 11. Checkpoint and restart track

The old archive contains a meaningful checkpoint implementation with valuable tested behavior. The new implementation should preserve those behavioral capabilities while replacing raw dictionary and filename coupling with typed, versioned snapshots and ownership-aware artifacts.

## CP0 — Legacy checkpoint behavior audit and characterization

### Objective

Turn the older implementation into an explicit behavioral specification before porting code.

### Prerequisites

F0.

### Primary files

- New checkpoint design/audit document
- Ported or adapted reference tests
- No current production checkpoint code

### Required changes

Document the old system’s behavior, including:

- optional null-object checkpointing;
- JSON and pickle formats;
- interval saves and final saves;
- schema envelope;
- RNG restoration;
- MC current accepted structure, temperature, rejection count, energy history, accepted indices, operation history, run ID, minimum steps, and cooldown restoration;
- MC extension after completion;
- GA generation-boundary snapshots;
- per-candidate intra-generation result cache;
- skipped reevaluation of completed candidates;
- pending next-generation structure files;
- cleanup ordering and missing-artifact failure;
- resumed-versus-continuous equivalence tests.

Classify each behavior as:

1. preserve;
2. redesign but preserve externally;
3. intentionally reject;
4. optional compatibility.

### Tests

- Reference tests may be copied into a non-executing specification area or adapted as expected-failure tests until implementation exists.

### Non-goals

- No production checkpoint code
- No promise of legacy pickle compatibility

### Exit criteria

- The checkpoint behavioral contract is explicit and reviewed.
- The new schema’s required state is enumerated.

---

## CP1 — Versioned checkpoint store and JSON codec

### Objective

Port the reusable persistence foundation independently of optimizers.

### Prerequisites

CP0.

### Primary files

- New `GBOpt/checkpoint/` package
- Persistence unit tests

### Required changes

1. Add checkpoint-specific error hierarchy.
2. Add a versioned envelope and strict schema-header validation.
3. Add `CheckpointStore` and disabled/null implementation.
4. Support configurable intervals and unconditional final saves.
5. Use JSON as the initial supported format.
6. Serialize approved NumPy arrays/scalars and paths explicitly.
7. Write to a temporary file, flush, optionally `fsync`, and atomically replace the destination.
8. Create parent directories only under a documented policy.
9. Remove temporary files after failed writes.
10. Define load behavior for absent, corrupted, unsupported-version, and wrong-minimizer files.
11. Omit pickle initially unless a concrete compatibility requirement is approved.

### Tests

- Disabled no-op behavior
- Interval behavior
- Lazy state function not called when not due
- Final save bypasses interval
- JSON NumPy/path conversion
- Atomic replacement
- Corrupted file
- Unsupported version
- Delete and existence behavior

### Non-goals

- No optimizer snapshots
- No candidate result cache
- No events

### Exit criteria

- Checkpoint persistence is independently testable and has no optimizer imports.

---

## CP2 — Typed MC, GA, population, and candidate-evaluation snapshots

### Objective

Define what can be safely restarted after the modern refactors.

### Prerequisites

CP1, F2, IO5, EVAL2, and MAN5.

### Primary files

- `GBOpt/checkpoint/model.py`
- `GBOpt/checkpoint/codec.py`
- Snapshot contract tests

### Required changes

1. Add immutable, versioned snapshot models:
   - `MonteCarloSnapshot`;
   - `GeneticAlgorithmSnapshot`;
   - `PopulationCandidateSnapshot`;
   - `CandidateEvaluationSnapshot`;
   - artifact and candidate references as needed.
2. Define completed safe boundaries:
   - MC after a completed accepted/rejected step;
   - GA after a completed generation;
   - candidate cache after an individual evaluation result is durably recorded.
3. Store RNG bit-generator type and state.
4. Store `RunContext`-compatible run identity without requiring the event system.
5. Store candidates through validated artifacts plus persistent interface state, or through an explicitly approved embedded representation.
6. Store operation/lineage information using structured metadata, not prose strings.
7. Store evaluation status and failure details, not only energy and dump path.
8. Prohibit serialization of callbacks, sinks, loggers, open files, scheduler clients, and live manipulator objects.
9. Define which run parameters are immutable, restored, or overrideable on resume.
10. Add semantic schema validation beyond JSON parsing.

### Tests

- Snapshot round trips
- Missing required fields
- Wrong algorithm/minimizer
- Unsupported schema version
- RNG state fidelity
- Artifact and ownership reference validation
- Prohibited arbitrary-object rejection

### Non-goals

- No MC or GA loop integration
- No legacy migration reader

### Exit criteria

- Restart state is explicit, typed, and independent of live Python objects.

---

## CP3 — Monte Carlo checkpoint and resume integration

### Objective

Restore the old MC checkpoint capabilities using modern candidate, evaluation, operation, and loader contracts.

### Prerequisites

CP2, EVAL2, MAN5, and IO5.

### Primary files

- `GBOpt/optimization/monte_carlo.py`
- `GBOpt/optimization/checkpointing.py`
- MC checkpoint tests

### Required changes

1. Add optional checkpoint configuration to `run_MC()` without changing default behavior.
2. Resume the accepted `InterfaceCandidate` through `CandidateLoader`.
3. Restore:
   - RNG state;
   - run ID;
   - resolved seed;
   - completed step;
   - temperature;
   - rejection count;
   - current and best energies;
   - current and best candidate artifacts;
   - accepted-index history;
   - operation/acceptance history;
   - minimum-step and cooldown settings according to approved policy.
4. Allow `max_steps` extension after normal completion.
5. Define which current-call convergence controls may override saved values.
6. Save only after a completed step or normal/early termination.
7. Keep the checkpoint after completion so extension remains possible.
8. Fail loudly if required candidate artifacts are missing or invalid.
9. Do not rely on event journaling for restart.

### Tests

- No file when checkpointing disabled
- Valid checkpoint retained on completion
- Interrupted resume
- Continuous-versus-resumed deterministic equivalence
- Extension after completion
- Restored run ID, RNG, cooldown, and minimum steps
- Corrupted checkpoint
- Missing current candidate artifact
- Interval behavior

### Non-goals

- No GA checkpointing
- No pickle
- No checkpoint events unless OBS2 is already available and integration remains optional

### Exit criteria

- MC resume is deterministic at completed-step boundaries.
- Checkpointing remains opt-in and default behavior is unchanged.

---

## CP4 — GA generation-boundary checkpoint and resume

### Objective

Restore robust generation-level GA restart before adding intra-generation recovery.

### Prerequisites

CP3, EVAL2, MAN5, and IO5.

### Primary files

- `GBOpt/optimization/genetic.py`
- `GBOpt/optimization/checkpointing.py`
- GA checkpoint tests

### Required changes

1. Save after a completed generation.
2. Store and restore:
   - RNG state;
   - run ID and resolved seed;
   - completed generation;
   - best result and artifact;
   - energy history;
   - structured population lineage;
   - ordered next-generation population candidate artifacts and interface state;
   - GA configuration needed for deterministic continuation.
3. Publish next-generation candidate artifacts before publishing the snapshot that references them.
4. Remove stale pending artifacts only after the new snapshot is durable.
5. Resume through `CandidateLoader`.
6. Keep checkpoint and required candidate artifacts after completion for run extension.
7. Fail loudly on missing or invalid population artifacts.
8. Preserve all-invalid-generation reseeding behavior.

### Tests

- Generation-boundary interrupted resume
- Continuous-versus-resumed equivalence
- History not duplicated
- Energy arrays not duplicated
- Completion followed by generation extension
- Missing population artifact failure
- Atomic pending-artifact rollover
- Scalar and batch evaluation compatibility

### Non-goals

- No per-candidate recovery within a generation
- No legacy schema reader

### Exit criteria

- GA resumes deterministically from completed generation boundaries.

---

## CP5 — GA intra-generation candidate recovery

### Objective

Restore the old implementation’s ability to skip already-completed candidate evaluations after interruption.

### Prerequisites

CP4 and EVAL2.

### Primary files

- `GBOpt/optimization/genetic.py`
- `GBOpt/checkpoint/candidate_cache.py`
- Candidate-cache tests

### Required changes

1. Add a per-generation candidate result cache keyed by stable candidate IDs.
2. Store full `CandidateEvaluationSnapshot` data:
   - status;
   - physical and selection energy;
   - artifact;
   - failure stage/code/message;
   - input index;
   - candidate identity.
3. Atomically record each completed candidate result.
4. Skip already-completed candidates on resume.
5. Preserve ordered generation result alignment.
6. Support scalar evaluators directly.
7. For batch evaluators:
   - support batch-return granularity automatically;
   - permit finer-grained recording only through an explicit callback/protocol;
   - do not inspect function signatures to inject undocumented semantics if a clearer adapter can be used.
8. Delete the transient candidate cache only after the authoritative generation checkpoint is durable.
9. Clean orphaned/stale cache files deterministically.
10. Preserve failures as typed results rather than converting them only to penalty energy.

### Tests

- Skip completed scalar candidates
- Batch-return recovery
- Fine-grained batch recovery adapter
- Ordered result reconstruction
- Failure result restoration
- Cache deletion after completed generation
- Orphan cleanup
- No cache when checkpointing disabled
- Continuous-versus-resumed candidate reconstruction equivalence

### Non-goals

- No distributed shared-file multi-writer cache
- No arbitrary evaluator-state serialization

### Exit criteria

- An interrupted GA generation resumes without repeating completed evaluations.
- Result provenance and ownership are retained.

---

# 12. Final integration PR

## INT1 — Cross-track architecture hardening, documentation, and release gate

### Objective

Verify that the independently developed tracks compose cleanly and remove only the compatibility scaffolding that is safe to remove in the current release.

### Prerequisites

GM8, IO5, MAN5, OBS3, and CP5.

### Primary files

- Architecture tests
- Public documentation and changelog
- Compatibility modules
- Examples

### Required changes

1. Add dependency-direction tests:
   - `GBOpt.gbmaker` does not import I/O or optimization;
   - `GBOpt.io` does not import `GBMaker` or minimizers;
   - manipulation operations do not perform I/O or evaluation;
   - checkpoint persistence does not import live optimizer classes;
   - events do not own restart state.
2. Verify compatibility import identity for `GBMaker`, `GBManipulator`, and minimizers.
3. Verify the same `CandidateLoader` is used by evaluator returns and checkpoint resumes.
4. Verify `WriteResult` mappings remain candidate-local.
5. Verify journal files are not accepted as checkpoints.
6. Add end-to-end tests covering:
   - build → manipulate → write → evaluate/reload → optimize;
   - interrupted/resumed MC;
   - interrupted/resumed GA;
   - logging plus journal plus checkpoint enabled simultaneously;
   - periodic and single-interface topologies.
7. Update examples to configure logging, journals, and checkpoints at the application layer.
8. Document extension points for readers, writers, manipulations, evaluators, event sinks, and checkpoint stores.
9. Document compatibility shims and their future removal conditions.
10. Produce a final verified source archive and SHA-256 handoff.

### Tests

- Focused tests for every track
- Full non-slow suite
- Approved slow/integration suite
- Deterministic uninterrupted-versus-resumed workflows
- Import/dependency architecture tests
- Documentation/static checks required by the project

### Non-goals

- No new format adapters beyond those already approved
- No plugin entry-point discovery
- No immutable public GBMaker redesign
- No OpenTelemetry or third-party logging framework

### Exit criteria

- All tracks compose without reverse dependencies.
- Every accepted behavioral invariant is tested.
- The repository is ready for release or for subsequent independent feature development.

---

## 13. Deferred or optional follow-on PRs

These should not be included in the core roadmap unless a concrete user requirement exists.

### OPT-IO1 — Versioned ownership sidecars and explicit lossy writes

Add `.gbopt.json` sidecars, content binding, transactional paired-file publication, and `allow_lossy` policy.

### OPT-IO2 — XYZ and CIF adapters

Add XYZ first to exercise explicit loss handling, followed by CIF after general cell semantics are stable.

### OPT-PLUGIN1 — Third-party entry-point discovery

Add package entry points for readers, writers, and manipulations only after built-in registries and contracts have stabilized.

### OPT-CP1 — Legacy checkpoint migration reader

Add a one-way schema-v1 migration only if real unmerged checkpoint files must be preserved. Do not add unsafe general pickle loading.

### OPT-OBS1 — Checkpoint lifecycle events

After checkpointing and events both exist, add `checkpoint_saved`, `checkpoint_loaded`, and `checkpoint_failed` as reports of checkpoint actions. The events remain non-authoritative.

### OPT-API1 — Public immutable construction API

Expose `GBBuildConfig`, `BicrystalResult`, and `build_bicrystal` as supported public APIs and begin a separately approved deprecation of mutable facade behavior.

---

## 14. Required PR template

Every implementation PR in this roadmap should include the following in its description:

### Scope

- Exact behavior or seam changed
- Owning architectural layer
- Explicit non-goals

### Prerequisites

- Required merged PR or branch
- Incoming source archive and SHA-256

### Compatibility

- Public imports preserved
- Warnings/exceptions preserved or intentionally changed
- Serialization/schema changes

### Invariants

- Mathematical and physical invariants
- Ownership and identity invariants
- Determinism requirements

### Files

- Files added
- Files moved
- Files modified
- Files removed

### Tests

- Focused commands and exact results
- Integration commands and exact results
- Non-slow suite result
- Slow tests run or explicitly deferred

### Risks and deferred work

- Known migration shims
- Remaining consumers of old paths
- Follow-up PR required

### Handoff

- Complete source archive
- SHA-256 file
- Closeout note

---

## 15. General testing gates

### Per-PR minimum

1. Compile/import check.
2. Focused tests for changed modules.
3. Relevant integration tests.
4. Full non-slow suite when the environment permits.
5. Exact report of commands and results.

### Major integration gates

Run required slow tests at minimum after:

- GM6 and GM7;
- IO5;
- MAN5;
- EVAL2;
- CP3, CP4, and CP5;
- INT1.

### Determinism gates

For refactors and checkpoint work, compare:

- order-sensitive arrays;
- order-insensitive physical structure equivalence;
- exact metadata;
- warning and exception behavior;
- serialized files;
- RNG-dependent optimizer histories;
- uninterrupted versus resumed results.

Do not widen tolerances or accept order-insensitive matches merely to conceal an unexplained behavior change.

---

## 16. Merge and branch strategy

1. Create each PR from the latest accepted `main` or from its single declared prerequisite.
2. Do not develop a PR from a long-lived integration branch containing unrelated unmerged work.
3. Rebase at each ownership-window boundary:
   - after F2 before I/O/manipulation work;
   - after IO3 before deep GBMaker facade work;
   - after IO4 before manipulation facade extraction;
   - after F1 before any minimizer-facing work.
4. Use compatibility re-exports rather than parallel duplicate implementations.
5. Keep integration PRs small by moving shared types and services into foundation PRs first.
6. Merge behavior-preserving extraction before behavior changes.
7. Never mix checkpoint schema design with logging/journal schema design.
8. Use one source archive and integrity record for every handoff.

---

## 17. Recommended immediate starting sequence

The most effective next actions are:

1. Implement **F0**.
2. Start **F1**, **F2**, **GM1**, and **CP0** in parallel.
3. Merge **F2**, then begin **IO1**.
4. Merge **F1**, then begin **OBS1**.
5. Complete **IO3** early, before GBMaker facade migration becomes extensive.
6. Complete **IO4** before beginning MAN1 changes to `GBManipulator.py`.
7. Build and prove the manipulation abstraction through MAN1–MAN4.
8. Integrate `OperationSpec` in MAN5.
9. Normalize evaluation through EVAL1–EVAL2.
10. Add events and journals through OBS2–OBS3.
11. Complete typed checkpoint schemas and optimizer integration through CP2–CP5.
12. Close with INT1.

---

## 18. Final decision summary

The recommended route is not one large refactor and not a set of nominally independent PRs that repeatedly edit the same monoliths. It is a controlled decomposition:

- freeze behavior first;
- split the minimizer mechanically;
- extract shared interface state;
- give I/O exclusive ownership of file syntax and transient mappings;
- give GBMaker exclusive ownership of construction;
- make manipulations operation objects over complete interface candidates;
- make evaluations typed before they are logged or checkpointed;
- use events for observation and snapshots for restart;
- preserve the old checkpoint system’s tested restart behavior while replacing its filename- and dictionary-coupled schema;
- use compatibility facades until all callers have migrated;
- finish with one small, explicit integration gate.

This sequence allows the GBMaker pipeline, I/O core, narrow logging work, and checkpoint persistence foundation to progress concurrently while keeping the high-conflict integration work ordered and reviewable.
