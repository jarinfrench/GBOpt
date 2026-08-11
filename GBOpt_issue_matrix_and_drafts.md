# GBOpt Refactoring Roadmap — GitHub Issue Matrix and Draft Issue Text

**Repository:** [IdahoLabResearch/GBOpt](https://github.com/IdahoLabResearch/GBOpt)  
**Prepared:** 2026-08-10  
**Source of implementation scope:** `MASTER_PLAN.md` / GBOpt Refactoring and Checkpointing Pull-Request Roadmap  
**Issue-template basis:** `.github/ISSUE_TEMPLATE/feature_request.md`

## Purpose

This document translates the master refactoring roadmap into a GitHub issue plan. The core recommendation is one independently closable implementation issue per roadmap PR, while existing broad behavioral/capability issues remain umbrella or reference issues rather than being duplicated.

The issue bodies below follow the repository's feature-request structure and add a final **Dependencies and related issues** section so each issue can stand alone while remaining connected to the implementation graph.

## Tracker strategy

1. Create **33 core roadmap issues**, corresponding one-to-one with F0–F2, GM1–GM8, IO1–IO5, MAN1–MAN5, OBS1/EVAL1/EVAL2/OBS2/OBS3, CP0–CP5, and INT1.
2. Keep existing issue **#59** as a cross-cutting behavioral/capstone reference. Smaller roadmap issues should say which part of #59 they implement/refactor.
3. Convert or edit existing issue **#36** into the checkpoint umbrella. The new CP0–CP5 issues should be linked as its implementation decomposition.
4. Reconcile **#27** before MAN3 merges. #27's current public soft-mode API specifies a single returned structure and `mode_index`; MAN3 must not silently reintroduce contradictory legacy behavior.
5. Keep the existing boundary-spec/exact-construction series (#42, #45–#49 and related issues) as behavioral/public-API context for the GBMaker refactor rather than reopening the same capability under new titles.
6. Do not open the six optional follow-on issues until a concrete requirement exists; copy-ready drafts are included in an appendix for later use.

## Important current-issue decisions

### #36 — checkpoint umbrella needs an update

The original #36 asks for JSON **and pickle**, and for successful runs to delete checkpoint files. The master roadmap intentionally changes that design:

- JSON is the initial supported format; pickle is deferred unless a real compatibility requirement appears.
- Completed checkpoints and required candidate artifacts are retained so a completed run can be extended.
- Persistence, typed snapshot schemas, MC integration, GA generation integration, and intra-generation recovery are split into separate issues/PRs.

The recommended action is to edit #36 so it describes the capability goal and links CP0–CP5, rather than treating its original implementation checklist as the final specification.

### #27 — MAN3 must preserve the already-specified public soft-mode behavior

#27 currently specifies:

- `displace_along_soft_modes()` returns a single `np.ndarray`;
- `num_children` is removed;
- `mode_index` selects the requested soft mode.

The generic manipulation layer may internally normalize operation results as a tuple of children, but MAN3 should not reverse the public API specified in #27 without a deliberate issue decision.

### #59 — keep as a behavioral capstone/reference

#59 already captures the invariant that a complete exact interface, its ownership, GB plane, physical grain bounds, periodicity, and topology remain intact through manipulation, LAMMPS round trips, and optimization. F2, GM6–GM7, IO4–IO5, MAN2, EVAL2, and related integration work should reference #59 rather than duplicate it.

---

# Core issue matrix

| Roadmap | Proposed issue title | Tracker action | Prerequisite(s) | Existing issue references |
|---|---|---|---|---|

| F0 | [Feature] Establish deterministic characterization baseline and architecture decisions | Create new | None | #40 (related testing umbrella) |
| F1 | [Feature] Decompose GBMinimizer into internal optimization modules with compatibility facade | Create new | F0 | None identified |
| F2 | [Feature] Extract shared interface-state domain model | Create new; reference umbrella | F0 | #59 |
| GM1 | [Feature] Establish internal GBMaker pipeline package and construction-state contracts | Create new | F0 | #45 (conceptually related typed scaffolding) |
| GM2 | [Feature] Separate GBMaker input normalization and material resolution | Create new; reference existing boundary-spec series | GM1 | #42, #45, #48 |
| GM3 | [Feature] Extract GBMaker orientation and periodicity resolution | Create new; reference existing boundary-spec series | GM2 | #42, #46, #49 |
| GM4 | [Feature] Extract GBMaker commensurability and dimension planning | Create new | GM3 | #46, #47 |
| GM5 | [Feature] Extract shared GBMaker geometry kernels | Create new | GM4 | #47, #59 (behavioral invariants) |
| GM6 | [Feature] Separate exact and approximate grain builders behind common contracts | Create new; reference umbrella | GM5 | #46, #47, #59 |
| GM7 | [Feature] Add pure bicrystal assembly and end-to-end GBMaker construction pipeline | Create new; reference umbrella | GM6 | #47, #59 |
| GM8 | [Feature] Migrate GBMaker facade state to the staged construction pipeline | Create new; reference existing API issue | GM7 and IO3 if not already merged | #48 |
| IO1 | [Feature] Add canonical StructureData model, I/O contracts, and LAMMPS data reader | Create new | F2 | #59 (round-trip invariants) |
| IO2 | [Feature] Add LAMMPS dump reader with explicit frame and coordinate semantics | Create new | IO1 | #59 |
| IO3 | [Feature] Extract LAMMPS writer and introduce WriteResult | Create new | IO1 | #59 (transient-ID semantics) |
| IO4 | [Feature] Add Parent.from_structure and remove file parsing from Parent | Create new; reference umbrella | IO2 and F2 | #59 |
| IO5 | [Feature] Add centralized ownership-aware candidate loader and validated round trips | Create new; reference umbrella | IO3, IO4, F2 | #59 |
| MAN1 | [Feature] Add manipulation operation protocol, registry, and GBManipulator facade seam | Create new | F2 and IO4 | #59 (domain behavior) |
| MAN2 | [Feature] Extract translation, termination, and interface-separation operations | Create new; reference umbrella | MAN1 | #59 |
| MAN3 | [Feature] Extract density and soft-mode manipulation operations | Create new; reconcile with existing issue before implementation | MAN2 | #27 |
| MAN4 | [Feature] Extract binary slice-and-merge operation with explicit compatibility validation | Create new | MAN3 | #59 (ownership propagation) |
| MAN5 | [Feature] Integrate OperationSpec-based manipulation dispatch into MC and GA | Create new | F1 and MAN4 | #59 (GA ownership/lineage context) |
| OBS1 | [Feature] Improve optimizer reproducibility and standard-library logging | Create new | F1 | None identified |
| EVAL1 | [Feature] Introduce algorithm-neutral evaluation and structure-artifact contracts | Create new | F1 and IO1 | #59 (evaluation failure provenance) |
| EVAL2 | [Feature] Normalize Monte Carlo and genetic-algorithm evaluation flows | Create new; reference umbrella | EVAL1 and IO5; MAN5 recommended | #59 |
| OBS2 | [Feature] Add RunContext and typed optimization event protocol | Create new | EVAL2 and MAN5 | None identified |
| OBS3 | [Feature] Add durable JSONL optimization journal and run manifest | Create new | OBS2 | None identified |
| CP0 | [Feature] Characterize and specify legacy checkpoint/restart behavior | Create child issue; update #36 as umbrella | F0 | #36 |
| CP1 | [Feature] Add versioned atomic JSON checkpoint store and codec | Create child issue; update #36 as umbrella | CP0 | #36 |
| CP2 | [Feature] Define typed restart snapshots for MC, GA, populations, and evaluations | Create child issue; update #36 as umbrella | CP1, F2, IO5, EVAL2, MAN5 | #36, #59 (candidate ownership semantics) |
| CP3 | [Feature] Add deterministic Monte Carlo checkpoint and resume support | Create child issue; update #36 as umbrella | CP2, EVAL2, MAN5, IO5 | #36 |
| CP4 | [Feature] Add generation-boundary genetic-algorithm checkpoint and resume support | Create child issue; update #36 as umbrella | CP3, EVAL2, MAN5, IO5 | #36 |
| CP5 | [Feature] Recover completed GA candidate evaluations within interrupted generations | Create child issue; update #36 as umbrella | CP4 and EVAL2 | #36 |
| INT1 | [Feature] Complete cross-track architecture hardening and refactor release gate | Create new integration issue | GM8, IO5, MAN5, OBS3, CP5 | #40, #57, #59 |

## Dependency summary

```text
F0
├─ F1 ────────────────────────────────┐
├─ F2 ── IO1 ── IO2 ── IO4 ── IO5 ──┼───────────────┐
│       └──────── IO3 ────────────────┘               │
├─ GM1 → GM2 → GM3 → GM4 → GM5 → GM6 → GM7 → GM8    │
└─ CP0 → CP1                                          │
                                                     │
F1 → OBS1                                             │
F1 + IO1 → EVAL1 → EVAL2 ── OBS2 → OBS3              │
F2 + IO4 → MAN1 → MAN2 → MAN3 → MAN4                 │
F1 + MAN4 → MAN5 ────────────────────────────────────┤
CP1 + F2 + IO5 + EVAL2 + MAN5 → CP2 → CP3 → CP4 → CP5
                                                     │
GM8 + IO5 + MAN5 + OBS3 + CP5 ──────────────────────→ INT1
```

The diagram is intentionally a dependency summary rather than a branch ancestry prescription. Each implementation PR should still be created from its declared prerequisite or the latest accepted integration base, consistent with the master plan.

---

# Copy-ready core issue drafts


## F0 — [Feature] Establish deterministic characterization baseline and architecture decisions

**Recommended tracker action:** Create new  
**Roadmap prerequisite:** None  
**Existing issue references:** #40 (related testing umbrella)


## What would you like GBOpt to do?

GBOpt should establish a deterministic characterization baseline and explicit architecture decisions before the refactoring roadmap changes production behavior.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [ ] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** The planned refactoring moves scientific construction, manipulation, evaluation, I/O, observability, and checkpointing across module boundaries. Without an accepted behavioral baseline, later PRs cannot distinguish an intended architectural move from an accidental scientific, numerical, warning, exception, ordering, or serialization change.

## Proposed behavior

Add characterization tests and architecture records that freeze the current accepted behavior of representative GBMaker construction, interface manipulation, scalar/batch evaluation, fixed-seed MC/GA execution, and existing ownership-aware reload behavior. Record both order-sensitive and order-insensitive structure fingerprints where appropriate. Import useful behavioral tests from the older checkpoint-enabled source as reference or expected-failure tests when the current implementation lacks the feature.

Document the authoritative domain contracts (`StructureData`, `GrainOwnership`, `InterfaceCandidate`, `WriteResult`/`StructureArtifact`, `ManipulationResult`, `EvaluationResult`, `RunContext`, `OptimizationSnapshot`, and `OptimizationEvent`) and the ownership boundaries between construction, I/O, manipulation, evaluation, observability, and checkpointing.

## Acceptance criteria

- [ ] Representative legacy, exact P/Q, exact CSL, exact fluorite/UO2, approximate non-CSL, mismatch-accommodation, periodic-bicrystal, slab, orthogonal-output, and triclinic-output construction cases have deterministic characterization coverage.
- [ ] Current `InterfaceCandidate`, grain-local cycling, slab cycling, translation, and interface-separation behavior is characterized.
- [ ] Scalar and batch GA evaluation, including failure penalties and ownership-aware reloads, is characterized.
- [ ] Fixed-seed MC and GA fixtures are added without changing optimizer policy.
- [ ] Two consecutive characterization runs produce identical manifests.
- [ ] Architecture decisions explicitly define the shared contracts and ownership boundaries required by later roadmap issues.
- [ ] The current source is recorded as the implementation baseline; the older checkpoint-enabled source is recorded as behavioral/design reference only.
- [ ] No production behavior changes are introduced beyond test-only hooks that do not alter results.

## API / breaking-change impact

- [x] No breaking changes expected. This issue adds tests and architecture documentation only.

## Alternatives considered

Starting with code movement was rejected because later failures would be difficult to classify as pre-existing behavior, intended change, or regression.

## Dependencies and related issues

Roadmap root issue. Related to #40, but narrower and more deterministic than general example testing.


---


## F1 — [Feature] Decompose GBMinimizer into internal optimization modules with compatibility facade

**Recommended tracker action:** Create new  
**Roadmap prerequisite:** F0  
**Existing issue references:** None identified


## What would you like GBOpt to do?

GBOpt should split the monolithic `GBMinimizer.py` into focused internal optimization modules while preserving existing public imports and behavior.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [x] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Checkpointing, evaluation normalization, manipulation integration, and observability all need to modify optimizer code. Keeping MC, GA, mutation, and evaluation in one monolith creates unnecessary merge conflicts and makes later changes harder to review independently.

## Proposed behavior

Create `GBOpt/optimization/` with separate modules for mutation, evaluation, Monte Carlo, genetic algorithms, and errors only where exact exception identity can be preserved. Move the existing implementations mechanically, then make `GBOpt/GBMinimizer.py` a compatibility facade that re-exports the same public class and exception objects.

This issue is extraction-only: constructor/method signatures, defaults, warnings, exception objects, numerical behavior, and import paths remain compatible.

## Acceptance criteria

- [ ] `Mutator`, current `CandidateEvaluation`, `MonteCarloMinimizer`, and `GeneticAlgorithmMinimizer` have canonical implementations under `GBOpt/optimization/`.
- [ ] `GBOpt.GBMinimizer` and `GBOpt.__init__` continue to expose the established public objects.
- [ ] Compatibility tests prove old and new import paths resolve to the same class objects.
- [ ] Constructor signatures, method signatures, defaults, warnings, and exception identity are unchanged.
- [ ] Reasonable pickle/import resolution for existing public paths is preserved where supported.
- [ ] Current MC and GA tests and the full non-slow suite pass without policy changes.

## API / breaking-change impact

- [x] No breaking changes expected; old imports remain supported through re-exports.

## Alternatives considered

Leaving the monolith intact was rejected because multiple later workstreams would repeatedly edit the same high-conflict file.

## Dependencies and related issues

Depends on F0. This should merge before optimizer-facing observability, evaluation, manipulation-dispatch, or checkpoint integration work.


---


## F2 — [Feature] Extract shared interface-state domain model

**Recommended tracker action:** Create new; reference umbrella  
**Roadmap prerequisite:** F0  
**Existing issue references:** #59


## What would you like GBOpt to do?

GBOpt should move persistent interface-domain state into a neutral package that can be shared by manipulation, I/O, evaluation, and checkpointing without reverse imports.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [x] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [ ] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [x] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** `InterfaceCandidate` and `GrainOwnership` represent persistent scientific/domain state, but they are currently owned by implementation modules that later tracks also need to refactor. Keeping them under `GBManipulator.py` or `FileGrainOwnership.py` would force I/O, evaluation, and checkpoint code to depend on those facades.

## Proposed behavior

Create `GBOpt/interface/` and move the authoritative immutable `InterfaceCandidate` and `GrainOwnership` types there without changing validation or immutability semantics. Keep `BoundaryNormalTopology` in its neutral topology module unless a separate move is justified. Preserve old imports by re-exporting from the legacy modules.

Serialization-local concepts such as transient file mappings stay out of the domain package.

## Acceptance criteria

- [ ] `InterfaceCandidate` and `GrainOwnership` have canonical definitions in a neutral interface-domain package.
- [ ] Old imports from `GBManipulator.py` and `FileGrainOwnership.py` continue to work and resolve to the same objects.
- [ ] `CandidateFileMapping`, parsers, and reload services are not moved into the domain model.
- [ ] Arrays remain defensively copied and read-only.
- [ ] Validation, equality, topology semantics, coordinate tolerance, and interface-separation state are preserved.
- [ ] I/O, manipulation, evaluation, and future checkpoint modules can import interface state without importing the legacy facades.

## API / breaking-change impact

- [x] No breaking changes expected; compatibility re-exports preserve established imports.

## Alternatives considered

Duplicating lightweight interface-state types in each subsystem was rejected because ownership and topology would drift across serialization, manipulation, and restart paths.

## Dependencies and related issues

Depends on F0. Implements the domain-model portion of the broader invariant tracked by #59.


---


## GM1 — [Feature] Establish internal GBMaker pipeline package and construction-state contracts

**Recommended tracker action:** Create new  
**Roadmap prerequisite:** F0  
**Existing issue references:** #45 (conceptually related typed scaffolding)


## What would you like GBOpt to do?

GBOpt should establish an internal `GBOpt.gbmaker` package with typed construction-state boundaries and one canonical supercell implementation.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [x] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [ ] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** The GBMaker refactor needs stable intermediate contracts before scientific orchestration is moved. Explicit construction states make each stage testable and prevent the facade from becoming a second source of truth during migration.

## Proposed behavior

Add typed construction dataclasses for configuration, resolved boundary input, material state, orientation state, axis accommodation, dimension planning, grain-build requests/results, and bicrystal results. Define units, array shapes, ownership, and immutability for each.

Move the stateless supercell implementation to `GBOpt/gbmaker/builders/supercell.py` and replace the old module implementation with a compatibility re-export. Do not move scientific orchestration yet.

## Acceptance criteria

- [ ] The internal `GBOpt.gbmaker` package exists with validated typed construction-state contracts.
- [ ] Array ownership, units, shapes, and immutability expectations are documented and tested.
- [ ] There is exactly one canonical supercell implementation.
- [ ] Legacy supercell imports continue to resolve to the canonical implementation.
- [ ] Construction output and baseline hashes remain unchanged.
- [ ] Scientific orchestration, input normalization, orientation logic, and LAMMPS writer logic remain in their pre-existing owners for now.

## API / breaking-change impact

- [x] No breaking changes expected; this is internal scaffolding plus compatibility re-exports.

## Alternatives considered

Moving orchestration and defining types simultaneously was rejected because it would combine extraction with behavioral risk and make review substantially harder.

## Dependencies and related issues

Depends on F0. Typed construction state is separate from the public boundary-spec types tracked by the existing boundary-spec issue series.


---


## GM2 — [Feature] Separate GBMaker input normalization and material resolution

**Recommended tracker action:** Create new; reference existing boundary-spec series  
**Roadmap prerequisite:** GM1  
**Existing issue references:** #42, #45, #48


## What would you like GBOpt to do?

GBOpt should isolate user-facing construction input normalization and material initialization from mutable `GBMaker` orchestration.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [x] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [ ] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [ ] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Constructor validation, legacy/boundary-spec adaptation, and unit-cell/material setup are currently intertwined with later geometric stages. This makes input behavior hard to test independently and encourages partially initialized facade state.

## Proposed behavior

Extract pure validation and normalization functions into `gbmaker/config.py`, `gbmaker/inputs.py`, and `gbmaker/material.py`. Normalize constructor inputs into `GBBuildConfig`, resolve legacy and `BoundarySpec` inputs into `ResolvedBoundaryInput`, and build material/unit-cell information into `MaterialState`.

Keep public constructors as adapters and preserve the existing legacy-constructor warning semantics and public exception translation.

## Acceptance criteria

- [ ] Scalar, sequence, mode, and boundary-input validation is directly testable without constructing a partial `GBMaker` object.
- [ ] Legacy and `BoundarySpec` inputs normalize to an explicit `ResolvedBoundaryInput`.
- [ ] Unit-cell and derived material state is produced by an isolated material stage.
- [ ] Legacy-constructor warning count, category, message fragment, and stack level are preserved according to the accepted baseline.
- [ ] Internal failures are translated to established public exception types.
- [ ] Full construction characterization remains equivalent.
- [ ] No duplicate migrated validation remains in the facade.

## API / breaking-change impact

- [x] No new breaking change is introduced by this refactor; existing deprecation behavior from the boundary-spec series is preserved rather than expanded.

## Alternatives considered

Adding more conditional branches directly to `GBMaker.__init__` was rejected because it would further couple public syntax to scientific construction stages.

## Dependencies and related issues

Depends on GM1. Preserve the user-facing behavior established by #42/#45/#48; this issue is architectural decomposition, not a replacement for those feature issues.


---


## GM3 — [Feature] Extract GBMaker orientation and periodicity resolution

**Recommended tracker action:** Create new; reference existing boundary-spec series  
**Roadmap prerequisite:** GM2  
**Existing issue references:** #42, #46, #49


## What would you like GBOpt to do?

GBOpt should compute orientation and in-plane periodicity as a pure construction stage that returns an explicit `OrientationState`.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [x] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [ ] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [ ] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Orientation selection currently relies on hidden mutable `GBMaker` state. Exact P/Q inputs, exactified five-DOF inputs, and approximate inputs all need predictable orientation semantics without float round trips or facade mutation.

## Proposed behavior

Move Miller-row reduction, row-angle calculations, approximate rotation-row selection, exact embedding-derived orientation selection, and x/in-plane periodicity calculations into `gbmaker/orientation.py`. Implement a pure `resolve_orientation(...) -> OrientationState`.

Preserve exact P/Q integer rows without converting them through floating-point approximations. The facade may temporarily mirror state required by unextracted downstream code, but orientation computation itself must not mutate `GBMaker`.

## Acceptance criteria

- [ ] Exact integer-row and approximate orientation paths have direct stage tests.
- [ ] Exact P/Q rows are preserved without float round trips.
- [ ] Left/right rotation results, primitive periods, and `inplane_periodic` flags match accepted behavior.
- [ ] Threshold and failure behavior is preserved.
- [ ] Baseline atom and dimension fingerprints remain equivalent.
- [ ] Orientation computation does not mutate a `GBMaker` instance.
- [ ] Private-method tests for migrated behavior are replaced by direct stage tests.

## API / breaking-change impact

- [x] No breaking changes expected; public construction APIs retain established behavior.

## Alternatives considered

Keeping orientation as a cluster of facade-private methods was rejected because it prevents independent testing and obscures exact-versus-approximate path ownership.

## Dependencies and related issues

Depends on GM2. Preserve exact-input semantics established by #46 and exactification semantics from #49.


---


## GM4 — [Feature] Extract GBMaker commensurability and dimension planning

**Recommended tracker action:** Create new  
**Roadmap prerequisite:** GM3  
**Existing issue references:** #46, #47


## What would you like GBOpt to do?

GBOpt should produce one explicit `DimensionPlan` from construction configuration and resolved orientation state.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [x] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [ ] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [ ] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Commensurate-pair search, strain accommodation, minimum in-plane sizing, box dimensions, and physical grain bounds are currently mixed into the mutable facade. These calculations need a single, testable owner before grain generation is split into exact and approximate builders.

## Proposed behavior

Move `_find_commensurate_pair`, strain-accommodation calculations, minimum in-plane dimension handling, and box/physical-grain-bound calculations into a pure dimension-planning stage. Preserve `_find_commensurate_pair` through a compatibility export if it is externally relied upon.

Warnings must remain explicit and in the accepted order; mismatch and exact/approximate policy must not change.

## Acceptance criteria

- [ ] Commensurate-pair selection is verified against brute-force reference cases.
- [ ] `both`, `left`, and `right` strain policies retain accepted behavior.
- [ ] Interaction-distance resizing and minimum dimensions are preserved.
- [ ] Exact no-pair failures and approximate fallback warnings retain established semantics and ordering.
- [ ] Box dimensions and physical grain bounds match baseline results.
- [ ] Dimension planning is pure and no longer performed directly by `GBMaker` for migrated paths.

## API / breaking-change impact

- [x] No breaking changes expected.

## Alternatives considered

Combining dimension planning with grain generation was rejected because it would make scientific path selection and cell-sizing policy difficult to test independently.

## Dependencies and related issues

Depends on GM3. Must preserve exact/approximate construction semantics established by the existing boundary-spec and CSL issues.


---


## GM5 — [Feature] Extract shared GBMaker geometry kernels

**Recommended tracker action:** Create new  
**Roadmap prerequisite:** GM4  
**Existing issue references:** #47, #59 (behavioral invariants)


## What would you like GBOpt to do?

GBOpt should move reusable coordinate, filtering, clipping, wrapping, and deduplication logic into explicit pure geometry functions.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [x] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [ ] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [ ] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Geometry helpers currently depend on hidden facade state and have historically been sensitive to periodic-boundary tolerances and atom completeness. A neutral geometry layer is needed so exact and approximate builders can share well-characterized kernels without reintroducing previous clipping defects.

## Proposed behavior

Extract reduced-coordinate tolerance calculation, periodic basis scaling, selection/box bases, reduced/Cartesian conversion, complete-origin filtering, Cartesian clipping, deterministic deduplication, upper-x trimming, and reduced-coordinate wrapping into `gbmaker/geometry.py`.

Every function receives its tolerances, bases, bounds, and periodicity explicitly; none accepts a `GBMaker` instance. This issue is extraction-only and must not silently retune tolerances.

## Acceptance criteria

- [ ] Geometry functions are pure and do not accept a `GBMaker` instance.
- [ ] Periodic and non-periodic axes and half-open box conventions have direct tests.
- [ ] Complete multi-species basis preservation is covered.
- [ ] Deduplication and output ordering are deterministic.
- [ ] Inputs are not mutated.
- [ ] Exact and approximate characterization remains equivalent to the accepted baseline.
- [ ] No tolerance is widened merely to make tests pass.

## API / breaking-change impact

- [x] No breaking changes expected.

## Alternatives considered

Duplicating geometry code inside exact and approximate builders was rejected because it would create divergent clipping, wrapping, and ordering semantics.

## Dependencies and related issues

Depends on GM4. The issue must preserve the completeness and deterministic-output invariants tracked by #47 and #59.


---


## GM6 — [Feature] Separate exact and approximate grain builders behind common contracts

**Recommended tracker action:** Create new; reference umbrella  
**Roadmap prerequisite:** GM5  
**Existing issue references:** #46, #47, #59


## What would you like GBOpt to do?

GBOpt should implement exact and approximate grain construction as separate builders behind a common request/result contract.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [x] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [ ] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [ ] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Exact crystallographic construction and approximate floating construction have different numerical requirements but currently share facade-owned orchestration. Separating them allows exact decorated-site completeness and approximate conservative enumeration to evolve independently while preserving one higher-level construction pipeline.

## Proposed behavior

Create common builder contracts plus focused exact and approximate builder modules. Move exact repeat calculation and integer supercell/decorated-site generation to the exact builder. Move floating/conservative conventional-cell enumeration to the approximate builder. Generalize build results to `GrainBuildResult`, and provide `build_grain`/`build_grain_pair` entry points.

Assembly, GB-region selection, and file writing remain outside the builders.

## Acceptance criteria

- [ ] Exact FCC, fluorite, and rocksalt construction cases pass direct builder tests.
- [ ] Approximate non-CSL cases pass direct builder tests.
- [ ] Atom counts, stoichiometry, bounds, origin groups, and deterministic ordering match accepted requirements.
- [ ] Exact construction preserves complete decorated sites and does not reintroduce float membership clipping.
- [ ] Generated structures contain no duplicate atoms.
- [ ] Exact and approximate path selection remains unchanged from the accepted public behavior.
- [ ] `GBMaker.py` no longer contains grain enumeration algorithms.

## API / breaking-change impact

- [x] No breaking changes expected in public construction APIs. Corrected exact-path populations established by #59 remain authoritative.

## Alternatives considered

Maintaining one large builder with mode flags was rejected because exact and approximate numerical invariants differ and deserve independently testable implementations.

## Dependencies and related issues

Depends on GM5. Implements/refactors the construction behavior covered by #46, #47, and the exact-completeness portion of #59.


---


## GM7 — [Feature] Add pure bicrystal assembly and end-to-end GBMaker construction pipeline

**Recommended tracker action:** Create new; reference umbrella  
**Roadmap prerequisite:** GM6  
**Existing issue references:** #47, #59


## What would you like GBOpt to do?

GBOpt should assemble and build a complete bicrystal through a pure pipeline that does not require a mutable `GBMaker` instance.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [x] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [ ] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Even after grain builders are separated, placement, concatenation, GB-region selection, and end-to-end orchestration can keep the facade as the true scientific implementation. A pure pipeline is needed before facade state can be safely reduced.

## Proposed behavior

Move left/right placement, concatenation, and GB-region selection into `gbmaker/assembly.py`, define atom ordering and half-open box conventions explicitly, and implement `assemble_bicrystal(...) -> BicrystalResult` plus `build_bicrystal(...) -> BicrystalResult`.

Route the facade rebuild path through this pipeline, using temporary mirrored facade fields only where legacy setters/properties still require them.

## Acceptance criteria

- [ ] The Phase-F0 representative construction cases pass through the pure pipeline.
- [ ] Stage-result consistency is tested.
- [ ] Periodic bicrystal and single-interface slab topology cases preserve accepted behavior.
- [ ] GB-plane and GB-region behavior is unchanged except where previously corrected behavior is explicitly authoritative.
- [ ] Exact and approximate integration paths both pass.
- [ ] The end-to-end scientific construction path works without instantiating `GBMaker`.
- [ ] Clean construction imports do not pull optimization modules.

## API / breaking-change impact

- [x] No breaking changes expected.

## Alternatives considered

Keeping assembly inside the facade was rejected because it would leave the mutable object as an unavoidable scientific dependency.

## Dependencies and related issues

Depends on GM6. Preserve the exact-completeness and topology invariants tracked by #59.


---


## GM8 — [Feature] Migrate GBMaker facade state to the staged construction pipeline

**Recommended tracker action:** Create new; reference existing API issue  
**Roadmap prerequisite:** GM7 and IO3 if not already merged  
**Existing issue references:** #48


## What would you like GBOpt to do?

GBOpt should reduce `GBMaker` to a compatibility/public API facade over validated configuration and a cached staged-pipeline result.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [x] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [ ] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [ ] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** After the scientific stages have been extracted, duplicated mirrored state and legacy private methods would otherwise leave two implementations in the codebase. The final GBMaker migration should establish one canonical state model and preserve compatibility setters deliberately.

## Proposed behavior

Store canonical `_config`, `_boundary`, and `_result` state. Delegate scientific properties to `_result`, configuration properties to `_config`, and configuration updates through validated replacement followed by a centralized `_rebuild()`.

Preserve setter semantics individually, remove transitional mirrors/dead private methods after migration, and keep `write_lammps()` as the thin I/O facade supplied by IO3.

## Acceptance criteria

- [ ] Each supported setter retains its characterized behavior.
- [ ] Scientific properties delegate to the pipeline result; configuration properties delegate to validated configuration.
- [ ] Regeneration is centralized in `_rebuild()`.
- [ ] Transitional mirrored scientific state and dead private methods are removed.
- [ ] Repeated rebuilds are deterministic.
- [ ] Downstream `GBManipulator` and minimizer compatibility tests pass.
- [ ] LAMMPS-specific implementation remains outside `GBOpt.gbmaker`.
- [ ] Dependency-direction tests prevent `GBOpt.gbmaker` from importing I/O or optimization implementation layers.

## API / breaking-change impact

- [x] No additional breaking changes expected; deprecation/public API behavior already established by #48 is preserved.

## Alternatives considered

Keeping both legacy facade-private algorithms and the new pipeline was rejected because dual implementations would inevitably diverge.

## Dependencies and related issues

Depends on GM7 and on IO3 for the final writer seam. Complements #48 rather than replacing its public-API/deprecation requirements.


---


## IO1 — [Feature] Add canonical StructureData model, I/O contracts, and LAMMPS data reader

**Recommended tracker action:** Create new  
**Roadmap prerequisite:** F2  
**Existing issue references:** #59 (round-trip invariants)


## What would you like GBOpt to do?

GBOpt should introduce a generic `StructureData` representation and reader/writer contracts, proven first with a strict LAMMPS data reader.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [ ] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [x] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** File syntax, persistent interface ownership, and optimization state are currently coupled. A neutral structure representation is required so readers/writers can evolve independently of `GBMaker`, `Parent`, and optimizer code.

## Proposed behavior

Create `GBOpt/io/` with `StructureData`, independent reader/writer protocols or ABCs, I/O-specific errors, and format-capability declarations. Extract/wrap the existing LAMMPS data parser as `LammpsDataReader`, preserving strict validation of counts, IDs, types/species, charges, coordinates, and box geometry.

External atom IDs are explicitly serialization identifiers, not persistent atom identities. Keep existing public reader functions as compatibility facades.

## Acceptance criteria

- [ ] `StructureData` carries atoms, full cell, origin, periodicity, optional external IDs, optional charges, and small metadata with defensive-copy semantics.
- [ ] Reader and writer contracts are independent.
- [ ] `LammpsDataReader` handles valid minimal files and rejects malformed sections, duplicate/missing IDs, invalid types/maps, non-finite coordinates, and invalid box data.
- [ ] Old and new reader paths produce equivalent results for supported inputs.
- [ ] Persistent grain ownership is not inferred or owned by the generic structure syntax layer.
- [ ] Existing public reader functions continue to work.

## API / breaking-change impact

- [x] No breaking changes expected; legacy reader entry points remain compatibility facades.

## Alternatives considered

Making `Parent` or `GBMaker` the canonical structure representation was rejected because it would keep generic file syntax coupled to GB-specific domain interpretation.

## Dependencies and related issues

Depends on F2 so generic structure data and persistent interface ownership remain distinct.


---


## IO2 — [Feature] Add LAMMPS dump reader with explicit frame and coordinate semantics

**Recommended tracker action:** Create new  
**Roadmap prerequisite:** IO1  
**Existing issue references:** #59


## What would you like GBOpt to do?

GBOpt should parse LAMMPS dump files into the canonical structure model with explicit frame-selection and coordinate semantics.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [ ] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [x] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [ ] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Dump parsing is needed for evaluator-returned structures and ownership-aware reloads, but trajectory syntax and coordinate conventions should not be embedded in `Parent` or manipulation code.

## Proposed behavior

Extract `read_lammps_dump_file()` into a focused dump reader. Preserve the current first-frame behavior initially, but document it explicitly. Define wrapped, unwrapped, and scaled coordinate handling; required columns; type-label resolution; and box/topology handling.

Do not invent persistent grain labels from dump coordinates.

## Acceptance criteria

- [ ] First-frame selection is explicit and tested on multi-frame inputs.
- [ ] Supported column permutations and wrapped/scaled coordinate variants are tested.
- [ ] Type labels and numeric type maps are handled consistently with the data reader.
- [ ] Malformed frames and bounds fail through the I/O error hierarchy.
- [ ] LAMMPS data and dump readers both return the canonical structure type.
- [ ] Legacy dump-reader entry points remain compatible.
- [ ] No persistent ownership is inferred from file coordinates.

## API / breaking-change impact

- [x] No breaking changes expected.

## Alternatives considered

Adding generic trajectory behavior immediately was rejected; the roadmap only requires the current dump semantics to be explicit and canonicalized first.

## Dependencies and related issues

Depends on IO1. Provides the dump syntax layer later consumed by the ownership-aware loader required by #59.


---


## IO3 — [Feature] Extract LAMMPS writer and introduce WriteResult

**Recommended tracker action:** Create new  
**Roadmap prerequisite:** IO1  
**Existing issue references:** #59 (transient-ID semantics)


## What would you like GBOpt to do?

GBOpt should move LAMMPS serialization and restricted-triclinic conversion out of `GBMaker` and return explicit serialization metadata through `WriteResult`.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [x] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [ ] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [x] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** `GBMaker.write_lammps()` currently owns both construction-facing API and file syntax. Ownership-aware reloads also need a candidate-local mapping between rows and transient LAMMPS IDs. Those are serialization concerns and should have one I/O owner.

## Proposed behavior

Add validated LAMMPS write options and a `WriteResult` containing target, assigned atom IDs, row-to-ID mapping, optional digest, and explicit loss declarations. Move orthogonal and restricted-triclinic serialization into `GBOpt/io/lammps/data_writer.py`.

The writer accepts neutral structure data, not a `GBMaker` instance. `GBMaker.write_lammps()` remains a thin compatibility wrapper.

## Acceptance criteria

- [ ] Orthogonal and restricted-triclinic golden outputs remain equivalent.
- [ ] Atom order, type labels, charges, precision, and established formatting are preserved.
- [ ] ID assignment is deterministic for a given input ordering.
- [ ] `WriteResult` exposes candidate-local row/ID mapping and declared losses.
- [ ] Standalone writer and facade wrapper produce equivalent files.
- [ ] The writer does not import the GBMaker facade.
- [ ] `GBMaker.py` contains no LAMMPS serialization implementation after migration.

## API / breaking-change impact

- [x] No breaking changes expected for `GBMaker.write_lammps()`; the new writer/result API is additive.

## Alternatives considered

Leaving serialization inside GBMaker and duplicating mapping logic in reload code was rejected because transient IDs must have a single authoritative owner.

## Dependencies and related issues

Depends on IO1. This issue owns the `write_lammps()` migration seam and should merge before deep GBMaker facade cleanup.


---


## IO4 — [Feature] Add Parent.from_structure and remove file parsing from Parent

**Recommended tracker action:** Create new; reference umbrella  
**Roadmap prerequisite:** IO2 and F2  
**Existing issue references:** #59


## What would you like GBOpt to do?

GBOpt should let `Parent` consume canonical structure data while retaining GB-domain interpretation and compatibility constructors.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [x] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [ ] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [x] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** `Parent` should interpret GB-domain structure state, not parse file syntax. Keeping parsing inside the class prevents reader reuse and makes manipulation depend directly on specific file formats.

## Proposed behavior

Add `Parent.from_structure(StructureData, ..., grain_ownership=None)`. Move generic structure validation to that path while preserving the distinction between persistent grain ownership and geometric GB-region membership. Make legacy filename construction choose a reader and delegate, and retain GBMaker-source adaptation as a compatibility path.

Unsupported cell geometry or topology should fail explicitly rather than being partially interpreted.

## Acceptance criteria

- [ ] Direct `Parent.from_structure()` construction is covered by tests.
- [ ] Legacy supported filename construction delegates through the I/O layer and remains equivalent.
- [ ] Explicit persistent ownership is not replaced by geometric GB-region membership.
- [ ] Unsupported geometry/topology fails clearly.
- [ ] Format-specific parsing implementation is removed from `Parent` after delegation.
- [ ] Public exception translation and downstream manipulator behavior remain compatible.

## API / breaking-change impact

- [x] No breaking changes expected; legacy constructors remain adapters.

## Alternatives considered

Keeping file parsing as a convenience inside `Parent` was rejected because it would preserve a reverse dependency from GB-domain interpretation to concrete syntax implementations.

## Dependencies and related issues

Depends on IO2 and F2. Implements the file/domain separation needed for the explicit-ownership behavior tracked by #59. MAN1 should start after this seam is stable.


---


## IO5 — [Feature] Add centralized ownership-aware candidate loader and validated round trips

**Recommended tracker action:** Create new; reference umbrella  
**Roadmap prerequisite:** IO3, IO4, F2  
**Existing issue references:** #59


## What would you like GBOpt to do?

GBOpt should provide one authoritative service for validating and reconstructing evaluator-returned or restart-restored interface candidates.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [x] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [x] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [x] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Ownership-aware reload currently risks being duplicated across evaluator and checkpoint paths. Explicit grain labels must survive row reordering and relaxation without being re-inferred from coordinates or the box midpoint.

## Proposed behavior

Promote existing explicit reload logic into a format-neutral `CandidateLoader`. Build transient `CandidateFileMapping` from a `WriteResult` plus persistent ownership, validate the returned structure, and reconstruct a validated `InterfaceCandidate` and/or `Parent`.

Validation includes atom count, unique expected IDs, per-ID species, finite coordinates, cell/box compatibility, topology, frame selection, ownership alignment, and the supported variable-cell policy. Explicit-mode failure must never fall back to geometric ownership inference.

## Acceptance criteria

- [ ] Row reordering is accepted when transient IDs and species still identify the same rows.
- [ ] Missing, extra, duplicate, species-mutated, stale-mapping, cell-incompatible, and topology-incompatible results are rejected.
- [ ] Atoms may cross `gb_plane_x` without losing persistent grain ownership.
- [ ] Explicit ownership never falls back to midpoint/geometric inference on failure.
- [ ] Compatibility wrappers route through one authoritative loader.
- [ ] The loader returns validated domain objects rather than raw file interpretations.
- [ ] Scalar and batch evaluator fixtures can use the same loader contract.

## API / breaking-change impact

- [x] No breaking changes expected; new loader service is additive and legacy wrappers remain available.

## Alternatives considered

Maintaining separate reload implementations for GA, MC, and checkpointing was rejected because they would inevitably disagree on ownership and validation semantics.

## Dependencies and related issues

Depends on IO3, IO4, and F2. Directly decomposes the file/optimizer handoff portion of #59.


---


## MAN1 — [Feature] Add manipulation operation protocol, registry, and GBManipulator facade seam

**Recommended tracker action:** Create new  
**Roadmap prerequisite:** F2 and IO4  
**Existing issue references:** #59 (domain behavior)


## What would you like GBOpt to do?

GBOpt should support manipulation operations through a generic, extensible protocol while preserving existing `GBManipulator` methods.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [x] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [ ] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Current manipulations are facade methods with optimizer-specific dispatch. This makes third-party operations difficult and ties arity, RNG behavior, lineage, and result shape to hard-coded method names.

## Proposed behavior

Create `GBOpt/manipulation/` with `Manipulation`, `ManipulationContext`, `ManipulationResult`, typed configuration/arity/compatibility/capability/execution errors, and an explicit registry without import-time side effects. Define `ManipulationResult.children` as complete `InterfaceCandidate` objects.

Add `GBManipulator.apply()` and `apply_named()` plus a current-state adapter. Existing public methods remain compatibility entry points. Centralize RNG handling so `seed=0` is valid.

## Acceptance criteria

- [ ] A test-defined operation outside GBOpt source can execute through `GBManipulator.apply()`.
- [ ] Operation arity is validated explicitly.
- [ ] Parents are not mutated and child storage is independent.
- [ ] Fixed-RNG replay is deterministic, including a zero seed.
- [ ] Registry duplicate-name handling is explicit.
- [ ] Existing public manipulation methods retain their established return shapes and behavior.
- [ ] No built-in scientific manipulation algorithm is moved in this issue.

## API / breaking-change impact

- [x] No breaking changes expected; generic operation APIs are additive and legacy methods remain supported.

## Alternatives considered

Building optimizer extensibility around strings and `match` statements was rejected because every new operation would still require optimizer source changes.

## Dependencies and related issues

Depends on F2 and IO4. Establishes the seam used by later manipulation extraction and by MAN5.


---


## MAN2 — [Feature] Extract translation, termination, and interface-separation operations

**Recommended tracker action:** Create new; reference umbrella  
**Roadmap prerequisite:** MAN1  
**Existing issue references:** #59


## What would you like GBOpt to do?

GBOpt should implement right-grain translation, termination cycling, and interface separation as standalone operations over complete interface candidates.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [x] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [ ] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** These transformations already depend on explicit grain ownership, physical bounds, periodicity, and boundary-normal topology. Making them operation objects provides the first complete vertical slice of the manipulation architecture while preserving the capstone interface invariant.

## Proposed behavior

Extract operations for right-grain translation, grain-local termination cycling, slab termination cycling, and interface separation. Each consumes explicit interface state, returns a complete `InterfaceCandidate`, declares topology/periodicity requirements, records operation parameters/metadata, avoids file I/O, and does not mutate parents.

Legacy methods remain wrappers returning their established result forms.

## Acceptance criteria

- [ ] Periodic translation behavior is preserved.
- [ ] Periodic-bicrystal and single-interface-slab termination behavior is preserved.
- [ ] Periodic interface separation expands the x box according to the accepted two-interface semantics; slab separation preserves outer vacuum widths.
- [ ] Persistent grain labels, actual GB plane, physical grain bounds, topology, and periodicity are preserved correctly.
- [ ] Composition-order and deterministic-replay tests pass.
- [ ] Legacy and operation-object paths are behaviorally equivalent for supported calls.
- [ ] No independent GB-only slab termination control is added.

## API / breaking-change impact

- [x] No breaking changes expected; legacy public methods remain compatible.

## Alternatives considered

Moving these methods only after every manipulation was redesigned was rejected because a focused vertical slice provides earlier validation of the operation contract.

## Dependencies and related issues

Depends on MAN1. This issue implements/refactors the topology-aware transformation requirements already captured in #59.


---


## MAN3 — [Feature] Extract density and soft-mode manipulation operations

**Recommended tracker action:** Create new; reconcile with existing issue before implementation  
**Roadmap prerequisite:** MAN2  
**Existing issue references:** #27


## What would you like GBOpt to do?

GBOpt should move insertion, removal, and soft-mode displacement into standalone manipulation operations with explicit RNG and ownership semantics.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [x] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [ ] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Unary stochastic operations need the same operation contract as translation/termination so optimizer dispatch can become generic. Density changes must also define grain ownership for inserted atoms and preserve label alignment after removal.

## Proposed behavior

Extract insertion/removal, Delaunay/grid site generation collaborators, and soft-mode displacement into focused modules. Route all stochastic choices through `ManipulationContext.rng`, preserve ownership of surviving rows, and define an explicit ownership policy for inserted atoms. Record inserted/removed atom details in result metadata.

**Important compatibility decision:** open issue #27 currently specifies that the legacy `GBManipulator.displace_along_soft_modes()` API returns a single structure and uses `mode_index` instead of `num_children`. This roadmap issue should preserve that public behavior if #27 remains authoritative. The generic operation layer may support a normalized tuple of children internally, but it must not reintroduce a contradictory legacy public return type.

## Acceptance criteria

- [ ] Insertion/removal preserve stoichiometric and ownership invariants defined by the existing behavior.
- [ ] Ownership remains aligned after row-count changes.
- [ ] All stochastic paths use the supplied generator; no global NumPy RNG is used.
- [ ] Fixed-seed replay works for zero and nonzero seeds.
- [ ] Soft-mode operation behavior is reconciled explicitly with #27 before merge.
- [ ] The legacy `displace_along_soft_modes()` API does not regress from the behavior accepted in #27.
- [ ] Legacy/new operation paths are behaviorally equivalent where their contracts overlap.

## API / breaking-change impact

- [x] Potential breaking-change risk is governed by #27. This issue should not independently change the already-decided legacy soft-mode signature/return type.

## Alternatives considered

Silently following the older roadmap wording about multiple soft-mode children was rejected because it conflicts with the currently open, already-specified API in #27.

## Dependencies and related issues

Depends on MAN2. Must explicitly reference and resolve #27 in the implementation PR.


---


## MAN4 — [Feature] Extract binary slice-and-merge operation with explicit compatibility validation

**Recommended tracker action:** Create new  
**Roadmap prerequisite:** MAN3  
**Existing issue references:** #59 (ownership propagation)


## What would you like GBOpt to do?

GBOpt should implement slice-and-merge crossover as a binary operation with explicit parent arity and compatibility checks.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [x] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [ ] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Crossover is currently special-cased and can rely on implicit one/two-parent assumptions. Generic optimizer dispatch requires binary operations to declare arity and validate whether two candidates are physically compatible before scientific execution.

## Proposed behavior

Implement `SliceAndMerge` with `parent_count = 2`. Validate cell, species, unit-cell, topology, and relevant region compatibility before executing the established x-slice scientific algorithm. Return complete child interface state and structured lineage metadata.

Keep the legacy `slice_and_merge()` method as a wrapper.

## Acceptance criteria

- [ ] Calling the operation with the wrong number of parents fails through a typed arity error.
- [ ] Incompatible parent cells/species/unit-cell/topology/region state fail before scientific execution.
- [ ] Fixed-RNG output is deterministic.
- [ ] Parents are not mutated.
- [ ] Complete child ownership/interface state and structured lineage are returned.
- [ ] Legacy and operation-object paths are behaviorally equivalent.

## API / breaking-change impact

- [x] No breaking changes expected; legacy crossover entry point remains supported.

## Alternatives considered

Retaining a GA-only crossover branch was rejected because arity belongs to the operation contract, not to the optimizer.

## Dependencies and related issues

Depends on MAN3. Ownership-preserving crossover must remain consistent with the broader invariant tracked by #59.


---


## MAN5 — [Feature] Integrate OperationSpec-based manipulation dispatch into MC and GA

**Recommended tracker action:** Create new  
**Roadmap prerequisite:** F1 and MAN4  
**Existing issue references:** #59 (GA ownership/lineage context)


## What would you like GBOpt to do?

GBOpt should let Monte Carlo and genetic-algorithm optimizers select manipulations through immutable `OperationSpec` objects rather than hard-coded string dispatch.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [x] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [x] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Even after operation objects exist, the optimizers remain closed to extension if they still map known strings to hand-written branches. Operation-declared arity and parameter samplers should drive parent selection and invocation.

## Proposed behavior

Add immutable `OperationSpec` with operation identity, weight, and parameter sampler. Convert legacy `choices: list[str]` inputs into compatibility specs, preserving current default probabilities and parameter distributions. Use operation-declared arity for parent selection, replace hard-coded dispatch/crossover paths, store structured lineage, and define explicit policy for any operation that can yield multiple internal children.

## Acceptance criteria

- [ ] Legacy string choices remain supported.
- [ ] Custom weighted `OperationSpec` objects participate in MC and GA.
- [ ] Unary and binary operations drive parent selection through declared arity.
- [ ] Current default mutation/crossover probabilities and sampled parameter distributions are preserved.
- [ ] Structured lineage from manipulation results is retained.
- [ ] Multi-child internal policy is explicit and does not contradict #27's legacy soft-mode API.
- [ ] A test-defined third-party operation can participate in both optimizers without modifying optimizer source.
- [ ] Fixed-seed MC/GA behavior remains deterministic.

## API / breaking-change impact

- [x] No breaking changes expected; new specs are additive and legacy `choices` remain accepted.

## Alternatives considered

Extending the existing `match`/hard-coded GA crossover logic was rejected because each new operation would require optimizer changes.

## Dependencies and related issues

Depends on F1 and MAN4. Recommended to merge before EVAL2 so lineage is normalized once.


---


## OBS1 — [Feature] Improve optimizer reproducibility and standard-library logging

**Recommended tracker action:** Create new  
**Roadmap prerequisite:** F1  
**Existing issue references:** None identified


## What would you like GBOpt to do?

GBOpt should fix narrow run-identity/reproducibility defects and replace optimizer console prints with standard-library logging without introducing the later event framework.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [x] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Long scientific runs need useful diagnostics and reproducible run identity now, but logging should remain a thin operational concern rather than becoming a second state system. Immediate cleanup also prevents later event work from codifying current print-based behavior.

## Proposed behavior

Resolve MC `unique_id` per run when not supplied, retain the resolved seed, replace MC termination `print()` calls with module logging, and add useful run-start/initial-evaluation/best-update/generation-summary/termination logs. Preserve GA evaluator/reload failure details before penalties are applied. Do not install handlers or call `basicConfig()` in library code, and leave `gb_params.py` requested-result stdout unchanged.

## Acceptance criteria

- [ ] Two fresh MC runs without explicit IDs receive different generated IDs.
- [ ] The resolved seed is retained and inspectable.
- [ ] MC termination does not use unsolicited `print()` output.
- [ ] Logging levels and contextual fields are tested.
- [ ] Library code remains quiet by default when the application does not configure logging.
- [ ] GA evaluation/reload failure diagnostics are retained before penalty application.
- [ ] `gb_params.py` result stdout remains machine-parseable.
- [ ] Numerical optimizer behavior is unchanged.

## API / breaking-change impact

- [x] No breaking changes expected except removal of unsolicited console printing in favor of logging.

## Alternatives considered

Jumping directly to durable structured events was rejected because immediate correctness/logging cleanup can merge independently and keeps the later event protocol smaller.

## Dependencies and related issues

Depends on F1. This issue intentionally does not implement RunContext, typed events, JSONL, or checkpoint lifecycle reporting.


---


## EVAL1 — [Feature] Introduce algorithm-neutral evaluation and structure-artifact contracts

**Recommended tracker action:** Create new  
**Roadmap prerequisite:** F1 and IO1  
**Existing issue references:** #59 (evaluation failure provenance)


## What would you like GBOpt to do?

GBOpt should represent candidate evaluation outcomes and structure artifacts through typed, algorithm-neutral contracts.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [x] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [x] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Scalar callbacks, batch callbacks, physical energies, optimizer penalties, output files, reload failures, and candidate reconstruction are currently represented differently across paths. Later observability and checkpointing need one authoritative result model that preserves failure provenance.

## Proposed behavior

Add `EvaluationStatus` and `FailureStage`, `StructureArtifact`, and immutable `EvaluationResult` containing candidate identity, status, physical energy, selection energy/policy field, artifact, failure stage/code/message, reconstructed candidate when available, and small metadata. Add adapters for legacy scalar tuples and batch dictionaries, validating malformed/non-finite/missing/index-misaligned results.

Keep current `CandidateEvaluation` as a compatibility alias/adapter until EVAL2.

## Acceptance criteria

- [ ] Valid scalar and batch callback results normalize to the same typed model.
- [ ] Evaluator exceptions, non-finite energies, missing artifacts, malformed entries, and index misalignment are represented/validated explicitly.
- [ ] Physical result data is distinct from optimizer selection penalty.
- [ ] Candidate/input-index alignment is stable.
- [ ] `StructureArtifact` carries stable artifact identity, path/format/digest metadata without becoming persistent atom identity.
- [ ] Result metadata is immutable/defensively copied as appropriate.
- [ ] Existing callback shapes remain adaptable.

## API / breaking-change impact

- [x] No breaking changes expected; typed contracts are introduced behind compatibility adapters.

## Alternatives considered

Using penalty energy as the only failure representation was rejected because it destroys diagnostic and restart provenance.

## Dependencies and related issues

Depends on F1 and IO1. Establishes the result contract used by EVAL2, observability, and typed checkpoints.


---


## EVAL2 — [Feature] Normalize Monte Carlo and genetic-algorithm evaluation flows

**Recommended tracker action:** Create new; reference umbrella  
**Roadmap prerequisite:** EVAL1 and IO5; MAN5 recommended  
**Existing issue references:** #59


## What would you like GBOpt to do?

GBOpt should make typed `EvaluationResult` authoritative across MC/GA scalar, batch, legacy, and explicit-ownership evaluation paths.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [x] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [x] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** The current optimizer paths can lose evaluator failure context, diverge between scalar and batch handling, and reconstruct structures through different code. This is particularly risky when explicit ownership must remain aligned across file-backed evaluation.

## Proposed behavior

Route scalar and batch callbacks through common adapters, reconstruct returned structures through `CandidateLoader`, and represent failures before applying optimizer penalty policy. Replace ad-hoc last-generation records with typed results and ensure GA selection uses `selection_energy` while physical result data remains available.

Establish the same typed evaluation boundary for MC accepted/proposed candidates without silently changing legacy callback signatures.

## Acceptance criteria

- [ ] Scalar and batch paths produce equivalent typed results for equivalent evaluations.
- [ ] Explicit-ownership round trips use the central `CandidateLoader`.
- [ ] Structure/ownership reconstruction failures retain stage, exception type/code, and message.
- [ ] Penalty policy does not overwrite the physical evaluation/failure record.
- [ ] GA result alignment remains one result per input candidate, including all-invalid/reseed behavior.
- [ ] MC and GA numerical behavior remains equivalent to the accepted baseline.
- [ ] Existing callback signatures remain supported through adapters.

## API / breaking-change impact

- [x] No breaking changes expected for legacy evaluator callbacks; internal result authority changes to `EvaluationResult`.

## Alternatives considered

Normalizing only GA was rejected because observability and checkpointing need the same evaluation semantics across algorithms.

## Dependencies and related issues

Depends on EVAL1 and IO5; MAN5 is recommended first. Directly decomposes the evaluator/GA handoff behavior included in #59.


---


## OBS2 — [Feature] Add RunContext and typed optimization event protocol

**Recommended tracker action:** Create new  
**Roadmap prerequisite:** EVAL2 and MAN5  
**Existing issue references:** None identified


## What would you like GBOpt to do?

GBOpt should expose optimizer lifecycle transitions through a typed, versioned event protocol with silent-by-default sinks.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [x] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Users need structured run observability without coupling logging to optimizer state or restart data. Events should report authoritative transitions that already occurred; they must not become the source of truth for optimizer continuation.

## Proposed behavior

Add immutable `RunContext`, versioned `OptimizationEvent`, `EventSink`, `NullEventSink`, `LoggingEventSink`, and `CompositeEventSink`. Emit common lifecycle events for optimization start, initial evaluation, proposal/evaluation, acceptance/rejection, best updates, generation start/completion, population reseed, termination, and failure.

Populate fields from typed evaluation/manipulation data, keep the default sink silent, exclude large structures/arrays, and define sink-failure policy.

## Acceptance criteria

- [ ] `NullEventSink` is inert and default optimizer behavior remains silent.
- [ ] `CompositeEventSink` preserves documented ordering.
- [ ] Logging sink level mapping is tested.
- [ ] Events carry schema version, timestamp, run identity, algorithm, and relevant candidate/generation metadata.
- [ ] MC and GA use a common vocabulary where the same concept exists.
- [ ] Events do not embed full atom arrays/structures.
- [ ] Sink failures follow an explicit policy.
- [ ] Numerical optimizer behavior is unchanged.

## API / breaking-change impact

- [x] No breaking changes expected; event configuration is additive and defaults to no output.

## Alternatives considered

Treating log lines or checkpoints as the event protocol was rejected because logs are presentation and checkpoints are restart state.

## Dependencies and related issues

Depends on EVAL2 and MAN5. Checkpoint lifecycle events remain deferred until both systems exist.


---


## OBS3 — [Feature] Add durable JSONL optimization journal and run manifest

**Recommended tracker action:** Create new  
**Roadmap prerequisite:** OBS2  
**Existing issue references:** None identified


## What would you like GBOpt to do?

GBOpt should provide opt-in durable JSONL run provenance and a separate run manifest without treating the journal as restart state.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [x] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Scientific workflows benefit from queryable, append-friendly provenance, but restart correctness requires typed checkpoints rather than reconstructing state from logs. Separating invariant run metadata from event records also keeps every journal line small.

## Proposed behavior

Add a versioned `JsonlEventSink` and separate run-manifest writer. Define UTF-8, one-JSON-object-per-line format, flush policy, append/overwrite behavior, and write-failure policy. Reference large structures through artifacts instead of embedding them. The optimization driver is the single authoritative journal writer; scheduler workers do not share a multi-writer JSONL file.

## Acceptance criteria

- [ ] Every journal line is a valid independent JSON object.
- [ ] Manifest and journal agree on run identity/schema metadata.
- [ ] Large atom arrays/structures are not embedded.
- [ ] Append/overwrite and flush policies are documented and tested.
- [ ] Write-failure behavior is explicit.
- [ ] Single-writer assumptions are documented.
- [ ] If a reader is included, a truncated final record is handled according to a documented policy.
- [ ] Documentation explicitly states that the journal is not a checkpoint.

## API / breaking-change impact

- [x] No breaking changes expected; durable journaling is opt-in.

## Alternatives considered

Using the event journal for restart was rejected because observational records may be lossy, append-oriented, or incomplete at process failure.

## Dependencies and related issues

Depends on OBS2. Remains independent of checkpoint persistence.


---


## CP0 — [Feature] Characterize and specify legacy checkpoint/restart behavior

**Recommended tracker action:** Create child issue; update #36 as umbrella  
**Roadmap prerequisite:** F0  
**Existing issue references:** #36


## What would you like GBOpt to do?

GBOpt should turn the older checkpoint-enabled implementation into an explicit behavioral specification before modern checkpoint code is ported.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [x] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Open issue #36 correctly identifies the need for resumable MC/GA runs, but its original implementation details no longer match the refactoring roadmap in several places. The older source contains valuable restart behavior that should be classified before choosing the new schema.

## Proposed behavior

Audit the older checkpoint implementation and document null/disabled behavior, formats, interval/final saves, schema envelope, RNG restoration, MC current/best state, run extension, GA generation snapshots, per-candidate recovery, pending artifacts, cleanup ordering, missing-artifact failures, and resumed-versus-continuous equivalence.

For each legacy behavior classify it as preserve, redesign-but-preserve-externally, intentionally reject, or optional compatibility. Do not add production checkpoint code here.

## Acceptance criteria

- [ ] The legacy checkpoint behavior is enumerated and reviewed.
- [ ] Required MC restart state is explicit.
- [ ] Required GA generation-boundary state is explicit.
- [ ] Intra-generation candidate recovery behavior is explicit.
- [ ] RNG/run-identity restoration and run-extension behavior are classified.
- [ ] Cleanup/artifact publication ordering is specified.
- [ ] Legacy pickle compatibility is explicitly classified rather than assumed.
- [ ] Reference tests are imported as specification/expected-failure tests where useful.

## API / breaking-change impact

- [x] No API change; documentation/tests only.

## Alternatives considered

Implementing directly from #36's original checklist was rejected because the newer architecture intentionally changes persistence format and completion semantics.

## Dependencies and related issues

Depends on F0. #36 should become the checkpoint umbrella and be updated to point to CP0–CP5.


---


## CP1 — [Feature] Add versioned atomic JSON checkpoint store and codec

**Recommended tracker action:** Create child issue; update #36 as umbrella  
**Roadmap prerequisite:** CP0  
**Existing issue references:** #36


## What would you like GBOpt to do?

GBOpt should provide optimizer-independent, versioned, atomic JSON checkpoint persistence with disabled/no-op behavior.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [x] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** The persistence mechanism can be tested independently of MC/GA state. A small storage layer also prevents algorithm modules from each implementing file replacement, schema headers, NumPy conversion, and corruption handling differently.

## Proposed behavior

Create `GBOpt/checkpoint/` with checkpoint-specific errors, a strict versioned envelope, `CheckpointStore`, and disabled/null implementation. Support configurable save intervals and unconditional final saves. Encode approved NumPy arrays/scalars and paths explicitly to JSON.

Write through a temporary file, flush, optionally `fsync`, and atomically replace the destination. Define behavior for absent, corrupt, unsupported-version, wrong-algorithm, and failed-write cases.

**Roadmap decision versus #36:** JSON is the initial supported format. General pickle support is not required unless a concrete compatibility requirement is approved.

## Acceptance criteria

- [ ] Disabled checkpointing is a true no-op.
- [ ] Interval behavior is tested, including lazy state construction not being called when no save is due.
- [ ] Final save can bypass the normal interval.
- [ ] Approved NumPy/path values round-trip through JSON.
- [ ] Writes use atomic replacement with temporary-file cleanup on failure.
- [ ] Corrupt and unsupported-version files fail clearly.
- [ ] Existence/delete behavior is tested.
- [ ] The persistence package does not import optimizer implementations.

## API / breaking-change impact

- [x] Additive checkpoint infrastructure. This intentionally narrows #36's original format requirement to JSON-first.

## Alternatives considered

Supporting arbitrary pickle from the start was rejected because it weakens schema discipline and introduces unsafe/opaque compatibility expectations before a real migration need is established.

## Dependencies and related issues

Depends on CP0 and is a child of the capability tracked by #36.


---


## CP2 — [Feature] Define typed restart snapshots for MC, GA, populations, and evaluations

**Recommended tracker action:** Create child issue; update #36 as umbrella  
**Roadmap prerequisite:** CP1, F2, IO5, EVAL2, MAN5  
**Existing issue references:** #36, #59 (candidate ownership semantics)


## What would you like GBOpt to do?

GBOpt should define immutable, versioned snapshot models that contain exactly the state needed for deterministic MC/GA restart at safe boundaries.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [x] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [x] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Raw dictionaries and live Python objects make checkpoint schemas difficult to validate and evolve. Restartable state must be separated from callbacks, sinks, open files, scheduler clients, and other non-authoritative runtime objects.

## Proposed behavior

Add `MonteCarloSnapshot`, `GeneticAlgorithmSnapshot`, `PopulationCandidateSnapshot`, and `CandidateEvaluationSnapshot`. Define completed safe boundaries (MC completed step, GA completed generation, durable individual evaluation). Store RNG bit-generator type/state, run identity, structured lineage/evaluation data, and validated candidate/artifact/interface references.

Define which run parameters are immutable, restored, or overrideable. Reject arbitrary live objects and perform semantic validation beyond JSON parsing.

## Acceptance criteria

- [ ] All snapshot types round-trip through the checkpoint codec.
- [ ] Missing required fields, wrong algorithm, and unsupported schema versions fail clearly.
- [ ] RNG bit-generator type/state round-trips with deterministic fidelity.
- [ ] Candidate/artifact/interface ownership references are semantically validated.
- [ ] Structured operation lineage and typed evaluation failure data are represented.
- [ ] Callbacks, event sinks, loggers, open files, scheduler clients, and live manipulator objects are prohibited.
- [ ] Resume-override policy is documented and testable.

## API / breaking-change impact

- [x] Additive internal/public checkpoint model depending on exposure decisions; no existing optimizer call must change yet.

## Alternatives considered

Serializing optimizer instances directly was rejected because it couples restart files to Python object layout and non-serializable runtime dependencies.

## Dependencies and related issues

Depends on CP1 plus stabilized interface, loader, evaluation, and operation contracts. Child of #36.


---


## CP3 — [Feature] Add deterministic Monte Carlo checkpoint and resume support

**Recommended tracker action:** Create child issue; update #36 as umbrella  
**Roadmap prerequisite:** CP2, EVAL2, MAN5, IO5  
**Existing issue references:** #36


## What would you like GBOpt to do?

GBOpt should resume Monte Carlo optimization deterministically from completed-step checkpoints using the modern candidate/evaluation/operation contracts.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [x] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [x] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** MC evaluations can be expensive and HPC interruptions should not force a complete restart. Resume must restore the accepted walker state—not merely evaluation history—and must preserve RNG/run identity so interrupted and uninterrupted runs are equivalent.

## Proposed behavior

Add optional checkpoint configuration to `run_MC()` without changing default behavior. At completed safe steps, snapshot the accepted candidate/artifact, RNG state, run identity/resolved seed, completed step, temperature, rejection count, current/best energies, accepted-index history, operation/acceptance history, and approved convergence/cooldown state. Resume candidates through `CandidateLoader`.

Allow `max_steps` extension after completion and retain the checkpoint/artifacts required for extension.

**Roadmap decision versus #36:** successful completion does not automatically delete the checkpoint; retention is intentional to support extension.

## Acceptance criteria

- [ ] No checkpoint artifacts are created when checkpointing is disabled.
- [ ] A valid final checkpoint is retained on normal completion.
- [ ] Interrupted runs resume from the last completed step.
- [ ] Interrupted+resumed and uninterrupted fixed-seed runs are deterministically equivalent.
- [ ] Run ID, resolved seed, RNG, cooldown/minimum-step state, histories, and accepted candidate state are restored according to policy.
- [ ] `max_steps` can be extended after a completed run.
- [ ] Missing/invalid required candidate artifacts fail loudly.
- [ ] Checkpoint interval behavior is tested.

## API / breaking-change impact

- [x] Additive optional `run_MC()` checkpoint configuration. Completion-retention semantics intentionally supersede the original deletion requirement in #36.

## Alternatives considered

Deleting checkpoints at successful completion was rejected because it prevents safe run extension and discards a useful restart boundary.

## Dependencies and related issues

Depends on CP2, EVAL2, MAN5, and IO5. Child of #36.


---


## CP4 — [Feature] Add generation-boundary genetic-algorithm checkpoint and resume support

**Recommended tracker action:** Create child issue; update #36 as umbrella  
**Roadmap prerequisite:** CP3, EVAL2, MAN5, IO5  
**Existing issue references:** #36


## What would you like GBOpt to do?

GBOpt should resume genetic-algorithm optimization deterministically from completed generation boundaries before adding intra-generation recovery.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [x] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [x] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Generation-boundary restart provides robust GA recovery with a much smaller consistency surface than per-candidate caching. It also establishes correct artifact publication/cleanup ordering before finer-grained recovery is introduced.

## Proposed behavior

After a completed generation, snapshot RNG/run identity, completed generation, best result/artifact, energy history, structured population lineage, ordered next-generation candidate artifacts/interface state, and required deterministic GA configuration. Publish next-generation artifacts before publishing the snapshot that references them, and clean stale pending artifacts only after the new snapshot is durable.

Resume all candidates through `CandidateLoader` and retain final checkpoints/artifacts for extension.

## Acceptance criteria

- [ ] Interrupted runs resume from completed generation boundaries.
- [ ] Fixed-seed uninterrupted and resumed runs are equivalent.
- [ ] History and energy arrays are not duplicated on resume.
- [ ] Generation count can be extended after completion.
- [ ] Missing/invalid population artifacts fail loudly.
- [ ] Pending-artifact rollover is atomic/order-safe.
- [ ] Scalar and batch evaluation modes remain supported.
- [ ] All-invalid generation reseeding behavior is preserved.

## API / breaking-change impact

- [x] Additive optional GA checkpoint configuration; default behavior remains unchanged.

## Alternatives considered

Implementing per-candidate recovery first was rejected because generation-boundary restart provides a simpler authoritative snapshot on which finer-grained caching can build.

## Dependencies and related issues

Depends on CP3 plus stabilized GA evaluation/loader/operation contracts. Child of #36.


---


## CP5 — [Feature] Recover completed GA candidate evaluations within interrupted generations

**Recommended tracker action:** Create child issue; update #36 as umbrella  
**Roadmap prerequisite:** CP4 and EVAL2  
**Existing issue references:** #36


## What would you like GBOpt to do?

GBOpt should resume an interrupted GA generation without repeating candidate evaluations that were already completed and durably recorded.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [x] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [x] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** A single GA generation may contain many expensive external calculations. Generation-boundary checkpoints still waste completed evaluations if interruption occurs mid-generation.

## Proposed behavior

Add a per-generation candidate result cache keyed by stable candidate IDs. Atomically record complete `CandidateEvaluationSnapshot` data after each durable result, skip already completed candidates on resume, and reconstruct ordered generation results without changing input alignment.

Support scalar evaluators directly. For batch evaluators, support batch-return granularity automatically and finer-grained recording only through an explicit adapter/protocol. Delete the transient cache only after the authoritative generation checkpoint is durable.

## Acceptance criteria

- [ ] Completed scalar candidate evaluations are skipped after resume.
- [ ] Batch-return recovery works at documented granularity.
- [ ] An explicit fine-grained batch recovery adapter is supported if provided.
- [ ] Ordered result alignment is reconstructed exactly.
- [ ] Failure results restore full typed provenance, not only penalty energy.
- [ ] Transient cache is deleted only after the completed-generation snapshot is durable.
- [ ] Orphan/stale cache cleanup is deterministic.
- [ ] No cache is created when checkpointing is disabled.
- [ ] Interrupted/resumed candidate reconstruction is equivalent to uninterrupted execution.

## API / breaking-change impact

- [x] No breaking changes expected; finer-grained batch recovery is opt-in through an explicit protocol.

## Alternatives considered

Inspecting evaluator function signatures to inject hidden callback semantics was rejected in favor of an explicit recovery adapter.

## Dependencies and related issues

Depends on CP4 and EVAL2. Final core child issue under #36.


---


## INT1 — [Feature] Complete cross-track architecture hardening and refactor release gate

**Recommended tracker action:** Create new integration issue  
**Roadmap prerequisite:** GM8, IO5, MAN5, OBS3, CP5  
**Existing issue references:** #40, #57, #59


## What would you like GBOpt to do?

GBOpt should verify that all refactoring tracks compose cleanly, enforce dependency direction, update documentation/examples, and pass an end-to-end release gate.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [x] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [x] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [x] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [x] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** The individual tracks intentionally minimize shared-file conflicts, but only a final integration issue can prove that construction, manipulation, I/O, evaluation, observability, and checkpointing use the same domain services and do not recreate reverse dependencies.

## Proposed behavior

Add architecture tests for package dependency direction, verify compatibility import identity, prove evaluator reload and checkpoint resume use the same `CandidateLoader`, verify transient mappings remain candidate-local, and reject journals as checkpoints.

Add end-to-end build → manipulate → write → evaluate/reload → optimize tests plus interrupted/resumed MC and GA with logging, JSONL journal, and checkpointing enabled together. Update application-layer examples/documentation for logging, journals, checkpoints, and extension points. Produce a final verified source archive and integrity record.

## Acceptance criteria

- [ ] `GBOpt.gbmaker` does not import I/O or optimization implementation layers.
- [ ] `GBOpt.io` does not import `GBMaker` or minimizer facades.
- [ ] Manipulation operations do not perform I/O/evaluation.
- [ ] Checkpoint persistence does not depend on live optimizer classes; events do not own restart state.
- [ ] Compatibility import identity for `GBMaker`, `GBManipulator`, and minimizers is verified.
- [ ] Evaluator-return and checkpoint-resume reconstruction use the same `CandidateLoader`.
- [ ] Journal files are not accepted as checkpoints.
- [ ] End-to-end periodic and slab workflows pass.
- [ ] Interrupted/resumed MC and GA deterministic gates pass.
- [ ] Full non-slow and approved slow/integration suites pass, and docs/examples are updated.

## API / breaking-change impact

- [x] No new feature-specific breaking changes should be introduced here; this is a hardening/release gate over previously approved changes.

## Alternatives considered

Allowing each track to declare itself complete independently was rejected because dependency-direction and cross-track state-identity failures only appear when the complete system is exercised.

## Dependencies and related issues

Depends on all core tracks. Related to #40 (examples/testing), #57 (release notes), and the capstone behavioral invariant #59.


---


# Deferred / optional issue matrix

These are intentionally **not recommended for immediate creation**. The drafts are retained so the scope is ready if a concrete requirement appears.

| Roadmap | Proposed issue title | Tracker action | Prerequisite(s) | Existing issue references |
|---|---|---|---|---|

| OPT-IO1 | [Feature] Add versioned ownership sidecars and explicit lossy-write policy | Draft only — do not open until concrete requirement | IO5 | #59 (ownership semantics) |
| OPT-IO2 | [Feature] Add XYZ and CIF structure adapters | Draft only — do not open until concrete requirement | IO1/IO5 stable | None required |
| OPT-PLUGIN1 | [Feature] Add third-party entry-point discovery for GBOpt extensions | Draft only — do not open until concrete requirement | Built-in registries/contracts stable | None required |
| OPT-CP1 | [Feature] Add one-way legacy checkpoint migration reader | Draft only — do not open until real legacy files must be preserved | CP1/CP2 stable | #36 |
| OPT-OBS1 | [Feature] Add checkpoint lifecycle optimization events | Draft only — open after checkpointing and events are both stable if useful | OBS2 and CP3/CP4 stable | #36 |
| OPT-API1 | [Feature] Expose public immutable bicrystal construction API | Draft only — open after GM8 when public demand exists | GM8 | #42/#48 may inform migration |

# Copy-ready deferred / optional issue drafts


## OPT-IO1 — [Feature] Add versioned ownership sidecars and explicit lossy-write policy

**Recommended tracker action:** Draft only — do not open until concrete requirement  
**Roadmap prerequisite:** IO5  
**Existing issue references:** #59 (ownership semantics)


## What would you like GBOpt to do?

GBOpt should optionally persist interface ownership/topology metadata beside structure files through a versioned sidecar format and require explicit opt-in for lossy writes.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [ ] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [x] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Some external file formats cannot represent GBOpt's persistent grain ownership, actual interface plane, physical grain bounds, or topology. A sidecar could preserve this state when a real interoperability workflow needs it.

## Proposed behavior

Define a versioned `.gbopt.json` ownership sidecar with content binding to the primary structure artifact, transactional paired publication, explicit loss declarations, and an `allow_lossy` policy. The sidecar must not redefine transient atom IDs as persistent identities.

## Acceptance criteria

- [ ] Sidecar schema is versioned and semantically validated.
- [ ] Sidecar is content-bound to the associated structure artifact.
- [ ] Paired structure/sidecar publication has a documented transactional policy.
- [ ] Losses are declared explicitly and writes that drop required state require `allow_lossy=True` or equivalent.
- [ ] Round-trip restores ownership/topology through the central candidate loader.

## API / breaking-change impact

- [x] Additive optional I/O capability.

## Alternatives considered

Opening this now was rejected because the core roadmap can preserve ownership through in-memory/artifact mappings without committing to a public sidecar schema.

## Dependencies and related issues

Deferred until a concrete user workflow requires portable ownership metadata outside checkpoint/evaluator round trips.


---


## OPT-IO2 — [Feature] Add XYZ and CIF structure adapters

**Recommended tracker action:** Draft only — do not open until concrete requirement  
**Roadmap prerequisite:** IO1/IO5 stable  
**Existing issue references:** None required


## What would you like GBOpt to do?

GBOpt should support additional structure formats through the canonical I/O contracts, beginning with XYZ and then CIF once cell semantics are stable.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [ ] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [x] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [ ] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Additional adapters are useful only after generic structure capabilities and explicit loss handling are established; otherwise each format risks inventing its own semantics.

## Proposed behavior

Implement an XYZ adapter first to exercise explicit loss reporting for limited metadata. Add CIF only after full-cell/origin/periodicity semantics are stable and tested. Both adapters must use `StructureData` and I/O capability declarations.

## Acceptance criteria

- [ ] Adapters implement the canonical reader/writer contracts.
- [ ] Unsupported metadata loss is reported explicitly.
- [ ] Round-trip behavior is documented per format capability.
- [ ] No format adapter imports GBMaker/minimizer implementation layers.

## API / breaking-change impact

- [x] Additive.

## Alternatives considered

Bundling new formats into the core I/O refactor was rejected because it would mix architecture stabilization with new format behavior.

## Dependencies and related issues

Deferred until a concrete format requirement exists.


---


## OPT-PLUGIN1 — [Feature] Add third-party entry-point discovery for GBOpt extensions

**Recommended tracker action:** Draft only — do not open until concrete requirement  
**Roadmap prerequisite:** Built-in registries/contracts stable  
**Existing issue references:** None required


## What would you like GBOpt to do?

GBOpt should optionally discover third-party readers, writers, and manipulation operations through package entry points.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [ ] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Explicit in-process registries are sufficient for the core refactor. Automatic package discovery adds packaging, naming, conflict, and import-side-effect policy that should only be standardized after extension contracts have proven stable.

## Proposed behavior

Define namespaced Python package entry points for supported extension categories, with deterministic discovery, duplicate/conflict handling, lazy/error-isolated loading, and a way to inspect discovered capabilities.

## Acceptance criteria

- [ ] Entry-point groups and naming rules are documented.
- [ ] Duplicate/conflicting names fail or resolve by an explicit policy.
- [ ] A broken third-party plugin does not silently corrupt built-in registries.
- [ ] Built-in operation/I/O behavior remains available without entry-point discovery.

## API / breaking-change impact

- [x] Additive plugin capability.

## Alternatives considered

Adding discovery to MAN1/IO1 was rejected because the core architecture only needs explicit registries to prove extensibility.

## Dependencies and related issues

Deferred until external packages need automatic discovery.


---


## OPT-CP1 — [Feature] Add one-way legacy checkpoint migration reader

**Recommended tracker action:** Draft only — do not open until real legacy files must be preserved  
**Roadmap prerequisite:** CP1/CP2 stable  
**Existing issue references:** #36


## What would you like GBOpt to do?

GBOpt should migrate specific legacy checkpoint files into the modern typed snapshot schema only if real unmerged runs need preservation.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [x] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** The older checkpoint implementation is a behavioral reference, but unsafe/general pickle compatibility would create a long-lived security and maintenance burden without demonstrated user value.

## Proposed behavior

Implement a narrowly scoped, one-way migration reader for explicitly supported legacy schema versions/formats. Validate expected fields and convert them into modern typed snapshots; reject arbitrary pickle object graphs and unknown schemas.

## Acceptance criteria

- [ ] Supported legacy schema versions are enumerated explicitly.
- [ ] Migration output passes modern semantic snapshot validation.
- [ ] Unknown/untrusted object graphs are rejected.
- [ ] Migration is one-way; modern writers do not emit legacy schemas.
- [ ] Representative real legacy files, if available, have regression fixtures.

## API / breaking-change impact

- [x] Additive compatibility utility.

## Alternatives considered

General pickle loading was rejected because it is unsafe and makes long-term restart compatibility depend on old Python object layouts.

## Dependencies and related issues

Deferred until a concrete legacy checkpoint preservation requirement exists.


---


## OPT-OBS1 — [Feature] Add checkpoint lifecycle optimization events

**Recommended tracker action:** Draft only — open after checkpointing and events are both stable if useful  
**Roadmap prerequisite:** OBS2 and CP3/CP4 stable  
**Existing issue references:** #36


## What would you like GBOpt to do?

GBOpt should optionally report checkpoint save/load/failure actions through the typed event system without making events authoritative restart state.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [x] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** Once both systems exist, lifecycle events can improve diagnostics and provenance, but adding them earlier would couple unfinished schemas and risk treating the event journal as a checkpoint.

## Proposed behavior

Add versioned `checkpoint_saved`, `checkpoint_loaded`, and `checkpoint_failed` event types populated from completed checkpoint actions. Include checkpoint identity/path/artifact metadata appropriate for observability, but never enough mutable state to make the event authoritative for restart.

## Acceptance criteria

- [ ] Checkpoint events are emitted only after/because of real checkpoint actions.
- [ ] Failure events preserve checkpoint-stage diagnostics.
- [ ] No checkpoint snapshot payload or large state arrays are embedded.
- [ ] Restart never reads event journals as checkpoint state.

## API / breaking-change impact

- [x] Additive optional event vocabulary.

## Alternatives considered

Including checkpoint events in OBS2 was rejected because checkpoint persistence semantics are not yet stable at that stage.

## Dependencies and related issues

Deferred follow-on after both observability and checkpointing are established.


---


## OPT-API1 — [Feature] Expose public immutable bicrystal construction API

**Recommended tracker action:** Draft only — open after GM8 when public demand exists  
**Roadmap prerequisite:** GM8  
**Existing issue references:** #42/#48 may inform migration


## What would you like GBOpt to do?

GBOpt should optionally expose the staged construction pipeline as a supported immutable public API and begin a separately approved deprecation path for mutable facade behavior.

## Environment

- **GBOpt version:** `main` at implementation start; record the exact base commit/SHA in the implementation PR
- **Python version:** Supported project versions
- **OS / platform:** Platform-independent unless a focused test requires otherwise

## Which part of the pipeline does this affect?

- [x] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [ ] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [x] **Other / cross-cutting** — architecture, domain contracts, tests, observability, or persistence as described below

## Motivation

**Capability / maintainability gap:** The core roadmap intentionally keeps the staged pipeline internal while preserving the established `GBMaker` facade. Making internal configuration/result types public is a separate compatibility commitment and should be evaluated after the architecture stabilizes.

## Proposed behavior

Promote selected immutable types such as `GBBuildConfig`, `BicrystalResult`, and `build_bicrystal` to supported public imports with documented stability guarantees. Any deprecation of mutable setters/facade behavior requires a separate migration policy and release cycle.

## Acceptance criteria

- [ ] Public immutable construction types/functions have documented stability and examples.
- [ ] Public imports are tested.
- [ ] Mutable facade deprecation, if approved, has a versioned migration plan.
- [ ] No internal-only intermediate type is accidentally promised as public API.

## API / breaking-change impact

- [x] Potential future breaking-change/deprecation program; must be separately approved.

## Alternatives considered

Making all pipeline dataclasses public during GM1–GM8 was rejected because internal boundaries need freedom to stabilize before becoming compatibility commitments.

## Dependencies and related issues

Deferred follow-on after GM8.


---


# Recommended issue-creation order

The issues do not all need to be opened simultaneously. A practical sequence is:

1. Create F0 first.
2. Once F0 is accepted, create F1, F2, GM1, and CP0.
3. Edit #36 into the checkpoint umbrella before creating CP1–CP5.
4. Create downstream issues as their prerequisites become active, so GitHub's open-issue list reflects actionable work rather than the entire long-range plan at once.
5. Before MAN3 implementation, resolve the #27 compatibility note explicitly in the issue/PR.
6. Keep #59 linked from every issue that refactors or implements one of its behavioral invariants.
7. Open INT1 only after the prerequisite tracks are substantially complete.

# Suggested labels / metadata

The repository feature template currently applies the `enhancement` label. If additional project labels are desirable, the following would make the roadmap easier to filter without changing issue semantics:

- `refactor`
- `roadmap:foundation`
- `roadmap:gbmaker`
- `roadmap:io`
- `roadmap:manipulation`
- `roadmap:evaluation`
- `roadmap:observability`
- `roadmap:checkpointing`
- `roadmap:integration`
- `deferred`

These are suggestions only; the drafts do not assume those labels already exist.

# Notes for converting drafts into GitHub issues

- Replace roadmap IDs in dependencies with actual GitHub issue numbers after creation.
- Keep the roadmap ID in the issue body or title prefix only if it is useful for project tracking; it does not need to be part of the user-facing title.
- When an implementation PR opens, add the exact base commit/SHA and executed test commands to the PR description rather than hard-coding a stale source SHA into a long-lived issue.
- Existing umbrella/capstone issues should be updated with links to the new child implementation issues so closure remains traceable.
- Do not close #36 merely because CP1 lands; close it only when the approved checkpoint capability represented by CP0–CP5 is complete.
- Do not close #59 based only on architectural extraction; close it when its behavioral acceptance criteria are genuinely satisfied.
