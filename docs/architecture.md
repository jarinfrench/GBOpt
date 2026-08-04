# GBOpt Architecture

## Purpose and authority

GBOpt constructs, transforms, evaluates, and optimizes grain-boundary structures while preserving explicit crystallographic, chemical, topological, ownership, and reproducibility invariants.

This document is the current system-wide architectural overview. It is subordinate to `MASTER_PLAN.md` for implementation sequence, prerequisites, branch ancestry, merge order, and file-ownership windows. Cross-cutting decisions are recorded in `architecture/adr/`.

The principal responsibilities are:

- `UnitCell` and crystallography modules: crystal definitions and exact crystallographic operations;
- `GBMaker` and `GBOpt.gbmaker`: deterministic bicrystal construction;
- `GBOpt.io`: external structure syntax and transient serialization identity;
- interface-domain types: persistent grain ownership and interface topology;
- `GBManipulator` and `GBOpt.manipulation`: structural operations;
- evaluator adapters: calculator invocation and normalized results;
- Monte Carlo and genetic algorithms: optimization policy;
- event sinks and journals: observation and provenance;
- checkpoint stores and snapshots: deterministic continuation.

Energy or another scalar objective is evaluated externally to the construction and manipulation layers.

## Architectural flow

```text
validated boundary and material inputs
                |
                v
       GBMaker staged pipeline
                |
                v
     StructureData + GrainOwnership
                |
                v
        InterfaceCandidate
                |
                v
       ManipulationResult
                |
                v
        EvaluationResult
                |
                v
      MC or GA policy decision
          |               |
          v               v
 OptimizationEvent   OptimizationSnapshot
          |               |
          v               v
 logging / journal     checkpoint store
```

Authoritative scientific and optimizer state flows through typed objects. Logs, events, journals, and checkpoints report or persist that state according to distinct contracts.

## Core domain contracts

The detailed decision is recorded in [ADR 0001](architecture/adr/0001-domain-contracts.md).

### `StructureData`

Owns generic atomic structure representation:

- atoms and coordinates;
- full cell and origin;
- per-axis periodicity;
- optional external atom IDs and charges;
- small format-neutral metadata.

It does not own persistent grain identity, interface topology, optimization lineage, calculator execution, or restart state.

### `GrainOwnership`

Owns persistent interface-domain identity:

- row-aligned grain labels;
- authoritative interface-plane location;
- physical grain bounds;
- in-plane periodicity;
- boundary-normal topology;
- coordinate tolerance.

Persistent ownership is independent of current coordinates. Relaxed atoms may cross the interface plane without changing grain ownership.

### `InterfaceCandidate`

Composes structure and persistent interface state into an optimization-ready candidate. New manipulation, evaluation, and checkpoint paths operate on complete candidates rather than raw position arrays.

### `WriteResult` and `StructureArtifact`

`WriteResult` describes one serialization operation, including its output and row-to-external-ID mapping. Those IDs are transient and candidate-local.

`StructureArtifact` identifies a produced or consumed artifact by path or storage reference, format, stable artifact ID, optional digest, and small metadata.

### `ManipulationResult`

Contains one or more complete `InterfaceCandidate` children plus the stable operation name, parameters, and structured lineage metadata.

### `EvaluationResult`

Contains normalized evaluator status, physical objective value, selection value where applicable, artifact, reconstructed candidate, and failure stage or reason. Optimizer penalties may be applied after a failure is represented, but must not erase its context.

### `RunContext`, `OptimizationEvent`, and `OptimizationSnapshot`

`RunContext` stores immutable run identity such as the run ID, resolved seed, and algorithm.

`OptimizationEvent` reports a state transition for logs, progress displays, or journals. It is not restart state.

`OptimizationSnapshot` stores the state required to continue an optimizer from a documented safe boundary. It must not serialize callbacks, sinks, loggers, open files, scheduler clients, or live calculator processes.

## Layer ownership and dependency direction

### Boundary and crystallography

`BoundarySpec.py` owns declarative user-facing boundary data and validation. Exact crystallographic arithmetic belongs in the dedicated crystallography modules and exact-integer helpers.

`GBMaker` may consume resolved crystallographic results, but must not become a second crystallography package.

### GBMaker construction

`GBOpt.gbmaker` owns calculator-neutral construction stages:

1. input normalization and material resolution;
2. orientation and periodicity resolution;
3. commensurability and dimension planning;
4. shared geometry kernels;
5. exact or approximate grain construction;
6. bicrystal assembly.

`GBMaker.py` remains a compatibility facade. Construction modules must not import optimization or I/O implementations.

### Structure I/O

External file syntax belongs to `GBOpt.io`, as recorded in [ADR 0002](architecture/adr/0002-io-owns-file-syntax.md).

Readers convert external representations to `StructureData`. Writers serialize neutral structure data and return `WriteResult`. I/O does not infer persistent grain ownership, execute calculators, or choose optimizer penalties.

`Parent` remains a compatibility and GB-domain interpretation object. It should construct from validated structure data and optional ownership metadata rather than implement format parsers.

### Manipulation

Individual operations are the extension point, as recorded in [ADR 0003](architecture/adr/0003-operation-level-manipulations.md).

Operations:

- declare exact parent arity;
- consume complete interface candidates or validated views;
- return complete candidate children;
- use a supplied `numpy.random.Generator`;
- do not perform I/O or objective evaluation;
- do not mutate parents;
- expose structured parameters and lineage.

`GBManipulator` remains a compatibility facade and operation coordinator.

### Evaluation and optimization

Evaluator adapters normalize legacy scalar and batch callback results into `EvaluationResult`. The evaluator boundary owns invocation, result parsing, artifact validation, and candidate reconstruction. MC and GA own acceptance, selection, reseeding, convergence, and penalty policy.

### Observability and restart

Logging, events, journals, and checkpoints are separate mechanisms, as recorded in [ADR 0004](architecture/adr/0004-events-are-not-checkpoints.md).

- logging provides ephemeral diagnostics;
- events provide typed semantic reports;
- journals retain optional durable provenance;
- checkpoints retain restartable optimizer state.

A journal is never accepted as a checkpoint.

## Physical and mathematical invariants

### Coordinate convention

The current public construction convention places the boundary normal along x. Grains are separated primarily along x. This is a current GBMaker convention, not permission to infer persistent ownership from x coordinates after serialization or relaxation.

Changing the configurable boundary-normal axis is a separate future API and scientific change.

### Topology and periodicity

Topology is explicit. Supported or planned states include:

- a periodic bicrystal with two interfaces along the boundary normal;
- a non-periodic slab with one interface and free outer surfaces;
- optional vacuum along the boundary normal;
- periodic or non-periodic in-plane axes where supported by the construction contract.

The boundary-normal direction is not universally periodic or universally non-periodic.

Periodic construction uses canonical reduced-coordinate representatives according to [ADR 0005](architecture/adr/0005-canonical-periodic-representatives.md). Raw Cartesian clipping must not replace reduced-coordinate membership for oblique periodic cells.

### Unit-cell authority

`UnitCell` owns:

- type mapping;
- stoichiometric ratio;
- bond-length and radius data;
- lattice geometry.

Other layers must not redefine those values independently.

### Exact arithmetic

Exact crystallographic quantities remain exact, normally using Python-sized integers and NumPy `dtype=object`. Exact identities must not be verified through floating-point approximations when an exact formulation is available.

The project row-vector rotation convention is preserved explicitly across row/column algorithm boundaries.

### Grain and atom integrity

Construction, filtering, wrapping, clipping, and deduplication preserve complete basis or origin groups where the construction contract requires them. Arbitrary atom deletion is not an acceptable repair for a gap, overlap, or periodic-selection defect.

Stoichiometry must be preserved by construction unless a higher-level explicit termination policy intentionally changes it and records that decision.

### Dimensions and bounds

- spacing and physical dimensions must remain positive;
- planned dimensions must agree with repeat factors and periodic spacing;
- declared cells and boxes must agree with generated geometry;
- generated atoms must satisfy the documented half-open or closed boundary convention;
- exact and approximate paths must preserve their distinct mode contracts;
- exact construction must not silently fall back to approximate construction.

### Type mapping and transient IDs

Format type maps must be validated, deterministic, and invertible where the format requires it. External atom IDs must survive the specific validated round trip for which they were produced, but are not persistent atom identities.

### Randomness

Global NumPy random functions are prohibited in production paths. All stochastic behavior uses `numpy.random.Generator`. A seed of zero is valid. Identical inputs, configuration, and RNG state must reproduce the same result.

## Output-channel policy

- return values and state objects carry algorithmic outcomes;
- exceptions indicate unsatisfied contracts;
- Python warnings communicate caller-visible, recoverable API conditions;
- logs provide operational diagnostics;
- stdout carries a command's requested result;
- stderr carries configured diagnostics and progress;
- journals carry optional durable provenance;
- checkpoints carry restart state;
- structure files carry scientific or calculator artifacts.

No channel silently substitutes for another.

## Compatibility policy

Existing public imports, exception identities, and compatibility facades remain stable during their declared migration windows. Implementations move once; old locations delegate or re-export rather than retain duplicate algorithms.

Compatibility does not require preserving behavior that violates an accepted physical, mathematical, ownership, or serialization invariant.

## Performance boundaries

Performance-sensitive areas include:

- exact and approximate grain construction;
- periodic representative selection and deduplication;
- neighbor and contact searches;
- insertion and removal;
- soft-mode calculations;
- large population evaluation and artifact reload.

Avoid unnecessary full-array copies, dense all-pairs distance matrices, unbounded searches, and Python loops over large atom populations where bounded vectorized or spatial methods exist. Correctness, exactness, and determinism take precedence over speed.

## Enforcement

Architecture is enforced through:

- the PR boundaries and ownership windows in `MASTER_PLAN.md`;
- the normative rules in `AI_AGENT_CODEBASE_RULES.md`;
- the test policy in `testing.md`;
- focused invariant and architecture tests;
- compatibility tests for public import identity;
- deterministic characterization manifests;
- uninterrupted-versus-resumed checkpoint tests;
- import-direction tests at the final integration gate.

A change to an accepted cross-cutting decision should normally create a new ADR that supersedes the previous decision rather than silently rewriting history.
