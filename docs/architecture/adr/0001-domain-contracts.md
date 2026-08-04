# ADR 0001: Shared Domain Contracts

* **Status:** Accepted
* **Date:** 2026-08-04
* **Decision owners:** GBOpt maintainers
* **Related roadmap items:** F2, IO1, IO3, IO5, MAN1, EVAL1, OBS2, CP2

## Context

GBOpt currently passes grain-boundary structures among construction, manipulation, serialization, evaluation, optimization, and restart code through a mixture of structured NumPy arrays, `Parent` objects, `GBMaker` instances, filenames, transient LAMMPS atom IDs, and dictionaries.

These representations overlap but do not own the same concepts.

In particular:

* generic atomic structure data is not the same as persistent grain ownership;
* a structure written to a file is not the same as an optimization candidate;
* a LAMMPS atom ID is not a persistent atom identity;
* an evaluator result is not an optimizer decision;
* an optimization event is not restart state;
* a journal is not a checkpoint.

Without explicit contracts, responsibilities can migrate into whichever class currently has access to the necessary data. That produces reverse dependencies, duplicated validation, and loss of interface identity during file round trips.

## Decision

GBOpt will use a set of narrow, typed domain contracts. Each contract owns one category of information and must not absorb neighboring responsibilities.

## Contracts

### `StructureData`

`StructureData` owns generic atomic structure representation:

* atoms and coordinates;
* full cell representation;
* origin;
* per-axis periodicity;
* optional external atom IDs;
* optional charges;
* small format-neutral metadata.

It does not own:

* persistent grain identity;
* interface topology;
* optimization lineage;
* calculator execution;
* restart state.

External atom IDs stored in `StructureData` are serialization identifiers. They must not be interpreted as optimizer-wide atom identity.

### `GrainOwnership`

`GrainOwnership` owns persistent interface-domain identity:

* row-aligned left/right grain labels;
* authoritative interface-plane position;
* physical grain bounds;
* boundary-normal topology;
* in-plane periodicity;
* coordinate tolerance;
* metadata needed to preserve ownership across supported transformations.

Persistent grain labels are independent of current atom coordinates. Atoms may cross the interface plane during relaxation without changing their grain ownership.

### `InterfaceCandidate`

`InterfaceCandidate` is the optimization-ready composition of structure and interface state.

It owns:

* candidate atoms;
* persistent grain labels;
* cell or box information;
* interface-plane location;
* topology and periodicity;
* accumulated interface-state metadata;
* validation required for a usable candidate.

It does not own:

* file parsing;
* calculator invocation;
* optimization acceptance policy;
* event sinks;
* checkpoint storage.

### `WriteResult`

`WriteResult` describes one serialization operation:

* output target;
* format;
* assigned external atom IDs;
* row-to-ID mapping;
* optional digest;
*[118;1:3u explicitly declared losses.

Its mappings have candidate-local and serialization-local lifetime. They must not be retained as permanent atom identity.

### `StructureArtifact`

`StructureArtifact` identifies a produced or consumed structure artifact:

* stable artifact ID;
* path or storage reference;
* format;
* optional digest;
* producing candidate ID;
* small metadata.

It does not imply that the artifact has been validated for ownership-aware reconstruction.

### `ManipulationResult`

`ManipulationResult` describes the result of applying one manipulation:

* one or more `InterfaceCandidate` children;
* stable operation name;
* explicit operation parameters;
* structured lineage and provenance metadata.

Manipulations must not return only raw arrays on the new internal path. Legacy public wrappers may temporarily preserve historical return types.

### `EvaluationResult`

`EvaluationResult` owns normalized evaluator outcomes:

* candidate identity;
* status;
* physical objective value, when available;
* selection value, when policy requires one;
* artifact;
* reconstructed candidate, when available;
* failure stage;
* failure code;
* failure message;
* small result metadata.

Penalty values are optimizer policy. They may be applied after an evaluation failure is represented, but they must not erase the original failure information.

### `RunContext`

`RunContext` owns immutable run-level identity:

* run ID;
* resolved random seed;
* algorithm;
* optional case, campaign, or boundary identifiers.

It does not own mutable optimizer state.

### `OptimizationEvent`

`OptimizationEvent` is a typed report of a completed or attempted domain action.

It may describe:

* lifecycle transitions;
* candidate proposal and evaluation;
* acceptance or rejection;
* generation completion;
* recovery or reseeding;
* termination.

An event must report authoritative state stored elsewhere. It must not be the sole representation of an algorithmically significant outcome.

### `OptimizationSnapshot`

`OptimizationSnapshot` owns restartable optimizer state at a documented safe boundary.

It may include:

* RNG state;
* completed step or generation;
* current and best candidates;
* population state;
* accepted-history or lineage state;
* algorithm configuration required for deterministic continuation;
* candidate-evaluation cache references.

It must not serialize:

* callbacks;
* event sinks;
* loggers;
* open file objects;
* scheduler clients;
* live calculator processes.

## Dependency direction

The intended dependency direction is:

```text
generic structure model
        +
interface-domain state
        ↓
InterfaceCandidate
        ↓
manipulation and validated I/O round trips
        ↓
EvaluationResult
        ↓
optimizer policy
        ↓
events and checkpoints
```

Events observe optimizer behavior. Checkpoints preserve optimizer state. Neither owns the scientific structure model.

## Consequences

### Positive

* Domain identity survives file round trips and relaxation.
* I/O, manipulation, evaluation, and checkpoint code can depend on neutral contracts.
* Serialization IDs receive an explicit and limited lifetime.
* Evaluation failures retain scientific and operational context.
* Restart state remains separate from logging and provenance.
* Parallel PR tracks can share stable seams without importing monolithic implementation modules.

### Negative

* More explicit types and conversion boundaries are required.
* Compatibility adapters must temporarily coexist with the new contracts.
* Some data currently carried informally in arrays, filenames, or dictionaries must be normalized.

## Enforcement

The contracts will be introduced through the roadmap PRs rather than in one large change.

Architecture tests should eventually verify that:

* interface-domain types do not import I/O or optimization;
* I/O models do not import minimizers;
* manipulation results contain complete candidates;
* evaluator failures remain typed before penalties are applied;
* journals are not accepted as checkpoints;
* checkpoint persistence does not serialize live runtime objects.

