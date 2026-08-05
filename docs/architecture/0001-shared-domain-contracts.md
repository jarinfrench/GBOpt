# ADR 0001: Shared domain contracts and ownership

- **Status:** Accepted
- **Date:** 2026-08-04
- **Decision owners:** Foundation track

## Context

Persistent interface state is currently distributed across `GBManipulator.py` and
`FileGrainOwnership.py`. I/O, manipulation, evaluation, and checkpointing all need the
same physical state, but none should import another high-level implementation module to
obtain it.

## Decision

The roadmap contracts are authoritative:

| Contract | Owns | Must not own |
|---|---|---|
| `StructureData` | Generic atoms, full cell, origin, periodicity, optional external IDs and charges | Grain identity, calculator execution, optimizer policy |
| `GrainOwnership` | Row-aligned persistent grain labels, authoritative interface plane, physical grain bounds, topology, coordinate tolerance | File syntax, objective values, optimizer history |
| `InterfaceCandidate` | Atoms plus persistent interface state required by optimization | Parsing, calculator execution, restart history |
| `WriteResult` / `StructureArtifact` | One serialization result, format/path/digest, transient ID mapping, declared losses | Persistent atom identity, selection policy |
| `ManipulationResult` | Child candidates, operation identity, parameters, lineage metadata | I/O, evaluation, logging backend |
| `EvaluationResult` | Status, physical result, selection result, artifact, failure stage/reason, reconstructed candidate | MC acceptance or GA population selection |
| `RunContext` | Run identity, resolved seed, algorithm and optional campaign/case identity | Mutable optimizer state |
| `OptimizationSnapshot` | Restartable state at a completed safe boundary | Loggers, callbacks, open files, scheduler clients |
| `OptimizationEvent` | Typed report of an already-authoritative transition | Restart state or otherwise unrecorded algorithmic state |

`InterfaceCandidate` and `GrainOwnership` are domain objects. F2 will move them to a
neutral interface-domain package while preserving compatibility imports and exact class
identity where required.

## Consequences

- I/O, evaluation, manipulation, and checkpoint modules may depend on the neutral domain
  layer.
- The domain layer must not depend on file formats or optimizers.
- Persistent labels remain attached to atom rows even when relaxed atoms cross the
  interface plane.
- Arrays in immutable domain objects remain defensively copied and read-only.
