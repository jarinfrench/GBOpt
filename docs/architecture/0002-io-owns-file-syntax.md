# ADR 0002: I/O owns file syntax and transient serialization identity

- **Status:** Accepted
- **Date:** 2026-08-04
- **Decision owners:** I/O track

## Context

LAMMPS parsing and writing currently span `GBMaker`, `Parent`, and
`FileGrainOwnership`. The repaired ownership-aware evaluator path depends on atom IDs,
but those IDs identify rows only within one serialized candidate round trip.

## Decision

Format adapters own representation conversion. A neutral `StructureData` owns generic
atomic and cell data. `GrainOwnership` owns persistent interface identity and topology.
`Parent` composes and interprets those objects but does not own file syntax.

LAMMPS atom IDs are **transient serialization identifiers**, never optimizer-wide atom
identity. Each writer call returns a candidate-local mapping through `WriteResult`; that
mapping is invalid for another candidate or another write.

Explicit ownership reload failures must fail. They must not fall back to geometric
left/right inference from current coordinates.

## Consequences

- IO1-IO3 move LAMMPS readers and writer behavior behind neutral contracts while
  preserving compatibility functions and `GBMaker.write_lammps()`.
- IO4 removes syntax parsing from `Parent` by delegating to readers.
- IO5 establishes one authoritative candidate loader for evaluator and checkpoint
  round trips.
- Loss of ownership, topology, charges, IDs, or cell information must eventually be
  declared or rejected rather than silently accepted.
