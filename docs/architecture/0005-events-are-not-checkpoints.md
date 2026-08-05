# ADR 0005: Events and journals are not checkpoints

- **Status:** Accepted
- **Date:** 2026-08-04
- **Decision owners:** Observability and checkpoint tracks

## Context

Operational diagnostics, durable provenance, and restart state have different
correctness requirements. Treating a log or JSONL event stream as a checkpoint creates
ambiguous ownership and incomplete restart guarantees.

## Decision

`OptimizationEvent` reports an already-authoritative transition. Event sinks may render
live logs or append durable JSONL provenance. Events must not be the only storage for
algorithmically significant state.

`OptimizationSnapshot` is separately versioned restart state published only at an
approved completed safe boundary. It stores the state required for deterministic
continuation and excludes loggers, event sinks, callbacks, open resources, and scheduler
clients.

A journal is never accepted as a checkpoint. Checkpoint lifecycle events may be added
only after checkpoint behavior exists, and remain reports rather than authority.

## Consequences

- OBS2/OBS3 can evolve without changing checkpoint schemas.
- CP1-CP5 can guarantee atomicity and deterministic resume without replaying logs.
- Application code configures sinks; reusable library imports remain silent.
