# ADR 0004: Evaluation failures are typed before optimizer penalties

- **Status:** Accepted
- **Date:** 2026-08-04
- **Decision owners:** Evaluation track

## Context

Legacy scalar and batch GA flows convert evaluator exceptions, malformed output, invalid
paths, reload failures, and ownership failures to `1.0e30`. That preserves selection
behavior but erases why and where the evaluation failed. The explicit-ownership path's
`CandidateEvaluation` is a useful first step but is algorithm-specific.

## Decision

Every evaluation is normalized to an algorithm-neutral typed `EvaluationResult` before
MC or GA applies policy. It records candidate identity, status, physical objective,
selection objective or penalty, artifact identity, reconstructed candidate where
available, and structured failure stage/code/message.

A penalty remains optimizer policy. It is applied **after** the failure result exists and
must not replace the original failure provenance.

Legacy tuple callbacks and batch dictionaries remain supported through adapters until a
separately approved compatibility change.

## Consequences

- Scalar, batch, local, subprocess, and scheduler-backed evaluators share one validation
  boundary.
- Selection uses `selection_energy`; scientific reporting retains the physical result or
  failure.
- Logging, events, journals, and checkpoints consume authoritative typed results rather
  than reconstructing outcomes from prose or filenames.
