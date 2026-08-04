# ADR 0004: Events and Journals Are Not Checkpoints

* **Status:** Accepted
* **Date:** 2026-08-04
* **Decision owners:** GBOpt maintainers
* **Related roadmap items:** OBS1–OBS3, CP0–CP5, INT1

## Context

Logging, scientific provenance, optimizer diagnostics, and restart persistence all describe an optimization run, but they serve different contracts.

A JSONL event journal may contain enough information to understand what happened without containing enough state to continue the algorithm deterministically.

Conversely, a checkpoint may contain restart-critical state that should not be emitted as routine logs or events.

Conflating these systems would create several risks:

* an event schema becoming an accidental restart API;
* logs becoming the only record of failed candidate evaluation;
* checkpoint compatibility becoming tied to human-readable output;
* excessive event payloads containing full structures or mutable optimizer state;
* replay claims that cannot be satisfied across external calculators and filesystems;
* multiple workers writing unsafely to a shared journal.

## Decision

GBOpt will maintain three distinct mechanisms:

1. **Logging** for ephemeral diagnostics.
2. **Events and journals** for structured observation and provenance.
3. **Checkpoints** for deterministic continuation from a documented safe boundary.

None substitutes for another.

## Authoritative state

Algorithmically significant facts must first exist in typed state or result objects.

Examples include:

* candidate evaluation status;
* physical objective value;
* failure stage and reason;
* accepted candidate;
* best candidate;
* current temperature;
* generation population;
* RNG state.

An event may report these facts, but the event must not be their only authoritative representation.

## Logging

Logging is intended for:

* terminal or notebook feedback;
* immediate debugging;
* operator awareness;
* low-level troubleshooting.

Library code:

* uses standard Python logging;
* does not call `basicConfig()`;
* does not install application handlers;
* does not redirect requested CLI results from stdout into logs.

Warnings remain warnings when they are caller-visible API conditions.

## Events

`OptimizationEvent` reports semantic lifecycle activity.

Examples include:

* optimization started;
* candidate proposed;
* candidate evaluated;
* candidate accepted or rejected;
* best candidate updated;
* generation completed;
* population reseeded;
* optimization terminated or failed.

Events should contain stable identifiers and small structured fields. They must not contain full structures, large atom arrays, unrestricted environment dumps, callbacks, or live objects.

The default event sink is silent.

## Journal

A journal is an optional durable event sink.

The initial journal format is versioned JSON Lines with:

* UTF-8 encoding;
* one complete JSON object per line;
* stable run ID;
* event schema version;
* documented append, overwrite, flush, and failure policy.

A separate manifest stores run-invariant metadata.

The optimization driver owns the authoritative journal. Scheduler workers should not append freely to one shared JSONL file.

A journal supports:

* provenance review;
* failure analysis;
* lineage inspection;
* campaign aggregation;
* report generation.

It does not guarantee restartability.

## Checkpoint

A checkpoint contains the minimum approved state required to continue an optimizer deterministically at a safe boundary.

Checkpoint state may include:

* optimizer algorithm and schema version;
* run identity;
* RNG bit-generator type and state;
* completed step or generation;
* current and best candidates;
* population candidate order;
* accepted or lineage history;
* convergence-control state;
* candidate-evaluation cache;
* required artifact references.

A checkpoint must not serialize:

* event sinks;
* loggers;
* callbacks;
* open files;
* scheduler clients;
* active calculator processes.

## Safe boundaries

Checkpoint publication occurs only after a documented complete boundary.

Initial boundaries are:

* Monte Carlo: after a completed accepted or rejected step;
* genetic algorithm: after a completed generation;
* candidate cache: after one candidate result has been durably recorded.

A checkpoint must never claim a partially completed optimizer transition is complete.

## Independence

Checkpointing must function when journaling is disabled.

Journaling must function when checkpointing is disabled.

Enabling both must not cause one to derive authoritative state from the other.

Checkpoint lifecycle events may be added only after the checkpoint implementation exists. Such events report checkpoint actions but remain non-authoritative.

## Failure behavior

Durable provenance and restart storage have different failure policies.

* A console logging failure may degrade with minimal disruption.
* A journal failure must follow an explicit configured policy because provenance guarantees may be affected.
* A checkpoint write must be atomic.
* A checkpoint load must reject corruption, unsupported schema versions, incompatible algorithms, and missing required artifacts.
* Missing checkpoint artifacts must not trigger silent reconstruction from unrelated files.

## Consequences

### Positive

* Event schemas can evolve without becoming restart schemas.
* Checkpoints can remain minimal and deterministic.
* Logs can stay readable and operational.
* Journals can be queried without carrying arbitrary live state.
* Scientific failures remain represented even when no journal is configured.

### Negative

* Some identifiers and metadata appear in both event and checkpoint records.
* Separate schemas and tests are required.
* Applications must configure multiple outputs when they need both provenance and restart.

## Enforcement

Integration tests should verify that:

* a journal cannot be loaded as a checkpoint;
* checkpointing works with a null event sink;
* journaling works with checkpointing disabled;
* events contain no full structures;
* checkpoints contain no sinks or callbacks;
* simultaneous logging, journaling, and checkpointing preserve numerical behavior;
* uninterrupted and resumed runs are equivalent at supported safe boundaries.

