# GBOpt Logging, Events, and Run-Provenance Plan

## Status

**Recommended long-term architecture, revised against the current GBOpt source.**


**Implementation authority:** `../MASTER_PLAN.md`  
**Roadmap mapping:** `OBS1`, `EVAL1`, `EVAL2`, `OBS2`, and `OBS3`  
**Role of this document:** architectural rationale and component-level acceptance criteria; it does not override roadmap sequencing or prerequisites.

This document supersedes the earlier logging-only implementation plan. It retains the useful parts of that plan—standard-library logging, preserved warning semantics, structured fields, and no import-time configuration—but broadens the design to match the direction of the GBOpt codebase.

The central recommendation is:

> **Represent algorithmically significant outcomes as typed results, represent optimization activity as typed domain events, and expose logging and durable journaling as replaceable event sinks configured by applications.**

This keeps live diagnostics, scientific provenance, optimizer behavior, command output, and warning semantics separate.

---

## 1. Objectives

The revised design should:

1. Improve terminal and notebook diagnostics without changing numerical behavior.
2. Preserve the distinction among warnings, exceptions, logs, CLI results, and scientific records.
3. Retain the reason and stage when candidate evaluation fails.
4. Support Monte Carlo, genetic, and future minimizer implementations through a common event vocabulary.
5. Work with scalar, batch, local, subprocess, and scheduler-backed evaluators.
6. Preserve stable run, candidate, artifact, and ownership identity across serialization boundaries.
7. Allow applications to select human-readable logging, JSONL journaling, notebook progress reporting, or multiple outputs at once.
8. Avoid a package-wide logging singleton or heavyweight observability manager.
9. Provide a gradual implementation path that does not block the current interface-refactoring work.

---

## 2. Current-Code Assessment

### 2.1 Elements of the previous plan that remain valid

The following recommendations remain appropriate:

- Use the Python standard-library `logging` module as the initial live-diagnostic backend.
- Use per-module loggers through `logging.getLogger(__name__)`.
- Do not call `logging.basicConfig()` or install handlers from library import paths.
- Preserve existing `warnings.warn(...)` behavior.
- Use short, stable event names and structured scalar fields.
- Keep frequent candidate- or mutation-level activity at `DEBUG`.
- Use `INFO` for meaningful lifecycle milestones and generation summaries.
- Test event names, levels, and selected fields rather than exact formatted strings.
- Do not add `structlog`, Eliot, Logbook, picologging, OpenTelemetry, or another framework without a demonstrated need.

### 2.2 Corrections required for the current source

#### `gb_params.py` stdout is a data contract

`GBOpt/Utils/gb_params.py` emits JSON and human-formatted query results through direct `print(...)` calls. Current tests parse stdout as a single JSON document.

Therefore:

- requested command results must remain on stdout;
- diagnostics and configured logs must go to stderr;
- logging must never be allowed to corrupt JSON output;
- converting the final command result to an `INFO` log would be incorrect.

`gb_params.py` should be excluded from the first logging patch except for carefully separated diagnostics.

#### The former initial-structure script has moved

The current example script is:

```text
examples/gb_optimization/scripts/make_initial.py
```

Its successful `Written initial.dat ...` message is user-facing script output, not problematic library output. It may remain until the examples adopt a deliberate command-output policy such as `--quiet` and `--verbose`.

#### Checkpoint events are premature

The earlier plan proposed `checkpoint_saved` and `checkpoint_loaded` events, but the current minimizer implementation does not provide a checkpoint system. The revised plan must not specify logging tests or events that imply nonexistent functionality.

Checkpoint events should be introduced only with an explicit checkpoint/restart design.

#### The largest observability gap is swallowed GA failure context

The current legacy GA evaluation path converts several failure conditions to the penalty energy without retaining the original reason:

- scalar evaluator exceptions;
- missing or invalid output paths;
- structure reload failures;
- malformed or incomplete batch results.

The explicit-ownership path already has a `CandidateEvaluation` record with `success` and `failure_reason`, which is a better foundation. The long-term design should normalize both legacy and explicit-ownership paths around typed evaluation results.

#### MC run identity and seed handling need correction

The current Monte Carlo method uses `uuid.uuid4()` as a function default, which resolves at function-definition time rather than once per run. The constructor also does not retain the resolved time-derived seed.

Before reliable run-context reporting is added:

- change the MC run identifier default to `None` and resolve it inside `run_MC()`;
- retain the resolved RNG seed as an attribute or immutable run-context field;
- use the resolved identifier and seed consistently in events and artifacts.

These are correctness and reproducibility improvements, not merely formatting changes.

---

## 3. Output-Channel Policy

GBOpt should establish the following explicit policy.

| Channel | Purpose | Examples |
|---|---|---|
| Return objects and state | Algorithmically significant outcomes | energy, candidate status, artifact identity, feasibility result |
| Exceptions | Operations that cannot satisfy their contract | invalid evaluator result, unrecoverable reload failure |
| Python warnings | Caller-visible, non-fatal API conditions | deprecated usage, questionable but accepted input |
| Logging | Ephemeral operational diagnostics and progress | run started, generation summary, evaluator failure detail |
| Stdout | A CLI command's requested result | JSON query response, final scalar/report |
| Stderr | CLI diagnostics and configured console logs | progress, warnings, troubleshooting context |
| Durable journal and manifest | Reproducibility and post-run analysis | normalized events, seed, configuration, candidate lineage |
| Structure/artifact files | Scientific and calculator artifacts | LAMMPS data/dump, future VASP/QE/GULP files |

No one channel should silently substitute for another.

In particular:

- logs are not an authoritative scientific result;
- warnings are not progress messages;
- stdout is not a general diagnostic stream;
- penalty energies must not be the sole representation of evaluation failure;
- a JSONL journal is not automatically a checkpoint system.

---

## 4. Recommended Architecture

### 4.1 Architectural layers

```text
GBMaker / GBManipulator / I/O adapters
        |
        | typed structures, mutations, artifacts, diagnostics
        v
Evaluator abstraction
        |
        | EvaluationResult
        v
Minimizer abstraction
        |
        | OptimizationEvent
        v
EventSink
   |             |                 |
   v             v                 v
stdlib logs   JSONL journal   notebook/progress UI
```

The authoritative data flow is through typed objects. Events describe that flow. Sinks determine how events are presented or retained.

### 4.2 No central logging manager

Do not create a singleton or manager that owns logging for the package.

Instead:

- library modules use normal module loggers where direct logging is still appropriate;
- minimizers emit through a narrow optional event-sink protocol;
- applications and campaign scripts configure concrete sinks;
- the default sink is silent and has no global side effects.

---

## 5. Typed Evaluation Results

### 5.1 Principle

Any fact that changes optimizer behavior must be represented in a result object, state object, or exception before it is logged.

A log record must never be the only place that records:

- whether a candidate succeeded;
- why it failed;
- which artifact was produced;
- whether ownership reconstruction succeeded;
- whether an energy was valid;
- whether a penalty was substituted.

### 5.2 Recommended result model

The current `CandidateEvaluation` can evolve toward an algorithm-neutral result such as:

```python
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping, TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class EvaluationStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INVALID = "invalid"
    CANCELLED = "cancelled"


class FailureStage(str, Enum):
    INPUT_VALIDATION = "input_validation"
    SUBMISSION = "submission"
    CALCULATOR_EXECUTION = "calculator_execution"
    RESULT_PARSING = "result_parsing"
    ARTIFACT_VALIDATION = "artifact_validation"
    STRUCTURE_RELOAD = "structure_reload"
    OWNERSHIP_RECONSTRUCTION = "ownership_reconstruction"
    ENERGY_VALIDATION = "energy_validation"


@dataclass(frozen=True, slots=True)
class StructureArtifact:
    path: Path
    format: str
    structure_id: str
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    candidate_id: str
    status: EvaluationStatus
    energy: float | None
    artifact: StructureArtifact | None
    failure_stage: FailureStage | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
```

The exact public API should be finalized with the evaluator and minimizer abstractions. The essential requirements are stable status, failure stage, failure reason, energy, and artifact identity.

### 5.3 Compatibility adapters

Existing evaluator callback contracts should not be broken in one step.

Provide adapters for existing forms such as:

```python
(energy, dump_file)
```

and:

```python
{"energy": ..., "final_dump": ...}
```

The adapters should normalize these values into `EvaluationResult` and perform validation in one place.

### 5.4 Penalty semantics

A penalty energy may remain an optimizer policy, but it should be applied after a typed failure result is created.

Conceptually:

```python
result = evaluator.evaluate(candidate)
selection_energy = result.energy if result.status is SUCCEEDED else penalty
```

The original result remains available for diagnostics and journaling.

---

## 6. Run Context and Identity

### 6.1 Immutable run context

Introduce an immutable context for values shared across an optimization run:

```python
@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str
    seed: int
    algorithm: str
    case_id: str | None = None
    boundary_spec_id: str | None = None
    campaign_id: str | None = None
```

Potential future manifest fields include:

- GBOpt version;
- source revision;
- evaluator backend;
- calculator and potential identifiers;
- input structure identifier;
- normalized configuration hash;
- ownership-representation version;
- host or scheduler job identifiers.

Run-invariant data should be stored once in a manifest rather than duplicated on every event.

### 6.2 Candidate identity and lineage

Candidate events should carry stable candidate identity and explicit lineage:

```text
candidate_id
parent_candidate_ids
generation
population_index
mutation_name
mutation_parameters
```

LAMMPS atom IDs must not be treated as persistent candidate or atom identity. They are serialization identifiers unless a separate ownership/identity contract explicitly preserves their meaning.

### 6.3 Artifact identity

Events should reference typed artifacts rather than infer meaning from filenames. Artifact metadata should include at least:

- stable structure/artifact ID;
- path;
- format;
- producing candidate ID;
- ownership metadata or mapping reference when required;
- optional content hash where practical.

---

## 7. Typed Optimization Events

### 7.1 Event record

A minimal event model is:

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping


@dataclass(frozen=True, slots=True)
class OptimizationEvent:
    schema_version: int
    event: str
    timestamp: datetime
    run_id: str
    component: str
    fields: Mapping[str, JSONValue]
```

The timestamp should be generated by the event-emission boundary, preferably as timezone-aware UTC.

### 7.2 Event sink protocol

```python
from typing import Protocol


class EventSink(Protocol):
    def emit(self, event: OptimizationEvent) -> None:
        ...
```

Recommended implementations:

```python
class NullEventSink:
    ...

class LoggingEventSink:
    ...

class JsonlEventSink:
    ...

class CompositeEventSink:
    ...
```

A minimizer receives an optional sink and defaults to `NullEventSink`.

### 7.3 Common event vocabulary

Prefer algorithm-neutral lifecycle events:

```text
optimization_started
initial_candidate_evaluated
candidate_proposed
candidate_evaluation_started
candidate_evaluated
candidate_accepted
candidate_rejected
best_candidate_updated
generation_started
generation_completed
population_reseeded
optimization_terminated
optimization_failed
```

Use fields such as `algorithm="monte_carlo"` or `algorithm="genetic"` rather than embedding the algorithm in every event name.

Algorithm-specific events remain acceptable where they represent a real domain concept, but the common lifecycle should be shared.

### 7.4 Recommended fields

#### Base fields

```text
event
schema_version
timestamp
component
run_id
algorithm
```

#### Candidate fields

```text
candidate_id
candidate_index
parent_candidate_ids
generation
step
mutation_name
mutation_parameters
```

#### Evaluation fields

```text
status
energy
selection_energy
best_energy
delta_energy
artifact_id
artifact_path
failure_stage
failure_code
failure_type
failure_message
```

#### Monte Carlo fields

```text
accepted
temperature
acceptance_probability
rejection_count
max_rejections
energy_tolerance
cooldown_rate
```

#### Genetic algorithm fields

```text
population_size
valid_candidate_count
failed_candidate_count
selected_parent_count
elite_count
reseeded_candidate_count
```

#### Termination fields

```text
termination_reason
completed_steps
completed_generations
best_candidate_id
best_energy
```

### 7.5 Field constraints

- Use scalar JSON-compatible values or small mappings/lists.
- Do not emit atom arrays, full structures, or large calculator outputs.
- Reference large data through artifact identifiers and paths.
- Keep event and field names stable once the journal schema is published.
- Do not include secrets, credentials, or unrestricted environment dumps.

---

## 8. Live Logging

### 8.1 Purpose

Live logging is for:

- terminal progress;
- notebook feedback;
- immediate debugging;
- operator awareness;
- troubleshooting failed evaluations.

It is not the authoritative run record.

### 8.2 Backend and configuration

Use standard-library `logging` initially.

Library requirements:

- use `logging.getLogger(__name__)`;
- do not call `basicConfig()`;
- do not install console or file handlers on import;
- do not force a format or log level on the host application;
- optionally install a `NullHandler` only at the package namespace boundary if needed to support older embedding patterns.

Application and campaign requirements:

- configure stderr console logging explicitly;
- choose human-readable or JSON formatting;
- set verbosity through command-line or configuration options;
- add file handlers only when the application owns the destination and retention policy.

### 8.3 Log levels

| Level | Intended use |
|---|---|
| `DEBUG` | individual candidate proposals, mutations, acceptance decisions, normalized failure details |
| `INFO` | run start, initial evaluation, generation summary, new best candidate, normal termination |
| `WARNING` | abnormal but recoverable run-level condition, such as an entirely invalid generation followed by reseeding |
| `ERROR` | run or major subsystem is about to abort; log immediately before raising only when useful context would otherwise be lost |

Avoid logging every hot-loop operation at `INFO`.

### 8.4 Human-readable format

A console sink may render an event as:

```text
INFO GBOpt.GBMinimizer generation_completed | run_id=... generation=4 valid_candidate_count=18 failed_candidate_count=2 best_energy=-3.58
```

Formatting is a sink concern. Core event producers should not construct presentation strings containing all context.

---

## 9. Durable Run Journal and Manifest

### 9.1 Purpose

A durable journal supports:

- reproducibility;
- post-run failure analysis;
- campaign aggregation;
- candidate lineage inspection;
- provenance review;
- later report generation.

It should answer questions such as:

- Why did a candidate receive a penalty?
- At which stage did evaluation fail?
- Which parent candidates produced the best structure?
- Which seed and configuration generated the run?
- How many candidates failed per generation?
- Was a population reseeded because all candidates were invalid?
- Which artifact corresponds to the reported minimum?

### 9.2 Recommended initial format

Use versioned JSON Lines for the event journal:

```json
{"schema_version":1,"event":"generation_completed","timestamp":"...","run_id":"...","component":"gbminimizer","fields":{"generation":4,"valid_candidate_count":18,"failed_candidate_count":2}}
```

Use a separate run manifest for invariant metadata:

```json
{
  "schema_version": 1,
  "run_id": "...",
  "algorithm": "genetic",
  "seed": 12345,
  "gbopt_version": "...",
  "source_revision": "...",
  "configuration": {},
  "input_artifacts": []
}
```

### 9.3 Journal guarantees

The initial journal implementation should define:

- UTF-8 encoding;
- one complete JSON object per line;
- explicit schema version;
- flush behavior;
- behavior on write failure;
- whether journal failure aborts the optimization or degrades to logging;
- path ownership and overwrite/append policy;
- deterministic serialization for tests where practical.

Atomicity should be considered at the record level. The journal should not claim to be a transactional database.

### 9.4 Not a checkpoint system

The journal records what happened. It does not necessarily contain enough state to restart an optimizer.

Checkpoint/restart requires a separate design for:

- optimizer state;
- RNG state;
- current population or accepted state;
- evaluator state, if any;
- artifact completeness;
- version compatibility;
- atomic checkpoint publication.

Do not add checkpoint event names until that functionality exists.

---

## 10. Parallel and Scheduler-Backed Evaluation

The examples and campaign work include batch and scheduler-oriented execution. The observability design should avoid assuming a single local process.

Recommended ownership model:

- the optimization driver owns the authoritative run journal;
- workers return typed `EvaluationResult` data or write candidate-local result artifacts;
- worker stdout and stderr remain job-specific diagnostics;
- the driver validates and records normalized candidate results after collection;
- worker events, if later required, are written to separate shards and merged deterministically;
- multiple workers should not append freely to one shared JSONL file on a distributed filesystem.

This avoids interleaved writes, file-lock assumptions, and ambiguous ownership of partial records.

Scheduler metadata such as job ID may be included in result metadata or the run manifest, but scheduler-specific types should not leak into the core minimizer API.

---

## 11. Placement Across GBOpt Abstractions

### 11.1 Minimizer abstraction

The minimizer owns optimization policy and should be the primary event producer.

It should emit:

- lifecycle events;
- candidate proposal and normalized result events;
- acceptance and selection decisions;
- generation summaries;
- best-candidate updates;
- recovery and reseeding events;
- termination and failure events.

The common minimizer layer may provide a protected helper:

```python
def _emit(self, event: str, **fields: JSONValue) -> None:
    ...
```

Concrete algorithms use the same emission path.

### 11.2 Evaluator abstraction

Evaluators should return `EvaluationResult` or an equivalent common contract.

They may produce low-level operational logs, but the minimizer or orchestration layer should emit the normalized semantic candidate result.

The result contract should distinguish evaluator invocation, calculator execution, output parsing, artifact validation, reload, ownership reconstruction, and energy validation.

### 11.3 Manipulator abstraction

Routine manipulations should return structured operation metadata rather than rely on internal logging:

```python
@dataclass(frozen=True, slots=True)
class MutationResult:
    operation: str
    parameters: Mapping[str, JSONValue]
    atoms: np.ndarray
    grain_labels: np.ndarray | None
```

The minimizer can then emit `candidate_proposed` with consistent mutation metadata.

Existing caller-visible questionable conditions should remain Python warnings where that behavior is already part of the interface.

### 11.4 I/O abstraction

I/O adapters should return typed artifact metadata and preserve required ownership information.

Useful events are coarse, meaningful boundaries such as:

```text
artifact_write_failed
artifact_validation_failed
artifact_reload_failed
ownership_reconstruction_failed
```

Do not log every parser operation or array conversion.

The design must remain calculator- and format-neutral; LAMMPS paths and dump terminology should not define the common abstraction.

### 11.5 GBMaker decomposition

As `GBMaker` is decomposed, a small number of construction-stage events may be useful:

```text
construction_started
supercell_enumerated
boundary_assembled
feasibility_evaluated
construction_completed
```

Construction outcomes and feasibility diagnostics must remain typed return data. Logging should not become the only explanation for construction failure.

### 11.6 Campaign tooling

Campaign code should own:

- console logging configuration;
- verbosity and formatting options;
- run-directory layout;
- manifest and journal destinations;
- candidate/job diagnostic paths;
- aggregation and reporting;
- retention and cleanup policy.

The reusable `GBOpt` package should remain silent unless the host application configures a sink or logger.

---

## 12. Alternatives Considered

### 12.1 Direct stdlib logging throughout the algorithms

**Advantages**

- smallest immediate patch;
- no dependency;
- familiar tooling;
- easy `caplog` testing.

**Disadvantages**

- couples algorithm code to one presentation mechanism;
- makes notebook/progress integration awkward;
- encourages duplicated call sites for logging and journaling;
- does not itself preserve swallowed evaluator failures;
- event consistency relies on individual call-site discipline.

**Decision**

Use direct module logging only for the narrow initial cleanup. Do not make it the final cross-cutting architecture.

### 12.2 Typed events with an observer/event sink

**Advantages**

- separates algorithm semantics from presentation;
- supports logging, JSONL, tests, notebooks, and future telemetry;
- fits the arbitrary-minimizer abstraction;
- allows a default no-op implementation;
- enables stable event-schema tests.

**Disadvantages**

- introduces a new internal protocol;
- requires decisions about event versioning and field ownership;
- can be overbuilt if applied indiscriminately to every low-level operation.

**Decision**

Recommended long-term architecture. Keep the protocol narrow and initially provisional.

### 12.3 Persistent JSONL logging only

**Advantages**

- machine-readable;
- useful for campaign analysis;
- easy initial implementation.

**Disadvantages**

- conflates live diagnostics with scientific provenance;
- does not establish typed in-memory results;
- can become an accidental checkpoint contract;
- distributed writers complicate correctness.

**Decision**

Use JSONL as an optional event sink after typed results and events exist, not as the core interface.

### 12.4 `structlog`

**Advantages**

- convenient contextual binding;
- mature structured rendering;
- good JSON support.

**Disadvantages**

- adds a dependency before event semantics are settled;
- does not define candidate identity, failure stages, artifacts, or evaluator contracts;
- still requires an application configuration policy.

**Decision**

Do not adopt now. It may later implement a logging sink if standard-library adapters and formatters become cumbersome.

### 12.5 Eliot or full workflow tracing

**Advantages**

- stronger nested-action and causal tracing;
- useful for complex distributed workflows.

**Disadvantages**

- substantial conceptual and implementation shift;
- more invasive than current needs justify;
- does not remove the need for typed domain results.

**Decision**

Reconsider only if GBOpt becomes a tracing-heavy distributed workflow system.

### 12.6 OpenTelemetry

**Advantages**

- integration with institutional telemetry systems;
- distributed traces and metrics.

**Disadvantages**

- operationally heavy;
- service-oriented concepts exceed current package needs;
- introduces deployment and exporter concerns.

**Decision**

Not appropriate now. A future OpenTelemetry event sink remains possible.

### 12.7 Full event sourcing

**Advantages**

- theoretical replay and auditability;
- complete state-transition history.

**Disadvantages**

- deterministic replay would require control of RNG, calculators, external executables, filesystems, implementation versions, and artifacts;
- much more complex than provenance journaling;
- risks making event schema changes equivalent to state-migration changes.

**Decision**

Do not adopt. The journal is an audit and provenance record, not the sole authoritative optimizer state.

### 12.8 Package-wide logging singleton or manager

**Advantages**

- superficially centralized configuration.

**Disadvantages**

- global mutable state;
- difficult embedding and testing;
- confuses application and library responsibilities;
- unnecessary with event sinks and normal Python logging.

**Decision**

Reject.

---

## 13. Roadmap alignment and phased implementation

The roadmap owns the actual PR boundaries:

| Roadmap PR | Scope represented in this document |
|---|---|
| `OBS1` | narrow correctness, seed/run-ID fixes, and standard-library logging cleanup |
| `EVAL1` | algorithm-neutral evaluation and artifact contracts |
| `EVAL2` | normalized MC and GA evaluation flows |
| `OBS2` | run context and typed event protocol |
| `OBS3` | durable JSONL journal and run manifest |

The phases below preserve the design reasoning. Implementation must use the roadmap PR identifiers above rather than creating a separate logging-plan branch sequence.

### Phase 0 — Policy and terminology

Document and agree on:

- output-channel policy;
- warning-versus-log semantics;
- event naming conventions;
- candidate, run, and artifact identity;
- failure stages;
- which interfaces are provisional versus public.

**Deliverable:** this document plus a concise contributor-facing policy.

### Phase 1 — Narrow correctness and diagnostic cleanup

Tasks:

1. Change MC `unique_id` to default to `None` and resolve it per invocation.
2. Retain the resolved RNG seed.
3. Replace the two MC `print(...)` termination messages with module logging.
4. Add run-start, initial-evaluation, best-update, and termination logs.
5. Preserve legacy GA evaluator exception and reload-failure details before applying penalties.
6. Add generation-level summaries.
7. Leave `gb_params.py` result output and `make_initial.py` success output unchanged.

This phase may use direct module logging because its scope is deliberately small.

**Non-goals:** event protocol, JSONL, checkpointing, broad GBMaker/GBManipulator logging.

### Phase 2 — Normalize evaluator results

Tasks:

1. Define an internal or provisional `EvaluationResult` contract.
2. Define failure stages and stable failure codes where useful.
3. Add adapters for scalar tuple evaluators and batch dictionary evaluators.
4. Normalize explicit-ownership and legacy paths.
5. Ensure penalties are an optimizer policy applied after the result is recorded.
6. Preserve backward compatibility for existing evaluator callbacks.

**Deliverable:** one normalized evaluation pathway with focused compatibility tests.

### Phase 3 — Add run context and event protocol

Tasks:

1. Add `RunContext`.
2. Add `OptimizationEvent` and `EventSink`.
3. Add `NullEventSink`, `LoggingEventSink`, and `CompositeEventSink`.
4. Provide a protected minimizer emission helper.
5. Translate existing minimizer logs to emitted semantic events.
6. Keep the protocol internal or explicitly provisional until exercised by both MC and GA.

**Deliverable:** common event emission across current minimizers without global configuration.

### Phase 4 — Integrate with minimizer and evaluator abstractions

Tasks:

1. Make run context and optional event sink part of common minimizer orchestration.
2. Define common lifecycle event names.
3. Ensure concrete minimizers provide algorithm-specific fields without redefining common semantics.
4. Align mutation metadata with the future manipulator result contract.
5. Align artifact metadata with the future I/O abstraction.

**Deliverable:** an abstraction-compatible event design rather than MC/GA-specific instrumentation.

### Phase 5 — Add durable journaling

Tasks:

1. Add a versioned `JsonlEventSink`.
2. Add a run-manifest writer owned by the application/campaign layer.
3. Define write, flush, append, overwrite, and failure policies.
4. Ensure structures and large outputs are referenced as artifacts.
5. Test truncated-last-record handling for readers if a reader is provided.
6. Keep checkpoint/restart explicitly out of scope.

**Deliverable:** opt-in, durable, queryable run history.

### Phase 6 — Campaign and scheduler integration

Tasks:

1. Configure console and journal sinks from campaign entry points.
2. Establish per-run and per-candidate diagnostic locations.
3. Have the driver record normalized worker results.
4. Avoid shared multi-writer JSONL files.
5. Add campaign aggregation based on manifests and journals rather than scraping prose logs.

**Deliverable:** coherent local and scheduler-backed observability.

### Phase 7 — Selective expansion

Add events to other modules only where a concrete operational or provenance need exists.

Potential targets:

- coarse GBMaker construction stages;
- I/O artifact boundaries;
- ownership reconstruction failures;
- expensive feasibility-validation stages.

Avoid:

- per-atom events;
- array dumps;
- blanket replacement of warnings;
- routine low-level parser chatter;
- mechanical instrumentation with no diagnostic value.

### Phase 8 — Reevaluate tooling

Consider `structlog`, OpenTelemetry, or another sink only when evidence shows a need such as:

- standardized JSON output across many applications;
- cumbersome standard-library contextual binding;
- institutional telemetry integration;
- distributed trace correlation.

Any migration should preserve the domain event and result contracts.

---

## 14. Testing Strategy

### 14.1 General principles

- Test GBOpt behavior and event semantics, not Python logging internals.
- Use deterministic evaluator fakes and fixed seeds.
- Do not assert complete human-formatted strings.
- Confirm that instrumentation does not change numerical outcomes.
- Confirm that default library use installs no global handlers and emits no unsolicited console output.

### 14.2 Immediate logging tests

Validate:

- MC termination produces the expected event/log level and reason;
- run IDs are resolved per invocation;
- the resolved seed is retained;
- GA evaluator exceptions retain exception type/message at `DEBUG` or in a typed result;
- invalid paths and reload failures are distinguishable;
- generation summaries report valid and failed counts;
- existing warning tests remain unchanged;
- `gb_params.py` JSON stdout remains parseable with logging configured.

### 14.3 Evaluation-result tests

Validate adapters for:

- valid scalar tuple results;
- evaluator exceptions;
- invalid energy values;
- missing output files;
- malformed batch entries;
- structure reload failure;
- ownership reconstruction failure;
- ordering and input-index alignment;
- penalty application without loss of the original failure result.

### 14.4 Event-sink tests

Validate:

- `NullEventSink` is inert;
- `CompositeEventSink` preserves emission order;
- `LoggingEventSink` maps event fields and levels correctly;
- optional fields do not break formatting;
- event timestamps and schema versions are present;
- sink exceptions follow a documented policy.

### 14.5 Journal tests

Validate:

- one valid JSON object per line;
- UTF-8 output;
- schema version in every record;
- stable run ID association;
- no large arrays embedded;
- append/overwrite policy;
- write-failure behavior;
- manifest and journal consistency;
- single-writer ownership in scheduler-oriented tests.

---

## 15. Documentation Requirements

Document for users:

- GBOpt does not auto-configure logging;
- warnings remain warnings;
- how to enable console logs in scripts and notebooks;
- how stdout and stderr are used by CLIs;
- how to opt into a durable journal;
- which event schema version is produced;
- that the journal is not a restart checkpoint.

Document for contributors:

- when to return typed diagnostics instead of logging;
- when to warn, log, or raise;
- approved common event names and field names;
- prohibition on arrays and full structures in events;
- identity and lineage requirements;
- compatibility expectations for evaluator adapters;
- application ownership of handlers, file destinations, and retention.

---

## 16. Risks and Mitigations

### Risk: over-engineering before abstractions stabilize

**Mitigation:** use direct logging only for the narrow first phase; keep event types internal/provisional until exercised by multiple minimizers.

### Risk: duplicate sources of truth

**Mitigation:** typed results and state are authoritative; events report those outcomes rather than recreate them independently.

### Risk: noisy or expensive instrumentation

**Mitigation:** use coarse lifecycle events at `INFO`, candidate detail at `DEBUG`, and artifact references instead of data dumps.

### Risk: breaking CLI output

**Mitigation:** preserve stdout as the requested-result channel and route diagnostics to stderr.

### Risk: journal mistaken for checkpointing

**Mitigation:** state explicitly that restart requires a separate contract; do not add checkpoint events prematurely.

### Risk: distributed file corruption

**Mitigation:** authoritative single-writer journal owned by the driver; worker-local diagnostics or deterministic shards only.

### Risk: unstable public schema

**Mitigation:** version events and manifests; keep the first event interface provisional until MC, GA, evaluator, and campaign integration have validated it.

### Risk: swallowed sink failures

**Mitigation:** define sink-failure policy explicitly. Console sink failure may degrade silently or minimally; durable-journal failure should be configurable and visible because it affects provenance guarantees.

---

## 17. Recommended Initial Scope

The first implementation should remain intentionally narrow:

- correct MC run-ID resolution;
- retain the resolved seed;
- add module-level stdlib logging to `GBOpt/GBMinimizer.py`;
- replace the two MC prints;
- retain legacy GA failure reason and failure stage;
- add generation summaries and termination reasons;
- add focused `caplog` and behavior-preservation tests;
- update documentation of stdout, stderr, warning, and logging semantics.

Do **not** include in the first patch:

- `gb_params.py` result conversion;
- checkpoint events;
- a public event framework;
- JSONL journaling;
- broad GBMaker or GBManipulator instrumentation;
- `structlog` or another dependency.

The second major step should be evaluator-result normalization, not formatter sophistication.

---

## 18. Decision Summary

### Immediate direction

- Use standard-library logging for narrow live diagnostics.
- Preserve warnings and CLI result output.
- Fix MC run identity and seed retention.
- Stop discarding GA candidate failure context.
- Log meaningful lifecycle and generation events without changing numerical behavior.

### Long-term direction

- Make typed results the authoritative representation of evaluation and mutation outcomes.
- Make typed, versioned optimization events the common observability boundary.
- Provide logging, JSONL journaling, and notebook/progress reporting as replaceable sinks.
- Configure all concrete sinks from applications and campaign tooling.
- Keep the design format-, calculator-, scheduler-, and minimizer-neutral.

### Final recommendation

> GBOpt should not build a larger logging framework. It should build a small domain-event boundary around typed optimizer results, use standard logging for ephemeral diagnostics, and add a separate opt-in journal for durable scientific provenance.
