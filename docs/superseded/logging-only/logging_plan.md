> **SUPERSEDED**
>
> Replaced by [`../../design/GBOPT_LOGGING_AND_OBSERVABILITY_PLAN.md`](../../design/GBOPT_LOGGING_AND_OBSERVABILITY_PLAN.md). This logging-only plan is retained for historical rationale and must not control implementation sequencing.

# Logging Implementation Plan for GBOpt

## Goal

Introduce logging in GBOpt using the Python standard library `logging` module in
a way that:

- improves human-readable diagnostics in terminals and notebooks
- preserves current library behavior and warning semantics
- avoids global logging side effects in library code
- keeps log call sites structured enough that migrating to `structlog` later is
  straightforward

This plan is intentionally phased so the repo can gain value early without
taking on an unnecessary framework migration up front.

## Current State

The codebase currently has three distinct output patterns:

1. Library warnings via `warnings.warn(...)`
2. Progress or status messages via `print(...)`
3. Script/utility output via `print(...)`

Observed current uses:

- `GBOpt/GBMaker.py`: warning-oriented library behavior
- `GBOpt/GBManipulator.py`: warning-oriented library behavior
- `GBOpt/GBMinimizer.py`: progress/status `print(...)`
- `GBOpt/Utils/gb_params.py`: script-style `print(...)`, including stderr output
- `write_initial_structure.py`: script-style `print(...)`
- `conftest.py`: pytest warning display formatting via `print(...)`

This means GBOpt should not treat logging as a blanket replacement for all
output. The library/script distinction should remain explicit.

## Design Principles

### 1. Keep warnings as warnings

Do not replace existing `warnings.warn(...)` calls in core library modules just
to make output uniform.

Reasons:

- tests already assert warning behavior
- warnings are part of the caller-visible API for non-fatal issues
- callers may filter, escalate, or capture warnings directly

Logging should complement warnings, not erase them.

### 2. Use `logging` as the backend

Use the standard library `logging` module for the first implementation.

Reasons:

- no new dependency
- standard library fit for reusable packages
- easiest low-risk adoption for this repo
- easiest to keep default output human-readable
- migration to `structlog` later remains feasible if call sites are structured

### 3. Avoid a heavyweight manager class

Do not build a central class that owns logging for the package.

Instead:

- use per-module loggers via `logging.getLogger(__name__)`
- add a small helper module for configuration and shared formatting
- use thin helpers or adapters for repeated contextual fields

This keeps the architecture simple and avoids threading a custom logging object
through constructors and APIs.

### 4. Treat logs as events with fields

Write log calls so they are structured from day one, even though the initial
formatter is human-readable.

That means each log should have:

- a short, stable message
- a predictable set of key-value fields

Example shape:

- message: `"Accepted mutation"`
- fields: `event="mutation_accepted"`, `step=12`, `mutation="translate"`,
  `energy=-3.42`, `best_energy=-3.58`

This gives readable output now and a clean path to JSON or `structlog` later.

### 5. Never configure global logging in library import paths

Core GBOpt modules should never call `logging.basicConfig()` or otherwise alter
process-global logging configuration on import.

Only script/CLI/notebook entry points should opt in to console logging setup.

## Recommended Architecture

Add a small helper module, for example:

- `GBOpt/Utils/logging_utils.py`

Responsibilities:

- define the package logger namespace conventions
- provide a console formatter for human-readable logs
- provide an optional `configure_logging(...)` helper for scripts/examples
- provide thin helpers for structured `extra` fields

Non-responsibilities:

- no global stateful manager object
- no ownership of warning behavior
- no mandatory configuration required for library use

### Proposed public helper surface

This is a design target, not exact code yet:

- `get_logger(name: str) -> logging.Logger`
- `configure_logging(level="INFO", stream=None, include_fields=True) -> None`
- `make_run_adapter(logger, **context) -> logging.LoggerAdapter`

If adapters feel too heavy in practice, helper functions that build `extra`
dictionaries are fine. The important part is consistent field naming.

## Event Schema

Standardize a small event schema up front. Keep it compact and stable.

Recommended base fields:

- `component`: high-level subsystem such as `gbminimizer`, `gbmaker`,
  `gbmanipulator`, `gb_params`
- `event`: machine-stable event name
- `unique_id`: run or artifact identifier when applicable
- `seed`: RNG seed when applicable

Recommended optimization/run fields:

- `step`
- `mutation`
- `accepted`
- `energy`
- `best_energy`
- `delta_energy`
- `temperature`
- `rejection_count`
- `max_rejections`
- `cooldown_rate`

Recommended file/checkpoint fields:

- `dump_file`
- `checkpoint_file`
- `checkpoint_format`
- `checkpoint_interval`

Recommended utility/script fields:

- `mode`
- `input_source`
- `warning_angle_deg`

Rules:

- prefer numbers and strings, not large arrays or full object dumps
- keep field names consistent across modules
- prefer one stable `event` string over many prose variants
- do not log entire atom arrays or full structures

## Formatting Strategy

### Default output

Default to human-readable console logs.

Recommended console format shape:

```text
INFO GBOpt.GBMinimizer Accepted mutation | event=mutation_accepted step=12 mutation=add1 energy=-3.42 best_energy=-3.58
```

This remains readable for humans while preserving machine-parseable key-value
structure.

### Future output modes

Design the helper layer so adding one of the following later is easy:

- JSON formatter with the same field schema
- `structlog` renderer with the same `event` and field names

The migration boundary should be the formatting/configuration layer, not every
call site.

## Alternative Libraries Considered

The standard library `logging` module remains the recommended first
implementation for GBOpt, but a few alternatives are worth noting explicitly.

### Eliot

Eliot is strongest when logs need to behave like structured workflow traces
with nested actions, causal relationships, and strong support for long-running
or distributed tasks.

Potential advantages for GBOpt later:

- better provenance for multi-step optimization workflows
- clearer tracing for checkpoint/restart and orchestration-heavy runs
- stronger event semantics than conventional leveled logging

Reasons not to adopt it now:

- it is a larger conceptual shift than GBOpt currently needs
- GBOpt's immediate need is readable diagnostics, not full action tracing
- it would require call-site patterns that are more invasive than stdlib
  logging

Recommendation:

- do not use Eliot for the initial implementation
- reconsider only if GBOpt grows into a tracing-heavy or multi-process workflow
  system

### Logbook

Logbook is a replacement logging framework with a cleaner API in some areas,
but it is still fundamentally an alternative logging stack rather than a small
increment over stdlib.

Potential advantages:

- somewhat friendlier logging ergonomics
- built-in features beyond the stdlib defaults

Reasons not to adopt it now:

- less standard than Python's built-in logging ecosystem
- adds dependency and migration cost without solving a pressing GBOpt problem
- does not materially improve the warning-vs-library-output distinction that
  GBOpt cares about

Recommendation:

- do not adopt Logbook for this package

### Picologging

Picologging is attractive mainly as a faster implementation of the stdlib
logging API.

Potential advantages:

- lower overhead for very high-volume logging
- partial drop-in compatibility with stdlib call sites

Reasons not to adopt it now:

- GBOpt's hot paths are dominated by structure generation and energy evaluation,
  so logging overhead is unlikely to be the bottleneck
- compatibility gaps and maturity risk are more important than raw logger
  throughput for this package

Recommendation:

- do not adopt Picologging unless profiling later shows logging itself is a
  meaningful runtime cost

### Summary

For GBOpt, the practical ranking is:

1. stdlib `logging` now
2. `structlog` later if richer structured output becomes necessary
3. Eliot only if workflow tracing becomes a real requirement
4. Logbook and Picologging are not currently justified

This keeps the initial implementation low-risk while preserving a clean upgrade
path if the package later needs richer structured logging or workflow tracing.

## Phased Implementation

## Phase 0: Logging policy and schema

Goal:

- agree on semantics before editing call sites

Tasks:

- document the distinction between warnings and logging
- define the base event schema and field names
- define default log levels by use case

Recommended log levels:

- `DEBUG`: internal state useful for debugging and reproducibility
- `INFO`: normal run progress and meaningful milestones
- `WARNING`: log-only warning conditions that are not part of the warnings API
- `ERROR`: failures before raising or re-raising exceptions

Deliverable:

- short policy section in repo docs or README

## Phase 1: Add logging helper module

Goal:

- establish the package-local logging foundation without changing behavior yet

Target files:

- new: `GBOpt/Utils/logging_utils.py`

Tasks:

- add a package-local formatter for human-readable event logs
- add `configure_logging(...)` for scripts/examples/tests
- add a helper for `LoggerAdapter` or structured `extra`
- ensure missing fields do not break formatting

Notes:

- do not configure logging at import time
- keep implementation small and stdlib-only

Deliverable:

- helper module plus unit tests for formatter/configuration behavior

## Phase 2: Convert minimizer progress output

Goal:

- replace the most obvious library-side `print(...)` progress messages first

Primary target:

- `GBOpt/GBMinimizer.py`

Current candidate messages:

- `"Meets energy tolerance criterion!"`
- `"Too many rejections!"`

Add structured log events around:

- MC run start
- initial energy evaluation complete
- mutation proposed
- mutation accepted or rejected
- new best energy found
- tolerance criterion met
- rejection threshold exceeded
- checkpoint saved
- checkpoint loaded
- run completed

Suggested event names:

- `mc_run_started`
- `initial_energy_evaluated`
- `mutation_proposed`
- `mutation_accepted`
- `mutation_rejected`
- `best_energy_updated`
- `energy_tolerance_met`
- `max_rejections_exceeded`
- `checkpoint_saved`
- `checkpoint_loaded`
- `mc_run_completed`

Suggested fields for minimizer logs:

- `component="gbminimizer"`
- `event`
- `step`
- `mutation`
- `accepted`
- `energy`
- `best_energy`
- `delta_energy`
- `temperature`
- `rejection_count`
- `unique_id`
- `checkpoint_file`

Notes:

- avoid logging on every hot-path detail at `INFO` if it becomes noisy
- put very frequent details at `DEBUG`
- do not change numerical behavior or run flow

Validation:

- add deterministic tests that capture logs
- preserve existing minimizer return values and file behavior

## Phase 3: Convert script/utility output

Goal:

- give standalone utilities nicer console output while preserving script
  friendliness

Target files:

- `GBOpt/Utils/gb_params.py`
- `write_initial_structure.py`

Tasks for `gb_params.py`:

- replace stderr warning `print(...)` with logger warning or keep stderr print if
  CLI ergonomics are better, but align message content with the event schema
- replace standalone success/status prints with `INFO` logs
- keep final primary user-facing result easy to read

Tasks for `write_initial_structure.py`:

- configure logging explicitly at script entry
- replace `"Written initial.dat ..."` print with a structured `INFO` log

Notes:

- script utilities are allowed to call `configure_logging(...)`
- library modules they import should still remain configuration-free

Validation:

- smoke tests if script tests exist
- otherwise keep changes small and manually verifiable

## Phase 4: Add optional contextual adapters

Goal:

- reduce repeated boilerplate in repeated-run contexts

Targets:

- `GBOpt/GBMinimizer.py`
- possibly future optimization entry points

Tasks:

- create a `LoggerAdapter` or equivalent helper that binds fields such as
  `component`, `unique_id`, and `seed`
- use that adapter inside long-running minimization routines

Context candidates:

- `component`
- `unique_id`
- `seed`
- `checkpoint_file`

Benefits:

- less repeated `extra` construction
- more consistent fields across related logs
- easier migration to `structlog.bind(...)` later

Constraint:

- keep this thin; do not grow it into a custom logging framework

## Phase 5: Expand into other modules only if justified

Goal:

- avoid mechanical logging churn in stable library code

Potential later targets:

- `GBOpt/GBMaker.py`
- `GBOpt/GBManipulator.py`

Recommendation:

- do not add broad new logging here immediately
- preserve current warning semantics
- only add debug/info logs if there is a concrete debugging or reproducibility
  need

Examples of justified future events:

- expensive construction stage start/end
- mutation strategy chosen
- geometry validation diagnostics before raising

Examples of unjustified churn:

- replacing every warning with a log
- logging per-atom or large-array internals
- adding logs in hot loops with no clear debugging value

## Phase 6: Document user-facing logging behavior

Goal:

- make logging discoverable and predictable for users

Documentation targets:

- `README.md`
- optionally `docs/testing.md`

Document:

- GBOpt does not auto-configure logging in library use
- scripts/examples may opt in to `configure_logging(...)`
- warnings remain warnings
- how to enable verbose output in notebooks or scripts

Example documentation topics:

- basic setup snippet for console logs
- how to increase to `DEBUG`
- how to redirect logs to a file

## Phase 7: Evaluate whether `structlog` is still needed

Goal:

- defer the dependency decision until there is evidence

Trigger conditions that would justify `structlog` later:

- need JSON/event logs across many optimization runs
- need richer contextual binding throughout workflows
- need integration with external telemetry/log aggregation
- stdlib `LoggerAdapter` patterns become cumbersome

If migration happens later:

- preserve event names and field names
- replace formatter/configuration layer first
- keep call-site messages and fields mostly unchanged

This is why event-style logging should be adopted from the start.

## File-by-File Execution Order

Recommended implementation order:

1. `GBOpt/Utils/logging_utils.py`
2. tests for logging utilities
3. `GBOpt/GBMinimizer.py`
4. minimizer log-capture tests
5. `GBOpt/Utils/gb_params.py`
6. `write_initial_structure.py`
7. docs updates in `README.md` and optionally `docs/testing.md`

This order gets the most value with the least risk.

## Testing Plan

Testing should focus on behavior, not exact internal implementation.

### Required tests

- log formatting does not crash when some optional fields are absent
- library code does not emit logs unless the caller configures handlers
- minimizer logs can be captured deterministically
- existing warning tests remain valid

### Minimizer-specific tests

Use fake deterministic energy functions and fixed RNG seeds.

Validate:

- an `INFO` log is emitted when convergence criteria are met
- an `INFO` or `WARNING` log is emitted when rejection limits terminate a run
- checkpoint save/load emits expected event names
- no change to numerical outcomes from logging instrumentation

### Non-goals for tests

- asserting full formatted strings character-for-character
- over-testing exact console presentation
- testing Python's `logging` internals

Prefer asserting:

- event presence
- level
- selected structured fields

## Risks and Mitigations

### Risk: noisy logs

Mitigation:

- keep high-frequency events at `DEBUG`
- reserve `INFO` for milestones and meaningful progress

### Risk: breaking tests that expect warnings

Mitigation:

- do not replace `warnings.warn(...)` in warning-driven library code

### Risk: accidental global logging side effects

Mitigation:

- never call `basicConfig()` from library import paths
- isolate configuration to helper functions and scripts

### Risk: over-engineering

Mitigation:

- no manager class
- no custom logging framework
- no premature `structlog` dependency

### Risk: making future migration harder

Mitigation:

- standardize `event` names and field keys now
- keep messages short and fields structured

## Recommended Initial Scope

The first implementation should be intentionally narrow:

- add `GBOpt/logging_utils.py`
- convert `GBOpt/GBMinimizer.py`
- convert `GBOpt/Utils/gb_params.py`
- convert `write_initial_structure.py`
- update docs
- add focused tests

Do not expand beyond that until the team has seen the output in real runs.

## Decision Summary

Recommended immediate direction:

- use stdlib `logging`
- keep warnings unchanged
- adopt event-style structured fields now
- default to human-readable console output
- keep the architecture thin so `structlog` can be adopted later with minimal
  call-site churn

This gives GBOpt a practical logging system now without committing to a
premature observability stack.
