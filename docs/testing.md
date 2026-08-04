# GBOpt Testing Policy

## Purpose and authority

This document defines the repository's test-writing conventions and normal execution gates. `MASTER_PLAN.md` controls PR-specific prerequisites and major integration gates. `AI_AGENT_CODEBASE_RULES.md` controls broader correctness, compatibility, and reporting discipline.

## Default execution order

Run tests from narrowest to broadest:

1. compile or import checks for touched modules;
2. the smallest relevant unit or regression tests;
3. related module-level and cross-layer integration tests;
4. the non-slow suite;
5. designated slow or end-to-end tests at roadmap integration gates.

Do not begin with the full slow suite when a focused failure can provide faster and clearer feedback.

## Normal commands

Use the repository's active Python environment and invoke pytest through Python where practical.

```bash
python -m compileall GBOpt tests

pytest -q tests/test_relevant_module.py

pytest -m "not slow"
```

Run a single test with its node ID:

```bash
pytest -q tests/test_example.py::test_specific_behavior
```

Run the full suite, including slow tests, only when required by the roadmap, requested by the maintainer, or appropriate for a release gate:

```bash
pytest
```

Do not claim a type check, formatter, linter, or documentation check passed unless that tool is configured in the repository and was actually run. Record the exact command and result.

## Test style

### New tests

New tests should normally use idiomatic pytest:

- test functions;
- fixtures;
- parametrization;
- `monkeypatch`;
- `pytest.raises`;
- `pytest.warns`;
- NumPy testing helpers.

Do not add a new `unittest.TestCase` merely because an older file uses that style. Extending an existing legacy `TestCase` is acceptable when conversion is unrelated to the current PR and keeping the test nearby improves reviewability.

Do not convert unrelated tests solely for style during a functional or extraction PR.

### Test placement

Place tests according to the owning behavior rather than a rigid one-file-per-source-file rule.

The suite should distinguish:

1. kernel unit tests;
2. stage or service unit tests;
3. cross-layer integration tests;
4. public compatibility tests;
5. end-to-end and slow tests.

As monoliths are decomposed, new tests should target the extracted module directly while retaining public-path compatibility coverage.

## Required qualities

### Determinism

Tests must not depend on:

- global NumPy RNG state;
- random iteration order;
- wall-clock timing;
- uncontrolled parallelism;
- nondeterministic file ordering;
- external network access.

Use a fixed `numpy.random.Generator` seed when randomness is part of the contract. Test that `seed=0` remains valid where applicable.

### Bounded execution

Exactification, repeat searches, enumeration, and other potentially expensive operations must use explicit test bounds. Mark genuinely expensive tests with `@pytest.mark.slow`.

### Behavior over implementation trivia

Prefer tests of:

- exact identities;
- physical and chemical invariants;
- deterministic structures and ordering;
- explicit topology and periodicity;
- ownership preservation;
- warning and failure policy;
- compatibility imports and public behavior;
- serialization round trips;
- optimizer histories and restart equivalence.

Avoid tests whose primary purpose is incidental interpreter state, private cache contents, exact internal call counts, or import order unless that behavior is itself an accepted architecture contract.

## Regression tests

Every bug fix requires a regression test that fails on the defective behavior and passes after the correction. The test name or a short comment should identify the protected invariant.

A bug fix should update the changelog or release notes when the repository's release process requires it.

## Warnings, exceptions, and logs

### Warnings

When warning behavior is part of the contract, assert:

- warning category;
- a meaningful message fragment;
- the expected number of warnings where material;
- absence of warnings on valid paths.

Do not assert complete prose unless exact wording is itself a compatibility requirement.

### Exceptions

Assert the layer-appropriate exception type and the condition that triggers it. When exception translation is part of the architecture, integration tests should verify the public exception and preserve chaining where useful.

### Logging and events

Logging and event tests should assert:

- semantic event name;
- level or sink mapping;
- selected structured fields;
- absence of unsolicited output by default.

Do not assert a complete human-formatted log line unless testing the formatter itself.

## NumPy and structure comparisons

Choose comparison semantics deliberately.

### Order-sensitive comparison

Use when atom or row ordering is part of deterministic behavior, serialization, identity alignment, or optimizer reproducibility.

### Order-insensitive comparison

Use only as an additional physical-equivalence diagnostic. It must not excuse an unexplained order-sensitive change.

### Floating-point comparison

Use existing project tolerances or a tolerance owned by the tested contract. Do not widen tolerances merely to make a refactor pass.

### Exact data

Compare exact integer crystallographic data exactly. Do not convert large exact arrays to float for testing.

### Structured arrays

Use existing shared helpers where available. Do not duplicate comparison logic across test modules.

## Characterization requirements for refactors

Behavior-preserving extraction PRs should characterize representative cases before moving code.

As applicable, record:

- normalized inputs;
- warning sequence;
- exact versus approximate path;
- topology and periodicity;
- dimensions and cell bounds;
- atom and species counts;
- order-sensitive structure hashes;
- order-insensitive physical hashes for diagnosis;
- serialized artifact hashes;
- fixed-seed optimizer histories.

Run the characterization at least twice to confirm determinism.

## I/O and evaluator testing

### I/O readers and writers

Use small, explicit fixtures and temporary directories. Cover:

- malformed syntax;
- missing or duplicate external IDs;
- type maps and charges;
- finite coordinates;
- frame-selection semantics;
- deterministic output;
- row-to-ID mappings;
- unsupported or lossy behavior;
- compatibility facades.

### External calculators

Prefer deterministic fake evaluators for normal unit and integration tests. They should emulate the documented legacy or typed result contract without invoking LAMMPS, a scheduler, or another external executable.

Use real calculator or scheduler tests only when the environment is explicitly configured and the test is marked appropriately.

### Evaluation failures

Test failure stages separately, including:

- invocation failure;
- malformed result;
- non-finite objective;
- missing artifact;
- reload failure;
- ownership reconstruction failure;
- penalty application without loss of the original failure record.

## Manipulation testing

Every operation should have reusable contract coverage for:

- exact parent arity;
- parent immutability;
- complete `InterfaceCandidate` output;
- independent child storage;
- deterministic replay;
- supplied RNG use;
- topology and capability validation;
- structured operation parameters and lineage;
- legacy-wrapper equivalence during migration.

At least one test-defined operation should prove extension without editing GBOpt source.

## Checkpoint testing

Checkpoint tests must distinguish persistence mechanics from optimizer integration.

Cover:

- disabled no-op behavior;
- schema and algorithm validation;
- atomic publication;
- corrupted or unsupported files;
- RNG restoration;
- required artifact validation;
- completed safe-boundary semantics;
- uninterrupted-versus-resumed equivalence;
- run extension after completion where supported;
- no duplicate history after resume;
- candidate-cache cleanup after authoritative checkpoint publication.

A journal file must never be accepted as a checkpoint.

## Temporary files

Prefer pytest's `tmp_path` or `tmp_path_factory` for new tests.

Legacy `TemporaryDirectory` usage may remain where conversion is unrelated:

```python
self.tmpdir = tempfile.TemporaryDirectory()

def tearDown(self):
    self.tmpdir.cleanup()
```

Tests must not write persistent artifacts into the repository tree unless the fixture is deliberately version-controlled.

## Mocking

Mock external failures, scheduler state, or file-system errors when testing translation and recovery behavior. Do not mock the core mathematical or physical algorithm in a test that claims to verify its correctness.

## Required reporting in each PR

The PR description or closeout note must state:

- exact commands run;
- pass, fail, skip, and xfail results;
- slow tests run or explicitly deferred;
- environment limitations;
- any test that could not be executed;
- whether characterization outputs changed;
- whether warnings, exceptions, public imports, or serialized files changed.

Never state that the suite passed when only a subset was run.

## Roadmap integration gates

Follow the gates named in `MASTER_PLAN.md`. At minimum, designated slow tests are expected after the major builder, candidate-loader, manipulation, evaluation, and checkpoint integrations, and at the final `INT1` release gate.
