# GBOpt Documentation

This directory contains the active implementation roadmap, architectural policy, component design documents, and retained historical material for the GBOpt refactoring program.

## Documentation authority

Use the following precedence order:

1. [`MASTER_PLAN.md`](MASTER_PLAN.md) — PR sequence, prerequisites, branch ancestry, merge waves, file-ownership windows, and integration gates.
2. [`architecture.md`](architecture.md) — current system-wide architecture and invariants.
3. [`architecture/adr/`](architecture/adr/) — accepted cross-cutting architecture decisions.
4. [`AI_AGENT_CODEBASE_RULES.md`](AI_AGENT_CODEBASE_RULES.md) — implementation and review discipline.
5. [`testing.md`](testing.md) — test-writing conventions and execution policy.
6. [`design/`](design/) — component-specific rationale, contracts, and detailed acceptance criteria.
7. [`history/`](history/) and [`superseded/`](superseded/) — non-authoritative retained material.

When two active documents conflict, `MASTER_PLAN.md` controls implementation sequencing. Resolve substantive architectural conflicts through a documentation-only change before implementation proceeds.

## Active documents

### Master roadmap

- [`MASTER_PLAN.md`](MASTER_PLAN.md)

### Architecture and policy

- [`architecture.md`](architecture.md)
- [`architecture/README.md`](architecture/README.md)
- [`AI_AGENT_CODEBASE_RULES.md`](AI_AGENT_CODEBASE_RULES.md)
- [`testing.md`](testing.md)

### Component design

- [`design/GBMAKER_INCREMENTAL_PIPELINE_REFACTOR_PLAN.md`](design/GBMAKER_INCREMENTAL_PIPELINE_REFACTOR_PLAN.md)
- [`design/GBMANIPULATOR_ABSTRACTION_PROPOSAL.md`](design/GBMANIPULATOR_ABSTRACTION_PROPOSAL.md)
- [`design/GBOPT_IO_ABSTRACTION_PROPOSAL.md`](design/GBOPT_IO_ABSTRACTION_PROPOSAL.md)
- [`design/GBOPT_LOGGING_AND_OBSERVABILITY_PLAN.md`](design/GBOPT_LOGGING_AND_OBSERVABILITY_PLAN.md)

## Historical and superseded material

- `history/` contains completed or earlier implementation work whose rationale remains useful.
- `superseded/` contains plans or designs replaced by active documents.
- Neither directory controls current implementation.

Every superseded document should identify its replacement. Historical trackers should state that they are closed and must not be used as current progress trackers.

## Backlog material

`backlog/IDEAS_TO_TRIAGE.md` is temporary. Each item should be converted into a GitHub issue, linked to a roadmap PR, or explicitly rejected. Remove the file when triage is complete.

## Updating status

Update the PR status table in `MASTER_PLAN.md` as work begins and merges. Do not create a separate session-log authority or long-lived integration-branch tracker.
