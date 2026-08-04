# `docs/architecture/README.md`

# GBOpt Architecture Decisions

This directory contains accepted architecture decision records for GBOpt.

The records document decisions that apply across multiple implementation tracks. They explain architectural constraints but do not replace the PR sequencing and branch rules in `docs/MASTER_PLAN.md`.

## Authority

Documentation precedence is:

1. `docs/MASTER_PLAN.md` — implementation sequence, prerequisites, merge order, and file-ownership windows.
2. `docs/architecture.md` — current system-wide architectural overview.
3. `docs/architecture/adr/*.md` — accepted cross-cutting architecture decisions.
4. `docs/AI_AGENT_CODEBASE_RULES.md` — implementation and review discipline.
5. `docs/testing.md` — testing policy and execution gates.
6. `docs/design/*.md` — component-specific rationale and detailed acceptance criteria.
7. `docs/history/` and `docs/superseded/` — non-authoritative historical material.

When an ADR conflicts with `MASTER_PLAN.md`, the master plan controls implementation sequencing. The conflict should then be resolved through a documentation-only change.

## Decision records

| ADR                                                        | Decision                                                                                                                                           | Status   |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| [ADR 0001](adr/0001-domain-contracts.md)                   | Shared domain contracts separate generic structure, persistent interface state, serialization, manipulation, evaluation, events, and restart state | Accepted |
| [ADR 0002](adr/0002-io-owns-file-syntax.md)                | File syntax and transient serialization identity belong to `GBOpt.io`                                                                              | Accepted |
| [ADR 0003](adr/0003-operation-level-manipulations.md)      | Individual manipulation operations are the extension point; `GBManipulator` remains a facade                                                       | Accepted |
| [ADR 0004](adr/0004-events-are-not-checkpoints.md)         | Logging, event journals, and checkpoints are separate mechanisms                                                                                   | Accepted |
| [ADR 0005](adr/0005-canonical-periodic-representatives.md) | Periodic grain construction uses canonical reduced-coordinate representatives                                                                      | Accepted |

## ADR status values

* **Proposed** — under review and not yet binding.
* **Accepted** — approved and binding for new implementation work.
* **Superseded** — replaced by a later ADR.
* **Rejected** — considered but not adopted.
* **Deprecated** — retained temporarily while migration completes.

## Adding or changing an ADR

An ADR change should:

1. describe the architectural problem;
2. state one clear decision;
3. identify affected roadmap PRs;
4. list positive and negative consequences;
5. define how the decision will be tested or enforced;
6. identify any previous document it supersedes.

Accepted ADRs should not be edited merely to hide historical disagreement. A materially different decision should normally receive a new ADR that supersedes the old one.

