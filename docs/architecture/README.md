# GBOpt architecture decisions

These records are the accepted F0 boundaries for the refactoring roadmap. Later PRs
may refine implementation details, but a change to one of these decisions requires a
new superseding ADR rather than an undocumented divergence.

| ADR | Decision |
|---|---|
| [0001](0001-shared-domain-contracts.md) | Shared domain contracts and ownership |
| [0002](0002-io-owns-file-syntax.md) | I/O owns representation syntax and transient IDs |
| [0003](0003-operation-level-manipulation.md) | Manipulations are operation-level strategies |
| [0004](0004-typed-evaluation-results.md) | Evaluation failures are typed before penalties |
| [0005](0005-events-are-not-checkpoints.md) | Events/journals and checkpoints are distinct |
| [0006](0006-source-baseline-authority.md) | Current source is the implementation baseline |
