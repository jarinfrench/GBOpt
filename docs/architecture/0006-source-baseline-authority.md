# ADR 0006: Current source is authoritative; legacy checkpoint source is reference-only

- **Status:** Accepted
- **Date:** 2026-08-04
- **Decision owners:** Foundation and checkpoint tracks

## Context

The roadmap names a current implementation archive and an older checkpoint-enabled
archive. This F0 task explicitly starts from a separately supplied archive named
`gbopt_source.tar(1).gz`, whose content and digest differ from the archive recorded in
the roadmap.

## Decision

For this F0 implementation, the user-specified source archive is authoritative:

```text
gbopt_source.tar(1).gz
d0c24898b334b26f5445304d209cb3bf3fbc1ac3882eb7a5b029045ac565d6b1
```

Its current behavior is frozen by the committed characterization manifest. The older
checkpoint-enabled archive is used only to enumerate behavior for CP0; its production
code and raw schemas must not be merged wholesale.

Later PRs begin from the latest accepted branch or verified handoff archive and record
that prerequisite explicitly. If project maintainers instead select another archive as
the F0 baseline, this ADR and the manifest must be regenerated together before any
production refactor merges.

## Consequences

- Baseline identity is explicit rather than inferred from an ambiguous filename.
- Characterization failures after later refactors identify behavioral drift relative to
  this exact source state.
- Legacy JSON/pickle formats are not compatibility promises unless separately approved.
