> **SUPERSEDED**
>
> Replaced by [`../../MASTER_PLAN.md`](../../MASTER_PLAN.md). This file is retained as design history for the April 2026 utilities-and-interfaces effort. It must not be used for branch creation, PR sequencing, progress status, or current API decisions.

# GBOpt — Implementation Master Plan

**Maintained by:** Jarin French
**Last updated:** (update on each human edit)
**Status:** Phase 0 — not yet started
**Branch:** `jarinfrench/GBOpt:feature/utils-and-interfaces`

SUPERSEDED

This plan was replaced by docs/MASTER_PLAN.md. It is retained as design
history and must not be used for branch creation, sequencing, or API decisions.

> This document is the source of truth for implementation sequencing and
> constraints. It does not duplicate the design documents — refer to those
> for architectural rationale. Changes to anything here must be recorded in
> `session_log_master.md` with reasoning.

---

## Design Documents

| Document | Purpose |
|---|---|
| `docs/gbopt_interface_design.md` | Abstract interface layer — Protocols, ABCs, `CalculatorResult`, `Parent` type |
| `docs/gbopt_utils_design.md` | `utils` subpackage — `io`, `geometry`, `validation`, `plotting`, `objective_templates` |

---

## Non-Negotiables

These constraints must be respected across all phases. Deviations require an
explicit decision recorded in `session_log_master.md`.

1. **Backward compatibility:** Existing class names (`GBMaker`, `GBManipulator`,
   `GBMinimizer`) must remain importable from their current paths for at least
   one full release cycle after any refactor.
2. **Callable compatibility:** Users currently passing a bare `gb_energy_function`
   callable to the minimizers must continue to work during the transition to
   `BaseCalculator`. The minimizer should accept either form, wrapping the bare
   callable with a deprecation warning.
3. **No silent moves:** Any method or function relocated from a core class to
   `utils/` must have a thin wrapper or import alias at the original location,
   with a `DeprecationWarning`, for at least one release cycle.
4. **`Parent` is the canonical structure type.** Do not introduce ASE `Atoms`
   or any other third-party type as an interface-level dependency.
5. **`CalculatorResult` is scalar-only.** No forces or stress tensors at the
   interface level. See `gbopt_interface_design.md` §3.5 for the full field
   specification.
6. **`interfaces/` is logic-free.** Protocols, ABCs, and type definitions only.
   No implementation logic belongs there.
7. **`CompositeManipulator` lives with the concrete manipulator implementations**
   (currently `GBManipulator.py`), not in `interfaces/` or `utils/`.
8. **All tests must pass at the end of every phase** before moving to the next.
9. **Test suite conventions:**
   - Fast tests only: `pytest -m "not slow and not known_bug"`
   - Full suite (slow tests included): `pytest` — only run when explicitly
     requested or at final phase exit
   - Codex verification sessions use fast tests only unless the session
     goal specifically involves slow test coverage

---

## Phases

### Phase 0 — Define interfaces + new utils (no existing code touched)

**Goal:** Establish the interface layer and the first new utility module.
Nothing in this phase touches existing classes, so nothing can break.

**Tasks:**
- Create `GBOpt/interfaces/` with:
  - `maker.py` — `GBMakerProtocol`, `BaseGBMaker`
  - `manipulator.py` — `GBManipulatorProtocol`, `BaseGBManipulator`
  - `minimizer.py` — `GBMinimizerProtocol`, `BaseGBMinimizer`
  - `io.py` — `IOProtocol`, `BaseIO`
  - `calculator.py` — `CalculatorProtocol`, `BaseCalculator` (with `is_ready`)
  - `types.py` — `CalculatorResult`, `MinimizerResult` dataclasses
  - `__init__.py`
- Create `GBOpt/utils/objective_templates.py`:
  - `make_lammps_subprocess_objective(...)` — single-eval, subprocess pattern
  - `make_lammps_slurm_batch_objective(...)` — batch, SLURM pattern

**Entry criteria:** Design docs approved (done).
**Exit criteria:** `interfaces/` and `objective_templates.py` exist; all
existing tests still pass; no imports in existing classes have changed.

---

### Phase 1 — Retrofit existing classes (highest risk)

**Goal:** Make `GBMaker`, `GBManipulator`, `MonteCarloMinimizer`, and
`GeneticAlgorithmMinimizer` subclass their respective ABCs.

**Tasks:**
- `GBMaker` subclasses `BaseGBMaker`; abstract properties confirmed satisfied
- `GBManipulator` subclasses `BaseGBManipulator`
- `MonteCarloMinimizer` and `GeneticAlgorithmMinimizer` subclass
  `BaseGBMinimizer`
- Verify no public API changes; add deprecation wrapper for bare callable
  interface

**Entry criteria:** Phase 0 complete and verified by Codex.
**Exit criteria:** All existing tests pass; no public API breakage; type
checker (mypy/pyright) passes on the retrofitted classes.
> **Known pre-existing failures:** `test_created_gbs` and
> `test_type_preservation_with_numeric_roundtrip` in
> `tests/test_gbmanipulator.py` are marked `@pytest.mark.known_bug` and
> excluded from Phase 1 exit criteria. See `session_log_master.md`
> 2026-04-30 for context.

---

### Phase 2 — First new concrete implementation + utils extraction

**Goal:** Validate the `BaseCalculator` interface in practice; extract
geometry helpers from `GBMaker`.

**Tasks:**
- Implement `LAMMPSCalculator` as the first formal `BaseCalculator` subclass
- Extract the six geometry helpers from `GBMaker` into `GBOpt/utils/geometry.py`:
  `reduce_integer_row`, `row_angle_error_deg`,
  `approximate_rotation_row_as_int`, `approximate_rotation_matrix_as_int`,
  `scaled_periodic_basis_vector`, `cartesian_from_box_coordinates`
- Leave deprecation wrappers at original call sites in `GBMaker`
- Begin `GBOpt/utils/io.py` with LAMMPS data and dump helpers (extracted from
  existing package code)

**Entry criteria:** Phase 1 complete and verified by Codex.
**Exit criteria:** `LAMMPSCalculator` passes a smoke test against the existing
example workflow; geometry helpers pass unit tests; existing tests unaffected.

---

### Phase 3 — Utils integration + validation consolidation

**Goal:** Connect the utils and interface layers; consolidate validation logic.

**Tasks:**
- `objective_templates.py` updated to return `BaseCalculator` instances
  rather than bare callables (while keeping callable wrappers for backward
  compat)
- `utils/io.py` functions become the implementation detail used by concrete
  `IO` classes (`LAMMPSDataIO`, `LAMMPSDumpIO`)
- `utils/validation.py` created with consolidated validators extracted from
  `GBMaker`, `GBManipulator`, `GBMinimizer`; deprecation wrappers left in
  place
- `utils/plotting.py` Tier 1 plots implemented

**Entry criteria:** Phase 2 complete and verified by Codex.
**Exit criteria:** All tests pass; full workflow smoke test passes end-to-end
using the new interface layer.

---

## Current Phase Tracker

| Phase | Status | Claude Code session(s) | Codex session(s) |
|---|---|---|---|
| 0 | Complete | — | — |
| 1 | Not started | — | — |
| 2 | Not started | — | — |
| 3 | Not started | — | — |
