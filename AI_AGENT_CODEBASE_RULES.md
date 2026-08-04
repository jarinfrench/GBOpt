# GBOpt Codebase Rules for AI Coding Agents

## 1. Purpose and interpretation

This document defines the default rules for modifying the GBOpt codebase. It is written for AI coding agents, but the same rules apply to human contributors.

The terms **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are normative:

- **MUST / MUST NOT**: required for correctness, maintainability, or compatibility.
- **SHOULD / SHOULD NOT**: expected unless the change has a documented reason to differ.
- **MAY**: optional and context-dependent.

When two rules appear to conflict, preserve the following priorities in order:

1. Physical and mathematical correctness.
2. Explicit domain invariants and user-visible behavior.
3. Determinism and reproducibility.
4. Backward compatibility.
5. Clear architecture and maintainability.
6. Performance.
7. Concision.

Do not optimize code by weakening an invariant, silently changing a construction mode, deleting required atoms, or replacing exact arithmetic with floating-point approximations.

---

## 2. Understand the owning layer before editing

Every change MUST be implemented in the layer that owns the behavior. Do not place logic in a convenient caller merely because that caller already has the necessary values.

### 2.1 Boundary specification layer

`BoundarySpec.py` owns user-facing boundary dataclasses, field validation, normalization, immutability, and boundary-spec exception types.

This layer SHOULD:

- Normalize mutable inputs into immutable tuples or copied read-only arrays.
- Reject invalid types, shapes, values, booleans masquerading as integers, and non-finite values.
- Define declarative boundary data, not perform lattice construction.

This layer MUST NOT:

- Construct CSLs.
- Generate grain atoms.
- Perform optimization.
- Contain package-level orchestration that belongs in a boundary adapter.

### 2.2 Crystallography data layer

`GBOpt/crystallography/types.py` owns crystallography exceptions, type aliases, and result/domain dataclasses.

This module MUST remain a data-definition layer. Arithmetic algorithms and cross-module orchestration do not belong here.

### 2.3 Exact integer utility layer

Exact integer validation and arithmetic MUST be centralized in the existing integer utility modules. Reuse helpers such as exact array conversion, exact determinant, exact adjugate, exact dot product, exact cross product, GCD reduction, and integer normal forms.

Do not add a second local implementation of an existing exact helper.

### 2.4 Crystallography algorithm modules

Keep responsibilities narrow:

- `csl.py`: exact CSL and DSC construction from validated scaled rotations.
- `rotation.py`: scaled-rotation validation, row/column convention conversion, and exact row-image helpers.
- `quaternion.py`: quaternion normalization, rationalization, and conversion to scaled rotations.
- `plane.py`: plane covectors, integer plane null spaces, in-plane bases, and plane-preservation logic.
- `reduction.py`: exact lattice-reduction algorithms.
- `pq.py`: paired P/Q canonicalization and exact rotation recovery.
- `embedding.py`: construction of `BoundaryEmbedding` objects from already normalized crystallographic data.
- `exactification.py`: conversion of supported floating-point five-DOF descriptions into bounded exact crystallographic forms.
- `boundary.py`: user-facing adapters and orchestration between boundary specs and lower-level crystallography modules.

A lower-level algorithm module MUST NOT import `GBMaker` or take responsibility for grain generation.

### 2.5 `GBMaker` layer

`GBMaker` owns bicrystal construction from validated boundary representations. It may select an exact or approximate construction path, build grains, establish box dimensions, apply explicitly configured commensurability/strain policy, and preserve atom-group invariants.

`GBMaker` MUST NOT become a second crystallography package. Exact orientation recovery, CSL arithmetic, P/Q canonicalization, and exactification belong in `GBOpt.crystallography`.

### 2.6 Clean generation versus optimization

Clean boundary generation and optimization are separate concerns.

Generation code MAY:

- Produce a deterministic initial bicrystal.
- Apply explicit termination and rigid-translation choices.
- Enforce construction limits.
- Reject invalid or infeasible structures.
- Report warnings and diagnostics.

Generation code MUST NOT:

- Perform energy minimization.
- Search an optimization landscape.
- Preempt the responsibilities of `GBManipulator` or `GBMinimizer`.
- Hide a poor starting structure by silently performing an optimization-like repair.

The output of generation should be a physically credible, auditable starting point with a clear path to downstream optimization.

---

## 3. Preserve exact arithmetic contracts

### 3.1 Use Python-sized integers for exact crystallography

Exact crystallographic arithmetic MUST use Python integers, normally through NumPy arrays with `dtype=object`.

Do not use fixed-width NumPy integer arithmetic when overflow is possible. Do not use floating-point arithmetic to verify identities that are exact by definition.

### 3.2 Never silently round exact input

Inputs documented as exact integers MUST be validated as exactly integer-valued. A float such as `3.0` may be accepted only through the established exact-integer validator. A value such as `3.0000001` MUST NOT be rounded into acceptance.

Boolean values MUST be rejected when an integer or real scalar is required.

### 3.3 Preserve the project rotation convention

The project convention for `ScaledRotation` is a row-vector mapping:

```text
q_row = p_row @ M / N
```

Column-vector CSL routines MUST receive the transposed numerator matrix. Any new function that crosses the row/column boundary MUST make that conversion explicit and test it.

### 3.4 Verify exact identities explicitly

Exact constructions SHOULD verify their defining identities before returning. Examples include:

- `M @ M.T == N**2 * I`.
- `det(M) == N**3`.
- CSL membership residuals are zero modulo `N`.
- `abs(det(CSL basis)) == sigma`.
- `P @ M == N * Q` row by row.
- Canonicalization preserves the intended exact rotation when that is part of the contract.

A failed internal invariant MUST raise a domain-specific exception rather than returning a partially valid result.

### 3.5 Do not convert large exact arrays to float for validation

Exact orientation rows may exceed floating-point range. Validate exact matrices with exact dot products and determinants. Floating-point normalization is appropriate only when constructing floating-point orientation frames for user-visible geometry.

---

## 4. Validation rules

### 4.1 Validate at public boundaries

Every public entry point MUST validate its own contract. Internal helpers MAY assume prevalidated inputs only when that assumption is documented and structurally enforced by callers.

Validation SHOULD cover, as applicable:

- Type.
- Shape.
- Finiteness.
- Boolean rejection.
- Positivity or non-negativity.
- Nonzero vectors.
- Nonsingularity.
- Orthogonality.
- Right-handedness.
- Exact divisibility.
- Supported enum or mode values.
- Resource/construction limits.

### 4.2 Centralize repeated validation

If the same condition appears in multiple modules, add or reuse a shared validator in the layer that owns the invariant. Error type, message, and condition SHOULD be defined once and tested once at the validator level, then integration-tested at public boundaries.

Do not copy validation snippets between modules.

### 4.3 Validate limits even when a path may not use them

User-facing bounds such as maximum primitive area index, maximum P/Q determinant, maximum repeat count, and tolerances MUST be validated consistently on entry. Whether a particular mode uses the bound is a separate question from whether the argument itself is valid.

### 4.4 Make unsupported behavior explicit

Recognized but unsupported behavior MUST raise the appropriate `NotImplementedError` subclass. It MUST NOT silently fall back to a different physical model.

For example, non-cubic hooks should remain explicit guards until non-cubic support is genuinely implemented.

---

## 5. Construction modes and fallback policy

Construction mode is part of the public contract.

- `mode="exact"` MUST either complete an exact construction or raise.
- `mode="prefer_exact"` MAY fall back only where the API explicitly defines a warning-backed approximate fallback.
- `mode="approximate"` MUST remain visibly approximate and MUST NOT claim exactness or coherence that was not established.

Do not introduce a silent fallback from exact to approximate behavior.

Warnings associated with a fallback MUST state:

1. What failed.
2. Which fallback was selected.
3. The relevant reason or exception.

---

## 6. Exception and warning discipline

### 6.1 Raise layer-appropriate exceptions

Each layer MUST expose its own exception vocabulary:

- Boundary-spec validation failures use `BoundarySpecError` subclasses.
- Crystallography failures use `CrystallographyError` subclasses.
- `GBMaker` validation and construction failures use `GBMakerError` subclasses.

Lower-level exceptions SHOULD be translated at architectural boundaries and chained with `raise ... from exc`.

### 6.2 Do not over-catch

Catch only exceptions that the current layer can translate, recover from, or annotate meaningfully. Do not use broad `except Exception` blocks around domain algorithms.

### 6.3 Warnings must be actionable

Warnings SHOULD indicate:

- The condition that triggered the warning.
- The policy selected.
- Whether physical validity, periodicity, stoichiometry, or usability may be affected.
- The relevant numeric diagnostic when useful.

Use an appropriate warning class and a `stacklevel` that points to the caller.

### 6.4 Warnings are not substitutes for invalid-state errors

Use warnings for recoverable policy decisions or degraded but explicitly supported output. Raise when an invariant required by the requested mode cannot be satisfied.

---

## 7. Immutability, ownership, and deterministic state

### 7.1 Copy caller-owned mutable input

Dataclasses and persistent result objects MUST NOT retain writable aliases to caller-owned lists or arrays.

Use immutable tuples for small integer/vector fields. Copy NumPy arrays before storage and mark arrays read-only when the object is intended to be immutable.

### 7.2 Normalize scalar types

Persistent public fields SHOULD contain Python `int`, `float`, and `bool` values rather than NumPy scalar subclasses, unless a NumPy scalar is specifically required.

### 7.3 Prefer frozen, slotted dataclasses for value objects

New domain/result dataclasses SHOULD normally use `frozen=True` and `slots=True`. Mutable operational state should be used only when mutation is part of the object’s explicit role.

### 7.4 Preserve deterministic output

Given the same inputs and configuration, generation and serialization MUST be deterministic.

Determinism includes:

- Canonical row ordering and signs.
- Stable atom ordering where order is exposed or serialized.
- Stable atom and grain identifiers.
- Stable metadata and provenance.
- Stable hashes.
- Explicit tie-breaking in searches.
- No dependence on set/dict iteration where order affects output.

---

## 8. Grain and atom integrity

### 8.1 Preserve complete basis/origin groups

For multi-atom conventional cells, trimming, clipping, wrapping, deduplication, and gap handling MUST preserve complete conventional-cell origin groups unless a higher-level algorithm explicitly performs a chemically informed termination operation.

Do not delete an arbitrary atom or a single species-resolved plane merely to improve a geometric metric.

### 8.2 Preserve stoichiometry by construction

For stoichiometric structures such as fluorite or rocksalt, each generated grain and the combined bicrystal SHOULD preserve the expected species ratio whenever the construction contract requires complete bulk cells.

Any intentional nonstoichiometric termination MUST be explicit, represented in metadata, and tested as a separate policy.

### 8.3 Do not hide geometry defects

Large voids, overlaps, non-commensurate exact boxes, incomplete unit-cell groups, or unusable grain widths MUST NOT be hidden by ad hoc atom deletion.

Prefer, in order:

1. Correcting the crystallographic construction.
2. Selecting an explicit termination or translation policy.
3. Rejecting the structure as infeasible.
4. Emitting a warning only when degraded output is an accepted public behavior.

### 8.4 Make topology explicit

Boundary conditions and topology MUST be explicit data, not inferred from historical defaults.

Represent, as applicable:

- Per-axis periodicity.
- Central and periodic interfaces.
- Free surfaces.
- Vacuum regions.
- Fixed and buffer regions.
- Grain membership.
- Interface/termination identity.
- Rigid translation.

Fully periodic boundaries are not a universal requirement.

---

## 9. Geometry, tolerances, and units

### 9.1 Name tolerances and state their units

Do not scatter magic tolerances through algorithms. Use named constants or validated keyword parameters at the layer that owns the comparison.

A tolerance’s units and meaning MUST be clear, for example:

- Angstroms.
- Radians.
- Degrees.
- Reduced-coordinate units.
- Relative mismatch.
- Dimensionless component error.

### 9.2 Do not reuse one tolerance for unrelated invariants

A coordinate snap tolerance, orthogonality tolerance, exactification angle tolerance, plane rationalization tolerance, and contact-distance threshold are different policies. Keep them separate.

### 9.3 Use scale-safe normalization

Direction normalization SHOULD avoid overflow and underflow for very large or very small finite vectors. Normalize by a scale factor before evaluating a Euclidean norm when needed.

### 9.4 Exact before approximate

When the domain provides an exact integer formulation, use it. Floating-point checks are appropriate for approximate orientation frames and physical coordinates, not as replacements for exact algebraic validation.

---

## 10. Public API rules

### 10.1 Curate exports deliberately

A function is not public merely because it lacks a leading underscore.

Public crystallography API symbols MUST be intentionally exported through the package `__init__.py` and, where used, the defining module’s `__all__`.

Private implementation modules, shared limits, and guard helpers SHOULD remain private.

### 10.2 A new public symbol requires a complete change

Adding a public function or type normally requires all of the following:

- Stable name and contract.
- Complete docstring.
- Domain-specific validation and exceptions.
- Unit tests and integration tests.
- Export updates.
- Compatibility review.
- User-facing documentation or example when appropriate.

### 10.3 Do not expose helpers prematurely

Keep a helper private until there is a real external use case and a stable contract. Avoid expanding the supported API to make one internal test easier.

---

## 11. Documentation rules

### 11.1 Module docstrings define boundaries

Every substantive module SHOULD begin with a concise docstring that states:

- What the module owns.
- What inputs it consumes.
- What it returns.
- Which neighboring concerns explicitly do not belong there.

### 11.2 Public and significant private functions require complete docstrings

Use the project’s reStructuredText field style:

```text
:param name: Description.
:return: Description.
:raises SomeError: Condition.
```

Docstrings MUST describe actual behavior, not intended future behavior.

### 11.3 Document propagated project exceptions

If a function directly propagates a documented project exception from a called project function, include the corresponding `:raises` field unless the function translates or suppresses it.

When the docstring checker requires a suppression, the suppression MUST include a specific reason. Multi-line reason-bearing suppressions are acceptable when needed for readability.

### 11.4 Comments explain why

Comments SHOULD explain invariants, conventions, physical policy, or non-obvious algorithmic choices. Do not narrate straightforward syntax.

### 11.5 Keep examples executable in spirit

Examples SHOULD use current public APIs, current mode names, correct units, and valid boundary specifications. Remove examples that depend on deprecated internals unless they are explicitly documenting a migration path.

---

## 12. Test rules

### 12.1 New tests use idiomatic pytest

New tests SHOULD use pytest functions, fixtures, parametrization, `monkeypatch`, `pytest.raises`, `pytest.warns`, and NumPy testing helpers.

Do not add new `unittest.TestCase` classes unless extending a legacy test area where conversion is outside the scope of the change.

### 12.2 Test behavior and invariants, not implementation trivia

Prefer tests that prove domain behavior:

- Exact algebraic identities.
- Proper rotations.
- Canonicalization idempotence.
- Round trips between representations.
- Deterministic output.
- Stoichiometry.
- Complete origin groups.
- Correct topology and periodicity.
- Boundary-mode dispatch.
- Immutability and input copying.
- Limit enforcement.
- Warning and fallback policy.

Avoid brittle tests of incidental Python interpreter state, import order, private cache contents, or internal call counts unless that behavior is itself part of the contract.

### 12.3 Every bug fix requires a regression test

A regression test MUST fail for the original bug and pass for the fix. Name or comment the test so the protected behavior is clear.

### 12.4 Parametrize input classes

Use parametrization for families of invalid inputs, mode combinations, structures, and crystallographic scenarios. Give cases descriptive IDs.

Do not over-consolidate tests when parametrization would obscure distinct invariants or make failures difficult to diagnose.

### 12.5 Separate unit and integration coverage

- Unit tests should isolate the owning algorithm with fixed inputs.
- Integration tests should prove that neighboring layers preserve the contract.
- End-to-end generation tests should cover representative boundary classes and structures without duplicating every lower-level arithmetic test.

### 12.6 Use mocking sparingly

Mock backend failures to verify exception translation or fallback policy. Do not mock the core algorithm in a test that claims to verify mathematical correctness.

### 12.7 Assert warnings precisely

When a warning is part of the contract, assert its category and meaningful message fragment. Also assert that warnings do not appear on paths where they are not expected.

### 12.8 Keep tests deterministic and bounded

Tests MUST NOT depend on random ordering, wall-clock timing, uncontrolled parallelism, or unbounded searches. Mark genuinely expensive tests as slow.

### 12.9 Run the appropriate suite

At minimum, an agent SHOULD run:

1. Focused tests for changed modules.
2. Relevant integration tests.
3. The non-slow suite when the environment permits:

```bash
pytest -m "not slow" tests/*
```

An agent MUST report exactly which tests were run and MUST NOT claim tests passed if they were not executed.

---

## 13. Performance rules

### 13.1 Correctness first

Do not trade exactness, determinism, or physical validity for speed.

### 13.2 Avoid avoidable quadratic work

When a bounded mathematical algorithm exists, prefer it over brute-force scanning. Preserve explicit tie-breaking and test equivalence to a brute-force reference on small cases.

### 13.3 Avoid redundant conversions and recomputation

Reuse already constructed CSLs, validated rotations, metadata, and exact embeddings within the same operation. Do not recompute them in multiple layers.

### 13.4 Be memory-aware for large structures

Large grain boundaries may contain hundreds of thousands of atoms. Avoid unnecessary full-array copies, Cartesian products, and dense pairwise-distance matrices.

Use spatial indexing or local-neighborhood methods for contact and duplicate checks.

### 13.5 Bound searches and constructions

Any potentially explosive exactification, determinant, area-index, repeat-count, or rationalization search MUST have explicit validated limits and a clear failure mode.

---

## 14. Compatibility and deprecation

### 14.1 Preserve established public behavior by default

A cleanup or refactor MUST NOT silently alter:

- Public argument defaults.
- Construction-mode semantics.
- Exact versus approximate status.
- Exception types.
- Serialized field meaning.
- Atom/species preservation.
- Periodicity behavior.

Intentional changes require focused tests and migration documentation.

### 14.2 Deprecate explicitly

Deprecated entry points SHOULD:

- Emit one clear `DeprecationWarning` per user call.
- Identify the replacement API.
- Remain covered by compatibility tests until removal.
- Avoid warning through the replacement path.

### 14.3 Do not preserve bugs as compatibility

If existing behavior violates a documented invariant, fix it. Preserve compatibility around valid contracts, not accidental corruption or physically invalid output.

---

## 15. Change discipline

### 15.1 Inspect before implementing

Before editing, an agent MUST inspect:

- The target module.
- Its direct callers and callees.
- Existing tests.
- Relevant exception and type definitions.
- Package exports.
- Shared constants and validators.

Do not implement a helper until confirming that an equivalent helper does not already exist.

### 15.2 Make the smallest coherent change

A change should be large enough to maintain architectural integrity but small enough to review.

Do not combine unrelated cleanup, formatting, renaming, and behavior changes in one patch.

### 15.3 Update all affected surfaces together

A behavior change may require coordinated updates to:

- Implementation.
- Tests.
- Docstrings.
- Public exports.
- CLI serialization/parsing.
- Defaults and shared limits.
- Migration notes.
- Generation or visual-review metadata.

Do not leave the repository in a state where one layer documents a contract that another layer no longer follows.

### 15.4 Do not duplicate repository code in generated patches

When contributing to an existing repository, patch the relevant files. Do not recreate large modules, copy entire subsystems into new files, or introduce parallel implementations.

### 15.5 Keep commits and reports factual

A change summary SHOULD state:

- What behavior changed.
- Why it changed.
- Which invariants are now enforced.
- Which tests were added or run.
- Any known limitations.

Do not claim performance improvement without measurement or correctness without tests or proof.

---

## 16. AI-agent workflow

An AI coding agent SHOULD follow this sequence for each task.

### Step 1: Identify the contract

State the user-visible behavior and the mathematical or physical invariants that must hold.

### Step 2: Identify the owning layer

Choose the module that owns the behavior. Avoid fixing symptoms in a caller when the defect belongs in a lower-level invariant or adapter.

### Step 3: Search for existing machinery

Find existing validators, exact helpers, dataclasses, constants, warnings, and tests before adding code.

### Step 4: Design failure behavior

Determine:

- Which exception type should escape.
- Whether translation is required.
- Whether fallback is allowed.
- Whether a warning is required.
- Which limits apply.

### Step 5: Implement the narrowest complete fix

Preserve conventions, exactness, determinism, immutability, and layer boundaries.

### Step 6: Add tests before broad cleanup

Add focused regression and invariant tests. Use integration tests where the bug crossed module boundaries.

### Step 7: Run targeted checks

Run the smallest useful test set first, then broader tests. Run documentation or static checks relevant to the touched code.

### Step 8: Review the diff as a maintainer

Check for:

- Duplicate logic.
- Silent fallbacks.
- Missing `:raises` fields.
- Incorrect exports.
- Mutable aliasing.
- Fixed-width overflow.
- Accidental float conversion.
- Unstable ordering.
- Ad hoc atom deletion.
- Tests of implementation trivia.
- Unrelated changes.

### Step 9: Report honestly

State what was changed, what was tested, and what remains uncertain. If a test could not be run, say so directly.

---

## 17. Prohibited patterns

An agent MUST NOT introduce the following without an explicit, reviewed exception:

- Silent exact-to-approximate fallback.
- Floating-point verification of an exact integer identity.
- Local copies of existing integer or validation helpers.
- Arbitrary atom deletion to repair a gap or overlap.
- Per-species deletion that breaks a complete conventional-cell group.
- Optimization logic inside clean structure generation.
- Public symbols that are undocumented, untested, or unexported inconsistently.
- Broad exception swallowing.
- Mutable aliases stored in frozen/domain result objects.
- Tests that primarily assert Python import internals rather than package behavior.
- Magic tolerances with no documented units or policy.
- Unbounded crystallographic searches.
- Claims that tests passed when they were not run.

---

## 18. Completion checklist

Before declaring a code change complete, verify all applicable items.

### Architecture

- [ ] The change is in the owning layer.
- [ ] No existing helper or subsystem was duplicated.
- [ ] Clean generation remains separate from optimization.

### Correctness

- [ ] Exact arithmetic remains exact.
- [ ] Row/column rotation conventions are preserved.
- [ ] Physical and chemical invariants are explicit.
- [ ] Complete atom-origin groups and stoichiometry are preserved where required.
- [ ] Construction modes retain their documented semantics.

### Validation and errors

- [ ] Public inputs are validated.
- [ ] Booleans are rejected where numeric inputs are expected.
- [ ] Exceptions use the correct hierarchy and are chained when translated.
- [ ] Fallbacks are explicit and warnings are actionable.
- [ ] Searches and construction sizes are bounded.

### API and documentation

- [ ] Public exports are intentional and consistent.
- [ ] Docstrings describe current behavior.
- [ ] All required `:raises` fields are present.
- [ ] Comments explain non-obvious reasons and invariants.

### Tests

- [ ] A regression test covers the defect or new behavior.
- [ ] Invariants are tested, not only example outputs.
- [ ] Tests are idiomatic pytest and deterministic.
- [ ] Relevant warnings and no-warning paths are asserted.
- [ ] Focused and broader test commands are recorded accurately.

### Maintainability

- [ ] The diff contains no unrelated refactor.
- [ ] No unnecessary array copies or large quadratic operations were added.
- [ ] Persistent results are immutable or have explicit mutation semantics.
- [ ] Output ordering, identifiers, metadata, and hashes remain deterministic.

---

## 19. Default decision rule

When uncertain, choose the implementation that is easiest to audit mathematically and physically:

- Validate early.
- Represent exact quantities exactly.
- Preserve complete structures rather than deleting around a problem.
- Make policy explicit.
- Fail clearly when the requested contract cannot be met.
- Keep generation deterministic and optimization separate.
- Prove the behavior with focused invariant tests.
