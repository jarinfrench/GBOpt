# Proposal: GBManipulator Abstraction Layer

## Executive decision

Adopt an **operator-level Strategy abstraction**:

1. Introduce a small abstract base class, `Manipulation`, representing one structural operation.
2. Keep `GBManipulator` concrete and convert it into a **coordination and compatibility facade** that owns normalized parent views, an RNG context, and a registry of available operations.
3. Represent each built-in operation as a separate concrete `Manipulation` implementation.
4. Replace the optimizer's hard-coded string dispatch with configurable operation specifications that can construct and parameterize any registered manipulation.
5. Place a narrow `StructureView` protocol or adapter between manipulation code and the current `Parent` representation, so this work does not freeze the known interface defects into the new public abstraction.

This is preferable to making the current `GBManipulator` class itself abstract. The existing class is not one operation; it is simultaneously a parent loader, state holder, random-number owner, geometry-algorithm container, and public facade. Turning that entire object into an ABC would formalize the current coupling rather than remove it.

## Basis in the paper

The paper defines the manipulator's architectural role as follows:

- it accepts one or two parent GB structures and creates a child structure;
- the currently implemented operations are slice-and-merge, right-grain translation, insertion, removal, and soft-mode displacement;
- structure creation, manipulation, optimization, objective evaluation, and eventually I/O are intended to remain modular;
- future abstract base classes are intended to formalize extension points and enable community-contributed manipulators.

The paper does **not** specify a concrete ABC signature, result type, plugin mechanism, or inheritance hierarchy. The design below therefore preserves the paper's stated separation of responsibilities while deriving the detailed API from the current implementation.

## Current source assessment

### What is already usable

The current implementation contains substantial, tested manipulation logic:

- `Parent` normalizes either a `GBMaker` instance or a LAMMPS file into left-grain, right-grain, full-system, GB-region, cell, and unit-cell data (`GBOpt/GBManipulator.py:101-612`).
- `GBManipulator` supports one- and two-parent construction and owns the random generator (`GBOpt/GBManipulator.py:874-967`).
- The existing operations are independently callable and generally avoid mutating the parent arrays in place (`GBOpt/GBManipulator.py:971-1803`).
- Tests cover direct manipulation behavior, parent parsing, translations, stoichiometric insertion/removal, soft-mode displacement, and minimizer integration.

These algorithms should be migrated, not rewritten wholesale.

### Where extension currently breaks down

#### 1. `GBManipulator` has too many responsibilities

The single module contains:

- LAMMPS data and dump parsing;
- reconstruction of grain ownership and the GB region;
- parent collection management;
- RNG management;
- numerical helper kernels;
- five manipulation algorithms;
- an unimplemented symmetry operation.

The class therefore has no narrow contract that a user can implement without inheriting unrelated behavior.

#### 2. Optimizers do not support arbitrary operations

`GBMinimizer.Mutator` accepts operation names, but its execution is a fixed `match` statement that recognizes only `insert_atoms`, `remove_atoms`, and `translate_right_grain` (`GBOpt/GBMinimizer.py:16-57`). Adding a method to `GBManipulator` is insufficient: the optimizer must also be edited.

The GA separately hard-codes `slice_and_merge`, including its 50 percent allocation and two-parent construction (`GBOpt/GBMinimizer.py:349-405`). This makes operation selection policy, parent arity, and operation implementation inseparable.

#### 3. Operation contracts are inconsistent

The current methods return several incompatible shapes:

- most operations return one structured NumPy array;
- insertion and removal can return `(child, changed_atoms)` tuples;
- soft-mode displacement returns a list of child arrays;
- operations communicate lineage through ad hoc strings in `GBMinimizer` rather than through the operation result.

An optimizer cannot consume arbitrary operations safely without knowing each method's special return convention.

#### 4. Parent arity is represented indirectly

The manipulator always stores a two-element mutable list and a separate `__one_parent` flag. Unary operations warn and ignore the second parent, while the binary operation raises if it is absent. A formal operation should declare its own arity and receive exactly the required number of parents.

#### 5. Manipulation is coupled to current interface assumptions

The current `Parent` reconstructs grain membership and the GB plane from x-coordinate partitions when loading files, and several operators consume fields such as `left_grain`, `right_grain`, `gb_indices`, and `gb_plane_x`. Those assumptions are precisely the area being corrected in the separate interface work. They should remain behind an adapter rather than become mandatory members of the new ABC.

#### 6. Reproducibility is not fully centralized

The class uses its stored generator in most paths, but some insertion/removal paths call the global `np.random.choice`. The constructor also treats `seed=0` as an absent seed. Extraction into operation objects should establish the invariant that all stochastic behavior uses the provided manipulation context.

#### 7. Algorithm implementation is mixed with orchestration

For example, insertion defines Delaunay and grid site-generation functions inside `insert_atoms`, contains unreachable code after a return, and combines site generation, type-count selection, stochastic selection, atom construction, and result formatting. This makes individual pieces difficult to test or reuse.

## Proposed architecture

### 1. `Manipulation`: the abstract extension point

Each structural operation becomes a small object with explicit configuration, parent requirements, validation, and application behavior.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ManipulationContext:
    rng: np.random.Generator


@dataclass(frozen=True)
class ManipulationResult:
    children: tuple[np.ndarray, ...]
    operation: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def child(self) -> np.ndarray:
        if len(self.children) != 1:
            raise ValueError("Result does not contain exactly one child")
        return self.children[0]


class Manipulation(ABC):
    name: ClassVar[str]
    parent_count: ClassVar[int]

    @abstractmethod
    def apply(
        self,
        parents: Sequence["StructureView"],
        *,
        context: ManipulationContext,
    ) -> ManipulationResult:
        """Return new child structures without mutating the parents."""
```

Concrete operations should normally be frozen dataclasses, so their parameters are explicit and serializable:

```python
@dataclass(frozen=True)
class TranslateRightGrain(Manipulation):
    dy: float
    dz: float
    dx: float = 0.0

    name: ClassVar[str] = "translate_right_grain"
    parent_count: ClassVar[int] = 1

    def apply(self, parents, *, context):
        ...
```

This object is both an executable operation and a complete description of what was requested. That makes lineage and reproducibility substantially cleaner than parsing strings such as `shift0.123dy0.456dz`.

### 2. `StructureView`: a narrow state seam

The ABC should not accept `GBMaker` or file paths. It should accept normalized structure views.

The first implementation should define a protocol around generic capabilities:

```python
class StructureView(Protocol):
    @property
    def atoms(self) -> np.ndarray: ...

    @property
    def box(self) -> np.ndarray: ...

    @property
    def periodicity(self) -> tuple[bool, bool, bool]: ...

    @property
    def unit_cell(self) -> UnitCell: ...

    def region_indices(self, name: str) -> np.ndarray: ...

    def metadata(self, key: str, default: Any = None) -> Any: ...
```

Recommended initial named regions are:

- `"grain:left"`;
- `"grain:right"`;
- `"interface:core"`.

The current `Parent` can be exposed through a `CurrentParentView` adapter. When the interface-fix project produces a stronger state object with explicit grain identity and boundary metadata, that object can implement `StructureView` directly or receive its own adapter.

The protocol should deliberately avoid requiring `gb_plane_x` or an x-contiguous grain partition. Operations that still need those assumptions during migration may request them as temporary metadata and raise a clear capability error when unavailable.

### 3. `GBManipulator`: concrete facade and coordinator

`GBManipulator` remains the public entry point in the first compatibility period. Its responsibilities become:

- normalize input sources into structure views through a source adapter/factory;
- hold an immutable tuple of parents;
- hold a `ManipulationContext`;
- validate operation arity;
- execute a `Manipulation` object;
- expose a registry-based convenience API;
- preserve legacy method calls as wrappers.

Illustrative interface:

```python
class GBManipulator:
    def __init__(
        self,
        *sources: StructureSource,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
        registry: ManipulationRegistry | None = None,
        source_adapter: StructureSourceAdapter | None = None,
    ) -> None:
        ...

    def apply(self, operation: Manipulation) -> ManipulationResult:
        ...

    def apply_named(self, name: str, /, **parameters: Any) -> ManipulationResult:
        ...
```

Legacy wrappers delegate to the new operations and preserve old return values:

```python
    def translate_right_grain(self, dy, dz, *, dx=0.0) -> np.ndarray:
        result = self.apply(TranslateRightGrain(dy=dy, dz=dz, dx=dx))
        return result.child
```

This allows existing examples and downstream users to continue working while new code uses the uniform result API.

### 4. Operation registry

A registry supports configuration-driven and user-contributed operations without requiring edits to `GBManipulator` or `GBMinimizer`.

```python
registry.register(TranslateRightGrain)
registry.register(InsertAtoms)
registry.register(RemoveAtoms)
```

Requirements:

- explicit registration rather than import-time global side effects;
- duplicate-name rejection unless replacement is explicitly requested;
- lookup by stable operation name;
- direct operation objects remain supported, so registration is optional for programmatic use;
- no packaging entry-point discovery in the first implementation. That can be added later if external plugin distribution becomes a real requirement.

### 5. Operation specifications for optimizers

Operation implementation and stochastic search policy should be separate.

```python
@dataclass(frozen=True)
class OperationSpec:
    name: str
    weight: float
    parameter_sampler: Callable[
        [Sequence[StructureView], np.random.Generator],
        Mapping[str, Any],
    ]
```

The optimizer selects an `OperationSpec`, selects the number of parents declared by the corresponding operation, samples parameters, constructs the operation, and calls `GBManipulator.apply`.

Built-in compatibility specifications can reproduce current behavior:

- insert one atom using the grid method;
- remove one atom;
- translate by sampled in-plane offsets;
- use slice-and-merge as a two-parent operation.

This eliminates the `match` statement and the GA's separate hard-coded crossover path. The GA may retain its current unary/binary proportions as policy defaults, but those proportions should no longer be embedded in the implementation of a specific operation.

### 6. Built-in operation modules

Recommended layout:

```text
GBOpt/
├── GBManipulator.py              # compatibility facade and public imports
└── manipulation/
    ├── __init__.py
    ├── base.py                   # Manipulation, context, result, errors
    ├── state.py                  # StructureView and current adapters
    ├── registry.py
    ├── specs.py                  # OperationSpec and parameter samplers
    ├── translation.py
    ├── crossover.py
    ├── density.py                # insertion/removal and shared helpers
    └── soft_modes.py
```

Use the lowercase `manipulation` package to avoid a name collision with the existing `GBOpt.GBManipulator` module and preserve the current import path.

### 7. Standardized errors

Recommended hierarchy:

```text
ManipulationError
├── ManipulationConfigurationError
├── ParentArityError
├── ParentCompatibilityError
├── MissingCapabilityError
└── ManipulationExecutionError
```

Existing `GBManipulatorValueError` can remain as a compatibility alias or wrapper during the transition.

### 8. Invariants of the new abstraction

Every concrete operation must satisfy:

1. **No parent mutation.** Parent atom arrays and metadata are read-only inputs.
2. **Explicit arity.** The operation declares the exact number of parents it consumes.
3. **Uniform output.** The operation returns `ManipulationResult`, even for zero, one, or multiple children.
4. **Central RNG.** All stochastic behavior uses `ManipulationContext.rng`; no global NumPy RNG calls.
5. **No I/O.** An operation does not load files, write files, submit calculator jobs, or evaluate objectives.
6. **Deterministic replay.** Given identical parents, operation configuration, and RNG state, the result is reproducible.
7. **Capability validation.** Missing regions or metadata produce typed errors, not silent fallback guesses.
8. **Serializable provenance.** Operation name, parameters, and important choices are available in the result metadata.
9. **Independent children.** Every returned child owns its array storage and does not alias a parent's writable data.

## Mapping the existing operations

| Existing method | New concrete operation | Arity | Result notes |
|---|---|---:|---|
| `translate_right_grain` | `TranslateRightGrain` | 1 | One child; requires right-grain region and cell/periodicity information |
| `slice_and_merge` | `SliceAndMerge` | 2 | One child; validates parent cell, species, and region compatibility |
| `remove_atoms` | `RemoveAtoms` | 1 | One child; removed atoms go in result metadata rather than changing return type |
| `insert_atoms` | `InsertAtoms` | 1 | One child; inserted atoms go in result metadata |
| `displace_along_soft_modes` | `DisplaceAlongSoftModes` | 1 | One or more children using the same result contract |
| `apply_group_symmetry` | Do not register until implemented | 1 | Avoid advertising an operation whose implementation always raises |

Site generation for insertion should be extracted into separate collaborators, for example `DelaunaySiteGenerator` and `GridSiteGenerator`, rather than represented by a string branch inside `InsertAtoms`. This is a second-level strategy seam and permits future user-supplied site-generation algorithms without subclassing the entire insertion operation.

## Interaction with the ongoing interface-fix work

This project should proceed now, but with a strict boundary:

### Safe to implement before interface fixes land

- `Manipulation`, `ManipulationContext`, and `ManipulationResult`;
- operation arity and registry;
- RNG centralization;
- optimizer operation specifications;
- extraction of algorithms into concrete operation classes;
- compatibility wrappers in `GBManipulator`;
- `StructureView` protocol and a temporary adapter over the current `Parent`;
- tests against current behavior.

### Do not stabilize yet

- x-coordinate reconstruction of grain identity as part of the public ABC;
- `gb_plane_x` as a required generic interface property;
- LAMMPS-specific file semantics in manipulation contracts;
- current assumptions that the interface is centered or that the GB region is a symmetric x window;
- a permanent serialized representation for manipulation state.

When the interface-fix state model is available, only the adapter and built-in capability queries should need revision. The operation ABC, result contract, registry, and optimizer integration should remain stable.

## Incremental implementation plan

### Phase 0 — Characterize and freeze current public behavior

- Inventory current constructors, methods, exceptions, warnings, return shapes, and import paths.
- Add focused characterization tests for every legacy wrapper.
- Add explicit tests for parent immutability and RNG replay.
- Decide which existing accidental behaviors are compatibility obligations and which are bugs to correct.
- Record that `seed=0` must be a valid deterministic seed.

**Exit criterion:** behavior is captured well enough that delegation can be introduced without silently changing existing users.

### Phase 1 — Add the core abstraction without moving algorithms

- Add `manipulation/base.py` with the ABC, context, result, and errors.
- Add `StructureView` and `CurrentParentView`.
- Add the registry.
- Add `GBManipulator.apply` while leaving existing methods untouched.
- Use an immutable parent tuple internally; retain the existing parent proxy only as a deprecated compatibility layer if tests or users require it.

**Exit criterion:** a tiny test-only custom operation can be applied through `GBManipulator` without modifying GBOpt source.

### Phase 2 — Extract right-grain translation as the vertical slice

Translation is the best first operation because it is relatively self-contained and already has detailed boundary-condition tests.

- Move the implementation to `TranslateRightGrain.apply`.
- Make the legacy method a thin wrapper.
- Confirm all existing translation tests pass unchanged.
- Add tests through the generic `apply` path and through registry lookup.

**Exit criterion:** one built-in operation uses the entire new path in production code.

### Phase 3 — Extract remaining unary operations

- Extract insertion and removal.
- Separate insertion-site generators from atom-selection logic.
- Move all stochastic calls to `context.rng`.
- Extract reusable local-order, neighbor, stoichiometry, and soft-mode helpers into focused private modules.
- Extract soft-mode displacement and normalize multiple-child results.
- Remove dead code and unreachable branches only after characterization tests are in place.

**Exit criterion:** all unary operations are concrete `Manipulation` implementations; legacy calls remain compatible.

### Phase 4 — Extract binary crossover and compatibility validation

- Implement `SliceAndMerge` with `parent_count = 2`.
- Validate box, species/type mapping, unit-cell assumptions, and region compatibility before creating a child.
- Preserve the current x-slice algorithm initially; changing the crossover geometry is a separate scientific enhancement.

**Exit criterion:** no operation depends on `GBManipulator.__one_parent` or silently ignores an extra parent.

### Phase 5 — Refactor minimizer dispatch

- Replace `Mutator`'s method-name dictionary and `match` statement with `OperationSpec` objects.
- Convert legacy `choices: list[str]` into built-in default specs.
- Let parent arity drive parent selection.
- Preserve current default GA proportions and mutation parameter distributions as compatibility policy.
- Store lineage from `ManipulationResult` rather than constructing ad hoc strings.

**Exit criterion:** a custom registered operation can participate in MC and GA without edits to `GBMinimizer.py`.

### Phase 6 — Consolidate source adaptation

- Move `GBMaker`/path normalization behind a `StructureSourceAdapter`.
- Keep LAMMPS parsing behavior available, but do not expand I/O scope in this project.
- Add an adapter for the interface-fixed state representation when that work lands.
- Deprecate direct assumptions that all non-`GBMaker` inputs are file paths.

**Exit criterion:** manipulation code is independent of how the structure entered GBOpt.

### Phase 7 — Public extension documentation and stabilization

- Document a minimal third-party manipulation example.
- Document arity, result, RNG, immutability, capability, and exception rules.
- Add API-reference exports from `GBOpt.manipulation`.
- Establish a deprecation timeline for mutable parent assignment and inconsistent legacy return options.

**Exit criterion:** an external user can implement, register, test, and use a custom operation from both direct and optimizer workflows.

## Testing strategy

### Contract tests for every operation

Create a reusable test suite that every operation can satisfy where applicable:

- rejects incorrect parent count;
- does not mutate parent atoms or metadata;
- returns `ManipulationResult`;
- returns correctly typed child arrays;
- provides stable operation name and parameters;
- reproduces results with the same RNG state;
- does not use global NumPy RNG;
- returns independent child arrays;
- raises typed capability or compatibility errors.

### Built-in regression tests

Retain current scientific and geometry tests, but run each behavior through two paths:

1. the existing legacy method;
2. the concrete operation through `GBManipulator.apply`.

The outputs should be equivalent for the same RNG state.

### Extension test

Add a small custom operation in tests, for example a deterministic displacement of a named region. It should be:

- defined entirely in the test module;
- optionally registered;
- callable directly;
- usable by the generic optimizer mutation path;
- absent from all hard-coded GBOpt dispatch statements.

This is the most important proof that the abstraction works.

### Optimizer tests

- legacy string choices still produce the current built-in defaults;
- weighted custom specs are selectable;
- unary and binary operations receive the correct number of parents;
- multi-child results have an explicit optimizer policy, such as selecting one child or expanding the candidate population;
- lineage is structured and replayable.

## Advantages of the selected approach

1. **It creates the extension point at the correct granularity.** Users implement one manipulation, not an entire replacement coordinator.
2. **It preserves the current public API.** Existing direct calls and examples can remain functional during migration.
3. **It removes optimizer hard-coding.** New operations do not require edits to `GBMinimizer`.
4. **It supports composition.** A single `GBManipulator` can use any combination of built-in and user-supplied operations.
5. **It handles one-parent, two-parent, and multi-child behavior explicitly.** These are contract properties rather than special cases.
6. **It improves reproducibility and provenance.** Operation objects and results carry explicit parameters and use one RNG context.
7. **It is compatible with the separate interface remediation.** The state adapter can change without redesigning the operation API.
8. **It keeps scientific algorithms independently testable.** Translation, density modification, crossover, and soft-mode logic can evolve separately.
9. **It aligns with the paper's modular intent.** Manipulation remains separate from creation, optimization, objective evaluation, calculator execution, and I/O.

## Disadvantages and costs

1. **More concepts and files.** The design introduces operation objects, contexts, results, views, specs, and a registry.
2. **Temporary dual API.** Legacy methods and the generic operation API must coexist until a later deprecation cycle.
3. **Migration effort.** The current 1,800-line module has intertwined helper functions and tests; extraction must be incremental.
4. **Result-wrapper friction.** Internal optimizer code and new users must adapt from raw arrays to `ManipulationResult`.
5. **Capability design requires discipline.** An overly broad `StructureView` would simply recreate `Parent`; an overly narrow one would force repeated type checks and metadata escape hatches.
6. **Plugin discovery is intentionally deferred.** Users can register external operations programmatically, but package-level automatic discovery would be a later feature.

These costs are acceptable because the alternative—adding more methods and more `match` cases—continues to increase coupling and makes the promised extension point ineffective.

## Alternatives considered and rejected

### Alternative 1 — Make the current `GBManipulator` class an ABC

Under this approach, `GBManipulator` would define abstract methods and application-specific subclasses would implement or override them.

**Pros**

- superficially matches the phrase "abstract base class for GBManipulator";
- small initial change;
- existing optimizer type annotations could remain similar.

**Why rejected**

- the current class is a coordinator and collection of unrelated operations, not one substitutable behavior;
- a subclass adding one operation would inherit file parsing, parent proxies, all other operations, and internal assumptions;
- multiple independently developed operations would not compose cleanly;
- `GBMinimizer.Mutator` would still require hard-coded knowledge of method names and parameter generation;
- it would stabilize the current flawed state assumptions in the inheritance contract.

This approach formalizes the monolith rather than creating an abstraction layer.

### Alternative 2 — One `GBManipulator` subclass per operation

For example, `TranslationManipulator`, `InsertionManipulator`, and `RemovalManipulator` would each inherit from a common manipulator base.

**Pros**

- each subclass has a recognizable scientific purpose;
- traditional object-oriented structure;
- operations can override common hooks.

**Why rejected**

- optimizers need several operations at once, causing object switching, wrapper objects, or multiple inheritance;
- each subclass still tends to own parents and RNG even when these should be shared;
- binary crossover and multi-child soft-mode displacement make a uniform manipulator-subclass lifecycle awkward;
- third-party composition is poorer than passing operation objects into one coordinator;
- it encourages inheritance for code reuse where composition is more appropriate.

### Alternative 3 — Use only a dictionary of callables

A registry could map strings to functions with a loose signature such as `func(parents, rng, **kwargs) -> np.ndarray`.

**Pros**

- smallest implementation;
- functions are easy to register and test;
- little object-oriented machinery.

**Why rejected as the primary public design**

- no formal place for parent arity, capability requirements, validation, result multiplicity, or serializable parameters;
- inconsistent callable signatures and return types would likely reappear;
- provenance and documentation are weak;
- configuration-driven construction becomes ad hoc;
- static analysis and discoverability are poorer.

A callable adapter can still be provided as a convenience, but it should adapt a function into the formal `Manipulation` contract rather than replace that contract.

### Alternative 4 — Defer all abstraction work until the interface fixes are complete

**Pros**

- avoids writing a temporary adapter;
- permits the final state representation to drive the API immediately.

**Why rejected**

- the operation/result/registry/optimizer seams are largely independent of the exact interface geometry;
- current hard-coded dispatch blocks arbitrary manipulation work now;
- a narrow adapter is cheaper than allowing the new abstraction to depend directly on the unfinished interface;
- delaying would couple two sizable refactors and increase integration risk.

## Why this approach was selected

The selected design is the only considered option that simultaneously:

- preserves the existing public behavior during an incremental migration;
- makes a single arbitrary manipulation independently implementable;
- allows optimizers to consume new operations without source edits;
- expresses parent arity and multi-child output explicitly;
- centralizes stochastic state and provenance;
- avoids making the current x-partition and file-loading assumptions permanent;
- maintains the paper's separation between creation, manipulation, optimization, objective evaluation, and I/O.

The central design judgment is that **the abstract entity is a manipulation operation, while `GBManipulator` is the service that applies operations to normalized parents**.

## Proposed acceptance criteria

The abstraction-layer project is complete when all of the following are true:

1. A third-party manipulation can be implemented outside the GBOpt package by subclassing `Manipulation`.
2. It can be applied directly through `GBManipulator.apply` without modifying GBOpt source.
3. It can participate in MC and GA through an `OperationSpec` without modifying `GBMinimizer.py`.
4. Parent count, configuration, RNG, result shape, and lineage are represented explicitly.
5. All built-in operations use the new abstraction internally.
6. Existing direct `GBManipulator` method calls remain compatible for the agreed deprecation window.
7. All stochastic paths use the supplied NumPy generator, including `seed=0`.
8. Manipulation code performs no file I/O or objective evaluation.
9. The abstraction accepts the current parent representation through an adapter and can accept the future interface-fixed state without changing the operation ABC.
10. Existing non-slow tests pass, and new contract, extension, and optimizer-integration tests pass.

## Recommended first implementation slice

Begin with Phases 0 through 2 only:

- add the core abstraction types and state adapter;
- add `GBManipulator.apply`;
- extract `TranslateRightGrain`;
- preserve the legacy translation method as a wrapper;
- add a test-only custom manipulation proving external extensibility.

That slice is small enough to review rigorously, establishes the end-to-end architecture, and avoids entangling the first change with the more complex insertion/removal and soft-mode algorithms.
