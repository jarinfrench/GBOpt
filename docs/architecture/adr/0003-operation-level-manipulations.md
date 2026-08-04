# ADR 0003: Manipulations Are Operation-Level Strategies

* **Status:** Accepted
* **Date:** 2026-08-04
* **Decision owners:** GBOpt maintainers
* **Related roadmap items:** F2, IO4, MAN1–MAN5

## Context

The current `GBManipulator` combines several responsibilities:

* parent normalization and file-backed construction;
* random-number ownership;
* structural state access;
* numerical helper functions;
* translation;
* termination cycling;
* interface separation;
* insertion and removal;
* soft-mode displacement;
* binary slice-and-merge crossover;
* compatibility behavior used by optimizers.

This is not one substitutable algorithm. Making the entire class abstract would formalize the monolith rather than create a useful extension point.

Optimizer integration is also hard-coded. Operation names are interpreted through fixed dispatch logic, and binary crossover is handled as a separate optimizer path. Adding a new manipulation therefore requires editing GBOpt internals even when the manipulation itself is independently implementable.

## Decision

The abstract extension point is one manipulation operation.

`GBManipulator` remains a concrete coordination and compatibility facade that applies operation objects to normalized candidates.

## Operation contract

Each operation must declare:

* a stable operation name;
* exact parent arity;
* explicit configuration;
* required capabilities;
* an `apply` implementation.

Conceptually:

```python
class Manipulation:
    name: ClassVar[str]
    parent_count: ClassVar[int]

    def apply(
        self,
        parents: Sequence[InterfaceCandidate],
        *,
        context: ManipulationContext,
    ) -> ManipulationResult:
        ...
```

Concrete operations should normally be immutable dataclasses whose fields describe all requested parameters.

## Result contract

The internal result must contain complete candidates:

```python
@dataclass(frozen=True, slots=True)
class ManipulationResult:
    children: tuple[InterfaceCandidate, ...]
    operation: str
    parameters: Mapping[str, JSONValue]
    metadata: Mapping[str, JSONValue]
```

Raw NumPy arrays are insufficient because they do not preserve:

* persistent grain labels;
* interface topology;
* authoritative interface-plane position;
* physical grain bounds;
* periodicity;
* accumulated interface state.

Legacy wrappers may continue returning historical array or tuple forms during the compatibility period.

## `GBManipulator` role

The facade may:

* normalize supported parent sources;
* own or receive a manipulation RNG context;
* validate operation arity;
* execute an operation;
* support explicit registry lookup;
* expose legacy methods as wrappers.

The facade must not require every extension author to inherit unrelated built-in algorithms.

## Registry policy

An explicit registry may associate stable names with operation classes.

Requirements:

* no mandatory import-time registration side effects;
* duplicate names rejected unless replacement is explicit;
* direct operation-object use remains supported;
* automatic package entry-point discovery is deferred;
* registration is not required for normal programmatic use.

## Randomness

All stochastic behavior must use `ManipulationContext.rng`, backed by `numpy.random.Generator`.

Operations must not call global `numpy.random` functions.

`seed=0` is a valid deterministic seed and must not be interpreted as an absent seed.

Given identical parents, operation configuration, and RNG state, an operation should reproduce the same result.

## Operation invariants

Every operation must satisfy:

1. Parent arrays and metadata are not mutated.
2. Parent arity is explicit.
3. Children have independent storage.
4. Complete interface state is preserved or intentionally updated.
5. Missing capabilities raise typed errors.
6. File I/O is not performed.
7. Objective evaluation is not performed.
8. Important choices and parameters are represented as structured metadata.
9. Topology restrictions are validated before modification.
10. Ownership is not reconstructed from atom coordinates.

## Built-in mapping

The planned operation modules include:

| Existing behavior               | Operation                      | Arity |
| ------------------------------- | ------------------------------ | ----: |
| Right-grain rigid translation   | `TranslateRightGrain`          |     1 |
| Grain-local termination cycling | grain-local cycling operation  |     1 |
| Slab termination cycling        | slab cycling operation         |     1 |
| Interface separation            | interface separation operation |     1 |
| Atom insertion                  | `InsertAtoms`                  |     1 |
| Atom removal                    | `RemoveAtoms`                  |     1 |
| Soft-mode displacement          | `DisplaceAlongSoftModes`       |     1 |
| Slice-and-merge crossover       | `SliceAndMerge`                |     2 |

Unimplemented operations must not be registered as available.

## Optimizer integration

Optimization policy is separate from operation implementation.

An immutable `OperationSpec` will describe:

* operation identity;
* selection weight;
* parameter sampler;
* optional policy metadata.

The optimizer will:

1. select an `OperationSpec`;
2. determine parent count from the operation;
3. select the required parents;
4. sample parameters through the optimizer RNG;
5. construct or resolve the operation;
6. apply it through `GBManipulator`;
7. consume structured lineage from `ManipulationResult`.

Legacy string choices may be translated into built-in specifications during the compatibility period.

## Sequencing constraint

Manipulation extraction begins only after:

* shared interface-domain types have been extracted;
* `Parent.from_structure()` and source adaptation are established.

This avoids stabilizing file parsing and coordinate-derived ownership as part of the operation API.

The required sequence is:

```text
F2 → IO4 → MAN1
```

## Consequences

### Positive

* Third parties can implement one operation without subclassing the entire facade.
* Unary, binary, and multi-child behavior share one contract.
* Optimizers no longer require hard-coded operation names.
* RNG and provenance are centralized.
* Interface state is preserved across every operation.
* Built-in algorithms become independently testable.

### Negative

* Legacy and new result forms temporarily coexist.
* Operation, context, result, registry, and specification types add concepts.
* Algorithm extraction must proceed incrementally to avoid scientific regressions.

## Enforcement

Contract tests should verify:

* correct arity;
* parent immutability;
* independent children;
* complete `InterfaceCandidate` output;
* deterministic replay;
* no global RNG use;
* typed capability failures;
* structured parameters and lineage;
* custom operation participation without editing GBOpt source.

