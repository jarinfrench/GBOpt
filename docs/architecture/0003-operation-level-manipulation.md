# ADR 0003: Manipulations are operation-level strategies

- **Status:** Accepted
- **Date:** 2026-08-04
- **Decision owners:** Manipulation track

## Context

`GBManipulator` currently coordinates parents and RNG state while also implementing
multiple unrelated algorithms. Optimizers dispatch by hard-coded strings and special
cases, so adding one operation requires editing several modules.

## Decision

The extension point is one manipulation operation, not an abstract replacement for the
entire `GBManipulator` facade.

Each operation declares its parent arity, validates capabilities, consumes explicit
interface state, uses a supplied `numpy.random.Generator`, does no I/O or evaluation,
and returns a uniform `ManipulationResult` containing complete `InterfaceCandidate`
children and structured provenance.

`GBManipulator` remains a concrete coordination and compatibility facade. Existing
methods retain their current return shapes during migration and delegate to operation
objects after extraction.

## Consequences

- Third-party operations can be defined without subclassing file loading, all built-in
  algorithms, or optimizer policy.
- Parent mutation and global NumPy RNG calls are prohibited.
- Unary, binary, and multi-child behavior become explicit contract properties.
- Optimizer integration occurs later through immutable `OperationSpec` values; it is not
  part of the initial abstraction PR.
