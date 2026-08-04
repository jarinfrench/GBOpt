# ADR 0005: Periodic Grain Construction Uses Canonical Reduced-Coordinate Representatives

* **Status:** Accepted
* **Date:** 2026-08-04
* **Decision owners:** GBOpt maintainers
* **Related roadmap items:** F0, GM3–GM7
* **Historical material:** `docs/history/gbmaker-periodic-selection/gbmaker_redesign.md`

## Context

Periodic grain construction must select exactly one representative from every periodic equivalence class inside the simulation cell.

Direct Cartesian clipping is not generally sufficient when the in-plane periodic vectors are oblique relative to the Cartesian y and z axes.

A point may be outside the raw Cartesian y/z rectangle while being periodically equivalent to a valid point inside the intended cell. Conversely, naïve wrapping and clipping can retain duplicate representatives on opposite periodic faces.

The previous GBMaker redesign work established a reduced-coordinate selection model. That decision should remain explicit as the staged GBMaker pipeline is extracted.

## Decision

Periodic membership and uniqueness in GBMaker grain construction are defined in reduced coordinates with respect to the selected in-plane basis.

Every periodic equivalence class must be represented exactly once in the canonical half-open cell.

## Selection basis

Construction stages must receive an explicit basis describing the simulation cell.

For periodic in-plane axes, the basis is derived from the primitive in-plane periodic vectors and the planned box dimensions.

For non-periodic axes, the basis uses the appropriate Cartesian direction or other explicitly defined physical selection direction.

Basis construction belongs to the GBMaker orientation, dimension-planning, or geometry stages according to the owning calculation. It must not be inferred from a file-format representation.

## Reduced coordinates

Candidate lattice origins or atoms are transformed into reduced coordinates:

```text
r_cartesian → r_reduced
```

Periodic reduced components are wrapped into the canonical interval:

```text
[0, 1)
```

The canonical representative is then mapped back to Cartesian coordinates.

A numerically equivalent upper periodic face must map to the lower face rather than survive as a second representative.

## Half-open convention

Periodic dimensions use a half-open selection convention.

Conceptually:

```text
0 <= u < 1
```

Tolerance handling may snap values close to zero or one, but it must preserve exactly one representative.

The tolerance must:

* be named;
* have documented units;
* be distinct from unrelated crystallographic and physical tolerances;
* be applied consistently across exact and approximate construction paths.

## Exact and approximate paths

Exact and approximate grain builders may enumerate lattice origins differently, but they must use the same canonical membership semantics.

The exact path must preserve exact integer crystallographic data for as long as possible. It must not convert exact orientation information through floating-point approximations merely to reuse an approximate selection method.

The approximate path may use floating-point geometry, but it must still produce deterministic canonical representatives.

## Complete basis groups

Selection, wrapping, clipping, and deduplication must preserve complete basis or conventional-cell origin groups when the construction contract requires complete chemical units.

The implementation must not:

* select individual species independently when that breaks stoichiometry;
* remove arbitrary atoms to repair a duplicate or gap;
* split a complete origin group across inconsistent periodic decisions.

Termination or deliberate nonstoichiometric behavior belongs to an explicit higher-level policy.

## Deterministic ordering

Canonicalization and deduplication must produce stable ordering.

The refactor must characterize and preserve order-sensitive output unless an ordering change is explicitly approved.

An order-insensitive physical match is useful for diagnosis but must not conceal an unexplained order-sensitive change.

## Topology

This decision concerns in-plane periodic representation during grain construction.

It does not assert that the boundary-normal direction is universally periodic or universally non-periodic.

GBOpt must support explicit topology such as:

* a periodic bicrystal with two interfaces;
* a non-periodic slab with one interface and free surfaces;
* optional vacuum along the boundary normal.

Boundary-normal topology is separate from in-plane reduced-coordinate membership.

## Consequences

### Positive

* Oblique periodic cells are represented correctly.
* Opposite periodic faces do not create duplicate atoms.
* Exact and approximate builders share one physical membership convention.
* Periodicity is explicit rather than inferred through Cartesian clipping.
* Construction remains deterministic and auditable.

### Negative

* Basis transformations and tolerance handling require careful tests.
* Exact and approximate builders cannot use simplistic raw-coordinate clipping.
* Historical output that depended on accidental duplicate or face-selection behavior may change only through an explicitly approved correction.

## Enforcement

Tests should cover:

* orthogonal and oblique in-plane bases;
* points on lower and upper periodic faces;
* points slightly inside and outside tolerance;
* periodic and non-periodic axis combinations;
* exact and approximate builders;
* complete multi-species origin groups;
* deterministic ordering;
* one representative per periodic equivalence class;
* atom coordinates within declared physical bounds after canonical mapping.

The staged GBMaker refactor must retain this decision while moving basis construction, coordinate transforms, filtering, and deduplication into separate modules.

