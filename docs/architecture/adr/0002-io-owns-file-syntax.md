# ADR 0002: I/O Owns File Syntax and Transient Serialization Identity

* **Status:** Accepted
* **Date:** 2026-08-04
* **Decision owners:** GBOpt maintainers
* **Related roadmap items:** F2, IO1–IO5, GM8, EVAL2, CP2–CP5

## Context

File parsing and writing responsibilities are currently distributed across `GBMaker`, `Parent`, `FileGrainOwnership`, minimizer evaluation paths, and user callbacks.

This coupling has several consequences:

* `GBMaker` contains LAMMPS-specific serialization;
* `Parent` detects and parses file formats while also interpreting grain-boundary state;
* optimizer reload paths may reconstruct physical grain ownership from coordinates;
* transient LAMMPS atom IDs can be mistaken for persistent identity;
* scalar, batch, and checkpoint reload paths can perform different validation;
* adding another structure format would require changes in classes that should not own file syntax.

GBOpt needs a format-neutral structural model while preserving strict ownership-aware round trips.

## Decision

All external structure syntax belongs to the `GBOpt.io` package.

The I/O layer will use independent reader and writer contracts. It will convert between external representations and neutral structure records, but it will not own grain-boundary interpretation, calculator execution, or optimization policy.

## Reader and writer separation

Readers and writers are independent because capabilities are asymmetric.

A format may be:

* readable but not writable;
* writable but not readable;
* single-frame or multi-frame;
* capable or incapable of preserving atom IDs, charges, cell geometry, or metadata.

The core contracts will therefore be conceptually equivalent to:

```python
class StructureReader:
    def read(...) -> StructureData:
        ...

class StructureWriter:
    def write(..., structure: StructureData) -> WriteResult:
        ...
```

A single monolithic read/write base class is not required.

## Format-neutral structure data

Readers return `StructureData`.

`StructureData` preserves:

* coordinates and species;
* cell and origin;
* periodicity;
* supported row-aligned fields;
* external IDs where present.

It does not infer:

* left/right grain ownership;
* interface topology;
* interface-plane location;
* optimization lineage.

A reader must preserve source information accurately even when a downstream GBOpt algorithm supports only a subset of that geometry. Unsupported downstream use should fail at the consuming boundary.

## Persistent state remains outside I/O syntax

Persistent interface identity belongs to `GrainOwnership` and `InterfaceCandidate`, not to a LAMMPS, XYZ, CIF, VASP, or other syntax adapter.

Formats that cannot represent all required interface state may later use a versioned sidecar or explicit lossy-write policy. Such support is deferred until the core reader, writer, and candidate-loader contracts have stabilized.

A failed explicit ownership load must not silently fall back to midpoint or coordinate-based grain inference.

## Transient identifier lifetime

Writers may assign external atom IDs.

Those IDs are valid only for the serialization operation and its corresponding validated return path.

The writer returns the exact mapping in `WriteResult`. A candidate-local mapping may then be constructed for ownership-aware reload.

The following are prohibited:

* treating LAMMPS atom IDs as permanent atom identity;
* reusing a mapping for a different candidate;
* assuming file row order preserves in-memory row order;
* reconstructing ownership from relaxed x coordinates when explicit ownership exists.

## `Parent` responsibility

`Parent` remains a GB-domain interpretation and compatibility object.

It should gain an explicit construction path equivalent to:

```python
Parent.from_structure(
    structure,
    *,
    unit_cell,
    gb_thickness,
    grain_ownership=None,
)
```

`Parent` may:

* validate structure compatibility with its domain needs;
* create whole-system and region views;
* use persistent ownership labels when supplied;
* compute geometric GB-region membership around an authoritative plane.

`Parent` must not retain format-specific parser implementation.

The legacy filename constructor may remain temporarily as a delegating compatibility facade.

## Candidate reload

All ownership-aware reconstruction will use one central candidate-loader service.

The loader validates, as applicable:

* atom count;
* unique and expected transient IDs;
* species associated with each ID;
* finite coordinates;
* cell and box compatibility;
* topology and periodicity;
* frame-selection semantics;
* ownership-array alignment;
* supported variable-cell behavior.

The same loader should be used for:

* evaluator-returned structures;
* scalar evaluation;
* batch evaluation;
* Monte Carlo resume;
* genetic-algorithm resume;
* population reconstruction.

## Calculator separation

I/O writes and reads structures. It does not:

* launch LAMMPS;
* submit scheduler jobs;
* parse objective values as optimizer policy;
* wait for jobs;
* choose penalty values;
* decide whether a candidate is accepted.

Calculator execution and evaluator normalization belong above the I/O layer.

## Consequences

### Positive

* New formats can be added without modifying `Parent`, `GBMaker`, or minimizers.
* LAMMPS-specific transformations leave the construction package.
* Ownership-aware reconstruction has one validation path.
* Candidate-local ID mappings become explicit.
* File syntax and physical interpretation are separately testable.

### Negative

* Compatibility constructors and wrappers are required during migration.
* Sidecar and lossy-format policy must be designed later.
* Some existing convenience behavior based on content sniffing may need to become explicitly legacy.

## Enforcement

Architecture tests should verify that:

* `GBOpt.io` does not import `GBMaker` or minimizers;
* `GBOpt.gbmaker` does not import I/O;
* a standalone writer accepts neutral structure data;
* `Parent` contains no format-specific parser implementation after IO4;
* ownership-aware reload paths use the central loader;
* explicit ownership failures never fall back to geometric grain inference.

