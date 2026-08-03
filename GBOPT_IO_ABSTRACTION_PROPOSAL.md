# GBOpt I/O Abstraction Layer Proposal

## Status and scope

This document proposes an abstraction layer for **structure input and output** in GBOpt. It is based on:

- the modular architecture and planned file-reader/file-writer extension points described in *GBOpt: Grain boundary structure optimization using Monte Carlo and evolutionary algorithms*;
- the current source archive, particularly `GBOpt/GBMaker.py`, `GBOpt/GBManipulator.py`, `GBOpt/GBMinimizer.py`, and `GBOpt/FileGrainOwnership.py`;
- the interface-repair invariants already present in the source, including explicit grain ownership, authoritative `gb_plane_x`, candidate-local LAMMPS atom-ID mappings, and validated evaluator-return reloads.

The proposal addresses representation conversion between GBOpt and external structure formats. It does **not** combine structure I/O with calculator execution, job submission, objective evaluation, or optimizer policy. Those are separate extension points in the architecture described by the paper.

## Executive recommendation

Build the I/O layer around two independent abstract base classes:

1. `StructureReader`, which converts an external structure representation into a canonical GBOpt structure record.
2. `StructureWriter`, which converts that canonical structure record into an external representation.

The canonical structure record should contain only format-neutral atomic and cell data. Grain-boundary-specific state—especially persistent left/right grain ownership, `gb_plane_x`, interface topology, and coordinate tolerances—should remain in a separate domain object and be composed with the structure record when constructing a `Parent`.

This preserves the most important invariant established by the current interface fixes:

> LAMMPS atom IDs are transient serialization identifiers used to align a particular file round trip; they are not persistent optimizer-wide atom identities.

The first implementation should extract the existing LAMMPS data and dump handling into the new abstraction without changing externally visible behavior. Once that migration is complete, formats such as XYZ, CIF, VASP, GULP, and Quantum ESPRESSO can be added as adapters rather than by modifying `Parent`, `GBMaker`, or the minimizers.

---

## 1. Architectural context

### 1.1 Direction established by the paper

The paper presents GBOpt as a modular workflow in which:

- `GBMaker` creates structures;
- `GBManipulator` changes structures;
- `GBMinimizer` controls the optimization process;
- external calculators or user callbacks evaluate structures;
- data are exchanged through `Parent` objects;
- future abstract base classes formalize extension points;
- future I/O support includes common atomistic formats such as XYZ and CIF while still allowing user-defined readers and writers.

The workflow diagram on page 3 places structure creation/loading, manipulation, external evaluation, and optimizer state updates in distinct parts of the process. The I/O layer should follow that separation. A reader or writer should translate representations, not infer optimization policy or perform grain-boundary manipulations.

The paper does not prescribe concrete class names or method signatures for the future I/O abstraction. The interfaces in this proposal are therefore a source-informed design that realizes the paper's intended extension point.

### 1.2 Current source-code boundary

The current code distributes I/O responsibilities across several classes:

- `GBMaker.write_lammps()` serializes structures to LAMMPS data files.
- `Parent` detects whether a file is a LAMMPS data file or dump file and then parses it.
- `Parent` also applies GB-specific interpretation, including constructing left/right grain views and determining GB-region membership.
- `FileGrainOwnership.py` contains stricter LAMMPS readers, persistent grain-label metadata, candidate-local serialization mappings, and the validated explicit-ownership reload path used after evaluator execution.
- `GBMinimizer` must repeatedly serialize, evaluate, and reload candidates.

This means syntax parsing, generic structure representation, grain-boundary interpretation, and evaluator handoff are not yet cleanly separated.

The interface fixes already point toward the correct abstraction. `GrainOwnership` stores persistent grain labels and interface topology separately from coordinates, while `CandidateFileMapping` creates a fresh candidate-local ID mapping for each serialized evaluator round trip. `reload_explicit_manipulator()` validates atom count, IDs, species, box bounds, boundary topology, and ownership before reconstructing the candidate.

The new I/O layer should generalize that separation rather than replacing it with a format-specific object model.

---

## 2. Design goals and non-goals

### 2.1 Goals

The abstraction should:

- support multiple structure formats without modifying `Parent` or optimizer algorithms;
- preserve explicit grain ownership and authoritative interface metadata;
- distinguish persistent domain identity from temporary file identifiers;
- support readable-only, writable-only, and bidirectional formats;
- permit strict validation of evaluator-returned structures;
- represent orthogonal and non-orthogonal cells without prematurely promising that all GBOpt algorithms support them;
- make lossy conversions explicit;
- retain backward compatibility while current public entry points migrate;
- make third-party readers and writers small and independently testable;
- avoid requiring a particular calculator package or atomistic ecosystem.

### 2.2 Non-goals

The initial abstraction should not:

- redesign `Atom.atom_dtype`;
- make ASE, pymatgen, or another external object model mandatory;
- combine file I/O with calculator execution;
- infer persistent grain ownership from relaxed coordinates;
- treat LAMMPS atom IDs as stable atom identities;
- require immediate support for every format named in the long-term roadmap;
- silently coerce unsupported geometry or discard GB metadata.

---

## 3. Selected architecture

## 3.1 Canonical structure record

Introduce a narrow canonical representation, provisionally named `StructureData`:

```python
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class StructureData:
    atoms: np.ndarray
    cell: np.ndarray
    origin: np.ndarray
    periodic: tuple[bool, bool, bool]

    atom_ids: np.ndarray | None = None
    charges: np.ndarray | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
```

### Required semantics

- `atoms` initially uses the existing `Atom.atom_dtype`.
- `cell` is a full `3 x 3` cell matrix.
- `origin` is explicit rather than assumed to be zero.
- `periodic` records boundary periodicity independently for all three axes.
- Optional per-atom arrays are row-aligned with `atoms`.
- `atom_ids` are external serialization identifiers, not persistent atom identities.
- Arrays are copied or made read-only at the object boundary.
- The object performs no left/right grain classification.
- Unsupported downstream geometry is rejected by the downstream consumer, not misrepresented by the reader.

A full cell matrix is preferable to using only `box_dims` because future CIF, VASP, Quantum ESPRESSO, and triclinic LAMMPS adapters need a format-neutral representation. GBOpt algorithms may initially continue to accept only the subset they currently support, but the reader should preserve the source geometry accurately.

## 3.2 Keep interface state separate

Continue to use `GrainOwnership` as a separate GB-domain object. Its current responsibilities are appropriate:

- persistent left/right grain labels aligned with in-memory atom rows;
- authoritative `gb_plane_x`;
- in-plane periodicity;
- supported right-grain x bounds;
- coordinate tolerance;
- outer-x interface topology;
- initial file IDs used only for verified row alignment.

Conceptually, a loaded interface structure is the composition:

```python
@dataclass(frozen=True, slots=True)
class LoadedInterfaceStructure:
    structure: StructureData
    grain_ownership: GrainOwnership | None
```

This answers two independent questions:

1. What atoms, cell, and boundary conditions were read?
2. What do those atoms mean within GBOpt's interface model?

Formats such as XYZ can answer the first question but cannot, by themselves, reliably answer the second. The API should represent that limitation explicitly rather than infer grain ownership from current x coordinates.

## 3.3 Independent reader and writer ABCs

Use separate abstract base classes:

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO


StructureSource = str | Path | IO[str] | IO[bytes]
StructureTarget = str | Path | IO[str] | IO[bytes]


class StructureReader(ABC):
    """Convert an external representation into canonical GBOpt structure data."""

    @abstractmethod
    def read(
        self,
        source: StructureSource,
        *,
        options: object | None = None,
    ) -> StructureData:
        raise NotImplementedError


class StructureWriter(ABC):
    """Serialize canonical GBOpt structure data."""

    @abstractmethod
    def write(
        self,
        target: StructureTarget,
        structure: StructureData,
        *,
        options: object | None = None,
    ) -> "WriteResult":
        raise NotImplementedError
```

Readers and writers should be separate because file-format capabilities are not symmetric. A format may be:

- readable and writable;
- readable only;
- writable only;
- single-structure;
- multi-frame;
- lossless for generic structure data but lossy for GBOpt interface state.

Separate ABCs avoid empty methods and routine `NotImplementedError` branches in otherwise valid adapters.

## 3.4 Explicit write results

A writer should return a result object rather than only producing a file:

```python
@dataclass(frozen=True, slots=True)
class WriteResult:
    target: Path | None
    atom_ids: np.ndarray | None
    row_to_atom_id: np.ndarray | None
    digest: str | None
    losses: tuple[str, ...]
```

This formalizes information already needed by the repaired evaluator handoff. A LAMMPS writer may assign IDs `1..N` in deterministic row order. The optimizer needs the exact mapping for validating the returned structure and restoring persistent grain labels, but that mapping must expire with the corresponding serialization event.

`WriteResult` should therefore be the source used to construct `CandidateFileMapping`; it should not be promoted into persistent atom identity.

## 3.5 Format-specific options

Avoid an untyped `**kwargs` contract. Define validated option dataclasses for each adapter:

```python
@dataclass(frozen=True, slots=True)
class LammpsDataWriteOptions:
    type_map: Mapping[str, int]
    charges: Mapping[str, float] | np.ndarray | None = None
    precision: int = 16
    include_type_labels: bool = True
    triclinic: bool = False
```

```python
@dataclass(frozen=True, slots=True)
class LammpsDumpReadOptions:
    frame: int | Literal["first", "last"] = "first"
    coordinate_style: Literal["wrapped", "unwrapped", "scaled"] | None = None
    type_map: Mapping[int, str] | None = None
```

This keeps adapter behavior explicit, testable, and type-checkable.

## 3.6 Capability declarations

Each adapter should expose capabilities such as:

```python
@dataclass(frozen=True, slots=True)
class FormatCapabilities:
    atom_ids: bool
    charges: bool
    periodic_cell: bool
    triclinic_cell: bool
    multiple_frames: bool
    arbitrary_metadata: bool
    interface_state: bool
```

Capability declarations support early validation and clear error messages. They are especially important for formats such as XYZ, which may serialize coordinates and species while losing cell, periodicity, IDs, or GB-specific metadata depending on the variant used.

---

## 4. Proposed package structure

```text
GBOpt/
    io/
        __init__.py
        base.py
        model.py
        errors.py
        registry.py

        lammps/
            __init__.py
            common.py
            data_reader.py
            data_writer.py
            dump_reader.py

        xyz.py
        cif.py

    FileGrainOwnership.py
    GBManipulator.py
    GBMaker.py
    GBMinimizer.py
```

`FileGrainOwnership.py` may be renamed or reorganized later, but its domain objects should remain outside `GBOpt/io`. Grain ownership and interface topology are physical/domain state, not file syntax.

### Intended dependency direction

```text
format adapter
    -> StructureData
    -> Parent.from_structure(...)
    -> GBManipulator
    -> GBMinimizer
```

The I/O package must not import optimizer or campaign policy. In particular, it should not depend on:

- `GBMinimizer`;
- a specific minimizer implementation;
- campaign screening rules;
- objective callbacks;
- calculator job-submission logic.

The ownership-aware candidate reload service may depend on both I/O results and GB-domain metadata, but should sit above the format adapters rather than inside them.

---

## 5. Registry and format selection

Provide an explicit registry:

```python
registry.register_reader("lammps-data", LammpsDataReader)
registry.register_reader("lammps-dump", LammpsDumpReader)
registry.register_writer("lammps-data", LammpsDataWriter)
```

User-facing helpers can then provide a compact API:

```python
structure = gbio.read(path, format="lammps-data")
result = gbio.write(path, structure, format="lammps-data")
```

Format resolution should use the following order:

1. an explicit `format=` argument;
2. an unambiguous registered file suffix;
3. content sniffing only when explicitly requested or through a legacy compatibility path.

The current `Parent` constructor examines the first several lines and selects a parser based on recognized tokens. That behavior is useful for compatibility, but it should not remain the primary interface because:

- LAMMPS data and dump files may use arbitrary suffixes;
- formats can share common header tokens;
- truncated files can be misclassified;
- error messages become format-detection errors rather than precise parser errors.

Third-party discovery through Python entry points should be added only after the built-in contracts are stable.

---

## 6. GB-specific metadata and lossy formats

## 6.1 Versioned sidecar metadata

For formats that cannot embed all GBOpt state, use a versioned sidecar:

```text
candidate.dat
candidate.gbopt.json
```

The sidecar should contain, at minimum:

- schema version;
- a digest or other binding to the corresponding structure file;
- expected atom and species counts;
- authoritative `gb_plane_x`;
- persistent left/right grain ownership;
- periodicity and interface topology;
- coordinate tolerance;
- relevant provenance.

The load sequence should be:

1. read the structure through a `StructureReader`;
2. read and validate the sidecar;
3. verify that the sidecar belongs to the exact structure;
4. align labels through the relevant transient serialization mapping;
5. construct `Parent` from `StructureData` plus `GrainOwnership`;
6. discard serialization identity when its round trip is complete.

Failure to validate explicit metadata must not fall back to midpoint-based ownership inference.

## 6.2 Explicit loss policy

When a writer cannot represent all requested state, it should either:

- write a structure and sidecar together;
- require an explicit `allow_lossy=True`; or
- raise a `LossyWriteError`.

It should not silently discard ownership, topology, charges, atom IDs, or cell information.

For example, writing an owned GB to basic XYZ without a sidecar may preserve species and Cartesian positions but lose the authoritative interface plane and persistent grain partition. That conversion is acceptable for visualization only when the caller explicitly accepts the loss.

---

## 7. Integration with existing classes

## 7.1 `Parent`

Add an explicit construction path:

```python
Parent.from_structure(
    structure: StructureData,
    *,
    unit_cell: UnitCell,
    gb_thickness: float,
    grain_ownership: GrainOwnership | None = None,
)
```

`Parent.from_structure()` should:

- validate required arrays and dimensions;
- build the whole-system view;
- select persistent grains from ownership labels when supplied;
- determine GB-region membership geometrically around the authoritative plane;
- preserve the distinction between grain ownership and GB-region membership;
- reject unsupported cell geometry or topology explicitly.

It should not parse files.

The existing `Parent(filename, ...)` constructor should remain temporarily as a compatibility shim that selects a reader and delegates to `Parent.from_structure()`.

## 7.2 `GBManipulator`

The current internal `_from_parents()` constructor is already a useful seam. The public integration can evolve toward:

```python
GBManipulator.from_parent(parent)
```

Internal optimizer paths should prefer already validated `Parent` objects. Repeatedly constructing `GBManipulator` from filenames encourages file parsing and physical reconstruction to remain coupled.

## 7.3 `GBMaker`

Retain `GBMaker.write_lammps()` as a public compatibility method, but make it a facade:

```python
def write_lammps(...):
    structure = self.to_structure_data(...)
    options = LammpsDataWriteOptions(...)
    return LammpsDataWriter().write(file_name, structure, options=options)
```

The extraction should preserve existing output semantics, including atom ordering, type labels, charges, precision, and supported triclinic conversion, until deliberate API changes are separately approved.

## 7.4 `GBMinimizer`

Centralize evaluator-return handling in one loader service:

```python
CandidateLoader.load(
    returned_structure,
    *,
    candidate_mapping,
    unit_cell,
    gb_thickness,
    type_dict=None,
) -> GBManipulator
```

The current `reload_explicit_manipulator()` function already implements the core validation required for explicit ownership. It should become the basis for this service rather than being duplicated across Monte Carlo, genetic algorithm, restart, reseeding, breeding, carryover, or batch-evaluation paths.

The central loader should validate:

- atom count;
- unique and expected transient IDs;
- species per ID;
- finite coordinates;
- cell or box compatibility;
- boundary topology;
- frame-selection rules;
- ownership-array alignment;
- unsupported variable-cell changes.

All ownership-aware reload paths should use this service. Direct filename reconstruction should remain only for legacy, ownership-free workflows during the transition.

---

## 8. Why this approach was selected

The selected architecture most directly satisfies both the paper's modular-extension objective and the invariants established by the interface fixes.

### Advantages

| Advantage | Consequence for GBOpt |
|---|---|
| Format syntax is isolated | Adding CIF, XYZ, VASP, GULP, or Quantum ESPRESSO does not require modifying `Parent` or minimizers. |
| Grain ownership remains explicit | Interface-plane and grain-partition fixes survive serialization and relaxation. |
| Readers and writers are independent | Read-only and write-only adapters are natural rather than exceptional. |
| The canonical record reuses the existing atom dtype | The I/O project does not expand into an unrelated atom-model rewrite. |
| Cell geometry is represented generally | Adapters can preserve source geometry even when a downstream algorithm supports only a subset. |
| Capabilities and losses are explicit | Formats cannot silently discard critical state. |
| Reload validation is centralized | Serial and batch evaluators use the same safety checks. |
| Transient IDs have a formal lifetime | Candidate-local mappings cannot accidentally become persistent identity. |
| Existing public APIs can delegate | Migration can be incremental and backward-compatible. |
| Third-party implementations have a small contract | Contributors can implement one reader or writer without subclassing `Parent`. |

### Disadvantages and mitigations

| Disadvantage | Mitigation |
|---|---|
| More classes and indirection | Keep the initial public interfaces small and focused. |
| Canonical models can accumulate unrelated fields | Add only format-neutral data required by implemented or near-term adapters. |
| Sidecars create paired-file management | Use deterministic names, schema versions, digests, and transactional publication. |
| Different adapters may implement inconsistent behavior | Provide reusable contract tests and capability declarations. |
| Backward-compatible shims temporarily duplicate entry points | Define a documented migration and deprecation sequence. |
| Full-cell representation exceeds some current algorithmic support | Preserve geometry at the I/O boundary and reject unsupported use downstream. |
| Strict loss handling may require more explicit user choices | Prefer explicit failures over silent corruption of GB-domain state. |

### Selection rationale

This design was selected because it creates a stable boundary between three distinct concerns:

1. **Representation conversion** — handled by format readers and writers.
2. **Generic atomic structure data** — held by `StructureData`.
3. **Grain-boundary identity and topology** — held by `GrainOwnership` and interpreted by `Parent`.

That division matches the direction of the current repairs and avoids reintroducing the original failure mode, where grain ownership or interface location could be reconstructed from coordinates after external relaxation.

---

## 9. Alternatives considered and rejected

## 9.1 One monolithic `GBIOBase` class

A single abstract base class could require both `read()` and `write()` for every format.

### Advantages

- fewer public classes;
- a simple one-format/one-adapter concept;
- straightforward registration.

### Why it was rejected

Read and write capabilities are not symmetric:

- LAMMPS dump files are naturally multi-frame inputs but are not necessarily the preferred output format;
- some calculator interfaces may be output-only from GBOpt's perspective;
- some formats can preserve IDs and charges while others cannot;
- restart-oriented and visualization-oriented formats have different requirements.

A combined class would require meaningless methods or make `NotImplementedError` a routine part of normal use. Independent reader and writer ABCs communicate capabilities more accurately.

## 9.2 Extend the existing `Parent` format dispatch

This approach would add more private methods such as:

```python
__init_from_xyz(...)
__init_from_cif(...)
__init_from_vasp(...)
```

### Advantages

- lowest immediate implementation cost;
- minimal changes to existing callers;
- no new canonical data type.

### Why it was rejected

`Parent` already combines:

- format detection;
- syntax parsing;
- type conversion;
- box interpretation;
- grain classification;
- GB-plane and GB-region construction.

Adding more formats there would enlarge an already broad class, duplicate parser validation, and require every format contributor to understand GB internals. More importantly, it would preserve the coupling that allowed file reload to reconstruct physical ownership from coordinates.

The compatibility constructor can remain, but only as a delegating facade.

## 9.3 Use ASE `Atoms` and ASE I/O as the core abstraction

### Advantages

- immediate access to many established atomistic formats;
- existing support for cells, periodicity, arrays, and trajectories;
- reduced parser maintenance for common formats.

### Why it was rejected as the core public contract

ASE does not define GBOpt's persistent grain ownership, authoritative interface plane, candidate-local ID lifetime, or evaluator-return validation. Making `ASE Atoms` mandatory would also introduce a broad dependency and require GBOpt invariants to be encoded in external metadata conventions.

ASE remains a strong optional implementation detail:

```text
CIF or XYZ file
    -> ASE reader
    -> GBOpt StructureData
```

This allows GBOpt to benefit from ASE's format coverage without making ASE's object model the package's public domain contract.

---

## 10. Incremental implementation plan

## Phase 0 — Freeze contracts and invariants

Create:

- `GBOpt/io/model.py`;
- `GBOpt/io/base.py`;
- `GBOpt/io/errors.py`;
- initial reader/writer contract tests.

Document the following invariants before moving code:

- persistent grain labels are independent of current coordinates;
- `gb_plane_x` is authoritative in explicit-ownership mode;
- file IDs are serialization-local;
- GB-region membership is geometric and distinct from grain ownership;
- unsupported loss or topology changes fail explicitly.

No existing public behavior should change in this phase.

## Phase 1 — Extract LAMMPS data reading

- Move or wrap `read_lammps_data_file()` as `LammpsDataReader`.
- Return `StructureData` while retaining access to the parsed IDs.
- Preserve strict validation of counts, types, IDs, charges, coordinates, and orthogonal bounds.
- Add adapter-level tests independent of `Parent`.
- Keep the existing function as a compatibility facade during migration.

## Phase 2 — Extract LAMMPS data writing

- Move the serialization logic from `GBMaker.write_lammps()` into `LammpsDataWriter`.
- Return a `WriteResult` containing the emitted transient ID mapping.
- Preserve existing formatting and output behavior unless a change is separately approved.
- Make `GBMaker.write_lammps()` delegate to the writer.

## Phase 3 — Extract LAMMPS dump reading

Define explicitly:

- frame-selection behavior;
- wrapped, unwrapped, and scaled coordinate handling;
- required atom columns;
- type-label resolution;
- orthogonal and triclinic support;
- boundary-periodicity interpretation.

The current strict reader deliberately selects the first frame. That behavior should remain stable during extraction, then be generalized through explicit options rather than changed accidentally.

## Phase 4 — Refactor `Parent`

- Add `Parent.from_structure()`.
- Move generic structure validation into that constructor.
- Keep GB-domain interpretation in `Parent`.
- Route legacy filename construction through the registry.
- Remove format-specific parser implementation from `Parent` once compatibility tests pass.

## Phase 5 — Centralize candidate reloads

- Promote `reload_explicit_manipulator()` into a general candidate-loader service.
- Make every ownership-aware optimizer reload use it.
- Construct `CandidateFileMapping` from `WriteResult` plus persistent ownership.
- Remove direct ownership-aware reconstruction from evaluator-return filenames.
- Cover serial and batch evaluators with the same contract tests.

## Phase 6 — Formalize metadata persistence

- Define a versioned sidecar schema.
- Bind metadata to exact structure content.
- Publish structure and metadata transactionally.
- Validate both before constructing a `Parent`.
- Make loss policy explicit in writer options.

## Phase 7 — Add common interchange formats

Add XYZ first because its limited metadata capacity exercises the loss policy clearly.

Add CIF after cell and periodicity semantics are stable.

Additional adapters for VASP, GULP, and Quantum ESPRESSO should initially focus on structure representation. Calculator setup, execution, and result extraction should remain separate interfaces.

## Phase 8 — Third-party plugin support

After the built-in interfaces have stabilized:

- expose reader/writer registration through package entry points;
- publish adapter contract tests;
- document capability declarations;
- define metadata-schema compatibility expectations;
- document which exceptions are part of the public API.

---

## 11. Testing strategy

### 11.1 Reader contract tests

Every reader should be tested for:

- valid minimal input;
- malformed headers and sections;
- missing required fields;
- duplicate or invalid IDs;
- non-finite coordinates;
- type-map errors;
- cell and boundary-condition parsing;
- defensive copying/read-only behavior;
- deterministic frame selection;
- consistent error types.

### 11.2 Writer contract tests

Every writer should be tested for:

- deterministic output;
- row-order preservation;
- emitted ID mapping;
- precision handling;
- charge handling;
- type-label handling;
- loss detection;
- unsupported-cell rejection;
- round-trip consistency within the format's declared capabilities.

### 11.3 Ownership and reload tests

The existing interface-fix behavior must remain covered:

- labels remain attached to rows even when atoms cross `gb_plane_x` during relaxation;
- evaluator output cannot change species for a known transient ID;
- atom additions or removals are rejected unless a workflow explicitly permits them;
- box or topology changes are rejected when unsupported;
- reordered file rows are realigned correctly by candidate-local IDs;
- stale mappings cannot be reused for another candidate;
- explicit-mode failure never falls back to geometric grain inference;
- GB-region membership remains geometric and separate from persistent ownership.

### 11.4 Compatibility tests

Compatibility tests should verify that:

- `GBMaker.write_lammps()` still produces accepted output;
- legacy `Parent(filename, ...)` calls still work through delegation;
- current LAMMPS data and dump inputs load equivalently before and after extraction;
- current Monte Carlo and GA workflows do not change behavior solely because of the refactor.

---

## 12. Definition of completion

The initial I/O abstraction is complete when:

- `Parent` contains no format-specific parsing implementation;
- `GBMaker.write_lammps()` delegates to a standalone writer;
- LAMMPS data and dump readers return the same canonical structure type;
- persistent grain labels are never inferred from current coordinates in explicit mode;
- `gb_plane_x` is not recomputed during ownership-aware evaluator reload;
- every ownership-aware minimizer reload uses one central loader;
- transient atom IDs have candidate-local lifetimes;
- unsupported or lossy operations fail explicitly;
- legacy public entry points remain functional through adapters;
- reader and writer implementations share reusable contract tests;
- the existing interface-fix regression suite continues to pass.

---

## 13. Final architectural statement

The recommended boundary is:

> **I/O adapters own representation conversion; `StructureData` owns generic structural data; `GrainOwnership` owns persistent interface identity and topology; `Parent` composes and interprets those objects; minimizers consume only validated domain objects.**

This design formalizes the extension point anticipated by the paper while preserving the interface-plane and grain-ownership corrections already implemented in the current source. It also creates a clean path from the existing LAMMPS-only workflow to calculator-independent structure interchange without turning `GBMaker`, `Parent`, or `GBMinimizer` into format dispatch layers.

---

## Reference

French, J. C., and Bhave, C. V. (2026). “GBOpt: Grain boundary structure optimization using Monte Carlo and evolutionary algorithms.” *SoftwareX*, 35, 102763. https://doi.org/10.1016/j.softx.2026.102763
