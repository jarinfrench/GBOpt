# Extending GBOpt

This document is for people adding new construction stages, structure formats,
manipulation operations, evaluators, event sinks, or checkpoint backends -- as
opposed to `README.md`, which covers day-to-day usage of the existing built-ins.
It reflects the architecture as of the R01-R30 refactor (issue #61): six
subpackages, each with a `types.py` leaf module holding its own exception
hierarchy and value types, and a curated `__init__.py` re-exporting a small
public surface. It also lists the compatibility shims the refactor retained,
so a caller migrating older code knows what is still supported and why.

## Construction stages (`GBOpt.gbmaker`)

`GBOpt.gbmaker` is a pure construction pipeline: it imports neither the
optimizer subpackage (`GBOpt.optimization`) nor any file-format implementation
(`GBOpt.io`), so a new construction stage never needs to know how its output
will be evaluated or serialized. `GBOpt.GBMaker.GBMaker` is a compatibility
facade over this subpackage -- it wires the pipeline's stages together and
adds `GBOpt.io.lammps` writing -- not the place to add new construction logic.

The pipeline's stages, leaf-first:

- `GBOpt.gbmaker.types` -- exception hierarchy (`GBMakerConstructionValueError`
  etc.) and value types (`GrainBuildRequest`, `GrainBuildResult`,
  `BicrystalResult`, `AxisAccommodation`, `MaterialState`, `GBBuildConfig`).
  Every other module here may import from `types.py`; it imports from nothing
  else in the subpackage.
- `GBOpt.gbmaker.material` / `orientation` / `dimension` / `geometry` -- pure
  per-concern kernels (crystal-structure/lattice-parameter validation,
  misorientation resolution, dimension/spacing planning, coordinate/rotation
  geometry).
- `GBOpt.gbmaker.exact_grain` / `approximate_grain` -- the two grain-builder
  strategies (`mode="exact"`, integer P/Q construction; `mode="approximate"`,
  floating-point construction), each consuming the kernels above.
- `GBOpt.gbmaker.assembly` -- composes a chosen grain builder with the
  planning/geometry stages into one complete bicrystal (`assemble_bicrystal`,
  `build_bicrystal`).

To add a new construction stage (e.g. a new grain-builder strategy), write it
as a pure function taking and returning this subpackage's own value types, add
it to the dependency graph above at the appropriate level (never importing a
sibling at the same level, and never importing `GBOpt.io`/`GBOpt.optimization`),
and add its own `tests/test_gbmaker_<module>.py`. See `CLAUDE.md`'s
"Subpackage architecture conventions" and "Refactor-issue discipline" sections
for the conventions this subpackage's own decomposition (R03-R10) established
in detail -- in particular, a legacy `positive=True`-shaped validator usually
means *non-negative*, not strictly positive; don't assume from the name.

## Readers/writers (`GBOpt.io`)

`GBOpt.io` does not import the `GBMaker`/`GBManipulator` facades or any
minimizer implementation -- it is a neutral structure-format layer both the
construction and manipulation sides depend on, not the reverse.

- `GBOpt.io.types.StructureData` is the format-neutral structure
  representation every reader produces and every writer consumes (atoms, a
  general 3x3 cell -- triclinic-capable since R11 -- and box origin).
- `GBOpt.io.types.StructureReader` / `StructureWriter` are the `Protocol`
  interfaces a new format implements: `read(path, **kwargs) -> StructureData`
  and `write(path, structure, **kwargs) -> None`. A writer additionally
  returns `WriteResult` describing what it wrote, including
  `atom_ids` -- the candidate-local row-to-external-ID mapping. These IDs are
  **local to one write** and never carry persistent identity across writes,
  reads, or reloads (see `GBOpt.CandidateLoader`, below).
- `GBOpt.io.lammps` is the only concrete implementation today
  (`LammpsDataReader`/`LammpsDumpReader`/`LammpsDataWriter`). A new format adds
  a sibling package under `GBOpt.io` implementing the same two protocols;
  nothing else in `GBOpt.io` needs to change.

## Manipulations (`GBOpt.manipulation`)

`GBOpt.manipulation` performs no file I/O and no evaluation -- it imports
neither `GBOpt.io` nor `GBOpt.evaluation`, and depends only on this
subpackage's own neutral domain state (`InterfaceCandidate`, from
`GBOpt.GBManipulator`, plus `GBOpt.BoundaryTopology`).

- `GBOpt.manipulation.types.Manipulation` is the `Protocol` a new operation
  implements: a `name` property (registry lookup / lineage key), an `arity`
  property (how many parent candidates it needs), and
  `execute(context: ManipulationContext) -> ManipulationResult`. An
  operation's `execute` trusts its caller (`GBManipulator.apply()`) to have
  already checked arity against `len(context.parents)`, unless (like
  `SliceAndMerge`) its own legacy delegate never routes through `apply()` --
  see `CLAUDE.md`'s R19 entries for when self-checking arity is warranted.
- `GBOpt.manipulation.registry.ManipulationRegistry` is an explicit
  name-to-operation lookup table (`register(name, manipulation)`,
  `resolve(name)`). `GBOpt.manipulation.default_registry` is the registry
  `GBManipulator.apply_named(...)` and the legacy `choices` adapter use when
  no explicit registry is given.
- `GBOpt.manipulation.builtins` registers this package's own built-in
  operations (translation, termination cycling, interface separation, atom
  insertion/removal, soft-mode displacement, slice-and-merge crossover) into
  `default_registry`.

To add a new operation: implement the `Manipulation` protocol against
`ManipulationContext`/`ManipulationResult` (both carry only
`InterfaceCandidate`s, RNG state, and JSON-safe parameters -- never a file
path, evaluator callback, or live manipulator/minimizer reference), register
it (`registry.register("my_op", MyOp())`), and it is reachable via
`manipulator.apply(MyOp(), **params)` or, once registered in
`default_registry`, via the legacy `choices=["my_op"]` list. A third-party
operation used through `choices` requires the manipulator's parent(s) to have
known boundary-normal topology; see `CLAUDE.md`'s R20 entry on this seam's one
documented limitation for Monte Carlo's legacy (non-owned) reload path.

## Evaluators (`GBOpt.evaluation`)

An evaluator is any callable matching the `gb_energy_func`/`gb_batch_energy_func`
contract both minimizers accept: called with `(GBMaker, GBManipulator, atom_positions,
unique_id)` (scalar) or the batch-shaped equivalent, returning `(objective, dump_path)`.
GBOpt does not run an external calculator itself -- the evaluator is the caller's own
integration point (LAMMPS today, in the initial release; any external tool that can be
driven this way tomorrow).

- `GBOpt.evaluation.types.EvaluationResult`/`StructureArtifact`/`EvaluationStatus`/
  `FailureStage` are the typed, classified representation both minimizers translate a raw
  evaluator return (or exception) into at the call boundary, via
  `GBOpt.evaluation.adapters.from_scalar_tuple`/`from_batch_dict`. Both minimizers wrap
  their own evaluator call in a recovery boundary (`except Exception` -> a penalized,
  classified `EvaluationResult`, not a crashed run) -- see `CLAUDE.md`'s R23 entries for
  the exact, deliberately-narrow scope of this behavior.
- Explicit-ownership (owned-mode) GA additionally routes every evaluator-returned
  structure through `GBOpt.CandidateLoader.CandidateLoader` -- the single, authoritative
  reload path for both evaluator-returned structures and checkpoint-restored candidates
  (`ExplicitOwnershipEvaluator._reload_mapping` and owned-mode `run_GA`'s resume path
  both call it). A new evaluator does not need to know about `CandidateLoader` directly;
  it only needs to write a structure file at the path it returns and let GBOpt's own
  reload machinery handle reconstruction.

## Event sinks (`GBOpt.observability`)

`GBOpt.observability` imports neither `GBOpt.snapshot` nor `GBOpt.Checkpoint` --
events/journals are observers of already-decided optimizer state, never a source of
restart-critical state, and a journal file cannot be loaded as a checkpoint (no
`schema_version`/`minimizer`/`progress_unit` envelope).

- `GBOpt.observability.sinks.EventSink` is the `Protocol` a new sink implements: one
  method, `emit(event: OptimizationEvent) -> None`. A sink whose `emit()` raises is
  logged and otherwise ignored by both minimizers -- event emission never aborts or
  alters a run.
- Built-in sinks: `NullEventSink` (the default; discards everything, so a run is silent
  unless a caller opts in), `LoggingEventSink` (routes events through a
  `logging.Logger`), `JsonlEventSink` (append-only durable JSONL journal, one line per
  event; a context manager -- flushes after every `emit()`), and `CompositeEventSink`
  (fans one event out to several sinks, e.g. logging and a journal simultaneously; see
  `tests/test_integration_end_to_end.py`'s
  `test_simultaneous_logging_journal_checkpoint_does_not_change_numerical_history` for a
  regression test that wiring several sinks in together never perturbs the accept/reject
  sequence a fixed seed produces).
- `GBOpt.observability.journal.write_run_manifest`/`read_journal_events` support
  provenance inspection of an existing journal; they are not part of any restart path.

Both minimizers accept an `event_sink: EventSink | None` constructor argument and stamp
every emitted `OptimizationEvent` with a `RunContext` (run/case/campaign identity, seed,
algorithm). A new sink is passed the same way; nothing else about a minimizer's
construction changes.

## Checkpoint stores (`GBOpt.Checkpoint`, `GBOpt.snapshot`)

Checkpoint persistence/model code does not import live optimizer classes --
`GBOpt.Checkpoint` and `GBOpt.snapshot` both import nothing from `GBOpt.optimization` at
module scope (the two exceptions, `GBOpt.snapshot.types`/`migration`'s deferred, call-time
imports of a few pure mapping-serialization helpers from `GBOpt.optimization.types`, exist
specifically to break an import cycle -- see `CLAUDE.md`'s R29 entry). State-transition
policy (when to checkpoint, what counts as a durable boundary) is owned by
`MonteCarloMinimizer`/`GeneticAlgorithmMinimizer` themselves, not by the checkpoint layer.

- `GBOpt.Checkpoint.CheckpointStore` is the schema-v1 (raw dictionary envelope)
  persistence layer every minimizer still reads and writes, in JSON or pickle format. It
  implements the null-object pattern (`CheckpointStore.from_optional(None)` returns a
  no-op instance), so a minimizer's own loop never needs `if checkpoint is not None:`
  guards. `GBOpt.Checkpoint.validate_checkpoint_envelope` centralizes envelope validation
  (`schema_version`/`minimizer`/`progress_unit`/`progress_index`) across all three restore
  paths (MC, legacy GA, owned GA).
- `GBOpt.snapshot` is the schema-v2 typed codec layered on top: immutable, validated
  value types (`MonteCarloSnapshot`, `GeneticAlgorithmSnapshot`, and their nested
  `CandidateEvaluationSnapshot`/`RngStateSnapshot`/`RunIdentitySnapshot`/etc.) with their
  own `to_state()`/`from_state()` JSON-safe serialization, plus `GBOpt.snapshot.migration`
  for reading an older schema-v1 checkpoint into the same typed shapes. Every snapshot
  type rejects any value that is not one of its own validated types, a small scalar, or a
  JSON-safe mapping/sequence at construction -- a callback, event sink, logger, open file,
  or live manipulator/minimizer object can never reach a constructed snapshot.
- `GBOpt.artifacts.store.ArtifactStore` (a separate, already-reviewed contract) owns
  artifact-retention state; both snapshot schemas carry its serialized form through
  opaquely (`retention_state`) rather than re-typing it.

A new checkpoint backend (e.g. a different storage medium) implements the same read/write
contract `CheckpointStore` does; it does not need to know about `GBOpt.snapshot` unless it
wants schema-v2 typed round-tripping specifically.

## Compatibility shims retained by the refactor

Per issue #90's own non-goal ("any compatibility removal requires separate approval"),
the following pre-refactor surfaces are still fully supported, unchanged:

- **`GBOpt.GBMaker.GBMaker`** -- a facade over the `GBOpt.gbmaker` construction
  pipeline. Both the deprecated direct `GBMaker(...)` constructor and the newer
  `GBMaker.from_boundary_spec(...)` remain supported.
- **`GBOpt.GBManipulator.GBManipulator`** -- both the deprecated coordinate-based
  (no `grain_ownership`) construction path and the explicit-ownership path remain
  supported; per-operation legacy methods (`translate_right_grain`,
  `cycle_grain_terminations`, `apply_interface_separation`, `insert_atoms`,
  `remove_atoms`, `displace_along_soft_modes`, `slice_and_merge`) remain callable
  directly, alongside the newer `apply()`/`apply_named()` generic seam.
- **The `choices: list[str]` compatibility adapter** -- the three legacy unary
  operation names (`translate_right_grain`, `cycle_grain_terminations`,
  `apply_interface_separation`) and `slice_and_merge` still dispatch to the
  same established `GBManipulator` methods directly, bypassing the generic
  `ManipulationContext` boundary where that boundary's stricter invariants
  would reject state the legacy methods have always tolerated (see
  `CLAUDE.md`'s R20 entry on `translate_right_grain` specifically).
- **`GBOpt.Checkpoint`'s schema-v1 envelope** -- both minimizers still read and
  write it by default; the schema-v2 typed codec (`GBOpt.snapshot`) is an
  additive migration path, not a replacement (a v1 checkpoint is read via
  `GBOpt.snapshot.migration`, not rejected).
- **Individual per-parameter GA constructor arguments** (`slice_and_merge_pct`,
  `crossover_surface`, `crossover_max_tilt_degrees`, etc.), rather than a single
  declarative operation-pool configuration.
- **Top-level re-exports** -- `GBOpt.GBMaker`, `GBOpt.GBManipulator`,
  `GBOpt.InterfaceCandidate`, `GBOpt.Atom`, `GBOpt.BoundaryNormalTopology`,
  `GBOpt.Position`, `GBOpt.UnitCell` all still resolve to the same objects as
  importing their owning submodules directly (see
  `tests/test_integration_import_boundaries.py`'s compatibility-identity tests).

None of these are expected to be removed before a deliberate, separately-approved
breaking pass (nominally "R31," not yet filed as an issue -- see `REFACTOR_CLEANUP.md`'s
own entry on this).
