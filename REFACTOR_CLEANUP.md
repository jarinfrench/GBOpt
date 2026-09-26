# Refactor cleanup backlog

## R28 net tooling deltas: ruff net 0, mypy net 0, bandit unchanged, pyscn unchanged (41 quality issues, 51 clone pairs)

Baseline taken at the R28 branch point (`4d429d5`, tip of `refactor/r27-checkpoint-
snapshots`, checked out in a separate worktree so mypy/pyscn resolve the real package
layout rather than a synthetic copy -- see the `CLAUDE.md` git-mechanics-adjacent entry
below): ruff (`GBOpt/snapshot/types.py`, `migration.py`, `GBOpt/optimization/
monte_carlo.py`) 4 errors; mypy (`GBOpt/snapshot`, `GBOpt/optimization` scopes, filtered
to these three files plus `GBOpt/artifacts/types.py`, which mypy pulls in
transitively) 15 lines; bandit 0; pyscn (whole-repo) 41 quality issues / 51 clone pairs
(this machine's real current pyscn output for the R27 tip -- not R27's own
`REFACTOR_CLEANUP.md`-recorded 41/47, which likely reflects a different pyscn version;
the "real baseline right now" is what this file's own tooling-comparison rule asks for,
not the historical recorded number). Current (after every R28 commit): ruff 4 errors
(same three findings -- one `BLE001`, two `RUF022` -- just at different line numbers,
since new code was added above them); mypy 15 lines (same 12 errors + 3 `note:` lines,
same message text, shifted line numbers only); bandit 0; pyscn exactly 41/51, unchanged.

**Resolve at**: no action needed.

## R27 net tooling deltas: ruff +3 (disclosed established-convention debt), mypy net 0, bandit unchanged, pyscn back to baseline after an in-step decomposition (41 quality issues, 47 clone pairs)

Baseline taken at the R27 branch point (`e7354db`, tip of `refactor/r26-checkpoint-
hardening`): ruff (`GBOpt`/`tests`) 241 errors; mypy `GBOpt/optimization` 288 errors in
31 files, `GBOpt/observability` 202 errors in 23 files; bandit 7 Low + 1 Medium; pyscn 41
quality issues / 47 clone pairs. Current (after all four R27 commits): ruff 244 (+3),
mypy unchanged in both existing scopes (`GBOpt/snapshot` itself: 0 findings), bandit
unchanged, pyscn back to exactly 41/47 -- it had temporarily grown to 43 quality issues
and 51 clone pairs after the first three commits, brought back down by an in-step
decomposition pass rather than left as new debt (see the dedicated entry below).

ruff +3 is `GBOpt/snapshot/__init__.py`/`migration.py`/`types.py`'s own `__all__` lists
tripping `RUF022` (not alphabetized), the same established, deliberate
grouped-not-alphabetized convention already reproduced by every other subpackage's
`__init__.py` in this codebase (see R20's/R22's/R24's/R25's own entries for the
precedent) -- not new-shape debt. Every other finding a bare tool run initially
surfaced in the new files (an unused import, an unsorted import block, two
`dict()`-as-literal findings, one nested-`with`, several `object`-typed indexing
findings, a `NoReturn`-shaped narrowing gap, a wrong `Mapping`-typed default, and a
`MappingProxyType` argument-type mismatch) was a genuine new-code defect, not a
reproduced pattern, and was fixed on the spot rather than disclosed as debt.

**Resolve at**: no action needed.

## R28 extended `MonteCarloSnapshot` with `min_steps`/`cooldown_rate` fields R27 omitted

Issue #88's acceptance criteria list "minimum steps, cooldown" among what a v2 restore
must recover, but R27's `MonteCarloSnapshot` (designed before any real MC write path
existed) carried neither field -- schema-v1's `run_params.min_steps`/`cooldown_rate`
had no v2 counterpart at all. Per `CLAUDE.md`'s own "a value type from an earlier step
isn't frozen when correcting it is driven by an actual behavior-preservation
requirement" precedent (the R04 `positive=True` case), both fields were added directly
to `MonteCarloSnapshot` (`GBOpt/snapshot/types.py`) with `to_state`/`from_state`
support, rather than inventing a side channel outside the typed schema.
`GeneticAlgorithmSnapshot` is untouched -- it has no `min_steps`/`cooldown_rate`
equivalent and issue #88 is MC-only.

`cooldown_rate`'s validator (`_normalize_cooldown_rate`) enforces the same `0 < x <= 1`
range `MonteCarloMinimizer.run_MC` itself already checks before ever reaching a
checkpoint write, so it can never reject a value that could have arrived through a real
run. `min_steps`'s validator (`_normalize_optional_min_steps`) deliberately does **not**
enforce non-negativity, even though every other integer field in this module does --
`run_MC` has no upstream validation on `min_steps` at all, so a stricter check here
would add a new save-time rejection path for a value the existing run loop has always
silently tolerated. This mirrors the `positive=True`-vs-`strictly_positive=True`
distinction this file already documents at length for `GBMaker`'s legacy validators, just
showing up as "validate at all" vs. "don't" instead of "loosely" vs. "strictly."

**Resolve at**: no action needed.

## R28's MC schema-v1 migration path is narrower than MC's own pre-migration seed fallback

`MonteCarloMinimizer.run_MC`'s resume branch used to fall back to `self.seed` when a
checkpoint's `run_params` had no `"seed"` key at all (`state["run_params"].get("seed",
self.seed)`) -- a leniency for a genuinely legacy checkpoint written before seed
persistence existed. R27's `_monte_carlo_snapshot_from_v1` (shared with GA, and
explicitly documented for GA as "a legacy checkpoint written before seed persistence
cannot be migrated") builds `RunIdentitySnapshot(seed=run_params["seed"])` with a bare
key lookup, so a schema-v1 checkpoint missing `seed` now fails migration outright
(`SnapshotMigrationError`/`GBMinimizerError`) instead of silently falling back. R28
reuses this already-established migrator as-is rather than special-casing MC's own
narrower fallback into it -- the R27 migrator was already reviewed and accepted for GA
with exactly this narrowing, and #88 says "resume v1 through the R27 migrator," not
"extend the R27 migrator's leniency." A genuinely pre-seed-persistence MC checkpoint (if
one still exists anywhere) can no longer resume; every checkpoint `run_MC` has written
since seed persistence was added always has the field, so this is a narrow, disclosed
gap rather than a live regression for any checkpoint this codebase itself has produced
recently.

**Resolve at**: no action needed unless a real pre-seed-persistence MC checkpoint needs
to keep resuming.

## R28's `MonteCarloSnapshot.retention_state` needs a tuple->list normalization before reaching `ArtifactStore.from_state`

`MonteCarloSnapshot`'s own JSON-safety validation (`_reject_live_objects`, R27) recursively
normalizes every nested sequence -- including `retention_state["records"]` -- into a
`tuple`, matching this module's own immutable-value-type discipline. `GBOpt.artifacts.
store.ArtifactStore.from_state` (a pre-existing, already-reviewed contract this schema
deliberately passes `retention_state` through opaquely rather than re-typing, per R27's
own entry below) enforces that `records` is literally a `list`, not any `Sequence`, and
raises `ArtifactStoreError` on a tuple. This only surfaces on MC's *resume* path (restoring
a live `ArtifactStore` from a snapshot's `retention_state`), not on save, since
`ArtifactStore.to_state()`'s own output is fed straight into the snapshot constructor
without a resume-side round trip in between. Fixed with a small `_tuples_to_lists`
helper in `GBOpt/optimization/monte_carlo.py`, applied only at the one call site that
hands `retention_state` to `ArtifactStore.from_state`; the snapshot's own tuple-based
representation is otherwise unchanged. Any future step that resumes `ArtifactStore`
state from a `GeneticAlgorithmSnapshot`'s `retention_state` will need the same fix.

**Resolve at**: no action needed unless a later step wires `GeneticAlgorithmSnapshot`
resume through `ArtifactStore.from_state` and hits the same tuple/list mismatch.

## R27's schema-v1 lineage entries are not uniformly `[operation_label, parent]` pairs

`LineageStepSnapshot` (`GBOpt/snapshot/types.py`) was first designed with a single
`parent_reference: str` field, modeled on the two-element shape every
`population_lineages.append([mutation, parent])` call site in `genetic.py` that was read
before starting this step seemed to establish. Writing the real-checkpoint round-trip
test against an owned-mode GA run that actually exercised crossover (not just mutation)
found a third shape neither read pass had surfaced: `_make_next_owned_generation`'s
slice-and-merge branch (`GBOpt/optimization/genetic.py` around line 1101) appends a
**four**-element list (`[label, parent1_path, parent2_path, repr(provenance)]`), and its
own inadmissible-crossover fallback branch appends a **three**-element list (`[label,
parent_path, "N inadmissible crossover attempts"]`) -- neither a single-parent pair nor
a JSON-safe-only tail. `LineageStepSnapshot` was revised (still before this step's first
commit landed, so the fix shipped in the same commit as the migrator, not as a
follow-up) to hold `parent_references: tuple[str, ...]` (one or two parents) plus an
explicit `diagnostic_note: str | None` field for the trailing free-text schema-v1 itself
already recorded alongside some entries -- disclosed on the type's own docstring as
non-authoritative diagnostic context, never a substitute for the structured
operation/parent identity. `GBOpt.snapshot.migration._lineage_step_from_v1` dispatches
on the entry's own length (2/3/4) to recover this shape, matching the same three
call-site shapes read directly out of `genetic.py`, rather than trusting either the
roadmap issue's prose or a first pass over only the mutation-only call sites. This is
the same "re-grep every call site, don't trust the description" discipline `CLAUDE.md`
already documents at the top of this file, applied to the roadmap issue's own field
description rather than to a `REFACTOR_CLEANUP.md` entry's.

**Resolve at**: no action needed; noted for the record.

## R28 confirmed none of R27's three deferred MC-adjacent gaps are actually in issue #88's scope

Rechecked each of R27's own entries against #88's real acceptance criteria before
starting, per this file's own top-of-file discipline:

- **`CandidateEvaluationSnapshot` construction from a live `EvaluationResult`.** #88's
  criteria never mention it, and `MonteCarloSnapshot` has no field of that type at all
  (only `MonteCarloStepRecordSnapshot`/`StructureArtifact`) -- MC's own restart schema
  never needed `from_evaluation_result()` to run for the first time. This gap (R27's
  entry above) remains open for whichever step actually wires GA's per-candidate
  bookkeeping onto `EvaluationResult`.
- **`CandidateCheckpoint`'s per-candidate `.iterN` sidecar file.** MC does not use
  `CandidateCheckpoint` at all (confirmed by grep -- only `GeneticAlgorithmMinimizer`
  constructs one), so this gap is entirely orthogonal to R28's MC-only scope and
  remains exactly as R27 left it.
- **`ArtifactStore`'s own checkpoint state as an opaque passthrough, not a re-typed
  schema-v2 value type.** #88 asks MC's restart-critical controls to be typed, not
  `ArtifactStore`'s own already-reviewed `to_state()`/`from_state()` contract to be
  duplicated as a second representation -- `MonteCarloSnapshot.retention_state` keeps
  R27's opaque-passthrough design unchanged (see the tuple/list normalization entry
  below for the one real wrinkle this surfaced).

**Resolve at**: no action needed at R28; unchanged from R27's own resolution notes.

## R27 partially resolves R22's/R23's/R26's deferred "`EvaluationResult` checkpoint-serialization form" gap -- the typed form now exists, but nothing writes it yet

R22's own entry (and R23's, R25's, R26's re-confirmations of it) left open the question
of what a checkpoint-serializable form of `EvaluationResult`/`CandidateEvaluation` should
look like, naming "R23 or R26+" and later "R27-R29" as the likely step. `Candidate
EvaluationSnapshot` (`GBOpt/snapshot/types.py`) is that form: a frozen, validated type
with the same field contract as `EvaluationResult` minus its live `manipulator`
reference, plus a `from_evaluation_result()` adapter that drops the manipulator rather
than ever reading it. This resolves the *shape* question. It does not resolve the
*wiring* question -- per this issue's own explicit "no optimizer loop is migrated"
non-goal, nothing in `MonteCarloMinimizer`/`GeneticAlgorithmMinimizer` constructs an
`EvaluationResult` and adapts it through `from_evaluation_result()` at any point; the
adapter is exercised today only by this step's own tests and, indirectly, by the schema-
v1 migrator (which builds `CandidateEvaluationSnapshot` instances directly from already-
serialized v1 dict state, never from a live `EvaluationResult`). One deliberate deviation
from `EvaluationResult`'s own stricter invariant, disclosed on the class docstring: a
successful `CandidateEvaluationSnapshot` does **not** require an `artifact`, unlike a
successful `EvaluationResult`, because this type also has to represent owned-mode GA's
artifact-independent `CandidateEvaluationSummary` (R22) -- a historical
"this candidate succeeded with this objective" record that never carried a structure
path to begin with. `GBOpt.optimization.genetic`'s own `CandidateEvaluation`/
`CandidateEvaluationSummary` types are themselves untouched by this step, per R23's own
"no action needed unless a later step needs `EvaluationResult` to be the type actually
flowing through GA's owned-mode bookkeeping" -- still true here, since the migrator reads
already-persisted dict state, not GA's live bookkeeping objects.

**Resolve at**: no action needed unless a later step wires `CandidateEvaluationSnapshot`
into a live MC/GA checkpoint *write* path (at which point `from_evaluation_result()`
would actually run against a real `EvaluationResult` for the first time) or wants GA's
owned-mode bookkeeping itself migrated onto `EvaluationResult`/this schema.

## R27's typed snapshots cover the run-level MC/GA checkpoint file only, not `CandidateCheckpoint`'s own per-candidate sidecar file

Issue #87's "Completed safe boundaries are explicit: ... candidate cache after a
completed candidate result is durable" was read as being satisfied by construction --
every `CandidateEvaluationSnapshot` instance is, by its own validated invariants,
necessarily a *completed* result (a definite `SUCCESS`/`FAILED` status, never a
pending/in-progress one), so the type itself cannot represent anything but a durable,
completed candidate outcome. This is distinct from, and does not migrate,
`GBOpt.Checkpoint.CandidateCheckpoint`'s own separate on-disk sidecar file (the
`{stem}.iter{N}{ext}` file recording each individual candidate's result as it completes,
*within* a still-in-progress generation, for intra-generation crash recovery) -- no
`CandidateCacheSnapshot`-shaped type or migrator exists for that file's own
`{iteration_index, unique_ids, results: {uid: {energy, dump, metadata}}}` shape. This
was a deliberate scoping choice, not an oversight: the migrator (`GBOpt.snapshot.
migration`) is a reader over an already-**completed**, at-rest run-level checkpoint --
per `GeneticAlgorithmMinimizer.run_GA`'s own resume logic, any stale `.iterN` sidecar for
the just-completed generation is deleted on resume, so a durable run-level checkpoint
never has a live sidecar to migrate at the boundary this step's migrator reads. A future
step wiring live checkpoint *writing* through this schema, where mid-generation
per-candidate durability genuinely matters for crash recovery, would need to decide
`CandidateCheckpoint`'s own typed shape at that point, informed by whichever write path
it is servicing -- not guessed here.

**Resolve at**: no action needed unless a later step's acceptance criteria require a
typed, migratable form of `CandidateCheckpoint`'s own per-candidate sidecar file
specifically.

## R27's `retention_state`/`retention_archive_mappings`/`claimed_paths` are carried opaquely or reused directly, not re-typed as new snapshot value types

Issue #87 names `MonteCarloSnapshot`/`GeneticAlgorithmSnapshot`/`PopulationCandidateSnapshot`/
`CandidateEvaluationSnapshot` plus "artifact/candidate references as needed" -- it does
not ask for `GBOpt.artifacts.store.ArtifactStore`'s own checkpoint state (the
`artifact_store` field in both MC's and GA's schema-v1 `state`) to be re-typed.
`ArtifactStore` already owns a reviewed `to_state()`/`from_state()` contract of its own
(pre-dating this step); `MonteCarloSnapshot.retention_state`/`GeneticAlgorithmSnapshot.
retention_state` carry that already-typed subsystem's serialized form through as an
opaque, JSON-safety-validated (never live-object-bearing) passthrough mapping rather than
duplicating its shape as a second, competing typed representation -- matching this
schema's own "validated artifacts plus persistent interface/ownership state **or another
explicitly reviewed typed representation**" contract, where `ArtifactStore`'s own
contract is that alternative. `retention_archive_mappings`, by contrast, *is* re-typed
(`Mapping[str, CandidateFileMapping]`, reusing `GBOpt.FileGrainOwnership.
CandidateFileMapping` directly) since it is exactly the persistent-ownership-state case
the schema's own contract calls out by name. `claimed_paths` (owned-mode's already-flat
`list[str]` of canonical evaluator paths) needed no typed wrapper at all -- a tuple of
validated non-empty path strings already is its own honest shape.

**Resolve at**: no action needed unless a later step wants `ArtifactStore`'s own
checkpoint state expressed as a schema-v2-native value type rather than an opaque,
already-typed-elsewhere passthrough.

## R26 net tooling deltas: ruff net 0, mypy net 0, bandit unchanged, pyscn unchanged (41 quality issues, 47 clone pairs)

Baseline taken at the R26 branch point (`a33cddd`, tip of `refactor/r25-event-journal`):
ruff (`GBOpt`/`tests`) 241 errors; mypy `GBOpt/optimization` 288 errors in 31 files (7
source files checked), `GBOpt/observability` 202 errors in 23 files (5 source files
checked); bandit 7 Low + 1 Medium + 0 High; pyscn 41 quality issues / 47 clone pairs.
Current (after all four R26 commits): identical on every metric -- ruff 241, mypy
unchanged in both scopes, bandit unchanged, pyscn unchanged at 41/47. Per-file ruff
counts for every file this step touched (`GBOpt/Checkpoint.py`,
`GBOpt/optimization/genetic.py`, `GBOpt/optimization/monte_carlo.py`,
`tests/test_checkpoint.py`, `tests/test_optimization_genetic.py`,
`tests/test_optimization_monte_carlo.py`) were diffed individually against their
pre-R26 copies, not just the repo-wide total, per `CLAUDE.md`'s own R23 lesson about a
per-file check being necessary even when the whole-repo total doesn't move -- every
file's own count matched exactly.

**Resolve at**: no action needed.

## R26 confirmed it is not the checkpoint-serialization step R22's/R23's `REFACTOR_CLEANUP.md` entries floated for "R23 or R26+"

Issue #86's own acceptance criteria are explicit: "No typed v2 snapshot model is
introduced yet." #86 is entirely about hardening the *existing* schema-v1 envelope
(uniform validation before MC/GA state access, atomic same-directory publish with
flush/fsync/`os.replace`) -- it adds no `to_state`/`from_state` serialization to
`EvaluationResult`/`CandidateEvaluation`, and doesn't touch the checkpoint *format* at
all. Confirmed by reading #86's acceptance criteria directly, the same discipline R25
used to rule itself out as this step. The gap remains open for whichever later step
actually is the checkpoint-serialization migration.

## None of R24's four new `REFACTOR_CLEANUP.md` entries, or R25's `OptimizationEvent` artifact-file-reference gap, needed any action at R26

Rechecked each against #86's real acceptance criteria before assuming so. #86 is
entirely scoped to checkpoint envelope validation and publish durability -- it says
nothing about `operation_parameters`, GA accept/reject semantics, legacy `run_GA`'s
`RUN_FAILED`-only try/except, `_evaluate_generation`'s reconstruction-failure accuracy,
or joining a journal entry back to its structure artifact. Each remains open for
whichever step's own criteria actually touch it.

## R26 found MC's and legacy GA's checkpoint-restore paths had *no* envelope validation at all before this step, not just uneven validation

Issue #86's motivation describes "uneven envelope validation" across MC, legacy GA,
and ownership-aware GA, which reads like all three had *some* validation that merely
differed in strictness. Reading the actual restore code before touching it found a
sharper asymmetry: `MonteCarloMinimizer.run_MC` and `GeneticAlgorithmMinimizer.run_GA`'s
legacy (unowned) path accessed `state["state"]["GBE_vals"]`-shaped fields directly, with
no check of `schema_version`/`minimizer`/`progress_unit` and no try/except translating a
resulting `KeyError`/`TypeError` into `GBMinimizerError` at all -- a corrupt or
wrong-minimizer checkpoint would surface a raw Python exception. Only the owned
(explicit-ownership) GA path validated the envelope inline before this step. The fix
(`GBOpt.Checkpoint.validate_checkpoint_envelope`, called from all three restore paths,
each wrapped in the same `except GBMinimizerError: raise` / `except
(CheckpointCompatibilityError, KeyError, TypeError, ValueError)` pattern the owned path
already used) treats "uniform validation" and "validation that currently doesn't exist
in two of the three places" as the same fix, since centralizing necessarily adds the
check where it was missing.

**Resolve at**: no action needed.

## R26's own `_save_checkpoint_payload` hardening initially left `mkdir` outside its own try/except, contradicting its docstring

While implementing the parent-directory-creation policy, the first draft called
`path.parent.mkdir(parents=True, exist_ok=True)` before the function's `try:` block that
translates every other write failure into `CheckpointError` -- so a failure to create
the parent directory (e.g. a path segment already occupied by a regular file) would
have leaked a raw `OSError`, contradicting the docstring's own `:raises CheckpointError:
If the parent directory cannot be created...`. Caught by writing the corresponding test
(`test_save_translates_parent_directory_creation_failure`) before assuming the
docstring was accurate, not by any tool. Fixed by moving the `mkdir` call inside the
try block, in a small follow-up commit rather than amending the commit that introduced
it. Any future edit to this function should keep every failure mode the docstring
promises to translate actually inside the translating `try`.

## R25 net tooling deltas: ruff +1 (disclosed established-convention debt), mypy net 0, bandit unchanged, pyscn unchanged (41 quality issues, 47 clone pairs)

Baseline taken at the R25 branch point (`96db29c`, tip of `refactor/r24-event-vocabulary`,
also the tip of `refactor/r25-event-journal` before any R25 commits): ruff (`GBOpt`/`tests`)
240 errors; mypy `GBOpt/optimization` 288 errors in 31 files, `GBOpt/observability` 202
errors in 23 files (4 own source files checked); bandit 7 Low + 1 Medium; pyscn 41 quality
issues / 47 clone pairs. Current (after all R25 commits): ruff 241 (+1), mypy unchanged in
both scopes (`GBOpt/observability` now checks 5 source files -- `journal.py` itself adds 0
new findings), bandit unchanged, pyscn unchanged at 41 quality issues / 47 clone pairs (the
new `journal.py`/`RunManifest` code trips neither the complexity/SLOC gates nor clone
detection).

ruff +1 is `journal.py`'s own `__all__` list tripping `RUF022` (not alphabetized), the same
established, deliberate convention already reproduced by `__init__.py`/`sinks.py`/`types.py`
in this exact subpackage (see R24's own entry for the precedent) -- not new-shape debt. A
`PYI034` finding on `JsonlEventSink.__enter__`'s return type (this is the first context
manager in the codebase) was fixed on the spot with `typing.Self` rather than disclosed as
debt, since it's a pure type-hint correction with no behavior change, not a new pattern
worth reproducing uncorrected.

**Resolve at**: no action needed.

## R25's `OptimizationEvent` still carries no artifact-file reference of its own

Issue #85's "no atom arrays/full structures are embedded; artifact references are used" is
satisfied by construction -- `OptimizationEvent` (R24) has never had an atom-array field to
begin with, and this step's own JSON serialization (`journal.py`'s `_event_to_json_dict`)
adds none. But it also means a journal reader cannot join a `PROPOSAL_EVALUATED`/
`CANDIDATE_ACCEPTED` line back to the structure file that produced it except indirectly, via
`candidate_id` (and only if some other, out-of-band record maps that ID to a path) --
`EvaluationResult.artifact` (a `StructureArtifact` reference: path/format/digest, from R22)
is never surfaced onto the event at all. This was checked against #85's actual acceptance
criteria before treating it as in-scope: the criterion constrains what a journal entry must
never embed (no atom arrays/full structures), it does not require adding a new authoritative
field to the event schema -- and `evaluation_event_fields()`'s R24-established field set is
out of this step's stated subject (durable storage of the existing schema, not a schema
change). Left as-is, not fabricated from an unrelated field.

**Resolve at**: no action needed unless a later step's acceptance criteria require joining a
journal entry back to its structure artifact without an external candidate-ID lookup, at
which point `OptimizationEvent` would need a new `artifact_path`-shaped field (bumping
`EVENT_SCHEMA_VERSION`) sourced from `EvaluationResult.artifact`, the same way
`evaluation_event_fields()` already sources every other authoritative field.

## R25's manifest overwrite guard and journal write-failure policy are this step's own design decisions, not spelled out by issue #85

Issue #85's "Proposed behavior" asks for append/overwrite rules, a flush policy, and a
write-failure policy to be "defined," without specifying what they should be. The choices
made, each documented on the relevant class/function docstring in
`GBOpt/observability/journal.py` rather than repeated here: `JsonlEventSink` defaults to
`mode="append"` (never silently truncates existing provenance) and flushes after every
`emit()` (no `fsync`, disclosed as a real durability limit, not a gap); a write/flush
`OSError` propagates to the caller unchanged rather than being caught internally, relying on
the recovery boundaries that already exist one layer up (`CompositeEventSink`,
`MonteCarloMinimizer._emit`/`GeneticAlgorithmMinimizer._emit`) to keep a journal failure from
ever reaching optimizer state; `write_run_manifest` defaults to refusing to replace an
existing manifest file (`overwrite=False`), since a manifest is one authoritative object per
run, not a line stream. `read_journal_events`'s truncated-final-line handling (skip with a
logged warning; any other malformed line raises) was added speculatively, matching #85's "if
a reader is provided" phrasing, since a reader is genuinely useful for provenance inspection
and the criterion anticipates exactly this failure mode.

**Resolve at**: no action needed; recorded so a later step doesn't mistake these documented
choices for gaps still needing a design decision.

## R24 net tooling deltas: ruff +6 (disclosed established-convention debt in the new package), mypy +15 (one disclosed root cause), bandit unchanged, pyscn unchanged (41 quality issues, +1 disclosed clone pair)

Baseline taken at the R24 branch point (`d2c95ba`, tip of `refactor/r23-mc-ga-
evaluation`): ruff 183 repo-wide; mypy `GBOpt/optimization` 273 errors in 31 files;
bandit 7 Low + 1 Medium; pyscn 41 quality issues / 46 clone pairs. Current (after both
R24 commits): ruff 189, mypy `GBOpt/optimization` 288 errors in 31 files (`GBOpt/
observability` itself: 0), bandit unchanged, pyscn unchanged at 41 quality issues / 47
clone pairs.

ruff +6 is entirely the new `GBOpt/observability/__init__.py`/`sinks.py`/`types.py`
reproducing this codebase's own established conventions at new call sites (3 `RUF022`
on grouped-not-alphabetized `__all__`, matching every other subpackage's `__init__.py`;
3 `UP042` on `(str, Enum)`, matching `EvaluationStatus`/`FailureStage`/
`BoundaryNormalTopology`/`ArtifactPin`) -- not new-shape debt. `monte_carlo.py`/
`genetic.py` are both net 0 (identical rule/message sets, only line numbers shifted).

mypy +15 (7 in `monte_carlo.py`, 8 in `genetic.py`) is one single root cause in both
files: `_emit()`'s `**fields: object` spread into `OptimizationEvent`'s individually-
typed keyword parameters. `**fields: object` is left as-is -- it is honestly typed (the
values really do vary in type by call site, and `_emit` is a private dispatch helper,
not a public contract mypy could usefully narrow) -- per this file's own "leave it typed
honestly and let mypy report the finding" mypy-debugging discipline, rather than
restructured into a `TypedDict`/`Unpack` scheme purely to appease the checker. `genetic.py`'s
+8 additionally includes one `[no-redef]` finding (a new `results` local declared in
both of `_evaluate_generation`'s existing two branches) reproducing that method's own
pre-existing "Name already defined" pattern verbatim (already present for
`gen_energies`/`gen_files`/`evaluated_manipulators` before R24). `GBOpt/observability`
itself carries 0 new-shape findings on any of its own paths.

bandit: unchanged (7 Low, 1 Medium, 0 High) -- the new package and both wired
minimizers introduce no subprocess/eval/pickle-shaped code for bandit to flag.

pyscn: 41 quality issues both before and after (`monte_carlo.py`/`genetic.py` each grew
in SLOC/complexity from the added straight-line `_emit()` calls and, in `genetic.py`'s
case, one new selection-membership branch per accept/reject decision, but neither
crossed either gate's threshold for the first time -- all four affected methods
(`run_MC`, `run_GA`, `_run_owned_GA`, `_evaluate_generation`) were already well over
both the complexity and SLOC gates before R24, so no decomposition pass was undertaken
for that pre-existing, out-of-scope debt). Clone pairs: 46 -> 47 (+1, informational
only) -- `MonteCarloMinimizer._emit`/`GeneticAlgorithmMinimizer._emit` are a genuine,
disclosed near-duplicate (identical shape and docstring, differing only in the log
message's "MC step"/"GA generation" wording), matching this codebase's existing,
un-refactored parallel-duplication convention between the two minimizer classes (which
already had its own pre-existing clone pair, e.g. their respective
`_make_initial_manipulator` methods) rather than introducing a shared base class or
mixin neither class currently has.

**Resolve at**: no action needed.

## R24's event `operation_parameters` is always `None` -- no call site retains a `ManipulationResult` to source it from

Issue #84's "candidate/evaluation/operation fields come from authoritative `EvaluationResult`
and `ManipulationResult` data" is satisfiable for candidate/evaluation fields (every event is
built from `evaluation_event_fields()`, itself sourced directly from an `EvaluationResult`/
`CandidateEvaluation`), but not literally for `operation_parameters` today. Auditing every MC/GA
dispatch path that could produce one: `Mutator.mutate()` (legacy `choices` adapter, both the
three-legacy-name table and the registry-resolved generic path) returns only
`(label: str, atoms: np.ndarray)`; `GeneticAlgorithmMinimizer`'s binary/crossover invokers
(`_owned_slice_and_merge_invoker`/`_legacy_slice_and_merge_invoker`/`_run_generic_binary_operation`)
return only `(label, manipulator, atoms)`. Every one of these discards the real
`ManipulationResult` (`result.parameters`/`.lineage`) `operation.execute(context)`/
`manipulator.apply(operation)` actually produced, keeping only the label string --
`GBOpt.optimization.dispatch`'s own established `(manipulator, rng) -> (label, atoms)` /
`(parent1, parent2, rng) -> (label, manipulator, atoms)` return shapes predate R24 and are used
by other callers too, so R24 does not change them. `OptimizationEvent.operation_name` is
therefore sourced from this same label (whatever `Mutator.mutate()`/the binary invokers already
return -- the mutation-dispatch label for the three legacy unary operations, or
`operation.name` for a registry-resolved one), the only operation-identity data actually
reaching MC/GA's own loops; `operation_parameters` has no authoritative source to draw from and
is left `None` everywhere, rather than fabricated from label-string parsing.

**Resolve at**: no action needed unless a later step changes `Mutator.mutate()`'s (or the binary
invokers') return contract to also surface the `ManipulationResult` itself, at which point
`operation_parameters` can be populated from its `.parameters` mapping the same way
`evaluation_event_fields()` already sources candidate/evaluation fields from `EvaluationResult`.

## R24's GA accept/reject events map to next-generation selection, not a Metropolis-style decision

Issue #84 asks for "accept/reject" as one of MC and GA's shared lifecycle events, modeled on
Monte Carlo's own single-current-state Metropolis criterion. The genetic algorithm has no
equivalent concept -- every evaluated population member is either carried over unchanged
(`keep_top_pct`), used as a crossover/mutation parent (`intermediate_pct`), or dropped, with no
notion of "the current state." `GeneticAlgorithmMinimizer._emit()`'s `CANDIDATE_ACCEPTED`/
`CANDIDATE_REJECTED` calls (both `run_GA`'s legacy path and `_run_owned_GA`) therefore mean
"this candidate's structure survives into the next generation's population" (selected via
`_select_indices_by_energy`'s `lowest_indices`/`intermediate_indices`, or -- for the legacy path
only -- also carried indirectly by contributing offspring through `_make_next_generation`) versus
"this candidate contributes nothing to the next generation" (a failed evaluation, or a
successful one outside the selected/bred set). This is disclosed on
`OptimizationEventType`'s own docstring in `GBOpt/observability/types.py` as the authoritative
definition; this entry exists so the interpretive fork itself -- not just its resolution -- is
on record.

**Resolve at**: no action needed; revisit only if a future step wants MC- and GA-comparable
accept/reject semantics badly enough to redefine one of the two algorithms' own established
selection/acceptance policy, which is out of scope for an event-vocabulary step.

## R24 legacy `run_GA`'s initial evaluation gains a try/except purely to emit `RUN_FAILED`, not a recovery boundary

Unlike `MonteCarloMinimizer.run_MC`'s initial evaluation and `_run_owned_GA`'s (both of which
already have some form of failure handling before R24), the legacy `run_GA` path's initial
`self.gb_energy_func(...)` call had no `try`/`except` at all -- any exception propagated
directly, uncaught, matching this file's own "GA docstring... is about the public call
signature, not... failure-handling shape" precedent (R21) that MC/GA's evaluator-boundary code
is not assumed symmetric without checking. Issue #84 asks only for a `RUN_FAILED` event at this
point, not a new recovery boundary, so the added `try/except Exception as exc: self._emit(...);
raise` re-raises the identical original exception unchanged (a bare `raise`, not
`raise GBMinimizerError(...) from exc`) -- the run still ends exactly as it did before R24,
just with one more event emitted along the way. Adding real recovery here (penalizing and
continuing, matching `MonteCarloMinimizer`'s own established pattern) would be a genuine
behavior change no acceptance criterion asked for.

**Resolve at**: no action needed unless a later step's acceptance criteria require a real
recovery boundary at legacy `run_GA`'s initial evaluation, the way R23 added one for MC's.

## R24 legacy `_evaluate_generation`'s returned `EvaluationResult` now reflects a candidate-reconstruction failure too

`_evaluate_generation`'s per-candidate `EvaluationResult` was, before R24, constructed and then
immediately discarded after reading only its `.selection_energy`/`.artifact.path` -- a
subsequent `_make_manipulator_from_file(dump)` failure (the evaluator's file exists but fails to
parse/load) was reflected only in the scalar `gen_energies`/`gen_files` outputs (mutated to
`ENERGY_PENALTY`/`None` in the `except` block), never in the `EvaluationResult` object itself,
which kept reporting `SUCCESS` with the pre-reconstruction-failure energy. This was invisible
because nothing outside the function ever read that `EvaluationResult`. R24 changes
`_evaluate_generation`'s return contract to also surface it (a fourth tuple element, `results:
list[EvaluationResult]`) so `run_GA` can emit `PROPOSAL_EVALUATED`/accept-reject/best-update
events from authoritative per-candidate data instead of re-deriving a coarser one from the
already-penalized scalar outputs -- once external code can actually observe this value, its
prior inaccuracy on the reconstruction-failure path becomes a real, disclosed defect to fix, not
just an internal implementation detail. Both the batch and scalar branches now replace `result`
with a new `EvaluationResult(status=FAILED, failure_stage=FailureStage.ARTIFACT, ...)` in that
`except` block, matching `FailureStage.ARTIFACT`'s existing R23-established meaning ("a missing/
invalid/reused structure path" family). This changes nothing about any value the function
already returned (`gen_energies`/`gen_files`/`evaluated_manipulators` are untouched) -- only the
newly-added, previously-nonexistent fourth return value's accuracy.

**Resolve at**: no action needed; noted for the record as a disclosed accuracy fix enabled by,
not required independently of, R24's own return-contract change.

## R23 net tooling deltas: ruff +4 (disclosed reproduced-pattern debt from new recovery boundaries), mypy net -2, bandit unchanged, pyscn unchanged (41 quality issues, 46 clone pairs)

Baseline taken at the R23 branch point (`d7b15c1`, R22 merged with R14): ruff 179,
mypy 300 errors in 32 files, bandit 7 Low + 1 Medium, pyscn 41 quality issues / 46
clone pairs. Current (after all four R23 commits): ruff 183, mypy 298 errors in 32
files, bandit unchanged, pyscn unchanged.

The ruff +4 is fully accounted for by two files, both from the same deliberate
pattern this file already documents at length (the new batch-callback and
MC-proposal recovery boundaries): `genetic.py` +3 (2 new `BLE001`, 1 new `B905`,
from the two new `except Exception` blocks wrapping `gb_batch_energy_func` and the
`zip(pending_idxs, pending_uids)` pairing inside one of them) and `monte_carlo.py`
+1 (1 new `BLE001`, from the new proposal-evaluation `except Exception` block; the
initial-evaluation one does not trip `BLE001` since it re-raises as `GBMinimizerError`
rather than swallowing). Per this file's own established discipline, these are not
narrowed -- `BLE001` is not automatically safe to narrow, and this codebase has a
deliberate, already-documented precedent for exactly this evaluator/reconstruction-
boundary shape. (An earlier commit message in this same step, "R23: legacy GA
scalar/batch path routes through EvaluationResult," inaccurately reported "ruff...
net 0" -- it verified the *test file's* ruff delta after fixing an unrelated `RUF059`
finding but never re-checked `genetic.py` itself for the +3 shown here; this entry is
the corrected, full account.)

The mypy net -2 is `genetic.py`'s only mypy change, already explained in the "legacy
GA scalar/batch path" entry below (`StructureArtifact.path`'s concrete `str` typing
replacing a `dict.get()`-inferred `object` in the batch branch removes 3 findings
whose argument-type errors no longer apply; one unrelated finding shape persists at a
shifted line, netting -2 once line-shift noise is excluded).

**Resolve at**: no action needed.

## R23 gives MC's evaluator calls the recovery boundary they never had, resolving R21's documented MC/GA asymmetry

R21's `CLAUDE.md` entry (docstring-parity/asymmetry) documented that GA's evaluator
callback has a recovery boundary (`except Exception` -> `ENERGY_PENALTY`) but MC's
`gb_energy_func` call has none at all -- any evaluator exception there propagated
directly and crashed the whole run, and R21 left resolving that asymmetry to whichever
later step's own acceptance criteria required it. Issue #83's "every MC/GA evaluation
produces an `EvaluationResult`" and "evaluator exceptions... retain typed failure
provenance" require it unconditionally, so R23 adds it at both of
`MonteCarloMinimizer.run_MC`'s evaluation call sites, each now classifying its raw
callback result through `from_scalar_tuple`/a directly-constructed `EvaluationResult`
(mirroring the legacy GA scalar path's pattern below), with a new module-level
`MC_ENERGY_PENALTY = 1.0e30` constant local to `monte_carlo.py` (not imported from
`genetic.py`'s own `ENERGY_PENALTY` -- `CLAUDE.md`'s subpackage conventions
already forbid sibling optimizer modules importing each other).

The two call sites are handled differently, deliberately:

- **Initial evaluation** (before the step loop) raises `GBMinimizerError` on failure,
  exactly like `GeneticAlgorithmMinimizer`'s owned-mode initial evaluation -- there is
  no sensible penalized state to start a whole MC run from (`T`/`min_gbe`/`prev_gbe` are
  all derived from it).
- **Proposal evaluation** (inside the step loop) is deterministically rejected on
  failure (`accepted = False`, skipping the Metropolis draw entirely rather than
  relying on `MC_ENERGY_PENALTY` alone making acceptance vanishingly unlikely) and is
  not registered for artifact retention (there is no real relaxed structure to
  register) -- otherwise falling through the loop's existing rejection path
  unchanged (`operation_list`, `rejection_count`, cooldown). This is a real,
  intentional behavior change (a previously-crashing run now degrades gracefully),
  pinned by `test_proposal_evaluator_exception_rejects_step_without_crashing`/
  `test_initial_evaluator_exception_raises` in `tests/test_optimization_monte_carlo.py`.

Five pre-existing checkpoint tests (`test_run_mc_checkpoint_file_is_valid_json`,
`..._format_pickle`, `..._resume_from_json`, `..._resume_from_pickle`,
`..._interval_respected`) used an evaluator exception purely as a mechanism to
interrupt a run mid-way so they could inspect intermediate checkpoint state -- not to
test evaluation semantics. Since an evaluator exception no longer interrupts anything,
they now use a `mutator.mutate` patch (`_install_mutate_crash`, still uncaught,
still propagates) as the interruption mechanism instead, preserving each test's
original step-count-to-checkpoint-state relationship.

**Resolve at**: no action needed; this closes R21's own asymmetry note.

## R23's legacy GA scalar/batch path adds a batch-callback recovery boundary that did not exist before, and does not route reconstruction through `CandidateLoader`

Issue #83 requires "every MC/GA evaluation produces an `EvaluationResult`" and asks to
"route all scalar/batch callbacks through R22 adapters and reconstruct returned
structures through R14 `CandidateLoader`." `GeneticAlgorithmMinimizer._evaluate_generation`
(the legacy, non-owned path) now classifies every candidate's raw callback result
through `from_scalar_tuple`/`from_batch_dict` regardless of source (fresh callback,
batch callback, checkpoint restore, or carryover cache), so `gen_energies` is always
`result.selection_energy` and reconstruction always reads `result.artifact.path`.

Two disclosed, deliberate deviations from a fully literal reading:

1. **The batch callback call itself (`self.gb_batch_energy_func(...)`) previously had
   no recovery boundary at all** -- only the scalar callback and both reconstruction
   points did (matching this file's/CLAUDE.md's own documented count of exactly three
   `except Exception` boundaries in this method before R23). An exception from
   `gb_batch_energy_func` used to propagate and abort the whole generation. It now
   penalizes only the pending candidates in that batch (`FailureStage.EVALUATOR`,
   `ENERGY_PENALTY`), at both of the method's two batch-call sites (checkpoint-enabled
   and not), matching the scalar callback's existing behavior and
   `ExplicitOwnershipEvaluator.evaluate_generation`'s own established batch-exception
   handling. This is a real, intentional behavior change (a previously-crashing run now
   degrades gracefully), required by "every MC/GA evaluation produces an
   `EvaluationResult`" and "evaluator exceptions... retain typed failure provenance"
   taken literally -- pinned by
   `test_batch_evaluator_exception_penalizes_pending_candidates_without_crashing`/
   `..._with_checkpoint_penalizes_and_records` in `tests/test_optimization_genetic.py`.
2. **Reconstruction (`_make_manipulator_from_file`) does not route through
   `CandidateLoader`.** `CandidateLoader.reload()` unconditionally requires a
   `CandidateFileMapping` (persistent grain-ownership state); the legacy path has none
   -- `_make_manipulator_from_file` itself asserts `self.initial_ownership is None` as a
   precondition. This is the same "share the pure computation, not the value-typed
   boundary" fork this file already documents for R16's `translate_right_grain` (a
   value type's constructor invariant is unconditional, with no bypass for a caller that
   genuinely lacks the state it requires): the owned-mode path already goes through
   `CandidateLoader` (via `reload_explicit_manipulator`, confirmed unchanged and
   already-resolved at R14); the legacy path's reconstruction stays a bare
   `GBManipulator(filename, ...)` construction, now just fed by the classified
   `EvaluationResult.artifact.path` instead of a raw dict/tuple value.

Also disclosed as a side effect, not a separate design choice: routing through
`from_scalar_tuple`/`from_batch_dict` is strictly more defensive than the code it
replaced, which did `float(gbe)` unconditionally (crashing on a non-numeric value) and
never checked for a non-finite energy at all (a NaN would previously flow silently into
`gen_energies` and selection comparisons). Both cases are now classified as an ordinary
`FailureStage.VALIDATION` failure with `ENERGY_PENALTY`, matching R12's own precedent for
routing existing code through a newer, already-validating value type.

**Resolve at**: no action needed unless a later step wants legacy-mode reconstruction to
carry real ownership state (at which point routing through `CandidateLoader` becomes
possible, not just desirable).

## R23's owned-GA selection reads `EvaluationResult.selection_energy` at the comparison sites only, not by migrating `CandidateEvaluation` out of GA's owned-mode bookkeeping

Issue #83 requires "every MC/GA evaluation produces an `EvaluationResult`" and "GA
selection consumes `selection_energy` while physical energy/failure metadata remains
available." `CandidateEvaluation` (R22/owned-mode GA's existing type) already has an
unambiguous `.success` flag and a `.objective` that is exactly the physical energy on
success or exactly the optimizer penalty on failure -- it does not itself conflate the
two the way a bare legacy scalar/batch return value does, which is the actual failure
mode `EvaluationResult.selection_energy`/`.energy` exist to prevent. A full migration of
GA's owned-mode internals (checkpoint state construction, lineage tracking, carryover
caching, `_clone_owned_record`, `_rebase_owned_evaluation`, retention registration) onto
`EvaluationResult` in place of `CandidateEvaluation` would touch far more surface than
this substance requires, for identical numeric behavior (`from_candidate_evaluation`
maps `selection_energy=record.objective` on both its success and failure branches, so
the two are never numerically distinguishable at any of these call sites) and materially
higher risk to checkpoint-shape preservation (criterion: "current checkpoint/resume
tests continue to pass until R27-R29 migrate serialization").

The actual selection decisions -- the per-generation energy lists feeding `GBE_vals`/
`history`/`_select_indices_by_energy`, and the best-record comparison choosing the next
`best_record` -- now read `from_candidate_evaluation(record).selection_energy` rather
than `record.objective` directly (`GBOpt/optimization/genetic.py`, `run_GA`'s owned
branch). Every other owned-mode call site (checkpoint state, lineage, carryover cache,
retention registration, cloning) is untouched and continues to read `CandidateEvaluation`
fields directly, since nothing about criterion 6 requires migrating code whose behavior
was already correct. This satisfies "every MC/GA evaluation produces an
`EvaluationResult`" (one is now always constructible, and is constructed, at the point
selection reads it) and "GA selection consumes `selection_energy`" literally, without the
larger, riskier rewrite.

**Resolve at**: no action needed unless a later step (R27-R29's checkpoint serialization
migration, most likely) needs `EvaluationResult` to be the type actually flowing through
GA's owned-mode bookkeeping rather than adapted at the point of use.

## R22's `EvaluationResult` scoped to the 8-item checklist, not issue #82's fuller prose

Issue #82's "Proposed behavior" describes `StructureArtifact` as carrying "stable
artifact ID/digest/metadata," but the acceptance criteria only require a
digest/identifier "where available" -- not that one always be computed, and not a
separate `metadata` field at all. `StructureArtifact` therefore has `path`/`format`/
`digest`, with `digest` defaulting to `None` and only computed on request via
`StructureArtifact.from_path(..., compute_digest=True)` (a SHA-256 hash of the file's
current contents) -- never eagerly, since hashing is not free and most callers (in
particular a `FAILED` result's diagnostic artifact) have no need for one. No `metadata`
field was added, per the same "acceptance criteria are literal constraints, not a
floor" discipline established for #62.

`EvaluationResult` also has no `to_state`/`from_state` serialization, matching
`CandidateEvaluation`'s own precedent (only the artifact-independent
`CandidateEvaluationSummary` variant has them) for the same reason: both hold a live,
non-JSON-serializable `manipulator` reference. A serializable summary type analogous to
`CandidateEvaluationSummary` was not added speculatively -- issue #82 doesn't ask for
one, and per the R08/R09 "speculative value types can be wrong-shaped" precedent, its
real shape should be driven by whichever later step (R23 or R26+) actually needs to
persist an `EvaluationResult` across a checkpoint boundary, not guessed here.

**Resolve at**: no action needed unless a later step needs `StructureArtifact` metadata
or an `EvaluationResult` checkpoint-serialization form.

## R22's adapters cannot always distinguish `CandidateEvaluation`'s collapsed failure origin -- RESOLVED at R23

`from_candidate_evaluation` adapted R22's `CandidateEvaluation.failure_reason` -- a
single opaque string that already collapses several distinct failure origins (evaluator
callback exception, artifact reload failure, ownership reconstruction failure, objective
validation) by the time `_explicit_ownership_evaluation.py` constructs it. Only the
ownership-construction failure is unambiguously recoverable from `record` alone:
`record.mapping is None` happens exactly when `_candidate_file_mapping` itself raised
`GrainOwnershipError`, before any evaluator callback ever runs, so that case maps to
`FailureStage.OWNERSHIP` with certainty. Every other failure shape (callback exception,
`_reload_mapping` failure, invalid/missing objective, reused/invalid structure path) is
attributed to `FailureStage.EVALUATOR` as a disclosed, best-effort default, since
distinguishing them would require regex-matching free-form diagnostic text that was
never a stable contract (the same kind of fragile inference this file's own
`positive=True`/CRLF-substring entries already warn against). Full-fidelity stage
attribution (`ARTIFACT`/`PARSE`/`VALIDATION` correctly separated from `EVALUATOR`) is
only possible from inside `ExplicitOwnershipEvaluator` itself, where the real exception
types are still visible -- not retrofittable after the fact from its already-flattened
output type. `from_scalar_tuple`/`from_batch_dict`, by contrast, see the raw,
undecided data directly and do classify precisely (`VALIDATION` for a missing/non-finite
energy, `ARTIFACT` for a missing/blank structure path).

**Resolved at R23**: `ExplicitOwnershipEvaluator._failed_evaluation` now takes a
required `stage: FailureStage` keyword, and every one of its 9 call sites across
`evaluate_candidate`/`_record_result`/`_restore_checkpointed_result`/
`evaluate_generation` passes the precise stage for the real exception type or
validation check that produced the failure at that point (`OWNERSHIP` for a
`GrainOwnershipError` from `_candidate_file_mapping` or from `_reload_mapping`;
`PARSE` for a `LammpsDataError` from `_reload_mapping`; `ARTIFACT` for an `OSError`
from `_reload_mapping` or a missing/invalid/reused structure path; `VALIDATION` for a
missing or non-finite objective; `EVALUATOR` for a raw evaluator-callback exception).
`CandidateEvaluation` itself gained a `failure_stage: FailureStage | None = None`
field (optional, not part of any checkpoint-serialized shape, so this doesn't touch
`CandidateEvaluationSummary`/`CandidateCheckpoint`'s own contracts) carrying this
attribution through to `from_candidate_evaluation`, which now reads `record.
failure_stage` directly instead of re-inferring it from `mapping is None`. The one
remaining fallback to `EVALUATOR` is `_restore_checkpointed_result`'s restore-from-
checkpoint path: pre-R23 checkpoint metadata never persisted a stage, so a failure
restored from an old checkpoint has no real stage to recover, and `EVALUATOR` is the
same disclosed default this entry originally described, now scoped to exactly that one
case instead of every non-ownership failure.

## R22's `EvaluationStatus`/`FailureStage` reproduce the existing `(str, Enum)` UP042 finding

`BoundaryNormalTopology`, `ArtifactPin`, and `ArtifactStatus` are all already
`class X(str, Enum)`, which ruff's `UP042` flags as preferring `enum.StrEnum` instead.
`EvaluationStatus`/`FailureStage` follow the same established `(str, Enum)` convention
(matching the enum shape already used throughout this codebase) and reproduce the exact
same finding shape at two more call sites -- not new-shape debt, per the "new code that
deliberately mirrors an existing pattern reproduces that pattern's pre-existing findings
verbatim" precedent already documented below for R15's `__current_parent_candidates`.

**Resolve at**: no action needed; revisit only if a future step migrates the codebase's
existing `(str, Enum)` classes to `enum.StrEnum` as a deliberate, disclosed batch change.

## R22 net tooling deltas: mypy net 0, ruff +3 (two disclosed reproduced-pattern, one disclosed established `__init__` convention), bandit unchanged, pyscn unchanged (42 quality issues, 46 clone pairs)

Baseline taken at the R22 branch point (`6d8eec4`, tip of `refactor/r21-optimizer-
logging`): ruff 218 repo-wide; mypy `GBOpt/_explicit_ownership_evaluation.py` 11,
`GBOpt/optimization` 66 (genetic.py 45, checkpointing.py 14, monte_carlo.py 5,
mutation.py 2, types.py 0); bandit 7 low/1 medium/0 high; pyscn 42 quality issues, 46
clone pairs.

The new `GBOpt/evaluation/` subpackage (`types.py`, `adapters.py`, `__init__.py`) adds
no new-shape mypy findings at all: `mypy GBOpt/evaluation` reports the identical 202
total findings across the same 22 pre-existing files that `mypy
GBOpt/_explicit_ownership_evaluation.py` alone already pulls in via
`GBOpt/__init__.py`'s import graph, with zero findings on any `GBOpt/evaluation/*` path
itself. `GBOpt/_explicit_ownership_evaluation.py`'s own 11 and `GBOpt/optimization`'s 66
are both unchanged, since this step adds new files without editing either.

ruff: 221 repo-wide (+3), all three disclosed above (`RUF022` on the new package's
`__init__.py`, matching this codebase's established grouped-not-alphabetized `__all__`
convention; two `UP042` on `EvaluationStatus`/`FailureStage` reproducing the existing
`(str, Enum)` pattern).

bandit: unchanged (7 low, 1 medium, 0 high) -- the new subpackage introduces no
subprocess/eval/pickle-shaped code for bandit to flag.

pyscn: unchanged at 42 quality issues and 46 clone pairs -- no new complexity/length/
dead-code finding crossed the gate's thresholds, and no new clone pair was introduced.

**Resolve at**: no action needed; noted for the record.

## R21 scoped its logging instrumentation to acceptance criteria, not the issue's full "Proposed behavior" prose

Issue #81's "Proposed behavior" text lists module loggers "for run start, initial
evaluation, best updates, generation summaries, failures, and termination," but its
9-item acceptance-criteria checklist only requires MC termination messages stop using
`print()` and that logging levels be "appropriate for lifecycle summaries versus failure
detail" -- it does not require instrumenting every lifecycle stage the prose mentions.
R21 instrumented exactly what the checklist requires: MC's two termination `print()`
calls became `logger.info(...)` (lifecycle summary, satisfies the "appropriate levels"
criterion by example) and GeneticAlgorithmMinimizer's three evaluator/reload-failure
`warnings.warn(RuntimeWarning, ...)` recovery-boundary blocks became `logger.warning(...)`
calls carrying the same exception type/message text they already had (satisfies "retain
exception type/message context"). Run start, initial evaluation, best-update, and
generation-summary logging for either minimizer were not added, per this file's own
"acceptance criteria are literal constraints, not a floor" -- and inverted here, "not
required just because prose gestures at it either" -- discipline established for #62.

**Resolve at**: no action needed unless a later step's issue explicitly asks for broader
lifecycle instrumentation; note this scoping decision if #81 (or a follow-up) is revisited.

## R21's checkpoint-serialized `seed` field assumes a JSON-safe seed value

`MonteCarloMinimizer`/`GeneticAlgorithmMinimizer.seed` (the resolved value actually
passed to `numpy.random.default_rng`) is stored directly in checkpoint `run_params` as
`"seed": self.seed`, matching how other scalar run parameters (`E_tol`, `cooldown_rate`)
are already serialized. `numpy.random.default_rng`'s `seed` parameter also accepts an
array-like of ints, a `SeedSequence`, a `BitGenerator`, or a `Generator` -- none of which
are JSON-serializable, so a caller passing one of those non-`int`/`None` seed forms would
have checkpointing break on the new `seed` field even though checkpointing worked
(without ever persisting the seed) before R21. No test in this codebase constructs either
minimizer with a non-`int`/non-`None` seed; every fixture and every `_make_minimizer`
helper passes a plain `int` (`0`, `101`, ...), matching real usage, so this wasn't treated
as in-scope defensive coercion for R21 (per "don't add validation for scenarios that
can't happen" applied to a scenario nothing in this codebase currently exercises).

**Resolve at**: no action needed unless a future step's caller wants to pass a non-`int`
seed through to either minimizer with checkpointing enabled.

## R20's generic `apply()`-routed third-party operation requires known boundary-normal topology across the whole MC/GA run

Issue #80's compatibility-adapter-vs-generic split (a legacy-named `choices`/
`binary_operations` entry calls the established `GBManipulator` method directly; any
other registry-resolved name routes through `GBManipulator.apply()`/`ManipulationContext`
directly) means a third-party operation is only as reliable as `apply()`'s own
established precondition: `GBManipulator.__parent_candidate()` requires
`normal_topology is not BoundaryNormalTopology.UNKNOWN` (R15). A manipulator built
directly from `GBMaker`/`GBMaker.from_boundary_spec` has known topology, but Monte Carlo
reloads a *new* manipulator from the evaluator's relaxed-structure file after every
accepted step via the deprecated coordinate-based grain-inference path (no
`grain_ownership` passed), which leaves `normal_topology` unknown. A third-party unary
operation used via `choices` therefore reliably participates only for a single MC step
(or for a run whose reload path supplies explicit `grain_ownership`) -- caught while
writing R20's own regression test (`test_third_party_unary_operation_participates_
without_editing_optimizer_source`, `tests/test_optimization_monte_carlo.py`), which had
to be scoped to `max_steps=1` for exactly this reason, with the finding left as a
comment in the test rather than silently working around it. This is a pre-existing
limitation of the `apply()` seam (established at R15/R16), not something R20 introduces;
GA's owned-mode path doesn't have this problem, since its parents already carry explicit
ownership (known topology) throughout.

**Resolve at**: no action needed for R20 itself (issue #80 doesn't require the generic
path to work across a full deprecated-inference MC run); revisit if a future step wants
third-party-operation participation to be reliable across MC's legacy (non-owned,
non-`grain_ownership`) reload path specifically -- would need MC's relaxed-file reload to
either infer topology or accept an explicit one.

## R20 net tooling deltas: mypy net 0 (two has-type cascades introduced and fixed the established way), ruff net +3 (one disclosed intentional re-export, two accepted pre-existing-shape), bandit unchanged, pyscn quality-issue count unchanged (42), clone pairs +2 (informational)

Baseline taken at the R20 branch point (`543a4c5`, R01+R19 merged): mypy
`GBOpt/manipulation` 186, `GBOpt/optimization` 66 (genetic.py 45, checkpointing.py 14,
monte_carlo.py 5, mutation.py 2); ruff 177 repo-wide; bandit 7 low/1 medium/0 high; pyscn
42 quality issues, 44 clone pairs.

mypy: adding `Parent.from_interface_candidate`/`GBManipulator._from_interface_candidate`
re-triggered the established "has-type cascade" pattern (`Cannot determine type of
"__rng"`, 13 findings across the file) purely from adding code to the class body, not
from a new attribute -- fixed the documented way (`self.__rng: np.random.Generator`
annotation on its first `__init__` assignment), back to exactly 186. Adding
`GeneticAlgorithmMinimizer._registry` triggered the same pattern once
(`Cannot determine type of "_registry"`), fixed identically, back to exactly 66 across
`GBOpt/optimization`'s own four files. Net delta across both subpackages: 0.

ruff: 180 repo-wide (+3). Two are pre-existing-shape debt in files R20 substantively
edited (`RUF022` `__all__` not sorted in the new `dispatch.py`, matching this codebase's
established grouped-not-alphabetized `__all__` convention already accepted for
`crystallography`/`artifacts`/`optimization`/`manipulation`'s own `__init__.py`s) --
accepted, not fixed, per that established precedent. One is deliberate:
`GBMinimizerError` imported but unused in `mutation.py` (F401) -- kept because
`tests/test_optimization_mutation.py` imports it from this module's namespace for
backward compatibility (it used to be referenced directly by the pre-OperationSpec
`mutate()`'s own error-raising code, now raised inside `dispatch.py` instead). All other
scoped findings in touched files (`GBManipulator.py`'s `RUF013`/`B028`/`PLC0414`/
`SIM118`, `genetic.py`'s `B905`/`BLE001`) were confirmed pre-existing by line number, not
introduced by this branch's edits.

bandit: unchanged (7 low, 1 medium, 0 high).

pyscn: quality-issue gate count unchanged at 42 (same as baseline, confirming no new
complexity/length/dead-code finding crossed the gate's thresholds). Clone pairs rose 44
-> 46, both informational: `_owned_slice_and_merge_invoker`/`_legacy_slice_and_merge_
invoker` and `_owned_generic_binary_invoker`/`_legacy_generic_binary_invoker`'s near-
identical shapes (matching R16's own established small-helper-duplication tradeoff
between the owned and legacy-file GA code paths, not a new pattern).

**Resolve at**: no action needed; noted for the record.

## R31 (post-roadmap): major version bump, drop legacy-interface compatibility, all-aspects run config

Per user direction during R20's design discussion: once the full R01-R30 roadmap has
landed, a final step (nominally R31, not yet filed as a GitHub issue) should deliberately
break compatibility with the pre-refactor optimizer-facing interface rather than continue
carrying compatibility adapters (`choices: list[str]`, individual constructor args like
`slice_and_merge_pct`/`crossover_surface`, etc.) indefinitely -- R20 through R30 keep
those adapters per each issue's own stated acceptance criteria, but the user was explicit
that over-indexing on compatibility mid-roadmap loses the benefit the whole PR series is
working toward, and that a single deliberate breaking pass at the end (informed by every
subpackage's finished shape, not just one roadmap step's narrow view) is preferable to an
early piecemeal one. This should coincide with a major version bump (currently 0.2.0 on
every branch; nothing has bumped it yet since nothing has merged past
`feature/artifact-retention`/PR #92 toward a real release).

Also raised in the same discussion: a future config file users write should describe
*all* aspects of an optimization run, not just manipulator operations/weights/percentages
-- also the minimization protocol (MC vs GA and their own parameters), logging behavior,
and artifact retention policy. R20's `OperationSpec` + `ManipulationRegistry`-based
operation discovery (see below) was deliberately designed so a future config loader only
needs to describe `{operation name, weight, params, module/path}` per entry and call
`registry.register(...)`/construct `OperationSpec`s from that -- nothing about R20's shape
should need to change to support this, but the config file itself (and equivalent
declarative config for the minimization protocol/logging/retention) is out of scope for
R20 and not yet designed.

**Resolve at**: R31 (or whatever the final roadmap step ends up numbered) -- file a
tracking GitHub issue on request before that step starts.

## R19's `SliceAndMerge` is the first built-in operation routed fully through the `InterfaceCandidate`-typed `execute()` boundary, with no pure/value-typed split

Every prior two-way fork between a legacy method and its `GBOpt.manipulation` operation
(`translate_right_grain`'s leniency conflict, the density/soft-mode operations' missing
`unit_cell`/`gb_thickness` schema fields) required sharing only the pure computation and
keeping the legacy method on its own, separately-packaged path. `slice_and_merge` needed
neither kind of split for its *packaging*: any parent this manipulator can construct an
interface candidate for (via its own existing concatenated-labels fallback for parents
without persistent ownership) can also be sliced-and-merged into another valid
candidate, so `SliceAndMerge.execute` constructs and returns a real `InterfaceCandidate`
unconditionally. The one real fork is `rng`: several existing regression tests assign a
duck-typed random-source double (exposing only `.random()`) directly to a manipulator's
own RNG slot, and `ManipulationContext.__init__` unconditionally rejects a
non-`np.random.Generator` value. `GBManipulator.slice_and_merge` therefore calls the
shared pure function (`crossover_slice_and_merge`) directly with its own parents and
RNG, never constructing a `ManipulationContext`/`InterfaceCandidate` -- the same
"share the pure computation" shape as the earlier cases, but driven by the `rng` type
constraint rather than a labeling/schema mismatch.

**Resolve at**: no action needed; noted for the record as a variant of the established
pattern.

## `SliceAndMerge.execute` self-checks its own arity, unlike every other built-in operation

Every other operation's `execute` trusts its caller (`GBManipulator.apply`) to have
already checked arity, and does not re-check it. `SliceAndMerge.execute` instead raises
`ManipulationArityError` itself for a non-2 parent count. This is a deliberate
deviation: `GBManipulator.slice_and_merge` never routes through `apply` (see the entry
above), so `apply`'s generic arity check never runs for the legacy entry point, and a
directly-constructed `ManipulationContext` with the wrong parent count would otherwise
fail with a plain `IndexError`/silently-ignored extra parent rather than a named arity
error.

**Resolve at**: no action needed; revisit only if a future step wants every operation's
arity-checking discipline unified one way or the other.

## `SliceAndMerge`'s preflight compatibility validation is unconditionally stricter than `GBManipulator.slice_and_merge`'s own no-ownership-mode callers

`crossover_slice_and_merge`'s topology/boundary-topology/affine-equivalent-geometry
checks are gated on `grain_labels is not None`, reproducing `GBManipulator.
slice_and_merge`'s exact existing behavior for both ownership modes. Since an
`InterfaceCandidate` always carries real `grain_labels`, `SliceAndMerge.execute` always
exercises the stricter, gated branch; only `GBManipulator.slice_and_merge`'s legacy
no-ownership callers (parents without persistent grain ownership) still skip it, exactly
as before. This is the same "stricter by construction once a caller can only reach the
operation through a real `InterfaceCandidate`" shape already documented for interface
separation's own parent-candidate conversion.

**Resolve at**: no action needed; noted for the record.

## `__translate_manipulation_error` gained a `capability_exception` parameter for `slice_and_merge`'s established `CompositionAwareCrossoverError` identity

`ManipulationCompatibilityError` (defined at R15, unused until now) and
`ManipulationCapabilityError` are both raised by `crossover_slice_and_merge`, but
`GBManipulator.slice_and_merge`'s own public contract distinguishes them: a
structural/topology mismatch surfaces as plain `GBManipulatorValueError`, while a
composition-specific failure (formula mismatch, no admissible cut interval) surfaces as
the existing `CompositionAwareCrossoverError`. Rather than write a second, near-duplicate
translation helper, `__translate_manipulation_error` took an optional
`capability_exception` keyword (defaulting to `GBManipulatorValueError`, unchanged for
every other existing call site) that replaces only the `ManipulationCapabilityError`
branch.

**Resolve at**: no action needed; noted for the record.

## R19 did not add a `make_slice_and_merge_candidate` facade method or register `SliceAndMerge` in `default_registry`

Matches R18's own deferral for `AtomInsertion`/`AtomRemoval`/`SoftModeDisplacement`: issue
#79's acceptance criteria ask for the operation class and a behaviorally-equivalent
legacy delegate, not a dedicated `InterfaceCandidate`-returning convenience method or
default registration. `manipulator.apply(SliceAndMerge(), unit_cell=(...), gb_thickness=...)`
already reaches it directly.

**Resolve at**: a standalone addition on request, following `make_translation_candidate`.

## R19 net tooling deltas: mypy net -29 (215 -> 186, decomposed across two files), ruff unchanged (21), pyscn quality-issue count unchanged (7, one finding relocated)

mypy: `GBManipulator.py` dropped 148 -> 113 findings, all 35 of the difference in the
`[attr-defined]` "`None` has no attribute" shape -- `slice_and_merge`'s body no longer
reads attributes directly off `self.__parents[0]`/`[1]` (typed `Parent | None`); it
passes them, unread, to `crossover_slice_and_merge`, which reads the same attributes
through its own `CrossoverParent`-typed parameters instead. The new
`GBOpt/manipulation/crossover.py` adds 6 findings, all matching the two shapes R18's own
`density.py`/`soft_mode.py` already established for this package (an `object`-typed
`context.params` value flowing into a narrower-typed argument or comparison; an
`X | None`-typed field read without the caller-side narrowing mypy can't see across an
`is None` check made on a *different*, correlated field). Net across the two files:
-29. ruff: unchanged at 21 (the extraction introduced one new `RUF023`
`__slots__`-not-sorted finding on a new class, fixed immediately by sorting the slots --
a trivial, would-make-regardless change, not tool appeasement). pyscn: unchanged at 7
quality issues -- `GBManipulator.slice_and_merge`'s own SLOC-length finding (191 SLOC)
disappeared with the method itself and reappeared as `crossover_slice_and_merge`'s (209
SLOC, the body plus its own preflight/composition logic in one function); clone pairs
rose 18 -> 28, all informational, matching the established small-helper-duplication
tradeoff (`_validate_finite_real`/`_require` duplicated into `crossover.py`, as every
sibling operation module already does).

Unlike R18's own insertion/removal/soft-mode SLOC findings (deliberately left
undecomposed because they had no dedicated direct tests to protect a further split),
`crossover_slice_and_merge` now has direct unit-level test coverage
(`tests/test_manipulation_crossover.py`), so a future decomposition pass would have
something to verify against if anyone wants to revisit it.

**Resolve at**: no action needed; a further decomposition of `crossover_slice_and_merge`
is a standalone option on request, now that it has direct test coverage.

## R18 did not add `make_removal_candidate`/`make_insertion_candidate`/`make_soft_mode_candidate` facade methods

R15/R16's translation/termination/separation operations each have a matching
`make_*_candidate()` convenience method on `GBManipulator` returning an
`InterfaceCandidate` directly (`make_translation_candidate`, `make_termination_candidate`).
R18 did not add analogous methods for `AtomInsertion`/`AtomRemoval`/
`SoftModeDisplacement` -- issue #78's acceptance criteria never ask for them (only for
the operation classes themselves and legacy-wrapper compatibility), and time was
prioritized on getting the shared pure-function extraction and its behavior-preservation
verification right. An external caller can still reach these three operations directly
via `manipulator.apply(AtomInsertion(), ...)` / `apply_named(...)` (once registered),
same as the R15 facade seam already supports for any operation not routed through a
dedicated legacy method.

**Resolve at**: a standalone addition on request, following the existing
`make_translation_candidate`/`make_termination_candidate` pattern exactly, if scripted
callers need direct `InterfaceCandidate` results from insertion/removal/soft-mode
without going through the legacy raw-array-returning methods.

## R18 increased `GBOpt.manipulation`'s mypy finding count (209 -> 215) by moving previously mypy-unscanned-as-a-unit private helpers into a checked module boundary

`GBOpt/GBManipulator.py`'s own findings (within the `mypy GBOpt/manipulation` scoped
run) dropped from 167 to 148 as extracted code left the file, but the two new modules
(`density.py`: 20, `soft_mode.py`: 14) introduce findings of the same long-established
shapes elsewhere in this codebase (dict/`object`-typed duck-typed parameters flowing
into `int()`/comparison operators, a numba `prange` "not iterable" false positive,
`Need type annotation for "..."` for a list built via `.extend()` in a loop,
`Incompatible types in assignment` for a list-then-ndarray reassignment, `min(...,
key=dict.get)`'s well-known mypy false-positive against `dict[int,int].get`, and one
genuine `No overload variant of "int" matches argument type "object"` following from
`context.params`'s `Mapping[str, object]` typing, unrelated to this step). None were
fixed with a `cast()` or a type-checker-motivated rewrite, per this file's own
established "would I make this change if mypy didn't exist?" discipline -- they are
disclosed, not silently accepted or hidden.

**Resolve at**: no action needed; a future step could add explicit type narrowing at
`context.params` read sites (e.g. a small typed-params helper) if this class of finding
recurs enough to be worth a shared fix.

## `displace_along_soft_modes`'s `threshold` parameter is computed but never used -- still open after R18

R18 (#78) touched `displace_along_soft_modes` (extracting its body into
`GBOpt.manipulation.soft_mode.soft_mode_displacement_atoms`) but deliberately did not
resolve this entry's underlying question. `threshold`'s validate-then-default
computation was kept, verbatim, in the legacy wrapper (it doesn't get passed to the
extracted pure function at all, matching its pre-R18 dead status exactly) rather than
guessing whether it was ever meant to clamp displacement magnitude somewhere. This is
a physics-intent question outside a refactor step's authority to decide unilaterally,
not something the extraction itself resolves by moving code around.

**Resolve at**: standalone question to the user (is `threshold` meant to clamp
displacement magnitude somewhere, and that clamp was lost, or was it always dead?) --
not tied to any specific future roadmap step.

## R18's soft-mode/density operations require `unit_cell` and `gb_thickness` as explicit `context.params`, since `InterfaceCandidate` carries neither

Issue #78 (R18) extracted `insert_atoms`/`remove_atoms`/`displace_along_soft_modes`
into `AtomInsertion`/`AtomRemoval`/`SoftModeDisplacement` (`GBOpt/manipulation/density.py`,
`GBOpt/manipulation/soft_mode.py`). All three need a unit cell (for ideal bond lengths,
radius, neighbor distances, ratios) and a GB-region definition, neither of which
`InterfaceCandidate` (R15) carries -- it only has `gb_plane_x`, not a thickness or a
unit cell. Rather than growing `InterfaceCandidate`'s schema (a bigger, cross-cutting
change touching every existing operation and out of #78's stated scope), both
operations take `unit_cell` and `gb_thickness` as required `context.params`, and
recompute GB-region membership from `gb_plane_x`/`gb_thickness`/the candidate's own
atoms (`_gb_region_indices`, duplicated identically in `density.py` and `soft_mode.py`
per this package's established small-helper-duplication convention). This is a genuine
fork between input-side and output-side data requirements for a not-yet-fully-general
value type, the same shape as R15's "ambiguous which side of a boundary a criterion
binds" entry -- disclosed here rather than silently deciding either way.

`_gb_region_indices`'s formula (whole-system atoms with x within `gb_thickness / 2` of
`gb_plane_x`) matches `Parent.__finish_init`'s `grain_ownership is not None` branch,
not its `grain_ownership is None` branch (which instead derives GB membership from
`left_grain`/`right_grain` arrays with a strict, not inclusive, bound comparison, and
can produce a different row order/membership set for edge-case atoms sitting exactly at
the GB-region boundary). Since `InterfaceCandidate` always represents persistent
explicit ownership, the `grain_ownership is not None` formula is the principled match --
but this means `AtomInsertion`/`AtomRemoval`/`SoftModeDisplacement`'s own GB-region
selection is not guaranteed byte-for-byte identical to `Parent.gb_atoms`/`gb_indices`
for a parent built without explicit `grain_ownership`. `GBManipulator.insert_atoms`/
`remove_atoms`/`displace_along_soft_modes` were **not** rerouted through these new
operations for exactly this reason -- they continue to read `Parent.gb_atoms`/
`gb_indices` directly, unchanged, and only share the stochastic-selection/site-generation/
displacement pure functions (`select_removal_indices`, `select_insertion_sites`,
`delaunay_insertion_sites`, `grid_insertion_sites`, `soft_mode_displacement_atoms`) with
the new operations -- the same "share the pure computation, not the value-typed
boundary" split R16 used for `translate_right_grain`, but for a different underlying
reason (no `InterfaceCandidate` leniency conflict here; insertion/removal/displacement
outputs are always `InterfaceCandidate`-constructible once GB-region membership is
resolved).

**Resolve at**: no action needed unless a future step wants `AtomInsertion`/
`AtomRemoval`/`SoftModeDisplacement`'s own GB-region formula to exactly match
`Parent`'s `grain_ownership is None` behavior too, which would need `InterfaceCandidate`
itself to grow a GB-thickness or GB-region field.

## R18 found and fixed two silent bugs in `insert_atoms`/`remove_atoms`'s `keep_ratio=False` multitype branch: global-RNG use and a mismatched probability-array length

Both bugs predate R18 and were not caught by any existing test (neither is exercised by
the current suite -- every `keep_ratio=False` test uses a single-type unit cell, which
takes a different code path entirely). Found while extracting this logic into
`GBOpt.manipulation.density`'s pure functions, since the acceptance criterion "All
stochastic selection uses `ManipulationContext.rng`; no global NumPy RNG is used by
migrated paths" made the first one directly relevant, and reading the two implementations
side by side (per this file's own established "read old and new side by side" discipline)
surfaced the second one as a latent crash the extraction would otherwise have
faithfully preserved:

- **Global RNG**: `remove_atoms`/`insert_atoms`'s `keep_ratio=False`-with-multiple-types
  branch called `np.random.choice(range(1, num_to_remove), ...)`/
  `np.random.choice(range(1, num_to_insert), ...)` directly -- the global NumPy RNG,
  never `self.__rng` -- to split the requested count randomly across types. A caller
  passing `seed=...` to the `GBManipulator` constructor got no determinism for this
  split. Fixed via a new shared helper, `_random_type_counts(total, num_types, rng)`
  (`GBOpt/manipulation/density.py`), which both the legacy methods (passing
  `self.__rng`) and `AtomInsertion`/`AtomRemoval` (passing `context.rng`) now call.
- **Mismatched probability array**: `insert_atoms`'s analogous branch called
  `self.__rng.choice(available_indices, num, replace=False, p=probabilities)`, where
  `available_indices` shrinks each iteration (sites already assigned to an earlier type
  are excluded) but `probabilities` stays the full, original-length array. `numpy`'s
  `Generator.choice` requires `len(p) == len(a)`, so any multitype `keep_ratio=False`
  insertion with more than one nonzero-count type after the first would raise
  `ValueError: a and p must have same size` at runtime. `select_insertion_sites`'s
  non-`keep_ratio` branch now renormalizes `probabilities[available_indices]` before
  each draw, matching `select_removal_indices`'s already-correct unweighted analog
  (removal's equivalent branch never passed `p=` at all, so it never had this bug).

Both fixes are visible only by reading the code (no regression test exercises either
path); disclosed here per this file's "a decrease/behavior change needs the same
explanation an increase would" discipline, extended to a fix with no test coverage to
verify it against.

**Resolve at**: no action needed; consider adding direct multitype `keep_ratio=False`
coverage for `select_removal_indices`/`select_insertion_sites` as a standalone
test-hygiene item, since neither bug nor its fix is currently exercised by any test.

## R18 fixed `insert_atoms`'s bare `raise GBManipulatorValueError` (dropped its message) for the `fill_fraction` bounds-check error

`insert_atoms`'s `fill_fraction` out-of-range check was `raise GBManipulatorValueError`
followed, on the next line, by an orphaned parenthesized f-string expression statement
that was never actually part of the `raise` (a typo -- missing parentheses directly after
`GBManipulatorValueError`) and, being unreachable code after a `raise`, was never
executed. This raised the bare exception class with no message. Its sibling check in
`remove_atoms` (`gb_fraction` bounds) has the correct
`raise GBManipulatorValueError(f"...")` form. `test_insert_atoms_fraction_error` only
asserts the exception type via `assertRaises`, never the message, so this was never
caught. Fixed as part of rewriting this validation block during extraction --
matching the sibling method's already-correct pattern is a "would make this change
regardless of the refactor" fix (same category as R10's `MakerConfig`-field-default
fixes), not a speculative improvement.

**Resolve at**: no action needed; noted for the record.

## R18's `select_removal_indices`/`select_insertion_sites`/`AtomInsertion.execute`/`soft_mode_displacement_atoms` each trip the SLOC-length quality check (>100 lines)

Same shape as R12's `LammpsDataWriter.write()` finding and R09's pipeline-composition
finding: `select_removal_indices` (138 SLOC), `select_insertion_sites` (101 SLOC),
`AtomInsertion.execute` (111 SLOC), and `soft_mode_displacement_atoms` (121 SLOC) each
exceed `pyscn`'s 100-line-per-function threshold. Unlike those precedents, a further
decomposition pass was not done in this step: `select_removal_indices`/
`select_insertion_sites` are near-verbatim extractions of `remove_atoms`/
`insert_atoms`'s own already-dense stochastic-selection logic (splitting them
further risked introducing exactly the kind of subtle selection-order/RNG-draw-order
bug this file's own "verify byte-for-byte formula equivalence" discipline warns
against, for functions with no dedicated direct unit tests to catch it), and
`soft_mode_displacement_atoms` is R17's own explicitly "no unrelated change to the
physical soft-mode calculation" body, extracted as a single unit on purpose.

Net effect confirmed via `pyscn check` CLI run against the pre-R18 branch point vs.
after (the MCP tool was unavailable mid-session; CLI numbers are not directly
comparable to an earlier MCP-recorded baseline, so this diffs CLI-against-CLI on the
same file set per this file's own documented tooling caveat): total complexity/length
quality issues dropped from 14 to 7 (both `insert_atoms` and `remove_atoms` fell out of
the complexity-threshold list entirely: 29->below 10 and 25->below 10 respectively; both
also dropped off the SLOC-length list except `insert_atoms`, whose own remaining body is
still 129 SLOC, itself over threshold and not resolved here either), and unreachable
dead code dropped from 4 findings to 1 (three fixed as side effects: two were the
grid-site-generation dead code below, one was the `insert_atoms` bare-raise fix above;
the fourth, `apply_group_symmetry`'s stub `NotImplementedError`, is unrelated
pre-existing debt, untouched). Clone-detection pairs rose from 12 to 18, all
informational: new pairs are between `density.py`/`soft_mode.py`'s duplicated small
helpers (`_create_neighbor_list`, `_gb_region_indices`) and between
`GBManipulator.py`'s legacy `new_atoms`-array-construction code and
`AtomInsertion.execute`'s own copy of the identical logic (not factored into a shared
helper, matching this package's established duplicate-small-block-across-the-legacy-
boundary tradeoff).

**Resolve at**: a standalone decomposition pass on request, ideally paired with adding
the direct unit tests these four functions currently lack (matching the still-open
`gbmaker/geometry.py` complete-origin-kernels entry above) so a further split can be
verified against real assertions, not just re-reading the diff.

## `insert_atoms`'s `grid_approach` had genuinely dead code after its early `return` -- removed during R18 extraction

The original `grid_approach` (a nested function inside `insert_atoms`) ended with
`return filtered_sites, probabilities`, followed by an `indices = self.__rng.choice(...)`
/ `return filtered_sites[indices]` block and a `raise GBManipulatorValueError(...)` --
all unreachable after the early return, confirmed via `pyscn`'s
`unreachable_after_return`/`unreachable_after_raise` findings (both present in the
pre-R18 baseline, both gone after). This dead code was not carried into
`GBOpt.manipulation.density.grid_insertion_sites` (the pure-function extraction of this
logic): it computed nothing observable (the `self.__rng.choice` call/`raise` were never
reached at runtime), so dropping it is a pure dead-code removal, not a behavior change,
consistent with this file's own "genuinely dead code" precedent for R17's `disp_vector`
no-op line.

**Resolve at**: no action needed; noted for the record.

## `displace_along_soft_modes`'s dense-`eigh` branch now raises `GBManipulatorValueError` instead of crashing on an oversized mode request

Before R17, requesting more eigenvectors than `3 * n_atoms` (via `num_children`) only
raised a clean `GBManipulatorValueError` on the sparse (`eigsh`) branch; the dense
(`np.linalg.eigh`) branch had no equivalent guard and would instead fail later with a
confusing `ValueError: could not broadcast input array from shape (X,) into shape
(Y,)` when slicing `freq_vals[:num_modes_needed]`/assigning into `freqs[i]`. R17 added
an explicit `num_modes_needed > 3 * n_atoms` check (right after `n_atoms` is computed)
so that an out-of-range `mode_index` always raises `GBManipulatorValueError`, per
issue #27's own acceptance criterion ("out-of-range or invalid mode_index raises the
established GBManipulatorValueError"). This is a genuine, required behavior addition
for the new API, not an incidental tightening -- flagged here only because it also
happens to close a latent crash-vs-clean-error gap that existed for the old
`num_children` parameter too (now moot, since `num_children` is gone).

**Resolve at**: no action needed; noted for the record.

## R16's built-in operations only reuse `right_grain_translation_atoms` for `translate_right_grain`'s raw-array leniency -- the InterfaceCandidate-based path is stricter by construction

Issue #77 (R16) asked for operation objects that "consume explicit `InterfaceCandidate`
state" for right-grain translation, termination cycling, and interface separation, with
legacy methods becoming thin delegating wrappers. `InterfaceCandidate.__init__`
unconditionally validates that every atom lies inside its labeled physical grain
bounds. Legacy `GBManipulator.translate_right_grain` (`GBOpt/GBManipulator.py`)
deliberately does not have this constraint: its own comment explains that explicit
ownership is persistent state, and a relaxed right-grain atom may legitimately already
lie -- or be translated to lie -- outside `right_grain_x_bounds` without changing
grains. Two currently-passing tests exercise exactly this
(`test_explicit_ownership_inplane_translation_allows_crossed_right_atom`,
`test_explicit_ownership_x_translation_rejects_crossed_right_atom`, both in
`tests/test_gbmanipulator.py`), including a case where the *parent's own current
state*, before any translation, already has a right-owned atom outside its physical
bound -- meaning even constructing an `InterfaceCandidate` for the parent (not just for
the translated result) fails for this scenario, independent of where in the pipeline
the construction happens.

Given this, `translate_right_grain` cannot be routed through `ManipulationContext`/
`Manipulation.execute()` at all (both require `InterfaceCandidate` parents and
produce `InterfaceCandidate` children). The resolution: `GBOpt/manipulation/
translation.py` factors the actual translation math into a pure function,
`right_grain_translation_atoms(*, left_atoms, right_atoms, right_grain_x_bounds,
box_dims, inplane_periodic, coordinate_tolerance, dx, dy, dz) -> np.ndarray`, that
takes plain arrays and geometry (not an `InterfaceCandidate`) and returns a plain
array (not an `InterfaceCandidate`) -- deliberately never touching
`InterfaceCandidate` construction or validation. `RightGrainTranslation.execute()`
calls this function and then wraps the result in a validated `InterfaceCandidate` (so
`apply()`/`apply_named()`/`make_translation_candidate()` -- which already enforced
this bound pre-R16 via `parent._to_interface_candidate(...)` -- keep exactly the
stricter behavior they always had). `GBManipulator.translate_right_grain` calls
`right_grain_translation_atoms` directly with `Parent.left_grain`/`right_grain`
arrays, never constructing an `InterfaceCandidate` at any point, preserving its
historic leniency exactly. The two entry points share the real computation (the
"delegates" half of the issue's ask) but intentionally diverge on packaging/validation
strictness, which is the only way to satisfy "outputs are equivalent for the current
regression cases" given `InterfaceCandidate`'s fixed validation contract.
`cycle_grain_terminations` did not need this split: it already validated
atoms-inside-bounds itself before cycling, so its result was always compatible with
`InterfaceCandidate` construction (see the next entry for a related but narrower
tolerance nuance in that check).

**Resolve at**: no action needed unless a future step wants `translate_right_grain`
itself to stop tolerating a crossed right-grain atom, which would be a deliberate
behavior change, not a refactor.

## R16's `cycle_grain_terminations`/`apply_interface_separation` wrappers now construct an `InterfaceCandidate` for the current parent, which is a strictly stricter (and, for termination cycling, slightly differently-toleranced) check than either method required before

Both methods' R16 wrappers call a new private helper, `GBManipulator.
__parent_candidate_geometry(index)`, to convert `self.__parents[index]` into an
`InterfaceCandidate` for use as `ManipulationContext.parents[0]`. This conversion
(`Parent._to_interface_candidate`) itself runs `InterfaceCandidate.__init__`'s
atoms-in-labeled-bounds check across all three axes, using a tolerance applied only on
each bound's lower side (`coordinates < lower - tolerance or coordinates >= upper`;
see `GBOpt/interface/model.py`).

- `apply_interface_separation`'s *original* implementation never converted the parent
  to an `InterfaceCandidate` at all -- it only read `parent.box_dims`/`gb_plane_x`/
  `left_grain_x_bounds`/`right_grain_x_bounds`/`normal_topology`/`inplane_periodic` as
  plain scalar/array attributes, with no dependency on `parent.whole_system` or its
  grain labels. The R16 wrapper's `__parent_candidate_geometry(0)` call is new,
  additional work with a new failure mode: a parent whose stored atoms don't
  self-consistently satisfy `InterfaceCandidate`'s bound check would now raise before
  `InterfaceSeparation.execute()` even inspects `candidate`. In practice this is very
  low risk: every test (and every realistic caller) sources `candidate` from
  `make_parent_candidate()`/`make_translation_candidate()`/`make_termination_candidate()`,
  all three of which already perform this exact same conversion successfully as their
  own precondition, so by the time a valid `candidate` argument exists, the parent's
  own conversion has already been proven to succeed. No test failure was observed from
  this change (full non-slow suite pass/fail set is unchanged from the pre-R16
  baseline).
- `cycle_grain_terminations`'s *original* implementation already had its own explicit
  atoms-in-bounds check (`np.any(left_x < left_bounds[0] - tolerance) or
  np.any(left_x >= left_bounds[1] + tolerance)`, i.e. tolerance applied on *both*
  sides of each bound) before cycling, so the R16 wrapper's additional
  `InterfaceCandidate`-construction check is very unlikely to newly reject anything the
  method's own subsequent check wouldn't already have rejected -- except for the narrow
  edge case of an atom sitting strictly between `upper` and `upper + tolerance`, which
  the legacy check tolerated but `InterfaceCandidate.__init__`'s lower-side-only
  tolerance does not. No test currently exercises an atom in that narrow band (full
  suite pass/fail set unchanged), so this is disclosed as a theoretical tightening, not
  a confirmed regression.

**Resolve at**: no action needed unless a future step surfaces a real case hitting
either narrowed edge; if so, the fix is likely `__parent_candidate_geometry` using a
two-sided-tolerance bounds check matching `cycle_grain_terminations`'s historic one,
or (for `apply_interface_separation`) reverting to reading `parent`'s raw scalar
attributes instead of a full candidate conversion for its reference-geometry checks.

## R16 did not route any built-in operation through `GBManipulator.apply()`/`apply_named()` -- the R15 facade-seam entry below is still open

R16 (#77) added three built-in operations (`RightGrainTranslation`,
`GrainTerminationCycle`, `InterfaceSeparation` in `GBOpt/manipulation/`) and wired
`GBManipulator`'s legacy methods to call them directly via a freshly constructed
`ManipulationContext`, not via `self.apply(operation)`. `apply()`'s generic arity-check
(`arity != len(parents): raise ManipulationArityError`) is incompatible with
`translate_right_grain`'s own legacy leniency (warn and still translate using parent 1
when the manipulator actually has two parents) and with `make_translation_candidate`/
`cycle_grain_terminations`/`apply_interface_separation`'s explicit
"exactly one parent required" checks, which raise a specific
`GBManipulatorValueError` message rather than `apply()`'s generic
`ManipulationArityError`. So the R15 entry "R15's `GBManipulator.apply()` seam has a
narrower 'preserved' contract than the issue prose might suggest" (below) remains
exactly as open as it was after R15: still no built-in operation is routed through
`apply()`/`apply_named()`, and this entry's own "Resolve at" (whichever step first
routes a built-in algorithm through the seam) has not yet been reached. An external
caller can still do `manipulator.apply(RightGrainTranslation(), dy=..., dz=...)`
today; it simply isn't how `GBManipulator`'s own legacy methods reach these
operations.

**Resolve at**: unchanged from the R15 entry -- whenever a roadmap step actually wants
`GBManipulator`'s own legacy methods to route through the generic `apply()` seam (which
would require either relaxing `apply()`'s arity check or changing these methods'
long-standing single-parent/two-parent semantics), or if a caller-facing need for
`apply_named()` access to these three built-ins (e.g. registering them in
`default_registry` under stable names) emerges.

## R16's `translation.py`/`termination.py`/`separation.py` duplicate small validation/construction helpers rather than importing across siblings

`_validate_finite_real` (the `TypeError`-for-bool-or-non-real /
`ManipulationConfigurationError`-for-non-finite split) and a `_build_candidate`-style
"construct `InterfaceCandidate`, translate its exceptions" helper are duplicated,
nearly verbatim, across `GBOpt/manipulation/translation.py`,
`GBOpt/manipulation/termination.py`, and `GBOpt/manipulation/separation.py` (the
latter's version is inlined in `InterfaceSeparation.execute` rather than factored into
a helper, since its geometry differs). `_translate_inplane` is duplicated between
`translation.py` and `termination.py` (both need in-plane right-grain translation).
This follows the same "sibling modules at the same level should not import each
other" / duplicate-small-helper precedent CLAUDE.md already documents for
`GBOpt/interface/model.py` vs `GBOpt/GBManipulator.py`'s own nearly-identical
`_validate_finite_real`/`_strict_float_array`/`_normalize_inplane_periodic`/
`_normalize_grain_labels`/`_readonly_copy` copies, rather than a new pattern.
`pyscn`'s clone detector flags these duplicates (9 clone pairs after R16, up from 4
before, all informational, not counted toward its quality-issue gate).

**Resolve at**: no action needed; this is the codebase's established tradeoff between
avoiding sibling-to-sibling imports and avoiding duplicate literal text. Revisit only
if a fourth sibling operation module needs the same helpers, at which point promoting
them to a small non-`types.py` shared module (still leaf-level, still no operation
logic) might be worth it.


## R15's `GBManipulator.apply()` seam has a narrower "preserved by the facade" contract than the issue prose might suggest

Issue #76 (R15)'s acceptance criteria say "Interface topology, ownership labels,
physical bounds, and separation state are preserved by the facade adapter." R15's
`GBManipulator.apply()`/`apply_named()` (`GBOpt/GBManipulator.py`) satisfy this only for
the *input* side: `__current_parent_candidates()` converts this manipulator's own
`Parent` state into `ManipulationContext.parents` using the exact same
`_to_interface_candidate()` call `make_parent_candidate()` already used, so topology,
ownership labels, physical grain bounds, and interface-separation are threaded through
correctly on the way in. Nothing in `apply()` re-validates that an operation's *returned*
children still carry matching topology/bounds/separation -- there is no built-in
operation yet (per this same issue's own "No built-in manipulation algorithm is moved
yet" criterion) whose output the facade could meaningfully check against, and an
externally defined test operation is free to construct whatever valid `InterfaceCandidate`
it wants. If a later roadmap step wires a real built-in algorithm through this seam and
wants the facade itself (not just the operation) to guarantee output geometry didn't
drift, that is new scope for that step, not an oversight here.

**Resolve at**: whichever roadmap step first routes a built-in manipulation algorithm
through `apply()`/`apply_named()` -- decide then whether output-side validation belongs
in the facade or stays the operation's own responsibility.

## R15's `ManipulationRegistry` intentionally has no `unregister`/`names`/`__contains__`

Issue #76 asks for "duplicate names and unknown names have explicit errors" only --
`register()`/`get()` cover that completely. `unregister`/`names`/`__contains__` were
considered and left out as unrequested surface area; tests that need isolation from the
shared `GBOpt.manipulation.default_registry` construct their own
`ManipulationRegistry()` instance instead (see `tests/test_gbmanipulator.py`'s
`TestGBManipulatorApply.test_apply_named_*`) rather than relying on the singleton plus a
removal method.

**Resolve at**: whenever a later roadmap step's operation lifecycle actually needs
deregistration or introspection -- add it then, against a concrete call site.

## R15's `GBManipulator.apply()`/`__current_parent_candidates()` reproduces `make_parent_candidate()`'s pre-existing mypy `Optional`-typing findings, not new debt

`self.__parents` is typed as a list that can hold `None` (`list[Parent | None]`), so any
code that iterates it and calls `.normal_topology`/`.grain_labels`/
`._to_interface_candidate()` on an element gets mypy's `"None" has no attribute ...`
family of findings -- `make_parent_candidate()` already carried 7 of these before R15.
The new `__current_parent_candidates()` helper (added for `apply()`) calls the identical
pattern for both parents and picks up 6 more of the same shape (verified: `mypy
GBOpt/manipulation` reports 202 `GBManipulator.py` findings after R15 versus 196 before,
and the +6 are all at the new method's lines, matching the existing 7's error text
exactly). This is the same untyped-`Optional`-list root cause already present in the
file, surfacing again because the new code follows an established pattern, not a new
category of typing debt R15 introduced.

**Resolve at**: whichever roadmap step next gives `self.__parents` a properly-typed
container (e.g. narrowing per `__one_parent`, or a small state object the way R10 did
for `GBMaker`) -- no specific step scheduled yet.

## `GBManipulator.__init__`'s `if not seed:` seed handling didn't accept `seed=0` -- RESOLVED at R20

`GBManipulator.__init__` (`GBOpt/GBManipulator.py`) resolved its own `self.__rng` with
`if not seed: self.__rng = np.random.default_rng()` -- `seed=0` is falsy in Python, so a
caller passing `seed=0` to the constructor silently got an unseeded (non-deterministic)
generator instead of the deterministic one they asked for. This predated R15 and was
unrelated to issue #76's scope (R15 didn't touch the constructor). R15's own `apply()`
seed parameter already avoided the same bug (`seed is None`, not `not seed`), so the two
call paths had inconsistent `seed=0` semantics until now.

Fixed at R20 as a standalone one-line commit (`if seed is None:`), on request, while R20
was already touching `GBManipulator.py` for other reasons. No caller in the existing test
suite passed `seed=0` to the constructor (only to `apply()`, which was already correct),
so nothing needed updating besides the fix itself; a new regression test
(`test_init_with_seed_zero_is_valid_and_deterministic`) covers it directly.

## `GBOpt.io.WriteResult`'s row-to-external-ID mapping is one array -- RESOLVED at R14

Confirmed at R14 (#75): `LammpsDataWriter.write()` always assigns
`np.arange(1, len(atoms) + 1)` as `WriteResult.atom_ids` (`GBOpt/io/lammps/data_writer.py`),
which is exactly the canonical-ID convention `CandidateFileMapping.__init__` already
requires (`GrainOwnership` rejects anything but consecutive `1..N` in row order). The
dense array is sufficient for `CandidateLoader`'s real consumption pattern -- no sparse or
inverse-lookup shape was needed, confirming the shape this entry originally flagged as
provisional.

`CandidateLoader.write_candidate()` (`GBOpt/CandidateLoader.py`, new at R14) now
demonstrates the pattern the issue's "Proposed behavior" describes end to end: it calls
`LammpsDataWriter` directly, and builds the returned `CandidateFileMapping` from that
call's real `WriteResult.atom_ids` rather than from an independently assumed
`np.arange`. Existing call sites that build a `CandidateFileMapping` *before* any file
exists (`_explicit_ownership_evaluation.py::_candidate_file_mapping`, called ahead of the
external evaluator callback that performs its own write) were deliberately left
unchanged: GBOpt does not control that write, so there is no real `WriteResult` available
at that call site to route through, and issue #75's own wording allows "WriteResult or an
equivalent explicit serialization result" -- the existing `np.arange(1, N+1)` there already
is that equivalent, verified byte-for-byte identical to what a real write would assign.
Rewiring `GBMinimizer.py`'s owned-checkpoint write (`GB.write_lammps`, which discards its
own internal `WriteResult`) to go through `CandidateLoader.write_candidate()` instead was
considered and deferred as a separate, riskier change touching checkpoint persistence and
composition-validation ordering unrelated to this entry's actual question (whether the
dense array shape suffices) -- no behavior gap motivates it, only a stylistic consistency
argument, so it wasn't made unilaterally per this file's own "don't fix pre-existing
findings as a drive-by" discipline extended to structural consistency, not just tool
findings.

## `GBOpt.FileGrainOwnership`'s ownership code still speaks orthogonal `box_dims` pairs, not `StructureData`'s cell/origin -- reconfirmed out of scope at R14

Re-checked against R14 (#75), whose own acceptance criteria are entirely about
consolidating *where* reload validation lives and closing the `GBManipulator` import
cycle, not about changing `CandidateFileMapping`/`FileGrainOwnership`'s internal
`box_dims`-pair representation. `CandidateLoader` (new at R14) builds on
`CandidateFileMapping` exactly as it was -- `_strict_box_dims`, `_remap_x_geometry`,
`CandidateFileMapping`'s box-bounds fields all still speak plain `(3, 2)` orthogonal
bounds, now consumed by `CandidateLoader.write_candidate()`/`reload()` in addition to
`FileGrainOwnership.reload_explicit_manipulator`, but unconverted. This still isn't "a
roadmap step that substantially revisits `GrainOwnership`/`FileGrainOwnership`'s own
internal representation" in the sense the original entry meant (changing
`CandidateFileMapping`'s public field shapes) -- R14 builds a consumer on top, it doesn't
touch the representation itself.

**Resolve at**: unchanged from the original entry -- whenever a roadmap step next
substantially revisits `GrainOwnership`/`FileGrainOwnership`'s own field shapes, or if/when
non-orthogonal cell support is added and `box_dims`'s orthogonal-only assumption needs to
go regardless.

## R14 built `CandidateLoader` as a new top-level module, not a subpackage

Issue #75 says "`CandidateLoader` (or equivalent higher-level I/O/domain service)"
without specifying its home. `GBOpt/CandidateLoader.py` was added as a new top-level
module (matching the naming and flat-file convention already used by
`GBManipulator.py`, `GBMaker.py`, `FileGrainOwnership.py`, `GrainOwnership.py`) rather
than a subpackage under the `types.py`-as-leaf convention CLAUDE.md documents for
`crystallography`/`artifacts`/`optimization`/`io`/`interface`. That convention applies to
subpackages that decompose one large module into several internally-layered files;
`CandidateLoader` is a single small service class with no internal submodule structure of
its own, so a subpackage would be over-structuring one file. It sits above
`FileGrainOwnership.py` in the dependency graph (imports `CandidateFileMapping`/
`GrainOwnershipError` from it, plus `GBManipulator` at module scope) and is the only
module in the ownership stack allowed to import `GBManipulator` at module scope --
verified with the stub-parent-package subprocess pattern (`tests/test_candidate_loader.py`)
that neither `FileGrainOwnership` nor `GBManipulator` import it back.

`FileGrainOwnership.reload_explicit_manipulator` is kept as the tested public entry point
(criterion 8: existing scalar/batch reload tests pass unchanged through it) but is now a
thin wrapper delegating to `CandidateLoader().reload()`, with a *local* (function-body)
import of `CandidateLoader` -- not `GBManipulator` -- replacing the previous local import.
This still satisfies issue #75's criterion 7 ("`FileGrainOwnership` no longer needs a
local import of `GBManipulator`") literally and in spirit: `FileGrainOwnership.py` never
references `GBManipulator` anywhere, at runtime or (outside a `TYPE_CHECKING` guard, which
carries no import-cycle risk since it never executes) for typing. The local import of
`CandidateLoader` itself is necessary, not a residual workaround: `CandidateLoader.py`
imports `FileGrainOwnership.py` at module scope for `CandidateFileMapping`, so the reverse
import at module scope would be directly circular (A imports B, B imports A) -- deferring
it to call time (long after both modules are fully loaded in any real usage) is the
correct fix, the same "defer to call time" mechanism the original workaround used, just
one layer up the dependency graph and pointed at a module that has no reason to import
`FileGrainOwnership` back.

## R13 collapsed Parent's three file-reading exception types to one, and two malformed-file tests turned out not to be malformed

Issue #74 (R13) asked to route `Parent`'s file-backed construction through
`GBOpt.io.lammps` readers instead of parsing LAMMPS syntax itself, while also requiring
"Existing Parent data/dump ... tests pass." These turned out to conflict: legacy
`Parent` parsing raised three distinct exception types depending on failure stage
(`ParentCorruptedFileError` for box-bounds/atoms-section problems,
`ParentFileMissingDataError` for missing columns/type mappings, `ParentValueError` for
unrecognized format), but `GBOpt.io.lammps`'s readers report every read failure as one
flat `LammpsDataError` -- there is no per-stage subclassing to translate back from.

Raised to the user explicitly rather than decided unilaterally (same fork shape as
R10's `_rebuild()` entry). Resolution, confirmed by the user: collapse to a single
exception (`ParentCorruptedFileError`, chosen since its existing docstring already
covered "an error occurs while reading a snapshot"; `ParentFileMissingDataError` is
kept importable for compatibility but Parent no longer raises it -- see both classes'
updated docstrings), on the condition that the reasoning for each behavior change is
made explicit rather than treated as incidental leniency.

Verifying against the actual test fixtures surfaced two cases where the readers are
also *more permissive* than legacy parsing -- not just differently exceptioned:

- `tests/inputs/file_with_invalid_box_bounds2.txt`: legacy `Parent` rejected it because
  its own box-bounds line-token-count check required an `ITEM: BOX BOUNDS` line to
  carry `pp pp pp`-style periodicity flags (6/8/9 tokens); this file's box-bounds line
  has none (3 tokens). Real LAMMPS dumps may omit these flags -- they are informational,
  not a required part of the format -- so `LammpsDumpReader` reading it successfully
  (with periodicity simply unknown) is the more correct behavior, not a defect. The
  legacy rejection was an artifact of that hand-rolled token-count check, not an actual
  format requirement.
- `tests/inputs/lammps_input_multiple_atom_types_wrong_num_types.txt`: legacy `Parent`
  rejected it because its "Atom Type Labels" section (1 entry) didn't match the
  declared atom-type count (3). But this file's `Atoms` section names species directly
  as strings (`Cu`/`Ni`/`Fe`), never by numeric type ID, so that label-count mismatch is
  irrelevant to resolving any atom's species unambiguously. `LammpsDataReader` only
  requires that any type ID actually used, and the number of distinct species present,
  not exceed the declared count -- both hold here.

Per the user's guidance ("define an expected format based on what LAMMPS can actually
do, rather than programming around potential corruption"), both behavior changes were
adopted as correct rather than worked around, and the two tests were rewritten to
assert successful reads (`test_box_bounds_without_periodicity_flags_reads_successfully`,
`test_atom_type_labels_partial_coverage_reads_successfully` in
`tests/test_gbmanipulator.py`), each with a comment explaining why the file is actually
valid. `test_unknown_file_type`/`test_file_too_short` were updated from
`ParentValueError` to `ParentCorruptedFileError` to match the single translated
exception; `test_read_lammps_input_errors`/`test_parent_snapshot_init_errors`'s other
cases were updated the same way with no behavior change (still rejected, just under the
unified exception type).

## R13 did not route GBMaker-backed Parent construction through `from_structure()`

Issue #74's "Proposed behavior" prose says to "Keep `Parent(filename, ...)` and
`GBMaker` source adaptation as compatibility paths that select/construct neutral
structure data first, then delegate [to `from_structure()`]." Only the file-backed path
was actually rearchitected this way. `__init_by_gbmaker` is untouched from before R13.

This is a deliberate deviation from the prose, not an oversight: `__init_by_gbmaker`
already gets fully-resolved GB geometry directly from the `GBMaker` instance
(`system.gb_plane_x`, `system.box_dims`, `system.normal_topology`, ...), which is exact
domain knowledge a generic `StructureData` cannot carry. `from_structure()`'s own
legacy-fallback inference path (used when `grain_ownership` is `None`) recomputes
`gb_plane_x` from the box x-midpoint -- routing the `GBMaker` path through it would
either silently lose precision (violating `test_parent_getters`'s exact
`gb_plane_x`-matches-source assertion) or require `from_structure()` to accept an
explicit-geometry override that would just reimplement `__init_by_gbmaker`'s existing
logic behind a different name. None of R13's actual acceptance criteria require this
(they're about file-backed construction specifically: "File-backed `Parent(...)` routes
through R11 readers," "Existing Parent data/dump and GBMaker-source tests pass"), so
per CLAUDE.md's "acceptance criteria win over conflicting prose" precedent, the
criteria were followed and this deviation is disclosed here rather than resolved
silently either way.

## R13 removed Parent's ability to read back a triclinic-written LAMMPS data file

Confirmed by a real, previously-passing test failure, not just a hypothesis: routing
`Parent`'s file-backed construction through `GBOpt.io.lammps.LammpsDataReader` means it
now refuses any LAMMPS data file containing an `xy xz yz` tilt line
(`"explicit ownership supports orthogonal LAMMPS data boxes only"`) -- this is not a new
restriction R13 introduced; R11/R12 already established that `GBOpt.io.lammps` readers
never support non-orthogonal cells (see the "no triclinic support (read side)" entry
above, resolved-at-R12 for the *write* side only). Legacy `Parent` parsing tolerated a
triclinic data file by mistake of implementation, not by design: it parsed the `xy xz
yz` line into an unused `self.__tilt` attribute and otherwise used plain orthogonal
`x_dims`/`y_dims`/`z_dims` for its own box bounds -- `Parent` has never actually used
tilt for anything.

Raised to the user explicitly, since this is a real capability loss (not just an
exception-type or leniency difference like the two cases in the entry above) affecting
a currently-passing acceptance test
(`tests/test_gbmaker.py::TestGBMakerTriclinic::test_triclinic_gbmanipulator_reads_back`).
Two options were on the table: keep a narrow Parent-side fallback that strips/ignores
the tilt line before re-reading (preserving the old capability but reintroducing a
sliver of file-syntax awareness into `Parent`, against this issue's core ask), or accept
the loss as inherited from R11/R12's already-established read-side contract and update
the test. **The user chose to accept the regression.** The test now asserts
`ParentCorruptedFileError` is raised instead of a clean read, with a comment explaining
why. If round-tripping a triclinic-written file through `Parent` is needed again later,
it requires teaching `GBOpt.io.lammps.LammpsDataReader`'s *read* side to tolerate
(not necessarily interpret) off-diagonal cell entries -- an `io.lammps` change, not a
`Parent`-side one, and out of scope for both R13 and the already-resolved R12 entry
above (which only added triclinic *write* support).

## R13 added one new bandit B101 (assert) finding, matching an existing precedent

`Parent.__init_from_owned_structure` asserts `structure.external_ids is not None`
before using it, since every `GBOpt.io.lammps` reader always populates `external_ids`
for LAMMPS data/dump formats -- the same invariant `GBOpt.io.lammps.compat`'s
`_to_lammps_atom_data` already asserts identically. This is a new bandit low-severity
B101 finding (2 -> 3 for `GBManipulator.py`/`FileGrainOwnership.py` combined), disclosed
per CLAUDE.md's tool-finding-diff discipline rather than silently accepted: it follows
the same established in-codebase pattern, not a new one.

## Three small pockets of real computation still lived directly in `GBMaker.py` after R10's first pass -- RESOLVED at R10

Resolved at R10, on request: all three were addressed.

- **`_normalize_vacuum_topology`**: moved to `GBOpt/gbmaker/dimension.py` (next to
  `_plan_box_dims`, the other small box/vacuum-geometry helper there), imported
  directly by `GBMaker.py` at both call sites, same pattern as `_find_commensurate_pair`.
  4 new direct unit tests added to `tests/test_gbmaker_dimension.py`.
- **`write_lammps`'s triclinic rotation-application block**: extracted as
  `_rotate_atoms_about_x` in `GBOpt/gbmaker/geometry.py`, next to
  `_triclinic_tilt_params` (the function that computes the `theta` it consumes).
  Returns a new array rather than mutating in place, preserving `write_lammps`'s
  existing "never mutate the caller's array" contract. 3 new direct unit tests added
  to `tests/test_gbmaker_geometry.py` (identity at `theta=0`, a quarter-turn's known
  y/z mapping, and a no-mutation check).
- **`GBMaker.get_supercell`**: removed outright, along with its dedicated test
  (`test_gbmaker.py::test_get_supercell`). Re-confirmed zero callers anywhere in the
  codebase (source, tests, and non-`.py` files) immediately before removing it, and no
  practical need for it to remain was found -- it was never part of the actual
  construction pipeline (exact/approximate grain builders enumerate sites internally
  via `GBOpt.gbmaker_supercell`, not through this method).

Verified: full non-slow suite passes (2693 passed, +6 net over the prior R10 checkpoint
-- +4 vacuum-topology tests, +3 rotation tests, -1 removed `get_supercell` test);
ruff/mypy/bandit/pyscn finding counts on every touched file are unchanged from the
prior R10 checkpoint.

## R10's `_config`/`_boundary`/`_result` state containers are facade-owned, not literal reuse of `gbmaker.types`'s pipeline dataclasses

Issue #71 (R10) says to "store canonical `_config`, `_boundary`, and `_result` state"
and "use validated dataclass replacement for setter updates." The natural first
instinct is to literally reuse `GBOpt.gbmaker.types`'s own pipeline dataclasses
(`GBBuildConfig`/`MaterialState` for `_config`, `BicrystalResult` for `_result`) rather
than inventing new ones. This was tried and deliberately abandoned for `_config`/
`_boundary`, for two concrete, verified reasons:

- **`MaterialState.a0` is strictly positive; `GBMaker`'s legacy `a0` setter is only
  non-negative (accepts `a0 == 0`).** This is the exact R04 divergence CLAUDE.md
  documents at length -- a *deliberate*, disclosed difference, not an oversight.
  Frozen dataclasses re-run `__post_init__` on every `dataclasses.replace(...)` call,
  with no way to bypass it for an already-validated value, so routing the `a0` setter
  through `MaterialState.replace(a0=...)` would silently reject `a0 == 0` where the
  setter has always accepted it -- a real regression against issue #71's "every
  existing setter... retains its current validation/rebuild semantics" criterion, not
  a hypothetical one.
- **`self.__radius` is set once at construction and never recomputed by any setter**,
  including `a0`'s and `structure`'s, even though both change quantities it depends on
  (`a0 * unit_cell.radius`). This is a genuine pre-existing staleness quirk, not
  something to "fix" as a drive-by. A derived/live-computed `radius` (the natural
  design if routing everything through the pipeline's own always-consistent
  dataclasses) would silently change this observable behavior.

Given these, `_config`/`_boundary` are new, facade-owned, **grouping-only** mutable
dataclasses (`_MakerConfig`/`_BoundaryState` in `GBOpt/GBMaker.py`) that perform no
independent validation of their own -- every field arrives already validated by
`GBMaker`'s own existing per-field validators, exactly as the flat instance attributes
they replace did. `_result` (`_AssembledResult`) is similarly a new small dataclass
rather than literally `BicrystalResult`: `BicrystalResult` is frozen and also carries
`box_dims`/`normal_topology`/`gb_id` (kept on `_boundary`/`_config` here instead, since
those are known before the very first bicrystal assembly ever runs, which would be an
awkward chicken-and-egg construction order against a frozen type), and
`vacuum_thickness`'s setter mutates the assembled atom arrays' contents in place
(`self._result.left_atoms["x"] += delta`) rather than reconstructing them, which a
frozen dataclass doesn't prevent (array mutation isn't attribute reassignment) but
which would be a strange fit for a type whose whole point is being replaced wholesale
on change.

This isn't a rejection of "validated dataclass replacement" as a concept -- the design
still centralizes what used to be ~36 flat, ungrouped private attributes into three
named containers, and getters/setters still delegate to them. It's a decision that the
validation for `_config`/`_boundary`'s fields stays exactly where it already lived
(`GBMaker.__validate` and its siblings, via `__translate_construction_error`), rather
than being re-delegated to `gbmaker.types`'s independently-evolving pipeline
dataclasses, which were designed for the *pipeline's* construction-time invariants, not
for preserving every quirk of the *facade's* already-established setter behavior.

## R10's state-model rewrite removed three more dead `GBMaker` geometry wrappers and ~40 name-mangled test call sites

Landed alongside the `_config`/`_boundary`/`_result` rewrite:

- `__scaled_periodic_basis_vector` and `__complete_origin_atom_mask` had zero internal
  callers left (the triclinic extraction earlier in R10 removed
  `__scaled_periodic_basis_vector`'s only caller; `__complete_origin_atom_mask`'s
  caller had apparently been gone since an earlier step) *and* zero name-mangled test
  callers -- confirmed by grep before deleting, per this file's own "grep both before
  deleting a now-dead wrapper" discipline. Removed outright.
- `__box_periodic_basis` had zero internal callers but one name-mangled test caller
  (`test_gbmaker.py`'s `test_triclinic_params_uses_grain_with_larger_y_period_norm`).
  Migrated that call site to `GBOpt.gbmaker.geometry._box_periodic_basis` directly
  (the real, unmangled function, called with the same arguments the wrapper used to
  supply from instance state) before removing the wrapper, per the established
  name-mangled-test-migration convention.
- Replacing ~36 flat private attributes with three grouped containers necessarily broke
  every other name-mangled test access to those attributes (`_GBMaker__a0`,
  `_GBMaker__left_x`, `_GBMaker__gb_region`, etc. -- about 40 call sites across
  `test_gbmaker.py` and `test_gbmaker_assembly.py`). Each was migrated to the
  corresponding `_config.*`/`_boundary.*`/`_result.*` path (e.g.
  `gb._GBMaker__left_x` -> `gb._boundary.left_x`). This is the extraction breaking
  them, not a separate cleanup pass -- same category as the geometry-kernel
  extractions' test migrations, just at facade-state-model scale. Method calls
  (`gb._GBMaker__get_triclinic_params()`, `gb._GBMaker__material_state()`) needed no
  change, since those methods themselves weren't moved or renamed, only the attributes
  they read.
- Giving `_MakerConfig`/`_BoundaryState`/`_AssembledResult`'s fields honest, non-`None`
  concrete defaults (empty arrays, `0.0`) instead of `None` placeholders eliminated 8 of
  mypy's pre-existing `GBMaker.py` findings as a side effect (every downstream
  `self._config.unit_cell.<attr>`-style `union-attr` error disappeared once `unit_cell`
  was typed as concrete `UnitCell` rather than the upstream `UnitCell | None` its
  source field allows for a different, not-yet-resolved use case). Per CLAUDE.md's "a
  finding count decrease needs the same explanation an increase would" rule: this was
  verified as a legitimate typing improvement I fully control (these fields are never
  actually `None` after `__init__` completes), not a cast or a suppressed check. One
  new explicit guard was added at the single point `config.material.unit_cell` (typed
  `UnitCell | None` upstream) flows into `_MakerConfig.unit_cell` (typed concrete
  `UnitCell`): `resolve_material_state` always attaches a `UnitCell`, so this raises a
  clear `GBMakerValueError` on an invariant violation that shouldn't be reachable,
  matching R08's precedent for this exact shape of boundary guard rather than a mypy-
  motivated cast.

## Several `GBMaker` setters change the system's identity, not just a parameter -- reconsider which properties should stay settable

While implementing R10's compatibility facade, it became clear the 9 settable
properties are not uniform: most (`interaction_distance`, `repeat_factor`, `x_dim_min`,
`misorientation`) trigger the full spacing/dims/GB-regeneration pipeline, but
`structure`'s setter only rebuilds `unit_cell` (no geometry regeneration at all), and
`vacuum_thickness`'s setter does an incremental atom-position shift instead of
regenerating from scratch. Several of these (`structure`, `a0`, `misorientation`) don't
just tweak a parameter of the existing bicrystal -- changing them produces what is, in
every physically meaningful sense, a different system (different crystal, different
misorientation) wearing the same `GBMaker` instance.

Raised during R10's design discussion: it may make more sense going forward to require
constructing a new `GBMaker`/`from_boundary_spec(...)` for identity-changing parameters,
rather than allowing them to be mutated in place via a setter, and to reserve settability
for parameters that are genuinely just tuning knobs on the same system (e.g.
`interaction_distance`, `epsilon`, `id`). Issue #71 explicitly scopes R10 to *preserve*
the mutable facade as-is (see the issue's "Alternatives considered": removing/restricting
it is deliberately deferred as "optional follow-on API work after the refactor
stabilizes"), so R10 keeps all 9 setters working exactly as before, special cases
included. This entry exists so that decision doesn't get silently forgotten once R10
ships.

**Resolve at**: a dedicated future roadmap step (post-R10, not yet numbered/scheduled)
that's explicitly scoped as an intentional, possibly-breaking API change -- deciding on
purpose which properties should remain settable in place vs. require constructing a new
instance, rather than letting the current "everything with a setter is settable" shape
persist by default.

## `GBMaker.__get_triclinic_params` extracted into `gbmaker.geometry._triclinic_tilt_params` -- RESOLVED at R10

Found while starting R10: `__get_triclinic_params` was still a real geometric
computation (grain selection by periodic-row norm, basis rotation, LAMMPS
restricted-triclinic tilt-factor/rotation-angle derivation) living directly in
`GBMaker.py`, not orchestration over an already-extracted pure function -- a direct
violation of #71's "`GBMaker.py` contains no canonical scientific construction
algorithm" acceptance criterion that R09's assembly extraction hadn't touched (R12's
entry on this method explicitly noted it as "unchanged").

Extracted as `_triclinic_tilt_params` (`GBOpt/gbmaker/geometry.py`), taking every input
explicitly (`inplane_periodic`, both grains' periodic Miller rows, `R_left`/`R_right`,
the shared `conventional_basis`, `y_dim`/`z_dim`, `epsilon`) rather than a `GBMaker`
instance, matching every other kernel in that module. `GBMaker.__get_triclinic_params`
is now a thin wrapper through `__translate_construction_error`, same pattern as its
siblings. One near-miss caught before landing: the extracted function's grain-selection
comparison originally used this module's own `_miller_row_norm` (attractive since it's
already imported here for other kernels) instead of the original `np.linalg.norm` --
but `_miller_row_norm` raises `GBMakerConstructionValueError` on any non-integer-dtype
row, which the periodic-Miller-row arrays are not guaranteed to be, so that swap risked
a new spurious exception on inputs the original silently handled. Reverted to
`np.linalg.norm` before landing. Verified byte-for-byte formula equivalence by reading
old and new side by side (not just running tests), plus 4 new direct unit tests in
`tests/test_gbmaker_geometry.py` (orthogonal-input zero-tilt case, the
not-periodic-raises error path, and one hand-verified-numeric test per grain-selection
branch) and the full non-slow suite passing unchanged.

Items identified during the R01-R30 architectural refactor (issue #61) that are real
issues or judgment calls, but were deliberately **not** addressed in the PR that found
them — because fixing them there would have exceeded that PR's stated scope, or because
they need a decision this file doesn't make on its own. Each entry names where the item
was found, why it's being deferred, and the natural point to resolve it. When an item is
resolved, remove its entry rather than leaving it marked done.

## `GBMaker.__validate`'s own `positive` parameter is still misleadingly named -- RESOLVED at R10

Resolved at R10: `GBMaker.__validate`'s `positive` keyword-only parameter was renamed to
`nonnegative` (matching `_validate_scalar`'s own already-correct name), along with its
9 current call sites (the ~20 figure in the original entry was from before R03-R09's
extraction shrank `GBMaker.py`). Purely mechanical -- no call site's actual
nonnegative-vs-strictly-positive semantics changed, confirmed by the full non-slow suite
passing unchanged before and after.

## `MaterialState.atom_types` validates more strictly than the legacy constructor ever did

The legacy `GBMaker.__init__` never validated `atom_types` itself — it passed the raw
value straight to `__init_unit_cell(atom_types)` -> `UnitCell.init_by_structure`, which
does its own type/count/species checking and raises `UnitCellTypeError`,
`UnitCellValueError`, or `AtomValueError` (all propagating unwrapped, confirmed by
`test_legacy_constructor_invalid_values_raise_exceptions`'s `atom_types="Invalid"`
case, which still expects a bare `AtomValueError`). `MaterialState.__post_init__`
(introduced in R03) now pre-validates `atom_types` as "a non-empty string or a tuple of
non-empty strings" and raises `GBMakerConstructionValueError` (translated to
`GBMakerValueError`) for anything that doesn't match that shape *before* `UnitCell` ever
sees it — e.g. `atom_types=123` would previously reach `UnitCell.init_by_structure` and
raise `UnitCellTypeError`; today it's rejected one layer earlier with a different
exception type and message. No test currently exercises this specific shape of invalid
input, so it hasn't surfaced as a failure the way `gb_id` did.

This may be a legitimate, desirable tightening (fail with a clearer error, one layer
earlier) rather than a bug, unlike the `positive=True` cases above — but it wasn't a
deliberate decision the way `a0`'s strict-positivity was; it was an assumption baked into
`MaterialState`'s R03 design before any legacy call site was checked against it.

**Resolve at**: whichever roadmap step next touches `MaterialState` or unit-cell
resolution directly (nothing currently scheduled specifically) — worth a deliberate
yes/no decision rather than leaving it as an unexamined side effect.

## Grain builders (R08) still call geometry kernels mid-computation, not composed beforehand -- RESOLVED at R08

R08 (issue #69) extracted `__build_exact_grain`/`__exact_grain_repeats` into
`GBOpt/gbmaker/exact_grain.py` (`build_exact_grain`, `_exact_grain_repeats`) and
`__generate_grain_result` into `GBOpt/gbmaker/approximate_grain.py`
(`build_approximate_grain`), plus the two `_FloatGrainBuildResult`-wrapping helpers
(`__filter_float_result_complete_origins`, `__trim_float_result_to_upper_x`) into
`approximate_grain.py` as `filter_grain_result_complete_origins`/
`trim_grain_result_to_upper_x`. Both new modules are leaves with respect to each other
(each imports only `gbmaker.geometry`/`gbmaker.types`, not one another), matching the
existing `geometry.py`/`orientation.py`/`dimension.py` layering. The kernels are still
called mid-computation, not composed beforehand -- that turned out to be inherent to
the algorithms (rotation/strain must happen before selection, and selection must
happen before clipping/deduplication), not an artifact of the previous file layout, so
no further restructuring was warranted.

The former `GBMaker.py`-local `_FloatGrainBuildResult` dataclass was collapsed into the
already-present (but previously unused) `gbmaker.types.GrainBuildResult` -- the two
were the same shape (`atoms`, `origin_ids`, `basis_size`) plus `GrainBuildResult`'s
`grain_side`, confirmed by re-reading both before merging them, per CLAUDE.md's
"decide deliberately" guidance rather than carrying both forward by default. The exact
path now also returns a `GrainBuildResult` (previously a bare `np.ndarray`); its
`origin_ids` assigns one id per contiguous `basis_size`-sized block of enumerated
sites (the established quotient-lattice origin order from
`gbmaker_supercell.enumerate_supercell_sites`) for contract uniformity, even though
the exact path never filters/clips/dedups by origin the way the approximate path does.

`GrainBuildRequest` (speculatively added in R03/R04) needed real field-shape
corrections once R08 wired it through actual call sites, per CLAUDE.md's "value types
aren't frozen" rule: its single `orientation` field (dtype conditional on `exact`)
conflated two distinct matrices both builders actually need simultaneously -- a proper
rotation matrix (`R_grain`, always float) and a canonical integer periodic-direction
matrix (`P`/`Q` on the exact path, the periodic Miller-row matrix on the approximate
path) -- so it was split into `rotation` and `periodic_matrix`. `box_dims` (a 3x2
bounds array) was never actually read by either builder; replaced with the fields they
do need: `x_offset`, `inplane_periodic`, `inplane_box_lengths`, `epsilon`, `y_scale`/
`z_scale`, and optional `y_repeats`/`z_repeats` (the exact path's mismatch-accommodation
repeat counts, when a caller supplies them explicitly instead of deriving them from box
length). `GrainBuildResult` needed no changes.

`__generate_grain_result`'s `grain_side: str | None = None` parameter (meaning "apply
no strain") was tightened to a required `str`: neither of its two call sites ever
passed `None`, and `GrainBuildRequest.grain_side` requires a concrete `"left"`/
`"right"` value, so preserving the `None` case would have meant carrying dead
flexibility through the new contract. Disclosed here rather than dropped silently.

`_miller_row_norm`'s and `_reduce_integer_row`'s R07-era duplication is now further
resolved: `_exact_grain_repeats` (in `exact_grain.py`) imports `_miller_row_norm` from
`geometry.py` as before; `build_approximate_grain`'s inline x-direction-lattice
reduction now imports `_reduce_integer_row` from `geometry.py` too (geometry is a leaf
relative to the new builder modules, same as it already is relative to
`orientation.py`/`dimension.py`), so `GBMaker.py`'s own `__reduce_integer_row` thin
wrapper (whose only caller was `__generate_grain_result`) is now dead and was removed.
`orientation.py`'s own `_reduce_integer_row` copy remains, for its own unrelated
internal use (row-rational approximation) -- still two copies, now geometry.py's and
orientation.py's, not three.

Several other `GBMaker.py` wrapper methods (`__selection_basis_vectors`,
`__reduced_box_coordinates`, `__reduced_coordinate_tolerance`,
`__cartesian_from_box_coordinates`, `__x_index_range`, `__filter_complete_origins`,
`__clip_complete_origins_to_cartesian_box`, `__deduplicate_complete_origins`,
`__select_complete_origins_in_box_basis`) had their only remaining caller removed by
this same extraction and were deleted as dead code. Four of them
(`__selection_basis_vectors`, `__reduced_box_coordinates`, `__reduced_coordinate_tolerance`,
`__cartesian_from_box_coordinates`) were still reachable from `tests/test_gbmaker.py`
via name-mangled access (`gb._GBMaker__selection_basis_vectors(...)`, used as
verification helpers in `TestGBMakerGenerateGrain`, not as the unit under test); per
CLAUDE.md's name-mangled-test-migration guidance, those call sites were migrated to
call `GBOpt.gbmaker.geometry`'s real unmangled functions directly instead of being left
broken. `__scaled_periodic_basis_vector` and `__complete_origin_atom_mask` were already
dead (zero callers) before this step and were left alone as pre-existing, out-of-scope
debt.

## Whether `GBMakerValueError`/`GBMakerTypeError` should eventually alias `GBMakerConstructionValueError`/`GBMakerConstructionTypeError` -- RESOLVED at R10

Decided at R10, on purpose, not by default: **stay independent; keep
`__translate_construction_error`.** `GBOpt/GBMinimizer.py:2951` does
`except (OSError, GBMakerError) as exc:` around a `self.GB.write_lammps(...)` call, a
real (non-test) reliance on `GBMakerValueError`/`GBMakerTypeError` both being subclasses
of `GBMakerError`. `GBMakerConstructionValueError`/`GBMakerConstructionTypeError`
subclass `GBMakerConstructionError`, not `GBMakerError`, and can't be made to without
reintroducing the exact circular import (`gbmaker/types.py` -> `GBMaker.py`) the
original translation wrapper exists to avoid. Aliasing `GBMakerValueError =
GBMakerConstructionValueError` would silently stop `GBMinimizer.py`'s except clause from
catching validation errors raised through the new pipeline -- a real behavior break, not
just a cosmetic one -- so the two hard constraints from the original entry (no circular
import, exception identities must not silently change) still hold at R10 and settle the
question the same way R04 provisionally chose. The translation wrapper is not
scaffolding to be removed later; it's the permanent boundary between `gbmaker`'s pure
construction-error hierarchy and `GBMaker`'s own public one.

## `gbmaker/geometry.py`'s complete-origin kernels have no direct unit tests

R07 (#68) moved `_complete_origin_atom_mask`, `_filter_complete_origins`,
`_deduplicate_complete_origins`, and the composed `_select_complete_origins_in_box_basis`
into `gbmaker/geometry.py` as thin-wrapper-backed pure functions, matching the other
extracted kernels. Unlike the other kernels moved in the same step (which got direct
`tests/test_gbmaker_geometry.py` coverage translated from the relocated
`_GBMaker__...`-mangled tests), these four had no pre-existing dedicated tests to
relocate — only `_clip_complete_origins_to_cartesian_box`'s epsilon-boundary test
existed standalone; the rest were exercised solely indirectly, through
`TestGBMakerGenerateGrain`/`TestGBMakerGenerateGB`'s full real-grain construction in
`tests/test_gbmaker.py`. That indirect coverage is real (every exact/approximate grain
built in the test suite exercises deduplication, complete-origin filtering, and
box-basis selection), but a defect isolated to one of these four functions specifically
would surface as a confusing downstream grain-shape assertion failure rather than a
targeted failure naming the function.

Not fixed here: writing thorough direct unit tests for four non-trivial array-shape
kernels (in particular `_select_complete_origins_in_box_basis`'s two code paths --
axis-aligned fast path vs. general mixed-basis path) is a real chunk of new test-writing
work, distinct from the mechanical "translate an existing test to call the unmangled
name" relocations CLAUDE.md's testing-conventions bullet describes, and risked
overrunning R07's actual scope.

R08 (#69) confirmed the parenthetical above: `build_approximate_grain` calls
`_select_complete_origins_in_box_basis` (and `_clip_complete_origins_to_cartesian_box`/
`_deduplicate_complete_origins`) directly now, imported straight from
`gbmaker.geometry` rather than through a `GBMaker` wrapper (the wrapper was removed as
dead code once its only caller moved out -- see the R08 entry above). This still didn't
add direct unit tests for these four kernels: R08's scope was the extraction itself
(moving `__generate_grain_result`'s orchestration, not adding new coverage for kernels
R07 already extracted unchanged), and the existing indirect coverage through real
grain construction continues to exercise all four via `test_gbmaker.py`.

**Resolve at**: as a standalone test-hygiene pass on request.

## `gbmaker/config.py` and `gbmaker/material.py` have no dedicated per-module test file

`CLAUDE.md`'s testing convention ("one test file per module, named
`test_<package>_<module>.py`") wasn't followed when R04 introduced `config.py`
(496 lines: `_validate_scalar`, `normalize_legacy_config`, `resolve_boundary_input`,
the `validate_*` wrappers) or `material.py` (`resolve_material_state`) — both are only
exercised indirectly today, through `GBMaker`'s public API in `test_gbmaker.py` and
`test_gbmaker_from_boundary_spec.py`. R05 followed the convention for the new
`gbmaker/orientation.py` (`tests/test_gbmaker_orientation.py`), which makes the gap in
`config.py`/`material.py` more visible as an inconsistency rather than the norm.

Not fixed here: retrofitting dedicated test files for two existing modules with no
behavior change is unrelated to R05's orientation extraction, and risks the same
"drive-by cleanup blurs what the PR did" problem `CLAUDE.md` warns against for
tooling-finding counts.

**Resolve at**: whichever roadmap step next substantially touches `config.py` or
`material.py` (no specific step scheduled), or as a standalone test-hygiene pass on
request.

## `GBMaker.__filter_float_result_complete_origins` has been dead code since R08 -- RESOLVED at R10

Resolved at R10: re-confirmed zero internal callers (`self.__filter_float_result_complete_origins(`
and the name-mangled test-access form both grep to nothing, including in
`tests/`) immediately before removing it, per this file's own "verify an entry's
call-graph claim against current code, don't trust the description" discipline -- the
method's shape hadn't changed since the R09 entry described it. Removed the method and
its now-solely-used-by-it imports (`GrainBuildResult` from `gbmaker.types`,
`filter_grain_result_complete_origins` from `gbmaker.approximate_grain`). Full non-slow
suite passes unchanged.

## `gbmaker/exact_grain.py` and `gbmaker/approximate_grain.py` have no dedicated per-module test file

Noticed while writing R09's (#70) `tests/test_gbmaker_assembly.py`: neither
`build_exact_grain` (R08, `exact_grain.py`) nor `build_approximate_grain` (R08,
`approximate_grain.py`) has a `test_gbmaker_exact_grain.py`/
`test_gbmaker_approximate_grain.py` file calling them directly, per CLAUDE.md's "one
test file per module" convention -- both are exercised only indirectly today, through
`GBMaker`'s real construction in `tests/test_gbmaker.py` and
`tests/test_gbmaker_exact_path.py`. Same shape of gap as the existing
`config.py`/`material.py` entry above, just for R08's two builder modules instead of
R04's two normalization modules.

Not fixed here: retrofitting dedicated test files for two existing modules with no
behavior change is unrelated to R09's assembly extraction.

**Resolve at**: whichever roadmap step next substantially touches `exact_grain.py` or
`approximate_grain.py` (no specific step scheduled), or as a standalone test-hygiene
pass on request.

## `GBOpt.io.StructureData.cell` is never populated off-diagonal (no triclinic support) -- RESOLVED at R12

R12 (#73) added real off-diagonal `cell` support on the *write* side:
`GBMaker.write_lammps()` now builds `cell` as LAMMPS box-vector rows
(`cell[1, 0]`/`cell[2, 0]`/`cell[2, 1]` carrying the `xy`/`xz`/`yz` tilt factors when
`triclinic=True`, computed by the still-`GBMaker`-private `__get_triclinic_params()`,
unchanged) and pre-rotates atom positions into the lab frame *before* constructing the
`StructureData` it hands to `LammpsDataWriter`. `StructureData.cell`'s general 3x3
validation (from R11) needed no changes -- it was already shaped correctly for this,
confirming R11's speculative choice not to assume diagonality was correct rather than
merely fortunate.

The *read* side is not part of this resolution: `LammpsDataReader`/`LammpsDumpReader`
still reject non-orthogonal boxes outright (unchanged from R11), so a `StructureData`
produced by reading a file is still always diagonal. Only writer-constructed
`StructureData` instances (built by `GBMaker.write_lammps()`, or directly by a caller of
`LammpsDataWriter`) can carry real tilt. Non-orthogonal *read* support remains
unscheduled.

## Import-boundary subprocess tests for `GBOpt.io.*` need a stub-parent-package workaround

CLAUDE.md's established pattern for "X does not import Y" acceptance criteria is
`subprocess.run([sys.executable, "-c", "import X; assert 'Y' not in sys.modules"])`,
to get a fresh interpreter with an empty `sys.modules` (R09, #70). R12 (#73) found this
literal pattern does not work for any "does not import `GBOpt.GBMaker`" check under
`GBOpt.io`: `GBOpt/__init__.py` itself unconditionally does `from GBOpt.GBMaker import
GBMaker` at module scope (it re-exports `GBMaker` as part of the package's public
surface), and Python always executes a package's `__init__.py` before any of its
submodules -- so `import GBOpt.io.lammps.data_writer` puts `GBOpt.GBMaker` into
`sys.modules` regardless of what `data_writer.py` itself imports. The check as written
would fail for *any* submodule of `GBOpt`, saying nothing about that submodule's own
import graph.

`tests/test_io_lammps_data_writer.py::test_data_writer_module_does_not_import_gbmaker`
works around this by stubbing `sys.modules["GBOpt"]` with an empty `types.ModuleType`
(with `__path__` populated from `importlib.util.find_spec("GBOpt")`, which locates the
package without executing it) before importing the submodule under test, so
`GBOpt/__init__.py`'s body never runs in that subprocess and only the submodule's own
imports are observed.

**Resolve at**: no action needed now, but any later roadmap step adding an import-
boundary test for a module under `GBOpt.io` (or anything else imported eagerly by
`GBOpt/__init__.py`) should reuse this stub-parent-package technique rather than the
plain R09 pattern, which will silently fail the same way.

## `LammpsDataReader.read`/`LammpsDumpReader.read` are still long, complex single methods

`pyscn check` flags both (`LammpsDataReader.read`: 40 cyclomatic complexity, 174 SLOC;
`LammpsDumpReader.read`: 29 complexity, 133 SLOC) as too-long/too-complex -- the same
shape of finding the pre-R11 combined `read_lammps_data_file`/`read_lammps_dump_file`
functions already had (43/167 and 32/135 respectively, checked against the
pre-refactor file in isolation), so this is relocated pre-existing debt, not new debt
R11 introduced, and decomposing either method's internal header-parsing/row-parsing
control flow further was out of R11's "preserve current parsing semantics exactly,
move don't rewrite" scope. `_resolve_type_id_species` (in `GBOpt.io.lammps.tokens`)
was extracted during R11 specifically because it was genuinely duplicated between the
two readers (an 85% pyscn clone finding) and both readers now call the same shared
helper for it -- a real complexity reduction, not just line movement -- but each
reader's header/section parsing and row loop remain large single methods.

**Resolve at**: a standalone `io.lammps` decomposition pass on request, or whichever
later roadmap step (R12/R13's writer work, or a future non-LAMMPS reader) next
substantially touches this package and has a concrete reason to split these methods
further.

## 9 of the 10 documented "baseline noise" test failures share one fixable root cause; the 10th isn't the same failure at all -- 9 RESOLVED at R12

While verifying R11 (#72) reproduced the documented 10-failure baseline exactly, each
of the 10 was actually reproduced individually rather than trusted from the CLAUDE.md
description (per this file's own "treat a description as a hypothesis to verify, not
an established fact" discipline). That surfaced two corrections, now folded into
CLAUDE.md's baseline-noise bullet in "Refactor-issue discipline":

- `test_checkpoint.py::test_json_serialization_normalizes_supported_values[path]` is
  not a `PermissionError` at all -- it's an `AssertionError` from a hardcoded POSIX
  path string (`'/some/dump.data'`) compared against `str(Path("/some/dump.data"))`'s
  Windows rendering (`'\some\dump.data'`), unrelated to file writes or locking.
  Nothing to fix in `GBOpt` for this one; it's a test assertion that was never
  cross-platform, most naturally fixed by asserting `Path(expected).as_posix()`-style
  normalization or by parametrizing the expected value per-OS.
- The other 9 (`test_gbmaker.py`'s `test_lammps_file_formatting`,
  `test_lammps_file_formatting_with_charge`, `test_write_lammps`,
  `TestGBMakerTriclinic::test_non_triclinic_no_tilt_line`,
  `test_triclinic_params_uses_grain_with_larger_y_period_norm` (both subtests),
  `test_triclinic_writes_tilt_line`; `test_gbmanipulator.py`'s
  `test_type_preservation_with_numeric_roundtrip`,
  `test_write_lammps_after_manipulate`) are all the same real `PermissionError` at
  `GBMaker.py:3328` (`open(file_name, "w")` inside `write_lammps`). Every one of these
  tests opens `tempfile.NamedTemporaryFile(delete=True)` as a context manager and
  passes `temp_file.name` to `write_lammps`, which reopens that same path for writing
  while the original handle is still open -- Windows denies the second open (no
  default file sharing), POSIX allows it. This is a test-fixture bug, not a
  `GBMaker`/`GBManipulator` bug: the fix is already used correctly once in the same
  file, by `test_csl_boundary_zero_tilt`'s helper around line 2853
  (`NamedTemporaryFile(delete=False, ...)`, closing the handle before `write_lammps`
  reopens it, `finally: os.unlink(fname)`). Applying that same pattern to the other 9
  call sites would make all 9 pass on Windows.

Not fixed at R11: this was unrelated to R11's read-only scope (`GBOpt.io` added readers
only; these failures are all in the write path), and touching `write_lammps`'s test
suite was unrelated to the structure-I/O extraction.

Fixed at R12 (#73): R12's writer extraction touches these exact test files
(`tests/test_gbmaker.py`, `tests/test_gbmanipulator.py`) anyway, so per this file's own
"don't leave it undisturbed by accident if you're touching those tests anyway"
discipline, all 9 `NamedTemporaryFile(delete=True)`-and-reopen call sites were swapped
for the `delete=False`-close-then-reopen-then-`finally: os.unlink` pattern. Verified by
re-running the full non-slow suite before and after: the only remaining failure is the
10th, unrelated `test_checkpoint.py` `AssertionError` (still not fixed here -- it's
genuinely out of scope for a LAMMPS-writer PR, and remains a standalone test-hygiene
item on request).

**Resolve at**: done (R12). The 10th failure (`test_checkpoint.py`'s POSIX-path
`AssertionError`) remains open as a standalone test-hygiene item on request.

## `GBMaker.write_lammps()` now raises `GBMakerValueError` (not silent garbage output) for non-finite `box_sizes`

Routing `write_lammps()`'s atoms/`box_sizes` through `GBOpt.io.types.StructureData`
construction (R12, #73) means `StructureData.__init__`'s existing finite-3x3/finite-
length-3 validation now runs on every call, translated to `GBMakerValueError` at the
writer boundary. Previously, a non-finite or malformed `box_sizes` (e.g. containing
`nan`/`inf`, or the wrong shape) would either crash with an unrelated `IndexError`/
`TypeError` deep in the old inline formatting code or, for `nan`/`inf` specifically,
silently write a garbage header line (`f"{nan}"` formats without error) -- neither path
raised a clear, typed error. No existing test exercised this shape of invalid input
(confirmed by the full non-slow suite passing unchanged before and after), so this is a
minor, incidental behavior tightening rather than a deliberate design decision the way
`a0`'s strict-positivity was -- flagged here per CLAUDE.md's "don't trust that a
behavior-preserving refactor didn't quietly become stricter" discipline, not because it
needs undoing.

**Resolve at**: no action needed; noted for the record. If a future step wants
`write_lammps()`'s error surface for malformed geometry to be a deliberate contract
(rather than an incidental side effect of using `StructureData`), decide then.
