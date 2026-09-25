# CLAUDE.md

Guidance for working in this repository, distilled from the R01 (#62,
`GBMinimizer` decomposition) refactor PR and its review. Applies to the rest of
the refactor roadmap (issue #61, implementation issues #62-#90 = R01-R30) as
much as to one-off work.

`REFACTOR_CLEANUP.md` (repo root) is the backlog of issues and judgment calls
identified while working a roadmap step but deliberately deferred rather than
fixed in that step's PR, each tagged with the roadmap step where it's expected
to be resolved. Check it before starting a roadmap step for anything tagged to
resolve there, and add to it — rather than fixing unilaterally or silently
letting it drop — whenever you notice a real issue that's out of the current
PR's scope per the "Refactor-issue discipline" rules below.

Both this file and `REFACTOR_CLEANUP.md` were originally untracked (this file
via the user's global git ignore, `REFACTOR_CLEANUP.md` via this repo's
`.git/info/exclude`) on the assumption they were local-only working notes for
sessions running directly in the user's own checkout. As of the R21 branch tip,
both are tracked instead: a cloud-run session gets a fresh clone with none of
the working directory's untracked state or ignore rules, so an untracked file
is invisible to it no matter how established the convention is locally. If a
future session (local or cloud) finds either file back to being untracked
again, don't assume that's a mistake to silently fix — the two setups have
genuinely different needs, and which one is current is itself worth
confirming with the user before proceeding as if either file's guidance is
visible to whatever session picks up the next roadmap step.

An entry's description of *what the code currently does* can go stale, not
just its line-number references — R07 found a `REFACTOR_CLEANUP.md` entry
claiming a specific method called a specific helper for its `d1`/`d2`
computation, written accurately when the entry was added, but by the time R07
started that method had been rewritten (during an earlier step, incidentally,
not maliciously) to call something else entirely. Treat an entry's call-graph
or behavior claims as a hypothesis to verify against the current code — the
same "re-grep every call site, don't trust the description" discipline this
file already asks for when mapping legacy validators — not as an established
fact just because it's tagged for resolution at this step.

## Code comments and docstrings

Comments and docstrings describe only what is in the file: the current contract,
behavior, and any non-obvious *why* behind it. They do not reference GitHub issues or
PR numbers, this roadmap's step names (R01-R30), `REFACTOR_CLEANUP.md` entries, or a
"legacy"/"pre-#NN" framing that describes a previous implementation instead of the
current one. That kind of provenance belongs in the commit message, the PR
description, or `REFACTOR_CLEANUP.md` — not in code, where it rots as soon as the
history it references is no longer the most recent thing that happened, and reads as
noise to anyone who opens the file without this roadmap's context. Write "delegates to
X, translating its exceptions to Y" rather than "matches the pattern from R16" or
"unchanged since #78" — say what the code does and relate it to other *code*, not to
the issue or step that produced it.

## Roadmap context

- Issue #61 is the coordination issue for a staged architectural refactor
  (R01-R30). Each roadmap item has its own implementation issue
  (`#62`-`#90`, plus `#27` as R17) with its own acceptance criteria and
  non-goals — read the specific issue, not just the roadmap summary, before
  starting.
- **Issues and pull requests are disabled on this fork (`jarinfrench/GBOpt`) —
  `#61`/`#62`-`#90` live on the upstream repo, `IdahoLabResearch/GBOpt`, not
  here.** Starting R22 in a fresh Claude Code on the web session, `GET
  /repos/jarinfrench/gbopt/issues/82` returned 404 and `list_issues`/
  `list_pull_requests` both returned empty for the whole fork — not a
  permission error, just genuinely nothing there, consistent with GitHub
  disabling Issues/PRs by default on a fork (this repo's history includes an
  "Add daily workflow to sync fork main with upstream" commit). The session's
  GitHub tool access was scoped only to `jarinfrench/gbopt`, and attaching
  `IdahoLabResearch/GBOpt` too failed on a same-basename checkout collision
  (`gbopt` already occupied) — spinning up a separate helper cloud session to
  fetch the issue text also didn't pan out, since cross-session messaging
  tools don't reliably bridge two independent cloud containers. The working
  fix was simply asking the user to paste issue #82/#61's text directly. Any
  cloud session starting a new roadmap step should expect this and either ask
  the user for the issue text up front or get `IdahoLabResearch/GBOpt`
  attached as this session's *only* GitHub-scoped repo (not alongside a
  same-named fork checkout).
- **A cloud session's checkout only has what's actually been pushed to
  `origin` — verify with `git ls-remote`/`git branch -a` before trusting any
  inherited summary of prior roadmap branch state, including claims about
  this file's own tracked status.** Starting R22, a detailed prior-session
  summary described `refactor/r11-structure-io` through `r21-optimizer-
  logging` as already existing and merged, and described `CLAUDE.md`/
  `REFACTOR_CLEANUP.md` in the "untracked, local-only" terms this file used
  to use (see the git-mechanics note below on that specific claim now being
  stale) — none of which existed in the fresh cloud clone (`git ls-remote`
  showed only `refactor/r01-...`; no issues at all on the fork). The
  branches, and these two files, turned out to be real but simply not yet
  pushed from the user's local machine to `origin`; asking the user to push
  them resolved it in minutes. The general check: when a task description's
  claims about branch topology or tracked files don't match a fresh
  `git ls-remote origin`/`git ls-files`, don't assume the description is
  fabricated or proceed around the mismatch — say what doesn't match and ask
  whether the missing state needs to be pushed first.
- PR branches are based on `feature/artifact-retention` (PR #92), not `main`,
  until that PR merges. **Branch topology depends on whether the issue lists
  a roadmap prerequisite.** R01 and R02 have no prerequisite between them
  (R02's issue doesn't list R01), so they're siblings both branched directly
  from `feature/artifact-retention` — confirmed via `git merge-base`, since
  neither branch contains the other's commits. R04's issue explicitly lists
  "Roadmap prerequisite: R03. #64", so `refactor/r04-...` branches from
  `refactor/r03-...` instead, to get R03's contracts (`GBBuildConfig`,
  `MaterialState`, ...). Check the target issue's "Dependencies and related
  issues" section before creating a branch — don't assume adjacency in the
  roadmap list implies a branch dependency, and don't assume no-prerequisite
  implies you can skip checking.
- **When an issue lists two prerequisites that are themselves sibling branches
  (neither contains the other's commits), create a new branch merging both,
  rather than picking one and hoping.** R14 (#75) listed R12 (`refactor/r12-
  lammps-writer`) and R13 (`refactor/r13-parent-from-structure`) as
  prerequisites; both branched from R11 and neither was an ancestor of the
  other (confirmed via `git merge-base --is-ancestor` both directions). The
  recipe: `git checkout -b refactor/r14-... refactor/r12-lammps-writer`, then
  `git merge --no-commit --no-ff refactor/r13-parent-from-structure` to
  inspect the merge before committing it. Pick the base branch (not the one
  merged in) by which one is more foundational to what the new step needs, or
  it genuinely doesn't matter when the diffs don't overlap in files (R14's
  didn't, beyond two shared test files, and merged with zero conflicts).
  **Don't generalize this into branching every roadmap step off whatever the
  most recent branch happens to be** — R15 (#76) lists only R02/R13 as
  prerequisites, not R14, so it branches from `refactor/r13-parent-from-
  structure` directly even though R14 is the most recently completed step;
  check each issue's own "Dependencies and related issues" section every
  time, per the rule above.
- **The roadmap's linear R01-R30 numbering does not imply a single linear
  branch chain — verify actual ancestry with `git merge-base`, don't infer it
  from adjacent numbers or a step's own narrative history in this file.**
  While starting R15, `git merge-base --is-ancestor` checks showed
  `refactor/r13-parent-from-structure` (R15's actual prerequisite branch)
  contains R02 and R11 but *not* R01, R03, R08, R09, R10, or R12 — the
  GBMaker-decomposition track (R03-R10, R12) and the Structure-I/O track (R11,
  R13, and now R15) are two independent branch lineages that only share
  R01/R02's common root, despite this file's own accumulated history
  describing R03-R12's `gbmaker` subpackage lessons at length just above.
  Neither track is "behind" the other; they're parallel, and a step whose
  issue lists only one track's branch as a prerequisite (R15 -> R13, not R12)
  simply never receives the other track's commits until something later
  explicitly merges both (the way R14 merged R12 and R13). Confirm the actual
  DAG for every new step with `git merge-base --is-ancestor` rather than
  assuming continuity with whatever roadmap narrative you just finished
  reading in this file.
- **`git log --all --grep="#N"` is not a reliable way to check for existing work
  on issue #N — a GitHub issue number can be reused across an issue's lifetime,
  and an old commit closing a since-closed, unrelated issue will match the same
  grep.** Starting R17 (#27), grepping for `#27` surfaced two old commits ("Added
  capability to specify repeat factors in y and z independently. Closes #27") for
  a completely different, already-shipped feature that happened to reuse issue
  number 27 at some point in the tracker's history — not #27's actual current
  title ("Simplify displace_along_soft_modes..."). The reliable check is
  content-based: `git branch -a` for a branch name/log matching the issue's real
  subject matter, and `git log --all -S "<a symbol/parameter the issue's contract
  would introduce>"` (e.g. `-S "mode_index"`) to catch any commit that touched
  that shape of change anywhere in history, not just ones that happened to
  mention the issue number in a commit message.
- **When a roadmap step's exact target code path has known, already-fixed bugs
  sitting on branches outside the roadmap's own branch DAG (not a roadmap
  prerequisite, not listed in any issue's "Dependencies" section), surface this
  to the user explicitly before choosing whether to merge them in, same as any
  other "real fork, not a mechanical judgment call."** R17 (#27) was about to
  refactor `displace_along_soft_modes`, and `origin/fix/soft_mode_fix1/fix2/fix3`
  (stacked, branched from `feature/artifact-retention`'s tip, not ancestors of
  any refactor branch) fix four real physics bugs in exactly that method's
  dynamical-matrix/q-point/neighbor-search code. Nothing in issue #27 mentions
  these branches — they wouldn't surface from reading the issue alone, only from
  independently checking what else touches the same file/method before starting.
  Refactoring the buggy version and merging the fixes in later would mean a
  second, avoidable pass over the same code. The user chose to merge the fix
  branch in as R17's own first commit, before making the API change — but the
  choice (merge now vs. defer vs. branch from the fix tip directly) was
  genuinely the user's to make, not something to infer from the roadmap
  structure.
- **A user's recollection that "several branches did use X as a base" is a
  hypothesis to verify against `git merge-base --is-ancestor`/`git branch -a
  --contains`, not a scope estimate to act on directly — the actual footprint
  can be much narrower than remembered.** Starting R20, the user believed
  several branches had picked up `refactor/r01-gbminimizer-decompose`'s
  mistaken merge of `tooling/lint-typecheck-experiment` (ruff/mypy/bandit/pyscn
  configs merged directly into a roadmap branch, contrary to the "Tooling"
  section below). Checking every branch (`git branch -a --contains
  <commit>`, plus `git merge-base --is-ancestor refactor/r01-... <branch>` for
  every branch in the repo) found only R01 itself and R20 (which had just
  merged R01) — no other roadmap branch descended from R01 at all, matching
  this file's own documented R03-R19 track independence. Fixing the actual,
  narrower footprint took a non-destructive `git rebase --onto
  <good-commit> <last-bad-commit>` on R01 (dropping the offending merge and its
  downstream VS Code wiring commit while preserving legitimate commits layered
  on top, resolving one incidental conflict where a later commit's
  format-on-save diff touched the same lines), then rebuilding R20's merge from
  the corrected R01. Always run the actual check before scoping a cleanup to
  match a stated belief, even when the belief comes from the user themself —
  it can be the "roadmap's linear numbering does not imply a linear branch
  chain" trap in a different guise.
- **A harness-created branch for a new roadmap step is not guaranteed to start from
  the right point in the DAG -- verify with `git rev-parse`/`git merge-base`, don't
  assume it was pre-built correctly.** Starting R23, the harness had already created
  `claude/mc-ga-eval-flows-f9cvnk`, but `git rev-parse` showed it was identical to
  `main`'s tip -- zero refactor-roadmap commits at all, not even R22's own branch tip.
  The fix was the same recipe as R14's sibling-prerequisite merge: `git checkout -B
  <branch> origin/<real-prerequisite-branch>`, confirm with `git rev-parse`/`git log`
  that the recreated branch now actually contains the expected lineage, then proceed.
  Don't infer a branch's contents from its name or its existence on `origin` --
  confirm the actual commit it points to before building anything on top of it.
- **When an issue lists multiple roadmap prerequisites and one is missing from the
  branch you're building on, check the issue's actual acceptance criteria for a
  literal dependency on that prerequisite's output before deciding whether to merge it
  in -- don't assume "listed as a prerequisite" alone settles it, and don't defer
  without checking either.** R23's issue (#83) listed R22, R14, and R20 as
  prerequisites; R22's branch already contained R20 but not R14. R14's own
  `CandidateLoader` module turned out to be named explicitly in #83's acceptance
  criteria ("Returned structures are validated through the authoritative
  `CandidateLoader`"), confirming the merge was actually required, not just nominally
  listed -- resolved via the same `git checkout -B <new> <base>` +
  `git merge --no-commit --no-ff <other-prerequisite>` recipe R14 itself used,
  verified with a full test run on the merged tip before committing.
- **Before reporting a tooling delta in a commit message, re-check every file the
  step actually touched -- verifying only the file you most recently edited (e.g. a
  test file, after fixing one specific finding in it) and generalizing that to "net
  0" for the whole change is a real mistake, not just an omission.** R23's "legacy GA
  scalar/batch path" commit message claimed "ruff... net 0" after confirming the test
  file's count was unchanged, but never re-checked `genetic.py` itself, which had
  genuinely gained 3 findings (2 `BLE001`, 1 `B905`) from a new recovery boundary
  added in the same commit. Caught only by a separate final full-repo baseline
  comparison against the branch point, after the commit had already been pushed;
  fixed with a follow-up commit recording the accurate, fully-reconciled delta rather
  than silently letting the inaccurate one stand. Run the exact per-file comparison
  for every file a commit touches, not just the one most recently in view.
- **Restoring the tooling config files (`pyproject.toml`/`mypy.ini`/`.pyscn.toml`)
  from `tooling/lint-typecheck-experiment` before a baseline run is not a one-time
  setup step -- it has to happen before *every* `pyscn` invocation specifically,
  since `pyscn` (unlike `ruff`/`mypy`) silently falls back to very different default
  thresholds when `.pyscn.toml` is missing, rather than erroring.** After deleting
  the three config files per this file's own "delete before staging" rule at the end
  of an earlier phase, a later full-repo `pyscn check` (re-copying only
  `pyproject.toml`/`mypy.ini`, forgetting `.pyscn.toml`) reported 81 quality issues
  against a baseline of 41 -- a seemingly large regression that was actually pyscn's
  default complexity threshold (10) silently replacing the repo's configured one
  (20), not a real code-quality change. Restoring `.pyscn.toml` and re-running
  reproduced the baseline's exact 41/46 counts. Don't read a large tool-count jump as
  a real regression before confirming the same config file was actually in place for
  both runs being compared.
  **Correction from R24:** the config file being genuinely present and correctly
  configured is not sufficient either -- `pyscn` resolves `.pyscn.toml` by searching
  upward from the *target file's own path*, not from the current working directory, so
  checking a pre-change baseline copy into a location outside the repo tree (e.g.
  `/tmp/baseline/monte_carlo.py`, copied there via `git show HEAD:... > /tmp/...` to
  avoid disturbing the working tree) silently falls back to `pyscn`'s hardcoded
  defaults for that one invocation, with no warning distinguishing it from the
  in-repo run that *did* find the real config -- both commands "succeed," but only one
  used the configured thresholds (confirmed by comparing `pyscn check
  /tmp/baseline/monte_carlo.py`'s complexity threshold, 10, against `pyscn check
  GBOpt/optimization/monte_carlo.py`'s, 20, on the exact same underlying file
  contents). Put every baseline comparison file inside the repo tree (a scratch
  subdirectory such as `.baseline_scratch/`, deleted before staging) rather than in
  `/tmp` or any other path outside it, so `.pyscn.toml`'s upward search finds the same
  config for both sides of the comparison.
- **A recovery boundary an earlier roadmap step's `CLAUDE.md` entry documented as
  missing (not a bug, just an asymmetry noted for "whichever later step's own
  acceptance criteria require it") should be added the moment a later step's
  acceptance criteria actually do require it, unconditionally -- not treated as
  optional scope just because it wasn't the step's main subject.** R21 documented
  that `MonteCarloMinimizer`'s evaluator call has no recovery boundary at all (unlike
  `GeneticAlgorithmMinimizer`'s established `except Exception` -> penalty pattern) and
  explicitly left resolving it to a later step. R23's "every MC/GA evaluation
  produces an `EvaluationResult`" and "evaluator exceptions... retain typed failure
  provenance" criteria require exactly this, read literally, so it was added at both
  of MC's evaluation call sites (initial: raises, matching GA-owned's own established
  fatal-initial-failure precedent; proposal: deterministically rejected, matching
  GA's per-candidate recovery-boundary pattern) -- disclosed as a real, intentional
  behavior change in `REFACTOR_CLEANUP.md`, with dedicated regression tests, rather
  than left as a subject for yet another future step.
- **A `REFACTOR_CLEANUP.md` entry's "no action needed unless a later step's issue
  explicitly asks for broader lifecycle instrumentation" is exactly the kind of thing
  to re-check against a new step's real acceptance criteria, not assume still
  deferred.** R21 scoped its logging instrumentation narrowly (only MC's termination
  `print()` calls and GA's three evaluator/reload-failure warnings), explicitly noting
  that broader "run start, initial evaluation, best updates, generation summaries"
  instrumentation the issue's own prose mentioned was left for a future step to decide
  it actually needs. R24 (#84, "introduce one versioned event vocabulary for MC and GA
  lifecycle reporting... silent by default") is exactly that future step -- its
  acceptance criteria literally ask for start/initial-evaluation/proposal/accept-
  reject/best-update/generation-boundary/reseed/termination/failure emissions, the
  same list R21's prose gestured at and its own criteria didn't require. Confirmed by
  re-reading R21's `REFACTOR_CLEANUP.md` entry before starting, not assumed from the
  roadmap summary alone; R24 does not touch R21's own narrower logging calls
  (`logger.info`/`logger.warning`), which remain as R21 left them -- it adds a
  separate, parallel event-emission mechanism, not a replacement for logging.
- **An earlier step's `REFACTOR_CLEANUP.md` entry gesturing at "a later step" needing
  a checkpoint-serialization form is not automatically that later step, even when the
  later step is squarely about the same subsystem -- check the later issue's actual
  acceptance criteria for what it says about checkpoints specifically.** R22's/R23's
  entries both flagged "no action needed unless a later step... needs `EvaluationResult`
  a checkpoint-serialization form," naming R23/R26+ as candidates. R24 is about MC/GA
  the same evaluation subsystem, but its own acceptance criteria explicitly wall events
  off from checkpoints in both directions ("checkpoint files are not used as event
  logs and events are not accepted as restart state") -- the literal opposite of a
  request to add checkpoint-serialization support to `EvaluationResult`. Confirmed by
  reading the criterion text itself rather than assuming "touches the same objects" is
  enough to resolve a vaguely-worded earlier deferral.
- **A harness-created branch for a new roadmap step reproduced the exact "identical to
  `main`'s tip" failure mode documented for R23, on the very next step.** Starting R24,
  the harness had created `claude/affectionate-darwin-oljxid`; `git rev-parse` showed it
  matched `origin/main` exactly, zero roadmap commits. Same fix as R23's own entry
  describes (`git checkout -B <branch> origin/<real-prerequisite-branch>`, verified with
  `git rev-parse`/`git log` before building anything on top of it) -- recorded here only
  to confirm this is a recurring harness behavior worth checking every single time, not
  a one-off from R23 specifically.
- **A harness-created branch for a new roadmap step reproduced the exact "identical to
  `main`'s tip" failure mode for the third step in a row.** Starting R25, the harness had
  created `claude/hopeful-allen-35k8q2`; `git rev-parse` showed it matched `origin/main`
  exactly, zero roadmap commits -- same as R23's and R24's own entries above. Same fix:
  `git checkout -B refactor/r25-event-journal origin/refactor/r24-event-vocabulary`,
  verified with `git rev-parse`/`git log` before building anything on top of it. This is
  not a one-off pattern to special-case around; expect it on every future step and check
  every single time before trusting a harness-assigned branch's contents.
- **R25's issue (#85) lists only R24 as a prerequisite, and R24's own branch already
  contains everything R25 needed -- no merge of a second prerequisite branch was
  required, unlike R14/R23.** Confirmed with `git merge-base --is-ancestor
  refactor/r23-mc-ga-evaluation refactor/r24-event-vocabulary` (true) before branching,
  so R25 branches directly from `refactor/r24-event-vocabulary`, carrying R23/R22/R20/R14
  transitively. Don't assume this is now the norm -- check every step's own "Dependencies
  and related issues" section per the standing rule above; this step simply happened to
  have one clean prerequisite.
- **R25 is the checkpoint-serialization step R22's/R23's `REFACTOR_CLEANUP.md` entries
  floated for "R23 or R26+" -- confirmed by rereading #85's actual acceptance criteria,
  and the answer is no.** Issue #85 ("provide opt-in durable scientific provenance
  through a versioned JSONL event journal plus a separate run manifest, explicitly
  distinct from checkpoint/restart files") explicitly requires the opposite: "Documentation
  states that journals cannot be used as checkpoints," matching R24's own criteria walling
  events off from checkpoint state in both directions. `EvaluationResult`/
  `CandidateEvaluation` still have no `to_state`/`from_state`; that gap remains open for
  whichever later step actually is the checkpoint-serialization migration.
- **None of R24's four new `REFACTOR_CLEANUP.md` entries (`operation_parameters` always
  `None`, GA accept/reject meaning next-generation selection, legacy `run_GA`'s
  `RUN_FAILED`-only try/except, `_evaluate_generation`'s reconstruction-failure accuracy
  fix) needed any action at R25.** Rechecked each against #85's real acceptance criteria
  (durable storage of the existing event schema, not a schema change) before assuming so
  -- #85 is silent on all four, so each remains open for whichever step's own criteria
  actually touch it.
- **A new durable-storage sink for an established `EventSink` protocol needs no changes
  to the classes that already emit through that protocol.** R25's `JsonlEventSink` plugs
  into `MonteCarloMinimizer`/`GeneticAlgorithmMinimizer` exactly like `NullEventSink`/
  `LoggingEventSink`/`CompositeEventSink` already do (R24) -- neither minimizer's own code
  changed. This is what "default library behavior remains unchanged unless a journal sink
  is configured" (#85's own acceptance criterion) means mechanically: the default stays
  `NullEventSink` because nothing wires `JsonlEventSink` in by default, not because of any
  new guard condition.
- Before opening a PR, run `ruff`, `mypy`, `bandit`, and `pyscn` (see
  "Tooling" below) and report the results.
- **Never open a PR without being explicitly told to.** The user reviews
  every change locally first (diffs, tests, tool output) before anything is
  proposed on GitHub. A branch being pushed, tests passing, and tooling being
  clean are not sufficient signals to open a PR on their own — treat
  "implement/commit this" and "open a PR for this" as two separate requests,
  and wait for the second one explicitly.

## Subpackage architecture conventions

Established by `GBOpt/crystallography/` and `GBOpt/artifacts/`, and now also
`GBOpt/optimization/`:

- **`types.py` is the lowest-level module in every subpackage.** It holds the
  exception hierarchy plus small immutable value types/dataclasses built on
  them (including their own validation and `to_state`/`from_state`
  serialization) — no optimizer/construction/IO policy. Everything else in
  the subpackage may import from `types.py`; `types.py` imports from nothing
  else in the subpackage. Don't split exceptions and their closely-related
  value types into separate files (e.g. a onetime `errors.py` +
  `evaluation.py` split was collapsed back into one `types.py` on review).
- **Write out the dependency graph explicitly** (leaf module → what depends
  on it) before/while decomposing a monolith, and verify with `grep` that no
  submodule imports a higher-level sibling back. Sibling modules at the same
  level (e.g. `monte_carlo.py` and `genetic.py`) should not import each
  other.
- **`__init__.py` curates a small, genuinely public surface** re-exported
  from the subpackage's modules, grouped by category with comments (not
  strictly alphabetized — that's a deliberate, consistent choice across
  `crystallography`, `artifacts`, and `optimization`, even though it trips
  ruff's `RUF022`). Document in the module docstring what is *not* promoted
  (underscore-prefixed internals, runtime-only helpers) and why.
- A compatibility facade for an old import path (e.g. `GBOpt/GBMinimizer.py`
  over `GBOpt/optimization/`) should import from the subpackage's `__init__`,
  not reach into individual submodules.
- **A new service that combines two existing modules' concerns doesn't
  automatically need a subpackage.** R14's `CandidateLoader` (combining
  `FileGrainOwnership`'s `CandidateFileMapping` with `GBManipulator` and
  `GBOpt.io.lammps.LammpsDataWriter`) was added as a single flat top-level
  file, `GBOpt/CandidateLoader.py`, matching the naming and layering of its
  siblings `GBManipulator.py`/`GBMaker.py`/`FileGrainOwnership.py`/
  `GrainOwnership.py` — not a `GBOpt/candidate_loader/` subpackage with its
  own `types.py`. The `types.py`-as-leaf subpackage convention above is for
  decomposing one large module into several internally-layered files; a
  single small service class with no internal submodule structure of its own
  doesn't need that scaffolding. Judge by whether the new code would ever be
  split into more than one file on its own merits, not by whether it "combines"
  existing concepts — combining is what most new modules in this codebase do.
- **A brand-new subpackage can hold zero extracted logic and still warrant the
  subpackage shape, if it has genuine internal layering of its own** — this is
  different from `CandidateLoader`'s flat-file case above, which combined two
  *existing* modules' concerns into one class with no internal submodule
  structure. R15's `GBOpt/manipulation/` package (issue #76) defines a new
  operation contract (`Manipulation` protocol, `ManipulationContext`/
  `ManipulationResult` value types, an 8-class exception hierarchy) plus a
  separate explicit registry, with "no built-in manipulation algorithm...
  moved yet" — nothing decomposed, nothing combined. It's still a real
  `types.py`-leaf + `registry.py` two-module dependency graph
  (`registry.py` imports only from `types.py`), so the subpackage shape was
  correct despite having no extracted content: judge by whether the new
  modules would have their own internal dependency edges, not by whether
  anything is being moved out of an existing file.
- **A `Protocol` interface contract belongs in `types.py` alongside the value
  types it exchanges, not a separate `protocol.py`** — matching
  `GBOpt/io/types.py`'s existing `StructureReader`/`StructureWriter`
  precedent. R15's `Manipulation` protocol lives in
  `GBOpt/manipulation/types.py` next to `ManipulationContext`/
  `ManipulationResult` for the same reason: the protocol's method signature
  references those types directly, and splitting them across files would
  just add an import back into the "lowest-level" module.

## Testing conventions

- **One test file per module**, named `test_<package>_<module>.py`
  (`test_crystallography_boundary.py`, `test_artifacts_cleanup.py`,
  `test_optimization_monte_carlo.py`, ...). Never a catch-all file for a
  whole subpackage or for "compatibility."
- Per-module test files import from the specific submodule under test
  (`from GBOpt.optimization.monte_carlo import MonteCarloMinimizer`), not
  from the subpackage's curated `__init__` re-export — even though the
  `__init__` re-export exists for external callers.
- **Test GBOpt behavior, not Python/import mechanics.** Red flags: a test
  whose only assertion is object identity (`is`) or that inspects source via
  `ast`/`inspect` to check code shape. Before adding this kind of test to
  guard a compatibility contract, check whether the codebase already tests
  that *kind* of thing anywhere — it usually doesn't (e.g. `GBOpt/__init__.py`
  re-exports `GBMaker`, `GBManipulator`, `InterfaceCandidate` and none of that
  is tested), and a trivial re-export module fails loudly (`ImportError`) on
  its own if it's ever wrong, without a dedicated test.
- When relocating existing tests during a decomposition (e.g. pulling
  `Mutator` tests out of a larger file), check for coverage gaps in the area
  while you're there — this surfaced two previously-untested validation
  paths (`Mutator.__init__` rejecting unknown/empty choices) worth adding.
- **Migrating off name-mangled private-method test access
  (`self.gbm._GBMaker__method(...)`) happens opportunistically per extraction,
  not as a standalone sweep.** When a roadmap step moves a private method's
  logic into a `gbmaker` (or other subpackage) submodule, relocate its tests
  to plain pytest functions calling the module's real, unmangled name — this
  is what eliminates the name-mangled access, as a side effect of the
  extraction itself, not a separate refactor (R05 did this for
  `__reduce_integer_row`/`__row_angle_error_deg`/`__approximate_rotation_*`
  into `test_gbmaker_orientation.py`). For whatever stays a private method
  long-term (orchestration that won't move, or logic not yet scheduled for
  extraction), don't reach into it directly at all — test it only through the
  owning class's public API/observable behavior, the way `GBMaker.py`'s
  `_validate_scalar`-wrapping validators already have no dedicated
  GBMaker-level tests and are exercised only via the constructor. Expect a
  residue of `_GBMaker__...`-style tests in `test_gbmaker.py` to persist
  until R10 decides `GBMaker`'s facade shape; don't treat that residue as a
  bug to batch-fix.
  **Update from R10 (#71):** the residue was larger than a single-extraction
  migration -- replacing ~36 flat private attributes with the `_config`/
  `_boundary`/`_result` state containers broke roughly 40 name-mangled call
  sites across `test_gbmaker.py` and `test_gbmaker_assembly.py` in one step,
  not the handful a per-method extraction produces. This is the same
  discipline at a different scale, not a different discipline: migrate each
  one to the corresponding `_config.*`/`_boundary.*`/`_result.*` path as a
  direct, expected consequence of the state-model change, budgeted as part
  of that same step rather than deferred. Method calls on `GBMaker` itself
  (`gb._GBMaker__get_triclinic_params()`, `gb._GBMaker__material_state()`)
  needed no change, since the methods weren't moved or renamed, only the
  attributes they read — don't assume every `_GBMaker__...` occurrence in a
  state-model diff needs touching; grep for the ones that resolve to
  attributes, not methods.
- **An "X does not import Y" acceptance criterion (import-boundary tests)
  needs a subprocess, not an in-process assertion.** `sys.modules` is shared
  across the whole pytest run, so a bare `import GBOpt.gbmaker; assert
  "GBOpt.optimization" not in sys.modules` can false-pass simply because no
  earlier test in the same process happened to import optimization first —
  or false-fail because one did, depending on test order, neither of which
  says anything about whether `gbmaker` itself pulls it in. R09's import-
  boundary test (issue #70's "clean construction doesn't import optimization
  modules" criterion) instead shells out
  (`subprocess.run([sys.executable, "-c", "import GBOpt.gbmaker; import sys;
  assert 'GBOpt.optimization' not in sys.modules"])`) to get a fresh
  interpreter with an empty `sys.modules`. Expect more of these as later
  roadmap steps (R10+) add further module-boundary acceptance criteria — the
  subprocess pattern is the reusable fix, not a one-off.
  **Correction from R12 (#73):** the plain subprocess pattern above silently
  fails for any "does not import `GBOpt.GBMaker`" check under `GBOpt.io` (or
  any other check where the thing you're asserting isn't imported is *also*
  something `GBOpt/__init__.py` itself eagerly imports at module scope —
  `GBOpt/__init__.py` does `from GBOpt.GBMaker import GBMaker` to re-export
  it). Python always executes a package's `__init__.py` before any of its
  submodules, so `import GBOpt.io.lammps.data_writer` puts `GBOpt.GBMaker`
  into `sys.modules` regardless of what `data_writer.py` itself imports — the
  check would fail for *any* submodule of `GBOpt`, saying nothing about that
  submodule's own import graph. The fix: stub `sys.modules["GBOpt"]` with an
  empty `types.ModuleType` (its `__path__` populated from
  `importlib.util.find_spec("GBOpt").submodule_search_locations`, which
  locates the package on disk without executing it) *before* importing the
  submodule under test, so `GBOpt/__init__.py`'s body never runs in that
  subprocess and only the submodule's own imports are observed. See
  `tests/test_io_lammps_data_writer.py::test_data_writer_module_does_not_import_gbmaker`
  for the full script. Use this stub whenever the "not imported" target is
  something `GBOpt/__init__.py` re-exports (`GBMaker`, `GBManipulator`,
  `Atom`, `BoundaryTopology`, `Position`, `UnitCell`); the plain R09 pattern
  is still correct for targets `GBOpt/__init__.py` doesn't import (e.g.
  `GBOpt.optimization`).
- **Before writing a new regression test to satisfy an "X behavior is
  unchanged and explicitly regression-tested" acceptance criterion, grep the
  existing test file for the behavior by name — an earlier roadmap step may
  already have added exactly that test for its own reasons, and the current
  step only needs to confirm it still passes, not duplicate it.** Issue #81
  (R21) required "current `unique_id=None` behavior remains unchanged and is
  explicitly regression-tested." `tests/test_optimization_monte_carlo.py`
  already had `test_two_fresh_runs_without_unique_id_use_different_labels`
  (added for an earlier step, unrelated to logging) that constructs two runs
  with `unique_id` omitted entirely and asserts the checkpointed labels
  differ — a real exercise of the `None` path, not just a name that sounds
  related. R21 added no new `unique_id`-specific test; running the existing
  suite (unchanged, still green) was the criterion's actual discharge. Don't
  assume a checklist item implies a new test is owed — check whether the
  regression coverage already exists first.

## Refactor-issue discipline

- A "mechanical, behavior-preserving" issue's acceptance criteria are literal
  constraints, not a floor. E.g. #62 says "warnings ... are unchanged" —
  that rules out adding a `DeprecationWarning` to a new compatibility facade
  even though it looks like good practice. If a seemingly-good addition
  isn't in the issue's acceptance criteria, flag it as a question or a
  separate follow-up rather than adding it unilaterally.
- Before introducing a new pattern (a deprecation warning, a new kind of
  test, a new module split), grep the codebase for how sibling
  packages/modules already handle the same concern and follow that instead
  of a generic best practice.
- Verify a "no behavior change" refactor by diffing tool output against the
  pre-refactor code, not just against zero: ruff/mypy/pyscn/bandit finding
  *counts* should match the original file exactly (same pre-existing debt,
  just possibly relocated), and `pytest -m "not slow"` should produce the
  same pass/fail set. New findings mean something changed; fixing
  pre-existing findings as a drive-by blurs what the PR actually did.
- **A handful of `PermissionError: ... tmp...` failures in
  `test_gbmaker.py`/`test_gbmanipulator.py`/`test_checkpoint.py` (LAMMPS
  write, JSON checkpoint write) and Windows-only file-locking are baseline
  noise on this machine, not a regression** — they showed up identically
  across every full-suite run in the R03/R04 sessions regardless of what was
  changed. When diffing pass/fail sets per the rule above, treat this
  specific failure shape (a `PermissionError` opening a temp file inside a
  write path) as expected noise rather than investigating it as caused by
  your change, but still confirm the *set* of such failures didn't grow.
  **Correction from R11 (#72):** the "10 failed" set is not homogeneous —
  don't describe all 10 as `PermissionError` without checking each one.
  `test_checkpoint.py::test_json_serialization_normalizes_supported_values[path]`
  is a plain `AssertionError`, not a `PermissionError`: it asserts a
  hardcoded POSIX path string (`'/some/dump.data'`) against
  `str(Path("/some/dump.data"))`'s Windows rendering (`'\some\dump.data'`),
  entirely unrelated to file writes or locking. The other 9 are the real
  `PermissionError`, and R11 traced them to one concrete, fixable root cause
  rather than an unexplained platform quirk: every one calls
  `tempfile.NamedTemporaryFile(delete=True)` as a context manager and then
  passes `temp_file.name` to code (`GBMaker.write_lammps`,
  `GBManipulator`'s write path) that reopens that same path with `open(...,
  "w")` *while the original handle is still open* — Windows refuses the
  second open (no default sharing), POSIX allows it. This is a test-fixture
  bug, not a `GBMaker`/`GBManipulator` production-code bug, and the fix is
  already used correctly once in the same file
  (`test_gbmaker.py::TestGBMakerTriclinic::test_triclinic_params_uses_grain_with_larger_y_period_norm`,
  via `NamedTemporaryFile(delete=False, ...)`, closing the handle before
  `write_lammps` reopens it, then `finally: os.unlink`): apply that pattern
  to the other 9 call sites. This is out of scope for a read-only step like
  R11 (`GBOpt.io` only added readers) and isn't tied to `write_lammps`'s
  production logic, so it doesn't need to wait for R12/R13's writer work
  either — see `REFACTOR_CLEANUP.md`'s entry for it as a standalone
  test-hygiene fix. Until it's fixed, keep diffing pass/fail *sets* by test
  name as before; just don't attribute the whole set to one failure shape
  without having actually reproduced each one.
- **`GBMaker.__validate(..., positive=True)` means "non-negative" (only
  rejects `value < 0`, so `0` passes), not what the name suggests. Its
  `strictly_positive=True` is the one that actually means `> 0`.** When R03
  designed `GBOpt/gbmaker/types.py`'s value contracts, every field that came
  from a legacy `positive=True` call was mapped to a strictly-positive
  (`> 0`) validator by name association — wrong for five of six such fields.
  R04 caught `gb_id` immediately (49 failing tests, `gb_id=0` is a common
  fixture value); auditing the rest by grepping every `positive=True` call
  site in the original `__validate` turned up three more silent instances —
  `x_dim_min`, `interaction_distance`, `mismatch_tol` — none caught by a
  test, all real legacy-legal values at exactly `0` (`mismatch_tol=0.0` in
  particular is semantically distinct from `None`: exact-match-required vs.
  accommodation-disabled). `a0` was the one exception, deliberately left
  strictly positive: unlike the other five, a zero lattice parameter was
  never a *usable* legacy value — nothing downstream produces a valid
  structure at `a0 == 0`, so rejecting it is a genuine improvement, not a
  behavior change anything could depend on. The general rule: when mapping
  a legacy `positive=True`/`nonneg`-shaped check onto a new value type,
  don't trust the parameter name — grep every call site of the thing being
  replaced, not just the one causing a visible test failure, and decide
  field by field whether `0` was ever a legal, meaningful legacy value
  before picking a stricter validator than the code you're replacing.
  `_validate_scalar` in `gbmaker/config.py` renamed its own copy of this
  parameter to `nonnegative` once this was understood — new code we own can
  just be named correctly; `GBMaker.__validate`'s own established `positive`
  parameter (~20 call sites) was left alone as a separate decision.
- **A value type from an earlier roadmap step isn't frozen just because an
  earlier PR introduced it — correcting it in a later step's branch is
  expected, not a scope violation, when the fix is driven by an actual
  behavior-preservation requirement** (the four `positive=True` fields
  above) **and not by taste.** R03's `types.py` was designed without full
  visibility into every legacy call site; R04, while actually wiring
  `GBMaker` through those contracts, is exactly where that gap surfaces and
  should be fixed on the spot rather than deferred.
- **When extracting a private method out of a class that has its own
  established exception hierarchy (e.g. `GBMaker.__validate` and its five
  static validators, which raise `GBMakerValueError`/`GBMakerTypeError`),
  and the class still has other callers of that method that aren't being
  touched (R04 left `GBMaker`'s ~10 property setters and misorientation
  validation alone since only constructor/factory normalization was in
  scope) — turn the original method into a thin wrapper that calls the new
  pure function and translates its exception type back**
  (`GBMakerConstructionValueError` -> `GBMakerValueError`), rather than
  duplicating the validation logic in two places or rewriting every
  remaining caller in the same PR. This isn't optional indirection: issue
  #61's compatibility requirements explicitly protect "established...
  public exception identities," and the pure function can't raise
  `GBMakerValueError` directly without a circular import
  (`GBMaker.py` -> `gbmaker/config.py` -> `GBMaker.py`), so translating at
  the boundary is the only way to keep both constraints. Consolidate the
  translation itself into one small shared helper
  (`__translate_construction_error(func, *args, **kwargs)`) rather than
  repeating the same two-line try/except at every wrapper — six near-copies
  of it accumulated in R04 before being collapsed to one. Expect this same
  wrapper shape to recur for R05-R09 as more of `GBMaker`'s ~3500 lines get
  extracted while some callers (setters, in particular) are intentionally
  left in place until a later issue's scope covers them; it's very likely
  temporary scaffolding that gets resolved once R10 ("reduce GBMaker to a
  compatibility facade") gets to decide on purpose whether
  `GBMakerValueError` stays its own class or becomes an alias of
  `GBMakerConstructionValueError`.
- **A speculative value type added in an earlier roadmap step "for later" can be
  wrong-shaped in ways beyond a single validator being too strict — a field can
  conflate two things a real call site needs separately, or exist but never
  actually get read.** R08 wired up `GrainBuildRequest`
  (`GBOpt/gbmaker/types.py`), added speculatively in R03/R04 alongside
  `AxisAccommodation` and left unused until R08 finally called it from real
  builder functions. Its single `orientation` field (dtype conditional on an
  `exact` flag) turned out to conflate two matrices both the exact and
  approximate builders need *simultaneously* — a proper rotation matrix and a
  separate canonical integer periodic-direction matrix — so it had to be split
  into two fields (`rotation`, `periodic_matrix`). Its `box_dims` field (a 3x2
  bounds array) was never read by either builder at all; grep the field's own
  usages inside the functions being wired up, not just its type annotation,
  before assuming a speculative field is correctly shaped — `AxisAccommodation`
  needed no changes when R06 finally used it for real, so "speculative types are
  usually fine" is not a safe prior; check each one on its own.
- **When two structurally-identical dataclasses exist — one added speculatively
  in `gbmaker/types.py` for a not-yet-wired contract, one already live as a
  private helper class inside the file being extracted — collapse them
  deliberately into the `types.py` one once the extraction makes them coexist,
  rather than carrying both forward by default.** R08 found `GBMaker.py`'s
  private `_FloatGrainBuildResult` (`atoms`/`origin_ids`/`basis_size`) was the
  same shape as `types.py`'s already-present `GrainBuildResult`
  (same three fields plus `grain_side`). Confirm the shapes really match by
  reading both dataclasses side by side before merging — don't assume from the
  names alone.
- **When an extraction removes a private method's only remaining caller,
  grep two things before deleting the now-dead wrapper: other
  `self.__method_name(` call sites within the class, and name-mangled test
  access (`grep -rn "_ClassName__method_name" tests/`).** A wrapper with zero
  internal callers left is not necessarily safe to delete outright — R08 found
  four `GBMaker.py` geometry wrapper methods
  (`__selection_basis_vectors`/`__reduced_box_coordinates`/
  `__reduced_coordinate_tolerance`/`__cartesian_from_box_coordinates`) whose
  only *internal* caller was the method being extracted, but which
  `tests/test_gbmaker.py` still called directly via
  `gb._GBMaker__selection_basis_vectors(...)` as verification helpers for real
  generated-grain output. Per the name-mangled-test-migration convention
  above, the fix is to migrate those test call sites to the real unmangled
  function in the module the logic now lives in (here,
  `GBOpt.gbmaker.geometry`), not to leave the tests broken or to keep the dead
  wrapper around indefinitely. Wrapper methods that already had zero callers
  *before* the extraction (`__scaled_periodic_basis_vector`,
  `__complete_origin_atom_mask` in R08) are pre-existing debt, out of scope —
  leave them alone rather than batch-cleaning everything that looks unused.
- **A `pyscn check` clone-detection finding disappearing between baseline and
  current is not automatically suspicious — check whether the clone pair
  itself was deleted.** R08's pre-existing clone finding (two near-identical
  `GBMaker.py` wrapper methods, 85% similarity) went away simply because both
  methods were removed as dead code in the same step; the finding count
  dropping is an expected side effect of deletion, not evidence a real
  duplication was silently "fixed" by rewriting shared logic.
- **A speculative value type can be wrong-shaped by being too coarse, not
  just by conflating two distinct fields into one.** R08's `GrainBuildRequest`
  fix was splitting one field into two (`rotation`/`periodic_matrix`). R09's
  `BicrystalResult` (also speculative, R03/R04) had the opposite problem: a
  single combined `atoms` field, when the real assembly call sites needed
  left-grain, right-grain, and GB-region atom arrays separately (mirrored
  individually onto `GBMaker.__left_grain`/`__right_grain`/`__gb_region` for
  existing property/setter compatibility) — so `left_atoms`/`right_atoms`/
  `gb_region_atoms` were added alongside `atoms`. Same underlying check as
  R08's ("grep the field's own usages at the real call sites before trusting
  a speculative type's shape"), just manifesting as under-granularity instead
  of over-conflation — don't assume a single-field type is automatically the
  right shape just because it isn't obviously conflating two matrices.
- **Composing several already-pure pipeline stages into one orchestration
  function (e.g. `build_bicrystal`) will likely trip `pyscn`'s per-function
  SLOC-length check even when every stage it calls is already extracted and
  well-factored** — the orchestration function's own line count grows with
  the number of stages it visibly sequences. R09 hit this on both
  `assemble_bicrystal` and `build_bicrystal` and resolved it by extracting
  further named helpers (`_build_grains_for_path`,
  `_plan_orientation_and_dimensions`) rather than collapsing logic or
  suppressing the check — which doubled as better documentation of the
  pipeline's actual stages. Expect this again for any later roadmap step
  that composes multiple prior steps' pure functions into one top-level
  pipeline entry point; budget for one more decomposition pass beyond the
  "obvious" stage extraction, not just the stage extraction itself.
- **The SLOC/complexity trip-up above isn't limited to *composing already-
  extracted* stages — a single new method written from scratch in one pass
  (not an extraction at all) can trip the same check just as easily,
  especially multi-section text formatting/serialization code.** R12's first
  draft of `LammpsDataWriter.write()` (new code, not moved from anywhere)
  was a single 156-line method (complexity 22) assembling a LAMMPS file's
  comment/header/box/tilt/type-label/atoms sections in sequence. Same fix as
  the pipeline case: decompose into named helper functions per section
  (`_write_header`, `_write_atoms_section`, `_resolve_name_to_int`,
  `_validate_and_resolve_charges`, `_declared_losses`) rather than
  suppressing the check — budget for this decomposition pass whenever a new
  roadmap step writes a new formatter/serializer/assembler method, not only
  when it's composing prior steps' extracted functions.
- **Routing existing inline logic through a new shared value type (e.g.
  `StructureData`) can incidentally add stricter runtime validation that
  didn't exist before, separate from and in addition to the mypy-level
  strictness changes covered above.** R12's `GBMaker.write_lammps()` now
  constructs a `StructureData` from its `atoms`/`box_sizes` arguments before
  formatting; `StructureData.__init__`'s existing finite-3x3/finite-length-3
  checks (from R11) now run on every call, translated to `GBMakerValueError`.
  Previously, non-finite/malformed `box_sizes` either crashed with an
  unrelated `IndexError`/`TypeError` deep in the old inline code or silently
  wrote a garbage header line (`f"{nan}"` formats without error) — neither
  path raised a clear, typed error. No existing test exercised this shape of
  input, so it wasn't caught by the pass/fail-set diff, only by reasoning
  through what changed. When wiring existing formatting/construction code
  through a newer subpackage's already-validating value type, check whether
  that type's constructor validates something the code being replaced never
  did, and disclose it (in `REFACTOR_CLEANUP.md` or the PR description) even
  when no test forces the question — the pass/fail-set diff only proves
  behavior *inside the range existing tests exercise* is unchanged.
- **An issue's "Proposed behavior" prose can conflict with its own acceptance
  criteria; when it does, the acceptance criteria win, and the deviation gets
  disclosed rather than silently resolved either way.** Issue #71 (R10) asked
  to "centralize regeneration in `_rebuild()`," but also required "every
  existing setter... retains its current validation/rebuild semantics" and
  "repeated rebuilds are deterministic and match current regression results."
  `GBMaker`'s 9 property setters don't share one rebuild behavior today —
  they split into distinct tiers (store-only; box-dims-only; dims+assembly;
  full spacing+dims+assembly; unit-cell-only with **no** geometry
  regeneration, for `structure`; an incremental atom-position patch that
  **never** calls the assembly pipeline, for `vacuum_thickness`). Forcing all
  nine through one literal `_rebuild()` that always does the "biggest" tier
  would have been a real behavior change (added geometry regeneration
  `structure` never triggered before) or an unverifiable numerical-drift risk
  (`vacuum_thickness`'s incremental shift vs. a full re-assembly through a
  pipeline full of `np.isclose` epsilon comparisons might not be bit-for-bit
  identical on every input, and today's test suite passing doesn't prove it
  never diverges). R10 kept the three pre-existing orchestration methods
  (`__generate_gb`, `__update_dims`, `update_spacing`) each setter already
  called at its own tier, rather than collapsing them — state is centralized
  (`_config`/`_boundary`/`_result`); regeneration orchestration is
  consolidated into named, composable methods; but there is no single
  `_rebuild()` symbol. This was raised to the user explicitly rather than
  decided unilaterally, since it's a real fork between literal-prose
  compliance and criteria compliance, not a mechanical judgment call.
- **A newer roadmap step's own pipeline dataclass is not automatically the
  right vessel for a legacy class's "canonical state," even when the field
  names overlap.** R10 tried routing `GBMaker`'s `_config` through
  `GBOpt.gbmaker.types.MaterialState`/`GBBuildConfig` directly (the
  dataclasses `normalize_legacy_config` already builds) before writing new
  facade-owned containers. This failed for a concrete reason: `MaterialState`
  requires `a0` strictly positive (the deliberate R04 divergence from
  `GBMaker`'s own legacy setter, which only rejects negative values — see the
  `positive=True` entry above), and a **frozen dataclass always re-runs
  `__post_init__` on `dataclasses.replace(...)`**, with no way to bypass it
  for an already-validated value. Routing the `a0` *setter* through
  `MaterialState.replace(a0=...)` would have silently started rejecting
  `a0 == 0`, a real regression, not a hypothetical one. The general check:
  before reusing an existing pipeline dataclass as a legacy-compatibility
  facade's own state container, confirm its validation rules are actually
  compatible with every caller that will construct/replace it through the
  facade's established (possibly looser, possibly historically-quirky)
  entry points — matching field names is not the same as matching contracts.
  `_result`, by contrast, safely reuses `assemble_bicrystal`'s real
  `BicrystalResult` return value directly, since nothing routes it through
  independent legacy validation.
- **When extracting a private method's logic into a pure function, don't
  assume a similarly-named helper already imported into the target module is
  a safe drop-in for the original's raw computation, even when it looks like
  exactly the right fit.** While extracting `GBMaker.__get_triclinic_params`
  into `gbmaker.geometry._triclinic_tilt_params` (R10), the grain-selection
  comparison originally used `np.linalg.norm` on a periodic-Miller-row array.
  `gbmaker.geometry` already has `_miller_row_norm`, imported there for other
  kernels, and swapping it in looked like a natural cleanup — but
  `_miller_row_norm` raises `GBMakerConstructionValueError` on any
  non-integer-dtype row, a stricter check than `np.linalg.norm`, which the
  periodic-Miller-row arrays are not guaranteed to satisfy. That swap would
  have added a spurious new exception path on inputs the original silently
  handled. Caught by reading the two implementations side by side before
  landing, not by the test suite. The general check is the same "would I
  make this change if the extraction didn't exist as an excuse?" test used
  elsewhere in this file for mypy/cast decisions — here applied to reusing a
  neighboring helper during a move, not to type-checker appeasement.

- **"Eliminate a local-import cycle workaround" doesn't mean the local import
  disappears — it means its target changes to whichever module can safely own
  the dependency at module scope, per correct layering.** R14 (#75) required
  `FileGrainOwnership` no longer locally import `GBManipulator` inside
  `reload_explicit_manipulator` to avoid a cycle. The fix wasn't to hoist that
  import to module scope in `FileGrainOwnership.py` (still circular — the new
  `CandidateLoader` module that owns the real reload logic imports
  `FileGrainOwnership` at module scope for `CandidateFileMapping`, so the
  reverse at module scope would be A-imports-B-imports-A) or to leave the
  local `GBManipulator` import in place (fails the criterion literally). The
  actual fix: move the validated-reload logic itself into `CandidateLoader`
  (a new, higher-level module allowed to import both `GBManipulator` and
  `FileGrainOwnership` at module scope, since nothing on either's import path
  reaches back into it — verified with the stub-parent-package subprocess
  pattern below, run in both directions), and have
  `FileGrainOwnership.reload_explicit_manipulator` become a thin wrapper with
  a *local* import of `CandidateLoader` instead. The local-import mechanism
  (defer resolution to call time, after all modules are fully loaded) is
  still exactly what breaks the cycle; what changed is which module is
  correctly positioned to own it at module scope. When an acceptance
  criterion names eliminating a specific local import, check whether the real
  fix is relocating the code that needed it to a module one layer up, not
  just moving where the import statement sits.
- **A code comment claiming "avoids a module cycle" is a claim about the
  current import graph, not a permanent fact — verify it by grep before
  trusting it, same as a `REFACTOR_CLEANUP.md` entry's call-graph claims.**
  `FileGrainOwnership.py`'s local `GBManipulator` import carried exactly this
  comment. At R14, `grep`ping `GBManipulator.py` for `FileGrainOwnership`
  found no reference at all — nothing on `GBManipulator`'s current import
  path reaches `FileGrainOwnership`, so the literal cycle the comment warned
  about no longer existed (it may have via a different, since-refactored
  path). The workaround was still worth keeping in spirit (a local import is
  cheap insurance against a future accidental cycle), but don't assume a
  comment like this is still describing a real constraint just because it's
  old and specific-sounding; re-derive the actual import graph.
- **An "X's mapping should originate from Y" acceptance criterion can be
  satisfied by making Y-sourcing available and demonstrably equivalent,
  without forcibly rewiring every existing call site that predates Y** —
  especially when a call site's own write is outside this codebase's control.
  Issue #75 (R14) asked for `CandidateFileMapping`'s candidate-local IDs to
  "originate from `WriteResult` or an equivalent explicit serialization
  result." `CandidateFileMapping.from_candidate`'s existing
  `np.arange(1, N+1)` assumption predates any real write in the one call site
  that matters most (`_explicit_ownership_evaluation.py`'s pre-evaluator
  mapping construction) because the actual file write there is performed by
  the external evaluator callback, not GBOpt — there is no real `WriteResult`
  available to route through at that point. Verifying
  `LammpsDataWriter.write()` always assigns exactly `np.arange(1, N+1)`
  confirmed the existing assumption is byte-for-byte what a real write would
  produce, i.e. genuinely "an equivalent explicit serialization result," so
  the new `CandidateLoader.write_candidate()` demonstrates the literal
  `WriteResult`-sourced pattern as a new capability (for call sites GBOpt
  *does* control the write for) rather than being forced through every
  existing speculative-mapping call site. Disclosed as a deliberate scoping
  decision in `REFACTOR_CLEANUP.md` rather than silently deciding either way,
  per the "acceptance criteria win, disclose the deviation" precedent above.
- **An acceptance criterion can be genuinely ambiguous about which side of a
  boundary it constrains, not just in conflict with the issue's prose (the
  R10 `_rebuild()` case above).** Issue #76's (R15) "Interface topology,
  ownership labels, physical bounds, and separation state are preserved by
  the facade adapter" reads naturally as either an input-side guarantee (the
  facade correctly threads a manipulator's own state into what it hands an
  operation) or an output-side one (the facade validates what an operation
  hands back). The same issue's own "No built-in manipulation algorithm is
  moved yet" criterion rules out a meaningful output-side check for R15
  specifically — there is no built-in operation whose output the facade
  could validate against yet, and an externally-defined test operation is
  free to return any valid `InterfaceCandidate` it wants. R15's
  `GBManipulator.apply()` reads the criterion as input-side only (it converts
  parent state into `ManipulationContext.parents` via the same
  `_to_interface_candidate()` path `make_parent_candidate()` already used,
  and does not re-validate `ManipulationResult.children`), and discloses that
  reading in `REFACTOR_CLEANUP.md` rather than guessing silently or inventing
  unrequested output validation just to cover both readings at once. When a
  criterion could bind either side of a not-yet-built pipeline, resolve the
  ambiguity by asking what's actually satisfiable within the issue's own
  stated scope.
- **A new API that accepts a `seed` parameter must treat `seed=0` as valid
  and distinct from `seed is None`, even when nearby/legacy code in the same
  class doesn't.** `GBManipulator.__init__`'s existing seed handling is
  `if not seed: self.__rng = np.random.default_rng()` — `seed=0` is falsy in
  Python, so it silently produces an *unseeded* generator instead of the
  deterministic one requested. Issue #76 (R15) explicitly requires
  `apply()`'s own `seed=0` to be valid and deterministic, so `apply()`'s seed
  handling checks `seed is None` instead of `not seed`. Fixing the legacy
  constructor was out of scope (issue #76 never touches `__init__`) and was
  disclosed in `REFACTOR_CLEANUP.md` rather than fixed unilaterally, per this
  file's own "acceptance criteria are literal constraints" discipline
  extended to *not* fixing an adjacent bug an issue didn't ask about either.
  Check sibling seed/RNG-handling code for this exact `if not seed` shape
  before copying it into any new entry point.
- **A "legacy methods delegate to the new value-typed operation" ask can be
  literally impossible for one specific method, when that method's own
  established leniency conflicts with the new value type's *unconditional*
  `__init__`-time invariant — not just a validation-strictness difference a
  translation wrapper can paper over.** Issue #77 (R16) asked
  `GBManipulator.translate_right_grain`/`cycle_grain_terminations`/
  `apply_interface_separation` to delegate to new `Manipulation`-protocol
  operations that "consume explicit `InterfaceCandidate` state." But
  `InterfaceCandidate.__init__` unconditionally rejects any atom outside its
  labeled physical grain bounds, while `translate_right_grain`'s own existing
  comment explains that explicit ownership is persistent state and a relaxed
  right-grain atom may legitimately already lie — or be translated to lie —
  outside `right_grain_x_bounds` without changing grains (two currently-
  passing regression tests exercise exactly this, one of them on the
  *parent's own pre-translation state*). Because `ManipulationContext`/
  `ManipulationResult` both require `InterfaceCandidate` on every boundary,
  there is no way to route this method through the protocol at all without
  losing that leniency — not even for constructing the *input* candidate,
  let alone the output one. The fix wasn't a stricter-vs-looser translation
  wrapper (the established pattern for exception-identity mismatches); it
  was extracting the actual math into a plain-arrays-in/plain-array-out pure
  function (`right_grain_translation_atoms` in
  `GBOpt/manipulation/translation.py`) that neither takes nor returns an
  `InterfaceCandidate`, called directly by the legacy method (bypassing the
  protocol entirely) and also called, then wrapped in a validating
  `InterfaceCandidate`, by the new operation's own `execute()` (which is
  correct there, since `make_translation_candidate` already enforced that
  same validation before R16). The general check: before assuming any
  "delegate to the new operation" ask is satisfiable for *every* legacy
  method an issue lists, check whether the new value type's own constructor
  invariants are unconditional (no bypass) and whether any legacy method's
  tested behavior depends on an atom/state configuration that invariant
  would reject — if so, share the pure computation, not the value-typed
  boundary, and disclose which methods actually route through the protocol
  versus which only share code with it.
- **Consolidating several call sites that independently accessed the same
  loosely-typed attribute (here, `self.__parents[index]` typed
  `list[Parent | None]`) into one shared private helper can legitimately
  drop dozens of mypy findings, even though every dropped finding is the
  exact same "None has no attribute X" shape the file already had.** R16's
  new `__parent_candidate_geometry`/`__parent_candidate` helpers replaced
  several inlined `parent = self.__parents[0]; parent.<attr>` blocks (in
  `translate_right_grain`, `cycle_grain_terminations`,
  `apply_interface_separation`, plus the pre-existing
  `make_parent_candidate`/`__current_parent_candidates`), and moved the
  bulk of the remaining attribute-heavy computation into
  `GBOpt/manipulation/termination.py`/`separation.py`, which operate on
  `InterfaceCandidate` (properly typed, no `| None`) instead of `Parent`.
  `GBManipulator.py`'s own mypy finding count dropped from 202 to 170 as a
  direct, expected consequence — confirmed by diffing finding *text*
  (message-shape counts), not just the total, per this file's own existing
  "verify a decrease the same way you'd verify an increase" rule: every
  dropped finding was the same `"None" has no attribute ...` /
  `Value of type variable "_SCT" of "asarray" cannot be "float"` shape
  already present before R16, at fewer call sites, not a new suppression.
  `GBOpt.manipulation` itself stayed mypy-clean both before and after.
- **Turning a "produce N results in one call" API into a "select result i" API
  is a faithful reframing, not a semantics change, only if the internal
  computation for index `i` is made to request exactly as much work as the old
  API would have needed to produce `i+1` results — not a fixed/maximal amount.**
  R17's `displace_along_soft_modes(num_children: int)` (computing the `num_children`
  softest eigenvectors per q-point, then globally sorting and returning all
  `num_children` children) became `displace_along_soft_modes(mode_index: int = 0)`
  by internally computing `num_modes_needed = mode_index + 1` eigenvectors per
  q-point — i.e. exactly what the old API would have computed for
  `num_children = mode_index + 1` — then taking the last element of the same
  global sort. This makes `mode_index=i` numerically identical to `children[i]`
  from the old `num_children=i+1` call, which is what let the old
  `num_children=2` test become two explicit `mode_index=0`/`mode_index=1` calls
  without changing expected values. Computing a fixed/maximal number of
  eigenvectors regardless of `mode_index` (simpler to write, and still
  "correct" in the sense of returning *a* valid mode) would have silently
  changed which physical mode `mode_index=i` refers to, since the eigensolver's
  per-q-point truncation depth was tied to how many eigenvectors the *old* API
  computed, not an independent hyperparameter.
- **A batch-of-N API's per-item loop can hide genuinely dead code that only
  becomes visible once the API is collapsed to a single item — check for it
  explicitly rather than assuming a line-for-line loop-to-single-value
  translation.** R17's original per-child loop body ended with
  `non_gb_indices = np.setdiff1d(...); pos[mode_index, non_gb_indices] =
  positions[non_gb_indices]` — entirely redundant, since `pos[mode_index]` was
  already initialized as a full copy of `positions` a few lines earlier and
  only the GB-indexed rows were ever mutated in between. It read as
  purposeful (defensive re-assignment of the "other" atoms) inside a loop with
  several other lines of real per-child logic; collapsing the loop to a single
  `mode_index` selection made it obviously a no-op and safe to drop.
- **When a new validation criterion is added (e.g. "raise a clean, typed error
  for an out-of-range selector"), check every execution branch that could reach
  the un-validated code path, not just the one branch that historically had a
  guard.** R17's `displace_along_soft_modes` had an existing bounds check for
  `num_children` (now `mode_index`) only on the sparse (`scipy.sparse.linalg.
  eigsh`) branch, gated behind `3 * n_atoms > sparse_threshold`; the dense
  (`np.linalg.eigh`) branch, taken for every normally-sized system, had no
  equivalent guard and would instead fail later with a confusing
  `ValueError: could not broadcast input array from shape (X,) into shape (Y,)`
  when slicing eigenvalues shorter than requested. Issue #27's own acceptance
  criterion ("out-of-range... mode_index raises the established
  GBManipulatorValueError") wouldn't be satisfied by copying the existing
  sparse-branch check pattern alone; the fix was a single check placed before
  either branch runs (`num_modes_needed > 3 * n_atoms`), not two branch-local
  copies of it.
- **"Share the pure computation, not the value-typed boundary" (R16's
  `translate_right_grain` precedent) has a second, distinct root cause besides an
  `InterfaceCandidate` leniency conflict: the value type can simply be missing a
  field the new operation genuinely needs.** R18's `AtomInsertion`/`AtomRemoval`/
  `SoftModeDisplacement` all need a unit cell and a GB-region definition;
  `InterfaceCandidate` (R15) carries neither (only `gb_plane_x`, no thickness, no
  unit cell). Rather than growing `InterfaceCandidate`'s schema for three new
  operations (a bigger, cross-cutting change touching every existing operation,
  out of scope for a step whose issue never asks for it), both `unit_cell` and
  `gb_thickness` became required `context.params` entries, and GB-region
  membership is recomputed from the candidate's own atoms/`gb_plane_x`/that
  param rather than threaded through as a value-type field. Unlike
  `translate_right_grain`, there is no leniency conflict here (insertion/removal/
  displacement outputs are always `InterfaceCandidate`-constructible), so the
  legacy methods and new operations diverge only in *how* they obtain GB-region
  membership (`Parent.gb_atoms`/`gb_indices` directly vs. recomputed from
  `gb_thickness`), not in what they compute once they have it -- still enough of
  a divergence to keep the legacy methods reading `Parent` directly rather than
  routing through the new operations' `execute()`.
- **A legacy class can already have two different formulas for "the same"
  derived quantity, split by an internal branch condition (e.g. `grain_ownership
  is not None`); when a new value type is unconditionally in one of those two
  regimes, match that specific formula and say so, don't assume "the" legacy
  formula is singular.** `Parent.__finish_init` computes GB-region membership one
  way when `grain_ownership is None` (via `left_grain`/`right_grain` array masks)
  and another way when it isn't (via `whole_system` indices within
  `gb_thickness / 2` of `gb_plane_x`). Since `InterfaceCandidate` always
  represents persistent explicit ownership, R18's `_gb_region_indices` (in both
  `GBOpt/manipulation/density.py` and `soft_mode.py`) matches only the second
  formula -- disclosed in `REFACTOR_CLEANUP.md` as a real, if narrow, behavior
  difference from the first formula's row order/boundary handling, since nothing
  proves the two are identical for every edge case.
- **A pure function shared between a `ManipulationContext`-validated path and a
  legacy path that accepts test doubles for its RNG must keep its `rng` parameter
  duck-typed (calling only `.choice(...)`), not typed/asserted as
  `np.random.Generator`.** `GBManipulator.rng`'s setter has never enforced a real
  `np.random.Generator` (tests routinely assign a small hand-rolled class exposing
  only `.choice`), while `ManipulationContext.__init__` does enforce it. R18's
  `select_removal_indices`/`select_insertion_sites`/`_random_type_counts` accept
  `rng` as an untyped, duck-typed parameter for exactly this reason: the legacy
  wrapper passes `self.__rng` (possibly a test double), the new operation passes
  `context.rng` (always real), and the shared function must work for both.
- **Re-`Read` the exact block being extracted immediately before extracting it,
  even if it was read earlier in the same session -- intervening edits shift
  every line number below them, and a decorator sitting on the line just above
  a function definition is exactly the kind of thing an off-by-one stale
  offset silently drops.** While extracting R18's `_gaussian` into
  `GBOpt/manipulation/density.py`, an earlier read (taken before a large
  import-block edit had shifted every subsequent line number by ~9) had its
  window start exactly on `_gaussian`'s `def` line, one line below its
  `@jit(float64(float64, float64), nopython=True, cache=True)` decorator --
  which sits on the line the stale offset no longer pointed at. The copy
  omitted the decorator, and since `_gaussian` is called from inside another
  `@jit(nopython=True)`-decorated function, this wasn't cosmetic: numba would
  either fail to compile the caller or silently fall back out of nopython
  mode. Caught by re-reading the source with fresh line numbers right before
  the copy, not after. The fix isn't "read more carefully" -- it's "never trust
  a line-numbered window from earlier in the conversation once anything above
  it has been edited; re-`Read` the block fresh at the moment of extraction."
- **When an extraction changes *where* a "resolve `None` to a concrete
  count/size" step happens relative to a *generator* it used to run inside
  of, check what quantity the resolution is measured against before assuming
  the reordering is a no-op.** R18's first pass at `insert_atoms` moved
  `num_to_insert`'s `fill_fraction`-based resolution from *before*
  `Delaunay_approach`/`grid_approach` ran (the actual pre-R18 control flow: an
  outer check resolves and early-returns on `num_to_insert == 0` using
  `len(gb_atoms)`, so the site generators' own internal, identically-shaped
  resolution logic was dead code, always called with an already-concrete,
  already-nonzero `num_to_insert`) to *after* the new pure site-generator
  functions ran, using `len(possible_sites)` instead -- a different, larger
  number (available empty sites vs. total GB atoms), silently changing what
  `fill_fraction` means. Re-tracing the original call order line by line (not
  just extracting each piece's "obvious" logic in isolation) caught this
  before it shipped; the fix was restoring the original order (resolve from
  `len(gb_atoms)`, early-return, only then generate sites) rather than the
  order that seemed natural once the site generators were pure functions
  returning `(sites, probabilities)` up front.
- **A test's `monkeypatch.setattr(module, "name", ...)` targets the attribute
  as looked up in *the module doing the calling*, not the module the thing
  was originally defined in or imported from -- when extraction moves the
  calling code to a new module, every such patch has to move with it, not
  just ones on directly-imported names.** R18 moved `_create_neighbor_list`/
  `_calculate_local_order` (insertion/removal) and
  `_calculate_bond_hardness`/`_calculate_dynamical_matrix`/`_soft_mode_q_points`/
  `spg` (soft-mode) out of `GBManipulator.py` into `GBOpt.manipulation.density`/
  `soft_mode`. Every existing test that did
  `monkeypatch.setattr(gbmanipulator_module, "_create_neighbor_list", ...)` (or
  patched `gbmanipulator_module.spg.find_primitive`) had to be repointed at the
  new module, because the code that now actually calls these names resolves
  them from its *own* module globals, not `GBManipulator.py`'s -- patching the
  old location becomes a silent no-op, not a test failure, since the mock
  simply never intercepts anything. This is the same "relocate tests during
  decomposition" discipline this file already documents for name-mangled
  private-method access, extended to `monkeypatch`/`unittest.mock.patch`
  targets specifically: grep every `patch(...)`/`monkeypatch.setattr(...)`
  string naming the module being extracted *from*, not just ones naming the
  specific function being moved, since a whole-module reference
  (`importlib.import_module("GBOpt.GBManipulator")` bound to a local variable,
  then `.spg`/`.<name>` accessed off it) won't show up in a plain grep for the
  function name alone.
- **"Share the pure computation, not the value-typed boundary" is not automatic
  just because a legacy method and a new operation coexist -- check whether a
  genuine leniency/schema conflict actually exists before assuming the split is
  needed, and separately, RNG duck-typing is a third, independent reason the
  split can be forced even when no such conflict exists.** R19's `SliceAndMerge`
  (`GBOpt/manipulation/crossover.py`) is the package's first `arity = 2`
  operation, built for issue #79. Unlike R16's `translate_right_grain`
  (an `InterfaceCandidate` leniency conflict) and R18's density/soft-mode
  operations (`InterfaceCandidate` missing `unit_cell`/`gb_thickness` fields),
  every parent `GBManipulator.slice_and_merge` can construct an interface
  candidate for can also be sliced-and-merged into another valid candidate, so
  `SliceAndMerge.execute` constructs and returns a real `InterfaceCandidate`
  unconditionally -- no schema/leniency split was needed at all. The real fork
  was `rng`: existing regression tests assign a duck-typed random-source double
  (exposing only `.random()`) directly to a manipulator's own RNG slot, and
  `ManipulationContext.__init__` unconditionally rejects a non-
  `np.random.Generator` value. So `GBManipulator.slice_and_merge` still calls
  the shared pure function directly with its own parents and RNG, never
  constructing a `ManipulationContext`, but for a type-constraint reason
  distinct from the earlier two cases. Check for *each* of these three reasons
  independently (leniency conflict, missing schema field, non-`Generator` RNG
  test doubles) rather than assuming any one of them by analogy to a prior
  step.
- **A new operation can deliberately deviate from "trust the caller for arity"
  when its own legacy delegate never routes through `apply()`'s generic arity
  check.** Every arity-1 operation's `execute()` trusts `GBManipulator.apply()`
  to have already checked arity, per the `Manipulation` protocol's own
  documented contract, and none of them self-check it. `SliceAndMerge.execute`
  (R19) is the first to self-check (`len(context.parents) != 2` ->
  `ManipulationArityError`), because `GBManipulator.slice_and_merge` never
  routes through `apply()` (see the RNG-duck-typing entry above), so `apply()`'s
  generic arity check never runs for the legacy entry point at all -- without a
  self-check, a directly-constructed `ManipulationContext` with the wrong
  parent count would fail with a plain `IndexError` instead of a named arity
  error. This is a deliberate, disclosed deviation from the established
  "operations trust the caller" convention, not an oversight; expect to make
  the same call again for any future operation whose legacy delegate similarly
  never reaches `apply()`.
- **`ManipulationCompatibilityError` (defined at R15) sat completely unused
  until R19's first two-parent operation needed a parent-vs-parent
  compatibility check** (mismatched ownership mode, boundary topology, or
  non-affine-equivalent physical grain geometry) **-- distinct from
  `ManipulationCapabilityError`, which covers whether an operation can succeed
  given the specific parents/parameters it was handed** (composition/formula
  mismatches, no admissible crossover interval). When a caller-facing
  established exception identity (here, `GBManipulator`'s own
  `CompositionAwareCrossoverError`) needs to keep mapping only to the
  capability-error case and not the compatibility-error case, extend the
  shared translation helper with an optional exception-override parameter
  (`__translate_manipulation_error`'s new `capability_exception` keyword,
  defaulting to `GBManipulatorValueError` for every other existing call site)
  rather than writing a second, near-duplicate translation helper.
- **"Route every operation through the generic `apply()`/`ManipulationContext`
  boundary uniformly" is unsafe for legacy-compatibility execution specifically,
  even when it's the right answer for genuinely new dispatch.** R20 (#80)
  initially planned uniform `apply()`-based dispatch for all `OperationSpec`s,
  including the `choices: list[str]` compatibility adapter's three legacy names.
  This breaks for `translate_right_grain` specifically: `ManipulationContext`
  unconditionally requires the *current* parent to already be
  `InterfaceCandidate`-constructible, but `translate_right_grain`'s own
  established tolerance (R16) exists precisely because a relaxed right-grain
  atom may legitimately have crossed the interface plane, which
  `InterfaceCandidate.__init__` rejects. Since MC/GA reload a fresh manipulator
  from the evaluator's *relaxed* structure output after every accepted step,
  this is not a theoretical edge case -- it is the manipulator's normal state
  between mutations. The fix: the compatibility adapter's dispatch, for all
  three legacy names, calls the established `GBManipulator` methods directly
  (bypassing `ManipulationContext` entirely), exactly matching what those
  methods already did pre-R20; only a spec reached *outside* the compatibility
  adapter (a built-in used directly, or a genuine third-party operation) has no
  legacy tolerance to preserve and safely uses the stricter, uniform boundary.
  The general check for a future step: before routing an *existing* legacy
  code path through a newer, stricter value-typed boundary "for uniformity,"
  verify whether that legacy path's own established behavior already depends on
  a state the newer boundary's constructor would reject -- "uniform dispatch"
  and "uniform execution boundary" are separable, and only the former is
  usually actually required.
- **Not every legacy method that predates a `Manipulation` operation bypasses
  `ManipulationContext` for the same reason, and some don't bypass it at
  all.** Auditing all seven of `GBManipulator`'s legacy per-operation methods
  during R20 found three distinct reasons for bypassing (`translate_right_grain`:
  the leniency conflict above; `slice_and_merge`: RNG duck-typing, R19;
  `insert_atoms`/`remove_atoms`/`displace_along_soft_modes`: missing
  `unit_cell`/`gb_thickness` fields, R18) -- but `cycle_grain_terminations` and
  `apply_interface_separation` already construct a real `ManipulationContext`
  and call their operation's `execute()` today, with no bypass at all. Whether a
  given legacy method is safe to route through the generic boundary is a
  per-operation fact to verify by reading that method's own body, not something
  inferable from "it's a legacy method" or from another operation's own reason.
- **Inserting a new, generic "select among N items" step into a code path that
  historically had no such step (because there was only ever one implicit
  choice) requires proving the new step consumes zero RNG state for the
  N=1 case, not just asserting it by analogy.** R20 generalized GA's
  always-exactly-`slice_and_merge` crossover into a weighted operation pool
  (today, still always one member by default) using the same
  `rng.permutation(n)`-based selection MC's `Mutator` already used. Before
  relying on this for a fixed-seed-history guarantee, this was verified
  empirically (`rng.permutation(1)` leaves `rng.bit_generator.state`
  byte-identical to not calling it at all) and pinned with a dedicated
  regression test -- not assumed from "well it's a single-element permutation,
  that should be a no-op." A different selection algorithm (e.g. weighted
  sampling via `rng.choice(..., p=..., replace=False)`) would *not* have had
  this property even for equal weights, since it uses a different internal
  draw pattern than `rng.permutation`.
- **An issue's "Proposed behavior" prose can ask for *more* instrumentation
  points than its own acceptance criteria require, not just conflict with
  them outright (the R10 `_rebuild()` case) -- the criteria still win, and
  the narrower scope is a deliberate choice to disclose, not an oversight.**
  Issue #81 (R21)'s prose said to "use module loggers for run start, initial
  evaluation, best updates, generation summaries, failures, and termination,"
  but its actual checklist only required MC's `print()` termination messages
  become logger calls and (separately) that logging levels be "appropriate
  for lifecycle summaries versus failure detail" -- satisfiable by the
  termination-message conversion alone, without adding run-start/initial-
  evaluation/best-update/generation-summary logging the prose also mentions.
  R21 instrumented only what the 9-item checklist named, disclosing the
  narrower scope in `REFACTOR_CLEANUP.md` rather than treating the prose list
  as a literal requirement -- the same "acceptance criteria are literal
  constraints" discipline established for #62, just showing up as "the
  criteria are also not a ceiling" this time instead of "not a floor."
- **`GeneticAlgorithmMinimizer`'s docstring claim that it "mirrors the
  interface of `MonteCarloMinimizer`" is about the public call signature, not
  about internal failure-handling shape -- don't assume a fix needed in one
  class's evaluator-boundary code is automatically needed in the other's.**
  Issue #81 (R21) asked that "evaluator exceptions and returned-structure
  reload failures retain exception type/message context before penalty
  application." Grepping `GeneticAlgorithmMinimizer` found three
  `except Exception as exc` recovery-boundary blocks matching this shape
  (evaluator-callback failure and two reconstruction-failure sites, all
  collapsing to `ENERGY_PENALTY`); grepping `MonteCarloMinimizer` for the
  same pattern (`except Exception`, `ENERGY_PENALTY`, `penalty`) found
  nothing at all -- MC's `gb_energy_func` call has no penalty-collapse
  wrapper and any evaluator exception there propagates directly. The
  acceptance criterion was therefore only meaningfully actionable for GA;
  confirmed by grep before writing any fix, not inferred from the two
  classes' shared docstring language or from one class's own fix pattern.
- **A new type's field that mirrors an existing authoritative field must copy that
  field's own validation contract, not a superficially-similar sibling field's.**
  R24's `OptimizationEvent` has both an `iteration` field (an MC step/GA generation
  index, always genuinely non-negative) and an `input_index` field (meant to mirror
  `EvaluationResult.input_index`/`CandidateEvaluation.input_index` exactly). The first
  draft normalized both through the same "non-negative integer" helper, since they
  look like the same kind of thing (a small integer index) -- but
  `CandidateEvaluation.input_index` uses `-1` as an established sentinel for "the
  owned-mode initial candidate, not a submitted population member"
  (`_run_owned_GA`'s `self._owned_evaluator.evaluate_candidate(..., -1)` call), and
  `EvaluationResult.input_index`'s own normalizer (`GBOpt/evaluation/types.py`'s
  `_normalize_input_index`) already accepts any integer, never enforcing
  non-negativity. Reusing the stricter helper meant every owned-mode
  `INITIAL_EVALUATION` event raised `ObservabilityValueError` the instant a real run
  tried to emit one -- caught only by writing a test that actually ran the owned-mode
  initial-evaluation path with a real event sink, not by the type's own unit tests
  (which never happened to construct that specific value). The fix was a second,
  separate normalizer matching the authoritative source's own actual contract, not a
  stricter one chosen because two fields both happen to be called "index." Before
  writing a new validator for a field that mirrors an existing one, check that
  existing field's own normalizer/constructor for exactly what it allows, rather than
  designing from the field's name or its resemblance to a different field on the same
  type.

## Mypy and lint debugging patterns

- **Never restructure working code just to make mypy quiet — including
  `typing.cast()`.** This isn't only about `cast()`; the same trap shows up
  as rewriting a `try: y, z = value / except (TypeError, ValueError)`
  unpack into an `isinstance(value, Sequence) and len(value) == 2` guard, or
  renaming a variable (`repeat_factor` -> `validated_repeat_factor`) purely
  so mypy stops complaining about reusing a parameter name at a different
  inferred type. Both happened in the same `gbmaker/config.py` normalization
  pass and were reverted on request: the guard version was strictly less
  permissive than the try/except it replaced (rejects any 2-unpackable
  non-`Sequence`, e.g. a generator, that the original accepted), and the
  rename changed nothing but which line mypy points its complaint at. Ask,
  for every diff motivated by "mypy was unhappy": would I make this exact
  change if mypy didn't exist? If not, it's the same move as a cast wearing
  a different disguise — don't make it. `cast()` is a runtime no-op
  (`return value`, nothing more) — it only changes what mypy believes, not
  what the code does, so reaching for it is a signal you're fixing the type
  checker's output instead of the code. If mypy can't verify something is
  safe (e.g. a runtime-validated string narrowed to a `Literal`, a
  dynamically-typed `object` parameter), leave the function typed honestly
  (`str`, `object`, etc.) and let mypy report the resulting error rather
  than papering over it with a cast. A real mypy finding is more useful
  left visible than hidden behind a cast that looks like a type guarantee
  but isn't one. This applies whether the cast would be new code
  or already exists — remove it and accept the finding.
- **A mypy finding *count decreasing* after a refactor is not automatically
  suspicious, either — the same "would I make this change if mypy didn't
  exist?" test applies in reverse.** R08 extracted two functions that each
  take a value typed `UnitCell | None` and start with an ordinary
  `if unit_cell is None: raise GBMakerConstructionValueError(...)` guard —
  written because a clear domain error beats a raw `AttributeError` at a
  function boundary, not to silence mypy. That guard happens to let mypy
  narrow the type for the rest of the function and removes several
  `union-attr` findings the old unguarded code had. Verify a decrease the
  same way you'd verify an increase: read the diff and confirm the guard is
  something you'd write anyway, not a cast-shaped workaround wearing an
  `if`-statement's clothes; don't assume "fewer findings" always means
  "coverage was lost" or always means "safe," just that it needs the same
  explanation an increase would.
- **`mypy GBOpt/<subpackage>` follows imports through the parent package's
  `__init__.py`, so it silently pulls in far more than the subpackage.**
  Importing `GBOpt.gbmaker.types` runs `GBOpt/__init__.py` first (parent
  packages always execute on submodule import), which imports `GBMaker` ->
  `GBManipulator`, so `mypy GBOpt/gbmaker` reports "Found N errors in M
  files" where M and N include pre-existing debt from files you didn't
  touch (272 errors across 18 files, for 4 files actually passed on the
  command line, in the R04 session). Don't read the raw summary count as a
  signal of anything; `grep` the output down to the exact paths you changed
  (`| grep "gbmaker\\\\"` on Windows) before comparing against a baseline.
- **A cascade of mypy "Cannot determine type of X" (has-type) errors across
  many unrelated attribute names in one class is a single root cause, not N
  separate bugs.** It happens when a self-attribute is assigned in a large,
  heavily-branching `__init__` and then *read from a different method*
  invoked from within that same `__init__` (e.g. `self.manipulator =
  self._make_initial_manipulator()`, where that method reads
  `self.initial_structure`). Confirmed by reproducing it in isolation on
  `GeneticAlgorithmMinimizer` (28 cascading errors, one root cause). The fix
  is always the same and always safe: add an explicit annotation on the
  self-attribute assignment (`self.x: T = value`) — this bypasses mypy's
  fragile inference entirely and has zero runtime effect. Expect this same
  pattern in other large constructors as later roadmap issues decompose
  `GBMaker`/`GBManipulator`.
- **Fixing a has-type cascade can unmask previously-hidden real findings.**
  mypy treats a value whose type it can't determine as effectively `Any` and
  skips checking operations on it. Once annotated, mypy can check real usage
  and may surface genuinely new-looking errors (e.g. `union-attr` on an
  `X | None` attribute) that were always latently there. These often reflect
  real runtime invariants mypy can't see statically (e.g. "this attribute is
  only accessed from code paths that only run when it's non-None"). Don't
  silently "fix" these with guards/asserts as part of an annotation pass —
  that's a control-flow change, not a mechanical one; flag them separately.
- **A has-type cascade on an *already-annotated-nowhere* attribute can be
  re-triggered just by adding new methods to a class, even when the new code
  never touches that attribute's assignment.** R20 added two new classmethods
  to `GBManipulator` (`_from_interface_candidate` and, on `Parent`,
  `from_interface_candidate`) that don't read or write `self.__rng` at all, yet
  this re-triggered the exact "Cannot determine type of `__rng`" cascade (13
  findings, every method reading it) that R15's own entry on this attribute
  already flagged as reproducing an existing pattern. mypy's fragile inference
  ordering is apparently sensitive to a class body's overall shape, not just to
  which attributes a specific edit touches. The fix is identical to the
  documented one (`self.__rng: np.random.Generator` annotation on its first
  `__init__` assignment) and equally safe, but the trigger condition is
  broader than previously documented: **run a scoped mypy check after adding
  *any* new method to a class with an established has-type-prone attribute,
  even a method that appears completely unrelated to that attribute.**
  `GeneticAlgorithmMinimizer._registry` (a brand-new attribute added the same
  step) triggered the same cascade the ordinary, already-documented way (read
  from a different method than the one that assigns it) — same fix, unrelated
  trigger, confirming the two trigger shapes are genuinely different and both
  worth checking for after any constructor/class-body change.
- **`ruff`'s `B904` (raise without `from` inside `except`) is always safe to
  fix** — it only changes `__cause__`/traceback-chaining metadata, never
  control flow. **`BLE001` (blind `except Exception`) is not automatically
  safe to narrow** — check for an existing precedent first. This codebase
  has a deliberate pattern for evaluator/reconstruction boundaries (e.g.
  `GBOpt/_explicit_ownership_evaluation.py:727,906`): keep the broad except,
  capture `as exc`, add a comment naming it a "deliberate recovery
  boundary," and surface the message (warning or structured failure field)
  instead of narrowing the exception type — narrowing risks missing a real
  failure mode from arbitrary user callback/file-parsing code and crashing
  the whole run instead of degrading one candidate.
- **Before deleting a flagged-unused variable, check whether a sibling
  implementation uses the analogous one for something real.**
  `_last_completed_gen` in `GeneticAlgorithmMinimizer.run_GA` looked dead,
  but `MonteCarloMinimizer.run_MC` has the identically-shaped
  `_last_completed_step`/`_early_exit` pair that's genuinely used to
  guarantee exactly one final checkpoint commit — GA's version is a stub for
  the not-yet-built convergence criterion (#41), not dead code. Also check
  for multiple same-named variables in different scopes in the same file
  before deleting (grep the exact assignment pattern, not just the name) —
  one occurrence can be genuinely dead while another with the same name
  elsewhere is live.
- **A new dataclass field's `None` "not yet initialized" placeholder default
  is new mypy debt you fully control, and fixing it isn't mypy appeasement.**
  R10's `_BoundaryState`/`_AssembledResult` (the new `GBMaker` state
  containers) initially typed several fields `X | None = None` as transient
  placeholders, filled in during `__init__` before any getter could
  reasonably read them. mypy doesn't know that invariant, so every
  downstream read (`self._boundary.x_dim`, `self._result.atoms["x"]`, ...)
  became a new `union-attr`/`index`-family finding that didn't exist for the
  flat, unannotated attributes these containers replaced — a real increase
  against the pre-refactor baseline, not a wash. Unlike the has-type-cascade
  fix above (where the *right* answer is an honest annotation and accepting
  whatever mypy then reports), here the fields are genuinely never `None` in
  practice and the type is under this session's own control: giving them
  concrete placeholder defaults instead (`0.0`, `np.empty(...)`) eliminated
  the new findings without hiding anything, since it's not a claim about an
  external invariant mypy can't see — it's just not lying about optionality
  that was never real. Applying the same fix to `_MakerConfig.unit_cell`
  (typed concrete `UnitCell`, not `UnitCell | None`) additionally *removed*
  8 of `GBMaker.py`'s pre-existing findings as a side effect, once every
  downstream `.unit_cell.<attr>` access no longer needed a union-attr guard
  — verified as a legitimate improvement (would make this change regardless
  of mypy) per the "count decreasing isn't automatically suspicious either"
  rule above, not asserted away. The one field that *is* legitimately
  `X | None` forever (`_BoundaryState.embedding` — the legacy misorientation
  construction path genuinely has no embedding) was left `| None`, since
  that's a real invariant, not a transient-placeholder one; every call site
  already null-checks it. The distinguishing question for any `| None`
  dataclass field: is this optional forever, or just before `__init__`
  finishes? Only the first case should stay `| None`.
- **New code that deliberately mirrors an existing method's pattern will
  reproduce that method's pre-existing mypy findings verbatim at the new call
  sites — this is not new debt to fix or guard away.** R15's
  `GBManipulator.__current_parent_candidates()` calls
  `parent._to_interface_candidate(...)` on each of `self.__parents`'s
  elements, the same way `make_parent_candidate()` already does for a single
  parent. `self.__parents` is typed as a list that can hold `None`, so both
  methods get the identical `"None" has no attribute ...` family of findings
  from mypy — `make_parent_candidate()` already carried 7 of these before
  R15; the new method reproduces 6 more of the exact same shape (verified:
  `mypy GBOpt/manipulation` reports 202 `GBManipulator.py` findings after R15
  versus 196 before, and the +6 are all at the new method's lines, matching
  the existing 7's error text). Verify a reproduced-pattern increase by
  diffing finding *text*, not just counting — if the new findings are the
  same message shape as an existing, unguarded occurrence of the same
  pattern, that's confirmation it's the same root cause surfacing again, not
  a fresh regression to cast or guard away (the established occurrence isn't
  guarded either, so guarding only the new one would be inconsistent, not a
  fix). Disclose the count increase in `REFACTOR_CLEANUP.md`.
- **Extracting a method body out of a class and into a pure function taking
  properly-typed parameters can eliminate a whole cascade of mypy
  `[attr-defined]` "`None` has no attribute ..." findings as a legitimate side
  effect, when the extracted body no longer reads attributes directly off a
  loosely-typed `Optional` container.** R19 moved `slice_and_merge`'s body
  (which read `.grain_labels`/`.coordinate_tolerance`/`.box_dims`/etc. directly
  off `self.__parents[0]`/`[1]`, typed `list[Parent | None]`) into
  `crossover_slice_and_merge`, which reads the same attributes through its own
  `CrossoverParent`-protocol-typed parameters instead -- `GBManipulator.py`'s
  own `slice_and_merge` now only *passes* `self.__parents[0]`/`[1]` through,
  unread. `mypy GBOpt/manipulation` confirmed `GBManipulator.py`'s findings
  dropped 148 -> 113, with the entire 35-finding difference in the
  `[attr-defined]` shape and every other error-code count identical before and
  after. Verify a decrease exactly like this by comparing per-file, per-error-
  code counts (not just totals) before and after, same as the increase-
  verification discipline above -- a full-category-code match on everything
  except the one shape the extraction should have affected is what confirms
  the decrease is a real, expected consequence of the extraction rather than
  a symptom of newly-hidden `Any`-typed values.
- **The has-type-cascade trigger ("assigned in `__init__`, read from a
  different method") is specifically about *methods*, not about nested
  closures defined and called inside the same method as the assignment.**
  R21 added `self.seed` to both `MonteCarloMinimizer.__init__` and
  `GeneticAlgorithmMinimizer.__init__` the same way (`self.seed: int =
  int(time()) if seed is None else seed`, right next to the existing
  `self.local_random` assignment). Only the GA one triggered the cascade
  (`Cannot determine type of "seed"`, 4 findings) -- `self.seed` there is
  read from `run_GA`, a separate top-level method. `MonteCarloMinimizer`
  reads `self.seed` only from inside `_build_state`/the checkpoint-restore
  block inside `run_MC` itself -- nested closures and code that all live
  inside the same method body as the `__init__` assignment don't count as
  "a different method" for mypy's inference-ordering purposes, and running a
  scoped check confirmed zero new findings there. The fix for the GA case
  was the documented one (`self.seed: int = ...` annotation); the general
  check from before still holds -- run a scoped mypy check after adding any
  new attribute -- but don't assume every new self-attribute needs the
  annotation defensively just because a sibling class with a similar
  assignment needed it; verify per class.
- **When diffing mypy finding *counts* against a baseline via `grep -c` on
  file-path lines, keep the exact same grep pattern for both sides, or a
  message that happens to also start with the file path (a `note:`
  continuation line, e.g. PEP 484 implicit-Optional guidance) gets counted
  inconsistently between runs and produces a false-looking delta.** R21's
  first baseline check for `monte_carlo.py` used `grep "monte_carlo.py\|
  genetic.py\|Found"` (a content filter, so `note:` lines matched too) and
  read the result informally as "3 errors" by eyeballing distinct line
  numbers; a later, more careful `grep -c "^GBOpt.*monte_carlo.py"` pass
  counted 5 (the 3 errors plus 2 `note:` continuation lines for the same
  `no_implicit_optional` finding) -- not a regression, just two different
  countings of the same unchanged baseline. Re-running the identical
  `grep -c` command against both the pre-change and post-change state (via
  `git stash` to get a true apples-to-apples baseline in the same file/
  environment, rather than trusting an earlier informal read) confirmed the
  real count was unchanged (5 -> 5, 45 -> 45 after the has-type-cascade fix
  above). Always take the numeric baseline from the same exact command
  you'll re-run afterward, not from a different, more readable invocation
  used only to inspect what the findings are.
- **mypy cannot narrow an `X | None` value through a derived `bool` flag
  computed in a separate statement, even when the flag is provably
  equivalent to an `is not None` check -- narrow through a typed local
  instead.** R22's first draft of `GBOpt/evaluation/adapters.py` computed
  `has_path = isinstance(structure_path, str) and bool(structure_path.strip())`
  and then branched on `if numeric_energy is None or not has_path:` before
  using `structure_path` (typed `Any | None` from a `dict.get(...)`/tuple
  unpack) in the success path below. mypy has no way to know `has_path`
  being `True` implies `structure_path` is a non-`None` `str` -- it isn't an
  `isinstance`/`is None` check on the value itself, so the success branch's
  `StructureArtifact(path=structure_path, ...)` still reported `path` as
  possibly `None`/`Any`. The fix was a small helper,
  `_normalize_structure_path(value) -> str | None`, returning the validated
  string or `None`; branching on `if ... or structure_path is None:` against
  its own typed return value let mypy correctly narrow `structure_path` to
  `str` in the success fallthrough, with no behavior change and no cast.
  Same root idea as this file's existing `UnitCell | None` guard-clause
  entry above (a domain check mypy can verify beats one it can't), just
  showing up as "restructure the boolean into the value's own type" instead
  of "add a `None`-check guard clause."

## Tooling

`ruff`, `mypy`, `bandit`, and `pyscn` configs (`pyproject.toml` `[tool.ruff]`,
`mypy.ini`, `.pyscn.toml`) exist only on the fork-local
`tooling/lint-typecheck-experiment` branch — they're an experiment not yet
proposed upstream, so don't add them to a refactor PR branch. To run the
tools against another branch, copy the three config files in temporarily,
run the tools, then delete them (`rm pyproject.toml mypy.ini .pyscn.toml`)
before staging/committing.

`git show <branch>:.pyscn.toml > .pyscn.toml` (a dotfile target) can fail in
Git Bash on Windows with `fatal: ambiguous argument
'tooling\lint-typecheck-experiment;.pyscn.toml'` — MSYS mangles the
`branch:.path` argument because it looks path-like. `pyproject.toml` and
`mypy.ini` aren't dotfiles and don't hit this. Prefix the command with
`MSYS_NO_PATHCONV=1` (or use `git cat-file -p <branch>:.pyscn.toml >
.pyscn.toml`) for the `.pyscn.toml` copy specifically.

Setting up a scratch venv to run tests (`setup.py` currently needs numpy
importable at *build* time, and no `[project]`/build-system table exists on
non-tooling branches):

```bash
python -m venv .venv-scratch
source .venv-scratch/Scripts/activate   # Windows Git Bash
python -m pip install -q "numpy<=2.1" scipy numba pandas matplotlib spglib pytest setuptools
python -m pip install -q --no-build-isolation -e .
```

## Cloud session environment notes

The sections above (Tooling's venv recipe aside, and all of Editor setup and
Git mechanics below) describe the user's own local Windows machine. A Claude
Code on the web / cloud-container session is a different environment with its
own quirks, first hit during R22:

- **The container's `python3`/`python` can resolve to an older interpreter
  than the one this codebase needs for its own syntax.** R22's container had
  `python3` at 3.11.15 and `python3.12` also on `PATH`; `UnitCell.py` uses a
  nested-same-quote f-string that's only valid syntax from Python 3.12
  onward (PEP 701). Creating the scratch venv with plain `python3 -m venv`
  produces a 3.11 venv that fails to even install the package
  (`SyntaxError: f-string: expecting '}'` from `setup.py`'s own import of
  `GBOpt`), and separately, a `mypy`/`bandit` installed globally via `uv tool
  install` (found on `PATH` already) runs under whatever Python `uv`
  provisioned it with — also 3.11 in this container — and hits the exact
  same parse error the instant it tries to follow an import into
  `UnitCell.py`, aborting with zero findings rather than reporting one.
  Check `python3.12 --version`/`ls /usr/bin/python3.1*` up front, create the
  scratch venv with `python3.12 -m venv` explicitly, and `pip install mypy
  bandit` *into that same venv* rather than trusting whatever `mypy`/
  `bandit` binary is already on `PATH` — `ruff` doesn't have this problem
  (its own Rust parser handles the syntax regardless of host Python), but
  mypy and bandit are only as new-syntax-capable as the interpreter running
  them.
- **`bandit` and `pyscn` are not preinstalled in a fresh cloud container**,
  unlike the user's machine (where `pyscn` is already on `PATH` and the
  Tooling section's `pyscn` MCP server may be installed). Both are
  `pip install`-able (`pip install bandit pyscn` — the `pyscn` PyPI package
  provides the same CLI the Tooling section describes as a fallback). Check
  `ToolSearch` for `mcp__plugin_pyscn-mcp_pyscn-mcp__*` first as usual, but
  expect it to be genuinely absent (not just unloaded) in a fresh cloud
  session and fall back to the CLI without spending time trying to install
  the plugin.
- **This repo's CRLF convention (see Git mechanics below) is a property of
  the user's local git config, not of the repository itself — a fresh cloud
  container has no reason to reproduce it.** `git config --get
  core.autocrlf`/`core.safecrlf` are unset in a fresh container, there's no
  committed `.gitattributes` forcing CRLF, and `git check-attr` on a tracked
  `.py` file confirms `eol: unspecified`. The repository's actual git blobs
  are LF (git stores `core.autocrlf=true` content as LF and only converts to
  CRLF on checkout on a machine configured to do so); a cloud container
  without that config checks files out as the plain LF they're stored as, and
  `Write`/`Edit` output (also LF) matches that natively. Applying Git
  mechanics' CRLF-conversion recipe in a cloud session would be solving a
  problem the container doesn't have, and would risk introducing the exact
  mismatched-line-ending state that section warns against. Confirm with
  `git config --get core.autocrlf` before assuming either convention applies.
- **A same-named repository can't be attached twice to one cloud session, and
  a separately-spawned helper cloud session isn't a reliable way around
  it.** See the "Issues and pull requests are disabled on this fork" entry
  under Roadmap context above for the concrete case (needing
  `IdahoLabResearch/GBOpt`'s issue tracker while already working in
  `jarinfrench/GBOpt`, both of which check out to a `gbopt`-named directory).

## Editor setup

`.vscode/settings.json` is tracked and wires the Ruff and Mypy Type Checker
extensions to `pyproject.toml`/`mypy.ini`. `.vscode/extensions.json`
(recommended extensions, including `PyCQA.bandit-pycqa` for bandit) and
`.vscode/tasks.json` (tasks for a full-repo ruff/mypy/bandit/pyscn run) exist
locally but are **not committed** — the user's global `~/.config/git/ignore`
excludes `.vscode/`, and `settings.json` is only tracked because it predates
that rule. Don't try to `git add` new files under `.vscode/` without asking;
recreate `extensions.json`/`tasks.json` locally if they're missing rather
than assuming they should be committed.

`.vscode/settings.json` now has format-on-save wired to Ruff. That means the
user's editor can reformat a whole file (collapsing parens, line-break
style, etc.) the moment they open/save it — independent of and invisible to
whatever Claude last `Read`. This already happened once: a full-file `ruff
format` pass got silently bundled into a commit meant to be four targeted
except-clause edits, because the file had been reformatted on disk between
the `Read` and the `Edit`/commit. Before committing any edit to a file the
user might have had open, diff the actual change against the last known-good
state (`git diff`, or `git show HEAD~1:<path>` vs. the working copy) rather
than trusting that only your own edits are present — don't assume `ruff
format --check` passing after your change means your change alone caused it.

The pyscn MCP server (`ludo-technologies/pyscn`, plugin
`pyscn-mcp@pyscn-marketplace`) is installed and available as
`mcp__plugin_pyscn-mcp_pyscn-mcp__*` tools (`get_health_score`,
`check_complexity`, `detect_clones`, `find_dead_code`, `check_coupling`,
`check_cohesion`, `check_di_antipatterns`, `analyze_code`). **Use these directly for pyscn checks instead of shelling out through a
scratch venv.** Setup was on the user's side, not Claude's: install the CLI
(`npm install -g @anthropic-ai/claude-code`), then
`claude plugin marketplace add ludo-technologies/pyscn` and
`claude plugin install pyscn-mcp@pyscn-marketplace`; the MCP tools then show
up automatically in-session. ruff/mypy/bandit still need the scratch-venv
route (see above) unless/until an equivalent MCP is set up for them.

The `mcp__plugin_pyscn-mcp_pyscn-mcp__*` tools are not guaranteed to be loaded
in every session (missing entirely in the R07 session despite being installed
per the setup above -- `ToolSearch` for the tool names found nothing). Check
for them first; if unavailable, fall back to the `pyscn` CLI directly (already
on `PATH`, confirmed via `pyscn check <paths>`), which reports the same kind
of signal (SLOC-length "too long" findings plus clone detection) in one
command, no scratch-venv needed. Its output format differs from the MCP
tools' (a flat list of findings, not separate health/complexity/dead-code
calls), so don't expect an apples-to-apples count against a baseline recorded
with the MCP tools -- diff against the same CLI command run on the pre-change
files instead (`git show <branch>:<path> > <tmp>` per a file, run `pyscn
check` on both sets, compare).

## Git mechanics

This repo uses CRLF line endings (`core.autocrlf=true`, `core.safecrlf=true`).
Files written by tools default to LF-only and `git add` will refuse them
("LF would be replaced by CRLF"). Convert before staging:

```bash
python -c "
p = 'path/to/file.py'
data = open(p, 'rb').read()
data = data.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
open(p, 'wb').write(data)
"
```

**`git show <branch>:<path> > <file>` redirection outputs LF-only content**,
even for a file that's CRLF in the working tree -- the redirect bypasses the
checkout filters that `core.autocrlf` normally applies, and `git show` prints
the blob exactly as git stores it (LF). This matters whenever a branch's file
is pulled in for comparison (a pre-refactor baseline, a temp file to diff
against): treat it as needing the same `\r\n`-conversion snippet above before
using it as a real working file, and don't conclude a diff is real just
because `git diff` against the *committed* version looks clean immediately
after a raw copy -- git's own diffing already normalizes CRLF/LF, so it can
mask a genuinely mixed-line-ending file that later corrupts under a
naive script.

**Never replace a large block of a CRLF file with a raw Python
read/split/join script -- use the Edit tool, even for an awkward multi-hundred
-line block.** This bit R07: splitting a CRLF file's bytes on `b'\n'` leaves a
trailing `\r` on every line (the split consumes only the `\n`), so rejoining
with `b'\r\n'` doubles that `\r` into `\r\r\n` for every line touched. A
follow-up "fix" pass that blindly does
`data.replace(b'\r\n', b'\n').replace(b'\r', b'\n')` compounds this: the
second `.replace(b'\r', b'\n')` also eats the `\r` out of valid `\r\n` pairs
elsewhere in the same normalization pass, so re-emitting with
`.replace(b'\n', b'\r\n')` turns every one of those into a spurious extra
blank line. The result is syntactically valid Python (blank lines carry no
meaning), so **the test suite passes with no sign anything is wrong** -- this
was only caught by `pyscn check`'s SLOC-length findings showing methods far
longer than their pre-change size, and confirmed by directly reading the file.
If a script-based block replacement is genuinely unavoidable, split and join
on `b'\r\n'` consistently (never `b'\n'` alone) and verify immediately
afterward with a lone-CR/lone-LF regex scan
(`re.findall(rb'\r(?!\n)', data)` and `re.findall(rb'(?<!\r)\n', data)` should
both be empty) -- do this before running anything else against the file, not
just before staging.

**A blind whole-file `str.replace()` for a "mechanical" rename can silently
corrupt a longer identifier that contains the search string as a substring.**
R10's `positive=True` -> `nonnegative=True` rename (a verified-safe, purely
mechanical keyword rename across `GBMaker.py`'s `__validate` call sites) was
done with `data.replace('positive=True', 'nonnegative=True')` across the
whole file. `"positive=True"` is also a substring of `"strictly_positive=True"`
(a different, unrelated keyword this same file uses), so the blind replace
turned it into `"strictly_nonnegative=True"` -- a parameter name that doesn't
exist, silently changing what `__validate` was called with. Caught
immediately by re-grepping every remaining `positive` occurrence in the file
right after the replace (the same "verify the actual diff, don't trust that
a mechanical script did only what you intended" discipline this section
already asks for CRLF scripts), not by the test suite, which would have
failed loudly here (`TypeError: unexpected keyword argument`) but wouldn't
have for a subtler substring collision. Prefer the `Edit` tool's
`old_string`/`new_string` matching (which requires the surrounding context to
be unique and won't touch an unrelated superstring) over a raw
find-and-replace script for renames in code, even ones that look too simple
to need it; if a script is genuinely faster for a large number of call sites,
grep every occurrence of the search string immediately after, not just spot
checks, before staging.

**Every branch should be committed before something branches off of it --
if the current branch has uncommitted changes when a new step needs to
branch from a different point, warn the user explicitly before doing
anything about it, rather than silently stashing (or committing) as a
workaround.** Starting R15 found `refactor/r14-candidate-loader` checked out
with uncommitted `CandidateLoader.py`/`FileGrainOwnership.py` changes, but
R15 needed to branch from `refactor/r13-parent-from-structure` instead. The
uncommitted state was stashed to preserve it without asking first; the user's
follow-up correction was that this situation should be surfaced to them
before acting on it, not handled quietly -- run `git status --short` on the
branch about to be left behind before any `git checkout -b <new> <parent>`
and flag anything relevant (ignore known scratch dirs like `.venv-scratch/`),
letting the user choose whether to commit it, stash it, or something else.
If stashing is the chosen path, `git stash push -u -- <specific real files>`
(naming the actual changed/untracked source and test files, not a blanket
`-u`) is the safe form -- a blanket `git stash push -u` can fail outright if
an untracked scratch dir's own files trip the same "LF would be replaced by
CRLF" refusal `git add` gives above, since `git stash` applies the same
checkout filters. A stash stays attached to the branch it was taken from and
can be popped when that earlier step is picked back up; it is not a
substitute for actually committing that step once it's ready.

**A brand-new file created with a plain file-write tool (not `Edit` on an
existing CRLF file) is LF-only from the moment it's written, same as any
`git show`-sourced blob -- convert it before `git add`, not just files that
went through a redirect or a script.** R18's two new modules
(`GBOpt/manipulation/density.py`, `soft_mode.py`) were written this way and
`git add` refused both with "LF would be replaced by CRLF," identical to the
`git show` case this section already documents. Since a freshly-written file
has no existing `\r` bytes at all, the safe fix is the same one-line global
conversion already used elsewhere here (`data.replace(b'\n', b'\r\n')`) --
confirm `data.count(b'\r') == 0` first so the conversion can't double up an
existing `\r\n` into `\r\r\n`, exactly like the mixed-line-ending corruption
case above. Do this as a matter of course for every new file this session
creates in a CRLF repo, not only when `git add` first complains.
