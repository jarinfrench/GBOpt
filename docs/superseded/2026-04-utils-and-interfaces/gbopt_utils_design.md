> **SUPERSEDED**
>
> Replaced by the owning-layer structure in [`../../MASTER_PLAN.md`](../../MASTER_PLAN.md) and [`../../architecture.md`](../../architecture.md). Its feature inventory remains historical input, but the proposed broad `utils` package must not be implemented as written.

# GBOpt `utils` Subpackage — Design Document

**Status:** Draft  
**Author:** Jarin French
**Last Updated:** 2026-04-30

---

## 1. Motivation

Several tools in GBOpt are more general than the classes they currently live in, and a growing body of external scripts has accumulated useful functionality that belongs inside the package. Rather than continuing to spread utilities across `GBMaker`, `GBManipulator`, `GBMinimizer`, and user-side scripts, this design introduces a dedicated `utils` subpackage to consolidate them.

Goals:
- Reduce duplication between the core classes and external scripts
- Make common operations (I/O, geometry math, plotting) accessible without instantiating a full class
- Provide first-class support for objective function templates (callables) that users can pass to `GBMinimizer`
- Keep the codebase readable and individually testable by area

Non-goals:
- This is not a rewrite of the core classes
- `utils` should not contain business logic specific to a single class (that stays where it is)

---

## 2. Proposed Structure

```
GBOpt/
└── utils/
    ├── __init__.py              # Curated re-exports of commonly used items
    ├── io.py                    # File I/O helpers (LAMMPS data, dump, XYZ, CIF, etc.)
    ├── geometry.py              # Rotation matrices, Miller indices, misorientation math
    ├── validation.py            # Shared argument checking and sanity assertions
    ├── plotting.py              # Reusable matplotlib/publication figure helpers
    └── objective_templates.py   # Pre-built scalar objective callables for GBMinimizer
```

---

## 3. Module Specifications

### 3.1 `io.py`

Handles reading and writing of atomistic structure file formats. Any format
that more than one core class might touch should live here rather than being
buried inside a single class.

**Candidates for inclusion:**
- [ ] LAMMPS dump file reader *(currently in package)*
- [ ] LAMMPS data file reader/writer *(currently in package)*
- [ ] XYZ reader/writer *(planned per manuscript)*
- [ ] CIF reader/writer *(planned per manuscript)*
- [ ] POSCAR/CONTCAR reader/writer *(not in manuscript but confirmed in scope)*

**Design notes:**
- Functions should be stateless (plain functions, not methods)
- I/O functions should accept and return `Parent` objects (or the specific
  attributes they need from one) — `Parent` is the canonical structure type
  per the interface design audit (see `gbopt_interface_design.md` §4)
- File format detection (by extension or header sniffing) is acceptable as a
  convenience wrapper, but explicit-format functions should also exist

**Open questions:**
- Are there any formats currently handled by external scripts that should be
  pulled in here? **Answered: None currently** — `io.py` starts fresh from
  the formats already in the package.
- Should there be a unified `Structure` data container, or keep raw arrays?
  **Answered:** Use `Parent` — see `gbopt_interface_design.md` §4.

---

### 3.2 `geometry.py`

Pure math and geometry operations relevant to grain boundary crystallography.
Nothing here should require file I/O or LAMMPS.

**Candidates for inclusion:**
- [ ] Rotation matrix construction (axis-angle, Euler angles, etc.)
- [ ] Misorientation angle calculation
- [ ] Miller index utilities (normalization, family equivalence, zone axis)
- [ ] CSL/DSC lattice helpers (Sigma value, O-lattice)
*The following are extracted from `GBMaker` (all currently private); each qualifies
because it performs no file I/O, makes no LAMMPS calls, and does not mutate `self`:*
- [ ] `reduce_integer_row(row: np.ndarray) -> np.ndarray` — reduce an integer row vector by its GCD *(currently `GBMaker.__reduce_integer_row`, static)*
- [ ] `row_angle_error_deg(reference: np.ndarray, candidate: np.ndarray) -> float` — angular error in degrees between two vectors *(currently `GBMaker.__row_angle_error_deg`, static)*
- [ ] `approximate_rotation_row_as_int(row: np.ndarray, angle_tol_deg: float, max_scale: int) -> np.ndarray` — smallest-norm integer vector within a given angle tolerance of a float row *(currently `GBMaker.__approximate_rotation_row_as_int`)*
- [ ] `approximate_rotation_matrix_as_int(m: np.ndarray, precision: float) -> np.ndarray` — approximate a float rotation matrix as an integer matrix row-by-row *(currently `GBMaker.__approximate_rotation_matrix_as_int`)*
- [ ] `scaled_periodic_basis_vector(period_vector: np.ndarray, box_length: float, axis_index: int) -> np.ndarray` — scale a periodic basis vector so its projection on one axis matches a target box length *(currently `GBMaker.__scaled_periodic_basis_vector`)*
- [ ] `cartesian_from_box_coordinates(box_coordinates: np.ndarray, box_basis: np.ndarray) -> np.ndarray` — convert mixed box coordinates `[x_cart, u_y, u_z]` to Cartesian *(currently `GBMaker.__cartesian_from_box_coordinates`)*

**Design notes:**
- All functions should operate on NumPy arrays
- Functions with non-obvious math should include a docstring citing the source
  formula or reference (paper, textbook)
- Consider grouping related functions with a comment block rather than
  sub-submodules (keep it flat unless it grows very large)

**Open questions:**
- Which of these currently live in `GBMaker` as methods and can be cleanly
  extracted?
- Are there geometry helpers in external scripts not yet in the package?
  **Answered: No**

---

### 3.3 `validation.py`

Centralized argument checking and sanity assertions. Keeping these out of the
core classes makes error messages consistent and the classes themselves easier
to read.

**Candidates for inclusion:**
- [ ] Miller index validator (integer, non-zero, correct length)
- [ ] Rotation matrix validator (orthogonality, determinant = +1)
- [ ] Lattice parameter validator (positive, physically reasonable)
- [ ] (others — to be identified during audit)

**Design notes:**
- Validators should raise `ValueError` or `TypeError` with descriptive messages
- Consider a consistent signature: `validate_<thing>(value, name="<thing>")` so
  that the argument name can appear in error messages
- These should be pure functions — no side effects, no logging

**Open questions:**
- Are there validation patterns currently duplicated across `GBMaker`,
  `GBManipulator`, and `GBMinimizer` that should be unified here?

---

### 3.4 `plotting.py`

Reusable matplotlib helpers for publication-quality grain boundary figures.
Avoids re-implementing the same axis formatting, colormap setup, and inset
logic in every notebook or script.

**Candidates for inclusion:**

*Tier 1 — Core (implement first; directly answer "did it work?"):*
- [ ] GBE vs. MC step — primary MCMC convergence view
- [ ] Minimum GBE vs. generation number — primary GA convergence view
- [ ] Cumulative best energy vs. step/generation (explicitly monotonic) — clean comparison baseline
- [ ] Best energy vs. number of objective evaluations, MC and GA on the same axes — fairest algorithm comparison
- [ ] γ-surface / grain boundary energy heatmap

*Tier 2 — Algorithm diagnostics (reveal why it worked or didn't):*
- [ ] Rolling acceptance rate vs. step — dropping rate → stuck; high rate → moves too small
- [ ] ΔE distribution: histogram of accepted vs. rejected proposed moves — diagnoses proposal scaling
- [ ] Population energy spread per generation: box or violin plot — narrowing → premature convergence
- [ ] Best / mean / worst energy band (ribbon) per generation — fuller picture than minimum alone
- [ ] Move type breakdown: acceptance rate per manipulator type — shows which operators are productive
- [ ] Improvement per step/generation: Δ(best energy) per iteration — shows diminishing returns

*Tier 3 — Advanced / multi-run (high value when you have the data):*
- [ ] Hitting time distribution: evaluations to reach a target energy threshold (CDF over multiple runs)
- [ ] Probability of improvement: fraction of steps/generations that beat the current best
- [ ] Energy distribution histogram/KDE: all energies ever sampled, MC vs. GA — breadth of exploration
- [ ] Energy variance vs. generation — scalar diversity proxy

*Tier 4 — Exploratory / research-grade (implement when specifically needed):*
- [ ] Energy landscape sampling density: 2D scatter/histogram of (step, energy) colored by acceptance
- [ ] Autocorrelation of MC energy — effective mixing length (independent-sample interval)
- [ ] Energy vs. structural change magnitude: ΔE vs. ||modification|| — diagnoses move size calibration
- [ ] Lineage / genealogy tree: which parents produced the eventual best individual
- [ ] Lineage survival: number of unique ancestors vs. generation — dominance vs. diversity
- [ ] Best energy vs. wall-clock time — meaningful for HPC cost comparisons

*Shared infrastructure (needed across multiple plot types):*
- [ ] Standard colormap/palette setup (`cmcrameri` wrappers)
- [ ] Inset axis creation helpers
- [ ] Reusable LaTeX axis label constants (GBE units, step labels, etc.)

**Design notes:**
- Functions should accept an optional `ax` argument (matplotlib `Axes`) so
  callers can place plots into their own figure layouts
- Avoid hardcoding figure sizes or DPI — accept as parameters with sensible
  defaults
- LaTeX label strings that are reused across plots should be defined as
  module-level constants rather than duplicated per function

**Open questions:**
- Which Tier 1/2 plots already exist in notebooks or scripts and should be
  extracted first? **Answer:** There are several example scripts in the examples directory (plot_\*.py, analyze_\*.ipynb)

---

### 3.5 `objective_templates.py`

Pre-built scalar objective callables (and factories) that users can pass
directly to `GBMinimizer`. The manuscript is explicit that the minimizer
accepts any callable mapping atomic positions to a scalar — not just energy.
This module provides ready-made implementations for the common cases.

**Candidates for inclusion:**
- [ ] A ready-to-use LAMMPS-based GB energy objective (standard EAM/MEAM
  workflow — the most common case)
- [ ] Factory function: `make_lammps_objective_fn(potential_path, elements, ...)`
  returns a configured callable
- [ ] (other objectives — VASP single-point, ML potentials, user-defined
  property proxies such as sink strength or migration barrier)

**Design notes:**
- The public interface is a **callable** matching the signature `GBMinimizer`
  already expects — no changes to `GBMinimizer` required for basic use
- Once the `Calculator` ABC is introduced (see `gbopt_interface_design.md`),
  this module transitions: templates return `BaseCalculator` instances rather
  than bare callables. The two are backward-compatible if the minimizer
  accepts either (see the migration plan in the interface design doc).
- Factory functions are preferred over classes for this: simpler to use and
  easier to document
- Each template should validate its own inputs at construction time (fail fast)
  rather than at the first call
- Templates should be clearly documented with what external setup they require
  (potential file format, LAMMPS unit style, etc.)
- Reference examples: `examples/run_ga.py` and `examples/run_mc.py`
  (submitted via `examples/submit.py`) show the current structure of objective
  function usage and are the primary source for extracting the first templates.

**Callable signatures (confirmed by audit of `examples/run_mc.py` and `examples/run_ga.py`):**

*Single-evaluation callable* — required by both `MonteCarloMinimizer` and
`GeneticAlgorithmMinimizer`. In both examples a `functools.partial` is used to
pre-fill LAMMPS-specific parameters, leaving the minimizer-facing signature as:

```python
def objective(
    GB: GBMaker,
    manipulator: GBManipulator,
    atom_positions: np.ndarray,
    unique_id: str,
    **kwargs,               # forwarded by MC; ignored by GA's _evaluate_generation
) -> tuple[float, str]:
    """Returns (objective_value, path_to_output_dump_file)."""
```

*Batch-evaluation callable* — optional GA-only extension (`gb_batch_energy_func`).
Effective signature after `partial` application:

```python
def batch_objective(
    gb: GBMaker,
    manipulators: list[GBManipulator],
    candidates: list[np.ndarray],
    lineages: list[list[str]],
    unique_ids: list[str],
) -> list[dict]:
    """Each dict must contain at least {"energy": float, "final_dump": str | None}.
    Optional keys: "num_atoms", "parents", "status", "fail_reason"."""
```

**LAMMPS process management (Q6 — confirmed by audit):**

Both examples manage LAMMPS process lifetime **entirely inside the callable**.
There is no external setup/teardown.

- `run_mc.py` (`get_gb_energy`): spawns LAMMPS via `subprocess.run(...)`,
  blocks until it exits, then parses the results file. One new process per call.
- `run_ga.py` (`evaluate_batch`): submits one SLURM job per candidate, then
  calls `wait_for_jobs(...)` inside the same function call to block until all
  finish. Each LAMMPS run is a stateless one-shot job.

Templates in `objective_templates.py` should follow the same pattern: own the
LAMMPS lifecycle (write input → launch → wait → parse → return). The
`setup()`/`teardown()` hooks in `BaseCalculator` (see `gbopt_interface_design.md`
§3.5) are the future path for persistent-process calculators; they are not needed
for the initial templates.

**Common patterns to extract as the first templates (confirmed by audit):**

1. `make_lammps_subprocess_objective(lmp_binary, input_script, n_threads, ...)` —
   wraps the `run_mc.py` pattern: single synchronous `subprocess.run` call per
   evaluation. Suitable for local or single-node runs.
2. `make_lammps_slurm_batch_objective(lmp_binary, input_script, slurm_cfg, ...)` —
   wraps the `run_ga.py` pattern: submits a SLURM array, waits for all jobs,
   collects results. Returns a batch callable suitable for `gb_batch_energy_func`.

---

## 4. `__init__.py` — Public API Surface

The `utils/__init__.py` re-exports the most commonly used items so users do not
need to know which submodule something lives in.

```python
# GBOpt/utils/__init__.py  (draft — update as modules are built out)

from .io import (
    # e.g. read_lammps_dump, read_lammps_data, write_lammps_data
)
from .geometry import (
    # e.g. rotation_matrix, misorientation_angle
)
from .objective_templates import (
    # e.g. lammps_gb_energy_fn, make_lammps_objective_fn
)
```

Items that are more "internal plumbing" (validators, low-level I/O primitives)
do not need to be re-exported here — they can be imported directly by module
path when needed.

---

## 5. Migration Plan

### Phase 1 — New functionality first (lowest risk)
- Build out `objective_templates.py` from scratch (no existing code to break)
- Build out `plotting.py` using helpers extracted from external scripts

### Phase 2 — Audit and extract
- Audit external scripts for I/O and geometry helpers → add to `io.py` and
  `geometry.py`
- Identify tools in `GBMaker`, `GBManipulator`, `GBMinimizer` that are
  general-purpose → move to `utils`, keep thin wrappers in the original classes
  if needed for backward compatibility

### Phase 3 — Validation consolidation
- After Phase 2, common validation patterns will be more visible → consolidate
  into `validation.py` and update callers

### Backward compatibility
- Any function moved out of a core class should remain accessible from its
  original location (as a wrapper or import alias) for at least one release
  cycle, with a deprecation warning if possible

---

## 6. Open Questions (Summary)

| # | Question | Owner | Status |
|---|----------|-------|--------|
| 1 | Exact signature `GBMinimizer` expects for its objective callable | — | **Answered:** `(GB, manipulator, atom_positions, unique_id, **kwargs) -> tuple[float, str]` for single-eval; `(gb, manipulators, candidates, lineages, unique_ids) -> list[dict]` for batch. See §3.5. |
| 2 | Which geometry helpers currently live in `GBMaker` as methods? | — | **Answered:** Six candidates identified: `reduce_integer_row`, `row_angle_error_deg`, `approximate_rotation_row_as_int`, `approximate_rotation_matrix_as_int`, `scaled_periodic_basis_vector`, `cartesian_from_box_coordinates`. See §3.2. |
| 3 | Which I/O formats exist in external scripts but not in the package? | — | **Answered: None** — `io.py` starts fresh from formats already in the package |
| 4 | Does the `Parent` object become the shared structure type, or is a protocol/wrapper needed? | — | **Answered: `Parent`** — see `gbopt_interface_design.md` §4 |
| 5 | Which plot types should be implemented first? | — | **Answered: Tier 1 list above** — confirm which already exist in notebooks before starting |
| 6 | Does LAMMPS process management belong in `objective_templates.py`? | — | **Answered: Yes — inside the callable.** Both examples spawn and await LAMMPS entirely within the callable (subprocess for MC, SLURM batch for GA). No external setup/teardown is used. Templates should own the LAMMPS lifecycle. See §3.5. |

---

## 7. Out of Scope (for now)

- Type stub files (`.pyi`) — desirable eventually, not in scope for this pass
- Sphinx autodoc integration — assume existing documentation conventions apply
- A `tests/utils/` test suite — should be added alongside each module but is
  not designed here
