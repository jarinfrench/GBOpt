> **SUPERSEDED**
>
> Replaced by [`../../MASTER_PLAN.md`](../../MASTER_PLAN.md), [`../../architecture.md`](../../architecture.md), and the accepted ADRs. This file is retained to explain the earlier broad ABC/Protocol proposal; its `Parent`-as-universal-structure and combined interface assumptions are not current architecture.

# GBOpt Abstract Interface Architecture — Design Document

**Status:** Draft  
**Author:** Jarin French
**Last Updated:** 2026-04-30  
**Related:** `gbopt_utils_design.md`

---

## 1. Motivation

GBOpt currently provides one concrete implementation of each major concern
(making, manipulating, minimizing grain boundaries). As the package grows,
users will want to swap components — a VASP-backed minimizer in place of a
LAMMPS one, a custom I/O adapter for a proprietary format, an ML potential
as the calculator — without rewriting their workflows.

This document designs an **abstract interface layer** that:
- Enforces consistent method contracts across all implementations of a role
- Makes concrete classes drop-in replaceable (swappable backends)
- Establishes clear boundaries between what each component is responsible for
- Leaves the existing `GBMaker`, `GBManipulator`, and `GBMinimizer` classes
  intact as the default concrete implementations

This design coexists with and complements the `utils` subpackage
(`gbopt_utils_design.md`). It does not replace it.

---

## 2. Design Approach: ABCs + Protocols

Python gives us two tools for this, with different trade-offs:

| | `abc.ABC` + `@abstractmethod` | `typing.Protocol` |
|---|---|---|
| Enforcement | At instantiation time — raises `TypeError` if abstract methods not implemented | At type-check time (mypy/pyright) — no runtime error |
| Subclassing required | Yes — must explicitly inherit | No — any class with matching methods qualifies ("structural subtyping") |
| Best for | Internal hierarchy you control | Third-party code / duck typing |

**Recommendation: Use both, layered.**

- Define `Protocol` classes as the *public interface specification* — these
  describe what any conforming object must look like, regardless of where it
  comes from. Power users who write their own implementations only need to
  satisfy the protocol; they don't have to inherit from anything.
- Define `ABC` base classes as the *convenient starting point* for
  implementations you ship — they enforce contracts at instantiation time and
  provide shared scaffolding (e.g., logging, common `__repr__`).

```
GBMakerProtocol        ← public contract (typing.Protocol)
    └── BaseGBMaker    ← ABC with shared scaffolding (abc.ABC)
            └── GBMaker  ← existing concrete class, now a subclass of BaseGBMaker
```

This means:
- Users who subclass `BaseGBMaker` get enforcement and shared helpers for free
- Users who write from scratch just need to satisfy `GBMakerProtocol`
- Type annotations use the `Protocol` everywhere, so the type checker accepts both

---

## 3. Component Roles and Proposed ABCs

### 3.1 `GBMakerProtocol` / `BaseGBMaker`

Responsible for constructing an initial grain boundary structure from
crystallographic inputs (orientation relationship, boundary plane, lattice).

**Core abstract methods (confirmed by audit):**

*Properties — required read access by `GBManipulator` and `GBMinimizer`:*
- `whole_system` — full bicrystal atom array
- `left_grain`, `right_grain` — per-grain atom arrays
- `gb_plane_x` — x-coordinate of the GB plane
- `box_dims` — 3×2 simulation cell extents
- `unit_cell` — nominal crystal unit cell (carries `type_map`, `a0`, `ratio`, `radius`, etc.)
- `radius` — atom radius (Å)
- `x_dim`, `y_dim`, `z_dim` — box dimensions
- `gb_thickness` — GB slab thickness
- `repeat_factor` — in-plane periodic repetition factors

*Methods:*
- No additional abstract methods beyond the properties above are required to satisfy the
  current consumer contracts.
- The proposed `make() -> Structure` entry point does not exist in the current `GBMaker`
  (construction happens inside `__init__`); a future `build()` abstract method could
  formalize this pattern, but it is not needed to satisfy existing consumers.

*What stays out of `BaseGBMaker`:*
- `write_lammps` — I/O; belongs in `IOProtocol` per the design (§3.4)
- `get_supercell`, `update_spacing` — implementation details; not part of the interface
  contract

**What it should NOT do:**
- Manipulate or relax the structure (that's `Manipulator`)
- Write files directly (that's `IO`)

---

### 3.2 `GBManipulatorProtocol` / `BaseGBManipulator`

Responsible for structural modifications to an existing grain boundary
(rigid body translation, atom deletion, cell manipulation, etc.).

**Core abstract methods:**
- `apply(structure: Structure) -> Structure` — apply the manipulation and
  return the result; implementations should treat input as immutable
- (others — to be identified during audit)

**Design notes:**
- Manipulations should be composable — it should be straightforward to chain
  multiple manipulators in sequence
- Consider whether a `Pipeline` or `CompositeManipulator` helper belongs in
  `utils` or here

**Open questions:**
- **Q**: Does the current `GBManipulator` mutate in place or return a new structure?
  The abstract interface should standardize one approach (return new is
  preferred for composability and safety).
  **A**: It returns a new structure.

---

### 3.3 `GBMinimizerProtocol` / `BaseGBMinimizer`

Responsible for driving an optimization loop over grain boundary structures
toward a user-specified scalar objective.

**Core abstract methods:**
- `minimize(structure: Structure) -> MinimizerResult` — run the optimization
  and return a result object
- `set_calculator(calculator: CalculatorProtocol) -> None` — attach the
  scalar objective backend (see §3.5)
- (others — to be identified during audit)

**Design notes:**
- The minimizer owns the optimization *algorithm* (GA, MC, basin hopping, etc.)
  but delegates scalar objective evaluation to a `Calculator`
- The objective is explicitly any scalar — not just GB energy. The manuscript
  cites thermal conductivity, hardness, and radiation tolerance as examples.
  `MinimizerResult` should therefore use neutral language: `best_value`, not
  `best_energy`
- `MinimizerResult` should be a dataclass or named container (not a raw dict)
  holding at minimum: best structure, best objective value, convergence history
- The minimizer must support both a single-evaluation callback and a
  batch-evaluation interface (e.g., SLURM/MPI dispatch for GA population
  members). Both modes exist in the current implementation and must be
  preserved in the abstract interface
- The two existing concrete implementations are `MonteCarloMinimizer`
  (Markov chain / Metropolis acceptance) and `GeneticAlgorithmMinimizer`
  (tournament/elitist selection). Both become subclasses of `BaseGBMinimizer`

**Current objective callable signatures (as of this audit):**

Both minimizers currently accept the objective as a bare `Callable` (not a `Calculator`
object). The exact signatures are:

*Single-evaluation callable* (`gb_energy_func`) — required by both minimizers:
```python
def gb_energy_func(
    GB: GBMaker,
    manipulator: GBManipulator,
    atom_positions: np.ndarray,
    unique_id: str,
    **kwargs,            # only forwarded by MonteCarloMinimizer, not by GA
) -> tuple[float, str]:
    """Returns (objective_value, path_to_output_dump_file)."""
```

*Batch-evaluation callable* (`gb_batch_energy_func`) — optional GA-only extension:
```python
def gb_batch_energy_func(
    GB: GBMaker,
    manipulators: list[GBManipulator],
    atom_positions_list: list[np.ndarray],
    lineages: list[list[str]],
    unique_ids: list[str],
) -> list[dict]:
    """Each dict must contain at least {"energy": float, "final_dump": str}."""
```

`GeneticAlgorithmMinimizer.__init__` accepts `gb_batch_energy_func` as a keyword
argument (defaults to `None`); when provided it replaces per-candidate calls with a
single batch call per generation.

The `BaseGBMinimizer` abstract interface should accept either form. Per §7 (backward
compatibility), a concrete base implementation can wrap the single-evaluation callable
in a `CalculatorProtocol`-conforming adapter with a deprecation warning during
migration.

---

### 3.4 `IOProtocol` / `BaseIO`

Responsible for reading and writing atomistic structure files.

**Core abstract methods:**
- `read(path: PathLike) -> Structure`
- `write(structure: Structure, path: PathLike) -> None`

**Design notes:**
- Current concrete implementations ship with GBOpt: `LAMMPSDataIO`
  (read/write LAMMPS data files) and `LAMMPSDumpIO` (read LAMMPS dump files).
  The manuscript identifies XYZ and CIF as the next planned formats.
  POSCAR/VASP formats are not mentioned in the manuscript, but they should also be included.
- This is distinct from the helper *functions* in `utils/io.py` — those are
  stateless utilities; `BaseIO` is a stateful object that can carry format-
  specific configuration (e.g., column ordering, unit conventions)
- `utils/io.py` functions may become the implementation detail that concrete
  `IO` classes delegate to

**Open questions:**
- Is there value in a `read_many / write_many` interface for trajectory-style
  output, or is that out of scope? **Tentative answer**: I don’t imagine a read_many/write_many approach would be needed.

---

### 3.5 `CalculatorProtocol` / `BaseCalculator`

Responsible for evaluating a scalar objective for a given structure. This is
the formalization of what the current user callback ("a callable that maps
atomic positions to a scalar objective") already is.

**Core abstract methods:**
- `calculate(structure: Structure) -> CalculatorResult` — evaluate and return
  the scalar result
- `setup() -> None` — initialize any external process or resource (e.g.,
  launch LAMMPS)
- `teardown() -> None` — clean up (e.g., terminate LAMMPS process)

**Design notes:**
- The objective is intentionally a **scalar** — the manuscript is explicit that
  GBOpt is not limited to energy. Forces and stress are out of scope for this
  interface; if a specific implementation needs them internally, that is an
  implementation detail, not part of the protocol.
- `CalculatorResult` is a dataclass holding the following fields, derived from
  the batch callable's return dict (audited from `examples/run_ga.py`):

  | Field | Type | Required | Notes |
  |---|---|---|---|
  | `value` | `float` | Yes | The scalar objective value |
  | `final_dump` | `str \| None` | Yes | Path to the output structure file |
  | `num_atoms` | `int \| None` | No | Atom count after relaxation (may differ from input) |
  | `parents` | `list[str] \| None` | No | Lineage IDs; used by GA for genealogy tracking |
  | `status` | `str \| None` | No | Job status string; critical for HPC robustness |
  | `fail_reason` | `str \| None` | No | Human-readable failure description if `status` indicates failure |

  `status` and `fail_reason` are particularly important for HPC use: they allow
  the minimizer to distinguish a failed SLURM job from a converged one without
  raising an exception. It does NOT include forces or stress tensors.
- Concrete implementations: `LAMMPSCalculator`, `VASPCalculator`,
  `MLPotentialCalculator`, etc.
- `objective_templates.py` in `utils` (see `gbopt_utils_design.md`) becomes
  the home for pre-built `Calculator` instances and factory functions — the two
  designs connect here
- `setup` / `teardown` allow the minimizer to manage calculator lifetime
  explicitly; consider also supporting use as a context manager
  (`__enter__` / `__exit__`) for convenience

**Open questions:**
- Does `BaseCalculator` need a `is_ready` / `health_check` method for
  long-running external processes? **A**: Yes

---

## 4. The `Structure` Type

All components pass structures between each other. The manuscript describes
the existing data container as a **`Parent` object**, which stores: the
simulation cell (unit cell and box dimensions), partitioned atomic coordinates
for each grain ("left" and "right"), the GB core, and the full bicrystal,
along with GB region metadata.

The abstract interfaces need to decide whether `Parent` becomes the canonical
shared type or whether a thinner wrapper is defined.

**Options:**

| Option | Pros | Cons |
|---|---|---|
| Promote `Parent` as the interface type | No new type; already used everywhere | `Parent` may carry GB-specific fields that non-GB use cases don't need |
| Define a minimal `GBStructure` dataclass | Lightweight; only what every component needs | Requires mapping to/from `Parent` in existing code |
| Define a `StructureProtocol` | Maximum flexibility; `Parent` and third-party types conform if they have matching attributes | Less discoverable; harder to document |

**Recommendation (updated after audit — see Q2, §8):** Promote `Parent` as the
interface type. Do not define a separate `StructureProtocol`.

The audit shows that `GBManipulator` uses nearly all of `Parent`'s GB-specific fields:
`whole_system`, `left_grain`, `right_grain`, `y_dim`, `z_dim`, `gb_atoms`,
`gb_indices`, `unit_cell`, `box_dims`, and `gb_thickness`. A thin `StructureProtocol`
would have to enumerate all of these fields anyway, yielding no practical difference
from using `Parent` directly. `GBMinimizer` accesses only `whole_system` (a strict
subset), so it is compatible with `Parent` as the interface type.

Concretely: `Parent` becomes the canonical `Structure` type used in all type
annotations. File-backed construction (from a LAMMPS dump or input file) is already
supported by `Parent.__init__`; no mapping layer is needed. A `StructureProtocol`
remains an option only if third-party types that cannot inherit from `Parent` need to
interoperate — treat that as a Phase 2 decision, not Phase 0 scope.

Note: ASE is **not** a dependency of GBOpt (Table 1 in the manuscript lists
numpy, scipy, numba, pandas, matplotlib, spglib). ASE `Atoms` should not be
treated as a candidate base type.

---

## 5. How the Pieces Fit Together

A typical workflow under this architecture (illustrating the *future* interface
after the Calculator ABC is introduced — the current interface uses a bare
callable instead of a `Calculator` object):

```python
from GBOpt import GBMaker, GBManipulator
from GBOpt.minimizers import GeneticAlgorithmMinimizer   # or MonteCarloMinimizer
from GBOpt.calculators import LAMMPSCalculator
from GBOpt.utils.objective_templates import make_lammps_objective_fn

# Each component satisfies its Protocol independently
maker       = GBMaker(sigma=5, boundary_plane=[0,1,0], ...)
manipulator = GBManipulator(...)
calculator  = LAMMPSCalculator(potential="path/to/potential", elements=["Fe"])
minimizer   = GeneticAlgorithmMinimizer(calculator=calculator, ...)

# Workflow
structure = maker.make()
structure = manipulator.apply(structure)

with calculator:                          # setup / teardown via context manager
    result = minimizer.minimize(structure)

print(result.best_value)                  # scalar objective — not necessarily energy
```

A power user with a custom calculator only needs to satisfy
`CalculatorProtocol` — no inheritance required:

```python
class MyMLCalculator:
    def calculate(self, structure): ...   # returns scalar CalculatorResult
    def setup(self): ...
    def teardown(self): ...

# Works anywhere a CalculatorProtocol is expected
minimizer = GeneticAlgorithmMinimizer(calculator=MyMLCalculator(), ...)
```

---

## 6. Package Layout Changes

```
GBOpt/
├── interfaces/               # NEW — Protocols and ABCs only, no logic
│   ├── __init__.py
│   ├── maker.py              # GBMakerProtocol, BaseGBMaker (see note below)
│   ├── manipulator.py        # GBManipulatorProtocol, BaseGBManipulator
│   ├── minimizer.py          # GBMinimizerProtocol, BaseGBMinimizer
│   ├── io.py                 # IOProtocol, BaseIO
│   ├── calculator.py         # CalculatorProtocol, BaseCalculator
│   └── types.py              # Parent / StructureProtocol, MinimizerResult, CalculatorResult
│
├── utils/                    # As described in gbopt_utils_design.md
│   └── ...
│
├── GBMaker.py                # Now subclasses BaseGBMaker (if in scope — see note)
├── GBManipulator.py          # Now subclasses BaseGBManipulator
└── GBMinimizer.py            # Contains MonteCarloMinimizer and GeneticAlgorithmMinimizer,
                              # both subclassing BaseGBMinimizer
```

**Note on `BaseGBMaker` scope:** The manuscript's explicit roadmap lists ABCs
for "manipulators, minimizers, I/O, and calculator interfaces" — `GBMaker` is
not named, but including it has been confirmed as in scope (see Q9, §8).

---

## 7. Migration Plan

### Phase 0 — Define interfaces first, touch nothing else
- Write `interfaces/` with all Protocols and empty ABCs
- Write `interfaces/types.py` with `GBStructure` etc.
- No existing code changes; nothing breaks

### Phase 1 — Retrofit existing classes
- Make `GBMaker`, `GBManipulator`, `GBMinimizer` subclass their respective ABCs
- Confirm they satisfy the abstract method contracts (add `@abstractmethod`
  stubs to ABCs based on what the existing classes already do)
- This is the highest-risk phase — audit carefully before merging

### Phase 2 — First new concrete implementation
- Write one alternative implementation (e.g., `LAMMPSCalculator` as a formal
  `BaseCalculator`) to validate the interface in practice
- Adjust the interface based on what was awkward

### Phase 3 — Utils integration
- `objective_templates.py` becomes factory helpers that return `BaseCalculator`
  instances rather than bare callables
- `utils/io.py` becomes the implementation detail used by concrete `IO` classes

### Backward compatibility
- The existing class names (`GBMaker`, `GBManipulator`, `GBMinimizer`) should
  remain importable from their current paths throughout this migration
- Users passing a `gb_energy_function` callable should continue to work during
  a transition period; `BaseMinimizer` can accept either a `CalculatorProtocol`
  or a bare callable and wrap the latter automatically with a deprecation
  warning

---

## 8. Open Questions (Summary)

| # | Question | Owner | Status |
|---|----------|-------|--------|
| 1 | Does `GBManipulator` currently mutate in place or return new? | — | **Answered:** Returns new |
| 2 | Which attributes of `Parent` does each component actually use? (Determines interface type — see §4) | — | **Answered:** `GBMaker` does not access `Parent` (it is the source). `GBManipulator` accesses: `whole_system`, `left_grain`, `right_grain`, `y_dim`, `z_dim`, `gb_atoms`, `gb_indices`, `unit_cell`, `box_dims`, `gb_thickness`. `GBMinimizer` accesses: `whole_system` only (via `manipulator.parents[0]`). GB-specific fields are central to `GBManipulator`, so `Parent` should be the interface type (see §4). |
| 3 | Is ASE used anywhere in GBOpt? | — | **Answered: No** — ASE is not a listed dependency |
| 4 | Should `CalculatorResult` include forces and stress? | — | **Answered: No** — scalar objective only; forces/stress are out of scope. However, allowing for vector objectives might be a later extension. |
| 5 | Does `BaseCalculator` need a health-check / `is_ready` method? | — | **Answered: Yes** - since this software will primarily be used with HPC systems, it should be able to check how a job is going, and if there is a need for user intervention. |
| 6 | Is a `read_many` / trajectory interface needed for `BaseIO`? | — | **Answered: Most likely not** - discuss |
| 7 | Exact abstract methods for `BaseGBMaker` — audit existing `GBMaker` | — | **Answered:** Abstract properties: `whole_system`, `left_grain`, `right_grain`, `gb_plane_x`, `box_dims`, `unit_cell`, `radius`, `x_dim`, `y_dim`, `z_dim`, `gb_thickness`, `repeat_factor`. `write_lammps` moves to `IOProtocol`; `get_supercell` and `update_spacing` are implementation details. No additional abstract methods are required beyond these properties. See §3.1. |
| 8 | Where does `CompositeManipulator` (pipeline) live — `utils` or `interfaces`? | — | **Answered:** It should live with the concrete manipulator implementations, not `utils` or `interfaces`|
| 9 | Should `BaseGBMaker` be in scope? (Not in the manuscript's explicit roadmap — see §6 note) | — | **Answered: Yes** - if it makes sense to have a `BaseGBMaker`, it should be included. |

---

## 9. Out of Scope (for now)

- Plugin auto-discovery (e.g., entry points for third-party implementations)
- Async / parallel calculator interfaces
- A formal `GBWorkflow` pipeline object (though §5 hints at what it could look
  like eventually)
- Sphinx autodoc / type stub generation
