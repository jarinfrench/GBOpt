# Temporary Ideas Triage Backlog

> **Status:** Temporary, non-authoritative backlog.  
> **Required disposition:** Convert each item into a GitHub issue, associate it with a roadmap PR, or record an explicit rejection. Delete this file when every item has a disposition.  
> **Implementation authority:** `../MASTER_PLAN.md`; entries here do not authorize scope expansion.

## Triage labels

Use one of the following labels while processing the original notes below:

- **Roadmap:** already owned by an active or optional roadmap PR;
- **Issue:** create a standalone GitHub issue;
- **Deferred:** valid idea without current priority;
- **Rejected:** conflicts with an accepted invariant or architecture decision;
- **Resolved:** already implemented or no longer applicable.

---

## Formatter / line-length hygiene

autopep8 wraps lines aggressively at 88 characters, which hurts readability
in long docstrings and complex expressions. Consider migrating to **Ruff**
(`ruff format`) as the formatter: it respects the 88-char target without
over-splitting, and its linter covers most of what autopep8 + flake8 provide.
Until Ruff is adopted, manually keep docstring prose close to the 88-char
limit rather than relying on autopep8's wrapping.

## Complete docstrings for minimizer methods

Most methods in `MonteCarloMinimizer` and `GeneticAlgorithmMinimizer` have
incomplete docstrings — typically missing `:param:` entries for one or more
parameters, or a `:return:` description. Notable gaps:

- `_evaluate_generation` — no `:param:` lines at all
- `run_MC` / `run_GA` — several parameters undocumented
- `_make_initial_manipulator`, `_make_manipulator_from_file` — no docstrings

Do a systematic pass: every public and private method should have a one-line
summary, full `:param:` / `:return:` / `:raises:` coverage, and type hints on
the signature where missing. Google-style with `:param name:` tags, consistent
with the rest of the codebase.

## Abstract `Minimizer` base class

`MonteCarloMinimizer` and `GeneticAlgorithmMinimizer` share setup boilerplate
that is currently duplicated:

- Checkpoint wiring (`CheckpointStore.from_optional`, unique_id derivation,
  `state = checkpoint.load()` / restore block)
- `_make_initial_manipulator` (near-identical in both classes; only the
  `hasattr(self, "local_random")` guard differed)
- `_is_valid_file` helper

A `Minimizer` abstract base class (e.g. `GBOpt/Minimizer.py`) with:

```python
class Minimizer(ABC):
    def _setup_checkpoint(self, checkpoint_file, fmt, interval): ...
    def _make_initial_manipulator(self): ...
    def _is_valid_file(self, p): ...

    @abstractmethod
    def run(self, ...): ...
```

would remove the duplication and provide a stable interface for future
minimizer variants (basin-hopping, simulated annealing with restarts, etc.).
`CheckpointStore` is already a clean primitive that this base class would own.

## Utils class

### GB parameter conversion

Add a conversion utility (initially in a `Utils` class, or as a standalone
module) that converts between different grain boundary orientation
representations.

**First target:** P/Q orientation matrices → GBOpt 5-angle array
`[alpha, beta, gamma, theta, phi]`.

Given orientation matrices P (left grain) and Q (right grain), where each
row is the crystal direction corresponding to the lab x, y, z axes:

1. Misorientation matrix: `R_mis = P @ Q^T`
   ZXZ Euler angles `[alpha, beta, gamma]` via `scipy.spatial.transform.Rotation`.

2. Inclination angles from P row 0 (boundary normal in the left grain's
   crystal frame), normalized to `n = [n_x, n_y, n_z]`:
   `theta = arctan(n_z / n_x)`
   `phi   = -arctan(n_y / sqrt(n_x² + n_z²))`

**Longer-term:** extend to convert between all common representations,
for example:
- Axis/angle + boundary normal
- Rodrigues vector + boundary normal
- GB database formats (e.g., Olmsted dataset convention, where inclination
  is referenced to the lab frame rather than the left grain)

## Test structure refactor

`tests/test_gbmanipulator.py` has a single large `TestGBManipulator` class
covering init, translation, slice/merge, remove/insert atoms, soft mode
displacement, parents, and file I/O. Break it up into focused classes per
feature group (e.g. `TestGBManipulatorRemoveAtoms`,
`TestGBManipulatorInsertAtoms`, `TestGBManipulatorSoftModeDisplacement`),
consistent with the class-per-feature pattern used in `test_gbmaker.py`.
Each class should carry only the `setUpClass` fixtures its tests actually need.
This separation should be applied across all tests.

## Logging

When logging is added to GBOpt, `displace_along_soft_modes` should log the
selected q-point and the mesh size used to generate it, so results are
reproducible and users can inspect which point in the Brillouin zone was used.

## `displace_along_soft_modes` performance

Several bottlenecks identified, in priority order:

1. **`_calculate_bond_hardness` — biggest gain.** Runs as interpreted Python
   (the `@jit` decorator is commented out). Constructs an `Atom` object and
   does dict lookups inside the inner loop over GB atom pairs. Refactor to
   precompute all per-atom scalar properties (r_cov, valence,
   valence_electrons, electronegativity) into plain numpy arrays before the
   loop so the inner loop touches only numbers and arrays — then re-enable
   `@jit`. Also replace the dense `Hij = np.zeros((len(atoms), len(atoms)))`
   with a sparse structure or dict, since only GB atom pairs are ever non-zero.

2. **`_calculate_dynamical_matrix` — O(n) lookups inside JIT loop.** Two
   linear scans of `gb_atom_indices` per neighbor pair:
   `if id2 not in gb_atom_indices` and `np.where(gb_atom_indices == id2)`.
   Replace with a precomputed `gb_index_to_d` array (global atom index →
   position in `gb_atom_indices`, -1 for non-GB atoms) for O(1) lookup.
   Also: `prange` is used but `parallel=True` is missing from the `@jit`
   decorator — add it for immediate multi-core speedup.

3. **`neighbor_list_typed` conversion — minor.** The Python list-of-lists is
   converted to a Numba `List` on every call. If `_create_neighbor_list`
   returned a typed List directly, this conversion would be eliminated.

4. **`np.linalg.eigh`** — unavoidable O(n³); the sparse fallback already
   handles large systems. Keeping `gb_thickness` small directly reduces cost.

## `displace_along_soft_modes` test structure

The current `test_simple_case` uses a single FCC Cu crystal with one deliberately
displaced atom. After the dynamical-matrix and `gb_indices` fixes, this structure
produces four GB-layer columns that collectively rotate rather than a localized
displacement centered on the displaced atom.

This is physically correct for the model: the Lyakhov bond-hardness dynamical
matrix finds the softest **collective** mode of the GB slab, not the mode that
minimises the energy of a specific displaced atom. For a nearly-perfect single
crystal the perturbation from one 0.5 Å displacement is too small to create a
localised gap mode.

For a more intuitive visual test, consider using a real bicrystal with a
well-characterised soft mode — e.g., a Σ5 (100) boundary in Cu — where structural
disorder at the GB genuinely produces soft, localised modes. A test against such a
structure would let you verify that displaced atoms are geometrically co-located
with the GB structural units rather than distributed uniformly across the entire
GB slab.

## GB orientation axis

The GB normal is currently hardcoded along the lab x-axis ("left" and "right"
grains). Add a parameter — something like `gb_axis` or `orientation` — that
lets the user choose which Cartesian direction the boundary normal lies along:

- `"x"` — left / right (current behaviour, default)
- `"y"` — bottom / top
- `"z"` — front / back

The grain labels reported and written out could follow the chosen axis so that
output files and logs make intuitive sense ("top grain", "bottom grain", etc.)
rather than always saying left/right.

Implementation note: the simplest approach is probably a post-rotation of the
entire assembled system rather than rewiring the internal x-axis logic, so that
the core GBMaker machinery stays unchanged.

## Invariant-tests branch follow-up

Update `tests/test_invariants.py` on the invariant-tests branch to match the
current `GBMaker` design:

- verify periodic membership in y/z using reduced box coordinates rather than
  Cartesian interval clipping
- verify exactly one representative is kept for each periodic equivalence class
- verify `y_dim` and `z_dim` remain consistent with repeat factors and periodic
  spacings
- verify left/right grain partitioning along x is preserved
- verify generated atoms remain within the declared box bounds after reduced-
  coordinate wrapping and Cartesian reconstruction
- verify triclinic handling remains allowed only when both in-plane axes are
  periodic

## Add jitter manipulator
CG minimization often results in atoms moving in different directions, while
most of the current manipulators move atoms as a group. It might be useful
to implement a jitter manipulator that displaces atoms in the GB region in
random directions by a small amount (maybe allowing for atoms to "cross
through" each other?).

## Spacing-pattern GB plane detection

For boundaries where the two grains have different x-interplanar spacings
(asymmetric tilt, mixed tilt/twist), the GB plane position could be inferred
purely from the atomic structure by analysing consecutive differences in the
unique x-plane positions:

1. `planes = np.unique(np.round(system["x"] / epsilon)) * epsilon`
2. `diffs = np.diff(planes)` — each grain produces a run of nearly-constant spacing
3. Split diffs at their median → low cluster (`d_left`) and high cluster (`d_right`)
4. The transition from one cluster to the other marks the GB; the GB plane is
   the midpoint of the gap straddling the transition

For a PBC bicrystal there are two interfaces; both appear as transitions and the
central one can be selected as the one closest to the box midpoint.

**Caveats:**

- **Does not work for symmetric tilt boundaries** (`d_left ≈ d_right`): the runs
  are indistinguishable, so no transition is detectable. A fallback to "largest
  gap" is needed, but for symmetric tilts the interface gap may not be the largest
  gap in the system.
- The `threshold` for deciding whether two clusters are "clearly different" requires
  a judgment call; no robust universal value is obvious.
- For the `GBMaker` path the geometric anchor `vacuum_thickness + left_x` is exact
  and strictly superior. This approach is most relevant for `GBManipulator`'s
  `__init_by_file` path, where grain provenance is not available — but even there
  the midpoint `(max(left_grain["x"]) + min(right_grain["x"])) / 2` already gives a
  reliable estimate once the grain split is correct.

## fragile pathname in GBMinimizer with best_dump = "min" + dump_file_name
if dump_file_name is a full path, prepending "min" will be a completely
different directory. Use Path of equivalent path-safe logic.

## Generator-based batch energy function interface

The current `gb_batch_energy_func` interface returns all results at once as a
list, making it impossible to record individual job completions from *outside*
the function. Per-job intra-batch checkpointing therefore requires the function
author to declare `checkpoint=None` and call `checkpoint.record()` manually.

An alternative: change `gb_batch_energy_func` to a **generator** that `yield`s
one result dict at a time as each job completes:

```python
def get_batch_gbe(GB, manips, structs, lineages, unique_ids):
    jobs = submit_all(manips, structs, unique_ids)
    for job in wait_for_completions(jobs):   # any completion order
        yield {"unique_id": job.uid, "energy": ..., "final_dump": ...}
```

`_evaluate_generation` would consume the generator and call
`gen_checkpoint.record()` after each yielded result — no change to user code
required beyond switching `return results` to individual `yield` statements.

**Considerations:**
- Breaking API change: existing batch functions return lists; would require a
  migration path or parallel support for both conventions (detect via
  `inspect.isgeneratorfunction`).
- The `"unique_id"` key in the yielded dict would be new (currently the caller
  assigns IDs to a positional list); ordering assumptions need to be dropped.
- Enables correct per-job recovery even for batches where jobs finish
  out-of-order, without any explicit checkpoint wiring in the batch function.

## Library of "energy_funcs"
I've already found myself annoyed with having to clean up the energy functions
to make them more robust to failed calls, and what-have-you. There should be
some sort of library (in Utils, or a separate 'library' directory?) that has
templates for these functions, and that users can directly import if they don't
want to come up with their own.

## Nudging overlapping atoms
We don't do any correctness checking on atomic positions when the manipulators
run. This can (often?) result in structures that have overlapping atoms, and
lead to failing simulations (e.g., "Lost atoms"). There should be a method or
Utility that double checks the atom positions and nudges any that are too close,
but that might require a user-definition for what that means.

## Describe the bug

  ga_data.json stores a field called gbe_vals whose structure is ambiguous and inconsistent with the dump file naming convention,
  making it impossible to reliably map a GBE value back to its corresponding dump file without also consulting the
  temp_GA_1_g*_c*.dump_GBE.txt sidecar files.

  gbe_vals[n] is a population snapshot — the GBE values of the candidates that survived into generation n — not a log of the
  candidates evaluated during generation n. As a result:

  - The outer index n is one generation later than the directory where the corresponding dump file lives (gen_{n-1}/).
  - The inner index k is a position in the current population ordering, not the _cK number in the dump filename. When any candidate in
   a generation fails evaluation it is omitted from gbe_vals without adjusting subsequent indices, so list position and filename
  suffix diverge.

  The sidecar files (temp_GA_1_g{G}_c{C}.dump_GBE.txt) are the only place where the generation, candidate number, and GBE value are
  stored together unambiguously, but their relationship to gbe_vals is undocumented.

  To Reproduce

  1. Run the GA optimizer for any boundary (e.g. Si/sigma5_mixed) and inspect the output ga_data.json.
  2. Note the minimum value in gbe_vals and its position — e.g. gbe_vals[13][44] = 1.132673.
  3. Expect the corresponding dump file to be workdir.1/gen_13/final_GA_1_g13_c44.dump.
  4. Observe that no such file exists; the actual dump is at workdir.1/gen_12/final_GA_1_g12_c44.dump.

  Expected behavior

  Either:

  - gbe_vals[n][k] should record the GBE of the k-th candidate evaluated at generation n, so that index n matches gen_N/ and index k
  matches _cK in the dump filename directly; or
  - ga_data.json should include a parallel field (e.g. best_candidate_history) that records, for each population update, the actual
  generation number, candidate number, and dump file path of the new best structure so downstream tools have a single unambiguous
  source.

  Screenshots

  N/A

  Environment:
  - OS: Linux
  - Python version: 3.12.13
  - NumPy version: N/A
  - Calculator: N/A
  - Installation method: N/A

  Additional context

  This ambiguity caused a silent data selection error in find_best_structures.py: the script used gbe_vals list indices as if they
  were dump file coordinates, produced wrong paths, and in some cases silently skipped entire runs (when the snapshot generation's
  directory contained no dump files), causing MC to be selected as the winner even when GA had found a lower GBE. The
  temp_GA_1_g*_c*.dump_GBE.txt sidecar files are the authoritative source and do not have this ambiguity, but relying on them requires
   scanning potentially thousands of small files rather than a single JSON.
