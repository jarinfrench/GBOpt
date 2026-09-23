# Refactor cleanup backlog

Items identified during the R01-R30 architectural refactor (issue #61) that are real
issues or judgment calls, but were deliberately **not** addressed in the PR that found
them — because fixing them there would have exceeded that PR's stated scope, or because
they need a decision this file doesn't make on its own. Each entry names where the item
was found, why it's being deferred, and the natural point to resolve it. When an item is
resolved, remove its entry rather than leaving it marked done.

## `GBMaker.__validate`'s own `positive` parameter is still misleadingly named

`GBMaker.__validate(..., positive=True)` (and its call sites in `__init__` and every
property setter, ~20 in total) only rejects `value < 0` — it means "non-negative," not
"positive." This was understood and fixed for the *new* code in
`GBOpt/gbmaker/config.py::_validate_scalar`, which renamed its own copy of the parameter
to `nonnegative`. `GBMaker.__validate`'s own parameter was left alone: renaming it is
still purely mechanical (a keyword rename, no behavior change), but touches ~20 call
sites in a file whose diffs are otherwise scoped tightly to "extraction only," per R04's
issue.

**Resolve at**: R10 ("Reduce `GBMaker` to a compatibility facade"), when `GBMaker`'s
whole legacy surface is being reconsidered anyway — or sooner, on request.

## `GBBuildConfig.repeat_factor` may have the same nonnegative-vs-strictly-positive bug as `gb_id`/`x_dim_min`/`interaction_distance`/`mismatch_tol`

Confirmed reproducible: `normalize_legacy_config(..., repeat_factor=0)` raises
`GBMakerConstructionValueError` from `gbmaker/types.py::_require_repeat_factor` (which
requires each axis value `> 0`), but the legacy validator it's meant to mirror
(`_validate_scalar`'s own `repeat_factor` branch, used by the same
`normalize_legacy_config` two lines earlier) only warns — "Recommended repeat factor is
at least 2." — for any value below 2, including `0`, and lets construction proceed. So
today, `GBBuildConfig`'s own re-validation is stricter than the exact value
`normalize_legacy_config` already approved moments earlier.

Unlike `mismatch_tol=0.0` (a meaningfully different, legal value from `None`),
`repeat_factor=0` zeroes out `self.__y_dim`/`self.__z_dim`
(`self.__repeat_factor[i] * self.__spacing[axis]`), which cascades into every downstream
box-dimension and grain-generation calculation — closer to the `a0=0` case: it's not
obvious `0` was ever a *usable* legacy value there, only a legacy-permitted one on the
way to a probably-confusing failure elsewhere. Deciding this needs the same tracing
`a0`'s case got (confirm nothing downstream produces a valid structure at
`repeat_factor=0`) before picking a validator, not a snap fix.

**Resolve at**: whichever roadmap step next touches `GBBuildConfig.repeat_factor` or
box-dimension planning (R06, "Extract commensurability, strain accommodation, and
dimension planning" — the natural place, since it owns `__update_dims`) — or sooner, on
request.

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

## Geometry kernels (R07) and grain builders (R08) may be more entangled than the roadmap summary suggests

Flagged while evaluating GBMaker split options (see conversation before R05): geometry
helpers (`wrap_reduced_coordinate`, `_miller_row_norm`, `__reduce_integer_row`,
`__selection_basis_vectors`, `__reduced_box_coordinates`, `__cartesian_from_box_coordinates`,
`__x_index_range`, `__row_angle_error_deg`) are called *from inside* both
`__build_exact_grain` (the exact path) and `__generate_grain_result`
(the approximate/float path), not just adjacent to them. The roadmap's R07-before-R08
sequencing already anticipates needing kernels before builders, but the actual call
shape (kernels invoked mid-computation by both builder paths, not composed before
calling them) may force an awkward callback or higher-order-function interface at the
R07/R08 seam that a purely bottom-up reading of the roadmap wouldn't predict.

**Resolve at**: R07/R08 — reread #69/#70's acceptance criteria closely against the
actual call graph (not just the roadmap summary) before finalizing the kernel module's
public interface.

## Whether `GBMakerValueError`/`GBMakerTypeError` should eventually alias `GBMakerConstructionValueError`/`GBMakerConstructionTypeError`

R04 introduced `__translate_construction_error`, a single wrapper that translates
`GBOpt.gbmaker.types`'s `GBMakerConstructionTypeError`/`GBMakerConstructionValueError`
back to `GBMaker`'s own established `GBMakerTypeError`/`GBMakerValueError` at every
extraction boundary. This exists because of two hard constraints that hold through at
least R09: `gbmaker/config.py` can't import `GBMakerValueError` from `GBMaker.py`
without a circular import, and issue #61's compatibility requirements explicitly protect
"established... public exception identities," so the exact exception type raised by
`GBMaker(...)` can't silently change type before that's authorized.

Once R10 is free to redesign `GBMaker`'s whole surface as a facade, it's worth deciding
on purpose whether `GBMakerValueError`/`GBMakerTypeError` should become thin aliases of
the `gbmaker` exceptions (collapsing the translation wrapper entirely) or stay
independent, rather than letting the R04-era translation pattern persist by default
because nobody revisited it.

**Resolve at**: R10 ("Reduce `GBMaker` to a compatibility facade").
