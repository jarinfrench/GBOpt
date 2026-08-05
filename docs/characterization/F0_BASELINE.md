# F0 characterization baseline

## Scope

F0 adds no production behavior. It records cross-cutting behavior before extraction and
establishes architecture decisions for later PRs.

The committed manifest is:

```text
tests/characterization/baseline_manifest.json
```

The generator and regression test are:

```text
tests/characterization/f0_manifest.py
tests/characterization/test_f0_characterization.py
```

## Source identity

| Field | Value |
|---|---|
| Archive | `gbopt_source.tar(1).gz` |
| SHA-256 | `d0c24898b334b26f5445304d209cb3bf3fbc1ac3882eb7a5b029045ac565d6b1` |

## Characterized behavior

The manifest covers:

- legacy FCC construction with a periodic outer-x interface;
- legacy FCC construction with nonzero vacuum and a single-interface slab;
- exact supplied P/Q FCC construction;
- exact CSL FCC construction;
- exact supplied P/Q fluorite/UO2 construction;
- approximate non-CSL construction;
- exact mismatch accommodation;
- orthogonal and restricted-triclinic LAMMPS output;
- immutable `InterfaceCandidate` geometry and labels;
- in-plane translation;
- periodic grain-local termination cycling;
- slab grain-local termination cycling;
- topology-aware interface separation and composition order;
- fixed-seed Monte Carlo history;
- fixed-seed legacy scalar GA history and failure penalty;
- ownership-aware batch GA reload, row reordering, species mutation rejection, and a
  left-owned atom crossing `gb_plane_x` without ownership loss.

## Hash policy

Structured atom arrays record:

- an order-sensitive SHA-256 over normalized rows;
- an order-insensitive SHA-256 over the same normalized row multiset;
- species counts;
- coordinate extrema;
- dtype field order.

Floating-point values are rounded to 12 decimal places before canonical JSON encoding.
This preserves meaningful ordering and geometry while avoiding platform-only text
variation below the accepted characterization precision. LAMMPS output files are hashed
as exact UTF-8 bytes at precision 12.

An order-insensitive match never excuses an unexplained order-sensitive change. Both are
recorded so a later PR can diagnose whether drift is only ordering or physical content.

## Regeneration

From the repository root:

```bash
PYTHONPATH=.:tests python -m characterization.f0_manifest --verify-repeat
```

`--verify-repeat` generates the behavior section twice and fails if the results differ.
The normal regression test regenerates behavior and compares it with the committed
manifest:

```bash
PYTHONPATH=. pytest -q tests/characterization/test_f0_characterization.py
```

Do not update the baseline merely to make a refactor pass. First explain every changed
field, warning, hash, optimizer history, or serialized byte sequence. An intentional
behavior change requires its own scoped PR and documentation.

## Recorded implementation environment

The manifest contains informational Python, platform, and package versions. Behavior
comparison intentionally excludes environment metadata so supported dependency updates
do not require a behavior-baseline rewrite when results remain identical.

During F0 implementation in the provided execution environment, `spglib` was not
installed and network installation was unavailable. A temporary external import stub
was used only to collect and run non-slow tests that do not invoke spglib. The stub is
not part of the repository or archive. Slow soft-mode tests were therefore not claimed
as executed in this environment.
