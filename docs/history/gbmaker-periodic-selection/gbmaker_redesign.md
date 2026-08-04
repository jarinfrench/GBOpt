> **HISTORICAL DESIGN RECORD**
>
> This document records the earlier periodic-selection redesign. The accepted cross-cutting decision is now summarized in [`../../architecture/adr/0005-canonical-periodic-representatives.md`](../../architecture/adr/0005-canonical-periodic-representatives.md). This file is non-authoritative and retained for detailed rationale.

# Robust Handling of Periodic Boundaries in Atomistic Structure Generation

## 1. The Original Problem

You are generating atomic structures with periodic boundary conditions (PBC) in the **y** and **z** directions. You observed that small changes in how you define the bounding box—specifically shifting both bounds by a small epsilon—dramatically affect which atoms are included.

Two examples:

```python
inside_box = (
    (atoms["x"] >= x_min - eps) &
    (atoms["x"] <  x_max - eps) &
    (atoms["y"] >= y_min - eps) &
    (atoms["y"] <  y_max - eps) &
    (atoms["z"] >= z_min - eps) &
    (atoms["z"] <  z_max - eps)
)
```

vs.

```python
inside_box = (
    (atoms["x"] >= x_min - eps) &
    (atoms["x"] <  x_max - eps) &
    (atoms["y"] >= y_min - eps) &
    (atoms["y"] <  y_max) &
    (atoms["z"] >= z_min - eps) &
    (atoms["z"] <  z_max)
)
```

The first works. The second produces overlapping atoms.

---

## 2. Why This Happens

### 2.1 Shifting vs Expanding the Domain

- First version:
  - Domain is shifted by `-eps`
  - Width remains the same
  - Still exactly one periodic cell

- Second version:
  - Lower bound shifted, upper bound unchanged
  - Domain width becomes `L + eps`
  - Now overlaps neighboring periodic images

### 2.2 Key Insight

Periodic systems require selecting **exactly one representative per equivalence class**.

If your domain is even slightly larger than one period, you may include:
- atoms from both sides of a periodic boundary
- duplicates from neighboring images

---

## 3. Why Small Shifts Cause Large Effects

Atom selection is **discrete**, not continuous.

A tiny shift:
- doesn’t slightly move atoms
- instead flips them from *included* to *excluded*

If a whole plane lies on a boundary, a tiny shift can swap entire layers.

---

## 4. Floating Point Reality

Even if mathematically:
```
y = 0 ≡ y = L
```

In floating point, you may see:
- `-1e-16`
- `0`
- `L - 1e-15`
- `L`
- `L + 1e-15`

So comparisons like:
```python
y < y_max
```
are inherently unstable near boundaries.

---

## 5. Why Epsilon Tweaks Don’t Solve It

No choice of:
- `[y_min, y_max)`
- `[y_min - eps, y_max)`
- `[y_min - eps, y_max - eps)`

is universally safe.

Because floating-point error means atoms near the boundary can land on either side.

---

## 6. Correct Conceptual Approach

### Stop using interval tests to enforce periodicity.

Instead:

1. **Map coordinates to a canonical periodic representation**
2. **Ensure one unique representative per atom**
3. **Avoid boundary ambiguity entirely**

---

## 7. Robust Solution Strategy

### 7.1 Wrap Periodic Coordinates

Convert to reduced coordinates:

```python
def wrap_periodic(coord, lo, hi, tol):
    L = hi - lo
    u = (coord - lo) / L
    u = u - np.floor(u)

    t = tol / L

    # snap boundary values
    u[(u < t) | (u > 1 - t)] = 0.0

    return lo + L * u
```

Apply only to periodic directions:
```python
atoms["y"] = wrap_periodic(atoms["y"], y_min, y_max, tol)
atoms["z"] = wrap_periodic(atoms["z"], z_min, z_max, tol)
```

---

### 7.2 Deduplicate Atoms (If Needed)

If atoms were generated from multiple periodic images:

```python
def dedupe_atoms(atoms, tol):
    key = np.round(atoms[["x","y","z"]].values / tol).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return atoms.iloc[np.sort(idx)]
```

---

## 8. Philosophical Clarification

### Should deduplication be necessary?

**No — not in a perfect construction.**

If your algorithm:
- generates exactly one periodic image per atom
- uses a canonical representation

Then deduplication is unnecessary.

### Why it appears in practice

Because many workflows:

1. Generate atoms from multiple images
2. Clip using floating-point comparisons

This creates duplicates artificially.

Deduplication then becomes a *repair step*.

---

## 9. Better Approach: Canonical Generation

Instead of:
> generate → rotate → clip

Use:
> generate directly in a canonical periodic cell

---

## 10. Redesign of GBMaker

### Current approach (problematic)

- Generate large cubic lattice
- Rotate
- Clip with inequalities
- Hope boundaries behave

### Proposed approach

#### 10.1 Work in crystal (lattice) coordinates

- Use integer lattice indices
- Use basis atoms
- Avoid floating-point boundary decisions

---

### 10.2 Define a Supercell Matrix

From integer grain vectors:

```
H = [nx*g_x, ny*g_y, nz*g_z]
```

Where:
- `g_x, g_y, g_z` come from approximated rotation
- `nx, ny, nz` are repeat counts

---

### 10.3 Enumerate Sites in Reduced Coordinates

For each lattice point `n` and basis atom `b`:

```
u = H⁻¹ (n + b)
```

Accept atom if:
```
0 ≤ u < 1   (half-open interval)
```

Apply tolerance:
- values near 0 → snap to 0
- values near 1 → exclude

---

### 10.4 Convert to Cartesian After Selection

```
r = R @ (a0 * (n + b)) + origin
```

---

## 11. Why This Works

- Periodicity handled **exactly**, not approximately
- No ambiguity at boundaries
- No dependence on epsilon tuning
- One atom per equivalence class guaranteed

---

## 12. Interface Handling (Grain Boundaries)

Use half-open convention:

- Left grain: `[0, 1)`
- Right grain: `[0, 1)`

This ensures:
- no overlapping atoms at interface
- no missing atoms

---

## 13. What to Remove

From current code:

- Cartesian clipping for periodic directions
- Boundary epsilon hacks
- reliance on `inside_box` for periodic filtering

---

## 14. Key Takeaways

### Core Principle

Periodic coordinates are not real intervals — they are **equivalence classes modulo L**.

### Therefore

- Interval comparisons are inherently unstable
- Canonical mapping is required
- Floating-point ambiguity must be resolved explicitly

---

## 15. Final Mental Model

Instead of asking:

> “Is this atom inside the box?”

Ask:

> “What is the unique periodic representative of this atom?”

Once you do that, the entire class of boundary problems disappears.
