# StepForge import performance review

A profile-driven review of the STEP import path, with the fixes it produced.
Everything here is measured on real data, not estimated; the scripts are small
and reproducible (see [Reproducing](#reproducing)).

- **Test file:** `Voron_2.4r2_Assembly.step` — 241 MB, 4.5 M source lines,
  3,645,173 parsed instances, 100,733 faces, 1314 solids, 1551
  `NEXT_ASSEMBLY_USAGE_OCCURRENCE`, 7192 styled colours, ~2.43 GB resident.
- **Machine:** 8 logical CPUs (4 cores / 8 threads), 31 GB RAM, Linux.
- **Importer:** StepForge 3.9.8, NumPy 2.5.3, Python 3.14.

The importer is pure Python + NumPy. It runs in two kinds of process: Blender's
own process (the UI) and background worker processes (`core/workers.py`) that do
parsing, tessellation and curve fitting. The core (`core/`) has no Blender
dependency and is what all of the numbers below exercise.

## Where the time went (before any fix)

| Phase | Cost | Note |
|---|---|---|
| Parse (`parser.parse_file`) | 90–110 s | single-threaded, per process |
| Assembly tree (`assembly.build_tree`) | **356 s** | O(NAUOs × CDSRs) |
| Surface records + planar-only edges | ~35 s | |
| Edge hints (`_register_edge_hints`) | ~94 s | single-threaded |
| `_pd_to_face_lists` / `face_colours` / `_plan_assembly` | 1.5 / 0.6 / 2.5 s | |
| Tessellation (`_process_face`) | ~14 ms/face typical | but see [pathological faces](#known-remaining-pathological-faces) |

Two structural problems dominated everything else:

1. **Every worker re-parsed the whole file.** `_tessellate_job` called
   `parse_file` itself, so an `n`-worker import paid `n × ~100 s` and
   `n × 2.43 GB` before the first face was tessellated.
2. **The pool never used more than half the cores.** `workers.worker_count()`
   returned `(os.cpu_count() or 2) // 2`, hard-capped, with no way to change it.

## Fixes

All changes are split into focused, independently reviewable PRs.

### 1. Index NAUO transforms once — `build_tree` O(n²) → O(n)

`build_tree` resolved each occurrence's placement by scanning **every**
`CONTEXT_DEPENDENT_SHAPE_REPRESENTATION` for each occurrence: on this file
1551 × 1551 ≈ **2.4 M** redundant lookups, 356 s. Building the whole
`NAUO → 4×4 matrix` map in one pass brings it to **0.8 s**, with an identical
assembly plan.

### 2. Fork-shared parse

Workers are now **forked after the parent has parsed**, so they inherit the
parsed `StepFile` copy-on-write instead of re-parsing it. `_tessellate_pooled`
tries `fork`, falls back to `spawn`, then to the sequential path.

| 400-face pool | spawn | fork |
|---|---|---|
| wall time | 134.2 s | **2.0 s** |
| per-worker parse | ~100 s × n | 0 (inherited) |

Parallel output hashes identically to the sequential output (SHA-256 over
verts/faces/face ids/metadata) for both spawn and fork.

### 3. Use all cores by default, expose a preference

The auto pool size is now every logical CPU (capped) rather than half, and a
`Worker cores` add-on preference (0 = automatic) is passed to import and BREP
export. Workers already run at below-normal priority, so the UI keeps a core.

On this 4-core/8-thread box the 4 → 8 change is nearly flat (fork pool: 2.0 s at
n=4, 1.9 s at n=8); the default matters most on machines with more *physical*
cores, and the preference is the escape hatch.

### 4. Edge hints on the worker pool

The edge-hint pre-pass (`_register_edge_hints`) walks every unique face's
boundary to space a cylinder/cone face's straight edges like its interior grid.
It was the largest remaining single-threaded block: **89 s**. It now runs on the
same fork pool as tessellation (batched per IPC message, since each face's own
work is tiny), with the identical sequential/spawn fallbacks.

The merge is a plain min over faces, so split across workers it produces the
same dict: on this file **57,137 keys, identical**, at **24.2 s** (memory-bound
on 4 physical cores, not IPC-bound).

## Known remaining hotspots

### Known remaining: pathological faces

A handful of faces in this file take **minutes each** even when run alone
(so it is not a scheduling or memory problem):

```
#1732409  #1732434  #1732436  #1732438  #1735056  #1735058
```

Profiling `#1732409` (a `BSplineSurface`) shows the cost is **not** the adaptive
grid sampler. It is constrained-Delaunay boundary recovery:

```
_recover_edge    tessellate.py:1143    scans EVERY edge, per constrained edge
_segs_cross      tessellate.py:1120    4.57 M calls in 25 s
```

`_recover_edge` forces each boundary edge into the triangulation by scanning
**all** edges and flipping the first one that crosses the target, up to
`guard_max = max(64, 4 * len(pts))` flips. The face has **723 boundary points /
723 constrained edges**; in 20 s only **56** were recovered (~318 ms each, ~5
full scans of ~12 k edges per edge), i.e. ~230 s for one pass — and because a
few recoveries fail, `_constrained_delaunay` retries the whole thing with a
larger helper triangle (up to 3 passes). That is the "hang" around the middle
of a full import.

This is **pre-existing** and independent of the fixes above: ordinary faces
(~14 ms) have few boundary points and no thrashing.

**Proposed follow-up (not in this review's PRs):** replace the
scan-everything-and-flip loop with a corridor walk that only visits the
triangles the segment actually crosses (the standard constrained-segment
insertion), turning each recovery from O(num_edges) into O(triangles crossed).
It must be validated by comparing geometry hashes against the current output,
since it can produce a different (still valid) interior triangulation. Lowering
`guard_max` back toward a small constant for non-degenerate cases is a
cheaper, lower-risk mitigation worth measuring first.

## Reproducing

Small harnesses (kept out of the repo deliberately; they need the real file):

- `bench_stepforge.py` — per-phase timing of a full import (`SF_VERBOSE=1` for
  live progress, `SF_NO_FORK=1` to force spawn).
- `bench_micro.py` — fork vs spawn on a bounded face sample, with a geometry
  digest for equality checking.
- `bench_plan.py` — parse / `build_tree` / `_plan_assembly` / `face_colours`.
- `bench_face_prof.py`, `bench_recover.py` — `cProfile` and counters for a
  single face, with a `SIGALRM` timeout.

Run them with a NumPy-capable interpreter, e.g. `/usr/bin/python3.14`, from the
repository root.
