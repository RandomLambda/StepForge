"""Reconstruct freeform B-spline surfaces (STEP `B_SPLINE_SURFACE_WITH_KNOTS`)
from the triangles `fit.py`'s primitive fitter could not identify.

On an organic, smoothly curved part no plane, sphere, cylinder or cone fits,
and every triangle would fall through to a one-triangle ADVANCED_FACE. This
tier sits between "fits a primitive" and that fallback: segment the leftover
region into patches a rectangular B-spline can represent, parameterize each
patch and fit a B-spline surface to it. Whatever cannot be fitted within
tolerance falls through to the `merge_coplanar` / per-triangle path
untouched, so this stage can only improve the output.

1. Segmentation: `vsa.partition` (variational shape approximation, compact
   regions) by default, or `grow_regions` (breadth-first over triangles whose
   normal stays within `max_normal_dev_deg` of the seed's). Every region must
   have a clean, simply connected boundary (`fit.boundary_loops`).
2. Parameterization, cheapest first (`_parameterizations`): orthogonal
   projection onto the region's mean-normal plane (valid for a height field,
   checked for folds in `parameterize`), then mean-value coordinates
   (`param.py`) for any disk-shaped patch.
3. `fit_bspline`: regularized linear least squares for the control net
   (Piegl & Tiller, *The NURBS Book* ch. 9) with a second-difference
   smoothing term, which keeps control points near uncovered corners of the
   parameter rectangle from running off. The control-point count climbs a
   ladder until the fit holds within tolerance; a region that never does is
   re-partitioned smaller (`fit_freeform_regions`).

Pure NumPy, no Blender dependency, same as `fit.py`.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import curvature as _curv
from . import fit as _fit
from . import param as _param
from . import vsa as _vsa
from .geometry import _basis_funs, _find_span

# Growth caps on the RAW mesh boundary. What must fit under the reader's
# constrained-Delaunay limit (`fit.MAX_BOUNDARY_PTS`) is the SIMPLIFIED
# boundary, checked in `fit_region`; a straight tessellation run collapses to
# one edge, so a region with 160 raw boundary points routinely writes well
# under 90.
MAX_REGION_BOUNDARY = 160
MAX_REGION_TRIS = 1500

# Segmentation used to propose patches: "vsa" (`vsa.py`; compact regions,
# which is what decides whether a patch can be flattened and fitted) or "grow"
# (breadth-first with a normal cone).
SEGMENTATION = "vsa"

# Triangles per region VSA aims for on its first pass; a region that misses
# the tolerance is re-partitioned smaller.
VSA_TARGET_TRIS = 400

# Below this many triangles a B-spline face costs more entities (control net
# and boundary) than it saves against the flat fallback, so small leftovers
# stay on the `merge_coplanar` path. 40 works for organic parts and for
# primitive-heavy CAD parts alike, whose leftovers are many small slivers
# rather than large curved regions.
MIN_REGION_TRIS = 40

# How far a triangle's normal may differ from its region's seed normal. Below
# 90 degrees the projection stays injective (`parameterize` verifies it per
# region); beyond that a wider cone grows bigger regions but foreshortens
# steep parts and degrades the fit.
MAX_NORMAL_DEV_DEG = 55.0

# How far a triangle's shape index (`curvature.shape_index`) may differ from
# the triangle it grew from before a region stops growing. It puts patch
# borders on ridges, valleys and inflection lines, where normals alone can be
# nearly identical across a dome-to-saddle transition. Scale-free, so it means
# the same on a 5 mm feature and a 5 m one. The test is local (against the
# parent triangle, not the seed), so a patch may change character slowly
# across its span. `None` disables it.
SHAPE_INDEX_TOL = 0.35

# Degree of the fitted surface in both parameter directions. Cubic is the
# CAD default and what every STEP reader is best-tested on.
DEGREE = 3

# Control-net ladder: total control points tried, smallest first. Stops at
# the first rung that holds the tolerance (or at the point where refining
# further has clearly stopped paying -- see `fit_bspline`).
CTRL_LADDER = (16, 25, 36, 49, 64, 81, 100, 144, 196, 256)

# Never ask for more than this share of the region's vertex count in control
# points: beyond it the fit turns into under-determined interpolation that can
# oscillate between samples while the error at the samples stays small.
MAX_CTRL_FRACTION = 0.6

# How many times the tolerance a region's plane-fit error must exceed before a
# B-spline face is worth writing. A nearly flat region is cheaper as a few
# merged planes than as a control net with a full boundary; 1.2 only guards
# the almost-flat case.
MIN_CURVATURE_GAIN = 1.2

# Weight of the second-difference smoothing term relative to the data term;
# small enough to bias the fit far below tolerance on real data (the final
# surface is re-checked against the tolerance anyway).
SMOOTH_WEIGHT = 0.02

# A control-net rung must bring the error below this fraction of the previous
# rung's to justify trying a bigger net (see `fit_bspline`). At 1.0 or more the
# test never fires and the ladder always runs to its cap, which measured better
# on coverage and face count at the same run time: runs that give up early
# get subdivided and re-fitted instead.
LADDER_STALL = 1.01


# A facet is "large" when its longest edge exceeds this many times the
# median longest edge of its region; only large facets get extra sample
# points (see `sample_points`), so a region of similar facets -- an organic
# mesh -- is fitted from exactly the points it always was.
LARGE_FACET_RATIO = 2.0

# Extra points on large facets are laid out at the spacing that would give a
# region about this many points if it were covered by them entirely, never
# finer than the tolerance. A control net of ~250 points has about ten knot
# spans across, so this puts several samples into each.
SAMPLE_TARGET = 2000

# Upper bound on those extra points per region; the spacing is widened until
# they fit.
MAX_EXTRA_SAMPLES = 3000


@dataclass
class BSplineFit:
    """A fitted freeform patch, in the same duck-typed shape as `fit.py`'s
    PlaneFit/CylinderFit/... so `brep_export` can treat it uniformly."""
    kind: str = "bspline"
    deg_u: int = DEGREE
    deg_v: int = DEGREE
    ctrl: np.ndarray = None        # (n_u, n_v, 3) control point grid
    knots_u: np.ndarray = None     # full (expanded) clamped knot vector
    knots_v: np.ndarray = None
    max_err: float = 0.0


# ---------------------------------------------------------------------------
# 1. Height-field region growing
# ---------------------------------------------------------------------------


def _face_normals(verts, faces, face_idx):
    n = {}
    for f in face_idx:
        a, b, c = faces[f]
        nrm = _fit._face_normal(verts[a], verts[b], verts[c])
        if nrm is not None:
            n[f] = nrm
    return n


def _adjacency(faces, face_idx):
    """Face adjacency restricted to `face_idx`, across 2-manifold edges only
    (identical rule to `fit.merge_coplanar`, so a region can never leak into
    triangles that another stage already claimed)."""
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for f in face_idx:
        a, b, c = faces[f]
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_faces.setdefault(key, []).append(f)
    adj: dict[int, list[int]] = {f: [] for f in face_idx}
    for fs in edge_faces.values():
        if len(fs) == 2:
            i, j = fs
            adj[i].append(j)
            adj[j].append(i)
    return adj


def _toggle_edges(bset, bdeg, faces, f):
    """Add or remove `f`'s three edges from the running boundary-edge set and
    keep each vertex's boundary degree with it: an edge shared with a triangle
    already in the region stops being boundary, a new edge becomes boundary.
    Boundary size and cleanliness stay available in O(1) per candidate."""
    a, b, c = faces[f]
    for u, v in ((a, b), (b, c), (c, a)):
        key = (u, v) if u < v else (v, u)
        if key in bset:
            bset.discard(key)
            bdeg[u] = bdeg.get(u, 0) - 1
            bdeg[v] = bdeg.get(v, 0) - 1
        else:
            bset.add(key)
            bdeg[u] = bdeg.get(u, 0) + 1
            bdeg[v] = bdeg.get(v, 0) + 1


def _would_pinch(bdeg, faces, f) -> bool:
    """After adding `f`, does any of its vertices carry more than two boundary
    edges? That is a pinch (two parts of the region touching at one vertex),
    which makes the boundary ambiguous for `fit.boundary_loops`. A pinched
    region cannot be repaired, only discarded and subdivided, so it is checked
    during growth."""
    return any(bdeg.get(v, 0) > 2 for v in faces[f])


def _absorb_enclosed(faces, adj, region, region_set, claimed, max_absorb):
    """Swallow any unclaimed island the region has completely surrounded.

    A region wrapped around a small patch of rejected triangles has a hole (a
    second boundary loop) and is no longer a disk, so it loses the mean-value
    parameterization's guarantee. Absorbing the island costs a handful of
    triangles that fit slightly worse. Only islands enclosed entirely by this
    region are taken, not components that also touch another region or the
    edge of the leftover set."""
    frontier = [nb for f in region for nb in adj.get(f, ())
                if nb not in region_set and nb not in claimed]
    absorbed = []
    seen = set()
    for start in frontier:
        if start in seen:
            continue
        comp = []
        stack = [start]
        seen.add(start)
        enclosed = True
        while stack:
            f = stack.pop()
            comp.append(f)
            neigh = adj.get(f, ())
            if len(neigh) < 3:
                enclosed = False      # touches the edge of the leftover set
                break
            for nb in neigh:
                if nb in region_set:
                    continue
                if nb in claimed:
                    enclosed = False  # touches another region
                    break
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
            if not enclosed:
                break
        if enclosed and len(comp) <= max_absorb:
            absorbed.extend(comp)
    return absorbed


def grow_regions(verts: np.ndarray, faces: Sequence[tuple[int, int, int]],
                 face_idx: Sequence[int],
                 max_normal_dev_deg: float = MAX_NORMAL_DEV_DEG,
                 max_boundary: int = MAX_REGION_BOUNDARY,
                 max_tris: int = MAX_REGION_TRIS,
                 shape_index=None,
                 shape_tol: float | None = SHAPE_INDEX_TOL,
                 absorb_holes: bool = True) -> list[list[int]]:
    """Cut `face_idx` into height-field regions (see module docstring).

    Breadth-first: a FIFO frontier grows a roughly circular geodesic disc,
    which has the smallest boundary for its area, and boundary size is the cap
    that limits growth (a LIFO frontier grows a snaking strip with several
    times the boundary).

    `shape_index`, if given, is the per-triangle shape index of the whole mesh
    (`curvature.face_shape_index`); growth then refuses to cross an edge where
    it jumps by more than `shape_tol` (a ridge, valley or inflection line),
    tested locally against the triangle the growth came from.

    Regions come back in a deterministic order (seeded in `face_idx` order),
    so the export is reproducible."""
    normals = _face_normals(verts, faces, face_idx)
    adj = _adjacency(faces, face_idx)
    cos_thresh = math.cos(math.radians(max_normal_dev_deg))
    claimed = set()
    regions: list[list[int]] = []

    for seed in face_idx:
        if seed in claimed or seed not in normals:
            continue
        seed_n = normals[seed]
        region = [seed]
        claimed.add(seed)
        bset = set()
        bdeg: dict[int, int] = {}
        _toggle_edges(bset, bdeg, faces, seed)
        queue = deque((nb, seed) for nb in adj[seed])
        while queue:
            if len(region) >= max_tris:
                break
            cand, parent = queue.popleft()
            if cand in claimed or cand not in normals:
                continue
            if float(np.dot(normals[cand], seed_n)) < cos_thresh:
                continue
            if (shape_index is not None and shape_tol is not None
                    and abs(float(shape_index[cand])
                            - float(shape_index[parent])) > shape_tol):
                continue  # a ridge/valley/inflection line runs between them
            _toggle_edges(bset, bdeg, faces, cand)
            if len(bset) > max_boundary or _would_pinch(bdeg, faces, cand):
                _toggle_edges(bset, bdeg, faces, cand)   # undo
                continue
            claimed.add(cand)
            region.append(cand)
            for nb in adj[cand]:
                if nb not in claimed:
                    queue.append((nb, cand))
        if absorb_holes:
            region_set = set(region)
            extra = _absorb_enclosed(faces, adj, region, region_set, claimed,
                                      max_absorb=max(len(region) // 4, 20))
            for f in extra:
                if f in claimed:
                    continue
                claimed.add(f)
                region.append(f)
        regions.append(region)
    return regions


# ---------------------------------------------------------------------------
# 2. Parameterization
# ---------------------------------------------------------------------------


def region_frame(verts, faces, region):
    """Area-weighted mean normal of the region plus an orthonormal (e1, e2)
    basis of the plane it projects onto. `e1 x e2 == normal` (see
    `fit._orthonormal_basis`), so a loop wound counter-clockwise around the
    outward normal stays counter-clockwise in (u, v) and the writer can keep
    `ADVANCED_FACE`'s sense flag `.T.`."""
    acc = np.zeros(3)
    for f in region:
        a, b, c = faces[f]
        # un-normalized cross product = 2 * area * unit normal, i.e. already
        # area-weighted
        acc += np.cross(verts[b] - verts[a], verts[c] - verts[a])
    n = float(np.linalg.norm(acc))
    if n < 1e-14:
        return None
    normal = acc / n
    e1, e2 = _fit._orthonormal_basis(normal)
    return normal, e1, e2


def _facet_samples(verts, faces, region, spacing, min_edge=0.0):
    """Points on the facets of `region` whose longest edge exceeds `spacing`
    and `min_edge`, on a lattice with at most that spacing. Returned as
    `(points, tri, w)`: `tri` is the index into `region` of the facet each
    point lies on and `w` its barycentric weights there (so a
    parameterization that is linear on a facet can place it without knowing
    how it was made)."""
    P = verts[np.array([faces[f] for f in region], dtype=np.int64)]   # (n, 3, 3)
    edge = np.stack([np.linalg.norm(P[:, 1] - P[:, 0], axis=1),
                     np.linalg.norm(P[:, 2] - P[:, 1], axis=1),
                     np.linalg.norm(P[:, 0] - P[:, 2], axis=1)], axis=1)
    longest = edge.argmax(axis=1)
    area2 = np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    pts, tri, wts = [], [], []
    for t in np.nonzero(edge.max(axis=1) > max(spacing, min_edge))[0]:
        k = int(longest[t])
        base = float(edge[t, k])
        rows = max(1, math.ceil(area2[t] / base / spacing - 1e-9))
        s = np.arange(rows) / rows          # rows parallel to the long edge
        m = np.maximum(1, np.ceil((1.0 - s) * base / spacing - 1e-9).astype(np.int64))
        row = np.repeat(np.arange(rows), m + 1)
        first = np.repeat(np.cumsum(m + 1) - (m + 1), m + 1)
        q = (np.arange(len(row)) - first) / m[row]
        keep = ~((row == 0) & ((q == 0.0) | (q == 1.0)))     # corners are vertices
        row, q = row[keep], q[keep]
        sr = s[row]
        w = np.empty((len(row), 3))
        w[:, k], w[:, (k + 1) % 3], w[:, (k + 2) % 3] = (1 - sr) * (1 - q), (1 - sr) * q, sr
        pts.append(w @ P[t])
        tri.append(np.full(len(row), t, dtype=np.int64))
        wts.append(w)
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64), np.zeros((0, 3))
    return np.vstack(pts), np.concatenate(tri), np.vstack(wts)


def sample_points(verts, faces, region, vids, tolerance=None):
    """The point set a region is fitted against: its vertices, every
    triangle's centroid, and extra points on triangles that are large next to
    the rest of the region.

    Vertices alone cannot make the tolerance claim honest: a least-squares
    surface with nearly as many control points as data points can pass every
    vertex yet bulge away from the mesh between them. Centroids sit on the
    facets and triple the constraint count cheaply. Where one facet spans a
    large part of the region (a flat face triangulated from its outline alone),
    centroids are 10 mm or more apart and are not enough, so facets longer than
    `LARGE_FACET_RATIO` times the region's median get a lattice of points
    (`SAMPLE_TARGET` for the region, no finer than the tolerance). A region of
    similar facets gets none.

    Returned as `(points, n_vertices, tri, w)`. Points are ordered vertices,
    centroids (in `region` order), extras; `tri`/`w` give the facet and
    barycentric weights of everything after the vertices."""
    faces_a = np.array([faces[f] for f in region], dtype=np.int64)
    P = verts[faces_a]                                          # (n, 3, 3)
    pts = [verts[vids], P.mean(axis=1)]
    tri = [np.arange(len(region), dtype=np.int64)]
    w = [np.full((len(region), 3), 1.0 / 3.0)]
    longest = np.max(np.stack([np.linalg.norm(P[:, 1] - P[:, 0], axis=1),
                               np.linalg.norm(P[:, 2] - P[:, 1], axis=1),
                               np.linalg.norm(P[:, 0] - P[:, 2], axis=1)]), axis=0)
    large = LARGE_FACET_RATIO * float(np.median(longest))
    if float(longest.max()) > large:
        area = 0.5 * float(np.linalg.norm(np.cross(P[:, 1] - P[:, 0],
                                                   P[:, 2] - P[:, 0]), axis=1).sum())
        spacing = math.sqrt(area / SAMPLE_TARGET)
        if tolerance:
            spacing = max(spacing, float(tolerance))
        ex, et, ew = _facet_samples(verts, faces, region, spacing, min_edge=large)
        while len(ex) > MAX_EXTRA_SAMPLES:
            spacing *= math.sqrt(len(ex) / MAX_EXTRA_SAMPLES) * 1.05
            ex, et, ew = _facet_samples(verts, faces, region, spacing, min_edge=large)
        if len(ex):
            pts.append(ex)
            tri.append(et)
            w.append(ew)
    return np.vstack(pts), len(vids), np.concatenate(tri), np.vstack(w)


def parameterize(verts, faces, region, vids, frame, pts_all=None):
    """Project the region onto its frame plane and rescale to [0, 1]^2.
    Returns `(uv, extent_u_mm, extent_v_mm)` or None if the projection
    degenerates (zero extent) or folds over itself (a triangle whose
    projected area flips sign -- the region is then not a height field
    after all and must not be fitted this way).

    `pts_all`, if given, is the full sample set (`sample_points`: vertices,
    facet centroids, extra points on large facets) to return parameters
    for; the fold check and the normalization always come from the
    vertices alone."""
    _, e1, e2 = frame
    pts = verts[vids]
    origin = pts.mean(axis=0)
    rel = pts - origin
    x = rel @ e1
    y = rel @ e2
    ext_u = float(x.max() - x.min())
    ext_v = float(y.max() - y.min())
    if ext_u < 1e-9 or ext_v < 1e-9:
        return None

    pos = {v: i for i, v in enumerate(vids)}
    for f in region:
        a, b, c = faces[f]
        ia, ib, ic = pos[a], pos[b], pos[c]
        area2 = ((x[ib] - x[ia]) * (y[ic] - y[ia])
                 - (x[ic] - x[ia]) * (y[ib] - y[ia]))
        if area2 <= 0.0:
            return None  # folded/degenerate in projection -- reject region

    if pts_all is None:
        px, py = x, y
    else:
        rel_all = pts_all - origin
        px = rel_all @ e1
        py = rel_all @ e2
    # Normalized against the VERTEX extent: every extra point is a convex
    # combination of three vertices, so it stays in range, and one shared
    # normalization keeps all groups in the same parameter frame.
    u = (px - x.min()) / ext_u
    v = (py - y.min()) / ext_v
    return np.stack([u, v], axis=1), ext_u, ext_v


# ---------------------------------------------------------------------------
# 3. Least-squares B-spline surface fit
# ---------------------------------------------------------------------------


def clamped_knots(n_ctrl: int, deg: int) -> np.ndarray:
    """Clamped, uniformly-spaced knot vector over [0, 1] for `n_ctrl`
    control points of degree `deg` (length n_ctrl + deg + 1)."""
    n_interior = n_ctrl - deg - 1
    interior = [(i + 1) / (n_interior + 1) for i in range(max(n_interior, 0))]
    return np.array([0.0] * (deg + 1) + interior + [1.0] * (deg + 1))


def basis_matrix(t: np.ndarray, knots: np.ndarray, deg: int,
                 n_ctrl: int) -> np.ndarray:
    """Dense (len(t), n_ctrl) matrix of B-spline basis values -- the design
    matrix for one parameter axis. The recurrence is the reader's own
    (`geometry._basis_funs`), run on all parameters at once, so the fitter
    and the evaluator can never disagree about what a knot vector means (and
    the values are the same to the last bit)."""
    t = np.asarray(t, dtype=float)
    m = len(t)
    n = n_ctrl - 1
    u = np.clip(t, float(knots[deg]), float(knots[n + 1]))
    span = np.clip(np.searchsorted(knots, u, side="right") - 1, deg, n)
    N = np.zeros((m, deg + 1))
    N[:, 0] = 1.0
    left = np.zeros((m, deg + 1))
    right = np.zeros((m, deg + 1))
    for j in range(1, deg + 1):
        left[:, j] = u - knots[span + 1 - j]
        right[:, j] = knots[span + j] - u
        saved = np.zeros(m)
        for r in range(j):
            denom = right[:, r + 1] + left[:, j - r]
            temp = np.divide(N[:, r], denom, out=np.zeros(m), where=denom != 0)
            N[:, r] = saved + right[:, r + 1] * temp
            saved = left[:, j - r] * temp
        N[:, j] = saved
    B = np.zeros((m, n_ctrl))
    rows = np.arange(m)
    for k in range(deg + 1):
        B[rows, span - deg + k] = N[:, k]
    return B


def _smoothing_matrix(ncu: int, ncv: int) -> np.ndarray:
    """Second-difference (discrete bending) operator over the control net, in
    both directions: rows are (1, -2, 1) stencils and `R @ P = 0` means the net
    is a plane. Penalizing `|R P|` biases the fit toward the smoothest net that
    explains the data and gives control points that no data point touches a
    defined value instead of leaving the system rank-deficient."""
    rows = []
    for j in range(ncv):
        for i in range(1, ncu - 1):
            r = np.zeros(ncu * ncv)
            r[(i - 1) * ncv + j] = 1.0
            r[i * ncv + j] = -2.0
            r[(i + 1) * ncv + j] = 1.0
            rows.append(r)
    for i in range(ncu):
        for j in range(1, ncv - 1):
            r = np.zeros(ncu * ncv)
            r[i * ncv + (j - 1)] = 1.0
            r[i * ncv + j] = -2.0
            r[i * ncv + (j + 1)] = 1.0
            rows.append(r)
    return np.array(rows) if rows else np.zeros((0, ncu * ncv))


def _solve_ctrl(uv, pts, ncu, ncv, deg_u, deg_v, smooth_weight):
    """One least-squares solve for a fixed control-net size. Returns
    `(ctrl, knots_u, knots_v, max_err)`."""
    U = clamped_knots(ncu, deg_u)
    V = clamped_knots(ncv, deg_v)
    Bu = basis_matrix(uv[:, 0], U, deg_u, ncu)
    Bv = basis_matrix(uv[:, 1], V, deg_v, ncv)
    A = (Bu[:, :, None] * Bv[:, None, :]).reshape(len(uv), ncu * ncv)
    R = _smoothing_matrix(ncu, ncv)
    if len(R):
        M = np.vstack([A, smooth_weight * R])
        rhs = np.vstack([pts, np.zeros((len(R), 3))])
    else:
        M, rhs = A, pts
    sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    err = np.linalg.norm(A @ sol - pts, axis=1)
    ctrl = sol.reshape(ncu, ncv, 3)
    return ctrl, U, V, float(err.max())


def fit_bspline(uv: np.ndarray, pts: np.ndarray, tolerance: float,
                ext_ratio: float = 1.0, deg: int = DEGREE,
                smooth_weight: float = SMOOTH_WEIGHT,
                ladder: Sequence[int] = CTRL_LADDER,
                n_free: int | None = None):
    """Least-squares fit a B-spline surface to `pts` at parameters `uv`,
    raising the control-net size along `ladder` until the worst deviation is
    within `tolerance`.

    `ext_ratio` is the patch's real u:v extent ratio; control points are
    distributed along it so a long, narrow patch gets its resolution where the
    geometry is.

    Returns `(fit_or_None, best_error_seen)`. On failure the second value is
    the closest this region could be approximated, which tells a user whose
    tolerance is unreachable for their mesh what would work (reported through
    `BrepStats`)."""
    m = len(pts)
    # The over-fitting cap counts VERTICES, not all samples: centroids are
    # determined by the vertices around them and carry no independent shape
    # information.
    cap = max((deg + 1) ** 2,
              int((m if n_free is None else n_free) * MAX_CTRL_FRACTION))
    prev_err = None
    best_err = float("inf")
    for total in ladder:
        ar = math.sqrt(max(ext_ratio, 1e-3))
        ncu = max(deg + 1, round(math.sqrt(total) * ar))
        ncv = max(deg + 1, round(math.sqrt(total) / ar))
        if ncu * ncv > cap:
            break
        ctrl, U, V, err = _solve_ctrl(uv, pts, ncu, ncv, deg, deg, smooth_weight)
        best_err = min(best_err, err)
        if err <= tolerance:
            return BSplineFit(deg_u=deg, deg_v=deg, ctrl=ctrl, knots_u=U,
                              knots_v=V, max_err=err), best_err
        # Stop early when a rung barely lowers the error (`LADDER_STALL`): the
        # region's own detail is what the tolerance fails against, not a lack
        # of freedom, and the caller subdivides the region instead.
        if prev_err is not None and err > prev_err * LADDER_STALL:
            break
        prev_err = err
    return None, best_err


# ---------------------------------------------------------------------------
# Region -> face, end to end
# ---------------------------------------------------------------------------


def _loop_signed_area(loop, verts, frame):
    _normal, e1, e2 = frame
    pts = verts[loop]
    x = pts @ e1
    y = pts @ e2
    n = len(loop)
    a2 = 0.0
    for i in range(n):
        j = (i + 1) % n
        a2 += x[i] * y[j] - x[j] * y[i]
    return 0.5 * a2


def _dp_keep(pts, i, j, keep, tolerance):
    """Douglas-Peucker: mark every point between `i` and `j` that is further
    than `tolerance` from the chord i->j (recursively)."""
    if j <= i + 1:
        return
    a = pts[i]
    ab = pts[j] - a
    ab_len = float(np.linalg.norm(ab))
    seg = pts[i + 1:j] - a
    if ab_len < 1e-12:
        d = np.linalg.norm(seg, axis=1)
    else:
        t = np.clip((seg @ ab) / (ab_len * ab_len), 0.0, 1.0)
        d = np.linalg.norm(seg - t[:, None] * ab, axis=1)
    k = int(np.argmax(d))
    if d[k] <= tolerance:
        return
    split = i + 1 + k
    keep[split] = True
    _dp_keep(pts, i, split, keep, tolerance)
    _dp_keep(pts, split, j, keep, tolerance)


def simplify_freeform_loop(loop, verts, tolerance):
    """Drop boundary vertices that lie within `tolerance` of the straight line
    between the vertices that survive around them (Douglas-Peucker on the
    closed loop, split at its two most distant points).

    Douglas-Peucker rather than `fit.merge_collinear`: that merges on turn
    angle alone, with no bound on how far the boundary moves, which suits CAD
    edges (straight or exact arcs) but would cut a long gentle freeform curve
    down to its chord. Here the error bound is the tolerance.

    Two neighbouring faces simplify their shared chain independently and can
    leave sub-tolerance T-junctions, which the reimporter's vertex weld closes;
    in exchange, straight tessellation runs along a patch border stop costing
    one EDGE_CURVE chain each."""
    n = len(loop)
    if n <= 4:
        return loop
    pts = verts[loop]
    # Anchor the two splits at the most distant pair, so neither half of the
    # closed loop starts out degenerate.
    i0 = 0
    d0 = np.linalg.norm(pts - pts[0], axis=1)
    i1 = int(np.argmax(d0))
    if i1 == i0:
        return loop
    keep = [False] * n
    keep[i0] = keep[i1] = True
    _dp_keep(pts, i0, i1, keep, tolerance)
    # second arc: wrap around by rotating indices
    order = list(range(i1, n)) + [i0]
    rot = pts[order]
    keep2 = [False] * len(order)
    keep2[0] = keep2[-1] = True
    _dp_keep(rot, 0, len(order) - 1, keep2, tolerance)
    for k, orig in enumerate(order):
        if keep2[k]:
            keep[orig] = True
    new_loop = [loop[i] for i in range(n) if keep[i]]
    return new_loop if len(new_loop) >= 3 else loop


def _line_segs(loop, verts):
    """Boundary segments for a freeform face: every edge is a straight LINE
    between two boundary vertices of the mesh, which is what the mesh had."""
    n = len(loop)
    return [_fit.EdgeSeg(kind="line", p0=verts[loop[i]],
                         p1=verts[loop[(i + 1) % n]], sagitta=0.0)
            for i in range(n)]


def _parameterizations(verts, faces, region, vids, pos, pts, n_verts,
                       raw_loops, frame, tri, w):
    """Yield `(uv, ext_ratio)` candidates for one region, cheapest first.

    The projection onto the region's mean-normal plane is free but only valid
    for a height field and rejects itself (fold check) otherwise. The
    mean-value parameterization (`param.py`) works for any patch that is
    topologically a disk, however far it wraps, at the cost of a linear solve.
    Yielding lets a region whose projection is valid but whose fit misses the
    tolerance try the second before being subdivided."""
    par = parameterize(verts, faces, region, vids, frame, pts_all=pts)
    if par is not None:
        uv, ext_u, ext_v = par
        yield uv, ext_u / max(ext_v, 1e-9)
    if len(raw_loops) != 1:
        return          # not a disk: Tutte's guarantee does not apply
    res = _param.mean_value_uv(verts, faces, region, vids, raw_loops[0])
    if res is None:
        return
    vuv, aspect = res
    # A sample on a facet takes the weighted mean of its corners' parameters:
    # exact for a piecewise-linear parameterization.
    f = np.array([faces[i] for i in region], dtype=np.int64)
    corner_uv = np.stack([vuv[[pos[v] for v in f[:, k]]] for k in range(3)], axis=1)
    cent = np.einsum("mk,mkc->mc", w, corner_uv[tri])
    yield np.vstack([vuv, cent]), aspect


# Why a rejected region is not written as a B-spline face. Only the reasons in
# `SPLITTABLE` can be fixed by cutting the region smaller (see
# `fit_freeform_regions`); the others are verdicts a smaller piece inherits.
SPLITTABLE = ("bad_boundary", "boundary_too_complex", "not_height_field",
              "no_net_within_tolerance")


def fit_region(verts, faces, region, tolerance,
               max_boundary_pts: int | None = None):
    """Try to turn one grown region into a B-spline ADVANCED_FACE.

    Returns `(fit, payload, best_err)`. On success `payload` is the boundary
    as `[(loop, segs), ...]`, outer loop first. On failure `fit` is None and
    `payload` is a reason string (see `SPLITTABLE`), which decides between
    subdividing the region and handing it back to the flat fallback.
    `best_err` is the closest net attempted (inf if none; see `fit_bspline`)."""
    if max_boundary_pts is None:
        max_boundary_pts = _fit.MAX_BOUNDARY_PTS
    if len(region) < MIN_REGION_TRIS:
        return None, "too_small", float("inf")

    vids = sorted({i for f in region for i in faces[f]})
    if len(vids) < 16:
        return None, "too_few_vertices", float("inf")
    pts, n_verts, tri, w = sample_points(verts, faces, region, vids, tolerance)
    pos = {v: i for i, v in enumerate(vids)}

    # Already flat (or nearly so) within tolerance? Then this is
    # `merge_coplanar`'s job -- see MIN_CURVATURE_GAIN.
    plane_err = _fit.plane_fit(pts[:n_verts]).max_err
    if plane_err <= tolerance:
        return None, "planar", float("inf")
    if plane_err < MIN_CURVATURE_GAIN * tolerance:
        return None, "nearly_planar", float("inf")

    raw_loops, ok = _fit.boundary_loops(faces, region)
    if not ok or not raw_loops:
        return None, "bad_boundary", float("inf")
    loops = [simplify_freeform_loop(l, verts, tolerance) for l in raw_loops]
    if sum(len(l) for l in loops) > max_boundary_pts:
        return None, "boundary_too_complex", float("inf")

    frame = region_frame(verts, faces, region)
    if frame is None:
        return None, "degenerate_frame", float("inf")

    fitted = None
    best_err = float("inf")
    tried_any = False
    for uv, ext_ratio in _parameterizations(verts, faces, region, vids, pos,
                                            pts, n_verts, raw_loops, frame,
                                            tri, w):
        tried_any = True
        fitted, err = fit_bspline(uv, pts, tolerance, ext_ratio=ext_ratio,
                                   n_free=n_verts)
        best_err = min(best_err, err)
        if fitted is not None:
            break
    if not tried_any:
        return None, "not_height_field", float("inf")
    if fitted is None:
        return None, "no_net_within_tolerance", best_err

    # Outer boundary first (largest enclosed area in the projection plane);
    # `boundary_loops` returns loops in traversal order, which says nothing
    # about which one is the outline and which are holes.
    ordered = sorted(loops, key=lambda l: -abs(_loop_signed_area(l, verts, frame)))
    simplified = [(loop, _line_segs(loop, verts)) for loop in ordered]
    return fitted, simplified, fitted.max_err


def fit_freeform_regions(verts, faces, leftover: Sequence[int], tolerance: float,
                         max_normal_dev_deg: float | None = None,
                         max_tris: int | None = None,
                         min_tris: int | None = None,
                         progress=None, stats: dict | None = None,
                         shape_index=None,
                         shape_tol: float | None = -1.0,
                         segmentation: str | None = None,
                         vsa_target: int | None = None):
    """Full freeform stage over one solid's leftover triangles, with adaptive
    subdivision: a region that no control net can hold within `tolerance` is
    re-partitioned in two and each piece tried again, recursively, down to
    `min_tris`. One fixed tolerance then works across wildly different inputs
    (a gently curved shell gets a few large patches, a coarsely tessellated one
    more, smaller ones), and only what is still unfittable at the floor falls
    through to the flat fallback.

    Returns `(accepted, remaining)`: `accepted` is a list of
    `(fit, simplified, region)` ready for the writer (`region` being the
    triangles it covers, which the caller needs for statistics and to register
    the face in the shared boundary graph); `remaining` is the triangles that
    still need the `merge_coplanar` / per-triangle treatment, in the caller's
    original order (see the comment on `used`)."""
    # Read the module constants here, not as keyword defaults: a default is
    # captured at import time, so a test that sets `freeform.MIN_REGION_TRIS`
    # would have no effect.
    if max_normal_dev_deg is None:
        max_normal_dev_deg = MAX_NORMAL_DEV_DEG
    if max_tris is None:
        max_tris = MAX_REGION_TRIS
    if min_tris is None:
        min_tris = MIN_REGION_TRIS
    if shape_tol == -1.0:      # sentinel: "not specified by the caller"
        shape_tol = SHAPE_INDEX_TOL
    if segmentation is None:
        segmentation = SEGMENTATION
    if vsa_target is None:
        vsa_target = VSA_TARGET_TRIS
    if shape_index is None and shape_tol is not None:
        shape_index = _curv.face_shape_index(verts, faces)

    def _split(face_idx, parts):
        """Cut a triangle set into `parts` pieces with whichever
        segmentation is configured."""
        if segmentation == "vsa":
            return _vsa.partition(verts, faces, face_idx, parts)
        budget = max(min_tris, len(face_idx) // max(parts, 2))
        return grow_regions(verts, faces, face_idx, max_normal_dev_deg,
                            max_tris=budget, shape_index=shape_index,
                            shape_tol=shape_tol)

    if segmentation == "vsa":
        first = _vsa.partition(verts, faces, leftover,
                               max(2, len(leftover) // max(vsa_target, 1)))
    else:
        first = grow_regions(verts, faces, leftover, max_normal_dev_deg,
                             max_tris=max_tris, shape_index=shape_index,
                             shape_tol=shape_tol)
    work = deque(first)
    accepted = []
    # Triangles this stage consumed. `remaining` is rebuilt at the end by
    # filtering the caller's own list, not by appending rejected regions, so
    # the stage preserves order: `merge_coplanar` seeds its greedy groups in
    # the order it receives triangles and is sensitive to it, and a stage that
    # fits nothing must hand it exactly what it received.
    used: set = set()
    reasons: dict[str, int] = {}
    reason_tris: dict[str, int] = {}
    done_tris = 0
    total_tris = max(len(leftover), 1)
    misses: list[float] = []
    while work:
        region = work.popleft()
        fitted, info, best_err = fit_region(verts, faces, region, tolerance)
        if best_err != float("inf") and fitted is None:
            misses.append(best_err)
        if fitted is not None:
            accepted.append((fitted, info, region))
            used.update(region)
            done_tris += len(region)
        elif info in SPLITTABLE and len(region) >= 2 * min_tris:
            subs = _split(region, 2)
            if len(subs) > 1:
                work.extend(subs)
                continue
            reasons[info] = reasons.get(info, 0) + 1   # wouldn't split
            reason_tris[info] = reason_tris.get(info, 0) + len(region)
            done_tris += len(region)
        else:
            reasons[info] = reasons.get(info, 0) + 1
            reason_tris[info] = reason_tris.get(info, 0) + len(region)
            done_tris += len(region)
        if progress is not None:
            progress(min(done_tris / total_tris, 1.0))
    if stats is not None:
        stats["reasons"] = reasons
        stats["reason_tris"] = reason_tris
        # The tolerance this mesh could have supported: only meaningful when
        # fits were attempted and missed; the median, so one unusually good
        # region does not suggest a tolerance most of the part cannot meet.
        stats["achievable_mm"] = (float(np.median(misses)) if misses else None)
    remaining = [f for f in leftover if f not in used]
    return accepted, remaining


# ---------------------------------------------------------------------------
# STEP output helpers
# ---------------------------------------------------------------------------


def compact_knots(knots: np.ndarray, tol: float = 1e-12):
    """Turn a full (expanded) knot vector into the distinct values +
    multiplicities pair STEP's `B_SPLINE_SURFACE_WITH_KNOTS` actually
    stores. Inverse of the reader's `geometry._expand_knots`."""
    vals: list[float] = []
    mults: list[int] = []
    for k in knots:
        k = float(k)
        if vals and abs(k - vals[-1]) <= tol:
            mults[-1] += 1
        else:
            vals.append(k)
            mults.append(1)
    return vals, mults


def eval_surface(fit: BSplineFit, u: float, v: float) -> np.ndarray:
    """Evaluate a fitted patch. Only used by tests/verification -- the
    exporter itself never needs to evaluate the surface it writes."""
    ncu, ncv = fit.ctrl.shape[0], fit.ctrl.shape[1]
    su = _find_span(ncu - 1, fit.deg_u, u, fit.knots_u)
    sv = _find_span(ncv - 1, fit.deg_v, v, fit.knots_v)
    Nu = _basis_funs(su, u, fit.deg_u, fit.knots_u)
    Nv = _basis_funs(sv, v, fit.deg_v, fit.knots_v)
    acc = np.zeros(3)
    for a in range(fit.deg_u + 1):
        for b in range(fit.deg_v + 1):
            acc += Nu[a] * Nv[b] * fit.ctrl[su - fit.deg_u + a, sv - fit.deg_v + b]
    return acc
