"""Fit analytic surfaces (plane / cylinder / cone / sphere / torus) to a
triangle soup, so a re-export can carry real curved STEP faces instead of
facets.

1. `segment` groups triangles into smooth patches: BFS across shared edges
   whose face normals differ by less than `smooth_angle_deg`, the rule the
   importer uses for smooth vs. flat shading, so sharp edges stay patch
   boundaries.
2. `fit_patch` tries each primitive and keeps the first whose max
   point-to-surface distance is within `tolerance`; a patch that fits nothing
   returns None and the caller falls back to planar faces.
3. `boundary_loops` walks each accepted patch's outer edge and hole loops, and
   `classify_edge` labels every boundary edge LINE, CIRCLE or, when neither
   applies, CHORD (a straight approximation whose sagitta error is reported
   so the caller can reject a patch that is too coarse).

Pure NumPy, no Blender dependency.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

# Cap on the combined boundary (outer ring and holes) of one face. Beyond it a
# reader's constrained-Delaunay triangulator fails to recover every
# constrained edge and falls back to plain earcut, whose multi-hole bridging
# can leave gaps around holes.
MAX_BOUNDARY_PTS = 90


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def loop_winding_sign(loop: list[int], verts: np.ndarray, fit_obj) -> int:
    """+1 if `loop` winds counter-clockwise around `fit_obj`'s axis (in the
    frame `classify_edge`'s circle branch and the STEP writer's
    AXIS2_PLACEMENT_3D use), -1 if clockwise. For cylinder, cone and torus
    fits (a torus "ring" loop of constant v winds like a cylinder rim).

    The two rim loops of a fitted cylinder or cone wind in opposite directions
    around the axis, like a flat annulus. The writer needs the direction once
    per loop to set each EDGE_CURVE's `same_sense`, so the reader samples arcs
    as they were meshed. From the signed area (shoelace formula) of the loop
    projected into the plane perpendicular to the axis."""
    axis = fit_obj.axis_dir
    axis_pt = fit_obj.apex if fit_obj.kind == "cone" else fit_obj.axis_point
    e1, e2 = _orthonormal_basis(axis)
    pts = verts[loop] - axis_pt
    xs = pts @ e1
    ys = pts @ e2
    n = len(loop)
    area2 = 0.0
    for i in range(n):
        j = (i + 1) % n
        area2 += xs[i] * ys[j] - xs[j] * ys[i]
    return 1 if area2 >= 0 else -1


def _face_normal(v0, v1, v2):
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n)
    if norm < 1e-14:
        return None
    return n / norm


def flip_degenerate_triangles(verts: np.ndarray,
                              faces: Sequence[tuple[int, int, int]],
                              face_ids: Sequence[int] | None = None):
    """Remove zero-area triangles that have a neighbour across their long edge.

    Three collinear points make a triangle without a normal, which `segment`
    leaves out of every patch; the faces on either side then disagree about the
    middle point and the shell opens. The triangle `(p, m, q)` (m between p
    and q) and the triangle `(q, p, d)` across the long edge become `(q, m, d)`
    and `(m, p, d)`: same surface, no zero-area triangle, every edge still
    used by two triangles. Both take the neighbour's entry of `face_ids`. A
    row of degenerate triangles is taken from the regular triangle inwards;
    one on the mesh edge, or whose middle point is not strictly between the
    other two, stays.

    Returns `(faces, face_ids)`; the inputs are not changed and are returned
    as they are when there is nothing to flip."""
    bad = [i for i, (a, b, c) in enumerate(faces)
           if _face_normal(verts[a], verts[b], verts[c]) is None]
    if not bad:
        return faces, face_ids
    tris = [tuple(int(v) for v in f) for f in faces]
    ids = list(face_ids) if face_ids is not None else None
    on_edge: dict[tuple[int, int], list[int]] = {}

    def _register(t):
        a, b, c = tris[t]
        for u, v in ((a, b), (b, c), (c, a)):
            on_edge.setdefault((u, v) if u < v else (v, u), []).append(t)

    def _forget(t):
        a, b, c = tris[t]
        for u, v in ((a, b), (b, c), (c, a)):
            on_edge[(u, v) if u < v else (v, u)].remove(t)

    for t in range(len(tris)):
        _register(t)
    is_bad = set(bad)

    def _flip(t):
        a, b, c = tris[t]
        # long edge p-q, with the point m between them
        p, q, m = max(((a, b, c), (b, c, a), (c, a, b)),
                      key=lambda e: float(np.linalg.norm(verts[e[0]] - verts[e[1]])))
        pq = verts[q] - verts[p]
        length2 = float(pq @ pq)
        if length2 <= 0.0:
            return False
        along = float((verts[m] - verts[p]) @ pq) / length2
        if not 0.0 < along < 1.0:
            return False
        across = [n for n in on_edge[(p, q) if p < q else (q, p)] if n != t]
        if len(across) != 1 or across[0] in is_bad:
            return False
        n = across[0]
        for u, v, d in ((tris[n][0], tris[n][1], tris[n][2]),
                        (tris[n][1], tris[n][2], tris[n][0]),
                        (tris[n][2], tris[n][0], tris[n][1])):
            if {u, v} == {p, q}:
                break
        _forget(t)
        _forget(n)
        tris[n] = (u, m, d)
        tris[t] = (m, v, d)
        _register(t)
        _register(n)
        is_bad.discard(t)
        if ids is not None:
            ids[t] = ids[n]
        return True

    # A run of collinear points can leave several such triangles side by side;
    # the one next to a regular triangle goes first.
    todo = bad
    while todo:
        left = [t for t in todo if not _flip(t)]
        if len(left) == len(todo):
            break
        todo = left
    return tris, ids


def segment(verts: np.ndarray, faces: Sequence[tuple[int, int, int]],
            smooth_angle_deg: float = 32.0) -> list[list[int]]:
    """Group triangle indices into smooth patches (BFS across low-dihedral
    manifold edges). Returns a list of patches, each a list of face indices.
    """
    n = len(faces)
    normals = [None] * n
    for i, (a, b, c) in enumerate(faces):
        normals[i] = _face_normal(verts[a], verts[b], verts[c])

    edge_faces: dict[tuple[int, int], list[int]] = {}
    for i, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_faces.setdefault(key, []).append(i)

    cos_thresh = math.cos(math.radians(smooth_angle_deg))
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for key, fs in edge_faces.items():
        if len(fs) != 2:
            continue  # non-manifold / boundary edge: never merge across it
        i, j = fs
        ni, nj = normals[i], normals[j]
        if ni is None or nj is None:
            continue
        if float(np.dot(ni, nj)) >= cos_thresh:
            adjacency[i].append(j)
            adjacency[j].append(i)

    visited = [False] * n
    patches: list[list[int]] = []
    for start in range(n):
        if visited[start] or normals[start] is None:
            continue
        stack = [start]
        visited[start] = True
        patch = []
        while stack:
            f = stack.pop()
            patch.append(f)
            for nb in adjacency[f]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        patches.append(patch)
    return patches


class _GrowingPointSet:
    """Amortized-growth point buffer for `merge_coplanar`'s per-group plane
    refit: a contiguous NumPy buffer grown by doubling, so a candidate's trial
    point set is a slice instead of a Python set union rebuilt per candidate.

    Only gains points: a rejected candidate's speculative write into the tail
    is overwritten by the next one (`_n` advances only on `commit()`). The
    point set equals the union it replaces, in commit order; the plane fit
    that is written out is always recomputed from `sorted(vids)` in
    `brep_export._fit_one_group`."""

    def __init__(self, verts: np.ndarray):
        self._verts = verts
        self._buf = np.empty((16, 3))
        self._n = 0

    def _reserve(self, extra: int) -> None:
        need = self._n + extra
        if need > len(self._buf):
            new_cap = max(need, len(self._buf) * 2)
            new_buf = np.empty((new_cap, 3))
            new_buf[:self._n] = self._buf[:self._n]
            self._buf = new_buf

    def trial_points(self, new_ids: Sequence[int]) -> np.ndarray:
        """Every committed point plus `new_ids`' own points, as a view --
        NOT committed until `commit(new_ids)` is called with the same ids."""
        if not new_ids:
            return self._buf[:self._n]
        self._reserve(len(new_ids))
        self._buf[self._n:self._n + len(new_ids)] = self._verts[new_ids]
        return self._buf[:self._n + len(new_ids)]

    def commit(self, new_ids: Sequence[int]) -> None:
        if not new_ids:
            return
        self._reserve(len(new_ids))
        self._buf[self._n:self._n + len(new_ids)] = self._verts[new_ids]
        self._n += len(new_ids)


FOLD_TOLERANCE = 0.05   # a triangle this close to edge-on (sine of its tilt) faces neither way
PEEL_ROUNDS = 8     # a group gives up at most this many times, then falls apart


def _peel_crossings(verts: np.ndarray, faces: Sequence[tuple[int, int, int]],
                    group: list[int], rounds: int = PEEL_ROUNDS) -> list[list[int]]:
    """`group`, cleared of whatever makes its outline cross itself.

    Of the two outline edges of each crossing the triangle with the smaller
    area leaves the group and becomes a face of its own; the rest of the
    group is tested again and, where the removal cut it in two, split into
    its connected parts. A group that still crosses after `rounds` rounds
    becomes one face per triangle. A group with a crossing-free outline, or
    one `boundary_loops` cannot read, is returned as it is."""
    if len(group) < 2:
        return [group]
    peeled: list[list[int]] = []
    for _ in range(rounds):
        loops, ok = boundary_loops(faces, group)
        if not (ok and loops):
            return [group] + peeled
        pairs = loop_crossing_edges(loops, verts)
        if not pairs:
            break
        owner = {}
        for t in group:
            a, b, c = faces[t]
            for u, v in ((a, b), (b, c), (c, a)):
                owner[(u, v) if u < v else (v, u)] = t

        def area(t):
            a, b, c = faces[t]
            return float(np.linalg.norm(np.cross(verts[b] - verts[a], verts[c] - verts[a])))

        victims = set()
        for e1, e2 in pairs:
            owners = [owner[(u, v) if u < v else (v, u)] for u, v in (e1, e2)]
            victims.add(min(owners, key=area))
        peeled.extend([t] for t in sorted(victims))
        group = [t for t in group if t not in victims]
        if len(group) < 2:
            return [group] + peeled if group else peeled
    else:
        return [[t] for t in group] + peeled
    # the removal can leave several pieces that only touch at a corner
    parent = {t: t for t in group}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    first_tri: dict[tuple[int, int], int] = {}
    for t in group:
        a, b, c = faces[t]
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            other = first_tri.setdefault(key, t)
            if other != t:
                parent[find(t)] = find(other)
    pieces: dict[int, list[int]] = {}
    for t in group:
        pieces.setdefault(find(t), []).append(t)
    if len(pieces) == 1:
        return [group] + peeled
    out = list(peeled)
    for piece in pieces.values():
        out.extend(_peel_crossings(verts, faces, piece, rounds))
    return out


def merge_coplanar(verts: np.ndarray, faces: Sequence[tuple[int, int, int]],
                    face_idx: Sequence[int], tolerance: float,
                    angle_tol_deg: float = 3.0,
                    progress: Callable[[float], None] | None = None) -> list[list[int]]:
    """Greedily grow connected groups of `face_idx` triangles into large flat
    patches: a triangle joins a group while the group's best-fit plane stays
    within `tolerance`.

    For triangles `fit_patch` could not identify as one primitive (a slightly
    bowed panel, say); one face per leftover triangle would bloat the file and
    make a re-import tessellate every tiny face on its own.

    Each candidate must also keep the group's boundary a clean simple
    silhouette (`boundary_loops` validity: no pinch where two regions touch at
    one vertex) and, as written (after simplification), within
    `MAX_BOUNDARY_PTS` so the reader's constrained-Delaunay triangulator can
    handle it. Past the cap the group stops growing and the excluded
    neighbours start their own, so no geometry is lost."""
    face_idx_set = set(face_idx)
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for f in face_idx:
        a, b, c = faces[f]
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_faces.setdefault(key, []).append(f)

    adjacency: dict[int, list[int]] = {f: [] for f in face_idx}
    for key, fs in edge_faces.items():
        if len(fs) == 2:
            i, j = fs
            adjacency[i].append(j)
            adjacency[j].append(i)

    # `claimed` grows by one triangle per accepted addition, so its size tracks
    # progress even while a single big group absorbs thousands of triangles.
    # Throttled to about 200 progress callbacks.
    total = max(len(face_idx), 1)
    report_stride = max(total // 200, 1)
    last_reported = 0

    def _maybe_report():
        nonlocal last_reported
        if progress is None:
            return
        if len(claimed) - last_reported >= report_stride:
            last_reported = len(claimed)
            progress(min(len(claimed) / total, 1.0))

    # --- incremental boundary tracking -----------------------------------
    # The directed-edge bookkeeping `boundary_loops` computes, updated per
    # added triangle in O(1) instead of rescanning the whole group for every
    # candidate (which made growth quadratic). `_try_add_face` mutates
    # `edge_count` / `boundary_set` and returns an undo log, so a rejected
    # candidate rolls back. Both track only the group being grown and are reset
    # for every new group.
    edge_count: dict[tuple[int, int], int] = {}
    boundary_set = set()

    def _try_add_face(fidx):
        a, b, c = faces[fidx]
        changes = []
        for u, v in ((a, b), (b, c), (c, a)):
            rev = (v, u)
            if rev in boundary_set:
                boundary_set.discard(rev)
                changes.append(("bset_add", rev))
            old = edge_count.get((u, v), 0)
            edge_count[(u, v)] = old + 1
            changes.append(("count", (u, v), old))
            if old == 0 and edge_count.get((v, u), 0) == 0:
                boundary_set.add((u, v))
                changes.append(("bset_remove", (u, v)))
        return changes

    def _undo(changes):
        for op in reversed(changes):
            if op[0] == "count":
                _, key, old = op
                if old == 0:
                    del edge_count[key]
                else:
                    edge_count[key] = old
            elif op[0] == "bset_add":
                boundary_set.add(op[1])
            else:  # "bset_remove"
                boundary_set.discard(op[1])

    def _loops_from_boundary():
        # Same walk `boundary_loops` does, just fed from the incrementally
        # maintained `boundary_set` instead of a freshly rescanned dict.
        src_count: dict[int, int] = {}
        dst_count: dict[int, int] = {}
        nxt: dict[int, int] = {}
        for u, v in boundary_set:
            src_count[u] = src_count.get(u, 0) + 1
            dst_count[v] = dst_count.get(v, 0) + 1
            nxt[u] = v
        if any(c > 1 for c in src_count.values()) or any(c > 1 for c in dst_count.values()):
            return [], False
        loops = []
        used = set()
        for start_v in list(nxt.keys()):
            if start_v in used:
                continue
            loop = []
            cur = start_v
            guard = 0
            while cur not in used and guard < len(nxt) + 2:
                used.add(cur)
                loop.append(cur)
                cur = nxt.get(cur)
                guard += 1
                if cur is None:
                    return [], False
            if loop and cur == start_v:
                loops.append(loop)
            else:
                return [], False
        return loops, True

    # Every triangle must face the same side of the group's plane: two thin
    # slivers pass any plane fit however they are twisted, and projected onto
    # the plane one lies folded over the other. `area_vec` holds each
    # triangle's area vector (normal times area).
    tri = np.asarray([faces[f] for f in face_idx], dtype=np.int64).reshape(-1, 3)
    area_vec = np.zeros((len(faces), 3))
    area_vec[list(face_idx)] = np.cross(verts[tri[:, 1]] - verts[tri[:, 0]],
                                        verts[tri[:, 2]] - verts[tri[:, 0]])

    claimed = set()
    groups: list[list[int]] = []
    for start in face_idx:
        if start in claimed:
            continue
        edge_count.clear()
        boundary_set.clear()
        group = [start]
        tri_area = _GrowingPointSet(area_vec)
        tri_area.commit([start])
        claimed.add(start)
        _try_add_face(start)  # committed immediately, never rolled back
        _maybe_report()
        group_vids = set(faces[start])
        points = _GrowingPointSet(verts)
        points.commit(list(group_vids))
        pending = list(dict.fromkeys(adjacency[start]))
        while pending:
            cand = pending.pop()
            if cand in claimed or cand not in face_idx_set:
                continue
            new_vids = [v for v in faces[cand] if v not in group_vids]
            pf = _fit_plane(points.trial_points(new_vids))
            if pf.max_err <= tolerance:
                area_rows = tri_area.trial_points([cand])
                side = (area_rows @ pf.normal) / np.maximum(
                    np.linalg.norm(area_rows, axis=1), 1e-300)
                if side.min() < -FOLD_TOLERANCE and side.max() > FOLD_TOLERANCE:
                    continue  # folded back over the group -- starts its own group
                changes = _try_add_face(cand)
                trial_loops, ok = _loops_from_boundary()
                if not ok:
                    _undo(changes)
                    continue  # would pinch the boundary -- reject, try the
                              # next pending candidate instead
                # Count the boundary as it will be written (after collinear and
                # full-circle simplification), not the raw per-vertex count, or
                # a large flat plate with a few bores stops growing because
                # its raw boundary is dense. `simplify_loop` only removes
                # points, so a raw count within the cap is already safe and
                # skips the O(boundary) simplification.
                raw_total = sum(len(l) for l in trial_loops)
                if raw_total <= MAX_BOUNDARY_PTS:
                    simplified_total = raw_total
                else:
                    simplified_total = sum(
                        len(simplify_loop(l, verts, pf, tolerance, angle_tol_deg)[0])
                        for l in trial_loops)
                if simplified_total > MAX_BOUNDARY_PTS:
                    _undo(changes)
                    continue  # would overwhelm the reader's triangulator --
                              # start a fresh group for this triangle instead
                group = group + [cand]
                tri_area.commit([cand])
                claimed.add(cand)
                _maybe_report()
                points.commit(new_vids)
                group_vids.update(new_vids)
                for nb in adjacency[cand]:
                    if nb not in claimed and nb in face_idx_set:
                        pending.append(nb)
        groups.append(group)
    # Even with every triangle facing one side, a group can lie over itself
    # once projected (a strip winding more than a turn, an edge-on sliver); its
    # outline then crosses itself. The triangles at the crossing leave.
    checked: list[list[int]] = []
    for group in groups:
        checked.extend(_peel_crossings(verts, faces, group))
    if progress is not None:
        progress(1.0)
    return checked


# ---------------------------------------------------------------------------
# Primitive fits
# ---------------------------------------------------------------------------


@dataclass
class PlaneFit:
    kind: str = "plane"
    origin: np.ndarray = None
    normal: np.ndarray = None
    max_err: float = 0.0


@dataclass
class SphereFit:
    kind: str = "sphere"
    center: np.ndarray = None
    radius: float = 0.0
    max_err: float = 0.0


@dataclass
class CylinderFit:
    kind: str = "cylinder"
    axis_point: np.ndarray = None
    axis_dir: np.ndarray = None
    radius: float = 0.0
    max_err: float = 0.0


@dataclass
class ConeFit:
    kind: str = "cone"
    apex: np.ndarray = None
    axis_dir: np.ndarray = None
    semi_angle: float = 0.0
    max_err: float = 0.0


@dataclass
class TorusFit:
    kind: str = "torus"
    axis_point: np.ndarray = None   # on the main axis, in the torus's centre plane
    axis_dir: np.ndarray = None
    major_radius: float = 0.0       # R: main-axis to tube-centre distance
    minor_radius: float = 0.0       # r: tube radius
    max_err: float = 0.0


def _orthonormal_basis(axis_dir: np.ndarray):
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis_dir, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis_dir, e1)
    return e1, e2


def plane_fit(pts: np.ndarray) -> PlaneFit:
    """Plane fit over a point set, for callers that already know it is flat."""
    return _fit_plane(pts)


def _fit_plane(pts: np.ndarray) -> PlaneFit:
    centroid = pts.mean(axis=0)
    d = pts - centroid
    # smallest-eigenvalue eigenvector of the covariance = the plane normal
    cov = d.T @ d
    _, v = np.linalg.eigh(cov)
    normal = v[:, 0]
    err = np.abs(d @ normal)
    return PlaneFit(origin=centroid, normal=normal, max_err=float(err.max()))


def _fit_sphere(pts: np.ndarray) -> SphereFit:
    # algebraic fit: x^2+y^2+z^2 = A x + B y + C z + D  (linear in A..D)
    n = len(pts)
    A = np.hstack([pts, np.ones((n, 1))])
    b = (pts ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, cz, D = sol
    center = np.array([cx, cy, cz]) / 2.0
    r2 = D + float(center @ center)
    if r2 <= 0:
        return SphereFit(center=center, radius=0.0, max_err=float("inf"))
    radius = math.sqrt(r2)
    err = np.abs(np.linalg.norm(pts - center, axis=1) - radius)
    return SphereFit(center=center, radius=radius, max_err=float(err.max()))


def _axis_candidates(normals: np.ndarray) -> list[np.ndarray]:
    cov = normals.T @ normals
    _, v = np.linalg.eigh(cov)
    # Every eigenvector is a candidate axis (for a cylinder the smallest
    # eigenvalue's, for a shallow cone possibly the largest); the caller
    # scores all three.
    return [v[:, 0], v[:, 1], v[:, 2]]


def _fit_cylinder_given_axis(pts: np.ndarray, axis_dir: np.ndarray) -> CylinderFit:
    p0 = pts.mean(axis=0)
    e1, e2 = _orthonormal_basis(axis_dir)
    rel = pts - p0
    u = rel @ e1
    v = rel @ e2
    # circle fit: u^2+v^2 = A u + B v + C
    n = len(pts)
    M = np.stack([u, v, np.ones(n)], axis=1)
    rhs = u ** 2 + v ** 2
    sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    A, B, C = sol
    a, b = A / 2.0, B / 2.0
    r2 = C + a * a + b * b
    if r2 <= 0:
        return CylinderFit(axis_point=p0, axis_dir=axis_dir, radius=0.0, max_err=float("inf"))
    radius = math.sqrt(r2)
    axis_point = p0 + a * e1 + b * e2
    d = pts - axis_point
    ax_comp = np.outer(d @ axis_dir, axis_dir)
    perp = d - ax_comp
    dist = np.linalg.norm(perp, axis=1)
    err = np.abs(dist - radius)
    return CylinderFit(axis_point=axis_point, axis_dir=axis_dir, radius=radius,
                        max_err=float(err.max()))


def _fit_cone_given_axis(pts: np.ndarray, axis_dir: np.ndarray) -> ConeFit:
    p0 = pts.mean(axis=0)
    e1, e2 = _orthonormal_basis(axis_dir)
    rel = pts - p0
    u = rel @ e1
    v = rel @ e2
    w = rel @ axis_dir
    # u^2+v^2 = A u + B v + C + D w + E w^2   (algebraic cone fit)
    n = len(pts)
    M = np.stack([u, v, np.ones(n), w, w ** 2], axis=1)
    rhs = u ** 2 + v ** 2
    sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    A, B, _, D, E = sol
    if E <= 1e-12:
        return ConeFit(apex=p0, axis_dir=axis_dir, semi_angle=0.0, max_err=float("inf"))
    k = math.sqrt(E)  # k = tan(semi_angle), magnitude only
    a, b = A / 2.0, B / 2.0
    r0 = D / (2.0 * k) if k > 1e-12 else 0.0
    # apex is where radius -> 0 along the axis: r0 + k*(w_apex) = 0
    w_apex = -r0 / k if k > 1e-12 else 0.0
    apex = p0 + a * e1 + b * e2 + w_apex * axis_dir
    axis_point = p0 + a * e1 + b * e2
    d = pts - axis_point
    ax_comp_scalar = d @ axis_dir
    perp = d - np.outer(ax_comp_scalar, axis_dir)
    dist = np.linalg.norm(perp, axis=1)
    expected_r = np.abs(r0 + k * ax_comp_scalar)
    err = np.abs(dist - expected_r)
    semi_angle = math.atan(k)
    return ConeFit(apex=apex, axis_dir=axis_dir, semi_angle=semi_angle,
                    max_err=float(err.max()))


def _fit_torus_given_axis(pts: np.ndarray, axis_dir: np.ndarray) -> TorusFit:
    """Torus fit for a given axis. The torus equation is quartic, so there is
    no single linear least-squares fit; instead every point is reduced to its
    meridian coordinates (rho, w) = (distance from the axis, position along
    it). With rho = R + r*cos(v) and w = r*sin(v) these lie on a circle of
    radius r centred at (R, 0) for every u, so a plain 2D circle fit gives the
    major radius (centre rho), the offset of the axis point from the centre
    plane (centre w) and the minor radius. The maximum error is exact: the
    nearest surface point of a surface of revolution lies in the point's own
    meridian half-plane, so the 3D distance equals the 2D distance to the
    meridian circle."""
    p0 = pts.mean(axis=0)
    e1, e2 = _orthonormal_basis(axis_dir)
    rel = pts - p0
    w = rel @ axis_dir
    x = rel @ e1
    y = rel @ e2
    rho = np.hypot(x, y)
    fitres = _fit_circle_2d(rho, w)
    if fitres is None:
        return TorusFit(axis_point=p0, axis_dir=axis_dir, max_err=float("inf"))
    major_r, w_center, minor_r, _ = fitres
    if major_r <= 1e-9 or minor_r <= 1e-9:
        return TorusFit(axis_point=p0, axis_dir=axis_dir, max_err=float("inf"))
    axis_point = p0 + w_center * axis_dir
    d = np.sqrt((rho - major_r) ** 2 + (w - w_center) ** 2)
    err = np.abs(d - minor_r)
    return TorusFit(axis_point=axis_point, axis_dir=axis_dir,
                    major_radius=float(major_r), minor_radius=float(minor_r),
                    max_err=float(err.max()))


def _fit_torus_refined(pts: np.ndarray, axis_dir: np.ndarray,
                        iters: int = 12) -> TorusFit:
    """`_fit_torus_given_axis` refined over the axis. A torus's normals are not
    all perpendicular to its axis (a fillet over a wide v range sweeps normals
    from radial to axial), so the normal-covariance axis guess is imprecise.
    Direct-search pattern refinement: move the axis by a small angle along two
    tangent directions, keep the move that lowers `max_err`, halve the step
    when none does, until the step is negligible or `iters` is spent."""
    best = _fit_torus_given_axis(pts, axis_dir)
    axis = axis_dir
    step = 0.1
    for _ in range(iters):
        if best.max_err == 0.0 or step < 1e-6:
            break
        e1, e2 = _orthonormal_basis(axis)
        improved = False
        for d in (e1, -e1, e2, -e2):
            cand_axis = axis + step * d
            cand_axis /= np.linalg.norm(cand_axis)
            cand = _fit_torus_given_axis(pts, cand_axis)
            if cand.max_err < best.max_err:
                best, axis = cand, cand_axis
                improved = True
        if not improved:
            step *= 0.5
    return best


def fit_patch(verts: np.ndarray, faces: Sequence[tuple[int, int, int]],
              face_idx: Sequence[int], tolerance: float,
              min_faces: int = 4):
    """Try Plane -> Sphere -> Cylinder -> Cone for one patch (a list of face
    indices). Returns the first fit whose max error is within `tolerance`,
    or None if nothing fits (caller should fall back to per-triangle planar
    faces for this patch)."""
    vert_ids = sorted({i for fidx in face_idx for i in faces[fidx]})
    pts = verts[vert_ids]
    normals_list = []
    for fidx in face_idx:
        a, b, c = faces[fidx]
        n = _face_normal(verts[a], verts[b], verts[c])
        if n is not None:
            normals_list.append(n)
    normals = np.array(normals_list)

    if len(face_idx) < min_faces or len(pts) < 3:
        return None

    plane = _fit_plane(pts)
    if plane.max_err <= tolerance:
        return plane

    if len(normals) >= 3:
        sphere = _fit_sphere(pts)
        if sphere.max_err <= tolerance and sphere.radius > 0:
            candidates = [_fit_cylinder_given_axis(pts, axis) for axis in _axis_candidates(normals)]
            best_cyl = min(candidates, key=lambda c: c.max_err)
            # prefer sphere unless a cylinder fits distinctly better
            if not (best_cyl.max_err < sphere.max_err * 0.5):
                return sphere

        cyl_candidates = [_fit_cylinder_given_axis(pts, axis) for axis in _axis_candidates(normals)]
        best_cyl = min(cyl_candidates, key=lambda c: c.max_err)
        if best_cyl.max_err <= tolerance and best_cyl.radius > 1e-9:
            return best_cyl

        cone_candidates = [_fit_cone_given_axis(pts, axis) for axis in _axis_candidates(normals)]
        best_cone = min(cone_candidates, key=lambda c: c.max_err)
        if best_cone.max_err <= tolerance:
            return best_cone

        sphere = _fit_sphere(pts)
        if sphere.max_err <= tolerance and sphere.radius > 0:
            return sphere

        # Last, after every simpler primitive failed: a fillet on a curved sweep
        # path (a rounded rim on a boss or bore) is only a torus.
        torus_candidates = [_fit_torus_refined(pts, axis) for axis in _axis_candidates(normals)]
        best_torus = min(torus_candidates, key=lambda t: t.max_err)
        if best_torus.max_err <= tolerance and best_torus.minor_radius > 1e-9:
            return best_torus

    return None


# ---------------------------------------------------------------------------
# Boundary loop extraction
# ---------------------------------------------------------------------------


def boundary_loops(faces: Sequence[tuple[int, int, int]],
                    face_idx: Sequence[int]):
    """Walk the outer edge (and any inner hole edges) of a patch in the
    winding direction the mesh already has. Returns `(loops, ok)`: `loops` is
    a list of closed loops, each a list of vertex indices (first == last
    implied); `ok` is False if the boundary is not a clean disjoint union of
    simple polygons, in which case callers must reject the patch or group.

    Edges shared inside the patch are skipped. A valid boundary needs every
    vertex to be the source of exactly one boundary edge and the destination
    of exactly one; otherwise the patch pinches (two loops, or a loop and a
    hole, touching at one vertex, as when `merge_coplanar` fuses triangles
    across the hub of a fan-triangulated n-gon)."""
    directed_count: dict[tuple[int, int], int] = {}
    for fidx in face_idx:
        a, b, c = faces[fidx]
        for u, v in ((a, b), (b, c), (c, a)):
            directed_count[(u, v)] = directed_count.get((u, v), 0) + 1

    boundary_edges = []
    for u, v in directed_count:
        if directed_count.get((v, u), 0) == 0:
            # no matching reverse edge inside the patch -> boundary edge
            boundary_edges.append((u, v))

    src_count: dict[int, int] = {}
    dst_count: dict[int, int] = {}
    nxt: dict[int, int] = {}
    for u, v in boundary_edges:
        src_count[u] = src_count.get(u, 0) + 1
        dst_count[v] = dst_count.get(v, 0) + 1
        nxt[u] = v
    if any(c > 1 for c in src_count.values()) or any(c > 1 for c in dst_count.values()):
        return [], False

    loops = []
    used = set()
    for start in list(nxt.keys()):
        if start in used:
            continue
        loop = []
        cur = start
        guard = 0
        while cur not in used and guard < len(nxt) + 2:
            used.add(cur)
            loop.append(cur)
            cur = nxt.get(cur)
            guard += 1
            if cur is None:
                return [], False
        if loop and cur == start:
            loops.append(loop)
        else:
            return [], False
    return loops, True


def loop_crossing_edges(loops: Sequence[Sequence[int]], verts: np.ndarray
                        ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """The pairs of edges of the closed `loops` (the outline and holes of one
    flat face, vertex indices into `verts`) that properly cross one another,
    seen along the normal of the plane through all their points. Edges that
    only touch (a shared vertex) do not count. A face whose outline crosses
    itself is read by Open CASCADE with the signed area of the loop and
    filled wrongly by the import. Each edge is a pair of vertex indices in
    the direction of its loop."""
    ids = [i for loop in loops for i in loop]
    if len(ids) < 4:
        return []
    verts = np.asarray(verts, dtype=float)
    centre = verts[ids].mean(axis=0)
    normal = np.linalg.svd(verts[ids] - centre, full_matrices=False)[2][2]
    e1, e2 = _orthonormal_basis(normal)
    starts, ends, edges = [], [], []
    for loop in loops:
        q = verts[list(loop)] - centre
        xy = np.stack([q @ e1, q @ e2], axis=1)
        starts.append(xy)
        ends.append(np.roll(xy, -1, axis=0))
        edges.extend((loop[k], loop[(k + 1) % len(loop)]) for k in range(len(loop)))
    a, b = np.vstack(starts), np.vstack(ends)
    d = b - a

    def side(p):        # [i, j]: which side of edge i the point p[j] lies on
        return d[:, 0, None] * (p[None, :, 1] - a[:, 1, None]) \
            - d[:, 1, None] * (p[None, :, 0] - a[:, 0, None])

    s1, s2 = side(a), side(b)
    crossing = (s1 * s2 < 0) & (s1.T * s2.T < 0)
    return [(edges[i], edges[j]) for i, j in np.argwhere(np.triu(crossing))]


def loop_crossings(loops: Sequence[Sequence[int]], verts: np.ndarray) -> int:
    """How many times the edges of the closed `loops` cross one another, see
    `loop_crossing_edges`."""
    return len(loop_crossing_edges(loops, verts))


# ---------------------------------------------------------------------------
# Edge classification against a fitted surface (for boundary curve output)
# ---------------------------------------------------------------------------


@dataclass
class EdgeSeg:
    kind: str  # "line" | "circle" | "chord"
    p0: np.ndarray = None
    p1: np.ndarray = None
    center: np.ndarray = None
    radius: float = 0.0
    axis: np.ndarray = None
    ref_dir: np.ndarray = None
    sagitta: float = 0.0


def _cyl_uv(fit_obj, p: np.ndarray):
    axis_pt = fit_obj.axis_point if fit_obj.kind == "cylinder" else fit_obj.apex
    axis_dir = fit_obj.axis_dir
    e1, e2 = _orthonormal_basis(axis_dir)
    d = p - axis_pt
    v = float(d @ axis_dir)
    x = float(d @ e1)
    y = float(d @ e2)
    u = math.atan2(y, x)
    return u, v, e1, e2


def _torus_uv(fit_obj, p: np.ndarray):
    """(u, v) for a point on a torus fit, in `geometry.Torus.eval`'s convention
    (u around the main axis, v around the tube); e1/e2 come from `axis_dir` by
    the same rule as in `_cyl_uv` and `_make_surface`, so angle zero agrees
    everywhere."""
    axis_pt = fit_obj.axis_point
    axis_dir = fit_obj.axis_dir
    e1, e2 = _orthonormal_basis(axis_dir)
    d = p - axis_pt
    w = float(d @ axis_dir)
    x = float(d @ e1)
    y = float(d @ e2)
    u = math.atan2(y, x)
    rho = math.hypot(x, y)
    v = math.atan2(w, rho - fit_obj.major_radius)
    return u, v, e1, e2, rho


def classify_edge(fit_obj, p0: np.ndarray, p1: np.ndarray,
                   angle_tol_deg: float = 2.0, axial_tol_ratio: float = 1e-3):
    """Classify one boundary edge (p0->p1) of a fitted patch as a straight
    LINE (constant angle, along the axis / in-plane), a CIRCLE arc (constant
    axial position, sweeping angle), or, failing both, a straight CHORD
    approximation whose sagitta error against the true surface is reported
    so the caller can reject the whole patch if it is too coarse."""
    if fit_obj.kind in ("plane", "bspline"):
        # A freeform (B-spline) patch's boundary is the original mesh polyline,
        # one LINE per mesh edge; there is no analytic curve to recover.
        # (`freeform.py` normally supplies these segments itself.)
        return EdgeSeg(kind="line", p0=p0, p1=p1, sagitta=0.0)

    if fit_obj.kind == "sphere":
        chord = float(np.linalg.norm(p1 - p0))
        r = fit_obj.radius
        sagitta = r - math.sqrt(max(r * r - (chord / 2.0) ** 2, 0.0)) if r > 0 else 0.0
        return EdgeSeg(kind="chord", p0=p0, p1=p1, sagitta=sagitta)

    if fit_obj.kind == "torus":
        u0, v0, e1, e2, _rho0 = _torus_uv(fit_obj, p0)
        u1, v1, _, _, _rho1 = _torus_uv(fit_obj, p1)
        axis_pt = fit_obj.axis_point
        axis_dir = fit_obj.axis_dir
        R = fit_obj.major_radius
        r = fit_obj.minor_radius
        dv = abs(math.atan2(math.sin(v1 - v0), math.cos(v1 - v0)))
        du = abs(math.atan2(math.sin(u1 - u0), math.cos(u1 - u0)))

        if math.degrees(dv) <= angle_tol_deg:
            # constant v: a "ring" arc, a circle of radius R + r*cos(v) around
            # the main axis, where a fillet meets the face it blends into
            ring_r = abs(R + r * math.cos(v0))
            center = axis_pt + (r * math.sin(v0)) * axis_dir
            return EdgeSeg(kind="circle", p0=p0, p1=p1, center=center,
                            radius=ring_r, axis=axis_dir, ref_dir=e1)

        if math.degrees(du) <= angle_tol_deg:
            # constant u: a meridian arc of radius r in the plane spanned by
            # axis_dir and this generatrix's radial direction
            u_dir = math.cos(u0) * e1 + math.sin(u0) * e2
            center = axis_pt + R * u_dir
            merid_axis = np.cross(u_dir, axis_dir)
            n = np.linalg.norm(merid_axis)
            if n > 1e-9:
                merid_axis = merid_axis / n
                return EdgeSeg(kind="circle", p0=p0, p1=p1, center=center,
                                radius=r, axis=merid_axis, ref_dir=axis_dir)

        # Neither (a diagonal cut across the fillet): a straight chord, with its
        # sagitta against the tighter principal radius.
        chord = float(np.linalg.norm(p1 - p0))
        local_r = min(r, abs(R + r * math.cos(v0)))
        sagitta = local_r - math.sqrt(max(local_r * local_r - (chord / 2.0) ** 2, 0.0)) \
            if local_r > 0 else 0.0
        return EdgeSeg(kind="chord", p0=p0, p1=p1, sagitta=sagitta)

    # cylinder / cone
    u0, v0, e1, e2 = _cyl_uv(fit_obj, p0)
    u1, v1, _, _ = _cyl_uv(fit_obj, p1)
    axis_pt = fit_obj.axis_point if fit_obj.kind == "cylinder" else fit_obj.apex
    axis_dir = fit_obj.axis_dir
    span = abs(v1 - v0)
    scale = max(abs(v0), abs(v1), 1.0)

    if span <= axial_tol_ratio * scale:
        # same axial position -> arc of a circle around the axis
        radius = fit_obj.radius if fit_obj.kind == "cylinder" else 0.0
        if fit_obj.kind == "cone":
            k = math.tan(fit_obj.semi_angle)
            radius = abs(k * v0)
        center = axis_pt + v0 * axis_dir
        return EdgeSeg(kind="circle", p0=p0, p1=p1, center=center, radius=radius,
                        axis=axis_dir, ref_dir=e1)

    du = abs(math.atan2(math.sin(u1 - u0), math.cos(u1 - u0)))
    if math.degrees(du) <= angle_tol_deg:
        # constant angle -> straight generatrix line (exact on a developable
        # surface: cylinders and cones are ruled along this direction)
        return EdgeSeg(kind="line", p0=p0, p1=p1, sagitta=0.0)

    # neither: approximate with a straight chord, report its true sagitta
    # error against the fitted surface so the caller can bail out.
    radius0 = fit_obj.radius if fit_obj.kind == "cylinder" else abs(
        math.tan(fit_obj.semi_angle) * v0)
    chord = float(np.linalg.norm(p1 - p0))
    sagitta = radius0 - math.sqrt(max(radius0 * radius0 - (chord / 2.0) ** 2, 0.0)) \
        if radius0 > 0 else 0.0
    return EdgeSeg(kind="chord", p0=p0, p1=p1, sagitta=sagitta)


def distance_to_surface(fit_obj, p: np.ndarray) -> float | None:
    """Exact unsigned distance from `p` to the analytic surface `fit_obj`, for
    checking that a curve fitted through a boundary chain hugs the surface it
    trims. None for a plane or freeform B-spline patch (a plane chain is exact
    as a LINE; a B-spline chain is checked against its raw points)."""
    kind = getattr(fit_obj, "kind", None)
    if kind == "cylinder":
        d = p - fit_obj.axis_point
        v = float(d @ fit_obj.axis_dir)
        radial = d - v * fit_obj.axis_dir
        return abs(float(np.linalg.norm(radial)) - fit_obj.radius)
    if kind == "cone":
        d = p - fit_obj.apex
        v = float(d @ fit_obj.axis_dir)
        radial = d - v * fit_obj.axis_dir
        r = float(np.linalg.norm(radial))
        alpha = fit_obj.semi_angle
        # Perpendicular distance from (v, r) to the meridian line r =
        # tan(alpha)*v; the radial error alone overstates it by ~1/cos(alpha).
        return abs(r * math.cos(alpha) - v * math.sin(alpha))
    if kind == "torus":
        d = p - fit_obj.axis_point
        w = float(d @ fit_obj.axis_dir)
        radial = d - w * fit_obj.axis_dir
        rho = float(np.linalg.norm(radial))
        return abs(math.hypot(rho - fit_obj.major_radius, w) - fit_obj.minor_radius)
    if kind == "sphere":
        return abs(float(np.linalg.norm(p - fit_obj.center)) - fit_obj.radius)
    return None


# ---------------------------------------------------------------------------
# Boundary-loop simplification: collapse dense tessellation into the handful
# of real lines and arcs, so ADVANCED_FACE boundaries (and their count against
# `MAX_BOUNDARY_PTS`) reflect the part's geometry rather than how finely it
# was tessellated.
# ---------------------------------------------------------------------------


def _vec_len(v: np.ndarray) -> float:
    """L2 norm of a 3-vector without `np.linalg.norm`'s per-call dispatch
    overhead (`merge_collinear` calls this hundreds of thousands of times)."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def merge_collinear(loop: list[int], verts: np.ndarray, tolerance: float,
                     angle_tol_deg: float = 3.0) -> list[int]:
    """Drop boundary vertices between two edges that are practically one
    straight edge: the turn angle at the vertex is below `angle_tol_deg`, or
    (for very short segments, where the angle is noisy) the point lies within
    `tolerance` of the line between its neighbours. Pure 3D check, so it
    serves flat faces and straight generatrices on a cylinder or cone.

    The angle test is primary and uses only the immediate neighbours: a fixed
    mm tolerance against an ever-longer merged chord stops merging early on a
    long, slightly noisy edge and leaves a comb of short segments."""
    n = len(loop)
    if n <= 3:
        return loop
    keep = [True] * n
    pts = verts[loop]
    cos_thresh = math.cos(math.radians(angle_tol_deg))
    # A pass removes at most every other surviving point of a straight run, so
    # collapsing n points takes ~log2(n) passes.
    max_passes = max(4, int(math.log2(max(n, 2))) + 3)
    for _ in range(max_passes):
        idxs = [i for i in range(n) if keep[i]]
        if len(idxs) <= 3:
            break
        m = len(idxs)
        removed_any = False
        for k in range(m):
            i = idxs[k]
            if not keep[i]:
                continue
            prev_i = idxs[k - 1]
            next_i = idxs[(k + 1) % m]
            if not keep[prev_i] or not keep[next_i]:
                continue  # neighbour removed this pass: re-check next pass
            a = pts[prev_i]
            b = pts[i]
            c = pts[next_i]
            d1 = b - a
            d2 = c - b
            l1 = _vec_len(d1)
            l2 = _vec_len(d2)
            if l1 < 1e-12 or l2 < 1e-12:
                continue
            cosang = float(np.dot(d1, d2) / (l1 * l2))
            cosang = max(-1.0, min(1.0, cosang))
            if cosang >= cos_thresh:
                keep[i] = False
                removed_any = True
                continue
            # fallback distance check, for short/noisy segments where the
            # angle test alone can be unreliable
            ac = c - a
            length = _vec_len(ac)
            if length < 1e-12:
                continue
            t = float(np.dot(b - a, ac) / (length * length))
            if t <= 0.0 or t >= 1.0:
                continue  # b isn't between a and c -- a real corner
            closest = a + t * ac
            if _vec_len(b - closest) <= tolerance:
                keep[i] = False
                removed_any = True
        if not removed_any:
            break
    new_loop = [loop[i] for i in range(n) if keep[i]]
    return new_loop if len(new_loop) >= 3 else loop


def _same_circle(seg1: EdgeSeg, seg2: EdgeSeg, tolerance: float) -> bool:
    """Whether two CIRCLE-classified segments lie on the same circle (centre
    and radius within `tolerance`). A torus boundary can be CIRCLE-classified
    as a "ring" (constant v) or a "meridian" (constant u) arc; these are
    different circles that meet at a corner, which merging must keep."""
    return (float(np.linalg.norm(seg1.center - seg2.center)) <= tolerance
            and abs(seg1.radius - seg2.radius) <= tolerance)


def merge_circular_arcs(loop: list[int], verts: np.ndarray, fit_obj,
                         tolerance: float) -> list[int]:
    """Drop boundary vertices between two edges that `classify_edge` calls
    CIRCLE on the same circle (`_same_circle`): both lie on one fitted circle,
    so the point between them only splits one longer arc."""
    n = len(loop)
    if n <= 3 or fit_obj is None or fit_obj.kind not in ("cylinder", "cone", "torus"):
        return loop
    keep = [True] * n
    kept_count = n
    # Pass count as in `merge_collinear`.
    max_passes = max(4, int(math.log2(max(n, 2))) + 3)
    for _ in range(max_passes):
        idxs = [i for i in range(n) if keep[i]]
        if len(idxs) <= 3:
            break
        m = len(idxs)
        removed_any = False
        for k in range(m):
            if kept_count <= 3:
                break  # never collapse below a valid polygon (a lone
                       # full circle would otherwise mark every point removable)
            i = idxs[k]
            if not keep[i]:
                continue
            prev_i = idxs[k - 1]
            next_i = idxs[(k + 1) % m]
            if not keep[prev_i] or not keep[next_i]:
                continue  # neighbour removed this pass: re-check next pass
            seg1 = classify_edge(fit_obj, verts[loop[prev_i]], verts[loop[i]])
            seg2 = classify_edge(fit_obj, verts[loop[i]], verts[loop[next_i]])
            if (seg1.kind == "circle" and seg2.kind == "circle"
                    and _same_circle(seg1, seg2, tolerance)):
                keep[i] = False
                kept_count -= 1
                removed_any = True
        if not removed_any:
            break
    new_loop = [loop[i] for i in range(n) if keep[i]]
    return new_loop if len(new_loop) >= 3 else loop


def _fit_circle_2d(x: np.ndarray, y: np.ndarray):
    n = len(x)
    M = np.stack([x, y, np.ones(n)], axis=1)
    rhs = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    A, B, C = sol
    cx, cy = A / 2.0, B / 2.0
    r2 = C + cx * cx + cy * cy
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    err = np.abs(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r)
    return cx, cy, r, float(err.max())


def try_collapse_circle_loop(loop: list[int], verts: np.ndarray, fit_obj,
                              tolerance: float):
    """If a closed boundary loop is, within `tolerance`, a full circle in a
    PLANE fit's own plane, collapse it to a 2-edge loop between two existing,
    roughly antipodal vertices: two CIRCLE edges instead of dozens of straight
    segments around a bore or boss (`classify_edge` already finds the rims of
    cylinders and cones, but not these). Returns `(new_loop, segs)` (segs
    aligned edge-by-edge with new_loop), or None if the loop is not a full
    circle."""
    if fit_obj is None or fit_obj.kind != "plane" or len(loop) < 6:
        return None
    normal = fit_obj.normal
    e1, e2 = _orthonormal_basis(normal)
    origin = fit_obj.origin
    pts = verts[loop]
    rel = pts - origin
    x = rel @ e1
    y = rel @ e2
    fitres = _fit_circle_2d(x, y)
    if fitres is None:
        return None
    cx, cy, r, max_err = fitres
    if max_err > tolerance:
        return None
    angles = np.arctan2(y - cy, x - cx)
    span = float(angles.max() - angles.min())
    if span < math.radians(180.0):
        return None  # not a real closed ring around the centre
    center = origin + cx * e1 + cy * e2
    i_a = 0
    target = angles[i_a] + math.pi
    diff = np.abs(np.arctan2(np.sin(angles - target), np.cos(angles - target)))
    i_b = int(np.argmin(diff))
    if i_b == i_a:
        return None

    def _seg(ia, ib):
        return EdgeSeg(kind="circle", p0=pts[ia], p1=pts[ib], center=center,
                        radius=r, axis=normal, ref_dir=e1)

    new_loop = [loop[i_a], loop[i_b]]
    segs = [_seg(i_a, i_b), _seg(i_b, i_a)]
    return new_loop, segs


def simplify_loop(loop: list[int], verts: np.ndarray, fit_obj, tolerance: float,
                   angle_tol_deg: float = 3.0):
    """Simplify one boundary loop for STEP emission. Returns `(new_loop,
    segs)`: `new_loop` is the reduced vertex-index list; `segs` is None (the
    caller classifies each edge) or a precomputed list of `EdgeSeg`s aligned
    with `new_loop` (the full-circle collapse, which `classify_edge` cannot
    derive for a plane fit).

    `angle_tol_deg` is how far two adjacent edges may differ in direction and
    still count as one straight edge (see `merge_collinear`)."""
    if fit_obj is not None and fit_obj.kind == "plane":
        collapsed = try_collapse_circle_loop(loop, verts, fit_obj, tolerance)
        if collapsed is not None:
            return collapsed
    new_loop = merge_collinear(loop, verts, tolerance, angle_tol_deg)
    if fit_obj is not None and fit_obj.kind in ("cylinder", "cone", "torus"):
        new_loop = merge_circular_arcs(new_loop, verts, fit_obj, tolerance)
    return new_loop, None
