"""Mesh parameterization: flatten a patch of triangles into (u, v) so a
B-spline surface can be fitted to it.

Mean-value coordinates (Floater 2003, "Mean value coordinates", CAGD 20(1)):
the patch's boundary loop is mapped onto the border of the unit square and
every interior vertex is placed at the weighted average of its neighbours.
With positive weights and a convex boundary, Tutte's theorem gives a fold-free
map for any disk-shaped patch, however much it wraps in 3D (a projection onto
a mean plane stops working beyond about 90 degrees).

The system is assembled dense and solved with `np.linalg.solve` up to
MAX_DENSE interior vertices, and by plain Jacobi sweeps above that. The
mean-value system is non-symmetric and over-relaxation diverges; omega = 1
converges, slowly, which is why the direct solve is the primary path.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Interior-vertex count up to which the system is solved directly. A dense
# n x n solve at n = 1500 is about 25 MB and a fraction of a second; above
# that the fallback iteration takes over.
MAX_DENSE = 1500

# Fallback iteration: plain Jacobi (see the module docstring on why not
# over-relaxed) and the movement, in unit-square units, below which it is
# considered converged.
MAX_SWEEPS = 4000
CONVERGE = 1e-6


def _pick_corners(loop_pts: np.ndarray) -> list[int]:
    """Choose the four boundary vertices that become the square's corners.

    Equal perimeter quarters would put a corner in the middle of a long side
    of a band-shaped patch. Instead: the point furthest from the centroid, the
    point furthest from that (the band's two ends), and on each arc between
    them the point furthest from their joining line. Returns four indices into
    `loop_pts`, in loop order."""
    n = len(loop_pts)
    centre = loop_pts.mean(axis=0)
    c0 = int(np.argmax(np.linalg.norm(loop_pts - centre, axis=1)))
    c2 = int(np.argmax(np.linalg.norm(loop_pts - loop_pts[c0], axis=1)))
    if c2 == c0:
        c2 = (c0 + n // 2) % n
    lo, hi = (c0, c2) if c0 < c2 else (c2, c0)
    axis = loop_pts[hi] - loop_pts[lo]
    alen = float(np.linalg.norm(axis))
    if alen < 1e-12:
        return sorted({0, n // 4, n // 2, 3 * n // 4})
    axis = axis / alen

    def furthest(a, b):
        """Index (in loop order) of the point on the arc a..b furthest from
        the c0-c2 axis; `b` may wrap past the end of the loop. Returns None
        for an arc with no interior point."""
        idx = [i % n for i in range(a + 1, b)]
        if not idx:
            return None
        rel = loop_pts[idx] - loop_pts[lo]
        perp = rel - np.outer(rel @ axis, axis)
        return idx[int(np.argmax(np.linalg.norm(perp, axis=1)))]

    c1 = furthest(lo, hi)
    c3 = furthest(hi, n + lo)
    corners = [c for c in (lo, c1, hi, c3) if c is not None]
    if len(corners) < 4:      # degenerate arc -- fall back to quarters
        return sorted({0, n // 4, n // 2, 3 * n // 4})
    return sorted(set(corners)) if len(set(corners)) == 4 else \
        sorted({0, n // 4, n // 2, 3 * n // 4})


def square_boundary_uv(loop_pts: np.ndarray):
    """Map a closed boundary polyline onto the perimeter of the unit square,
    the (u, v) domain of the B-spline being fitted (convex, as Tutte's theorem
    needs).

    Returns `(uv, aspect)`; `aspect` is the patch's u:v extent ratio from the
    3D lengths of the four sides, used to shape the fitter's control net."""
    n = len(loop_pts)
    seg = np.linalg.norm(np.roll(loop_pts, -1, axis=0) - loop_pts, axis=1)
    if float(seg.sum()) <= 0:
        return np.zeros((n, 2)), 1.0
    corners = _pick_corners(loop_pts)
    uv = np.zeros((n, 2))
    side_ends = ((0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 1.0, 1.0),
                 (1.0, 1.0, 0.0, 1.0), (0.0, 1.0, 0.0, 0.0))
    side_len = []
    for k in range(4):
        a = corners[k]
        b = corners[(k + 1) % 4]
        idx = list(range(a, b)) if b > a else list(range(a, n)) + list(range(b))
        lens = np.array([seg[i] for i in idx])
        total = float(lens.sum())
        side_len.append(total)
        if total <= 0:
            frac = np.linspace(0.0, 1.0, len(idx), endpoint=False)
        else:
            frac = np.concatenate([[0.0], np.cumsum(lens)[:-1]]) / total
        x0, y0, x1, y1 = side_ends[k]
        uv[idx, 0] = x0 + (x1 - x0) * frac
        uv[idx, 1] = y0 + (y1 - y0) * frac
    du = 0.5 * (side_len[0] + side_len[2])
    dv = 0.5 * (side_len[1] + side_len[3])
    aspect = float(du / dv) if dv > 1e-12 else 1.0
    return uv, max(min(aspect, 20.0), 0.05)


def _mean_value_weights(verts, faces, region, pos):
    """Floater's mean-value weights as flat (row, col, value) arrays, built per
    triangle corner: tan(theta/2) / edge_length for each of its two edges.
    Summed over the two triangles sharing an edge this gives
    w_ij = (tan(alpha/2) + tan(beta/2)) / |p_j - p_i|."""
    f = np.array([faces[i] for i in region], dtype=np.int64)
    a, b, c = f[:, 0], f[:, 1], f[:, 2]
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for i, j, k in ((a, b, c), (b, c, a), (c, a, b)):
        e1 = verts[j] - verts[i]
        e2 = verts[k] - verts[i]
        l1 = np.linalg.norm(e1, axis=1)
        l2 = np.linalg.norm(e2, axis=1)
        l1s = np.maximum(l1, 1e-12)
        l2s = np.maximum(l2, 1e-12)
        cos = np.clip(np.einsum("ij,ij->i", e1, e2) / (l1s * l2s), -1.0, 1.0)
        theta = np.arccos(cos)
        # tan(theta/2) = (1 - cos) / sin, numerically safe near 0 and pi
        sin = np.maximum(np.sin(theta), 1e-9)
        half = (1.0 - cos) / sin
        ri = np.array([pos[v] for v in i])
        rj = np.array([pos[v] for v in j])
        rk = np.array([pos[v] for v in k])
        rows.append(ri)
        cols.append(rj)
        vals.append(half / l1s)
        rows.append(ri)
        cols.append(rk)
        vals.append(half / l2s)
    return (np.concatenate(rows), np.concatenate(cols),
            np.concatenate(vals))


def mean_value_uv(verts: np.ndarray,
                  faces: Sequence[tuple[int, int, int]],
                  region: Sequence[int],
                  vids: Sequence[int],
                  loop: Sequence[int],
                  init_uv: np.ndarray | None = None,
                  max_dense: int = MAX_DENSE,
                  max_sweeps: int = MAX_SWEEPS) -> np.ndarray | None:
    """Parameterize a disk-shaped patch into [0, 1]^2. Returns `(uv, aspect)`
    (a (len(vids), 2) array and the patch's u:v extent ratio), or None if the
    patch is unusable: boundary too small, iteration did not settle, or the
    result contains a folded triangle.

    `loop` is the patch's boundary loop (vertex indices, in order); `vids` is
    every vertex of the patch, and the returned rows line up with it.
    `init_uv` is an optional starting guess that shortens the fallback
    iteration."""
    n = len(vids)
    if n < 4 or len(loop) < 4:
        return None
    pos: dict[int, int] = {v: i for i, v in enumerate(vids)}
    if any(v not in pos for v in loop):
        return None

    uv = np.full((n, 2), 0.5) if init_uv is None else np.array(init_uv, float)
    bidx = np.array([pos[v] for v in loop])
    boundary_uv, aspect = square_boundary_uv(verts[list(loop)])
    uv[bidx] = boundary_uv
    interior = np.ones(n, bool)
    interior[bidx] = False
    if not interior.any():
        return uv, aspect   # every vertex is on the boundary: already placed

    rows, cols, vals = _mean_value_weights(verts, faces, region, pos)
    den = np.zeros(n)
    np.add.at(den, rows, vals)
    den[den <= 0] = 1.0

    int_ids = np.nonzero(interior)[0]
    n_int = len(int_ids)
    solved = False
    if n_int <= max_dense:
        # Direct solve: A[i,i] = sum_j w_ij, A[i,j] = -w_ij for interior j,
        # and every boundary neighbour's (already fixed) position moves to
        # the right-hand side.
        remap = np.full(n, -1, dtype=np.int64)
        remap[int_ids] = np.arange(n_int)
        A = np.zeros((n_int, n_int))
        rhs = np.zeros((n_int, 2))
        ri = remap[rows]
        keep = ri >= 0
        ri = ri[keep]
        cj = cols[keep]
        vj = vals[keep]
        rj = remap[cj]
        A[np.arange(n_int), np.arange(n_int)] = den[int_ids]
        inner = rj >= 0
        np.add.at(A, (ri[inner], rj[inner]), -vj[inner])
        outer = ~inner
        np.add.at(rhs, ri[outer], vj[outer, None] * uv[cj[outer]])
        try:
            uv[int_ids] = np.linalg.solve(A, rhs)
            solved = True
        except np.linalg.LinAlgError:
            solved = False

    if not solved:
        for _ in range(max_sweeps):
            num = np.zeros((n, 2))
            np.add.at(num, rows, vals[:, None] * uv[cols])
            delta = (num / den[:, None] - uv)[interior]
            uv[interior] += delta
            if float(np.abs(delta).max()) < CONVERGE:
                break
        else:
            return None      # never settled -- do not hand back garbage

    # The no-fold guarantee needs a clean disk, which region growing does not
    # ensure, so check.
    if not is_valid(faces, region, pos, uv):
        return None
    return uv, aspect


def is_valid(faces, region, pos, uv, min_area: float = 0.0) -> bool:
    """Every triangle keeps a positive area in (u, v) -- i.e. the
    parameterization is a genuine one-to-one flattening, not a folded one."""
    f = np.array([faces[i] for i in region], dtype=np.int64)
    ia = np.array([pos[v] for v in f[:, 0]])
    ib = np.array([pos[v] for v in f[:, 1]])
    ic = np.array([pos[v] for v in f[:, 2]])
    d1 = uv[ib] - uv[ia]
    d2 = uv[ic] - uv[ia]
    area2 = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    # Orientation follows the mesh winding; accept either sign as long as it
    # is CONSISTENT, since only folding matters here.
    pos_n = int((area2 > min_area).sum())
    neg_n = int((area2 < -min_area).sum())
    return pos_n == 0 or neg_n == 0
