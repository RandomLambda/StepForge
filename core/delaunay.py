"""Incremental constrained Delaunay triangulation on an explicit
triangle-adjacency structure. Pure Python, no NumPy in the inner loops.

* Point location by walking: from any triangle, step across whichever edge
  separates you from the target. With points inserted in Hilbert order
  (`hilbert_order`) each walk takes a couple of steps; no spatial index.
* Every insertion restores the Delaunay property locally (Lawson flips), so
  triangles stay compact and the result does not depend on insertion order.
* Orientation and in-circle tests are floating-point filtered with an exact
  rational fallback, so near-degenerate input cannot corrupt the structure.

Data layout: triangles live in two flat lists, three entries per triangle
(faster in CPython than tuples or objects):

    tv[3t+0], tv[3t+1], tv[3t+2]    vertex indices of triangle t
    nb[3t+k]                        the triangle across the edge OPPOSITE
                                    vertex tv[3t+k], or -1 if none

"Opposite vertex k" means the edge (tv[3t+(k+1)%3], tv[3t+(k+2)%3]). Every
triangle is counter-clockwise; a deleted triangle has tv[3t+0] == -1.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

# Relative error bounds of the floating-point filters below (conservative, for
# IEEE doubles): a result larger than bound * (sum of the magnitudes involved)
# cannot have had its sign changed by rounding.
_ORIENT_BOUND = 3.4e-16
_INCIRCLE_BOUND = 1.2e-15

EXACT_CALLS = [0, 0]   # [orient2d fallbacks, in-circle fallbacks], for tests


def _fr(x):
    from fractions import Fraction
    return Fraction(*float(x).as_integer_ratio())


def _orient2d_exact(ax, ay, bx, by, cx, cy):
    """Exact sign of the orientation determinant, expanded in the original
    coordinates (not the rounded differences); floats are dyadic rationals."""
    EXACT_CALLS[0] += 1
    ax, ay, bx, by, cx, cy = (_fr(ax), _fr(ay), _fr(bx),
                              _fr(by), _fr(cx), _fr(cy))
    d = (bx * cy - bx * ay - ax * cy - by * cx + by * ax + ay * cx)
    return 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)


def orient2d(ax, ay, bx, by, cx, cy):
    """> 0 if a->b->c turns counter-clockwise. The sign is exact: evaluated in
    floating point, with an exact rational fallback when the result is too
    close to zero for rounding error to be ruled out. A wrong sign on a
    near-degenerate configuration corrupts the mesh (a missing hull
    triangle, i.e. a hole)."""
    detleft = (bx - ax) * (cy - ay)
    detright = (by - ay) * (cx - ax)
    det = detleft - detright
    if (detleft > 0.0) != (detright > 0.0) or detleft == 0.0 or detright == 0.0:
        # No cancellation possible, so the sign is already certain.
        return det
    err = _ORIENT_BOUND * (abs(detleft) + abs(detright))
    if det > err or -det > err:
        return det
    return _orient2d_exact(ax, ay, bx, by, cx, cy)


def _in_circle_exact(ax, ay, bx, by, cx, cy, dx, dy):
    """Exact sign of the in-circle determinant (lifted 4x4 form, expanded by
    minors on the last column, in the original coordinates)."""
    EXACT_CALLS[1] += 1
    ax, ay, bx, by, cx, cy, dx, dy = (_fr(ax), _fr(ay), _fr(bx), _fr(by),
                                      _fr(cx), _fr(cy), _fr(dx), _fr(dy))
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    d2 = dx * dx + dy * dy

    def det3(p, q, r, s, t, u, v, w, x):
        return p * (t * x - u * w) - q * (s * x - u * v) + r * (s * w - t * v)

    d = (det3(bx, by, b2, cx, cy, c2, dx, dy, d2)
         - det3(ax, ay, a2, cx, cy, c2, dx, dy, d2)
         + det3(ax, ay, a2, bx, by, b2, dx, dy, d2)
         - det3(ax, ay, a2, bx, by, b2, cx, cy, c2))
    return 1 if d > 0 else (-1 if d < 0 else 0)


def in_circle(ax, ay, bx, by, cx, cy, dx, dy):
    """True if d is strictly inside the circumcircle of CCW (a, b, c).
    Filtered like `orient2d`, with the same exact fallback."""
    adx = ax - dx
    ady = ay - dy
    bdx = bx - dx
    bdy = by - dy
    cdx = cx - dx
    cdy = cy - dy
    alift = adx * adx + ady * ady
    blift = bdx * bdx + bdy * bdy
    clift = cdx * cdx + cdy * cdy
    bc = bdx * cdy - cdx * bdy
    ca = cdx * ady - adx * cdy
    ab = adx * bdy - bdx * ady
    det = alift * bc + blift * ca + clift * ab
    permanent = (alift * (abs(bdx * cdy) + abs(cdx * bdy))
                 + blift * (abs(cdx * ady) + abs(adx * cdy))
                 + clift * (abs(adx * bdy) + abs(bdx * ady)))
    err = _INCIRCLE_BOUND * permanent
    if det > err:
        return True
    if -det > err:
        return False
    return _in_circle_exact(ax, ay, bx, by, cx, cy, dx, dy) > 0


def hilbert_order(pts: Sequence[tuple[float, float]], order: int = 16) -> list[int]:
    """Indices of `pts` sorted along a Hilbert curve, so consecutive inserted
    points are close together and `locate`'s walk stays short."""
    n = len(pts)
    if n < 2:
        return list(range(n))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    dx = (maxx - minx) or 1.0
    dy = (maxy - miny) or 1.0
    side = (1 << order) - 1
    keyed = []
    for i in range(n):
        x = int((xs[i] - minx) / dx * side)
        y = int((ys[i] - miny) / dy * side)
        # Standard xy -> Hilbert distance (Wikipedia's iterative form).
        rx = ry = 0
        d = 0
        s = 1 << (order - 1)
        while s > 0:
            rx = 1 if (x & s) > 0 else 0
            ry = 1 if (y & s) > 0 else 0
            d += s * s * ((3 * rx) ^ ry)
            # rotate
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            s >>= 1
        keyed.append((d, i))
    keyed.sort()
    return [i for _, i in keyed]


class Triangulation:
    """A 2D triangulation with incremental (constrained) Delaunay insertion.

    Build it with `delaunay(points)` (super-triangle and spatial sort), or
    wrap an existing triangle list with `from_triangles`, which is how
    `tessellate.py` inserts interior points into a cropped boundary
    triangulation.
    """

    __slots__ = ("_last", "constrained", "nb", "px", "py", "tv")

    def __init__(self, px: list[float], py: list[float]):
        self.px = px
        self.py = py
        self.tv: list[int] = []
        self.nb: list[int] = []
        # Undirected vertex pairs (lo, hi) that must never be flipped away;
        # empty for an unconstrained triangulation.
        self.constrained: set = set()
        self._last = 0

    # -- construction ----------------------------------------------------
    def _new_tri(self, a: int, b: int, c: int) -> int:
        """Append a triangle, forcing counter-clockwise winding."""
        if orient2d(self.px[a], self.py[a], self.px[b], self.py[b],
                    self.px[c], self.py[c]) < 0.0:
            b, c = c, b
        t = len(self.tv) // 3
        self.tv.append(a)
        self.tv.append(b)
        self.tv.append(c)
        self.nb.append(-1)
        self.nb.append(-1)
        self.nb.append(-1)
        return t

    def _slot_of(self, t: int, other: int) -> int:
        """Which of t's three neighbour slots points at `other`."""
        b = 3 * t
        nb = self.nb
        if nb[b] == other:
            return 0
        if nb[b + 1] == other:
            return 1
        if nb[b + 2] == other:
            return 2
        return -1

    def _link(self, t: int, k: int, s: int, j: int) -> None:
        """Make t's slot k and s's slot j point at each other."""
        if t >= 0:
            self.nb[3 * t + k] = s
        if s >= 0:
            self.nb[3 * s + j] = t

    def _build_adjacency(self) -> None:
        """Derive `nb` from `tv` by hashing every directed edge (used when
        triangles arrive from outside)."""
        tv = self.tv
        nb = self.nb
        for i in range(len(nb)):
            nb[i] = -1
        # Directed edges of a CCW triangle are (v1,v2), (v2,v0), (v0,v1) for
        # slots 0, 1, 2; the neighbour carries the same edge reversed.
        seen = {}
        nt = len(tv) // 3
        for t in range(nt):
            b = 3 * t
            if tv[b] < 0:
                continue
            v0, v1, v2 = tv[b], tv[b + 1], tv[b + 2]
            for k, (p, q) in enumerate(((v1, v2), (v2, v0), (v0, v1))):
                mate = seen.pop((q, p), None)
                if mate is None:
                    seen[(p, q)] = (t, k)
                else:
                    mt, mk = mate
                    nb[3 * t + k] = mt
                    nb[3 * mt + mk] = t

    @classmethod
    def from_triangles(cls, points: Sequence[tuple[float, float]],
                       tris: Sequence[Sequence[int]],
                       constrained: Iterable | None = None) -> Triangulation:
        tri = cls([float(p[0]) for p in points], [float(p[1]) for p in points])
        for t in tris:
            tri._new_tri(int(t[0]), int(t[1]), int(t[2]))
        tri._build_adjacency()
        if constrained:
            tri.set_constrained(constrained)
        return tri

    def set_constrained(self, edges: Iterable) -> None:
        s = self.constrained
        s.clear()
        for e in edges:
            i, j = tuple(e)[:2]
            s.add((i, j) if i < j else (j, i))

    # -- predicates ------------------------------------------------------
    def _in_circle(self, a: int, b: int, c: int, d: int) -> bool:
        """True if d lies strictly inside the circumcircle of the
        counter-clockwise triangle (a, b, c).

        The super-triangle's corners are treated as ordinary far-away points:
        symbolic points at infinity are ambiguous when a triangle has an
        infinite vertex on both sides of the tested edge, and `in_circle`'s
        exact fallback covers the conditioning cost of real coordinates.
        """
        px, py = self.px, self.py
        return in_circle(px[a], py[a], px[b], py[b],
                         px[c], py[c], px[d], py[d])

    # -- location --------------------------------------------------------
    def locate(self, x: float, y: float, start: int = -1) -> tuple[int, int]:
        """Walk to the triangle containing (x, y).

        Returns `(t, k)`: `t` is the containing triangle, `k` the slot of the
        edge the point lies ON (exact zero orientation), else -1. Returns
        `(-1, -1)` if the walk leaves the triangulation.

        Straight walk: if the point is to the right of one of a triangle's
        counter-clockwise edges (negative orient2d), step across that edge.
        """
        px, py, tv, nb = self.px, self.py, self.tv, self.nb
        t = start if start >= 0 else self._last
        nt = len(tv) // 3
        if t < 0 or t >= nt or tv[3 * t] < 0:
            t = 0
            while t < nt and tv[3 * t] < 0:
                t += 1
            if t >= nt:
                return -1, -1
        # Bound only, the walk is monotone; guards against a cycle caused by
        # inconsistent orientation.
        for _ in range(3 * nt + 16):
            b = 3 * t
            v0, v1, v2 = tv[b], tv[b + 1], tv[b + 2]
            d0 = orient2d(px[v1], py[v1], px[v2], py[v2], x, y)
            if d0 < 0.0:
                nxt = nb[b]
                if nxt < 0:
                    return -1, -1
                t = nxt
                continue
            d1 = orient2d(px[v2], py[v2], px[v0], py[v0], x, y)
            if d1 < 0.0:
                nxt = nb[b + 1]
                if nxt < 0:
                    return -1, -1
                t = nxt
                continue
            d2 = orient2d(px[v0], py[v0], px[v1], py[v1], x, y)
            if d2 < 0.0:
                nxt = nb[b + 2]
                if nxt < 0:
                    return -1, -1
                t = nxt
                continue
            self._last = t
            if d0 == 0.0:
                return t, 0
            if d1 == 0.0:
                return t, 1
            if d2 == 0.0:
                return t, 2
            return t, -1
        return -1, -1

    def _locate_scan(self, x: float, y: float) -> tuple[int, int]:
        """Exhaustive point location. Fallback for `locate`; see `insert`."""
        px, py, tv = self.px, self.py, self.tv
        for t in range(len(tv) // 3):
            b = 3 * t
            v0 = tv[b]
            if v0 < 0:
                continue
            v1, v2 = tv[b + 1], tv[b + 2]
            d0 = orient2d(px[v1], py[v1], px[v2], py[v2], x, y)
            if d0 < 0.0:
                continue
            d1 = orient2d(px[v2], py[v2], px[v0], py[v0], x, y)
            if d1 < 0.0:
                continue
            d2 = orient2d(px[v0], py[v0], px[v1], py[v1], x, y)
            if d2 < 0.0:
                continue
            self._last = t
            if d0 == 0.0:
                return t, 0
            if d1 == 0.0:
                return t, 1
            if d2 == 0.0:
                return t, 2
            return t, -1
        return -1, -1

    # -- insertion -------------------------------------------------------
    def insert(self, pi: int, start: int = -1) -> int:
        """Insert point index `pi`, restoring the Delaunay property.

        Returns the index of one triangle incident to the new point, or -1
        if the point could not be located (outside the triangulation) or is
        an exact duplicate of a corner of its containing triangle.
        """
        px, py, tv, nb = self.px, self.py, self.tv, self.nb
        x, y = px[pi], py[pi]
        t, on_edge = self.locate(x, y, start)
        if t < 0:
            # The walk left the triangulation: possible when this class wraps
            # a cropped, non-convex face domain (the straight line to an
            # interior point can pass outside it). Fall back to a scan.
            t, on_edge = self._locate_scan(x, y)
            if t < 0:
                return -1
        b = 3 * t
        # Already present (same index or same coordinates): inserting would
        # create zero-area triangles that no flip removes. Leave the mesh as is.
        for k in range(3):
            v = tv[b + k]
            if v == pi or (px[v] == x and py[v] == y):
                return t

        stack: list[tuple[int, int]] = []
        if on_edge >= 0 and nb[b + on_edge] >= 0:
            first = self._split_edge(t, on_edge, pi, stack)
        else:
            first = self._split_tri(t, pi, stack)
        self._legalize(pi, stack)
        self._last = first
        return first

    def _split_tri(self, t: int, pi: int, stack) -> int:
        """1 -> 3 split of triangle t at interior point pi."""
        tv, nb = self.tv, self.nb
        b = 3 * t
        v0, v1, v2 = tv[b], tv[b + 1], tv[b + 2]
        n0, n1, n2 = nb[b], nb[b + 1], nb[b + 2]

        # Rewrite t in place as (v0, v1, pi) and add the other two, so the
        # triangle list grows by exactly two per insertion.
        tv[b], tv[b + 1], tv[b + 2] = v0, v1, pi
        ta = t
        tb_ = self._new_tri_raw(v1, v2, pi)
        tc = self._new_tri_raw(v2, v0, pi)

        # Slot layout of (a, b, pi): slot 0 is opposite a -> edge (b, pi),
        # slot 1 is opposite b -> edge (pi, a), slot 2 is opposite pi ->
        # edge (a, b), which is the original outer edge.
        self._link(ta, 2, n2, self._slot_of(n2, t) if n2 >= 0 else -1)
        self._link(tb_, 2, n0, self._slot_of(n0, t) if n0 >= 0 else -1)
        self._link(tc, 2, n1, self._slot_of(n1, t) if n1 >= 0 else -1)
        self._link(ta, 0, tb_, 1)
        self._link(tb_, 0, tc, 1)
        self._link(tc, 0, ta, 1)

        stack.append((ta, 2))
        stack.append((tb_, 2))
        stack.append((tc, 2))
        return ta

    def _split_edge(self, t: int, k: int, pi: int, stack) -> int:
        """2 -> 4 split across the edge in slot k of triangle t. A point on an
        existing edge must split both neighbours, or a zero-area sliver
        remains that no flip can remove."""
        tv, nb = self.tv, self.nb
        s = nb[3 * t + k]
        j = self._slot_of(s, t)
        if j < 0:
            return self._split_tri(t, pi, stack)

        b = 3 * t
        # t's vertices relabelled so that (a, b) is the split edge and c is
        # opposite it; likewise d opposite on the neighbour s.
        a = tv[b + (k + 1) % 3]
        bb = tv[b + (k + 2) % 3]
        c = tv[b + k]
        d = tv[3 * s + j]
        # Outer neighbours, named after the edge they sit across.
        n_ca = nb[b + (k + 2) % 3]      # across (c, a)
        n_bc = nb[b + (k + 1) % 3]      # across (b, c)
        n_bd = nb[3 * s + (j + 2) % 3]  # across (b, d)
        n_da = nb[3 * s + (j + 1) % 3]  # across (d, a)

        t0 = self._rewrite(t, c, a, pi)
        t1 = self._rewrite(s, bb, c, pi)
        t2 = self._new_tri_raw(d, bb, pi)
        t3 = self._new_tri_raw(a, d, pi)

        self._link(t0, 2, n_ca, self._slot_of(n_ca, t) if n_ca >= 0 else -1)
        self._link(t1, 2, n_bc, self._slot_of(n_bc, t) if n_bc >= 0 else -1)
        self._link(t2, 2, n_bd, self._slot_of(n_bd, s) if n_bd >= 0 else -1)
        self._link(t3, 2, n_da, self._slot_of(n_da, s) if n_da >= 0 else -1)
        # Internal spokes. For a triangle written (X, Y, pi): slot 0 is the
        # edge (Y, pi), slot 1 is (pi, X) and slot 2 is (X, Y). Pairing the
        # four triangles up by the spoke they actually share:
        #   (a, pi)  -> t0 slot 0 and t3 slot 1
        #   (c, pi)  -> t0 slot 1 and t1 slot 0
        #   (b, pi)  -> t1 slot 1 and t2 slot 0
        #   (d, pi)  -> t2 slot 1 and t3 slot 0
        self._link(t0, 0, t3, 1)
        self._link(t0, 1, t1, 0)
        self._link(t1, 1, t2, 0)
        self._link(t2, 1, t3, 0)

        # The split edge (a, b) is gone, replaced by (a, pi) and (pi, b).
        # If it was constrained, both halves inherit that.
        key = (a, bb) if a < bb else (bb, a)
        if key in self.constrained:
            self.constrained.discard(key)
            self.constrained.add((a, pi) if a < pi else (pi, a))
            self.constrained.add((bb, pi) if bb < pi else (pi, bb))

        stack.append((t0, 2))
        stack.append((t1, 2))
        stack.append((t2, 2))
        stack.append((t3, 2))
        return t0

    def _new_tri_raw(self, a: int, b: int, c: int) -> int:
        """Append (a, b, c) without re-checking winding; callers derive the
        triple from a counter-clockwise triangle."""
        t = len(self.tv) // 3
        self.tv.append(a)
        self.tv.append(b)
        self.tv.append(c)
        self.nb.append(-1)
        self.nb.append(-1)
        self.nb.append(-1)
        return t

    def _rewrite(self, t: int, a: int, b: int, c: int) -> int:
        base = 3 * t
        self.tv[base] = a
        self.tv[base + 1] = b
        self.tv[base + 2] = c
        self.nb[base] = -1
        self.nb[base + 1] = -1
        self.nb[base + 2] = -1
        return t

    def _legalize(self, pi: int, stack) -> None:
        """Lawson flips until every edge opposite `pi` is Delaunay.

        `stack` holds (triangle, slot) pairs naming an edge that has just
        become suspect. The invariant throughout is that the slot's triangle
        has `pi` as its third vertex, so a flip always replaces the suspect
        edge with one through `pi` and pushes the two new outer edges.
        """
        tv, nb = self.tv, self.nb
        constrained = self.constrained
        while stack:
            t, k = stack.pop()
            base = 3 * t
            if tv[base] < 0:
                continue
            s = nb[base + k]
            if s < 0:
                continue
            a = tv[base + (k + 1) % 3]
            b = tv[base + (k + 2) % 3]
            if constrained:
                key = (a, b) if a < b else (b, a)
                if key in constrained:
                    continue
            j = self._slot_of(s, t)
            if j < 0:
                continue
            d = tv[3 * s + j]
            c = tv[base + k]          # == pi for every edge we push
            if not self._in_circle(a, b, c, d):
                continue
            # Flip (a, b) to (c, d). The two triangles become (c, a, d) and
            # (c, d, b), which stay counter-clockwise given (a, b, c) was.
            n_bd = nb[3 * s + (j + 2) % 3]   # neighbour across (b, d)
            n_da = nb[3 * s + (j + 1) % 3]   # neighbour across (d, a)
            n_ca = nb[base + (k + 2) % 3]    # neighbour across (c, a)
            n_bc = nb[base + (k + 1) % 3]    # neighbour across (b, c)

            self._rewrite(t, c, a, d)
            self._rewrite(s, c, d, b)
            # t becomes (c, a, d): slot 0 is the edge (a, d), slot 1 is
            # (d, c) -- the new diagonal -- and slot 2 is (c, a).
            # s becomes (c, d, b): slot 0 is (d, b), slot 1 is (b, c) and
            # slot 2 is (c, d), the other side of the new diagonal.
            self._link(t, 2, n_ca, self._slot_of(n_ca, t) if n_ca >= 0 else -1)
            self._link(t, 0, n_da, self._slot_of(n_da, s) if n_da >= 0 else -1)
            self._link(s, 0, n_bd, self._slot_of(n_bd, s) if n_bd >= 0 else -1)
            self._link(s, 1, n_bc, self._slot_of(n_bc, t) if n_bc >= 0 else -1)
            self._link(t, 1, s, 2)

            # (c, a, d): slot 0 is the edge (a, d) -- newly exposed.
            # (c, d, b): slot 0 is the edge (d, b) -- newly exposed.
            stack.append((t, 0))
            stack.append((s, 0))

    # -- output ----------------------------------------------------------
    def triangles(self, exclude_above: int = -1) -> list[list[int]]:
        """Live triangles as mutable [i, j, k] lists.

        `exclude_above`, when >= 0, drops every triangle touching a vertex
        index at or above it -- how the super-triangle's three artificial
        corners are removed.
        """
        out = []
        tv = self.tv
        for t in range(len(tv) // 3):
            b = 3 * t
            v0 = tv[b]
            if v0 < 0:
                continue
            v1, v2 = tv[b + 1], tv[b + 2]
            if exclude_above >= 0 and (v0 >= exclude_above or v1 >= exclude_above
                                       or v2 >= exclude_above):
                continue
            out.append([v0, v1, v2])
        return out

    # -- self-check ------------------------------------------------------
    def check(self) -> list[str]:
        """Structural audit, for tests. Returns a list of problems found."""
        problems = []
        tv, nb = self.tv, self.nb
        nt = len(tv) // 3
        for t in range(nt):
            b = 3 * t
            if tv[b] < 0:
                continue
            v = (tv[b], tv[b + 1], tv[b + 2])
            if len(set(v)) != 3:
                problems.append(f"tri {t} has a repeated vertex {v}")
                continue
            if orient2d(self.px[v[0]], self.py[v[0]], self.px[v[1]],
                        self.py[v[1]], self.px[v[2]], self.py[v[2]]) < 0:
                problems.append(f"tri {t} is clockwise")
            for k in range(3):
                s = nb[b + k]
                if s < 0:
                    continue
                j = self._slot_of(s, t)
                if j < 0:
                    problems.append(f"tri {t} slot {k} -> {s}, not mutual")
                    continue
                e0 = {v[(k + 1) % 3], v[(k + 2) % 3]}
                sv = (tv[3 * s], tv[3 * s + 1], tv[3 * s + 2])
                e1 = {sv[(j + 1) % 3], sv[(j + 2) % 3]}
                if e0 != e1:
                    problems.append(f"tris {t}/{s} share slots but not an edge")
        return problems

    def delaunay_violations(self, exclude_above: int = -1) -> int:
        """Count edges that are not locally Delaunay (constrained edges and
        edges touching an excluded vertex are not counted)."""
        bad = 0
        tv, nb = self.tv, self.nb
        for t in range(len(tv) // 3):
            b = 3 * t
            if tv[b] < 0:
                continue
            for k in range(3):
                s = nb[b + k]
                if s < 0 or s < t:
                    continue
                a = tv[b + (k + 1) % 3]
                bb = tv[b + (k + 2) % 3]
                c = tv[b + k]
                j = self._slot_of(s, t)
                if j < 0:
                    continue
                d = tv[3 * s + j]
                if exclude_above >= 0 and max(a, bb, c, d) >= exclude_above:
                    continue
                key = (a, bb) if a < bb else (bb, a)
                if key in self.constrained:
                    continue
                if self._in_circle(a, bb, c, d):
                    bad += 1
        return bad


def delaunay(points: Sequence[tuple[float, float]]) -> list[list[int]]:
    """Delaunay-triangulate `points`, returning [i, j, k] triangles: points are
    inserted in Hilbert order into a super-triangle, each legalized as it
    goes, and the super-triangle's corners are removed at the end."""
    n = len(points)
    if n < 3:
        return []
    px = [float(p[0]) for p in points]
    py = [float(p[1]) for p in points]
    minx, maxx = min(px), max(px)
    miny, maxy = min(py), max(py)
    midx = (minx + maxx) * 0.5
    midy = (miny + maxy) * 0.5
    r = 0.5 * math.hypot(maxx - minx, maxy - miny)
    if r <= 0.0:
        r = 1.0
    # Far enough out that no corner falls inside a real triangle's
    # circumcircle, so deleting every triangle touching a corner leaves
    # exactly the convex hull.
    R = 1000.0 * r
    for ang in (math.pi * 0.5, math.pi * 7.0 / 6.0, math.pi * 11.0 / 6.0):
        px.append(midx + R * math.cos(ang))
        py.append(midy + R * math.sin(ang))

    tri = Triangulation(px, py)
    tri._new_tri(n, n + 1, n + 2)
    last = 0
    for i in hilbert_order(points):
        r = tri.insert(i, last)
        if r >= 0:
            last = r
    return tri.triangles(exclude_above=n)
