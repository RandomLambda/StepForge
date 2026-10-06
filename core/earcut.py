"""Ear-clipping polygon triangulation with hole support.

A pure-Python simplification of mapbox/earcut (ISC licence, below): the
z-order hash is omitted because B-rep faces have only a few hundred boundary
points. The triangulation keeps the input boundary edges exactly, so faces
that share an edge get identical boundary vertices and the mesh welds
watertight.

`keep_collinear=True` keeps vertices that are collinear in 2D. In a curved
surface's (u, v) space, points on a straight iso-line (a cylinder's circular
edge, say) still carry curvature in 3D.

    earcut(points, hole_indices=None, keep_collinear=False)
        -> list of (i, j, k) index triples into `points`

Original work Copyright (c) 2016, Mapbox. ISC License:

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
"""
from __future__ import annotations


class _Node:
    __slots__ = ("i", "next", "prev", "steiner", "x", "y")

    def __init__(self, i, x, y):
        self.i = i
        self.x = x
        self.y = y
        self.prev = None
        self.next = None
        self.steiner = False


def earcut(points, hole_indices=None, keep_collinear=False):
    has_holes = bool(hole_indices)
    outer_len = hole_indices[0] if has_holes else len(points)

    outer = _linked_list(points, 0, outer_len, True)
    triangles: list[tuple[int, int, int]] = []
    if outer is None or outer.next is outer.prev:
        return triangles

    if has_holes:
        outer = _eliminate_holes(points, hole_indices, outer)

    if keep_collinear:
        _mark_steiner(outer)

    _earcut_linked(outer, triangles, 0)
    return triangles


def _mark_steiner(node):
    p = node
    while True:
        p.steiner = True
        p = p.next
        if p is node:
            break


def _signed_area(points, start, end):
    s = 0.0
    j = end - 1
    for i in range(start, end):
        s += (points[j][0] - points[i][0]) * (points[i][1] + points[j][1])
        j = i
    return s


def _linked_list(points, start, end, clockwise):
    last = None
    area = _signed_area(points, start, end)
    if clockwise == (area > 0):
        for i in range(start, end):
            last = _insert_node(i, points[i][0], points[i][1], last)
    else:
        for i in range(end - 1, start - 1, -1):
            last = _insert_node(i, points[i][0], points[i][1], last)
    if last is not None and _equals(last, last.next):
        _remove_node(last)
        last = last.next
    return last


def _insert_node(i, x, y, last):
    p = _Node(i, x, y)
    if last is None:
        p.prev = p
        p.next = p
    else:
        p.next = last.next
        p.prev = last
        last.next.prev = p
        last.next = p
    return p


def _remove_node(p):
    p.next.prev = p.prev
    p.prev.next = p.next


def _equals(a, b):
    return a.x == b.x and a.y == b.y


def _area(p, q, r):
    return (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)


def _point_in_triangle(ax, ay, bx, by, cx, cy, px, py):
    return ((cx - px) * (ay - py) - (ax - px) * (cy - py) >= 0 and
            (ax - px) * (by - py) - (bx - px) * (ay - py) >= 0 and
            (bx - px) * (cy - py) - (cx - px) * (by - py) >= 0)


def _is_ear(ear):
    a, b, c = ear.prev, ear, ear.next
    if _area(a, b, c) >= 0:
        return False  # reflex or collinear -> not a valid (CCW) ear
    p = c.next
    while p is not a:
        if _point_in_triangle(a.x, a.y, b.x, b.y, c.x, c.y, p.x, p.y) and \
                _area(p.prev, p, p.next) >= 0:
            return False
        p = p.next
    return True


def _earcut_linked(ear, triangles, pass_):
    if ear is None:
        return
    stop = ear
    guard = 0
    count = _count(ear)
    while ear.prev is not ear.next:
        guard += 1
        if guard > count * count + 16:
            break  # safety: malformed polygon
        prev = ear.prev
        nxt = ear.next
        if _is_ear(ear):
            triangles.append((prev.i, ear.i, nxt.i))
            _remove_node(ear)
            ear = nxt.next
            stop = nxt.next
            count -= 1
            guard = 0
            continue
        ear = nxt
        if ear is stop:
            # no ear found in a full pass -> try recovery
            if pass_ == 0:
                _earcut_linked(_filter_points(ear), triangles, 1)
            elif pass_ == 1:
                _split_earcut(ear, triangles)
            break


def _count(node):
    n = 0
    p = node
    while True:
        n += 1
        p = p.next
        if p is node:
            break
    return n


def _filter_points(start, end=None):
    if start is None:
        return start
    if end is None:
        end = start
    p = start
    while True:
        again = False
        if not p.steiner and (_equals(p, p.next) or _area(p.prev, p, p.next) == 0):
            _remove_node(p)
            p = end = p.prev
            if p is p.next:
                break
            again = True
        else:
            p = p.next
        if not again and p is end:
            break
    return end


def _split_earcut(start, triangles):
    """Find a valid diagonal that splits a hard polygon into two simpler ones."""
    a = start
    while True:
        b = a.next.next
        while b is not a.prev:
            if a.i != b.i and _is_valid_diagonal(a, b):
                c = _split_polygon(a, b)
                _earcut_linked(_filter_points(a), triangles, 0)
                _earcut_linked(_filter_points(c), triangles, 0)
                return
            b = b.next
        a = a.next
        if a is start:
            break


def _is_valid_diagonal(a, b):
    return (a.next.i != b.i and a.prev.i != b.i and
            not _intersects_polygon(a, b) and
            _locally_inside(a, b) and _locally_inside(b, a) and
            _middle_inside(a, b))


def _intersects(p1, q1, p2, q2):
    o1 = _sign(_area(p1, q1, p2))
    o2 = _sign(_area(p1, q1, q2))
    o3 = _sign(_area(p2, q2, p1))
    o4 = _sign(_area(p2, q2, q1))
    return o1 != o2 and o3 != o4


def _sign(n):
    return (1 if n > 0 else 0) - (1 if n < 0 else 0)


def _intersects_polygon(a, b):
    p = a
    while True:
        if (p.i != a.i and p.next.i != a.i and p.i != b.i and p.next.i != b.i and
                _intersects(p, p.next, a, b)):
            return True
        p = p.next
        if p is a:
            break
    return False


def _locally_inside(a, b):
    if _area(a.prev, a, a.next) < 0:
        return _area(a, b, a.next) >= 0 and _area(a, a.prev, b) >= 0
    return _area(a, b, a.prev) < 0 or _area(a, a.next, b) < 0


def _middle_inside(a, b):
    p = a
    inside = False
    px = (a.x + b.x) / 2.0
    py = (a.y + b.y) / 2.0
    while True:
        if ((p.y > py) != (p.next.y > py)) and p.next.y != p.y and \
                (px < (p.next.x - p.x) * (py - p.y) / (p.next.y - p.y) + p.x):
            inside = not inside
        p = p.next
        if p is a:
            break
    return inside


def _split_polygon(a, b):
    a2 = _Node(a.i, a.x, a.y)
    b2 = _Node(b.i, b.x, b.y)
    a2.steiner = a.steiner
    b2.steiner = b.steiner
    an = a.next
    bp = b.prev
    a.next = b
    b.prev = a
    a2.next = an
    an.prev = a2
    b2.next = a2
    a2.prev = b2
    bp.next = b2
    b2.prev = bp
    return b2


# --- hole elimination ------------------------------------------------------

def _eliminate_holes(points, hole_indices, outer):
    queue = []
    n = len(hole_indices)
    for i in range(n):
        start = hole_indices[i]
        end = hole_indices[i + 1] if i + 1 < n else len(points)
        lst = _linked_list(points, start, end, False)
        if lst is lst.next:
            lst.steiner = True
        queue.append(_get_leftmost(lst))
    queue.sort(key=lambda nd: nd.x)
    for h in queue:
        outer = _eliminate_hole(h, outer)
    return outer


def _eliminate_hole(hole, outer):
    bridge = _find_hole_bridge(hole, outer)
    if bridge is None:
        return outer
    _split_polygon(bridge, hole)
    return outer


def _point_in_triangle_except_first(ax, ay, bx, by, cx, cy, px, py):
    return not (ax == px and ay == py) and _point_in_triangle(ax, ay, bx, by, cx, cy, px, py)


def _sector_contains_sector(m, p):
    return _area(m.prev, m, p.prev) < 0 and _area(p.next, m, m.next) < 0


def _find_hole_bridge(hole, outer):
    """Pick a vertex on `outer` to bridge `hole` to (after mapbox/earcut's
    findHoleBridge): cast a ray from the hole's leftmost point in -x and take
    the nearest edge crossing, then look for a closer vertex inside the
    (hole, crossing, m) triangle that nothing blocks. The second stage keeps
    one distant bridge point from being chosen for every hole of a face."""
    p = outer
    hx, hy = hole.x, hole.y
    qx = -1e30
    m = None
    while True:
        if _equals(hole, p.next):
            return p.next
        if p.next.y != p.y and p.y >= hy >= p.next.y:
            x = p.x + (hy - p.y) * (p.next.x - p.x) / (p.next.y - p.y)
            if x <= hx and x > qx:
                qx = x
                if x == hx:
                    if hy == p.y:
                        return p
                    if hy == p.next.y:
                        return p.next
                m = p if p.x < p.next.x else p.next
        p = p.next
        if p is outer:
            break
    if m is None:
        return None
    if hx == qx:
        return m  # hole touches outer segment; pick leftmost endpoint

    # Stage 2: look for a closer, unobstructed bridge point near m.
    stop = m
    mx, my = m.x, m.y
    tan_min = float("inf")
    p = m
    while True:
        if (hx >= p.x >= mx and hx != p.x and
                _point_in_triangle_except_first(
                    hx if hy < my else qx, hy, mx, my,
                    qx if hy < my else hx, hy, p.x, p.y)):
            tan = abs(hy - p.y) / (hx - p.x)
            if (_locally_inside(p, hole) and
                    (tan < tan_min or
                     (tan == tan_min and (p.x > m.x or
                      (p.x == m.x and _sector_contains_sector(m, p)))))):
                m = p
                tan_min = tan
        p = p.next
        if p is stop:
            break
    return m


def _get_leftmost(start):
    p = start
    leftmost = start
    while True:
        if p.x < leftmost.x or (p.x == leftmost.x and p.y < leftmost.y):
            leftmost = p
        p = p.next
        if p is start:
            break
    return leftmost
