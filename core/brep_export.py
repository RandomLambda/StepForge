"""Write triangle meshes to STEP as a real ADVANCED_BREP shape representation.

Analytic PLANE / CYLINDRICAL_SURFACE / CONICAL_SURFACE / SPHERICAL_SURFACE /
TOROIDAL_SURFACE faces are reconstructed wherever `fit.py` identifies them
within the chosen tolerance, freeform B_SPLINE_SURFACE_WITH_KNOTS faces
(`freeform.py`) wherever the shape is smooth but not a primitive, and
per-triangle planar ADVANCED_FACEs (always exact) everywhere else. The
counterpart to `exporter.py`'s tessellated writer: use that one for robustness
and speed, this one to get curved surfaces back as curves.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from . import boundary as _boundary
from . import fit as _fit
from . import freeform as _freeform
from . import workers as _workers
from .exporter import step_header, step_str
from .geometry import _basis_funs, _find_span


def _num(x: float) -> str:
    if x == 0:
        return "0."
    s = repr(float(x)).upper()
    mant, _, exp = s.partition("E")
    if "." not in mant:
        mant += "."          # Part 21 reals need the point, "5.E-06" too
    return mant + ("E" + exp if exp else "")


def _dir(v) -> str:
    return f"({_num(v[0])},{_num(v[1])},{_num(v[2])})"


class _Writer:
    def __init__(self):
        self.lines: list[str] = []
        self.id = 0

    def add(self, body: str) -> int:
        self.id += 1
        self.lines.append(f"#{self.id}={body};")
        return self.id


class _FaceBuilder:
    """Accumulates shared VERTEX_POINT / EDGE_CURVE entities for one solid so
    adjoining faces stitch into a watertight CLOSED_SHELL instead of each
    carrying its own disconnected copy of the boundary."""

    def __init__(self, w: _Writer, verts: np.ndarray, scale: float):
        self.w = w
        self.verts = verts
        self.scale = scale
        self._vertex_cache: dict[int, int] = {}
        self._edge_cache: dict[tuple[int, int], tuple[int, int, int]] = {}
        # How many boundary edges were written as a fitted curve instead of a
        # straight LINE (see `_try_curve_from_span`).
        self.bspline_curve_count = 0

    def _pos(self, vidx: int) -> np.ndarray:
        return self.verts[vidx]

    def vertex(self, vidx: int) -> int:
        cached = self._vertex_cache.get(vidx)
        if cached is not None:
            return cached
        p = self._pos(vidx) * self.scale
        cp = self.w.add(f"CARTESIAN_POINT('',{_dir(p)})")
        vp = self.w.add(f"VERTEX_POINT('',#{cp})")
        self._vertex_cache[vidx] = vp
        return vp

    def _emit_line(self, a: int, b: int) -> int:
        p0 = self._pos(a) * self.scale
        p1 = self._pos(b) * self.scale
        d = p1 - p0
        length = float(np.linalg.norm(d))
        if length < 1e-9:
            length = 1.0
            d = np.array([1.0, 0.0, 0.0])
        udir = d / length
        cp = self.w.add(f"CARTESIAN_POINT('',{_dir(p0)})")
        dirid = self.w.add(f"DIRECTION('',{_dir(udir)})")
        vec = self.w.add(f"VECTOR('',#{dirid},{_num(length)})")
        return self.w.add(f"LINE('',#{cp},#{vec})")

    def _emit_circle(self, seg: _fit.EdgeSeg) -> int:
        center = seg.center * self.scale
        cp = self.w.add(f"CARTESIAN_POINT('',{_dir(center)})")
        axisdir = self.w.add(f"DIRECTION('',{_dir(seg.axis)})")
        refdir = self.w.add(f"DIRECTION('',{_dir(seg.ref_dir)})")
        placement = self.w.add(
            f"AXIS2_PLACEMENT_3D('',#{cp},#{axisdir},#{refdir})")
        return self.w.add(f"CIRCLE('',#{placement},{_num(seg.radius * self.scale)})")

    def _emit_bspline_curve(self, degree: int, ctrl_scaled: np.ndarray,
                             knots: np.ndarray) -> int:
        ids = [self.w.add(f"CARTESIAN_POINT('',{_dir(p)})") for p in ctrl_scaled]
        pts_str = "(" + ",".join(f"#{i}" for i in ids) + ")"
        vals, mults = _freeform.compact_knots(knots)
        mult_str = ",".join(str(m) for m in mults)
        knot_str = ",".join(_num(k) for k in vals)
        return self.w.add(
            f"B_SPLINE_CURVE_WITH_KNOTS('',{degree},{pts_str},.UNSPECIFIED.,"
            f".F.,.F.,({mult_str}),({knot_str}),.UNSPECIFIED.)")

    def _try_curve_from_span(self, raw_span, fit_obj, tolerance: float | None) -> int | None:
        """Fit a `B_SPLINE_CURVE_WITH_KNOTS` through the raw mesh-vertex run
        `raw_span` (see `boundary._spans_for_chain`) and emit it, returning the
        curve entity id; `None` if there is nothing useful to fit (too few
        points, degenerate) or the curve would stray further than `tolerance`
        from the analytic surface it trims, in which case the caller writes the
        straight `LINE` it would have written anyway.

        For a boundary edge that `classify_edge` could not recognise as a
        generatrix or circle (a diagonal chord across a curved surface, or any
        edge of a freeform patch), a straight LINE between two distant
        simplified points ignores how far the surface curves away from the
        chord. A curve through the raw, pre-simplification points stays within
        the mesh's own deflection of the true surface."""
        if raw_span is None or len(raw_span) < 4:
            return None
        pts = np.array([self._pos(v) for v in raw_span], dtype=float)
        fitted = _fit_interpolating_bspline(pts)
        if fitted is None:
            return None
        degree, ctrl, knots = fitted
        surf_kind = getattr(fit_obj, "kind", None) if fit_obj is not None else None
        if tolerance is not None and surf_kind in ("cylinder", "cone", "torus", "sphere"):
            n_check = max(4 * (len(pts) - 1), 12)
            umin, umax = float(knots[degree]), float(knots[len(ctrl)])
            for t in np.linspace(umin, umax, n_check):
                p = _eval_bspline_point(degree, ctrl, knots, t)
                d = _fit.distance_to_surface(fit_obj, p)
                if d is not None and d > tolerance:
                    return None
        curve_id = self._emit_bspline_curve(degree, ctrl * self.scale, knots)
        self.bspline_curve_count += 1
        return curve_id

    def _circle_same_sense(self, seg: _fit.EdgeSeg, a: int, b: int,
                            winding_sign: int) -> bool:
        """Whether EDGE_CURVE.same_sense should be True (curve parameter
        increases from vertex a to vertex b) for this circular arc.

        Hardcoding True is only right for a loop that winds counter-clockwise
        in `seg`'s own frame. A cylinder or cone's second boundary loop (the
        FACE_BOUND "hole") winds the other way (see `fit.loop_winding_sign`),
        and a wrong flag makes the reader sample its arcs in the wrong
        direction, losing the side wall on reimport. `winding_sign` (+1 CCW,
        -1 CW) is computed once per loop from all its vertices, which is robust
        to any single segment's arc size."""
        return winding_sign >= 0

    def _circle_from_span(self, raw_span, tolerance) -> _fit.EdgeSeg | None:
        """Circle through a raw boundary run whose points leave the straight
        chord by more than `tolerance` but all lie within `tolerance` of
        the circle through its first, middle and last point; else None."""
        if not raw_span or len(raw_span) < 3 or not tolerance:
            return None
        pts = np.array([self._pos(v) for v in raw_span], dtype=float)
        a, m, b = pts[0], pts[len(pts) // 2], pts[-1]
        ab = b - a
        L2 = float(np.dot(ab, ab))
        if L2 <= 0.0:
            return None
        t = np.clip((pts - a) @ ab / L2, 0.0, 1.0)
        off_chord = np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)
        if float(off_chord.max()) <= tolerance:
            return None
        u, v = m - a, b - a
        n = np.cross(u, v)
        nn = float(np.dot(n, n))
        if nn <= 1e-24:
            return None
        # circumcentre of a, m, b (u = m - a, v = b - a, n = u x v)
        center = a + (float(np.dot(u, u)) * np.cross(v, n) + float(np.dot(v, v)) * np.cross(n, u)) / (2.0 * nn)
        axis = n / math.sqrt(nn)
        r = float(np.linalg.norm(a - center))
        d = pts - center
        h = d @ axis
        radial = np.linalg.norm(d - h[:, None] * axis[None, :], axis=1)
        if float(np.max(np.hypot(radial - r, h))) > tolerance:
            return None
        ref = (a - center) / r
        return _fit.EdgeSeg(kind="circle", p0=a, p1=b, center=center, radius=r,
                            axis=axis, ref_dir=ref)

    def _arc_sense_from_span(self, seg, raw_span) -> bool | None:
        """Direction of a circular arc from the mesh vertices it was simplified
        from: True if going a -> b counter-clockwise (in the circle's own
        frame) passes through the span's middle vertex; None without a span. A
        loop-wide winding sign cannot answer this for a partial face (a quarter
        cylinder's two rim arcs overlap once projected onto the axis plane),
        and a wrong answer makes the reader sweep the other 270 degrees. A span
        of a single mesh edge means the short arc."""
        if not raw_span or len(raw_span) < 2:
            return None
        e1 = seg.ref_dir
        e2 = np.cross(seg.axis, e1)
        c = seg.center

        def ang(p):
            d = p - c
            return math.atan2(float(np.dot(d, e2)), float(np.dot(d, e1)))
        ta, tb = ang(self._pos(raw_span[0])), ang(self._pos(raw_span[-1]))
        s_ab = (tb - ta) % (2.0 * math.pi)
        if len(raw_span) == 2:
            # one mesh edge: it can only stand for the short way round
            return s_ab < math.pi
        tm = ang(self._pos(raw_span[len(raw_span) // 2]))
        s_am = (tm - ta) % (2.0 * math.pi)
        return s_am < s_ab

    def edge(self, a: int, b: int, seg, winding_sign: int = 1,
             raw_span=None, fit_obj=None, tolerance: float | None = None) -> tuple[int, bool]:
        """Return (EDGE_CURVE id, sense) for the directed edge a->b, creating it
        on first use and reusing or flipping it on later references so shared
        boundaries stay watertight.

        `winding_sign` only matters for a fresh circular-arc edge (see
        `_circle_same_sense`); `raw_span`, `fit_obj` and `tolerance` for a fresh
        non-circular edge that classified as a diagonal chord or belongs to a
        freeform patch (see `_try_curve_from_span`)."""
        key = (a, b) if a < b else (b, a)
        cached = self._edge_cache.get(key)
        if cached is not None:
            master_a, _, edge_id = cached
            return edge_id, (a == master_a)
        va = self.vertex(a)
        vb = self.vertex(b)
        same_sense = True
        if seg is not None and seg.kind == "circle" and seg.radius > 1e-9:
            curve = self._emit_circle(seg)
            same_sense = self._arc_sense_from_span(seg, raw_span)
            if same_sense is None:
                same_sense = self._circle_same_sense(seg, a, b, winding_sign)
        else:
            curve = None
            circ = self._circle_from_span(raw_span, tolerance)
            if circ is not None:
                # The face writing this edge first may not know it is an arc
                # (a plane classifies every edge as a line), yet the mesh
                # between the two simplified ends is round, as the neighbouring
                # cylinder's shared chain shows. A chord would cut a notch of up
                # to its sagitta out of both faces.
                curve = self._emit_circle(circ)
                same_sense = self._arc_sense_from_span(circ, raw_span)
            else:
                wants_curve = (seg is not None and seg.kind == "chord") or (
                    fit_obj is not None and getattr(fit_obj, "kind", None) == "bspline")
                if wants_curve:
                    curve = self._try_curve_from_span(raw_span, fit_obj, tolerance)
            if curve is None:
                curve = self._emit_line(a, b)
        flag = ".T." if same_sense else ".F."
        edge_id = self.w.add(f"EDGE_CURVE('',#{va},#{vb},#{curve},{flag})")
        self._edge_cache[key] = (a, b, edge_id)
        return edge_id, True


def _fit_interpolating_bspline(pts: np.ndarray):
    """Fit a clamped, non-rational cubic B-spline curve that exactly
    interpolates every point in `pts`, in order: chord-length parameterisation
    plus knot averaging (Piegl & Tiller, "The NURBS Book", Algorithm A9.1).
    Returns `(degree, control_points, knot_vector)`, or `None` if `pts` is too
    short or degenerate to fit.

    Interpolating rather than approximating, the curve passes through every raw
    tessellation vertex (already within the mesh's deflection of the true
    surface), so a chain that simplification collapsed to one chord keeps its
    shape between the endpoints."""
    n_pts = len(pts)
    if n_pts < 4:
        return None
    seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg_lens.sum())
    if total < 1e-9:
        return None

    degree = 3
    n = n_pts - 1
    t = np.zeros(n_pts)
    np.cumsum(seg_lens, out=t[1:])
    t /= total
    t[-1] = 1.0

    p = degree
    U = np.zeros(n + p + 2)
    U[-(p + 1):] = 1.0
    for j in range(1, n - p + 1):
        U[j + p] = float(np.mean(t[j:j + p]))

    A = np.zeros((n_pts, n_pts))
    for k in range(n_pts):
        span = _find_span(n, p, float(t[k]), U)
        N = _basis_funs(span, float(t[k]), p, U)
        A[k, span - p:span + 1] = N
    try:
        ctrl = np.linalg.solve(A, pts)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(ctrl)):
        return None
    return degree, ctrl, U


def _eval_bspline_point(degree: int, ctrl: np.ndarray, knots: np.ndarray, t: float) -> np.ndarray:
    n = len(ctrl) - 1
    span = _find_span(n, degree, t, knots)
    N = _basis_funs(span, t, degree, knots)
    idx = span - degree
    p = np.zeros(3)
    for i in range(degree + 1):
        p += N[i] * ctrl[idx + i]
    return p


def _oriented_edge(w: _Writer, edge_id: int, sense: bool) -> int:
    flag = ".T." if sense else ".F."
    # ORIENTED_EDGE's edge_start/edge_end are written as * (derive from the
    # edge), which keeps them consistent with the EDGE_CURVE whatever `sense`.
    return w.add(f"ORIENTED_EDGE('',*,*,#{edge_id},{flag})")


def _make_loop(fb: _FaceBuilder, w: _Writer, loop_verts: list[int],
               fit_obj, verts: np.ndarray, segs=None, winding_sign=1,
               spans=None, tolerance: float | None = None) -> int:
    """`segs`, if given, is a precomputed list of `EdgeSeg`s aligned
    edge-by-edge with `loop_verts` (the full-circle collapse in
    `fit.simplify_loop`, which `classify_edge` cannot derive for a plane fit).
    Otherwise each edge is classified on the fly.

    `winding_sign` (see `fit.loop_winding_sign`) is this loop's rotational
    direction around a cylinder or cone fit's axis, +1 counter-clockwise, -1
    clockwise. It reaches `fb.edge()` so a circular arc's `same_sense` reflects
    the direction THIS loop travels; the two rim loops of a cylinder or cone
    wind in opposite directions, and a wrong flag turns every arc of one into
    the reflex sweep on reimport. It may also be a per-edge sequence (same
    length as `loop_verts`), needed for a seam-merged loop (see
    `_make_seamed_loop`) whose halves wind in opposite directions.

    `spans`, if given, is a `{(a, b): [raw vertex ids]}` map (see
    `boundary.BoundaryGraph.simplify_loop`'s `with_spans=True`): the raw
    mesh-vertex run behind a simplified edge, passed to `fb.edge()` so a chord
    or freeform edge can be written as a curve instead of a straight line.
    `tolerance` is the export tolerance to check that curve against the
    analytic surface, where one exists."""
    per_edge = winding_sign if hasattr(winding_sign, "__len__") else None
    oriented = []
    n = len(loop_verts)
    for i in range(n):
        a = loop_verts[i]
        b = loop_verts[(i + 1) % n]
        if segs is not None:
            seg = segs[i]
        elif fit_obj is None:
            seg = None
        else:
            seg = _fit.classify_edge(fit_obj, fb._pos(a), fb._pos(b))
        ws = per_edge[i] if per_edge is not None else winding_sign
        raw_span = spans.get((a, b)) if spans is not None else None
        edge_id, sense = fb.edge(a, b, seg, ws, raw_span=raw_span,
                                  fit_obj=fit_obj, tolerance=tolerance)
        oriented.append(_oriented_edge(w, edge_id, sense))
    edges_str = ",".join(f"#{e}" for e in oriented)
    return w.add(f"EDGE_LOOP('',({edges_str}))")


def _make_surface(w: _Writer, fit_obj, scale: float) -> int:
    kind = fit_obj.kind
    if kind == "plane":
        origin = fit_obj.origin * scale
        normal = fit_obj.normal
        ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        ref = ref - normal * float(np.dot(ref, normal))
        ref /= np.linalg.norm(ref)
        cp = w.add(f"CARTESIAN_POINT('',{_dir(origin)})")
        axisdir = w.add(f"DIRECTION('',{_dir(normal)})")
        refdir = w.add(f"DIRECTION('',{_dir(ref)})")
        placement = w.add(f"AXIS2_PLACEMENT_3D('',#{cp},#{axisdir},#{refdir})")
        return w.add(f"PLANE('',#{placement})")

    if kind == "cylinder":
        origin = fit_obj.axis_point * scale
        axis = fit_obj.axis_dir
        ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        ref = ref - axis * float(np.dot(ref, axis))
        ref /= np.linalg.norm(ref)
        cp = w.add(f"CARTESIAN_POINT('',{_dir(origin)})")
        axisdir = w.add(f"DIRECTION('',{_dir(axis)})")
        refdir = w.add(f"DIRECTION('',{_dir(ref)})")
        placement = w.add(f"AXIS2_PLACEMENT_3D('',#{cp},#{axisdir},#{refdir})")
        return w.add(f"CYLINDRICAL_SURFACE('',#{placement},{_num(fit_obj.radius * scale)})")

    if kind == "cone":
        origin = fit_obj.apex * scale
        axis = fit_obj.axis_dir
        ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        ref = ref - axis * float(np.dot(ref, axis))
        ref /= np.linalg.norm(ref)
        cp = w.add(f"CARTESIAN_POINT('',{_dir(origin)})")
        axisdir = w.add(f"DIRECTION('',{_dir(axis)})")
        refdir = w.add(f"DIRECTION('',{_dir(ref)})")
        placement = w.add(f"AXIS2_PLACEMENT_3D('',#{cp},#{axisdir},#{refdir})")
        return w.add(
            f"CONICAL_SURFACE('',#{placement},0.,{_num(fit_obj.semi_angle)})")

    if kind == "torus":
        origin = fit_obj.axis_point * scale
        axis = fit_obj.axis_dir
        ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        ref = ref - axis * float(np.dot(ref, axis))
        ref /= np.linalg.norm(ref)
        cp = w.add(f"CARTESIAN_POINT('',{_dir(origin)})")
        axisdir = w.add(f"DIRECTION('',{_dir(axis)})")
        refdir = w.add(f"DIRECTION('',{_dir(ref)})")
        placement = w.add(f"AXIS2_PLACEMENT_3D('',#{cp},#{axisdir},#{refdir})")
        return w.add(
            f"TOROIDAL_SURFACE('',#{placement},"
            f"{_num(fit_obj.major_radius * scale)},"
            f"{_num(fit_obj.minor_radius * scale)})")

    if kind == "bspline":
        # A freeform patch (see freeform.py). Control points are written row by
        # row in u, each row spanning v: the (i = u, j = v) grid order the
        # reader's `_build_bspline_surface` expects and `BSplineSurface` indexes.
        ctrl = fit_obj.ctrl * scale
        rows = []
        for i in range(ctrl.shape[0]):
            ids = [w.add(f"CARTESIAN_POINT('',{_dir(ctrl[i, j])})")
                   for j in range(ctrl.shape[1])]
            rows.append("(" + ",".join(f"#{p}" for p in ids) + ")")
        grid = "(" + ",".join(rows) + ")"
        uk, um = _freeform.compact_knots(fit_obj.knots_u)
        vk, vm = _freeform.compact_knots(fit_obj.knots_v)
        mult_u = ",".join(str(m) for m in um)
        mult_v = ",".join(str(m) for m in vm)
        knot_u = ",".join(_num(k) for k in uk)
        knot_v = ",".join(_num(k) for k in vk)
        # Non-rational, non-periodic, not self-intersecting: every optional
        # NURBS feature is declared off rather than left for a reader to infer.
        return w.add(
            f"B_SPLINE_SURFACE_WITH_KNOTS('',{fit_obj.deg_u},{fit_obj.deg_v},"
            f"{grid},.UNSPECIFIED.,.F.,.F.,.F.,"
            f"({mult_u}),({mult_v}),({knot_u}),({knot_v}),.UNSPECIFIED.)")

    if kind == "sphere":
        origin = fit_obj.center * scale
        axis = np.array([0.0, 0.0, 1.0])
        ref = np.array([1.0, 0.0, 0.0])
        cp = w.add(f"CARTESIAN_POINT('',{_dir(origin)})")
        axisdir = w.add(f"DIRECTION('',{_dir(axis)})")
        refdir = w.add(f"DIRECTION('',{_dir(ref)})")
        placement = w.add(f"AXIS2_PLACEMENT_3D('',#{cp},#{axisdir},#{refdir})")
        return w.add(f"SPHERICAL_SURFACE('',#{placement},{_num(fit_obj.radius * scale)})")

    raise ValueError(f"unknown fit kind {kind!r}")


def _loop_area3(loop, verts):
    """Real 3D area of a closed polygon loop (vertex indices into `verts`):
    the sum of triangle-fan cross products about the centroid, valid for any
    near-planar loop whatever its winding or orientation. Picks the outer
    perimeter among a face's boundary loops (see `iter_build_brep_faces`): an outer
    loop encloses more area than any hole inside it, unlike the order in which
    `boundary_loops` or `simplify_loop` happen to return them."""
    if len(loop) < 3:
        return 0.0
    pts = verts[loop]
    c = pts.mean(axis=0)
    total = np.zeros(3)
    n = len(pts)
    for i in range(n):
        total += np.cross(pts[i] - c, pts[(i + 1) % n] - c)
    return 0.5 * float(np.linalg.norm(total))


def _triangle_plane(a, b, c):
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n)
    if norm < 1e-14:
        return None
    n = n / norm
    return _fit.PlaneFit(origin=a, normal=n, max_err=0.0)


def _cyl_frame(fitted):
    axis_pt = fitted.apex if fitted.kind == "cone" else fitted.axis_point
    axis = fitted.axis_dir
    e1, e2 = _fit._orthonormal_basis(axis)
    return axis_pt, axis, e1, e2


def _seg_forward_sign(seg: _fit.EdgeSeg, p_from: np.ndarray, p_to: np.ndarray) -> int:
    """+1 if travelling p_from -> p_to along `seg`'s own circle (its center,
    axis, ref_dir parametrization) goes in that circle's increasing-parameter
    direction, -1 if backward. Generic over any CIRCLE `EdgeSeg`; used for the
    seam connector edge in `_make_seamed_loop`, which for a torus is a meridian
    arc with no fixed direction convention."""
    axis2 = np.cross(seg.axis, seg.ref_dir)

    def _ang(p):
        d = p - seg.center
        return math.atan2(float(np.dot(d, axis2)), float(np.dot(d, seg.ref_dir)))

    dt = math.atan2(math.sin(_ang(p_to) - _ang(p_from)), math.cos(_ang(p_to) - _ang(p_from)))
    return 1 if dt >= 0 else -1


def _loop_is_full_circle(loop, segs, verts, fit_obj):
    """True if every edge of this (already simplified) boundary loop is a
    circular arc of `fit_obj`: a full, untrimmed rim ring, not a partial arc
    bounded by straight cut edges. `segs`, if given (the plane full-circle
    collapse), is trusted; otherwise each edge is classified here, as
    `_make_loop` would."""
    n = len(loop)
    if n < 2:
        return False
    if segs is not None:
        return all(s.kind == "circle" for s in segs)
    for i in range(n):
        a, b = loop[i], loop[(i + 1) % n]
        if _fit.classify_edge(fit_obj, verts[a], verts[b]).kind != "circle":
            return False
    return True


def _make_seamed_loop(fb: _FaceBuilder, verts: np.ndarray, fitted,
                      ring0, ring1, tolerance: float,
                      winding_signs: tuple[int, int] = (1, -1)):
    """Merge the two full closed rim loops of an untrimmed cylinder, cone or
    torus lateral face into ONE seamed loop, so it can be written as a simple
    FACE_OUTER_BOUND. Two full loops of the same periodic strip do not nest (a
    hole must lie inside its outer loop once flattened to (u, v)), which is not
    a valid outer-boundary-plus-hole decomposition and breaks the reader's
    re-triangulation.

    `ring0`/`ring1` are `(loop, segs)` pairs: with shared boundaries on, the
    `BoundaryGraph` loops, i.e. the exact vertex sequences the neighbouring
    face (typically an end cap) also uses. The merged loop keeps every one of
    those shared vertices and adds a single connecting element, a "pinch"
    where the loop touches one vertex of each ring. Collapsing each ring to one
    point would sidestep the seam angle but drop the shared
    vertices, so the faces would stop sharing EDGE_CURVEs and the shell would
    open there.

    For ring0 = [A, B, C] and ring1 = [D, E] (D being ring1's vertex closest in
    angle to A), the merged loop is [A, B, C, A, D, E, D]: ring0 closed back to
    A (its real closing edge, shared with ring0's neighbour), a connector
    A->D, ring1 closed back to D, and a connector D->A. The two connectors are
    the same undirected pair in opposite directions, so `_FaceBuilder.edge()`
    makes them one EDGE_CURVE referenced exactly twice by this face alone,
    which needs nothing from any neighbour.

    The connector (A, D) is rarely an exact generatrix. `classify_edge` reports
    a LINE inside its angle tolerance (exact for a cylinder or cone), or for a
    torus a meridian CIRCLE with a small chord error bounded by how finely the
    ring was tessellated. D is chosen as the angularly closest vertex to
    minimise that gap.

    Returns `(loop, None, per_edge_winding)` (segs `None`, so `_make_loop`
    classifies every edge, ring and connector alike), or `None` if the two
    rings are unusable, in which case the caller writes two separate loops."""
    loop0, _segs0 = ring0
    loop1, _segs1 = ring1
    if len(loop0) < 1 or len(loop1) < 1:
        return None
    axis_pt, _, e1, e2 = _cyl_frame(fitted)

    def _angle_of(vidx):
        d = fb._pos(vidx) - axis_pt
        return math.atan2(float(np.dot(d, e2)), float(np.dot(d, e1)))

    a_cut = loop0[0]
    theta_a = _angle_of(a_cut)
    best_k, best_dtheta = 0, None
    for k, v in enumerate(loop1):
        dtheta = abs(math.atan2(math.sin(_angle_of(v) - theta_a),
                                math.cos(_angle_of(v) - theta_a)))
        if best_dtheta is None or dtheta < best_dtheta:
            best_dtheta, best_k = dtheta, k
    d_cut = loop1[best_k]
    loop1_reordered = loop1[best_k:] + loop1[:best_k]

    merged_loop = list(loop0) + [a_cut] + loop1_reordered + [d_cut]
    n0 = len(loop0)
    n1 = len(loop1_reordered)

    connector_seg = _fit.classify_edge(fitted, fb._pos(a_cut), fb._pos(d_cut))
    if connector_seg.kind == "chord" and connector_seg.sagitta > tolerance:
        return None  # tessellation too coarse around this angle for the
                      # connector to stay within tolerance: two-loop fallback
    connector_winding = (_seg_forward_sign(connector_seg, fb._pos(a_cut), fb._pos(d_cut))
                         if connector_seg.kind == "circle" else 0)

    per_edge_winding = ([winding_signs[0]] * n0 + [connector_winding]
                        + [winding_signs[1]] * n1 + [0])
    return merged_loop, None, per_edge_winding


STAT_FIELDS = ("plane_faces", "cylinder_faces", "cone_faces", "sphere_faces",
               "torus_faces",
               "bspline_faces", "bspline_triangle_faces",
               "leftover_triangle_faces", "leftover_faces", "rejected_patches",
               "source_faces")


class BrepStats:
    def __init__(self):
        self.plane_faces = 0
        self.cylinder_faces = 0
        self.cone_faces = 0
        self.sphere_faces = 0
        self.torus_faces = 0
        self.bspline_faces = 0            # freeform B_SPLINE_SURFACE faces
        self.bspline_triangle_faces = 0   # triangles they replaced
        self.leftover_triangle_faces = 0  # original triangles that ended up as fallback
        self.leftover_faces = 0           # ADVANCED_FACE entities written for them (after merge)
        self.rejected_patches = 0
        # faces written with the exact surface they were imported with
        # (unedited source faces; also counted in the per-kind totals)
        self.source_faces = 0
        # Set when the freeform tier attempted fits but none held the
        # tolerance: roughly the tolerance this mesh COULD have supported.
        # Not a count, so not in STAT_FIELDS (see `export_step_brep`).
        self.freeform_achievable_mm = None
        # Shared-boundary bookkeeping (see boundary.py): distinct chains and
        # the edges they collapsed to. Not counts of faces.
        self.boundary_chains = 0
        self.boundary_points_raw = 0
        self.boundary_points_shared = 0
        # How many faces took their loops from the shared boundary graph and
        # how many fell back to simplifying on their own, which is what the
        # graph exists to prevent: such a face stops referencing the same
        # EDGE_CURVEs as its neighbours and opens the shell. Counted because
        # the graph itself reports healthy while faces quietly decline to use
        # it. Not counts of faces written.
        self.shared_loop_faces = 0
        self.shared_loop_skipped_no_loops = 0
        self.shared_loop_skipped_short = 0
        self.shared_loop_seamed = 0
        # BoundaryGraph's own health counters, all 0 on a well-formed mesh:
        # collisions and protected are automatic construction-time repairs,
        # fallbacks a should-be-impossible fallback to raw vertices in
        # `simplify_loop`. Exposed so tests/curved_topology_gate.py can assert
        # on them.
        self.graph_collisions = 0
        self.graph_protected = 0
        self.graph_fallbacks = 0
        # Boundary edges written as a fitted B_SPLINE_CURVE_WITH_KNOTS instead
        # of a straight LINE (`_FaceBuilder._try_curve_from_span`).
        self.boundary_curve_edges = 0
        # How many primitive patches `_merge_adjacent_patches` folded into a
        # neighbour.
        self.patches_merged = 0


def _accept_patch(verts, faces, patch_face_idx, fit_obj, tolerance, angle_tol_deg=3.0):
    """Re-check every boundary edge's chord-approximation error against
    `tolerance`; reject the whole patch (the caller falls back to per-triangle
    planar faces) if any edge deviates too much, if the boundary is not a
    clean, simply connected silhouette, or if the combined boundary (outer
    ring and holes) exceeds `_fit.MAX_BOUNDARY_PTS` (see `merge_coplanar`).

    Returns `(simplified, winding_signs)` on acceptance, `None` on rejection.
    `winding_signs` has one `fit.loop_winding_sign(...)` per loop (only
    meaningful for cylinder, cone and torus fits; `1` otherwise)."""
    loops, ok = _fit.boundary_loops(faces, patch_face_idx)
    if not ok:
        return None  # pinched/invalid boundary -- never trust it
    if not loops:
        return None  # fully closed patch (e.g. a whole sphere): caller handles it

    if fit_obj.kind in ("cylinder", "cone", "torus"):
        winding_signs = [_fit.loop_winding_sign(loop, verts, fit_obj) for loop in loops]
    else:
        winding_signs = [1] * len(loops)

    # Simplify each loop (collinear-point merge, circular-arc merge, and, for a
    # plane fit, full-circle collapse) before counting against MAX_BOUNDARY_PTS:
    # the raw per-vertex count overstates a patch's complexity and would reject
    # large flat faces for being finely tessellated.
    simplified = [_fit.simplify_loop(loop, verts, fit_obj, tolerance, angle_tol_deg)
                  for loop in loops]
    if sum(len(nl) for nl, _ in simplified) > _fit.MAX_BOUNDARY_PTS:
        return None  # too complex for the reader's triangulator
    for new_loop, segs in simplified:
        n = len(new_loop)
        for i in range(n):
            a = new_loop[i]
            b = new_loop[(i + 1) % n]
            seg = segs[i] if segs is not None else _fit.classify_edge(
                fit_obj, verts[a], verts[b])
            if seg.kind == "chord" and seg.sagitta > tolerance:
                return None
    return simplified, winding_signs


def _fits_compatible(f1, f2, tolerance: float, angle_tol_deg: float) -> bool:
    """Whether two fitted primitive patches plausibly belong to one larger
    surface of the same kind, checked geometrically (axis or normal alignment,
    coincidence, matching radii) so two different cylinders that merely share a
    boundary are never merged. A pre-filter for `_merge_adjacent_patches`; a
    merge is only trusted after the combined patch is re-fitted and passes
    `_accept_patch`.

    Axes and normals are compared with `abs(dot(...))`: a fit's axis sign is an
    arbitrary PCA eigenvector choice."""
    if f1.kind != f2.kind:
        return False
    kind = f1.kind
    cos_tol = math.cos(math.radians(angle_tol_deg))
    if kind == "plane":
        if abs(float(np.dot(f1.normal, f2.normal))) < cos_tol:
            return False
        return abs(float(np.dot(f2.origin - f1.origin, f1.normal))) <= tolerance
    if kind == "sphere":
        if abs(f1.radius - f2.radius) > tolerance:
            return False
        return float(np.linalg.norm(f1.center - f2.center)) <= tolerance
    if kind == "cylinder":
        if abs(float(np.dot(f1.axis_dir, f2.axis_dir))) < cos_tol:
            return False
        if abs(f1.radius - f2.radius) > tolerance:
            return False
        d = f2.axis_point - f1.axis_point
        perp = d - float(np.dot(d, f1.axis_dir)) * f1.axis_dir
        return float(np.linalg.norm(perp)) <= tolerance
    if kind == "cone":
        if abs(float(np.dot(f1.axis_dir, f2.axis_dir))) < cos_tol:
            return False
        if abs(f1.semi_angle - f2.semi_angle) > math.radians(angle_tol_deg):
            return False
        return float(np.linalg.norm(f1.apex - f2.apex)) <= tolerance
    if kind == "torus":
        if abs(float(np.dot(f1.axis_dir, f2.axis_dir))) < cos_tol:
            return False
        if abs(f1.major_radius - f2.major_radius) > tolerance:
            return False
        if abs(f1.minor_radius - f2.minor_radius) > tolerance:
            return False
        d = f2.axis_point - f1.axis_point
        perp = d - float(np.dot(d, f1.axis_dir)) * f1.axis_dir
        return float(np.linalg.norm(perp)) <= tolerance
    return False


def _refit_same_kind(verts, faces, face_idx, seed_fit):
    """Re-fit `face_idx` directly as `seed_fit.kind`, seeded with its axis,
    instead of running the full plane -> sphere -> cylinder -> cone -> torus
    cascade (`fit.fit_patch`): the two patches being merged were already
    classified as the same kind by `_fits_compatible`."""
    vert_ids = sorted({i for fidx in face_idx for i in faces[fidx]})
    pts = verts[vert_ids]
    kind = seed_fit.kind
    if kind == "plane":
        return _fit.plane_fit(pts)
    if kind == "sphere":
        return _fit._fit_sphere(pts)
    if kind == "cylinder":
        return _fit._fit_cylinder_given_axis(pts, seed_fit.axis_dir)
    if kind == "cone":
        return _fit._fit_cone_given_axis(pts, seed_fit.axis_dir)
    if kind == "torus":
        return _fit._fit_torus_refined(pts, seed_fit.axis_dir)
    return None


def _merge_adjacent_patches(verts, faces, entries, tolerance, angle_tol_deg=3.0):
    """Merge two adjacent, already fitted PRIMITIVE patches into one larger
    face when they plausibly belong to the same surface (`_fits_compatible`)
    and the combined, re-fitted patch still passes `_accept_patch` (boundary
    stays simple, within `MAX_BOUNDARY_PTS`, every edge within `tolerance`).
    Repeats until nothing changes, so a run of 3+ over-segmented patches
    collapses in one call.

    A smooth cylindrical face that `fit.segment` split in two would otherwise
    stay split in the written file. Only `entry["kind"] == "primitive"` entries
    take part; freeform and flat patches have their own merge passes
    (`freeform.fit_freeform_regions`, `fit.merge_coplanar`). Mutates and
    returns `entries`."""
    changed = True
    while changed:
        changed = False
        group_of: dict[int, int] = {}
        for ei, e in enumerate(entries):
            if e["kind"] != "primitive" or e["fit"] is None:
                continue
            for fidx in e["group"]:
                group_of[fidx] = ei

        edge_tris: dict[tuple[int, int], list[int]] = {}
        for fidx in group_of:
            a, b, c = faces[fidx]
            for u, v in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                edge_tris.setdefault(key, []).append(fidx)

        pairs = set()
        for tris in edge_tris.values():
            if len(tris) != 2:
                continue
            ei, ej = group_of[tris[0]], group_of[tris[1]]
            if ei != ej:
                pairs.add((min(ei, ej), max(ei, ej)))

        for i, j in sorted(pairs):
            f1, f2 = entries[i]["fit"], entries[j]["fit"]
            if not _fits_compatible(f1, f2, tolerance, angle_tol_deg):
                continue
            combined = sorted(set(entries[i]["group"]) | set(entries[j]["group"]))
            refit = _refit_same_kind(verts, faces, combined, f1)
            if refit is None or refit.max_err > tolerance:
                continue
            accepted = _accept_patch(verts, faces, combined, refit, tolerance, angle_tol_deg)
            if accepted is not None:
                simplified, winding_signs = accepted
            else:
                check_loops, check_ok = _fit.boundary_loops(faces, combined)
                if not (check_ok and len(check_loops) == 0):
                    continue  # not usable merged either way -- leave both as they were
                simplified, winding_signs = [], []
            entries[i] = {"kind": "primitive", "fit": refit, "group": combined,
                          "simplified": simplified, "winding": winding_signs}
            del entries[j]
            changed = True
            break  # `entries` indices shifted -- rebuild group_of and restart
    return entries


# Whether a full untrimmed cylinder, cone or torus lateral face is written as
# one seamed loop (see `_make_seamed_loop`) instead of two separate rim loops.
# Two full periodic rim loops cannot be written as outer boundary plus hole
# and survive a reader's re-triangulation (a hole must lie inside its outer
# loop once flattened to (u, v)), and a full cylinder is lost within a few
# export/reimport cycles without it, so it stays on. The seamed loop is built
# from the shared chains and keeps every shared vertex of both rings, so it
# does not break shared boundaries.
SEAMED_LOOPS = True


class _SourceBSpline:
    """An imported B-spline surface in the shape `_make_surface` writes."""
    kind = "bspline"

    def __init__(self, rec, scale):
        self.deg_u = int(rec["deg_u"])
        self.deg_v = int(rec["deg_v"])
        self.ctrl = np.asarray(rec["ctrl"], dtype=float) * scale
        self.knots_u = np.asarray(rec["knots_u"], dtype=float)
        self.knots_v = np.asarray(rec["knots_v"], dtype=float)
        self._surf = None

    def distance(self, p):
        from .geometry import BSplineSurface
        if self._surf is None:
            self._surf = BSplineSurface(self.deg_u, self.deg_v, self.ctrl,
                                        self.knots_u, self.knots_v)
        u, v = self._surf.invert(p)
        return float(np.linalg.norm(self._surf.eval(u, v) - p))


def source_fit(rec: dict, scale: float):
    """Fit object (fit.PlaneFit, ...) for a surface record from import
    (`convert.surface_record`), in the exporter's units."""
    k = rec.get("kind")
    def a(key):
        return np.asarray(rec[key], dtype=float)
    try:
        if k == "plane":
            return _fit.PlaneFit(origin=a("origin") * scale, normal=a("normal"))
        if k == "cylinder":
            return _fit.CylinderFit(axis_point=a("axis_point") * scale, axis_dir=a("axis_dir"),
                                    radius=float(rec["radius"]) * scale)
        if k == "cone":
            return _fit.ConeFit(apex=a("apex") * scale, axis_dir=a("axis_dir"),
                                semi_angle=float(rec["semi_angle"]))
        if k == "sphere":
            return _fit.SphereFit(center=a("center") * scale, radius=float(rec["radius"]) * scale)
        if k == "torus":
            return _fit.TorusFit(axis_point=a("axis_point") * scale, axis_dir=a("axis_dir"),
                                 major_radius=float(rec["major_radius"]) * scale,
                                 minor_radius=float(rec["minor_radius"]) * scale)
        if k == "bspline":
            return _SourceBSpline(rec, scale)
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _source_distance(fit_obj, p):
    k = fit_obj.kind
    if k == "plane":
        return abs(float(np.dot(p - fit_obj.origin, fit_obj.normal)))
    if k == "sphere":
        return abs(float(np.linalg.norm(p - fit_obj.center)) - fit_obj.radius)
    if k == "bspline":
        return fit_obj.distance(p)
    d = _fit.distance_to_surface(fit_obj, p)
    return float("inf") if d is None else d


def _seg_dist(p, a, b):
    ab = b - a
    L2 = float(np.dot(ab, ab))
    t = 0.0 if L2 <= 0.0 else min(1.0, max(0.0, float(np.dot(p - a, ab)) / L2))
    return float(np.linalg.norm(p - (a + t * ab)))


def _still_on_source(verts, faces, group, fit_obj, tolerance, max_checks=400):
    """True if the triangle group still has its source surface's shape, so
    the original surface can be written back unchanged.

    Every interior vertex must lie on the surface (within `tolerance`). A
    boundary vertex may instead lie on the straight line between the nearest
    on-surface vertices either side of it along the rim: that is where the
    importer puts the points of an edge a previous Curved export wrote as a
    straight chord (the chord leaves a curved surface by its sagitta). A
    vertex moved by an edit is on neither."""
    loops, ok = _fit.boundary_loops(faces, group)
    if not ok:
        return False
    bnd = {i for l in loops for i in l}
    interior = sorted({i for t in group for i in faces[t]} - bnd)
    if len(interior) > max_checks and fit_obj.kind == "bspline":
        step = len(interior) / max_checks
        interior = [interior[int(k * step)] for k in range(max_checks)]
    if any(_source_distance(fit_obj, verts[i]) > tolerance for i in interior):
        return False
    for loop in loops:
        n = len(loop)
        on = [_source_distance(fit_obj, verts[i]) <= tolerance for i in loop]
        if all(on):
            continue
        if not any(on):
            return False
        for k in range(n):
            if on[k]:
                continue
            j = k
            while not on[j % n]:
                j -= 1
            a = verts[loop[j % n]]
            j = k
            while not on[j % n]:
                j += 1
            b = verts[loop[j % n]]
            if _seg_dist(verts[loop[k]], a, b) > tolerance:
                return False
    return True


def _orient_plane(fit_obj, verts, faces, group):
    """Point a source plane's normal the way its triangles face."""
    n = np.zeros(3)
    for t in group:
        a, b, c = faces[t]
        n += np.cross(verts[b] - verts[a], verts[c] - verts[a])
    if float(np.dot(n, fit_obj.normal)) < 0:
        fit_obj.normal = -fit_obj.normal
    return fit_obj


def _outward_normals(fit_obj, pts):
    """Unit normal of an analytic surface at `pts` in STEP's convention
    (cylinder, cone, sphere: away from the axis or centre; torus: away from
    the tube's centre circle). Rows that fall on the axis stay zero."""
    kind = fit_obj.kind
    if kind == "sphere":
        n = pts - fit_obj.center
    else:
        axis = fit_obj.axis_dir
        origin = fit_obj.apex if kind == "cone" else fit_obj.axis_point
        d = pts - origin
        h = d @ axis
        radial = d - np.outer(h, axis)
        ln = np.linalg.norm(radial, axis=1)
        ur = radial / np.maximum(ln, 1e-300)[:, None]
        if kind == "cylinder":
            n = ur
        elif kind == "cone":
            ca, sa = math.cos(fit_obj.semi_angle), math.sin(fit_obj.semi_angle)
            n = ca * ur - (np.sign(h) * sa)[:, None] * axis
        elif kind == "torus":
            n = pts - (origin + fit_obj.major_radius * ur)
        else:
            return np.zeros_like(pts)
    ln = np.linalg.norm(n, axis=1)
    return n / np.maximum(ln, 1e-300)[:, None]


def _face_sense(fit_obj, verts, faces, group) -> bool:
    """`ADVANCED_FACE.same_sense` for a curved analytic face: True when the
    surface normal points the way the group's triangles face (out of the
    material). A bore, for instance, has its cylinder normal pointing at the
    material and needs .F.. Writing it explicitly keeps the face orientation
    from depending on how a reader guesses it from the loops."""
    tri = verts[np.array([faces[t] for t in group], dtype=np.int64)]
    area_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n = _outward_normals(fit_obj, tri.mean(axis=1))
    return float(np.einsum("ij,ij->", area_n, n)) >= 0.0


# ---------------------------------------------------------------------------
# Per-patch / per-group fitting, shared by the sequential path and the worker
# processes (`core/workers.py`), so both classify a patch identically.
# ---------------------------------------------------------------------------

# Below this many patches/groups, starting worker processes is not worth it.
MIN_ITEMS_FOR_POOL = 24


def _fit_one_patch(verts, faces, patch, tolerance, line_angle_tol_deg):
    """Fit + accept one smooth patch. Returns `(kind, fitted, simplified,
    winding_signs)`; kind is "fitted", "closed" (a surface without boundary,
    e.g. a whole sphere) or "leftover" (no usable fit)."""
    fitted = _fit.fit_patch(verts, faces, patch, tolerance)
    if fitted is None:
        return "leftover", None, None, None
    accepted = _accept_patch(verts, faces, patch, fitted, tolerance,
                             line_angle_tol_deg)
    if accepted is None:
        loops, ok = _fit.boundary_loops(faces, patch)
        if ok and len(loops) == 0:
            return "closed", fitted, [], []
        return "leftover", None, None, None
    simplified, winding_signs = accepted
    return "fitted", fitted, simplified, winding_signs


def _fit_one_group(verts, faces, group, tolerance, line_angle_tol_deg):
    """Plane-fit + simplify the boundary of one re-flattened group. Returns
    `(usable, plane_fit, simplified)`; not usable means the caller writes
    the group's triangles one by one."""
    vids = sorted({i for fidx in group for i in faces[fidx]})
    pf = _fit.plane_fit(verts[vids])
    loops, ok = _fit.boundary_loops(faces, group)
    simplified = [_fit.simplify_loop(loop, verts, pf, tolerance,
                                     line_angle_tol_deg)
                  for loop in loops] if (ok and loops) else []
    usable = bool(ok and loops) and \
        sum(len(nl) for nl, _ in simplified) <= _fit.MAX_BOUNDARY_PTS
    return usable, pf, simplified


def _pack(obj):
    """Fit / edge-segment dataclass -> plain dict (for the process boundary)."""
    return None if obj is None else (type(obj).__name__, dict(vars(obj)))


def _unpack(packed):
    if packed is None:
        return None
    name, fields = packed
    obj = getattr(_fit, name).__new__(getattr(_fit, name))
    obj.__dict__.update(fields)
    return obj


def _pack_simplified(simplified):
    return None if simplified is None else [
        (list(loop), None if segs is None else [_pack(sg) for sg in segs])
        for loop, segs in simplified]


def _unpack_simplified(packed):
    return None if packed is None else [
        (loop, None if segs is None else [_unpack(sg) for sg in segs])
        for loop, segs in packed]


def _patch_job(state):
    """Worker-side factory (see `workers.serve`): one patch index in, plain
    data out."""
    verts, faces, patches = state["verts"], state["faces"], state["items"]
    tol, ang = state["tolerance"], state["line_angle_tol_deg"]

    def run(pi):
        kind, fitted, simplified, winding = _fit_one_patch(
            verts, faces, patches[pi], tol, ang)
        return kind, _pack(fitted), _pack_simplified(simplified), winding
    return run


def _group_job(state):
    verts, faces, groups = state["verts"], state["faces"], state["items"]
    tol, ang = state["tolerance"], state["line_angle_tol_deg"]

    def run(gi):
        usable, pf, simplified = _fit_one_group(verts, faces, groups[gi], tol, ang)
        return usable, _pack(pf), _pack_simplified(simplified)
    return run


def _fit_pooled(job, verts, faces, items, tolerance, line_angle_tol_deg,
                max_workers, base, span, label):
    """Generator: run `job` for every index of `items` on worker processes.
    Yields `(frac, msg)` (frac within [base, base + span]); returns the
    results in `items` order, or None if the workers failed (the caller then
    falls back to the sequential path)."""
    total = len(items)
    n = _workers.worker_count(max_workers, total)
    state = {"verts": np.asarray(verts, dtype=float),
             "faces": [tuple(int(i) for i in f) for f in faces],
             "items": [[int(i) for i in it] for it in items],
             "tolerance": float(tolerance),
             "line_angle_tol_deg": float(line_angle_tol_deg)}
    results = [None] * total
    done = 0
    msg = f"Fitting {total} {label} on {n} core(s)..."
    try:
        with _workers.WorkerPool(job, state, n) as pool:
            for out in pool.imap(range(total)):
                if out is not None:
                    i, res = out
                    if job.endswith("_patch_job"):
                        kind, fitted, simplified, winding = res
                        results[i] = (kind, _unpack(fitted),
                                      _unpack_simplified(simplified), winding)
                    else:
                        usable, pf, simplified = res
                        results[i] = (usable, _unpack(pf),
                                      _unpack_simplified(simplified))
                    done += 1
                    msg = f"Fitting {total} {label} on {n} core(s): {done}/{total}"
                yield base + span * (done / max(total, 1)), msg
    except Exception:  # noqa: BLE001 - any worker failure means "run sequentially"
        return None
    return results


SMALL_FACE_AREA = 10.0     # a flat face this many tolerance^2 or less is small


def _absorb_small_flat_faces(verts, faces, entries, tolerance):
    """Join each small flat face to the flat face next to it.

    A flat face of a few hundredths of a square millimetre is what a fit
    leaves along the seam of two patches. Written on its own it hands both
    neighbours a boundary of a few tenths of a millimetre, and its outline,
    a few hundredths of a millimetre off a straight line, does not survive
    the reader. A face joins the neighbouring plane that shares the longest
    edge with it and that all its vertices lie within half the tolerance of,
    provided the joined outline is still a clean set of loops. Faces that
    keep their source surface never take part. Returns the remaining
    entries, in their order."""
    limit = SMALL_FACE_AREA * tolerance * tolerance
    owner = {t: i for i, e in enumerate(entries) for t in e["group"]}
    edge_tris: dict[tuple[int, int], list[int]] = {}
    for t, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_tris.setdefault((u, v) if u < v else (v, u), []).append(t)

    def is_plane(e):
        return e["kind"] != "source" and e["fit"] is not None and e["fit"].kind == "plane"

    def area(group):
        p = verts[[faces[t][0] for t in group]]
        q = verts[[faces[t][1] for t in group]]
        r = verts[[faces[t][2] for t in group]]
        return 0.5 * float(np.linalg.norm(np.cross(q - p, r - p), axis=1).sum())

    alive = [True] * len(entries)
    order = sorted((i for i, e in enumerate(entries) if is_plane(e)),
                   key=lambda i: area(entries[i]["group"]))
    for i in order:
        e = entries[i]
        if not alive[i] or area(e["group"]) > limit:
            continue
        shared: dict[int, float] = {}
        for t in e["group"]:
            a, b, c = faces[t]
            for u, v in ((a, b), (b, c), (c, a)):
                for t2 in edge_tris[(u, v) if u < v else (v, u)]:
                    j = owner.get(t2)
                    if j is not None and j != i and alive[j] and is_plane(entries[j]):
                        shared[j] = shared.get(j, 0.0) + float(np.linalg.norm(verts[u] - verts[v]))
        pts = verts[sorted({k for t in e["group"] for k in faces[t]})]
        for j in sorted(shared, key=shared.get, reverse=True):
            plane = entries[j]["fit"]
            if float(np.abs((pts - plane.origin) @ plane.normal).max()) > 0.5 * tolerance:
                continue
            merged = list(entries[j]["group"]) + list(e["group"])
            loops, ok = _fit.boundary_loops(faces, merged)
            if not (ok and loops):
                continue
            entries[j]["group"] = merged
            for t in e["group"]:
                owner[t] = j
            alive[i] = False
            break
    return [e for i, e in enumerate(entries) if alive[i]]


def iter_build_brep_faces(w: _Writer, fb: _FaceBuilder, verts: np.ndarray,
                           faces: Sequence[tuple[int, int, int]],
                           tolerance: float, smooth_angle_deg: float,
                           stats: BrepStats,
                           line_angle_tol_deg: float = 3.0,
                           freeform: bool = True,
                           shared_boundaries: bool = True,
                           freeform_tolerance: float | None = None,
                           merge_patches: bool = True,
                           tri_face_ids: Sequence[int] | None = None,
                           source_fits: dict[int, object] | None = None,
                           parallel: bool = False,
                           max_workers: int | None = None):
    """Generator: segment + fit `faces`, emit ADVANCED_FACE entities.

    Yields `(frac_0_1, message)` between units of work (per patch, per fallback
    group, per written face) so a caller can stay responsive, and returns the
    list of face ids. With `parallel=True` the per-patch fitting (and the
    per-group fallback fitting) runs on worker processes (`core/workers.py`);
    writing the STEP entities stays sequential, in the original order, so the
    output is identical either way.

    `tri_face_ids` / `source_fits`: the STEP face each triangle was imported
    from, and that face's exact surface (see `source_fit`). A group of
    triangles whose vertices all still lie on its source surface is written as
    one face with that surface instead of being re-segmented and re-fitted, so
    an unedited imported part keeps its faces one to one.

    Three phases:

    1. **Decide every face.** Fit smooth patches to primitives, fit freeform
       B-spline surfaces to what is left (`freeform.py`), re-flatten the rest
       (`fit.merge_coplanar`), splitting a group into single triangles where
       its boundary is unusable. Nothing is written yet; the result is one
       group of triangles per future face.
    2. **Build the shared boundary graph** (`boundary.py`) over all groups at
       once and simplify each shared chain exactly once.
    3. **Write** every face, taking its loops from that graph.

    `shared_boundaries=False` makes each face simplify its own boundary
    (neighbours then disagree about their shared edge and the shell opens),
    for A/B comparison.

    `line_angle_tol_deg` is how far two adjacent boundary edges may differ in
    direction and still be merged into one straight edge (see
    `fit.merge_collinear`). It applies to the per-face path only; the
    shared-chain path is error-bounded by `tolerance`
    (`boundary.simplify_chain`).

    `freeform_tolerance`, if given, replaces `tolerance` for the freeform tier.
    It decides whether a part comes back as a few large surfaces or many small
    ones, independently of how tightly cylinders and planes are fitted.

    `freeform` (default on): cover the leftover triangles with fitted B-spline
    surfaces before the flat re-merging. On an organic part that is a handful
    of curved faces instead of tens of thousands of one-triangle ones.

    `merge_patches` (default on): after the primitive patches are fitted and
    before the boundary graph is built, merge adjacent patches that plausibly
    belong to one larger surface of the same kind (`_merge_adjacent_patches`).
    """
    # A zero-area triangle belongs to no patch (`fit.segment` skips it), which
    # leaves the faces around it disagreeing about its middle point.
    faces, tri_face_ids = _fit.flip_degenerate_triangles(verts, faces, tri_face_ids)

    # ---- phase 0: unedited source faces keep their exact surface ----------
    entries: list[dict] = []      # one dict per future ADVANCED_FACE
    leftover: list[int] = []
    taken = set()
    if tri_face_ids is not None and source_fits and len(tri_face_ids) == len(faces):
        yield 0.0, "Checking faces against their original surfaces..."
        groups: dict[int, list[int]] = {}
        for t, fid in enumerate(tri_face_ids):
            if fid and int(fid) in source_fits:
                groups.setdefault(int(fid), []).append(t)
        for fid, group in groups.items():
            fit_obj = source_fits[fid]
            if not _still_on_source(verts, faces, group, fit_obj, tolerance):
                continue
            if fit_obj.kind == "plane":
                _orient_plane(fit_obj, verts, faces, group)
            # The surface is exact, so unlike a fitted patch there is no
            # chord-vs-tolerance test on the boundary: its edges are the
            # mesh's own, sampled at the import's (possibly coarser) chord
            # tolerance. Only a usable, not over-complex boundary is needed.
            loops, ok = _fit.boundary_loops(faces, group)
            if not ok:
                continue
            if fit_obj.kind == "bspline":
                simplified, winding_signs = [(l, None) for l in loops], None
            else:
                simplified = [_fit.simplify_loop(l, verts, fit_obj, tolerance,
                                                 line_angle_tol_deg) for l in loops]
                if sum(len(nl) for nl, _ in simplified) > _fit.MAX_BOUNDARY_PTS:
                    continue
                winding_signs = ([_fit.loop_winding_sign(l, verts, fit_obj) for l in loops]
                                 if fit_obj.kind in ("cylinder", "cone", "torus")
                                 else [1] * len(loops))
            entries.append({"kind": "source", "fit": fit_obj, "group": group,
                            "simplified": simplified, "winding": winding_signs})
            taken.update(group)

    # ---- phase 1: decide every other face ----------------------------------
    yield 0.0, "Segmenting mesh into smooth patches..."
    if taken:
        rest = [t for t in range(len(faces)) if t not in taken]
        sub = [faces[t] for t in rest]
        patches = [[rest[i] for i in p] for p in _fit.segment(verts, sub, smooth_angle_deg)] \
            if sub else []
    else:
        patches = _fit.segment(verts, faces, smooth_angle_deg)

    n_patches = max(len(patches), 1)

    patch_results = None
    if parallel and len(patches) >= MIN_ITEMS_FOR_POOL:
        patch_results = yield from _fit_pooled(
            "brep_export:_patch_job", verts, faces, patches, tolerance,
            line_angle_tol_deg, max_workers, 0.0, 0.5, "patches")

    for pi, patch in enumerate(patches):
        if patch_results is None:
            yield 0.5 * (pi / n_patches), f"Fitting patch {pi + 1}/{n_patches}..."
            res = _fit_one_patch(verts, faces, patch, tolerance, line_angle_tol_deg)
        else:
            res = patch_results[pi]
        kind, fitted, simplified, winding_signs = res
        if kind == "leftover":
            leftover.extend(patch)
            stats.rejected_patches += 1
            continue
        entries.append({"kind": "primitive", "fit": fitted, "group": patch,
                        "simplified": simplified, "winding": winding_signs})

    # Second tier: whatever no rigid primitive could hold gets one more chance
    # as a freeform surface before the flat fallback; without it an organic
    # part lands entirely on one-triangle faces.
    if leftover and freeform:
        yield 0.5, (f"Fitting freeform surfaces to {len(leftover)} "
                    f"leftover triangle(s)...")
        ff_stats: dict[str, object] = {}
        ff_tol = tolerance if not freeform_tolerance else freeform_tolerance
        accepted_ff, leftover = _freeform.fit_freeform_regions(
            verts, faces, leftover, ff_tol, stats=ff_stats)
        achievable = ff_stats.get("achievable_mm")
        if achievable is not None:
            prev = stats.freeform_achievable_mm
            stats.freeform_achievable_mm = (achievable if prev is None
                                            else max(prev, achievable))
        for fitted, simplified, region in accepted_ff:
            entries.append({"kind": "bspline", "fit": fitted, "group": region,
                            "simplified": simplified, "winding": None})

    # Leftover triangles that fit no primitive as a whole patch (a sheet-metal
    # panel with a few mm of bow, say): one ADVANCED_FACE per triangle would
    # bloat the file and make a re-import tessellate every tiny face on its
    # own. Greedily re-flatten them into the largest local patches that stay
    # within tolerance.
    if leftover:
        yield 0.65, f"Re-flattening {len(leftover)} leftover triangle(s)..."
        groups = _fit.merge_coplanar(verts, faces, leftover, tolerance,
                                      line_angle_tol_deg)
        n_groups = max(len(groups), 1)

        group_results = None
        if parallel and len(groups) >= MIN_ITEMS_FOR_POOL:
            group_results = yield from _fit_pooled(
                "brep_export:_group_job", verts, faces, groups, tolerance,
                line_angle_tol_deg, max_workers, 0.75, 0.15, "fallback faces")

        for gi, group in enumerate(groups):
            if group_results is None:
                yield (0.75 + 0.15 * (gi / n_groups),
                       f"Preparing fallback face {gi + 1}/{n_groups}...")
                res = _fit_one_group(verts, faces, group, tolerance,
                                     line_angle_tol_deg)
            else:
                res = group_results[gi]
            usable, pf, simplified = res
            if usable:
                entries.append({"kind": "flat", "fit": pf, "group": group,
                                "simplified": simplified, "winding": None})
                continue
            # `merge_coplanar` already rejects merges that pinch the boundary
            # or exceed the reader's safe complexity, so this should be
            # unreachable; never drop geometry if it happens anyway. A lone
            # triangle cannot pinch or overwhelm anything, so this is always
            # valid, just less compact.
            for fidx in group:
                a, b, c = faces[fidx]
                pf_tri = _triangle_plane(verts[a], verts[b], verts[c])
                if pf_tri is None:
                    continue
                entries.append({"kind": "triangle", "fit": pf_tri,
                                "group": [fidx],
                                "simplified": [([a, b, c], None)],
                                "winding": None})

    if merge_patches and entries:
        yield 0.87, "Merging adjacent same-surface patches..."
        n_before = sum(1 for e in entries if e["kind"] == "primitive")
        entries = _merge_adjacent_patches(verts, faces, entries, tolerance,
                                           line_angle_tol_deg)
        n_after = sum(1 for e in entries if e["kind"] == "primitive")
        stats.patches_merged = n_before - n_after

    if entries:
        entries = _absorb_small_flat_faces(verts, faces, entries, tolerance)

    # ---- phase 2: one shared, simplified boundary per neighbour pair -----
    graph = None
    if shared_boundaries and entries:
        yield 0.90, "Stitching shared face boundaries..."
        arc_fits = {gid: e["fit"] for gid, e in enumerate(entries)
                    if e["fit"] is not None
                    and getattr(e["fit"], "kind", None) in ("cylinder", "cone", "torus")}
        curved_gids = {gid for gid, e in enumerate(entries)
                       if e["fit"] is not None
                       and getattr(e["fit"], "kind", None) != "plane"}
        graph = _boundary.BoundaryGraph(
            verts, faces, [e["group"] for e in entries], tolerance,
            arc_fits=arc_fits, curved_gids=curved_gids)
        st = graph.stats()
        stats.boundary_chains = st["chains"]
        stats.boundary_points_raw = st["raw_edges"]
        stats.boundary_points_shared = st["simplified_edges"]

    # ---- phase 3: write --------------------------------------------------
    face_ids: list[int] = []
    n_entries = max(len(entries), 1)
    for ei, entry in enumerate(entries):
        if ei % 100 == 0:
            yield (0.92 + 0.08 * (ei / n_entries),
                   f"Writing face {ei + 1}/{n_entries}...")
        fitted = entry["fit"]
        simplified = entry["simplified"]
        winding_signs = entry["winding"]
        entry_spans: dict[tuple[int, int], list[int]] | None = None

        if graph is not None:
            loops, ok = _fit.boundary_loops(faces, entry["group"])
            if not (ok and loops):
                stats.shared_loop_skipped_no_loops += 1
            if ok and loops:
                shared_pairs = [graph.simplify_loop(loop, with_spans=True) for loop in loops]
                shared = [sl for sl, _ in shared_pairs]
                entry_spans = {}
                for _, sp in shared_pairs:
                    entry_spans.update(sp)
                if not all(len(l) >= 3 for l in shared):
                    stats.shared_loop_skipped_short += 1
                if all(len(l) >= 3 for l in shared):
                    stats.shared_loop_faces += 1
                    # No `try_collapse_circle_loop` here, deliberately: collapsing
                    # a round hole in a plane face to two arc edges is a per-FACE
                    # rewrite that the face on the other side of the ring does
                    # not do, so the two stop sharing edges and the shell opens.
                    # The ring's straight edges are shared instead of
                    # duplicated, at the same file size.
                    simplified = [(l, None) for l in shared]
                    if fitted is not None and fitted.kind in ("cylinder", "cone", "torus"):
                        winding_signs = [
                            _fit.loop_winding_sign(l, verts, fitted)
                            for l, _ in simplified]
                    else:
                        winding_signs = [1] * len(simplified)
        if winding_signs is None:
            winding_signs = [1] * len(simplified)

        same_sense = True
        if fitted.kind == "plane":
            _orient_plane(fitted, verts, faces, entry["group"])
        elif fitted.kind in ("cylinder", "cone", "sphere", "torus"):
            same_sense = _face_sense(fitted, verts, faces, entry["group"])
        surface_id = _make_surface(w, fitted, fb.scale)

        # An untrimmed full cylinder, cone or torus lateral face has exactly two
        # closed rim loops, each a pure full-circle boundary (for a torus, the
        # common fillet running 360 degrees around a hole or boss). Written as
        # outer boundary plus hole they would not nest in (u, v), which breaks
        # the reader's re-triangulation, so they are merged into one seamed loop
        # (`_make_seamed_loop`), as real CAD STEP exports do. Trimmed patches
        # (straight cut edges, or any other loop count) are untouched.
        seamed = None
        if (SEAMED_LOOPS
                and fitted is not None and fitted.kind in ("cylinder", "cone", "torus")
                and len(simplified) == 2
                and all(_loop_is_full_circle(loop, segs, verts, fitted)
                        for loop, segs in simplified)):
            stats.shared_loop_seamed += 1
            seamed = _make_seamed_loop(fb, verts, fitted,
                                       simplified[0], simplified[1], tolerance,
                                       winding_signs=(winding_signs[0],
                                                      winding_signs[1]))

        bounds = []
        if seamed is not None:
            loop, segs, per_edge_winding = seamed
            loop_id = _make_loop(fb, w, loop, fitted, verts, segs=segs,
                                 winding_sign=per_edge_winding,
                                 spans=entry_spans, tolerance=tolerance)
            bounds.append(w.add(f"FACE_OUTER_BOUND('',#{loop_id},.T.)"))
        else:
            # Which loop is the outer perimeter (FACE_OUTER_BOUND) and which a
            # hole (FACE_BOUND) is decided by 3D polygon area, not by index:
            # `boundary_loops` and `BoundaryGraph.simplify_loop` return loops in
            # an order unrelated to nesting, and a hole written as the outer
            # bound makes an invalid face whose re-tessellation comes back as
            # slivers. The outer loop of a valid decomposition encloses more
            # area than any hole in it.
            order = sorted(range(len(simplified)),
                           key=lambda i: _loop_area3(simplified[i][0], verts),
                           reverse=True)
            for out_i, i in enumerate(order):
                loop, segs = simplified[i]
                loop_id = _make_loop(fb, w, loop, fitted, verts, segs=segs,
                                     winding_sign=winding_signs[i],
                                     spans=entry_spans, tolerance=tolerance)
                bound_kind = "FACE_OUTER_BOUND" if out_i == 0 else "FACE_BOUND"
                bounds.append(w.add(f"{bound_kind}('',#{loop_id},.T.)"))
        bounds_str = ",".join(f"#{b}" for b in bounds)
        sense_str = ".T." if same_sense else ".F."
        face_ids.append(w.add(f"ADVANCED_FACE('',({bounds_str}),#{surface_id},{sense_str})"))

        kind = entry["kind"]
        if kind == "source":
            stats.source_faces += 1
            kind = "primitive" if fitted.kind != "bspline" else "bspline"
        if kind == "primitive":
            if fitted.kind == "plane":
                stats.plane_faces += 1
            elif fitted.kind == "cylinder":
                stats.cylinder_faces += 1
            elif fitted.kind == "cone":
                stats.cone_faces += 1
            elif fitted.kind == "sphere":
                stats.sphere_faces += 1
            elif fitted.kind == "torus":
                stats.torus_faces += 1
        elif kind == "bspline":
            stats.bspline_faces += 1
            stats.bspline_triangle_faces += len(entry["group"])
        else:
            stats.leftover_faces += 1
            stats.leftover_triangle_faces += len(entry["group"])

    if graph is not None:
        # Read after every simplify_loop() call above -- `fallbacks` only
        # accumulates as faces actually use the graph during writing.
        stats.graph_collisions = graph.collisions
        stats.graph_protected = graph.protected
        stats.graph_fallbacks = graph.fallbacks
    stats.boundary_curve_edges = fb.bspline_curve_count

    return face_ids


def export_step_brep(solids, path: str, *args, progress=None, **kwargs):
    """Write `solids` to `path` as a real ADVANCED_BREP (see
    `iter_export_step_brep` for the parameters) and return the BrepStats.

    `progress`, if given, is called as `progress(frac_0_1, message)`."""
    gen = iter_export_step_brep(solids, path, *args, **kwargs)
    try:
        while True:
            frac, msg = next(gen)
            if progress is not None:
                progress(frac, msg)
    except StopIteration as done:
        return done.value


def iter_export_step_brep(solids, path: str, tolerance_mm: float = 0.1,
                           smooth_angle_deg: float = 32.0,
                           author: str = "StepForge", scale: float = 1000.0,
                           line_angle_tol_deg: float = 3.0,
                           freeform: bool = True, shared_boundaries: bool = True,
                           freeform_tolerance: float | None = None,
                           merge_patches: bool = True,
                           keep_source_surfaces: bool = True,
                           tolerance_pct: float | None = None,
                           freeform_tolerance_pct: float | None = None,
                           parallel: bool = False,
                           max_workers: int | None = None):
    """Generator: write `solids` to `path` as a real ADVANCED_BREP with
    analytic planes, cylinders, cones, spheres and tori reconstructed from the
    mesh wherever they fit within `tolerance_mm` (in the output file's units,
    mm when scale=1000). Returns a BrepStats to report to the user.

    Yields `(frac_0_1, message)` between units of work; each solid gets a slice
    of the bar sized by its share of the total triangle count. A Blender
    operator runs it a few milliseconds at a time, so the UI stays responsive.
    `parallel=True` moves the patch fitting to worker processes (see
    `iter_build_brep_faces`).

    `line_angle_tol_deg` controls how aggressively adjacent straight boundary
    edges are merged (see `fit.merge_collinear`).

    `freeform` (default on) enables the B-spline surface tier for regions no
    rigid primitive fits (`freeform.py`). Off, those regions go straight to flat
    re-merging and per-triangle faces: only useful for A/B comparison or if a
    downstream tool cannot read B_SPLINE_SURFACE_WITH_KNOTS.

    `keep_source_surfaces` (default on): a solid whose mesh carries `face_ids`
    and `face_surfaces` from a StepForge import writes each unedited face back
    with its original surface (see `iter_build_brep_faces`).

    `tolerance_pct` / `freeform_tolerance_pct`, if given, replace
    `tolerance_mm` / `freeform_tolerance` with that percentage of each solid's
    own bounding-box longest edge, so a small part in an assembly is fitted to
    a tolerance that suits it and not the whole assembly.
    """
    w = _Writer()

    app_ctx = w.add("APPLICATION_CONTEXT('automotive design')")
    w.add("APPLICATION_PROTOCOL_DEFINITION('international standard',"
          f"'automotive_design',2010,#{app_ctx})")
    luni = w.add("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))")
    auni = w.add("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
    suni = w.add("(NAMED_UNIT(*)SOLID_ANGLE_UNIT()SI_UNIT($,.STERADIAN.))")
    unc = w.add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-6),#{luni},"
                "'distance_accuracy_value','confusion accuracy')")
    geo_ctx = w.add(
        "(GEOMETRIC_REPRESENTATION_CONTEXT(3)"
        f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc}))"
        f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{luni},#{auni},#{suni}))"
        "REPRESENTATION_CONTEXT('Context','3D'))")
    mpc = w.add(f"PRODUCT_CONTEXT('',#{app_ctx},'mechanical')")
    pdc = w.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app_ctx},'design')")

    total_stats = BrepStats()

    usable = [s for s in solids if s.mesh.verts and s.mesh.faces]
    total_tris = sum(len(s.mesh.faces) for s in usable) or 1
    done_tris = 0

    for si, solid in enumerate(usable):
        raw_verts = solid.mesh.verts
        raw_faces = solid.mesh.faces
        name = step_str(solid.name or "Part")

        # fit in *output* (mm) units so `tolerance_mm` means what it says
        verts_mm = np.asarray(raw_verts, dtype=float) * scale
        edge_mm = float((verts_mm.max(axis=0) - verts_mm.min(axis=0)).max())
        tolerance = (edge_mm * tolerance_pct / 100.0
                     if tolerance_pct is not None and edge_mm > 0 else tolerance_mm)
        ff_tolerance = (edge_mm * freeform_tolerance_pct / 100.0
                        if freeform_tolerance_pct is not None and edge_mm > 0
                        else freeform_tolerance)

        prod = w.add(f"PRODUCT('{name}','{name}','',(#{mpc}))")
        pdf = w.add(f"PRODUCT_DEFINITION_FORMATION('','',#{prod})")
        pd = w.add(f"PRODUCT_DEFINITION('design','',#{pdf},#{pdc})")
        pds = w.add(f"PRODUCT_DEFINITION_SHAPE('','',#{pd})")

        fb = _FaceBuilder(w, verts_mm, scale=1.0)  # already in mm
        stats = BrepStats()
        solid_span = len(raw_faces) / total_tris

        src = getattr(solid, "face_surfaces", None) or {}
        source_fits = {}
        if keep_source_surfaces:
            for fid, rec in src.items():
                fo = source_fit(rec, scale)
                if fo is not None:
                    source_fits[int(fid)] = fo
        face_gen = iter_build_brep_faces(
            w, fb, verts_mm, raw_faces, tolerance, smooth_angle_deg, stats,
            line_angle_tol_deg=line_angle_tol_deg, freeform=freeform,
            shared_boundaries=shared_boundaries,
            freeform_tolerance=ff_tolerance, merge_patches=merge_patches,
            tri_face_ids=getattr(solid.mesh, "face_ids", None),
            source_fits=source_fits, parallel=parallel,
            max_workers=max_workers)
        base = done_tris / total_tris
        while True:
            try:
                local_frac, msg = next(face_gen)
            except StopIteration as done:
                face_ids = done.value
                break
            yield base + solid_span * local_frac, f"{name}: {msg}"
        done_tris += len(raw_faces)
        for attr in STAT_FIELDS:
            setattr(total_stats, attr, getattr(total_stats, attr) + getattr(stats, attr))
        for attr in ("boundary_chains", "boundary_points_raw",
                     "boundary_points_shared", "shared_loop_faces",
                     "shared_loop_skipped_no_loops",
                     "shared_loop_skipped_short", "shared_loop_seamed",
                     "graph_collisions", "graph_protected", "graph_fallbacks",
                     "boundary_curve_edges", "patches_merged"):
            setattr(total_stats, attr,
                    getattr(total_stats, attr) + getattr(stats, attr))
        if stats.freeform_achievable_mm is not None:
            prev = total_stats.freeform_achievable_mm
            total_stats.freeform_achievable_mm = (
                stats.freeform_achievable_mm if prev is None
                else max(prev, stats.freeform_achievable_mm))

        if not face_ids:
            continue
        faces_str = ",".join(f"#{f}" for f in face_ids)
        shell = w.add(f"CLOSED_SHELL('',({faces_str}))")
        brep = w.add(f"MANIFOLD_SOLID_BREP('{name}',#{shell})")
        rep = w.add(f"ADVANCED_BREP_SHAPE_REPRESENTATION('{name}',(#{brep}),#{geo_ctx})")
        w.add(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})")

    yield 1.0, "Writing file..."

    body = "\n".join(w.lines)
    text = (
        step_header("StepForge analytic B-rep export", path, author) +
        "DATA;\n"
        f"{body}\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return total_stats
