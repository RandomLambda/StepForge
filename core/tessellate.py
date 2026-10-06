"""Build geometry objects from STEP entities and tessellate faces into
triangles. Pure Python + NumPy.
"""
from __future__ import annotations

import math
from collections import deque
from itertools import pairwise

import numpy as np

from . import delaunay as _delaunay
from . import earcut
from . import geometry as G
from .geometry import (
    TWO_PI,
    BSplineCurve,
    BSplineSurface,
    Circle,
    Cone,
    Cylinder,
    Ellipse,
    Frame,
    Line,
    Plane,
    Sphere,
    Torus,
)

# ---------------------------------------------------------------------------
# Builders: STEP instance -> geometry object
# ---------------------------------------------------------------------------

# A SURFACE_CURVE and its subtypes carry the edge's real 3D geometry in their
# first attribute (`curve_3d`); the rest are parameter-space copies (PCURVEs)
# that a 3D tessellator does not need.
_SURFACE_CURVE_TYPES = ("SURFACE_CURVE", "SEAM_CURVE", "INTERSECTION_CURVE",
                        "BOUNDED_SURFACE_CURVE")

# Curve / surface entities that could not be built, per process:
# (kind, entity name) -> set of entity ids. See take_unsupported().
_UNSUPPORTED = {}


def _entity_label(inst):
    return inst.name or "(" + " ".join(r.name for r in (inst.records or [])) + ")"


def _note_unsupported(kind, inst):
    _UNSUPPORTED.setdefault((kind, _entity_label(inst)), set()).add(int(inst.id))


def take_unsupported():
    """Entities met since the last call that could not be turned into a
    curve or surface, as {(kind, entity name): set of entity ids}; clears
    the record. A curve that could not be built is drawn as a straight
    chord, a surface that could not be built drops its face."""
    out = {k: set(v) for k, v in _UNSUPPORTED.items()}
    _UNSUPPORTED.clear()
    return out


def build_curve(sf, ref, _depth=0):
    """Geometry object for a curve entity, or None (recorded, see
    take_unsupported)."""
    return build_curve_sense(sf, ref, _depth)[0]


def build_curve_sense(sf, ref, _depth=0):
    """(curve, sense): `sense` is False when the returned geometry runs
    opposite to the referenced entity (a TRIMMED_CURVE whose
    sense_agreement is .F.), so an edge's same_sense must be flipped."""
    inst = sf.get(ref)
    if inst is None or _depth > 8:
        return None, True
    name = inst.name
    p = inst.params
    if name in _SURFACE_CURVE_TYPES and len(p) > 1:
        return build_curve_sense(sf, p[1], _depth + 1)
    if name == "TRIMMED_CURVE" and len(p) > 4:
        # The edge's own vertices do the trimming; only the direction of
        # the trimmed curve relative to its basis matters here.
        curve, sense = build_curve_sense(sf, p[1], _depth + 1)
        return curve, (sense == _bool(p[4]))
    if name == "" and inst.records:
        for r in inst.records:
            if r.name in _SURFACE_CURVE_TYPES and r.params:
                return build_curve_sense(sf, r.params[0], _depth + 1)
    curve = None
    try:
        curve = _build_curve(sf, inst, name, p)
    except Exception:  # noqa: BLE001 - a malformed entity counts as unsupported
        curve = None
    if curve is None:
        _note_unsupported("curve", inst)
    return curve, True


_BSPLINE_CURVE_NAMES = ("B_SPLINE_CURVE_WITH_KNOTS", "B_SPLINE_CURVE",
                        "RATIONAL_B_SPLINE_CURVE", "BOUNDED_CURVE",
                        "BEZIER_CURVE", "UNIFORM_CURVE", "QUASI_UNIFORM_CURVE")


def _build_curve(sf, inst, name, p):
    if name == "LINE":
        pnt = G.get_point(sf, p[1])
        vinst = sf.get(p[2])  # VECTOR(name, dir, mag)
        d = G.get_dir(sf, vinst.params[1]) * float(vinst.params[2])
        return Line(pnt, d)
    if name == "CIRCLE":
        f = Frame.from_axis2(sf, p[1])
        return Circle(f, float(p[2]))
    if name == "ELLIPSE":
        f = Frame.from_axis2(sf, p[1])
        return Ellipse(f, float(p[2]), float(p[3]))
    if name == "POLYLINE":
        pts = [G.get_point(sf, r) for r in p[1]]
        pts = [q for q in pts if q is not None]
        return G.Polyline(pts) if len(pts) >= 2 else None
    if name in _BSPLINE_CURVE_NAMES or name == "":
        return _build_bspline_curve(sf, inst)
    return None


def _implicit_knots(kind, n_ctrl, deg):
    """Knots and multiplicities of the knot-less B-spline subtypes
    (ISO 10303-42): UNIFORM (all simple, evenly spaced), QUASI_UNIFORM
    (clamped ends, evenly spaced inside) and BEZIER (clamped, one Bezier
    segment per `deg` control points)."""
    if kind == "UNIFORM":
        n = n_ctrl + deg + 1
        return list(range(n)), [1] * n
    if kind == "BEZIER" and deg > 0 and (n_ctrl - 1) % deg == 0:
        n_seg = (n_ctrl - 1) // deg
        mults = [deg + 1] + [deg] * (n_seg - 1) + [deg + 1]
        return list(range(n_seg + 1)), mults
    n_int = max(n_ctrl - deg - 1, 0)
    return list(range(n_int + 2)), [deg + 1] + [1] * n_int + [deg + 1]


def _knot_kind(names):
    for n in names:
        if n.startswith("BEZIER_"):
            return "BEZIER"
        if n.startswith("QUASI_UNIFORM_"):
            return "QUASI_UNIFORM"
        if n.startswith("UNIFORM_"):
            return "UNIFORM"
    return "QUASI_UNIFORM"


def _build_bspline_curve(sf, inst):
    """Handle both a simple B_SPLINE_CURVE_WITH_KNOTS (carries a leading name and
    the full parameter set) and a complex/rational entity (where the data is
    split across B_SPLINE_CURVE / B_SPLINE_CURVE_WITH_KNOTS / RATIONAL subrecords
    that have NO name), plus the knot-less subtypes (BEZIER_CURVE,
    UNIFORM_CURVE, QUASI_UNIFORM_CURVE) in either form."""
    deg = cps = knots = mults = weights = None
    if inst.records:
        names = [r.name for r in inst.records]
        for r in inst.records:
            if r.name == "B_SPLINE_CURVE":
                deg = int(r.params[0])
                cps = [G.get_point(sf, c) for c in r.params[1]]
            elif r.name == "B_SPLINE_CURVE_WITH_KNOTS":
                mults = [int(m) for m in r.params[0]]
                knots = [float(k) for k in r.params[1]]
            elif r.name == "RATIONAL_B_SPLINE_CURVE":
                weights = [float(w) for w in r.params[0]]
    else:
        names = [inst.name]
        p = inst.params  # 0=name,1=deg,2=ctrlpts,3=form,4=closed,5=selfint,6=mult,7=knots
        deg = int(p[1])
        cps = [G.get_point(sf, c) for c in p[2]]
        if inst.name == "B_SPLINE_CURVE_WITH_KNOTS":
            mults = [int(m) for m in p[6]]
            knots = [float(k) for k in p[7]]
    if deg is None or cps is None:
        return None
    if knots is None:
        knots, mults = _implicit_knots(_knot_kind(names), len(cps), deg)
    U = G._expand_knots(knots, mults)
    return BSplineCurve(deg, cps, U, weights)


def build_surface(sf, ref):
    inst = sf.get(ref)
    if inst is None:
        return None
    surface = None
    try:
        surface = _build_surface(sf, inst, inst.name, inst.params)
    except Exception:  # noqa: BLE001 - a malformed entity counts as unsupported
        surface = None
    if surface is None:
        _note_unsupported("surface", inst)
    return surface


_BSPLINE_SURFACE_NAMES = ("B_SPLINE_SURFACE_WITH_KNOTS", "B_SPLINE_SURFACE",
                          "BEZIER_SURFACE", "UNIFORM_SURFACE", "QUASI_UNIFORM_SURFACE")


def _angle(sf, x):
    """A plane-angle value in radians (files may use degrees)."""
    return float(x) * getattr(sf, "angle_scale", 1.0)


def _build_surface(sf, inst, name, p):
    if name == "PLANE":
        return Plane(Frame.from_axis2(sf, p[1]))
    if name == "CYLINDRICAL_SURFACE":
        return Cylinder(Frame.from_axis2(sf, p[1]), float(p[2]))
    if name == "CONICAL_SURFACE":
        return Cone(Frame.from_axis2(sf, p[1]), float(p[2]), _angle(sf, p[3]))
    if name == "SPHERICAL_SURFACE":
        return Sphere(Frame.from_axis2(sf, p[1]), float(p[2]))
    if name == "TOROIDAL_SURFACE":
        return Torus(Frame.from_axis2(sf, p[1]), float(p[2]), float(p[3]))
    if name == "SURFACE_OF_LINEAR_EXTRUSION":
        curve = build_curve(sf, p[1])
        vinst = sf.get(p[2])  # VECTOR(name, dir, magnitude)
        if curve is None or vinst is None:
            return None
        d = G.get_dir(sf, vinst.params[1]) * float(vinst.params[2])
        return G.SurfaceOfLinearExtrusion(curve, d)
    if name == "SURFACE_OF_REVOLUTION":
        curve = build_curve(sf, p[1])
        ax = sf.get(p[2])  # AXIS1_PLACEMENT(name, location, axis)
        if curve is None or ax is None:
            return None
        o = G.get_point(sf, ax.params[1])
        a = (G.get_dir(sf, ax.params[2]) if len(ax.params) > 2 and ax.params[2] is not None
             else np.array([0.0, 0.0, 1.0]))
        return G.SurfaceOfRevolution(curve, o, a)
    if name == "OFFSET_SURFACE":
        basis = build_surface(sf, p[1])
        return G.make_offset_surface(basis, float(p[2])) if basis is not None else None
    if name == "RECTANGULAR_TRIMMED_SURFACE":
        basis = build_surface(sf, p[1])
        if basis is None:
            return None
        u1, u2, v1, v2 = (float(x) for x in p[2:6])
        if basis.periodic_u:
            u1, u2 = _angle(sf, u1), _angle(sf, u2)
        if isinstance(basis, (Sphere, Torus)):
            v1, v2 = _angle(sf, v1), _angle(sf, v2)
        surf = G.trimmed_domain(basis, u1, u2, v1, v2)
        if len(p) > 7 and _bool(p[6]) != _bool(p[7]):
            surf.sense_flip = not surf.sense_flip
        return surf
    if name in _BSPLINE_SURFACE_NAMES or name == "":
        return _build_bspline_surface(sf, inst)
    return None


def _build_bspline_surface(sf, inst):
    deg_u = deg_v = grid_refs = None
    uk = vk = um = vm = weights = None
    if inst.records:
        names = [r.name for r in inst.records]
        for r in inst.records:
            if r.name == "B_SPLINE_SURFACE":
                deg_u = int(r.params[0])
                deg_v = int(r.params[1])
                grid_refs = r.params[2]
            elif r.name == "B_SPLINE_SURFACE_WITH_KNOTS":
                um = [int(m) for m in r.params[0]]
                vm = [int(m) for m in r.params[1]]
                uk = [float(k) for k in r.params[2]]
                vk = [float(k) for k in r.params[3]]
            elif r.name == "RATIONAL_B_SPLINE_SURFACE":
                weights = [[float(w) for w in row] for row in r.params[0]]
    else:
        names = [inst.name]
        p = inst.params  # 0=name,1=u_deg,2=v_deg,3=grid,...,8=umult,9=vmult,10=uknots,11=vknots
        deg_u = int(p[1])
        deg_v = int(p[2])
        grid_refs = p[3]
        if inst.name == "B_SPLINE_SURFACE_WITH_KNOTS":
            um = [int(m) for m in p[8]]
            vm = [int(m) for m in p[9]]
            uk = [float(k) for k in p[10]]
            vk = [float(k) for k in p[11]]
    if deg_u is None or grid_refs is None:
        return None
    nu = len(grid_refs)
    nv = len(grid_refs[0])
    grid = np.zeros((nu, nv, 3))
    for i in range(nu):
        for j in range(nv):
            grid[i, j] = G.get_point(sf, grid_refs[i][j])
    if uk is None:
        kind = _knot_kind(names)
        uk, um = _implicit_knots(kind, nu, deg_u)
        vk, vm = _implicit_knots(kind, nv, deg_v)
    U = G._expand_knots(uk, um)
    V = G._expand_knots(vk, vm)
    w = np.asarray(weights) if weights is not None else None
    return BSplineSurface(deg_u, deg_v, grid, U, V, w)


# ---------------------------------------------------------------------------
# Edge / loop sampling
# ---------------------------------------------------------------------------

def _sample_edge_inclusive(curve, v0, v1, deflection, increasing=True, max_edge=None):
    """Return 3D points from v0 to v1 along curve, INCLUDING both v0 and v1.

    The canonical, direction-fixed sampler: always called with v0/v1 in one
    reference edge's own (vertex_start, vertex_end) order, cached once, and then
    used forward or reversed by whichever faces traverse the edge. Two faces
    re-deriving points from opposite ends would differ by up to a whole sample
    step, a real gap between them. One inclusive sampling per edge, reversed as
    a list for the opposite face, makes both sides bit-identical.
    """
    if curve is None:
        return [v0, v1]
    if isinstance(curve, Line):
        # Straight edges are subsampled by `max_edge` alone, while curved edges
        # use `deflection`, which is usually several times finer. A face bounded
        # by both (a narrow cylindrical bend strip with two short curved sides
        # and two long straight ones) then has a density cliff along the
        # straight sides and a comb of thin triangles there. Dividing the target
        # by 3 closes most of it, but only for edges already short relative to
        # `max_edge` (4 segments or fewer: the profile of a bend or fillet
        # transition edge), so long panel edges stay coarse. It depends only on
        # this edge's own length and `max_edge`, so both faces sharing the edge
        # still resolve it to the same cached point list.
        if max_edge:
            length = float(np.linalg.norm(v1 - v0))
            n_coarse = max(1, math.ceil(length / max_edge))
            line_target = (max_edge / 3.0) if n_coarse <= 4 else max_edge
            n = max(1, min(math.ceil(length / line_target), 200))
        else:
            n = 1
        return [v0 + (v1 - v0) * (k / n) for k in range(n + 1)]
    if isinstance(curve, (Circle, Ellipse)):
        t0 = curve.param_of(v0)
        t1 = curve.param_of(v1)
        full = np.linalg.norm(v0 - v1) < 1e-7
        if increasing:
            sweep = TWO_PI if full else (t1 - t0)
            while sweep <= 1e-9:
                sweep += TWO_PI
            while sweep > TWO_PI + 1e-9:
                sweep -= TWO_PI
        else:
            sweep = -TWO_PI if full else (t1 - t0)
            while sweep >= -1e-9:
                sweep -= TWO_PI
            while sweep < -TWO_PI - 1e-9:
                sweep += TWO_PI
        r = getattr(curve, "r", None) or max(getattr(curve, "ra", 1.0), getattr(curve, "rb", 1.0))
        ratio = max(-1.0, min(1.0, 1.0 - deflection / max(r, 1e-6)))
        dtheta = min(max(2.0 * math.acos(ratio), 0.08), G.MAX_SEGMENT_ANGLE)
        n = max(2, math.ceil(abs(sweep) / dtheta))
        n = min(n, 96)
        pts = [curve.eval(t0 + sweep * k / n) for k in range(n)]
        pts.append(v1 if not full else pts[0])
        return pts
    if isinstance(curve, G.Polyline):
        return _sample_polyline(curve, v0, v1, increasing, max_edge)
    if curve.domain is not None:
        return _sample_param_curve(curve, v0, v1, deflection, increasing)
    return [v0, v1]


def _param_sweep(curve, v0, v1, increasing):
    """(t0, sweep) for the edge v0 -> v1 on a bounded parametric curve. On a
    closed curve the edge may run across the curve's start, and an edge
    whose ends coincide is the whole curve; `increasing` (the edge's
    same_sense) says which way round."""
    lo, hi = curve.domain
    t0 = curve.param_of(v0)
    t1 = curve.param_of(v1)
    if not curve.closed:
        return t0, t1 - t0
    span = hi - lo
    if np.linalg.norm(v0 - v1) <= 1e-9 * max(1.0, float(np.linalg.norm(v0))):
        return t0, (span if increasing else -span)
    sweep = t1 - t0
    if increasing and sweep <= 0:
        sweep += span
    elif not increasing and sweep >= 0:
        sweep -= span
    return t0, sweep


def _sample_param_curve(curve, v0, v1, deflection, increasing, max_pts=400):
    """Points v0 .. v1 along a parametric curve (B-spline, ...): evenly in
    parameter first, then any piece whose midpoint strays more than
    `deflection` from its chord is halved until none does. The ends are the
    vertices themselves, so the edge meets its neighbours exactly."""
    t0, sweep = _param_sweep(curve, v0, v1, increasing)
    ncp = getattr(curve, "n", 7) + 1
    n = max(6, min(ncp * 3, 48))
    ts = [t0 + sweep * k / n for k in range(n + 1)]
    pts = [curve.eval_wrapped(t) for t in ts]
    pts[0], pts[-1] = v0, v1
    for _ in range(6):
        new_ts, new_pts = [ts[0]], [pts[0]]
        split = False
        for k in range(len(ts) - 1):
            tm = 0.5 * (ts[k] + ts[k + 1])
            pm = curve.eval_wrapped(tm)
            if (len(ts) + len(new_ts) < max_pts and
                    np.linalg.norm(pm - 0.5 * (pts[k] + pts[k + 1])) > deflection):
                new_ts.append(tm)
                new_pts.append(pm)
                split = True
            new_ts.append(ts[k + 1])
            new_pts.append(pts[k + 1])
        ts, pts = new_ts, new_pts
        if not split:
            break
    return pts


def _sample_polyline(curve, v0, v1, increasing, max_edge):
    """The polyline's own corner points between the two vertices, each
    straight piece subdivided the way a LINE edge is."""
    t0, sweep = _param_sweep(curve, v0, v1, increasing)
    t1 = t0 + sweep
    step = 1 if sweep >= 0 else -1
    corners = []
    k = math.floor(t0) + 1 if step > 0 else math.ceil(t0) - 1
    while (k < t1 - 1e-9) if step > 0 else (k > t1 + 1e-9):
        corners.append(curve.eval_wrapped(float(k)))
        k += step
    pts = [v0] + corners + [v1]
    out = [pts[0]]
    for a, b in pairwise(pts):
        out.extend(_sample_edge_inclusive(Line(a, b - a), a, b, 0.0, True, max_edge)[1:])
    return out


def _walk_loop(sf, loop_ref, deflection, max_edge=None, edge_cache=None,
              edge_hints=None):
    """Return ordered list of 3D points for an EDGE_LOOP.

    `edge_cache` (shared across a whole file's conversion, keyed by EDGE_CURVE
    reference) makes whichever of the two faces bordering an edge asks first
    compute the canonical point list and the other reuse it (reversed if
    needed); see `_sample_edge_inclusive`.

    `edge_hints` (also file-wide, EDGE_CURVE reference -> metric target sample
    spacing) lets a face whose curvature-driven interior resolution is finer
    than `max_edge` (a small-radius or narrow-angle cylinder or cone, see
    `convert.py`'s pre-pass) demand that its own straight boundary edges be
    sampled at least that densely. Otherwise a boundary edge can be far coarser
    than the interior grid and Delaunay connects a cluster of interior points
    to one distant boundary vertex: long sliver triangles along the edge. It is
    computed once per file before any edge is sampled, so it applies whichever
    of the two faces samples the edge first.
    """
    if edge_cache is None:
        edge_cache = {}
    loop = sf.get(loop_ref)
    if loop is None or loop.name == "VERTEX_LOOP":
        return []
    if loop.name == "POLY_LOOP":
        pts = [G.get_point(sf, r) for r in loop.params[-1]]
        pts = [p for p in pts if p is not None]
        if len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) < 1e-12:
            pts.pop()
        return pts
    pts = []
    for oe_ref in loop.params[-1]:  # EDGE_LOOP(name, (oriented_edges))
        oe = sf.get(oe_ref)
        if oe is None or oe.name != "ORIENTED_EDGE":
            continue
        edge_ref = oe.params[3]
        orient = oe.params[4]
        edge = sf.get(edge_ref)
        if edge is None or edge.name != "EDGE_CURVE":
            continue
        ec_same = _bool(edge.params[4])   # EDGE_CURVE.same_sense
        o = _bool(orient)                  # ORIENTED_EDGE.orientation
        full = edge_cache.get(edge_ref)
        if full is None:
            vs = _vertex_point(sf, edge.params[1])
            ve = _vertex_point(sf, edge.params[2])
            curve, sense = build_curve_sense(sf, edge.params[3])
            if not sense:
                ec_same = not ec_same
            eff_max_edge = max_edge
            hint = edge_hints.get(edge_ref) if edge_hints else None
            if hint == math.inf:
                eff_max_edge = None      # only flat faces on either side
            elif hint is not None:
                eff_max_edge = min(eff_max_edge, hint) if eff_max_edge else hint
            full = _sample_edge_inclusive(curve, vs, ve, deflection, ec_same, eff_max_edge)
            edge_cache[edge_ref] = full
        # ORIENTED_EDGE.orientation True means this face travels the edge in
        # its own (vertex_start -> vertex_end) direction, i.e. forward through
        # the cached list; False means backward. The endpoint this segment must
        # not include (it arrives via the next edge) is sliced off the shared
        # list, never a recomputed one.
        seg = full[:-1] if o else list(reversed(full))[:-1]
        pts.extend(seg)
    return pts


def _vertex_point(sf, ref):
    vp = sf.get(ref)
    if vp is None:
        return np.zeros(3)
    # VERTEX_POINT(name, point)
    return G.get_point(sf, vp.params[1])


def _bool(enum):
    return str(enum).upper().startswith("T")


# ---------------------------------------------------------------------------
# Boundary loops -> parameter-space polygons
# ---------------------------------------------------------------------------
#
# A face's boundary loops are walked in 3D and inverted onto the surface. On
# a periodic or singular surface that inversion is not a polygon yet:
#
#   * At a pole (sphere pole, cone apex, a B-spline row collapsed to a
#     point) the parameter along the collapsed direction is undefined, so
#     the single 3D point stands for a whole segment of parameter space.
#   * A loop can wind once around a periodic direction (the rim circle of a
#     cylinder written without a seam edge, a sphere cap's circle). Its
#     parameter image is then an open curve, not a closed polygon, and the
#     face region only closes up via a second such loop, a pole, or the
#     surface's own natural bounds.
#
# `_loop_uv` resolves the first case and measures the second; the
# `close_*` helpers turn winding loops into one closed polygon by adding a
# cut (a seam) whose two sides share the same 3D samples, so the mesh welds
# shut across it.

_SINGULAR_RATIO = 1e-5


def _singular_axis(surface, u, v, hu, hv):
    """'u' if a small step in u does not move the point (a pole, where a
    whole u-isoline collapses to one point), 'v' likewise, else None."""
    p = surface.eval(u, v)
    a = (np.linalg.norm(surface.eval(u + hu, v) - p) +
         np.linalg.norm(surface.eval(u - hu, v) - p))
    b = (np.linalg.norm(surface.eval(u, v + hv) - p) +
         np.linalg.norm(surface.eval(u, v - hv) - p))
    if a <= _SINGULAR_RATIO * b:
        return "u"
    if b <= _SINGULAR_RATIO * a:
        return "v"
    return None


def _nearest_copy(x, ref, period):
    """`x` shifted by whole periods to lie closest to `ref`."""
    if not period:
        return x
    return x + round((ref - x) / period) * period


def _loop_uv(surface, pts3d):
    """Parameter-space image of one boundary loop.

    Returns `(uv, xyz, net_u, net_v)`: the polygon (poles expanded into a
    segment along the collapsed direction), the matching 3D points, and how
    far the loop winds around each periodic direction (0 for an ordinary
    closed loop, +/- one period for a loop around a cylinder)."""
    n = len(pts3d)
    raw = [surface.invert(p) for p in pts3d]
    Pu = surface.period_u if surface.periodic_u else None
    Pv = surface.period_v if surface.periodic_v else None

    sing = [None] * n
    if getattr(surface, "may_have_poles", False) and n:
        us = [r[0] for r in raw]
        vs = [r[1] for r in raw]
        ru = max(us) - min(us)
        rv = max(vs) - min(vs)
        # A rim circle (constant v on a cone or sphere, written without a seam)
        # has no extent in v to scale its v step by: with the floor alone that
        # step is ~1e-7 beside the u step of 6e-4, and any rim of more than
        # ~18 mm radius looked like a pole in v. Borrow the other direction's
        # extent instead.
        if ru < 1e-6 * rv:
            ru = rv
        elif rv < 1e-6 * ru:
            rv = ru
        hu = 1e-4 * max(ru, 1e-3)
        hv = 1e-4 * max(rv, 1e-3)
        sing = [_singular_axis(surface, u, v, hu, hv) for (u, v) in raw]
    regular = [i for i in range(n) if sing[i] is None]
    if not regular:
        return None
    k0 = regular[0]
    order = list(range(k0, n)) + list(range(k0))

    # Walk the loop: continuous within runs of regular points, a jump of
    # still-undecided size across each pole.
    seq = []          # ("pt", i, u, v) or ("pole", [i, ...], axis)
    prev = None
    for i in order:
        if sing[i] is not None:
            if seq and seq[-1][0] == "pole":
                seq[-1][1].append(i)
            else:
                seq.append(["pole", [i], sing[i]])
            continue
        u, v = raw[i]
        if prev is not None:
            u = _nearest_copy(u, prev[0], Pu)
            v = _nearest_copy(v, prev[1], Pv)
        seq.append(["pt", i, u, v])
        prev = (u, v)

    first = seq[0]
    last_pt = next(e for e in reversed(seq) if e[0] == "pt")
    net_u = last_pt[2] + (_nearest_copy(first[2], last_pt[2], Pu) - last_pt[2]) - first[2]
    net_v = last_pt[3] + (_nearest_copy(first[3], last_pt[3], Pv) - last_pt[3]) - first[3]
    net_u = round(net_u / Pu) * Pu if Pu else 0.0
    net_v = round(net_v / Pv) * Pv if Pv else 0.0

    # A loop through a pole is contractible: the jump across its last pole
    # absorbs whatever winding the rest of the loop accumulated. (Taking the
    # "nearest" copy there would collapse a hemisphere bounded by its equator
    # and a seam into a zero-area polygon.)
    poles = [k for k, e in enumerate(seq) if e[0] == "pole"]
    if poles:
        last_pole = poles[-1]
        if last_pole != len(seq) - 1:
            for e in seq[last_pole + 1:]:
                e[2] -= net_u
                e[3] -= net_v
        net_u = net_v = 0.0

    uv_out, xyz_out = [], []
    for k, e in enumerate(seq):
        if e[0] == "pt":
            uv_out.append((e[2], e[3]))
            xyz_out.append(pts3d[e[1]])
            continue
        before = next((x for x in reversed(seq[:k]) if x[0] == "pt"), None)
        after = next((x for x in seq[k + 1:] if x[0] == "pt"), None)
        if after is None:                 # pole closes the loop
            after = ["pt", first[1], first[2], first[3]]
        pi = e[1][0]
        pu, pv = raw[pi]
        if e[2] == "u":
            a = (before[2], pv)
            b = (after[2], pv)
        else:
            a = (pu, before[3])
            b = (pu, after[3])
        uv_out.append(a)
        xyz_out.append(pts3d[pi])
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1e-12:
            uv_out.append(b)
            xyz_out.append(pts3d[pi])
    return uv_out, xyz_out, net_u, net_v


def _sample_uv_segment(surface, a, b, deflection, target_len, max_n=128, count=None):
    """Interior points (excluding both ends) of the straight parameter-space
    segment a->b, dense enough that the 3D polyline stays within
    `deflection` of the surface curve and no piece is longer than
    `target_len` (or, given `count`, at least that many equal pieces).
    Returns (uv_list, xyz_list)."""
    probe = 16
    pts = [surface.eval(a[0] + (b[0] - a[0]) * k / probe,
                        a[1] + (b[1] - a[1]) * k / probe) for k in range(probe + 1)]
    length = sum(float(np.linalg.norm(pts[k + 1] - pts[k])) for k in range(probe))
    m = max(1, math.ceil(length / max(target_len, 1e-9))) if target_len else 1
    if count:
        m = count
    while m < max_n:
        ok = True
        for k in range(m):
            t0, t1 = k / m, (k + 1) / m
            p0 = surface.eval(a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0)
            p1 = surface.eval(a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)
            tm = 0.5 * (t0 + t1)
            pm = surface.eval(a[0] + (b[0] - a[0]) * tm, a[1] + (b[1] - a[1]) * tm)
            if np.linalg.norm(pm - 0.5 * (p0 + p1)) > deflection:
                ok = False
                break
        if ok:
            break
        m *= 2
    m = min(m, max_n)
    uv = [(a[0] + (b[0] - a[0]) * k / m, a[1] + (b[1] - a[1]) * k / m) for k in range(1, m)]
    return uv, [surface.eval(u, v) for (u, v) in uv]


def _swap(uv):
    return [(v, u) for (u, v) in uv]


def _rotate_closed(uv, xyz, j, net):
    """Start a closed winding loop at index j, keeping it continuous: points
    that were before j move to the end, shifted by the loop's winding."""
    uv2 = uv[j:] + [(u + net, v) for (u, v) in uv[:j]]
    return uv2, xyz[j:] + xyz[:j]


def _capped_rows(surface, u0, u1, v0, v1, deflection):
    """Number of pieces for a seam cut along the axis of a cylinder or cone,
    or None. Where the axial cell count is capped (`G.MAX_AXIAL_CELLS`) the
    rows of the interior grid are further apart than the seam's own samples
    would be, and on a rod narrower than a row the Delaunay triangles then
    reach from one side of the seam to the other, through the rod. The seam
    gets the rows of the grid instead, at the same heights."""
    if not isinstance(surface, (Cylinder, Cone)):
        return None
    lo, hi = min(v0, v1), max(v0, v1)
    _, nv = surface.grid_cells(deflection, ((min(u0, u1), lo), (max(u0, u1), hi)))
    return nv if nv >= G.MAX_AXIAL_CELLS else None


def _close_band(surface, A, B, period, deflection, target_len):
    """Two loops winding around the periodic u direction in opposite senses
    (a cylinder's two rims) -> one polygon, cut open along a seam from A's
    first point to the nearest point of B."""
    uvA, xyzA, dA = A
    uvB, xyzB, dB = B
    if dA * dB > 0:                       # mislabeled orientation
        uvB, xyzB, dB = uvB[::-1], xyzB[::-1], -dB
    a0 = uvA[0]
    target = a0[0] + dA
    j = min(range(len(uvB)),
            key=lambda k: abs(uvB[k][0] - _nearest_copy(target, uvB[k][0], period)))
    uvB, xyzB = _rotate_closed(uvB, xyzB, j, dB)
    s = _nearest_copy(uvB[0][0], target, period) - uvB[0][0]
    uvB = [(u + s, v) for (u, v) in uvB]
    a_end = (a0[0] + dA, a0[1])
    b0 = uvB[0]
    b_end = (b0[0] + dB, b0[1])
    rows = _capped_rows(surface, a0[0], a0[0] + dA, a_end[1], b0[1], deflection)
    s_uv, s_xyz = _sample_uv_segment(surface, a_end, b0, deflection, target_len,
                                     count=rows)
    back_uv = [(u + dB, v) for (u, v) in reversed(s_uv)]
    uv = list(uvA) + [a_end] + s_uv + list(uvB) + [b_end] + back_uv
    xyz = list(xyzA) + [xyzA[0]] + s_xyz + list(xyzB) + [xyzB[0]] + list(reversed(s_xyz))
    return uv, xyz


def _close_to_pole(surface, A, v_pole, pole_xyz, deflection, target_len):
    """One loop winding around u plus a pole at v_pole (a sphere cap's
    circle and its pole, a cone's base circle and its apex) -> one polygon."""
    uvA, xyzA, dA = A
    a0 = uvA[0]
    a_end = (a0[0] + dA, a0[1])
    top_end = (a0[0] + dA, v_pole)
    top_start = (a0[0], v_pole)
    s_uv, s_xyz = _sample_uv_segment(surface, a_end, top_end, deflection, target_len)
    down_uv = [(u - dA, v) for (u, v) in reversed(s_uv)]
    uv = list(uvA) + [a_end] + s_uv + [top_end, top_start] + down_uv
    xyz = (list(xyzA) + [xyzA[0]] + s_xyz + [pole_xyz, pole_xyz] +
           list(reversed(s_xyz)))
    return uv, xyz


def _collapsed_row(surface, fixed_axis, value, lo, hi, tol):
    """True if the parameter line at `fixed_axis` = value (the other
    parameter running lo..hi) is a single 3D point."""
    pts = []
    for k in range(5):
        t = lo + (hi - lo) * k / 4
        pts.append(surface.eval(value, t) if fixed_axis == "u" else surface.eval(t, value))
    return max(float(np.linalg.norm(p - pts[0])) for p in pts) <= tol


def pole_candidates(surface, uv_extent_u, tol):
    """v values where the whole u-isoline collapses to a point, as
    (v, xyz) pairs."""
    out = []
    lo, hi = uv_extent_u
    if isinstance(surface, Sphere):
        for v in (-math.pi / 2, math.pi / 2):
            out.append((v, surface.eval(0.0, v)))
        return out
    if isinstance(surface, Cone):
        if abs(surface.tan) > 1e-12:
            v = -surface.r_ref / surface.tan
            out.append((v, surface.eval(0.0, v)))
        return out
    dom = surface.domain()
    if dom is None:
        return out
    (_, _), (v0, v1) = dom
    for v in (v0, v1):
        if v is not None and _collapsed_row(surface, "v", v, lo, hi, tol):
            out.append((v, surface.eval(lo, v)))
    return out


def close_winding_loops(surface, winding, vertex_poles, deflection, target_len, tol):
    """Turn the loops that wind around a periodic direction into one closed
    parameter-space polygon, or return None if the face cannot be closed
    (e.g. a single rim circle on a cylinder, which has no pole).

    `winding`: list of (uv, xyz, net_u, net_v, domain_left) per loop, where
    `domain_left` is +1 if the face lies to the left of the loop in (u, v)
    (STEP's loop orientation, combined with the face's own sense)."""
    around_u = [w for w in winding if w[2] != 0.0]
    around_v = [w for w in winding if w[2] == 0.0 and w[3] != 0.0]
    if around_u:
        loops = around_u
        swapped = False
        period = surface.period_u
    elif around_v:
        loops = [(_swap(w[0]), w[1], w[3], w[2], -w[4]) for w in around_v]
        swapped = True
        period = surface.period_v
    else:
        return None
    if len(loops) >= 2:
        A = (loops[0][0], loops[0][1], loops[0][2])
        B = (loops[1][0], loops[1][1], loops[1][2])
        uv, xyz = _close_band(surface, A, B, period, deflection, target_len)
    else:
        if swapped:
            return None
        uvA, xyzA, dA, _, left = loops[0]
        v_loop = sum(v for (_, v) in uvA) / len(uvA)
        cand = []
        for p in vertex_poles:
            _, v = surface.invert(p)
            cand.append((v, p))
        if not cand:
            lo = min(u for (u, _) in uvA)
            cand = pole_candidates(surface, (lo, lo + abs(dA)), tol)
        if not cand:
            return None
        # The face lies left of the loop: going +u with the face on the
        # left means the face is at larger v.
        side = (1 if dA > 0 else -1) * left
        pick = [c for c in cand if (c[0] - v_loop) * side > 0] or cand
        v_pole, pxyz = min(pick, key=lambda c: abs(c[0] - v_loop))
        uv, xyz = _close_to_pole(surface, (uvA, xyzA, dA), v_pole, pxyz,
                                 deflection, target_len)
    if swapped:
        uv = _swap(uv)
    return uv, xyz


def natural_bounds(surface, deflection, target_len, tol):
    """Boundary of a face that uses its surface's whole parameter domain (a
    full sphere or torus written with no edge loop, only a VERTEX_LOOP or
    nothing). Opposite sides that coincide in 3D (seams) share their
    samples, and a side collapsed to a point (a pole) is kept as just its
    first corner, so the mesh closes up."""
    dom = surface.domain()
    if dom is None:
        return None
    (u0, u1), (v0, v1) = dom
    corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]

    def side(k):
        a, b = corners[k], corners[(k + 1) % 4]
        uv, xyz = _sample_uv_segment(surface, a, b, deflection, target_len)
        return [a] + uv, [surface.eval(*a)] + xyz

    def coincide(pairs):
        return all(float(np.linalg.norm(surface.eval(*p) - surface.eval(*q))) <= tol
                   for p, q in pairs)

    ts = [k / 6 for k in range(7)]
    seam_u = coincide([((u0, v0 + (v1 - v0) * t), (u1, v0 + (v1 - v0) * t)) for t in ts])
    seam_v = coincide([((u0 + (u1 - u0) * t, v0), (u0 + (u1 - u0) * t, v1)) for t in ts])

    sides = [side(0), side(1), None, None]
    if seam_v:
        full_uv = sides[0][0] + [corners[1]]
        full_xyz = sides[0][1] + [surface.eval(*corners[1])]
        sides[2] = ([(u, v1) for (u, _) in reversed(full_uv)][:-1],
                    list(reversed(full_xyz))[:-1])
    else:
        sides[2] = side(2)
    if seam_u:
        full_uv = sides[1][0] + [corners[2]]
        full_xyz = sides[1][1] + [surface.eval(*corners[2])]
        sides[3] = ([(u0, v) for (_, v) in reversed(full_uv)][:-1],
                    list(reversed(full_xyz))[:-1])
    else:
        sides[3] = side(3)
    for k in range(4):
        uv, xyz = sides[k]
        end = surface.eval(*corners[(k + 1) % 4])
        if max(float(np.linalg.norm(p - xyz[0])) for p in xyz + [end]) <= tol:
            sides[k] = ([uv[0]], [xyz[0]])
    uv, xyz = [], []
    for s_uv, s_xyz in sides:
        uv.extend(s_uv)
        xyz.extend(s_xyz)
    return uv, xyz


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------

def _unwrap(seq, period):
    out = [seq[0]]
    for x in seq[1:]:
        prev = out[-1]
        while x - prev > period / 2:
            x -= period
        while x - prev < -period / 2:
            x += period
        out.append(x)
    return out


def _point_in_poly(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside


def _points_in_poly_batch(pts, poly):
    """Vectorized ray-casting point-in-polygon test: many points against one
    polygon in a single NumPy pass. The same parity computation and edge order
    as `_point_in_poly`, so the result is identical, only evaluated in bulk.
    Returns a bool array, one entry per row of `pts`.
    """
    pts = np.asarray(pts, dtype=np.float64)
    poly = np.asarray(poly, dtype=np.float64)
    if pts.shape[0] == 0 or poly.shape[0] < 3:
        return np.zeros(pts.shape[0], dtype=bool)
    x = pts[:, 0][:, None]
    y = pts[:, 1][:, None]
    xi = poly[:, 0][None, :]
    yi = poly[:, 1][None, :]
    xj = np.roll(poly[:, 0], 1)[None, :]
    yj = np.roll(poly[:, 1], 1)[None, :]
    cond = (yi > y) != (yj > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi
    hit = cond & (x < xint)
    return np.logical_xor.reduce(hit, axis=1)


def _bowyer_watson(points, super_scale=20):
    """Delaunay triangulation. points: (N,2). Returns list of (i,j,k).

    The construction starts from a helper triangle `super_scale` times the
    extent of the points. A hull triangle whose three points are nearly
    collinear has a circumcircle far bigger than that, so the hull edge it
    carries is lost; `_constrained_delaunay` repeats the construction with a
    larger helper triangle.

    Grid-accelerated incremental (Bowyer-Watson) construction: a uniform grid
    over the point bounding box maps each triangle to the cells its
    circumcircle's bounding box overlaps, so a new point only tests the
    triangles registered under its own cell. Falls back to a full scan on the
    rare cell miss.
    """
    n = len(points)
    if n < 3:
        return []
    pts = list(points)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    dx = maxx - minx or 1.0
    dy = maxy - miny or 1.0
    dmax = max(dx, dy) * super_scale + 1.0
    midx = (minx + maxx) / 2
    midy = (miny + maxy) / 2
    pts.append((midx - dmax, midy - dmax))
    pts.append((midx, midy + dmax))
    pts.append((midx + dmax, midy - dmax))
    i0, i1, i2 = n, n + 1, n + 2

    def circum(a, b, c):
        ax, ay = pts[a]
        bx, by = pts[b]
        cx, cy = pts[c]
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-20:
            return None
        ux = ((ax*ax+ay*ay)*(by-cy) + (bx*bx+by*by)*(cy-ay) + (cx*cx+cy*cy)*(ay-by)) / d
        uy = ((ax*ax+ay*ay)*(cx-bx) + (bx*bx+by*by)*(ax-cx) + (cx*cx+cy*cy)*(bx-ax)) / d
        r2 = (ax-ux)**2 + (ay-uy)**2
        return (ux, uy, r2)

    cell = 2.0 * max(dx, dy) / max(1.0, math.sqrt(n)) or 1.0
    # Only the n real points are looked up in the grid, and all lie in
    # [minx,maxx]x[miny,maxy]. Clamping registration to that box (+1 cell
    # margin) keeps the huge super-triangle (~20x the data extent) from
    # registering itself into tens of thousands of grid cells.
    gminx, gmaxx = minx - cell, maxx + cell
    gminy, gmaxy = miny - cell, maxy + cell

    def cell_of(x, y):
        return (math.floor(x / cell), math.floor(y / cell))

    def cells_for_bbox(cx, cy, r):
        bx0 = max(cx - r, gminx)
        bx1 = min(cx + r, gmaxx)
        by0 = max(cy - r, gminy)
        by1 = min(cy + r, gmaxy)
        if bx0 > bx1 or by0 > by1:
            return
        gi0, gi1 = math.floor(bx0 / cell), math.floor(bx1 / cell)
        gj0, gj1 = math.floor(by0 / cell), math.floor(by1 / cell)
        for gi in range(gi0, gi1 + 1):
            for gj in range(gj0, gj1 + 1):
                yield (gi, gj)

    grid = {}
    circ = {}

    def add_tri(t):
        c = circum(*t)
        circ[t] = c
        if c is not None:
            cx, cy, r2 = c
            r = math.sqrt(max(r2, 0.0))
            for k in cells_for_bbox(cx, cy, r):
                grid.setdefault(k, set()).add(t)

    def remove_tri(t):
        c = circ.pop(t, None)
        if c is not None:
            cx, cy, r2 = c
            r = math.sqrt(max(r2, 0.0))
            for k in cells_for_bbox(cx, cy, r):
                s = grid.get(k)
                if s is not None:
                    s.discard(t)

    tris = set()
    t0 = (i0, i1, i2)
    tris.add(t0)
    add_tri(t0)

    for ip in range(n):
        px, py = pts[ip]
        candidates = grid.get(cell_of(px, py))
        bad = []
        if candidates:
            for t in candidates:
                if t not in tris:
                    continue
                c = circ.get(t)
                if c is None:
                    continue
                if (px - c[0]) ** 2 + (py - c[1]) ** 2 <= c[2] + 1e-12:
                    bad.append(t)
        if not bad:
            for t in tris:
                c = circ.get(t)
                if c is None:
                    continue
                if (px - c[0]) ** 2 + (py - c[1]) ** 2 <= c[2] + 1e-12:
                    bad.append(t)
        edges = {}
        for t in bad:
            for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                key = tuple(sorted(e))
                edges[key] = edges.get(key, 0) + 1
        for t in bad:
            tris.discard(t)
            remove_tri(t)
        for (a, b), cnt in edges.items():
            if cnt == 1:
                nt = (a, b, ip)
                tris.add(nt)
                add_tri(nt)

    result = []
    for t in tris:
        if t[0] >= n or t[1] >= n or t[2] >= n:
            continue
        result.append(t)
    return result


# ---------------------------------------------------------------------------
# Constrained Delaunay triangulation (boundary + holes, no bridge artifacts)
# ---------------------------------------------------------------------------
#
# `earcut`'s hole "bridging" stitches each hole into the outer loop through one
# bridge edge that plain ear-clipping walks twice, producing thin fan triangles
# anchored at one hub vertex. Each slit wall borders only ONE triangle, so
# Delaunay-improving flips can never touch it and the fan is permanent: on a
# face with many holes (a flange with a row of bolt holes) the hole vertices
# all connect to one or two hub points instead of their nearest neighbours.
#
# Instead, a real unconstrained Delaunay triangulation of every boundary point
# is built at once (`_bowyer_watson`) and each required boundary edge (outer
# ring and every hole ring) is forced to exist by flipping the non-constrained
# edges it crosses: the standard flip algorithm for constrained Delaunay
# triangulation. Every triangle is shared by two neighbours, so nothing is
# permanently un-refinable.
def _build_edge_map(tris):
    edge_map = {}
    for ti, t in enumerate(tris):
        for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_map.setdefault(frozenset(e), []).append(ti)
    return edge_map


def _has_edge(edge_map, i, j):
    return frozenset((i, j)) in edge_map


def _segs_cross(p1, p2, p3, p4):
    """Proper intersection test (segments sharing an endpoint don't count),
    with a bounding-box rejection first: `_recover_edge` calls this against
    every unconstrained edge of the face and most are nowhere near.
    """
    if max(p1[0], p2[0]) < min(p3[0], p4[0]) or max(p3[0], p4[0]) < min(p1[0], p2[0]):
        return False
    if max(p1[1], p2[1]) < min(p3[1], p4[1]) or max(p3[1], p4[1]) < min(p1[1], p2[1]):
        return False
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)
    return (d1 > 0) != (d2 > 0) and d1 != 0 and d2 != 0 \
        and (d3 > 0) != (d4 > 0) and d3 != 0 and d4 != 0


def _normalize_winding(pts, tris):
    for t in tris:
        if _orient(pts[t[0]], pts[t[1]], pts[t[2]]) < 0:
            t[0], t[1] = t[1], t[0]


def _recover_edge(pts, tris, edge_map, i, j, guard_max=64):
    """Force edge (i, j) to exist in `tris` by flipping whatever
    non-constrained edges it crosses. Best effort: gives up, leaving the
    triangulation valid but without this one edge, if `guard_max` flips are
    not enough, which only happens with pathological or self-intersecting
    input loops, and beats hanging on bad STEP data.
    """
    pi, pj = pts[i], pts[j]
    for _ in range(guard_max):
        if _has_edge(edge_map, i, j):
            return True
        flipped = False
        # Iterates the live dict: it is mutated once per pass, immediately
        # followed by `break`.
        for e, ts in edge_map.items():
            if i in e or j in e:
                continue
            if len(ts) != 2:
                continue
            t0, t1 = ts
            a, b = _directed_edge(tris[t0], *tuple(e))
            if not _segs_cross(pi, pj, pts[a], pts[b]):
                continue
            c = _third(tris[t0], a, b)
            d = _third(tris[t1], a, b)
            if c is None or d is None or c == d:
                continue
            if _orient(pts[a], pts[c], pts[d]) * _orient(pts[b], pts[c], pts[d]) >= 0:
                continue
            if _orient(pts[c], pts[a], pts[b]) * _orient(pts[d], pts[a], pts[b]) >= 0:
                continue
            _flip_edge(tris, edge_map, e, t0, t1, a, b, c, d)
            flipped = True
            break
        if not flipped:
            return _has_edge(edge_map, i, j)
    return _has_edge(edge_map, i, j)


# Helper-triangle sizes (times the extent of the points) `_constrained_delaunay`
# tries in turn. A strip 350 mm long and 5 mm wide, with a point 0.085 mm from
# the end of its long edge, needs a factor of about 400.
SUPER_TRIANGLE_SCALES = (20, 400, 8000)


def _constrained_delaunay(pts, constrained_edges):
    """Delaunay-triangulate `pts` (2D) such that every edge in
    `constrained_edges` (an iterable of 2-element index sequences) is present
    as a triangle edge. Returns `(tris, all_recovered)`: `tris` is a list of
    mutable [i, j, k] triangles (or [] for fewer than 3 points);
    `all_recovered` is False if a constrained edge could not be forced into
    existence (pathological or self-intersecting loops), in which case callers
    that rely on the ring edges as walls (`_crop_to_domain`'s flood fill) must
    fall back to the exact test.

    `guard_max` for each edge recovery scales with `len(pts)` instead of a fixed
    64: on a long, densely and near-collinearly sampled straight edge the
    triangulation has no unique diagonal, and recovering one boundary edge can
    need flips across several near-zero-area slivers in a row; giving up there
    would leave a boundary point unconnected to its neighbour (a comb of
    slivers). The common case keeps 64.
    """
    constrained_edges = list(constrained_edges)
    guard_max = max(64, 4 * len(pts))
    first = None
    for attempt, scale in enumerate(SUPER_TRIANGLE_SCALES):
        tris = [list(t) for t in _bowyer_watson(pts, scale)]
        if not tris:
            return tris, True
        _normalize_winding(pts, tris)
        edge_map = _build_edge_map(tris)
        all_recovered = True
        for e in constrained_edges:
            i, j = tuple(e)
            if not _recover_edge(pts, tris, edge_map, i, j, guard_max=guard_max):
                all_recovered = False
        # Where two ring edges cross each other, recovering the second flips
        # the first away again; a larger helper triangle is only taken when
        # every edge is really there at the end.
        if attempt and all_recovered:
            all_recovered = all(_has_edge(edge_map, *tuple(e)) for e in constrained_edges)
        if all_recovered:
            return tris, True
        if first is None:
            first = (tris, False)
    return first


def _poly_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def _point_ok(cx, cy, outer_poly, outer_bbox, hole_data):
    ox0, oy0, ox1, oy1 = outer_bbox
    if cx < ox0 or cx > ox1 or cy < oy0 or cy > oy1:
        return False
    if not _point_in_poly((cx, cy), outer_poly):
        return False
    for h, (hx0, hy0, hx1, hy1) in hole_data:
        if cx < hx0 or cx > hx1 or cy < hy0 or cy > hy1:
            continue
        if _point_in_poly((cx, cy), h):
            return False
    return True


def _crop_to_domain(pts, tris, outer_poly, hole_polys, constrained=None):
    """Keep only the triangles inside `outer_poly` and outside every hole in
    `hole_polys`.

    With every ring edge present in `tris` (see `_constrained_delaunay`) those
    edges are impassable walls, so the triangles of the face form one
    edge-connected region. One point-in-polygon test places a seed triangle and
    a flood fill across the non-constrained edges recovers the region in
    O(triangles), instead of testing every triangle against every hole. Falls
    back to per-triangle classification if the seed search or the flood comes
    up empty (a degenerate boundary loop): correctness first, this is a speed
    path.
    """
    if not tris:
        return tris
    outer_bbox = _poly_bbox(outer_poly)
    hole_data = [(h, _poly_bbox(h)) for h in hole_polys]

    if constrained is not None:
        edge_map = _build_edge_map(tris)
        seed = None
        fallback = None
        for ti, t in enumerate(tris):
            a, b, c = t
            cx = (pts[a][0] + pts[b][0] + pts[c][0]) / 3.0
            cy = (pts[a][1] + pts[b][1] + pts[c][1]) / 3.0
            if _point_ok(cx, cy, outer_poly, outer_bbox, hole_data):
                # The centre of a near-flat triangle on a straight run of
                # boundary points lies on the boundary itself, and the test
                # can call it inside: seed from a triangle with some area.
                (ax, ay), (bx, by), (qx, qy) = pts[a], pts[b], pts[c]
                area2 = abs((bx - ax) * (qy - ay) - (by - ay) * (qx - ax))
                longest2 = max((bx - ax) ** 2 + (by - ay) ** 2,
                               (qx - bx) ** 2 + (qy - by) ** 2,
                               (ax - qx) ** 2 + (ay - qy) ** 2)
                if area2 >= 0.02 * longest2:
                    seed = ti
                    break
                if fallback is None:
                    fallback = ti
        if seed is None:
            seed = fallback
        if seed is not None:
            visited = {seed}
            stack = [seed]
            while stack:
                ti = stack.pop()
                t = tris[ti]
                for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                    key = frozenset(e)
                    if key in constrained:
                        continue
                    for tj in edge_map.get(key, ()):
                        if tj not in visited:
                            visited.add(tj)
                            stack.append(tj)
            if visited:
                return [tris[i] for i in visited]

    out = []
    for t in tris:
        a, b, c = t
        cx = (pts[a][0] + pts[b][0] + pts[c][0]) / 3.0
        cy = (pts[a][1] + pts[b][1] + pts[c][1]) / 3.0
        if _point_ok(cx, cy, outer_poly, outer_bbox, hole_data):
            out.append(t)
    return out


def triangulate_face(surface, bounds, deflection, uv_extent, max_edge=None,
                     debug=False, face_label="", log=print):
    """Triangulate one trimmed face into well-shaped triangles.

    `bounds` : list of (uv_polygon, xyz_polygon, is_outer).
    Returns (verts3d, tris). Boundary points use exact sampled edge coords;
    interior Steiner points are evaluated on the surface. The boundary is
    preserved exactly so adjacent faces weld. Debug messages go through `log`
    (default `print`), so the caller controls where they are emitted.
    """
    outer = next((b for b in bounds if b[2]), None)
    if outer is None:
        if not bounds:
            return None, None
        outer = max(bounds, key=lambda b: _poly_area(b[0]))
    holes = [b for b in bounds if b is not outer]

    # The geometric predicates below (circumcircle tests, Delaunay flips) work
    # on raw (u, v), which is only "well shaped in 3D" when (u, v) is an
    # isometry: true for a Plane, false for nearly every curved surface (a
    # cylinder's u is an angle, its v a length), and the result is long, badly
    # angled triangles. `local_scale` (see geometry.py) gives the per-axis
    # multiplier that makes (u*su, v*sv) about isometric (exactly for
    # developable types). It is applied to every coordinate that feeds a
    # predicate; `surface.eval` always takes the original, unscaled (u, v).
    su, sv = surface.local_scale(uv_extent)

    # Merge near-coincident consecutive ring points first (see `_dedupe_ring`):
    # a near-zero-length constrained ring edge leaves permanent zero-area
    # slivers, because constrained edges are never touched by the flip-based
    # repair. The tolerance is a small fraction of the chord tolerance, so only
    # points that are negligible at the requested quality are removed.
    dedupe_tol = max(deflection * 0.02, 1e-9)
    total_removed = 0
    new_outer_uv, new_outer_xyz, removed = _dedupe_ring(outer[0], outer[1], su, sv, dedupe_tol)
    outer = (new_outer_uv, new_outer_xyz, outer[2])
    total_removed += removed
    new_holes = []
    for h in holes:
        hu, hx, r = _dedupe_ring(h[0], h[1], su, sv, dedupe_tol)
        new_holes.append((hu, hx, h[2]))
        total_removed += r
    holes = new_holes
    if isinstance(surface, BSplineSurface):
        outer = (_spread_collapsed_ring(outer[0], outer[1], su, sv), outer[1], outer[2])
        holes = [(_spread_collapsed_ring(h[0], h[1], su, sv), h[1], h[2]) for h in holes]
    if debug and total_removed:
        log(f"[StepForge] {face_label}: dedupe removed {total_removed} "
            f"near-duplicate boundary point(s) (tol={dedupe_tol:.6g})")

    uv = list(outer[0])
    xyz = list(outer[1])
    hole_indices = []
    for h in holes:
        hole_indices.append(len(uv))
        uv.extend(h[0])
        xyz.extend(h[1])
    if len(uv) < 3:
        return None, None
    n_boundary = len(uv)

    def to_metric(points):
        return [(u * su, v * sv) for (u, v) in points]

    ext = max((uv_extent[1][0] - uv_extent[0][0]) * su,
              (uv_extent[1][1] - uv_extent[0][1]) * sv, 1e-6)

    # The tie-breaking jitter must be small relative to the CLOSEST spacing of
    # consecutive boundary points, not just the face's extent: on a long,
    # densely and near-collinearly sampled straight edge a fixed `ext * 1e-7`
    # can flip the order of two adjacent points and turn a degenerate test into
    # a wrong one, which starves `_recover_edge`. Bounding eps by the minimum
    # consecutive boundary spacing (metric units) keeps it negligible.
    def _min_ring_spacing(ring_uv):
        n = len(ring_uv)
        if n < 2:
            return float("inf")
        m = ring_uv
        best = float("inf")
        for i in range(n):
            (u0, v0), (u1, v1) = m[i], m[(i + 1) % n]
            d = math.hypot((u1 - u0) * su, (v1 - v0) * sv)
            if 0.0 < d < best:
                best = d
        return best

    min_spacing = min([_min_ring_spacing(outer[0])] +
                      [_min_ring_spacing(h[0]) for h in holes])
    eps = ext * 1e-7
    if min_spacing < float("inf"):
        eps = min(eps, min_spacing * 1e-3)
    eps = max(eps, ext * 1e-12)

    def perturbed(points):
        out = []
        for k, (u, v) in enumerate(points):
            a = ((k * 2654435761) & 0xFFFF) / 65535.0 - 0.5
            b = ((k * 40503 + 7) & 0xFFFF) / 65535.0 - 0.5
            out.append((float(u + a * eps), float(v + b * eps)))
        return out

    # Kept in raw (u, v) for `_interior_points`/`_adaptive_grid_uv` (which
    # already work correctly in real 3D via `surface.eval`, and whose
    # near-boundary de-dup tolerance is calibrated in raw parameter units).
    outer_poly = list(outer[0])
    hole_polys = [h[0] for h in holes]
    # Metric-scaled copies for the triangulation predicates below, so they
    # stay in the same coordinate space as `pts_boundary`/`pts2d`.
    outer_poly_m = to_metric(outer_poly)
    hole_polys_m = [to_metric(h) for h in hole_polys]

    constrained = set()
    _add_ring_edges(constrained, range(len(outer[0])))
    base = len(outer[0])
    for h in holes:
        _add_ring_edges(constrained, range(base, base + len(h[0])))
        base += len(h[0])

    # Constrained Delaunay first: well-shaped triangles everywhere, without
    # earcut's permanent hole-bridge slivers (see the comment above
    # `_build_edge_map`).
    pts_boundary = perturbed(to_metric(uv))
    tris, all_recovered = _constrained_delaunay(pts_boundary, constrained)
    tris = _crop_to_domain(pts_boundary, tris, outer_poly_m, hole_polys_m,
                           constrained if all_recovered else None) if tris else []
    if not all_recovered and debug:
        log(f"[StepForge] {face_label}: constrained-edge recovery failed "
            f"for at least one boundary edge -- falling back to "
            f"per-triangle domain classification for this face")

    if not tris:
        # Retry once with a different jitter seed: a fresh jitter often clears
        # a numerical near-degeneracy that survived the ring dedupe (usually a
        # coincidence between points on DIFFERENT rings).
        if debug:
            log(f"[StepForge] {face_label}: first CDT attempt produced no "
                f"usable triangles, retrying with a different jitter seed")

        def perturbed_retry(points):
            eps = ext * 3e-6
            if min_spacing < float("inf"):
                eps = min(eps, min_spacing * 1e-2)
            eps = max(eps, ext * 1e-12)
            out = []
            for k, (u, v) in enumerate(points):
                a = (((k + 1) * 2246822519) & 0xFFFF) / 65535.0 - 0.5
                b = (((k + 1) * 3266489917 + 11) & 0xFFFF) / 65535.0 - 0.5
                out.append((float(u + a * eps), float(v + b * eps)))
            return out

        pts_retry = perturbed_retry(to_metric(uv))
        tris2, all_recovered2 = _constrained_delaunay(pts_retry, constrained)
        tris2 = _crop_to_domain(pts_retry, tris2, outer_poly_m, hole_polys_m,
                                constrained if all_recovered2 else None) if tris2 else []
        if tris2:
            pts_boundary = pts_retry
            tris = tris2
            all_recovered = all_recovered2
            if debug:
                log(f"[StepForge] {face_label}: retry succeeded "
                    f"({len(tris)} triangles)")

    if not tris:
        # Last resort for degenerate or self-intersecting boundary loops where
        # edge recovery could not converge: ear clipping with holes almost
        # never fails outright. It can reintroduce the hub-fan slivers the CDT
        # exists to avoid (see the comment above `_build_edge_map`) and is
        # rare, so it is flagged whenever it triggers.
        if debug:
            log(f"[StepForge] {face_label}: WARNING falling back to plain "
                f"earcut triangulation -- expect possible sliver "
                f"triangles on this face ({len(uv)} boundary pts, "
                f"{len(holes)} hole(s))")
        try:
            tris = earcut.earcut(pts_boundary, hole_indices or None)
        except Exception:  # noqa: BLE001 - last resort, any failure means no triangles
            tris = []
        tris = [list(t) for t in tris]
    if not tris:
        return None, None

    interior_uv = _interior_points(surface, outer_poly, hole_polys, uv_extent,
                                   deflection, max_edge)
    if interior_uv:
        for p in interior_uv:
            uv.append(p)
            xyz.append(surface.eval(p[0], p[1]))
        pts2d = perturbed(to_metric(uv))
        # Interior points go in through incremental Delaunay insertion
        # (`delaunay.py`): each point is located by walking from the previous
        # one and legalized on the spot (splitting each containing triangle
        # 1->3 without legalizing leaves most triangles of a B-spline face over
        # 6:1 aspect ratio until a full-face refine repairs them). A Delaunay
        # triangulation of points in general position is unique (`perturbed`
        # puts them there), so the result does not depend on how it is
        # reached.
        _tri = _delaunay.Triangulation.from_triangles(
            pts2d, tris, constrained=constrained)
        n_interior = len(uv) - n_boundary
        if n_interior > 1:
            # Spatially coherent order, so each walk starts next door to where
            # it has to end. It cannot change the result: legalizing at every
            # step makes it independent of insertion order.
            interior_order = [n_boundary + k for k
                              in _delaunay.hilbert_order(pts2d[n_boundary:])]
        else:
            interior_order = list(range(n_boundary, len(uv)))
        for i in interior_order:
            _tri.insert(i)
        tris[:] = _tri.triangles()
    else:
        pts2d = perturbed(to_metric(uv))

    # Single repair pass: one worklist flip pass over a merged criterion (plain
    # Delaunay in-circle, 3D degenerate collinearity and 3D aspect ratio; see
    # `_flip_gain_3d`). The incremental insertion above already legalized the
    # 2D Delaunay property, so this cleans up the residual 3D-aware defects it
    # cannot see.
    # A flip judged by 3D triangle shape alone can pick a diagonal that cuts
    # through the part: on a coarsely sampled narrow cylinder or cone band,
    # every triangle is a sliver and a chord straight across the axis scores
    # "fatter". Only take a new diagonal whose midpoint stays within the
    # chord tolerance of the surface (or no worse than the edge it replaces).
    def _chord_dev(i, j):
        um = 0.5 * (uv[i][0] + uv[j][0])
        vm = 0.5 * (uv[i][1] + uv[j][1])
        mid = 0.5 * (np.asarray(xyz[i], dtype=float) + np.asarray(xyz[j], dtype=float))
        return float(np.linalg.norm(surface.eval(um, vm) - mid))

    def _edge_ok(a, b, c, d):
        new = _chord_dev(c, d)
        return new <= deflection or new <= _chord_dev(a, b) + 1e-12

    edge_ok = None if isinstance(surface, Plane) else _edge_ok
    total_flipped, bad_final, edge_map = _unify_repair_3d(pts2d, xyz, tris, constrained,
                                                          edge_ok=edge_ok)

    # Some residual bad triangles (near-zero-area collinear clusters, spiky but
    # valid slivers) are unflippable in place, every candidate flip blocked by
    # 2D convexity, yet have a free interior point that can move off the
    # isoline or into a better position. Nudge each, then run another seeded
    # pass of the same worklist to pick up the connectivity the move allows.
    # Relocation and reflip repeat while they make progress, up to 6 rounds:
    # one round clears most faces, and only the rare hard face (dense NURBS
    # geometry) pays for more.
    for _ in range(6):
        if not (n_boundary < len(uv) and any(not touches_boundary
                                             for (*_, touches_boundary) in bad_final)):
            break
        moved = _relocate_bad_points_3d(uv, pts2d, xyz, tris, constrained, n_boundary,
                                        surface, ext, su, sv, to_metric, deflection,
                                        debug=debug, face_label=face_label, log=log)
        if not moved:
            break
        seed_edges = [e for e in edge_map if not e.isdisjoint(moved)]
        _, bad_final, edge_map = _unify_repair_3d(
            pts2d, xyz, tris, constrained, edge_map=edge_map, seed_edges=seed_edges,
            edge_ok=edge_ok)

    out_tris = [tuple(t) for t in tris
                if t[0] != t[1] and t[1] != t[2] and t[0] != t[2]]

    if debug:
        bad = 0
        thin = 0
        for (a, b, c) in out_tris:
            area = _tri_area3(xyz, a, b, c)
            maxe = max(_edge_len3(xyz, a, b), _edge_len3(xyz, b, c), _edge_len3(xyz, c, a))
            if maxe > 1e-12 and area < 1e-6 * maxe * maxe:
                bad += 1
            if maxe > 1e-12:
                alt = (2.0 * area) / maxe
                if alt <= 1e-12 or maxe / alt > 6.0:
                    thin += 1
        if bad:
            log(f"[StepForge] {face_label}: {bad}/{len(out_tris)} residual "
                f"sliver triangle(s) could not be repaired (likely a "
                f"near-duplicate point pair on two DIFFERENT rings, or a "
                f"genuinely degenerate STEP edge)")
        if total_flipped or thin:
            n_boundary_touch = sum(1 for *_, tb in bad_final if tb)
            n_interior_only = len(bad_final) - n_boundary_touch
            log(f"[StepForge] {face_label}: unified repair pass flipped "
                f"{total_flipped} edge(s), {thin}/{len(out_tris)} "
                f"triangle(s) still over 6:1 aspect ratio afterwards "
                f"({n_boundary_touch} touch a boundary/constrained edge "
                f"and cannot be flip-repaired without breaking welding, "
                f"{n_interior_only} are purely interior and unexpected)")
    return xyz, out_tris


def _poly_area(poly):
    a = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2


def _add_ring_edges(constrained, idxs):
    idxs = list(idxs)
    n = len(idxs)
    for i in range(n):
        constrained.add(frozenset((idxs[i], idxs[(i + 1) % n])))


def _dedupe_ring(uv_ring, xyz_ring, su, sv, tol):
    """Drop points from one boundary ring (outer or hole) that sit closer than
    `tol` (real units) to the PREVIOUS kept point, wrapping around to check
    the closing edge too.

    Every ring edge is a constrained Delaunay edge that the flip-based repair
    never touches (flipping a real boundary edge would desync this face from
    the neighbour that welds to it). Two consecutive near-coincident ring
    points (a degenerate EDGE_CURVE in the source, or a closed loop whose start
    and end samples nearly coincide) would bake a near-zero-length edge into
    the mesh and a permanent sliver at every triangle touching it, so they are
    merged away before triangulation. `tol` is a small fraction of the chord
    tolerance, so real small features (a 0.2 mm fillet edge at Fine quality)
    stay.

    Distance is measured in real 3D (`xyz_ring`), not in this face's scaled
    (u, v): `xyz_ring` comes from `_walk_loop`'s shared `edge_cache` and is
    bit-identical on both faces bordering an EDGE_CURVE, whereas `su`/`sv` are
    a per-face approximation, so one side could merge a point away while the
    other keeps it (a T-junction crack on reimport). Both faces therefore reach
    the same decision for every point of a shared edge.
    """
    n = len(uv_ring)
    if n < 4:
        return uv_ring, xyz_ring, 0
    keep = [0]
    for i in range(1, n):
        px, py, pz = xyz_ring[i]
        lx, ly, lz = xyz_ring[keep[-1]]
        dx = px - lx
        dy = py - ly
        dz = pz - lz
        if dx * dx + dy * dy + dz * dz > tol * tol:
            keep.append(i)
    # Closing edge: last kept point back to the first.
    if len(keep) >= 3:
        px, py, pz = xyz_ring[keep[0]]
        lx, ly, lz = xyz_ring[keep[-1]]
        dx = px - lx
        dy = py - ly
        dz = pz - lz
        if dx * dx + dy * dy + dz * dz <= tol * tol:
            keep.pop()
    removed = n - len(keep)
    if len(keep) < 3 or removed == 0:
        return uv_ring, xyz_ring, 0
    return [uv_ring[i] for i in keep], [xyz_ring[i] for i in keep], removed


# A ring step whose length in parameter space (metric units) is below this
# share of its 3D length is collapsed: the surface does not reach the
# boundary points on it.
COLLAPSED_STEP = 0.1


def _ring_crosses_itself(uv, segments):
    """True if any of the ring segments `segments` (indices i: point i to
    i + 1) properly crosses another segment of the closed ring `uv`."""
    n = len(uv)
    a = uv
    b = np.roll(uv, -1, axis=0)
    ab = b - a
    for i in segments:
        p, q = a[i], b[i]
        pq = q - p
        d1 = ab[:, 0] * (p[1] - a[:, 1]) - ab[:, 1] * (p[0] - a[:, 0])
        d2 = ab[:, 0] * (q[1] - a[:, 1]) - ab[:, 1] * (q[0] - a[:, 0])
        d3 = pq[0] * (a[:, 1] - p[1]) - pq[1] * (a[:, 0] - p[0])
        d4 = pq[0] * (b[:, 1] - p[1]) - pq[1] * (b[:, 0] - p[0])
        hit = (d1 * d2 < 0) & (d3 * d4 < 0)
        hit[[(i - 1) % n, i, (i + 1) % n]] = False
        if hit.any():
            return True
    return False


def _spread_collapsed_ring(uv_ring, xyz_ring, su, sv):
    """Give boundary points that the surface does not reach distinct
    parameter positions.

    A fitted B-spline patch matches its boundary curves only to the export
    tolerance, and near a corner or a narrow strip the curve can run past the
    end of the patch. Projecting such a point onto the patch (`invert`)
    clamps it to the parameter border, so several 3D points that lie tenths
    of a millimetre apart land on one (u, v), and the triangulation, which
    works in (u, v), leaves them out. The neighbouring face still has them,
    and the seam opens (a triangle-sized hole at the corner).

    The points on collapsed steps (see `COLLAPSED_STEP`) are placed on the
    straight parameter segment between the nearest well-projected points on
    either side, in proportion to their 3D arc length. Only the parameter
    positions change: the 3D boundary points stay exactly those the shared
    edge cache delivered, so both faces of a seam keep the same points. A
    ring is left as it is when it has fewer than two well-projected points,
    or when the new positions would make it cross itself."""
    n = len(uv_ring)
    if n < 4:
        return uv_ring
    uv = np.asarray(uv_ring, dtype=float)
    xyz = np.asarray(xyz_ring, dtype=float)
    step3 = np.linalg.norm(np.roll(xyz, -1, axis=0) - xyz, axis=1)
    d = np.roll(uv, -1, axis=0) - uv
    collapsed = np.hypot(d[:, 0] * su, d[:, 1] * sv) < COLLAPSED_STEP * step3
    if not collapsed.any():
        return uv_ring
    anchors = np.nonzero(~(collapsed | np.roll(collapsed, 1)))[0]
    if len(anchors) < 2:
        return uv_ring
    out = uv.copy()
    touched = []
    for k, a in enumerate(anchors):
        b = anchors[(k + 1) % len(anchors)]
        gap = (b - a) % n
        if gap < 2:
            continue
        steps = (a + np.arange(gap)) % n
        arc = np.cumsum(step3[steps])
        gap_uv = uv[b] - uv[a]
        if arc[-1] <= 0.0 or math.hypot(gap_uv[0] * su, gap_uv[1] * sv) < COLLAPSED_STEP * arc[-1]:
            continue
        inner = (a + np.arange(1, gap)) % n
        out[inner] = uv[a] + (arc[:-1] / arc[-1])[:, None] * gap_uv
        touched.extend(steps.tolist())
    if not touched or _ring_crosses_itself(out, touched):
        return uv_ring
    return [tuple(p) for p in out.tolist()]


def _adaptive_grid_uv(surface, uv_extent, deflection, max_edge, max_depth=3, cap=1500,
                       outer=None, holes=None):
    """Sample (u, v) points across the whole face so the true surface,
    evaluated at this resolution, never deviates from the flat (triangulated)
    approximation by more than `deflection`, for any surface type: one
    mechanism instead of per-surface-type formulas.

    Splitting each base cell recursively on its own would let neighbouring
    cells reach independent depths, 3-4 refinement levels apart, and Delaunay
    would then have only long, badly angled slivers to connect them. So the
    decision is split into passes, which makes the neighbours known before any
    point is emitted:

      1.  For every base cell, the scalar refinement depth it needs on its own
          (increasing until every sub-cell passes the bilinear-flatness test).
      1b. Bump a cell's depth if the face's own boundary (`outer` / `holes`)
          runs through it more densely than the cell's depth-0 size would
          produce, e.g. a short bend-transition edge on an otherwise flat,
          coarsely gridded panel. Only the few cells a dense stretch of
          boundary falls into are touched, and `max_depth` still caps it.
          (Densifying the boundary edge itself would break welding with the
          neighbouring face, and matching the interior to the boundary
          face-wide would over-refine the whole panel.)
      1c. Cap a cell's depth back DOWN if the boundary through it is much
          coarser than pass 1 would refine to: a fitted B-spline face's
          boundary is a handful of long straight segments, but its
          curvature-driven interior would refine far past what they can
          connect to cleanly (a comb of slivers). This never touches a
          boundary point or calls `surface.invert`; it only holds the interior
          back near the coarse segment, and pass 2 grades the cap into a ramp.
          That relaxes the flatness guarantee near a coarse boundary segment,
          which is already at least as coarse an approximation of the surface
          there.
      2.  Enforce the quadtree 2:1 balance rule over the depth grid: no cell
          more than one level coarser than any 4-connected neighbour. Raising
          a cell can cascade, so it repeats until nothing changes.
      3.  Emit each cell's uniform (2**depth+1)^2 sample grid. Same-depth
          neighbours share an identical edge; a neighbour one level coarser
          differs by a single hanging midpoint, which Delaunay handles cleanly.
    """
    (umin, vmin), (umax, vmax) = uv_extent
    du = umax - umin
    dv = vmax - vmin
    if du <= 0 or dv <= 0:
        return [], (umin, vmin, du, dv, 1, 1, [[0]])

    base_nu, base_nv = surface.grid_cells(deflection, uv_extent)
    if max_edge:
        base_nu = max(base_nu, math.ceil(du / max_edge))
        base_nv = max(base_nv, math.ceil(dv / max_edge))
    base_nu = max(1, min(base_nu, 160))
    base_nv = max(1, min(base_nv, 160))

    cache = {}

    def ev(u, v):
        key = (round(u, 10), round(v, 10))
        p = cache.get(key)
        if p is None:
            p = surface.eval(u, v)
            cache[key] = p
        return p

    def cell_extent(I, J):
        u0 = umin + du * I / base_nu
        u1 = umin + du * (I + 1) / base_nu
        v0 = vmin + dv * J / base_nv
        v1 = vmin + dv * (J + 1) / base_nv
        return u0, v0, u1, v1

    def flat_enough(u0, v0, u1, v1, depth):
        n = 1 << depth
        for si in range(n):
            su0 = u0 + (u1 - u0) * si / n
            su1 = u0 + (u1 - u0) * (si + 1) / n
            for sj in range(n):
                sv0 = v0 + (v1 - v0) * sj / n
                sv1 = v0 + (v1 - v0) * (sj + 1) / n
                p00 = ev(su0, sv0)
                p01 = ev(su0, sv1)
                p10 = ev(su1, sv0)
                p11 = ev(su1, sv1)
                pm = ev((su0 + su1) * 0.5, (sv0 + sv1) * 0.5)
                bilinear = (p00 + p01 + p10 + p11) * 0.25
                if np.linalg.norm(pm - bilinear) > deflection:
                    return False
        return True

    # Pass 1: scalar depth per base cell, independent of neighbours.
    depth = [[0] * base_nv for _ in range(base_nu)]
    for I in range(base_nu):
        for J in range(base_nv):
            u0, v0, u1, v1 = cell_extent(I, J)
            d = 0
            while d < max_depth and not flat_enough(u0, v0, u1, v1, d):
                d += 1
            depth[I][J] = d

    # Pass 1b: bump depth for cells a locally dense stretch of this face's own
    # boundary passes through (see docstring). `su`/`sv` convert the boundary's
    # (u, v) to the metric space `local_scale` gives every other predicate, so
    # "dense" is judged in real units; on a cylinder or cone u is an angle and
    # v a length.
    if outer or holes:
        su, sv = surface.local_scale(uv_extent)
        cu0 = du / base_nu * su
        cv0 = dv / base_nv * sv
        cell_dim0 = max(cu0, cv0)

        def cell_index(u, v):
            i = int((u - umin) / du * base_nu)
            j = int((v - vmin) / dv * base_nv)
            i = max(0, min(base_nu - 1, i))
            j = max(0, min(base_nv - 1, j))
            return i, j

        def bump(I, J, seg_len, trigger, dim):
            if seg_len <= 1e-12 or seg_len >= trigger:
                return
            d = 0
            while d < max_depth and dim / (1 << d) > seg_len:
                d += 1
            depth[I][J] = max(depth[I][J], d)

        rings = ([outer] if outer else []) + list(holes or [])
        for ring in rings:
            n = len(ring)
            if n < 3:
                continue
            seg_lens = []
            for k in range(n):
                pu, pv = ring[k]
                qu, qv = ring[(k + 1) % n]
                seg_lens.append(math.hypot((qu - pu) * su, (qv - pv) * sv))
            # A plain "seg_len < this cell's own size" trigger fires on almost
            # any boundary with average curvature, refining broadly across the
            # face instead of only for a short bend-transition edge on an
            # otherwise coarse boundary. Compare each segment to this ring's
            # OWN median length instead: only one markedly shorter (< 1/4) than
            # typical for the ring is a locally dense outlier. A uniformly fine
            # ring (a dense spline boundary) has none, and its interior density
            # already comes from the flatness pass.
            sorted_lens = sorted(seg_lens)
            median = sorted_lens[len(sorted_lens) // 2]
            trigger = min(median * 0.25, cell_dim0)
            if trigger <= 1e-12:
                continue
            for k in range(n):
                pu, pv = ring[k]
                qu, qv = ring[(k + 1) % n]
                seg_len = seg_lens[k]
                # Match the cell's size along the segment's own direction:
                # a cell much taller than wide (a long thin rod) needs no
                # extra rows along its length for a dense circle at its end.
                dim = cu0 if abs(qu - pu) * su >= abs(qv - pv) * sv else cv0
                Ip, Jp = cell_index(pu, pv)
                bump(Ip, Jp, seg_len, trigger, dim)
                Iq, Jq = cell_index(qu, qv)
                if (Iq, Jq) != (Ip, Jp):
                    bump(Iq, Jq, seg_len, trigger, dim)

        # Pass 1c: cap depth back DOWN for cells whose local boundary is much
        # coarser than pass 1's flatness test alone would refine to (see
        # docstring). It never adds, moves or re-inverts a boundary point (an
        # off-surface point inverts to a wrong (u, v) on a fitted B-spline,
        # folding the ring) and only holds the interior back, using the
        # always robust forward `surface.eval`. Pass 2's 2:1 balance then
        # grades the cap into a ramp. The interior next to a coarse boundary
        # segment gives up some accuracy there, but the segment is already at
        # least as coarse an approximation of the surface.
        for ring in rings:
            n = len(ring)
            if n < 3:
                continue
            for k in range(n):
                pu, pv = ring[k]
                qu, qv = ring[(k + 1) % n]
                seg_len = math.hypot((qu - pu) * su, (qv - pv) * sv)
                if seg_len <= 1e-12:
                    continue
                # Walk the whole segment, not just its two endpoints: a long
                # segment can cross several base cells, and capping only the
                # endpoint cells leaves a single-cell dent that pass 2's 2:1
                # balance (symmetric: it also raises a coarse cell back up to
                # within one level of a fine neighbour) mostly erases.
                nsteps = max(1, math.ceil(seg_len / max(cell_dim0 / (1 << max_depth), 1e-9)))
                for step in range(nsteps + 1):
                    t = step / nsteps
                    I, J = cell_index(pu + (qu - pu) * t, pv + (qv - pv) * t)
                    # Compare against THIS cell's depth as passes 1/1b left it,
                    # not the depth-0 size: a cell already refined past depth 0
                    # by flatness has a much smaller real size, and that is the
                    # size the boundary segment through it must be compared to.
                    cur_size = cell_dim0 / (1 << depth[I][J])
                    if seg_len <= cur_size:
                        continue
                    d = 0
                    while d < max_depth and cell_dim0 / (1 << d) > seg_len:
                        d += 1
                    cap_d = max(0, d - 1)
                    depth[I][J] = min(depth[I][J], cap_d)

    # Pass 2: 2:1 neighbour balance, propagated to a global fixed point.
    changed = True
    while changed:
        changed = False
        for I in range(base_nu):
            for J in range(base_nv):
                d = depth[I][J]
                nmax = d
                if I > 0:
                    nmax = max(nmax, depth[I - 1][J])
                if I < base_nu - 1:
                    nmax = max(nmax, depth[I + 1][J])
                if J > 0:
                    nmax = max(nmax, depth[I][J - 1])
                if J < base_nv - 1:
                    nmax = max(nmax, depth[I][J + 1])
                if nmax - d > 1:
                    depth[I][J] = nmax - 1
                    changed = True

    # Pass 3: emit each cell's uniform grid at its balanced depth.
    for I in range(base_nu):
        for J in range(base_nv):
            if len(cache) > cap:
                break
            u0, v0, u1, v1 = cell_extent(I, J)
            n = 1 << depth[I][J]
            for si in range(n + 1):
                u = u0 + (u1 - u0) * si / n
                for sj in range(n + 1):
                    v = v0 + (v1 - v0) * sj / n
                    ev(u, v)

    # Return the balanced depth grid with the points: the near-boundary de-dup
    # in `_interior_points` sizes its rejection tolerance to each point's own
    # local resolution. A flat estimate would reject the deliberately dense
    # points of a bumped cell as "too close to the boundary", leaving a sparse,
    # badly placed few.
    grid_meta = (umin, vmin, du, dv, base_nu, base_nv, depth)
    return list(cache.keys()), grid_meta


def _circular_boundary_center(outer, holes, rel_tol=0.02):
    """Return a single interior (u, v) point, the fitted circle's own centre, if
    `outer` is a closed polygon whose points all sit within `rel_tol` of their
    mean radius from their centroid (a near-circular outline); `None` otherwise
    (nearly all real flat panels are straight-edged).

    A Plane's `grid_cells()` is always (1, 1), no interior points, which is
    right for an ordinary polygon. A full circle (a round blank, a washer, a
    full cylinder's flat end cap on re-import) has cocircular points, where the
    in-circumcircle predicate has no preference between triangulations and
    floating-point noise decides; this reliably collapsed the whole boundary
    into a one-vertex fan, which made `fit.boundary_loops()` reject the face as
    "pinched" and broke the cylinder re-fit on the next export cycle. The
    circle's own centre as one interior point gives the even fan a CAD kernel
    would draw and removes the degeneracy.

    Holes are not tested for circularity; the check only bails out if the
    centre would fall inside one."""
    if len(outer) < 8:
        return None  # too few points for "is this a circle" to mean
                      # anything (e.g. a real polygon corner run)
    pts = np.asarray(outer, dtype=np.float64)
    centroid = pts.mean(axis=0)
    d = pts - centroid
    r = np.hypot(d[:, 0], d[:, 1])
    rmean = float(r.mean())
    if rmean <= 1e-12 or (r.max() - r.min()) > rel_tol * rmean:
        return None
    for h in holes:
        if len(h) < 3:
            continue
        hpts = np.asarray(h, dtype=np.float64)
        hc = hpts.mean(axis=0)
        hd = hpts - hc
        hr = float(np.hypot(hd[:, 0], hd[:, 1]).mean())
        if math.hypot(centroid[0] - hc[0], centroid[1] - hc[1]) < hr:
            return None  # center would fall inside a hole -- leave alone
    return (float(centroid[0]), float(centroid[1]))


def _interior_points(surface, outer, holes, uv_extent, deflection, max_edge):
    """Thin wrapper around `_interior_points_impl`: if the adaptive-grid
    pipeline yields NO interior points (correct for an ordinary flat panel, see
    `Surface.grid_cells`), check whether this is the cocircular case
    `_circular_boundary_center` handles. It has to be a post-hoc check: for a
    1x1 base grid `_adaptive_grid_uv` does emit one candidate (the bounding-box
    centre), but the near-boundary de-dup tolerance (0.4x the base cell)
    swallows it on a circular boundary, where every boundary point is
    equidistant from it."""
    pts = _interior_points_impl(surface, outer, holes, uv_extent, deflection, max_edge)
    if not pts:
        center = _circular_boundary_center(outer, holes)
        if center is not None:
            return [center]
    return pts


def _interior_points_impl(surface, outer, holes, uv_extent, deflection, max_edge):
    (umin, vmin), (umax, vmax) = uv_extent
    du = umax - umin
    dv = vmax - vmin
    if du <= 0 or dv <= 0:
        return []

    uv_pts, grid_meta = _adaptive_grid_uv(surface, uv_extent, deflection, max_edge,
                                          outer=outer, holes=holes)
    if not uv_pts:
        return []
    g_umin, g_vmin, g_du, g_dv, g_nu, g_nv, g_depth = grid_meta

    # Near-boundary de-dup tolerance, per axis from the same base grid
    # `_adaptive_grid_uv` uses (`Surface.grid_cells` plus the `max_edge`
    # floor), not from one isotropic estimate sqrt(point count): on a long,
    # narrow face (a sheet-metal bend strip; nu=2, nv=27 was observed) u and v
    # need base cell counts an order of magnitude apart, and a single scalar
    # would be too loose on the finely divided axis and too tight on the
    # coarse one.
    base_nu, base_nv = surface.grid_cells(deflection, uv_extent)
    if max_edge:
        base_nu = max(base_nu, math.ceil(du / max_edge))
        base_nv = max(base_nv, math.ceil(dv / max_edge))
    base_nu = max(1, min(base_nu, 160))
    base_nv = max(1, min(base_nv, 160))
    tol_u = du / base_nu * 0.4
    tol_v = dv / base_nv * 0.4

    # A cell the boundary-density pass (`_adaptive_grid_uv`) bumped several
    # levels deeper produces points far closer than the flat `tol_u`/`tol_v`
    # expects, and that tolerance would reject nearly all of them as "too close
    # to the boundary". The per-point local tolerance (`ltol_u`/`ltol_v`,
    # computed vectorized below from `g_depth`) scales it down by the point's
    # own cell depth.

    bpts = list(outer)
    for h in holes:
        bpts.extend(h)

    # Each hole's bbox is precomputed once, so a candidate nowhere near a hole
    # is rejected with four comparisons instead of a point-in-polygon walk of
    # its ring (the same bbox-then-exact pattern as `_point_ok`).
    hole_data = [(h, _poly_bbox(h)) for h in holes]

    # The candidates are tested as one (N, 2) array with vectorized NumPy ops
    # (domain edge, outer polygon, holes, near-boundary de-dup, in that order):
    # no test depends on another candidate's outcome.
    P = np.asarray(uv_pts, dtype=np.float64)
    U = P[:, 0]
    V = P[:, 1]

    keep = ~((U <= umin + 1e-9) | (U >= umax - 1e-9) |
             (V <= vmin + 1e-9) | (V >= vmax - 1e-9))

    outer_arr = np.asarray(outer, dtype=np.float64)
    idx = np.nonzero(keep)[0]
    if idx.size:
        inside_outer = _points_in_poly_batch(P[idx], outer_arr)
        keep = np.zeros(len(P), dtype=bool)
        keep[idx[inside_outer]] = True
    else:
        keep[:] = False

    for h, (hx0, hy0, hx1, hy1) in hole_data:
        idx = np.nonzero(keep)[0]
        if idx.size == 0:
            break
        Uc, Vc = U[idx], V[idx]
        bbox_hit = (Uc >= hx0) & (Uc <= hx1) & (Vc >= hy0) & (Vc <= hy1)
        sub_idx = idx[bbox_hit]
        if sub_idx.size == 0:
            continue
        in_hole = _points_in_poly_batch(P[sub_idx], np.asarray(h, dtype=np.float64))
        keep[sub_idx[in_hole]] = False

    idx = np.nonzero(keep)[0]
    if idx.size == 0:
        return []

    # Near-boundary de-dup with the per-point local tolerance above, vectorized
    # over the surviving candidates and every boundary point.
    Uc, Vc = U[idx], V[idx]
    if g_du > 0 and g_dv > 0:
        depth_arr = np.asarray(g_depth, dtype=np.int64)
        gi = np.clip(((Uc - g_umin) / g_du * g_nu).astype(np.int64), 0, g_nu - 1)
        gj = np.clip(((Vc - g_vmin) / g_dv * g_nv).astype(np.int64), 0, g_nv - 1)
        n_local = (1 << depth_arr[gi, gj]).astype(np.float64)
        ltol_u = tol_u / n_local
        ltol_v = tol_v / n_local
    else:
        ltol_u = np.full(idx.shape, tol_u)
        ltol_v = np.full(idx.shape, tol_v)

    bpts_arr = np.asarray(bpts, dtype=np.float64)
    if bpts_arr.size:
        near = np.any(
            (np.abs(Uc[:, None] - bpts_arr[None, :, 0]) < ltol_u[:, None]) &
            (np.abs(Vc[:, None] - bpts_arr[None, :, 1]) < ltol_v[:, None]),
            axis=1,
        )
    else:
        near = np.zeros(idx.shape, dtype=bool)

    final_idx = idx[~near]
    # The test above only looks at boundary VERTICES. On a flat face a
    # candidate can still sit on a boundary SEGMENT: a triangle whose long
    # edge joins two opposite corners of the parameter box has that edge's
    # midpoint as the box centre. Inserted there, it makes a zero-area
    # triangle with the two ends of that segment (the segment's own edge
    # stays constrained, so no later flip can remove it).
    if final_idx.size and isinstance(surface, Plane):
        final_idx = final_idx[~_near_ring_segments(P[final_idx], [outer] + list(holes))]

    # Performance safety valve: at most 6001 interior points per face.
    if final_idx.size > 6000:
        # Thin out evenly over the whole face. Keeping the first 6001 in
        # grid order instead left the rest of the face with no interior
        # point at all: a whole sphere at 0.1 % detail lost the interior of
        # its upper half, and its big boundary-only triangles cut 44 % of
        # the area away.
        final_idx = final_idx[np.linspace(0, final_idx.size - 1, 6001).round().astype(np.int64)]
    return [(float(P[i, 0]), float(P[i, 1])) for i in final_idx]


def _near_ring_segments(points, rings, rel=1e-3):
    """Boolean per row of `points` (N, 2): True if it lies within `rel` times
    a segment's own length of that segment, for any segment of any ring.
    A triangle from that point to the segment's two ends has a longest-edge
    to height ratio of at least 1/`rel`."""
    hit = np.zeros(len(points), dtype=bool)
    for ring in rings:
        A = np.asarray(ring, dtype=np.float64)
        if len(A) < 2:
            continue
        D = np.roll(A, -1, axis=0) - A
        L2 = (D * D).sum(axis=1)
        ok = L2 > 0
        A, D, L2 = A[ok], D[ok], L2[ok]
        if not len(A):
            continue
        step = max(1, 2_000_000 // len(A))
        for s0 in range(0, len(points), step):
            Q = points[s0:s0 + step]
            R = Q[:, None, :] - A[None, :, :]
            t = np.clip((R * D[None, :, :]).sum(axis=2) / L2, 0.0, 1.0)
            off = R - t[:, :, None] * D[None, :, :]
            hit[s0:s0 + step] |= ((off * off).sum(axis=2) < rel * rel * L2).any(axis=1)
    return hit


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _in_circumcircle(pts, a, b, c, d):
    ax, ay = pts[a]
    bx, by = pts[b]
    cx, cy = pts[c]
    dx, dy = pts[d]
    if _orient(pts[a], pts[b], pts[c]) < 0:
        bx, by, cx, cy = cx, cy, bx, by
    adx, ady = ax - dx, ay - dy
    bdx, bdy = bx - dx, by - dy
    cdx, cdy = cx - dx, cy - dy
    det = ((adx * adx + ady * ady) * (bdx * cdy - cdx * bdy)
           - (bdx * bdx + bdy * bdy) * (adx * cdy - cdx * ady)
           + (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady))
    return det > 1e-12


def _third(tri, a, b):
    for x in tri:
        if x != a and x != b:
            return x
    return None


def _directed_edge(tri, p, q):
    """Return (p, q) or (q, p), whichever is `tri`'s own forward traversal of
    that edge. `tuple(some_frozenset)` has no defined order, and using it
    directly reverses the winding of the rebuilt triangles about half the time,
    which shows up as a patch of inverted-looking shading.
    """
    a, b, c = tri
    for (u, v) in ((a, b), (b, c), (c, a)):
        if u == p and v == q:
            return p, q
        if u == q and v == p:
            return q, p
    return p, q


def _flip_edge(tris, edge_map, e, t0, t1, a, b, c, d):
    """Apply the (a,b,c,d) flip, t0=(a,b,c)/t1=(b,a,d) becoming t0=(c,a,d)/
    t1=(d,b,c), and patch `edge_map` in place instead of rebuilding it.

    Only 3 entries change: the diagonal moves from {a,b} to {c,d} (still owned
    by t0 and t1), {b,c} moves from t0 to t1 and {a,d} from t1 to t0. {a,c} and
    {b,d} keep their owner.
    """
    tris[t0] = [c, a, d]
    tris[t1] = [d, b, c]
    del edge_map[e]
    edge_map[frozenset((c, d))] = [t0, t1]
    bc = frozenset((b, c))
    lst = edge_map.get(bc)
    if lst is not None:
        for i, v in enumerate(lst):
            if v == t0:
                lst[i] = t1
                break
    ad = frozenset((a, d))
    lst = edge_map.get(ad)
    if lst is not None:
        for i, v in enumerate(lst):
            if v == t1:
                lst[i] = t0
                break


def _tri_metrics_3d(xyz, a, b, c, ratio_threshold):
    """(area, aspect, is_bad) for triangle (a,b,c) in real 3D, in one pass from
    a shared cross product and edge-length set. The caller (`_unify_repair_3d`)
    caches the result per triangle, so a triangle examined from each of its 3
    edges costs one computation.
    """
    ax, ay, az = xyz[a]
    bx, by, bz = xyz[b]
    cx, cy, cz = xyz[c]
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    area = 0.5 * (nx * nx + ny * ny + nz * nz) ** 0.5
    maxe = max(_edge_len3(xyz, a, b), _edge_len3(xyz, b, c), _edge_len3(xyz, c, a))
    if maxe < 1e-12:
        return area, 0.0, False
    if area < 1e-6 * maxe * maxe:
        return area, float("inf"), True
    alt = (2.0 * area) / maxe
    aspect = maxe / alt if alt > 1e-12 else float("inf")
    return area, aspect, aspect > ratio_threshold


def _flip_gain_3d(pts2d, xyz, a, b, c, d, cur_area0, cur_aspect0,
                  cur_area1, cur_aspect1, ratio_threshold):
    """Single merged flip-acceptance test for candidate diagonal a-b -> c-d,
    given the caller's metrics for the two current triangles (t0=(a,b,c),
    t1=(b,a,d); `_unify_repair_3d` caches them). Takes the flip if ANY of these
    clearly helps:

      1. Degenerate repair: one current triangle is a near-zero-area 3D sliver
         (collinear in real space, invisible to the 2D in-circle test) and the
         flip would give it real area.
      2. Aspect-ratio repair: one current triangle is spiky (aspect over
         `ratio_threshold`) and the flip clearly improves the worst case.
      3. Plain Delaunay in-circle, as a tie-breaker, only if it does not make
         the 3D aspect ratio worse (on a doubly curved surface a 2D-optimal
         and a 3D-well-shaped flip can disagree).
    """
    cur_min_area = min(cur_area0, cur_area1)
    new_area0, new_aspect0, _ = _tri_metrics_3d(xyz, c, a, d, ratio_threshold)
    new_area1, new_aspect1, _ = _tri_metrics_3d(xyz, d, b, c, ratio_threshold)
    new_min_area = min(new_area0, new_area1)

    # A triangle `_tri_metrics_3d` reports as near-zero-area always carries
    # aspect == inf, so that is reused instead of re-deriving the threshold.
    if (cur_aspect0 == float("inf") or cur_aspect1 == float("inf")) and new_min_area > cur_min_area + 1e-12:
        return True

    cur_worst_aspect = max(cur_aspect0, cur_aspect1)
    new_worst_aspect = max(new_aspect0, new_aspect1)
    if cur_worst_aspect > ratio_threshold and new_worst_aspect < cur_worst_aspect - 1e-6:
        return True

    return _in_circumcircle(pts2d, a, b, c, d) and new_worst_aspect <= cur_worst_aspect + 1e-6


def _unify_repair_3d(pts2d, xyz, tris, constrained, ratio_threshold=6.0,
                     edge_map=None, seed_edges=None, edge_ok=None):
    """One worklist sweep with one merged flip criterion (`_flip_gain_3d`), one
    edge map and one per-triangle metrics cache shared across every edge a
    triangle touches.

    The expensive combined test only runs for an edge whose triangle on either
    side is already flagged bad (degenerate or spiky), so the plain in-circle
    criterion inside `_flip_gain_3d` fires only on an edge that needs attention
    anyway. That is enough because the incremental Delaunay insertion already
    keeps the mesh close to 2D-optimal; this pass fixes what insertion order
    alone could not.

    `edge_map`, when passed in (built and flip-patched by a prior call on the
    same `tris`), is reused instead of rebuilt: the one follow-up pass after
    `_relocate_bad_points_3d` moves a few points then only pays for what needs
    re-examining.

    `seed_edges`, when given, restricts the initial worklist to that subset
    (the edges touching a just-moved vertex; every other edge's verdict has not
    changed).

    Returns `(total_flipped, bad_final, edge_map)`: `bad_final` is
    `(a, b, c, touches_boundary)` for every triangle still bad once the
    worklist drains; `edge_map` is handed back (patched, still valid) for a
    follow-up call.
    """
    if edge_map is None:
        edge_map = {}
        for ti, t in enumerate(tris):
            for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                edge_map.setdefault(frozenset(e), []).append(ti)

    metrics_cache: dict = {}

    def metrics(ti):
        v = metrics_cache.get(ti)
        if v is None:
            a, b, c = tris[ti]
            v = _tri_metrics_3d(xyz, a, b, c, ratio_threshold)
            metrics_cache[ti] = v
        return v

    queue = deque(seed_edges if seed_edges is not None else edge_map.keys())
    queued = set(queue)
    budget = 25 * max(len(edge_map), 1)
    total_flipped = 0
    examined = 0
    while queue and examined < budget:
        examined += 1
        e = queue.popleft()
        queued.discard(e)
        ts = edge_map.get(e)
        if not ts or len(ts) != 2 or e in constrained:
            continue
        t0, t1 = ts
        area0, aspect0, bad0 = metrics(t0)
        area1, aspect1, bad1 = metrics(t1)
        if not (bad0 or bad1):
            continue
        a, b = _directed_edge(tris[t0], *tuple(e))
        c = _third(tris[t0], a, b)
        d = _third(tris[t1], a, b)
        if c is None or d is None or c == d:
            continue
        if _orient(pts2d[a], pts2d[c], pts2d[d]) * _orient(pts2d[b], pts2d[c], pts2d[d]) >= 0:
            continue
        if _orient(pts2d[c], pts2d[a], pts2d[b]) * _orient(pts2d[d], pts2d[a], pts2d[b]) >= 0:
            continue
        cd = frozenset((c, d))
        if cd in edge_map:
            # c and d already share a DIFFERENT edge elsewhere in this face:
            # flipping would create a second, disconnected {c, d} edge, two
            # unrelated triangle pairs claiming one vertex pair, which is not a
            # valid 2-manifold (non-manifold edges, count > 2). The local 2D
            # convexity checks above cannot see that.
            continue
        if not _flip_gain_3d(pts2d, xyz, a, b, c, d, area0, aspect0, area1, aspect1,
                             ratio_threshold):
            continue
        if edge_ok is not None and not edge_ok(a, b, c, d):
            continue
        _flip_edge(tris, edge_map, e, t0, t1, a, b, c, d)
        metrics_cache.pop(t0, None)
        metrics_cache.pop(t1, None)
        total_flipped += 1
        for ti in (t0, t1):
            ta, tb, tc = tris[ti]
            for e2 in (frozenset((ta, tb)), frozenset((tb, tc)), frozenset((tc, ta))):
                if e2 in edge_map and e2 not in queued:
                    queue.append(e2)
                    queued.add(e2)

    bad_final = []
    for ti, t in enumerate(tris):
        _, _, bad = metrics(ti)
        if bad:
            a, b, c = t
            touches_boundary = any(frozenset(e) in constrained
                                   for e in ((a, b), (b, c), (c, a)))
            bad_final.append((t[0], t[1], t[2], touches_boundary))
    return total_flipped, bad_final, edge_map


def _relocate_bad_points_3d(uv, pts2d, xyz, tris, constrained, n_boundary,
                            surface, ext, su, sv, to_metric, deflection,
                            ratio_threshold=6.0, debug=False, face_label="",
                            log=print):
    """One pass over every currently near-zero-area triangle
    (`_tri_metrics_3d`'s aspect == inf), nudging each one's free interior
    (Steiner) vertex once. Boundary vertices (index < `n_boundary`) are never
    moved: they stay exactly where the edge sampler put them so the
    neighbouring face across that edge still welds. The step is capped at half
    the chord deflection.

    Returns the set of moved vertex indices, to seed a cheap follow-up
    `_unify_repair_3d` pass with the edges that could have a new verdict.
    """
    cap = (deflection * 0.5) if deflection else float("inf")
    step = max(min(ext * 2e-3, cap), 1e-9)

    # Vertex -> incident-triangle adjacency, built once. A move that flips a
    # triangle inside out in 2D (parameter space) would not be noticed, let
    # alone repaired, by the follow-up flip pass, so each move is verified: a
    # victim is kept only if every triangle it touches keeps its 2D orientation
    # sign, and reverted on the spot otherwise (an unchecked move produced
    # overlapping and duplicate triangles, i.e. non-manifold edges).
    vert_tris: dict = {}
    for ti, t in enumerate(tris):
        for v in t:
            vert_tris.setdefault(v, []).append(ti)

    def signs_for(victim):
        out = []
        for ti in vert_tris.get(victim, ()):
            p0, p1, p2 = tris[ti]
            out.append((ti, _orient(pts2d[p0], pts2d[p1], pts2d[p2]) >= 0))
        return out

    moved = set()
    # Only near-zero-area triangles are worth a nudge: a spiky triangle with
    # real area is a property of the point set (a rod thinner than the
    # tolerance is spiky by design), and a random step of a point in it only
    # wanders, carrying points across such a rod.
    bad_tris = [ti for ti in range(len(tris))
               if _tri_metrics_3d(xyz, *tris[ti], ratio_threshold)[1] == float("inf")]
    for ti in bad_tris:
        a, b, c = tris[ti]
        victim = next((v for v in (a, b, c) if v >= n_boundary), None)
        if victim is None or victim in moved:
            continue  # every vertex is a boundary point, or already nudged this round
        seed = victim * 2654435761 + 13
        ang = ((seed & 0xFFFF) / 65535.0) * TWO_PI
        mag = step
        if deflection:
            mag = min(mag, deflection)
        du_m = math.cos(ang) * mag
        dv_m = math.sin(ang) * mag
        u0, v0 = uv[victim]
        before = signs_for(victim)
        new_u = u0 + du_m / max(su, 1e-9)
        new_v = v0 + dv_m / max(sv, 1e-9)
        uv[victim] = (new_u, new_v)
        xyz_new = surface.eval(new_u, new_v)
        mu, mv = to_metric(((new_u, new_v),))[0]
        xyz_before = xyz[victim]
        pts2d_before = pts2d[victim]
        xyz[victim] = xyz_new
        pts2d[victim] = (mu, mv)
        if signs_for(victim) != before:
            # This move turned an incident triangle inside out (or exactly
            # degenerate) in 2D: revert instead of leaving a self-overlapping
            # patch that the follow-up flip pass cannot repair.
            uv[victim] = (u0, v0)
            xyz[victim] = xyz_before
            pts2d[victim] = pts2d_before
            continue
        moved.add(victim)
    if debug and moved:
        log(f"[StepForge] {face_label}: relocated {len(moved)} interior "
            f"point(s) in the single repair round ({len(bad_tris)} bad "
            f"triangle(s) seen)")
    return moved


def _tri_area3(xyz, a, b, c):
    ax, ay, az = xyz[a]
    bx, by, bz = xyz[b]
    cx, cy, cz = xyz[c]
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    return 0.5 * (nx * nx + ny * ny + nz * nz) ** 0.5


def _edge_len3(xyz, a, b):
    ax, ay, az = xyz[a]
    bx, by, bz = xyz[b]
    return ((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2) ** 0.5
