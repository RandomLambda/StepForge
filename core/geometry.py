"""Geometry: placements, curves and surfaces for STEP B-rep.

Pure Python + NumPy. Each Surface exposes:
    eval(u, v)            -> (3,) point
    invert(p)             -> (u, v) parameters for a 3D point on/near the surface
    periodic_u / period_u -> seam handling info (and likewise v)
    uv_step(deflection)   -> (du, dv) parameter step giving ~chord <= deflection

Each Curve exposes sample(p_start, p_end, n) returning 3D points; for trimmed
edges we project the edge vertices to parameters and sample between them.
"""
from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Low-level resolve helpers
# ---------------------------------------------------------------------------

def vec(seq) -> np.ndarray:
    return np.asarray(seq, dtype=float)


def get_point(sf, ref) -> np.ndarray | None:
    inst = sf.get(ref)
    if inst is None:
        return None
    # CARTESIAN_POINT(name, (x,y,z))
    coords = inst.params[1] if len(inst.params) > 1 else inst.params[0]
    return vec(coords)


def get_dir(sf, ref) -> np.ndarray | None:
    inst = sf.get(ref)
    if inst is None:
        return None
    coords = inst.params[1] if len(inst.params) > 1 else inst.params[0]
    d = vec(coords)
    n = np.linalg.norm(d)
    return d / n if n > 1e-12 else d


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


class Frame:
    """Right-handed local coordinate frame from AXIS2_PLACEMENT_3D."""
    __slots__ = ("o", "x", "y", "z")

    def __init__(self, o, z, x):
        self.o = o
        z = normalize(z)
        if x is None:
            # pick an arbitrary x perpendicular to z
            x = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(x, z)) > 0.9:
                x = np.array([0.0, 1.0, 0.0])
        # Gram-Schmidt x against z
        x = normalize(x - np.dot(x, z) * z)
        y = np.cross(z, x)
        self.x, self.y, self.z = x, y, z

    @classmethod
    def from_axis2(cls, sf, ref):
        inst = sf.get(ref)
        if inst is None:
            return cls(np.zeros(3), np.array([0, 0, 1.0]), np.array([1.0, 0, 0]))
        o = get_point(sf, inst.params[1]) if len(inst.params) > 1 else np.zeros(3)
        z = get_dir(sf, inst.params[2]) if len(inst.params) > 2 and inst.params[2] is not None else np.array([0, 0, 1.0])
        x = get_dir(sf, inst.params[3]) if len(inst.params) > 3 and inst.params[3] is not None else None
        if o is None:
            o = np.zeros(3)
        if z is None:
            z = np.array([0, 0, 1.0])
        return cls(o, z, x)

    def local(self, p):
        d = p - self.o
        return np.array([np.dot(d, self.x), np.dot(d, self.y), np.dot(d, self.z)])


# ===========================================================================
# Curves
# ===========================================================================

class Curve:
    period = TWO_PI
    # (t0, t1) for a bounded parametric curve (B-spline, polyline), else
    # None; `closed` if its two ends meet, so parameters wrap around.
    domain = None
    closed = False

    def param_of(self, p):  # default
        return 0.0

    def eval(self, t):
        raise NotImplementedError

    def eval_wrapped(self, t):
        """eval(t), with t wrapped into the domain on a closed curve."""
        if self.closed and self.domain is not None:
            t0, t1 = self.domain
            span = t1 - t0
            if span > 0 and not (t0 <= t <= t1):
                t = t0 + (t - t0) % span
        return self.eval(t)


class Line(Curve):
    def __init__(self, pnt, direction):
        self.p = pnt
        self.d = direction  # not necessarily unit (carries magnitude)

    def param_of(self, p):
        dd = np.dot(self.d, self.d)
        return float(np.dot(p - self.p, self.d) / dd) if dd > 1e-20 else 0.0

    def eval(self, t):
        return self.p + t * self.d


class Conic(Curve):
    period = TWO_PI

    def __init__(self, frame: Frame):
        self.f = frame

    def angle_of(self, p):
        loc = self.f.local(p)
        return math.atan2(loc[1], loc[0])

    def param_of(self, p):
        return self.angle_of(p)


class Circle(Conic):
    def __init__(self, frame, r):
        super().__init__(frame)
        self.r = r

    def eval(self, t):
        return self.f.o + self.r * (math.cos(t) * self.f.x + math.sin(t) * self.f.y)


class Ellipse(Conic):
    def __init__(self, frame, ra, rb):
        super().__init__(frame)
        self.ra, self.rb = ra, rb

    def angle_of(self, p):
        loc = self.f.local(p)
        return math.atan2(loc[1] / self.rb if self.rb else 0.0,
                          loc[0] / self.ra if self.ra else 0.0)

    def eval(self, t):
        return self.f.o + self.ra * math.cos(t) * self.f.x + self.rb * math.sin(t) * self.f.y


def _expand_knots(knots, mults):
    out = []
    for k, m in zip(knots, mults):
        out.extend([k] * int(m))
    return np.asarray(out, dtype=float)


def _find_span(n, degree, u, U):
    if u >= U[n + 1]:
        return n
    if u <= U[degree]:
        return degree
    lo, hi = degree, n + 1
    mid = (lo + hi) // 2
    while u < U[mid] or u >= U[mid + 1]:
        if u < U[mid]:
            hi = mid
        else:
            lo = mid
        mid = (lo + hi) // 2
    return mid


def _basis_funs(span, u, degree, U):
    # Plain lists and floats rather than NumPy: the degree is tiny and this
    # runs very often (grid samples, Newton iterations), where NumPy's
    # per-call overhead outweighs the arithmetic.
    N = [0.0] * (degree + 1)
    N[0] = 1.0
    left = [0.0] * (degree + 1)
    right = [0.0] * (degree + 1)
    for j in range(1, degree + 1):
        left[j] = u - U[span + 1 - j]
        right[j] = U[span + j] - u
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            temp = N[r] / denom if denom != 0 else 0.0
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved
    return N


class BSplineCurve(Curve):
    def __init__(self, degree, ctrlpts, knots, weights=None):
        self.p = degree
        self.P = np.asarray(ctrlpts, dtype=float)  # (n+1, 3)
        self.U = np.asarray(knots, dtype=float)
        self.n = len(self.P) - 1
        self.w = np.asarray(weights, dtype=float) if weights is not None else None
        self.umin = self.U[degree]
        self.umax = self.U[self.n + 1]
        # Plain-Python mirrors of the control points and weights for the hot
        # `eval()` loop (see `_basis_funs`).
        self._Pl = [tuple(p) for p in self.P]
        self._wl = list(self.w) if self.w is not None else None
        self.domain = (float(self.umin), float(self.umax))
        size = float(np.linalg.norm(self.P.max(axis=0) - self.P.min(axis=0))) if len(self.P) else 0.0
        self.closed = bool(np.linalg.norm(self.eval(self.umin) - self.eval(self.umax))
                           <= 1e-9 * max(size, 1e-9))

    def eval(self, u):
        u = min(max(u, self.umin), self.umax)
        span = _find_span(self.n, self.p, u, self.U)
        N = _basis_funs(span, u, self.p, self.U)
        idx = span - self.p
        Pl = self._Pl
        if self._wl is None:
            x = y = z = 0.0
            for i in range(self.p + 1):
                n = N[i]
                px, py, pz = Pl[idx + i]
                x += n * px
                y += n * py
                z += n * pz
            return np.array((x, y, z))
        wl = self._wl
        x = y = z = den = 0.0
        for i in range(self.p + 1):
            wn = N[i] * wl[idx + i]
            px, py, pz = Pl[idx + i]
            x += wn * px
            y += wn * py
            z += wn * pz
            den += wn
        if den != 0:
            return np.array((x / den, y / den, z / den))
        return np.array((x, y, z))

    def param_of(self, p):
        """Parameter of the curve point nearest `p`: coarse sampling, then
        Newton on the squared distance, so an edge starts and ends on its own
        vertices."""
        n = max(28, 4 * (self.n + 1))
        us = np.linspace(self.umin, self.umax, n, endpoint=not self.closed)
        best_u, best_d = self.umin, 1e30
        for u in us:
            d = np.sum((self.eval(u) - p) ** 2)
            if d < best_d:
                best_d, best_u = d, u
        if not self.closed:
            return _refine_curve_param(self, p, float(best_u), self.umin, self.umax)
        # A closed curve's parameter wraps: refine without the domain ends in
        # the way, then wrap back.
        span = self.umax - self.umin
        t = _refine_curve_param(_Wrapped(self), p, float(best_u),
                                float(best_u) - 0.5 * span, float(best_u) + 0.5 * span)
        return self.umin + (t - self.umin) % span


def _refine_curve_param(curve, p, t, lo, hi, iters=30):
    """Gauss-Newton on |C(t) - p|^2 from a starting parameter (tangent by
    finite differences, one-sided at the domain ends), clamped to [lo, hi],
    halving any step that does not bring the point closer."""
    span = hi - lo
    if span <= 0:
        return t
    h = span * 1e-7
    c = curve.eval(t)
    d = float(np.linalg.norm(c - p))
    for _ in range(iters):
        t0, t1 = max(t - h, lo), min(t + h, hi)
        if t1 <= t0:
            break
        d1 = (curve.eval(t1) - curve.eval(t0)) / (t1 - t0)
        g = float(np.dot(c - p, d1))
        H = float(np.dot(d1, d1))
        if H <= 1e-300:
            break
        step = max(-0.25 * span, min(0.25 * span, -g / H))
        improved = False
        for _ in range(8):
            t_new = min(max(t + step, lo), hi)
            c_new = curve.eval(t_new)
            d_new = float(np.linalg.norm(c_new - p))
            if d_new < d:
                improved = True
                break
            step *= 0.5
        if not improved:
            break
        moved = abs(t_new - t)
        t, c, d = t_new, c_new, d_new
        if moved < span * 1e-13:
            break
    return t


class _Wrapped:
    """eval() of a closed curve with its parameter wrapped into the domain."""

    def __init__(self, curve):
        self.c = curve

    def eval(self, t):
        return self.c.eval_wrapped(t)


class Polyline(Curve):
    """POLYLINE: straight segments through its points; parameter k + f on
    segment k."""

    def __init__(self, pts):
        self.P = np.asarray(pts, dtype=float)
        self.domain = (0.0, float(len(self.P) - 1))
        self.closed = bool(len(self.P) > 2 and np.linalg.norm(self.P[0] - self.P[-1]) < 1e-12)

    def eval(self, t):
        n = len(self.P) - 1
        t = min(max(t, 0.0), float(n))
        k = min(math.floor(t), n - 1)
        f = t - k
        return self.P[k] * (1.0 - f) + self.P[k + 1] * f

    def param_of(self, p):
        best = (1e300, 0.0)
        for k in range(len(self.P) - 1):
            a, b = self.P[k], self.P[k + 1]
            ab = b - a
            L2 = float(np.dot(ab, ab))
            f = 0.0 if L2 <= 0 else min(1.0, max(0.0, float(np.dot(p - a, ab)) / L2))
            d = float(np.linalg.norm(a + f * ab - p))
            if d < best[0]:
                best = (d, k + f)
        return best[1]


# ===========================================================================
# Surfaces
# ===========================================================================

class Surface:
    periodic_u = False
    periodic_v = False
    period_u = TWO_PI
    period_v = TWO_PI
    # True where the surface can collapse a parameter line to a point (a
    # sphere's poles, a cone's apex, a degenerate B-spline row).
    may_have_poles = False

    # Set by a RECTANGULAR_TRIMMED_SURFACE (see trimmed_domain); wins over
    # natural_domain() wherever the face's whole domain is needed.
    _domain_override = None
    # True if the STEP surface's normal is opposite to this object's Su x Sv
    # (a trimmed surface whose u and v senses disagree with its basis).
    sense_flip = False

    def natural_domain(self):
        """((u0, u1), (v0, v1)) if the surface has a finite parameter
        domain a face can use whole, else None (planes, cylinders, ...)."""
        return

    def domain(self):
        return self._domain_override or self.natural_domain()

    def eval(self, u, v):
        raise NotImplementedError

    def invert(self, p):
        raise NotImplementedError

    def grid_cells(self, deflection, uv_extent):
        """Return (nu_cells, nv_cells) param-rectangle subdivisions so each
        cell's chord error <= deflection. (1, 1) means no interior points."""
        return (1, 1)

    def local_scale(self, uv_extent):
        """Return (su, sv): per-axis multipliers so that, for small steps, 3D
        distance is about sqrt((du*su)^2 + (dv*sv)^2). The default (1, 1) is
        exact for a Plane. Curved surfaces override it: raw (u, v) is not
        isometric (a cylinder's u is an angle, its v a length), and a
        triangulation that sees (u, v) unscaled produces long, thin triangles.
        The triangulator works in scaled (u, v); `surface.eval` takes the real
        ones. Exact for developable surfaces (cylinder, cone), first order
        for the others.
        """
        return 1.0, 1.0


# Largest angle one chord may span on a circular edge or across one cell of a
# curved face, whatever the chord tolerance allows (pi/3: a full circle is at
# least a hexagon). Without it a radius at or below the tolerance collapses to
# a two-point "circle", and a face one cell wide gets triangles that run from
# the seam straight through the axis.
MAX_SEGMENT_ANGLE = math.pi / 3


# A cylinder or cone gets axial cells as wide as its angular ones (see
# `Cylinder.grid_cells`), at most this many, so a long thin rod is one set of
# long thin cells rather than hundreds of squares. 64 keeps the worst sliver
# ratio of the reference parts unchanged; 32 adds slivers on long tubes.
MAX_AXIAL_CELLS = 64


def _angular_cells(extent, radius, deflection, cap=512):
    if radius <= 1e-9:
        return 1
    ratio = max(-1.0, min(1.0, 1.0 - deflection / radius))
    dtheta = min(max(2.0 * math.acos(ratio), 1e-3), MAX_SEGMENT_ANGLE)
    return int(max(1, min(cap, math.ceil(abs(extent) / dtheta))))


class Plane(Surface):
    def __init__(self, frame: Frame):
        self.f = frame

    def eval(self, u, v):
        return self.f.o + u * self.f.x + v * self.f.y

    def invert(self, p):
        loc = self.f.local(p)
        return loc[0], loc[1]


class Cylinder(Surface):
    periodic_u = True

    def __init__(self, frame: Frame, r):
        self.f = frame
        self.r = r

    def eval(self, u, v):
        return self.f.o + self.r * (math.cos(u) * self.f.x + math.sin(u) * self.f.y) + v * self.f.z

    def invert(self, p):
        loc = self.f.local(p)
        return math.atan2(loc[1], loc[0]), loc[2]

    def grid_cells(self, deflection, uv_extent):
        (umin, vmin), (umax, vmax) = uv_extent
        nu = _angular_cells(umax - umin, self.r, deflection)
        # v runs along the axis (zero curvature), so accuracy alone would
        # allow nv=1. Matching v's cell size to u's real angular cell size
        # keeps interior cells roughly square, which triangle quality needs;
        # capped at MAX_AXIAL_CELLS rows. The straight boundary edges along v
        # get matching sample spacing from `convert.py`'s boundary-density
        # pass and `_adaptive_grid_uv`'s pass 1b.
        world_step = max(self.r * (umax - umin) / nu, 1e-6)
        nv = int(max(1, min(MAX_AXIAL_CELLS, math.ceil((vmax - vmin) / world_step))))
        return (nu, nv)

    def local_scale(self, uv_extent):
        # u is radians: times r it is arc length. A cylinder is developable,
        # so this unrolling is an exact isometry.
        return self.r, 1.0


class Cone(Surface):
    periodic_u = True
    may_have_poles = True

    def __init__(self, frame: Frame, r_ref, semi_angle):
        self.f = frame
        self.r_ref = r_ref
        self.tan = math.tan(semi_angle)

    def radius_at(self, v):
        return self.r_ref + v * self.tan

    def eval(self, u, v):
        r = self.radius_at(v)
        return self.f.o + r * (math.cos(u) * self.f.x + math.sin(u) * self.f.y) + v * self.f.z

    def invert(self, p):
        loc = self.f.local(p)
        return math.atan2(loc[1], loc[0]), loc[2]

    def grid_cells(self, deflection, uv_extent):
        (umin, vmin), (umax, vmax) = uv_extent
        rmax = max(abs(self.radius_at(vmin)), abs(self.radius_at(vmax)), 1e-6)
        nu = _angular_cells(umax - umin, rmax, deflection)
        # Same square-cell coupling as Cylinder.grid_cells.
        world_step = max(rmax * (umax - umin) / nu, 1e-6)
        nv = int(max(1, min(MAX_AXIAL_CELLS, math.ceil((vmax - vmin) / world_step))))
        return (nu, nv)

    def local_scale(self, uv_extent):
        (_, vmin), (_, vmax) = uv_extent
        rmid = max(self.radius_at((vmin + vmax) * 0.5), 1e-6)
        # Also developable, but the slant surface is longer than the axial
        # v-step by 1/cos(semi-angle); u uses the radius at mid-v.
        slant = math.sqrt(1.0 + self.tan * self.tan)
        return rmid, slant


class Sphere(Surface):
    periodic_u = True
    may_have_poles = True

    def __init__(self, frame: Frame, r):
        self.f = frame
        self.r = r

    def natural_domain(self):
        return (-math.pi, math.pi), (-math.pi / 2, math.pi / 2)

    def eval(self, u, v):
        # u = longitude, v = latitude (-pi/2..pi/2)
        cv = math.cos(v)
        return self.f.o + self.r * (cv * math.cos(u) * self.f.x +
                                    cv * math.sin(u) * self.f.y +
                                    math.sin(v) * self.f.z)

    def invert(self, p):
        loc = self.f.local(p)
        u = math.atan2(loc[1], loc[0])
        v = math.atan2(loc[2], math.hypot(loc[0], loc[1]))
        return u, v

    def grid_cells(self, deflection, uv_extent):
        (umin, vmin), (umax, vmax) = uv_extent
        nu = _angular_cells(umax - umin, self.r, deflection)
        nv = _angular_cells(vmax - vmin, self.r, deflection)
        return (nu, nv)

    def local_scale(self, uv_extent):
        # Not developable: equirectangular first-order approximation at the
        # extent's mid-latitude (u shrinks by cos(latitude), v is a real
        # great-circle distance), good enough for the small patches of real faces.
        (_, vmin), (_, vmax) = uv_extent
        vmid = (vmin + vmax) * 0.5
        return self.r * max(math.cos(vmid), 1e-3), self.r


class Torus(Surface):
    periodic_u = True
    periodic_v = True

    def __init__(self, frame: Frame, major_r, minor_r):
        self.f = frame
        self.R = major_r
        self.r = minor_r

    def natural_domain(self):
        return (0.0, TWO_PI), (0.0, TWO_PI)

    def eval(self, u, v):
        # u around main axis, v around tube
        ring = (self.R + self.r * math.cos(v))
        return self.f.o + ring * (math.cos(u) * self.f.x + math.sin(u) * self.f.y) + \
            self.r * math.sin(v) * self.f.z

    def invert(self, p):
        loc = self.f.local(p)
        u = math.atan2(loc[1], loc[0])
        # distance from main axis in plane
        rho = math.hypot(loc[0], loc[1]) - self.R
        v = math.atan2(loc[2], rho)
        return u, v

    def grid_cells(self, deflection, uv_extent):
        (umin, vmin), (umax, vmax) = uv_extent
        nu = _angular_cells(umax - umin, self.R + self.r, deflection)
        nv = _angular_cells(vmax - vmin, self.r, deflection)
        return (nu, nv)

    def local_scale(self, uv_extent):
        # Not developable. u's real arc length follows the ring radius
        # R + r*cos(v), taken at the extent's mid-v; v is a real tube arc length.
        (_, vmin), (_, vmax) = uv_extent
        vmid = (vmin + vmax) * 0.5
        return max(self.R + self.r * math.cos(vmid), 1e-6), self.r


class BSplineSurface(Surface):
    may_have_poles = True

    def natural_domain(self):
        return (self.umin, self.umax), (self.vmin, self.vmax)

    def __init__(self, deg_u, deg_v, ctrlgrid, knots_u, knots_v, weights=None):
        self.pu = deg_u
        self.pv = deg_v
        self.P = np.asarray(ctrlgrid, dtype=float)  # (nu+1, nv+1, 3)
        self.U = np.asarray(knots_u, dtype=float)
        self.V = np.asarray(knots_v, dtype=float)
        self.nu = self.P.shape[0] - 1
        self.nv = self.P.shape[1] - 1
        self.w = np.asarray(weights, dtype=float) if weights is not None else None
        self.umin, self.umax = self.U[deg_u], self.U[self.nu + 1]
        self.vmin, self.vmax = self.V[deg_v], self.V[self.nv + 1]
        self._grid_cache = None
        # Plain-Python mirror of the control grid and weights for the hot
        # `eval()` loop (see `_basis_funs`).
        self._Pl = [[tuple(self.P[a, b]) for b in range(self.P.shape[1])]
                    for a in range(self.P.shape[0])]
        self._wl = ([[float(self.w[a, b]) for b in range(self.w.shape[1])]
                     for a in range(self.w.shape[0])]
                    if self.w is not None else None)

    def eval(self, u, v):
        u = min(max(u, self.umin), self.umax)
        v = min(max(v, self.vmin), self.vmax)
        su = _find_span(self.nu, self.pu, u, self.U)
        sv = _find_span(self.nv, self.pv, v, self.V)
        Nu = _basis_funs(su, u, self.pu, self.U)
        Nv = _basis_funs(sv, v, self.pv, self.V)
        iu = su - self.pu
        iv = sv - self.pv
        Pl = self._Pl
        if self._wl is None:
            x = y = z = 0.0
            for a in range(self.pu + 1):
                row = Pl[iu + a]
                na = Nu[a]
                for b in range(self.pv + 1):
                    n = na * Nv[b]
                    px, py, pz = row[iv + b]
                    x += n * px
                    y += n * py
                    z += n * pz
            return np.array((x, y, z))
        wl = self._wl
        x = y = z = den = 0.0
        for a in range(self.pu + 1):
            row = Pl[iu + a]
            wrow = wl[iu + a]
            na = Nu[a]
            for b in range(self.pv + 1):
                n = na * Nv[b]
                wn = n * wrow[iv + b]
                px, py, pz = row[iv + b]
                x += wn * px
                y += wn * py
                z += wn * pz
                den += wn
        if den != 0:
            return np.array((x / den, y / den, z / den))
        return np.array((x, y, z))

    def invert(self, p, iters=16):
        """Nearest (u, v) to `p`: seed from a sampled grid, then damped Newton.

        The seed grid is deliberately fine and the step damped rather than a
        plain solve: on a near-singular Jacobian (a flat or degenerate spot of
        a fitted surface) a failed solve returned the seed unchanged, so two
        different boundary points could get the same (u, v) and produce
        zero-area triangles."""
        if self._grid_cache is None:
            gu = np.linspace(self.umin, self.umax, 21)
            gv = np.linspace(self.vmin, self.vmax, 21)
            pts = [[self.eval(u, v) for v in gv] for u in gu]
            # One flat (441, 3) array per surface: the seed search below is a
            # single vectorized distance computation instead of 441 NumPy calls.
            flat_pts = np.array([p for row in pts for p in row])  # (441, 3)
            flat_uv = [(u, v) for u in gu for v in gv]
            self._grid_cache = (gu, gv, pts, flat_pts, flat_uv)
        gu, gv, pts, flat_pts, flat_uv = self._grid_cache
        d2 = np.sum((flat_pts - p) ** 2, axis=1)
        best = flat_uv[int(np.argmin(d2))]
        u, v = best
        du = (self.umax - self.umin) * 1e-4 + 1e-9
        dv = (self.vmax - self.vmin) * 1e-4 + 1e-9
        for _ in range(iters):
            s = self.eval(u, v)
            r = s - p
            su_ = (self.eval(u + du, v) - s) / du
            sv_ = (self.eval(u, v + dv) - s) / dv
            J = np.array([[np.dot(su_, su_), np.dot(su_, sv_)],
                          [np.dot(su_, sv_), np.dot(sv_, sv_)]])
            g = np.array([np.dot(r, su_), np.dot(r, sv_)])
            # Levenberg-style damping: a near-singular J gives a small finite
            # step instead of an exception.
            damp = 1e-9 * (float(J[0, 0] + J[1, 1]) + 1e-30)
            try:
                delta = np.linalg.solve(J + np.eye(2) * damp, -g)
            except np.linalg.LinAlgError:
                try:
                    delta = np.linalg.lstsq(J, -g, rcond=None)[0]
                except np.linalg.LinAlgError:
                    break
            u = min(max(u + delta[0], self.umin), self.umax)
            v = min(max(v + delta[1], self.vmin), self.vmax)
            if abs(delta[0]) < du and abs(delta[1]) < dv:
                break
        return u, v

    def grid_cells(self, deflection, uv_extent):
        """Base grid for the adaptive tessellation of a fitted patch.

        The ceiling is `(nu+1)*2` cells per axis (capped at 48), from the
        control-point count. At that size a flat patch spanning metres would
        cost thousands of triangles whatever `deflection` is, so the curvature
        is estimated numerically at nine points, turned with `deflection` into
        a chord length via the sagitta relation s ~ kappa L^2 / 8, and the real
        extent divided by it. The result only ever shrinks the ceiling
        (`min(ceiling, needed)`): `_adaptive_grid_uv` refines curved cells
        itself, and a base sized to the curvature's own scale costs far more
        total refinement than a coarse uniform one. So it steps in only for a
        patch that is flat or gently curved over its whole extent."""
        (umin, vmin), (umax, vmax) = uv_extent
        du = max(umax - umin, 1e-12)
        dv = max(vmax - vmin, 1e-12)
        su, sv = self.local_scale(uv_extent)
        old_nu = int(max(2, min(48, (self.nu + 1) * 2)))
        old_nv = int(max(2, min(48, (self.nv + 1) * 2)))

        # Sagitta (bilinear-midpoint deviation, as in `_adaptive_grid_uv`'s
        # flatness check) at nine spots, with a step that is a fixed fraction
        # of the patch's extent so the estimate scales with the patch.
        eps_u = du * 0.02
        eps_v = dv * 0.02
        max_ku = max_kv = 0.0
        for fu in (0.25, 0.5, 0.75):
            for fv in (0.25, 0.5, 0.75):
                u = umin + du * fu
                v = vmin + dv * fv
                p0 = self.eval(u, v)
                pu0 = self.eval(max(u - eps_u, umin), v)
                pu1 = self.eval(min(u + eps_u, umax), v)
                pv0 = self.eval(u, max(v - eps_v, vmin))
                pv1 = self.eval(u, min(v + eps_v, vmax))
                sag_u = float(np.linalg.norm((pu0 + pu1) * 0.5 - p0))
                sag_v = float(np.linalg.norm((pv0 + pv1) * 0.5 - p0))
                # sagitta s ~= kappa * eps^2 / 2 (eps in REAL units) =>
                # kappa ~= 2*s / eps_real^2
                eps_u_real = eps_u * su
                eps_v_real = eps_v * sv
                if eps_u_real > 1e-12:
                    max_ku = max(max_ku, 2.0 * sag_u / (eps_u_real ** 2))
                if eps_v_real > 1e-12:
                    max_kv = max(max_kv, 2.0 * sag_v / (eps_v_real ** 2))

        def cells_for(old_n, extent_real, kappa):
            if kappa <= 1e-9 or extent_real <= 1e-12:
                # No measurable curvature (a flat patch): shrink hard, but
                # not to 1 -- `_adaptive_grid_uv` can still refine up to 8x
                # to catch a sharp feature the nine samples missed.
                return max(1, old_n // 8)
            # chord length L so a bulge of curvature kappa sagging across it
            # deviates by at most `deflection`: L = sqrt(8 * deflection / kappa)
            chord = math.sqrt(8.0 * max(deflection, 1e-9) / kappa)
            needed = int(max(1, math.ceil(extent_real / max(chord, 1e-9))))
            return min(old_n, needed)

        nu = cells_for(old_nu, du * su, max_ku)
        nv = cells_for(old_nv, dv * sv, max_kv)
        return (nu, nv)

    def local_scale(self, uv_extent):
        # No closed-form metric for a general NURBS patch: average |dP/du| and
        # |dP/dv| by finite differences at nine points of the face's own
        # uv_extent (a trimmed face can sit in a small corner of a larger
        # patch where the metric differs from the average).
        (umin, vmin), (umax, vmax) = uv_extent
        du = max(umax - umin, 1e-9)
        dv = max(vmax - vmin, 1e-9)
        eps_u = du * 1e-3
        eps_v = dv * 1e-3
        su_vals = []
        sv_vals = []
        for fu in (0.25, 0.5, 0.75):
            for fv in (0.25, 0.5, 0.75):
                u = umin + du * fu
                v = vmin + dv * fv
                p0 = self.eval(u, v)
                pu = self.eval(min(u + eps_u, self.umax), v)
                pv = self.eval(u, min(v + eps_v, self.vmax))
                su_vals.append(np.linalg.norm(pu - p0) / eps_u)
                sv_vals.append(np.linalg.norm(pv - p0) / eps_v)
        su = sum(su_vals) / len(su_vals) if su_vals else 1.0
        sv = sum(sv_vals) / len(sv_vals) if sv_vals else 1.0
        return max(su, 1e-6), max(sv, 1e-6)


# ===========================================================================
# Swept and offset surfaces (numeric: no closed-form inverse)
# ===========================================================================

def _numeric_local_scale(surface, uv_extent):
    """Average |dS/du| and |dS/dv| over a 3x3 sample of the extent."""
    (umin, vmin), (umax, vmax) = uv_extent
    du = max(umax - umin, 1e-9)
    dv = max(vmax - vmin, 1e-9)
    eu, ev = du * 1e-3, dv * 1e-3
    su_vals, sv_vals = [], []
    for fu in (0.25, 0.5, 0.75):
        for fv in (0.25, 0.5, 0.75):
            u = umin + du * fu
            v = vmin + dv * fv
            p0 = surface.eval(u, v)
            su_vals.append(np.linalg.norm(surface.eval(u + eu, v) - p0) / eu)
            sv_vals.append(np.linalg.norm(surface.eval(u, v + ev) - p0) / ev)
    return (max(sum(su_vals) / len(su_vals), 1e-6),
            max(sum(sv_vals) / len(sv_vals), 1e-6))


def _numeric_grid_cells(surface, deflection, uv_extent, cap=96):
    """Cells per direction so a chord of one cell sags at most `deflection`,
    from curvature sampled across the extent (sagitta s ~ kappa L^2 / 8)."""
    (umin, vmin), (umax, vmax) = uv_extent
    du = max(umax - umin, 1e-12)
    dv = max(vmax - vmin, 1e-12)
    su, sv = surface.local_scale(uv_extent)
    eu, ev = du * 0.02, dv * 0.02
    ku = kv = 0.0
    for fu in (0.1, 0.3, 0.5, 0.7, 0.9):
        for fv in (0.1, 0.3, 0.5, 0.7, 0.9):
            u = umin + du * fu
            v = vmin + dv * fv
            p0 = surface.eval(u, v)
            sag_u = np.linalg.norm((surface.eval(u - eu, v) + surface.eval(u + eu, v)) * 0.5 - p0)
            sag_v = np.linalg.norm((surface.eval(u, v - ev) + surface.eval(u, v + ev)) * 0.5 - p0)
            if eu * su > 1e-12:
                ku = max(ku, 2.0 * float(sag_u) / (eu * su) ** 2)
            if ev * sv > 1e-12:
                kv = max(kv, 2.0 * float(sag_v) / (ev * sv) ** 2)

    def cells(extent_real, kappa):
        if kappa <= 1e-12 or extent_real <= 1e-12:
            return 1
        # angular cap as for circles: at most MAX_SEGMENT_ANGLE of turn per cell
        chord = min(math.sqrt(8.0 * max(deflection, 1e-9) / kappa),
                    MAX_SEGMENT_ANGLE / kappa)
        return int(max(1, min(cap, math.ceil(extent_real / max(chord, 1e-12)))))

    return cells(du * su, ku), cells(dv * sv, kv)


def _curve_period(curve):
    """Parameter period of a closed curve, else None."""
    if isinstance(curve, Conic):
        return TWO_PI
    if curve.closed and curve.domain is not None:
        return curve.domain[1] - curve.domain[0]
    return None


class SurfaceOfLinearExtrusion(Surface):
    """S(u, v) = C(u) + v * d  (ISO 10303-42 surface_of_linear_extrusion;
    d is the extrusion VECTOR including its magnitude)."""

    def __init__(self, curve: Curve, d):
        self.c = curve
        self.d = np.asarray(d, dtype=float)
        self._dd = float(np.dot(self.d, self.d)) or 1.0
        per = _curve_period(curve)
        if per:
            self.periodic_u = True
            self.period_u = per

    def eval(self, u, v):
        return self.c.eval_wrapped(u) + v * self.d

    def invert(self, p):
        u = self.c.param_of(p)
        v = 0.0
        for _ in range(3):
            v = float(np.dot(p - self.c.eval_wrapped(u), self.d)) / self._dd
            u = self.c.param_of(p - v * self.d)
        v = float(np.dot(p - self.c.eval_wrapped(u), self.d)) / self._dd
        return u, v

    def local_scale(self, uv_extent):
        return _numeric_local_scale(self, uv_extent)

    def grid_cells(self, deflection, uv_extent):
        nu, _ = _numeric_grid_cells(self, deflection, uv_extent)
        # straight along v: keep cells roughly square, as Cylinder does
        (umin, vmin), (umax, vmax) = uv_extent
        su, sv = self.local_scale(uv_extent)
        step = max(su * (umax - umin) / max(nu, 1), 1e-6)
        nv = int(max(1, min(512, math.ceil(sv * (vmax - vmin) / step))))
        return nu, nv


class SurfaceOfRevolution(Surface):
    """S(u, v) = C(v) rotated by angle u about the axis (ISO 10303-42
    surface_of_revolution: u is the angle, v the swept curve's parameter)."""
    periodic_u = True
    may_have_poles = True

    def __init__(self, curve: Curve, origin, axis):
        self.c = curve
        self.o = np.asarray(origin, dtype=float)
        self.a = normalize(np.asarray(axis, dtype=float))
        per = _curve_period(curve)
        if per:
            self.periodic_v = True
            self.period_v = per

    def _rot(self, x, ang):
        a = self.a
        ca, sa = math.cos(ang), math.sin(ang)
        return x * ca + np.cross(a, x) * sa + a * float(np.dot(a, x)) * (1.0 - ca)

    def eval(self, u, v):
        return self.o + self._rot(self.c.eval_wrapped(v) - self.o, u)

    def _radial(self, x):
        d = x - self.o
        return d - self.a * float(np.dot(d, self.a))

    def invert(self, p):
        rp = self._radial(p)
        v = self.c.param_of(p)
        u = 0.0
        for _ in range(4):
            rc = self._radial(self.c.eval_wrapped(v))
            nc, npn = np.linalg.norm(rc), np.linalg.norm(rp)
            if nc < 1e-12 or npn < 1e-12:
                u = 0.0
                break
            ec, ep = rc / nc, rp / npn
            u = math.atan2(float(np.dot(self.a, np.cross(ec, ep))), float(np.dot(ec, ep)))
            v_new = self.c.param_of(self.o + self._rot(p - self.o, -u))
            if abs(v_new - v) < 1e-12:
                v = v_new
                break
            v = v_new
        return u, v

    def natural_domain(self):
        if self.c.domain is None:
            return None
        return (0.0, TWO_PI), tuple(self.c.domain)

    def local_scale(self, uv_extent):
        return _numeric_local_scale(self, uv_extent)

    def grid_cells(self, deflection, uv_extent):
        return _numeric_grid_cells(self, deflection, uv_extent)


class OffsetSurface(Surface):
    """S(u, v) = B(u, v) + d * N(u, v), N the unit normal Bu x Bv of the
    basis surface. Analytic bases are replaced by their exact offset in
    `make_offset_surface`; this numeric form covers the rest."""

    def __init__(self, basis: Surface, dist):
        self.b = basis
        self.dist = float(dist)
        self.periodic_u = basis.periodic_u
        self.periodic_v = basis.periodic_v
        self.period_u = basis.period_u
        self.period_v = basis.period_v
        self.may_have_poles = getattr(basis, "may_have_poles", False)
        dom = basis.domain()
        self._h = 1e-6 * max(((dom[0][1] - dom[0][0]) + (dom[1][1] - dom[1][0])) if dom else 1.0, 1e-9)

    def _normal(self, u, v):
        h = self._h
        for _ in range(4):
            du = self.b.eval(u + h, v) - self.b.eval(u - h, v)
            dv = self.b.eval(u, v + h) - self.b.eval(u, v - h)
            n = np.cross(du, dv)
            ln = np.linalg.norm(n)
            if ln > 1e-300:
                return n / ln
            u, v = u + 10 * h, v + 10 * h   # singular point: step off it
        return np.zeros(3)

    def eval(self, u, v):
        return self.b.eval(u, v) + self.dist * self._normal(u, v)

    def invert(self, p):
        # The foot of the normal from a point of the offset surface is the
        # same (u, v) on the basis.
        return self.b.invert(p)

    def natural_domain(self):
        return self.b.domain()

    def local_scale(self, uv_extent):
        return _numeric_local_scale(self, uv_extent)

    def grid_cells(self, deflection, uv_extent):
        return _numeric_grid_cells(self, deflection, uv_extent)


def make_offset_surface(basis: Surface, dist: float) -> Surface:
    """Exact offset for the analytic bases, numeric OffsetSurface otherwise.
    STEP's offset direction is the basis normal Bu x Bv, which points away
    from the axis/centre for StepForge's cylinder, cone, sphere and torus
    parameterisations and along +z for a plane."""
    if isinstance(basis, Plane):
        f = basis.f
        return Plane(Frame(f.o + dist * f.z, f.z, f.x))
    if isinstance(basis, Cylinder) and basis.r + dist > 0:
        return Cylinder(basis.f, basis.r + dist)
    if isinstance(basis, Sphere) and basis.r + dist > 0:
        return Sphere(basis.f, basis.r + dist)
    if isinstance(basis, Torus) and basis.r + dist > 0:
        return Torus(basis.f, basis.R, basis.r + dist)
    return OffsetSurface(basis, dist)


def trimmed_domain(surface: Surface, u1, u2, v1, v2):
    """RECTANGULAR_TRIMMED_SURFACE: the basis surface with its natural
    domain set to the trim rectangle (the face's own loops still trim it)."""
    surface._domain_override = ((min(u1, u2), max(u1, u2)), (min(v1, v2), max(v1, v2)))
    return surface
