"""Convert a parsed STEP file into welded triangle meshes, one per solid,
named from the owning PRODUCT. Pure Python + NumPy.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from . import geometry as G
from . import tessellate as T
from . import workers as W
from .parser import Instance, StepFile, parse_file


@dataclass
class Mesh:
    verts: list[tuple[float, float, float]] = field(default_factory=list)
    faces: list[tuple[int, ...]] = field(default_factory=list)
    # STEP ADVANCED_FACE id each triangle came from (0 = unknown), parallel
    # to `faces`: carries per-face colours and the source-face attribute.
    face_ids: list[int] = field(default_factory=list)

    def add_triangles(self, verts3d, tris, weld, face_id=0):
        idx_map = {}
        for i, v in enumerate(verts3d):
            key = (round(v[0] * weld), round(v[1] * weld), round(v[2] * weld))
            gi = self._global.get(key)
            if gi is None:
                gi = len(self.verts)
                self.verts.append((float(v[0]), float(v[1]), float(v[2])))
                self._global[key] = gi
            idx_map[i] = gi
        for (a, b, c) in tris:
            ia, ib, ic = idx_map[a], idx_map[b], idx_map[c]
            if ia == ib or ib == ic or ia == ic:
                continue
            self.faces.append((ia, ib, ic))
            self.face_ids.append(int(face_id))

    _global: dict = field(default_factory=dict, repr=False)


def _stitch_boundary_gaps(mesh: Mesh, deflection: float | None = None):
    """Close small gaps where two faces meet along the same physical curve but
    do not share vertices (e.g. a full cylinder's rim and its end cap each
    sample the same circle from their own reference angle).

    Only vertices that are already exposed are candidates: those on an edge
    used by exactly one triangle (a real gap; a watertight edge is used by
    two). So it can close gaps, never erase real geometry. The merge distance
    is locally adaptive (a fraction of each candidate's own boundary edge
    length), so it scales from a 1 mm bore to a 1 m cylinder. A no-op on a
    watertight mesh, which includes every default Tessellated-mode export."""
    faces = mesh.faces
    if not faces:
        return
    edge_count: dict[tuple[int, int], int] = {}
    for (a, b, c) in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_count[key] = edge_count.get(key, 0) + 1
    boundary_verts = set()
    for (u, v), cnt in edge_count.items():
        if cnt == 1:
            boundary_verts.add(u)
            boundary_verts.add(v)
    if len(boundary_verts) < 2:
        return  # already watertight (or a single dangling vertex, not a gap)

    verts = mesh.verts
    # Local scale = the length of the vertex's own open (boundary) edges: two
    # samplings of one curve are offset by at most about one of those.
    edge_lens: dict[int, list[float]] = {v: [] for v in boundary_verts}
    for (u, v), cnt in edge_count.items():
        if cnt == 1:
            d = math.dist(verts[u], verts[v])
            edge_lens[u].append(d)
            edge_lens[v].append(d)

    bv_list = list(boundary_verts)
    local_scale = {v: (sum(edge_lens[v]) / len(edge_lens[v]) if edge_lens[v] else 0.0)
                   for v in bv_list}
    parent = {v: v for v in bv_list}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # Which physical boundary RING (one face's own loop of used-once edges)
    # each candidate vertex belongs to, via a separate union-find over the
    # boundary-edge graph. A real gap is two vertices of two DIFFERENT rings at
    # almost the same point; two vertices adjacent along the same ring are
    # joined by a real boundary edge and are never a gap. Without this check
    # the search would close a fine ring onto its own neighbour, and
    # single-linkage chaining would collapse long runs of a legitimate boundary
    # into one hub vertex.
    ring_parent = {v: v for v in bv_list}

    def ring_find(x):
        while ring_parent[x] != x:
            ring_parent[x] = ring_parent[ring_parent[x]]
            x = ring_parent[x]
        return x

    for (u, v), cnt in edge_count.items():
        if cnt == 1:
            ru, rv = ring_find(u), ring_find(v)
            if ru != rv:
                ring_parent[rv] = ru

    pts = np.array([verts[v] for v in bv_list])
    # Bucket candidates into a uniform grid instead of an O(n^2) scan.
    # `thresh` for any pair is 1.5 * max(local_scale[vi], local_scale[vj]) and
    # so never exceeds `cell` (1.5 * the global max local_scale); a match is
    # therefore always within vi's own grid cell or one of its 26 neighbours.
    max_scale = max(local_scale.values(), default=0.0)
    cell = 1.5 * max_scale if max_scale > 0.0 else 1.0
    grid: dict[tuple[int, int, int], list[int]] = {}

    def cell_of(p):
        return (math.floor(p[0] / cell), math.floor(p[1] / cell),
                math.floor(p[2] / cell))

    for i in range(len(bv_list)):
        grid.setdefault(cell_of(pts[i]), []).append(i)

    # Candidate pairs first, then merge closest-first into clusters whose
    # diameter stays within the pair threshold. Merging each vertex straight
    # into its nearest neighbour's set lets single-linkage chains run along a
    # boundary: next to a face that failed to tessellate (a real hole, not a
    # seam gap) whole rims collapsed onto one vertex. With a chord tolerance
    # known, a pair must also be a real seam gap: two samplings of one curve,
    # so each vertex lies on the other ring's polyline to within the chord
    # tolerance. Vertices on opposite sides of a hole are near each other but
    # not on each other's rim, and must stay apart.
    ring_nbrs: dict[int, list[int]] = {}
    for (u, v), cnt in edge_count.items():
        if cnt == 1:
            ring_nbrs.setdefault(u, []).append(v)
            ring_nbrs.setdefault(v, []).append(u)
    on_tol = 2.0 * deflection if deflection else None

    def _seg_dist(p, a, b):
        ab = b - a
        L2 = float(np.dot(ab, ab))
        t = 0.0 if L2 <= 0.0 else min(1.0, max(0.0, float(np.dot(p - a, ab)) / L2))
        return float(np.linalg.norm(p - (a + t * ab)))

    def _on_rim(p, v):
        a = np.asarray(verts[v], dtype=float)
        return any(_seg_dist(p, a, np.asarray(verts[w], dtype=float)) <= on_tol
                   for w in ring_nbrs.get(v, ()))

    root_index = {v: k for k, v in enumerate(bv_list)}
    pairs = []
    for i, vi in enumerate(bv_list):
        if local_scale[vi] <= 0.0:
            continue
        vi_ring = ring_find(vi)
        cx, cy, cz = cell_of(pts[i])
        candidates = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    candidates.update(grid.get((cx + dx, cy + dy, cz + dz), ()))
        for j in candidates:
            if j <= i:
                continue
            vj = bv_list[j]
            if ring_find(vj) == vi_ring:
                continue  # same physical ring -- never a gap, see above
            thresh = 1.5 * max(local_scale[vi], local_scale[vj])
            if thresh <= 0.0:
                continue
            d = float(np.linalg.norm(pts[i] - pts[j]))
            if d >= thresh:
                continue
            if on_tol is not None and not (_on_rim(pts[i], vj) or _on_rim(pts[j], vi)):
                continue
            pairs.append((d, i, j, thresh))
    pairs.sort()
    members = {i: [i] for i in range(len(bv_list))}
    for d, i, j, thresh in pairs:
        ri, rj = find(bv_list[i]), find(bv_list[j])
        if ri == rj:
            continue
        mi = members[root_index[ri]]
        mj = members[root_index[rj]]
        if any(float(np.linalg.norm(pts[a] - pts[b])) >= thresh for a in mi for b in mj):
            continue
        union(ri, rj)
        keep, drop = (root_index[ri], root_index[rj])
        members[keep] = mi + mj
        members[drop] = []

    remap = {v: find(v) for v in bv_list}
    if all(k == v for k, v in remap.items()):
        return  # nothing within tolerance -- leave the mesh untouched

    new_faces = []
    new_ids = []
    ids = mesh.face_ids if len(mesh.face_ids) == len(faces) else [0] * len(faces)
    for (a, b, c), fid in zip(faces, ids):
        na, nb, nc = remap.get(a, a), remap.get(b, b), remap.get(c, c)
        if na == nb or nb == nc or na == nc:
            continue  # merge collapsed this triangle to a line -- drop it
        new_faces.append((na, nb, nc))
        new_ids.append(fid)
    mesh.faces = new_faces
    mesh.face_ids = new_ids
    return remap


def _drop_coincident_triangles(mesh: Mesh) -> int:
    """Remove a triangle that two faces both claim.

    Where a fitted patch reaches its neighbour's boundary, both faces can
    triangulate the same three vertices, wound in opposite directions: a fin
    of no thickness whose three edges are used four times. Both triangles go,
    and the surrounding faces then use each of those edges twice. Two
    triangles of equal winding are one triangle listed twice; one stays.
    Returns the number of triangles removed."""
    faces = mesh.faces
    by_set: dict[frozenset, list[int]] = {}
    for t, f in enumerate(faces):
        by_set.setdefault(frozenset(f), []).append(t)

    def canon(f):
        a, b, c = f
        return min((a, b, c), (b, c, a), (c, a, b))

    drop = set()
    for ts in by_set.values():
        if len(ts) == 2:
            t0, t1 = ts
            if canon(faces[t0]) == canon(faces[t1]):
                drop.add(t1)
            else:
                drop.update(ts)
    if not drop:
        return 0
    ids = mesh.face_ids if len(mesh.face_ids) == len(faces) else [0] * len(faces)
    mesh.faces = [f for t, f in enumerate(faces) if t not in drop]
    mesh.face_ids = [i for t, i in enumerate(ids) if t not in drop]
    return len(drop)


HOLE_MAX_POINTS = 6
HOLE_WIDTH_CHORDS = 8.0     # widest hole to close, in chord tolerances


def _fill_small_holes(mesh: Mesh, max_width: float) -> int:
    """Close a hole in the mesh that is narrower than `max_width`.

    Two faces that meet along one curve are tessellated on their own, and
    where one of them ends a few hundredths of a millimetre beside the other
    (a boundary vertex next to a long straight edge, or a triangle-sized hole
    where three faces meet), the mesh stays open although the file is closed.
    A hole is a loop of at most `HOLE_MAX_POINTS` edges that each belong to
    one triangle only; its width is 4 x area / perimeter, about the height of
    a thin triangle and the size of a compact one. The loop is triangulated
    with the winding that closes it, and the new triangles take the face of
    the triangle at the loop's longest edge. Returns the number of holes
    closed."""
    faces = mesh.faces
    verts = mesh.verts
    uses: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            uses[key] = uses.get(key, 0) + 1
    out_edges: dict[int, list[int]] = {}
    owner: dict[tuple[int, int], int] = {}
    for t, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            if uses[(u, v) if u < v else (v, u)] == 1:
                out_edges.setdefault(u, []).append(v)
                owner[(u, v)] = t
    if not out_edges:
        return 0
    ids = mesh.face_ids if len(mesh.face_ids) == len(faces) else [0] * len(faces)
    P = np.asarray(verts, dtype=float)
    seen = set()
    new_faces: list[tuple[int, int, int]] = []
    new_ids: list[int] = []
    n_holes = 0
    for start in sorted(out_edges):
        if start in seen:
            continue
        loop: list[int] | None = [start]
        while True:
            nxt = out_edges.get(loop[-1], [])
            if len(nxt) != 1 or nxt[0] in loop[1:] or (nxt[0] != start and len(loop) == HOLE_MAX_POINTS):
                loop = None
                break
            if nxt[0] == start:
                break
            loop.append(nxt[0])
        if loop is None or len(loop) < 3:
            continue
        seen.update(loop)
        k = len(loop)
        edge_len = [float(np.linalg.norm(P[loop[(i + 1) % k]] - P[loop[i]])) for i in range(k)]
        poly = loop[::-1]                       # the winding that closes the hole
        normal = np.zeros(3)
        for i in range(k):
            normal += np.cross(P[poly[i]], P[poly[(i + 1) % k]])
        area = 0.5 * float(np.linalg.norm(normal))
        perimeter = sum(edge_len)
        if area <= 0.0 or 4.0 * area / perimeter > max_width:
            continue
        normal /= 2.0 * area
        best, best_min = None, 0.0
        for apex in range(k):
            tris = [(poly[apex], poly[(apex + j) % k], poly[(apex + j + 1) % k])
                    for j in range(1, k - 1)]
            smallest = min(0.5 * float(np.cross(P[b] - P[a], P[c] - P[a]) @ normal)
                           for a, b, c in tris)
            if smallest > best_min:
                best, best_min = tris, smallest
        if best is None:
            continue
        longest = max(range(k), key=edge_len.__getitem__)
        fid = ids[owner[(loop[longest], loop[(longest + 1) % k])]]
        new_faces += best
        new_ids += [fid] * len(best)
        n_holes += 1
    if n_holes:
        mesh.faces = list(faces) + new_faces
        mesh.face_ids = list(ids) + new_ids
    return n_holes


@dataclass
class Solid:
    name: str
    mesh: Mesh
    n_faces_in: int = 0
    n_faces_failed: int = 0
    matrix: list | None = None  # 4x4 world transform (assembly instances)
    # Display colours from STYLED_ITEMs, sRGB (r, g, b, alpha) in 0..1:
    # `colour` for the whole solid, `face_colours` per ADVANCED_FACE id
    # (see Mesh.face_ids) where a face carries its own.
    colour: tuple | None = None
    face_colours: dict[int, tuple] = field(default_factory=dict)
    # Exact source surface per ADVANCED_FACE id, in this solid's own mesh
    # coordinates (see `surface_record`): lets Curved export write the
    # original surface back for faces nobody edited.
    face_surfaces: dict[int, dict] = field(default_factory=dict)


class ConvertResult(list):
    """`convert_step`'s return value: the list of Solids, plus what went
    wrong on the way, so a caller can tell the user instead of leaving
    silent holes.

    n_faces        -- distinct B-rep faces in the file
    failed_faces   -- ids of the faces that produced no triangles
    unsupported    -- {(kind, entity name): number of entities} of curves
                      and surfaces that could not be built (a curve falls
                      back to a straight chord, a surface drops its face)
    """
    n_faces = 0

    def __init__(self, *args):
        super().__init__(*args)
        self.failed_faces: list[int] = []
        self.unsupported: dict[tuple[str, str], int] = {}

    def summary_lines(self, label=""):
        """Human-readable warnings (empty when nothing went wrong)."""
        out = []
        pre = f"{label}: " if label else ""
        if self.failed_faces:
            ids = ", ".join(f"#{int(f)}" for f in self.failed_faces[:12])
            more = " ..." if len(self.failed_faces) > 12 else ""
            out.append(f"{pre}{len(self.failed_faces)} of {self.n_faces} faces "
                       f"could not be tessellated and are missing (holes): "
                       f"{ids}{more}")
        for (kind, name), n in sorted(self.unsupported.items()):
            what = ("drawn as straight lines" if kind == "curve"
                    else "their faces are missing")
            out.append(f"{pre}unsupported {kind} type {name} "
                       f"({n} entit{'y' if n == 1 else 'ies'}, {what})")
        return out


# ---------------------------------------------------------------------------

_SI_PREFIX_SCALE = {
    "MICRO": 1e-6, "MILLI": 1e-3, "CENTI": 1e-2, "DECI": 1e-1,
    "DECA": 10.0, "HECTO": 100.0, "KILO": 1000.0,
}


def _si_unit_scale(r):
    """Scale-to-metres for a single SI_UNIT(prefix, .METRE.) record, or None
    if this record isn't a metre-based SI_UNIT at all."""
    if r.name != "SI_UNIT" or not any(str(p).upper() == "METRE" for p in r.params):
        return None
    for p in r.params:
        s = str(p).upper()
        if s in _SI_PREFIX_SCALE:
            return _SI_PREFIX_SCALE[s]
    return 1.0


def _length_scale(sf: StepFile) -> float:
    """Return factor to convert file units to metres (Blender default).

    STEP expresses the modelling length unit either as a plain
    SI_UNIT(prefix, .METRE.) (millimetres, say) or as a CONVERSION_BASED_UNIT
    (e.g. 'INCH') with a scale factor against some SI unit
    (LENGTH_MEASURE_WITH_UNIT). Both count only when the record is tagged
    LENGTH_UNIT(), which marks *the* document length unit: the millimetre base
    unit inside an inch unit's own conversion factor must not be taken for it.
    """
    # The unit that counts is the one of the context the geometry itself is
    # defined in; a file can declare several (an Inventor inch file also
    # carries the centimetre unit the inch is defined against, and a
    # millimetre context for its tolerances).
    geo_unit = _geometry_length_unit(sf)
    if geo_unit is not None:
        scale = _unit_record_scale(sf, geo_unit)
        if scale is not None:
            return scale
    for inst in sf.instances.values():
        recs = inst.records
        if not recs or not any(r.name == "LENGTH_UNIT" for r in recs):
            continue
        scale = _unit_record_scale(sf, inst)
        if scale is not None:
            return scale
    return 0.001  # CREO/most CAD default mm; treat as mm->m


# Items that carry a part's geometry: whichever representation holds one of
# these defines the length unit the geometry is written in.
_GEOMETRY_ITEMS = {"MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS", "FACETED_BREP",
                   "SHELL_BASED_SURFACE_MODEL", "TRIANGULATED_FACE_SET",
                   "COMPLEX_TRIANGULATED_FACE_SET", "TESSELLATED_SOLID",
                   "TESSELLATED_SHELL"}


def _geometry_length_unit(sf: StepFile):
    """The LENGTH_UNIT instance of the representation context that the first
    geometry-carrying representation uses, or None."""
    return _geometry_unit(sf, "LENGTH_UNIT")


# What a conversion-based unit is called when its conversion factor cannot be
# read: (metres per unit, radians per unit).
_NAMED_LENGTH_UNITS = {"INCH": 0.0254, "FOOT": 0.3048, "YARD": 0.9144,
                       "MILE": 1609.344, "THOU": 2.54e-5, "MIL": 2.54e-5}
_NAMED_ANGLE_UNITS = {"DEGREE": math.pi / 180.0, "GRAD": math.pi / 200.0,
                      "GON": math.pi / 200.0, "RADIAN": 1.0}


def _number(v):
    """`v` as a float, or None. A STEP measure value is a plain number or a
    typed one (`LENGTH_MEASURE(25.4)`, an Instance with the number inside)."""
    if isinstance(v, Instance):
        v = v.params[0] if v.params else None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _measure_with_unit(sf: StepFile, ref):
    """(value, unit reference) of a `*_MEASURE_WITH_UNIT` entity, either may
    be None. The value is written both ways in files from mainstream
    writers: typed, `LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#9)`
    (Inventor, Pro/E, CATIA), and as a bare number,
    `LENGTH_MEASURE_WITH_UNIT(25.4,#9)` (Open CASCADE, hence FreeCAD and
    CadQuery); in a complex entity the attributes sit in a
    `MEASURE_WITH_UNIT` record."""
    inst = sf.get(ref)
    if inst is None:
        return None, None
    for r in (inst.records or [inst]):
        if r.name.endswith("MEASURE_WITH_UNIT") and r.params:
            value = _number(r.params[0])
            if value is not None:
                return value, (r.params[1] if len(r.params) > 1 else None)
    for r in (inst.records or []):
        if r.name.endswith("_MEASURE") and r.params:
            value = _number(r.params[0])
            if value is not None:
                return value, (inst.params[-1] if inst.params else None)
    return None, None


def _plane_angle_scale(sf: StepFile) -> float:
    """Radians per plane-angle unit of the geometry's context (1.0 for
    radians; files from Inventor, CATIA, Pro/E often use degrees, which
    matter for a cone's semi-angle and trimmed-surface parameters)."""
    u = _geometry_unit(sf, "PLANE_ANGLE_UNIT")
    if u is None:
        return 1.0
    scale = _angle_unit_scale(sf, u)
    return 1.0 if scale is None else scale


def _angle_unit_scale(sf: StepFile, inst, _depth=0):
    """Radians per unit of one PLANE_ANGLE_UNIT instance (SI radian or
    conversion-based, e.g. DEGREE), or None if it cannot be resolved."""
    recs = inst.records or [inst]
    for r in recs:
        if r.name == "SI_UNIT":
            return 1.0
    if _depth > 4:
        return None
    for r in recs:
        if r.name != "CONVERSION_BASED_UNIT":
            continue
        factor, base_ref = _measure_with_unit(sf, r.params[1] if len(r.params) > 1 else None)
        base = sf.get(base_ref)
        base_scale = _angle_unit_scale(sf, base, _depth + 1) if base is not None else None
        if factor is not None:
            return factor * (1.0 if base_scale is None else base_scale)
        named = _NAMED_ANGLE_UNITS.get(str(r.params[0]).upper() if r.params else "")
        if named is not None:
            return named
    return None


def _geometry_unit(sf: StepFile, unit_record: str):
    """The unit instance tagged `unit_record` (LENGTH_UNIT, PLANE_ANGLE_UNIT)
    of the context the first geometry-carrying representation uses."""
    for rep in sf.instances.values():
        recs = rep.records or [rep]
        for r in recs:
            if not r.name.endswith("REPRESENTATION") or len(r.params) < 3:
                continue
            items = r.params[1]
            if not isinstance(items, list):
                continue
            if not any((sf.get(it) is not None and sf.get(it).name in _GEOMETRY_ITEMS)
                       for it in items):
                continue
            ctx = sf.get(r.params[2])
            if ctx is None:
                return None
            for cr in (ctx.records or [ctx]):
                if cr.name != "GLOBAL_UNIT_ASSIGNED_CONTEXT" or not cr.params:
                    continue
                for uref in cr.params[0]:
                    u = sf.get(uref)
                    if u is not None and any(x.name == unit_record for x in (u.records or [])):
                        return u
            return None
    return None


def _unit_record_scale(sf: StepFile, inst, _depth=0):
    """Scale to metres of one LENGTH_UNIT instance (plain SI or
    conversion-based), or None if it cannot be resolved."""
    recs = inst.records or [inst]
    for r in recs:
        scale = _si_unit_scale(r)
        if scale is not None:
            return scale
    if _depth > 4:
        return None
    for r in recs:
        if r.name != "CONVERSION_BASED_UNIT":
            continue
        # CONVERSION_BASED_UNIT(name, ref) -> ref is a
        # LENGTH_MEASURE_WITH_UNIT(factor, base_unit_ref), the factor typed
        # or bare (see _measure_with_unit).
        factor, base_ref = _measure_with_unit(sf, r.params[1] if len(r.params) > 1 else None)
        if factor is None:
            named = _NAMED_LENGTH_UNITS.get(str(r.params[0]).upper() if r.params else "")
            if named is not None:
                return named
            continue
        base_inst = sf.get(base_ref)
        base_scale = None
        if base_inst is not None:
            base_scale = _unit_record_scale(sf, base_inst, _depth + 1)
        return factor * (1.0 if base_scale is None else base_scale)
    return None


def _product_name(sf: StepFile) -> str:
    prods = [i for i in sf.instances.values() if i.name == "PRODUCT"]
    if prods:
        nm = prods[0].params[0] or prods[0].params[1]
        if nm:
            return str(nm)
    # fall back to FILE_NAME header
    fn = sf.header.get("FILE_NAME")
    if fn and fn.params and fn.params[0]:
        return str(fn.params[0])
    return "STEP_Object"


def _all_product_names(sf: StepFile) -> list[str]:
    out = []
    for i in sf.instances.values():
        if i.name == "PRODUCT":
            nm = i.params[0] or i.params[1]
            out.append(str(nm) if nm else "Part")
    return out


def _collect_solids(sf: StepFile):
    """Return list of (solid_name_hint, [advanced_face_refs])."""
    solids = []
    breps = sf.of_type("MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS")
    if breps:
        for b in breps:
            shell_ref = b.params[1]
            faces = _faces_of_shell(sf, shell_ref)
            if b.name == "BREP_WITH_VOIDS" and len(b.params) > 2:
                for void in b.params[2]:
                    faces += _faces_of_shell(sf, void)
            solids.append(faces)
    else:
        # fall back: open/closed shells directly
        for sh in sf.of_type("CLOSED_SHELL", "OPEN_SHELL"):
            solids.append(_faces_of_shell(sf, sh.id))
    return solids


def _faces_of_shell(sf, shell_ref):
    sh = sf.get(shell_ref)
    if sh is None:
        return []
    faces = []
    for fref in sh.params[-1]:
        f = sf.get(fref)
        if f and f.name in ("ADVANCED_FACE", "FACE_SURFACE"):
            faces.append(fref)
    return faces


def _pd_to_face_lists(sf):
    pd_name = {}
    for pd in sf.of_type("PRODUCT_DEFINITION"):
        pdf = sf.get(pd.params[2]) if len(pd.params) > 2 else None
        prod = sf.get(pdf.params[2]) if pdf and len(pdf.params) > 2 else None
        nm = None
        if prod and prod.name == "PRODUCT":
            nm = prod.params[0] or prod.params[1]
        pd_name[pd.id] = str(nm) if nm else f"Part_{pd.id}"
    related = _plain_rep_relations(sf)
    pd_solids = {}
    for sdr in sf.of_type("SHAPE_DEFINITION_REPRESENTATION"):
        pds = sf.get(sdr.params[0])
        sr = sf.get(sdr.params[1])
        if pds is None or sr is None:
            continue
        pd_ref = pds.params[-1] if pds.params else None
        pd = sf.get(pd_ref)
        if pd is None or pd.name != "PRODUCT_DEFINITION":
            continue
        items = list(sr.params[1]) if len(sr.params) > 1 else []
        # AP203 exporters (Pro/E, CATIA, ...) often hang the solid off a
        # separate ADVANCED_BREP_SHAPE_REPRESENTATION, linked to the part's
        # own SHAPE_REPRESENTATION by a SHAPE_REPRESENTATION_RELATIONSHIP.
        for rid in related.get(int(sr.id), ()):
            other = sf.get(rid)
            if other is not None and len(other.params) > 1 and isinstance(other.params[1], list):
                items.extend(other.params[1])
        face_lists = []
        for it in items:
            item = sf.get(it)
            if item is None:
                continue
            if item.name in ("MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS"):
                face_lists.append(_faces_of_shell(sf, item.params[1]))
        if face_lists:
            pd_solids.setdefault(pd.id, []).extend(face_lists)
    return pd_name, pd_solids


# ---------------------------------------------------------------------------
# Colours (STYLED_ITEM -> ... -> COLOUR_RGB)
# ---------------------------------------------------------------------------

# DRAUGHTING_PRE_DEFINED_COLOUR names (ISO 10303-46), sRGB.
_PREDEFINED_COLOURS = {
    "black": (0.0, 0.0, 0.0), "red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0), "yellow": (1.0, 1.0, 0.0), "magenta": (1.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0), "white": (1.0, 1.0, 1.0),
}


def _colour_of(sf, ref, depth=0):
    """(r, g, b) of a COLOUR_RGB / DRAUGHTING_PRE_DEFINED_COLOUR, else None."""
    c = sf.get(ref)
    if c is None or depth > 4:
        return None
    for r in (c.records or [c]):
        if r.name == "COLOUR_RGB" and len(r.params) >= 3:
            vals = r.params[-3:]
            try:
                return tuple(min(1.0, max(0.0, float(v))) for v in vals)
            except (TypeError, ValueError):
                return None
        if r.name == "DRAUGHTING_PRE_DEFINED_COLOUR" and r.params:
            return _PREDEFINED_COLOURS.get(str(r.params[0]).lower())
    return None


def _surface_style_colour(sf, psa_ref):
    """Walk one PRESENTATION_STYLE_ASSIGNMENT down to a surface colour and
    transparency: SURFACE_STYLE_USAGE -> SURFACE_SIDE_STYLE ->
    SURFACE_STYLE_FILL_AREA -> FILL_AREA_STYLE -> FILL_AREA_STYLE_COLOUR,
    or SURFACE_STYLE_RENDERING(_WITH_PROPERTIES) (colour, transparency).
    Returns (r, g, b, a) or None. Curve and text styles are ignored."""
    psa = sf.get(psa_ref)
    if psa is None or not psa.params:
        return None
    styles = psa.params[0] if isinstance(psa.params[0], list) else psa.params
    rgb, alpha = None, 1.0
    for st_ref in styles:
        st = sf.get(st_ref)
        if st is None or st.name != "SURFACE_STYLE_USAGE" or len(st.params) < 2:
            continue
        side = sf.get(st.params[1])
        if side is None or len(side.params) < 2:
            continue
        for el_ref in side.params[1]:
            el = sf.get(el_ref)
            if el is None:
                continue
            if el.name == "SURFACE_STYLE_FILL_AREA" and el.params:
                fas = sf.get(el.params[0])
                if fas is None or len(fas.params) < 2:
                    continue
                for fc_ref in fas.params[1]:
                    fc = sf.get(fc_ref)
                    if fc is not None and fc.name == "FILL_AREA_STYLE_COLOUR" and len(fc.params) > 1:
                        rgb = _colour_of(sf, fc.params[1]) or rgb
            elif el.name in ("SURFACE_STYLE_RENDERING",
                             "SURFACE_STYLE_RENDERING_WITH_PROPERTIES") and len(el.params) > 1:
                rgb = rgb or _colour_of(sf, el.params[1])
                if len(el.params) > 2 and isinstance(el.params[2], list):
                    for pr_ref in el.params[2]:
                        pr = sf.get(pr_ref)
                        if pr is not None and pr.name == "SURFACE_STYLE_TRANSPARENT" and pr.params:
                            try:
                                alpha = 1.0 - min(1.0, max(0.0, float(pr.params[0])))
                            except (TypeError, ValueError):
                                pass
    if rgb is None:
        return None
    return (rgb[0], rgb[1], rgb[2], alpha)


def styled_colours(sf):
    """item id -> (r, g, b, a) for every item a STYLED_ITEM colours. An
    OVER_RIDING_STYLED_ITEM (e.g. one face of a coloured solid) wins over a
    plain one."""
    out: dict[int, tuple] = {}
    for over in (False, True):
        for si in sf.of_type("OVER_RIDING_STYLED_ITEM" if over else "STYLED_ITEM"):
            p = si.params
            if len(p) < 3:
                continue
            col = None
            for psa_ref in (p[1] if isinstance(p[1], list) else [p[1]]):
                col = _surface_style_colour(sf, psa_ref) or col
            if col is None:
                continue
            try:
                out[int(p[2])] = col
            except (TypeError, ValueError):
                continue
    return out


def face_colours(sf):
    """ADVANCED_FACE id -> colour: the face's own style, else its shell's,
    else its solid's."""
    items = styled_colours(sf)
    if not items:
        return {}
    out: dict[int, tuple] = {}

    def paint(face_ids, col):
        for f in face_ids:
            out.setdefault(int(f), col)
    faces_first = sorted(items.items(), key=lambda kv: 0 if (sf.get(kv[0]) is not None and
                                                              sf.get(kv[0]).name in ("ADVANCED_FACE", "FACE_SURFACE")) else 1)
    for item_id, col in faces_first:
        it = sf.get(item_id)
        if it is None:
            continue
        if it.name in ("ADVANCED_FACE", "FACE_SURFACE"):
            out[int(item_id)] = col
        elif it.name in ("CLOSED_SHELL", "OPEN_SHELL"):
            paint(_faces_of_shell(sf, item_id), col)
    for item_id, col in items.items():
        it = sf.get(item_id)
        if it is None:
            continue
        if it.name in ("MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS", "FACETED_BREP") and len(it.params) > 1:
            paint(_faces_of_shell(sf, it.params[1]), col)
        elif it.name == "SHELL_BASED_SURFACE_MODEL" and len(it.params) > 1:
            for sh in it.params[1]:
                paint(_faces_of_shell(sf, sh), col)
    return out


# ---------------------------------------------------------------------------
# Source surfaces (for a lossless round trip of unedited faces)
# ---------------------------------------------------------------------------

def surface_record(surface) -> dict | None:
    """JSON-safe description of an imported surface in the vocabulary of the
    Curved exporter's fit objects (plane/cylinder/cone/sphere/torus/
    bspline), or None for kinds it cannot write exactly (swept, numeric
    offset, rational B-spline)."""
    f = getattr(surface, "f", None)
    if isinstance(surface, G.Plane):
        return {"kind": "plane", "origin": f.o.tolist(), "normal": f.z.tolist()}
    if isinstance(surface, G.Cylinder):
        return {"kind": "cylinder", "axis_point": f.o.tolist(), "axis_dir": f.z.tolist(),
                "radius": float(surface.r)}
    if isinstance(surface, G.Cone):
        if abs(surface.tan) < 1e-9:
            return None
        apex = f.o - (surface.r_ref / surface.tan) * f.z
        axis = f.z if surface.tan > 0 else -f.z
        return {"kind": "cone", "apex": apex.tolist(), "axis_dir": axis.tolist(),
                "semi_angle": float(math.atan(abs(surface.tan)))}
    if isinstance(surface, G.Sphere):
        return {"kind": "sphere", "center": f.o.tolist(), "radius": float(surface.r)}
    if isinstance(surface, G.Torus):
        return {"kind": "torus", "axis_point": f.o.tolist(), "axis_dir": f.z.tolist(),
                "major_radius": float(surface.R), "minor_radius": float(surface.r)}
    if isinstance(surface, G.BSplineSurface) and surface.w is None:
        return {"kind": "bspline", "deg_u": int(surface.pu), "deg_v": int(surface.pv),
                "ctrl": surface.P.tolist(), "knots_u": surface.U.tolist(),
                "knots_v": surface.V.tolist()}
    return None


def transform_surface_record(rec: dict, matrix=None, scale: float = 1.0) -> dict | None:
    """`rec` moved by the 4x4 `matrix` (applied first) and then scaled by
    `scale`. Analytic kinds need a rigid motion times a uniform scale (a
    cylinder stretched one way is no cylinder); a plane or B-spline takes
    any affine map. Returns None when the record cannot follow."""
    M = np.eye(4) if matrix is None else np.asarray(matrix, dtype=float)
    A = M[:3, :3] * scale
    t = M[:3, 3] * scale
    cols = np.linalg.norm(A, axis=0)
    s_uni = float(cols.mean())
    similar = (s_uni > 0 and np.allclose(cols, s_uni, rtol=1e-6) and
               np.allclose(A.T @ A, np.eye(3) * s_uni * s_uni, atol=1e-9 * s_uni * s_uni + 1e-15))

    def pt(x):
        return (A @ np.asarray(x, dtype=float) + t).tolist()

    def vec(x):
        v = A @ np.asarray(x, dtype=float)
        n = float(np.linalg.norm(v))
        return (v / n).tolist() if n > 0 else None

    k = rec.get("kind")
    out = dict(rec)
    if k == "plane":
        n = np.linalg.solve(A.T, np.asarray(rec["normal"], dtype=float)) if abs(np.linalg.det(A)) > 1e-300 else None
        if n is None or np.linalg.norm(n) == 0:
            return None
        out["origin"] = pt(rec["origin"])
        out["normal"] = (n / np.linalg.norm(n)).tolist()
        return out
    if k == "bspline":
        out["ctrl"] = [[pt(c) for c in row] for row in rec["ctrl"]]
        return out
    if not similar:
        return None
    if k in ("cylinder", "torus"):
        out["axis_point"] = pt(rec["axis_point"])
        out["axis_dir"] = vec(rec["axis_dir"])
        for key in ("radius", "major_radius", "minor_radius"):
            if key in rec:
                out[key] = float(rec[key]) * s_uni
        return out
    if k == "cone":
        out["apex"] = pt(rec["apex"])
        out["axis_dir"] = vec(rec["axis_dir"])
        return out
    if k == "sphere":
        out["center"] = pt(rec["center"])
        out["radius"] = float(rec["radius"]) * s_uni
        return out
    return None


def _plain_rep_relations(sf):
    """rep id -> ids of representations linked to it by a
    SHAPE_REPRESENTATION_RELATIONSHIP that carries no transformation (those
    with one are assembly placements, handled in assembly.py)."""
    rel: dict[int, set] = {}
    for inst in sf.instances.values():
        recs = inst.records or [inst]
        names = {r.name for r in recs}
        if "SHAPE_REPRESENTATION_RELATIONSHIP" not in names:
            continue
        if "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION" in names:
            continue
        r = inst if inst.name else next((x for x in recs if x.name == "REPRESENTATION_RELATIONSHIP"), None)
        if r is None or len(r.params) < 4:
            continue
        try:
            a, b = int(r.params[2]), int(r.params[3])
        except (TypeError, ValueError):
            continue
        rel.setdefault(a, set()).add(b)
        rel.setdefault(b, set()).add(a)
    return rel


def _collect_tessellated(sf, scale):
    """Meshes from AP242 tessellated geometry: TESSELLATED_SOLID /
    TESSELLATED_SHELL of TRIANGULATED_FACEs over a COORDINATES_LIST (what
    Open CASCADE and StepForge >= 3.7 write), and the older
    TRIANGULATED_FACE_SET. Vertex order follows the coordinate list and
    triangle order the file, so a StepForge export reads back unchanged."""
    out = []
    names = {}
    for sdr in sf.of_type("SHAPE_DEFINITION_REPRESENTATION"):
        rep = sf.get(sdr.params[1]) if len(sdr.params) > 1 else None
        pds = sf.get(sdr.params[0]) if sdr.params else None
        pd = sf.get(pds.params[-1]) if pds is not None and pds.params else None
        pdf = sf.get(pd.params[2]) if pd is not None and len(pd.params) > 2 else None
        prod = sf.get(pdf.params[2]) if pdf is not None and len(pdf.params) > 2 else None
        if rep is None or prod is None or len(rep.params) < 2 or not isinstance(rep.params[1], list):
            continue
        for it in rep.params[1]:
            names[int(it)] = str(prod.params[0] or prod.params[1] or "")

    for item in sf.of_type("TESSELLATED_SOLID", "TESSELLATED_SHELL"):
        p = item.params
        if len(p) < 2 or not isinstance(p[1], list):
            continue
        mesh = Mesh()
        base_of = {}
        for fref in p[1]:
            f = sf.get(fref)
            if f is None or f.name != "TRIANGULATED_FACE" or len(f.params) < 7:
                continue
            fp = f.params
            cl = sf.get(fp[1])
            if cl is None or len(cl.params) < 3:
                continue
            base = base_of.get(int(cl.id))
            if base is None:
                base = base_of[int(cl.id)] = len(mesh.verts)
                mesh.verts.extend((float(c[0]) * scale, float(c[1]) * scale, float(c[2]) * scale)
                                  for c in cl.params[2])
            n_cl = len(cl.params[2])
            pnindex = fp[5] if isinstance(fp[5], list) else []
            for tri in fp[6]:
                idx = []
                for t in tri:
                    k = int(t) - 1
                    ci = (int(pnindex[k]) - 1) if pnindex else k
                    idx.append(ci)
                if len(idx) == 3 and all(0 <= i < n_cl for i in idx):
                    mesh.faces.append((base + idx[0], base + idx[1], base + idx[2]))
                    mesh.face_ids.append(int(fref))
        if mesh.faces:
            name = str(p[0]) if p[0] else (names.get(int(item.id)) or "Mesh")
            sol = Solid(name=name, mesh=mesh, n_faces_in=len(p[1]))
            sol._item = int(item.id)
            out.append(sol)

    for tfs in sf.of_type("TRIANGULATED_FACE_SET", "COMPLEX_TRIANGULATED_FACE_SET"):
        p = tfs.params
        name = str(p[0]) if p and p[0] else "Mesh"
        cpl = sf.get(p[1])
        if cpl is None:
            continue
        coords = cpl.params[1] if len(cpl.params) > 1 else cpl.params[0]
        coords = [(float(c[0]) * scale, float(c[1]) * scale, float(c[2]) * scale) for c in coords]
        pnindex = p[4] if len(p) > 4 and p[4] else None
        triangles = p[5] if len(p) > 5 else []
        mesh = Mesh()
        for tri in triangles:
            idx = []
            for t in tri:
                ti = int(t) - 1
                ci = (int(pnindex[ti]) - 1) if pnindex else ti
                idx.append(ci)
            if len(idx) == 3 and all(0 <= i < len(coords) for i in idx):
                mesh.faces.append((idx[0], idx[1], idx[2]))
                mesh.face_ids.append(int(tfs.id))
        mesh.verts = coords
        if mesh.faces:
            sol = Solid(name=name, mesh=mesh, n_faces_in=len(mesh.faces))
            sol._item = int(tfs.id)
            out.append(sol)
    return out


def _process_face(sf, face_ref, deflection, max_edge=None, edge_cache=None, debug=False, log=print,
                  edge_hints=None, flat_max_edge=True):
    """Return (verts3d_list, tris) for one ADVANCED_FACE, or (None, None)."""
    face = sf.get(face_ref)
    if face is None:
        return None, None
    bounds_refs = face.params[1]
    surf_ref = face.params[2]
    same_sense = True
    if len(face.params) > 3:
        same_sense = T._bool(face.params[3])
    surface = T.build_surface(sf, surf_ref)
    if surface is None:
        return None, None
    if surface.sense_flip:
        same_sense = not same_sense

    edge_sample_target = max_edge if max_edge else deflection * 400.0

    # Face normal = surface normal if same_sense; each bound's loop, with
    # its FACE_BOUND orientation applied, has the face on its left when
    # seen against that normal. Only needed to pick the side of a loop that
    # winds around a periodic direction (see T.close_winding_loops).
    face_sign = 1 if same_sense else -1
    bounds = []
    winding = []
    vertex_poles = []
    for bref in bounds_refs:
        b = sf.get(bref)
        if b is None or len(b.params) < 2:
            continue
        is_outer = (b.name == "FACE_OUTER_BOUND")
        loop_ref = b.params[1]
        loop = sf.get(loop_ref)
        if loop is None:
            continue
        if loop.name == "VERTEX_LOOP":
            p = T._vertex_point(sf, loop.params[1])
            if p is not None:
                vertex_poles.append(p)
            continue
        orient = T._bool(b.params[2]) if len(b.params) > 2 else True
        pts3d = T._walk_loop(sf, loop_ref, deflection, edge_sample_target, edge_cache,
                             edge_hints=edge_hints)
        if len(pts3d) < 2:
            continue
        res = T._loop_uv(surface, pts3d)
        if res is None:
            continue
        uv, pts3d, net_u, net_v = res
        if net_u or net_v:
            winding.append((uv, pts3d, net_u, net_v, face_sign * (1 if orient else -1)))
            continue
        if len(uv) < 3:
            continue
        bounds.append([uv, pts3d, is_outer])

    tol = max(deflection * 1e-3, 1e-9)
    closed = None
    if winding:
        closed = T.close_winding_loops(surface, winding, vertex_poles, deflection,
                                       edge_sample_target, tol)
        if closed is None and debug:
            log(f"[StepForge] face {face_ref}: {len(winding)} boundary loop(s) wind "
                f"around the {surface.__class__.__name__} with nothing to close "
                f"them against -- face skipped")
    elif not bounds:
        closed = T.natural_bounds(surface, deflection, edge_sample_target, tol)
    if closed is not None:
        for bd in bounds:
            bd[2] = False
        bounds.insert(0, [closed[0], closed[1], True])

    # Every loop must sit in the same copy of a periodic parameter domain as
    # the first one: independently unwrapped loops of one cylinder can land
    # whole periods apart, which breaks the hole-inside-outer tests below (CDT
    # boundary recovery, point-in-polygon). Shifting whole periods is always
    # safe, since the parameter space wraps. Holes are moved to where their
    # centre lies inside the span of the outline: a cylinder written with a
    # seam runs over a whole period, and a hole whose first point is nearest to
    # the outline's first point can land on the wrong side of the seam, outside
    # the outline, where it is not cut (the second hole of two crossing
    # cylinders).
    if bounds:
        us0 = [u for (u, _) in bounds[0][0]]
        vs0 = [v for (_, v) in bounds[0][0]]
        ref_u = 0.5 * (min(us0) + max(us0))
        ref_v = 0.5 * (min(vs0) + max(vs0))
        for bd in bounds[1:]:
            uv = bd[0]
            cu = sum(u for (u, _) in uv) / len(uv)
            cv = sum(v for (_, v) in uv) / len(uv)
            du = (T._nearest_copy(cu, ref_u, surface.period_u) - cu) if surface.periodic_u else 0.0
            dv = (T._nearest_copy(cv, ref_v, surface.period_v) - cv) if surface.periodic_v else 0.0
            if du or dv:
                bd[0] = [(u + du, v + dv) for (u, v) in uv]

    if not bounds:
        return None, None
    all_u = [u for bd in bounds for (u, _) in bd[0]]
    all_v = [v for bd in bounds for (_, v) in bd[0]]
    bounds = [tuple(bd) for bd in bounds]

    uv_extent = ((min(all_u), min(all_v)), (max(all_u), max(all_v)))
    face_max_edge = max_edge
    if not flat_max_edge and isinstance(surface, G.Plane):
        face_max_edge = None   # a flat face needs no interior points
    verts3d, tris = T.triangulate_face(surface, bounds, deflection, uv_extent,
                                       max_edge=face_max_edge, debug=debug,
                                       face_label=f"face {face_ref}", log=log)
    if verts3d is None or not tris:
        if debug:
            log(f"[StepForge] face {face_ref}: triangulation failed entirely "
                f"({surface.__class__.__name__}, {len(bounds)} loop(s)) -- "
                f"this face will be skipped (a hole in the mesh)")
        return None, None
    if not same_sense:
        tris = [(a, c, b) for (a, b, c) in tris]
    return verts3d, tris



def _face_edge_refs(sf, face):
    for bref in face.params[1]:
        b = sf.get(bref)
        loop = sf.get(b.params[1]) if b is not None and len(b.params) > 1 else None
        if loop is None or loop.name != "EDGE_LOOP":
            continue
        for oe_ref in loop.params[-1]:
            oe = sf.get(oe_ref)
            if oe is not None and oe.name == "ORIENTED_EDGE":
                yield int(oe.params[3])


def _planar_only_edges(sf, face_refs):
    """EDGE_CURVE ids whose every face is a PLANE. A straight edge between
    two flat faces needs no points between its vertices; one that borders
    a curved face keeps the `max_edge` sampling, which that face's own grid
    relies on."""
    planar_uses: dict[int, bool] = {}
    for fref in face_refs:
        face = sf.get(fref)
        if face is None or len(face.params) < 3:
            continue
        surf = sf.get(face.params[2])
        is_plane = surf is not None and surf.name == "PLANE"
        for e in _face_edge_refs(sf, face):
            planar_uses[e] = planar_uses.get(e, True) and is_plane
    return [e for e, flat in planar_uses.items() if flat]


def _register_edge_hints(sf, face_ref, deflection, edge_hints, max_edge=None):
    face = sf.get(face_ref)
    if face is None:
        return
    surf_ref = face.params[2]
    surface = T.build_surface(sf, surf_ref)
    # B-spline faces get no hint: densifying their boundary edges measured
    # worse every time (a new boundary point that lies slightly off a fitted
    # surface can invert to a wrong (u, v) and fold the boundary ring).
    if not isinstance(surface, (G.Cylinder, G.Cone)):
        return
    bounds_refs = face.params[1]
    tmp_cache = {}
    all_u = []
    all_v = []
    line_edge_refs = []
    for bref in bounds_refs:
        b = sf.get(bref)
        if b is None:
            continue
        loop_ref = b.params[1]
        loop = sf.get(loop_ref)
        if loop is None or loop.name != "EDGE_LOOP":
            continue
        for oe_ref in loop.params[-1]:
            oe = sf.get(oe_ref)
            if oe is None or oe.name != "ORIENTED_EDGE":
                continue
            edge_ref = oe.params[3]
            edge = sf.get(edge_ref)
            if edge is None or edge.name != "EDGE_CURVE":
                continue
            curve = T.build_curve(sf, edge.params[3])
            if isinstance(curve, G.Line):
                line_edge_refs.append(edge_ref)
        pts3d = T._walk_loop(sf, loop_ref, deflection, None, tmp_cache)
        for p in pts3d:
            try:
                u, v = surface.invert(p)
            except Exception:  # noqa: BLE001, S112 - an uninvertible point is skipped
                continue
            all_u.append(u)
            all_v.append(v)
    if not line_edge_refs or not all_u:
        return
    if surface.periodic_u:
        all_u = T._unwrap(all_u, surface.period_u)
    u_extent = max(all_u) - min(all_u)
    if u_extent <= 1e-9:
        return
    r = surface.r if isinstance(surface, G.Cylinder) else surface.r_ref
    r = max(abs(r), 1e-6)
    nu = G._angular_cells(u_extent, r, deflection)
    cell_width = max(r * u_extent / max(nu, 1), 1e-6)
    # the interior grid has at most MAX_AXIAL_CELLS cells along the axis
    # (`Cylinder.grid_cells`): spacing the straight edges the same way
    cell_width = max(cell_width, (max(all_v) - min(all_v)) / G.MAX_AXIAL_CELLS)
    for edge_ref in line_edge_refs:
        prev = edge_hints.get(edge_ref)
        if prev is None or cell_width < prev:
            edge_hints[edge_ref] = cell_width


def _all_points_extent(sf: StepFile):
    """(lo, hi) corners of the box around every point the file lists, or
    None if it lists none."""
    lo = np.full(3, math.inf)
    hi = np.full(3, -math.inf)

    def _points():
        for inst in sf.of_type("CARTESIAN_POINT"):
            yield inst.params[1] if len(inst.params) > 1 else inst.params[0]
        for inst in sf.of_type("COORDINATES_LIST", "CARTESIAN_POINT_LIST_3D"):
            lst = inst.params[-1] if inst.params else []
            if isinstance(lst, list):
                yield from lst

    for coords in _points():
        try:
            c = np.array([float(x) for x in coords])
        except (TypeError, ValueError):
            continue
        if c.shape != (3,):
            continue
        lo = np.minimum(lo, c)
        hi = np.maximum(hi, c)
    return (lo, hi) if lo[0] <= hi[0] else None


def _geometry_extent(sf: StepFile):
    """(lo, hi) corners of a box around the geometry of the part itself, or None
    if the file has no edges, faces or tessellation to take it from.

    Taken from the vertices of the edges, the control points of B-spline edges
    and faces, points along each circular or elliptic edge (a circle is not
    covered by its one vertex), a box around a sphere or torus face with no
    edge at all, and the points of a faceted outline and of a tessellation. Not
    the other points of the file: the location of a plane or of a cylinder's
    axis sits anywhere on that infinite surface (in one test part up to 2300 mm
    from a part of 690 mm), a line's origin anywhere on the line, and the
    placements of the file's own axes at the global origin, however far from it
    the part lies."""
    lo = np.full(3, math.inf)
    hi = np.full(3, -math.inf)
    state = {"n": 0}

    def point(ref_or_coords):
        nonlocal lo, hi
        if isinstance(ref_or_coords, list):
            c = np.array([float(x) for x in ref_or_coords])
        else:
            c = G.get_point(sf, ref_or_coords)
        if c is None or c.shape != (3,):
            return
        lo = np.minimum(lo, c)
        hi = np.maximum(hi, c)
        state["n"] += 1

    def box(centre, half):
        nonlocal lo, hi
        lo = np.minimum(lo, centre - half)
        hi = np.maximum(hi, centre + half)
        state["n"] += 1

    def refs(v):
        if isinstance(v, list):
            for x in v:
                yield from refs(x)
        elif v is not None:
            yield v

    def bspline_points(inst, surface):
        if inst.records:
            for r in inst.records:
                if r.name == "B_SPLINE_SURFACE" and len(r.params) > 2:
                    for ref in refs(r.params[2]):
                        point(ref)
                elif r.name == "B_SPLINE_CURVE" and len(r.params) > 1:
                    for ref in refs(r.params[1]):
                        point(ref)
        elif surface and len(inst.params) > 3:
            for ref in refs(inst.params[3]):
                point(ref)
        elif not surface and len(inst.params) > 2:
            for ref in refs(inst.params[2]):
                point(ref)

    def base_curve(ref, depth=0):
        """The curve under SURFACE_CURVE / TRIMMED_CURVE wrappers."""
        inst = sf.get(ref)
        if inst is None or depth > 6:
            return None
        if inst.name in T._SURFACE_CURVE_TYPES and len(inst.params) > 1:
            return base_curve(inst.params[1], depth + 1)
        if inst.name == "TRIMMED_CURVE" and len(inst.params) > 1:
            return base_curve(inst.params[1], depth + 1)
        if inst.name == "" and inst.records:
            for r in inst.records:
                if r.name in T._SURFACE_CURVE_TYPES and r.params:
                    return base_curve(r.params[0], depth + 1)
        return inst

    def conic(frame, a, b, v1, v2, same_sense):
        """Points along the circle or ellipse from vertex v1 to v2 (the
        whole curve if they coincide), in the edge's direction."""
        def angle(q):
            loc = frame.local(q)
            return math.atan2(loc[1] / b, loc[0] / a)
        if v1 is None or v2 is None or float(np.linalg.norm(v1 - v2)) <= 1e-6 * max(a, b):
            start, sweep = 0.0, G.TWO_PI
        else:
            t1, t2 = angle(v1), angle(v2)
            if not same_sense:
                t1, t2 = t2, t1
            start, sweep = t1, (t2 - t1) % G.TWO_PI
        for k in range(33):
            t = start + sweep * k / 32.0
            q = frame.o + a * math.cos(t) * frame.x + b * math.sin(t) * frame.y
            lo_hi(q)

    def lo_hi(q):
        nonlocal lo, hi
        lo = np.minimum(lo, q)
        hi = np.maximum(hi, q)
        state["n"] += 1

    def edge(inst):
        p = inst.params
        inst_c = base_curve(p[3])
        if inst_c is None:
            return
        name, cp = inst_c.name, inst_c.params
        v1 = v2 = None
        for k, ref in ((1, "v1"), (2, "v2")):
            vi = sf.get(p[k])
            if vi is not None and vi.name == "VERTEX_POINT" and len(vi.params) > 1:
                q = G.get_point(sf, vi.params[1])
                if ref == "v1":
                    v1 = q
                else:
                    v2 = q
        same = T._bool(p[4]) if len(p) > 4 else True
        if name == "CIRCLE" and len(cp) > 2:
            r = float(cp[2])
            conic(G.Frame.from_axis2(sf, cp[1]), r, r, v1, v2, same)
        elif name == "ELLIPSE" and len(cp) > 3:
            conic(G.Frame.from_axis2(sf, cp[1]), float(cp[2]), float(cp[3]), v1, v2, same)
        elif name in T._BSPLINE_CURVE_NAMES or name == "":
            bspline_points(inst_c, False)

    def has_edges(face):
        for bref in (face.params[1] if len(face.params) > 1 else []):
            b = sf.get(bref)
            loop = sf.get(b.params[1]) if b is not None and len(b.params) > 1 else None
            if loop is not None and loop.name == "EDGE_LOOP":
                return True
        return False

    def surface(face):
        inst = sf.get(face.params[2])
        if inst is None:
            return
        name, p = inst.name, inst.params
        if name in T._BSPLINE_SURFACE_NAMES or name == "":
            bspline_points(inst, True)
        elif has_edges(face):
            return                      # its edges bound it
        elif name == "SPHERICAL_SURFACE" and len(p) > 2:
            f = G.Frame.from_axis2(sf, p[1])
            box(f.o, np.full(3, float(p[2])))
        elif name == "TOROIDAL_SURFACE" and len(p) > 3:
            f = G.Frame.from_axis2(sf, p[1])
            box(f.o, np.full(3, float(p[2]) + float(p[3])))

    for inst in sf.instances.values():
        name, p = inst.name, inst.params
        try:
            if name == "VERTEX_POINT" and len(p) > 1:
                point(p[1])
            elif name == "EDGE_CURVE" and len(p) > 4:
                edge(inst)
            elif name == "ADVANCED_FACE" and len(p) > 2:
                surface(inst)
            elif name == "POLY_LOOP" and len(p) > 1:
                for ref in refs(p[1]):
                    point(ref)
            elif name in ("COORDINATES_LIST", "CARTESIAN_POINT_LIST_3D") and p:
                for coords in refs(p[-1]):
                    point(coords)
        except (TypeError, ValueError, IndexError, AttributeError):
            continue
    return (lo, hi) if state["n"] and lo[0] <= hi[0] else None


def _quick_bbox_longest_edge(sf: StepFile) -> float:
    """Longest edge of the part's own bounding box, in FILE units (the space
    `deflection`/`max_edge` are compared in; `_process_face` tessellates before
    `convert_step`'s `scale` is applied). It needs no tessellation and no
    second parse, and turns the relative-tolerance percentage into an absolute
    value, as `export_step_brep` does for its percentage-of-bounding-box
    tolerance.

    The box is taken around the geometry (`_geometry_extent`), not every
    CARTESIAN_POINT of the file: those include the location of each plane and
    cylinder axis, which lie anywhere on the infinite object, and the global
    origin, so a part 1 m from the origin looked 1 m across and the "0.1 % of
    the part" came out far too coarse. Files with no edges, faces or
    tessellation to read fall back to all points.

    A B-spline surface's control points can lie a little outside the surface,
    so the result can overestimate the part size slightly; that only makes the
    derived tolerance a little coarser, never finer or slower. Returns 0.0 if
    the file has no points at all (the caller then keeps a fixed default)."""
    ext = _geometry_extent(sf) or _all_points_extent(sf)
    if ext is None:
        return 0.0
    lo, hi = ext
    return float(np.max(hi - lo))


# Below this many faces, starting worker processes is not worth it.
MIN_FACES_FOR_POOL = 48


def _tessellate_job(state):
    """Worker-side factory (see `workers.serve`): parse the file once, then
    tessellate one face per call. Mirrors the sequential per-face code in
    `iter_convert_step`. Returns plain data only."""
    sf = parse_file(state["path"])
    sf.angle_scale = state["angle_scale"]

    def run(face_ref):
        T.take_unsupported()
        msgs: list[str] = []
        try:
            verts3d, tris = _process_face(
                sf, face_ref, state["deflection"], state["max_edge"],
                edge_cache={}, debug=state["debug"], log=msgs.append,
                edge_hints=state["edge_hints"],
                flat_max_edge=state["flat_max_edge"])
        except Exception:  # noqa: BLE001 - a face that raises is skipped (and reported)
            verts3d, tris = None, None
            if state["debug"]:
                import traceback
                msgs.append(f"[StepForge] face {face_ref}: raised an exception, "
                            f"skipped:\n{traceback.format_exc()}")
        unsup = {k: {int(i) for i in ids} for k, ids in T.take_unsupported().items()}
        return verts3d, tris, msgs, unsup
    return run


def _tessellate_pooled(path, sf, face_refs, deflection, max_edge, edge_hints,
                       debug, debug_log, flat_max_edge, max_workers):
    """Generator: tessellate `face_refs` on worker processes. Yields
    `(frac, msg)`; returns `(results, unsupported)` -- results maps face ref
    to `(verts3d, tris)` -- or None if the workers failed (the caller then
    falls back to the sequential path). Faces are independent of each other
    (`edge_hints` is complete before any is dispatched and a shared edge is
    always sampled in its own fixed direction), so the output equals the
    sequential one; the welding stays sequential in the caller."""
    refs = [int(r) for r in face_refs]
    total = len(refs)
    n = W.worker_count(max_workers, total)
    state = {"path": path, "angle_scale": sf.angle_scale,
             "deflection": deflection, "max_edge": max_edge, "debug": debug,
             "flat_max_edge": flat_max_edge,
             "edge_hints": {int(k): float(v) for k, v in edge_hints.items()}}
    results: dict[int, tuple] = {}
    unsupported: dict[tuple[str, str], set] = {}
    done = 0
    msg = f"Tessellating {total} face(s) on {n} core(s)..."
    try:
        with W.WorkerPool("convert:_tessellate_job", state, n) as pool:
            for out in pool.imap(refs):
                if out is not None:
                    i, (verts3d, tris, msgs, unsup) = out
                    results[refs[i]] = (verts3d, tris)
                    if debug:
                        for m in msgs:
                            debug_log(m)
                    for k, ids in unsup.items():
                        unsupported.setdefault(k, set()).update(ids)
                    done += 1
                    msg = (f"Tessellating {total} face(s) on {n} core(s): "
                           f"{done}/{total}")
                yield 0.05 + 0.9 * (done / total), msg
    except Exception:  # noqa: BLE001 - any worker failure means "run single-core"
        if debug:
            import traceback
            debug_log("[StepForge] parallel tessellation failed, falling back "
                      f"to single-core:\n{traceback.format_exc()}")
        return None
    return results, unsupported


def convert_step(src, deflection: float = 0.05, scale: float | None = None,
                 max_edge: float | None = None, progress=None,
                 debug: bool = False, debug_log=print,
                 relative_pct: float | None = None,
                 flat_max_edge: bool = False,
                 keep_surfaces: bool = True,
                 parallel: bool = False,
                 max_workers: int | None = None) -> list[Solid]:
    """Convert a STEP file (path or StepFile) into a list of Solid meshes.

    Runs to completion and returns the result. `progress`, if given, is
    called as `progress(frac_0_1, message)`. See `iter_convert_step` for the
    same work as a generator (what the Blender operator drives, so the UI
    stays responsive).
    """
    gen = iter_convert_step(src, deflection, scale, max_edge, debug, debug_log,
                            relative_pct, flat_max_edge, keep_surfaces,
                            parallel, max_workers)
    try:
        while True:
            frac, msg = next(gen)
            if progress:
                progress(frac, msg)
    except StopIteration as done:
        return done.value


def iter_convert_step(src, deflection: float = 0.05,
                      scale: float | None = None,
                      max_edge: float | None = None,
                      debug: bool = False, debug_log=print,
                      relative_pct: float | None = None,
                      flat_max_edge: bool = False,
                      keep_surfaces: bool = True,
                      parallel: bool = False,
                      max_workers: int | None = None):
    """Generator behind `convert_step`: yields `(frac_0_1, message)` between
    units of work (once per face) and returns the list of Solid meshes.

    A caller that must stay responsive (a Blender modal operator) runs it a few
    milliseconds at a time. With `parallel=True` (and a file path as `src`, and
    enough faces) the per-face tessellation runs on worker processes
    (`core/workers.py`); the generator keeps yielding while it polls for
    results.

    `relative_pct`, when given (and > 0), overrides `deflection`/`max_edge` with
    values scaled off the file's OWN bounding box: `deflection =
    bbox_longest_edge * relative_pct / 100`, and `max_edge` is rescaled by the
    same factor so it keeps the ratio of the caller's `deflection`/`max_edge`
    pair (e.g. a quality preset's 8 mm : 0.15 mm). It mirrors the export side's
    `freeform_tolerance_pct` (see `_quick_bbox_longest_edge`). A fixed absolute
    deflection is far finer than needed for a part spanning metres and far
    coarser than needed for one a few millimetres across; this makes the
    requested detail independent of part size.

    `flat_max_edge`: also apply `max_edge` to planar faces (interior grid and
    edges shared only by planar faces). Off by default: a plane is exact with
    just its boundary, and the grid made most of the triangles of a typical
    sheet-metal part (see `_planar_only_edges`)."""
    yield 0.0, "Parsing STEP file..."
    T.take_unsupported()  # start this file's record from empty
    sf = src if isinstance(src, StepFile) else parse_file(src)
    sf.angle_scale = _plane_angle_scale(sf)
    if scale is None:
        scale = _length_scale(sf)

    if relative_pct:
        yield 0.01, "Measuring part size..."
        bbox_edge = _quick_bbox_longest_edge(sf)
        if bbox_edge > 0:
            new_deflection = bbox_edge * relative_pct / 100.0
            if new_deflection > 1e-9:
                if max_edge:
                    max_edge = max_edge * (new_deflection / max(deflection, 1e-9))
                deflection = new_deflection
                if debug_log:
                    debug_log(
                        f"[StepForge] Relative tolerance: bbox longest edge "
                        f"{bbox_edge:.3g} (file units) x {relative_pct:.3g}% "
                        f"-> deflection={deflection:.4g}, "
                        f"max_edge={max_edge if max_edge else 'auto'}")

    main_name = _product_name(sf)

    yield 0.015, "Planning assembly structure..."
    plan = _plan_assembly(sf) or _plan_flat(sf, main_name)
    if not plan:
        tess = _collect_tessellated(sf, scale)
        item_col = styled_colours(sf)
        for sol in tess:
            sol.colour = item_col.get(getattr(sol, "_item", -1))
        return ConvertResult(tess)

    colours = face_colours(sf)
    surf_records: dict[int, dict] = {}
    if keep_surfaces:
        for fref in {int(f) for _, fr, _ in plan for f in fr}:
            face = sf.get(fref)
            try:
                srf = T.build_surface(sf, face.params[2]) if face is not None else None
            except Exception:  # noqa: BLE001 - an unbuildable surface just has no record
                srf = None
            rec = surface_record(srf) if srf is not None and not srf.sense_flip else None
            if rec is not None:
                surf_records[fref] = rec
        T.take_unsupported()   # already reported by the tessellation pass
    total = sum(len(fr) for _, fr, _ in plan) or 1
    done = 0
    weld = 1.0 / max(deflection * 0.1, 1e-6)
    cache = {}
    edge_cache = {}
    edge_hints = {}
    out: list[Solid] = []

    # Deduplicated before the edge-hints pre-pass: an assembly that repeats a
    # component N times (bolts, washers) reuses the same `face_refs` list for
    # every instance, and walking `plan` directly would re-run
    # `_register_edge_hints` (which tessellates every boundary loop at full
    # resolution) once per instance instead of once per unique component.
    unique_lists = {}
    for _, face_refs, _ in plan:
        unique_lists.setdefault(id(face_refs), face_refs)
    all_face_refs: list[int] = []
    seen_refs = set()
    for face_refs in unique_lists.values():
        for fref in face_refs:
            if fref not in seen_refs:
                seen_refs.add(fref)
                all_face_refs.append(fref)

    if not flat_max_edge:
        # math.inf = "never subdivide this straight edge" (see T._walk_loop)
        for e in _planar_only_edges(sf, all_face_refs):
            edge_hints[e] = math.inf
    n_hint_faces = len(all_face_refs)
    for i, fref in enumerate(all_face_refs):
        try:
            _register_edge_hints(sf, fref, deflection, edge_hints, max_edge)
        except Exception:  # noqa: BLE001, S110 - a face without hints is tessellated without them
            pass
        yield (0.02 + 0.03 * ((i + 1) / max(n_hint_faces, 1)),
               f"Preparing edge hints: {i + 1}/{n_hint_faces} face(s)...")

    unsupported: dict[tuple[str, str], set] = {}
    failed_faces: list[int] = []
    face_results = None
    if (parallel and isinstance(src, (str, os.PathLike))
            and len(all_face_refs) >= MIN_FACES_FOR_POOL):
        pooled = yield from _tessellate_pooled(
            os.fspath(src), sf, all_face_refs, deflection, max_edge,
            edge_hints, debug, debug_log, flat_max_edge, max_workers)
        if pooled is not None:
            face_results, unsupported = pooled

    failed_set: set = set()
    for name, face_refs, matrix in plan:
        key = id(face_refs)
        base = cache.get(key)
        n_fail = 0
        if base is None:
            base = Mesh()
            for i, fref in enumerate(face_refs):
                if face_results is not None:
                    verts3d, tris = face_results.get(int(fref), (None, None))
                else:
                    try:
                        verts3d, tris = _process_face(sf, fref, deflection, max_edge,
                                                      edge_cache, debug=debug, log=debug_log,
                                                      edge_hints=edge_hints,
                                                      flat_max_edge=flat_max_edge)
                    except Exception:  # noqa: BLE001 - a face that raises is skipped (and reported)
                        verts3d, tris = None, None
                        if debug:
                            import traceback
                            debug_log(f"[StepForge] face {fref}: raised an exception, "
                                      f"skipped:\n{traceback.format_exc()}")
                if verts3d and tris:
                    base.add_triangles(verts3d, tris, weld, face_id=fref)
                    if debug:
                        surf_ref = sf.get(fref).params[2]
                        surf = sf.get(surf_ref)
                        surf_name = surf.name if surf else "?"
                        debug_log(f"[StepForge] face {fref} ({surf_name}): "
                                  f"{len(verts3d)} verts, {len(tris)} tris")
                else:
                    n_fail += 1
                    failed_faces.append(int(fref))
                    failed_set.add(int(fref))
                done += 1
                frac = done / total
                yield ((0.95 + 0.05 * frac) if face_results is not None
                       else (0.05 + 0.95 * frac),
                       (f"{name}: face {i + 1}/{len(face_refs)} "
                        f"({len(base.verts)} verts so far)"))
            _stitch_boundary_gaps(base, deflection)
            _drop_coincident_triangles(base)
            if deflection:
                _fill_small_holes(base, HOLE_WIDTH_CHORDS * deflection)
            cache[key] = base
        verts = base.verts
        if matrix is not None:
            R = matrix[:3, :3]
            t = matrix[:3, 3]
            verts = [tuple(R @ np.asarray(v) + t) for v in verts]
        if scale != 1.0:
            verts = [(x * scale, y * scale, z * scale) for (x, y, z) in verts]
        m = Mesh()
        m.verts = list(verts)
        m.faces = list(base.faces)
        m.face_ids = list(base.face_ids)
        fsurf = {}
        for f in face_refs:
            rec = surf_records.get(int(f))
            if rec is not None and int(f) not in failed_set:
                tr = transform_surface_record(rec, matrix, scale)
                if tr is not None:
                    fsurf[int(f)] = tr
        fcol = {int(f): colours[int(f)] for f in face_refs if int(f) in colours}
        solid_col = None
        if fcol and len(fcol) == len(face_refs) and len(set(fcol.values())) == 1:
            solid_col, fcol = next(iter(fcol.values())), {}
        out.append(Solid(name=name, mesh=m, n_faces_in=len(face_refs),
                         n_faces_failed=n_fail,
                         matrix=matrix.tolist() if matrix is not None else None,
                         colour=solid_col, face_colours=fcol,
                         face_surfaces=fsurf))

    for k, ids in T.take_unsupported().items():   # sequential path / hints
        unsupported.setdefault(k, set()).update(ids)
    out = ConvertResult(out)
    out.n_faces = len(all_face_refs)
    out.failed_faces = failed_faces
    out.unsupported = {k: len(v) for k, v in unsupported.items()}
    if debug:
        total_in = sum(s.n_faces_in for s in out)
        total_failed = sum(s.n_faces_failed for s in out)
        total_tris = sum(len(s.mesh.faces) for s in out)
        total_verts = sum(len(s.mesh.verts) for s in out)
        debug_log(f"[StepForge] done: {len(out)} solid(s), {total_in} STEP face(s) "
                  f"({total_failed} failed/skipped), {total_verts} verts, "
                  f"{total_tris} triangles -- see above for any per-face warnings")
    return out



def _plan_flat(sf, main_name):
    solids_faces = _collect_solids(sf)
    if not any(solids_faces):
        return []
    names = _all_product_names(sf)
    plan = []
    for si, fr in enumerate(solids_faces):
        if len(solids_faces) == 1:
            nm = main_name
        elif si < len(names):
            nm = names[si]
        else:
            nm = f"{main_name}_{si + 1:03d}"
        plan.append((nm, fr, None))
    return plan


def _plan_assembly(sf):
    try:
        from . import assembly as _asm
        tree = _asm.build_tree(sf)
        if tree is None:
            return None
        roots, children = tree
        pd_name, pd_solids = _pd_to_face_lists(sf)
        if len(pd_solids) < 2:
            return None
        plan = []

        def walk(pd, mat, depth):
            if depth > 32:
                return
            for fl in pd_solids.get(pd, []):
                plan.append((pd_name.get(pd, f"Part_{pd}"), fl, mat))
            for child, cmat, _ in children.get(pd, []):
                walk(child, mat @ cmat, depth + 1)

        eye = np.eye(4)
        for r in roots:
            walk(r, eye, 0)
        return plan or None
    except Exception:  # noqa: BLE001 - an unresolvable assembly falls back to a flat list
        return None
