"""Write triangle meshes to a STEP file (ISO 10303-21).

Blender works with polygon meshes, so the natural and standards-compliant way
to put that back into STEP is the AP242 *tessellated* geometry model. Each
solid is written the way Open CASCADE (FreeCAD, CadQuery, ...) writes and
reads it:

    COORDINATES_LIST              every vertex of the solid, once
    TRIANGULATED_FACE (per face)  normals per point, pnindex into the list
    TESSELLATED_SOLID / _SHELL    closed / open mesh
    TESSELLATED_SHAPE_REPRESENTATION + product chain (keeps the part name)

A "face" is a group of triangles: the STEP face each came from, when the mesh
still carries that (see `Mesh.face_ids`), otherwise a region bounded by
edges sharper than 30 degrees -- so other programs shade it the way it looks
in Blender. Normals are the area-weighted average over the face's triangles
at each point. StepForge's own importer reads this layout and the older
TRIANGULATED_FACE_SET files. Pure Python + NumPy.
"""
from __future__ import annotations

import datetime
import math
from collections.abc import Sequence

import numpy as np

# Decimal places a coordinate is written with, in output units (nanometres at
# the default mm scale). Rounding to a fixed decimal grid makes export ->
# re-import a fixed point from the first cycle: the writer scales metres by
# 1000.0 and the reader millimetres by 0.001, and (x * 1000.0) * 0.001 != x in
# binary floating point, whereas round(x * 1000, 6) recovers the written value
# exactly. Costs at most half a nanometre of geometric error.
COORD_DECIMALS = 6


def _num(x: float) -> str:
    # STEP reals always need a decimal point.
    if x == 0:
        return "0."
    s = repr(float(x)).upper()
    mant, _, exp = s.partition("E")
    if "." not in mant:
        mant += "."          # Part 21 reals need the point, "5.E-06" too
    return mant + ("E" + exp if exp else "")


def _coord(x: float) -> str:
    """Write one coordinate, snapped to the canonical decimal grid."""
    return _num(round(float(x), COORD_DECIMALS))


def step_str(text) -> str:
    """Body of a Part 21 string: ' doubled, reverse solidus doubled, and
    every non-ASCII character as \\X2\\hhhh\\X0\\ (so a name like
    "Träger" or "Bob's part" survives any reader)."""
    out = []
    for ch in str(text):
        if ch == "'":
            out.append("''")
        elif ch == "\\":
            out.append("\\\\")
        elif 32 <= ord(ch) < 127:
            out.append(ch)
        elif ord(ch) <= 0xFFFF:
            out.append(f"\\X2\\{ord(ch):04X}\\X0\\")
        else:
            out.append(f"\\X4\\{ord(ch):08X}\\X0\\")
    return "".join(out)


def step_header(description: str, path: str, author: str) -> str:
    """HEADER section. FILE_NAME carries the file's base name only (not the
    full path, which would publish the local folder and user name)."""
    now = datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S")
    name = step_str(str(path).replace("\\", "/").rsplit("/", 1)[-1])
    return (
        "ISO-10303-21;\n"
        "HEADER;\n"
        f"FILE_DESCRIPTION(('{step_str(description)}'),'2;1');\n"
        f"FILE_NAME('{name}','{now}',('{step_str(author)}'),(''),"
        "'StepForge','StepForge','');\n"
        "FILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF'));\n"
        "ENDSEC;\n"
    )


# Edges sharper than this split the triangles into separate faces when the
# mesh carries no STEP face ids (Blender's default auto-smooth angle).
REGION_ANGLE_DEG = 30.0


def _face_groups(verts: np.ndarray, faces: np.ndarray,
                 face_ids) -> tuple[list[np.ndarray], np.ndarray]:
    """Partition triangle indices into faces: connected triangles with the
    same source face id, or (id 0 / no ids) meeting at less than
    REGION_ANGLE_DEG. Groups are ordered by their first triangle and keep
    the mesh's triangle order, so a re-imported file writes back unchanged."""
    n = len(faces)
    ids = (np.asarray(face_ids, dtype=np.int64) if face_ids is not None and len(face_ids) == n
           else np.zeros(n, dtype=np.int64))
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    nrm = np.cross(b - a, c - a)
    ln = np.linalg.norm(nrm, axis=1)
    unit = nrm / np.where(ln > 0, ln, 1.0)[:, None]
    cos_lim = math.cos(math.radians(REGION_ANGLE_DEG))

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edge_tris = {}
    for t in range(n):
        f = faces[t]
        for i in range(3):
            u, v = int(f[i]), int(f[(i + 1) % 3])
            edge_tris.setdefault((u, v) if u < v else (v, u), []).append(t)
    for tris in edge_tris.values():
        if len(tris) != 2:
            continue
        t0, t1 = tris
        if ids[t0] != ids[t1]:
            continue
        if ids[t0] == 0 and float(np.dot(unit[t0], unit[t1])) < cos_lim:
            continue
        r0, r1 = find(t0), find(t1)
        if r0 != r1:
            parent[max(r0, r1)] = min(r0, r1)
    groups = {}
    order = []
    for t in range(n):
        r = find(t)
        if r not in groups:
            groups[r] = []
            order.append(r)
        groups[r].append(t)
    return [np.asarray(groups[r], dtype=np.int64) for r in order], nrm


def _is_closed(faces: np.ndarray) -> bool:
    """Every edge used by exactly two triangles."""
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool(len(counts)) and bool(np.all(counts == 2))


def _normal_str(v) -> str:
    return f"({','.join(_num(round(float(x), 6)) for x in v)})"


class _Writer:
    def __init__(self):
        self.lines: list[str] = []
        self.id = 0
        # Memo slots for the singleton PRODUCT_CONTEXT and
        # PRODUCT_DEFINITION_CONTEXT entities (see _pctx / _pdc).
        self._pctx_id: int | None = None
        self._pdc_id: int | None = None

    def add(self, body: str) -> int:
        self.id += 1
        self.lines.append(f"#{self.id}={body};")
        return self.id


def export_step(solids: Sequence, path: str, author: str = "StepForge",
                scale: float = 1000.0, progress=None) -> None:
    """Write `solids` (objects with `.name` and `.mesh.verts/.faces`, and
    optionally `.mesh.face_ids`) to `path`.

    scale : multiply coordinates on the way out. Blender works in metres; STEP
            mechanical data is conventionally millimetres, so default 1000.
    progress : optional callable(frac_0_1, message) -- writing is a single
               fast string-formatting pass, so this only reports coarse
               per-solid checkpoints rather than fine-grained sub-progress.
    """
    w = _Writer()
    usable = [s for s in solids if s.mesh.verts and s.mesh.faces]
    n_solids = max(len(usable), 1)

    # --- shared application context ---------------------------------------
    app_ctx = w.add("APPLICATION_CONTEXT('managed model based 3d engineering')")
    w.add("APPLICATION_PROTOCOL_DEFINITION('international standard',"
          f"'ap242_managed_model_based_3d_engineering',2014,#{app_ctx})")

    # length / angle / solid-angle units in a geometric context
    luni = w.add("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))")
    auni = w.add("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
    suni = w.add("(NAMED_UNIT(*)SOLID_ANGLE_UNIT()SI_UNIT($,.STERADIAN.))")
    unc = w.add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-06),#{luni},"
                "'distance_accuracy_value','confusion accuracy')")
    geo_ctx = w.add(
        "(GEOMETRIC_REPRESENTATION_CONTEXT(3)"
        f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc}))"
        f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{luni},#{auni},#{suni}))"
        "REPRESENTATION_CONTEXT('Context','3D'))")

    for si, solid in enumerate(usable):
        verts = solid.mesh.verts
        faces = solid.mesh.faces
        name = step_str(solid.name or "Part")
        if progress is not None:
            progress(si / n_solids, f"Writing {solid.name or 'Part'}...")

        # product / definition chain (gives the part its name on re-import)
        prod = w.add(f"PRODUCT('{name}','{name}','',(#{_pctx(w, app_ctx)}))")
        pdf = w.add(f"PRODUCT_DEFINITION_FORMATION('','',#{prod})")
        pd = w.add(f"PRODUCT_DEFINITION('design','',#{pdf},#{_pdc(w, app_ctx)})")
        pds = w.add(f"PRODUCT_DEFINITION_SHAPE('','',#{pd})")

        coord_strs = ",".join(
            f"({_coord(x * scale)},{_coord(y * scale)},{_coord(z * scale)})"
            for (x, y, z) in verts)
        cl = w.add(f"COORDINATES_LIST('',{len(verts)},({coord_strs}))")

        # normals from the coordinates as written, so a re-imported file
        # produces the very same normals (fixed point from the first cycle)
        V = np.round(np.asarray(verts, dtype=float) * scale, COORD_DECIMALS)
        F = np.asarray(faces, dtype=np.int64)
        groups, tri_nrm = _face_groups(V, F, getattr(solid.mesh, "face_ids", None))
        face_refs = []
        for g in groups:
            gf = F[g]
            # keep points in order of first use (stable across round trips)
            first = {}
            for vid in gf.ravel():
                first.setdefault(int(vid), len(first))
            pts = sorted(first, key=first.get)
            pos = {v: k for k, v in enumerate(pts)}
            acc = np.zeros((len(pts), 3))
            for t, (i0, i1, i2) in zip(g, gf):
                for vid in (i0, i1, i2):
                    acc[pos[int(vid)]] += tri_nrm[t]
            ln = np.linalg.norm(acc, axis=1)
            acc = np.where(ln[:, None] > 0, acc / np.where(ln > 0, ln, 1.0)[:, None],
                           np.array([0.0, 0.0, 1.0]))
            normals = ",".join(_normal_str(v) for v in acc)
            pnindex = ",".join(str(v + 1) for v in pts)
            tris = ",".join(f"({pos[int(i0)] + 1},{pos[int(i1)] + 1},{pos[int(i2)] + 1})"
                            for (i0, i1, i2) in gf)
            face_refs.append(w.add(
                f"TRIANGULATED_FACE('',#{cl},{len(pts)},({normals}),$,"
                f"({pnindex}),({tris}))"))
        kind = "TESSELLATED_SOLID" if _is_closed(F) else "TESSELLATED_SHELL"
        item = w.add(f"{kind}('{name}',({','.join(f'#{f}' for f in face_refs)}),$)")
        rep = w.add(f"TESSELLATED_SHAPE_REPRESENTATION('{name}',(#{item}),"
                    f"#{geo_ctx})")
        w.add(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})")

    body = "\n".join(w.lines)
    text = (step_header("StepForge tessellated export", path, author) +
            "DATA;\n"
            f"{body}\n"
            "ENDSEC;\n"
            "END-ISO-10303-21;\n")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    if progress is not None:
        progress(1.0, "Done")


# Singleton support entities, created on first use and memoized per writer.


def _pctx(w, app_ctx):
    if w._pctx_id is None:
        w._pctx_id = w.add(f"PRODUCT_CONTEXT('',#{app_ctx},'mechanical')")
    return w._pctx_id


def _pdc(w, app_ctx):
    if w._pdc_id is None:
        w._pdc_id = w.add(
            f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app_ctx},'design')")
    return w._pdc_id
