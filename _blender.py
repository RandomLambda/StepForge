"""Blender-facing operators, file handler and menus for StepForge.

Imported only when running inside Blender (see __init__.py). All bpy access
lives here so that `step_forge.core` stays importable in a plain interpreter.
"""
import json
import math
import os
import time
import traceback

import blf
import bpy
import gpu
import numpy as np
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    StringProperty,
)
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper, ImportHelper
from gpu_extras.batch import batch_for_shader

from .core import jobs, workers
from .core.convert import transform_surface_record as _transform_surface_record

# ---------------------------------------------------------------------------
# Progress bar overlay, drawn in every 3D viewport during import and export by
# a `SpaceView3D` `POST_PIXEL` draw handler (labelled, unlike the cursor
# readout of `wm.progress_begin`).
# ---------------------------------------------------------------------------
_BAR_W = 340
_BAR_H = 22
_BAR_MARGIN = 18

# The import/export operator that is currently running, if any. Only one runs
# at a time (`_ModalWork._begin` refuses to start a second): the conversion
# code keeps a little per-run state at module level.
_ACTIVE_PROGRESS_OPS: list = []


def _draw_progress_bar(op):
    frac, msg = op._progress
    frac = max(0.0, min(1.0, frac))
    region = bpy.context.region
    if region is None:
        return
    x0 = _BAR_MARGIN
    y0 = _BAR_MARGIN
    x1 = x0 + _BAR_W
    y1 = y0 + _BAR_H

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')

    def rect(rx0, ry0, rx1, ry1, color):
        verts = ((rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1))
        indices = ((0, 1, 2), (2, 3, 0))
        batch = batch_for_shader(shader, 'TRIS', {"pos": verts}, indices=indices)
        shader.uniform_float("color", color)
        batch.draw(shader)

    # Backdrop, track, then the filled portion on top.
    rect(x0 - 4, y0 - 4, x1 + 4, y1 + 4, (0.0, 0.0, 0.0, 0.55))
    rect(x0, y0, x1, y1, (1.0, 1.0, 1.0, 0.12))
    if frac > 0.0:
        rect(x0, y0, x0 + _BAR_W * frac, y1, (0.25, 0.55, 1.0, 0.9))

    gpu.state.blend_set('NONE')

    font_id = 0
    blf.size(font_id, 12)
    blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
    blf.position(font_id, x0 + 6, y0 + 6, 0)
    label = f"StepForge: {msg}  ({frac * 100:.0f}%)"
    blf.draw(font_id, label)


def _tag_all_view3d_redraw():
    for wm_win in bpy.context.window_manager.windows:
        for area in wm_win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


# Quality presets -> (deflection mm, max_edge mm for flat faces)
QUALITY_PRESETS = {
    "ROUGH": (0.5, 20.0),
    "MEDIUM": (0.15, 8.0),
    "FINE": (0.05, 4.0),
    "VERY_FINE": (0.015, 2.0),
}


# ---------------------------------------------------------------------------
# Running long work from a modal operator
#
# Import and export are jobs (`core/jobs.py`): generators that yield plain-data
# messages. They run in a background process (`core/workers.py`), so the heavy
# work (parsing, tessellation, curve fitting, writing) never runs in Blender's
# own process and cannot slow the UI down. What stays here is what only Blender
# can do: reading mesh data and building it, in short slices.
#
# An operator's work is a generator too. It yields `(fraction, text)` for the
# progress bar, or None for "nothing to do right now". A timer runs it for a
# few milliseconds per tick and then hands control back to Blender, so the UI
# keeps redrawing. It only polls the background process and never waits for it.
# ---------------------------------------------------------------------------

# How long one timer tick may work before yielding to the UI, and how often
# the timer fires.
_SLICE_SECONDS = 0.01
_TIMER_SECONDS = 0.02


# False runs everything inside Blender's own process (the work is then divided
# into slices, but single steps such as parsing can still block the UI).
_BACKGROUND_PROCESSES = True


def _can_use_workers():
    """False if background processes are switched off, or while a script runs as
    `__main__` (`blender --python`, or a text run from the editor):
    `multiprocessing` would execute it again in every process it starts, so
    those runs stay in this process."""
    import __main__
    return _BACKGROUND_PROCESSES and not getattr(__main__, "__file__", None)


class _InProcessTask:
    """Steps a job generator in this process, with the interface of
    `workers.TaskProcess` (`poll`, `close`). Used where no background process
    can be used: headless Blender, scripts, a process that failed to start."""

    def __init__(self, gen):
        self._gen = gen
        self._finished = False

    def poll(self):
        if self._finished:
            return None
        try:
            return next(self._gen)
        except StopIteration as stop:
            self._finished = True
            return ("result", stop.value)

    def close(self):
        self._gen.close()


def _run_job(name, state, make_messages=None):
    """Start the job `name` of `core/jobs.py`; returns `(task, in_background)`.

    In a window session it runs in a background process. Otherwise (headless,
    a script) or if that process cannot be started, the same generator is
    stepped in this process. `make_messages()` yields what the job reads from
    its inbox (see `_drive` for how it is sent to a process)."""
    state = dict(state)
    options = dict(state.get("options") or {})
    state["options"] = options
    if not bpy.app.background and bpy.context.window is not None and _can_use_workers():
        if "parallel" in options:
            options["parallel"] = True
        try:
            return workers.TaskProcess(f"jobs:{name}", state), True
        except Exception:  # noqa: BLE001 - no process available: run in Blender instead
            print("[StepForge] Could not start the background process; running in "
                  "Blender (the window stays responsive only between steps):\n"
                  + traceback.format_exc())
    if "parallel" in options:
        options["parallel"] = _can_use_workers()
    inbox = workers.Inbox(None, list(make_messages()) if make_messages else ())
    return _InProcessTask(getattr(jobs, name)(state, inbox)), False


def _drive(task, on_message, outgoing=None, span=(0.0, 1.0)):
    """Generator: talk to `task` until its result arrives, and return it.

    Progress messages move the bar inside `span`; any other message goes to
    `on_message`. `outgoing`, an iterable of messages for a background
    process, is sent one message per step once the process says it is ready.
    Yields progress tuples, or None when nothing is to be done right now."""
    lo, hi = span
    progress = (lo, "Starting the background process...")
    out = iter(outgoing if outgoing is not None else ())
    ready = outgoing is None
    while True:
        msg = task.poll()
        if msg is not None:
            kind = msg[0]
            if kind == "result":
                return msg[1]
            if kind == "error":
                raise workers.WorkerError("the background process failed:\n" + msg[1])
            if kind == "ready":
                ready = True
            elif kind == "progress":
                progress = (lo + (hi - lo) * msg[1], msg[2])
            else:
                on_message(msg)
            yield progress
            continue
        if ready:
            nxt = next(out, None)
            if nxt is not None:
                task.send(nxt)
                yield progress
                continue
        yield None


class _ModalWork:
    """Mixin for an operator that runs a generator from a modal timer.

    Subclasses implement `_done(context, result)` (called with the generator's
    return value) and `_failed_message`. Esc cancels.
    """
    _failed_message = "StepForge failed (see console)"
    _work = None
    _timer = None
    _draw_handle = None

    def _begin(self, context, work):
        """Start `work`, a generator yielding `(frac_0_1, message)` (or None
        while it waits) and returning the value passed to `_done`."""
        if _ACTIVE_PROGRESS_OPS:
            self.report({"ERROR"}, "StepForge is already running an import or export")
            return {"CANCELLED"}
        self._work = work
        self._progress = (0.0, "Starting...")

        if bpy.app.background or context.window is None:
            # Scripts and command-line runs have no event loop to drive a
            # modal operator: run to the end right here.
            try:
                while True:
                    item = next(work)
                    if item is None:
                        time.sleep(0.005)
                    else:
                        self._progress = item
            except StopIteration as stop:
                return self._done(context, stop.value)
            except Exception:  # noqa: BLE001 - any failure is reported to the user
                return self._failed(context)

        wm = context.window_manager
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_progress_bar, (self,), 'WINDOW', 'POST_PIXEL')
        _ACTIVE_PROGRESS_OPS.append(self)
        _tag_all_view3d_redraw()
        self._timer = wm.event_timer_add(_TIMER_SECONDS, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC" and event.value == "PRESS":
            self._stop_modal(context)
            self._work.close()
            self.report({"WARNING"}, "StepForge: cancelled")
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        deadline = time.monotonic() + _SLICE_SECONDS
        try:
            while time.monotonic() < deadline:
                item = next(self._work)
                if item is None:
                    break
                self._progress = item
        except StopIteration as stop:
            self._stop_modal(context)
            return self._done(context, stop.value)
        except Exception:  # noqa: BLE001 - any failure is reported to the user
            self._stop_modal(context)
            return self._failed(context)
        frac, msg = self._progress
        context.workspace.status_text_set(f"StepForge: {msg}  ({frac * 100:.0f}%)")
        _tag_all_view3d_redraw()
        return {"PASS_THROUGH"}

    def cancel(self, context):
        # Blender ended the modal operator (e.g. a new file was loaded).
        self._stop_modal(context)
        if self._work is not None:
            self._work.close()

    def _stop_modal(self, context):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if self._draw_handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
            self._draw_handle = None
        if self in _ACTIVE_PROGRESS_OPS:
            _ACTIVE_PROGRESS_OPS.remove(self)
        _tag_all_view3d_redraw()
        context.workspace.status_text_set(None)

    def _failed(self, context):
        print(traceback.format_exc())
        self.report({"ERROR"}, self._failed_message)
        return {"CANCELLED"}


# ---------------------------------------------------------------------------
# Mesh building (main thread only)
#
# Building a mesh is a generator that yields between its steps, so the timer
# can give the UI a turn in between. Only a single Blender call (`from_mesh`,
# a bmesh operator) cannot be divided.
# ---------------------------------------------------------------------------

# Mesh attribute holding each polygon's source STEP face id, and the mesh
# property holding those faces' exact surfaces (JSON, mesh-local metres):
# Curved export writes a face's original surface back while its polygons
# still lie on it.
FACE_ID_ATTR = "stepforge_face"
SURFACES_PROP = "stepforge_surfaces"

# Elements handled between two yields in the loops over a bmesh.
_LOOP_STEP = 4096


def _fill_mesh(me, verts, faces):
    """Put a payload's vertices and triangles into `me`, as `Mesh.from_pydata`
    does, but from the NumPy arrays (a memory copy instead of a Python loop).
    Falls back to `from_pydata` if this Blender does not take them."""
    nv, nf = len(verts), len(faces)
    try:
        me.vertices.add(nv)
        me.loops.add(nf * 3)
        me.polygons.add(nf)
        me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
        me.polygons.foreach_set("loop_start", np.arange(0, nf * 3, 3, dtype=np.int32))
        me.polygons.foreach_set("vertices", faces.ravel())
        me.update(calc_edges=True)
        if len(me.polygons) == nf and (nf == 0 or (
                tuple(me.polygons[0].vertices) == tuple(faces[0])
                and tuple(me.polygons[nf - 1].vertices) == tuple(faces[-1]))):
            return
    except (RuntimeError, TypeError, ValueError, AttributeError):
        pass
    me.clear_geometry()
    me.from_pydata(verts.tolist(), [], faces.tolist())


def _renumber_faces(scene, face_ids):
    """Ids unique across the scene for one solid's STEP faces, numbered in order
    of first appearance. Returns (new id per triangle, {old id: new id})."""
    start = int(scene.get("stepforge_next_face_id", 1)) if scene is not None else 1
    used = face_ids != 0
    uniq, first = np.unique(face_ids[used], return_index=True)
    new = np.empty(len(uniq), dtype=np.int64)
    new[np.argsort(first, kind="stable")] = start + np.arange(len(uniq))
    per_tri = np.zeros(len(face_ids), dtype=np.int32)
    if len(uniq):
        per_tri[used] = new[np.searchsorted(uniq, face_ids[used])]
    mapping = dict(zip(uniq.tolist(), new.tolist(), strict=True))
    mapping[0] = 0
    if scene is not None:
        scene["stepforge_next_face_id"] = start + len(mapping)
    return per_tri, mapping


def _srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _colour_material(rgba):
    """One shared material per distinct STEP colour (STEP colours are sRGB
    display values; Blender material colours are linear)."""
    r, g, b, a = rgba
    name = f"StepForge #{round(r * 255):02X}{round(g * 255):02X}{round(b * 255):02X}"
    if a < 0.999:
        name += f" {round(a * 100)}%"
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    lin = (_srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b))
    mat.diffuse_color = (lin[0], lin[1], lin[2], a)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (lin[0], lin[1], lin[2], 1.0)
        if a < 0.999 and "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = a
    if a < 0.999:
        if hasattr(mat, "surface_render_method"):
            mat.surface_render_method = "BLENDED"
        elif hasattr(mat, "blend_method"):
            mat.blend_method = "BLEND"
    return mat


def _assign_colours(me, payload, face_ids):
    """Material slots from the solid's STYLED_ITEM colours: one slot for the
    solid colour (or an empty slot for uncoloured faces), one per distinct
    face colour. Slots are numbered in order of first appearance."""
    base = payload["colour"]
    fcol = payload["face_colours"]
    if base is None and not fcol:
        return
    uniq, first, inverse = np.unique(face_ids, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    ids = uniq[order].tolist()                 # STEP faces, by first appearance
    if base is None and len(face_ids) and all(f in fcol for f in ids):
        # every face coloured on its own: the commonest colour is the base
        counts = {}
        sizes = np.bincount(inverse, minlength=len(uniq))[order].tolist()
        for f, n in zip(ids, sizes, strict=True):
            counts[fcol[f]] = counts.get(fcol[f], 0) + n
        base = max(counts, key=counts.get)
    me.materials.append(_colour_material(base) if base is not None else None)
    if not fcol or not len(face_ids):
        return
    slots = {}
    slot_of = np.zeros(len(uniq), dtype=np.int32)
    for rank, f in enumerate(ids):
        col = fcol.get(f)
        if col is None or col == base:
            continue
        k = slots.get(col)
        if k is None:
            me.materials.append(_colour_material(col))
            k = slots[col] = len(me.materials) - 1
        slot_of[order[rank]] = k
    me.polygons.foreach_set("material_index", slot_of[inverse])


def _iter_cleanup_mesh(me, merge_distance, shading, smooth_angle_deg, progress):
    """Generator: weld coincident verts, drop degenerate faces, make normals
    point consistently outward, and shade the mesh; yields `progress` between
    the steps. Uses bmesh, so it works headless without an ops context.

    `shading`: FLAT (every facet hard), SMOOTH (every facet smooth) or AUTO:
    faces whose normals differ by less than `smooth_angle_deg` are smooth
    across their shared edge and anything sharper stays a hard edge, so
    fillets, cylinders and cones read as smooth while real corners stay crisp.
    It uses bmesh face-normal angles and sharp-edge marking rather than the
    Smooth by Angle modifier, so it behaves the same on every supported
    Blender version.
    """
    import bmesh
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        yield progress
        if merge_distance > 0.0:
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_distance)
            yield progress
        bmesh.ops.dissolve_degenerate(bm, dist=1e-7, edges=bm.edges)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        yield progress

        if shading in ("FLAT", "SMOOTH"):
            smooth = shading == "SMOOTH"
            for i, f in enumerate(bm.faces):
                f.smooth = smooth
                if not i % _LOOP_STEP:
                    yield progress
        else:  # AUTO
            thresh = math.cos(math.radians(max(0.0, min(180.0, smooth_angle_deg))))
            for i, f in enumerate(bm.faces):
                f.smooth = True
                if not i % _LOOP_STEP:
                    yield progress
            for i, e in enumerate(bm.edges):
                lf = e.link_faces
                if len(lf) != 2:
                    e.smooth = False
                else:
                    e.smooth = lf[0].normal.dot(lf[1].normal) >= thresh
                if not i % _LOOP_STEP:
                    yield progress
        bm.to_mesh(me)
    finally:
        bm.free()
    me.update()


def _iter_build_mesh(payload, build, scene, progress):
    """Generator: the Blender mesh for one solid's payload, built in steps that
    yield `progress` in between; returns the mesh."""
    me = bpy.data.meshes.new(payload["name"])
    try:
        _fill_mesh(me, payload["verts"], payload["faces"])
        yield progress
        face_ids = payload["face_ids"]
        if len(face_ids) == len(me.polygons) and len(face_ids):
            if build["colours"]:
                _assign_colours(me, payload, face_ids)
            # Which STEP face each polygon came from; survives the cleanup
            # below and ordinary editing (new polygons get 0 = unknown).
            # Renumbered to ids unique in the scene, so joining parts (or
            # instances of one part) never mixes two faces under one id.
            per_tri, mapping = _renumber_faces(scene, face_ids)
            attr = me.attributes.new(FACE_ID_ATTR, "INT", "FACE")
            attr.data.foreach_set("value", per_tri)
            entries = [f'"{mapping[f]}":{text}' for f, text in payload["surfaces"].items()
                       if f in mapping]
            if entries:
                me[SURFACES_PROP] = "{" + ",".join(entries) + "}"
        elif build["colours"]:
            _assign_colours(me, payload, face_ids[:0])
        me.validate(verbose=False)
        yield progress
        yield from _iter_cleanup_mesh(me, build["merge_distance"], build["shading"],
                                      build["smooth_angle"], progress)
    except BaseException:
        bpy.data.meshes.remove(me)
        raise
    return me


def _iter_build_objects(payloads, file_name, build, created, lo, hi):
    """Generator: one object per solid, named from its STEP product; the
    progress bar moves from `lo` to `hi`. Everything made is appended to
    `created` (so a cancelled import can remove it again); returns the objects.

    A wrapper collection is created ONLY for assemblies / multi-body results;
    a single solid is linked straight into the active collection.
    """
    context = bpy.context
    target = context.collection  # active collection
    if len(payloads) > 1:
        target = bpy.data.collections.new(file_name)
        context.scene.collection.children.link(target)
        created.append(target)

    objs = []
    for i, payload in enumerate(payloads):
        progress = (lo + (hi - lo) * i / len(payloads),
                    f"Building {file_name}: {i + 1}/{len(payloads)}")
        yield progress
        me = yield from _iter_build_mesh(payload, build, context.scene, progress)
        created.append(me)
        # The mesh already carries the instance placement (convert_step
        # applies it to the vertices), so the object keeps an identity matrix.
        ob = bpy.data.objects.new(payload["name"], me)
        target.objects.link(ob)
        created.append(ob)
        objs.append(ob)
    return objs


def _discard(created):
    """Remove what `_iter_build_objects` made (a cancelled or failed import)."""
    for idb in reversed(created):
        try:
            if isinstance(idb, bpy.types.Object):
                bpy.data.objects.remove(idb)
            elif isinstance(idb, bpy.types.Mesh):
                bpy.data.meshes.remove(idb)
            elif isinstance(idb, bpy.types.Collection):
                bpy.data.collections.remove(idb)
        except ReferenceError:
            pass  # already gone (joined into another object)


# Share of the progress bar for the background conversion; building the
# objects in Blender takes the rest.
_CONVERT_SHARE = 0.9


def _import_work(paths, options, build):
    """Generator behind the import operator: the files are converted in the
    background, then the objects are built here. Returns what `_done` reports."""
    task, _ = _run_job("import_job", {"paths": paths, "options": options})
    receiver = jobs.SolidReceiver()
    payloads = [[] for _ in paths]

    def collect(msg):
        if msg[0] in jobs.SOLID_KINDS:
            payload = receiver.feed(msg)
            if payload is not None:
                payloads[payload["tag"]].append(payload)

    try:
        infos = yield from _drive(task, collect, span=(0.0, _CONVERT_SHARE))
    finally:
        task.close()

    created, imported = [], []
    parts = built = 0
    n_solids = sum(len(group) for group in payloads) or 1
    try:
        for path, group in zip(paths, payloads, strict=True):
            file_name = os.path.splitext(os.path.basename(path))[0]
            lo = _CONVERT_SHARE + (1.0 - _CONVERT_SHARE) * built / n_solids
            built += len(group)
            hi = _CONVERT_SHARE + (1.0 - _CONVERT_SHARE) * built / n_solids
            objs = yield from _iter_build_objects(group, file_name, build, created, lo, hi)
            group.clear()
            parts += len(objs)
            if build["join"] and len(objs) > 1:
                yield hi, f"Joining {file_name}..."
                _join(bpy.context, objs, file_name)
                objs = objs[:1]
            imported.extend(objs)
        _select_objects(bpy.context, imported)
    except BaseException:
        _discard(created)
        raise
    return {
        "parts": parts,
        "n_faces": sum(info["n_faces"] for info in infos),
        "n_failed": sum(info["n_failed"] for info in infos),
        "lines": [line for info in infos for line in info["lines"]],
    }


# ---------------------------------------------------------------------------
# Import operator (modal timer, so the UI stays responsive)
# ---------------------------------------------------------------------------

class STEPFORGE_OT_import(_ModalWork, Operator, ImportHelper):
    """Import a STEP (.step/.stp) file"""
    bl_idname = "import_scene.stepforge"
    bl_label = "Import STEP (StepForge)"
    bl_options = {"REGISTER", "UNDO", "PRESET"}  # noqa: RUF012
    _failed_message = "StepForge import failed (see console)"

    filename_ext = ".step"
    filter_glob: StringProperty(default="*.step;*.stp", options={"HIDDEN"})
    files: CollectionProperty(type=bpy.types.OperatorFileListElement,
                              options={"HIDDEN", "SKIP_SAVE"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN", "SKIP_SAVE"})

    quality: EnumProperty(
        name="Accuracy",
        description="How finely curved surfaces are approximated (rough = fast "
                    "and light, fine = dense and slower)",
        items=[
            ("ROUGH", "Rough", "Fastest, coarse facets"),
            ("MEDIUM", "Medium", "Balanced (default)"),
            ("FINE", "Fine", "Smooth curves, denser mesh"),
            ("VERY_FINE", "Very Fine", "Highest fidelity, slowest"),
            ("CUSTOM", "Custom", "Use the exact values below"),
            ("RELATIVE", "Relative to Part Size",
             "Tolerance as a percentage of the part's own bounding box"),
        ],
        default="MEDIUM")
    deflection: FloatProperty(
        name="Chord Tolerance (mm)", default=0.15, min=0.001, max=10.0,
        description="Custom max deviation of a facet from the true surface")
    max_edge: FloatProperty(
        name="Max Flat Edge (mm)", default=8.0, min=0.1, max=1000.0,
        description="Custom target triangle edge length along straight "
                    "edges next to curved faces, and on flat faces when "
                    "Subdivide Flat Faces is on")
    flat_max_edge: BoolProperty(
        name="Subdivide Flat Faces", default=False,
        description="Also split flat faces into triangles no longer than Max "
                    "Flat Edge (denser, evenly sized mesh, e.g. for "
                    "deformation). Off: a flat face uses only its outline, "
                    "which is exact and much lighter")
    relative_pct: FloatProperty(
        name="Detail", subtype="PERCENTAGE", default=0.1, min=0.005, max=5.0,
        precision=3,
        description="Chord tolerance as a percentage of the part's own "
                    "longest bounding-box dimension (Max Flat Edge scales "
                    "with it, at the ratio Medium quality uses). Lower = "
                    "finer and slower, higher = coarser and faster")
    unit: EnumProperty(
        name="File Units",
        items=[("AUTO", "Auto", "Read the unit from the file header"),
               ("MM", "Millimetres", "Treat file as mm"),
               ("M", "Metres", "Treat file as metres")],
        default="AUTO")
    merge_distance: FloatProperty(
        name="Weld Distance (m)", default=0.0001, min=0.0, max=0.01, precision=5,
        description="Merge verts closer than this after import")
    shading: EnumProperty(
        name="Shading",
        description="How faces are shaded",
        items=[
            ("AUTO", "Smooth by Angle (recommended)",
             "Sharp corners stay crisp, curved surfaces shade smooth"),
            ("FLAT", "Flat", "Every triangle facet visible: the classic hard-edge CAD look"),
            ("SMOOTH", "Smooth", "Every face smooth-shaded, including flat panels"),
        ],
        default="AUTO")
    smooth_angle: FloatProperty(
        name="Smooth Angle", default=30.0, min=0.0, max=180.0, subtype="NONE",
        description="Faces meeting at less than this angle (degrees) are "
                    "shaded smooth across their shared edge; only used when "
                    "Shading = Smooth by Angle")
    join_solids: BoolProperty(name="Join Into Single Object", default=False)
    import_colours: BoolProperty(
        name="Import Colours", default=True,
        description="Create materials from the colours stored in the STEP "
                    "file (per part and per face)")
    debug: BoolProperty(
        name="Debug Prints (Console)", default=False,
        description="Print per-face tessellation diagnostics to the system "
                    "console while importing; useful when reporting a bad "
                    "STEP file")
    manual: BoolProperty(
        name="Manual Settings", default=False,
        description="Show every import quality option (fixed presets, units, "
                    "weld distance, shading, debug) instead of just the "
                    "Detail slider")

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "manual")
        if not self.manual:
            layout.prop(self, "relative_pct")
            layout.prop(self, "join_solids")
            layout.prop(self, "import_colours")
            return
        layout.prop(self, "quality")
        if self.quality == "CUSTOM":
            layout.prop(self, "deflection")
            layout.prop(self, "max_edge")
        elif self.quality == "RELATIVE":
            layout.prop(self, "relative_pct")
        layout.prop(self, "flat_max_edge")
        layout.prop(self, "unit")
        layout.prop(self, "merge_distance")
        layout.prop(self, "shading")
        if self.shading == "AUTO":
            layout.prop(self, "smooth_angle")
        layout.prop(self, "join_solids")
        layout.prop(self, "import_colours")
        layout.prop(self, "debug")

    def execute(self, context):
        paths = [os.path.join(self.directory, f.name) for f in self.files] \
            if self.files else [self.filepath]
        paths = [p for p in paths if p]
        relative_pct = None
        # With Manual Settings off the Accuracy dropdown is hidden: behave as
        # if RELATIVE were picked, with the current Detail %.
        quality = self.quality if self.manual else "RELATIVE"
        if quality == "CUSTOM":
            defl, max_edge = self.deflection, self.max_edge
        elif quality == "RELATIVE":
            # MEDIUM's (deflection, max_edge) pair is the fallback if the file
            # has no points to measure, and otherwise only the ratio that
            # `convert_step` keeps once the bounding-box-relative deflection is
            # known (see its `relative_pct` handling).
            defl, max_edge = QUALITY_PRESETS["MEDIUM"]
            relative_pct = self.relative_pct
        else:
            defl, max_edge = QUALITY_PRESETS[quality]
        scale = {"MM": 0.001, "M": 1.0, "AUTO": None}[self.unit]
        # Read the properties now: the work below runs after execute()
        # returns. ("parallel" is only a switch: `_run_job` sets it.)
        options = {
            "deflection": defl, "scale": scale, "max_edge": max_edge,
            "debug": self.debug, "relative_pct": relative_pct,
            "flat_max_edge": bool(self.flat_max_edge and self.manual),
            "parallel": True,
        }
        build = {
            "merge_distance": self.merge_distance, "shading": self.shading,
            "smooth_angle": self.smooth_angle, "colours": self.import_colours,
            "join": self.join_solids,
        }
        return self._begin(context, _import_work(paths, options, build))

    def _done(self, context, summary):
        total, n_faces, n_failed = summary["parts"], summary["n_faces"], summary["n_failed"]
        warnings = summary["lines"]
        for line in warnings:
            print("[StepForge] " + line)
        if n_failed:
            self.report({"WARNING"},
                        f"StepForge: imported {total} part(s), but {n_failed} of "
                        f"{n_faces} faces failed and are missing (holes in the "
                        f"mesh). Details in the system console.")
        elif warnings:
            self.report({"WARNING"},
                        f"StepForge: imported {total} part(s); some edges use "
                        f"curve types StepForge cannot read and are drawn "
                        f"straight. Details in the system console.")
        else:
            self.report({"INFO"}, f"StepForge: imported {total} part(s)")
        return {"FINISHED"}


def _join(context, objs, name):
    table = {}
    for o in objs:
        raw = o.data.get(SURFACES_PROP)
        if raw:
            # each part's table is in its own mesh space; join moves every
            # part into the active object's space
            rel = objs[0].matrix_world.inverted() @ o.matrix_world
            for k, rec in json.loads(raw).items():
                tr = _transform_surface_record(rec, [list(r) for r in rel])
                if tr is not None:
                    table[k] = tr
    for o in context.selected_objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    context.view_layer.objects.active = objs[0]
    objs[0].name = name
    bpy.ops.object.join()
    if table:
        objs[0].data[SURFACES_PROP] = json.dumps(table, separators=(",", ":"))


def _select_objects(context, objs):
    """Select `objs` and make the first one active, as Blender's own importers
    do, so the result can be exported or moved right away."""
    if not objs:
        return
    for o in context.selected_objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    context.view_layer.objects.active = objs[0]


# ---------------------------------------------------------------------------
# Drag-and-drop support (Blender 4.1+)
# ---------------------------------------------------------------------------

class STEPFORGE_FH_import(bpy.types.FileHandler):
    bl_idname = "STEPFORGE_FH_import"
    bl_label = "StepForge STEP"
    bl_import_operator = "import_scene.stepforge"
    bl_file_extensions = ".step;.stp"

    @classmethod
    def poll_drop(cls, context):
        return context.area is not None and context.area.type == "VIEW_3D"


# ---------------------------------------------------------------------------
# Export operator
# ---------------------------------------------------------------------------

class STEPFORGE_OT_export(_ModalWork, Operator, ExportHelper):
    """Export mesh objects to a STEP (.step) file"""
    bl_idname = "export_scene.stepforge"
    bl_label = "Export STEP (StepForge)"
    bl_options = {"REGISTER", "PRESET"}  # noqa: RUF012
    _failed_message = "StepForge export failed (see console)"

    filename_ext = ".stp"
    filter_glob: StringProperty(default="*.stp;*.step", options={"HIDDEN"})
    use_selection: BoolProperty(name="Selection Only", default=True)
    scale: FloatProperty(name="Scale (to mm)", default=1000.0, min=1e-6)
    optimize_mesh: BoolProperty(
        name="Optimize Mesh (merge coplanar facets)", default=True,
        description="Before writing, merge near-coplanar triangles into larger "
                    "flat faces and re-triangulate, which removes zig-zag "
                    "diagonals on flat panels for a cleaner, lighter file. "
                    "Tessellated mode only; Curved mode merges flat regions "
                    "itself")
    optimize_angle: FloatProperty(
        name="Coplanar Angle", default=2.0, min=0.0, max=45.0,
        description="Triangles whose face normals differ by less than this "
                    "(degrees) are merged when Optimize Mesh is on")
    write_mode: EnumProperty(
        name="Surface Type",
        items=(
            ("BREP", "Curved (analytic, experimental)",
             ("Reconstruct real cylinders, cones, spheres, tori and planes "
              "from the mesh wherever they fit within the tolerance, so radii "
              "and fillets round-trip as curves, not facets. Smooth regions "
              "that are none of those become B-spline surfaces; whatever "
              "still does not fit becomes exact per-triangle planar faces. "
              "Newer and less battle-tested than Tessellated")),
            ("TESSELLATED", "Tessellated (triangle mesh)",
             ("Write the mesh as-is, as an AP242 tessellated solid (the "
              "layout Open CASCADE and FreeCAD read). Matches Blender "
              "exactly, but curved surfaces come back as facets on the next "
              "STEP import")),
        ),
        default="TESSELLATED",
        description="Tessellated is the safe default: it round-trips through "
                    "any STEP reader, StepForge's own included, identical to "
                    "the mesh in Blender however often you export and "
                    "re-import. Curved mode re-fits primitives from the mesh "
                    "on every export, so repeated export-reimport-export "
                    "cycles can lose fitted cylinders and cones and grow the "
                    "triangle count: fine for a one-off hand-off to a CAD "
                    "tool, not for round-tripping",
    )
    fit_tolerance: FloatProperty(
        name="Curve Fit Tolerance (mm)", default=0.3, min=1e-4, soft_max=2.0,
        description="Manual mode only. How far a mesh region may deviate from "
                    "a plane, cylinder, cone or sphere and still be written "
                    "as that surface. Smaller = stricter (more flat "
                    "triangles), larger = more curved surfaces recovered at "
                    "looser accuracy")
    fit_tolerance_pct: FloatProperty(
        name="Curve Fit Tolerance", subtype="PERCENTAGE", default=1.0,
        min=0.01, max=20.0, precision=3,
        description="Curve fit tolerance as a percentage of the exported "
                    "part's bounding-box longest edge, computed at export "
                    "time, so it scales with part size")
    smooth_angle: FloatProperty(
        name="Patch Break Angle", default=32.0, min=1.0, max=89.0,
        description="Adjacent triangles belong to the same curved patch only "
                    "if their face-normal angle is below this; keeps sharp "
                    "edges and corners from fusing into one surface. Rarely "
                    "needs changing")
    freeform_tolerance_pct: FloatProperty(
        name="Freeform Tolerance", default=10.0, min=1.0, max=100.0,
        subtype="PERCENTAGE",
        description="Tolerance for freeform (B-spline) surfaces, as a "
                    "percentage of the part's bounding-box longest edge. It "
                    "sets how large the patches are: smaller = many small "
                    "patches, larger = few large ones. Raise it for a coarse "
                    "or noisy mesh (scan, sculpt, topology-optimised part) to "
                    "turn hundreds of facets into a handful of clean surfaces")
    keep_source_surfaces: BoolProperty(
        name="Keep Original Surfaces", default=True,
        description="Faces imported by StepForge whose polygons still lie on "
                    "their original STEP surface are written back with that "
                    "exact surface (one face each, as in the source file) "
                    "instead of being re-fitted. Edited faces are re-fitted "
                    "as usual")
    freeform_surfaces: BoolProperty(
        name="Freeform Surfaces (B-spline)", default=True,
        description="Reconstruct smoothly curved regions that are not a "
                    "plane, cylinder, cone or sphere as B-spline surface "
                    "faces instead of thousands of flat per-triangle ones. "
                    "Turn off only if a downstream tool cannot read B-spline "
                    "surfaces")
    line_angle_tolerance: FloatProperty(
        name="Line Merge Angle", default=3.0, min=0.0, max=15.0,
        description="Adjacent straight boundary edges are merged into one "
                    "when they differ in direction by less than this many "
                    "degrees. Higher = fewer, longer edges (smaller file, "
                    "faster reimport); too high rounds off a sharp corner")
    manual: BoolProperty(
        name="Manual Settings", default=False,
        description="Show every raw tuning option (fixed-mm curve tolerance, "
                    "patch break angle, line merge angle, mesh optimize "
                    "options) instead of the automatic percentage-based "
                    "defaults")

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "use_selection")
        layout.prop(self, "scale")
        layout.prop(self, "write_mode")
        layout.prop(self, "manual")
        if self.write_mode == "BREP":
            layout.prop(self, "keep_source_surfaces")
            if self.manual:
                layout.prop(self, "fit_tolerance")
                layout.prop(self, "smooth_angle")
                layout.prop(self, "line_angle_tolerance")
                layout.prop(self, "freeform_surfaces")
                if self.freeform_surfaces:
                    layout.prop(self, "freeform_tolerance_pct")
            else:
                layout.prop(self, "fit_tolerance_pct")
                layout.prop(self, "freeform_tolerance_pct")
        else:
            if self.manual:
                layout.prop(self, "optimize_mesh")
                if self.optimize_mesh:
                    layout.prop(self, "optimize_angle")

    def execute(self, context):
        objs = (context.selected_objects if self.use_selection
                else context.scene.objects)
        names = [o.name for o in objs if o.type == "MESH"]
        if not names:
            self.report({"ERROR"}, "No mesh objects to export")
            return {"CANCELLED"}

        # Reading the meshes needs bpy (mesh evaluation, optional bmesh
        # coplanar merge) and happens here in short steps; the curve fitting
        # and the writing run in the background (see `_export_work`).
        angle = self.optimize_angle if (self.write_mode == "TESSELLATED"
                                         and self.optimize_mesh) else None
        job = {"path": self.filepath, "mode": self.write_mode, "scale": self.scale,
               "options": {}}
        if self.write_mode == "BREP":
            # Freeform Tolerance (always) and, when Manual Settings is off,
            # Curve Fit Tolerance too are percentages of each solid's own
            # bounding-box longest edge (see `export_step_brep`), so a small
            # part in an assembly is not fitted to a tolerance sized for the
            # whole scene. ("parallel" is only a switch: `_run_job` sets it.)
            job["options"] = {
                "tolerance_mm": self.fit_tolerance,
                "smooth_angle_deg": self.smooth_angle,
                "scale": self.scale,
                "line_angle_tol_deg": self.line_angle_tolerance,
                "freeform": self.freeform_surfaces,
                "tolerance_pct": None if self.manual else self.fit_tolerance_pct,
                "freeform_tolerance_pct": (self.freeform_tolerance_pct
                                           if self.freeform_surfaces else None),
                "keep_source_surfaces": self.keep_source_surfaces,
                "parallel": True,
            }
        return self._begin(context, _export_work(
            names, angle, self.keep_source_surfaces, job))

    def _done(self, context, result):
        self._report_result(result["n_solids"], self.write_mode, result["stats"])
        return {"FINISHED"}

    def _report_result(self, n_solids, write_mode, stats):
        if write_mode == "BREP" and stats:
            curved = stats["cylinder_faces"] + stats["cone_faces"] + stats["sphere_faces"]
            bspline = stats.get("bspline_faces", 0)
            self.report(
                {"INFO"},
                f"StepForge: exported {n_solids} object(s) - "
                f"{curved} curved face(s) ({stats['cylinder_faces']} cyl, "
                f"{stats['cone_faces']} cone, {stats['sphere_faces']} sphere), "
                f"{bspline} freeform B-spline face(s) "
                f"(from {stats.get('bspline_triangle_faces', 0)} triangles), "
                f"{stats['plane_faces']} planar, "
                f"{stats['leftover_faces']} flattened fallback face(s) "
                f"(from {stats['leftover_triangle_faces']} triangles); "
                f"{stats.get('source_faces', 0)} kept their original surface")
            # A tolerance the mesh cannot meet (a coarse organic mesh whose
            # facets are hundreds of times the tolerance) rejects every
            # freeform fit and leaves the whole part on per-triangle faces.
            # Say so, with the tolerance that would work.
            achievable = stats.get("freeform_achievable_mm")
            if bspline == 0 and achievable:
                self.report(
                    {"WARNING"},
                    "StepForge: no freeform surface could be fitted within the "
                    f"chosen tolerance. This mesh supports roughly "
                    f"{achievable:.2g} mm - raise 'Curve Fit Tolerance' to "
                    "about that to reconstruct curved faces (a coarse mesh on "
                    "a large part cannot be approximated more closely than "
                    "its own facet size).")
        else:
            self.report({"INFO"}, f"StepForge: exported {n_solids} object(s)")


def _export_work(names, optimize_angle_deg, keep_surfaces, job):
    """Generator behind the export operator: read each mesh object here (one per
    step), then fit and write in the background. Returns the job's result."""
    payloads = []
    for i, name in enumerate(names):
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            continue
        progress = (0.1 * i / len(names), f"Reading {name}...")
        yield progress
        payloads.append((yield from _iter_object_payload(
            obj, optimize_angle_deg, keep_surfaces, progress)))
    job["n_solids"] = len(payloads)

    def messages():
        for i, payload in enumerate(payloads):
            yield from jobs.payload_messages(i, payload)

    task, in_background = _run_job("export_job", job, messages)
    try:
        return (yield from _drive(task, lambda msg: None,
                                  messages() if in_background else None,
                                  span=(0.1, 1.0)))
    finally:
        task.close()


def _iter_object_payload(obj, optimize_angle_deg, keep_surfaces, progress):
    """Generator: the plain-data form (see `core/jobs.py`) of a mesh object, in
    world space, n-gons fan-triangulated; read from the mesh with NumPy. Yields
    `progress` between its steps and returns the payload."""
    source = me = obj.to_mesh()
    mat = obj.matrix_world
    try:
        yield progress
        if optimize_angle_deg is not None:
            me = yield from _iter_optimize_mesh_copy(source, optimize_angle_deg, progress)

        co = np.empty(len(me.vertices) * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        m32 = np.array(mat, dtype=np.float32)
        verts = (co.reshape(-1, 3) @ m32[:3, :3].T + m32[:3, 3]).astype(np.float64)

        n_poly, n_loop = len(me.polygons), len(me.loops)
        starts = np.empty(n_poly, dtype=np.int32)
        me.polygons.foreach_get("loop_start", starts)
        corner = np.empty(n_loop, dtype=np.int32)
        me.loops.foreach_get("vertex_index", corner)
        poly_ids = None
        attr = me.attributes.get(FACE_ID_ATTR)
        if attr is not None and attr.domain == "FACE" and attr.data_type == "INT":
            poly_ids = np.empty(n_poly, dtype=np.int32)
            attr.data.foreach_get("value", poly_ids)

        # fan-triangulate: polygon p with k corners gives (0, i, i + 1), i = 1..k-2
        sizes = np.diff(np.append(starts, n_loop))
        n_tri = np.maximum(sizes - 2, 0)
        tri_poly = np.repeat(np.arange(n_poly), n_tri)
        i = np.arange(len(tri_poly)) - np.repeat(np.cumsum(n_tri) - n_tri, n_tri)
        base = starts[tri_poly]
        faces = np.stack([corner[base], corner[base + i + 1], corner[base + i + 2]],
                         axis=1).astype(np.int32)
        face_ids = (poly_ids[tri_poly].astype(np.int64) if poly_ids is not None
                    else np.zeros(len(tri_poly), dtype=np.int64))
    finally:
        obj.to_mesh_clear()
        if me is not source:
            bpy.data.meshes.remove(me)

    raw = obj.data.get(SURFACES_PROP) if poly_ids is not None and keep_surfaces else None
    return {
        "name": obj.name,
        "verts": verts,
        "faces": faces,
        "face_ids": face_ids,
        "surfaces_json": str(raw) if raw else "",
        "matrix": [list(r) for r in mat],
    }


def _iter_optimize_mesh_copy(me, angle_deg, progress):
    """Generator: a NEW mesh datablock with `me`'s triangles, near-coplanar
    faces merged (limited dissolve) and re-triangulated. Leaves `me` untouched;
    the caller frees the returned datablock. Yields `progress` between steps."""
    import bmesh
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        yield progress
        bmesh.ops.dissolve_limit(
            bm, angle_limit=math.radians(max(0.0, angle_deg)),
            use_dissolve_boundaries=False,
            verts=bm.verts, edges=bm.edges, delimit={"NORMAL"})
        yield progress
        bmesh.ops.triangulate(bm, faces=bm.faces)
        yield progress
        out = bpy.data.meshes.new(me.name + "_optimized")
        bm.to_mesh(out)
    finally:
        bm.free()
    return out


# ---------------------------------------------------------------------------
# Menus / registration
# ---------------------------------------------------------------------------

def _menu_import(self, context):
    self.layout.operator(STEPFORGE_OT_import.bl_idname,
                         text="STEP - StepForge (.step/.stp)")


def _menu_export(self, context):
    self.layout.operator(STEPFORGE_OT_export.bl_idname,
                         text="STEP - StepForge (.step)")


_classes = (
    STEPFORGE_OT_import,
    STEPFORGE_FH_import,
    STEPFORGE_OT_export,
)


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.TOPBAR_MT_file_import.append(_menu_import)
    bpy.types.TOPBAR_MT_file_export.append(_menu_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(_menu_export)
    bpy.types.TOPBAR_MT_file_import.remove(_menu_import)
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
