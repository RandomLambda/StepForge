"""Import and export as jobs: generators that yield plain-data messages.

A job runs in a background process (`workers.TaskProcess`), so that Blender's
own process only has to poll for messages and build its mesh data. When no
process can be used (a script, headless Blender, a failed start) the same
generator is stepped inside the caller's process instead.

Messages, all plain tuples:

    ("progress", fraction_0_1, text)
    ("solid", tag, header)  ("part", key, array)  ("dict", key, entries)
    ("text", key, string)   ("end", tag)         -- one solid, in chunks
                                                    (see `payload_messages`)

A job returns its result value (the generator's return value).

A solid travels as a "payload" dict of plain data: `name`, `verts` (n, 3)
float64, `faces` (m, 3) int32 and `face_ids` (m,) int64 arrays, `colour`,
`face_colours`, and its source surfaces as JSON text (`surfaces`: STEP face id
to the record's JSON, on import; `surfaces_json` plus `matrix`, on export).
"""
from __future__ import annotations

import json
import os

import numpy as np

from .brep_export import STAT_FIELDS, iter_export_step_brep
from .convert import Mesh, Solid, iter_convert_step, transform_surface_record
from .exporter import export_step

CHUNK_BYTES = 1 << 20     # an array or text travels in pieces of about this size
CHUNK_ITEMS = 2000        # dict entries per message

ARRAY_KEYS = ("verts", "faces", "face_ids")
DICT_KEYS = ("face_colours", "surfaces")
TEXT_KEYS = ("surfaces_json",)
SOLID_KINDS = frozenset({"solid", "part", "dict", "text", "end"})

_EMPTY = {"verts": (0, 3, np.float64), "faces": (0, 3, np.int32), "face_ids": (0, 0, np.int64)}


def solid_payload(solid) -> dict:
    """The plain-data form of a converted `Solid`."""
    m = solid.mesh
    return {
        "name": solid.name,
        "verts": np.asarray(m.verts, dtype=np.float64).reshape(-1, 3),
        "faces": np.asarray(m.faces, dtype=np.int32).reshape(-1, 3),
        "face_ids": np.asarray(m.face_ids, dtype=np.int64),
        "colour": solid.colour,
        "face_colours": dict(solid.face_colours),
        "surfaces": {int(f): json.dumps(rec, separators=(",", ":"))
                     for f, rec in solid.face_surfaces.items()},
        "n_faces_in": solid.n_faces_in,
        "n_faces_failed": solid.n_faces_failed,
    }


def payload_messages(tag, payload: dict):
    """Yield the messages that carry `payload`: a header, the arrays, dicts and
    texts in pieces, and an end marker."""
    skip = ARRAY_KEYS + DICT_KEYS + TEXT_KEYS
    yield "solid", tag, {k: v for k, v in payload.items() if k not in skip}
    for key in ARRAY_KEYS:
        arr = payload[key]
        if len(arr):
            row = max(arr.nbytes // len(arr), 1)
            step = max(CHUNK_BYTES // row, 1)
            for i in range(0, len(arr), step):
                yield "part", key, arr[i:i + step]
    for key in DICT_KEYS:
        items = list(payload.get(key, {}).items())
        for i in range(0, len(items), CHUNK_ITEMS):
            yield "dict", key, dict(items[i:i + CHUNK_ITEMS])
    for key in TEXT_KEYS:
        text = payload.get(key) or ""
        for i in range(0, len(text), CHUNK_BYTES):
            yield "text", key, text[i:i + CHUNK_BYTES]
    yield "end", tag


class SolidReceiver:
    """Rebuilds payloads from `payload_messages`: `feed(message)` returns the
    finished payload (with its `tag`) on the "end" message, else None."""

    def __init__(self):
        self._cur = None

    def feed(self, msg):
        kind = msg[0]
        if kind == "solid":
            self._cur = {"tag": msg[1], "head": msg[2],
                         "arrays": {k: [] for k in ARRAY_KEYS},
                         "dicts": {k: {} for k in DICT_KEYS},
                         "texts": {k: [] for k in TEXT_KEYS}}
        elif kind == "part":
            self._cur["arrays"][msg[1]].append(msg[2])
        elif kind == "dict":
            self._cur["dicts"][msg[1]].update(msg[2])
        elif kind == "text":
            self._cur["texts"][msg[1]].append(msg[2])
        elif kind == "end":
            cur, self._cur = self._cur, None
            out = dict(cur["head"])
            out["tag"] = cur["tag"]
            for key in ARRAY_KEYS:
                parts = cur["arrays"][key]
                if not parts:
                    n, cols, dtype = _EMPTY[key]
                    out[key] = np.empty((n, cols) if cols else (n,), dtype=dtype)
                else:
                    out[key] = parts[0] if len(parts) == 1 else np.concatenate(parts)
            out.update(cur["dicts"])
            for key in TEXT_KEYS:
                out[key] = "".join(cur["texts"][key])
            return out
        return None


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def import_job(state, inbox=None):
    """Convert every file in `state["paths"]` (`state["options"]` go to
    `iter_convert_step`). Sends each solid as a payload (tag = file index);
    returns one info dict per file."""
    paths, options = state["paths"], state["options"]
    n_files = len(paths) or 1
    infos = []
    for file_i, path in enumerate(paths):
        gen = iter_convert_step(path, **options)
        while True:
            try:
                frac, msg = next(gen)
            except StopIteration as done:
                result = done.value
                break
            yield "progress", (file_i + frac) / n_files, msg
        for solid in result:
            if solid.mesh.verts:
                yield from payload_messages(file_i, solid_payload(solid))
        infos.append({
            "n_faces": getattr(result, "n_faces", 0),
            "n_failed": len(getattr(result, "failed_faces", ())),
            "lines": (result.summary_lines(os.path.basename(path))
                      if hasattr(result, "summary_lines") else []),
        })
    return infos


def _payload_to_solid(p: dict) -> Solid:
    mesh = Mesh(verts=[tuple(r) for r in p["verts"].tolist()],
                faces=[tuple(r) for r in p["faces"].tolist()],
                face_ids=p["face_ids"].tolist())
    surfaces = {}
    raw = p.get("surfaces_json")
    if raw:
        try:
            for key, rec in json.loads(raw).items():
                tr = transform_surface_record(rec, p["matrix"])
                if tr is not None:
                    surfaces[int(key)] = tr
        except (ValueError, TypeError, KeyError):
            surfaces = {}
    return Solid(name=p["name"], mesh=mesh, face_surfaces=surfaces)


def export_job(state, inbox):
    """Write the `state["n_solids"]` solids that arrive through `inbox` to
    `state["path"]`: `state["mode"]` "BREP" runs `iter_export_step_brep` with
    `state["options"]`, anything else writes the mesh as it is (`scale`).
    Returns `{"n_solids": ..., "stats": ...}` (stats only for BREP)."""
    receiver = SolidReceiver()
    solids = []
    n_expected = state["n_solids"]
    while len(solids) < n_expected:
        payload = receiver.feed(inbox.get())
        if payload is not None:
            solids.append(_payload_to_solid(payload))
            yield "progress", 0.0, f"Reading the mesh data: {len(solids)}/{n_expected}"
    if state["mode"] != "BREP":
        yield "progress", 0.5, "Writing the STEP file..."
        export_step(solids, state["path"], scale=state["scale"])
        return {"n_solids": len(solids), "stats": None}
    gen = iter_export_step_brep(solids, state["path"], **state["options"])
    while True:
        try:
            frac, msg = next(gen)
        except StopIteration as done:
            stats_obj = done.value
            break
        yield "progress", frac, msg
    stats = {f: getattr(stats_obj, f) for f in STAT_FIELDS}
    stats["freeform_achievable_mm"] = stats_obj.freeform_achievable_mm
    return {"n_solids": len(solids), "stats": stats}
