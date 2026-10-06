"""STEP reading, B-rep tessellation and STEP writing in pure Python and NumPy.

Importable outside Blender, so it can be tested from a plain interpreter.
"""

# Reload support ("Reload Scripts"): on a repeated load the submodules are
# reloaded too. Same pattern as in
# https://developer.blender.org/docs/handbook/extensions/addon_dev_setup/#reloading-scripts
_needs_reload = "parser" in locals()

from . import (
    assembly,
    boundary,
    brep_export,
    convert,
    curvature,
    delaunay,
    earcut,
    exporter,
    fit,
    freeform,
    geometry,
    param,
    parser,
    tessellate,
    vsa,
    workers,
)

if _needs_reload:
    import importlib

    # Dependencies first: a module that does `from .x import y` must be
    # reloaded after x.
    geometry = importlib.reload(geometry)
    assembly = importlib.reload(assembly)
    fit = importlib.reload(fit)
    boundary = importlib.reload(boundary)
    exporter = importlib.reload(exporter)
    curvature = importlib.reload(curvature)
    param = importlib.reload(param)
    vsa = importlib.reload(vsa)
    freeform = importlib.reload(freeform)
    workers = importlib.reload(workers)
    brep_export = importlib.reload(brep_export)
    parser = importlib.reload(parser)
    delaunay = importlib.reload(delaunay)
    earcut = importlib.reload(earcut)
    tessellate = importlib.reload(tessellate)
    convert = importlib.reload(convert)

from .convert import Mesh, Solid, convert_step
from .parser import StepFile, parse_file, parse_string

__all__ = [
    "Mesh",
    "Solid",
    "StepFile",
    "convert_step",
    "parse_file",
    "parse_string",
]
