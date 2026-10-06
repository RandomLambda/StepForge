"""StepForge: STEP (.step/.stp) import and export for Blender.

Pure Python and NumPy, no wheels. The conversion code lives in `core` and has
no Blender dependency; the operators are in `_blender` and are only loaded
inside Blender.
"""

# Reload support ("Reload Scripts"). See
# https://developer.blender.org/docs/handbook/extensions/addon_dev_setup/#reloading-scripts
_needs_reload = "bpy" in locals()

try:
    import bpy
except ModuleNotFoundError:  # plain Python interpreter (tests, benchmarks)
    bpy = None

from . import core

if bpy is not None:
    from . import _blender

if _needs_reload:
    import importlib

    core = importlib.reload(core)
    if bpy is not None:
        _blender = importlib.reload(_blender)

from .core import convert_step  # noqa: F401
from .core.exporter import export_step  # noqa: F401


def register():
    if bpy is not None:
        _blender.register()


def unregister():
    if bpy is not None:
        _blender.unregister()
