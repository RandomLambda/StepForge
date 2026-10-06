"""Entry point of StepForge's worker processes -- see `core/workers.py`.

Not meant to be run by hand: `runpy.run_path` executes this file in a
spawned worker and injects `_sf_job`, `_sf_state` and `_sf_conn`. It stays
free of Blender imports."""
if "_sf_job" in globals():
    from core import workers
    workers.serve(_sf_job, _sf_state, _sf_conn)  # noqa: F821
