"""Entry point of StepForge's background processes -- see `core/workers.py`.

Not meant to be run by hand: `runpy.run_path` executes this file in a
spawned process and injects `_sf_job`, `_sf_state` and `_sf_conn` (and
`_sf_task` for a task process rather than a pool worker). It stays free of
Blender imports."""
if "_sf_job" in globals():
    from core import workers
    if globals().get("_sf_task"):
        workers.serve_task(_sf_job, _sf_state, _sf_conn)  # noqa: F821
    else:
        workers.serve(_sf_job, _sf_state, _sf_conn)  # noqa: F821
