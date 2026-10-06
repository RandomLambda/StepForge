"""Multi-core helper: a small pool of spawned worker processes.

Real OS processes via `multiprocessing` (never `threading`), always started
with the 'spawn' method: a fresh interpreter that imports only this
Blender-free package, never Blender's process state.

The add-on lives in the virtual package `bl_ext.<repo>.<id>`, which a fresh
interpreter cannot import. So nothing of ours is pickled by reference, and the
add-on does not touch `sys.path` or `sys.modules`:

  * the child's entry point is the stdlib `runpy.run_path`, pointed at the
    add-on folder and its `__main__.py`; inside the child the package is
    simply `core`;
  * only plain data crosses the process boundary (numbers, strings, lists,
    dicts, NumPy arrays), never instances of our own classes;
  * `WorkerPool.imap` is a generator that yields `None` while no result is
    ready, so the modal operator keeps running its timer and the UI stays
    responsive.

A job names a factory "module:function" in this package; the child calls
`factory(state)` once and gets back a function that turns one work item into
one result (see `serve`).
"""
from __future__ import annotations

import contextlib
import importlib
import multiprocessing
import multiprocessing.connection
import os
import runpy

# Add-on folder, the one that holds `__main__.py` (parent of `core/`).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAX_WORKERS = 32
_IN_FLIGHT = 3           # items handed to one worker ahead of its results
_POLL_SECONDS = 0.01     # how long an idle poll waits for a result


class WorkerError(RuntimeError):
    """A worker process died or failed to start."""


def worker_count(requested=None, n_jobs=None) -> int:
    """Pool size: `requested` if given, else the CPU count (capped), never
    more than `n_jobs` and never less than 1."""
    if requested and requested > 0:
        n = int(requested)
    else:
        n = max(1, min(os.cpu_count() or 1, MAX_WORKERS))
    if n_jobs is not None:
        n = max(1, min(n, n_jobs))
    return n


# ---------------------------------------------------------------------------
# Child side
# ---------------------------------------------------------------------------

def serve(job: str, state, conn) -> None:
    """Child main loop (called from `__main__.py`). `job` is
    "module:factory" with `module` relative to this package."""
    mod, _, name = job.partition(":")
    make = getattr(importlib.import_module(f"{__package__}.{mod}"), name)
    fn = make(state)
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg is None:
            return
        idx, item = msg
        conn.send((idx, fn(item)))


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------

class WorkerPool:
    """`n` worker processes running one job. Use as a context manager (or
    call `close`); leaving early terminates the workers."""

    def __init__(self, job: str, state, n: int):
        ctx = multiprocessing.get_context("spawn")
        self._procs = []
        self._conns = []
        try:
            for _ in range(n):
                parent, child = ctx.Pipe()
                proc = ctx.Process(
                    target=runpy.run_path,
                    args=(_ROOT, {"_sf_job": job, "_sf_state": state,
                                  "_sf_conn": child}, "__stepforge_worker__"),
                    daemon=True)
                proc.start()
                child.close()
                self._procs.append(proc)
                self._conns.append(parent)
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        for c in self._conns:
            with contextlib.suppress(OSError):
                c.send(None)
        for p in self._procs:
            p.join(0.5)
            if p.is_alive():
                p.terminate()
        for c in self._conns:
            with contextlib.suppress(OSError):
                c.close()
        self._procs, self._conns = [], []

    def imap(self, items):
        """Generator over `(index, result)` in completion order. Yields
        `None` after waiting a moment when no result was ready, so a caller
        that must stay responsive can hand control back."""
        items = list(items)
        pending = list(range(len(items) - 1, -1, -1))   # pop() = next item
        load = [0] * len(self._conns)
        done = 0
        while done < len(items):
            for w, c in enumerate(self._conns):
                while pending and load[w] < _IN_FLIGHT:
                    i = pending.pop()
                    c.send((i, items[i]))
                    load[w] += 1
            got = False
            for w, c in enumerate(self._conns):
                try:
                    while c.poll():
                        i, res = c.recv()
                        load[w] -= 1
                        done += 1
                        got = True
                        yield i, res
                except (EOFError, OSError):
                    raise WorkerError("a worker process ended unexpectedly")
            if not got:
                if any(not p.is_alive() for p in self._procs):
                    raise WorkerError("a worker process ended unexpectedly")
                multiprocessing.connection.wait(self._conns, _POLL_SECONDS)
                yield None
