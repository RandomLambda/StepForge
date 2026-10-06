"""Background processes: a task process for a whole import/export, and a small
pool of worker processes for the per-face tessellation inside it.

Real OS processes via `multiprocessing` (never `threading`), always started
with the 'spawn' method: a fresh interpreter that imports only this
Blender-free package, never Blender's process state. Blender's own process
therefore never runs the heavy work and never competes with it for the GIL;
it only polls for messages.

The add-on is a virtual package inside Blender that a fresh interpreter cannot
import. So nothing of ours is pickled by reference, and the add-on does not
touch `sys.path` or `sys.modules`:

  * the child's entry point is the stdlib `runpy.run_path`, pointed at the
    add-on folder and its `__main__.py`; inside the child the package is
    simply `core`;
  * only plain data crosses the process boundary (numbers, strings, lists,
    dicts, NumPy arrays), never instances of our own classes.

`TaskProcess` runs one job (a generator in `core/jobs.py`) and streams its
messages back; `poll()` never waits. `WorkerPool.imap` is a generator that
yields `None` while no result is ready, for a caller that must stay
responsive. A job names a factory "module:function" in this package.
"""
from __future__ import annotations

import atexit
import collections
import contextlib
import importlib
import multiprocessing
import multiprocessing.connection
import os
import runpy
import time
import traceback
import weakref

# Add-on folder, the one that holds `__main__.py` (parent of `core/`).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAX_WORKERS = 32
_IN_FLIGHT = 3           # items handed to one worker ahead of its results
_POLL_SECONDS = 0.01     # how long an idle poll waits for a result
_PROGRESS_SECONDS = 0.05  # a task sends progress at most this often


class WorkerError(RuntimeError):
    """A background process died, failed to start, or reported an error."""


def worker_count(requested=None, n_jobs=None) -> int:
    """Pool size: `requested` if given, else half the logical CPUs (capped), so
    that Blender's own threads always have cores to run on; never more than
    `n_jobs` and never less than 1."""
    if requested and requested > 0:
        n = int(requested)
    else:
        n = max(1, min((os.cpu_count() or 2) // 2, MAX_WORKERS))
    if n_jobs is not None:
        n = max(1, min(n, n_jobs))
    return n


# ---------------------------------------------------------------------------
# Child side
# ---------------------------------------------------------------------------

def _lower_priority() -> None:
    """Run this process below normal priority, so that the operating system
    always serves Blender's user interface first, however many cores the
    background work keeps busy. Processes started from here inherit it."""
    try:
        if os.name == "nt":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
        else:
            os.nice(10)
    except (OSError, AttributeError):
        pass    # no way to lower it here: the work just runs at normal priority


def serve(job: str, state, conn) -> None:
    """Pool worker main loop (called from `__main__.py`). `job` is
    "module:factory" with `module` relative to this package."""
    _lower_priority()
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


class _Cancelled(Exception):
    """The parent asked the task to stop."""


class Inbox:
    """Messages from the parent to a running task. A job reads them with
    `get()` (blocking); between its steps `serve_task` calls `check()`, which
    notices a cancel request without waiting. Also usable without a process:
    `Inbox(None, messages)` hands out a ready-made list."""

    def __init__(self, conn=None, messages=()):
        self._conn = conn
        self._buf = collections.deque(messages)

    def _pull(self, block: bool) -> None:
        while self._conn is not None and (block or self._conn.poll(0)):
            msg = self._conn.recv()
            if msg[0] == "cancel":
                raise _Cancelled
            self._buf.append(msg)
            block = False

    def get(self):
        while not self._buf:
            if self._conn is None:
                raise EOFError("no more messages")
            self._pull(True)
        return self._buf.popleft()

    def check(self) -> None:
        self._pull(False)


def serve_task(job: str, state, conn) -> None:
    """Task process main loop (called from `__main__.py`): run the generator
    `factory(state, inbox)` and send everything it yields to the parent, then
    its return value as `("result", value)`. A failure is sent as
    `("error", traceback text)`."""
    _lower_priority()
    inbox = Inbox(conn)
    try:
        mod, _, name = job.partition(":")
        make = getattr(importlib.import_module(f"{__package__}.{mod}"), name)
        conn.send(("ready",))
        gen = make(state, inbox)
        last_progress = 0.0
        while True:
            try:
                msg = next(gen)
            except StopIteration as stop:
                conn.send(("result", stop.value))
                return
            if msg[0] == "progress":
                now = time.monotonic()
                if now - last_progress < _PROGRESS_SECONDS:
                    msg = None
                else:
                    last_progress = now
            if msg is not None:
                conn.send(msg)
            inbox.check()
    except _Cancelled:
        return
    except (EOFError, BrokenPipeError, ConnectionError):
        return                      # the parent is gone
    except BaseException:  # noqa: BLE001 - reported to the parent
        with contextlib.suppress(OSError):
            conn.send(("error", traceback.format_exc()))


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------

_live_tasks: weakref.WeakSet = weakref.WeakSet()


@atexit.register
def _stop_live_tasks() -> None:
    # A task process is not a daemon (it starts the pool's daemon workers), so
    # `multiprocessing` would wait for it when the interpreter exits.
    for task in list(_live_tasks):
        task.close()


class TaskProcess:
    """One background process running a job. `poll()` returns the next message
    that is already there, or None, and never waits for the job; `send()`
    writes a message to it. The job's messages are plain tuples:
    ("ready",), ("progress", fraction, text), job-specific ones, then
    ("result", value) or ("error", traceback text)."""

    def __init__(self, job: str, state):
        ctx = multiprocessing.get_context("spawn")
        parent, child = ctx.Pipe()
        self._conn = parent
        self._proc = ctx.Process(
            target=runpy.run_path,
            args=(_ROOT, {"_sf_job": job, "_sf_state": state, "_sf_conn": child,
                          "_sf_task": True}, "__stepforge_worker__"),
            daemon=False)
        try:
            self._proc.start()
        except BaseException:
            parent.close()
            child.close()
            raise
        child.close()
        _live_tasks.add(self)

    def poll(self):
        try:
            if self._conn.poll(0):
                return self._conn.recv()
            if not self._proc.is_alive():
                if self._conn.poll(0):
                    return self._conn.recv()
                raise WorkerError("the background process ended unexpectedly")
        except (EOFError, OSError) as exc:
            raise WorkerError("the background process ended unexpectedly") from exc
        return None

    def send(self, msg) -> None:
        try:
            self._conn.send(msg)
        except OSError as exc:
            raise WorkerError("the background process ended unexpectedly") from exc

    def close(self) -> None:
        """Stop the process if it still runs. Its pool workers notice the
        closed pipe and end by themselves."""
        proc, self._proc = getattr(self, "_proc", None), None
        if proc is None:
            return
        proc.join(0.02)
        if proc.is_alive():
            proc.terminate()
            proc.join(0.5)
        with contextlib.suppress(OSError):
            self._conn.close()
        _live_tasks.discard(self)


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
