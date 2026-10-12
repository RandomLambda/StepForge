"""Tests for the worker pool's fork/spawn selection.

The import path forks its workers so they inherit the already-parsed STEP
file (copy-on-write) instead of each re-parsing it. These tests check the
platform probe, that the tessellation job reuses an inherited `sf` rather
than parsing again, and that a forked pool actually runs jobs.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core import convert as C
    from core import workers as W
    _HAVE_CORE = True
except ImportError:      # NumPy (or the package) unavailable
    _HAVE_CORE = False


@unittest.skipUnless(_HAVE_CORE, "core (NumPy) not importable")
class ForkSupportTests(unittest.TestCase):
    def test_supports_fork_matches_platform(self):
        self.assertEqual(W.supports_fork(),
                         "fork" in multiprocessing.get_all_start_methods())

    def test_job_reuses_shared_sf_without_parsing(self):
        """With `state['sf']` present, `_tessellate_job` must not parse the
        file: parsing again is exactly what the fork path removes."""
        class Stub:
            angle_scale = 1.0

        def boom(_path):
            raise AssertionError("worker re-parsed the file")

        orig = C.parse_file
        C.parse_file = boom
        try:
            state = {"path": "/nonexistent/never.step", "angle_scale": 1.0,
                     "deflection": 0.05, "max_edge": None, "debug": False,
                     "flat_max_edge": False, "edge_hints": {}, "sf": Stub()}
            run = C._tessellate_job(state)
            # A stub `sf` makes `_process_face` raise, which the job swallows;
            # the assertion is only that `parse_file` was never reached.
            self.assertIsInstance(run(1), tuple)
        finally:
            C.parse_file = orig

    def test_fork_pool_runs_jobs(self):
        if not W.supports_fork():
            self.skipTest("fork not available on this platform")

        class Stub:
            angle_scale = 1.0

        state = {"path": "/nonexistent/never.step", "angle_scale": 1.0,
                 "deflection": 0.05, "max_edge": None, "debug": False,
                 "flat_max_edge": False, "edge_hints": {}, "sf": Stub()}
        with W.WorkerPool("convert:_tessellate_job", state, 1,
                          start_method="fork") as pool:
            got = [r for r in pool.imap([1, 2]) if r is not None]
        self.assertEqual([i for i, _ in got], [0, 1])


@unittest.skipUnless(_HAVE_CORE, "core (NumPy) not importable")
class WorkerCountTests(unittest.TestCase):
    def test_auto_uses_every_cpu_capped(self):
        n = W.worker_count(None, None)
        self.assertEqual(n, max(1, min(os.cpu_count() or 2, W.MAX_WORKERS)))

    def test_requested_value_wins(self):
        self.assertEqual(W.worker_count(3, 100), 3)

    def test_zero_or_negative_means_auto(self):
        self.assertEqual(W.worker_count(0, None), W.worker_count(None, None))
        self.assertEqual(W.worker_count(-5, None), W.worker_count(None, None))

    def test_clamped_to_job_count(self):
        self.assertEqual(W.worker_count(8, 2), 2)
        self.assertEqual(W.worker_count(8, 0), 1)


if __name__ == "__main__":
    unittest.main()
