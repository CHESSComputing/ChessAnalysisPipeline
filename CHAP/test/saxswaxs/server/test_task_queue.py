"""Unit tests for CHAP.saxswaxs.server.task_queue."""

import queue
import threading
from unittest.mock import MagicMock, call, patch

import pytest

import CHAP.saxswaxs.server.task_queue as tq


def drain(q, timeout=2):
    """Block until q is empty or timeout expires."""
    q.join()


class TestPut:
    def test_put_enqueues_tuple(self):
        task = MagicMock()
        args = (1, 2)
        kwargs = {"a": 3}
        tq.put(task, args, kwargs)
        item = tq._task_queue.get_nowait()
        tq._task_queue.task_done()
        assert item == (task, args, kwargs)


class TestWorker:
    def _run_worker_once(self, task, args=(), kwargs={}):
        """Put one task, wait for it to complete."""
        tq.put(task, args, kwargs)
        tq._task_queue.join()

    def test_worker_calls_task(self):
        called = threading.Event()
        task = MagicMock(side_effect=lambda: called.set())
        self._run_worker_once(task)
        assert called.wait(timeout=5), "worker never called the task"
        task.assert_called_once()

    def test_worker_passes_args_and_kwargs(self):
        received = {}

        def capture(*args, **kwargs):
            received["args"] = args
            received["kwargs"] = kwargs

        tq.put(capture, (10, 20), {"x": 99})
        tq._task_queue.join()
        assert received["args"] == (10, 20)
        assert received["kwargs"] == {"x": 99}

    def test_worker_retries_on_exception(self):
        """Worker should retry a failing task until it succeeds."""
        attempt_count = [0]
        done = threading.Event()

        def flaky():
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                raise RuntimeError("transient failure")
            done.set()

        tq.put(flaky, (), {})
        assert done.wait(timeout=15), "worker never retried the task"
        assert attempt_count[0] == 2

    def test_worker_marks_task_done_after_success(self):
        sentinel = threading.Event()
        tq.put(lambda: sentinel.set(), (), {})
        # join() blocks until task_done() is called; it would hang on failure
        tq._task_queue.join()
        assert sentinel.is_set()
