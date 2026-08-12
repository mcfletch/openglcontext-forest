"""The off-thread stream recompute worker (:class:`AsyncStreamer`).

These drive the real worker thread with plain compute/apply callables (no GL): a
request is computed off the caller thread and handed back by ``drain`` on the
caller thread, the newest request wins when several arrive while the worker is
busy, and a compute that raises does not kill the worker.
"""
import threading
import time

from openglcontext_forest_demo.streaming import AsyncStreamer


def _drain_until(streamer, applied, want, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        streamer.drain()
        if want in applied:
            return True
        time.sleep(0.005)
    return False


def test_request_computes_off_thread_and_drain_applies():
    applied = []
    s = AsyncStreamer(lambda x: x * 2, applied.append, "test")
    try:
        s.request(21)
        assert _drain_until(s, applied, 42)
        assert applied == [42]
    finally:
        s.stop()


def test_drain_returns_false_when_nothing_pending():
    s = AsyncStreamer(lambda: 0, lambda v: None, "test")
    try:
        assert s.drain() is False
    finally:
        s.stop()


def test_newest_request_wins_when_worker_is_busy():
    applied = []
    started = threading.Event()
    gate = threading.Event()

    def compute(x):
        if x == 1:
            started.set()
            gate.wait(2.0)     # hold the first compute so 2 and 3 queue behind it
        return x

    s = AsyncStreamer(compute, applied.append, "test")
    try:
        s.request(1)
        assert started.wait(2.0)
        s.request(2)
        s.request(3)           # supersedes the unstarted 2
        gate.set()
        assert _drain_until(s, applied, 3)
        assert applied[-1] == 3
        assert 2 not in applied     # coalesced away, never computed
    finally:
        s.stop()


def test_worker_survives_a_failing_compute():
    applied = []

    def compute(x):
        if x == 0:
            raise ValueError("boom")
        return x

    s = AsyncStreamer(compute, applied.append, "test")
    try:
        s.request(0)               # raises in the worker; must be logged, not fatal
        time.sleep(0.1)
        s.request(5)
        assert _drain_until(s, applied, 5)
        assert applied == [5]
    finally:
        s.stop()


def test_stop_is_idempotent_and_joins():
    s = AsyncStreamer(lambda: 0, lambda v: None, "test")
    s.stop()
    s.stop()                       # second stop must not hang or raise
    assert not s._thread.is_alive()
