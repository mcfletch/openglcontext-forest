"""Off-thread camera-following stream recompute.

The forest's vegetation follows the camera: every time the viewer crosses a
movement threshold, a streamer re-scatters a disc of grass/impostors around the
new position. That scatter is pure numpy (``world_grid_scatter``, the near-mesh
species masks) — no GL — and it is the demo's one heavy per-move cost, tens of
milliseconds that otherwise land whole on a single render frame as a visible hitch.

:class:`AsyncStreamer` moves that compute onto a worker thread. A move submits the
new ``(x, z, …)`` to :meth:`request`; the worker runs the supplied ``compute`` and
leaves the result in a latest-wins slot; the render loop calls :meth:`drain` once a
frame to hand the newest result to ``apply``, which does only the cheap
``update_instances`` staging on the GL thread. numpy releases the GIL across the
scatter, so the worker genuinely overlaps rendering, and the field trailing the
camera by a frame or two is hidden by the LOD fade bands. ``compute`` must touch no
GL and no shared mutable state; ``apply`` runs on the render thread and is the only
side that mutates nodes.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)


class AsyncStreamer:
    """A worker that recomputes one camera-following field off the render thread.

    :param compute: ``compute(*args) -> payload``. Pure, GL-free; runs on the
        worker. ``args`` are whatever :meth:`request` is given (e.g. ``(x, z)`` or
        ``(x, z, fx, fz)``).
    :param apply: ``apply(payload) -> None``. Runs on the render thread via
        :meth:`drain`; does the GL-side ``update_instances`` staging.
    :param name: thread name, for logs and debugging.
    """

    def __init__(self, compute: Callable[..., Any], apply: Callable[[Any], None],
                 name: str = "stream") -> None:
        self._compute = compute
        self._apply = apply
        self._cv = threading.Condition()
        self._req: tuple[Any, ...] | None = None   # newest pending request
        self._result: Any = None                       # newest computed, undrained
        self._have_result = False
        self._stop = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def request(self, *args: Any) -> None:
        """Ask for a recompute at ``args``; a newer request supersedes an unstarted one."""
        with self._cv:
            self._req = args
            self._cv.notify()

    def drain(self) -> bool:
        """Apply the newest computed result, if any (render thread). Return whether it did."""
        with self._cv:
            if not self._have_result:
                return False
            payload = self._result
            self._result = None
            self._have_result = False
        self._apply(payload)
        return True

    def stop(self) -> None:
        """Stop the worker and wait for it to exit (for a clean shutdown or a test)."""
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._req is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                args = self._req
                self._req = None
            try:
                payload = self._compute(*args)
            except Exception:
                log.exception("stream recompute failed")
                continue
            with self._cv:
                self._result = payload
                self._have_result = True
