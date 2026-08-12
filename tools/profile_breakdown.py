#!/usr/bin/env python
"""Instrumented forest bench: split streaming / draw-submit / GPU, attribute by subsystem.

Env knobs (all optional):
  PB_FRAMES=220          frames to time (after 30 warmup)
  PB_DISABLE=grass,clumps,impostors,near,terrain   subsystems to skip (GPU attribution)
  OPENGLCONTEXT_SHADOWS=0   turn terrain sun-shadows off
Any forest CLI flag also works (e.g. --near-mesh-radius 30 --tree-radius 1200).
"""
import os, sys, time, math
os.environ.setdefault("OPENGLCONTEXT_BACKEND", "glfw")
os.environ["OPENGLCONTEXT_DISABLE_FPS_DISPLAY"] = "1"
import numpy as np
import OpenGL.GL as gl
from openglcontext_forest_demo import run as demo

FRAMES = int(os.environ.get("PB_FRAMES", "220"))
WARM = 30
DISABLE = set(f for f in os.environ.get("PB_DISABLE", "").split(",") if f)


class PB(demo.Forest):
    def OnInit(self):
        r = super().OnInit()
        sc = self.scene
        # Subsystem disable for GPU attribution: make chosen nodes no-op draws.
        def kill(node):
            if node is not None:
                node._disabled = True
        if "grass" in DISABLE:
            kill(sc.grass); kill(sc.grass_far)
        if "clumps" in DISABLE:
            kill(sc.clumps_near); kill(sc.clumps_far)
        if "impostors" in DISABLE:
            for node, *_ in sc.impostors:
                kill(node)
        if "near" in DISABLE:
            kill(sc.near)
        if "terrain" in DISABLE:
            kill(sc.terrain_geom)
        self._idle_ms = []
        self._sub_ms = []
        self._gpu_ms = []
        self._all_ms = []
        self._stalls = []
        self._n = 0
        self._walk = 0.0
        self._t_idle = 0.0
        return r

    def OnIdle(self, *a):
        t0 = time.perf_counter()
        if self.platform is not None:
            self._walk += 0.6
            x = -200.0 + math.sin(self._walk * 0.02) * 40
            z = 300.0 - self._walk
            self.platform.setPosition((x, self.hf.height_at(x, z) + self.eye_height, z))
            self._stream_near(x, z); self._stream_far(x, z); self._stream_impostors(x, z)
            self.triggerRedraw(1)
        self._t_idle = (time.perf_counter() - t0) * 1000.0
        return None

    def OnDraw(self, *a, **k):
        t0 = time.perf_counter()
        r = super().OnDraw(*a, **k)
        t1 = time.perf_counter()            # CPU submit done
        gl.glFinish()
        t2 = time.perf_counter()            # GPU drained
        self._n += 1
        if self._n > WARM:
            self._idle_ms.append(self._t_idle)
            self._sub_ms.append((t1 - t0) * 1000.0)
            self._gpu_ms.append((t2 - t1) * 1000.0)
            frame = self._t_idle + (t2 - t0) * 1000.0
            self._all_ms.append(frame)
            if frame > 80.0:
                self._stalls.append((self._n, frame))
        if self._n >= WARM + FRAMES:
            self._report()
            os._exit(0)
        return r

    def _report(self):
        def stat(a):
            a = np.array(a)
            return np.median(a), np.percentile(a, 95), a.max()
        im, ip, ix = stat(self._idle_ms)
        sm, sp, sx = stat(self._sub_ms)
        gm, gp, gx = stat(self._gpu_ms)
        am, ap, ax = stat(self._all_ms)
        print("\n=== BREAKDOWN (%d frames, disable=%s, shadows=%s) ===" % (
            len(self._all_ms), ",".join(sorted(DISABLE)) or "none",
            os.environ.get("OPENGLCONTEXT_SHADOWS", "1")))
        print("                 median      p95       max")
        print("stream/idle CPU  %6.1f   %6.1f   %6.1f ms" % (im, ip, ix))
        print("draw submit CPU  %6.1f   %6.1f   %6.1f ms" % (sm, sp, sx))
        print("GPU (glFinish)   %6.1f   %6.1f   %6.1f ms" % (gm, gp, gx))
        print("FULL FRAME       %6.1f   %6.1f   %6.1f ms" % (am, ap, ax))
        print("               -> %.1f fps median, %.1f fps p95" % (1000.0 / am, 1000.0 / ap))
        if self._stalls:
            print("STALLS >80ms: %d  ->" % len(self._stalls),
                  ", ".join("f%d=%.0fms" % s for s in self._stalls[:12]))
        else:
            print("no frames >80ms")
        sys.stdout.flush()


if __name__ == "__main__":
    from openglcontext_forest_demo.config import config_from_args
    PB.config = config_from_args()   # honor forest CLI knobs (--clump-density, etc.)
    PB.ContextMainLoop()
