#!/usr/bin/env python
"""Benchmark the forest demo: render N frames offscreen and report fps.
    /workspaces/OpenGL-dev/.venv/bin/python /workspaces/OpenGL-dev/forest-demo/bench.py
"""
import math
import os
import sys
import time
from typing import ClassVar

os.environ.setdefault("OPENGLCONTEXT_BACKEND","glfw")
os.environ["OPENGLCONTEXT_DISABLE_FPS_DISPLAY"]="1"
import numpy as np

from openglcontext_forest_demo import run as demo  # needs `pip install -e .`


class Bench(demo.Forest):
    _t: ClassVar[list] = []
    _n = 0; _walk = 0.0
    def OnDraw(self, *a, **k):
        t0=time.perf_counter()
        r=super().OnDraw(*a,**k)
        import OpenGL.GL as gl
        gl.glFinish()
        self._t.append(time.perf_counter()-t0); self._n+=1
        if self._n>=160:
            warm=self._t[20:]                       # drop warmup frames
            ms=np.array(warm)*1000.0
            print("\n=== FOREST DEMO BENCHMARK ===")
            print(f"frames timed: {len(warm)}")
            print(f"median frame: {np.median(ms):.1f} ms  -> {1000.0/np.median(ms):.1f} fps")
            print(f"p95 frame:    {np.percentile(ms,95):.1f} ms  -> {1000.0/np.percentile(ms,95):.1f} fps")
            sys.stdout.flush()   # os._exit skips the buffers: piped, the numbers vanish
            os._exit(0)
        return r
    def OnIdle(self,*a):
        # auto-walk forward so the benchmark reflects motion (streaming/redraw cost)
        if self.platform is not None:
            self._walk+=0.6
            x=-200.0+math.sin(self._walk*0.02)*40; z=300.0-self._walk
            self.platform.setPosition((x, self.hf.height_at(x,z)+self.eye_height, z))
            self._stream_near(x,z); self._stream_far(x,z)   # keep the veg field populated
            self._stream_impostors(x,z)                     # cull impostors to the view cone
            self.triggerRedraw(1)


if __name__=="__main__":
    Bench.ContextMainLoop()
