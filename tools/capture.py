#!/usr/bin/env python
"""Render a few frames from fixed viewpoints to PNGs (offscreen) for review.
    /workspaces/OpenGL-dev/.venv/bin/python /workspaces/OpenGL-dev/forest-demo/capture.py
Writes shots/shot_*.png near fir (high/steep) clusters to check foliage/trunk bleed
and the distant-grass fill.
"""
import os, math
os.environ.setdefault("OPENGLCONTEXT_BACKEND", "glfw")
os.environ["OPENGLCONTEXT_DISABLE_FPS_DISPLAY"] = "1"
import numpy as np
from openglcontext_forest_demo import run as demo   # needs `pip install -e .`
import OpenGL.GL as gl
from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shots"); os.makedirs(OUT, exist_ok=True)


class Cap(demo.Forest):
    _shots = []     # (x,z) viewpoints, filled in OnInit
    _i = 0; _warm = 0

    def OnInit(self):
        r = super().OnInit()
        h = self.hf.grid; sc = demo.RES
        gz, gx = np.gradient(h.astype('f')); slope = np.hypot(gx, gz)
        hi = (h > np.percentile(h, 68)) & (slope > np.percentile(slope, 60))
        ys, xs = np.where(hi); pick = np.linspace(0, len(xs) - 1, 6).astype(int)
        for k in pick:
            wx = (xs[k] / (sc - 1) - 0.5) * demo.EXTENT; wz = (ys[k] / (sc - 1) - 0.5) * demo.EXTENT
            self._shots.append((wx, wz))
        return r

    def OnIdle(self, *a):
        if self.platform is None:
            return None
        wx, wz = self._shots[self._i]
        cx, cz = wx - 18 * math.sin(self._i), wz - 18 * math.cos(self._i)
        y = self.hf.height_at(cx, cz) + self.eye_height
        self.platform.setPosition((cx, y, cz))
        yaw = math.atan2(wx - cx, wz - cz)
        try:
            self.platform.setOrientation((0, 1, 0, yaw))
        except Exception:
            pass
        self._stream_near(cx, cz); self._stream_far(cx, cz)
        self.triggerRedraw(1); return None

    def OnDraw(self, *a, **k):
        r = super().OnDraw(*a, **k)
        self._warm += 1
        if self._warm < 4:
            return r          # let the near-set upload settle
        self._warm = 0
        w, h = self.getViewPort()
        try:
            gl.glReadBuffer(gl.GL_BACK)
        except Exception:
            pass
        buf = gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        img = Image.frombytes("RGB", (w, h), buf).transpose(Image.FLIP_TOP_BOTTOM)
        p = os.path.join(OUT, "shot_%d.png" % self._i); img.save(p); print("wrote", p)
        self._i += 1
        if self._i >= len(self._shots):
            os._exit(0)
        return r


if __name__ == "__main__":
    Cap.ContextMainLoop()
