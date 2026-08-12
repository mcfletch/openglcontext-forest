"""Render-quality presets and the auto-quality picker for the forest demo.

The forest's frame cost on an integrated GPU is dominated by the near-field grass:
the real-geometry clumps are ~13M tris/frame at the shipped density, ~70% of GPU
time on an Intel UHD 630 (profiled). Every knob that moves it — clump density and
disc radius, the mid/far grass billboard density and follow-radius, and the
near-mesh follow radius — is read fresh at *stream* time from the live
`ForestConfig`/`ForestScene`, so a preset can be applied to a running scene without
rebuilding the 230k-tree world. `apply_to_scene` does exactly that and keeps the
LOD cross-fade windows consistent with the new radii.

Presets:
- ``high``   — the shipped look (equals the `ForestConfig` defaults), max fidelity.
- ``medium`` — tuned to ~60 fps on a UHD-630-class iGPU.
- ``low``    — extra headroom for the weakest parts.

``auto`` is not a preset but a mode: :class:`AutoQuality` measures the live frame
rate and steps between the presets until the rate is at target, keeping shadows on
(they cost <1 ms here — not worth dropping). A user who picks high/medium/low
explicitly overrides and freezes the auto picker so they can see each level's cost.
"""
from __future__ import annotations

from dataclasses import dataclass

from openglcontext_forest_demo.config import ForestConfig


@dataclass(frozen=True)
class QualityPreset:
    """A named set of the live grass/LOD knobs that drive integrated-GPU frame cost."""
    name: str
    clump_density: float       # real grass clumps per m^2 (dominant GPU lever)
    clump_radius: float        # real-clump follow-disc radius, metres
    grass_mid_density: float   # mid grass billboards per m^2
    grass_far_radius: float    # coarse far-grass follow-disc radius, metres
    grass_far_density: float   # far grass billboards per m^2
    near_mesh_radius: float    # radius the near-mesh trees follow the camera, metres


_D = ForestConfig()   # shipped defaults define the `high` preset so the two never drift

#: The presets, cheapest first (see :data:`ORDER`).
PRESETS = {
    "high": QualityPreset(
        "high", clump_density=_D.clump_density, clump_radius=_D.clump_radius,
        grass_mid_density=_D.grass_mid_density, grass_far_radius=_D.grass_far_radius,
        grass_far_density=_D.grass_far_density, near_mesh_radius=_D.near_mesh_radius),
    "medium": QualityPreset(
        "medium", clump_density=4.0, clump_radius=18.0, grass_mid_density=2.5,
        grass_far_radius=400.0, grass_far_density=0.07, near_mesh_radius=50.0),
    "low": QualityPreset(
        "low", clump_density=2.0, clump_radius=12.0, grass_mid_density=1.5,
        grass_far_radius=260.0, grass_far_density=0.05, near_mesh_radius=42.0),
}

#: Presets from cheapest to most expensive — the ladder the auto picker walks.
ORDER = ["low", "medium", "high"]

#: Valid ``--quality`` values.
CHOICES = ["auto", "high", "medium", "low"]


def apply_to_scene(scene, preset: QualityPreset) -> None:
    """Point a live :class:`~openglcontext_forest_demo.scene.ForestScene` at ``preset``.

    Mutates the streaming knobs the scene reads each frame and re-pushes the LOD
    fade windows that were baked from the old radii, so the geometry->billboard
    hand-off stays seamless at the new distances. Call on the GL thread (it touches
    node uniforms); the caller restreams at the current camera position afterwards.
    """
    cfg = scene.config
    cfg.clump_density = preset.clump_density
    cfg.grass_mid_density = preset.grass_mid_density
    cfg.grass_far_density = preset.grass_far_density
    scene.clump_radius = preset.clump_radius
    scene.grass_far_radius = preset.grass_far_radius
    scene.near_mesh_radius = preset.near_mesh_radius

    # Keep the cross-fade windows consistent with the new radii. Each node sends
    # these as constants at GL init, so re-commit to push the new values; guard on
    # _gl so this is a no-op before first render (the fresh values upload then).
    def recommit(node) -> None:
        if node is not None and getattr(node, "_gl", None) is not None:
            node._commit_constants()

    if scene.clumps is not None:
        scene.clumps.fade_start = preset.clump_radius * 0.8
        scene.clumps.fade_end = preset.clump_radius
        recommit(scene.clumps)
    # mid grass billboards fade IN exactly where the clumps fade out
    scene.grass.near_cut = preset.clump_radius
    recommit(scene.grass)
    # far grass billboards dissolve at their follow-disc edge
    scene.grass_far.far_fade = preset.grass_far_radius
    recommit(scene.grass_far)


class AutoQuality:
    """Measured-frame-rate quality picker — down-only, so it never oscillates.

    Fed one frame time per rendered frame, it holds a rolling median and steps the
    preset *down* a rung whenever that median is slower than the target for a whole
    window. It deliberately never steps *up*: the 60 fps target sits between what
    two adjacent presets can sustain (e.g. medium clears it easily while high cannot
    on a UHD 630), so an up-step would just flap medium<->high forever. It starts at
    ``high`` (the shipped look) and drops only as far as the hardware needs — a
    discrete GPU holds ``high``, a UHD 630 lands on ``medium``, a weaker part on
    ``low``; a user who wants a different level picks one by hand (F8). It reports
    *when* a down-step is wanted; the owner applies it and calls :meth:`note_applied`.

    :attr:`decided` flips true only once a window lands in-band (or at the ``low``
    rail): the owner force-feeds frames until then so start-up converges through each
    down-step, and lets natural frames drive any later drop in dense forest.
    """

    def __init__(self, target_fps: float = 60.0, start: str = "high",
                 window: int = 48, warmup: int = 45) -> None:
        self.level = start
        self.target_fps = target_fps
        self.window = window
        self.warmup = warmup
        # Step DOWN when the median frame is slower than this. The measurement is a
        # static view (the picker force-redraws a still camera), which reads ~30%
        # heavier than the same scene in motion, so the band sits ~12 fps under the
        # target: ~48 fps standing ≈ the 60 fps target while walking.
        self._down_ms = 1000.0 / (target_fps - 12.0)   # ~20.8 ms at a 60 fps target
        self._samples: list[float] = []
        self._seen = 0
        self.decided = False
        self.last_median = 0.0   # median frame time (ms) of the most recent full window

    def add_frame(self, dt_ms: float) -> str | None:
        """Record a frame time; return a lower level to apply, or None.

        Ignores warmup frames (first-use shader compiles) and stall frames (a
        streaming burst on one frame must not drag the median it is judged by).
        """
        self._seen += 1
        if self._seen <= self.warmup:
            return None
        if dt_ms > 80.0:            # streaming/GC stall — not representative of steady cost
            return None
        self._samples.append(dt_ms)
        if len(self._samples) < self.window:
            return None
        med = sorted(self._samples)[len(self._samples) // 2]
        self._samples.clear()
        self.last_median = med
        idx = ORDER.index(self.level)
        if med > self._down_ms and idx > 0:
            return ORDER[idx - 1]        # still converging; not yet decided
        self.decided = True              # in-band, or at the low rail: settled
        return None

    def note_applied(self, level: str) -> None:
        """Acknowledge that ``level`` is now live; start a fresh measurement window."""
        self.level = level
        self._samples.clear()


def next_level(level: str, step: int) -> str:
    """The preset ``step`` places from ``level`` on the low->high ladder, clamped."""
    idx = max(0, min(len(ORDER) - 1, ORDER.index(level) + step))
    return ORDER[idx]


def next_level_cycle(level: str) -> str:
    """The next preset on a wrapping low->medium->high->low cycle (for a toggle key)."""
    return ORDER[(ORDER.index(level) + 1) % len(ORDER)]
