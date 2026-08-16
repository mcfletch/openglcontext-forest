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

``auto`` is not a preset but a mode: :class:`AutoQuality` measures the frame rate
once at ``high`` and picks the level that should hold the target, keeping shadows on
(they cost <1 ms here — not worth dropping). A capable GPU holds ``high`` and nothing
changes; a slower one has its pick applied deferred (folded into the next streaming
move) so the field is never seen re-scattering in place. A user who picks
high/medium/low explicitly freezes that choice.
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
    #: GPU cost of this preset relative to ``high`` (=1.0). Used by the auto picker to
    #: estimate a level's frame time from a measurement taken at another level, so it
    #: can land on the *highest* level that holds the target rather than over-dropping.
    #: Calibrated from measured GPU time; recheck if the preset knobs change materially.
    cost_weight: float


_D = ForestConfig()   # shipped defaults define the `high` preset so the two never drift

#: The quality ladder, cheapest first. `high`/`medium`/`low` are the named rungs a
#: user picks (F8, ``--quality``); the two intermediate rungs exist only to give the
#: auto picker somewhere to land between them, so it need not jump a big fidelity step
#: to hold ~60 fps. Each rung's ``cost_weight`` is its GPU cost relative to `high`,
#: measured **fullscreen** (fill-bound) -- the regime where the picker actually
#: downgrades. Windowed the ratios differ (the clump vertex cost is per-primitive, not
#: per-pixel), but windowed `high` holds, so the weights only matter fill-bound.
PRESETS = {
    "high": QualityPreset(
        "high", clump_density=_D.clump_density, clump_radius=_D.clump_radius,
        grass_mid_density=_D.grass_mid_density, grass_far_radius=_D.grass_far_radius,
        grass_far_density=_D.grass_far_density, near_mesh_radius=_D.near_mesh_radius,
        cost_weight=1.0),
    "medhigh": QualityPreset(
        "medhigh", clump_density=6.0, clump_radius=24.0, grass_mid_density=3.0,
        grass_far_radius=540.0, grass_far_density=0.08, near_mesh_radius=53.0,
        cost_weight=0.78),
    "medium": QualityPreset(
        "medium", clump_density=4.0, clump_radius=18.0, grass_mid_density=2.5,
        grass_far_radius=400.0, grass_far_density=0.07, near_mesh_radius=50.0,
        cost_weight=0.67),
    "medlow": QualityPreset(
        "medlow", clump_density=3.0, clump_radius=15.0, grass_mid_density=2.0,
        grass_far_radius=330.0, grass_far_density=0.06, near_mesh_radius=46.0,
        cost_weight=0.64),
    "low": QualityPreset(
        "low", clump_density=2.0, clump_radius=12.0, grass_mid_density=1.5,
        grass_far_radius=260.0, grass_far_density=0.05, near_mesh_radius=42.0,
        cost_weight=0.53),
}

#: Every rung, cheapest first — the ladder the auto picker chooses from and F8 cycles.
LADDER = ["low", "medlow", "medium", "medhigh", "high"]
#: Backwards-compatible alias (the old three-rung name).
ORDER = LADDER

#: Valid ``--quality`` values: auto, or any ladder rung (F8 cycles the same set).
CHOICES = ["auto", *reversed(LADDER)]


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

    # clump LOD fade/cut windows (both nodes) follow clump_radius
    scene.retune_clumps()
    # mid grass billboards fade IN exactly where the clumps fade out
    scene.grass.near_cut = preset.clump_radius
    recommit(scene.grass)
    # far grass billboards dissolve at their follow-disc edge
    scene.grass_far.far_fade = preset.grass_far_radius
    recommit(scene.grass_far)


class AutoQuality:
    """Measured-frame-rate quality picker — one measurement, one decision.

    It measures a single window at the start level (``high`` at first launch, the
    shipped look, so a capable GPU is already where it belongs) and picks the level
    that should hold the target by scaling the measured frame time by each rung's
    :attr:`~QualityPreset.cost_weight` — landing on the *highest* rung that fits, so a
    GPU that just misses `high` drops to `medhigh`, not all the way to `low`. It does
    **not** step level-by-level, because measuring a lower level means *showing* it,
    and a whole field of grass changing density on screen is the very pop we are
    avoiding: the one target is applied deferred (folded into the next streaming move)
    by the owner, then the picker stops. The owner re-runs it on a resolution change
    (fill scales with pixels), starting from the current level, which is also how it
    climbs back up when a window shrinks.

    :attr:`decided` flips true once the window is judged; :attr:`target` is the level
    to settle at (None means "keep the start level"). The owner force-feeds frames
    until decided, applies :attr:`target` deferred, and locks.
    """

    def __init__(self, target_fps: float = 60.0, start: str = "high",
                 window: int = 48, warmup: int = 45) -> None:
        self.level = start
        self.target_fps = target_fps
        self.window = window
        self.warmup = warmup
        # Over this frame time the start level misses target. The measurement is a
        # static view (the picker force-redraws a still camera), which reads ~30%
        # heavier than the same scene in motion, so the band sits ~12 fps under the
        # target: ~48 fps standing ≈ the 60 fps target while walking.
        self._down_ms = 1000.0 / (target_fps - 12.0)   # ~20.8 ms at a 60 fps target
        self._samples: list[float] = []
        self._seen = 0
        self.decided = False
        self.target: str | None = None
        self.last_median = 0.0   # median frame time (ms) of the measurement window

    def add_frame(self, dt_ms: float) -> str | None:
        """Record a frame time; once the window fills, decide and return the target level.

        Ignores warmup frames (first-use shader compiles) and stall frames (a
        streaming burst on one frame must not drag the median it is judged by).
        Returns the target level to settle at, or None to keep the start level.
        """
        if self.decided:
            return None
        self._seen += 1
        if self._seen <= self.warmup:
            return None
        if dt_ms > 80.0:            # streaming/GC stall — not representative of steady cost
            return None
        self._samples.append(dt_ms)
        if len(self._samples) < self.window:
            return None
        med = sorted(self._samples)[len(self._samples) // 2]
        self.last_median = med
        self.decided = True
        # Estimate each rung's frame time from the one measured (time scales with the
        # cost_weight ratio) and take the highest rung that holds the band; if none do,
        # the cheapest rung.
        start_w = PRESETS[self.level].cost_weight
        tgt = LADDER[0]
        for name in reversed(LADDER):                      # high -> low
            if med * PRESETS[name].cost_weight / start_w <= self._down_ms:
                tgt = name
                break
        self.target = tgt if tgt != self.level else None
        return self.target


def next_level(level: str, step: int) -> str:
    """The rung ``step`` places from ``level`` on the low->high ladder, clamped."""
    idx = max(0, min(len(LADDER) - 1, LADDER.index(level) + step))
    return LADDER[idx]


def next_level_cycle(level: str) -> str:
    """The next rung on a wrapping low->...->high->low cycle (for the F8 toggle)."""
    return LADDER[(LADDER.index(level) + 1) % len(LADDER)]
