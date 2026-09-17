"""Render-quality presets and the auto-quality picker for the forest demo.

The forest's frame cost on an integrated GPU is dominated by the near-field ground
cover: the real-geometry rung is millions of triangles a frame at the shipped
density, ~70% of GPU time on an Intel UHD 630 (profiled). Every knob that moves it
— how much cover there is and how far each rung of its distance chain reaches, plus
the near-mesh follow radius — is read fresh at *stream* time from the live
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
from typing import Any

from openglcontext_forest_demo.config import ForestConfig


@dataclass(frozen=True)
class QualityPreset:
    """A named set of the live cover/LOD knobs that drive integrated-GPU frame cost."""
    name: str
    #: Multiplier on every cover species' own density -- the dominant GPU lever.
    #: A multiplier rather than a figure because the field is several kinds of
    #: plant at several densities: less cover means less of all of it, in
    #: proportion, not a different set of plants on a slower machine.
    cover_density: float
    clump_radius: float        # real-geometry cover disc radius, metres
    grass_card_radius: float   # how far the cover cards reach past it, metres
    grass_far_radius: float    # coarse far-cover follow-disc radius, metres
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
        "high", cover_density=_D.cover_density, clump_radius=_D.clump_radius,
        grass_card_radius=_D.grass_card_radius,
        grass_far_radius=_D.grass_far_radius, near_mesh_radius=_D.near_mesh_radius,
        cost_weight=1.0),
    "medhigh": QualityPreset(
        "medhigh", cover_density=0.67, clump_radius=24.0, grass_card_radius=86.0,
        grass_far_radius=540.0, near_mesh_radius=53.0,
        cost_weight=0.78),
    "medium": QualityPreset(
        "medium", cover_density=0.44, clump_radius=18.0, grass_card_radius=71.0,
        grass_far_radius=400.0, near_mesh_radius=50.0,
        cost_weight=0.67),
    "medlow": QualityPreset(
        "medlow", cover_density=0.33, clump_radius=15.0, grass_card_radius=57.0,
        grass_far_radius=330.0, near_mesh_radius=46.0,
        cost_weight=0.64),
    "low": QualityPreset(
        "low", cover_density=0.22, clump_radius=12.0, grass_card_radius=43.0,
        grass_far_radius=260.0, near_mesh_radius=42.0,
        cost_weight=0.53),
}

#: Every rung, cheapest first — the ladder the auto picker chooses from and F8 cycles.
LADDER = ["low", "medlow", "medium", "medhigh", "high"]
#: Backwards-compatible alias (the old three-rung name).
ORDER = LADDER

#: Valid ``--quality`` values: auto, or any ladder rung (F8 cycles the same set).
CHOICES = ["auto", *reversed(LADDER)]


def apply_to_scene(scene: Any, preset: QualityPreset) -> None:
    """Point a live :class:`~openglcontext_forest_demo.scene.ForestScene` at ``preset``.

    Mutates the streaming knobs the scene reads each frame and re-pushes the LOD
    fade windows that were baked from the old radii, so the geometry->billboard
    hand-off stays seamless at the new distances. Call on the GL thread (it touches
    node uniforms); the caller restreams at the current camera position afterwards.
    """
    cfg = scene.config
    cfg.cover_density = preset.cover_density
    cfg.clump_radius = preset.clump_radius
    cfg.grass_card_radius = preset.grass_card_radius
    cfg.grass_far_radius = preset.grass_far_radius
    scene.near_mesh_radius = preset.near_mesh_radius

    cover = scene.cover
    cover.density_scale = preset.cover_density
    cover.clump_radius = preset.clump_radius
    cover.card_radius = preset.grass_card_radius
    cover.far_radius = preset.grass_far_radius
    # Every rung's fade window was set from the radii the last preset had, and a
    # window left where those put it is a seam at the new distance. `retune` is
    # what puts them back, and re-sends them: a node uploads them as constants at
    # GL init, so nothing else would.
    cover.retune()


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
