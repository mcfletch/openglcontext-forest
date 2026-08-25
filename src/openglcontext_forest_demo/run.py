#!/usr/bin/env python
"""Walkable forest demo — runtime-splat terrain + instanced forest.

Run:
    oglc-forest [--help for the tunable knobs]

Controls:
- the mouse steers
- W/A/S/D move
- Shift runs
- Space jumps
- ``m`` cycles to the walk and fly modes
- ``f`` flies
- ``g`` hands the camera to the free-fly navigator and back
- ``F6`` shows the keys and lets them be rebound
- ``F8`` cycles the render-quality preset (low / medium / high)
- ``F10`` the rendering settings
- ``F2`` saves a screenshot
- ``Alt+F`` the developer overlay
- ``Escape`` the menu

You walk on the terrain (height-clamped, blocked by trunks). Real
Great-Smoky-Mountains elevation drives the terrain; the ground is a runtime
multi-layer splat (forest_floor/grass/rock/moss). The forest is GPU-instanced with
distance LOD: real 3D tree *meshes* near the camera, baked *impostor* billboards
farther out (cross-faded), plus camera-following grass (two LODs) and terrain
sun-shadows + tree canopy shade. Biome mix: fir conifers on high/steep ground,
maples in valleys, realistic deciduous between. ~230k trees, comfortably above 60fps
on a 3060Ti. First run downloads CC0 ground textures (cached after).

This is only the *navigation* and the screens around it: the scene itself is built
by :func:`openglcontext_forest_demo.scene.build_forest_scene` from a
:class:`~openglcontext_forest_demo.config.ForestConfig`, and the same scene +
streamers back a future "driving through a forest" demo — that demo reuses
``build_forest_scene`` and attaches its own controls. The generic parts live in
OpenGLContext::

  OpenGLContext.scenegraph.terrain      — HeightField, SplatTerrain
  OpenGLContext.scenegraph.vegetation   — InstancedBillboards, InstancedMeshLOD,
                                          InstancedClumps, world_grid_scatter
  OpenGLContext.move.terrainwalk        — TerrainWalkMixin (the avatar, the
                                          ground, the trunks and the streaming)
  OpenGLContext.ui                      — the overlay screens

Tree assets (CC-BY, full attribution in CREDITS.txt, which is what the credits
screen and the launch notice both show):

- "Fir tree" by Georgeous
- "Noel Pine Tree" by 3D Error 404
- "Maple trees pack" by LOLIPOP
- "Realistic Trees Collection" by Jungle Jim
- "Low Poly Forest Tree Pack" by 99.Miles.
"""
import logging
import math
import os
import pathlib
import sys
import time
from typing import Any

os.environ.setdefault("OPENGLCONTEXT_PROFILE", "core")
os.environ.setdefault("OPENGLCONTEXT_RENDERER", "pbr")
os.environ.setdefault("OPENGLCONTEXT_BACKEND", "glfw")
from OpenGLContext import quaternion, testingcontext
from OpenGLContext.contextdefinition import ContextDefinition
from OpenGLContext.move import modes as movemodes
from OpenGLContext.move.terrainwalk import TerrainWalkMixin
from OpenGLContext.ui import bindings, settings
from OpenGLContext.ui.overlay import OverlayMixin

BaseContext = testingcontext.getInteractive()

from openglcontext_forest_demo import menu
from openglcontext_forest_demo import quality as qual
from openglcontext_forest_demo.config import ForestConfig, config_from_args
from openglcontext_forest_demo.scene import build_forest_scene
from openglcontext_forest_demo.streaming import AsyncStreamer

log = logging.getLogger(__name__)

#: The window this opens, before anyone resizes it.
WINDOW_SIZE = (1280, 800)

#: Walking speeds, in metres per second. A walk and a jog through undergrowth.
WALK_SPEED = 3.0
RUN_SPEED = 6.5
#: Flying is for looking at the forest rather than being in it, and the forest is
#: four kilometres across, so it is much faster than a run.
FLY_SPEED = 30.0

#: Keys that open a screen, and the method each opens.  On ``keyboard`` key-downs
#: rather than ``keypress``: a function key produces no character, so a keypress
#: binding for one is accepted and then never fires.
SCREEN_KEYS = (
    ('<F6>', 'showBindings'),
    ('<F8>', 'cycleQuality'),
    ('<F10>', 'showSettings'),
    ('m', 'cycleMovementMode'),
)


def movement_modes() -> list[Any]:
    """The ways of moving this demo offers, as declared nodes.

    Declared rather than hand-rolled so the F6 page can present them and rebind
    their keys, and so a game embedding this scene retunes them by setting
    fields.  ``FPSMode`` walks exactly as ``WalkMode`` does and differs only in
    taking the pointer to steer with; **it is declared first, so it is the mode
    the demo starts in** — the navigation manager takes the first selectable
    mode, and a forest is a place you look around. Walking with ``q``/``e`` stays
    one ``m`` away for anyone who would rather keep the pointer.
    """
    return [
        movemodes.FPSMode(name='fps', walkSpeed=WALK_SPEED, runSpeed=RUN_SPEED),
        movemodes.WalkMode(name='walk', walkSpeed=WALK_SPEED, runSpeed=RUN_SPEED),
        movemodes.FlyMode(name='fly', flySpeed=FLY_SPEED),
    ]


class Forest(OverlayMixin, TerrainWalkMixin, BaseContext):
    """Walk the forest: build the shared scene, then walk, look and stream.

    Thin glue over :func:`build_forest_scene` — swapping this navigation for a
    vehicle is the whole point of keeping the scene build reusable.  ``config``
    is stashed by :func:`main` from the parsed command line before the context is
    created.

    :class:`~OpenGLContext.ui.overlay.OverlayMixin` comes first so its event
    routing runs before the navigation mix-in's: while a screen is up the
    movement sampler is not fed at all, which is what stops the walk continuing
    under a menu somebody is reading.
    """

    config: ForestConfig | None = None   # set by main() (defaults for a bare run)

    def OnInit(self):
        try:
            import glfw
            glfw.swap_interval(0)   # best-effort: uncap the loop for benching
        except Exception:  # noqa: BLE001, S110 -- glfw optional; leave vsync as-is on any failure
            pass
        cfg = self.config if self.config is not None else ForestConfig()
        self.eye_height = cfg.eye_height
        scene = build_forest_scene(cfg)
        self.scene = scene
        self.sg = scene.sceneGraph
        self.hf = scene.hf   # convenience for subclasses/tools (bench, capture, a driving demo)

        ex, ez = scene.spawn
        if self.platform is not None:
            self.platform.setFrustum(math.radians(cfg.fov_deg), None, cfg.near, cfg.far)
            self.platform.setPosition((ex, scene.hf.height_at(ex, ez) + self.eye_height, ez))
            self.platform.setOrientation(quaternion.fromXYZR(0, 1, 0, 2.3))

        # Walk: the avatar stands on the height field, is blocked by the trunks,
        # and streams the vegetation as it moves.  After the camera is placed,
        # because that is where the avatar is stood up.
        self.init_walk(scene.hf, scene.collider_pos, scene.collider_radius)
        self.setupPhysics(enable=True)
        self.setupScreenshots()
        self.bindScreenKeys()
        # Register the SAME reusable streamers a driving demo would; impostor
        # culling needs the camera forward vector, which this navigation supplies.
        # Off-thread stream recompute: the scatter is pure numpy (GIL-released while
        # it runs) and latency-tolerant — a field trailing the camera a frame or two
        # is hidden by the LOD fade bands — so it runs on worker threads and the
        # render loop only drains the finished arrays (see OnDraw). This is what
        # keeps a step-across-a-threshold from landing a ~50 ms scatter on one frame.
        self._near_stream = AsyncStreamer(scene.compute_near, scene.apply_near, "stream-near")
        self._far_stream = AsyncStreamer(scene.compute_far, scene.apply_far, "stream-far")
        self._imp_stream = AsyncStreamer(scene.compute_impostors, scene.apply_impostors, "stream-imp")
        self._streamers = (self._near_stream, self._far_stream, self._imp_stream)
        self.add_stream(10.0, lambda x, z: self._near_stream.request(x, z))
        self.add_stream(40.0, lambda x, z: self._far_stream.request(x, z))
        self.add_stream(20.0, self._request_impostors, turn=math.radians(9.0))

        # --- render quality (Tier 1: hold ~60 fps on integrated GPUs) ---
        # The presets are demo scatter knobs (clump/grass density + radii, near-mesh
        # radius) the scene already reads each frame, so a change applies live with
        # no rebuild.  'auto' starts at high and lets OnDraw's frame-rate measure
        # step it down to what the GPU holds; high/medium/low freeze the choice.
        # F8 cycles it by hand.
        self._quality = 'medium'
        self._auto = None
        self._frame_t = None
        self._pending_quality = None    # a deferred auto downgrade, applied on first move
        self._last_xz = None
        self._render_wh = None          # last render size, to re-measure quality on a resize
        mode = (getattr(cfg, 'quality', 'auto') or 'auto').lower()
        self._auto_mode = (mode == 'auto')   # re-measure on resize only while auto is in charge
        if mode == 'auto':
            # Start at the shipped 'high' look: a capable GPU (the primary target)
            # holds it and never changes. A GPU that can't sustain it measures once,
            # picks a lower target, and that target is applied deferred -- folded into
            # the next streaming move so it is not seen popping in place.
            self._auto = qual.AutoQuality(target_fps=60.0, start='high')
            self._apply_quality('high', announce=False)
        else:
            self._apply_quality(mode, announce=False)
        measuring = " — measuring" if self._auto is not None else ""
        print(f"quality: {mode}{measuring}  (F8 cycles the quality rungs)")

        print_controls()

    # -- render quality ---------------------------------------------------
    def _apply_quality(self, level, announce=True, defer=False):
        """Point the live scene at preset ``level``.

        With ``defer`` the change is held until the next streaming move
        (:meth:`_flush_pending_quality`), so an auto downgrade folds into the churn
        the field already does as the camera walks; re-scattering the whole field in
        place while standing still is what reads as the grass popping to a sparser
        set. A direct pick (F8) applies at once."""
        self._quality = level
        if defer:
            self._pending_quality = level
            return
        self._pending_quality = None
        qual.apply_to_scene(self.scene, qual.PRESETS[level])
        try:
            x, z = self._tw_xz()
        except Exception:  # noqa: BLE001 -- nav not live yet at first apply; fall back to spawn
            x, z = self.scene.spawn
        try:
            self._stream_near(x, z); self._stream_far(x, z); self._stream_impostors(x, z)
        except Exception:  # noqa: BLE001, S110 -- before nav is live the first OnIdle streams instead
            pass
        self.triggerRedraw(1)
        if announce:
            print(f"quality -> {level}"); sys.stdout.flush()

    def _flush_pending_quality(self):
        """Apply a deferred quality change at the next move (see :meth:`_apply_quality`)."""
        level = self._pending_quality
        if level is None:
            return
        self._pending_quality = None
        qual.apply_to_scene(self.scene, qual.PRESETS[level])
        x, z = self._tw_xz()
        self._near_stream.request(x, z)
        self._far_stream.request(x, z)
        self._request_impostors(x, z)

    def cycleQuality(self, event=None):
        """F8: step low -> medium -> high -> low.  A manual pick freezes auto."""
        self._auto = None
        self._auto_mode = False        # a hand pick wins; stop re-measuring on resize
        self._apply_quality(qual.next_level_cycle(self._quality))

    def OnResize(self, width, height, *a):
        """Re-measure quality at a new render size (fill cost scales with pixels).

        A GPU that held ``high`` in a window may not at fullscreen and vice versa, so
        a material size change restarts the one-shot auto measurement from the current
        level; the new pick applies deferred, like any auto change. Only while auto is
        in charge -- a hand F8 pick is left alone."""
        r = super().OnResize(width, height, *a)
        if self._auto_mode:
            prev = self._render_wh
            px, ppx = int(width) * int(height), (prev[0] * prev[1] if prev else 0)
            if prev is None or abs(px - ppx) > 0.15 * max(px, 1):
                self._auto = qual.AutoQuality(target_fps=60.0, start=self._quality)
                self._frame_t = None
            self._render_wh = (int(width), int(height))
        return r

    def _request_impostors(self, x, z):
        """Ask the impostor worker to re-cull to the current forward view cone."""
        fx, fz = self._tw_forward()
        self._imp_stream.request(x, z, fx, fz)

    def OnIdle(self, *a):
        """Apply a deferred auto-quality change on the first move, then walk/stream."""
        if self._pending_quality is not None and self.platform is not None:
            xz = self._tw_xz()
            if self._last_xz is not None and (abs(xz[0] - self._last_xz[0])
                                              + abs(xz[1] - self._last_xz[1])) > 0.4:
                self._flush_pending_quality()
            self._last_xz = xz
        sup = super()
        return sup.OnIdle(*a) if hasattr(sup, 'OnIdle') else None

    def OnDraw(self, *a, **k):
        """Render, and while auto-quality is measuring, time one window and decide."""
        # Hand any finished off-thread stream result to its node before drawing, so
        # the fresh instances upload in this frame's render.
        for streamer in self._streamers:
            streamer.drain()
        # Re-center the clump LOD on the live camera every frame (cheap mask of the
        # cached scatter), so the disc tracks the walk with no streaming lag -- the
        # leading clumps fade in through the LOD band instead of popping up in density.
        if self.platform is not None:
            try:
                self.scene.update_clump_lod(*self._tw_xz())
            except Exception:  # noqa: BLE001, S110 -- best-effort re-center; never crash the draw
                pass
        r = super().OnDraw(*a, **k)
        if self._auto is not None:
            now = time.perf_counter()
            if self._frame_t is not None:
                self._auto.add_frame((now - self._frame_t) * 1000.0)
                if self._auto.decided:
                    m = self._auto.last_median
                    fps = 1000.0 / m if m else 0.0
                    if self._auto.target is not None and self._auto.target != self._quality:
                        self._apply_quality(self._auto.target, announce=False, defer=True)
                        print(f"quality: auto measured {fps:.0f} fps at high -> "
                              f"{self._auto.target} (applies as you move)")
                    else:
                        print(f"quality: auto kept {self._quality} ({fps:.0f} fps)")
                    sys.stdout.flush()
                    self._auto = None       # one measurement, one decision, then done
            self._frame_t = now
            if self._auto is not None:      # keep force-redrawing until the window fills
                self.triggerRedraw(1)
        return r

    # -- the screens ------------------------------------------------------
    def bindScreenKeys(self, context: Any = None) -> None:
        """Bind the keys that open a screen or change the way you move.

        ``context`` is what to bind on, defaulting to this one, so a test can
        hand in a recorder rather than standing up a window.
        """
        context = context if context is not None else self
        for name, attribute in SCREEN_KEYS:
            context.addEventHandler('keyboard', name=name, state=1,
                                    function=getattr(self, attribute))

    def OnEscape(self, event: Any = None) -> Any:
        """Escape puts the menu up rather than throwing the walk away.

        A camera somewhere in a forest that took a moment to build is a session
        worth something, and a key pressed to back out of *something else* must
        not be what ends it.  Resume and Quit are both one click away from here.
        """
        return self.showMenu(event)

    def showMenu(self, event: Any = None) -> Any:
        """Raise the menu, or bring up the one already showing."""
        existing = self.overlays.named(menu.MENU_NAME)
        if existing is not None:
            return existing
        return self.pushOverlay(menu.main_menu(
            on_resume=self.closeMenu,
            on_bindings=self.showBindings,
            on_settings=self.showSettings,
            on_credits=self.showCredits,
            on_quit=self.OnQuit,
            subtitle=self.menuSubtitle()))

    def menuSubtitle(self) -> str:
        """What the menu says under its title: how big the world behind it is."""
        cfg = self.config if self.config is not None else ForestConfig()
        return (f"{len(self.scene.collider_pos)} trees over "
                f"{cfg.extent / 1000.0:.1f} km of Great Smoky Mountains elevation")

    def closeMenu(self, event: Any = None) -> None:
        """Put the menu away, leaving the forest showing."""
        panel = self.overlays.named(menu.MENU_NAME)
        if panel is not None and not panel.closed:
            panel.close(True)

    def showSettings(self, event: Any = None) -> Any:
        """Raise the shared settings screen (F10)."""
        return settings.open_settings(self)

    def showBindings(self, event: Any = None) -> Any:
        """Raise the shared key-bindings screen (F6)."""
        return bindings.open_bindings(self)

    def showCredits(self, event: Any = None) -> Any:
        """Raise the asset attribution the tree models are licensed on."""
        existing = self.overlays.named(menu.CREDITS_NAME)
        if existing is not None:
            return existing
        return self.pushOverlay(menu.credits_screen(credits_text()))

    def cycleMovementMode(self, event: Any = None) -> Any:
        """Step to the next declared movement mode, as ``m`` does in twitch."""
        navigation = self.getNavigation()
        return navigation.cycle() if navigation is not None else None

    # -- streaming --------------------------------------------------------
    # Streamer methods delegate to the scene so a Forest subclass (bench, capture, a
    # driving demo) can drive the veg fields with the familiar self._stream_* names.
    def _stream_near(self, x, z):
        self.scene.stream_near(x, z)

    def _stream_far(self, x, z):
        self.scene.stream_far(x, z)

    def _stream_impostors(self, x, z):
        fx, fz = self._tw_forward()
        self.scene.stream_impostors(x, z, fx, fz)


#: What is shown if ``CREDITS.txt`` cannot be read. CC-BY asks for attribution
#: wherever the work appears, so a packaging mistake must not be able to turn the
#: credits screen into an empty one; this names every work, its author and the
#: licence, and the file carries the authors' own longer wording.
FALLBACK_CREDITS = """Forest demo — asset credits

Tree models, CC-BY 4.0 (http://creativecommons.org/licenses/by/4.0/):
  * "Fir tree" by Georgeous — https://skfb.ly/pA8TG
  * "Noel_Pine_Tree" by 3D Error 404 — https://skfb.ly/6XHoJ
  * "Maple trees pack" by LOLIPOP — https://skfb.ly/p9tGx
  * "Realistic Trees Collection" by Jungle Jim — https://skfb.ly/pDzJR
  * "Low Poly Forest Tree Pack" by 99.Miles — https://skfb.ly/pJXrH

Ground textures: CC0 via ambientCG.com.
Terrain: public AWS Terrain Tiles."""


def credits_path() -> pathlib.Path:
    """Where the packaged ``CREDITS.txt`` lives."""
    return pathlib.Path(__file__).parent / 'CREDITS.txt'


def credits_text() -> str:
    """The asset notices, as one block of text.

    Read from ``CREDITS.txt`` so the file is the single place an attribution is
    written: the launch notice and the credits screen are the same text, and one
    cannot fall behind the other. Falls back to :data:`FALLBACK_CREDITS` if the
    file is unreadable, since an attribution that silently disappeared would be a
    licence breach rather than a cosmetic fault.
    """
    try:
        return credits_path().read_text(encoding='utf-8').strip()
    except OSError as error:
        log.warning('cannot read %s (%s); using the built-in attribution',
                    credits_path(), error)
        return FALLBACK_CREDITS


def print_credits():
    """Print the CC-BY attribution notices required for the tree models used."""
    print("=" * 74)
    print(credits_text())
    print("=" * 74)
    sys.stdout.flush()   # CC-BY attribution must reach the user even if stdout is piped


def print_controls():
    """Say what the keys do, for someone who ran it from a terminal."""
    sys.stdout.write(
        "  mouse steers, w a s d move, shift runs, space jumps\n"
        "  m cycles walk / fly / mouse-look, f flies, g free-fly camera\n"
        "  F6 keys, F8 quality, F10 rendering settings, F2 "
        "screenshot, alt+f developer overlay, escape menu\n")
    sys.stdout.flush()


def main():
    """Console entry point (``oglc-forest``): parse knobs, print attributions, run."""
    Forest.config = config_from_args()
    print_credits()
    Forest.ContextMainLoop(definition=ContextDefinition(
        title=menu.TITLE, size=WINDOW_SIZE, movementModes=movement_modes()))


if __name__ == "__main__":
    main()
