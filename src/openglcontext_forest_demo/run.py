#!/usr/bin/env python
"""Walkable forest demo — runtime-splat terrain + instanced forest.

Run:
    oglc-forest [--help for the tunable knobs]

Controls: W/A/S/D move, mouse-look, arrows/PageUp-Down also navigate. You walk on
the terrain (height-clamped, blocked by trunks). Real Great-Smoky-Mountains
elevation drives the terrain; the ground is a crisp runtime multi-layer splat
(forest_floor/grass/rock/moss). The forest is GPU-instanced with distance LOD: real
3D tree *meshes* near the camera, baked *impostor* billboards farther out
(cross-faded), plus camera-following grass (two LODs) and terrain sun-shadows + tree
canopy shade. Biome mix: fir conifers on high/steep ground, maples in valleys,
realistic deciduous between. ~640k trees, 3x+ above 60fps. First run downloads CC0
ground textures (cached after).

This is only the *navigation*: the scene itself is built by
:func:`openglcontext_forest_demo.scene.build_forest_scene` from a
:class:`~openglcontext_forest_demo.config.ForestConfig`, and the same scene +
streamers back a future "driving through a forest" demo — that demo reuses
``build_forest_scene`` and attaches its own controls. The generic rendering lives in
OpenGLContext:
  OpenGLContext.scenegraph.terrain      — HeightField, SplatTerrain
  OpenGLContext.scenegraph.vegetation   — InstancedBillboards, InstancedMeshLOD,
                                          InstancedClumps, world_grid_scatter
  OpenGLContext.move.terrainwalk        — TerrainWalkMixin (clamp/collide/stream)

Tree assets (CC-BY, see CREDITS-trees.txt): "Fir tree" by Georgeous, "Noel Pine
Tree" by 3D Error 404, "Maple trees pack" by LOLIPOP, "Realistic Trees Collection"
by Jungle Jim, "Low Poly Forest Tree Pack" by 99.Miles.
"""
import os, sys, math
os.environ.setdefault("OPENGLCONTEXT_PROFILE", "core")
os.environ.setdefault("OPENGLCONTEXT_RENDERER", "pbr")
os.environ.setdefault("OPENGLCONTEXT_BACKEND", "glfw")
from OpenGLContext import testingcontext, quaternion
from OpenGLContext.move.terrainwalk import TerrainWalkMixin
BaseContext = testingcontext.getInteractive()

from openglcontext_forest_demo.config import ForestConfig, config_from_args
from openglcontext_forest_demo.scene import build_forest_scene


class Forest(TerrainWalkMixin, BaseContext):
    """Walk the forest: build the shared scene, then clamp/collide/stream on the move.

    Thin glue over :func:`build_forest_scene` — swapping this navigation for a vehicle
    is the whole point of keeping the scene build reusable. ``config`` is stashed by
    :func:`main` from the parsed command line before the context is created."""

    config = None   # ForestConfig; set by main() (falls back to defaults for a bare run)

    def OnInit(self):
        try:
            import glfw; glfw.swap_interval(0)
        except Exception:
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

        # walk: clamp to terrain, block only the trunks, stream veg on the move.
        self.init_walk(scene.hf, scene.collider_pos, scene.collider_radius)
        # Register the SAME reusable streamers a driving demo would; impostor culling
        # needs the camera forward vector, which this navigation supplies.
        self.add_stream(10.0, scene.stream_near)
        self.add_stream(40.0, scene.stream_far)
        self.add_stream(20.0, self._stream_impostors, turn=math.radians(9.0))

    # Streamer methods delegate to the scene so a Forest subclass (bench, capture, a
    # driving demo) can drive the veg fields with the familiar self._stream_* names.
    def _stream_near(self, x, z):
        self.scene.stream_near(x, z)

    def _stream_far(self, x, z):
        self.scene.stream_far(x, z)

    def _stream_impostors(self, x, z):
        fx, fz = self._tw_forward()
        self.scene.stream_impostors(x, z, fx, fz)


def print_credits():
    """Print the CC-BY attribution notices required for the tree models used."""
    print("=" * 74)
    print("Forest demo — asset credits (full text in CREDITS.txt)")
    print("Tree models, CC-BY 4.0 (http://creativecommons.org/licenses/by/4.0/):")
    for line in ('"Fir tree" by Georgeous (https://skfb.ly/pA8TG)',
                 '"Noel_Pine_Tree" by 3D Error 404 (https://skfb.ly/6XHoJ)',
                 '"Maple trees pack" by LOLIPOP (https://skfb.ly/p9tGx)',
                 '"Realistic Trees Collection" by Jungle Jim (https://skfb.ly/pDzJR)',
                 '"Low Poly Forest Tree Pack" by 99.Miles (https://skfb.ly/pJXrH)'):
        print("  * " + line)
    print("Ground textures: CC0 via ambientCG.com. Terrain: public AWS Terrain Tiles.")
    print("=" * 74)
    sys.stdout.flush()   # CC-BY attribution must reach the user even if stdout is piped


def main():
    """Console entry point (``oglc-forest``): parse knobs, print attributions, run."""
    Forest.config = config_from_args()
    print_credits()
    Forest.ContextMainLoop()


if __name__ == "__main__":
    main()
