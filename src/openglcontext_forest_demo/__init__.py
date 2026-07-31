"""Walkable near-photoreal forest demo for OpenGLContext.

A scene built on the OpenGLContext terrain/vegetation engine
(``OpenGLContext.scenegraph.terrain`` / ``.vegetation`` /
``OpenGLContext.move.terrainwalk``). This package holds the concrete scene — the
biome mix and the bundled geometry/textures/heightmaps — plus a reusable builder so
other navigation demos can share the exact same world.

Run it with the ``oglc-forest`` console script (``--help`` lists the tunable knobs),
or ``python -m openglcontext_forest_demo``.

Reuse the scene from a different navigation layer (e.g. a "driving through a forest"
demo) by building it inside your own context and attaching your own controls::

    from openglcontext_forest_demo import ForestConfig, build_forest_scene

    class Drive(SomeVehicleMixin, BaseContext):
        def OnInit(self):
            scene = build_forest_scene(ForestConfig(clump_radius=45.0))
            self.sg = scene.sceneGraph
            # drive over scene.hf; reuse the identical streamers:
            self.add_stream(10.0, scene.stream_near)
            self.add_stream(40.0, scene.stream_far)
            self.add_stream(20.0, lambda x, z: scene.stream_impostors(x, z, *self.heading()))

``build_forest_scene`` only creates scenegraph nodes and numpy data (GL buffers
upload lazily on first render), so call it inside a live GL context's ``OnInit``.

Asset licensing is documented in ``ASSET-LICENSES.md`` (tree models are CC-BY 4.0
and require attribution, printed on launch).
"""
from openglcontext_forest_demo.run import main, Forest, print_credits
from openglcontext_forest_demo.config import ForestConfig, config_from_args, build_arg_parser
from openglcontext_forest_demo.scene import build_forest_scene, ForestScene

__all__ = [
    "main", "Forest", "print_credits",
    "ForestConfig", "config_from_args", "build_arg_parser",
    "build_forest_scene", "ForestScene",
]
