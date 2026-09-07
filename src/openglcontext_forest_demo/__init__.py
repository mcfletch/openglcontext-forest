"""Walkable near-photoreal forest demo for OpenGLContext.

A scene built on the OpenGLContext terrain/vegetation engine
(``OpenGLContext.scenegraph.terrain`` / ``.vegetation`` /
``OpenGLContext.move.terrainwalk``), walked with the same avatar, movement modes
and overlay screens as ``oglc-view`` and ``twitch``. This package holds the
concrete scene — the biome mix and the bundled geometry/textures/heightmaps —
plus a reusable builder so other navigation demos can share the exact same world.

Run it with the ``oglc-forest`` console script (``--help`` lists the tunable knobs),
or ``python -m openglcontext_forest_demo``. The mouse steers; ``Escape`` opens the
menu and ``F6``/``F10`` the key bindings and the rendering settings.

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

#: The distribution's version, which `pyproject.toml` reads through
#: `[tool.setuptools.dynamic]` -- one place to bump, and the place the release
#: tooling looks.
__version__ = "1.0.0a1"

from openglcontext_forest_demo.config import (
    ForestConfig,
    build_arg_parser,
    config_from_args,
)
from openglcontext_forest_demo.run import (
    Forest,
    credits_text,
    main,
    movement_modes,
    print_credits,
)
from openglcontext_forest_demo.scene import ForestScene, build_forest_scene

__all__ = [
    "Forest",
    "ForestConfig",
    "ForestScene",
    "build_arg_parser",
    "build_forest_scene",
    "config_from_args",
    "credits_text",
    "main",
    "movement_modes",
    "print_credits",
    "__version__",
]
