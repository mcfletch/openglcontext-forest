"""Walkable near-photoreal forest demo for OpenGLContext.

A scene built on the OpenGLContext terrain/vegetation engine
(``OpenGLContext.scenegraph.terrain`` / ``.vegetation`` /
``OpenGLContext.move.terrainwalk``). This package holds only the concrete scene:
the biome mix and the bundled geometry/textures/heightmaps. The generic rendering
lives in OpenGLContext.

Run it with the ``oglc-forest`` console script, or ``python -m
openglcontext_forest_demo``.

Asset licensing is documented in ``ASSET-LICENSES.md`` (tree models are CC-BY 4.0
and require attribution, printed on launch).
"""
from openglcontext_forest_demo.run import main, Forest

__all__ = ["main", "Forest"]
