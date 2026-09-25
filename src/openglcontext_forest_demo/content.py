"""Where this demo's art comes from.

66 MB of terrain, trees and impostors is not something an index should be asked
to serve, so it does not travel in the wheel: it is a **base pack**, attached to
a GitHub release and fetched through :mod:`OpenGLContext.contentpacks` before
the first frame (:func:`openglcontext_forest_demo.run.fetch_art`). The demo
opens its heightmap before it opens a window, so the fetch happens on the
console, before the window.

    >>> from openglcontext_forest_demo import content
    >>> content.art_directory()                            # doctest: +SKIP

:data:`CONTENT` is the engine's
:class:`~OpenGLContext.contentpacks.application.Application` for this demo:
which registry, which namespace, which pack is the art, and the copy of the
art in a checkout, used while the pack is not installed. Where the art is, is
asked each time a scene is built.
"""

from __future__ import annotations

import os

from OpenGLContext.contentpacks import Application, ContentPack, ContentStore

__all__ = ["BASE", "CATALOG_PATH", "CONTENT", "IN_WHEEL", "NAMESPACE",
           "art_directory", "needed_to_start", "registry", "store"]

#: The registry shipped with the demo.
CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "packs.json")

#: The namespace this demo's packs sit under.
NAMESPACE = "openglcontext-forest"

#: The pack the demo cannot start without.
BASE = f"{NAMESPACE}/art"

#: The art in a checkout (Git LFS). An installed wheel does not carry it.
IN_WHEEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

#: This demo's content.
CONTENT = Application(NAMESPACE, CATALOG_PATH, base=BASE, fallback=IN_WHEEL)


def registry() -> list[ContentPack]:
    """Every pack this build offers."""
    return CONTENT.registry()


def store(root: str | None = None) -> ContentStore:
    """Where this demo's content lives on this machine."""
    return CONTENT.store(root)


def needed_to_start(where: ContentStore | None = None) -> list[ContentPack]:
    """What a first run has to fetch before the demo can draw anything."""
    return CONTENT.needed_to_start(where)


def art_directory(where: ContentStore | None = None) -> str:
    """Where the art is read from, as of now.

    The base pack once it is installed, else the copy in a checkout;
    :class:`~OpenGLContext.contentpacks.application.NotInstalled` if neither.
    """
    return CONTENT.base_directory(where)
