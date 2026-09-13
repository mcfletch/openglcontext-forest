"""Where this demo's art comes from.

66 MB of terrain, trees and impostors is not something an index should be asked
to serve, so it does not travel in the wheel: it is a **base pack**, attached to
a GitHub release and fetched through :mod:`OpenGLContext.contentpacks` before
the first frame. The demo opens its heightmap before it opens a window, so
"before the first frame" is not a nicety.

    >>> from openglcontext_forest_demo import content
    >>> content.art_directory()                            # doctest: +SKIP

The facility is the engine's; what is here is which registry, which namespace,
and the rule that the pack wins over the copy in the wheel while there is one.
"""

from __future__ import annotations

import os

from OpenGLContext.contentpacks import ContentPack, ContentStore, catalog, fetch

__all__ = ["BASE", "CATALOG_PATH", "IN_WHEEL", "NAMESPACE", "art_directory",
           "needed_to_start", "registry", "store"]

#: The registry shipped with the demo.
CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "packs.json")

#: The namespace this demo's packs sit under.
NAMESPACE = "openglcontext-forest"

#: The pack the demo cannot start without.
BASE = f"{NAMESPACE}/art"

#: The copy inside the package, while there is one.
IN_WHEEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

_registry: list[ContentPack] | None = None


def registry() -> list[ContentPack]:
    """Every pack this build offers."""
    global _registry
    if _registry is None:
        _registry = catalog.merge(catalog.load(CATALOG_PATH))
    return _registry


def store(root: str | None = None) -> ContentStore:
    """Where this demo's content lives on this machine."""
    return ContentStore(NAMESPACE, root=root)


def needed_to_start(where: ContentStore | None = None) -> list[ContentPack]:
    """What a first run has to fetch before the demo can draw anything."""
    return list(fetch.missing_base(registry(),
                                   where if where is not None else store()))


def art_directory(where: ContentStore | None = None) -> str:
    """Where the art is read from.

    The base pack once it has been fetched, and the copy inside the wheel until
    then. Both, deliberately: the art leaves the wheel when the release holding
    it exists, and until that day an install has to work anyway. When it does
    leave, this is the only place that has to stop looking there.
    """
    found = where if where is not None else store()
    pack = catalog.pack_for_key(BASE, registry())
    if pack is not None:
        root = found.root_for(pack)
        if root is not None:
            return str(root)
    return IN_WHEEL
