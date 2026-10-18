#! /usr/bin/env python3
"""Build the art this demo fetches, install it here, and publish it.

66 MB of terrain, tree and impostor art is not something an index should be
asked to serve, so it does not travel in the wheel: it is attached to a GitHub
release and fetched through :mod:`OpenGLContext.contentpacks`. One command
covers the whole of that:

    ./release-assets.py                   # build the archive and its registry
    ./release-assets.py --install         # ...and put it in this machine's store
    ./release-assets.py --reinstall       # ...replacing the copy installed there
    ./release-assets.py --push            # ...attach it to its release and
                                          #    write the shipped packs.json

``--install`` is what makes a content release testable before it is a release:
the demo then finds the pack in its own store and runs as an installed copy
would, with nothing published and no network reached.

The registry is written from the archive this built, so its digest and its size
cannot describe a file that was never made. It is written beside the archive;
``src/openglcontext_forest_demo/packs.json``, the one the wheel ships, is
rewritten only by ``--push`` or ``--write-registry``. The options, the install
and the push are :func:`OpenGLContext.contentpacks.publish.main`'s.

Run it from a checkout with the LFS objects fetched (``git lfs pull``): the art
is in Git LFS, and :func:`OpenGLContext.contentpacks.archive.write` refuses a
tree holding pointers in place of the files.
"""

from __future__ import annotations

import os

from OpenGLContext.contentpacks import publish

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "src", "openglcontext_forest_demo", "assets")
CATALOG = os.path.join(HERE, "src", "openglcontext_forest_demo", "packs.json")

#: Whose the art is, in the one line a consent screen has room for;
#: CREDITS.txt travels inside the pack for the rest.
CREDIT = ("Tree models by Georgeous, 3D Error 404 and LOLIPOP (CC-BY 4.0); "
          "ground cover plants from Poly Haven and ground materials from "
          "ambientCG (both CC0 1.0); terrain and impostors generated for "
          "this demo. Full attribution in CREDITS.txt inside the pack.")


def declare(build: publish.Build) -> list[dict]:
    """Build the art pack; what the registry says about it."""
    if not os.path.isdir(ASSETS):
        raise SystemExit(f"no assets at {ASSETS}")
    built = build.archive(ASSETS, "forest-art")
    return [build.entry(
        "art", built, title="Forest demo art", copyright=CREDIT, base=True,
        # The first thing the scene opens, so its presence is what proves the
        # pack is unpacked.
        marker="forest_height.png",
        notes="The terrain, the trees and their impostors, and the plants "
              "that grow between them. Fetched before the first frame, "
              "because the demo opens its heightmap before it opens a "
              "window.")]


RELEASE = publish.Release(
    namespace="openglcontext-forest",
    url="https://github.com/mcfletch/openglcontext-forest/releases/download/"
        "%s/%s",
    catalog=CATALOG, declare=declare,
    into=os.path.join(HERE, "dist", "content"),
    description=__doc__.split("\n\n")[0], title="Forest demo art",
    notes="The art `openglcontext-forest` fetches on its first run. Terms are "
          "in the registry and in CREDITS.txt inside the pack.")


if __name__ == "__main__":
    raise SystemExit(publish.main(RELEASE))
