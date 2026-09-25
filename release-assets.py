#! /usr/bin/env python3
"""Build the art this demo fetches, install it here, and publish it.

66 MB of terrain, tree and impostor art is not something an index should be
asked to serve, so it does not travel in the wheel: it is attached to a GitHub
release and fetched through :mod:`OpenGLContext.contentpacks`. One command
covers the whole of that:

    ./release-assets.py                 # build the archive, write the registry
    ./release-assets.py --install       # ...and put it in this machine's store
    ./release-assets.py --reinstall     # ...over whatever that store already holds
    ./release-assets.py --push          # ...and attach it to the release tag

``--install`` is what makes a content release testable before it is a release:
the demo then finds the pack in its own store and runs as an installed copy
would, with nothing published and no network reached.

The registry is written from the archive this built, so its digest and its size
cannot describe a file that was never made. The archive comes from
:func:`OpenGLContext.contentpacks.archive.write`, whose bytes are a function of
the content and of nothing else, so the same art gives the same digest.

**Run it from a checkout with the LFS objects fetched.** The art is in Git LFS,
and a checkout without them holds 132-byte pointers wearing the right names;
`tests/test_assets_are_real.py` is what refuses to let those be packed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from OpenGLContext.contentpacks import archive, catalog, publish

#: Where a release's artefacts are fetched from.
URL = "https://github.com/mcfletch/openglcontext-forest/releases/download/%s/%s"

#: The namespace this demo's packs sit under.
NAMESPACE = "openglcontext-forest"

#: What proves the pack is unpacked: the first thing the scene opens.
MARKER = "forest_height.png"

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "src", "openglcontext_forest_demo", "assets")
CATALOG = os.path.join(HERE, "src", "openglcontext_forest_demo", "packs.json")


def build(where: str, name: str, into: str) -> tuple[str, int, str]:
    """Archive ``where`` as ``name``; return its path, size and digest."""
    path = archive.write(where, os.path.join(into, f"{name}.tar.gz"))
    return path, os.path.getsize(path), archive.digest(path)


def refuse_pointers(where: str) -> None:
    """Stop a pack of Git LFS pointers before it is built, not after.

    A pointer keeps the name and the extension of the art it stands for, so an
    archive of them looks right in every way but the one that matters -- which
    is how 57 of them reached PyPI wearing the names of 66 MB of art.
    """
    pointer = b"version https://git-lfs.github.com/spec/v1"
    found = []
    for root, _, files in os.walk(where):
        for leaf in files:
            path = os.path.join(root, leaf)
            with open(path, "rb") as handle:
                if handle.read(len(pointer)) == pointer:
                    found.append(os.path.relpath(path, where))
    if found:
        raise SystemExit(
            f"{len(found)} files under {where} are Git LFS pointers rather "
            f"than content: {', '.join(found[:5])}. Run `git lfs pull`."
        )


def entry(tag: str, path: str, size: int, sha: str) -> dict[str, object]:
    """What the registry says about the art this built."""
    return {
        "key": f"{NAMESPACE}/art",
        "title": "Forest demo art",
        "url": URL % (tag, os.path.basename(path)),
        "directory": "art",
        "archive": "tar",
        "approximate_bytes": size,
        "sha256": sha,
        "base": True,
        "copyright": credits(),
        "marker": MARKER,
        "notes": "The terrain, the trees and their impostors, and the plants "
                 "that grow between them. Fetched before the first frame, "
                 "because the demo opens its heightmap before it opens a "
                 "window.",
    }


def credits() -> str:
    """Whose the art is, from the file that already says so.

    One line, because a pack's terms are read on a screen with room for one;
    CREDITS.txt travels inside the pack for the rest.
    """
    return ("Tree models by Georgeous, 3D Error 404 and LOLIPOP (CC-BY 4.0); "
            "ground cover plants from Poly Haven and ground materials from "
            "ambientCG (both CC0 1.0); terrain and impostors generated for "
            "this demo. Full attribution in CREDITS.txt inside the pack.")


def install(into: str, replace: bool = False) -> int:
    """Put what was built into the store the demo reads, and say where.

    ``replace`` throws away what is installed under each key first, which is
    what a second build of a world wants: the store holds the last one, and an
    install that leaves it there shows the world before the change.

    Through the demo's own :mod:`~openglcontext_forest_demo.content`, so what
    is installed is what it will look for -- the registry it ships, the
    namespace it declares and the store it opens, rather than a second opinion
    about any of the three.
    """
    from openglcontext_forest_demo import content

    store = content.store()
    packs = catalog.merge(catalog.load(CATALOG))
    print(f"store: {store.root}")
    for chosen in catalog.offered(packs):
        # As a download does: the choice, and whatever it is incomplete
        # without, into the chosen pack's own directory.
        for pack in catalog.with_needed(chosen, packs):
            where = publish.install(pack, store, into, within=chosen,
                                    replace=replace)
            print(f"  {pack.key:<28} {os.path.relpath(where, store.root)}")
    return 0


def push(tag: str, paths: list[str]) -> int:
    """Attach the built archives to the release the registry names."""
    publish.push(publish.repository(URL % (tag, "x")), tag, paths,
                 title=f"Forest demo art {tag}",
                 notes="The art `openglcontext-forest` fetches on its first "
                       "run. Terms are in the registry and in CREDITS.txt "
                       "inside the pack.")
    print(f"attached {len(paths)} file(s) to {tag}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", default="content-v2",
                        help="the release tag the artefact is attached to "
                             "(default: %(default)s)")
    parser.add_argument("--into", default=os.path.join(HERE, "dist", "content"),
                        help="where to write the archive")
    parser.add_argument("--install", action="store_true",
                        help="install what was built into this machine's own "
                             "store, so the demo runs against it with nothing "
                             "published")
    parser.add_argument("--reinstall", action="store_true",
                        help="install, throwing away what is already in the "
                             "store under this key first, which is what a "
                             "rebuilt world needs to be the one that opens")
    parser.add_argument("--push", action="store_true",
                        help="attach the archives to the release at --tag, "
                             "creating it if it is not there yet (needs the "
                             "GitHub CLI, and an account that may write here)")
    options = parser.parse_args(argv)
    if not os.path.isdir(ASSETS):
        print(f"no assets at {ASSETS}", file=sys.stderr)
        return 2
    refuse_pointers(ASSETS)
    os.makedirs(options.into, exist_ok=True)

    path, size, sha = build(ASSETS, "forest-art", options.into)
    declared = entry(options.tag, path, size, sha)
    with open(CATALOG, "w", encoding="utf-8") as handle:
        json.dump({"namespace": NAMESPACE, "packs": [declared]}, handle,
                  indent=1)
        handle.write("\n")
    print(f"forest art: {size / 1048576:.1f} MB, sha256 {sha[:12]}, "
          f"registry written to {os.path.relpath(CATALOG, HERE)}")

    if options.install or options.reinstall:
        install(options.into, replace=options.reinstall)
    if options.push:
        return push(options.tag, [path])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
