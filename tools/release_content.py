#! /usr/bin/env python3
"""Build the art this demo fetches, and the registry that names it.

66 MB of terrain, tree and impostor art is not something an index should be
asked to serve, so it does not travel in the wheel: it is attached to a GitHub
release and fetched through :mod:`OpenGLContext.contentpacks`.

    tools/release_content.py --tag content-v1

The registry is written from the archive this built, so its digest and its size
cannot describe a file that was never made. The archive is reproducible --
fixed timestamps, sorted entries, no owner and no mode beyond read/write -- so
the same art gives the same bytes and the same digest.

**Run it from a checkout with the LFS objects fetched.** The art is in Git LFS,
and a checkout without them holds 132-byte pointers wearing the right names;
`tests/test_assets_are_real.py` is what refuses to let those be packed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile

#: Where a release's artefacts are fetched from.
URL = "https://github.com/mcfletch/openglcontext-forest/releases/download/%s/%s"

#: The namespace this demo's packs sit under.
NAMESPACE = "openglcontext-forest"

#: Fixed, so the bytes do not depend on when they were built.
EPOCH = 1600000000

#: What proves the pack is unpacked: the first thing the scene opens.
MARKER = "forest_height.png"

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(HERE, "src", "openglcontext_forest_demo", "assets")
CREDITS = os.path.join(HERE, "src", "openglcontext_forest_demo", "CREDITS.txt")


def build(where: str, name: str, into: str) -> tuple[str, int, str]:
    """Archive ``where`` as ``name``; return its path, size and digest."""
    path = os.path.join(into, f"{name}.tar.gz")
    with tarfile.open(path, "w:gz", compresslevel=9) as archive:
        for root, directories, files in os.walk(where):
            directories.sort()
            for leaf in sorted(files):
                full = os.path.join(root, leaf)
                info = archive.gettarinfo(full, os.path.relpath(full, where))
                info.mtime = EPOCH
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mode = 0o644
                with open(full, "rb") as handle:
                    archive.addfile(info, handle)
    return path, os.path.getsize(path), digest_of(path)


def digest_of(path: str) -> str:
    found = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            found.update(block)
    return found.hexdigest()


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", default="content-v2",
                        help="the release tag the artefact is attached to")
    parser.add_argument("--into", default="dist/content",
                        help="where to write the archive")
    options = parser.parse_args(argv)
    if not os.path.isdir(ASSETS):
        print(f"no assets at {ASSETS}", file=sys.stderr)
        return 2
    refuse_pointers(ASSETS)
    os.makedirs(options.into, exist_ok=True)

    path, size, sha = build(ASSETS, "forest-art", options.into)
    entry = {
        "key": f"{NAMESPACE}/art",
        "title": "Forest demo art",
        "url": URL % (options.tag, os.path.basename(path)),
        "directory": "art",
        "archive": "tar",
        "approximate_bytes": size,
        "sha256": sha,
        "base": True,
        "copyright": _credits(),
        "marker": MARKER,
        "notes": "The terrain, the trees and their impostors, and the plants "
                 "that grow between them. Fetched before the first frame, "
                 "because the demo opens its heightmap before it opens a "
                 "window.",
    }
    out = os.path.join("src", "openglcontext_forest_demo", "packs.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"namespace": NAMESPACE, "packs": [entry]}, handle, indent=1)
        handle.write("\n")
    print(f"forest art: {size / 1048576:.1f} MB, sha256 {sha[:12]}, "
          f"registry written to {out}")
    return 0


def _credits() -> str:
    """Whose the art is, from the file that already says so.

    One line, because a pack's terms are read on a screen with room for one;
    CREDITS.txt travels inside the pack for the rest.
    """
    return ("Tree models by Georgeous, 3D Error 404 and LOLIPOP (CC-BY 4.0); "
            "ground cover plants from Poly Haven and ground materials from "
            "ambientCG (both CC0 1.0); terrain and impostors generated for "
            "this demo. Full attribution in CREDITS.txt inside the pack.")


if __name__ == "__main__":
    raise SystemExit(main())
