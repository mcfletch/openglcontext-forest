"""The art has to be the art, not a note saying where the art is.

`*.png`, `*.glb` and `*.npz` are in Git LFS. `actions/checkout` fetches
*pointers* unless it is told `lfs: true`, and a pointer is a 132-byte text file
that keeps the name and the extension of the thing it stands for. A tree built
from such a checkout looks right -- every file present, every path resolving --
and the demo stops at its first image with `UnidentifiedImageError`.

The content pack is built from this tree, so these tests check that every file
in it is what its name says. `release-assets.py` refuses pointers on the same
rule, through the engine's archive writer, before it writes an archive.
"""

import os

import pytest

try:
    from importlib import resources
    ASSETS = str(resources.files('openglcontext_forest_demo') / 'assets')
except (ImportError, ModuleNotFoundError):          # pragma: no cover
    ASSETS = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'src', 'openglcontext_forest_demo',
        'assets')

#: What a Git LFS pointer starts with. The whole file is three short lines.
POINTER = b'version https://git-lfs.github.com/spec/v1'

#: How big a pointer can be. Anything at or under this that is meant to be art
#: is worth looking at twice; the real files are hundreds of kilobytes upward.
POINTER_BYTES = 200


def assets(suffix):
    if not os.path.isdir(ASSETS):
        pytest.skip('no assets directory to check')
    return sorted(os.path.join(ASSETS, name) for name in os.listdir(ASSETS)
                  if name.endswith(suffix))


class TestNothingIsAPointer:
    @pytest.mark.parametrize('suffix', ['.png', '.glb', '.npz'])
    def test_no_asset_is_a_stand_in_for_itself(self, suffix) -> None:
        found = assets(suffix)
        assert found, f'no {suffix} assets at all'
        pointers = [path for path in found
                    if open(path, 'rb').read(len(POINTER)) == POINTER]
        named = ', '.join(os.path.basename(one) for one in pointers[:5])
        assert not pointers, (
            f'{len(pointers)} {suffix} files are Git LFS pointers rather than '
            f'content -- the checkout that built this did not fetch LFS: '
            f'{named}')

    @pytest.mark.parametrize('suffix', ['.png', '.glb', '.npz'])
    def test_nothing_is_suspiciously_small(self, suffix) -> None:
        """A pointer is 132 bytes; so is anything else that went wrong."""
        small = [path for path in assets(suffix)
                 if os.path.getsize(path) <= POINTER_BYTES]
        assert not small, [os.path.basename(one) for one in small]


class TestTheImagesAreImages:
    """The check the crash would have needed: a file that opens."""

    def test_every_png_can_be_opened(self) -> None:
        Image = pytest.importorskip('PIL.Image')
        for path in assets('.png'):
            with Image.open(path) as im:
                im.load()

    def test_the_heightmap_is_the_one_the_scene_asks_for(self) -> None:
        """The first thing the demo opens, and where it died."""
        Image = pytest.importorskip('PIL.Image')
        where = os.path.join(ASSETS, 'forest_height.png')
        assert os.path.isfile(where)
        with Image.open(where) as im:
            assert im.size == (1024, 1024)
            assert im.mode == 'I;16', 'the terrain wants 16-bit height'
