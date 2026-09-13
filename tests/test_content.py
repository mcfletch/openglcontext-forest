"""Where the demo's art comes from, and what a first run has to fetch.

66 MB of terrain, trees and impostors does not travel in the wheel: it is a
base pack fetched through the engine's content facility. This is the demo's
half -- which registry, which namespace, and the rule that a fetched pack wins
over the copy in the wheel while there is one.
"""

import os

import pytest

from openglcontext_forest_demo import content


@pytest.fixture
def store(tmp_path):
    return content.store(root=str(tmp_path / "content"))


class TestTheRegistry:
    def test_it_loads(self) -> None:
        assert content.registry()

    def test_it_offers_one_pack_and_it_is_the_base(self) -> None:
        """Everything the demo draws is in it; there is nothing to choose."""
        packs = content.registry()
        assert [one.key for one in packs] == [content.BASE]
        assert packs[0].base

    def test_it_carries_a_digest(self) -> None:
        """Ours to publish, so the bytes are known and are checked. The last
        release shipped 132-byte stand-ins wearing the right names."""
        assert len(content.registry()[0].sha256) == 64

    def test_it_states_whose_the_art_is(self) -> None:
        """CC-BY tree models: a pack that cannot say so is one redistributed
        unattributed."""
        assert "CC-BY" in content.registry()[0].copyright

    def test_its_size_is_the_archive_s(self) -> None:
        """`approximate_bytes` caps the fetch; a wrong one fails a download on
        its last megabyte."""
        assert content.registry()[0].approximate_bytes > 10 * 1024 * 1024

    def test_its_marker_is_the_first_thing_the_scene_opens(self) -> None:
        from openglcontext_forest_demo import scene
        assert content.registry()[0].marker == os.path.basename(scene.HEIGHTMAP)


class TestWhatAFirstRunNeeds:
    def test_it_asks_for_the_art(self, store) -> None:
        assert [one.key for one in content.needed_to_start(store)] \
            == [content.BASE]

    def test_and_stops_once_it_is_here(self, store) -> None:
        self.unpack(store)
        assert content.needed_to_start(store) == []

    def test_the_art_is_the_wheel_s_until_the_pack_is_here(self, store) -> None:
        where = content.art_directory(store)
        assert where == content.IN_WHEEL

    def test_and_the_pack_s_once_it_is(self, store) -> None:
        root = self.unpack(store)
        assert content.art_directory(store) == root

    def unpack(self, store):
        pack = content.registry()[0]
        root = store.directory_for(pack)
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, pack.marker), "wb") as handle:
            handle.write(b"\x89PNG\r\n\x1a\n")
        return root
