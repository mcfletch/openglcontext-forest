"""Where the demo's art comes from, and what a first run has to fetch.

66 MB of terrain, trees and impostors does not travel in the wheel: it is a
base pack fetched through the engine's content facility. This is the demo's
half -- which registry, which namespace, and the rule that a fetched pack wins
over the copy in the wheel while there is one.
"""

import os

import pytest
from OpenGLContext import userpaths
from OpenGLContext.loaders import resolver

from openglcontext_forest_demo import content, run, scene


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


class TestAPackFetchedAfterImport:
    """A first run imports the scene before the art is here, then fetches it.

    Whatever the scene opens is read from where the art is when it is opened,
    not from where it was when the module was imported.
    """

    def test_the_cover_is_read_from_the_pack(self, tmp_path,
                                             monkeypatch) -> None:
        pack = content.registry()[0]
        root = tmp_path / "local" / content.NAMESPACE / pack.directory
        root.mkdir(parents=True)
        (root / pack.marker).write_bytes(b"\x89PNG\r\n\x1a\n")
        (root / "cover.json").write_text(
            '{"species": [{"name": "only_one", "card": "card.png"}]}',
            encoding="utf-8")
        monkeypatch.setenv("OPENGLCONTEXT_CONTENT", str(tmp_path / "local"))
        species = scene.cover_species()
        assert [one.name for one in species] == ["only_one"]
        assert species[0].card == str(root / "card.png")


class TestTheFirstRunOnTheConsole:
    """The art is fetched before the window opens, so the console is where the
    player is asked, and where a failure is explained."""

    @pytest.fixture
    def nothing_here(self, tmp_path, monkeypatch):
        monkeypatch.setattr(userpaths, "appdatadirectory",
                            lambda: str(tmp_path / "appdata"))
        monkeypatch.delenv("OPENGLCONTEXT_CONTENT", raising=False)
        fetched = []

        def offline(url, *_args, **_named):
            fetched.append(url)
            raise OSError("no route to host")
        monkeypatch.setattr(resolver, "fetch_to_cache", offline)
        return fetched

    def test_the_player_is_asked_and_no_fetches_nothing(
            self, nothing_here, monkeypatch, capsys) -> None:
        monkeypatch.setattr("builtins.input", lambda _prompt: "n")
        with pytest.raises(SystemExit) as stopped:
            run.fetch_art()
        assert nothing_here == []
        assert "OPENGLCONTEXT_CONTENT" in str(stopped.value)
        said = capsys.readouterr().out
        assert "Forest demo art" in said and "CC-BY" in said

    def test_a_failed_download_is_explained_rather_than_a_traceback(
            self, nothing_here, monkeypatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _prompt: "y")
        with pytest.raises(SystemExit) as stopped:
            run.fetch_art()
        assert nothing_here, "it tried"
        message = str(stopped.value)
        assert "no route to host" in message
        assert "OPENGLCONTEXT_CONTENT" in message

    def test_yes_on_the_command_line_does_not_ask(
            self, nothing_here, monkeypatch) -> None:
        def never(_prompt):
            raise AssertionError("asked")
        monkeypatch.setattr("builtins.input", never)
        with pytest.raises(SystemExit):
            run.fetch_art(assume_yes=True)
        assert nothing_here, "it went ahead"
