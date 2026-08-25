"""The demo's screens and its controls, with no GL context.

Building a panel and declaring a movement mode touch no GL, so everything a
test needs to know about the interface -- which buttons a screen has, what each
one does, which keys open what, and that the demo starts in mouse-look -- is
checked here rather than against a window.
"""
import pathlib

import pytest
from OpenGLContext.move import modes as movemodes

from openglcontext_forest_demo import menu, run
from openglcontext_forest_demo.config import ForestConfig

# -- how the demo moves -------------------------------------------------------

def test_the_demo_starts_in_mouse_look():
    """The navigation manager takes the first selectable mode, so first is what
    a launch lands in -- and a forest is a place you look around."""
    declared = run.movement_modes()
    assert isinstance(declared[0], movemodes.FPSMode)
    assert declared[0].capturePointer


def test_walking_and_flying_are_offered_too():
    assert [mode.name for mode in run.movement_modes()] == ['fps', 'walk', 'fly']


def test_mouse_look_and_walking_move_at_the_same_speed():
    """Taking the pointer changes how you steer, not how fast you go."""
    fps, walk, fly = run.movement_modes()
    assert fps.walkSpeed == pytest.approx(walk.walkSpeed)
    assert fps.runSpeed == pytest.approx(walk.runSpeed)
    assert fly.flySpeed > walk.runSpeed          # flying is for covering ground


def test_the_avatar_is_the_height_the_config_asks_for():
    """The scene is framed for a camera at ``eye_height``, so the avatar's eye
    goes there rather than to the physics default."""
    demo = run.Forest.__new__(run.Forest)
    demo.eye_height = ForestConfig().eye_height
    capabilities = demo.characterCapabilities(1.0)
    assert capabilities.eyeHeight == pytest.approx(ForestConfig().eye_height)
    assert capabilities.standHeight > capabilities.eyeHeight


# -- the keys that open a screen ---------------------------------------------

class _Recorder:
    """Stands in for the context the handlers are bound on."""

    def __init__(self):
        self.bound = []

    def addEventHandler(self, kind, name=None, function=None, **named):
        self.bound.append((kind, name, named.get('state'), function))


def _bindings():
    demo = run.Forest.__new__(run.Forest)
    recorder = _Recorder()
    demo.bindScreenKeys(recorder)
    return demo, recorder


def test_the_screen_keys_are_the_ones_every_other_program_here_uses():
    demo, recorder = _bindings()
    bound = {name: function for _kind, name, _state, function in recorder.bound}
    assert bound['<F6>'] == demo.showBindings
    assert bound['<F10>'] == demo.showSettings
    assert bound['m'] == demo.cycleMovementMode


def test_the_screen_keys_are_bound_as_key_downs():
    """A function key produces no character, so a keypress binding for one is
    accepted and then never fires."""
    _demo, recorder = _bindings()
    assert recorder.bound
    for kind, _name, state, _function in recorder.bound:
        assert (kind, state) == ('keyboard', 1)


def test_the_screenshot_key_is_the_one_every_context_has():
    """F2 comes from the engine; the demo binds nothing of its own for it."""
    assert run.Forest.screenshotKey == '<F2>'


def test_the_shot_is_named_for_the_demo():
    """The window title names the file, so a picture folder says which took it."""
    from types import SimpleNamespace

    from OpenGLContext.screenshot import ScreenshotMixin

    named = ScreenshotMixin()
    named.contextDefinition = SimpleNamespace(title=menu.TITLE)
    assert 'forest' in named.screenshotTitle()


# -- the menu -----------------------------------------------------------------

def _names(panel):
    return [str(widget.name) for widget in panel.walk()
            if str(widget.name)]


def test_the_menu_offers_resume_first_and_quit_last():
    """There is always a forest behind this screen, so leaving it is what
    Escape means and quitting is what the button is for."""
    panel = menu.main_menu(on_resume=lambda: None)
    offered = [name for name in _names(panel)
               if name in ('resume', 'bindings', 'settings', 'credits', 'quit')]
    assert offered == ['resume', 'bindings', 'settings', 'credits', 'quit']


def test_the_menu_buttons_call_what_they_were_given():
    called = []
    panel = menu.main_menu(
        on_resume=lambda: called.append('resume'),
        on_bindings=lambda: called.append('bindings'),
        on_settings=lambda: called.append('settings'),
        on_credits=lambda: called.append('credits'),
        on_quit=lambda: called.append('quit'))
    for name in ('resume', 'bindings', 'settings', 'credits', 'quit'):
        panel.find(name).on_activate(panel.find(name))
    assert called == ['resume', 'bindings', 'settings', 'credits', 'quit']


def test_escaping_the_menu_resumes_rather_than_quitting():
    called = []
    panel = menu.main_menu(on_resume=lambda: called.append('resume'))
    panel.on_close(panel)
    assert called == ['resume']


def test_the_menu_says_what_is_behind_it():
    panel = menu.main_menu(subtitle='230161 trees')
    assert any('230161' in str(widget.text)
               for widget in panel.walk() if hasattr(widget, 'text'))


def test_the_menu_is_pushed_under_a_name_so_it_cannot_stack():
    assert menu.main_menu().name == menu.MENU_NAME
    assert menu.credits_screen('anything').name == menu.CREDITS_NAME


# -- the attribution ----------------------------------------------------------

#: Every work the demo has to name, by author, so a dropped attribution shows up
#: as a failure rather than as a quietly shorter screen.
CREDITED_WORKS = [
    ('Fir tree', 'Georgeous'),
    ('Noel_Pine_Tree', '3D Error 404'),
    ('Maple trees pack', 'LOLIPOP'),
    ('Realistic Trees Collection', 'Jungle Jim'),
    ('Low Poly Forest Tree Pack', '99.Miles'),
]


def test_the_credits_file_ships_with_the_package():
    """The notice is read from CREDITS.txt at runtime, so a wheel that dropped it
    would show a screen with no attribution on it at all."""
    assert run.credits_path().is_file()


def test_every_tree_model_is_credited_in_the_notice():
    """CC-BY asks for attribution wherever the work appears, and the launch
    notice and the credits screen are one text so neither can fall behind."""
    text = run.credits_text()
    for title, author in CREDITED_WORKS:
        assert title in text, title
        assert author in text, author
    assert 'creativecommons.org/licenses/by/4.0' in text


def test_the_notice_is_the_credits_file():
    """One source, so an attribution cannot be added in one place and missed in
    the other."""
    assert run.credits_text().strip() == run.credits_path().read_text(
        encoding='utf-8').strip()


def test_attribution_survives_a_missing_credits_file(monkeypatch):
    """A packaging mistake must not silently drop the attribution: the fallback
    still names every work, its author and the licence."""
    monkeypatch.setattr(run, 'credits_path',
                        lambda: pathlib.Path('/nonexistent/CREDITS.txt'))
    text = run.credits_text()
    for title, author in CREDITED_WORKS:
        assert title in text, title
        assert author in text, author
    assert 'creativecommons.org/licenses/by/4.0' in text


def test_the_credits_screen_carries_the_same_notice():
    text = run.credits_text()
    panel = menu.credits_screen(text)
    assert any(str(getattr(widget, 'text', '')) == text
               for widget in panel.walk())


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
