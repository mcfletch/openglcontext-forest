"""The screens around the outside of the walk.

Escape puts :func:`main_menu` up: Resume, the controls, the rendering settings,
the asset credits and Quit. Everything here is a plain
:class:`~OpenGLContext.ui.panel.Panel` built from OpenGLContext's shared widgets,
so it takes the player's interface scale, the skin and the input routing that
every other screen in the project takes -- and can be checked with no window at
all, since building a panel touches no GL.

Settings and key bindings are not reimplemented: they are
:func:`OpenGLContext.ui.settings.open_settings` and
:func:`OpenGLContext.ui.bindings.open_bindings`, the same pages ``oglc-view``
and ``twitch`` show.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional

from OpenGLContext.ui import dialogs
from OpenGLContext.ui.layout import Column
from OpenGLContext.ui.panel import Panel
from OpenGLContext.ui.widgets import Button, Label, Separator

__all__ = ['MENU_NAME', 'CREDITS_NAME', 'TITLE', 'main_menu', 'credits_screen']

#: What the demo calls itself on its own front screen.
TITLE = 'OpenGLContext forest demo'

#: Names the screens are pushed under, so a second press of a key raises the one
#: already up rather than stacking another over it.
MENU_NAME = 'forest-menu'
CREDITS_NAME = 'forest-credits'

#: How wide the menu is, in characters.
MENU_COLUMNS = 44

#: How wide the credits are.  Wider, because the notices are attribution lines
#: with URLs in them and wrapping those makes them harder to copy than to read.
CREDITS_COLUMNS = 78


def main_menu(on_resume: Optional[Callable[[], None]] = None,
              on_bindings: Optional[Callable[[], None]] = None,
              on_settings: Optional[Callable[[], None]] = None,
              on_credits: Optional[Callable[[], None]] = None,
              on_quit: Optional[Callable[[], None]] = None,
              subtitle: str = '') -> Panel:
    """The screen Escape brings up, with the walk still behind it.

    Resume is offered first and Escape leaves the screen, because there is
    always a forest to go back to: somebody who pressed Escape meaning "close
    this" must never find they have quit instead.
    """
    children: List[Any] = [Label(text=TITLE, name='title')]
    if subtitle:
        children.append(Label(text=subtitle, wrap=True, name='subtitle'))
    children.append(Separator(top=6))
    for name, text, handler, primary in (
            ('resume', 'Resume', on_resume, True),
            ('bindings', 'Controls', on_bindings, False),
            ('settings', 'Settings', on_settings, False),
            ('credits', 'Asset credits', on_credits, False),
            ('quit', 'Quit', on_quit, False)):
        button = Button(text=text, name=name, role='primary' if primary else '')
        if handler is not None:
            button.on_activate = lambda _widget, call=handler: call()
        children.append(button)
    panel = Panel(title='', name=MENU_NAME, scrim=True, modal=True,
                  preferredColumns=MENU_COLUMNS,
                  children=[Column(children=children, spacing=4)])
    if on_resume is not None:
        # Escaping out of the menu means what the Resume button means.
        panel.on_close = lambda _closing: on_resume()
    return panel


def credits_screen(text: str) -> Panel:
    """The attribution the tree models are licensed on, where it can be read.

    The same notices the demo prints on launch.  CC-BY asks for attribution
    wherever the work appears, and a line that scrolled past in a terminal
    before the window opened is not somewhere anyone can find it.
    """
    panel = dialogs.notice('Asset credits', text, columns=CREDITS_COLUMNS)
    panel.name = CREDITS_NAME
    return panel
