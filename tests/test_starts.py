"""The demo starts: its scene is built, its walker is bound and frames are drawn.

The other tests in this directory build the pieces of the demo from a handful
of trunks and a toy height field. Nothing in them reaches the place where the
whole forest meets the physics world, so a world too small for the forest's
trunks was found by running the demo. This test runs it.
"""

import os
import subprocess
import sys

import pytest
from OpenGLContext.testing.glcontext import gl_available

#: The frames the demo draws before it exits.
FRAMES = 5

#: What the console says when the demo stops because its art is not installed.
NO_ART = ("Not downloading", "did not download")


@pytest.fixture(scope="module")
def started() -> subprocess.CompletedProcess[str]:
    """The demo run for a few frames in a hidden window; its exit status and everything it printed."""
    if not gl_available():
        pytest.skip("no GL context can be made here")
    environment = dict(os.environ, OPENGLCONTEXT_HIDDEN="1",
                       OPENGLCONTEXT_AUTO_EXIT_FRAMES=str(FRAMES))
    done = subprocess.run(
        [sys.executable, "-m", "openglcontext_forest_demo.run"],
        env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, timeout=240, check=False)
    if any(phrase in done.stdout for phrase in NO_ART):
        pytest.skip("the demo's art is not installed on this machine")
    return done


def test_it_runs_its_frames_and_exits(started) -> None:
    assert started.returncode == 0, started.stdout[-3000:]


def test_it_raises_nothing_on_the_way(started) -> None:
    assert "Traceback" not in started.stdout, started.stdout[-3000:]


def test_it_builds_the_forest(started) -> None:
    """The forest is more trees than a physics world holds bodies."""
    assert " trees " in started.stdout, started.stdout[-3000:]
