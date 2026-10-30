"""Boulders thrown into the forest are bodies in the physics world that lands on the ground and strikes the trunks."""
import numpy as np
import pytest
from OpenGLContext.physics.landscape import Landscape
from OpenGLContext.physics.service import Physics
from OpenGLContext.physics.settings import PhysicsDefinition
from OpenGLContext.scenegraph.group import Group
from OpenGLContext.scenegraph.physicalmaterial import PhysicalMaterial
from OpenGLContext.scenegraph.terrain import HeightField

from openglcontext_forest_demo.boulders import Boulders


@pytest.fixture
def physics():
    service = Physics(PhysicsDefinition(mode='inline', maxCatchUp=2.0, tickHz=100.0))
    yield service
    service.close()


def run(service, seconds):
    for _ in range(int(seconds * 50)):
        service.frame(0.02)


def boulders(physics, **named):
    return Boulders(physics, Group(), PhysicalMaterial(), **named)


def test_a_thrown_boulder_is_a_node_in_the_scene_and_a_dynamic_body_in_the_world(physics):
    thrown = boulders(physics)
    body = thrown.throw((0.0, 10.0, 0.0), (1.0, 0.0, 0.0), speed=5.0)
    assert body.kind == 'dynamic'
    assert body.node in thrown.scene.children
    assert body.node.translation[1] == pytest.approx(10.0)
    physics.frame(0.02)
    assert body.velocity[0] == pytest.approx(5.0, abs=0.3)


def test_a_boulder_thrown_at_the_ground_comes_to_rest_on_it(physics):
    Landscape(physics, HeightField(np.ones((8, 8)), 400.0, 5.0))
    body = boulders(physics).throw((0.0, 8.0, 0.0), (1.0, -0.2, 0.0), speed=6.0)
    run(physics, 8.0)
    assert 5.0 < body.position[1] < 6.0
    assert body.speed < 0.5


def test_a_boulder_thrown_at_a_trunk_is_stopped_by_it(physics):
    Landscape(physics, HeightField(np.ones((8, 8)), 400.0, 5.0),
              np.array([[10.0, 0.0, 0.0]]), np.array([0.5]))
    body = boulders(physics).throw((0.0, 6.0, 0.0), (1.0, 0.0, 0.0), speed=12.0)
    run(physics, 4.0)
    assert body.position[0] < 10.0 - 0.5
    assert body.position[0] > 6.0


def test_a_boulder_thrown_past_the_trunk_goes_on(physics):
    Landscape(physics, HeightField(np.ones((8, 8)), 400.0, 5.0),
              np.array([[10.0, 0.0, 4.0]]), np.array([0.5]))
    body = boulders(physics).throw((0.0, 6.0, 0.0), (1.0, 0.0, 0.0), speed=12.0)
    run(physics, 3.0)
    assert body.position[0] > 12.0


def test_the_oldest_boulder_goes_when_there_are_too_many(physics):
    thrown = boulders(physics, limit=3)
    bodies = [thrown.throw((float(x), 10.0, 0.0), (0.0, 0.0, 1.0)) for x in range(5)]
    assert len(thrown) == 3
    assert [body.node in thrown.scene.children for body in bodies] == [False, False, True, True, True]
    assert not bodies[0].alive and bodies[4].alive


def test_boulders_differ_in_size_and_shape(physics):
    thrown = boulders(physics)
    first = thrown.throw((0.0, 10.0, 0.0), (1.0, 0.0, 0.0))
    second = thrown.throw((0.0, 10.0, 5.0), (1.0, 0.0, 0.0))
    assert first.proxy.params['radius'] != second.proxy.params['radius']


def test_throwing_in_a_direction_with_no_length_is_refused(physics):
    with pytest.raises(ValueError, match='direction'):
        boulders(physics).throw((0.0, 10.0, 0.0), (0.0, 0.0, 0.0))
