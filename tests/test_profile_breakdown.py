"""The profiling tool leaves out the subsystems it is asked to, by name."""

import importlib.util
import os
from types import SimpleNamespace

import pytest

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'tools', 'profile_breakdown.py')


@pytest.fixture(scope='module')
def tool():
    spec = importlib.util.spec_from_file_location('profile_breakdown', TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scene():
    rungs = [SimpleNamespace(cards=f'cards{one}', far_cards=f'far{one}',
                             clumps_near=None if one else 'near-clumps0',
                             clumps_far=f'far-clumps{one}') for one in (0, 1)]
    return SimpleNamespace(cover=SimpleNamespace(rungs=rungs),
                           impostors=[('impostor', 'cards')], near='trees',
                           terrain_geom='ground')


@pytest.mark.parametrize('name, nodes', [
    ('grass', ['cards0', 'far0', 'cards1', 'far1']),
    ('clumps', ['near-clumps0', 'far-clumps0', 'far-clumps1']),
    ('impostors', ['impostor']),
    ('near', ['trees']),
    ('terrain', ['ground']),
])
def test_each_subsystem_names_the_nodes_it_draws_with(tool, name, nodes):
    assert tool.subsystem_nodes(_scene(), {name}) == nodes


def test_a_scene_with_no_cover_has_no_grass_or_clumps(tool):
    scene = _scene()
    scene.cover = None
    assert tool.subsystem_nodes(scene, {'grass', 'clumps'}) == []


def test_every_subsystem_the_tool_documents_is_one_it_knows(tool):
    assert set(tool.SUBSYSTEMS) == {'grass', 'clumps', 'impostors', 'near', 'terrain'}
