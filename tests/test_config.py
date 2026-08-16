"""Config/CLI and reusable-API tests that need no GL context.

Covers the argparse round-trip, the documented `ForestConfig` defaults, and that
the reusable scene API imports and the parser accepts an empty argv. A synthetic
`build_forest_scene` smoke build is attempted but skipped if assets/deps make it
impossible — it must never require a real GL context."""
import dataclasses

import pytest

import openglcontext_forest_demo as fd
from openglcontext_forest_demo.config import (
    ForestConfig,
    build_arg_parser,
    config_from_args,
)


def test_reusable_api_is_importable():
    # Everything a driving demo would import off the top-level package.
    assert fd.build_forest_scene is not None
    assert fd.ForestScene is not None
    assert fd.ForestConfig is ForestConfig
    for name in ("build_forest_scene", "ForestScene", "ForestConfig",
                 "config_from_args", "build_arg_parser", "Forest", "main"):
        assert name in fd.__all__


def test_parser_accepts_empty_argv():
    # The smoke check the task calls out: build_arg_parser().parse_args([]) must work.
    ns = build_arg_parser().parse_args([])
    assert ns.tree_density == ForestConfig().tree_density


def test_defaults_match_documented_values():
    c = ForestConfig()
    expected = {
        "extent": 4096.0, "relief": 450.0, "res": 513,
        "tree_density": 0.09, "tree_radius": 2200.0, "spacing_base": 0.8, "spacing_per_m": 0.06,
        "tree_scale_min": 0.75, "tree_scale_max": 1.45,
        "clump_radius": 30.0, "grass_far_radius": 700.0, "grass_mid_density": 3.5, "clump_density": 9.0,
        "clump_scale": 0.42, "clump_length_samples": 4, "clump_far_length_samples": 1,
        "grass_sun": 0.36, "grass_mid_scale": 0.42,
        "grass_far_density": 0.09,
        "impostor_cone_deg": 75.0, "near_mesh_radius": 56.0,
        "fov_deg": 62.0, "eye_height": 1.7, "near": 0.25, "far": 9000.0,
        "seed": 1,
        "quality": "auto",
    }
    got = {f.name: getattr(c, f.name) for f in dataclasses.fields(c)}
    assert got == expected


def test_config_from_args_defaults_round_trip():
    # No flags -> the dataclass defaults, byte for byte.
    assert config_from_args([]) == ForestConfig()


def test_config_from_args_overrides():
    cfg = config_from_args([
        "--clump-radius", "45",
        "--grass-far-radius", "1200",
        "--tree-density", "0.05",
        "--fov", "70",
        "--seed", "99",
        "--clump-length-samples", "6",
    ])
    assert cfg.clump_radius == 45.0
    assert cfg.grass_far_radius == 1200.0
    assert cfg.tree_density == 0.05
    assert cfg.fov_deg == 70.0
    assert cfg.seed == 99
    assert cfg.clump_length_samples == 6
    # untouched knobs keep their defaults
    assert cfg.relief == ForestConfig().relief
    assert cfg.near_mesh_radius == ForestConfig().near_mesh_radius


def test_alias_flags():
    # --trees is an alias for --tree-density; --impostor-cone for --impostor-cone-deg.
    assert config_from_args(["--trees", "0.2"]).tree_density == 0.2
    assert config_from_args(["--impostor-cone", "60"]).impostor_cone_deg == 60.0


def test_int_flags_are_ints():
    cfg = config_from_args(["--res", "257", "--clump-length-samples", "3", "--seed", "7"])
    assert isinstance(cfg.res, int) and cfg.res == 257
    assert isinstance(cfg.clump_length_samples, int) and cfg.clump_length_samples == 3
    assert isinstance(cfg.seed, int) and cfg.seed == 7


def test_build_forest_scene_smoke():
    """GL-free smoke build on a tiny config. Skipped if assets/deps are unavailable.

    build_forest_scene only makes scenegraph nodes + numpy data (GL uploads lazily on
    render), so this must run with no context. A small tree_radius keeps it fast."""
    import os
    try:
        from openglcontext_forest_demo.scene import (
            CONTROL,
            HEIGHTMAP,
            ForestScene,
            build_forest_scene,
        )
    except Exception as exc:  # noqa: BLE001 -- any import-time GL/deps problem -> skip
        pytest.skip(f"scene import unavailable: {exc!r}")
    if not (os.path.exists(HEIGHTMAP) and os.path.exists(CONTROL)):
        pytest.skip("terrain assets not bundled")
    cfg = ForestConfig(res=129, tree_radius=60.0, grass_far_radius=80.0)
    try:
        scene = build_forest_scene(cfg)
    except Exception as exc:  # noqa: BLE001 -- missing runtime resources here -> skip
        pytest.skip(f"build_forest_scene needs resources unavailable here: {exc!r}")
    assert isinstance(scene, ForestScene)
    assert scene.hf is not None
    assert scene.sceneGraph is not None
    assert scene.spawn is not None and len(scene.spawn) == 2
    assert scene.collider_pos.shape[0] == scene.collider_radius.shape[0]
    # streamers stage numpy data without touching GL
    ex, ez = scene.spawn
    scene.stream_near(ex, ez)
    scene.stream_far(ex, ez)
    scene.stream_impostors(ex, ez, 1.0, 0.0)
