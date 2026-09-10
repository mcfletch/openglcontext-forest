"""User-tunable configuration for the forest demo.

`ForestConfig` gathers every knob the demo exposes — terrain size, tree and grass
scatter density, LOD windows, camera frustum, spawn seed — with the shipped
defaults as field defaults. It is the single source of user config: the command
line (`build_arg_parser` / `config_from_args`) maps flags onto these fields,
replacing the older per-knob `os.environ` reads. The scene builder
(`openglcontext_forest_demo.scene.build_forest_scene`) takes one `ForestConfig`
and produces the whole terrain+forest+grass scene, so an alternate navigation
demo constructs its own config and gets the identical world.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, fields


@dataclass
class ForestConfig:
    """Every user-facing knob for the forest scene, with the shipped defaults.

    Defaults reproduce the demo exactly as it ships (same seeds, densities and LOD
    windows). Pass an instance to `build_forest_scene`; get one from the command
    line with `config_from_args`."""

    # --- terrain ---
    extent: float = 4096.0        # world span of the square terrain patch, metres
    relief: float = 450.0         # vertical relief mapped from the heightmap, metres
    res: int = 513                # heightfield grid resolution, samples per side

    # --- tree scatter ---
    tree_density: float = 0.09    # trees per m^2 before blue-noise thinning
    tree_radius: float = 2200.0   # tree scatter disc radius about the origin, metres
    spacing_base: float = 0.8     # blue-noise keep-out floor between trunks, metres
    spacing_per_m: float = 0.06   # extra keep-out radius per metre of tree height
    tree_scale_min: float = 0.75  # smallest per-tree scale multiplier
    tree_scale_max: float = 1.45  # largest per-tree scale multiplier

    # --- grass ---
    clump_radius: float = 30.0        # real-geometry grass-clump disc radius, metres
    grass_far_radius: float = 700.0   # coarse far-grass disc radius, metres
    grass_mid_density: float = 3.5    # mid grass billboards per m^2
    clump_density: float = 9.0        # real grass clumps per m^2
    clump_scale: float = 0.42         # grass-clump scale multiplier
    clump_length_samples: int = 4     # blade-length subdivisions baked per near clump
    clump_far_length_samples: int = 1  # coarse blade subdivisions for the distance-LOD far clumps
    grass_sun: float = 0.36           # flat sun term for grass billboards (matches clumps)
    grass_mid_scale: float = 0.42     # mid grass billboard scale multiplier
    grass_far_density: float = 0.09   # far grass billboards per m^2

    # --- LOD ---
    impostor_cone_deg: float = 75.0   # forward view-cone half-angle for impostor culling, degrees
    near_mesh_radius: float = 56.0    # radius the near-mesh trees follow the camera, metres

    # --- camera ---
    fov_deg: float = 62.0     # vertical field of view, degrees
    eye_height: float = 1.7   # camera height above the ground, metres
    near: float = 0.25        # near clip plane, metres
    far: float = 9000.0       # far clip plane, metres

    # --- spawn ---
    seed: int = 1             # RNG seed for the walkable spawn-point search

    # --- render quality ---
    # 'auto' measures the live frame rate and picks high/medium/low to hold ~60 fps
    # (integrated GPUs land on medium); high/medium/low freeze that choice so the
    # cost of each level is visible. The shipped defaults above ARE the 'high' look.
    quality: str = "auto"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the demo's command-line parser; every flag defaults to a `ForestConfig` field.

    Flag names are the dataclass field names with underscores as dashes (plus a few
    short aliases: `--trees`, `--fov`, `--impostor-cone`). Parsed into a namespace
    whose attribute names match the fields, so `config_from_args` can rebuild the
    dataclass directly."""
    d = ForestConfig()
    p = argparse.ArgumentParser(
        prog="oglc-forest",
        description="Walkable near-photoreal forest demo for OpenGLContext.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    t = p.add_argument_group("terrain")
    t.add_argument("--extent", type=float, default=d.extent,
                   help="world span of the square terrain patch (m)")
    t.add_argument("--relief", type=float, default=d.relief,
                   help="vertical relief mapped from the heightmap (m)")
    t.add_argument("--res", type=int, default=d.res,
                   help="heightfield grid resolution (samples per side)")

    tr = p.add_argument_group("trees")
    tr.add_argument("--tree-density", "--trees", dest="tree_density", type=float,
                    default=d.tree_density, help="trees per m^2 before blue-noise thinning")
    tr.add_argument("--tree-radius", dest="tree_radius", type=float, default=d.tree_radius,
                    help="tree scatter disc radius about the origin (m)")
    tr.add_argument("--spacing-base", dest="spacing_base", type=float, default=d.spacing_base,
                    help="blue-noise keep-out floor between trunks (m)")
    tr.add_argument("--spacing-per-m", dest="spacing_per_m", type=float, default=d.spacing_per_m,
                    help="extra keep-out radius per metre of tree height")
    tr.add_argument("--tree-scale-min", dest="tree_scale_min", type=float, default=d.tree_scale_min,
                    help="smallest per-tree scale multiplier")
    tr.add_argument("--tree-scale-max", dest="tree_scale_max", type=float, default=d.tree_scale_max,
                    help="largest per-tree scale multiplier")

    g = p.add_argument_group("grass")
    g.add_argument("--clump-radius", dest="clump_radius", type=float, default=d.clump_radius,
                   help="real-geometry grass-clump disc radius (m)")
    g.add_argument("--grass-far-radius", dest="grass_far_radius", type=float, default=d.grass_far_radius,
                   help="coarse far-grass disc radius (m)")
    g.add_argument("--grass-mid-density", dest="grass_mid_density", type=float, default=d.grass_mid_density,
                   help="mid grass billboards per m^2")
    g.add_argument("--clump-density", dest="clump_density", type=float, default=d.clump_density,
                   help="real grass clumps per m^2")
    g.add_argument("--clump-scale", dest="clump_scale", type=float, default=d.clump_scale,
                   help="grass-clump scale multiplier")
    g.add_argument("--clump-length-samples", dest="clump_length_samples", type=int,
                   default=d.clump_length_samples, help="blade-length subdivisions baked per near clump")
    g.add_argument("--clump-far-length-samples", dest="clump_far_length_samples", type=int,
                   default=d.clump_far_length_samples,
                   help="coarse blade subdivisions for the distance-LOD far clumps")
    g.add_argument("--grass-sun", dest="grass_sun", type=float, default=d.grass_sun,
                   help="flat sun term for grass billboards (matches the clump geometry)")
    g.add_argument("--grass-mid-scale", dest="grass_mid_scale", type=float, default=d.grass_mid_scale,
                   help="mid grass billboard scale multiplier")
    g.add_argument("--grass-far-density", dest="grass_far_density", type=float, default=d.grass_far_density,
                   help="far grass billboards per m^2")

    lod = p.add_argument_group("LOD")
    lod.add_argument("--impostor-cone-deg", "--impostor-cone", dest="impostor_cone_deg", type=float,
                     default=d.impostor_cone_deg,
                     help="forward view-cone half-angle for impostor culling (deg)")
    lod.add_argument("--near-mesh-radius", dest="near_mesh_radius", type=float, default=d.near_mesh_radius,
                     help="radius the near-mesh trees follow the camera (m)")

    c = p.add_argument_group("camera")
    c.add_argument("--fov-deg", "--fov", dest="fov_deg", type=float, default=d.fov_deg,
                   help="vertical field of view (deg)")
    c.add_argument("--eye-height", dest="eye_height", type=float, default=d.eye_height,
                   help="camera height above the ground (m)")
    c.add_argument("--near", type=float, default=d.near, help="near clip plane (m)")
    c.add_argument("--far", type=float, default=d.far, help="far clip plane (m)")

    p.add_argument("--seed", type=int, default=d.seed, help="RNG seed for the spawn-point search")

    # Ladder rungs kept in sync with quality.LADDER (imported lazily to avoid a cycle:
    # quality imports ForestConfig from here).
    p.add_argument("--quality",
                   choices=["auto", "high", "medhigh", "medium", "medlow", "low"],
                   default=d.quality,
                   help="render quality: auto measures fps to hold ~60; a rung name freezes it")
    return p


def config_from_args(argv: Sequence[str] | None = None) -> ForestConfig:
    """Parse `argv` (default `sys.argv`) into a `ForestConfig`.

    The parser's namespace attribute names match the dataclass fields, so this is a
    direct rebuild — no per-knob wiring to drift out of sync."""
    ns = build_arg_parser().parse_args(argv)
    names = {f.name for f in fields(ForestConfig)}
    return ForestConfig(**{k: v for k, v in vars(ns).items() if k in names})
