"""Reusable construction of the forest scene.

`build_forest_scene(config)` builds the runtime-splat terrain, the GPU-instanced
forest and the grass LOD chain from one `ForestConfig`, and returns a
`ForestScene` bundling everything a navigation layer needs: the scene graph to
render, the height field to clamp and collide against, the per-tree collider, and
the camera-following streamers. The build creates scenegraph nodes and numpy data
only — GL buffers upload lazily on first render — so a caller constructs it inside
its own context's `OnInit`, adds `scene.sceneGraph`, sets its frustum from the
config, and registers `scene.stream_near` / `stream_far` / `stream_impostors` on
its move loop. The walking demo (`run.py`) and a future driving demo share the
exact same world this way.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from OpenGLContext.loaders.tiles3d import vegetation
from OpenGLContext.scenegraph.terrain import HeightField, SplatTerrain
from OpenGLContext.scenegraph.vegetation import (
    InstancedBillboards, InstancedMeshLOD, InstancedClumps, load_clump_glb, world_grid_scatter)
from OpenGLContext.scenegraph.basenodes import (
    sceneGraph, Background, Shape, Appearance, Material)

from openglcontext_forest_demo.config import ForestConfig

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
HEIGHTMAP = os.path.join(ASSETS, "forest_height.png")
CONTROL = os.path.join(ASSETS, "forest_control.png")
CLUMP_GLB = os.path.join(ASSETS, "basic-clump.glb")
LAYERS = ["forest_floor", "grass", "rock", "moss"]

# Fixed handoff geometry that keeps the LOD cross-fades seamless — not user knobs:
# nudging these desyncs the geometry->billboard blend, so they stay constants.
TREE_SCATTER_SEED = 7          # tree layout seed (holds the shipped forest arrangement)
BIOME_SEED = 11                # conifer/maple/deciduous mix seed
MID_GRASS_FADE = 95.0          # mid grass billboard disc + fade-out distance, metres
FAR_GRASS_NEAR_CUT = 90.0      # far grass fades IN here, just inside the mid fade-out
FAR_GRASS_SCALE_MUL = 1.5      # far grass billboards run larger to stay readable at range
IMPOSTOR_NEAR_DISC = 60.0      # always-drawn impostor disc so nothing pops at the feet on a spin
GRASS_BILLBOARD_WIDTH = 2.6    # grass card width (matches the baked clump impostor aspect)
CLUMP_SUN = (-0.5, -0.72, -0.48)   # per-fragment sun direction for the clump geometry

# Per-species trunk-base sink so root flares/buttresses embed (fir flare, real3 buttress).
# Indexed by near-mesh species id (0 fir, 1 noel, 2-5 maple, 6-12 realistic).
SINK = np.array([0.10, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.17, 0.02, 0.02, 0.02], np.float32)


def A(name):
    """Path to a bundled demo asset."""
    return os.path.join(ASSETS, name)


def _shape(geometry):
    return Shape(geometry=geometry, appearance=Appearance(material=Material()))


def _trunk_metrics(npz_path, part_key):
    """Trunk axis (cx, cz) and core radius for a species mesh, in local units.

    Measured over a mid-height band (above the ground flare/buttress roots, below the
    crown) so the collider lands on the pole the player actually bumps, not on the
    root skirt. Radius is a low percentile of the band's distance-to-axis (the dense
    trunk core, excluding splayed limbs). Multiply by the per-tree height to get world
    units. Returns (cx, cz, radius)."""
    P = np.load(npz_path)[part_key]
    y = P[:, 1]; y0 = y.min(); h = np.ptp(y) or 1.0
    band = P[(y > y0 + 0.15 * h) & (y < y0 + 0.45 * h)]
    if len(band) < 8:
        band = P
    cx = float(np.median(band[:, 0])); cz = float(np.median(band[:, 2]))
    r = float(np.percentile(np.hypot(band[:, 0] - cx, band[:, 2] - cz), 20))
    return cx, cz, r


@dataclass
class ForestScene:
    """Everything a navigation layer needs to render, walk and stream the forest.

    Built by `build_forest_scene`. `sceneGraph` goes into the context; `hf` clamps
    and collides the camera; `stream_near`/`stream_far`/`stream_impostors` follow the
    camera each move. The same instance drives the walking demo and a driving demo —
    only the navigation on top differs."""

    config: ForestConfig
    sceneGraph: object
    hf: HeightField
    terrain_geom: SplatTerrain
    near: InstancedMeshLOD
    # (node, positions, yaws, scales) per impostor species, for view-cone culling
    impostors: list
    grass: InstancedBillboards
    grass_far: InstancedBillboards
    clumps: object                   # InstancedClumps, or None if the clump asset is missing
    grass_mask: Callable
    clump_radius: float
    grass_far_radius: float
    impostor_cos: float
    near_mesh_radius: float
    collider_pos: np.ndarray
    collider_radius: np.ndarray
    near_species: list
    spawn: tuple                     # (ex, ez) walkable spawn point

    # --- reusable camera-following streamers (register on the move loop) ---

    def stream_near(self, x, z):
        """Restream the near-field: mid grass billboards, near tree meshes, real clumps."""
        cfg = self.config
        # mid grass billboards: short like the clumps, dense enough to overlap into
        # continuous cover so the geometry->billboard band doesn't read as sparse tufts.
        p, y, s = world_grid_scatter(x, z, MID_GRASS_FADE, cfg.grass_mid_density, self.hf,
                                     scale_mul=cfg.grass_mid_scale, mask=self.grass_mask)
        self.grass.update_instances(p, y, s)
        self.near.update(x, z, radius=self.near_mesh_radius)   # near meshes cover the LOD band
        if self.clumps is not None:
            # real-geometry clumps: dense, short, small radius (they are heavy)
            pc, yc, sc = world_grid_scatter(x, z, self.clump_radius, cfg.clump_density, self.hf,
                                            scale_mul=cfg.clump_scale, mask=self.grass_mask)
            self.clumps.update_instances(pc, yc, sc)

    def stream_far(self, x, z):
        """Restream the coarse far-grass disc that follows the camera out to the horizon."""
        p, y, s = world_grid_scatter(x, z, self.grass_far_radius, self.config.grass_far_density,
                                     self.hf, scale_mul=FAR_GRASS_SCALE_MUL, mask=self.grass_mask)
        self.grass_far.update_instances(p, y, s)

    def stream_impostors(self, x, z, fx, fz):
        """Submit only the impostor cards inside the forward view cone `(fx, fz)`.

        A forward cone (dot > 0, dot^2 > cos^2 * dist^2 tests the angle without a
        per-tree sqrt) plus a small always-drawn near disc. Trees behind and beside
        the camera — roughly half the forest — never reach the GPU. Callers pass the
        camera's ground-plane forward vector so this stays independent of any one
        navigation scheme."""
        cc = self.impostor_cos * self.impostor_cos
        near2 = IMPOSTOR_NEAR_DISC * IMPOSTOR_NEAR_DISC
        for node, P, Y, S in self.impostors:
            dx = P[:, 0] - x; dz = P[:, 2] - z
            d2 = dx * dx + dz * dz
            dot = dx * fx + dz * fz
            keep = (d2 < near2) | ((dot > 0.0) & (dot * dot > cc * d2))
            node.update_instances(P[keep], Y[keep], S[keep])


def build_forest_scene(config: ForestConfig) -> ForestScene:
    """Build the terrain + instanced forest + grass LOD chain from `config`.

    Creates scenegraph nodes and numpy instance data only (GL uploads on first
    render), so call this inside a GL context's `OnInit`. Returns a `ForestScene`
    the caller renders and streams; the concrete navigation lives on top."""
    hf = HeightField.from_image(HEIGHTMAP, config.res, config.extent, config.relief)
    hfn = hf.sample
    terrain_geom = SplatTerrain(hf, LAYERS, CONTROL)
    terrain = _shape(terrain_geom)

    # Grass grows on any soft ground (forest floor, meadow, moss) but NOT on rock:
    # scatter is thinned by (1 - rock weight) from the control map, so it stays lush
    # where you walk yet vanishes on bare mountainsides instead of dotting them with
    # green blobs. Reuses the HeightField bilinear sampler over the terrain extent.
    from PIL import Image
    _ctl = np.asarray(Image.open(CONTROL).convert("RGBA").resize((config.res, config.res), Image.LANCZOS),
                      np.float32) / 255.0
    _grass_allowed = np.clip(1.0 - _ctl[..., LAYERS.index("rock")], 0.0, 1.0)
    _rock_mask = HeightField(_grass_allowed, config.extent, 1.0).sample

    def grass_mask(px, pz):
        # thin grass out on steep ground: on a slope a wide clump's uphill side
        # buries under the terrain (only blade tips show), and steep rock faces
        # shouldn't be grassed anyway. slope is rise/run; full grass below ~29deg,
        # gone by ~49deg.
        steep = 1.0 - np.clip((hf.slope(px, pz) - 0.55) / 0.60, 0.0, 1.0)
        return _rock_mask(px, pz) * steep

    # spawn on a gentle, walkable spot (so we stand among trees, not on a cliff)
    ex, ez = -200.0, 300.0
    rng = np.random.default_rng(config.seed); bs = 9e9
    for _ in range(800):
        x = rng.uniform(-700, 700); z = rng.uniform(-700, 700)
        sl = float(hf.slope(x, z))
        if sl < bs:
            bs = sl; ex, ez = float(x), float(z)
        if bs < 0.05:
            break

    # dense forest across gentle-to-moderate slopes (conifers tolerate steeper ground)
    def gentle(p):
        x = p[:, 0]; z = p[:, 2]
        clear = np.hypot(x - ex, z - ez) > 13.0     # small clearing at the spawn
        return (hf.slope(x, z) < 0.72) & clear
    trees = vegetation.scatter_disc((0, 0, 0), config.tree_radius, config.tree_density,
                                    TREE_SCATTER_SEED, hfn, keep=gentle,
                                    scale_range=(config.tree_scale_min, config.tree_scale_max))
    tp = trees.positions; N = len(tp); yaws = trees.yaws; sc = trees.scales

    # --- biome mix: conifers high/steep, maples in valleys, realistic deciduous
    #     between, with coarse spatial noise so each area has a dominant type plus
    #     a minority mix ---
    elev = np.clip(tp[:, 1] / config.relief, 0, 1)
    slope = np.clip(hf.slope(tp[:, 0], tp[:, 2], eps=10.0), 0, 1.5)
    nz = np.sin(tp[:, 0] * 0.004 + 1.3) * np.cos(tp[:, 2] * 0.0035) + 0.5 * np.sin(tp[:, 2] * 0.010)
    nz = (nz - nz.min()) / (nz.max() - nz.min() + 1e-6)
    rng3 = np.random.default_rng(BIOME_SEED)
    fir = rng3.random(N) < np.clip(elev * 1.4 + slope * 1.2 + (nz - 0.5) * 1.1 - 0.28, 0.03, 0.97)
    maple = (~fir) & (rng3.random(N) < np.clip((0.55 - elev) * 1.7 + (0.5 - nz) * 0.7, 0.15, 0.9))
    realistic = (~fir) & (~maple)
    heights = (sc * np.where(fir, 16.0, 12.5)).astype(np.float32)
    noel = fir & (rng3.random(N) < 0.45); fir_only = fir & ~noel

    # De-clump: a uniform scatter piles trees into branch soup. Thin to blue-noise
    # so no two survivors are closer than their combined keep-out radius (which
    # grows with tree height), and the biggest tree in a crowd wins the spot.
    keep = vegetation.poisson_thin(tp, config.spacing_base + config.spacing_per_m * heights,
                                   priority=heights)
    tp = tp[keep]; yaws = yaws[keep]; sc = sc[keep]; heights = heights[keep]
    fir = fir[keep]; maple = maple[keep]; realistic = realistic[keep]
    noel = noel[keep]; fir_only = fir_only[keep]; N = int(keep.sum())

    # near-mesh species table (npz + opaque/foliage part keys + texture paths).
    # ids: 0 fir, 1 noel, 2-5 maple, 6-12 realistic.
    near_species = [
        dict(npz=A("fir.npz"), o_keys=("oP", "oN", "oU", "oI"), o_tex=A("fir_bark.png"),
             b_keys=("bP", "bN", "bU", "bI"), b_tex=A("fir_branch.png")),
        dict(npz=A("noel.npz"), o_keys=("oP", "oN", "oU", "oI"), o_tex=A("noel_bark.png"),
             b_keys=("bP", "bN", "bU", "bI"), b_tex=A("noel_branch.png")),
    ]
    near_id = np.full(N, 0, int)
    near_id[fir_only] = 0; near_id[noel] = 1
    mi = np.where(maple)[0]; msp = np.arange(len(mi)) % 4
    for k in range(4):
        near_id[mi[msp == k]] = 2 + k
        near_species.append(dict(npz=A("maple%d.npz" % k), o_keys=("bP", "bN", "bU", "bI"),
                                 o_tex=A("maple_bark.png"), b_keys=("cP", "cN", "cU", "cI"),
                                 b_tex=A("maple_leaves.png")))
    ri = np.where(realistic)[0]; rsp = np.arange(len(ri)) % 7
    for k in range(7):
        near_id[ri[rsp == k]] = 6 + k
        near_species.append(dict(npz=A("real%d.npz" % k), o_keys=("oP", "oN", "oU", "oI"),
                                 o_tex=A("real_br%d.png" % k), b_keys=("bP", "bN", "bU", "bI"),
                                 b_tex=A("real_lf%d.png" % k)))
    # per-species sink so the TRUNK base sits on the ground and root flares/buttresses
    # embed (fir has a root flare; real3 has big buttress roots). idx = near_id.
    tp = tp.copy(); tp[:, 1] -= SINK[near_id] * heights

    tree_nodes = []
    # (node, positions, yaws, scales) per impostor species, for view-cone culling in
    # stream_impostors: the full forest's impostor cards are otherwise all submitted
    # every frame regardless of where the camera looks.
    impostors = []
    impostor_cos = math.cos(math.radians(config.impostor_cone_deg))

    def impostor(mask, tex, width):   # every species fades near -> replaced by its near-mesh
        if mask.any():
            node = InstancedBillboards(tp[mask], yaws[mask], heights[mask], A(tex + ".png"),
                                       width=width, near_fade=True)
            tree_nodes.append(_shape(node))
            impostors.append((node, tp[mask].copy(), yaws[mask].copy(), heights[mask].copy()))
    impostor(fir_only, "fir_imp", 0.50); impostor(noel, "noel_imp", 0.55)
    for k in range(4):
        impostor(near_id == 2 + k, "maple_imp%d" % k, 0.72)
    for k in range(7):
        impostor(near_id == 6 + k, "imp%d" % k, 0.60)

    near = InstancedMeshLOD(tp, yaws, heights, near_species, species_id=near_id)
    tree_nodes.append(_shape(near))
    terrain_geom.canopy = tp   # bake tree-canopy shade into the ground shadow
    print("forest: %d trees (%d fir + %d noel pine conifers, %d maple, %d realistic) — all with near-mesh LOD" % (
        N, fir_only.sum(), noel.sum(), maple.sum(), realistic.sum()))

    # Grass LOD chain (all camera-following, streamed as the viewer moves):
    #   near  = real-geometry clumps 0->R, dither-dissolving out at the disc edge;
    #   mid   = impostor billboards baked from the clump, fading IN at R (near_cut)
    #           exactly where the clumps fade out, out to 95m;
    #   far   = coarse sparse impostor billboards 90->R_far.
    # Mid/far use the clump-baked impostor (grass_clump_imp.png) so their colour
    # matches the clumps and the geometry->billboard handoff is invisible. Falls
    # back to the legacy grass.png if the impostor hasn't been baked.
    clump_radius = config.clump_radius
    # The far grass disc's outer edge follows the camera; against open ground its
    # fade band reads as a "carpet growing in" as you walk. Pushing the radius out
    # puts that edge far away (smaller on screen, fade band ~15% of the radius so it
    # widens with it) where perspective + fog hide it.
    grass_far_radius = config.grass_far_radius
    imp = "grass_clump_imp.png" if os.path.exists(A("grass_clump_imp.png")) else "grass.png"
    # grass billboards use a low flat sun term (config.grass_sun) so they match the
    # dark, per-fragment-lit clump geometry rather than reading as bright neon lumps.
    gsun = config.grass_sun
    grass = InstancedBillboards(np.zeros((0, 3), 'f4'), np.zeros(0, 'f4'), np.zeros(0, 'f4'),
                                A(imp), width=GRASS_BILLBOARD_WIDTH, far_fade=MID_GRASS_FADE,
                                near_cut=clump_radius, sun_level=gsun)
    grass_far = InstancedBillboards(np.zeros((0, 3), 'f4'), np.zeros(0, 'f4'), np.zeros(0, 'f4'),
                                    A(imp), width=GRASS_BILLBOARD_WIDTH, far_fade=grass_far_radius,
                                    near_cut=FAR_GRASS_NEAR_CUT, sun_level=gsun)

    # near-field real-geometry grass clumps: authored blade clumps instanced within
    # R, fading out across [0.8R, R] as the mid billboards fade in. Bundled as a
    # package asset; if it is missing the demo simply runs without near clumps.
    clumps = None
    if os.path.exists(CLUMP_GLB):
        cP, cN, cUV, cIdx, cTex = load_clump_glb(CLUMP_GLB, length_samples=config.clump_length_samples)
        clumps = InstancedClumps(cP, cN, cUV, cIdx, cTex, sun=CLUMP_SUN,
                                 fade_start=clump_radius * 0.8, fade_end=clump_radius)
        print("grass clumps: %d tris/clump from %s" % (len(cIdx) // 3, os.path.basename(CLUMP_GLB)))

    clump_nodes = [_shape(clumps)] if clumps is not None else []
    bb_grass = [_shape(grass_far), _shape(grass)]
    sg = sceneGraph(children=[
        Background(skyColor=[[0.28, 0.46, 0.72], [0.52, 0.66, 0.82], [0.80, 0.87, 0.93]],
                   skyAngle=[1.15, 1.5708],
                   groundColor=[[0.68, 0.76, 0.80], [0.62, 0.70, 0.74]], groundAngle=[1.5708]),
        terrain] + bb_grass + clump_nodes + tree_nodes)

    # collider: the cylinder must sit on the *visible* trunk, not the mesh origin.
    # Some models put the trunk off-centre (buttress roots, lean), so we measure each
    # species' trunk axis + radius from its mesh and place a per-tree cylinder there,
    # transformed by the same scale+yaw the vertex shader uses.
    metrics = np.array([_trunk_metrics(s["npz"], s["o_keys"][0]) for s in near_species])
    off = metrics[near_id, :2]                       # (N,2) local trunk offset
    c = np.cos(yaws); s = np.sin(yaws)
    col_x = tp[:, 0] + heights * (c * off[:, 0] + s * off[:, 1])     # matches veg_mesh.vert
    col_z = tp[:, 2] + heights * (-s * off[:, 0] + c * off[:, 1])
    collider_pos = np.stack([col_x, tp[:, 1], col_z], 1).astype(np.float32)
    collider_radius = np.clip(metrics[near_id, 2] * heights, 0.15, 0.55).astype(np.float32)

    return ForestScene(
        config=config, sceneGraph=sg, hf=hf, terrain_geom=terrain_geom, near=near,
        impostors=impostors, grass=grass, grass_far=grass_far, clumps=clumps,
        grass_mask=grass_mask, clump_radius=clump_radius, grass_far_radius=grass_far_radius,
        impostor_cos=impostor_cos, near_mesh_radius=config.near_mesh_radius,
        collider_pos=collider_pos, collider_radius=collider_radius,
        near_species=near_species, spawn=(ex, ez))
