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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from OpenGLContext.loaders.tiles3d import vegetation
from OpenGLContext.scenegraph.basenodes import (
    Appearance,
    Background,
    Material,
    Shape,
    sceneGraph,
)
from OpenGLContext.scenegraph.terrain import HeightField, SplatTerrain
from OpenGLContext.scenegraph.vegetation import (
    CoverSpecies,
    GroundCover,
    InstancedBillboards,
    InstancedMeshLOD,
)
from OpenGLContext.scenegraph.vegetation.cover import read_cover_manifest
from PIL import Image

from openglcontext_forest_demo import content
from openglcontext_forest_demo.config import ForestConfig

#: The files the scene opens, by name within the art. Where the art is comes
#: from :func:`openglcontext_forest_demo.content.art_directory` when a scene is
#: built, since a first run imports this module before the art has arrived.
HEIGHTMAP = "forest_height.png"
CONTROL = "forest_control.png"
LAYERS = ["forest_floor", "grass", "rock", "moss"]

# Fixed handoff geometry that keeps the LOD cross-fades seamless — not user knobs:
# nudging these desyncs the geometry->billboard blend, so they stay constants.
TREE_SCATTER_SEED = 7          # tree layout seed (holds the shipped forest arrangement)
BIOME_SEED = 11                # conifer/maple/deciduous mix seed
IMPOSTOR_NEAR_DISC = 60.0      # always-drawn impostor disc so nothing pops at the feet on a spin
#: Where the sun is for the cover geometry, which is lit per fragment. The same
#: direction the terrain and the trees are lit from.
COVER_SUN = (-0.5, -0.72, -0.48)

# Per-species trunk-base sink so root flares/buttresses embed (fir flare, real3 buttress).
# Indexed by near-mesh species id (0 fir, 1 noel, 2-5 maple, 6-12 realistic).
SINK = np.array([0.10, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.17, 0.02, 0.02, 0.02], np.float32)


def _shape(geometry: Any) -> Shape:
    return Shape(geometry=geometry, appearance=Appearance(material=Material()))


def _trunk_metrics(npz_path: str,
                   part_key: str) -> tuple[float, float, float]:
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
    #: Everything that grows between the trees: several kinds of plant, each
    #: through the whole distance chain. The engine owns the chain; this scene
    #: says what grows, where it may, and how dark it is under the canopy.
    cover: GroundCover
    grass_mask: Callable
    impostor_cos: float
    near_mesh_radius: float
    collider_pos: np.ndarray
    collider_radius: np.ndarray
    near_species: list[dict[str, Any]]
    spawn: tuple                     # (ex, ez) walkable spawn point

    # --- reusable camera-following streamers (register on the move loop) ---
    # Each field splits into a pure ``compute_*`` (the numpy scatter/cull — GL-free,
    # so an AsyncStreamer can run it off the render thread) and an ``apply_*`` (the
    # cheap ``update_instances`` staging, which must run on the GL thread). The
    # ``stream_*`` wrappers do both in line, for the first fill and for a live
    # quality change where the restream should show at once.

    @property
    def clump_radius(self) -> float:
        """How far the cover is drawn as real geometry, in metres."""
        return self.cover.clump_radius

    @clump_radius.setter
    def clump_radius(self, metres: float) -> None:
        self.cover.clump_radius = float(metres)

    @property
    def grass_far_radius(self) -> float:
        """How far the coarse cover cards follow the camera, in metres."""
        return self.cover.far_radius

    @grass_far_radius.setter
    def grass_far_radius(self, metres: float) -> None:
        self.cover.far_radius = float(metres)

    def compute_near(self, x: float, z: float) -> tuple[Any, Any]:
        """Scatter the near field at ``(x, z)``: the cover and the near tree meshes.

        Both halves are pure numpy, so a worker thread runs this and the render
        thread takes the result through :meth:`apply_near`."""
        return self.cover.compute_near(x, z), self.near.compute_pending(
            x, z, radius=self.near_mesh_radius)

    def apply_near(self, payload: tuple[Any, Any]) -> None:
        """Stage a :meth:`compute_near` result on the GL thread."""
        cover, near_pending = payload
        self.cover.apply_near(cover)
        self.near.stage(near_pending)

    def update_clump_lod(self, x: float, z: float) -> None:
        """Re-centre the drawn cover geometry on the live camera. Every frame.

        Cheap -- distance masks over a cached scatter -- and what keeps the disc
        from lagging the walk; see
        :meth:`~OpenGLContext.scenegraph.vegetation.cover.GroundCover.select`."""
        self.cover.select(x, z)

    def stream_near(self, x: float, z: float) -> None:
        """Restream the near field synchronously (first fill / quality change)."""
        self.apply_near(self.compute_near(x, z))
        self.update_clump_lod(x, z)

    def compute_far(self, x: float, z: float) -> Any:
        """Scatter the coarse cover disc that follows the camera to the horizon."""
        return self.cover.compute_far(x, z)

    def apply_far(self, payload: Any) -> None:
        """Stage a :meth:`compute_far` result on the GL thread."""
        self.cover.apply_far(payload)

    def stream_far(self, x: float, z: float) -> None:
        """Restream the far cover disc synchronously."""
        self.apply_far(self.compute_far(x, z))

    def compute_impostors(self, x: float, z: float, fx: float,
                          fz: float) -> list[tuple[Any, Any, Any, Any]]:
        """The impostor cards inside the forward view cone `(fx, fz)`, per species.

        A forward cone (dot > 0, dot^2 > cos^2 * dist^2 tests the angle without a
        per-tree sqrt) plus a small always-drawn near disc. Trees behind and beside
        the camera — roughly half the forest — never reach the GPU. Callers pass the
        camera's ground-plane forward vector so this stays independent of any one
        navigation scheme. Returns ``[(node, pos, yaw, scale), …]`` for
        :meth:`apply_impostors`."""
        cc = self.impostor_cos * self.impostor_cos
        near2 = IMPOSTOR_NEAR_DISC * IMPOSTOR_NEAR_DISC
        out = []
        for node, P, Y, S in self.impostors:
            dx = P[:, 0] - x; dz = P[:, 2] - z
            d2 = dx * dx + dz * dz
            dot = dx * fx + dz * fz
            keep = (d2 < near2) | ((dot > 0.0) & (dot * dot > cc * d2))
            out.append((node, P[keep], Y[keep], S[keep]))
        return out

    def apply_impostors(
        self, payload: list[tuple[Any, Any, Any, Any]]) -> None:
        """Stage a :meth:`compute_impostors` result on the GL thread."""
        for node, p, y, s in payload:
            node.update_instances(p, y, s)

    def stream_impostors(self, x: float, z: float, fx: float,
                         fz: float) -> None:
        """Cull and submit the impostor cards synchronously."""
        self.apply_impostors(self.compute_impostors(x, z, fx, fz))


def build_forest_scene(config: ForestConfig,
                       art: str | None = None) -> ForestScene:
    """Build the terrain + instanced forest + grass LOD chain from `config`.

    Creates scenegraph nodes and numpy instance data only (GL uploads on first
    render), so call this inside a GL context's `OnInit`. Returns a `ForestScene`
    the caller renders and streams; the concrete navigation lives on top.

    ``art`` is the directory the art is read from, by default wherever
    :func:`~openglcontext_forest_demo.content.art_directory` finds it now."""
    art = art if art is not None else content.art_directory()

    def A(name: str) -> str:
        return os.path.join(art, name)

    hf = HeightField.from_image(A(HEIGHTMAP), config.res, config.extent, config.relief)
    hfn = hf.sample
    terrain_geom = SplatTerrain(hf, LAYERS, A(CONTROL))
    terrain = _shape(terrain_geom)

    # Grass grows on any soft ground (forest floor, meadow, moss) but NOT on rock:
    # scatter is thinned by (1 - rock weight) from the control map, so it stays lush
    # where you walk yet vanishes on bare mountainsides instead of dotting them with
    # green blobs. Reuses the HeightField sampler to read the map over the extent.
    _ctl = np.asarray(Image.open(A(CONTROL)).convert("RGBA").resize((config.res, config.res), Image.Resampling.LANCZOS),
                      np.float32) / 255.0
    _grass_allowed = np.clip(1.0 - _ctl[..., LAYERS.index("rock")], 0.0, 1.0)
    _rock_mask = HeightField(_grass_allowed, config.extent, 1.0).sample

    def grass_mask(px: Any, pz: Any) -> Any:
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
    def gentle(p: Any) -> Any:
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
    # The shape `InstancedMeshLOD` declares for a species: a mesh file, the
    # four array keys for each of its two parts, and a texture for each.
    near_species: list[dict[str, Any]] = [
        {"npz": A("fir.npz"), "o_keys": ("oP", "oN", "oU", "oI"), "o_tex": A("fir_bark.png"),
             "b_keys": ("bP", "bN", "bU", "bI"), "b_tex": A("fir_branch.png")},
        {"npz": A("noel.npz"), "o_keys": ("oP", "oN", "oU", "oI"), "o_tex": A("noel_bark.png"),
             "b_keys": ("bP", "bN", "bU", "bI"), "b_tex": A("noel_branch.png")},
    ]
    near_id = np.full(N, 0, int)
    near_id[fir_only] = 0; near_id[noel] = 1
    mi = np.where(maple)[0]; msp = np.arange(len(mi)) % 4
    for k in range(4):
        near_id[mi[msp == k]] = 2 + k
        near_species.append({"npz": A(f"maple{k}.npz"), "o_keys": ("bP", "bN", "bU", "bI"),
                                 "o_tex": A("maple_bark.png"), "b_keys": ("cP", "cN", "cU", "cI"),
                                 "b_tex": A("maple_leaves.png")})
    ri = np.where(realistic)[0]; rsp = np.arange(len(ri)) % 7
    for k in range(7):
        near_id[ri[rsp == k]] = 6 + k
        near_species.append({"npz": A(f"real{k}.npz"), "o_keys": ("oP", "oN", "oU", "oI"),
                                 "o_tex": A(f"real_br{k}.png"), "b_keys": ("bP", "bN", "bU", "bI"),
                                 "b_tex": A(f"real_lf{k}.png")})
    # per-species sink so the TRUNK base sits on the ground and root flares/buttresses
    # embed (fir has a root flare; real3 has big buttress roots). idx = near_id.
    tp = tp.copy(); tp[:, 1] -= SINK[near_id] * heights

    tree_nodes = []
    # (node, positions, yaws, scales) per impostor species, for view-cone culling in
    # stream_impostors: the full forest's impostor cards are otherwise all submitted
    # every frame regardless of where the camera looks.
    impostors = []
    impostor_cos = math.cos(math.radians(config.impostor_cone_deg))

    def impostor(mask: Any, tex: Any, width: float) -> None:
        # every species fades near -> replaced by its near-mesh
        if mask.any():
            node = InstancedBillboards(tp[mask], yaws[mask], heights[mask], A(tex + ".png"),
                                       width=width, near_fade=True)
            tree_nodes.append(_shape(node))
            impostors.append((node, tp[mask].copy(), yaws[mask].copy(), heights[mask].copy()))
    impostor(fir_only, "fir_imp", 0.50); impostor(noel, "noel_imp", 0.55)
    for k in range(4):
        impostor(near_id == 2 + k, f"maple_imp{k}", 0.72)
    for k in range(7):
        impostor(near_id == 6 + k, f"imp{k}", 0.60)

    near = InstancedMeshLOD(tp, yaws, heights, near_species, species_id=near_id)
    tree_nodes.append(_shape(near))
    terrain_geom.canopy = tp   # bake tree-canopy shade into the ground shadow
    print(f"forest: {N} trees ({fir_only.sum()} fir + {noel.sum()} noel pine conifers, "
          f"{maple.sum()} maple, {realistic.sum()} realistic) — all with near-mesh LOD")

    # What grows between the trees. The engine owns the whole distance chain --
    # two geometry rungs, the cards that replace them, and the coarse field out
    # to the haze -- so what is decided here is only what grows (the baked
    # species set), where it may (the mask), how dark it is under the canopy
    # (the terrain's shading) and how much tree stands over it (the terrain's
    # closure, which is what puts the shrubs in the gaps and along the edges).
    cover = GroundCover(
        hf, cover_species(art), clump_radius=config.clump_radius,
        card_radius=config.grass_card_radius,
        far_radius=config.grass_far_radius,
        mask=grass_mask, shade=terrain_geom.shade,
        canopy=terrain_geom.canopy_cover, sun=COVER_SUN)
    cover.density_scale = config.cover_density
    kinds = sorted({one.name.rsplit('_', 1)[0] for one in cover.species})
    print(f"ground cover: {len(cover.species)} kinds — {', '.join(kinds)}")

    sg = sceneGraph(children=[
        Background(skyColor=[[0.28, 0.46, 0.72], [0.52, 0.66, 0.82], [0.80, 0.87, 0.93]],
                   skyAngle=[1.15, 1.5708],
                   groundColor=[[0.68, 0.76, 0.80], [0.62, 0.70, 0.74]], groundAngle=[1.5708]),
        terrain, cover] + tree_nodes)

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
        impostors=impostors, cover=cover, grass_mask=grass_mask,
        impostor_cos=impostor_cos, near_mesh_radius=config.near_mesh_radius,
        collider_pos=collider_pos, collider_radius=collider_radius,
        near_species=near_species, spawn=(ex, ez))


def cover_species(art: str | None = None) -> list[CoverSpecies]:
    """What grows between the trees, read from the baked art in ``art``.

    A set rather than one plant: grass, fern, nettle, periwinkle and shrub, each
    at its own density and its own size. ``cover.json`` is written by
    ``oglc-bake-plants`` beside the models it names, so adding a plant to the
    forest is a re-bake rather than an edit here. ``art`` defaults to wherever
    :func:`~openglcontext_forest_demo.content.art_directory` finds the art now.
    """
    art = art if art is not None else content.art_directory()
    return read_cover_manifest(art)
