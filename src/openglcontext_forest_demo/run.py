#!/usr/bin/env python
"""Walkable forest demo — runtime-splat terrain + instanced forest.

Run:
    /workspaces/OpenGL-dev/.venv/bin/python /workspaces/OpenGL-dev/forest-demo/run.py

Controls: W/A/S/D move, mouse-look, arrows/PageUp-Down also navigate. You walk on
the terrain (height-clamped, blocked by trunks). Real Great-Smoky-Mountains
elevation drives the terrain; the ground is a crisp runtime multi-layer splat
(forest_floor/grass/rock/moss). The forest is GPU-instanced with distance LOD: real
3D tree *meshes* near the camera, baked *impostor* billboards farther out
(cross-faded), plus camera-following grass (two LODs) and terrain sun-shadows + tree
canopy shade. Biome mix: fir conifers on high/steep ground, maples in valleys,
realistic deciduous between. ~640k trees, 3x+ above 60fps. First run downloads CC0
ground textures (cached after).

This is a *scene* built from reusable engine pieces:
  OpenGLContext.scenegraph.terrain      — HeightField, SplatTerrain
  OpenGLContext.scenegraph.vegetation   — InstancedBillboards, InstancedMeshLOD,
                                          world_grid_scatter
  OpenGLContext.move.terrainwalk        — TerrainWalkMixin (clamp/collide/stream)

Tree assets (CC-BY, see CREDITS-trees.txt): "Fir tree" by Georgeous, "Noel Pine
Tree" by 3D Error 404, "Maple trees pack" by LOLIPOP, "Realistic Trees Collection"
by Jungle Jim, "Low Poly Forest Tree Pack" by 99.Miles.
"""
import os, sys, math, numpy as np
os.environ.setdefault("OPENGLCONTEXT_PROFILE", "core")
os.environ.setdefault("OPENGLCONTEXT_RENDERER", "pbr")
os.environ.setdefault("OPENGLCONTEXT_BACKEND", "glfw")
from OpenGLContext import testingcontext, quaternion
from OpenGLContext.loaders.tiles3d import vegetation
from OpenGLContext.scenegraph.terrain import HeightField, SplatTerrain
from OpenGLContext.scenegraph.vegetation import InstancedBillboards, InstancedMeshLOD, world_grid_scatter
from OpenGLContext.move.terrainwalk import TerrainWalkMixin
BaseContext = testingcontext.getInteractive()
from OpenGLContext.scenegraph.basenodes import sceneGraph, Background, Shape, Appearance, Material

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
EXTENT = 4096.0; RELIEF = 450.0; RES = 513
HEIGHTMAP = os.path.join(ASSETS, "forest_height.png")
CONTROL = os.path.join(ASSETS, "forest_control.png")
LAYERS = ["forest_floor", "grass", "rock", "moss"]
# Tree keep-out radius = SPACING_BASE + SPACING_PER_M * height. Two neighbours must
# be at least the sum of their radii apart, so mature (tall) trees hold more room.
# Base 0.8 + 0.06/m gives ~3.5m between two 16m trees, ~2.7m between two 9m ones.
SPACING_BASE = 0.8
SPACING_PER_M = 0.06


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


class Forest(TerrainWalkMixin, BaseContext):
    eye_height = 1.7

    def OnInit(self):
        try:
            import glfw; glfw.swap_interval(0)
        except Exception:
            pass
        hf = HeightField.from_image(HEIGHTMAP, RES, EXTENT, RELIEF)
        self.hf = hf
        hfn = hf.sample
        terrain_geom = SplatTerrain(hf, LAYERS, CONTROL)
        terrain = _shape(terrain_geom)

        # spawn on a gentle, walkable spot (so we stand among trees, not on a cliff)
        ex, ez = -200.0, 300.0
        rng = np.random.default_rng(1); bs = 9e9
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
        trees = vegetation.scatter_disc((0, 0, 0), 2200.0, 0.09, 7, hfn,
                                        keep=gentle, scale_range=(0.75, 1.45))
        tp = trees.positions; N = len(tp); yaws = trees.yaws; sc = trees.scales

        # --- biome mix: conifers high/steep, maples in valleys, realistic deciduous
        #     between, with coarse spatial noise so each area has a dominant type plus
        #     a minority mix ---
        elev = np.clip(tp[:, 1] / RELIEF, 0, 1)
        slope = np.clip(hf.slope(tp[:, 0], tp[:, 2], eps=10.0), 0, 1.5)
        nz = np.sin(tp[:, 0] * 0.004 + 1.3) * np.cos(tp[:, 2] * 0.0035) + 0.5 * np.sin(tp[:, 2] * 0.010)
        nz = (nz - nz.min()) / (nz.max() - nz.min() + 1e-6)
        rng3 = np.random.default_rng(11)
        fir = rng3.random(N) < np.clip(elev * 1.4 + slope * 1.2 + (nz - 0.5) * 1.1 - 0.28, 0.03, 0.97)
        maple = (~fir) & (rng3.random(N) < np.clip((0.55 - elev) * 1.7 + (0.5 - nz) * 0.7, 0.15, 0.9))
        realistic = (~fir) & (~maple)
        heights = (sc * np.where(fir, 16.0, 12.5)).astype(np.float32)
        noel = fir & (rng3.random(N) < 0.45); fir_only = fir & ~noel

        # De-clump: a uniform scatter piles trees into branch soup. Thin to blue-noise
        # so no two survivors are closer than their combined keep-out radius (which
        # grows with tree height), and the biggest tree in a crowd wins the spot.
        keep = vegetation.poisson_thin(tp, SPACING_BASE + SPACING_PER_M * heights, priority=heights)
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
        SINK = np.array([0.10, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.17, 0.02, 0.02, 0.02], np.float32)
        tp = tp.copy(); tp[:, 1] -= SINK[near_id] * heights

        tree_nodes = []

        def impostor(mask, tex, width):   # every species fades near -> replaced by its near-mesh
            if mask.any():
                tree_nodes.append(_shape(InstancedBillboards(
                    tp[mask], yaws[mask], heights[mask], A(tex + ".png"),
                    width=width, near_fade=True)))
        impostor(fir_only, "fir_imp", 0.50); impostor(noel, "noel_imp", 0.55)
        for k in range(4):
            impostor(near_id == 2 + k, "maple_imp%d" % k, 0.72)
        for k in range(7):
            impostor(near_id == 6 + k, "imp%d" % k, 0.60)

        self._near = InstancedMeshLOD(tp, yaws, heights, near_species, species_id=near_id)
        tree_nodes.append(_shape(self._near))
        terrain_geom.canopy = tp   # bake tree-canopy shade into the ground shadow
        print("forest: %d trees (%d fir + %d noel pine conifers, %d maple, %d realistic) — all with near-mesh LOD" % (
            N, fir_only.sum(), noel.sum(), maple.sum(), realistic.sum()))

        # camera-following grass, two LODs (streamed as the viewer moves):
        #  near = fine dense tufts out to 95m; far = coarse sparse clumps 90->340m
        #  that fill the mid-distance ground so it isn't bare texture past the near disc.
        self._grass = InstancedBillboards(np.zeros((0, 3), 'f4'), np.zeros(0, 'f4'), np.zeros(0, 'f4'),
                                          A("grass.png"), width=0.5, far_fade=95.0)
        self._grass_far = InstancedBillboards(np.zeros((0, 3), 'f4'), np.zeros(0, 'f4'), np.zeros(0, 'f4'),
                                              A("grass.png"), width=1.7, far_fade=340.0, near_cut=90.0)

        self.sg = sceneGraph(children=[
            Background(skyColor=[[0.28, 0.46, 0.72], [0.52, 0.66, 0.82], [0.80, 0.87, 0.93]],
                       skyAngle=[1.15, 1.5708],
                       groundColor=[[0.68, 0.76, 0.80], [0.62, 0.70, 0.74]], groundAngle=[1.5708]),
            terrain, _shape(self._grass_far), _shape(self._grass)] + tree_nodes)

        if self.platform is not None:
            self.platform.setFrustum(math.radians(62), None, 0.25, 9000.0)
            self.platform.setPosition((ex, hf.height_at(ex, ez) + self.eye_height, ez))
            self.platform.setOrientation(quaternion.fromXYZR(0, 1, 0, 2.3))

        # walk: clamp to terrain, block only the trunks, stream veg on the move.
        # The collider must sit on the *visible* trunk, not the mesh origin: some
        # models put the trunk off-centre (buttress roots, lean), so we measure each
        # species' trunk axis + radius from its mesh and place a per-tree cylinder
        # there, transformed by the same scale+yaw the vertex shader uses.
        metrics = np.array([_trunk_metrics(s["npz"], s["o_keys"][0]) for s in near_species])
        off = metrics[near_id, :2]                       # (N,2) local trunk offset
        c = np.cos(yaws); s = np.sin(yaws)
        col_x = tp[:, 0] + heights * (c * off[:, 0] + s * off[:, 1])     # matches veg_mesh.vert
        col_z = tp[:, 2] + heights * (-s * off[:, 0] + c * off[:, 1])
        collider_pos = np.stack([col_x, tp[:, 1], col_z], 1).astype(np.float32)
        collider_r = np.clip(metrics[near_id, 2] * heights, 0.15, 0.55).astype(np.float32)
        self.init_walk(hf, collider_pos, collider_r)
        self.add_stream(10.0, self._stream_near)
        self.add_stream(40.0, self._stream_far)

    def _stream_near(self, x, z):
        p, y, s = world_grid_scatter(x, z, 95.0, 2.6, self.hf)
        self._grass.update_instances(p, y, s)
        self._near.update(x, z, radius=56.0)   # near meshes follow the camera (cover the LOD band)

    def _stream_far(self, x, z):
        p, y, s = world_grid_scatter(x, z, 340.0, 0.09, self.hf, scale_mul=1.5)
        self._grass_far.update_instances(p, y, s)


def print_credits():
    """Print the CC-BY attribution notices required for the tree models used."""
    print("=" * 74)
    print("Forest demo — asset credits (full text in CREDITS.txt)")
    print("Tree models, CC-BY 4.0 (http://creativecommons.org/licenses/by/4.0/):")
    for line in ('"Fir tree" by Georgeous (https://skfb.ly/pA8TG)',
                 '"Noel_Pine_Tree" by 3D Error 404 (https://skfb.ly/6XHoJ)',
                 '"Maple trees pack" by LOLIPOP (https://skfb.ly/p9tGx)',
                 '"Realistic Trees Collection" by Jungle Jim (https://skfb.ly/pDzJR)',
                 '"Low Poly Forest Tree Pack" by 99.Miles (https://skfb.ly/pJXrH)'):
        print("  * " + line)
    print("Ground textures: CC0 via ambientCG.com. Terrain: public AWS Terrain Tiles.")
    print("=" * 74)
    sys.stdout.flush()   # CC-BY attribution must reach the user even if stdout is piped


def main():
    """Console entry point (``oglc-forest``): print attributions, then run the demo."""
    print_credits()
    Forest.ContextMainLoop()


if __name__ == "__main__":
    main()
