"""Quality presets and the single-measurement auto picker.

``high`` equals the shipped `ForestConfig`, the ladder gets cheaper going down,
``apply_to_scene`` mutates the live streaming knobs and re-commits the LOD fade
windows, and :class:`AutoQuality` measures once and picks the highest rung that holds
target by scaling the measurement by each rung's ``cost_weight``.
"""
import types
from itertools import pairwise

import numpy as np

from openglcontext_forest_demo import quality as q
from openglcontext_forest_demo import scene as sc
from openglcontext_forest_demo.config import ForestConfig


def test_high_preset_matches_shipped_defaults():
    d = ForestConfig()
    h = q.PRESETS["high"]
    assert (h.clump_density, h.clump_radius, h.grass_mid_density,
            h.grass_far_radius, h.grass_far_density, h.near_mesh_radius) == (
        d.clump_density, d.clump_radius, d.grass_mid_density,
        d.grass_far_radius, d.grass_far_density, d.near_mesh_radius)


def test_ladder_is_ordered_cheapest_first_by_every_lever():
    rungs = [q.PRESETS[n] for n in q.LADDER]
    for a, b in pairwise(rungs):               # each rung fuller (and pricier) than the last
        assert a.cost_weight < b.cost_weight
        assert a.clump_density <= b.clump_density
        assert a.clump_radius <= b.clump_radius
    assert q.LADDER[0] == "low" and q.LADDER[-1] == "high"
    assert q.PRESETS["high"].cost_weight == 1.0


def test_next_level_clamps_at_the_rails():
    assert q.next_level("low", -1) == "low"
    assert q.next_level("high", 1) == "high"
    assert q.next_level("low", 1) == q.LADDER[1]      # one rung up from the bottom


def test_f8_cycle_visits_every_rung():
    seen, cur = [], "low"
    for _ in range(len(q.LADDER)):
        seen.append(cur)
        cur = q.next_level_cycle(cur)
    assert seen == q.LADDER                            # cheapest-first, every rung
    assert q.next_level_cycle("high") == "low"         # wraps


def test_every_rung_is_selectable():
    # No hidden rung: every ladder rung (plus auto) is a --quality/F8 choice, and the
    # command line actually accepts each (catches config<->quality drift).
    from openglcontext_forest_demo.config import config_from_args
    assert set(q.CHOICES) == {"auto", *q.LADDER}
    for name in q.CHOICES:
        assert config_from_args(["--quality", name]).quality == name


# --- AutoQuality (single measurement, one decision) -------------------------
# down_ms = 1000/(60-12) = 20.8 ms; the pick is the highest rung whose estimated
# frame time (measured_ms * cost_weight, since start=high has weight 1.0) is <= that.

def test_auto_keeps_the_level_when_it_holds_target():
    a = q.AutoQuality(start="high", window=3, warmup=0)
    results = [a.add_frame(8.0) for _ in range(3)]     # ~125 fps at high: it holds
    assert results == [None, None, None]
    assert a.decided and a.target is None


def test_auto_lands_on_the_highest_rung_that_holds():
    # 25 ms: high (25) misses, medhigh (0.78*25=19.5) holds -> medhigh.
    a = q.AutoQuality(start="high", window=3, warmup=0)
    out = [a.add_frame(25.0) for _ in range(3)]
    assert a.decided and a.target == "medhigh"
    assert out[-1] == "medhigh"
    # 30 ms: medhigh (0.78*30=23.4) misses, medium (0.67*30=20.1) holds -> medium.
    b = q.AutoQuality(start="high", window=2, warmup=0)
    b.add_frame(30.0)
    assert b.add_frame(30.0) == "medium"


def test_auto_drops_to_low_when_far_over_the_band():
    a = q.AutoQuality(start="high", window=2, warmup=0)
    a.add_frame(60.0)
    assert a.add_frame(60.0) == "low"                  # nothing holds -> cheapest rung
    assert a.decided and a.target == "low"


def test_auto_decides_only_once():
    a = q.AutoQuality(start="high", window=2, warmup=0)
    a.add_frame(30.0)
    a.add_frame(30.0)                                  # decides here (medium)
    assert a.add_frame(5.0) is None                    # further frames are ignored
    assert a.target == "medium"


def test_auto_ignores_warmup_and_stall_frames():
    a = q.AutoQuality(start="high", window=2, warmup=1)
    assert a.add_frame(5.0) is None      # warmup frame, dropped
    assert a.add_frame(500.0) is None    # stall frame, not counted toward the window
    assert a.add_frame(30.0) is None     # first real sample; window not full
    assert a.add_frame(30.0) == "medium"  # window full and slow -> highest holding rung


def test_auto_target_clamps_at_the_low_rail():
    a = q.AutoQuality(start="low", window=2, warmup=0)
    a.add_frame(60.0)
    assert a.add_frame(60.0) is None     # already at low: nothing lower to pick
    assert a.decided and a.target is None


# --- apply_to_scene ---------------------------------------------------------

class _Node:
    def __init__(self, gl=True):
        self._gl = 1 if gl else None
        self.committed = 0
        self.fade_start = self.fade_end = self.near_cut = self.far_fade = None
        self.cut_start = self.cut_end = None

    def _commit_constants(self):
        self.committed += 1


class _Scene:
    def __init__(self):
        self.config = ForestConfig()
        self.grass = _Node()
        self.grass_far = _Node()
        self.clump_radius = self.grass_far_radius = self.near_mesh_radius = 0.0
        self.retuned = 0

    def retune_clumps(self):
        self.retuned += 1


def test_apply_to_scene_sets_live_knobs_and_retunes():
    s = _Scene()
    p = q.PRESETS["low"]
    q.apply_to_scene(s, p)
    assert s.config.clump_density == p.clump_density
    assert s.config.grass_mid_density == p.grass_mid_density
    assert s.config.grass_far_density == p.grass_far_density
    assert s.clump_radius == p.clump_radius
    assert s.grass_far_radius == p.grass_far_radius
    assert s.near_mesh_radius == p.near_mesh_radius
    assert s.retuned == 1                       # clump LOD windows follow clump_radius
    assert s.grass.near_cut == p.clump_radius
    assert s.grass_far.far_fade == p.grass_far_radius
    assert s.grass.committed == s.grass_far.committed == 1


def test_apply_to_scene_skips_grass_commit_before_gl_init():
    s = _Scene()
    for node in (s.grass, s.grass_far):
        node._gl = None
    q.apply_to_scene(s, q.PRESETS["medium"])
    assert s.grass.committed == 0
    assert s.grass.near_cut == q.PRESETS["medium"].clump_radius   # value still set


def test_retune_clumps_sets_near_and_far_lod_windows():
    near, far = _Node(), _Node()
    ns = types.SimpleNamespace(clumps_near=near, clumps_far=far, clump_radius=30.0)
    sc.ForestScene.retune_clumps(ns)
    r1 = 30.0 * sc.CLUMP_LOD_FRAC
    assert near.fade_end == r1 and near.fade_start == r1 * sc.CLUMP_FADE_FRAC   # overlay fades out at R1
    assert far.cut_start == 0.0 and far.cut_end == 0.0        # full coarse base: no inner cut
    assert far.fade_end == 30.0 and far.fade_start == 30.0 * sc.CLUMP_FADE_FRAC
    assert near.committed == far.committed == 1


def test_retune_clumps_is_a_noop_without_clumps():
    sc.ForestScene.retune_clumps(types.SimpleNamespace(clumps_near=None, clump_radius=30.0))


# --- update_clump_lod: the per-frame camera-relative disc re-selection ----------
# The clump scatter is cached over a disc wider than the draw radius; each frame the
# drawn near/far subsets are re-picked from it against the *live* camera, so the disc
# tracks the walk with no streaming lag (what read as the mid-distance clumps popping
# up in density when a lagged disc recentred).

class _InstNode:
    def __init__(self):
        self.pos = None
        self.updates = 0

    def update_instances(self, p, y, s):
        self.pos = p
        self.updates += 1


def _clump_grid(half=45.0, step=0.5):
    xs = np.arange(-half, half, step)
    gx, gz = np.meshgrid(xs, xs)
    p = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], 1).astype("f4")
    z = np.zeros(len(p), "f4")
    return (p, z, z + 1.0)


def _scene_with_cache():
    near, far = _InstNode(), _InstNode()
    ns = types.SimpleNamespace(clumps_near=near, clumps_far=far, clump_radius=30.0,
                               _clump_cache=_clump_grid(), _clump_lod_center=None)
    return ns, near, far


def test_update_clump_lod_selects_the_live_camera_disc():
    ns, near, far = _scene_with_cache()
    sc.ForestScene.update_clump_lod(ns, 0.0, 5.0)          # camera at (0, 5)
    dfar = np.hypot(far.pos[:, 0], far.pos[:, 2] - 5.0)
    dnear = np.hypot(near.pos[:, 0], near.pos[:, 2] - 5.0)
    assert dfar.max() <= 30.0                              # coarse base = whole disc
    assert dnear.max() <= 30.0 * sc.CLUMP_LOD_FRAC         # full overlay = out to R1
    assert len(near.pos) < len(far.pos)                    # overlay is a subset


def test_update_clump_lod_tracks_camera_without_a_density_jump():
    # Walking small steps, the clumps in the forward mid-distance band (where the pop
    # showed) stay ~constant: the disc follows the camera continuously instead of
    # lagging and snapping forward at a stream boundary.
    ns, _near, far = _scene_with_cache()
    bands = []
    for z in np.arange(0.0, 12.0, 0.6):                    # across two former 10 m boundaries
        sc.ForestScene.update_clump_lod(ns, 0.0, float(z))
        ahead = far.pos[:, 2] - z
        bands.append(int(((ahead > 20.0) & (ahead < 30.0)).sum()))
    bands = np.array(bands)
    # frame-to-frame change stays a small fraction of the band; a lagged disc would
    # drop this near zero then jump back to full at each boundary.
    assert np.abs(np.diff(bands)).max() < 0.15 * bands.mean()


def test_update_clump_lod_skips_reselection_while_the_camera_holds_still():
    ns, _near, far = _scene_with_cache()
    sc.ForestScene.update_clump_lod(ns, 0.0, 0.0)
    sc.ForestScene.update_clump_lod(ns, 0.0, 0.0)          # unmoved: no second select
    sc.ForestScene.update_clump_lod(ns, 0.01, 0.0)         # sub-threshold jitter: still skipped
    assert far.updates == 1
    sc.ForestScene.update_clump_lod(ns, 5.0, 0.0)          # a real move re-selects
    assert far.updates == 2


def test_update_clump_lod_is_a_noop_without_clumps_or_cache():
    sc.ForestScene.update_clump_lod(               # no clump nodes at all
        types.SimpleNamespace(clumps_near=None, _clump_cache=None, clump_radius=30.0), 0.0, 0.0)
    ns, _near, far = _scene_with_cache()
    ns._clump_cache = None                         # nodes present, nothing cached yet
    sc.ForestScene.update_clump_lod(ns, 0.0, 0.0)
    assert far.pos is None


def test_apply_to_scene_delegates_clump_windows_to_the_scene():
    s = _Scene()
    q.apply_to_scene(s, q.PRESETS["low"])         # must not touch clump nodes directly
    assert s.retuned == 1
    assert s.grass.near_cut == q.PRESETS["low"].clump_radius
