"""Quality presets and the down-only auto picker.

``high`` equals the shipped `ForestConfig`, the ladder gets cheaper going down,
``apply_to_scene`` mutates the live streaming knobs and re-commits the LOD fade
windows, and :class:`AutoQuality` only ever steps down (so the 60 fps target
sitting between medium and high cannot make it oscillate).
"""
from openglcontext_forest_demo import quality as q
from openglcontext_forest_demo.config import ForestConfig


def test_high_preset_matches_shipped_defaults():
    d = ForestConfig()
    h = q.PRESETS["high"]
    assert (h.clump_density, h.clump_radius, h.grass_mid_density,
            h.grass_far_radius, h.grass_far_density, h.near_mesh_radius) == (
        d.clump_density, d.clump_radius, d.grass_mid_density,
        d.grass_far_radius, d.grass_far_density, d.near_mesh_radius)


def test_ladder_gets_cheaper_going_down():
    lo, me, hi = q.PRESETS["low"], q.PRESETS["medium"], q.PRESETS["high"]
    assert lo.clump_density < me.clump_density < hi.clump_density
    assert lo.clump_radius < me.clump_radius < hi.clump_radius
    assert lo.grass_far_radius < me.grass_far_radius < hi.grass_far_radius
    assert q.ORDER == ["low", "medium", "high"]


def test_next_level_cycle_wraps():
    assert q.next_level_cycle("low") == "medium"
    assert q.next_level_cycle("medium") == "high"
    assert q.next_level_cycle("high") == "low"


def test_next_level_clamps_at_the_rails():
    assert q.next_level("low", -1) == "low"
    assert q.next_level("high", 1) == "high"
    assert q.next_level("low", 1) == "medium"


# --- AutoQuality (down-only) ------------------------------------------------

def test_auto_steps_down_when_median_is_slow():
    a = q.AutoQuality(start="medium", window=3, warmup=0)
    results = [a.add_frame(30.0) for _ in range(3)]   # 30 ms > 18.2 ms target band
    assert "low" in results
    assert not a.decided        # a down-step is not a settle; it keeps converging


def test_auto_decides_once_a_window_lands_in_band():
    a = q.AutoQuality(start="high", window=2, warmup=0)
    a.add_frame(10.0)
    assert a.add_frame(10.0) is None   # ~100 fps at high: in band, no down-step
    assert a.decided


def test_auto_holds_and_decides_when_fast():
    a = q.AutoQuality(start="medium", window=3, warmup=0)
    results = [a.add_frame(8.0) for _ in range(3)]     # ~125 fps: comfortably in band
    assert results == [None, None, None]
    assert a.decided


def test_auto_never_steps_up_even_with_huge_headroom():
    a = q.AutoQuality(start="medium", window=3, warmup=0)
    results = [a.add_frame(2.0) for _ in range(9)]     # 500 fps would tempt an up-step
    assert all(r != "high" for r in results)


def test_auto_ignores_warmup_and_stall_frames():
    a = q.AutoQuality(start="medium", window=2, warmup=1)
    assert a.add_frame(5.0) is None      # warmup frame, dropped
    assert a.add_frame(500.0) is None    # stall frame, not counted toward the window
    assert a.add_frame(30.0) is None     # first real sample; window not full
    assert a.add_frame(30.0) == "low"    # window full and slow -> step down


def test_auto_at_low_cannot_go_lower():
    a = q.AutoQuality(start="low", window=2, warmup=0)
    results = [a.add_frame(60.0) for _ in range(4)]
    assert all(r is None for r in results)


# --- apply_to_scene ---------------------------------------------------------

class _Node:
    def __init__(self, gl=True):
        self._gl = 1 if gl else None
        self.committed = 0
        self.fade_start = self.fade_end = self.near_cut = self.far_fade = None

    def _commit_constants(self):
        self.committed += 1


class _Scene:
    def __init__(self, with_clumps=True):
        self.config = ForestConfig()
        self.clumps = _Node() if with_clumps else None
        self.grass = _Node()
        self.grass_far = _Node()
        self.clump_radius = self.grass_far_radius = self.near_mesh_radius = 0.0


def test_apply_to_scene_sets_live_knobs_and_recommits():
    s = _Scene()
    p = q.PRESETS["low"]
    q.apply_to_scene(s, p)
    assert s.config.clump_density == p.clump_density
    assert s.config.grass_mid_density == p.grass_mid_density
    assert s.config.grass_far_density == p.grass_far_density
    assert s.clump_radius == p.clump_radius
    assert s.grass_far_radius == p.grass_far_radius
    assert s.near_mesh_radius == p.near_mesh_radius
    assert s.clumps.fade_end == p.clump_radius
    assert s.clumps.fade_start == p.clump_radius * 0.8
    assert s.grass.near_cut == p.clump_radius
    assert s.grass_far.far_fade == p.grass_far_radius
    assert s.clumps.committed == s.grass.committed == s.grass_far.committed == 1


def test_apply_to_scene_skips_commit_before_gl_init():
    s = _Scene()
    for node in (s.clumps, s.grass, s.grass_far):
        node._gl = None
    p = q.PRESETS["medium"]
    q.apply_to_scene(s, p)
    assert s.clumps.committed == 0
    assert s.clumps.fade_end == p.clump_radius   # value still set for first render


def test_apply_to_scene_tolerates_absent_clumps():
    s = _Scene(with_clumps=False)
    q.apply_to_scene(s, q.PRESETS["low"])         # must not raise
    assert s.grass.near_cut == q.PRESETS["low"].clump_radius
