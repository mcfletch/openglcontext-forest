"""Quality presets and the single-measurement auto picker.

``high`` equals the shipped `ForestConfig`, the ladder gets cheaper going down,
``apply_to_scene`` mutates the live streaming knobs and re-commits the LOD fade
windows, and :class:`AutoQuality` measures once and picks the highest rung that holds
target by scaling the measurement by each rung's ``cost_weight``.
"""
from itertools import pairwise

from openglcontext_forest_demo import quality as q
from openglcontext_forest_demo.config import ForestConfig, config_from_args


def test_high_preset_matches_shipped_defaults():
    d = ForestConfig()
    h = q.PRESETS["high"]
    assert (h.cover_density, h.clump_radius, h.grass_card_radius,
            h.grass_far_radius, h.near_mesh_radius) == (
        d.cover_density, d.clump_radius, d.grass_card_radius,
        d.grass_far_radius, d.near_mesh_radius)


def test_ladder_is_ordered_cheapest_first_by_every_lever():
    rungs = [q.PRESETS[n] for n in q.LADDER]
    for a, b in pairwise(rungs):               # each rung fuller (and pricier) than the last
        assert a.cost_weight < b.cost_weight
        assert a.cover_density <= b.cover_density
        assert a.clump_radius <= b.clump_radius
        assert a.grass_card_radius <= b.grass_card_radius
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


class _Cover:
    """Stands in for the engine's GroundCover, which needs art and a context."""

    def __init__(self):
        self.density_scale = 1.0
        self.clump_radius = self.card_radius = self.far_radius = 0.0
        self.retuned = 0

    def retune(self):
        self.retuned += 1


class _Scene:
    def __init__(self):
        self.config = ForestConfig()
        self.cover = _Cover()
        self.near_mesh_radius = 0.0


def test_apply_to_scene_sets_live_knobs_and_retunes():
    s = _Scene()
    p = q.PRESETS["low"]
    q.apply_to_scene(s, p)
    assert s.config.cover_density == p.cover_density
    assert s.config.clump_radius == p.clump_radius
    assert s.config.grass_card_radius == p.grass_card_radius
    assert s.config.grass_far_radius == p.grass_far_radius
    assert s.near_mesh_radius == p.near_mesh_radius
    assert s.cover.density_scale == p.cover_density
    assert s.cover.clump_radius == p.clump_radius
    assert s.cover.card_radius == p.grass_card_radius
    assert s.cover.far_radius == p.grass_far_radius
    # Every rung's fade window was set from the radii the last preset had.
    assert s.cover.retuned == 1


def test_every_rung_of_the_chain_moves_together():
    """A preset that moved the geometry disc and left the cards where they were
    would open a ring of bare ground between the two."""
    for name in q.LADDER:
        p = q.PRESETS[name]
        assert p.clump_radius < p.grass_card_radius < p.grass_far_radius


# The geometry level-of-detail and the per-frame re-centring it used to do here
# are the engine's now (`OpenGLContext.scenegraph.vegetation.cover`), and are
# covered by `tests/unit/test_ground_cover_species.py` there. What is left here
# is what this demo decides: which preset moves which knob.
