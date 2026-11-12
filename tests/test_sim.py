import numpy as np

from aniston.trade.roster import RosterValuer
from aniston.model.lineup import LineupSpec


def test_legalize_drops_lowest_surplus_and_picks_up():
    spec = LineupSpec({"QB": 1, "RB": 1})
    mat = np.array([[20.0], [15.0], [5.0], [3.0], [12.0]])  # QB, RB, RB, QB, RB(FA)
    pos = np.array(["QB", "RB", "RB", "QB", "RB"])
    v = RosterValuer(mat, pos, np.array(["ACTIVE"] * 5), spec, weeks=[7], reg_last=13, active_cap=3, ir_slots=0)
    roster, dropped, added = v.legalize((0, 1, 2, 3), [4])
    assert dropped == [3] and added == [] and roster == (0, 1, 2)
    roster, dropped, added = v.legalize((0, 1), [4])
    assert added == [4] and roster == (0, 1, 4)


def test_eight_team_bracket_fixed_no_reseed():
    from aniston.sim.season import Simulator
    sim = Simulator.__new__(Simulator)
    sim.n_playoff, sim.reg_last, sim.weeks = 8, 13, [14, 15, 16]
    sim.meta = {"last_week": 16}
    seeds = np.arange(8)[None, :]
    scores = np.zeros((1, 3, 8))
    scores[0, 0] = [1, 1, 1, 1, 9, 1, 1, 1]  # 5 seed upsets 4, chalk elsewhere
    scores[0, 1] = [1, 9, 1, 1, 5, 1, 1, 1]  # 5 beats 1 (1/8 vs 4/5 side), 2 beats 3
    scores[0, 2] = [1, 3, 1, 1, 2, 1, 1, 1]
    assert sim._bracket(seeds, scores).tolist() == [1]
    scores[0, 2, 4] = 4
    assert sim._bracket(seeds, scores).tolist() == [4]


def test_injury_hazard_hits_configured_missed_share():
    from aniston.sim.season import MEAN_INJURY_WEEKS, onset_hazard
    rng = np.random.default_rng(0)
    for rate, n_weeks in [(0.11, 6), (0.06, 10)]:
        q = onset_hazard(rate, n_weeks)
        onset = rng.geometric(q, 400_000) - 1
        dur = rng.geometric(1.0 / MEAN_INJURY_WEEKS, 400_000)
        w = np.arange(n_weeks)[None, :]
        hurt = (w >= onset[:, None]) & (w < (onset + dur)[:, None])
        assert abs(hurt.mean() - rate) < 0.002
    assert onset_hazard(0.0, 6) == 0.0


def test_legalize_keeps_ir_player_instead_of_double_dropping():
    # Cutting the OUT player (lowest value) would free no active spot and force a second, healthy drop.
    spec = LineupSpec({"RB": 1})
    mat = np.array([[10.0], [8.0], [6.0], [0.0], [9.0]])
    injury = np.array(["ACTIVE", "ACTIVE", "ACTIVE", "OUT", "ACTIVE"])
    v = RosterValuer(mat, np.array(["RB"] * 5), injury, spec, weeks=[11], reg_last=13, active_cap=3, ir_slots=1)
    roster, dropped, _ = v.legalize((0, 1, 2, 3, 4), [])
    assert dropped == [2] and roster == (0, 1, 3, 4)


def test_remaining_sd_scales_with_sqrt_of_points_left():
    from aniston.sim.season import CURRENT_WEEK_TEAM_CV, remaining_sd
    assert np.isclose(remaining_sd(120.0, 0.0), CURRENT_WEEK_TEAM_CV * 120)
    assert np.isclose(remaining_sd(120.0, 90.0), CURRENT_WEEK_TEAM_CV * 60)
    assert remaining_sd(120.0, 130.0) == 0.0
