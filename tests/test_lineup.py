import numpy as np

from aniston.model.lineup import LineupSpec

SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "RB/WR/TE": 1, "D/ST": 1, "K": 1}


def test_flex_takes_best_leftover():
    spec = LineupSpec(SLOTS)
    pos = np.array(["QB", "RB", "RB", "RB", "WR", "WR", "WR", "TE", "TE", "D/ST", "K"])
    pts = np.array([20, 15, 14, 13, 12, 11, 9, 8, 10.5, 6, 7], dtype=float)
    # QB20 + RB15,14 + WR12,11 + TE10.5 + FLEX13 (RB3 beats TE8 and WR9) + D6 + K7
    assert spec.evaluate(pts, pos) == 20 + 15 + 14 + 12 + 11 + 10.5 + 13 + 6 + 7
    mask = spec.starters(pts, pos)
    assert mask.sum() == 9 and not mask[6] and not mask[7]


def test_missing_position_scores_zero_slot():
    spec = LineupSpec(SLOTS)
    pos = np.array(["QB", "RB", "WR"])
    assert spec.evaluate(np.array([10.0, 5.0, 4.0]), pos) == 19.0


def test_vectorized_matches_scalar():
    spec = LineupSpec(SLOTS, {"QB": 0.5, "RB": 0.6, "WR": 0.6, "TE": 0.6, "K": 0.4, "D/ST": 0.7})
    rng = np.random.default_rng(0)
    pos = np.array(["QB", "QB", "RB", "RB", "RB", "RB", "WR", "WR", "WR", "WR", "TE", "TE", "D/ST", "K"])
    vals = rng.uniform(0, 25, (50, 3, len(pos)))
    total, var = spec.evaluate(vals, pos, with_var=True)
    for s in range(5):
        for w in range(3):
            assert np.isclose(total[s, w], spec.evaluate(vals[s, w], pos))
            starters = spec.starters(vals[s, w], pos)
            assert np.isclose(total[s, w], vals[s, w][starters].sum())
    assert (var > 0).all()
