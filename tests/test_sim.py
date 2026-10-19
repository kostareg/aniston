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
