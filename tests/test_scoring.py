from aniston.scoring import score_sleeper

PPR = {"PY": 0.04, "PTD": 4, "INTT": -2, "RY": 0.1, "RTD": 6, "REC": 1, "REY": 0.1, "RETD": 6, "FUML": -2, "2PC": 2}


def test_score_sleeper_line():
    stats = {"pass_yd": 250, "pass_td": 2, "pass_int": 1, "rush_yd": 20, "fum_lost": 1, "pass_2pt": 1}
    assert abs(score_sleeper(stats, PPR) - (10 + 8 - 2 + 2 - 2 + 2)) < 1e-9


def test_receiver_line():
    assert abs(score_sleeper({"rec": 6, "rec_yd": 85, "rec_td": 1}, PPR) - (6 + 8.5 + 6)) < 1e-9
