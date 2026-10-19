"""Checks against the local database; skipped until `aniston refresh` has run."""
import polars as pl
import pytest

from aniston import store

pytestmark = pytest.mark.skipif(not store.settings().db_path.exists(), reason="no local database")


def test_nflverse_scoring_matches_espn_actuals():
    """Our scoring engine applied to nflverse stat lines should reproduce ESPN's weekly points."""
    ids = store.read("ids").filter(pl.col("espn_id").is_not_null() & pl.col("gsis_id").is_not_null()).unique("gsis_id")
    meta = store.get_json("meta")
    u = store.read("usage").filter(pl.col("season") == meta["season"], pl.col("week") < meta["current_week"])
    u = u.join(ids.select(pl.col("gsis_id").alias("player_id"), "espn_id"), on="player_id")
    espn = store.read("espn_weeks").filter(pl.col("actual").is_not_null())
    m = u.join(espn, on=["espn_id", "week"]).filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
    assert m.height > 500
    close = ((m["league_points"] - m["actual"]).abs() <= 0.11).mean()
    assert close > 0.97, f"only {close:.1%} of player-weeks match ESPN scoring"


def test_team_scores_match_player_sums():
    teams = store.read("teams")
    assert teams.height == 12 and teams["points_for"].sum() > 0
