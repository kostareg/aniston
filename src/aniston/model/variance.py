"""Weekly scoring volatility by position, fitted on last season's nflverse data."""
import polars as pl

from aniston import store

DEFAULT_CV = {"QB": 0.40, "RB": 0.55, "WR": 0.60, "TE": 0.65, "K": 0.45, "D/ST": 0.70}


def position_cv(season: int) -> dict[str, float]:
    """Median within-player coefficient of variation for fantasy-relevant players (mean >= 8, 8+ games)."""
    cv = dict(DEFAULT_CV)
    if not store.has("usage"):
        return cv
    u = store.read("usage").filter(pl.col("season") == season - 1)
    per_player = (
        u.group_by("player_id", "position")
        .agg(pl.len().alias("g"), pl.col("league_points").mean().alias("m"), pl.col("league_points").std().alias("s"))
        .filter((pl.col("g") >= 8) & (pl.col("m") >= 8))
        .group_by("position").agg((pl.col("s") / pl.col("m")).median().alias("cv"), pl.len().alias("n"))
    )
    for pos, value, n in per_player.iter_rows():
        if pos in cv and n >= 10:
            cv[pos] = float(value)
    return cv
