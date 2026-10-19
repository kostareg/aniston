"""nflverse data: weekly player stats/usage, expected fantasy points, ID map, schedules, ECR."""
import nflreadpy as nfl
import polars as pl

from aniston.scoring import nflverse_points_expr


def player_ids() -> pl.DataFrame:
    ids = nfl.load_ff_playerids().select("espn_id", "sleeper_id", "gsis_id", "fantasypros_id", "name", "position")
    return ids.with_columns(pl.col("espn_id").cast(pl.Int64, strict=False),
                            pl.col("sleeper_id").cast(pl.Utf8), pl.col("fantasypros_id").cast(pl.Utf8))


def weekly_usage(seasons: list[int], scoring: dict[str, float]) -> pl.DataFrame:
    stats = nfl.load_player_stats(seasons).filter(pl.col("season_type") == "REG")
    stats = stats.select(
        "player_id", "player_display_name", "position", "season", "week", "team", "opponent_team",
        "targets", "receptions", "carries", "attempts", "target_share", "air_yards_share", "wopr",
        "receiving_tds", "rushing_tds", "passing_tds",
        nflverse_points_expr(scoring).alias("league_points"),
    )
    opp = nfl.load_ff_opportunity(seasons=seasons).select(
        "player_id", "season", "week", "total_fantasy_points_exp", "total_fantasy_points")
    opp = opp.with_columns(pl.col("season").cast(pl.Int32, strict=False), pl.col("week").cast(pl.Int32, strict=False))
    stats = stats.with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    return stats.join(opp, on=["player_id", "season", "week"], how="left")


def schedules(season: int) -> pl.DataFrame:
    return nfl.load_schedules([season]).filter(pl.col("game_type") == "REG").select(
        "week", "home_team", "away_team", "home_score", "away_score")


def ecr_weekly() -> pl.DataFrame:
    rk = nfl.load_ff_rankings("week")
    return rk.select(pl.col("fantasypros_id").cast(pl.Utf8), "player_name", "pos", "ecr", "sd", "scrape_date")
