"""Per-player signals that point at mispricing: TD/efficiency luck, usage trends, source disagreement,
schedule strength (rest of season and fantasy playoff weeks)."""
import numpy as np
import polars as pl

from aniston import store
from aniston.config import settings
from aniston.model.projections import Projections, nfl_team


def defense_vs_position(season: int, through_week: int) -> pl.DataFrame:
    """Fantasy points a defense allows to each position per game, relative to league average (1.0 = average)."""
    u = store.read("usage").filter((pl.col("season") == season) & (pl.col("week") <= through_week)
                                   & pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
    per_game = u.group_by("opponent_team", "position", "week").agg(pl.col("league_points").sum())
    allowed = per_game.group_by("opponent_team", "position").agg(pl.col("league_points").mean().alias("fpa"),
                                                                 pl.len().alias("g"))
    avg = allowed.group_by("position").agg(pl.col("fpa").mean().alias("avg"))
    # Shrink toward 1.0: few games of defensive data are noisy.
    return allowed.join(avg, on="position").with_columns(
        (1 + (pl.col("fpa") / pl.col("avg") - 1) * pl.col("g") / (pl.col("g") + 4)).alias("factor")
    ).select(pl.col("opponent_team").alias("defense"), "position", "factor")


def schedule_strength(proj: Projections, weeks: list[int]) -> np.ndarray:
    """Mean opponent factor (>1 = softer matchups) over `weeks` for every universe player."""
    s = settings()
    meta = store.get_json("meta")
    dvp = defense_vs_position(s.season, meta["current_week"])
    factor = {(d, p): f for d, p, f in dvp.iter_rows()}
    sched = store.read("nfl_schedule").filter(pl.col("week").is_in(weeks))
    opp: dict[tuple[str, int], str] = {}
    for w, h, a in sched.select("week", "home_team", "away_team").iter_rows():
        opp[(h, w)], opp[(a, w)] = a, h
    out = np.full(proj.players.height, np.nan)
    for i, (team, pos) in enumerate(proj.players.select("pro_team", "pos").iter_rows()):
        vals = [factor.get((opp[(nfl_team(team), w)], pos)) for w in weeks if (nfl_team(team), w) in opp]
        vals = [v for v in vals if v is not None]
        if vals:
            out[i] = float(np.mean(vals))
    return out


def player_signals(proj: Projections) -> pl.DataFrame:
    meta = store.get_json("meta")
    p = proj.players
    reg_last = meta["reg_season_weeks"]
    playoff_weeks = list(range(reg_last + 1, meta["last_week"] + 1))
    espn_ros = proj.espn.sum(1)
    ens_ros = proj.ens.sum(1)
    sleeper_ros = np.nansum(proj.sources["sleeper"], 1) if "sleeper" in proj.sources else np.full(p.height, np.nan)

    fa = p["fantasy_team_id"].to_numpy() == 0
    pos = p["pos"].to_numpy()
    repl = {}
    for q in np.unique(pos):
        vals = np.sort(ens_ros[fa & (pos == q)])[::-1]
        repl[q] = vals[min(2, len(vals) - 1)] if len(vals) else 0.0
    vorp = ens_ros - np.array([repl[q] for q in pos])

    df = p.select("espn_id", "name", "pos", "pro_team", "fantasy_team_id", "injury_status", "pct_owned",
                  "games", "fp_pg", "xfp_pg", "last3_pg", "target_share", "target_share_l3", "carries_pg",
                  "carries_l3", "skill_tds").with_columns(
        pl.Series("espn_ros", espn_ros),
        pl.Series("sleeper_ros", sleeper_ros),
        pl.Series("ens_ros", ens_ros),
        pl.Series("vorp", vorp),
        pl.Series("sos_ros", schedule_strength(proj, proj.weeks)),
        pl.Series("sos_playoffs", schedule_strength(proj, playoff_weeks)),
    )
    df = df.with_columns(
        (pl.col("fp_pg") - pl.col("xfp_pg")).alias("luck_pg"),
        ((pl.col("espn_ros") - pl.col("ens_ros")) / pl.col("ens_ros").clip(lower_bound=1)).alias("espn_vs_consensus"),
        (pl.col("target_share_l3") - pl.col("target_share")).alias("target_share_trend"),
        (pl.col("carries_l3") - pl.col("carries_pg")).alias("carries_trend"),
    )
    # Buy low: underperforming expected points or trending up in usage, and ESPN not already pricing it in.
    # Sell high: scoring above expectation, ESPN projecting above consensus.
    return df.with_columns(
        pl.when((pl.col("luck_pg") <= -2.5) | (pl.col("target_share_trend") >= 0.04) | (pl.col("carries_trend") >= 3))
        .then(pl.lit("buy-low"))
        .when((pl.col("luck_pg") >= 3) | (pl.col("espn_vs_consensus") >= 0.15))
        .then(pl.lit("sell-high"))
        .otherwise(pl.lit(None, pl.Utf8))
        .alias("flag")
    )
