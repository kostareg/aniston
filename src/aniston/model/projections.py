"""Rest-of-season projection ensemble.

For every player and every future week we blend ESPN, Sleeper (RotoWire) and a usage model built from
nflverse expected fantasy points (xFP), shrunk toward the projection sources. Weights renormalize over
whichever sources exist for that player-week.
"""
from dataclasses import dataclass

import numpy as np
import polars as pl

from aniston import store
from aniston.config import settings

ESPN_TO_NFLVERSE = {"LAR": "LA", "WSH": "WAS"}
USAGE_PRIOR_GAMES = 3.0  # how many games of the projection-source prior the usage model starts with
XFP_WEIGHT = 0.6  # usage ppg = 0.6 * xFP/game + 0.4 * actual FP/game (xFP is the stickier signal)


def nfl_team(espn_abbrev: str) -> str:
    return ESPN_TO_NFLVERSE.get(espn_abbrev, espn_abbrev)


def bye_weeks() -> dict[str, list[int]]:
    sched = store.read("nfl_schedule")
    teams = set(sched["home_team"]) | set(sched["away_team"])
    byes = {t: [] for t in teams}
    for w in sorted(sched["week"].unique()):
        playing = set(sched.filter(pl.col("week") == w)["home_team"]) | set(sched.filter(pl.col("week") == w)["away_team"])
        for t in teams - playing:
            byes[t].append(int(w))
    return byes


def usage_table(season: int, through_week: int) -> pl.DataFrame:
    """Per-player season usage summary from nflverse, keyed by espn_id."""
    ids = store.read("ids").filter(pl.col("espn_id").is_not_null() & pl.col("gsis_id").is_not_null()).unique("gsis_id")
    u = store.read("usage").filter((pl.col("season") == season) & (pl.col("week") <= through_week))
    agg = (
        u.sort("week")
        .group_by("player_id")
        .agg(
            pl.len().alias("games"),
            pl.col("league_points").mean().alias("fp_pg"),
            pl.col("total_fantasy_points_exp").mean().alias("xfp_pg"),
            pl.col("league_points").tail(3).mean().alias("last3_pg"),
            pl.col("target_share").mean().alias("target_share"),
            pl.col("target_share").tail(3).mean().alias("target_share_l3"),
            pl.col("carries").mean().alias("carries_pg"),
            pl.col("carries").tail(3).mean().alias("carries_l3"),
            (pl.col("receiving_tds").fill_null(0) + pl.col("rushing_tds").fill_null(0)).sum().alias("skill_tds"),
        )
    )
    return agg.join(ids.select(pl.col("gsis_id").alias("player_id"), "espn_id"), on="player_id").drop("player_id")


@dataclass
class Projections:
    """Dense matrices over the player universe for the future weeks."""

    weeks: list[int]  # future weeks, e.g. [7..16]
    players: pl.DataFrame  # one row per universe player, row order == matrix row order
    ens: np.ndarray  # (P, W) ensemble projection
    espn: np.ndarray  # (P, W) ESPN projection (what other managers see)
    sources: dict[str, np.ndarray]  # per-source matrices with NaN where missing

    def idx(self) -> dict[int, int]:
        return {pid: i for i, pid in enumerate(self.players["espn_id"].to_list())}


def build(fa_per_pos: int = 15, through_week: int | None = None) -> Projections:
    """Assemble the universe (all rostered players + best free agents per position) and its projections."""
    s = settings()
    meta = store.get_json("meta")
    cur = meta["current_week"]
    weeks = list(range(cur + 1, meta["last_week"] + 1))
    through_week = through_week if through_week is not None else cur

    players = store.read("players")
    ew = store.read("espn_weeks").filter(pl.col("week").is_in(weeks))
    ids = store.read("ids").filter(pl.col("espn_id").is_not_null()).unique("espn_id").select("espn_id", "sleeper_id")
    sw = store.read("sleeper_weeks").filter(pl.col("week").is_in(weeks)) if store.has("sleeper_weeks") else None

    espn_ros = ew.group_by("espn_id").agg(pl.col("espn_proj").sum().alias("espn_ros"))
    players = players.join(espn_ros, on="espn_id", how="left").with_columns(pl.col("espn_ros").fill_null(0.0))
    rostered = players.filter(pl.col("fantasy_team_id") > 0)
    fas = (players.filter(pl.col("fantasy_team_id") == 0)
           .sort("espn_ros", descending=True).group_by("pos", maintain_order=True).head(fa_per_pos).select(players.columns))
    universe = pl.concat([rostered, fas]).join(ids, on="espn_id", how="left")
    usage = usage_table(s.season, through_week)
    universe = universe.join(usage, on="espn_id", how="left")

    P, W = universe.height, len(weeks)
    widx = {w: j for j, w in enumerate(weeks)}
    pidx = {pid: i for i, pid in enumerate(universe["espn_id"].to_list())}

    def dense(df: pl.DataFrame, key: str, col: str, keymap: dict) -> np.ndarray:
        m = np.full((P, W), np.nan)
        for k, w, v in df.select(key, "week", col).iter_rows():
            i = keymap.get(k)
            if i is not None and w in widx and v is not None:
                m[i, widx[w]] = v
        return m

    espn = np.nan_to_num(dense(ew, "espn_id", "espn_proj", pidx), nan=0.0)
    sources = {"espn": espn.copy()}
    if sw is not None and sw.height:
        sleeper_idx = {sid: pidx[e] for e, sid in universe.select("espn_id", "sleeper_id").iter_rows() if sid}
        sl = dense(sw, "sleeper_id", "sleeper_proj", sleeper_idx)
        sources["sleeper"] = sl

    # Usage model: per-game rate from xFP and actual FP, shrunk toward the projection sources' ROS rate,
    # spread over the weeks where the player is expected to play (ESPN shows > 0).
    plays = espn > 0
    src_rate = np.where(plays.sum(1) > 0, espn.sum(1) / np.maximum(plays.sum(1), 1), 0.0)
    g = universe["games"].fill_null(0).to_numpy().astype(float)
    xfp = universe["xfp_pg"].fill_null(np.nan).to_numpy().astype(float)
    fp = universe["fp_pg"].fill_null(np.nan).to_numpy().astype(float)
    obs = np.where(np.isnan(xfp), fp, XFP_WEIGHT * xfp + (1 - XFP_WEIGHT) * np.nan_to_num(fp, nan=0.0))
    usage_rate = np.where(np.isnan(obs), src_rate, (g * np.nan_to_num(obs) + USAGE_PRIOR_GAMES * src_rate) / (g + USAGE_PRIOR_GAMES))
    usage = np.where(plays, usage_rate[:, None], 0.0)
    usage[universe["pos"].is_in(["K", "D/ST"]).to_numpy()] = np.nan  # no usage model for K/DST
    usage[g == 0] = np.nan
    sources["usage"] = usage

    weights = {"espn": s.w_espn, "sleeper": s.w_sleeper, "usage": s.w_usage}
    num = np.zeros((P, W))
    den = np.zeros((P, W))
    for name, m in sources.items():
        have = ~np.isnan(m)
        # A source that is missing while ESPN says the player is out stays missing; otherwise contribute.
        num += np.where(have, weights[name] * np.nan_to_num(m), 0.0)
        den += np.where(have, weights[name], 0.0)
    ens = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
    ens = np.where(plays | (np.nan_to_num(sources.get("sleeper", np.zeros_like(espn))) > 0), ens, 0.0)

    universe = universe.with_columns(
        pl.Series("ens_ros", ens.sum(1)),
        pl.Series("usage_rate", usage_rate),
    )
    return Projections(weeks=weeks, players=universe, ens=ens, espn=espn, sources=sources)
