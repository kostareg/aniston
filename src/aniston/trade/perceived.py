"""Perceived value: what a league-mate sees in the ESPN app, in rest-of-season points.

Blend of ESPN's own ROS projection, season points per game, last-3-games points per game (recency bias),
and draft cost (this league's draft slot, else ESPN ADP), each scaled to the games the player has left.
Perceived value over replacement (PVOR) subtracts what a comparable free agent looks like.
"""
import numpy as np
import polars as pl

from aniston import store
from aniston.config import settings
from aniston.model.projections import Projections, bye_weeks, nfl_team

UNDRAFTED_PICK = 220.0
REPLACEMENT_FA_RANK = 2  # third-best free agent at the position looks like "what I could pick up"


def _espn_form(proj: Projections, current_week: int) -> pl.DataFrame:
    ew = store.read("espn_weeks").filter((pl.col("week") <= current_week) & pl.col("actual").is_not_null())
    return (
        ew.sort("week")
        .group_by("espn_id")
        .agg(pl.len().alias("gp"), pl.col("actual").mean().alias("ppg"), pl.col("actual").tail(3).mean().alias("last3"))
    )


def perceived_values(proj: Projections) -> pl.DataFrame:
    s = settings()
    meta = store.get_json("meta")
    p = proj.players.join(_espn_form(proj, meta["current_week"]), on="espn_id", how="left", maintain_order="left")

    byes = bye_weeks()
    games_left = np.array([
        sum(1 for w in proj.weeks if w not in byes.get(nfl_team(t), [])) for t in p["pro_team"].to_list()
    ], dtype=float)

    draft = store.read("draft").select("espn_id", pl.col("overall").cast(pl.Float64).alias("league_pick"))
    p = p.join(draft, on="espn_id", how="left", maintain_order="left")
    pick = p["league_pick"].fill_null(p["adp"]).fill_null(UNDRAFTED_PICK).to_numpy().astype(float)

    # Draft-cost curve: what a pick at this slot was "supposed" to score over a full season (ESPN preseason-ish
    # season projection), fit on log(pick). Monotone by construction of the fit range.
    has = (p["adp"].fill_null(0).to_numpy() > 0) & (p["espn_season_proj"].to_numpy() > 0)
    x = np.log(np.clip(p["adp"].fill_null(UNDRAFTED_PICK).to_numpy()[has], 1, 300))
    y = p["espn_season_proj"].to_numpy()[has]
    coef = np.polyfit(x, y, 1) if has.sum() > 20 else np.array([-40.0, 300.0])
    season_games = 17.0
    adp_ros = np.clip(np.polyval(coef, np.log(np.clip(pick, 1, 300))), 0, None) * games_left / season_games

    espn_ros = proj.espn.sum(1)
    ppg = p["ppg"].fill_null(np.nan).to_numpy().astype(float)
    last3 = p["last3"].fill_null(np.nan).to_numpy().astype(float)
    comps = np.vstack([espn_ros, ppg * games_left, last3 * games_left, adp_ros])
    weights = np.array([s.pv_w_espn_ros, s.pv_w_season_ppg, s.pv_w_last3_ppg, s.pv_w_adp])[:, None]
    avail = ~np.isnan(comps)
    pv = (np.nan_to_num(comps) * weights * avail).sum(0) / (weights * avail).sum(0)
    kdst = p["pos"].is_in(["K", "D/ST"]).to_numpy()
    pv = np.where(kdst, espn_ros, pv)

    fa = p["fantasy_team_id"].to_numpy() == 0
    pos = p["pos"].to_numpy()
    repl = {}
    for q in np.unique(pos):
        vals = np.sort(pv[fa & (pos == q)])[::-1]
        repl[q] = vals[min(REPLACEMENT_FA_RANK, len(vals) - 1)] if len(vals) else 0.0
    pvor = pv - np.array([repl[q] for q in pos])

    return p.select("espn_id", "name", "pos", "fantasy_team_id", "gp", "ppg", "last3", "league_pick", "adp").with_columns(
        pl.Series("games_left", games_left),
        pl.Series("espn_ros", espn_ros),
        pl.Series("adp_ros", adp_ros),
        pl.Series("pv", pv),
        pl.Series("pvor", pvor),
    )
