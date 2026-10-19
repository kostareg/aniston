"""ESPN Fantasy: league settings, teams, schedule, rosters, players (weekly projections + actuals), draft, trades."""
import json
from datetime import datetime
from functools import lru_cache

import polars as pl
from espn_api.football import League
from espn_api.football.constant import PRO_TEAM_MAP

from aniston.config import settings

POSITIONS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "D/ST"}
IR_SLOT = 21


@lru_cache
def league(year: int | None = None) -> League:
    s = settings()
    return League(league_id=s.league_id, year=year or s.season, espn_s2=s.espn_s2, swid=s.swid)


def my_team_id() -> int:
    s = settings()
    if s.my_team_id:
        return s.my_team_id
    for t in league().teams:
        if any((o.get("id") if isinstance(o, dict) else o) == s.swid for o in t.owners):
            return t.team_id
    raise RuntimeError("SWID does not own a team in this league; set MY_TEAM_ID in .env")


def fetch_meta() -> dict:
    lg = league()
    st = lg.settings
    slots = {k: v for k, v in st.position_slot_counts.items() if v and k not in ("BE", "IR")}
    weeks_per_period = {int(k): v for k, v in st.matchup_periods.items()}
    return {
        "league_name": st.name,
        "season": settings().season,
        "current_week": lg.current_week,
        "nfl_week": lg.nfl_week,
        "reg_season_weeks": st.reg_season_count,
        "last_week": max(w for ws in weeks_per_period.values() for w in ws),
        "playoff_teams": st.playoff_team_count,
        "tie_rule": st.playoff_seed_tie_rule,
        "trade_deadline": datetime.fromtimestamp(st.trade_deadline / 1000).isoformat() if st.trade_deadline else None,
        "faab": bool(st.faab),
        "slots": slots,
        "bench": st.position_slot_counts.get("BE", 0),
        "ir": st.position_slot_counts.get("IR", 0),
        "scoring": {x["abbr"]: x["points"] for x in st.scoring_format},
        "my_team_id": my_team_id(),
        "espn_playoff_pct": {t.team_id: t.playoff_pct for t in lg.teams},
    }


def fetch_teams() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "team_id": t.team_id,
                "name": t.team_name,
                "abbrev": t.team_abbrev,
                "owner": (t.owners[0].get("id") if t.owners and isinstance(t.owners[0], dict) else None),
                "wins": t.wins,
                "losses": t.losses,
                "ties": t.ties,
                "points_for": float(t.points_for),
                "points_against": float(t.points_against),
                "standing": t.standing,
                "acquisitions": t.acquisitions,
                "trades": t.trades,
                "drops": t.drops,
                "logo_url": t.logo_url,
            }
            for t in league().teams
        ]
    )


def fetch_schedule() -> pl.DataFrame:
    """One row per team-week of the regular season, with final scores where known."""
    rows = []
    for t in league().teams:
        for i, (opp, score, outcome) in enumerate(zip(t.schedule, t.scores, t.outcomes)):
            rows.append({"week": i + 1, "team_id": t.team_id, "opp_id": opp.team_id,
                         "score": float(score), "outcome": outcome})
    return pl.DataFrame(rows)


def fetch_live(week: int) -> pl.DataFrame:
    """Current-week live score and ESPN's live projected final for each team."""
    rows = []
    for b in league().box_scores(week):
        for side, other in (("home", "away"), ("away", "home")):
            team = getattr(b, f"{side}_team")
            if not team:
                continue
            rows.append({"week": week, "team_id": team.team_id, "opp_id": getattr(b, f"{other}_team").team_id,
                         "live_score": float(getattr(b, f"{side}_score") or 0),
                         "live_projected": float(getattr(b, f"{side}_projected") or 0)})
    return pl.DataFrame(rows)


def _kona(filt: dict, scoring_period: int) -> list[dict]:
    lg = league()
    data = lg.espn_request.league_get(
        params={"view": ["kona_player_info", "players_wl"], "scoringPeriodId": scoring_period},
        headers={"x-fantasy-filter": json.dumps(filt)},
    )
    return data["players"]


def fetch_players(current_week: int, last_week: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns (players, player_weeks). player_weeks has ESPN weekly projections and actuals in league scoring."""
    season = settings().season
    filt = {
        "players": {
            "limit": 2000,
            "filterActive": {"value": True},
            "sortPercOwned": {"sortPriority": 1, "sortAsc": False},
            "filterStatsForSourceIds": {"value": [0, 1]},
            "filterStatsForSplitTypeIds": {"value": [0, 1]},
            "filterStatsForScoringPeriodIds": {"value": list(range(0, max(last_week, 17) + 1))},
        }
    }
    raw = _kona(filt, current_week)
    roster_slot = _roster_slots()
    players, weeks = [], []
    for entry in raw:
        p = entry["player"]
        pos = POSITIONS.get(p.get("defaultPositionId"))
        if pos is None:
            continue
        own = p.get("ownership") or {}
        season_proj = season_actual = 0.0
        for s in p.get("stats", []):
            if s.get("seasonId") != season:
                continue
            pts = float(s.get("appliedTotal") or 0.0)
            if s["statSplitTypeId"] == 0:
                if s["statSourceId"] == 1:
                    season_proj = pts
                else:
                    season_actual = pts
            elif 1 <= s["scoringPeriodId"] <= max(last_week, 17):
                kind = "espn_proj" if s["statSourceId"] == 1 else "actual"
                weeks.append({"espn_id": p["id"], "week": s["scoringPeriodId"], "kind": kind, "points": pts})
        players.append({
            "espn_id": p["id"],
            "name": p["fullName"],
            "pos": pos,
            "pro_team": PRO_TEAM_MAP.get(p.get("proTeamId"), "FA"),
            "fantasy_team_id": entry.get("onTeamId") or 0,
            "lineup_slot_id": roster_slot.get(p["id"]),
            "injury_status": p.get("injuryStatus") or ("ACTIVE" if not p.get("injured") else "INJURED"),
            "pct_owned": float(own.get("percentOwned") or 0),
            "pct_started": float(own.get("percentStarted") or 0),
            "adp": float(own.get("averageDraftPosition") or 0) or None,
            "espn_season_proj": season_proj,
            "espn_season_actual": season_actual,
            "espn_pos_rank": (entry.get("ratings") or {}).get("0", {}).get("positionalRanking"),
            "tradeable": not entry.get("tradeLocked", False),
        })
    pw = (
        pl.DataFrame(weeks)
        .group_by("espn_id", "week", "kind").agg(pl.col("points").sum())  # sums split-game rows defensively
        .pivot(on="kind", index=["espn_id", "week"], values="points")
    )
    for col in ("espn_proj", "actual"):
        if col not in pw.columns:
            pw = pw.with_columns(pl.lit(None, pl.Float64).alias(col))
    return pl.DataFrame(players), pw.select("espn_id", "week", "espn_proj", "actual").sort("espn_id", "week")


def _roster_slots() -> dict[int, int]:
    data = league().espn_request.league_get(params={"view": "mRoster"})
    return {e["playerId"]: e["lineupSlotId"] for t in data["teams"] for e in t["roster"]["entries"]}


def fetch_draft() -> pl.DataFrame:
    n = len(league().teams)
    return pl.DataFrame([
        {"espn_id": d.playerId, "name": d.playerName, "team_id": d.team.team_id, "round": d.round_num,
         "round_pick": d.round_pick, "overall": (d.round_num - 1) * n + d.round_pick}
        for d in league().draft
    ])


def fetch_trades(year: int | None = None) -> pl.DataFrame:
    """One row per player moved in an executed trade."""
    rows = []
    for i, act in enumerate(league(year).recent_activity(size=1000, msg_type="TRADED")):
        for team, action, player, *_ in act.actions:
            if action == "TRADE_RECEIVED":
                rows.append({"trade_idx": i, "date": datetime.fromtimestamp(act.date / 1000), "team_id": team.team_id,
                             "espn_id": getattr(player, "playerId", None), "name": getattr(player, "name", str(player))})
    schema = {"trade_idx": pl.Int64, "date": pl.Datetime, "team_id": pl.Int64, "espn_id": pl.Int64, "name": pl.Utf8}
    return pl.DataFrame(rows, schema=schema)
