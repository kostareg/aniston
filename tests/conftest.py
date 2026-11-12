"""A small synthetic league in a temporary DuckDB, for tests that need the whole pipeline."""
from datetime import datetime, timedelta

import polars as pl
import pytest

TEAMS = [1, 2, 3, 4]
SLOTS = {"QB": 1, "RB": 1, "WR": 1, "RB/WR/TE": 1}
POS_BASE = {"QB": 18.0, "RB": 14.0, "WR": 13.0, "TE": 8.0}
ROSTER = ["QB", "QB", "RB", "RB", "WR", "WR", "TE"]  # 7 players a team: 4 starters + 3 bench
FREE = ["QB", "RB", "RB", "WR", "WR", "TE"]
NFL = ["BUF", "KC", "DAL", "SF", "DET", "PHI", "BAL", "GB"]


def build_league(current_week: int = 5, started: bool = False, reg_weeks: int = 8, last_week: int = 10,
                 playoff_teams: int = 4, deadline_in_days: int = 10) -> None:
    from aniston import store

    players, espn_weeks, sleeper, ids, usage, draft = [], [], [], [], [], []
    pid = 0
    for team, positions in [(t, ROSTER) for t in TEAMS] + [(0, FREE)]:
        for k, pos in enumerate(positions):
            pid += 1
            # Team 1 is strongest; ESPN underrates every player with an even id by 20% against Sleeper.
            mean = POS_BASE[pos] * (1.0 - 0.07 * k) * (1.15 - 0.08 * team if team else 0.55)
            espn_mean = mean * (0.8 if pid % 2 == 0 else 1.0)
            players.append({
                "espn_id": pid, "name": f"P{pid} {pos}", "pos": pos, "pro_team": NFL[pid % len(NFL)],
                "fantasy_team_id": team, "lineup_slot_id": 0, "injury_status": "ACTIVE", "pct_owned": 50.0,
                "pct_started": 30.0, "adp": float(pid), "espn_season_proj": espn_mean * 17,
                "espn_season_actual": mean * (current_week - 1), "espn_pos_rank": 1, "tradeable": True})
            ids.append({"espn_id": pid, "sleeper_id": f"s{pid}", "gsis_id": f"g{pid}", "fantasypros_id": None,
                        "name": f"P{pid} {pos}", "position": pos})
            if team:
                draft.append({"espn_id": pid, "name": f"P{pid} {pos}", "team_id": team, "round": k + 1,
                              "round_pick": team, "overall": k * len(TEAMS) + team})
            for w in range(1, last_week + 1):
                wobble = 1.0 + 0.1 * ((pid + w) % 3 - 1)
                played = w < current_week
                espn_weeks.append({"espn_id": pid, "week": w, "espn_proj": espn_mean,
                                   "actual": mean * wobble if played else None})
                sleeper.append({"sleeper_id": f"s{pid}", "week": w, "sleeper_proj": mean, "sleeper_company": "rw"})
                if played:
                    usage.append({"player_id": f"g{pid}", "player_display_name": f"P{pid} {pos}", "position": pos,
                                  "season": 2026, "week": w, "team": NFL[pid % len(NFL)],
                                  "opponent_team": NFL[(pid + w) % len(NFL)], "targets": 5, "receptions": 3,
                                  "carries": 8, "attempts": 0, "target_share": 0.2, "air_yards_share": 0.2,
                                  "wopr": 0.4, "receiving_tds": 0, "rushing_tds": 0, "passing_tds": 0,
                                  "league_points": mean * wobble, "total_fantasy_points_exp": mean,
                                  "total_fantasy_points": mean * wobble})

    schedule, live = [], []
    for w in range(1, reg_weeks + 1):
        pairs = [(1, 2), (3, 4)] if w % 2 else [(1, 3), (2, 4)]
        for a, b in pairs:
            for t, o in ((a, b), (b, a)):
                done = w < current_week
                score = (120.0 - 10 * t + w) if done else 0.0
                opp = (120.0 - 10 * o + w) if done else 0.0
                schedule.append({"week": w, "team_id": t, "opp_id": o, "score": score,
                                 "outcome": ("W" if score > opp else "L") if done else "U"})
                if w == current_week:
                    live.append({"week": w, "team_id": t, "opp_id": o, "live_score": 30.0 if started else 0.0,
                                 "live_projected": 110.0 - 8 * t})
    for w in range(reg_weeks + 1, current_week + 1):  # playoff weeks played or under way: 1v4, 2v3 then the final
        for t, o in ((1, 4), (4, 1), (2, 3), (3, 2)):
            if w < current_week:
                schedule.append({"week": w, "team_id": t, "opp_id": o, "score": 100.0 + 5 * t, "outcome": "U"})
            else:
                live.append({"week": w, "team_id": t, "opp_id": o, "live_score": 30.0 if started else 0.0,
                             "live_projected": 110.0 - 8 * t})

    played = [r for r in schedule if r["outcome"] != "U" and r["week"] <= reg_weeks]
    teams = [{"team_id": t, "name": f"Team {t}", "abbrev": f"T{t}", "owner": None,
              "wins": sum(r["outcome"] == "W" for r in played if r["team_id"] == t),
              "losses": sum(r["outcome"] == "L" for r in played if r["team_id"] == t), "ties": 0,
              "points_for": sum(r["score"] for r in played if r["team_id"] == t), "points_against": 0.0,
              "standing": t, "acquisitions": 0, "trades": 0, "drops": 0, "logo_url": ""} for t in TEAMS]
    kickoff = datetime.now() + timedelta(days=-1 if started else 1)
    nfl = [{"week": w, "gameday": (kickoff + timedelta(weeks=w - current_week)).strftime("%Y-%m-%d"),
            "gametime": "13:00", "home_team": NFL[i], "away_team": NFL[i + 1], "home_score": None, "away_score": None}
           for w in range(1, last_week + 1) for i in range(0, len(NFL), 2)]

    store.put_json("meta", {
        "league_name": "Test", "season": 2026, "current_week": current_week, "nfl_week": current_week,
        "reg_season_weeks": reg_weeks, "last_week": last_week, "playoff_teams": playoff_teams, "tie_rule": "",
        "trade_deadline": (datetime.now() + timedelta(days=deadline_in_days)).isoformat(), "faab": False,
        "slots": SLOTS, "bench": 3, "ir": 0, "scoring": {}, "my_team_id": 2,
        "espn_playoff_pct": {str(t): 50.0 for t in TEAMS}})
    store.write("teams", pl.DataFrame(teams))
    store.write("schedule", pl.DataFrame(schedule))
    store.write("live", pl.DataFrame(live, schema={"week": pl.Int64, "team_id": pl.Int64, "opp_id": pl.Int64,
                                                   "live_score": pl.Float64, "live_projected": pl.Float64}))
    store.write("players", pl.DataFrame(players))
    store.write("espn_weeks", pl.DataFrame(espn_weeks, schema_overrides={"actual": pl.Float64}))
    store.write("sleeper_weeks", pl.DataFrame(sleeper))
    store.write("ids", pl.DataFrame(ids, schema_overrides={"fantasypros_id": pl.Utf8}))
    store.write("usage", pl.DataFrame(usage))
    store.write("draft", pl.DataFrame(draft))
    store.write("nfl_schedule", pl.DataFrame(nfl, schema_overrides={"home_score": pl.Int32, "away_score": pl.Int32}))
    store.write("trades", pl.DataFrame(schema={"trade_idx": pl.Int64, "date": pl.Datetime, "team_id": pl.Int64,
                                               "espn_id": pl.Int64, "name": pl.Utf8}))


@pytest.fixture
def league(tmp_path, monkeypatch):
    """Points the store at an empty temp database; call the returned function to (re)build the league in it."""
    from aniston.config import settings

    for key, value in {"DB_PATH": str(tmp_path / "test.duckdb"), "ESPN_S2": "x", "SWID": "{x}", "LEAGUE_ID": "1",
                       "SEASON": "2026", "N_SIMS": "400", "NOTIFY_URL": ""}.items():
        monkeypatch.setenv(key, value)
    settings.cache_clear()
    yield build_league
    settings.cache_clear()
