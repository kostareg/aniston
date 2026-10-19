"""Pull every source into DuckDB. Each non-ESPN source fails independently."""
import logging
import traceback

import polars as pl

from aniston import store
from aniston.config import settings
from aniston.ingest import espn, nflverse, sleeper

log = logging.getLogger(__name__)


def refresh() -> dict:
    s = settings()
    meta = espn.fetch_meta()
    store.put_json("meta", meta)
    report = {"meta": "ok"}

    store.write("teams", espn.fetch_teams())
    store.write("schedule", espn.fetch_schedule())
    store.write("live", espn.fetch_live(meta["current_week"]))
    players, player_weeks = espn.fetch_players(meta["current_week"], meta["last_week"])
    store.write("players", players)
    store.write("espn_weeks", player_weeks)
    store.write("draft", espn.fetch_draft())
    store.write("trades", espn.fetch_trades())
    report["espn"] = f"{players.height} players"

    sources = {
        "ids": lambda: nflverse.player_ids(),
        "usage": lambda: nflverse.weekly_usage([s.season - 1, s.season], meta["scoring"]),
        "nfl_schedule": lambda: nflverse.schedules(s.season),
        "ecr": lambda: nflverse.ecr_weekly(),
        "sleeper_weeks": lambda: sleeper.fetch_projections(
            s.season, range(1, meta["last_week"] + 1), meta["scoring"]),
    }
    for name, fn in sources.items():
        try:
            df: pl.DataFrame = fn()
            store.write(name, df)
            report[name] = f"{df.height} rows"
        except Exception as e:  # noqa: BLE001 - a broken source must not block the rest
            log.warning("source %s failed: %s", name, traceback.format_exc())
            report[name] = f"FAILED: {e}"
    store.put_json("refresh_report", report)
    return report
