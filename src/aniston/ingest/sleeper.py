"""Sleeper's free weekly projections (sourced from RotoWire). Undocumented endpoint, so failures are non-fatal."""
import httpx
import polars as pl

from aniston.scoring import score_sleeper

URL = "https://api.sleeper.app/projections/nfl/{season}/{week}"


def fetch_projections(season: int, weeks: range, scoring: dict[str, float]) -> pl.DataFrame:
    rows = []
    with httpx.Client(timeout=30) as client:
        for week in weeks:
            r = client.get(URL.format(season=season, week=week),
                           params={"season_type": "regular", "position[]": ["QB", "RB", "WR", "TE"]})
            r.raise_for_status()
            for x in r.json():
                stats = x.get("stats") or {}
                if not stats.get("pts_ppr"):
                    continue
                rows.append({"sleeper_id": str(x["player_id"]), "week": week,
                             "sleeper_proj": score_sleeper(stats, scoring), "sleeper_company": x.get("company")})
    return pl.DataFrame(rows, schema={"sleeper_id": pl.Utf8, "week": pl.Int64, "sleeper_proj": pl.Float64,
                                      "sleeper_company": pl.Utf8})
