"""Apply the league's ESPN scoring settings to stat lines from other sources."""

# ESPN scoring abbreviation -> (sleeper stat key, nflverse player_stats column)
OFFENSE_MAP = {
    "PY": ("pass_yd", "passing_yards"),
    "PTD": ("pass_td", "passing_tds"),
    "INTT": ("pass_int", "passing_interceptions"),
    "RY": ("rush_yd", "rushing_yards"),
    "RTD": ("rush_td", "rushing_tds"),
    "REC": ("rec", "receptions"),
    "REY": ("rec_yd", "receiving_yards"),
    "RETD": ("rec_td", "receiving_tds"),
    "FUML": ("fum_lost", None),  # nflverse splits lost fumbles across three columns; handled below
    "2PC": ("pass_2pt", "passing_2pt_conversions"),
    "2PR": ("rush_2pt", "rushing_2pt_conversions"),
    "2PRE": ("rec_2pt", "receiving_2pt_conversions"),
    "KRTD": ("kr_td", None),
    "PRTD": ("pr_td", None),
}

NFLVERSE_FUMBLES_LOST = ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost")


def score_sleeper(stats: dict, scoring: dict[str, float]) -> float:
    return sum(scoring.get(abbr, 0.0) * float(stats.get(key) or 0.0) for abbr, (key, _) in OFFENSE_MAP.items())


def nflverse_points_expr(scoring: dict[str, float]):
    """Polars expression computing league points from an nflverse player_stats row."""
    import polars as pl

    expr = pl.lit(0.0)
    for abbr, (_, col) in OFFENSE_MAP.items():
        if col and abbr in scoring:
            expr = expr + pl.col(col).fill_null(0) * scoring[abbr]
    if "FUML" in scoring:
        expr = expr + sum(pl.col(c).fill_null(0) for c in NFLVERSE_FUMBLES_LOST) * scoring["FUML"]
    return expr
