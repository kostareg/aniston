"""Projection accuracy on completed weeks of this season, per source, plus fitted blend weights.

ESPN and Sleeper both expose their projections for past weeks; the usage model is rebuilt for each week
using only games before it (with that week's ESPN projection as its prior), so nothing leaks.
"""
import numpy as np
import polars as pl
from scipy.optimize import nnls

from aniston import store
from aniston.config import settings
from aniston.model.projections import USAGE_PRIOR_GAMES, XFP_WEIGHT


def _usage_projection(week: int, season: int, ids: pl.DataFrame) -> pl.DataFrame:
    u = store.read("usage").filter((pl.col("season") == season) & (pl.col("week") < week))
    agg = u.group_by("player_id").agg(pl.len().alias("g"), pl.col("league_points").mean().alias("fp"),
                                      pl.col("total_fantasy_points_exp").mean().alias("xfp"))
    return agg.join(ids, left_on="player_id", right_on="gsis_id").select("espn_id", "g", "fp", "xfp")


def run(min_week: int = 2) -> dict:
    s = settings()
    meta = store.get_json("meta")
    ids = store.read("ids").filter(pl.col("espn_id").is_not_null() & pl.col("gsis_id").is_not_null()) \
        .unique("gsis_id").select("gsis_id", "espn_id", "sleeper_id")
    pos = store.read("players").select("espn_id", "pos")
    ew = store.read("espn_weeks").filter(pl.col("actual").is_not_null() & (pl.col("week") < meta["current_week"]))
    sw = store.read("sleeper_weeks")
    frames = []
    for week in range(min_week, meta["current_week"]):
        base = ew.filter(pl.col("week") == week).join(pos, on="espn_id").filter(pl.col("pos").is_in(["QB", "RB", "WR", "TE"]))
        base = base.join(ids.select("espn_id", "sleeper_id").unique("espn_id"), on="espn_id", how="left")
        base = base.join(sw.filter(pl.col("week") == week).select("sleeper_id", "sleeper_proj"), on="sleeper_id", how="left")
        use = _usage_projection(week, s.season, ids)
        base = base.join(use, on="espn_id", how="left").with_columns(
            pl.when(pl.col("g").is_null()).then(pl.col("espn_proj")).otherwise(
                (pl.col("g") * (XFP_WEIGHT * pl.col("xfp").fill_null(pl.col("fp")) + (1 - XFP_WEIGHT) * pl.col("fp"))
                 + USAGE_PRIOR_GAMES * pl.col("espn_proj")) / (pl.col("g") + USAGE_PRIOR_GAMES)
            ).alias("usage_proj"))
        frames.append(base)
    df = pl.concat(frames).filter((pl.col("espn_proj") > 0) & pl.col("sleeper_proj").is_not_null())

    y = df["actual"].to_numpy()
    X = df.select("espn_proj", "sleeper_proj", "usage_proj").to_numpy()
    ens_w = np.array([s.w_espn, s.w_sleeper, s.w_usage])
    current = X @ (ens_w / ens_w.sum())
    w_fit, _ = nnls(X, y)
    fitted = w_fit / w_fit.sum() if w_fit.sum() > 0 else ens_w
    mae = lambda pred: float(np.mean(np.abs(pred - y)))  # noqa: E731
    by_pos = (df.with_columns(pl.Series("ensemble", current))
              .group_by("pos").agg(*(( (pl.col(c) - pl.col("actual")).abs().mean().alias(c))
                                     for c in ("espn_proj", "sleeper_proj", "usage_proj", "ensemble")),
                                   pl.len().alias("n")).sort("pos"))
    result = {
        "n": int(len(y)),
        "weeks": [min_week, meta["current_week"] - 1],
        "mae": {"espn": mae(X[:, 0]), "sleeper": mae(X[:, 1]), "usage": mae(X[:, 2]),
                "ensemble_current": mae(current), "ensemble_fitted": mae(X @ fitted)},
        "fitted_weights": {"espn": float(fitted[0]), "sleeper": float(fitted[1]), "usage": float(fitted[2])},
        "fitted_scale": float(w_fit.sum()),
    }
    store.write("backtest_by_pos", by_pos)
    store.put_json("backtest", result)
    return result
