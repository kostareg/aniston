"""Compute every model output the dashboard reads and persist it to DuckDB."""
from datetime import datetime

import numpy as np
import polars as pl

from aniston import store
from aniston.model.signals import player_signals
from aniston.trade.engine import TradeContext


def team_position_strength(ctx: TradeContext) -> pl.DataFrame:
    """Average weekly starting points each team gets from each lineup position (consensus projections)."""
    rows = []
    spec = ctx.sim.spec
    for t, roster in ctx.base.items():
        r = np.array(roster)
        mat = ctx.proj.ens[r].T  # (W, n)
        starters = np.zeros(len(r))
        for w in range(mat.shape[0]):
            starters += spec.starters(mat[w], ctx.pos[r]) * mat[w]
        for q in np.unique(ctx.pos[r]):
            rows.append({"team_id": t, "pos": q, "pts_per_week": float(starters[ctx.pos[r] == q].sum() / mat.shape[0])})
    df = pl.DataFrame(rows)
    return df.with_columns((pl.col("pts_per_week") - pl.col("pts_per_week").mean().over("pos")).alias("vs_avg"))


def waiver_targets(ctx: TradeContext, top_sim: int = 12) -> pl.DataFrame:
    me = ctx.me
    base = ctx.base[me]
    base_v = ctx.true.value(base)
    rows = []
    for fa in ctx.free_agents:
        roster, dropped, _ = ctx.true.legalize(base + (fa,), [], protected={fa}, allow_pickup=False)
        if fa not in roster:
            continue
        rows.append({"i": fa, "gain": ctx.true.value(roster) - base_v, "drop": dropped, "roster": roster})
    rows.sort(key=lambda r: -r["gain"])
    out = []
    for k, r in enumerate(rows[:40]):
        d_playoffs = d_title = None
        if k < top_sim and r["gain"] > 0:
            rosters = dict(ctx.base)
            rosters[me] = r["roster"]
            res = ctx.sim.run(rosters)
            mi = ctx.sim.tix[me]
            d_playoffs = float(res.p_playoffs[mi] - ctx.base_result.p_playoffs[mi])
            d_title = float(res.p_title[mi] - ctx.base_result.p_title[mi])
        out.append({"espn_id": int(ctx.players["espn_id"][r["i"]]), "name": ctx.names[r["i"]], "pos": ctx.pos[r["i"]],
                    "ros_lineup_gain": r["gain"], "drop": ", ".join(ctx.names[i] for i in r["drop"]),
                    "d_playoffs": d_playoffs, "d_title": d_title})
    return pl.DataFrame(out)


def projection_weeks(ctx: TradeContext) -> pl.DataFrame:
    p = ctx.proj
    ids = p.players["espn_id"].to_numpy()
    rows = []
    for j, w in enumerate(p.weeks):
        rows.append(pl.DataFrame({
            "espn_id": ids, "week": w, "ens": p.ens[:, j], "espn": p.espn[:, j],
            "sleeper": p.sources["sleeper"][:, j] if "sleeper" in p.sources else np.nan,
            "usage": p.sources["usage"][:, j],
        }))
    return pl.concat(rows)


def analyze(with_trades: bool = True, workers: int = 1) -> dict:
    ctx = TradeContext()
    teams = store.read("teams").select("team_id", "name")
    store.write("sim_baseline", ctx.base_result.frame().join(teams, on="team_id"))
    sig = player_signals(ctx.proj).join(
        ctx.pv.select("espn_id", "gp", "ppg", "last3", "league_pick", "adp", "games_left", "pv", "pvor"),
        on="espn_id", how="left", maintain_order="left")
    store.write("player_model", sig)
    store.write("projection_weeks", projection_weeks(ctx))
    store.write("team_strength", team_position_strength(ctx))
    store.write("waivers", waiver_targets(ctx))
    report = {"players": sig.height}
    try:
        from aniston.backtest import run as backtest

        report["backtest_mae"] = round(backtest()["mae"]["ensemble_current"], 3)
    except Exception as e:  # noqa: BLE001 - backtest is diagnostic only
        report["backtest_mae"] = f"failed: {e}"
    if with_trades:
        trades = ctx.frame(ctx.find(workers=workers))
        store.write("trade_ideas", trades)
        report["trades"] = trades.height
    store.put_json("analysis", {"at": datetime.now().isoformat(timespec="minutes"), **report})
    return report
