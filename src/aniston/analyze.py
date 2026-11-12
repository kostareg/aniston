"""Compute every model output the dashboard reads and persist it to DuckDB."""
from datetime import datetime

import numpy as np
import polars as pl

from aniston import store
from aniston.config import deadline_passed
from aniston.model.projections import STREAM_POS, STREAM_WEEKS
from aniston.model.signals import player_signals
from aniston.model.startsit import lineup_advice
from aniston.trade.engine import TradeContext


def team_position_strength(ctx: TradeContext) -> pl.DataFrame:
    """Average weekly points each team gets from each lineup slot (consensus projections). Flex slots are their own
    rows, so a team starting a third RB at flex doesn't look strong at RB and weak at WR next to teams flexing a WR."""
    rows = []
    spec = ctx.sim.spec
    slot_names = list(spec.dedicated) + [spec.flex_names[e] for e, _ in spec.flex]
    for t, roster in ctx.base.items():
        r = np.array(roster)
        mat = ctx.proj.ens[r].T  # (W, n)
        pts = dict.fromkeys(slot_names, 0.0)
        for w in range(mat.shape[0]):
            for slot, v in zip(spec.slots(mat[w], ctx.pos[r]), mat[w]):
                if slot:
                    pts[slot] += v
        rows += [{"team_id": t, "pos": q, "pts_per_week": v / max(mat.shape[0], 1)} for q, v in pts.items()]
    df = pl.DataFrame(rows)
    return df.with_columns((pl.col("pts_per_week") - pl.col("pts_per_week").mean().over("pos")).alias("vs_avg"))


def waiver_targets(ctx: TradeContext, top_sim: int = 12) -> pl.DataFrame:
    me = ctx.me
    base = ctx.base[me]
    base_v = ctx.true.value(base)
    is_playoff = np.array(ctx.proj.weeks) > ctx.meta["reg_season_weeks"]
    rows = []
    for fa in ctx.free_agents:
        roster, dropped, _ = ctx.true.legalize(base + (fa,), [], protected={fa}, allow_pickup=False)
        if fa not in roster:
            continue
        by_week = ctx.true.weekly(roster) - ctx.true.weekly(base)
        rows.append({"i": fa, "gain": ctx.true.value(roster) - base_v, "drop": dropped, "roster": roster,
                     "playoff_gain": float(by_week[is_playoff].sum())})
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
                    "ros_lineup_gain": r["gain"], "playoff_gain": r["playoff_gain"],
                    "drop": ", ".join(ctx.names[i] for i in r["drop"]),
                    "d_playoffs": d_playoffs, "d_title": d_title})
    schema = {"espn_id": pl.Int64, "name": pl.Utf8, "pos": pl.Utf8, "ros_lineup_gain": pl.Float64,
              "playoff_gain": pl.Float64, "drop": pl.Utf8, "d_playoffs": pl.Float64, "d_title": pl.Float64}
    return pl.DataFrame(out, schema=schema)


def _week_vectors(ctx: TradeContext) -> dict[int, np.ndarray]:
    """Consensus projection (P,) for the current week and every later one."""
    return {ctx.proj.cur_week: ctx.proj.ens_cur, **{w: ctx.proj.ens[:, j] for j, w in enumerate(ctx.proj.weeks)}}


def streamers(ctx: TradeContext, per_week: int = 5) -> pl.DataFrame:
    """Best free agents at the streaming positions for each of the next few weeks, against who I'd start there."""
    vec = _week_vectors(ctx)
    mine = list(ctx.base[ctx.me])
    rows = []
    for w in sorted(vec)[:STREAM_WEEKS]:
        v = vec[w]
        for q in STREAM_POS:
            have = [i for i in mine if ctx.pos[i] == q]
            starter = max(have, key=lambda i: v[i]) if have else None
            fas = sorted((i for i in ctx.free_agents if ctx.pos[i] == q), key=lambda i: -v[i])[:per_week]
            for i in fas:
                rows.append({"week": w, "pos": q, "name": ctx.names[i], "pro_team": ctx.players["pro_team"][i],
                             "proj": float(v[i]), "my_starter": ctx.names[starter] if starter is not None else None,
                             "my_proj": float(v[starter]) if starter is not None else 0.0})
    schema = {"week": pl.Int64, "pos": pl.Utf8, "name": pl.Utf8, "pro_team": pl.Utf8, "proj": pl.Float64,
              "my_starter": pl.Utf8, "my_proj": pl.Float64}
    return pl.DataFrame(rows, schema=schema).with_columns((pl.col("proj") - pl.col("my_proj")).alias("gain"))


def blocks(ctx: TradeContext, top: int = 8) -> pl.DataFrame:
    """Free agents who would raise this week's opponent's lineup: worth a bench spot just to keep them away."""
    schema = {"opponent": pl.Utf8, "name": pl.Utf8, "pos": pl.Utf8, "opp_gain": pl.Float64, "my_gain": pl.Float64,
              "drop": pl.Utf8}
    live = store.read("live").filter(pl.col("team_id") == ctx.me) if store.has("live") else None
    if live is None or not live.height or live["opp_id"][0] not in ctx.base:
        return pl.DataFrame(schema=schema)
    opp = live["opp_id"][0]
    v, spec = ctx.proj.ens_cur, ctx.sim.spec

    def gain(team: int, fa: int) -> float:
        r = np.array(ctx.base[team])
        with_fa = np.append(r, fa)
        return float(spec.evaluate(v[with_fa], ctx.pos[with_fa]) - spec.evaluate(v[r], ctx.pos[r]))

    drop = ctx.names[ctx.true.drop_candidate(ctx.base[ctx.me])]
    by_week = sorted(ctx.free_agents, key=lambda i: -v[i])
    seen: dict[str, int] = {}
    rows = []
    for fa in by_week:  # the opponent only cares about the few best options at each position this week
        if seen.get(ctx.pos[fa], 0) >= 5:
            continue
        seen[ctx.pos[fa]] = seen.get(ctx.pos[fa], 0) + 1
        g = gain(opp, fa)
        if g > 0.5:
            rows.append({"opponent": ctx.teams[opp], "name": ctx.names[fa], "pos": ctx.pos[fa], "opp_gain": g,
                         "my_gain": gain(ctx.me, fa), "drop": drop})
    rows.sort(key=lambda r: -r["opp_gain"])
    return pl.DataFrame(rows[:top], schema=schema)


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
    schema = {"espn_id": pl.Int64, "week": pl.Int64, "ens": pl.Float64, "espn": pl.Float64, "sleeper": pl.Float64,
              "usage": pl.Float64}
    return pl.concat(rows).cast(schema) if rows else pl.DataFrame(schema=schema)


def analyze(with_trades: bool = True, workers: int = 1) -> dict:
    ctx = TradeContext()
    now, week = datetime.now(), ctx.meta["current_week"]
    teams = store.read("teams").select("team_id", "name")
    sim = ctx.base_result.frame().join(teams, on="team_id").with_columns(
        pl.lit(ctx.sim.next_week, pl.Int64).alias("next_week"))
    store.write("sim_baseline", sim)
    sig = player_signals(ctx.proj).join(
        ctx.pv.select("espn_id", "gp", "ppg", "last3", "league_pick", "adp", "games_left", "pv", "pvor"),
        on="espn_id", how="left", maintain_order="left")
    store.write("player_model", sig)
    weeks = projection_weeks(ctx)
    store.write("projection_weeks", weeks)
    store.write("team_strength", team_position_strength(ctx))
    store.write("waivers", waiver_targets(ctx))
    store.write("streamers", streamers(ctx))
    store.write("blocks", blocks(ctx))
    lineup, swaps, lineup_info = lineup_advice(ctx)
    store.write("my_lineup", lineup)
    store.write("start_sit", swaps)
    store.put_json("lineup_info", lineup_info)
    # History: what the model believed on this day, to score later against what happened (see backtest).
    store.snapshot("sim", sim, now, week)
    store.snapshot("proj", weeks, now, week)
    report = {"players": sig.height}
    try:
        from aniston.backtest import run as backtest

        report["backtest_mae"] = round(backtest()["mae"]["ensemble_current"], 3)
    except Exception as e:  # noqa: BLE001 - backtest is diagnostic only
        report["backtest_mae"] = f"failed: {e}"
    if with_trades and deadline_passed(ctx.meta, now):
        report["trades"] = "skipped: trade deadline has passed"
    elif with_trades:
        trades = ctx.frame(ctx.find(workers=workers))
        store.write("trade_ideas", trades)
        store.snapshot("trades", trades, now, week)
        report["trades"] = trades.height
    store.put_json("analysis", {"at": now.isoformat(timespec="minutes"), **report})
    return report
