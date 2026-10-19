import math

import polars as pl
import streamlit as st

from aniston.ui import load, meta, page, ready, team_names, trade_context

page("Matchup", "🆚")
st.title("Matchup")
if not ready():
    st.stop()

m = meta()
me, names = m["my_team_id"], team_names()
live = load("live")


def win_prob(mu_a, sd_a, mu_b, sd_b):
    sd = math.hypot(sd_a, sd_b)
    if sd == 0:  # every game is final: the result is decided
        return 1.0 if mu_a > mu_b else 0.0 if mu_a < mu_b else 0.5
    return 0.5 * (1 + math.erf((mu_a - mu_b) / (sd * math.sqrt(2))))


row = live.filter(pl.col("team_id") == me)
if row.height:
    r = row.row(0, named=True)
    o = live.filter(pl.col("team_id") == r["opp_id"]).row(0, named=True)
    sd = lambda x: 0.35 * max(x["live_projected"] - x["live_score"], 0)  # noqa: E731
    p = win_prob(r["live_projected"], sd(r), o["live_projected"], sd(o))
    st.subheader(f"Week {m['current_week']} (live): you vs {names[r['opp_id']]}")
    c = st.columns(3)
    c[0].metric("You", f"{r['live_score']:.1f}", f"proj. {r['live_projected']:.1f}", delta_color="off")
    c[1].metric(names[r["opp_id"]], f"{o['live_score']:.1f}", f"proj. {o['live_projected']:.1f}", delta_color="off")
    c[2].metric("Win probability", f"{p * 100:.0f}%")

ctx = trade_context()
sched = load("schedule")
st.subheader("Upcoming regular-season matchups")
rows = []
for w in range(m["current_week"] + 1, m["reg_season_weeks"] + 1):
    opp = sched.filter((pl.col("week") == w) & (pl.col("team_id") == me))["opp_id"][0]
    j = ctx.proj.weeks.index(w)
    mu_me, var_me = ctx.sim.spec.evaluate(ctx.proj.ens[list(ctx.base[me]), j], ctx.pos[list(ctx.base[me])], with_var=True)
    r_opp = list(ctx.base[opp])
    mu_o, var_o = ctx.sim.spec.evaluate(ctx.proj.ens[r_opp, j], ctx.pos[r_opp], with_var=True)
    rows.append({"week": w, "opponent": names[opp], "you": float(mu_me), "them": float(mu_o),
                 "win_prob": 100 * win_prob(float(mu_me), math.sqrt(var_me), float(mu_o), math.sqrt(var_o))})
st.dataframe(pl.DataFrame(rows), hide_index=True, width="stretch", column_config={
    "week": "Week", "opponent": "Opponent",
    "you": st.column_config.NumberColumn("You (proj.)", format="%.1f"),
    "them": st.column_config.NumberColumn("Them (proj.)", format="%.1f"),
    "win_prob": st.column_config.ProgressColumn("Win prob.", format="%.0f%%", min_value=0, max_value=100)})
st.caption("Both sides at their optimal lineup by consensus projections; weekly score volatility from last season's data.")
