import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from aniston.ui import DIVERGING, load, meta, page, ready, team_names, trade_context

page("My Team", "🧢")
st.title("My Team")
if not ready():
    st.stop()

m = meta()
me = m["my_team_id"]
pm = load("player_model")
pw = load("projection_weeks")
next_week = m["current_week"] + 1

ctx = trade_context()
roster = list(ctx.base[me])
j = ctx.proj.weeks.index(next_week)
starters = ctx.sim.spec.starters(ctx.proj.ens[roster, j], ctx.pos[roster])
start_ids = {int(ctx.players["espn_id"][i]) for i, s in zip(roster, starters) if s}

mine = pm.filter(pl.col("espn_id").is_in([int(ctx.players["espn_id"][i]) for i in roster]))
nw = pw.filter(pl.col("week") == next_week).select("espn_id", pl.col("ens").alias("next_week"))
mine = mine.join(nw, on="espn_id", how="left").with_columns(pl.col("espn_id").is_in(list(start_ids)).alias("start"))

st.subheader(f"Optimal lineup for week {next_week}")
order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "D/ST": 4, "K": 5}
lineup = mine.sort(["start", pl.col("pos").replace_strict(order, default=9), "next_week"], descending=[True, False, True])
st.dataframe(
    lineup.select("start", "name", "pos", "pro_team", "injury_status", "next_week", "ens_ros", "espn_ros", "vorp",
                  "flag", "luck_pg", "sos_ros", "sos_playoffs"),
    hide_index=True, width="stretch",
    column_config={
        "start": st.column_config.CheckboxColumn("Start"), "name": "Player", "pos": "Pos", "pro_team": "NFL",
        "injury_status": "Status",
        "next_week": st.column_config.NumberColumn(f"Wk {next_week} proj", format="%.1f"),
        "ens_ros": st.column_config.NumberColumn("ROS (consensus)", format="%.0f"),
        "espn_ros": st.column_config.NumberColumn("ROS (ESPN)", format="%.0f"),
        "vorp": st.column_config.NumberColumn("ROS over replacement", format="%+.0f"),
        "flag": "Signal",
        "luck_pg": st.column_config.NumberColumn("Pts/g vs expected", format="%+.1f"),
        "sos_ros": st.column_config.NumberColumn("SoS ROS", format="%.2f", help=">1 = softer opponents"),
        "sos_playoffs": st.column_config.NumberColumn("SoS playoffs", format="%.2f"),
    },
)

st.subheader("Positional strength vs the league")
st.caption("Average weekly starting points each team gets from each position (consensus ROS), relative to the league "
           "average. Red = below average, blue = above.")
ts = load("team_strength").filter(~pl.col("pos").is_in(["K", "D/ST"]))
names = team_names()
piv = ts.pivot(on="pos", index="team_id", values="vs_avg").fill_null(0)
pos_cols = [p for p in ["QB", "RB", "WR", "TE"] if p in piv.columns]
piv = piv.with_columns(pl.sum_horizontal(pos_cols).alias("total")).sort("total", descending=True)
z = piv.select(pos_cols).to_numpy()
lim = float(np.abs(z).max()) or 1
labels = [("▶ " if t == me else "") + names[t] for t in piv["team_id"]]
fig = go.Figure(go.Heatmap(z=z, x=pos_cols, y=labels, colorscale=DIVERGING, zmin=-lim, zmax=lim, xgap=2, ygap=2,
                           text=np.round(z, 1), texttemplate="%{text:+.1f}",
                           hovertemplate="%{y} %{x}: %{z:+.1f} pts/week vs avg<extra></extra>",
                           colorbar=dict(title="pts/wk")))
fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, width="stretch")

st.subheader("Bye-week and depth check")
mat = ctx.proj.ens[roster]
weekly = ctx.sim.spec.evaluate(mat.T, ctx.pos[roster])
fig2 = go.Figure(go.Bar(x=[f"W{w}" for w in ctx.proj.weeks], y=weekly, marker_color="#2a78d6", marker_line_width=0,
                        hovertemplate="%{x}: %{y:.1f} projected<extra></extra>"))
fig2.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Optimal lineup pts", bargap=0.4)
st.plotly_chart(fig2, width="stretch")
st.caption("Projected optimal-lineup points each remaining week; dips reveal bye weeks or thin spots.")
