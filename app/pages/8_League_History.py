import plotly.graph_objects as go
import polars as pl
import streamlit as st

from aniston.ui import BLUE, ORANGE, load, meta, page, ready, team_names

page("League History", "📜")
st.title("League History")
if not ready():
    st.stop()

m = meta()
names = team_names()
draft = load("draft")
pm = load("player_model").select("espn_id", "ens_ros", "pos", "fantasy_team_id")
season = load("players").select("espn_id", "espn_season_actual")
d = (draft.join(season, on="espn_id", how="left").join(pm, on="espn_id", how="left")
     .with_columns(pl.col("espn_season_actual").fill_null(0), pl.col("ens_ros").fill_null(0))
     .with_columns((pl.col("espn_season_actual") + pl.col("ens_ros")).alias("season_value")))
# Expected season value for a pick slot = average of the 12 picks around it.
d = d.sort("overall").with_columns(pl.col("season_value").rolling_mean(12, center=True, min_samples=4).alias("slot_avg"))
d = d.with_columns((pl.col("season_value") - pl.col("slot_avg")).alias("vs_slot"))

st.subheader("Draft value: points so far + projected rest of season, by pick")
fig = go.Figure()
fig.add_scatter(x=d["overall"], y=d["season_value"], mode="markers", marker=dict(size=8, color=BLUE, line=dict(width=1, color="white")),
                text=d["name"] + " · " + d["team_id"].replace_strict(names), name="Pick",
                hovertemplate="Pick %{x}: %{text}<br>%{y:.0f} season pts<extra></extra>")
fig.add_scatter(x=d["overall"], y=d["slot_avg"], mode="lines", line=dict(color=ORANGE, width=2), name="Slot average",
                hovertemplate="Pick %{x}: slot avg %{y:.0f}<extra></extra>")
fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Overall pick", yaxis_title="Season points",
                  legend=dict(orientation="h", y=1.08))
st.plotly_chart(fig, width="stretch")

c1, c2 = st.columns(2)
cols = {"overall": "Pick", "name": "Player", "team_id": "Drafted by",
        "vs_slot": st.column_config.NumberColumn("vs slot", format="%+.0f")}
show = lambda df: df.select("overall", "name", pl.col("team_id").replace_strict(names), "vs_slot")  # noqa: E731
c1.markdown("**Best values**")
c1.dataframe(show(d.sort("vs_slot", descending=True).head(10)), hide_index=True, width="stretch", column_config=cols)
c2.markdown("**Biggest busts**")
c2.dataframe(show(d.sort("vs_slot").head(10)), hide_index=True, width="stretch", column_config=cols)

st.subheader("Draft grades by team")
grade = d.group_by("team_id").agg(pl.col("vs_slot").sum().alias("value_vs_slot")).sort("value_vs_slot", descending=True)
st.dataframe(grade.with_columns(pl.col("team_id").replace_strict(names).alias("team")).select("team", "value_vs_slot"),
             hide_index=True, width="stretch",
             column_config={"value_vs_slot": st.column_config.NumberColumn("Total pts vs slot", format="%+.0f")})

st.subheader("Trades graded by rest-of-season value (consensus)")
st.caption("ROS = consensus rest-of-season points (what each side actually gets). Perceived = what ESPN makes it "
           "look like today. Accepted trades with balanced perceived value but lopsided ROS are exactly the edge the "
           "Trade Finder hunts for.")
pv = load("player_model").select("espn_id", "pv")
trades = load("trades").join(pm.select("espn_id", "ens_ros"), on="espn_id", how="left").join(pv, on="espn_id", how="left")
if trades.height:
    g = trades.group_by("trade_idx", "team_id").agg(pl.col("name").str.join(", "), pl.col("ens_ros").fill_null(0).sum(),
                                                    pl.col("pv").fill_null(0).sum())
    for idx in sorted(g["trade_idx"].unique().to_list()):
        sides = g.filter(pl.col("trade_idx") == idx).sort("ens_ros", descending=True)
        st.markdown("- " + " vs ".join(
            f"**{names[r['team_id']]}** got {r['name']} ({r['ens_ros']:.0f} ROS, {r['pv']:.0f} perceived)"
            for r in sides.iter_rows(named=True)))
