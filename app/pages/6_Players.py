import plotly.graph_objects as go
import polars as pl
import streamlit as st

from aniston.ui import AQUA, BLUE, ORANGE, load, meta, page, ready, team_names

page("Players", "📈")
st.title("Players")
if not ready():
    st.stop()

m = meta()
names = team_names()
pm = load("player_model").with_columns(
    pl.col("fantasy_team_id").replace_strict({0: "Free agent", **names}, default="?").alias("owner"))

c1, c2, c3 = st.columns(3)
pos = c1.multiselect("Position", ["QB", "RB", "WR", "TE", "K", "D/ST"], default=["QB", "RB", "WR", "TE"])
owner = c2.multiselect("Owner", ["Free agent", *names.values()])
flag = c3.multiselect("Signal", ["buy-low", "sell-high"])
v = pm.filter(pl.col("pos").is_in(pos)) if pos else pm
if owner:
    v = v.filter(pl.col("owner").is_in(owner))
if flag:
    v = v.filter(pl.col("flag").is_in(flag))
v = v.sort("ens_ros", descending=True)

st.dataframe(
    v.select("name", "pos", "pro_team", "owner", "injury_status", "ens_ros", "espn_ros", "sleeper_ros",
             "espn_vs_consensus", "vorp", "pv", "flag", "fp_pg", "xfp_pg", "luck_pg", "last3", "target_share",
             "target_share_trend", "carries_trend", "sos_ros", "sos_playoffs", "league_pick"),
    hide_index=True, width="stretch", height=480,
    column_config={
        "name": "Player", "pos": "Pos", "pro_team": "NFL", "owner": "Owner", "injury_status": "Status",
        "ens_ros": st.column_config.NumberColumn("ROS consensus", format="%.0f"),
        "espn_ros": st.column_config.NumberColumn("ROS ESPN", format="%.0f"),
        "sleeper_ros": st.column_config.NumberColumn("ROS Sleeper", format="%.0f"),
        "espn_vs_consensus": st.column_config.NumberColumn("ESPN vs consensus", format="percent",
                                                           help="Positive = ESPN (what others see) is higher"),
        "vorp": st.column_config.NumberColumn("Over replacement", format="%+.0f"),
        "pv": st.column_config.NumberColumn("Perceived value", format="%.0f", help="What a league-mate sees, ROS pts"),
        "flag": "Signal",
        "fp_pg": st.column_config.NumberColumn("Pts/g", format="%.1f"),
        "xfp_pg": st.column_config.NumberColumn("Expected pts/g", format="%.1f"),
        "luck_pg": st.column_config.NumberColumn("Luck/g", format="%+.1f"),
        "last3": st.column_config.NumberColumn("Last 3", format="%.1f"),
        "target_share": st.column_config.NumberColumn("Tgt share", format="%.2f"),
        "target_share_trend": st.column_config.NumberColumn("Tgt trend", format="%+.2f"),
        "carries_trend": st.column_config.NumberColumn("Carry trend", format="%+.1f"),
        "sos_ros": st.column_config.NumberColumn("SoS ROS", format="%.2f"),
        "sos_playoffs": st.column_config.NumberColumn("SoS playoffs", format="%.2f"),
        "league_pick": st.column_config.NumberColumn("Drafted at", format="%.0f"),
    },
)
st.caption("Pts/g vs expected uses nflverse expected fantasy points (usage-based): big positive gaps tend to regress. "
           "SoS > 1 means softer opponents for that position.")

choice = st.selectbox("Player detail", v["name"].to_list())
if choice:
    pid = v.filter(pl.col("name") == choice)["espn_id"][0]
    pw = load("projection_weeks").filter(pl.col("espn_id") == pid).sort("week")
    act = load("espn_weeks").filter((pl.col("espn_id") == pid) & pl.col("actual").is_not_null()).sort("week")
    fig = go.Figure()
    fig.add_bar(x=act["week"], y=act["actual"], name="Actual", marker_color="rgba(128,128,128,0.45)", marker_line_width=0,
                hovertemplate="Week %{x}: %{y:.1f} actual<extra></extra>")
    for col, color, label in [("espn", ORANGE, "ESPN"), ("sleeper", AQUA, "Sleeper"), ("ens", BLUE, "Consensus")]:
        fig.add_scatter(x=pw["week"], y=pw[col], name=label, mode="lines+markers", line=dict(color=color, width=2),
                        marker=dict(size=8), hovertemplate=f"Week %{{x}}: %{{y:.1f}} {label}<extra></extra>")
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified", xaxis_title="Week",
                      yaxis_title="Fantasy points", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")
