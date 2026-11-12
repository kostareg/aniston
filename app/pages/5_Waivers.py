import polars as pl
import streamlit as st

from aniston.ui import as_pct, load, load_opt, page, ready

page("Waivers", "📋")
st.title("Waivers")
st.caption("Free agents ranked by how much they raise *your* rest-of-season starting lineup (after dropping your "
           "least valuable player), with simulated playoff and title odds for the top options.")
if not ready():
    st.stop()

w = load("waivers").join(load("player_model").select("espn_id", "pro_team", "injury_status", "pct_owned", "ens_ros",
                                                      "espn_ros", "vorp", "flag", "luck_pg", "target_share_trend"),
                         on="espn_id", how="left")
st.dataframe(as_pct(w.drop("espn_id"), ["d_playoffs", "d_title"]), hide_index=True, width="stretch",
             column_config={
                 "name": "Player", "pos": "Pos", "pro_team": "NFL", "injury_status": "Status",
                 "ros_lineup_gain": st.column_config.NumberColumn("Your ROS lineup gain", format="%+.1f"),
                 "playoff_gain": st.column_config.NumberColumn(
                     "Playoff-week gain", format="%+.1f",
                     help="The part of the gain that lands in the fantasy playoff weeks: a stash worth holding"),
                 "drop": "You'd drop",
                 "d_playoffs": st.column_config.NumberColumn("Δ Playoffs", format="%+.1f%%"),
                 "d_title": st.column_config.NumberColumn("Δ Title", format="%+.1f%%"),
                 "pct_owned": st.column_config.NumberColumn("% rostered (ESPN)", format="%.0f%%"),
                 "ens_ros": st.column_config.NumberColumn("ROS (consensus)", format="%.0f"),
                 "espn_ros": st.column_config.NumberColumn("ROS (ESPN)", format="%.0f"),
                 "vorp": st.column_config.NumberColumn("Over replacement", format="%+.0f"),
                 "flag": "Signal",
                 "luck_pg": st.column_config.NumberColumn("Pts/g vs expected", format="%+.1f"),
                 "target_share_trend": st.column_config.NumberColumn("Target share trend", format="%+.2f"),
             })

streams = load_opt("streamers")
if streams is not None and streams.height:
    st.subheader("Streamers")
    st.caption("Best free agents at QB, K and D/ST for each of the next weeks, against who you'd start there. "
               "Positive gain = worth a one-week pickup.")
    weeks = sorted(streams["week"].unique().to_list())
    for tab, wk in zip(st.tabs([f"Week {w}" for w in weeks]), weeks):
        tab.dataframe(streams.filter(pl.col("week") == wk).sort(["pos", "proj"], descending=[False, True]).drop("week"),
                      hide_index=True, width="stretch", column_config={
                          "pos": "Pos", "name": "Free agent", "pro_team": "NFL",
                          "proj": st.column_config.NumberColumn("Projected", format="%.1f"),
                          "my_starter": "Your starter",
                          "my_proj": st.column_config.NumberColumn("His projection", format="%.1f"),
                          "gain": st.column_config.NumberColumn("Gain", format="%+.1f")})

blocks = load_opt("blocks")
if blocks is not None and blocks.height:
    st.subheader(f"Blocks against {blocks['opponent'][0]}")
    st.caption("Free agents who would raise this week's opponent's best lineup. If your last bench spot is dead "
               "weight, picking one up costs you nothing and takes the points away from them.")
    st.dataframe(blocks.drop("opponent"), hide_index=True, width="stretch", column_config={
        "name": "Free agent", "pos": "Pos",
        "opp_gain": st.column_config.NumberColumn("Their gain this week", format="%+.1f"),
        "my_gain": st.column_config.NumberColumn("Your gain this week", format="%+.1f"),
        "drop": "You'd drop"})
