import polars as pl
import streamlit as st

from aniston.ui import as_pct, load, page, ready

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
