import polars as pl
import streamlit as st

from aniston.ui import load, meta, page, ready

page("Managers", "🧑‍💼")
st.title("Managers")
st.caption("Who trades, what they need, and what they have to spare. Needs are positions where a team's starters "
           "trail the league average; surpluses are bench players who would start elsewhere.")
if not ready():
    st.stop()

m = meta()
teams = load("teams")
ts = load("team_strength").filter(~pl.col("pos").is_in(["K", "D/ST"]))
pm = load("player_model")
trades = load("trades")

needs = (ts.sort("vs_avg").group_by("team_id", maintain_order=True)
         .agg(pl.col("pos").first().alias("biggest_need"), pl.col("vs_avg").first().alias("need_gap"),
              pl.col("pos").last().alias("biggest_strength"), pl.col("vs_avg").last().alias("strength_gap")))
depth = (pm.filter(pl.col("fantasy_team_id") > 0, pl.col("pos").is_in(["RB", "WR", "TE", "QB"]))
         .sort("ens_ros", descending=True)
         .with_columns(pl.col("ens_ros").rank("ordinal", descending=True).over(["fantasy_team_id", "pos"]).alias("depth"))
         .filter(((pl.col("pos") == "QB") & (pl.col("depth") == 2)) | ((pl.col("pos").is_in(["RB", "WR"])) & (pl.col("depth") == 3))
                 | ((pl.col("pos") == "TE") & (pl.col("depth") == 2)))
         .group_by("fantasy_team_id").agg(pl.col("name").head(3).str.join(", ").alias("spare_parts")))
traded = trades.group_by("team_id").agg(pl.col("trade_idx").n_unique().alias("trades_made"))
view = (teams.select("team_id", "name", "wins", "losses", "acquisitions", "drops")
        .join(traded, on="team_id", how="left").join(needs, on="team_id", how="left")
        .join(depth, left_on="team_id", right_on="fantasy_team_id", how="left")
        .with_columns(pl.col("trades_made").fill_null(0)).filter(pl.col("team_id") != m["my_team_id"]))
st.dataframe(view.drop("team_id"), hide_index=True, width="stretch", column_config={
    "name": "Team", "wins": "W", "losses": "L", "acquisitions": "Pickups", "drops": "Drops",
    "trades_made": "Trades", "biggest_need": "Biggest need",
    "need_gap": st.column_config.NumberColumn("Need (pts/wk vs avg)", format="%+.1f"),
    "biggest_strength": "Strength", "strength_gap": st.column_config.NumberColumn("Strength (pts/wk)", format="%+.1f"),
    "spare_parts": "Spare parts (first bench option)"})

st.subheader("Trade history this season")
if trades.height:
    names = dict(teams.select("team_id", "name").iter_rows())
    for idx in sorted(trades["trade_idx"].unique().to_list()):
        t = trades.filter(pl.col("trade_idx") == idx)
        sides = [f"**{names[tid]}** got {', '.join(g['name'].to_list())}" for (tid,), g in t.group_by("team_id")]
        st.markdown(f"- {t['date'][0]:%b %d}: " + " · ".join(sides))
else:
    st.write("No trades yet.")
