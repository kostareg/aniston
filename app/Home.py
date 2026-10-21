import plotly.graph_objects as go
import polars as pl
import streamlit as st

from aniston.ui import BLUE, ORANGE, as_pct, load, meta, page, pct, ready

page("League")
with connect() as con:
    con.execute("""
        CREATE TABLE IF NOT EXISTS kv (
            key VARCHAR PRIMARY KEY,
            value VARCHAR
        )
    """)
m = meta()
st.title(m.get("league_name", "Aniston"))

with st.sidebar:
    st.caption("Data")
    from aniston import store

    info = store.get_json("analysis") if store.settings().db_path.exists() else None
    st.write(f"Analysis: **{(info or {}).get('at', 'never')}**")
    if st.button("Refresh data + rerun models", type="primary", width="stretch"):
        from aniston.analyze import analyze
        from aniston.refresh import refresh

        with st.spinner("Pulling ESPN, Sleeper, nflverse..."):
            refresh()
        with st.spinner("Projections, simulation, trade search (about 30s)..."):
            analyze()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

if not ready():
    st.stop()

me = m["my_team_id"]
sim = load("sim_baseline")
teams = load("teams")
sched = load("schedule")

# ---- headline tiles
mine = sim.filter(pl.col("team_id") == me).row(0, named=True)
my_team = teams.filter(pl.col("team_id") == me).row(0, named=True)
c = st.columns(5)
c[0].metric("Record", f"{my_team['wins']}-{my_team['losses']}", f"#{my_team['standing']} in standings", delta_color="off")
c[1].metric("Projected wins", f"{mine['exp_wins']:.1f}", f"of {m['reg_season_weeks']} games", delta_color="off")
c[2].metric("Playoff odds", pct(mine["p_playoffs"]), f"ESPN says {m['espn_playoff_pct'][str(me)]:.0f}%", delta_color="off")
c[3].metric("Bye odds", pct(mine["p_bye"]))
c[4].metric("Title odds", pct(mine["p_title"]), f"{mine['p_title'] * len(teams):.1f}x an average team", delta_color="off")
st.caption(f"Week {m['current_week']} in progress · trade deadline {m['trade_deadline'][:10]} · "
           f"{m['playoff_teams']} make the playoffs (weeks {m['reg_season_weeks'] + 1}-{m['last_week']})")

# ---- luck: actual wins vs all-play
played = sched.filter(pl.col("outcome").is_in(["W", "L", "T"]))
allplay = (
    played.with_columns(((pl.col("score").rank("average").over("week") - 1) / (pl.len().over("week") - 1)).alias("ap"))
    .group_by("team_id").agg(pl.col("ap").sum().alias("allplay_wins"))
)
table = (
    teams.select("team_id", "name", "wins", "losses", "points_for", "points_against")
    .join(allplay, on="team_id", how="left")
    .join(sim.select("team_id", "exp_wins", "p_playoffs", "p_bye", "p_title"), on="team_id")
    .with_columns((pl.col("wins") - pl.col("allplay_wins")).alias("luck"))
    .sort(["wins", "points_for"], descending=True)
)

left, right = st.columns([3, 2])
with left:
    st.subheader("Standings")
    st.dataframe(
        as_pct(table, ["p_playoffs", "p_bye", "p_title"]).drop("team_id"),
        hide_index=True, width="stretch",
        column_config={
            "name": "Team", "wins": "W", "losses": "L",
            "points_for": st.column_config.NumberColumn("PF", format="%.1f"),
            "points_against": st.column_config.NumberColumn("PA", format="%.1f"),
            "allplay_wins": st.column_config.NumberColumn("All-play W", format="%.1f",
                                                          help="Wins expected if you played every team every week"),
            "luck": st.column_config.NumberColumn("Luck", format="%+.1f", help="Actual wins minus all-play wins"),
            "exp_wins": st.column_config.NumberColumn("Proj. W", format="%.1f"),
            "p_playoffs": st.column_config.ProgressColumn("Playoffs", format="%.0f%%", min_value=0, max_value=100),
            "p_bye": st.column_config.NumberColumn("Bye", format="%.0f%%"),
            "p_title": st.column_config.NumberColumn("Title", format="%.1f%%"),
        },
    )
with right:
    st.subheader("Title odds")
    t = table.sort("p_title")
    colors = [ORANGE if tid == me else BLUE for tid in t["team_id"]]
    fig = go.Figure(go.Bar(x=t["p_title"] * 100, y=t["name"], orientation="h", marker_color=colors,
                           marker_line_width=0, hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
                           text=[f"{v * 100:.0f}%" if v >= 0.05 else "" for v in t["p_title"]],
                           textposition="outside"))
    fig.update_layout(height=420, margin=dict(l=0, r=30, t=10, b=0), xaxis_title="Championship probability (%)",
                      bargap=0.35, showlegend=False)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Your team in orange. 5,000 simulated seasons of the remaining schedule and playoff bracket.")

st.subheader("Seed distribution")
seed_cols = [c for c in sim.columns if c.startswith("seed_")]
sd = sim.join(teams.select("team_id", "wins"), on="team_id").sort("exp_wins", descending=True)
st.dataframe(as_pct(sd.select("name", *seed_cols), seed_cols), hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(c.replace("seed_", "Seed "), format="%.0f%%") for c in seed_cols})
