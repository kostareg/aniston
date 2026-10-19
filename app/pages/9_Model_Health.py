import polars as pl
import streamlit as st

from aniston.ui import load, page, ready

page("Model Health", "🩺")
st.title("Model Health")
if not ready():
    st.stop()

from aniston import store  # noqa: E402
from aniston.config import settings  # noqa: E402

s = settings()
st.subheader("Data freshness")
st.json({"analysis": store.get_json("analysis"), "refresh": store.get_json("refresh_report")}, expanded=False)

bt = store.get_json("backtest")
if bt:
    st.subheader(f"Projection accuracy, weeks {bt['weeks'][0]}–{bt['weeks'][1]} ({bt['n']} player-weeks)")
    mae = bt["mae"]
    c = st.columns(5)
    c[0].metric("ESPN MAE", f"{mae['espn']:.2f}")
    c[1].metric("Sleeper MAE", f"{mae['sleeper']:.2f}")
    c[2].metric("Usage model MAE", f"{mae['usage']:.2f}")
    c[3].metric("Ensemble MAE", f"{mae['ensemble_current']:.2f}")
    c[4].metric("Best-fit blend MAE", f"{mae['ensemble_fitted']:.2f}")
    fw = bt["fitted_weights"]
    st.caption(f"Mean absolute error in fantasy points per player-week, lower is better. Current weights: ESPN "
               f"{s.w_espn:.2f} / Sleeper {s.w_sleeper:.2f} / usage {s.w_usage:.2f}. Best fit on this sample: "
               f"{fw['espn']:.2f} / {fw['sleeper']:.2f} / {fw['usage']:.2f}. Small samples overfit; adjust in `.env` "
               f"(W_ESPN, W_SLEEPER, W_USAGE) only once several more weeks agree.")
    st.dataframe(load("backtest_by_pos"), hide_index=True, width="stretch")

st.subheader("Simulation vs ESPN's playoff odds")
meta = store.get_json("meta")
sim = load("sim_baseline").with_columns(
    pl.col("team_id").cast(pl.Utf8).replace_strict(meta["espn_playoff_pct"], default=None).cast(pl.Float64).alias("espn"))
st.dataframe(sim.select("name", (pl.col("p_playoffs") * 100).alias("ours"), "espn").sort("ours", descending=True),
             hide_index=True, width="stretch",
             column_config={"ours": st.column_config.NumberColumn("Our playoff %", format="%.1f%%"),
                            "espn": st.column_config.NumberColumn("ESPN playoff %", format="%.1f%%")})
st.caption("ESPN's odds lean mostly on record; ours also use roster strength from consensus projections, so they "
           "spread wider. Model knobs (in .env): PLAYER_SHOCK_SD, TEAM_SHOCK_SD, INJURY_RATE, N_SIMS.")

st.subheader("Model settings")
st.json(s.model_dump(exclude={"espn_s2", "swid"}), expanded=False)
