import polars as pl
import streamlit as st

from aniston.ui import load, load_opt, page, ready

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
    rm = bt.get("rmse")
    st.caption(f"Mean absolute error in fantasy points per player-week, lower is better. Current weights: ESPN "
               f"{s.w_espn:.2f} / Sleeper {s.w_sleeper:.2f} / usage {s.w_usage:.2f}. Best fit on this sample (by "
               f"squared error, since the blend is used as a mean): {fw['espn']:.2f} / {fw['sleeper']:.2f} / "
               f"{fw['usage']:.2f}"
               + (f", RMSE {rm['ensemble_fitted']:.3f} against {rm['ensemble_current']:.3f} now" if rm else "")
               + ". The optimum is flat, so differences of a few hundredths in a weight are noise; change "
                 "W_ESPN, W_SLEEPER, W_USAGE in `.env` only when the fit moves well away from the current values.")
    st.dataframe(load("backtest_by_pos"), hide_index=True, width="stretch")

hz = load_opt("backtest_horizon")
if hz is not None and hz.height:
    st.subheader("Accuracy by how far ahead the projection was made")
    st.dataframe(hz, hide_index=True, width="stretch", column_config={
        "weeks_ahead": "Weeks ahead",
        "ens_mae": st.column_config.NumberColumn("Consensus MAE", format="%.2f"),
        "espn_mae": st.column_config.NumberColumn("ESPN MAE", format="%.2f"),
        "sleeper_mae": st.column_config.NumberColumn("Sleeper MAE", format="%.2f"),
        "n": "Player-weeks"})
    st.caption("From the daily snapshots of every player's weekly projections, scored once the games are played. "
               "The trade engine's edge is consensus beating ESPN at 1+ weeks ahead; this table is the evidence "
               "for or against it, and it fills in as snapshots accumulate.")

cal = store.get_json("sim_calibration")
if cal:
    st.subheader("Simulation calibration: weekly win probabilities")
    st.metric("Brier score", f"{cal['brier']:.3f}", "0.250 = coin flip, lower is better", delta_color="off")
    st.dataframe(load("sim_calibration"), hide_index=True, width="stretch", column_config={
        "bucket": "Predicted", "predicted": st.column_config.NumberColumn("Mean predicted", format="percent"),
        "won": st.column_config.NumberColumn("Actually won", format="percent"), "games": "Team-games"})
    st.caption(f"{cal['n']} team-games over weeks {cal['weeks'][0]}–{cal['weeks'][-1]}, each scored against the last "
               "snapshot before the week ended. Well calibrated = the two percentage columns agree.")

from aniston.offers import calibration  # noqa: E402

oc = calibration()
if oc:
    st.subheader("Acceptance model vs your logged offers")
    c = st.columns(3)
    c[0].metric("Resolved offers", oc["n"])
    c[1].metric("Predicted accept rate", f"{oc['predicted'] * 100:.0f}%")
    c[2].metric("Observed accept rate", f"{oc['observed'] * 100:.0f}%")
    if oc["suggested_acc_bias"] is not None:
        st.caption(f"ACC_BIAS={oc['suggested_acc_bias']} in `.env` (now {s.acc_bias}) would make the two match. "
                   "With this few offers treat it as a nudge, not a fit.")
    else:
        st.caption("Too few resolved offers (or all one outcome) to suggest a correction yet.")

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
