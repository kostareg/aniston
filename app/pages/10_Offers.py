import polars as pl
import streamlit as st

from aniston.offers import STATUSES, calibration, list_offers, set_status
from aniston.ui import page

page("Offers", "📨")
st.title("Offers")
st.caption("Trades you actually proposed and how they ended. Log one from the Trade Finder or Trade Analyzer, then "
           "set its outcome here. A counter-offer counts as the original being turned down; log the counter as a "
           "new offer if you send one back. Withdrawn offers are left out of the acceptance check.")

offers = list_offers()
if not offers.height:
    st.info("No offers logged yet.")
    st.stop()

shown = offers.select("id", "created", "partner", "give", "get", "p_accept", "d_title", "status", "note").with_columns(
    pl.col("p_accept") * 100, pl.col("d_title") * 100)
edited = st.data_editor(
    shown.to_pandas(), hide_index=True, width="stretch", key="offers_editor",
    disabled=["id", "created", "partner", "give", "get", "p_accept", "d_title"],
    column_config={
        "id": "#", "created": st.column_config.DatetimeColumn("Sent", format="MMM D"), "partner": "Partner",
        "give": "You give", "get": "You get",
        "p_accept": st.column_config.NumberColumn("P(accept) when sent", format="%.0f%%"),
        "d_title": st.column_config.NumberColumn("Δ title", format="%+.1f%%"),
        "status": st.column_config.SelectboxColumn("Outcome", options=list(STATUSES), required=True),
        "note": "Note",
    })
if st.button("Save changes", type="primary"):
    changed = 0
    for old, new in zip(shown.iter_rows(named=True), edited.to_dict("records")):
        note = new["note"] if isinstance(new["note"], str) else ""
        if (old["status"], old["note"] or "") != (new["status"], note):
            set_status(old["id"], new["status"], note if (old["note"] or "") != note else None)
            changed += 1
    st.success(f"Saved {changed} change(s).")
    st.rerun()

cal = calibration()
if cal:
    c = st.columns(3)
    c[0].metric("Resolved offers", cal["n"])
    c[1].metric("Predicted accept rate", f"{cal['predicted'] * 100:.0f}%")
    c[2].metric("Observed accept rate", f"{cal['observed'] * 100:.0f}%")
