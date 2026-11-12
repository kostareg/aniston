import polars as pl
import streamlit as st

from aniston.ui import as_pct, load, log_offer_button, meta, page, ready, trade_context, trades_closed

page("Trade Finder", "🤝")
st.title("Trade Finder")
st.caption("Trades that look good to the other manager on ESPN but raise your championship odds by consensus "
           "projections and simulation. Ranked by expected value = Δ title odds × P(accept) × P(no veto).")
if not ready():
    st.stop()

m = meta()
if trades_closed():
    st.info(f"The trade deadline ({m['trade_deadline'][:10]}) has passed, so the trade search no longer runs. "
            "Waivers, streamers and start/sit are on the Waivers and My Team pages.")
    st.stop()
ideas = load("trade_ideas")

with st.expander("Search again with constraints"):
    ctx = trade_context()
    my_idx = sorted(ctx.base[ctx.me], key=lambda i: -ctx.true.ros[i])
    label = lambda i: f"{ctx.names[i]} ({ctx.pos[i]})"  # noqa: E731
    others = sorted((i for t, r in ctx.base.items() if t != ctx.me for i in r if ctx.tradeable[i]),
                    key=lambda i: -ctx.true.ros[i])
    owner = {i: t for t, r in ctx.base.items() for i in r}
    targets = st.multiselect("Target: you get at least one of", others,
                             format_func=lambda i: f"{label(i)} · {ctx.teams[owner[i]]}",
                             help="Finds the best packages that land this player; only their owner is searched.")
    c1, c2, c3 = st.columns(3)
    untouch = c1.multiselect("Untouchable", my_idx, format_func=label)
    must = c2.multiselect("You give at least one of", my_idx, format_func=label)
    partners = c3.multiselect("Only these partners", [t for t in ctx.base if t != ctx.me],
                              format_func=lambda t: ctx.teams[t])
    max_players = st.radio("Players per side", [1, 2], index=1, horizontal=True)
    if st.button("Search", type="primary"):
        with st.spinner("Searching packages and simulating the best ones..."):
            found = ctx.find(partners=partners or None, max_players=max_players,
                             untouchable=set(untouch), must_include=set(must), must_get=set(targets))
            st.session_state["custom_ideas"] = ctx.frame(found) if found else None
    if "custom_ideas" in st.session_state:
        if st.session_state["custom_ideas"] is None:
            st.info("No trades found that help you and clear the acceptance threshold.")
        else:
            ideas = st.session_state["custom_ideas"]
            st.success(f"Showing {ideas.height} results from your custom search.")

if ideas.height == 0:
    st.info("No trade ideas stored. Run `aniston analyze`.")
    st.stop()

f1, f2, f3 = st.columns(3)
partner_filter = f1.multiselect("Partner", sorted(ideas["partner"].unique().to_list()))
min_accept = f2.slider("Min P(accept)", 0, 100, 25, 5)
min_dtitle = f3.slider("Min Δ title odds (pts)", 0.0, 10.0, 0.5, 0.5)
view = ideas.filter(pl.col("p_accept") * 100 >= min_accept, pl.col("d_title") * 100 >= min_dtitle)
if partner_filter:
    view = view.filter(pl.col("partner").is_in(partner_filter))
view = view.sort("ev", descending=True)

st.dataframe(
    as_pct(view.select("partner", "give", "get", "d_title", "d_playoffs", "p_accept", "p_no_veto", "ev",
                       "my_ros_gain", "their_seen_gain", "their_true_gain", "you_add", "you_drop"),
           ["d_title", "d_playoffs", "p_accept", "p_no_veto", "ev"]),
    hide_index=True, width="stretch",
    column_config={
        "partner": "Partner", "give": "You give", "get": "You get",
        "d_title": st.column_config.NumberColumn("Δ Title", format="%+.1f%%"),
        "d_playoffs": st.column_config.NumberColumn("Δ Playoffs", format="%+.1f%%"),
        "p_accept": st.column_config.ProgressColumn("P(accept)", format="%.0f%%", min_value=0, max_value=100),
        "p_no_veto": st.column_config.NumberColumn("P(no veto)", format="%.0f%%"),
        "ev": st.column_config.NumberColumn("EV", format="%+.2f%%", help="Δ title × P(accept) × P(no veto)"),
        "my_ros_gain": st.column_config.NumberColumn("Your ROS pts", format="%+.0f",
                                                     help="Change in your rest-of-season starting lineup points (consensus)"),
        "their_seen_gain": st.column_config.NumberColumn("They see", format="%+.0f",
                                                         help="Their lineup change by ESPN projections — what they'll look at"),
        "their_true_gain": st.column_config.NumberColumn("They really get", format="%+.0f",
                                                         help="Their lineup change by consensus projections"),
        "you_add": "You pick up", "you_drop": "You drop",
    },
)

st.subheader("Why each trade works")
uidx = ctx.proj.idx()
for n, r in enumerate(view.head(10).iter_rows(named=True)):
    with st.expander(f"{r['partner']}: give {r['give']} → get {r['get']}  ·  title {r['d_title'] * 100:+.1f} pts, "
                     f"accept {r['p_accept'] * 100:.0f}%"):
        for line in r["why"].split(" | "):
            st.markdown(f"- {line}")
        st.caption(f"Simulation standard error on Δ title: ±{r['d_title_se'] * 100:.1f} pts. "
                   f"Partner's own title odds change: {r['partner_d_title'] * 100:+.1f} pts.")
        give, get = [uidx.get(e) for e in r["give_ids"]], [uidx.get(e) for e in r["get_ids"]]
        if None not in give + get and set(give) <= set(ctx.base[ctx.me]) and set(get) <= set(ctx.base[r["partner_id"]]):
            if st.checkbox("I sent this offer", key=f"sent_{n}"):
                log_offer_button(ctx, ctx.evaluate(ctx.build_trade(r["partner_id"], tuple(give), tuple(get))), f"f{n}")
