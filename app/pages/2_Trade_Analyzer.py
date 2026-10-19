import polars as pl
import streamlit as st

from aniston.ui import page, ready, trade_context

page("Trade Analyzer", "⚖️")
st.title("Trade Analyzer")
st.caption("Build any trade and see the full before/after: your odds, theirs, what each side sees, and the league.")
if not ready():
    st.stop()

ctx = trade_context()
partner = st.selectbox("Trade partner", [t for t in ctx.base if t != ctx.me], format_func=lambda t: ctx.teams[t])
label = lambda i: f"{ctx.names[i]} ({ctx.pos[i]}, {ctx.true.ros[i]:.0f} ROS)"  # noqa: E731
c1, c2 = st.columns(2)
give = c1.multiselect("You give", sorted(ctx.base[ctx.me], key=lambda i: -ctx.true.ros[i]), format_func=label)
get = c2.multiselect("You get", sorted(ctx.base[partner], key=lambda i: -ctx.true.ros[i]), format_func=label)

if not give or not get:
    st.info("Pick at least one player on each side.")
    st.stop()

t = ctx.evaluate(ctx.build_trade(partner, tuple(give), tuple(get)))
k = st.columns(5)
k[0].metric("Δ Title odds", f"{t.d_title * 100:+.1f} pts", f"±{t.d_title_se * 100:.1f}", delta_color="off")
k[1].metric("Δ Playoff odds", f"{t.d_playoffs * 100:+.1f} pts")
k[2].metric("Δ Projected wins", f"{t.d_wins:+.2f}")
k[3].metric("P(they accept)", f"{t.p_accept * 100:.0f}%")
k[4].metric("P(no veto)", f"{t.p_no_veto * 100:.0f}%")

nm = lambda ids: ", ".join(ctx.names[i] for i in ids) or "–"  # noqa: E731
st.markdown(
    f"""
| | You | {ctx.teams[partner]} |
|---|---|---|
| ROS lineup points (consensus) | **{t.my_gain:+.0f}** | {t.their_true_gain:+.0f} |
| ROS lineup points as ESPN shows them | – | **{t.their_seen_gain:+.0f}** |
| Perceived value received − given | {-t.their_pvor_gain:+.0f} | {t.their_pvor_gain:+.0f} |
| Roster moves after the trade | add {nm(t.my_moves['add'])}; drop {nm(t.my_moves['drop'])} | add {nm(t.their_moves['add'])}; drop {nm(t.their_moves['drop'])} |
| Δ title odds | **{t.d_title * 100:+.1f} pts** | {t.partner_d_title * 100:+.1f} pts |
"""
)

from aniston.trade.explain import rationale  # noqa: E402

st.subheader("Rationale")
for line in rationale(ctx, t).split(" | "):
    st.markdown(f"- {line}")

st.subheader("League-wide effect")
rosters = dict(ctx.base)
rosters[ctx.me], rosters[partner] = t.my_roster, t.their_roster
after = ctx.sim.run(rosters).frame().select("team_id", pl.col("p_playoffs").alias("playoffs_after"),
                                             pl.col("p_title").alias("title_after"))
before = ctx.base_result.frame().select("team_id", "p_playoffs", "p_title")
lw = before.join(after, on="team_id").with_columns(
    pl.col("team_id").replace_strict(ctx.teams).alias("team"),
    ((pl.col("title_after") - pl.col("p_title")) * 100).alias("Δ title"),
    ((pl.col("playoffs_after") - pl.col("p_playoffs")) * 100).alias("Δ playoffs"),
).sort("Δ title")
st.dataframe(lw.select("team", (pl.col("p_title") * 100).alias("title before"), "Δ title",
                       (pl.col("p_playoffs") * 100).alias("playoffs before"), "Δ playoffs"),
             hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(c, format="%+.1f" if "Δ" in c else "%.1f%%")
                            for c in ["title before", "Δ title", "playoffs before", "Δ playoffs"]})
