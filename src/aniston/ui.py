"""Shared dashboard helpers: cached loading, palette, formatting."""
import time

import polars as pl
import streamlit as st

from aniston import store

# Reference data-viz palette (categorical order fixed; diverging blue <-> red with gray midpoint).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
DIVERGING = [[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]
SEQUENTIAL = [[0.0, "#cde2fb"], [1.0, "#104281"]]


def page(title: str, icon: str = "🏈"):
    st.set_page_config(page_title=f"{title} · Aniston", page_icon=icon, layout="wide")


def _stamp() -> str:
    try:
        return (store.get_json("analysis") or {}).get("at", "") + str(store.get_json("meta") and 1)
    except Exception:  # noqa: BLE001 - database busy or empty
        return str(time.time())


@st.cache_data(show_spinner=False)
def _load(name: str, stamp: str) -> pl.DataFrame:
    for attempt in range(5):  # the CLI may be writing; DuckDB allows one writer
        try:
            return store.read(name)
        except Exception:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(0.5)


def load(name: str) -> pl.DataFrame:
    return _load(name, _stamp())


def meta() -> dict:
    return store.get_json("meta") or {}


def team_names() -> dict[int, str]:
    return dict(load("teams").select("team_id", "name").iter_rows())


def ready() -> bool:
    try:
        ok = store.has("sim_baseline")
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        st.warning("No analysis yet. Run `aniston refresh && aniston analyze`, or use the button on the Home page.")
    return ok


def pct(x, signed: bool = False) -> str:
    if x is None:
        return "–"
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"


def pct_cols(cols: list[str], signed: bool = False) -> dict:
    fmt = "%+.1f%%" if signed else "%.1f%%"
    return {c: st.column_config.NumberColumn(c, format=fmt) for c in cols}


def as_pct(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return df.with_columns(*(pl.col(c) * 100 for c in cols if c in df.columns))


@st.cache_resource(show_spinner="Building projections and the season simulation...")
def _ctx(stamp: str):
    from aniston.trade.engine import TradeContext

    return TradeContext()


def trade_context():
    return _ctx(_stamp())
