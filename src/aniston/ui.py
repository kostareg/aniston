"""Shared dashboard helpers: cached loading, palette, formatting."""
import subprocess
import sys
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


def load_opt(name: str) -> pl.DataFrame | None:
    """A table that only exists once a newer `aniston analyze` has run; None until then."""
    try:
        return load(name) if store.has(name) else None
    except Exception:  # noqa: BLE001 - database busy
        return None


_RELOAD: dict = {"proc": None}  # the running `aniston reload`; this module outlives page reruns


def reload_log():
    return store.settings().db_path.parent / "reload.log"


def start_reload() -> None:
    """Run `aniston reload` (refresh + analyze) as its own process: the trade search can then use every core
    (forking inside Streamlit's threaded server is unsafe) and the dashboard stays responsive meanwhile."""
    if reload_state() == "running":
        return
    log = reload_log()
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as out:
        _RELOAD["proc"] = subprocess.Popen([sys.executable, "-m", "aniston.cli", "reload"], stdout=out,
                                           stderr=subprocess.STDOUT, start_new_session=True)


def reload_state() -> str:
    """'idle' (never started here), 'running', 'done' or 'failed'."""
    proc = _RELOAD["proc"]
    if proc is None:
        return "idle"
    code = proc.poll()
    return "running" if code is None else "done" if code == 0 else "failed"


def reload_clear() -> None:
    _RELOAD["proc"] = None


def trades_closed() -> bool:
    from aniston.config import deadline_passed

    return deadline_passed(meta())


def log_offer_button(ctx, t, key: str):
    """'Log as proposed' control for a scored Trade; writes to the offer log."""
    from aniston.offers import log_offer

    c1, c2 = st.columns([3, 1])
    note = c1.text_input("Note", key=f"note_{key}", label_visibility="collapsed", placeholder="Note (optional)")
    if c2.button("Log as proposed", key=f"log_{key}", width="stretch"):
        st.success(f"Logged as offer #{log_offer(ctx, t, note)}. Update its outcome on the Offers page.")


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
