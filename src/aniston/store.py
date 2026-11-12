"""DuckDB persistence. Refresh and analysis tables are replaced wholesale on every run; history lives in the
`snap_*` tables (one snapshot per calendar day, see `snapshot`) and in `offers` (trades actually proposed)."""
import json
import time
from datetime import datetime

import duckdb
import polars as pl

from aniston.config import settings


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path = settings().db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def _writer(attempts: int = 10) -> duckdb.DuckDBPyConnection:
    """Write connection, retried: DuckDB allows one writing process, and the dashboard and the timer both write."""
    for attempt in range(attempts):
        try:
            return connect()
        except duckdb.IOException:
            if attempt == attempts - 1:
                raise
            time.sleep(0.5)


def write(name: str, df: pl.DataFrame) -> None:
    with _writer() as con:
        con.register("_df", df.to_arrow())
        con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM _df')
        con.unregister("_df")


def read(name: str) -> pl.DataFrame:
    with connect(read_only=True) as con:
        return con.execute(f'SELECT * FROM "{name}"').pl()  # unlike .arrow(), keeps the schema of an empty table


def has(name: str) -> bool:
    with connect(read_only=True) as con:
        return bool(con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]).fetchone()[0])


def _columns(con, name: str) -> list[str]:
    return [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ?", [name]).fetchall()]


def append(name: str, df: pl.DataFrame, replace_where: tuple[str, object] | None = None) -> None:
    """Append rows to a history table, creating it (or adding new columns) as needed. `replace_where` =
    (column, value) first deletes the rows with that value, so re-running replaces instead of duplicating."""
    with _writer() as con:
        con.register("_df", df.to_arrow())
        existing = _columns(con, name)
        if not existing:
            con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM _df')
        else:
            for col, dtype, *_ in con.execute("DESCRIBE SELECT * FROM _df").fetchall():
                if col not in existing:
                    con.execute(f'ALTER TABLE "{name}" ADD COLUMN "{col}" {dtype}')
            if replace_where:
                con.execute(f'DELETE FROM "{name}" WHERE "{replace_where[0]}" = ?', [replace_where[1]])
            con.execute(f'INSERT INTO "{name}" BY NAME SELECT * FROM _df')
        con.unregister("_df")


def snapshot(name: str, df: pl.DataFrame, at: datetime, week: int) -> None:
    """Keep today's copy of an analysis table in `snap_<name>`; a later run on the same day replaces it."""
    if df.height == 0:
        return
    day = at.date()
    stamped = df.with_columns(pl.lit(day).alias("snap_date"), pl.lit(week, pl.Int64).alias("snap_week"))
    append(f"snap_{name}", stamped, replace_where=("snap_date", day))


def execute(sql: str, params: list | None = None) -> None:
    with _writer() as con:
        con.execute(sql, params or [])


def put_json(key: str, value) -> None:
    write_kv = pl.DataFrame({"key": [key], "value": [json.dumps(value)], "updated": [datetime.now()]})
    with _writer() as con:
        cols = _columns(con, "kv")
        if cols and "updated" not in cols:  # stale schema; kv is rebuilt by refresh, so dropping is safe
            con.execute("DROP TABLE kv")
        con.execute("CREATE TABLE IF NOT EXISTS kv (key VARCHAR PRIMARY KEY, value VARCHAR, updated TIMESTAMP)")
        con.register("_kv", write_kv.to_arrow())
        con.execute("INSERT OR REPLACE INTO kv SELECT * FROM _kv")
        con.unregister("_kv")


def get_json(key: str):
    if not settings().db_path.exists() or not has("kv"):
        return None
    with connect(read_only=True) as con:
        row = con.execute("SELECT value FROM kv WHERE key = ?", [key]).fetchone()
    return json.loads(row[0]) if row else None
