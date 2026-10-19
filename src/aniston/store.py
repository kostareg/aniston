"""DuckDB persistence. Every table is replaced wholesale on refresh; history lives in `snapshots`."""
import json
from datetime import datetime

import duckdb
import polars as pl

from aniston.config import settings


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path = settings().db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def write(name: str, df: pl.DataFrame) -> None:
    with connect() as con:
        con.register("_df", df.to_arrow())
        con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM _df')
        con.unregister("_df")


def read(name: str) -> pl.DataFrame:
    with connect(read_only=True) as con:
        return pl.from_arrow(con.execute(f'SELECT * FROM "{name}"').arrow())


def has(name: str) -> bool:
    with connect(read_only=True) as con:
        return bool(con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]).fetchone()[0])


def put_json(key: str, value) -> None:
    write_kv = pl.DataFrame({"key": [key], "value": [json.dumps(value)], "updated": [datetime.now()]})
    with connect() as con:
        con.execute("CREATE TABLE IF NOT EXISTS kv (key VARCHAR PRIMARY KEY, value VARCHAR, updated TIMESTAMP)")
        con.register("_kv", write_kv.to_arrow())
        con.execute("INSERT OR REPLACE INTO kv SELECT * FROM _kv")
        con.unregister("_kv")


def get_json(key: str):
    with connect(read_only=True) as con:
        row = con.execute("SELECT value FROM kv WHERE key = ?", [key]).fetchone()
    return json.loads(row[0]) if row else None
