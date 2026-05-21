import os
from typing import AsyncIterator
import pandas as pd
from sqlalchemy import create_engine, text
from ..plugins.protocol import Connection, DataProfile, Chunk


class SQLConnector:
    name = "sql"

    def connect(self, config: dict) -> Connection:
        dsn = os.path.expandvars(config["dsn"])
        return Connection(
            connector_name="sql",
            handle={
                "dsn": dsn,
                "query": config.get("query", ""),
                "chunk_size": config.get("chunk_size", 10_000),
            },
        )

    def _read_df(self, handle: dict) -> pd.DataFrame:
        engine = create_engine(handle["dsn"])
        with engine.connect() as conn:
            return pd.read_sql(text(handle["query"]), conn)

    def sample(self, conn: Connection, n: int) -> pd.DataFrame:
        engine = create_engine(conn.handle["dsn"])
        with engine.connect() as db_conn:
            return pd.read_sql(
                text(f"SELECT * FROM ({conn.handle['query']}) q LIMIT :n"),
                db_conn,
                params={"n": n},
            )

    def profile(self, conn: Connection) -> DataProfile:
        df = self._read_df(conn.handle)
        return DataProfile(
            row_count=len(df),
            column_count=len(df.columns),
            columns={col: str(df[col].dtype) for col in df.columns},
            nulls={col: int(df[col].isna().sum()) for col in df.columns},
            data_types=["tabular"],
            sample_path=conn.handle["dsn"],
        )

    async def stream(self, conn: Connection) -> AsyncIterator[Chunk]:
        df = self._read_df(conn.handle)
        chunk_size = conn.handle.get("chunk_size", 10_000)
        for i, start in enumerate(range(0, len(df), chunk_size)):
            yield Chunk(data=df.iloc[start: start + chunk_size], sequence=i)
