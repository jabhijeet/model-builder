import glob as _glob
from typing import AsyncIterator
import duckdb
import pandas as pd
from ..plugins.protocol import Connection, DataProfile, Chunk


class FileConnector:
    name = "file"

    def connect(self, config: dict) -> Connection:
        paths = config.get("paths", [])
        resolved = []
        for pattern in paths:
            matches = _glob.glob(str(pattern), recursive=True)
            resolved.extend(matches if matches else [str(pattern)])
        chunk_size = config.get("chunk_size", 10_000)
        return Connection(
            connector_name="file",
            handle={"paths": resolved, "chunk_size": chunk_size},
        )

    def _read_df(self, handle: dict) -> pd.DataFrame:
        paths = handle["paths"]
        if not paths:
            return pd.DataFrame()
        frames = []
        for p in paths:
            p = str(p)
            if p.endswith(".parquet"):
                frames.append(duckdb.read_parquet(p).df())
            elif p.endswith(".json") or p.endswith(".jsonl"):
                frames.append(duckdb.read_json(p).df())
            elif p.endswith((".xlsx", ".xls")):
                frames.append(pd.read_excel(p))
            elif p.endswith(".feather") or p.endswith(".arrow"):
                import pyarrow.feather as feather
                frames.append(feather.read_feather(p))
            else:
                frames.append(duckdb.read_csv(p).df())
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def sample(self, conn: Connection, n: int) -> pd.DataFrame:
        return self._read_df(conn.handle).head(n)

    def profile(self, conn: Connection) -> DataProfile:
        df = self._read_df(conn.handle)
        return DataProfile(
            row_count=len(df),
            column_count=len(df.columns),
            columns={col: str(df[col].dtype) for col in df.columns},
            nulls={col: int(df[col].isna().sum()) for col in df.columns},
            data_types=["tabular"],
            sample_path=conn.handle["paths"][0] if conn.handle["paths"] else "",
        )

    async def stream(self, conn: Connection) -> AsyncIterator[Chunk]:
        df = self._read_df(conn.handle)
        chunk_size = conn.handle.get("chunk_size", 10_000)
        for i, start in enumerate(range(0, len(df), chunk_size)):
            yield Chunk(data=df.iloc[start: start + chunk_size], sequence=i)
