import asyncio
from typing import AsyncIterator
import httpx
import pandas as pd
from ..plugins.protocol import Connection, DataProfile, Chunk


def _extract_records(payload, records_path: str) -> list:
    if isinstance(payload, list):
        return payload
    if records_path and records_path in payload:
        val = payload[records_path]
        return val if isinstance(val, list) else [val]
    return [payload]


class RestPollConnector:
    name = "rest_poll"

    def connect(self, config: dict) -> Connection:
        return Connection(
            connector_name="rest_poll",
            handle={
                "url": config["url"],
                "records_path": config.get("records_path", ""),
                "interval_seconds": config.get("interval_seconds", 10),
                "max_batches": config.get("max_batches", 100),
                "headers": config.get("headers", {}),
            },
        )

    def _fetch_df(self, handle: dict) -> pd.DataFrame:
        resp = httpx.get(handle["url"], headers=handle.get("headers", {}))
        resp.raise_for_status()
        records = _extract_records(resp.json(), handle["records_path"])
        return pd.DataFrame(records)

    def sample(self, conn: Connection, n: int) -> pd.DataFrame:
        return self._fetch_df(conn.handle).head(n)

    def profile(self, conn: Connection) -> DataProfile:
        df = self._fetch_df(conn.handle)
        return DataProfile(
            row_count=len(df),
            column_count=len(df.columns),
            columns={col: str(df[col].dtype) for col in df.columns},
            nulls={col: int(df[col].isna().sum()) for col in df.columns},
            data_types=["stream", "tabular"],
            sample_path=conn.handle["url"],
        )

    async def stream(self, conn: Connection) -> AsyncIterator[Chunk]:
        handle = conn.handle
        for i in range(handle["max_batches"]):
            df = self._fetch_df(handle)
            yield Chunk(data=df, sequence=i)
            if i < handle["max_batches"] - 1 and handle["interval_seconds"] > 0:
                await asyncio.sleep(handle["interval_seconds"])
