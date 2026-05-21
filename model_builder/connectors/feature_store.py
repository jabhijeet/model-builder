"""Feature store connector — loads named feature sets into a pipeline."""
import asyncio
from typing import AsyncIterator
import pandas as pd
from ..plugins.protocol import Connection, DataProfile, Chunk
from ..feature_store.store import FeatureStore


class FeatureStoreConnector:
    name = "feature_store"

    def connect(self, config: dict) -> Connection:
        name = config.get("name")
        if not name:
            raise ValueError("FeatureStoreConnector requires 'name' in config")
        return Connection(
            connector_name="feature_store",
            handle={
                "name": name,
                "version": config.get("version"),
                "project_dir": config.get("project_dir", "."),
                "chunk_size": config.get("chunk_size", 10_000),
            },
        )

    def _load(self, handle: dict) -> pd.DataFrame:
        from pathlib import Path
        store = FeatureStore(Path(handle["project_dir"]))
        return asyncio.run(_async_load(store, handle["name"], handle.get("version")))

    def sample(self, conn: Connection, n: int) -> pd.DataFrame:
        return self._load(conn.handle).head(n)

    def profile(self, conn: Connection) -> DataProfile:
        from pathlib import Path
        store = FeatureStore(Path(conn.handle["project_dir"]))
        meta = asyncio.run(_async_get(store, conn.handle["name"], conn.handle.get("version")))
        return DataProfile(
            row_count=meta.row_count,
            column_count=meta.column_count,
            columns=meta.columns,
            nulls={col: 0 for col in meta.columns},
            data_types=["tabular"],
            sample_path=meta.data_path,
        )

    async def stream(self, conn: Connection) -> AsyncIterator[Chunk]:
        from pathlib import Path
        store = FeatureStore(Path(conn.handle["project_dir"]))
        await store.init()
        df = await _async_load(store, conn.handle["name"], conn.handle.get("version"))
        chunk_size = conn.handle["chunk_size"]
        for i, start in enumerate(range(0, len(df), chunk_size)):
            yield Chunk(data=df.iloc[start: start + chunk_size], sequence=i)


async def _async_load(store: FeatureStore, name: str, version) -> pd.DataFrame:
    await store.init()
    return await store.load(name, version)


async def _async_get(store: FeatureStore, name: str, version):
    await store.init()
    return await store.get(name, version)
