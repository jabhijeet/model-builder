from __future__ import annotations
"""Project-local feature store — versioned, named feature sets backed by SQLite + parquet."""
import datetime
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import aiosqlite
import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feature_sets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    run_id INTEGER,
    created_at TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    columns_json TEXT NOT NULL,
    tags_json TEXT NOT NULL DEFAULT '[]',
    data_path TEXT NOT NULL,
    UNIQUE(name, version)
);
"""


@dataclass
class FeatureSetMeta:
    id: int
    name: str
    version: int
    row_count: int
    column_count: int
    columns: dict[str, str]
    created_at: datetime.datetime
    data_path: str
    run_id: Optional[int] = None
    tags: list[str] = field(default_factory=list)


class FeatureStore:
    """Project-scoped feature store. DB + data live in <project>/.modelbuilder/features/."""

    def __init__(self, project_dir: Path):
        self._base = project_dir / ".modelbuilder" / "features"
        self._base.mkdir(parents=True, exist_ok=True)
        self._db_path = project_dir / ".modelbuilder" / "feature_store.db"

    async def init(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()

    async def save(
        self,
        name: str,
        df: pd.DataFrame,
        run_id: Optional[int] = None,
        tags: Optional[list[str]] = None,
    ) -> FeatureSetMeta:
        """Save DataFrame as a new version of feature set <name>."""
        version = await self._next_version(name)
        data_dir = self._base / name / f"v{version:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / "data.parquet"
        df.to_parquet(data_path, index=False)

        columns = {col: str(df[col].dtype) for col in df.columns}
        now = datetime.datetime.now(datetime.UTC).isoformat()
        tags_list = tags or []

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO feature_sets
                  (name, version, run_id, created_at, row_count, column_count,
                   columns_json, tags_json, data_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (name, version, run_id, now, len(df), len(df.columns),
                 json.dumps(columns), json.dumps(tags_list), str(data_path)),
            )
            await db.commit()
            row_id = cursor.lastrowid

        return FeatureSetMeta(
            id=row_id, name=name, version=version,
            row_count=len(df), column_count=len(df.columns),
            columns=columns, created_at=datetime.datetime.fromisoformat(now),
            data_path=str(data_path), run_id=run_id, tags=tags_list,
        )

    async def load(self, name: str, version: Optional[int] = None) -> pd.DataFrame:
        """Load feature set by name. Latest version if version is None."""
        meta = await self.get(name, version)
        return pd.read_parquet(meta.data_path)

    async def get(self, name: str, version: Optional[int] = None) -> FeatureSetMeta:
        """Get metadata for a feature set. Latest version if version is None."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            if version is not None:
                cursor = await db.execute(
                    "SELECT * FROM feature_sets WHERE name=? AND version=?",
                    (name, version),
                )
            else:
                cursor = await db.execute(
                    "SELECT * FROM feature_sets WHERE name=? ORDER BY version DESC LIMIT 1",
                    (name,),
                )
            row = await cursor.fetchone()
        if row is None:
            ver_str = f" v{version}" if version else ""
            raise KeyError(f"Feature set '{name}{ver_str}' not found")
        return _row_to_meta(row)

    async def list(self) -> list[FeatureSetMeta]:
        """List all feature sets (latest version per name)."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT * FROM feature_sets
                WHERE version = (
                    SELECT MAX(version) FROM feature_sets f2
                    WHERE f2.name = feature_sets.name
                )
                ORDER BY name
            """)
            rows = await cursor.fetchall()
        return [_row_to_meta(r) for r in rows]

    async def list_versions(self, name: str) -> list[FeatureSetMeta]:
        """List all versions of a feature set."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM feature_sets WHERE name=? ORDER BY version",
                (name,),
            )
            rows = await cursor.fetchall()
        return [_row_to_meta(r) for r in rows]

    async def delete(self, name: str, version: Optional[int] = None) -> int:
        """Delete a feature set version (or all versions if version is None). Returns count deleted."""
        if version is not None:
            meta = await self.get(name, version)
            _delete_data(meta.data_path)
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    "DELETE FROM feature_sets WHERE name=? AND version=?", (name, version)
                )
                await db.commit()
            return 1

        versions = await self.list_versions(name)
        for m in versions:
            _delete_data(m.data_path)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM feature_sets WHERE name=?", (name,))
            await db.commit()
        return len(versions)

    async def _next_version(self, name: str) -> int:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT COALESCE(MAX(version), 0) FROM feature_sets WHERE name=?", (name,)
            )
            max_ver = (await cursor.fetchone())[0]
        return max_ver + 1


def _row_to_meta(row: aiosqlite.Row) -> FeatureSetMeta:
    return FeatureSetMeta(
        id=row["id"],
        name=row["name"],
        version=row["version"],
        run_id=row["run_id"],
        row_count=row["row_count"],
        column_count=row["column_count"],
        columns=json.loads(row["columns_json"]),
        tags=json.loads(row["tags_json"]),
        created_at=datetime.datetime.fromisoformat(row["created_at"]),
        data_path=row["data_path"],
    )


def _delete_data(data_path: str) -> None:
    p = Path(data_path).parent  # version dir
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)


