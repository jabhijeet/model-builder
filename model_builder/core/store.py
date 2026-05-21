import datetime
from pathlib import Path
import aiosqlite
from .models import NodeRun, NodeState, Run

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    from_node_id TEXT,
    parent_run_id INTEGER
);
CREATE TABLE IF NOT EXISTS node_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    started_at TEXT,
    finished_at TEXT,
    error TEXT,
    output_path TEXT,
    UNIQUE(run_id, node_id)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT,
    created_at TEXT NOT NULL
);
"""


class ProjectStore:
    def __init__(self, project_dir: Path):
        self.db_path = project_dir / "project.db"

    async def init(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()

    async def next_run_name(self) -> str:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM runs")
            count = (await cursor.fetchone())[0]
            return f"run_{count + 1:03d}"

    async def create_run(
        self,
        name: str,
        from_node_id: str | None = None,
        parent_run_id: int | None = None,
    ) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "INSERT INTO runs (name, created_at, from_node_id, parent_run_id) VALUES (?, ?, ?, ?)",
                (name, datetime.datetime.now(datetime.UTC).isoformat(), from_node_id, parent_run_id),
            )
            await db.commit()
            return cursor.lastrowid

    async def upsert_node_run(self, nr: NodeRun) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO node_runs (run_id, node_id, state, started_at, finished_at, error, output_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, node_id) DO UPDATE SET
                    state=excluded.state,
                    started_at=excluded.started_at,
                    finished_at=excluded.finished_at,
                    error=excluded.error,
                    output_path=excluded.output_path
                """,
                (
                    nr.run_id, nr.node_id, nr.state.value,
                    nr.started_at.isoformat() if nr.started_at else None,
                    nr.finished_at.isoformat() if nr.finished_at else None,
                    nr.error, nr.output_path,
                ),
            )
            await db.commit()

    async def get_node_run(self, run_id: int, node_id: str) -> NodeRun | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM node_runs WHERE run_id=? AND node_id=?", (run_id, node_id)
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_node_run(row)

    async def get_all_node_runs(self, run_id: int) -> list[NodeRun]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM node_runs WHERE run_id=?", (run_id,)
            )
            rows = await cursor.fetchall()
        return [_row_to_node_run(r) for r in rows]

    async def list_runs(self) -> list[Run]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM runs ORDER BY id")
            rows = await cursor.fetchall()
        return [_row_to_run(r) for r in rows]

    async def get_latest_run(self) -> Run | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1")
            row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_run(row)

    async def get_run_by_name(self, name: str) -> Run | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM runs WHERE name=?", (name,))
            row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_run(row)


def _row_to_node_run(row: aiosqlite.Row) -> NodeRun:
    return NodeRun(
        run_id=row["run_id"],
        node_id=row["node_id"],
        state=NodeState(row["state"]),
        started_at=datetime.datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
        finished_at=datetime.datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
        error=row["error"],
        output_path=row["output_path"],
    )


def _row_to_run(row: aiosqlite.Row) -> Run:
    return Run(
        id=row["id"],
        name=row["name"],
        created_at=datetime.datetime.fromisoformat(row["created_at"]),
        from_node_id=row["from_node_id"],
        parent_run_id=row["parent_run_id"],
    )

