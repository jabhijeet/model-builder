import pytest
import pandas as pd
from pathlib import Path
from model_builder.connectors.sql import SQLConnector


@pytest.fixture
def sqlite_db(tmp_path: Path) -> str:
    from sqlalchemy import create_engine, text
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                age REAL,
                score REAL,
                label INTEGER
            )
        """))
        for i in range(5):
            conn.execute(text(
                "INSERT INTO customers VALUES (:id, :age, :score, :label)"
            ), {"id": i, "age": 20.0 + i, "score": 0.5 + i * 0.1, "label": i % 2})
        conn.commit()
    return f"sqlite:///{db_path}"


def test_connect_sqlite(sqlite_db: str):
    c = SQLConnector()
    conn = c.connect({"dsn": sqlite_db, "query": "SELECT * FROM customers"})
    assert conn.connector_name == "sql"


def test_sample_returns_dataframe(sqlite_db: str):
    c = SQLConnector()
    conn = c.connect({"dsn": sqlite_db, "query": "SELECT * FROM customers"})
    df = c.sample(conn, n=100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert "age" in df.columns


def test_sample_respects_n(sqlite_db: str):
    c = SQLConnector()
    conn = c.connect({"dsn": sqlite_db, "query": "SELECT * FROM customers"})
    df = c.sample(conn, n=3)
    assert len(df) <= 3


def test_profile_sqlite(sqlite_db: str):
    c = SQLConnector()
    conn = c.connect({"dsn": sqlite_db, "query": "SELECT * FROM customers"})
    profile = c.profile(conn)
    assert profile.row_count == 5
    assert profile.column_count == 4
    assert "tabular" in profile.data_types


async def test_stream_yields_chunks(sqlite_db: str):
    c = SQLConnector()
    conn = c.connect({"dsn": sqlite_db, "query": "SELECT * FROM customers", "chunk_size": 2})
    chunks = []
    async for chunk in c.stream(conn):
        chunks.append(chunk)
    assert sum(len(ch.data) for ch in chunks) == 5
