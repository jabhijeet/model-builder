"""S3 connector — reads files from S3 via DuckDB httpfs extension."""
import os
from typing import AsyncIterator
import duckdb
import pandas as pd
from ..plugins.protocol import Connection, DataProfile, Chunk

_SUPPORTED_FORMATS = {"csv", "json", "parquet", "arrow", "jsonl"}


def _detect_format(path: str) -> str:
    lower = path.lower()
    for fmt in ("parquet", "json", "jsonl", "arrow", "csv"):
        if lower.endswith(f".{fmt}") or f".{fmt}" in lower:
            return fmt
    return "csv"


def _s3_url(bucket: str, path: str) -> str:
    path = path.lstrip("/")
    return f"s3://{bucket}/{path}"


def _configure_duckdb(con: duckdb.DuckDBPyConnection, handle: dict) -> None:
    con.execute("INSTALL httpfs; LOAD httpfs;")
    region = handle.get("region") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    key = handle.get("aws_access_key_id") or os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret = handle.get("aws_secret_access_key") or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    token = handle.get("aws_session_token") or os.environ.get("AWS_SESSION_TOKEN", "")
    con.execute(f"SET s3_region='{region}';")
    if key:
        con.execute(f"SET s3_access_key_id='{key}';")
    if secret:
        con.execute(f"SET s3_secret_access_key='{secret}';")
    if token:
        con.execute(f"SET s3_session_token='{token}';")
    endpoint = handle.get("endpoint_url") or os.environ.get("AWS_ENDPOINT_URL", "")
    if endpoint:
        con.execute(f"SET s3_endpoint='{endpoint}';")
        con.execute("SET s3_use_ssl=false; SET s3_url_style='path';")


def _read_df(handle: dict) -> pd.DataFrame:
    url = _s3_url(handle["bucket"], handle["path"])
    fmt = handle["format"]
    con = duckdb.connect()
    _configure_duckdb(con, handle)
    if fmt == "parquet":
        return con.execute(f"SELECT * FROM read_parquet('{url}')").df()
    if fmt in ("json", "jsonl"):
        return con.execute(f"SELECT * FROM read_json_auto('{url}')").df()
    return con.execute(f"SELECT * FROM read_csv_auto('{url}')").df()


class S3Connector:
    name = "s3"

    def connect(self, config: dict) -> Connection:
        bucket = config.get("bucket") or config.get("s3_bucket", "")
        path = config.get("path") or config.get("s3_path", "")
        if not bucket:
            raise ValueError("S3Connector requires 'bucket' in config")
        if not path:
            raise ValueError("S3Connector requires 'path' in config")
        fmt = config.get("format") or _detect_format(path)
        return Connection(
            connector_name="s3",
            handle={
                "bucket": bucket,
                "path": path,
                "format": fmt,
                "chunk_size": config.get("chunk_size", 10_000),
                "region": config.get("region"),
                "aws_access_key_id": config.get("aws_access_key_id"),
                "aws_secret_access_key": config.get("aws_secret_access_key"),
                "aws_session_token": config.get("aws_session_token"),
                "endpoint_url": config.get("endpoint_url"),
            },
        )

    def sample(self, conn: Connection, n: int) -> pd.DataFrame:
        url = _s3_url(conn.handle["bucket"], conn.handle["path"])
        fmt = conn.handle["format"]
        con = duckdb.connect()
        _configure_duckdb(con, conn.handle)
        if fmt == "parquet":
            return con.execute(f"SELECT * FROM read_parquet('{url}') LIMIT {n}").df()
        if fmt in ("json", "jsonl"):
            return con.execute(f"SELECT * FROM read_json_auto('{url}') LIMIT {n}").df()
        return con.execute(f"SELECT * FROM read_csv_auto('{url}') LIMIT {n}").df()

    def profile(self, conn: Connection) -> DataProfile:
        df = _read_df(conn.handle)
        url = _s3_url(conn.handle["bucket"], conn.handle["path"])
        return DataProfile(
            row_count=len(df),
            column_count=len(df.columns),
            columns={col: str(df[col].dtype) for col in df.columns},
            nulls={col: int(df[col].isna().sum()) for col in df.columns},
            data_types=["tabular"],
            sample_path=url,
        )

    async def stream(self, conn: Connection) -> AsyncIterator[Chunk]:
        df = _read_df(conn.handle)
        chunk_size = conn.handle["chunk_size"]
        for i, start in enumerate(range(0, len(df), chunk_size)):
            yield Chunk(data=df.iloc[start: start + chunk_size], sequence=i)
