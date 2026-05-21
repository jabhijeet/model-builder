"""GCS connector — reads files from Google Cloud Storage via DuckDB httpfs."""
import os
from typing import AsyncIterator
import duckdb
import pandas as pd
from ..plugins.protocol import Connection, DataProfile, Chunk


def _detect_format(path: str) -> str:
    lower = path.lower()
    for fmt in ("parquet", "json", "jsonl", "arrow", "csv"):
        if lower.endswith(f".{fmt}") or f".{fmt}" in lower:
            return fmt
    return "csv"


def _gcs_url(bucket: str, path: str) -> str:
    path = path.lstrip("/")
    return f"gs://{bucket}/{path}"


def _configure_duckdb(con: duckdb.DuckDBPyConnection, handle: dict) -> None:
    con.execute("INSTALL httpfs; LOAD httpfs;")
    key_path = (
        handle.get("service_account_key_path")
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    )
    hmac_key = handle.get("hmac_access_key") or os.environ.get("GCS_HMAC_ACCESS_KEY", "")
    hmac_secret = handle.get("hmac_secret") or os.environ.get("GCS_HMAC_SECRET", "")
    if hmac_key and hmac_secret:
        # HMAC credentials (S3-compatible GCS access)
        con.execute("SET s3_endpoint='storage.googleapis.com';")
        con.execute(f"SET s3_access_key_id='{hmac_key}';")
        con.execute(f"SET s3_secret_access_key='{hmac_secret}';")
    elif key_path and os.path.exists(key_path):
        # Service account JSON key
        con.execute(f"SET gcs_auth_type='service_account';")
        con.execute(f"SET gcs_service_account_key_file='{key_path}';")


def _read_df(handle: dict) -> pd.DataFrame:
    url = _gcs_url(handle["bucket"], handle["path"])
    fmt = handle["format"]
    con = duckdb.connect()
    _configure_duckdb(con, handle)
    if fmt == "parquet":
        return con.execute(f"SELECT * FROM read_parquet('{url}')").df()
    if fmt in ("json", "jsonl"):
        return con.execute(f"SELECT * FROM read_json_auto('{url}')").df()
    return con.execute(f"SELECT * FROM read_csv_auto('{url}')").df()


class GCSConnector:
    name = "gcs"

    def connect(self, config: dict) -> Connection:
        bucket = config.get("bucket") or config.get("gcs_bucket", "")
        path = config.get("path") or config.get("gcs_path", "")
        if not bucket:
            raise ValueError("GCSConnector requires 'bucket' in config")
        if not path:
            raise ValueError("GCSConnector requires 'path' in config")
        fmt = config.get("format") or _detect_format(path)
        return Connection(
            connector_name="gcs",
            handle={
                "bucket": bucket,
                "path": path,
                "format": fmt,
                "chunk_size": config.get("chunk_size", 10_000),
                "service_account_key_path": config.get("service_account_key_path"),
                "hmac_access_key": config.get("hmac_access_key"),
                "hmac_secret": config.get("hmac_secret"),
            },
        )

    def sample(self, conn: Connection, n: int) -> pd.DataFrame:
        url = _gcs_url(conn.handle["bucket"], conn.handle["path"])
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
        url = _gcs_url(conn.handle["bucket"], conn.handle["path"])
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
