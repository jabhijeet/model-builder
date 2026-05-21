"""Image connector — reads PNG/JPG/TIFF from a directory into a DataFrame."""
import asyncio
from pathlib import Path
from typing import AsyncIterator
import pandas as pd
from PIL import Image as PILImage
from ..plugins.protocol import Connection, DataProfile, Chunk

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def _collect_images(image_dir: Path, label_from_dir: bool) -> pd.DataFrame:
    records = []
    for f in sorted(image_dir.rglob("*")):
        if f.suffix.lower() not in _IMAGE_EXTS:
            continue
        label = f.parent.name if label_from_dir else None
        try:
            with PILImage.open(f) as img:
                w, h = img.size
                mode = img.mode
        except Exception:
            w = h = 0
            mode = "unknown"
        records.append({
            "image_path": str(f),
            "label": label,
            "width": w,
            "height": h,
            "mode": mode,
            "filename": f.name,
        })
    df = pd.DataFrame(records)
    if not label_from_dir:
        df = df.drop(columns=["label"])
    return df


class ImageConnector:
    name = "image"

    def connect(self, config: dict) -> Connection:
        image_dir = Path(config["image_dir"])
        if not image_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")
        return Connection(
            connector_name="image",
            handle={
                "image_dir": str(image_dir),
                "label_from_dir": config.get("label_from_dir", True),
                "chunk_size": config.get("chunk_size", 100),
            },
        )

    def sample(self, conn: Connection, n: int) -> pd.DataFrame:
        df = _collect_images(
            Path(conn.handle["image_dir"]),
            conn.handle["label_from_dir"],
        )
        return df.head(n)

    def profile(self, conn: Connection) -> DataProfile:
        df = _collect_images(
            Path(conn.handle["image_dir"]),
            conn.handle["label_from_dir"],
        )
        return DataProfile(
            row_count=len(df),
            column_count=len(df.columns),
            columns={col: str(df[col].dtype) for col in df.columns},
            nulls={col: int(df[col].isna().sum()) for col in df.columns},
            data_types=["image"],
            sample_path=conn.handle["image_dir"],
        )

    async def stream(self, conn: Connection) -> AsyncIterator[Chunk]:
        df = _collect_images(
            Path(conn.handle["image_dir"]),
            conn.handle["label_from_dir"],
        )
        chunk_size = conn.handle["chunk_size"]
        for i, start in enumerate(range(0, len(df), chunk_size)):
            yield Chunk(data=df.iloc[start: start + chunk_size], sequence=i)
