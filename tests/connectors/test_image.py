import pytest
import numpy as np
from pathlib import Path
from PIL import Image
from model_builder.connectors.image import ImageConnector


@pytest.fixture
def image_dir_flat(tmp_path: Path) -> Path:
    """8 images in one dir (no label subdirs)."""
    d = tmp_path / "images"
    d.mkdir()
    for i in range(8):
        arr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        Image.fromarray(arr).save(d / f"img_{i}.jpg")
    return d


@pytest.fixture
def image_dir_labeled(tmp_path: Path) -> Path:
    """2 classes, 5 images each."""
    for cls in ["cats", "dogs"]:
        cls_dir = tmp_path / "labeled" / cls
        cls_dir.mkdir(parents=True)
        for i in range(5):
            arr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"{cls}_{i}.png")
    return tmp_path / "labeled"


def test_connect_returns_connection(image_dir_flat):
    c = ImageConnector()
    conn = c.connect({"image_dir": str(image_dir_flat), "label_from_dir": False})
    assert conn.connector_name == "image"


def test_connect_missing_dir_raises():
    c = ImageConnector()
    with pytest.raises(FileNotFoundError):
        c.connect({"image_dir": "/nonexistent/path"})


def test_sample_flat_no_labels(image_dir_flat):
    c = ImageConnector()
    conn = c.connect({"image_dir": str(image_dir_flat), "label_from_dir": False})
    df = c.sample(conn, n=100)
    assert len(df) == 8
    assert "image_path" in df.columns
    assert "label" not in df.columns
    assert "width" in df.columns
    assert "height" in df.columns


def test_sample_labeled(image_dir_labeled):
    c = ImageConnector()
    conn = c.connect({"image_dir": str(image_dir_labeled), "label_from_dir": True})
    df = c.sample(conn, n=100)
    assert len(df) == 10
    assert "label" in df.columns
    assert set(df["label"].unique()) == {"cats", "dogs"}


def test_sample_respects_n(image_dir_labeled):
    c = ImageConnector()
    conn = c.connect({"image_dir": str(image_dir_labeled), "label_from_dir": True})
    df = c.sample(conn, n=3)
    assert len(df) <= 3


def test_profile_data_types(image_dir_labeled):
    c = ImageConnector()
    conn = c.connect({"image_dir": str(image_dir_labeled), "label_from_dir": True})
    profile = c.profile(conn)
    assert profile.row_count == 10
    assert "image" in profile.data_types


def test_profile_dimensions_recorded(image_dir_flat):
    c = ImageConnector()
    conn = c.connect({"image_dir": str(image_dir_flat), "label_from_dir": False})
    df = c.sample(conn, n=1)
    assert df.iloc[0]["width"] == 16
    assert df.iloc[0]["height"] == 16


async def test_stream_yields_chunks(image_dir_labeled):
    c = ImageConnector()
    conn = c.connect({"image_dir": str(image_dir_labeled), "label_from_dir": True,
                      "chunk_size": 3})
    chunks = []
    async for chunk in c.stream(conn):
        chunks.append(chunk)
    assert len(chunks) >= 1
    assert sum(len(ch.data) for ch in chunks) == 10
