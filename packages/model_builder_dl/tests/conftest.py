import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from model_builder.plugins.protocol import DataProfile, DataBundle


@pytest.fixture
def tabular_bundle(tmp_path: Path) -> DataBundle:
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "feat_a": np.random.uniform(0, 10, n),
        "feat_b": np.random.uniform(-1, 1, n),
        "feat_c": np.random.uniform(0, 100, n),
        "label": np.random.randint(0, 2, n),
    })
    data_path = tmp_path / "data.parquet"
    df.to_parquet(data_path, index=False)
    profile = DataProfile(
        row_count=n, column_count=4,
        columns={"feat_a": "float64", "feat_b": "float64", "feat_c": "float64", "label": "int64"},
        nulls={k: 0 for k in ["feat_a", "feat_b", "feat_c", "label"]},
        data_types=["tabular"],
        sample_path=str(data_path),
    )
    return DataBundle(profile=profile, data_path=str(data_path))


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    """Create a tiny image dataset: 2 classes, 10 images each (8x8 px)."""
    for cls in ["cat", "dog"]:
        cls_dir = tmp_path / "images" / cls
        cls_dir.mkdir(parents=True)
        for i in range(10):
            arr = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i}.jpg")
    return tmp_path / "images"


@pytest.fixture
def image_bundle(tmp_path: Path, image_dir: Path) -> DataBundle:
    # For image plugin, data_path is used as a base; image_dir comes from config
    data_path = tmp_path / "placeholder.parquet"
    pd.DataFrame({"path": ["placeholder"]}).to_parquet(data_path)
    profile = DataProfile(
        row_count=20, column_count=1, columns={}, nulls={},
        data_types=["image"], sample_path=str(image_dir),
    )
    return DataBundle(profile=profile, data_path=str(data_path))
