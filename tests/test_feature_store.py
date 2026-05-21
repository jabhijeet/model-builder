"""Feature store tests — FeatureStore, FeatureStoreConnector, FeatureStoreSavePlugin, CLI."""
import asyncio
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from typer.testing import CliRunner
from model_builder.cli.main import app
from model_builder.feature_store.store import FeatureStore
from model_builder.plugins.registry import PluginRegistry
from model_builder.plugins.core_protocol import CoreContext

runner = CliRunner()


@pytest.fixture
async def store(tmp_project: Path) -> FeatureStore:
    s = FeatureStore(tmp_project)
    await s.init()
    return s


@pytest.fixture
def sample_df() -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.uniform(20, 60, 50),
        "income": np.random.uniform(20000, 100000, 50),
        "label": np.random.randint(0, 2, 50),
    })


# --- FeatureStore core ---

async def test_save_creates_parquet(store: FeatureStore, sample_df: pd.DataFrame, tmp_project: Path):
    meta = await store.save("customer_features", sample_df)
    assert meta.name == "customer_features"
    assert meta.version == 1
    assert meta.row_count == 50
    assert meta.column_count == 3
    assert Path(meta.data_path).exists()


async def test_save_increments_version(store: FeatureStore, sample_df: pd.DataFrame):
    await store.save("feats", sample_df)
    meta2 = await store.save("feats", sample_df)
    assert meta2.version == 2


async def test_load_latest(store: FeatureStore, sample_df: pd.DataFrame):
    await store.save("feats", sample_df)
    df2 = pd.DataFrame({"x": range(10)})
    await store.save("feats", df2)
    loaded = await store.load("feats")
    assert len(loaded) == 10  # latest version


async def test_load_specific_version(store: FeatureStore, sample_df: pd.DataFrame):
    await store.save("feats", sample_df)
    small = pd.DataFrame({"x": range(5)})
    await store.save("feats", small)
    loaded_v1 = await store.load("feats", version=1)
    assert len(loaded_v1) == 50


async def test_get_missing_raises(store: FeatureStore):
    with pytest.raises(KeyError, match="not found"):
        await store.get("nonexistent")


async def test_list_returns_latest_per_name(store: FeatureStore, sample_df: pd.DataFrame):
    await store.save("a", sample_df)
    await store.save("a", sample_df)  # v2
    await store.save("b", sample_df)
    sets = await store.list()
    assert len(sets) == 2
    names = {s.name for s in sets}
    assert names == {"a", "b"}
    a_meta = next(s for s in sets if s.name == "a")
    assert a_meta.version == 2


async def test_list_versions(store: FeatureStore, sample_df: pd.DataFrame):
    await store.save("feats", sample_df)
    await store.save("feats", sample_df)
    await store.save("feats", sample_df)
    versions = await store.list_versions("feats")
    assert len(versions) == 3
    assert [v.version for v in versions] == [1, 2, 3]


async def test_save_with_tags(store: FeatureStore, sample_df: pd.DataFrame):
    meta = await store.save("tagged", sample_df, tags=["v1", "production"])
    assert meta.tags == ["v1", "production"]
    loaded = await store.get("tagged")
    assert loaded.tags == ["v1", "production"]


async def test_delete_specific_version(store: FeatureStore, sample_df: pd.DataFrame):
    await store.save("feats", sample_df)
    await store.save("feats", sample_df)
    count = await store.delete("feats", version=1)
    assert count == 1
    versions = await store.list_versions("feats")
    assert len(versions) == 1
    assert versions[0].version == 2


async def test_delete_all_versions(store: FeatureStore, sample_df: pd.DataFrame):
    await store.save("feats", sample_df)
    await store.save("feats", sample_df)
    count = await store.delete("feats")
    assert count == 2
    sets = await store.list()
    assert not any(s.name == "feats" for s in sets)


async def test_save_with_run_id(store: FeatureStore, sample_df: pd.DataFrame):
    meta = await store.save("feats", sample_df, run_id=42)
    assert meta.run_id == 42
    loaded = await store.get("feats")
    assert loaded.run_id == 42


# --- FeatureStoreConnector ---

def test_connector_missing_name_raises():
    from model_builder.connectors.feature_store import FeatureStoreConnector
    c = FeatureStoreConnector()
    with pytest.raises(ValueError, match="name"):
        c.connect({})


def test_connector_stores_config():
    from model_builder.connectors.feature_store import FeatureStoreConnector
    c = FeatureStoreConnector()
    conn = c.connect({"name": "customer_features", "version": 2})
    assert conn.connector_name == "feature_store"
    assert conn.handle["name"] == "customer_features"
    assert conn.handle["version"] == 2


def test_connector_registered():
    reg = PluginRegistry()
    reg.register_built_ins()
    assert reg.get_connector("connectors.feature_store").name == "feature_store"


# --- FeatureStoreSavePlugin ---

@pytest.fixture
def save_ctx(tmp_project: Path, sample_df: pd.DataFrame) -> CoreContext:
    run_dir = tmp_project / "runs" / "run_001"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (run_dir / "logs").mkdir()
    sample_df.to_parquet(artifacts_dir / "merged.parquet", index=False)

    reg = PluginRegistry()
    reg.register_built_ins()
    return CoreContext(
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        logs_dir=run_dir / "logs",
        run_id=1,
        registry=reg,
        node_config={"feature_name": "customer_features", "tags": ["test"]},
    )


def test_feature_store_save_plugin_writes_receipt(save_ctx: CoreContext, tmp_project: Path):
    from model_builder.core_plugins.feature_store_save_plugin import FeatureStoreSavePlugin
    out = FeatureStoreSavePlugin().run(save_ctx)
    assert out.exists()
    receipt = json.loads(out.read_text())
    assert receipt["feature_name"] == "customer_features"
    assert receipt["version"] == 1
    assert receipt["row_count"] == 50


def test_feature_store_save_creates_parquet_in_store(save_ctx: CoreContext, tmp_project: Path):
    from model_builder.core_plugins.feature_store_save_plugin import FeatureStoreSavePlugin
    FeatureStoreSavePlugin().run(save_ctx)
    # Verify parquet exists in feature store dir
    features_dir = tmp_project / ".modelbuilder" / "features" / "customer_features"
    assert features_dir.exists()
    assert len(list(features_dir.rglob("*.parquet"))) == 1


def test_feature_store_save_no_name_raises(save_ctx: CoreContext):
    from model_builder.core_plugins.feature_store_save_plugin import FeatureStoreSavePlugin
    save_ctx.node_config = {}
    with pytest.raises(ValueError, match="feature_name"):
        FeatureStoreSavePlugin().run(save_ctx)


def test_feature_store_save_registered():
    reg = PluginRegistry()
    reg.register_built_ins()
    plugin = reg.get_core_plugin("core.feature_store_save")
    assert plugin.name == "core.feature_store_save"


# --- CLI ---

def test_features_list_empty(tmp_project: Path):
    async def seed():
        await FeatureStore(tmp_project).init()

    asyncio.run(seed())
    result = runner.invoke(app, ["features", "list"],
                           env={"MB_PROJECT_DIR": str(tmp_project)})
    assert result.exit_code == 0
    assert "No feature sets" in result.output


def test_features_list_shows_saved(tmp_project: Path, sample_df: pd.DataFrame):
    async def seed():
        store = FeatureStore(tmp_project)
        await store.init()
        await store.save("customer_features", sample_df, tags=["v1"])

    asyncio.run(seed())
    result = runner.invoke(app, ["features", "list"],
                           env={"MB_PROJECT_DIR": str(tmp_project)})
    assert result.exit_code == 0
    assert "customer_features" in result.output


def test_features_info_shows_columns(tmp_project: Path, sample_df: pd.DataFrame):
    async def seed():
        store = FeatureStore(tmp_project)
        await store.init()
        await store.save("feats", sample_df)

    asyncio.run(seed())
    result = runner.invoke(app, ["features", "info", "feats"],
                           env={"MB_PROJECT_DIR": str(tmp_project)})
    assert result.exit_code == 0
    assert "age" in result.output
    assert "income" in result.output


def test_features_versions(tmp_project: Path, sample_df: pd.DataFrame):
    async def seed():
        store = FeatureStore(tmp_project)
        await store.init()
        await store.save("feats", sample_df)
        await store.save("feats", sample_df)

    asyncio.run(seed())
    result = runner.invoke(app, ["features", "versions", "feats"],
                           env={"MB_PROJECT_DIR": str(tmp_project)})
    assert result.exit_code == 0
    assert "1" in result.output
    assert "2" in result.output


def test_features_help():
    result = runner.invoke(app, ["features", "--help"])
    assert result.exit_code == 0
    for cmd in ["list", "info", "versions", "delete"]:
        assert cmd in result.output
