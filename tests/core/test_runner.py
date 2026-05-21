import pytest
from pathlib import Path
from model_builder.core.runner import NodeRunner
from model_builder.core.models import NodeDef, NodeType
from model_builder.plugins.registry import PluginRegistry
from model_builder.plugins.protocol import DataProfile, Connection
import pandas as pd


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "logs").mkdir()
    return tmp_path


@pytest.fixture
def registry_with_connector(run_dir: Path) -> PluginRegistry:
    reg = PluginRegistry()
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "label": [0, 1]})

    class FakeConnector:
        name = "fake"
        def connect(self, config): return Connection("fake", None)
        def sample(self, conn, n): return df
        def profile(self, conn): return DataProfile(2, 3, {}, {}, ["tabular"], "")
        async def stream(self, conn): ...

    reg.register_connector("connectors.fake", FakeConnector())
    return reg


async def test_connector_node_writes_parquet(run_dir: Path, registry_with_connector: PluginRegistry):
    runner = NodeRunner(registry_with_connector, run_dir)
    node_def = NodeDef("ingest", NodeType.TASK, plugin="connectors.fake", config={})
    output = await runner.execute(node_def, run_id=1)
    assert output is not None
    assert output.exists()
    assert output.suffix == ".parquet"


async def test_node_with_no_plugin_returns_none(run_dir: Path):
    reg = PluginRegistry()
    runner = NodeRunner(reg, run_dir)
    node_def = NodeDef("join", NodeType.PARALLEL_JOIN)
    output = await runner.execute(node_def, run_id=1)
    assert output is None


async def test_unknown_plugin_raises(run_dir: Path):
    reg = PluginRegistry()
    runner = NodeRunner(reg, run_dir)
    node_def = NodeDef("ingest", NodeType.TASK, plugin="connectors.missing", config={})
    with pytest.raises(KeyError, match="not installed"):
        await runner.execute(node_def, run_id=1)
