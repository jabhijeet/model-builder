import pytest
from model_builder.plugins.registry import PluginRegistry
from model_builder.plugins.protocol import DataProfile


def test_get_missing_connector_raises():
    reg = PluginRegistry()
    with pytest.raises(KeyError, match="not installed"):
        reg.get_connector("connectors.file")


def test_get_missing_ml_plugin_raises():
    reg = PluginRegistry()
    with pytest.raises(KeyError, match="model-builder-classical"):
        reg.get_ml_plugin("ml.classical.random_forest")


def test_register_and_get_connector():
    reg = PluginRegistry()

    class FakeConnector:
        name = "fake"
        def connect(self, config): return None
        def sample(self, conn, n): return None
        def profile(self, conn): return None
        async def stream(self, conn): ...

    reg.register_connector("connectors.fake", FakeConnector())
    result = reg.get_connector("connectors.fake")
    assert result.name == "fake"


def test_register_and_get_ml_plugin():
    reg = PluginRegistry()

    class FakeML:
        name = "fake_ml"
        data_types = ["tabular"]
        requires_gpu = False
        def detect(self, profile): return 0.9
        def train(self, data, config, callbacks): return None
        def evaluate(self, artifact, data): return None
        def export(self, artifact, fmt): return None

    reg.register_ml_plugin("ml.fake", FakeML())
    result = reg.get_ml_plugin("ml.fake")
    assert result.name == "fake_ml"


def test_all_ml_plugins_empty():
    reg = PluginRegistry()
    assert reg.all_ml_plugins() == []
