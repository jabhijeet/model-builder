from importlib.metadata import entry_points


class PluginRegistry:
    def __init__(self):
        self._connectors: dict = {}
        self._ml_plugins: dict = {}
        self._core_plugins: dict = {}

    def register_core_plugin(self, name: str, plugin: object) -> None:
        self._core_plugins[name] = plugin

    def get_core_plugin(self, name: str) -> object:
        if name not in self._core_plugins:
            raise KeyError(f"Core plugin '{name}' not found. Available: {list(self._core_plugins)}")
        return self._core_plugins[name]

    def register_built_ins(self) -> None:
        from ..connectors.file import FileConnector
        from ..connectors.sql import SQLConnector
        from ..connectors.rest_poll import RestPollConnector
        from ..connectors.websocket_conn import WebSocketConnector
        self.register_connector("connectors.file", FileConnector())
        self.register_connector("connectors.sql", SQLConnector())
        self.register_connector("connectors.rest_poll", RestPollConnector())
        self.register_connector("connectors.websocket", WebSocketConnector())
        from ..connectors.image import ImageConnector
        from ..connectors.audio import AudioConnector
        from ..connectors.kafka_conn import KafkaConnector
        self.register_connector("connectors.image", ImageConnector())
        self.register_connector("connectors.audio", AudioConnector())
        self.register_connector("connectors.kafka", KafkaConnector())
        from ..connectors.s3 import S3Connector
        from ..connectors.gcs import GCSConnector
        self.register_connector("connectors.s3", S3Connector())
        self.register_connector("connectors.gcs", GCSConnector())
        from ..connectors.feature_store import FeatureStoreConnector
        self.register_connector("connectors.feature_store", FeatureStoreConnector())
        from ..core_plugins.merge_plugin import MergePlugin
        from ..core_plugins.profile_plugin import ProfilePlugin
        from ..core_plugins.validator_plugin import SchemaValidatorPlugin
        from ..core_plugins.automl_ranker_plugin import AutoMLRankerPlugin
        from ..core_plugins.export_plugin import ExportPlugin
        from ..core_plugins.deploy_advisor_plugin import DeployAdvisorPlugin
        from ..core_plugins.tuner_plugin import TunerPlugin
        from ..core_plugins.feature_store_save_plugin import FeatureStoreSavePlugin
        from ..core_plugins.model_update_plugin import ModelUpdatePlugin
        for p in [MergePlugin(), ProfilePlugin(), SchemaValidatorPlugin(),
                  AutoMLRankerPlugin(), ExportPlugin(), DeployAdvisorPlugin(), TunerPlugin(),
                  FeatureStoreSavePlugin(), ModelUpdatePlugin()]:
            self.register_core_plugin(p.name, p)

    def discover(self) -> None:
        self.register_built_ins()
        for ep in entry_points(group="model_builder.connectors"):
            self._connectors[f"connectors.{ep.name}"] = ep.load()()
        for ep in entry_points(group="model_builder.ml_plugins"):
            self._ml_plugins[f"ml.{ep.name}"] = ep.load()()

    def register_connector(self, name: str, connector: object) -> None:
        self._connectors[name] = connector

    def register_ml_plugin(self, name: str, plugin: object) -> None:
        self._ml_plugins[name] = plugin

    def get_connector(self, name: str) -> object:
        if name not in self._connectors:
            raise KeyError(
                f"Connector '{name}' not installed. Available: {list(self._connectors)}"
            )
        return self._connectors[name]

    def get_ml_plugin(self, name: str) -> object:
        if name not in self._ml_plugins:
            raise KeyError(
                f"ML plugin '{name}' not installed. Run: uv pip install aimodelground-classical"
            )
        return self._ml_plugins[name]

    def all_ml_plugins(self) -> list:
        return list(self._ml_plugins.values())







