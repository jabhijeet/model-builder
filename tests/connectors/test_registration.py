from model_builder.plugins.registry import PluginRegistry


def test_built_in_connectors_registered():
    reg = PluginRegistry()
    reg.register_built_ins()
    assert reg.get_connector("connectors.file").name == "file"
    assert reg.get_connector("connectors.sql").name == "sql"
    assert reg.get_connector("connectors.rest_poll").name == "rest_poll"
    assert reg.get_connector("connectors.websocket").name == "websocket"
