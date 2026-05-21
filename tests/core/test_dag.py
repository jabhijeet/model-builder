import pytest
from pathlib import Path
from model_builder.core.dag import DAG, DAGValidationError
from model_builder.core.models import NodeDef, NodeType


def _make_dag(nodes: list) -> DAG:
    return DAG(nodes)


def test_dag_loads_nodes():
    nodes = [
        NodeDef("a", NodeType.TASK),
        NodeDef("b", NodeType.TASK, depends_on=["a"]),
    ]
    dag = _make_dag(nodes)
    assert "a" in dag.nodes
    assert "b" in dag.nodes


def test_dag_rejects_unknown_dependency():
    nodes = [NodeDef("b", NodeType.TASK, depends_on=["nonexistent"])]
    with pytest.raises(DAGValidationError, match="unknown node"):
        _make_dag(nodes)


def test_dag_rejects_cycle():
    nodes = [
        NodeDef("a", NodeType.TASK, depends_on=["b"]),
        NodeDef("b", NodeType.TASK, depends_on=["a"]),
    ]
    with pytest.raises(DAGValidationError, match="cycle"):
        _make_dag(nodes)


def test_upstream_of_direct():
    nodes = [
        NodeDef("a", NodeType.TASK),
        NodeDef("b", NodeType.TASK, depends_on=["a"]),
        NodeDef("c", NodeType.TASK, depends_on=["b"]),
    ]
    dag = _make_dag(nodes)
    assert dag.upstream_of("c") == {"a", "b"}


def test_upstream_of_no_deps():
    nodes = [NodeDef("a", NodeType.TASK)]
    dag = _make_dag(nodes)
    assert dag.upstream_of("a") == set()


def test_from_file(tmp_path: Path):
    yaml_content = """
nodes:
  - id: ingest
    type: task
    plugin: connectors.file
    config:
      paths: ["data/*.csv"]
  - id: validate
    type: task
    plugin: validators.schema
    depends_on: [ingest]
  - id: review
    type: gate
    depends_on: [validate]
    message: "Check data before training"
"""
    pipeline_file = tmp_path / "pipeline.yaml"
    pipeline_file.write_text(yaml_content)
    dag = DAG.from_file(pipeline_file)
    assert len(dag.nodes) == 3
    assert dag.nodes["review"].type == NodeType.GATE
    assert dag.nodes["review"].message == "Check data before training"
    assert dag.nodes["validate"].depends_on == ["ingest"]
