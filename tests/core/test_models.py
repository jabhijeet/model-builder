from model_builder.core.models import NodeType, NodeState, NodeDef, NodeRun, Run
import datetime


def test_node_type_values():
    assert NodeType.TASK == "task"
    assert NodeType.GATE == "gate"
    assert NodeType.PARALLEL_JOIN == "parallel_join"


def test_node_state_values():
    assert NodeState.PENDING == "pending"
    assert NodeState.AWAITING_HUMAN == "awaiting_human"


def test_node_def_defaults():
    node = NodeDef(id="test", type=NodeType.TASK)
    assert node.depends_on == []
    assert node.config == {}
    assert node.plugin is None
    assert node.message is None


def test_node_run_defaults():
    nr = NodeRun(run_id=1, node_id="test", state=NodeState.PENDING)
    assert nr.started_at is None
    assert nr.error is None
    assert nr.output_path is None


def test_run_fields():
    now = datetime.datetime.utcnow()
    r = Run(id=1, name="run_001", created_at=now)
    assert r.from_node_id is None
    assert r.parent_run_id is None
