import pytest
from pathlib import Path
from model_builder.core.store import ProjectStore
from model_builder.core.gates import GateManager
from model_builder.core.models import NodeRun, NodeState, NodeDef, NodeType


@pytest.fixture
async def store_with_run(tmp_project: Path):
    store = ProjectStore(tmp_project)
    await store.init()
    run_id = await store.create_run("run_001")
    return store, run_id


async def test_approve_gate(store_with_run):
    store, run_id = store_with_run
    gate_def = NodeDef("review", NodeType.GATE)
    nr = NodeRun(run_id=run_id, node_id="review", state=NodeState.AWAITING_HUMAN)
    await store.upsert_node_run(nr)

    mgr = GateManager(store)
    await mgr.approve(run_id, "review", gate_def)

    result = await store.get_node_run(run_id, "review")
    assert result.state == NodeState.APPROVED
    assert result.finished_at is not None


async def test_approve_non_gate_raises(store_with_run):
    store, run_id = store_with_run
    task_def = NodeDef("task1", NodeType.TASK)
    nr = NodeRun(run_id=run_id, node_id="task1", state=NodeState.RUNNING)
    await store.upsert_node_run(nr)

    mgr = GateManager(store)
    with pytest.raises(ValueError, match="not a GATE"):
        await mgr.approve(run_id, "task1", task_def)


async def test_approve_wrong_state_raises(store_with_run):
    store, run_id = store_with_run
    gate_def = NodeDef("review", NodeType.GATE)
    nr = NodeRun(run_id=run_id, node_id="review", state=NodeState.PENDING)
    await store.upsert_node_run(nr)

    mgr = GateManager(store)
    with pytest.raises(ValueError, match="not awaiting"):
        await mgr.approve(run_id, "review", gate_def)


async def test_skip_node(store_with_run):
    store, run_id = store_with_run
    nr = NodeRun(run_id=run_id, node_id="optional_step", state=NodeState.PENDING)
    await store.upsert_node_run(nr)

    mgr = GateManager(store)
    await mgr.skip(run_id, "optional_step")

    result = await store.get_node_run(run_id, "optional_step")
    assert result.state == NodeState.SKIPPED


async def test_skip_creates_node_run_if_missing(store_with_run):
    store, run_id = store_with_run
    mgr = GateManager(store)
    await mgr.skip(run_id, "brand_new_node")
    result = await store.get_node_run(run_id, "brand_new_node")
    assert result.state == NodeState.SKIPPED
