import pytest
import datetime
from pathlib import Path
from model_builder.core.store import ProjectStore
from model_builder.core.models import NodeRun, NodeState, Run


@pytest.fixture
async def store(tmp_project: Path) -> ProjectStore:
    s = ProjectStore(tmp_project)
    await s.init()
    return s


async def test_init_creates_db(tmp_project: Path):
    store = ProjectStore(tmp_project)
    await store.init()
    assert (tmp_project / "project.db").exists()


async def test_next_run_name_first(store: ProjectStore):
    name = await store.next_run_name()
    assert name == "run_001"


async def test_next_run_name_increments(store: ProjectStore):
    await store.create_run("run_001")
    await store.create_run("run_002")
    name = await store.next_run_name()
    assert name == "run_003"


async def test_create_and_get_run(store: ProjectStore):
    run_id = await store.create_run("run_001")
    runs = await store.list_runs()
    assert len(runs) == 1
    assert runs[0].name == "run_001"
    assert runs[0].id == run_id


async def test_upsert_and_get_node_run(store: ProjectStore):
    run_id = await store.create_run("run_001")
    nr = NodeRun(run_id=run_id, node_id="ingest", state=NodeState.PENDING)
    await store.upsert_node_run(nr)
    result = await store.get_node_run(run_id, "ingest")
    assert result.state == NodeState.PENDING


async def test_upsert_updates_existing(store: ProjectStore):
    run_id = await store.create_run("run_001")
    nr = NodeRun(run_id=run_id, node_id="ingest", state=NodeState.PENDING)
    await store.upsert_node_run(nr)
    nr.state = NodeState.RUNNING
    nr.started_at = datetime.datetime.utcnow()
    await store.upsert_node_run(nr)
    result = await store.get_node_run(run_id, "ingest")
    assert result.state == NodeState.RUNNING
    assert result.started_at is not None


async def test_get_node_run_not_found(store: ProjectStore):
    run_id = await store.create_run("run_001")
    result = await store.get_node_run(run_id, "nonexistent")
    assert result is None


async def test_get_all_node_runs(store: ProjectStore):
    run_id = await store.create_run("run_001")
    for nid in ["a", "b", "c"]:
        await store.upsert_node_run(NodeRun(run_id=run_id, node_id=nid, state=NodeState.PENDING))
    results = await store.get_all_node_runs(run_id)
    assert len(results) == 3


async def test_get_latest_run(store: ProjectStore):
    await store.create_run("run_001")
    await store.create_run("run_002")
    latest = await store.get_latest_run()
    assert latest.name == "run_002"


async def test_get_latest_run_empty(store: ProjectStore):
    result = await store.get_latest_run()
    assert result is None
