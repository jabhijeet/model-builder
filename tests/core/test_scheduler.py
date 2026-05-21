import asyncio
import pytest
from pathlib import Path
from model_builder.core.store import ProjectStore
from model_builder.core.dag import DAG
from model_builder.core.models import NodeDef, NodeType, NodeState
from model_builder.core.events import EventBus
from model_builder.core.scheduler import Scheduler
from model_builder.plugins.registry import PluginRegistry
from model_builder.plugins.protocol import Connection, DataProfile
from pathlib import Path as _Path


@pytest.fixture
async def store(tmp_project: Path) -> ProjectStore:
    s = ProjectStore(tmp_project)
    await s.init()
    return s


def _make_registry(tmp_path: Path) -> PluginRegistry:
    reg = PluginRegistry()

    class _FastDF:
        """Minimal stand-in that avoids pyarrow cold-start in thread pool."""
        def to_parquet(self, path): _Path(path).touch()

    class FakeConn:
        name = "fake"
        def connect(self, config): return Connection("fake", None)
        def sample(self, conn, n): return _FastDF()
        def profile(self, conn): return DataProfile(2, 1, {}, {}, ["tabular"], "")
        async def stream(self, conn): ...

    reg.register_connector("connectors.fake", FakeConn())
    return reg


async def test_scheduler_runs_tasks_to_gate(tmp_project: Path, store: ProjectStore):
    dag = DAG([
        NodeDef("a", NodeType.TASK, plugin="connectors.fake"),
        NodeDef("b", NodeType.TASK, plugin="connectors.fake", depends_on=["a"]),
        NodeDef("gate", NodeType.GATE, depends_on=["b"], message="Review"),
        NodeDef("c", NodeType.TASK, plugin="connectors.fake", depends_on=["gate"]),
    ])
    events = EventBus()
    reg = _make_registry(tmp_project)
    run_id = await store.create_run("run_001")
    run_dir = tmp_project / "runs" / "run_001"
    run_dir.mkdir(parents=True)

    scheduler = Scheduler(dag, store, reg, events, run_id, run_dir, poll_interval=0.05)
    await asyncio.wait_for(scheduler.run(), timeout=30.0)

    nr_a = await store.get_node_run(run_id, "a")
    nr_b = await store.get_node_run(run_id, "b")
    nr_gate = await store.get_node_run(run_id, "gate")
    nr_c = await store.get_node_run(run_id, "c")

    assert nr_a.state == NodeState.SUCCEEDED
    assert nr_b.state == NodeState.SUCCEEDED
    assert nr_gate.state == NodeState.AWAITING_HUMAN
    assert nr_c.state == NodeState.PENDING


async def test_parallel_join_waits_for_all(tmp_project: Path, store: ProjectStore):
    dag = DAG([
        NodeDef("a", NodeType.TASK, plugin="connectors.fake"),
        NodeDef("b", NodeType.TASK, plugin="connectors.fake"),
        NodeDef("join", NodeType.PARALLEL_JOIN, depends_on=["a", "b"]),
        NodeDef("c", NodeType.TASK, plugin="connectors.fake", depends_on=["join"]),
    ])
    events = EventBus()
    reg = _make_registry(tmp_project)
    run_id = await store.create_run("run_001")
    run_dir = tmp_project / "runs" / "run_001"
    run_dir.mkdir(parents=True)

    scheduler = Scheduler(dag, store, reg, events, run_id, run_dir, poll_interval=0.05)
    await asyncio.wait_for(scheduler.run(), timeout=5.0)

    for nid in ["a", "b", "join", "c"]:
        nr = await store.get_node_run(run_id, nid)
        assert nr.state == NodeState.SUCCEEDED, f"{nid} state: {nr.state}"


async def test_events_emitted(tmp_project: Path, store: ProjectStore):
    dag = DAG([NodeDef("a", NodeType.TASK, plugin="connectors.fake")])
    events = EventBus()
    emitted = []

    async def capture(*args): emitted.append(args)
    events.subscribe(capture)

    reg = _make_registry(tmp_project)
    run_id = await store.create_run("run_001")
    run_dir = tmp_project / "runs" / "run_001"
    run_dir.mkdir(parents=True)

    scheduler = Scheduler(dag, store, reg, events, run_id, run_dir, poll_interval=0.05)
    await asyncio.wait_for(scheduler.run(), timeout=3.0)

    event_types = [e[2] for e in emitted]
    assert "state_change" in event_types
