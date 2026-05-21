import asyncio
import datetime
import logging
from pathlib import Path
from .models import NodeDef, NodeRun, NodeState, NodeType, TERMINAL_STATES, UNBLOCKING_STATES
from .store import ProjectStore
from .dag import DAG
from .events import EventBus
from .runner import NodeRunner
from ..plugins.registry import PluginRegistry

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(
        self,
        dag: DAG,
        store: ProjectStore,
        registry: PluginRegistry,
        events: EventBus,
        run_id: int,
        run_dir: Path,
        poll_interval: float = 2.0,
    ):
        self.dag = dag
        self.store = store
        self.registry = registry
        self.events = events
        self.run_id = run_id
        self.run_dir = run_dir
        self.poll_interval = poll_interval
        self._runner = NodeRunner(registry, run_dir)
        self._active_tasks: dict[str, asyncio.Task] = {}

    async def run(self) -> None:
        await self._init_pending_nodes()
        while True:
            await self._tick()
            if await self._is_done():
                break
            await asyncio.sleep(self.poll_interval)

    async def _init_pending_nodes(self) -> None:
        existing = {nr.node_id for nr in await self.store.get_all_node_runs(self.run_id)}
        for node_id in self.dag.nodes:
            if node_id not in existing:
                await self.store.upsert_node_run(
                    NodeRun(run_id=self.run_id, node_id=node_id, state=NodeState.PENDING)
                )

    async def _tick(self) -> None:
        node_runs = {nr.node_id: nr for nr in await self.store.get_all_node_runs(self.run_id)}
        for node_id, node_def in self.dag.nodes.items():
            nr = node_runs.get(node_id)
            if nr is None or nr.state != NodeState.PENDING:
                continue
            if node_id in self._active_tasks:
                continue
            if self._deps_satisfied(node_def, node_runs):
                await self._dispatch(node_id, node_def)

    def _deps_satisfied(self, node_def: NodeDef, node_runs: dict) -> bool:
        for dep_id in node_def.depends_on:
            dep = node_runs.get(dep_id)
            if dep is None or dep.state not in UNBLOCKING_STATES:
                return False
        return True

    async def _dispatch(self, node_id: str, node_def: NodeDef) -> None:
        if node_def.type == NodeType.GATE:
            nr = NodeRun(
                run_id=self.run_id, node_id=node_id,
                state=NodeState.AWAITING_HUMAN,
                started_at=datetime.datetime.now(datetime.UTC),
            )
            await self.store.upsert_node_run(nr)
            await self.events.emit(self.run_id, node_id, "awaiting_human",
                                   {"message": node_def.message})
            return

        if node_def.type == NodeType.PARALLEL_JOIN:
            now = datetime.datetime.now(datetime.UTC)
            nr = NodeRun(
                run_id=self.run_id, node_id=node_id,
                state=NodeState.SUCCEEDED,
                started_at=now, finished_at=now,
            )
            await self.store.upsert_node_run(nr)
            await self.events.emit(self.run_id, node_id, "state_change",
                                   {"state": NodeState.SUCCEEDED.value})
            return

        task = asyncio.create_task(self._run_task(node_id, node_def))
        self._active_tasks[node_id] = task

    async def _run_task(self, node_id: str, node_def: NodeDef) -> None:
        nr = NodeRun(
            run_id=self.run_id, node_id=node_id,
            state=NodeState.RUNNING,
            started_at=datetime.datetime.now(datetime.UTC),
        )
        await self.store.upsert_node_run(nr)
        await self.events.emit(self.run_id, node_id, "state_change",
                               {"state": NodeState.RUNNING.value})
        try:
            output_path = await self._runner.execute(node_def, self.run_id)
            nr.state = NodeState.SUCCEEDED
            nr.output_path = str(output_path) if output_path else None
        except Exception as exc:
            logger.exception("Node %s failed", node_id)
            nr.state = NodeState.FAILED
            nr.error = str(exc)
        finally:
            nr.finished_at = datetime.datetime.now(datetime.UTC)
            await self.store.upsert_node_run(nr)
            await self.events.emit(self.run_id, node_id, "state_change",
                                   {"state": nr.state.value})
            self._active_tasks.pop(node_id, None)

    async def _is_done(self) -> bool:
        node_runs = await self.store.get_all_node_runs(self.run_id)
        if len(node_runs) < len(self.dag.nodes):
            return False
        if all(nr.state in TERMINAL_STATES for nr in node_runs):
            return True
        if self._active_tasks:
            return False
        # No tasks running — done if no pending node can be dispatched
        nr_map = {nr.node_id: nr for nr in node_runs}
        for node_id, node_def in self.dag.nodes.items():
            nr = nr_map.get(node_id)
            if nr and nr.state == NodeState.PENDING and self._deps_satisfied(node_def, nr_map):
                return False
        return True

