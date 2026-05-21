import datetime
from .models import NodeRun, NodeState, NodeDef, NodeType
from .store import ProjectStore


class GateManager:
    def __init__(self, store: ProjectStore):
        self.store = store

    async def approve(self, run_id: int, node_id: str, node_def: NodeDef) -> None:
        if node_def.type != NodeType.GATE:
            raise ValueError(f"Node '{node_id}' is not a GATE node")
        nr = await self.store.get_node_run(run_id, node_id)
        if nr is None or nr.state != NodeState.AWAITING_HUMAN:
            state = nr.state if nr else "not found"
            raise ValueError(
                f"Node '{node_id}' is not awaiting human approval (state: {state})"
            )
        nr.state = NodeState.APPROVED
        nr.finished_at = datetime.datetime.now(datetime.UTC)
        await self.store.upsert_node_run(nr)

    async def skip(self, run_id: int, node_id: str) -> None:
        nr = await self.store.get_node_run(run_id, node_id)
        if nr is None:
            nr = NodeRun(
                run_id=run_id, node_id=node_id, state=NodeState.SKIPPED,
                finished_at=datetime.datetime.now(datetime.UTC),
            )
        else:
            nr.state = NodeState.SKIPPED
            nr.finished_at = datetime.datetime.now(datetime.UTC)
        await self.store.upsert_node_run(nr)

