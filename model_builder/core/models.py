from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import datetime


class NodeType(str, Enum):
    TASK = "task"
    GATE = "gate"
    PARALLEL_JOIN = "parallel_join"


class NodeState(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    AWAITING_HUMAN = "awaiting_human"
    APPROVED = "approved"
    SKIPPED = "skipped"


UNBLOCKING_STATES: frozenset = frozenset({
    NodeState.SUCCEEDED, NodeState.SKIPPED, NodeState.APPROVED
})

TERMINAL_STATES: frozenset = frozenset({
    NodeState.SUCCEEDED, NodeState.FAILED, NodeState.SKIPPED, NodeState.APPROVED
})


@dataclass
class NodeDef:
    id: str
    type: NodeType
    depends_on: list = field(default_factory=list)
    plugin: Optional[str] = None
    config: dict = field(default_factory=dict)
    message: Optional[str] = None


@dataclass
class NodeRun:
    run_id: int
    node_id: str
    state: NodeState
    started_at: Optional[datetime.datetime] = None
    finished_at: Optional[datetime.datetime] = None
    error: Optional[str] = None
    output_path: Optional[str] = None


@dataclass
class Run:
    id: int
    name: str
    created_at: datetime.datetime
    from_node_id: Optional[str] = None
    parent_run_id: Optional[int] = None
