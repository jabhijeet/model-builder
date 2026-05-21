from pathlib import Path
import yaml
from .models import NodeDef, NodeType


class DAGValidationError(Exception):
    pass


class DAG:
    def __init__(self, nodes: list):
        self.nodes: dict[str, NodeDef] = {n.id: n for n in nodes}
        self._validate()

    def _validate(self) -> None:
        for node in self.nodes.values():
            for dep in node.depends_on:
                if dep not in self.nodes:
                    raise DAGValidationError(
                        f"Node '{node.id}' depends on unknown node '{dep}'"
                    )
        if self._has_cycle():
            raise DAGValidationError("Pipeline contains a cycle")

    def _has_cycle(self) -> bool:
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            for dep in self.nodes[node_id].depends_on:
                if dep not in visited:
                    if dfs(dep):
                        return True
                elif dep in rec_stack:
                    return True
            rec_stack.discard(node_id)
            return False

        return any(dfs(n) for n in list(self.nodes) if n not in visited)

    def upstream_of(self, node_id: str) -> set[str]:
        result: set[str] = set()
        queue = list(self.nodes[node_id].depends_on)
        while queue:
            nid = queue.pop()
            if nid not in result:
                result.add(nid)
                queue.extend(self.nodes[nid].depends_on)
        return result

    @classmethod
    def from_file(cls, path: Path) -> "DAG":
        data = yaml.safe_load(path.read_text())
        nodes = [
            NodeDef(
                id=n["id"],
                type=NodeType(n["type"]),
                depends_on=n.get("depends_on", []),
                plugin=n.get("plugin"),
                config=n.get("config", {}),
                message=n.get("message"),
            )
            for n in data["nodes"]
        ]
        return cls(nodes)
