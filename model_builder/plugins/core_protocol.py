from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import PluginRegistry


@dataclass
class CoreContext:
    run_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    run_id: int
    registry: "PluginRegistry"
    node_config: dict
