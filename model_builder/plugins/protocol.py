from typing import Protocol, AsyncIterator, runtime_checkable
from pathlib import Path
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class DataProfile:
    row_count: int
    column_count: int
    columns: dict
    nulls: dict
    data_types: list
    sample_path: str


@dataclass
class DataBundle:
    profile: DataProfile
    data_path: str


@dataclass
class Artifact:
    model_path: str
    plugin_name: str
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalReport:
    plugin_name: str
    metrics: dict
    confusion_matrix: list | None = None
    feature_importance: dict | None = None
    extra: dict = field(default_factory=dict)


class ExportFormat(str):
    pass


ONNX = ExportFormat("onnx")
PICKLE = ExportFormat("pickle")
SAFETENSORS = ExportFormat("safetensors")
GGUF = ExportFormat("gguf")


@dataclass
class TrainCallbacks:
    on_progress: object
    on_log: object


@dataclass
class Connection:
    connector_name: str
    handle: object


@dataclass
class Chunk:
    data: pd.DataFrame
    sequence: int


@runtime_checkable
class Connector(Protocol):
    name: str

    def connect(self, config: dict) -> Connection: ...
    def sample(self, conn: Connection, n: int) -> pd.DataFrame: ...
    def profile(self, conn: Connection) -> DataProfile: ...
    async def stream(self, conn: Connection) -> AsyncIterator[Chunk]: ...


@runtime_checkable
class MLPlugin(Protocol):
    name: str
    data_types: list
    requires_gpu: bool

    def detect(self, profile: DataProfile) -> float: ...
    def train(self, data: DataBundle, config: dict, callbacks: TrainCallbacks) -> Artifact: ...
    def evaluate(self, artifact: Artifact, data: DataBundle) -> EvalReport: ...
    def export(self, artifact: Artifact, fmt: ExportFormat) -> Path: ...
