import asyncio
from pathlib import Path
from .models import NodeDef
from ..plugins.registry import PluginRegistry
from ..plugins.protocol import TrainCallbacks, DataBundle, DataProfile, Artifact
from ..plugins.core_protocol import CoreContext


class NodeRunner:
    def __init__(self, registry: PluginRegistry, run_dir: Path):
        self.registry = registry
        self.run_dir = run_dir

    async def execute(self, node_def: NodeDef, run_id: int) -> Path | None:
        if node_def.plugin is None:
            return None

        artifacts_dir = self.run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = self.run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        if node_def.plugin.startswith("connectors."):
            return await self._run_connector(node_def, artifacts_dir)

        if node_def.plugin.startswith("ml."):
            return await self._run_ml_plugin(node_def, artifacts_dir, logs_dir)

        if node_def.plugin.startswith("core.") or node_def.plugin.startswith("validators."):
            return await self._run_core_plugin(node_def, artifacts_dir, logs_dir, run_id)

        return None

    def _resolve_config(self, config: dict) -> dict:
        """Resolve relative file paths in connector config against project dir."""
        project_dir = self.run_dir.parent.parent
        if "paths" not in config:
            return config
        resolved = dict(config)
        resolved["paths"] = [
            str(project_dir / p) if not Path(p).is_absolute() else p
            for p in config["paths"]
        ]
        return resolved

    async def _run_connector(self, node_def: NodeDef, artifacts_dir: Path) -> Path:
        connector = self.registry.get_connector(node_def.plugin)
        config = self._resolve_config(node_def.config)
        conn = connector.connect(config)
        df = connector.sample(conn, n=100_000)
        output_path = artifacts_dir / f"{node_def.id}.parquet"
        await asyncio.to_thread(df.to_parquet, output_path)
        return output_path

    async def _run_core_plugin(
        self, node_def: NodeDef, artifacts_dir: Path, logs_dir: Path, run_id: int
    ) -> Path | None:
        plugin = self.registry.get_core_plugin(node_def.plugin)
        ctx = CoreContext(
            run_dir=self.run_dir,
            artifacts_dir=artifacts_dir,
            logs_dir=logs_dir,
            run_id=run_id,
            registry=self.registry,
            node_config=node_def.config,
        )
        return await asyncio.to_thread(plugin.run, ctx)

    async def _run_ml_plugin(
        self, node_def: NodeDef, artifacts_dir: Path, logs_dir: Path
    ) -> Path:
        ml_plugin = self.registry.get_ml_plugin(node_def.plugin)
        data_path = artifacts_dir / "merged.parquet"
        log_file = logs_dir / f"{node_def.id}.log"

        profile = DataProfile(
            row_count=0, column_count=0, columns={}, nulls={},
            data_types=["tabular"], sample_path=str(data_path),
        )
        bundle = DataBundle(profile=profile, data_path=str(data_path))

        def on_progress(epoch: int, total: int, metrics: dict) -> None:
            with open(log_file, "a") as f:
                f.write(f"[epoch {epoch}/{total}] {metrics}\n")

        def on_log(msg: str) -> None:
            with open(log_file, "a") as f:
                f.write(f"{msg}\n")

        callbacks = TrainCallbacks(on_progress=on_progress, on_log=on_log)
        artifact: Artifact = await asyncio.to_thread(
            ml_plugin.train, bundle, node_def.config, callbacks
        )
        return Path(artifact.model_path)
