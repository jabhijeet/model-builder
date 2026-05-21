"""
End-to-end integration test: CSV -> ingest -> validate -> profile -> rank ->
train RF -> eval -> export -> DEPLOY.md
"""
import asyncio
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from model_builder.core.dag import DAG
from model_builder.core.store import ProjectStore
from model_builder.core.events import EventBus
from model_builder.core.gates import GateManager
from model_builder.core.models import NodeState
from model_builder.core.scheduler import Scheduler
from model_builder.plugins.registry import PluginRegistry


@pytest.fixture
def e2e_project(tmp_path: Path) -> Path:
    """Full project with sample CSV and complete pipeline."""
    project_dir = tmp_path / "e2e"
    (project_dir / "data" / "raw").mkdir(parents=True)
    (project_dir / "data" / "processed").mkdir()
    (project_dir / "runs").mkdir()
    (project_dir / ".modelbuilder").mkdir()
    (project_dir / ".modelbuilder" / "config.yaml").write_text("default_metric: accuracy\n")

    # Generate sample classification data
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        "age": np.random.uniform(20, 60, n),
        "income": np.random.uniform(20000, 100000, n),
        "score": np.random.uniform(0, 1, n),
        "label": np.random.randint(0, 2, n),
    })
    df.to_csv(project_dir / "data" / "raw" / "customers.csv", index=False)

    pipeline = """
nodes:
  - id: ingest
    type: task
    plugin: connectors.file
    config:
      paths: ["data/raw/customers.csv"]

  - id: merge
    type: task
    plugin: core.merge
    depends_on: [ingest]

  - id: validate
    type: task
    plugin: validators.schema
    depends_on: [merge]
    config:
      required_columns: [age, income, score, label]
      max_null_pct: 0.1

  - id: profile
    type: task
    plugin: core.profile
    depends_on: [merge]

  - id: rank_algos
    type: task
    plugin: core.automl_ranker
    depends_on: [profile]

  - id: review_data
    type: gate
    depends_on: [rank_algos, validate]
    message: "Review data profile and algorithm rankings"

  - id: train_rf
    type: task
    plugin: ml.classical.random_forest
    depends_on: [review_data]
    config:
      target_col: label
      n_estimators: 20

  - id: eval_join
    type: parallel_join
    depends_on: [train_rf]

  - id: review_results
    type: gate
    depends_on: [eval_join]
    message: "Review training results"

  - id: deploy_advisor
    type: task
    plugin: core.deploy_advisor
    depends_on: [review_results]
"""
    (project_dir / "pipeline.yaml").write_text(pipeline)
    return project_dir


async def _run_until_gate(project_dir: Path, store: ProjectStore, run_id: int,
                           run_dir: Path, registry: PluginRegistry) -> None:
    dag = DAG.from_file(project_dir / "pipeline.yaml")
    events = EventBus()
    scheduler = Scheduler(dag, store, registry, events, run_id, run_dir, poll_interval=0.05)
    await asyncio.wait_for(scheduler.run(), timeout=60.0)


async def test_full_pipeline_e2e(e2e_project: Path):
    store = ProjectStore(e2e_project)
    await store.init()

    registry = PluginRegistry()
    registry.discover()  # loads classical ML plugins

    run_name = await store.next_run_name()
    run_id = await store.create_run(run_name)
    run_dir = e2e_project / "runs" / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "artifacts").mkdir()
    (run_dir / "logs").mkdir()

    dag = DAG.from_file(e2e_project / "pipeline.yaml")
    gates = GateManager(store)

    # Phase 1: run until first gate (review_data)
    await _run_until_gate(e2e_project, store, run_id, run_dir, registry)

    nr_ingest = await store.get_node_run(run_id, "ingest")
    assert nr_ingest.state == NodeState.SUCCEEDED, f"ingest: {nr_ingest.state}"

    nr_profile = await store.get_node_run(run_id, "profile")
    assert nr_profile.state == NodeState.SUCCEEDED, f"profile: {nr_profile.state}"

    nr_rank = await store.get_node_run(run_id, "rank_algos")
    assert nr_rank.state == NodeState.SUCCEEDED, f"rank_algos: {nr_rank.state}"

    nr_gate1 = await store.get_node_run(run_id, "review_data")
    assert nr_gate1.state == NodeState.AWAITING_HUMAN, f"review_data: {nr_gate1.state}"

    # Verify profile.json produced
    profile_path = run_dir / "artifacts" / "profile.json"
    assert profile_path.exists()
    profile = json.loads(profile_path.read_text())
    assert profile["row_count"] == 300
    assert profile["column_count"] == 4

    # Verify ranking.json produced
    ranking_path = run_dir / "artifacts" / "ranking.json"
    assert ranking_path.exists()
    ranking = json.loads(ranking_path.read_text())
    assert len(ranking["rankings"]) >= 1

    # Approve gate 1
    await gates.approve(run_id, "review_data", dag.nodes["review_data"])

    # Phase 2: run until second gate (review_results)
    await _run_until_gate(e2e_project, store, run_id, run_dir, registry)

    nr_train = await store.get_node_run(run_id, "train_rf")
    assert nr_train.state == NodeState.SUCCEEDED, f"train_rf: {nr_train.state}"

    nr_gate2 = await store.get_node_run(run_id, "review_results")
    assert nr_gate2.state == NodeState.AWAITING_HUMAN, f"review_results: {nr_gate2.state}"

    # Check model artifact exists
    model_dirs = list((run_dir / "artifacts").glob("*_model"))
    assert len(model_dirs) >= 1

    # Approve gate 2
    await gates.approve(run_id, "review_results", dag.nodes["review_results"])

    # Phase 3: finish (deploy advisor)
    await _run_until_gate(e2e_project, store, run_id, run_dir, registry)

    nr_deploy = await store.get_node_run(run_id, "deploy_advisor")
    assert nr_deploy.state == NodeState.SUCCEEDED, f"deploy_advisor: {nr_deploy.state}"

    # Verify DEPLOY.md generated
    deploy_path = run_dir / "DEPLOY.md"
    assert deploy_path.exists()
    content = deploy_path.read_text()
    assert "Deployment Guide" in content
    assert "Option 1" in content
    assert "Option 2" in content
    assert "Option 3" in content

    # All nodes done
    all_runs = await store.get_all_node_runs(run_id)
    for nr in all_runs:
        assert nr.state.value in ("succeeded", "approved", "skipped"), \
            f"Node {nr.node_id} stuck in {nr.state}"
