import asyncio
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from httpx import AsyncClient, ASGITransport
from model_builder.web.app import create_app
from model_builder.core.store import ProjectStore
from model_builder.core.models import NodeRun, NodeState


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    p = tmp_path / "proj"
    (p / "data" / "raw").mkdir(parents=True)
    (p / "data" / "processed").mkdir()
    (p / "runs").mkdir()
    (p / ".modelbuilder").mkdir()
    (p / ".modelbuilder" / "config.yaml").write_text("default_metric: f1\n")
    (p / "pipeline.yaml").write_text("""
nodes:
  - id: ingest
    type: task
    plugin: connectors.file
    config:
      paths: ["data/raw/*.csv"]
  - id: review
    type: gate
    depends_on: [ingest]
    message: "Check data before training"
""")
    return p


@pytest.fixture
async def seeded_project(project_dir: Path) -> Path:
    store = ProjectStore(project_dir)
    await store.init()
    run_id = await store.create_run("run_001")
    await store.upsert_node_run(NodeRun(run_id=run_id, node_id="ingest", state=NodeState.SUCCEEDED,
                                        output_path="runs/run_001/artifacts/ingest.parquet"))
    await store.upsert_node_run(NodeRun(run_id=run_id, node_id="review", state=NodeState.AWAITING_HUMAN))
    return project_dir


@pytest.fixture
def app(seeded_project):
    return create_app(seeded_project)


async def test_pipeline_page_returns_200(app, seeded_project):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/")
    assert r.status_code == 200
    assert "ingest" in r.text
    assert "review" in r.text


async def test_pipeline_page_shows_state(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/")
    assert "succeeded" in r.text
    assert "awaiting_human" in r.text


async def test_approve_gate_updates_state(app, seeded_project):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/approve/review")
    assert r.status_code == 200

    store = ProjectStore(seeded_project)
    await store.init()
    run = await store.get_latest_run()
    nr = await store.get_node_run(run.id, "review")
    assert nr.state == NodeState.APPROVED


async def test_skip_node(app, seeded_project):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/skip/review")
    assert r.status_code == 200

    store = ProjectStore(seeded_project)
    await store.init()
    run = await store.get_latest_run()
    nr = await store.get_node_run(run.id, "review")
    assert nr.state == NodeState.SKIPPED


async def test_data_page_returns_200(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/data")
    assert r.status_code == 200


async def test_data_page_shows_uploaded_files(app, seeded_project):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.to_csv(seeded_project / "data" / "raw" / "sample.csv", index=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/data")
    assert "sample.csv" in r.text


async def test_upload_file(app, seeded_project):
    csv_content = b"a,b\n1,2\n3,4\n"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/upload",
                              files={"file": ("test.csv", csv_content, "text/csv")})
    assert r.status_code == 200
    assert (seeded_project / "data" / "raw" / "test.csv").exists()


async def test_results_page_returns_200(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/results")
    assert r.status_code == 200


async def test_results_page_shows_eval_report(app, seeded_project):
    run_dir = seeded_project / "runs" / "run_001"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "eval_report.json").write_text(json.dumps({
        "best_plugin": "random_forest",
        "metrics": {"accuracy": 0.92, "f1": 0.91},
        "feature_importance": {"age": 0.6, "score": 0.4},
    }))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/results")
    assert "0.9200" in r.text or "accuracy" in r.text


async def test_deploy_page_returns_200(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/deploy")
    assert r.status_code == 200


async def test_deploy_page_shows_deploy_md(app, seeded_project):
    run_dir = seeded_project / "runs" / "run_001"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "DEPLOY.md").write_text("# Deployment Guide\n\n## Option 1 — Python\n## Option 2 — FastAPI\n## Option 3 — Docker\n")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/deploy")
    assert "Deployment Guide" in r.text


async def test_pipeline_nodes_partial(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/pipeline-nodes")
    assert r.status_code == 200
    assert "ingest" in r.text
