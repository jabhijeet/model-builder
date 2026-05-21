# tests/web/test_new_endpoints.py
import asyncio
import json
import pytest
import pandas as pd
from pathlib import Path
from httpx import AsyncClient, ASGITransport
from model_builder.web.app import create_app


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    p = tmp_path / "proj"
    (p / "data" / "raw").mkdir(parents=True)
    (p / "runs").mkdir()
    (p / ".modelbuilder").mkdir()
    (p / ".modelbuilder" / "config.yaml").write_text("default_metric: f1\n")
    (p / "pipeline.yaml").write_text(
        'nodes:\n  - id: ingest\n    plugin: connectors.file\n    config:\n      paths: ["data/raw/*.csv"]\n'
    )
    return p


@pytest.fixture
def app(project_dir):
    return create_app(project_dir)


# ── /api/file-info ──────────────────────────────────────────────

async def test_file_info_csv_returns_columns(app, project_dir):
    df = pd.DataFrame({
        "sepal_length": [5.1, 4.9],
        "sepal_width": [3.5, 3.0],
        "species": ["setosa", "versicolor"],
    })
    df.to_csv(project_dir / "data" / "raw" / "iris.csv", index=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/file-info/iris.csv")
    assert r.status_code == 200
    data = r.json()
    assert data["rows"] == 2
    assert any(col["name"] == "sepal_length" for col in data["columns"])
    assert data["detected_target"] == "species"


async def test_file_info_missing_returns_404(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/file-info/nope.csv")
    assert r.status_code == 404


# ── /api/yaml ───────────────────────────────────────────────────

async def test_get_yaml_returns_content(app, project_dir):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/yaml")
    assert r.status_code == 200
    assert "connectors.file" in r.text


async def test_get_yaml_missing_returns_404(app, project_dir):
    (project_dir / "pipeline.yaml").unlink()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/yaml")
    assert r.status_code == 404


async def test_post_yaml_writes_file(app, project_dir):
    new_yaml = 'nodes:\n  - id: new_node\n    plugin: connectors.file\n    config:\n      paths: ["data/raw/*.csv"]\n'
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/yaml", content=new_yaml.encode(), headers={"Content-Type": "text/plain"})
    assert r.status_code == 200
    assert r.json()["saved"] is True
    assert (project_dir / "pipeline.yaml").read_text() == new_yaml


async def test_validate_yaml_valid(app):
    valid = 'nodes:\n  - id: t\n    plugin: connectors.file\n    config:\n      paths: ["data/raw/*.csv"]\n'
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/yaml/validate", content=valid.encode(), headers={"Content-Type": "text/plain"})
    assert r.status_code == 200
    assert r.json()["valid"] is True


async def test_validate_yaml_missing_nodes_key(app):
    bad = 'foo: bar\n'
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/yaml/validate", content=bad.encode(), headers={"Content-Type": "text/plain"})
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is False
    assert "error" in body


async def test_validate_yaml_parse_error(app):
    broken = ': [unclosed\n'
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/yaml/validate", content=broken.encode(), headers={"Content-Type": "text/plain"})
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is False


# ── /api/run ────────────────────────────────────────────────────

async def test_api_run_missing_yaml_returns_400(app, project_dir):
    (project_dir / "pipeline.yaml").unlink()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/run")
    assert r.status_code == 400
    assert "error" in r.json()


async def test_api_run_starts_background_task(app, project_dir, monkeypatch):
    spawned = []

    def fake_create_task(coro, **kwargs):
        spawned.append(True)
        coro.close()
        return asyncio.get_event_loop().create_task(asyncio.sleep(0))

    monkeypatch.setattr("model_builder.web.app.asyncio.create_task", fake_create_task)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/run")
    assert r.status_code == 200
    assert r.json()["started"] is True
    assert len(spawned) == 1


async def test_api_run_from_node_also_starts(app, project_dir, monkeypatch):
    spawned = []

    def fake_create_task(coro, **kwargs):
        spawned.append(True)
        coro.close()
        return asyncio.get_event_loop().create_task(asyncio.sleep(0))

    monkeypatch.setattr("model_builder.web.app.asyncio.create_task", fake_create_task)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/run?from_node=validate")
    assert r.status_code == 200
    assert r.json()["started"] is True


# ── /api/predict ────────────────────────────────────────────────

async def test_predict_no_run_returns_404(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/predict", json={"features": {"f1": 1.0}})
    assert r.status_code == 404


async def test_predict_returns_class_and_confidence(app, project_dir):
    import joblib
    from sklearn.dummy import DummyClassifier
    from model_builder.core.store import ProjectStore

    run_dir = project_dir / "runs" / "run_001"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)

    model = DummyClassifier(strategy="most_frequent")
    model.fit([[1.0, 2.0]], ["setosa"])
    model_path = artifacts / "model.pkl"
    joblib.dump(model, model_path)

    (artifacts / "export_meta.json").write_text(json.dumps({
        "format": "pickle", "path": str(model_path), "plugin": "random_forest",
    }))
    (artifacts / "profile.json").write_text(json.dumps({
        "row_count": 1, "column_count": 3,
        "columns": {"f1": "float64", "f2": "float64", "species": "object"},
        "nulls": {}, "data_types": ["float64", "object"],
        "target_col": "species",
    }))

    store = ProjectStore(project_dir)
    await store.init()
    await store.create_run("run_001")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/predict", json={"features": {"f1": 1.0, "f2": 2.0}})
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert "confidence" in data
    assert data["prediction"] == "setosa"


# ── /api/explain ────────────────────────────────────────────────

async def test_explain_no_run_returns_404(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/explain")
    assert r.status_code == 404


async def test_explain_returns_metrics_and_insights(app, project_dir):
    from model_builder.core.store import ProjectStore

    run_dir = project_dir / "runs" / "run_001"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)

    (run_dir / "eval_report.json").write_text(json.dumps({
        "metrics": {"accuracy": 0.94, "f1": 0.93},
        "feature_importance": {"petal_length": 0.85, "sepal_width": 0.15},
        "plugin_name": "random_forest",
    }))
    (artifacts / "profile.json").write_text(json.dumps({
        "row_count": 150, "column_count": 5,
        "columns": {"petal_length": "float64", "sepal_width": "float64"},
        "nulls": {"petal_length": 0, "sepal_width": 0},
        "data_types": ["float64"],
    }))

    store = ProjectStore(project_dir)
    await store.init()
    await store.create_run("run_001")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/explain")
    assert r.status_code == 200
    data = r.json()
    assert data["metrics"]["accuracy"] == pytest.approx(0.94)
    assert "petal_length" in data["feature_importance"]
    assert isinstance(data["insights"], list)
    # petal_length dominance > 0.8 should trigger insight
    assert any("petal_length" in i for i in data["insights"])
