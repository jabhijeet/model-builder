"""FastAPI web UI for aimodelground. Launched via `aimodelground UI`."""
import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

import yaml as _yaml
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..core.store import ProjectStore
from ..core.dag import DAG
from ..core.gates import GateManager
from ..core.models import NodeState

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"

_project_dir: Path = Path(".")
_sse_queues: list[asyncio.Queue] = []
_model_cache: dict = {}


def _tr(templates: Jinja2Templates, request: Request, name: str, ctx: dict):
    """Starlette 1.0 TemplateResponse — injects mb_version into every context."""
    return templates.TemplateResponse(request, name, {"mb_version": __version__, **ctx})


def create_app(project_dir: Path) -> FastAPI:
    global _project_dir
    _project_dir = project_dir

    app = FastAPI(title="aimodelground UI")
    tmpl = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    async def _store() -> ProjectStore:
        store = ProjectStore(_project_dir)
        await store.init()
        return store

    def _dag() -> DAG | None:
        pf = _project_dir / "pipeline.yaml"
        return DAG.from_file(pf) if pf.exists() else None

    async def _node_context(store: ProjectStore, dag: DAG | None, run_id: int | None) -> list:
        node_runs = {}
        if run_id:
            node_runs = {nr.node_id: nr for nr in await store.get_all_node_runs(run_id)}
        if not dag:
            return []
        nodes = []
        for node_id, node_def in dag.nodes.items():
            nr = node_runs.get(node_id)
            nodes.append({
                "id": node_id,
                "state": nr.state.value if nr else "pending",
                "plugin": node_def.plugin,
                "message": node_def.message,
                "error": nr.error if nr else None,
                "output_path": nr.output_path if nr else None,
            })
        return nodes

    def _broadcast(event: dict) -> None:
        for q in list(_sse_queues):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def _detect_available_algos() -> list[dict]:
        algos = []
        try:
            import model_builder_classical  # noqa: F401
            algos += [
                {"id": "ml.random_forest.classifier", "label": "RandomForest"},
                {"id": "ml.xgboost.classifier", "label": "XGBoost"},
                {"id": "ml.lightgbm.classifier", "label": "LightGBM"},
            ]
        except ImportError:
            pass
        try:
            import model_builder_dl  # noqa: F401
            algos += [
                {"id": "ml.cnn.classifier", "label": "CNN"},
                {"id": "ml.lstm.classifier", "label": "LSTM"},
            ]
        except ImportError:
            pass
        return algos

    async def _step_context() -> dict:
        raw_dir = _project_dir / "data" / "raw"
        has_files = raw_dir.exists() and any(True for _ in raw_dir.iterdir())
        has_yaml = (_project_dir / "pipeline.yaml").exists()
        store = await _store()
        run = await store.get_latest_run()
        has_run = run is not None
        has_deploy = False
        has_model = False
        if run:
            has_deploy = (_project_dir / "runs" / run.name / "DEPLOY.md").exists()
            has_model = (_project_dir / "runs" / run.name / "artifacts" / "export_meta.json").exists()
        if has_model:
            max_step = 6
        elif has_deploy:
            max_step = 5
        elif has_run:
            max_step = 4
        elif has_yaml:
            max_step = 3
        elif has_files:
            max_step = 2
        else:
            max_step = 1
        return {"project_name": _project_dir.name, "max_step": max_step}

    # --- SSE ---

    @app.get("/events")
    async def sse_endpoint():
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        _sse_queues.append(q)

        async def gen() -> AsyncIterator[str]:
            try:
                yield 'data: {"type":"connected"}\n\n'
                while True:
                    try:
                        event = await asyncio.wait_for(q.get(), timeout=30)
                        yield f"data: {json.dumps(event)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"
            finally:
                if q in _sse_queues:
                    _sse_queues.remove(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ── /api/file-info ────────────────────────────────────────────

    @app.get("/api/file-info/{filename}")
    async def api_file_info(filename: str):
        path = _project_dir / "data" / "raw" / filename
        if not path.exists():
            return JSONResponse({"error": "File not found"}, status_code=404)
        try:
            import pandas as pd
            if filename.endswith(".csv"):
                df = pd.read_csv(path, nrows=1000)
            elif filename.endswith(".parquet"):
                df = pd.read_parquet(path).head(1000)
            else:
                return JSONResponse({"error": "Unsupported format"}, status_code=422)
            columns = [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns]
            target_kw = {"target", "label", "y", "class", "output", "species", "outcome", "result"}
            detected_target = next(
                (c["name"] for c in columns if c["name"].lower() in target_kw), None
            )
            if detected_target is None:
                last = columns[-1]
                if df[last["name"]].dtype == object or df[last["name"]].nunique() < 20:
                    detected_target = last["name"]
            return JSONResponse({"rows": len(df), "columns": columns, "detected_target": detected_target})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)

    # ── /api/yaml ─────────────────────────────────────────────────

    @app.get("/api/yaml")
    async def api_get_yaml():
        pf = _project_dir / "pipeline.yaml"
        if not pf.exists():
            return JSONResponse({"error": "pipeline.yaml not found"}, status_code=404)
        return Response(content=pf.read_text(encoding="utf-8"), media_type="text/plain")

    @app.post("/api/yaml")
    async def api_post_yaml(request: Request):
        content = (await request.body()).decode()
        try:
            _yaml.safe_load(content)
        except _yaml.YAMLError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        (_project_dir / "pipeline.yaml").write_text(content, encoding="utf-8")
        return JSONResponse({"saved": True})

    @app.post("/api/yaml/validate")
    async def api_validate_yaml(request: Request):
        content = (await request.body()).decode()
        try:
            parsed = _yaml.safe_load(content)
            if not isinstance(parsed, dict) or "nodes" not in parsed:
                return JSONResponse({"valid": False, "error": "Missing required 'nodes' key"})
            return JSONResponse({"valid": True})
        except _yaml.YAMLError as exc:
            return JSONResponse({"valid": False, "error": str(exc)})

    # ── /api/run ──────────────────────────────────────────────────

    @app.post("/api/run")
    async def api_run(from_node: str = None):
        if not (_project_dir / "pipeline.yaml").exists():
            return JSONResponse({"error": "pipeline.yaml not found"}, status_code=400)

        async def _run_bg() -> None:
            from model_builder.core.store import ProjectStore as _Store
            from model_builder.core.dag import DAG as _DAG
            from model_builder.core.events import EventBus as _EventBus
            from model_builder.core.scheduler import Scheduler as _Scheduler
            from model_builder.plugins.registry import PluginRegistry as _Registry

            bg_store = _Store(_project_dir)
            await bg_store.init()
            dag = _DAG.from_file(_project_dir / "pipeline.yaml")
            registry = _Registry()
            registry.discover()
            events = _EventBus()

            run_name = await bg_store.next_run_name()
            parent_run = await bg_store.get_latest_run()
            parent_run_id = parent_run.id if parent_run and from_node else None
            run_id = await bg_store.create_run(
                run_name, from_node_id=from_node, parent_run_id=parent_run_id
            )
            run_dir = _project_dir / "runs" / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "artifacts").mkdir(exist_ok=True)
            (run_dir / "logs").mkdir(exist_ok=True)

            if from_node and parent_run_id:
                upstream = dag.upstream_of(from_node)
                from model_builder.core.models import NodeRun as _NR, NodeState as _NS
                for nr in await bg_store.get_all_node_runs(parent_run_id):
                    if nr.node_id in upstream and nr.state.value in ("succeeded", "approved", "skipped"):
                        await bg_store.upsert_node_run(
                            _NR(run_id=run_id, node_id=nr.node_id, state=_NS(nr.state.value),
                                started_at=nr.started_at, finished_at=nr.finished_at, output_path=nr.output_path)
                        )

            def on_event(rid, node_id, event_type, payload):
                _broadcast({"type": event_type, "node_id": node_id, **payload})

            events.subscribe(on_event)
            scheduler = _Scheduler(dag, bg_store, registry, events, run_id, run_dir)
            await scheduler.run()

        asyncio.create_task(_run_bg())
        return JSONResponse({"started": True})

    # ── /api/predict ──────────────────────────────────────────────

    @app.post("/api/predict")
    async def api_predict(request: Request):
        import joblib
        store = await _store()
        run = await store.get_latest_run()
        if not run:
            return JSONResponse({"error": "No run found"}, status_code=404)

        artifacts = _project_dir / "runs" / run.name / "artifacts"
        meta_path = artifacts / "export_meta.json"
        if not meta_path.exists():
            return JSONResponse({"error": "No exported model found"}, status_code=404)

        export_meta = json.loads(meta_path.read_text())
        model_path = Path(export_meta["path"])
        if not model_path.exists():
            return JSONResponse({"error": f"Model file not found: {model_path}"}, status_code=404)

        profile = {}
        profile_path = artifacts / "profile.json"
        if profile_path.exists():
            profile = json.loads(profile_path.read_text())
        target_col = profile.get("target_col")
        feature_order = [c for c in profile.get("columns", {}) if c != target_col] if profile else None

        body = await request.json()
        features: dict = body.get("features", {})

        cache_key = str(model_path)
        if cache_key not in _model_cache:
            _model_cache[cache_key] = joblib.load(model_path)
        model = _model_cache[cache_key]

        if feature_order:
            X = [[float(features.get(f, 0)) for f in feature_order]]
        else:
            X = [[float(v) for v in features.values()]]

        prediction = model.predict(X)[0]
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            confidence = float(max(proba))

        top_feature = top_value = None
        eval_path = _project_dir / "runs" / run.name / "eval_report.json"
        if eval_path.exists() and feature_order:
            fi = json.loads(eval_path.read_text()).get("feature_importance", {})
            if fi:
                top_feature = max(fi, key=fi.get)
                top_value = features.get(top_feature)

        return JSONResponse({
            "prediction": str(prediction),
            "confidence": confidence,
            "top_feature": top_feature,
            "top_feature_value": top_value,
        })

    # ── /api/explain ──────────────────────────────────────────────

    @app.get("/api/explain")
    async def api_explain():
        store = await _store()
        run = await store.get_latest_run()
        if not run:
            return JSONResponse({"error": "No run found"}, status_code=404)

        run_dir = _project_dir / "runs" / run.name
        eval_path = run_dir / "eval_report.json"
        if not eval_path.exists():
            return JSONResponse({"error": "No eval report found"}, status_code=404)

        eval_report = json.loads(eval_path.read_text())
        metrics = eval_report.get("metrics", {})
        feature_importance = eval_report.get("feature_importance", {})

        profile: dict = {}
        pp = run_dir / "artifacts" / "profile.json"
        if pp.exists():
            profile = json.loads(pp.read_text())

        insights: list[str] = []
        acc = metrics.get("accuracy")
        if acc is not None and acc < 0.7:
            insights.append("Accuracy below 70% — consider adding more data or tuning hyperparameters.")
        if feature_importance:
            top_score = max(feature_importance.values())
            if top_score > 0.8:
                top_feat = max(feature_importance, key=feature_importance.get)
                insights.append(
                    f"'{top_feat}' dominates predictions (score {top_score:.2f}) — model may overfit."
                )
        nulls = profile.get("nulls", {})
        row_count = profile.get("row_count", 0)
        if nulls and row_count:
            high_null = [c for c, n in nulls.items() if n / row_count > 0.1]
            if high_null:
                insights.append(f"High null rate in: {', '.join(high_null)}. Clean source data.")

        return JSONResponse({
            "run_name": run.name,
            "metrics": metrics,
            "feature_importance": feature_importance,
            "profile": profile,
            "insights": insights,
        })

    # --- Pipeline ---

    @app.get("/", response_class=HTMLResponse)
    async def pipeline_page(request: Request):
        store = await _store()
        dag = _dag()
        run = await store.get_latest_run()
        nodes = await _node_context(store, dag, run.id if run else None)
        done = sum(1 for n in nodes if n["state"] in ("succeeded", "approved", "skipped"))
        awaiting = sum(1 for n in nodes if n["state"] == "awaiting_human")
        sc = await _step_context()
        return _tr(tmpl, request, "run.html", {
            "page": "run", "page_title": "Run",
            "current_step": 3,
            "run_name": run.name if run else None,
            "nodes": nodes, "done": done, "total": len(nodes),
            "awaiting_gates": awaiting,
            **sc,
        })

    @app.get("/pipeline-nodes", response_class=HTMLResponse)
    async def pipeline_nodes_partial(request: Request):
        store = await _store()
        dag = _dag()
        run = await store.get_latest_run()
        nodes = await _node_context(store, dag, run.id if run else None)
        return _tr(tmpl, request, "pipeline_nodes.html", {"nodes": nodes})

    # --- API ---

    @app.post("/api/approve/{node_id}", response_class=HTMLResponse)
    async def api_approve(request: Request, node_id: str):
        store = await _store()
        dag = _dag()
        run = await store.get_latest_run()
        if run and dag and node_id in dag.nodes:
            try:
                await GateManager(store).approve(run.id, node_id, dag.nodes[node_id])
                _broadcast({"type": "state_change", "node_id": node_id, "state": "approved"})
            except ValueError:
                pass
        nodes = await _node_context(store, dag, run.id if run else None)
        return _tr(tmpl, request, "pipeline_nodes.html", {"nodes": nodes})

    @app.post("/api/skip/{node_id}", response_class=HTMLResponse)
    async def api_skip(request: Request, node_id: str):
        store = await _store()
        dag = _dag()
        run = await store.get_latest_run()
        if run:
            await GateManager(store).skip(run.id, node_id)
            _broadcast({"type": "state_change", "node_id": node_id, "state": "skipped"})
        nodes = await _node_context(store, dag, run.id if run else None)
        return _tr(tmpl, request, "pipeline_nodes.html", {"nodes": nodes})

    @app.post("/api/retry/{node_id}", response_class=HTMLResponse)
    async def api_retry(request: Request, node_id: str):
        store = await _store()
        dag = _dag()
        run = await store.get_latest_run()
        if run:
            nr = await store.get_node_run(run.id, node_id)
            if nr and nr.state == NodeState.FAILED:
                nr.state = NodeState.PENDING
                nr.error = nr.started_at = nr.finished_at = None
                await store.upsert_node_run(nr)
                _broadcast({"type": "state_change", "node_id": node_id, "state": "pending"})
        nodes = await _node_context(store, dag, run.id if run else None)
        return _tr(tmpl, request, "pipeline_nodes.html", {"nodes": nodes})

    # --- Data ---

    @app.get("/upload", response_class=HTMLResponse)
    async def upload_page(request: Request):
        raw_dir = _project_dir / "data" / "raw"
        files = []
        if raw_dir.exists():
            for f in sorted(raw_dir.iterdir()):
                if f.is_file():
                    files.append({"name": f.name, "size_kb": round(f.stat().st_size / 1024, 1)})
        sc = await _step_context()
        return _tr(tmpl, request, "upload.html", {
            "page": "upload", "page_title": "Upload",
            "current_step": 1, "files": files, **sc,
        })

    @app.get("/configure", response_class=HTMLResponse)
    async def configure_page(request: Request):
        raw_dir = _project_dir / "data" / "raw"
        files = [f.name for f in sorted(raw_dir.iterdir()) if f.is_file()] if raw_dir.exists() else []
        yaml_content = ""
        pf = _project_dir / "pipeline.yaml"
        if pf.exists():
            yaml_content = pf.read_text(encoding="utf-8")
        sc = await _step_context()
        return _tr(tmpl, request, "configure.html", {
            "page": "configure", "page_title": "Configure",
            "current_step": 2,
            "files": files,
            "yaml_content": yaml_content,
            "available_algos": _detect_available_algos(),
            **sc,
        })

    @app.get("/data", response_class=HTMLResponse)
    async def data_page(request: Request):
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/upload", status_code=302)

    @app.get("/query", response_class=HTMLResponse)
    async def query_page(request: Request):
        store = await _store()
        run = await store.get_latest_run()
        features: list[dict] = []
        if run:
            pp = _project_dir / "runs" / run.name / "artifacts" / "profile.json"
            if pp.exists():
                profile = json.loads(pp.read_text())
                target_col = profile.get("target_col")
                for col, dtype in profile.get("columns", {}).items():
                    if col != target_col:
                        features.append({
                            "name": col,
                            "dtype": dtype,
                            "is_numeric": "int" in dtype or "float" in dtype,
                        })
        sc = await _step_context()
        return _tr(tmpl, request, "query.html", {
            "page": "query", "page_title": "Query",
            "current_step": 6,
            "features": features,
            "has_model": (
                run is not None and
                (_project_dir / "runs" / run.name / "artifacts" / "export_meta.json").exists()
            ),
            **sc,
        })

    @app.post("/api/upload")
    async def api_upload(file: UploadFile = File(...)):
        raw_dir = _project_dir / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        content = await file.read()
        (raw_dir / file.filename).write_bytes(content)
        return HTMLResponse(
            f'<div class="alert alert-success">Uploaded {file.filename} ({len(content)//1024} KB)</div>'
        )

    # --- Results ---

    @app.get("/results", response_class=HTMLResponse)
    async def results_page(request: Request, run: str = None, compare: str = None):
        store = await _store()
        all_runs = await store.list_runs()
        selected_run = run or (all_runs[-1].name if all_runs else None)

        eval_report = compare_report = None
        all_metrics: list[str] = []

        if selected_run:
            ep = _project_dir / "runs" / selected_run / "eval_report.json"
            if ep.exists():
                eval_report = json.loads(ep.read_text())
        if compare:
            cp = _project_dir / "runs" / compare / "eval_report.json"
            if cp.exists():
                compare_report = json.loads(cp.read_text())
        if eval_report or compare_report:
            ka = set((eval_report or {}).get("metrics", {}).keys())
            kb = set((compare_report or {}).get("metrics", {}).keys())
            all_metrics = sorted(ka | kb)

        sc = await _step_context()
        return _tr(tmpl, request, "results.html", {
            "page": "results", "page_title": "Results",
            "current_step": 4,
            "runs": all_runs, "selected_run": selected_run,
            "eval_report": eval_report, "compare_report": compare_report,
            "compare_run": compare, "all_metrics": all_metrics,
            **sc,
        })

    # --- Deploy ---

    @app.get("/deploy", response_class=HTMLResponse)
    async def deploy_page(request: Request):
        store = await _store()
        run = await store.get_latest_run()
        deploy_md = export_meta = ranking = None
        run_name = run.name if run else None

        if run:
            dp = _project_dir / "runs" / run.name / "DEPLOY.md"
            if dp.exists():
                deploy_md = dp.read_text()
            ep = _project_dir / "runs" / run.name / "artifacts" / "export_meta.json"
            if ep.exists():
                export_meta = json.loads(ep.read_text())
            rp = _project_dir / "runs" / run.name / "artifacts" / "ranking.json"
            if rp.exists():
                ranking = json.loads(rp.read_text()).get("rankings", [])

        sc = await _step_context()
        return _tr(tmpl, request, "deploy.html", {
            "page": "deploy", "page_title": "Deploy",
            "current_step": 5,
            "deploy_md": deploy_md, "export_meta": export_meta,
            "ranking": ranking, "run_name": run_name,
            "export_format": export_meta.get("format") if export_meta else None,
            **sc,
        })

    return app

