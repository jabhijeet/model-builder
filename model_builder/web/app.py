"""FastAPI web UI for aimodelground. Launched via `aimodelground UI`."""
import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
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

    # --- Pipeline ---

    @app.get("/", response_class=HTMLResponse)
    async def pipeline_page(request: Request):
        store = await _store()
        dag = _dag()
        run = await store.get_latest_run()
        nodes = await _node_context(store, dag, run.id if run else None)
        done = sum(1 for n in nodes if n["state"] in ("succeeded", "approved", "skipped"))
        awaiting = sum(1 for n in nodes if n["state"] == "awaiting_human")
        return _tr(tmpl, request, "pipeline.html", {
            "page": "pipeline", "page_title": "Pipeline",
            "project_name": _project_dir.name,
            "run_name": run.name if run else None,
            "nodes": nodes, "done": done, "total": len(nodes),
            "awaiting_gates": awaiting,
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

    @app.get("/data", response_class=HTMLResponse)
    async def data_page(request: Request):
        raw_dir = _project_dir / "data" / "raw"
        files = []
        if raw_dir.exists():
            for f in sorted(raw_dir.iterdir()):
                if f.is_file():
                    files.append({"name": f.name, "size_kb": round(f.stat().st_size / 1024, 1)})
        profile = None
        store = await _store()
        run = await store.get_latest_run()
        if run:
            pp = _project_dir / "runs" / run.name / "artifacts" / "profile.json"
            if pp.exists():
                profile = json.loads(pp.read_text())
        return _tr(tmpl, request, "data.html", {
            "page": "data", "page_title": "Data", "files": files, "profile": profile,
        })

    @app.post("/api/upload")
    async def api_upload(file: UploadFile = File(...)):
        raw_dir = _project_dir / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        content = await file.read()
        (raw_dir / file.filename).write_bytes(content)
        return HTMLResponse(
            f'<div class="alert alert-success py-1 px-2 mb-0">Uploaded {file.filename} ({len(content)//1024} KB)</div>'
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

        return _tr(tmpl, request, "results.html", {
            "page": "results", "page_title": "Results",
            "runs": all_runs, "selected_run": selected_run,
            "eval_report": eval_report, "compare_report": compare_report,
            "compare_run": compare, "all_metrics": all_metrics,
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

        return _tr(tmpl, request, "deploy.html", {
            "page": "deploy", "page_title": "Deploy",
            "deploy_md": deploy_md, "export_meta": export_meta,
            "ranking": ranking, "run_name": run_name,
            "export_format": export_meta.get("format") if export_meta else None,
        })

    return app

