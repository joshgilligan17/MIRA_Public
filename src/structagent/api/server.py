"""FastAPI service for local MIRA batch triage jobs."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import secrets
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from structagent.jobs.models import JobConfig
from structagent.jobs.runner import PROVIDER_DEFAULTS, JobRunner
from structagent.jobs.store import JobStore
from structagent.profiles import list_analysis_profiles


JOB_ROOT = Path(os.environ.get("MIRA_JOB_ROOT", ".mira/jobs"))
MAX_UPLOAD_BYTES = int(float(os.environ.get("MIRA_MAX_UPLOAD_MB", "250")) * 1024 * 1024)
BASIC_AUTH_USERNAME = os.environ.get("MIRA_BASIC_AUTH_USERNAME")
BASIC_AUTH_PASSWORD = os.environ.get("MIRA_BASIC_AUTH_PASSWORD")
STORE = JobStore(JOB_ROOT)
RUNNER = JobRunner(STORE)

app = FastAPI(title="MIRA Batch Lab", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get(
            "MIRA_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173",
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def optional_basic_auth(request: Request, call_next):
    if not BASIC_AUTH_USERNAME or not BASIC_AUTH_PASSWORD or request.url.path == "/api/health":
        return await call_next(request)

    authorization = request.headers.get("Authorization", "")
    if not _valid_basic_auth(authorization):
        return PlainTextResponse(
            "Authentication required",
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": 'Basic realm="MIRA"'},
        )
    return await call_next(request)


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"status": "ok", "synthesis": _synthesis_status()}


@app.get("/api/profiles")
def profiles() -> dict[str, object]:
    return {"profiles": list_analysis_profiles()}


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    files: Annotated[list[UploadFile] | None, File(description="PDB, CIF, mmCIF, or zip files")] = None,
    query: Annotated[str, Form()] = "Rank these structures for filtering.",
    profile: Annotated[str, Form()] = "triage_default",
    rank_by: Annotated[str, Form()] = "stability",
    glob_pattern: Annotated[str, Form()] = "*",
    chain_a: Annotated[str | None, Form()] = None,
    chain_b: Annotated[str | None, Form()] = None,
    max_workers: Annotated[int, Form()] = 2,
    enable_llm_synthesis: Annotated[bool, Form()] = True,
    llm_provider: Annotated[str | None, Form()] = None,
    llm_model: Annotated[str | None, Form()] = None,
    llm_base_url: Annotated[str | None, Form()] = None,
    llm_api_key: Annotated[str | None, Form()] = None,
    llm_temperature: Annotated[float, Form()] = 0.2,
) -> dict[str, object]:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one .pdb, .cif, .mmcif, or .zip file.")

    config = JobConfig(
        query=query,
        profile=profile,
        rank_by=rank_by,
        glob_pattern=glob_pattern,
        chain_a=chain_a or None,
        chain_b=chain_b or None,
        max_workers=max(1, max_workers),
        enable_llm_synthesis=enable_llm_synthesis,
        llm_provider=llm_provider or None,
        llm_model=llm_model or None,
        llm_base_url=llm_base_url or None,
        llm_temperature=llm_temperature,
    )
    record = STORE.create_job(config)
    input_files = await _save_uploads(record.id, files)
    if not input_files:
        STORE.set_status(record.id, "failed", "No supported structure files were uploaded.", error="No files")
        raise HTTPException(status_code=400, detail="No supported structure files were uploaded.")

    record.input_files = input_files
    STORE.write_record(record)
    STORE.append_event(record.id, "uploads_saved", f"Saved {len(input_files)} input file(s).")
    background_tasks.add_task(RUNNER.run_job, record.id, llm_api_key or None)
    return {"job_id": record.id, "job": record.to_dict()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    record = _get_record_or_404(job_id)
    progress = _progress(record.completed_count, record.total_count)
    return {"job": record.to_dict(), "progress": progress}


@app.get("/api/jobs/{job_id}/events")
async def stream_events(job_id: str) -> StreamingResponse:
    _get_record_or_404(job_id)

    async def event_stream():
        sent = 0
        while True:
            events = STORE.list_events(job_id)
            for event in events[sent:]:
                yield f"data: {json.dumps(event)}\n\n"
            sent = len(events)
            status = STORE.get_record(job_id).status
            if status in {"completed", "failed"} and sent >= len(events):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/jobs/{job_id}/results")
def get_results(job_id: str) -> JSONResponse:
    _get_record_or_404(job_id)
    results = STORE.load_results(job_id)
    if not results:
        raise HTTPException(status_code=404, detail="Results are not ready yet.")
    return JSONResponse(results)


@app.get("/api/jobs/{job_id}/structures/{structure_id}")
def get_structure(job_id: str, structure_id: str) -> FileResponse:
    _get_record_or_404(job_id)
    results = STORE.load_results(job_id)
    for item in results.get("structures", []):
        if item.get("id") == structure_id:
            path = Path(item.get("source_path") or "")
            if _is_job_path(job_id, path) and path.exists():
                return FileResponse(path, media_type="chemical/x-pdb", filename=path.name)
    raise HTTPException(status_code=404, detail="Structure not found.")


@app.get("/api/jobs/{job_id}/report.md")
def get_report(job_id: str) -> PlainTextResponse:
    _get_record_or_404(job_id)
    path = STORE.report_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report is not ready yet.")
    return PlainTextResponse(path.read_text(), media_type="text/markdown")


def _get_record_or_404(job_id: str):
    try:
        return STORE.get_record(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


async def _save_uploads(job_id: str, uploads: list[UploadFile]) -> list[str]:
    input_dir = STORE.input_dir(job_id)
    saved: list[str] = []
    for upload in uploads:
        filename = Path(upload.filename or "structure").name
        content = await upload.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"{filename} is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
            )
        if filename.lower().endswith(".zip"):
            saved.extend(_extract_zip(input_dir, content))
        else:
            path = _unique_path(input_dir / filename)
            path.write_bytes(content)
            saved.append(path.name)
    return saved


def _extract_zip(input_dir: Path, content: bytes) -> list[str]:
    saved: list[str] = []
    extracted_bytes = 0
    with zipfile.ZipFile(BytesIO(content)) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            member_path = Path(member.filename)
            if member_path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
                continue
            extracted_bytes += member.file_size
            if extracted_bytes > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Extracted structures exceed the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
                )
            safe_parts = [part for part in member_path.parts if part not in {"", ".", ".."}]
            if not safe_parts:
                continue
            target = _unique_path(input_dir.joinpath(*safe_parts))
            if not _is_under(target, input_dir):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
            saved.append(str(target.relative_to(input_dir)))
    return saved


def _unique_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 1000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise ValueError(f"Could not create unique filename for {path.name}")


def _is_job_path(job_id: str, path: Path) -> bool:
    return _is_under(path, STORE.job_dir(job_id))


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _progress(completed: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(completed / total, 3)


def _valid_basic_auth(authorization: str) -> bool:
    if not authorization.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(authorization.removeprefix("Basic ").strip()).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return False
    username, separator, password = decoded.partition(":")
    if not separator:
        return False
    return secrets.compare_digest(username, BASIC_AUTH_USERNAME or "") and secrets.compare_digest(
        password, BASIC_AUTH_PASSWORD or ""
    )


def _synthesis_status() -> dict[str, object]:
    provider = (os.getenv("MIRA_REPORT_PROVIDER") or _infer_report_provider()).lower()
    defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["openai"])
    model = os.getenv("MIRA_REPORT_MODEL") or defaults.get("model")
    configured = bool(os.getenv("MIRA_REPORT_API_KEY") or _first_env_value(defaults.get("env_vars", ())))
    return {"configured": configured, "provider": provider, "model": model}


def _infer_report_provider() -> str:
    for provider, defaults in PROVIDER_DEFAULTS.items():
        if _first_env_value(defaults.get("env_vars", ())):
            return provider
    return "openai"


def _first_env_value(env_vars: tuple[str, ...] | list[str]) -> str | None:
    for env_var in env_vars:
        value = os.getenv(env_var)
        if value:
            return value
    return None


dist_dir = Path(os.environ.get("MIRA_WEB_DIST", "")) if os.environ.get("MIRA_WEB_DIST") else None
for candidate in [
    dist_dir,
    Path(__file__).resolve().parents[3] / "webapp" / "dist",
    Path.cwd() / "webapp" / "dist",
    Path("/app/webapp/dist"),
]:
    if candidate and candidate.exists():
        dist_dir = candidate
        break

if dist_dir and dist_dir.exists():
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="mira-web")
