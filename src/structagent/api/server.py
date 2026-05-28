"""FastAPI service for local MIRA batch triage jobs."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import secrets
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from structagent.jobs.models import JobConfig
from structagent.jobs.runner import PROVIDER_DEFAULTS, JobRunner
from structagent.jobs.store import JobStore
from structagent.projects import ProjectRecord, ProjectStore
from structagent.providers import create_provider
from structagent.profiles import list_analysis_profiles


JOB_ROOT = Path(os.environ.get("MIRA_JOB_ROOT", ".mira/jobs"))
PROJECT_ROOT = Path(os.environ.get("MIRA_PROJECT_ROOT", str(JOB_ROOT.parent / "projects")))
MAX_UPLOAD_BYTES = int(float(os.environ.get("MIRA_MAX_UPLOAD_MB", "250")) * 1024 * 1024)
BASIC_AUTH_USERNAME = os.environ.get("MIRA_BASIC_AUTH_USERNAME")
BASIC_AUTH_PASSWORD = os.environ.get("MIRA_BASIC_AUTH_PASSWORD")
STORE = JobStore(JOB_ROOT)
PROJECTS = ProjectStore(PROJECT_ROOT)
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


class ProjectCreateRequest(BaseModel):
    name: str = "Untitled project"
    description: str = ""


class ProjectUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    selected_job_id: str | None = None
    selected_structure_id: str | None = None


class ProjectChatRequest(BaseModel):
    message: str
    selected_job_id: str | None = None
    selected_structure_id: str | None = None


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


@app.get("/api/projects")
def list_projects() -> dict[str, object]:
    return {"projects": [_project_response(project) for project in PROJECTS.list_projects()]}


@app.post("/api/projects")
def create_project(payload: ProjectCreateRequest) -> dict[str, object]:
    project = PROJECTS.create_project(payload.name, payload.description)
    return {"project": _project_response(project)}


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict[str, object]:
    return {"project": _project_response(_get_project_or_404(project_id))}


@app.patch("/api/projects/{project_id}")
def update_project(project_id: str, payload: ProjectUpdateRequest) -> dict[str, object]:
    _get_project_or_404(project_id)
    project = PROJECTS.update_project(
        project_id,
        name=payload.name,
        description=payload.description,
        selected_job_id=payload.selected_job_id,
        selected_structure_id=payload.selected_structure_id,
    )
    return {"project": _project_response(project)}


@app.post("/api/projects/{project_id}/target")
async def upload_project_target(
    project_id: str,
    file: Annotated[UploadFile, File(description="Target PDB, CIF, or mmCIF structure")],
) -> dict[str, object]:
    _get_project_or_404(project_id)
    filename = Path(file.filename or "target.pdb").name
    if Path(filename).suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
        raise HTTPException(status_code=400, detail="Upload a .pdb, .cif, or .mmcif target structure.")
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"{filename} is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )
    project = PROJECTS.save_target(project_id, filename, content)
    return {"project": _project_response(project)}


@app.get("/api/projects/{project_id}/target")
def get_project_target(project_id: str) -> FileResponse:
    project = _get_project_or_404(project_id)
    path = PROJECTS.target_path(project)
    if not path or not path.exists() or not _is_under(path, PROJECTS.target_dir(project_id)):
        raise HTTPException(status_code=404, detail="Target structure not found.")
    return FileResponse(path, media_type="chemical/x-pdb", filename=project.target_original_name or path.name)


@app.get("/api/projects/{project_id}/jobs")
def list_project_jobs(project_id: str) -> dict[str, object]:
    project = _get_project_or_404(project_id)
    jobs = []
    for job_id in project.job_ids:
        try:
            jobs.append(STORE.get_record(job_id).to_dict())
        except FileNotFoundError:
            continue
    jobs.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return {"jobs": jobs}


@app.post("/api/projects/{project_id}/jobs")
async def create_project_job(
    project_id: str,
    background_tasks: BackgroundTasks,
    files: Annotated[list[UploadFile] | None, File(description="PDB, CIF, mmCIF, or zip files")] = None,
    query: Annotated[str, Form()] = "Rank candidate binders for this project target.",
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
    _get_project_or_404(project_id)
    return await _queue_upload_job(
        background_tasks,
        files,
        query,
        profile,
        rank_by,
        glob_pattern,
        chain_a,
        chain_b,
        max_workers,
        enable_llm_synthesis,
        llm_provider,
        llm_model,
        llm_base_url,
        llm_api_key,
        llm_temperature,
        project_id=project_id,
    )


@app.get("/api/projects/{project_id}/chat")
def get_project_chat(project_id: str) -> dict[str, object]:
    project = _get_project_or_404(project_id)
    return {"messages": [message.to_dict() for message in project.chat_messages]}


@app.post("/api/projects/{project_id}/chat")
def send_project_chat(project_id: str, payload: ProjectChatRequest) -> dict[str, object]:
    project = _get_project_or_404(project_id)
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    PROJECTS.append_chat_message(project_id, "user", message, payload.selected_job_id, payload.selected_structure_id)
    project = PROJECTS.get_project(project_id)
    assistant_content = _generate_project_chat_response(
        project, message, payload.selected_job_id, payload.selected_structure_id
    )
    assistant_message = PROJECTS.append_chat_message(
        project_id,
        "assistant",
        assistant_content,
        payload.selected_job_id,
        payload.selected_structure_id,
    )
    return {
        "message": assistant_message.to_dict(),
        "messages": [message.to_dict() for message in PROJECTS.get_project(project_id).chat_messages],
    }


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
    return await _queue_upload_job(
        background_tasks,
        files,
        query,
        profile,
        rank_by,
        glob_pattern,
        chain_a,
        chain_b,
        max_workers,
        enable_llm_synthesis,
        llm_provider,
        llm_model,
        llm_base_url,
        llm_api_key,
        llm_temperature,
    )


async def _queue_upload_job(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] | None,
    query: str,
    profile: str,
    rank_by: str,
    glob_pattern: str,
    chain_a: str | None,
    chain_b: str | None,
    max_workers: int,
    enable_llm_synthesis: bool,
    llm_provider: str | None,
    llm_model: str | None,
    llm_base_url: str | None,
    llm_api_key: str | None,
    llm_temperature: float,
    project_id: str | None = None,
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
    record = STORE.create_job(config, project_id=project_id)
    input_files = await _save_uploads(record.id, files)
    if not input_files:
        STORE.set_status(record.id, "failed", "No supported structure files were uploaded.", error="No files")
        raise HTTPException(status_code=400, detail="No supported structure files were uploaded.")

    record.input_files = input_files
    STORE.write_record(record)
    if project_id:
        PROJECTS.add_job(project_id, record.id)
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


def _project_response(project: ProjectRecord) -> dict[str, object]:
    data = project.to_dict()
    data["target_structure"] = _target_structure_response(project)
    data["job_count"] = len(project.job_ids)
    return data


def _target_structure_response(project: ProjectRecord) -> dict[str, object] | None:
    path = PROJECTS.target_path(project)
    if not path or not path.exists():
        return None
    filename = project.target_original_name or path.name
    pdb_id = Path(filename).stem.upper() or "TARGET"
    return {
        "id": "target",
        "pdb_id": pdb_id,
        "filename": filename,
        "success": True,
        "error": None,
        "profile": "project_target",
        "chains": [],
        "metrics": {},
        "features": {},
        "warnings": [],
        "summary": "Project target structure.",
        "structure_url": f"/api/projects/{project.id}/target",
    }


def _get_project_or_404(project_id: str) -> ProjectRecord:
    try:
        return PROJECTS.get_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _generate_project_chat_response(
    project: ProjectRecord,
    user_message: str,
    selected_job_id: str | None,
    selected_structure_id: str | None,
) -> str:
    context = _project_chat_context(project, selected_job_id, selected_structure_id)
    fallback = _deterministic_chat_response(context)
    provider_name, model, base_url, api_key = _resolve_chat_llm_config()
    if not api_key or not model:
        return fallback

    try:
        provider = create_provider(
            provider_name,
            api_key=api_key,
            base_url=base_url,
            timeout=120.0,
            temperature=0.2,
        )
        response = provider.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are MIRA, a molecular structure reasoning assistant inside a project workspace. "
                        "Answer from the supplied project context only. Be concise, scientific, and practical. "
                        "If you mention a residue or region, copy one of the supplied markdown region links exactly "
                        "so the UI can highlight it. Do not invent residues, scores, targets, affinities, wet-lab "
                        "claims, or biological mechanisms. Do not include hidden reasoning or <think> blocks."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User message: {user_message}\n\n"
                        "Project context JSON:\n"
                        f"{json.dumps(context, indent=2, sort_keys=True)}"
                    ),
                },
            ],
            model=model,
            temperature=0.2,
        )
        content = _clean_chat_response(response.content)
        return content or fallback
    except Exception as exc:
        return f"{fallback}\n\n_Chat synthesis fell back to local context because the provider returned {type(exc).__name__}._"


def _project_chat_context(
    project: ProjectRecord,
    selected_job_id: str | None,
    selected_structure_id: str | None,
) -> dict[str, object]:
    target = _target_structure_response(project)
    job_id = selected_job_id or project.selected_job_id or (project.job_ids[-1] if project.job_ids else None)
    job = None
    results: dict[str, object] = {}
    report_excerpt = ""
    selected_structure = None
    if job_id:
        try:
            record = STORE.get_record(job_id)
            job = record.to_dict()
            results = STORE.load_results(job_id)
            selected_structure = _select_structure(results, selected_structure_id or project.selected_structure_id)
            report_path = STORE.report_path(job_id)
            if report_path.exists():
                report_excerpt = report_path.read_text()[:5000]
        except FileNotFoundError:
            job = None

    return {
        "project": {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "target": target,
            "job_count": len(project.job_ids),
        },
        "selected_job": job,
        "batch_summary": results.get("summary") if isinstance(results, dict) else None,
        "ranking": (results.get("ranking") or [])[:8] if isinstance(results, dict) else [],
        "selected_structure": _structure_chat_context(selected_structure),
        "report_excerpt": report_excerpt,
        "recent_chat": [message.to_dict() for message in project.chat_messages[-8:]],
    }


def _select_structure(results: dict[str, object], selected_structure_id: str | None) -> dict[str, object] | None:
    structures = results.get("structures") if isinstance(results, dict) else None
    if not isinstance(structures, list) or not structures:
        return None
    if selected_structure_id:
        for item in structures:
            if not isinstance(item, dict):
                continue
            if selected_structure_id in {str(item.get("id")), str(item.get("pdb_id"))}:
                return item
    return structures[0] if isinstance(structures[0], dict) else None


def _structure_chat_context(structure: dict[str, object] | None) -> dict[str, object] | None:
    if not structure:
        return None
    metrics = structure.get("metrics") if isinstance(structure.get("metrics"), dict) else {}
    return {
        "id": structure.get("id"),
        "pdb_id": structure.get("pdb_id"),
        "success": structure.get("success"),
        "error": structure.get("error"),
        "metrics": {key: value for key, value in metrics.items() if key != "total_execution_time"},
        "evidence_links": {
            "interface": _first_refs(structure, "interface_residues", 6),
            "hotspots": _first_refs(structure, "hotspots", 4),
            "flexible": _first_refs(structure, "high_bfactor_residues", 4),
            "geometry": _first_refs(structure, "ramachandran_outliers", 4),
            "charge": _first_charge_refs(structure, 5),
        },
    }


def _deterministic_chat_response(context: dict[str, object]) -> str:
    project = context.get("project") if isinstance(context.get("project"), dict) else {}
    selected_structure = (
        context.get("selected_structure") if isinstance(context.get("selected_structure"), dict) else None
    )
    batch_summary = context.get("batch_summary") if isinstance(context.get("batch_summary"), dict) else {}
    target = project.get("target") if isinstance(project, dict) else None

    lines = [f"I have the `{project.get('name', 'project')}` workspace loaded."]
    if isinstance(target, dict):
        lines.append(f"The current target is `{target.get('pdb_id')}`.")
    else:
        lines.append("No project target has been uploaded yet.")

    if selected_structure:
        metrics = selected_structure.get("metrics") if isinstance(selected_structure.get("metrics"), dict) else {}
        lines.append(f"The selected structure is `{selected_structure.get('pdb_id')}` with {_metric_clause(metrics)}.")
        refs = []
        evidence = selected_structure.get("evidence_links")
        if isinstance(evidence, dict):
            for values in evidence.values():
                if isinstance(values, list):
                    refs.extend(str(value) for value in values)
        if refs:
            lines.append(f"Useful highlighted regions include {', '.join(refs[:4])}.")
    elif batch_summary:
        lines.append(
            "A batch is available; select a ranked structure to ground the next answer in residue-level evidence."
        )
    else:
        lines.append("Run a candidate batch to add ranked metrics, report synthesis, and clickable residue evidence.")
    return "\n\n".join(lines)


def _resolve_chat_llm_config() -> tuple[str, str | None, str | None, str | None]:
    provider_name = (os.getenv("MIRA_REPORT_PROVIDER") or _infer_report_provider()).lower()
    defaults = PROVIDER_DEFAULTS.get(provider_name, PROVIDER_DEFAULTS["openai"])
    model = os.getenv("MIRA_REPORT_MODEL") or defaults.get("model")
    base_url = os.getenv("MIRA_REPORT_BASE_URL") or defaults.get("base_url")
    api_key = os.getenv("MIRA_REPORT_API_KEY") or _first_env_value(defaults.get("env_vars", ()))
    return provider_name, model, base_url, api_key


def _clean_chat_response(markdown: str) -> str:
    content = re.sub(r"<think>.*?</think>", "", markdown, flags=re.DOTALL | re.IGNORECASE).strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("markdown"):
            content = content[len("markdown") :].strip()
    return content.strip()


def _metric_clause(metrics: dict[str, object]) -> str:
    clauses = []
    for key, label, suffix in [
        ("buried_surface_area", "buried surface area", " A^2"),
        ("n_interface_residues", "interface residues", ""),
        ("mean_relative_sasa_percent", "mean relative SASA", "%"),
        ("mean_bfactor", "mean B-factor", ""),
        ("charge_cluster_count", "charge clusters", ""),
    ]:
        value = _metric(metrics, key)
        if value is not None:
            clauses.append(f"{label} {_fmt(value)}{suffix}")
    return ", ".join(clauses) if clauses else "no rankable metrics available"


def _metric(metrics: dict[str, object], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _first_refs(item: dict[str, object], evidence_key: str, limit: int) -> list[str]:
    features = item.get("features")
    if not isinstance(features, dict):
        return []
    refs = [_region_link(evidence_key, feature) for feature in features.get(evidence_key, [])[:limit]]
    return _dedupe([ref for ref in refs if ref])


def _first_charge_refs(item: dict[str, object], limit: int) -> list[str]:
    features = item.get("features")
    if not isinstance(features, dict):
        return []
    refs = []
    for cluster in features.get("charge_clusters", []):
        if not isinstance(cluster, dict):
            continue
        for residue in cluster.get("residues") or []:
            link = _region_link("charge_clusters", residue)
            if link and link not in refs:
                refs.append(link)
            if len(refs) >= limit:
                return refs
    return refs


def _region_link(evidence_key: str, feature: object) -> str | None:
    if not isinstance(feature, dict):
        return None
    residue_number = feature.get("residue_number")
    if residue_number is None:
        return None
    chain = feature.get("chain") or "any"
    residue_name = feature.get("residue_name") or "Residue"
    label = f"{residue_name}-{residue_number}"
    if chain != "any":
        label = f"{label} chain {chain}"
    href = f"mira://region/{quote(evidence_key)}/{quote(str(chain))}/{quote(str(residue_number))}"
    return f"[{label}]({href})"


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


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
    assets_dir = dist_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="mira-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str) -> FileResponse:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        requested = dist_dir / full_path
        if requested.is_file() and _is_under(requested, dist_dir):
            return FileResponse(requested)
        return FileResponse(dist_dir / "index.html")
