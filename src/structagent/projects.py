"""Filesystem-backed project storage for the hosted MIRA workspace."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4


ChatRole = Literal["user", "assistant"]


@dataclass
class ProjectStructure:
    """Persisted standalone structure in a project chat workspace."""

    id: str
    filename: str
    original_name: str
    uploaded_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectStructure":
        return cls(
            id=data["id"],
            filename=data["filename"],
            original_name=data.get("original_name") or data["filename"],
            uploaded_at=data["uploaded_at"],
        )


@dataclass
class ProjectAnalysis:
    """Persisted analysis produced by project chat tools."""

    id: str
    kind: str
    query: str
    status: str
    created_at: str
    updated_at: str
    selected_job_id: str | None = None
    selected_structure_id: str | None = None
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "id": self.id,
                "kind": self.kind,
                "query": self.query,
                "status": self.status,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "selected_job_id": self.selected_job_id,
                "selected_structure_id": self.selected_structure_id,
                "tool_events": self.tool_events,
                "metrics": self.metrics,
                "features": self.features,
                "summary": self.summary,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectAnalysis":
        return cls(
            id=data["id"],
            kind=data.get("kind") or "analysis",
            query=data.get("query") or "",
            status=data.get("status") or "completed",
            created_at=data["created_at"],
            updated_at=data.get("updated_at") or data["created_at"],
            selected_job_id=data.get("selected_job_id"),
            selected_structure_id=data.get("selected_structure_id"),
            tool_events=data.get("tool_events") or [],
            metrics=data.get("metrics") or {},
            features=data.get("features") or {},
            summary=data.get("summary") or "",
        )


@dataclass
class ProjectDesignRun:
    """Persisted generative design-library invocation."""

    id: str
    library: str
    prompt: str
    status: str
    created_at: str
    updated_at: str
    target_structure_id: str | None = None
    output_dir: str | None = None
    command: str | None = None
    num_designs: int = 0
    parameters: dict[str, Any] = field(default_factory=dict)
    generated_structure_ids: list[str] = field(default_factory=list)
    generated_sequences: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    logs: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "id": self.id,
                "library": self.library,
                "prompt": self.prompt,
                "status": self.status,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "target_structure_id": self.target_structure_id,
                "output_dir": self.output_dir,
                "command": self.command,
                "num_designs": self.num_designs,
                "parameters": self.parameters,
                "generated_structure_ids": self.generated_structure_ids,
                "generated_sequences": self.generated_sequences,
                "artifacts": self.artifacts,
                "logs": self.logs,
                "error": self.error,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectDesignRun":
        return cls(
            id=data["id"],
            library=data.get("library") or "unknown",
            prompt=data.get("prompt") or "",
            status=data.get("status") or "queued",
            created_at=data["created_at"],
            updated_at=data.get("updated_at") or data["created_at"],
            target_structure_id=data.get("target_structure_id"),
            output_dir=data.get("output_dir"),
            command=data.get("command"),
            num_designs=int(data.get("num_designs") or 0),
            parameters=data.get("parameters") or {},
            generated_structure_ids=data.get("generated_structure_ids") or [],
            generated_sequences=data.get("generated_sequences") or [],
            artifacts=data.get("artifacts") or [],
            logs=data.get("logs") or "",
            error=data.get("error"),
        )


@dataclass
class ChatMessage:
    """Persisted project chat message."""

    id: str
    role: ChatRole
    content: str
    created_at: str
    selected_job_id: str | None = None
    selected_structure_id: str | None = None
    tool_events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "id": self.id,
                "role": self.role,
                "content": self.content,
                "created_at": self.created_at,
                "selected_job_id": self.selected_job_id,
                "selected_structure_id": self.selected_structure_id,
                "tool_events": self.tool_events,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatMessage":
        return cls(
            id=data["id"],
            role=data["role"],
            content=data["content"],
            created_at=data["created_at"],
            selected_job_id=data.get("selected_job_id"),
            selected_structure_id=data.get("selected_structure_id"),
            tool_events=data.get("tool_events") or [],
        )


@dataclass
class ProjectRecord:
    """Persisted project metadata and lightweight workspace state."""

    id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    target_file: str | None = None
    target_original_name: str | None = None
    target_uploaded_at: str | None = None
    structures: list[ProjectStructure] = field(default_factory=list)
    job_ids: list[str] = field(default_factory=list)
    analysis_ids: list[str] = field(default_factory=list)
    design_run_ids: list[str] = field(default_factory=list)
    chat_messages: list[ChatMessage] = field(default_factory=list)
    selected_job_id: str | None = None
    selected_structure_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "target_file": self.target_file,
            "target_original_name": self.target_original_name,
            "target_uploaded_at": self.target_uploaded_at,
            "structures": [structure.to_dict() for structure in self.structures],
            "job_ids": self.job_ids,
            "analysis_ids": self.analysis_ids,
            "design_run_ids": self.design_run_ids,
            "chat_messages": [message.to_dict() for message in self.chat_messages],
            "selected_job_id": self.selected_job_id,
            "selected_structure_id": self.selected_structure_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectRecord":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description") or "",
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            target_file=data.get("target_file"),
            target_original_name=data.get("target_original_name"),
            target_uploaded_at=data.get("target_uploaded_at"),
            structures=[ProjectStructure.from_dict(item) for item in data.get("structures", [])],
            job_ids=data.get("job_ids") or [],
            analysis_ids=data.get("analysis_ids") or [],
            design_run_ids=data.get("design_run_ids") or [],
            chat_messages=[ChatMessage.from_dict(item) for item in data.get("chat_messages", [])],
            selected_job_id=data.get("selected_job_id"),
            selected_structure_id=data.get("selected_structure_id"),
        )


class ProjectStore:
    """Small JSON store rooted beside MIRA jobs."""

    def __init__(self, root: str | Path = ".mira/projects"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_project(self, name: str, description: str = "") -> ProjectRecord:
        now = _now()
        project = ProjectRecord(
            id=uuid4().hex[:12],
            name=name.strip() or "Untitled project",
            description=description.strip(),
            created_at=now,
            updated_at=now,
        )
        self.project_dir(project.id).mkdir(parents=True, exist_ok=True)
        self.write_project(project)
        return project

    def list_projects(self) -> list[ProjectRecord]:
        projects = []
        for path in sorted(self.root.glob("*/project.json")):
            try:
                projects.append(ProjectRecord.from_dict(json.loads(path.read_text())))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return sorted(projects, key=lambda project: project.updated_at, reverse=True)

    def get_project(self, project_id: str) -> ProjectRecord:
        path = self.metadata_path(project_id)
        if not path.exists():
            raise FileNotFoundError(f"Unknown project: {project_id}")
        return ProjectRecord.from_dict(json.loads(path.read_text()))

    def write_project(self, project: ProjectRecord) -> None:
        project.updated_at = _now()
        _write_json(self.metadata_path(project.id), project.to_dict())

    def update_project(self, project_id: str, **updates: Any) -> ProjectRecord:
        project = self.get_project(project_id)
        for key, value in updates.items():
            if hasattr(project, key):
                setattr(project, key, value)
        self.write_project(project)
        return project

    def delete_project(self, project_id: str) -> ProjectRecord:
        project = self.get_project(project_id)
        project_dir = self.project_dir(project_id).resolve()
        root = self.root.resolve()
        if project_dir == root or root not in project_dir.parents:
            raise ValueError(f"Refusing to delete unsafe project path: {project_id}")
        shutil.rmtree(project_dir)
        return project

    def save_target(self, project_id: str, filename: str, content: bytes) -> ProjectRecord:
        project = self.get_project(project_id)
        safe_name = Path(filename or "target.pdb").name
        path = _unique_path(self.target_dir(project_id) / safe_name)
        path.write_bytes(content)
        project.target_file = path.name
        project.target_original_name = safe_name
        project.target_uploaded_at = _now()
        project.selected_job_id = None
        project.selected_structure_id = "target"
        self.write_project(project)
        return project

    def save_structure(self, project_id: str, filename: str, content: bytes) -> tuple[ProjectRecord, ProjectStructure]:
        project = self.get_project(project_id)
        structure_id = uuid4().hex[:12]
        safe_name = Path(filename or "structure.pdb").name
        stored_name = f"{structure_id}{Path(safe_name).suffix.lower() or '.pdb'}"
        path = self.structure_dir(project_id) / stored_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        structure = ProjectStructure(
            id=structure_id,
            filename=stored_name,
            original_name=safe_name,
            uploaded_at=_now(),
        )
        project.structures.append(structure)
        project.selected_job_id = None
        project.selected_structure_id = structure.id
        self.write_project(project)
        return project, structure

    def add_job(self, project_id: str, job_id: str) -> ProjectRecord:
        project = self.get_project(project_id)
        if job_id not in project.job_ids:
            project.job_ids.append(job_id)
        project.selected_job_id = job_id
        self.write_project(project)
        return project

    def set_selection(
        self, project_id: str, selected_job_id: str | None, selected_structure_id: str | None
    ) -> ProjectRecord:
        project = self.get_project(project_id)
        project.selected_job_id = selected_job_id
        project.selected_structure_id = selected_structure_id
        self.write_project(project)
        return project

    def append_chat_message(
        self,
        project_id: str,
        role: ChatRole,
        content: str,
        selected_job_id: str | None = None,
        selected_structure_id: str | None = None,
        tool_events: list[dict[str, Any]] | None = None,
    ) -> ChatMessage:
        project = self.get_project(project_id)
        message = ChatMessage(
            id=uuid4().hex[:12],
            role=role,
            content=content.strip(),
            created_at=_now(),
            selected_job_id=selected_job_id,
            selected_structure_id=selected_structure_id,
            tool_events=tool_events or [],
        )
        project.chat_messages.append(message)
        project.selected_job_id = selected_job_id
        project.selected_structure_id = selected_structure_id
        self.write_project(project)
        return message

    def save_analysis(
        self,
        project_id: str,
        *,
        kind: str,
        query: str,
        status: str,
        selected_job_id: str | None = None,
        selected_structure_id: str | None = None,
        tool_events: list[dict[str, Any]] | None = None,
        metrics: dict[str, Any] | None = None,
        features: dict[str, Any] | None = None,
        summary: str = "",
    ) -> ProjectAnalysis:
        project = self.get_project(project_id)
        now = _now()
        analysis = ProjectAnalysis(
            id=uuid4().hex[:12],
            kind=kind,
            query=query,
            status=status,
            created_at=now,
            updated_at=now,
            selected_job_id=selected_job_id,
            selected_structure_id=selected_structure_id,
            tool_events=tool_events or [],
            metrics=metrics or {},
            features=features or {},
            summary=summary,
        )
        _write_json(self.analysis_path(project_id, analysis.id), analysis.to_dict())
        if analysis.id not in project.analysis_ids:
            project.analysis_ids.append(analysis.id)
        self.write_project(project)
        return analysis

    def list_analyses(self, project_id: str) -> list[ProjectAnalysis]:
        analyses = []
        for path in sorted(self.analysis_dir(project_id).glob("*.json")):
            try:
                analyses.append(ProjectAnalysis.from_dict(json.loads(path.read_text())))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return sorted(analyses, key=lambda analysis: analysis.updated_at, reverse=True)

    def create_design_run(
        self,
        project_id: str,
        *,
        library: str,
        prompt: str,
        target_structure_id: str | None,
        output_dir: str | None,
        command: str | None,
        num_designs: int = 0,
        parameters: dict[str, Any] | None = None,
        status: str = "queued",
        error: str | None = None,
    ) -> ProjectDesignRun:
        project = self.get_project(project_id)
        now = _now()
        run = ProjectDesignRun(
            id=uuid4().hex[:12],
            library=library,
            prompt=prompt,
            status=status,
            created_at=now,
            updated_at=now,
            target_structure_id=target_structure_id,
            output_dir=output_dir,
            command=command,
            num_designs=num_designs,
            parameters=parameters or {},
            error=error,
        )
        _write_json(self.design_run_path(project_id, run.id), run.to_dict())
        if run.id not in project.design_run_ids:
            project.design_run_ids.append(run.id)
        self.write_project(project)
        return run

    def update_design_run(self, project_id: str, run_id: str, **updates: Any) -> ProjectDesignRun:
        run = ProjectDesignRun.from_dict(json.loads(self.design_run_path(project_id, run_id).read_text()))
        for key, value in updates.items():
            if hasattr(run, key):
                setattr(run, key, value)
        run.updated_at = _now()
        _write_json(self.design_run_path(project_id, run.id), run.to_dict())
        return run

    def list_design_runs(self, project_id: str) -> list[ProjectDesignRun]:
        runs = []
        for path in sorted(self.design_runs_dir(project_id).glob("*.json")):
            try:
                runs.append(ProjectDesignRun.from_dict(json.loads(path.read_text())))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return sorted(runs, key=lambda run: run.updated_at, reverse=True)

    def project_dir(self, project_id: str) -> Path:
        return self.root / project_id

    def target_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "target"

    def structure_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "structures"

    def analysis_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "analyses"

    def analysis_path(self, project_id: str, analysis_id: str) -> Path:
        return self.analysis_dir(project_id) / f"{analysis_id}.json"

    def designs_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "designs"

    def design_run_output_dir(self, project_id: str, run_id: str) -> Path:
        return self.designs_dir(project_id) / run_id / "outputs"

    def design_runs_dir(self, project_id: str) -> Path:
        return self.designs_dir(project_id) / "runs"

    def design_run_path(self, project_id: str, run_id: str) -> Path:
        return self.design_runs_dir(project_id) / f"{run_id}.json"

    def target_path(self, project: ProjectRecord) -> Path | None:
        if not project.target_file:
            return None
        return self.target_dir(project.id) / project.target_file

    def structure_path(self, project: ProjectRecord, structure_id: str) -> Path | None:
        structure = next((item for item in project.structures if item.id == structure_id), None)
        if not structure:
            return None
        return self.structure_dir(project.id) / structure.filename

    def metadata_path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "project.json"


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


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(_json_safe(data), indent=2, sort_keys=True))
    temp_path.replace(path)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _now() -> str:
    return datetime.now(UTC).isoformat()
