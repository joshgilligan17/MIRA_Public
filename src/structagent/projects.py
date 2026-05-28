"""Filesystem-backed project storage for the hosted MIRA workspace."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4


ChatRole = Literal["user", "assistant"]


@dataclass
class ChatMessage:
    """Persisted project chat message."""

    id: str
    role: ChatRole
    content: str
    created_at: str
    selected_job_id: str | None = None
    selected_structure_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatMessage":
        return cls(
            id=data["id"],
            role=data["role"],
            content=data["content"],
            created_at=data["created_at"],
            selected_job_id=data.get("selected_job_id"),
            selected_structure_id=data.get("selected_structure_id"),
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
    job_ids: list[str] = field(default_factory=list)
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
            "job_ids": self.job_ids,
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
            job_ids=data.get("job_ids") or [],
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
            if hasattr(project, key) and value is not None:
                setattr(project, key, value)
        self.write_project(project)
        return project

    def save_target(self, project_id: str, filename: str, content: bytes) -> ProjectRecord:
        project = self.get_project(project_id)
        safe_name = Path(filename or "target.pdb").name
        path = _unique_path(self.target_dir(project_id) / safe_name)
        path.write_bytes(content)
        project.target_file = path.name
        project.target_original_name = safe_name
        project.target_uploaded_at = _now()
        self.write_project(project)
        return project

    def add_job(self, project_id: str, job_id: str) -> ProjectRecord:
        project = self.get_project(project_id)
        if job_id not in project.job_ids:
            project.job_ids.append(job_id)
        project.selected_job_id = job_id
        self.write_project(project)
        return project

    def append_chat_message(
        self,
        project_id: str,
        role: ChatRole,
        content: str,
        selected_job_id: str | None = None,
        selected_structure_id: str | None = None,
    ) -> ChatMessage:
        project = self.get_project(project_id)
        message = ChatMessage(
            id=uuid4().hex[:12],
            role=role,
            content=content.strip(),
            created_at=_now(),
            selected_job_id=selected_job_id,
            selected_structure_id=selected_structure_id,
        )
        project.chat_messages.append(message)
        project.selected_job_id = selected_job_id or project.selected_job_id
        project.selected_structure_id = selected_structure_id or project.selected_structure_id
        self.write_project(project)
        return message

    def project_dir(self, project_id: str) -> Path:
        return self.root / project_id

    def target_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "target"

    def target_path(self, project: ProjectRecord) -> Path | None:
        if not project.target_file:
            return None
        return self.target_dir(project.id) / project.target_file

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
