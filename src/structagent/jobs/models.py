"""Dataclasses for persisted MIRA batch jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


JobStatus = Literal["queued", "running", "completed", "failed"]


@dataclass
class JobConfig:
    """User-selected settings for one batch job."""

    query: str
    profile: str = "triage_default"
    rank_by: str = "stability"
    glob_pattern: str = "*"
    chain_a: str | None = None
    chain_b: str | None = None
    max_workers: int = 2
    enable_llm_synthesis: bool = True
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_temperature: float = 0.2


@dataclass
class JobRecord:
    """Persisted job metadata."""

    id: str
    status: JobStatus
    config: JobConfig
    created_at: str
    updated_at: str
    input_files: list[str] = field(default_factory=list)
    total_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "config": self.config.__dict__,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "input_files": self.input_files,
            "total_count": self.total_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobRecord":
        config = JobConfig(**data["config"])
        return cls(
            id=data["id"],
            status=data["status"],
            config=config,
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            input_files=data.get("input_files") or [],
            total_count=data.get("total_count", 0),
            completed_count=data.get("completed_count", 0),
            failed_count=data.get("failed_count", 0),
            error=data.get("error"),
        )
