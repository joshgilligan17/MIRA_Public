"""Filesystem-backed job storage for local MIRA batch jobs."""

from __future__ import annotations

import json
import math
from dataclasses import is_dataclass, asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from structagent.jobs.models import JobConfig, JobRecord, JobStatus


class JobStore:
    """Small JSON store rooted at `.mira/jobs` by default."""

    def __init__(self, root: str | Path = ".mira/jobs"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_job(self, config: JobConfig) -> JobRecord:
        job_id = uuid4().hex[:12]
        now = _now()
        record = JobRecord(id=job_id, status="queued", config=config, created_at=now, updated_at=now)
        self.job_dir(job_id).mkdir(parents=True, exist_ok=True)
        self.input_dir(job_id).mkdir(parents=True, exist_ok=True)
        self.write_record(record)
        self.append_event(job_id, "queued", "Job queued.")
        return record

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def input_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "inputs"

    def metadata_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "metadata.json"

    def results_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "results.json"

    def report_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "report.md"

    def events_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "events.jsonl"

    def get_record(self, job_id: str) -> JobRecord:
        path = self.metadata_path(job_id)
        if not path.exists():
            raise FileNotFoundError(f"Unknown job: {job_id}")
        return JobRecord.from_dict(json.loads(path.read_text()))

    def write_record(self, record: JobRecord) -> None:
        record.updated_at = _now()
        _write_json(self.metadata_path(record.id), record.to_dict())

    def update_record(self, job_id: str, **updates: Any) -> JobRecord:
        record = self.get_record(job_id)
        for key, value in updates.items():
            if hasattr(record, key):
                setattr(record, key, value)
        self.write_record(record)
        return record

    def set_status(self, job_id: str, status: JobStatus, message: str, error: str | None = None) -> JobRecord:
        record = self.update_record(job_id, status=status, error=error)
        self.append_event(job_id, status, message, error=error)
        return record

    def save_results(self, job_id: str, results: dict[str, Any]) -> None:
        _write_json(self.results_path(job_id), results)

    def load_results(self, job_id: str) -> dict[str, Any]:
        path = self.results_path(job_id)
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def save_report(self, job_id: str, report: str) -> None:
        self.report_path(job_id).write_text(report)

    def append_event(self, job_id: str, event_type: str, message: str, **extra: Any) -> None:
        event = {"timestamp": _now(), "type": event_type, "message": message, **extra}
        path = self.events_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(_json_safe(event), sort_keys=True) + "\n")

    def list_events(self, job_id: str) -> list[dict[str, Any]]:
        path = self.events_path(job_id)
        if not path.exists():
            return []
        events = []
        for line in path.read_text().splitlines():
            if line.strip():
                events.append(json.loads(line))
        return events


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
