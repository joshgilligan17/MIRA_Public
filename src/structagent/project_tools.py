"""Constrained project tools for hosted chat workflows."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import requests

from structagent.projects import ProjectRecord, ProjectStore
from structagent.registry import ToolResult


PDB_ID_PATTERN = re.compile(r"\b([0-9][A-Za-z0-9]{3})\b")

PROJECT_CHAT_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "load_pdb_id",
        "description": "Fetch an RCSB PDB/mmCIF structure by 4-character PDB ID and select it in the project viewer.",
        "parameters": {
            "type": "object",
            "properties": {
                "pdb_id": {
                    "type": "string",
                    "description": "A 4-character RCSB PDB identifier, e.g. 1UBQ.",
                }
            },
            "required": ["pdb_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "select_project_structure",
        "description": "Select an already-loaded project structure or target for the structure viewer.",
        "parameters": {
            "type": "object",
            "properties": {
                "structure_id_or_pdb_id": {
                    "type": "string",
                    "description": "The project structure id, 'target', or a visible PDB/file stem such as 1UBQ.",
                }
            },
            "required": ["structure_id_or_pdb_id"],
            "additionalProperties": False,
        },
    },
]


def message_may_need_project_tool(message: str) -> bool:
    lowered = message.lower()
    action_words = ("load", "open", "pull up", "show", "display", "select", "view")
    return bool(PDB_ID_PATTERN.search(message)) or any(word in lowered for word in action_words)


def fallback_project_tool_calls(message: str) -> list[dict[str, Any]]:
    pdb_id = extract_pdb_id(message)
    if not pdb_id:
        return []
    return [{"tool": "load_pdb_id", "args": {"pdb_id": pdb_id}, "purpose": "Detected explicit PDB ID."}]


def extract_pdb_id(message: str) -> str | None:
    match = PDB_ID_PATTERN.search(message)
    return match.group(1).upper() if match else None


def execute_project_chat_tool(
    project_store: ProjectStore,
    project: ProjectRecord,
    tool_name: str,
    args: dict[str, Any],
) -> tuple[ProjectRecord, ToolResult]:
    if tool_name == "load_pdb_id":
        return _load_pdb_id(project_store, project, args)
    if tool_name == "select_project_structure":
        return _select_project_structure(project_store, project, args)
    return (
        project,
        ToolResult(
            success=False,
            data=f"Unknown project chat tool: {tool_name}",
            raw={"tool": tool_name},
            error=f"Unknown project chat tool: {tool_name}",
            tool_name=tool_name,
        ),
    )


def project_structure_identity(project: ProjectRecord, structure_id_or_pdb_id: str) -> dict[str, Any] | None:
    requested = structure_id_or_pdb_id.strip().upper()
    if not requested:
        return None
    if requested == "TARGET" and project.target_file:
        return {"id": "target", "pdb_id": _pdb_id_from_name(project.target_original_name or "target")}

    if project.target_file:
        target_pdb_id = _pdb_id_from_name(project.target_original_name or "target")
        if requested in {"TARGET", target_pdb_id}:
            return {"id": "target", "pdb_id": target_pdb_id}

    for structure in project.structures:
        pdb_id = _pdb_id_from_name(structure.original_name)
        if requested in {structure.id.upper(), pdb_id}:
            return {"id": structure.id, "pdb_id": pdb_id}
    return None


def _load_pdb_id(
    project_store: ProjectStore, project: ProjectRecord, args: dict[str, Any]
) -> tuple[ProjectRecord, ToolResult]:
    pdb_id = str(args.get("pdb_id") or "").strip().upper()
    if not PDB_ID_PATTERN.fullmatch(pdb_id):
        return (
            project,
            ToolResult(
                success=False,
                data="load_pdb_id requires a valid 4-character RCSB PDB ID.",
                raw={"pdb_id": pdb_id},
                error="Invalid PDB ID",
                tool_name="load_pdb_id",
            ),
        )

    existing = project_structure_identity(project, pdb_id)
    if existing:
        selected_project = project_store.set_selection(project.id, None, str(existing["id"]))
        return (
            selected_project,
            ToolResult(
                success=True,
                data=f"Selected existing project structure {pdb_id}.",
                raw={
                    "status": "selected_existing",
                    "pdb_id": pdb_id,
                    "selected_job_id": None,
                    "selected_structure_id": existing["id"],
                },
                tool_name="load_pdb_id",
            ),
        )

    try:
        content = _download_rcsb_cif(pdb_id)
    except ValueError as exc:
        return (
            project,
            ToolResult(
                success=False,
                data=f"Could not load {pdb_id} from RCSB.",
                raw={"status": "failed", "pdb_id": pdb_id},
                error=str(exc),
                tool_name="load_pdb_id",
            ),
        )

    updated_project, structure = project_store.save_structure(project.id, f"{pdb_id}.cif", content)
    return (
        updated_project,
        ToolResult(
            success=True,
            data=f"Loaded {pdb_id} from RCSB and selected it in the structure viewer.",
            raw={
                "status": "loaded",
                "pdb_id": pdb_id,
                "selected_job_id": None,
                "selected_structure_id": structure.id,
            },
            tool_name="load_pdb_id",
        ),
    )


def _select_project_structure(
    project_store: ProjectStore, project: ProjectRecord, args: dict[str, Any]
) -> tuple[ProjectRecord, ToolResult]:
    requested = str(args.get("structure_id_or_pdb_id") or "").strip()
    structure = project_structure_identity(project, requested)
    if not structure:
        return (
            project,
            ToolResult(
                success=False,
                data=f"No project structure matched {requested!r}.",
                raw={"requested": requested},
                error="Structure not found",
                tool_name="select_project_structure",
            ),
        )
    selected_project = project_store.set_selection(project.id, None, str(structure["id"]))
    return (
        selected_project,
        ToolResult(
            success=True,
            data=f"Selected structure {structure['pdb_id']}.",
            raw={
                "status": "selected_existing",
                "pdb_id": structure["pdb_id"],
                "selected_job_id": None,
                "selected_structure_id": structure["id"],
            },
            tool_name="select_project_structure",
        ),
    )


def _download_rcsb_cif(pdb_id: str) -> bytes:
    pdb_id = pdb_id.upper()
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    try:
        response = requests.get(url, timeout=30.0)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError(f"failed to download {pdb_id} from RCSB") from exc
    return response.content


def _pdb_id_from_name(filename: str) -> str:
    return Path(filename).stem.upper()
