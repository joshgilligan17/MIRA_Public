"""Constrained project tools for hosted chat workflows."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from structagent.jobs.models import JobConfig
from structagent.jobs.runner import JobRunner
from structagent.jobs.store import JobStore
from structagent.projects import ProjectRecord, ProjectStore
from structagent.registry import ToolRegistry, ToolResult


PDB_ID_PATTERN = re.compile(r"\b([0-9][A-Za-z0-9]{3})\b")
SUPPORTED_STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif"}


@dataclass
class ProjectToolRuntime:
    project_store: ProjectStore
    job_store: JobStore | None = None
    runner: JobRunner | None = None
    background_tasks: Any = None
    llm_api_key: str | None = None


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
    {
        "name": "analyze_structure",
        "description": (
            "Run project-scoped target analysis on the selected or named structure. "
            "Use this for chain inventory, residue numbering, flexibility, SASA, charge, and geometry checks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "structure_id_or_pdb_id": {
                    "type": "string",
                    "description": "Project structure id, 'target', or PDB/file stem. Defaults to selected structure.",
                },
                "chain_id": {
                    "type": "string",
                    "description": "Optional chain to analyze. If omitted, MIRA analyzes up to three chains.",
                },
                "analyses": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "load_structure",
                            "list_residues",
                            "bfactors",
                            "sasa",
                            "charge",
                            "ramachandran",
                            "secondary_structure",
                        ],
                    },
                    "description": "Subset of analyses to run. Omit for the default target-analysis bundle.",
                },
                "residue_range": {"type": "string", "description": "Optional range such as 1-80."},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "analyze_contacts",
        "description": "Analyze contacts around a specific residue in a selected project structure.",
        "parameters": {
            "type": "object",
            "properties": {
                "structure_id_or_pdb_id": {"type": "string"},
                "chain_id": {"type": "string"},
                "residue_number": {"type": "integer"},
                "cutoff_angstroms": {"type": "number", "default": 4.5},
            },
            "required": ["chain_id", "residue_number"],
            "additionalProperties": False,
        },
    },
    {
        "name": "analyze_interface",
        "description": "Analyze a two-chain protein interface in a selected project structure.",
        "parameters": {
            "type": "object",
            "properties": {
                "structure_id_or_pdb_id": {"type": "string"},
                "chain_a": {"type": "string"},
                "chain_b": {"type": "string"},
                "distance_cutoff": {"type": "number", "default": 5.0},
            },
            "required": ["chain_a", "chain_b"],
            "additionalProperties": False,
        },
    },
    {
        "name": "start_batch_from_project",
        "description": (
            "Start a MIRA batch screen from already-loaded project structures or generated candidates. "
            "Do this when the user wants to rank/evaluate candidates from chat."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "structure_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Project structure ids or PDB/file stems. Omit to use all non-target project structures.",
                },
                "query": {"type": "string", "description": "Batch analysis query."},
                "profile": {"type": "string", "default": "triage_default"},
                "rank_by": {"type": "string", "default": "stability"},
                "max_workers": {"type": "integer", "default": 2},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "generate_design_candidates",
        "description": (
            "Invoke a configured local generative design library such as BindCraft, RFdiffusion, or ProteinMPNN. "
            "If the library command is not configured, create a design setup record explaining what is missing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "library": {
                    "type": "string",
                    "enum": ["bindcraft", "rfdiffusion", "proteinmpnn", "custom"],
                    "default": "custom",
                },
                "target_structure_id_or_pdb_id": {
                    "type": "string",
                    "description": "Target structure id, 'target', or PDB/file stem. Defaults to selected/target.",
                },
                "chain_id": {"type": "string", "description": "Optional target chain or receptor chain."},
                "num_designs": {"type": "integer", "default": 8},
                "design_prompt": {
                    "type": "string",
                    "description": "Short design objective, constraints, or strategy from the user.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
]


def message_may_need_project_tool(message: str) -> bool:
    lowered = message.lower()
    action_words = (
        "load",
        "open",
        "pull up",
        "show",
        "display",
        "select",
        "view",
        "analyze",
        "analyse",
        "flexible",
        "b-factor",
        "sasa",
        "surface",
        "charge",
        "contact",
        "interface",
        "residue",
        "ramachandran",
        "batch",
        "rank",
        "screen",
        "candidate",
        "binder",
        "design",
        "generate",
        "rfdiffusion",
        "bindcraft",
        "proteinmpnn",
    )
    return bool(PDB_ID_PATTERN.search(message)) or any(word in lowered for word in action_words)


def fallback_project_tool_calls(message: str) -> list[dict[str, Any]]:
    lowered = message.lower()
    pdb_id = extract_pdb_id(message)
    calls: list[dict[str, Any]] = []
    if pdb_id:
        calls.append({"tool": "load_pdb_id", "args": {"pdb_id": pdb_id}, "purpose": "Detected explicit PDB ID."})
    if any(word in lowered for word in ("analyze", "analyse", "flexible", "sasa", "charge", "ramachandran")):
        calls.append({"tool": "analyze_structure", "args": {}, "purpose": "Detected target-analysis request."})
    if any(word in lowered for word in ("batch", "rank", "screen")):
        calls.append({"tool": "start_batch_from_project", "args": {}, "purpose": "Detected batch-screening request."})
    if any(word in lowered for word in ("design", "generate", "rfdiffusion", "bindcraft", "proteinmpnn")):
        calls.append(
            {
                "tool": "generate_design_candidates",
                "args": {"design_prompt": message},
                "purpose": "Detected candidate design request.",
            }
        )
    return calls[:6]


def extract_pdb_id(message: str) -> str | None:
    match = PDB_ID_PATTERN.search(message)
    return match.group(1).upper() if match else None


def execute_project_chat_tool(
    runtime: ProjectToolRuntime,
    project: ProjectRecord,
    tool_name: str,
    args: dict[str, Any],
) -> tuple[ProjectRecord, ToolResult]:
    if tool_name == "load_pdb_id":
        return _load_pdb_id(runtime.project_store, project, args)
    if tool_name == "select_project_structure":
        return _select_project_structure(runtime.project_store, project, args)
    if tool_name == "analyze_structure":
        return _analyze_structure(runtime, project, args)
    if tool_name == "analyze_contacts":
        return _analyze_contacts(runtime, project, args)
    if tool_name == "analyze_interface":
        return _analyze_interface(runtime, project, args)
    if tool_name == "start_batch_from_project":
        return _start_batch_from_project(runtime, project, args)
    if tool_name == "generate_design_candidates":
        return _generate_design_candidates(runtime, project, args)
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


def _analyze_structure(
    runtime: ProjectToolRuntime, project: ProjectRecord, args: dict[str, Any]
) -> tuple[ProjectRecord, ToolResult]:
    ref = _resolve_structure(runtime.project_store, project, args.get("structure_id_or_pdb_id"))
    if not ref:
        return _tool_error(project, "analyze_structure", "No selected project structure is available.")

    registry = _structure_registry()
    requested = args.get("analyses") if isinstance(args.get("analyses"), list) else None
    analyses = requested or ["load_structure", "list_residues", "bfactors", "sasa", "charge", "ramachandran"]
    chain_ids = _requested_chain_ids(ref, args.get("chain_id"))
    residue_range = str(args.get("residue_range") or "").strip() or None

    tool_events: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    features: dict[str, Any] = {}
    summaries: list[str] = []

    if "load_structure" in analyses:
        result = registry.call_tool("load_structure", pdb_path=str(ref["path"]))
        _append_tool_event(tool_events, result, {"structure_id": ref["id"]})
        _merge_tool_output(metrics, features, result, chain_id=None)
        summaries.append(_short_text(result.data))
        if result.success and isinstance(result.raw, dict) and result.raw.get("chains") and not chain_ids:
            chain_ids = [str(chain["id"]) for chain in result.raw["chains"][:3] if chain.get("id")]

    if not chain_ids:
        chain_ids = ["A"]

    tool_map = {
        "list_residues": "list_residues",
        "bfactors": "analyze_bfactors",
        "sasa": "compute_sasa",
        "charge": "compute_charge_distribution",
        "ramachandran": "check_ramachandran",
        "secondary_structure": "get_secondary_structure",
    }
    for analysis in analyses:
        tool_name = tool_map.get(str(analysis))
        if not tool_name:
            continue
        for chain_id in chain_ids[:3]:
            kwargs: dict[str, Any] = {"pdb_path": str(ref["path"]), "chain_id": chain_id}
            if residue_range and tool_name != "list_residues":
                kwargs["residue_range"] = residue_range
            result = registry.call_tool(tool_name, **kwargs)
            _append_tool_event(tool_events, result, {"structure_id": ref["id"], "chain_id": chain_id})
            _merge_tool_output(metrics, features, result, chain_id=chain_id)
            summaries.append(_short_text(result.data))

    success = any(event.get("success") for event in tool_events)
    analysis = runtime.project_store.save_analysis(
        project.id,
        kind="structure_analysis",
        query=str(args),
        status="completed" if success else "failed",
        selected_structure_id=str(ref["id"]),
        tool_events=tool_events,
        metrics=metrics,
        features=features,
        summary="\n\n".join(part for part in summaries if part)[:8000],
    )
    updated_project = runtime.project_store.set_selection(project.id, None, str(ref["id"]))
    return (
        updated_project,
        ToolResult(
            success=success,
            data=f"Completed structure analysis for {ref['pdb_id']} using {len(tool_events)} tool call(s).",
            raw={
                "status": "completed" if success else "failed",
                "analysis_id": analysis.id,
                "pdb_id": ref["pdb_id"],
                "selected_structure_id": ref["id"],
                "selected_job_id": None,
                "metrics": metrics,
                "features": features,
                "tool_events": tool_events,
            },
            error=None if success else "All analysis tools failed.",
            tool_name="analyze_structure",
        ),
    )


def _analyze_contacts(
    runtime: ProjectToolRuntime, project: ProjectRecord, args: dict[str, Any]
) -> tuple[ProjectRecord, ToolResult]:
    ref = _resolve_structure(runtime.project_store, project, args.get("structure_id_or_pdb_id"))
    if not ref:
        return _tool_error(project, "analyze_contacts", "No selected project structure is available.")
    chain_id = str(args.get("chain_id") or "").strip()
    residue_number = args.get("residue_number")
    if not chain_id or residue_number is None:
        return _tool_error(project, "analyze_contacts", "chain_id and residue_number are required.")
    result = _structure_registry().call_tool(
        "get_residue_contacts",
        pdb_path=str(ref["path"]),
        chain_id=chain_id,
        residue_number=int(residue_number),
        cutoff_angstroms=float(args.get("cutoff_angstroms") or 4.5),
    )
    tool_events: list[dict[str, Any]] = []
    _append_tool_event(tool_events, result, {"structure_id": ref["id"], "chain_id": chain_id})
    analysis = runtime.project_store.save_analysis(
        project.id,
        kind="contact_analysis",
        query=str(args),
        status="completed" if result.success else "failed",
        selected_structure_id=str(ref["id"]),
        tool_events=tool_events,
        summary=_short_text(result.data, 8000),
    )
    return (
        runtime.project_store.set_selection(project.id, None, str(ref["id"])),
        ToolResult(
            success=result.success,
            data=f"Completed contact analysis for {ref['pdb_id']} chain {chain_id} residue {residue_number}.",
            raw={
                "analysis_id": analysis.id,
                "selected_structure_id": ref["id"],
                "selected_job_id": None,
                "tool_events": tool_events,
            },
            error=result.error,
            tool_name="analyze_contacts",
        ),
    )


def _analyze_interface(
    runtime: ProjectToolRuntime, project: ProjectRecord, args: dict[str, Any]
) -> tuple[ProjectRecord, ToolResult]:
    ref = _resolve_structure(runtime.project_store, project, args.get("structure_id_or_pdb_id"))
    if not ref:
        return _tool_error(project, "analyze_interface", "No selected project structure is available.")
    chain_a = str(args.get("chain_a") or "").strip()
    chain_b = str(args.get("chain_b") or "").strip()
    if not chain_a or not chain_b:
        return _tool_error(project, "analyze_interface", "chain_a and chain_b are required.")
    result = _structure_registry().call_tool(
        "compute_interface",
        pdb_path=str(ref["path"]),
        chain_a=chain_a,
        chain_b=chain_b,
        distance_cutoff=float(args.get("distance_cutoff") or 5.0),
    )
    tool_events: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    features: dict[str, Any] = {}
    _append_tool_event(tool_events, result, {"structure_id": ref["id"], "chain_a": chain_a, "chain_b": chain_b})
    _merge_tool_output(metrics, features, result, chain_id=None)
    analysis = runtime.project_store.save_analysis(
        project.id,
        kind="interface_analysis",
        query=str(args),
        status="completed" if result.success else "failed",
        selected_structure_id=str(ref["id"]),
        tool_events=tool_events,
        metrics=metrics,
        features=features,
        summary=_short_text(result.data, 8000),
    )
    return (
        runtime.project_store.set_selection(project.id, None, str(ref["id"])),
        ToolResult(
            success=result.success,
            data=f"Completed interface analysis for {ref['pdb_id']} chains {chain_a}/{chain_b}.",
            raw={
                "analysis_id": analysis.id,
                "selected_structure_id": ref["id"],
                "selected_job_id": None,
                "metrics": metrics,
                "features": features,
                "tool_events": tool_events,
            },
            error=result.error,
            tool_name="analyze_interface",
        ),
    )


def _start_batch_from_project(
    runtime: ProjectToolRuntime, project: ProjectRecord, args: dict[str, Any]
) -> tuple[ProjectRecord, ToolResult]:
    if not runtime.job_store or not runtime.runner:
        return _tool_error(project, "start_batch_from_project", "Batch runtime is not configured.")

    requested_ids = args.get("structure_ids") if isinstance(args.get("structure_ids"), list) else []
    refs = [_resolve_structure(runtime.project_store, project, item) for item in requested_ids]
    refs = [ref for ref in refs if ref]
    if not refs:
        refs = [
            _resolve_structure(runtime.project_store, project, structure.id)
            for structure in project.structures
            if structure.id != "target"
        ]
        refs = [ref for ref in refs if ref]
    if not refs:
        return _tool_error(
            project,
            "start_batch_from_project",
            "No project candidate structures are available. Upload or generate candidates first.",
        )

    config = JobConfig(
        query=str(args.get("query") or "Rank candidate binders from project chat."),
        profile=str(args.get("profile") or "triage_default"),
        rank_by=str(args.get("rank_by") or "stability"),
        max_workers=max(1, int(args.get("max_workers") or 2)),
        enable_llm_synthesis=True,
    )
    record = runtime.job_store.create_job(config, project_id=project.id)
    saved = []
    for ref in refs:
        source = Path(ref["path"])
        target = _unique_copy_path(runtime.job_store.input_dir(record.id) / source.name)
        shutil.copy2(source, target)
        saved.append(target.name)
    record.input_files = saved
    runtime.job_store.write_record(record)
    updated_project = runtime.project_store.add_job(project.id, record.id)
    runtime.job_store.append_event(record.id, "uploads_saved", f"Saved {len(saved)} project structure input file(s).")
    if runtime.background_tasks is not None:
        runtime.background_tasks.add_task(runtime.runner.run_job, record.id, runtime.llm_api_key)
        status = "queued"
    else:
        runtime.runner.run_job(record.id, runtime.llm_api_key)
        status = runtime.job_store.get_record(record.id).status
    return (
        updated_project,
        ToolResult(
            success=True,
            data=f"Started batch job {record.id} over {len(saved)} project structure(s).",
            raw={
                "status": status,
                "job_id": record.id,
                "selected_job_id": record.id,
                "selected_structure_id": updated_project.selected_structure_id,
                "structure_count": len(saved),
            },
            tool_name="start_batch_from_project",
        ),
    )


def _generate_design_candidates(
    runtime: ProjectToolRuntime, project: ProjectRecord, args: dict[str, Any]
) -> tuple[ProjectRecord, ToolResult]:
    library = str(args.get("library") or "custom").strip().lower()
    target_ref = _resolve_structure(
        runtime.project_store,
        project,
        args.get("target_structure_id_or_pdb_id") or project.selected_structure_id or "target",
    )
    if not target_ref:
        return _tool_error(project, "generate_design_candidates", "Upload or load a target structure before design.")

    prompt = str(args.get("design_prompt") or "Generate candidate binders for the selected target.")
    num_designs = max(1, min(int(args.get("num_designs") or 8), int(os.getenv("MIRA_MAX_DESIGNS_PER_CHAT", "64"))))
    command_template = _design_command_template(library)
    run = runtime.project_store.create_design_run(
        project.id,
        library=library,
        prompt=prompt,
        target_structure_id=str(target_ref["id"]),
        output_dir=None,
        command=command_template,
        status="queued" if command_template else "configuration_required",
        error=None if command_template else _design_configuration_message(library),
    )
    output_dir = runtime.project_store.design_run_output_dir(project.id, run.id)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = _render_design_command(
        command_template,
        project=project,
        run_id=run.id,
        target_path=Path(target_ref["path"]),
        output_dir=output_dir,
        chain_id=str(args.get("chain_id") or ""),
        num_designs=num_designs,
        prompt=prompt,
    )
    runtime.project_store.update_design_run(project.id, run.id, output_dir=str(output_dir), command=command)

    if not command:
        return (
            project,
            ToolResult(
                success=True,
                data=_design_configuration_message(library),
                raw={
                    "status": "configuration_required",
                    "design_run_id": run.id,
                    "library": library,
                    "target_structure_id": target_ref["id"],
                },
                tool_name="generate_design_candidates",
            ),
        )

    if runtime.background_tasks is not None:
        runtime.background_tasks.add_task(
            _execute_design_command,
            runtime.project_store,
            project.id,
            run.id,
            command,
            output_dir,
        )
        data = f"Queued {library} design run {run.id} for {num_designs} candidate(s)."
        status = "queued"
    else:
        _execute_design_command(runtime.project_store, project.id, run.id, command, output_dir)
        status = runtime.project_store.update_design_run(project.id, run.id).status
        data = f"Completed {library} design run {run.id}."

    return (
        project,
        ToolResult(
            success=True,
            data=data,
            raw={
                "status": status,
                "design_run_id": run.id,
                "library": library,
                "target_structure_id": target_ref["id"],
                "output_dir": str(output_dir),
            },
            tool_name="generate_design_candidates",
        ),
    )


def _resolve_structure(
    project_store: ProjectStore, project: ProjectRecord, structure_id_or_pdb_id: object | None
) -> dict[str, Any] | None:
    requested = str(structure_id_or_pdb_id or project.selected_structure_id or "").strip()
    if not requested and project.target_file:
        requested = "target"
    identity = project_structure_identity(project, requested)
    if not identity:
        return None
    if identity["id"] == "target":
        path = project_store.target_path(project)
        filename = project.target_original_name or (path.name if path else "target.pdb")
    else:
        path = project_store.structure_path(project, str(identity["id"]))
        structure = next((item for item in project.structures if item.id == identity["id"]), None)
        filename = structure.original_name if structure else (path.name if path else str(identity["id"]))
    if not path or not path.exists():
        return None
    return {"id": identity["id"], "pdb_id": identity["pdb_id"], "path": path, "filename": filename}


def _requested_chain_ids(ref: dict[str, Any], chain_id: object | None) -> list[str]:
    if chain_id:
        return [str(chain_id)]
    try:
        result = _structure_registry().call_tool("load_structure", pdb_path=str(ref["path"]))
        chains = result.raw.get("chains") if isinstance(result.raw, dict) else []
        return [str(chain["id"]) for chain in chains[:3] if isinstance(chain, dict) and chain.get("id")]
    except Exception:
        return []


def _structure_registry() -> ToolRegistry:
    _import_structure_tools()
    return ToolRegistry()


def _import_structure_tools() -> None:
    from structagent.tools import bfactor as _bfactor  # noqa: F401
    from structagent.tools import charge as _charge  # noqa: F401
    from structagent.tools import contacts as _contacts  # noqa: F401
    from structagent.tools import interface as _interface  # noqa: F401
    from structagent.tools import ramachandran as _ramachandran  # noqa: F401
    from structagent.tools import sasa as _sasa  # noqa: F401
    from structagent.tools import secondary_structure as _secondary_structure  # noqa: F401
    from structagent.tools import structure_io as _structure_io  # noqa: F401


def _append_tool_event(events: list[dict[str, Any]], result: ToolResult, args: dict[str, Any]) -> None:
    events.append(
        {
            "tool": result.tool_name,
            "args": args,
            "success": result.success,
            "data": _short_text(result.data, 2500),
            "error": result.error,
            "raw": _compact_raw(result.raw),
        }
    )


def _merge_tool_output(
    metrics: dict[str, Any], features: dict[str, Any], result: ToolResult, chain_id: str | None
) -> None:
    if not result.success or not isinstance(result.raw, dict):
        return
    raw = result.raw
    tool = result.tool_name
    if tool == "load_structure":
        metrics["chain_count"] = len(raw.get("chains") or [])
        metrics["ligand_count"] = raw.get("ligand_count") or 0
        if raw.get("resolution"):
            metrics["resolution"] = raw.get("resolution")
    elif tool == "analyze_bfactors":
        stats = raw.get("statistics") or {}
        metrics["mean_bfactor"] = stats.get("mean")
        metrics["max_bfactor"] = stats.get("max")
        metrics["highly_flexible_count"] = (raw.get("classification_counts") or {}).get("highly_flexible")
        _extend_feature_list(
            features,
            "high_bfactor_residues",
            [
                _feature("high_bfactor_residues", chain_id, item.get("residue_number"), item.get("residue_name"), item)
                for item in raw.get("residues", [])
                if item.get("classification") == "highly_flexible" or item.get("potentially_disordered")
            ][:30],
        )
    elif tool == "compute_sasa":
        residues = raw.get("residues") or []
        metrics["n_buried"] = sum(1 for item in residues if item.get("classification") == "buried")
        metrics["n_exposed"] = sum(1 for item in residues if item.get("classification") == "exposed")
        if residues:
            metrics["mean_relative_sasa_percent"] = round(
                sum(float(item.get("relative_sasa_percent") or 0) for item in residues) / len(residues), 2
            )
        _extend_feature_list(
            features,
            "buried_residues",
            [
                _feature("buried_residues", chain_id, item.get("residue_number"), item.get("resname"), item)
                for item in residues
                if item.get("classification") == "buried"
            ][:30],
        )
        _extend_feature_list(
            features,
            "exposed_residues",
            [
                _feature("exposed_residues", chain_id, item.get("residue_number"), item.get("resname"), item)
                for item in residues
                if item.get("classification") == "exposed"
            ][:30],
        )
    elif tool == "compute_charge_distribution":
        metrics["net_charge"] = raw.get("total_charge")
        metrics["charge_cluster_count"] = raw.get("cluster_count")
        _extend_feature_list(features, "charge_clusters", raw.get("clusters") or [])
    elif tool == "check_ramachandran":
        metrics["ramachandran_outlier_pct"] = raw.get("outlier_pct")
        _extend_feature_list(
            features,
            "ramachandran_outliers",
            [
                _feature("ramachandran_outliers", chain_id, item.get("res_num"), item.get("res_name"), item)
                for item in raw.get("outliers", [])
            ],
        )
    elif tool == "compute_interface":
        for key in ("buried_surface_area", "n_interface_residues", "interface_residue_count"):
            if key in raw:
                metrics[key] = raw[key]
        for key in ("interface_residues", "hotspots"):
            if isinstance(raw.get(key), list):
                _extend_feature_list(features, key, raw[key])


def _extend_feature_list(features: dict[str, Any], key: str, values: list) -> None:
    current = features.setdefault(key, [])
    if isinstance(current, list):
        current.extend(values)
        features[key] = current[:60]


def _feature(kind: str, chain: str | None, residue_number: object, residue_name: object, raw: dict[str, Any]) -> dict:
    residue_label = f"{residue_name or 'Residue'}-{residue_number}"
    return {
        "kind": kind,
        "chain": chain,
        "residue_number": residue_number,
        "residue_name": residue_name,
        "label": residue_label,
        "score": raw.get("avg_bfactor") or raw.get("relative_sasa_percent") or raw.get("phi"),
    }


def _compact_raw(raw: object, max_items: int = 20) -> object:
    if isinstance(raw, dict):
        compact = {}
        for key, value in raw.items():
            if isinstance(value, list):
                compact[key] = value[:max_items]
            else:
                compact[key] = value
        return compact
    return raw


def _download_rcsb_cif(pdb_id: str) -> bytes:
    pdb_id = pdb_id.upper()
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    try:
        response = requests.get(url, timeout=30.0)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError(f"failed to download {pdb_id} from RCSB") from exc
    return response.content


def _execute_design_command(
    project_store: ProjectStore, project_id: str, run_id: str, command: str, output_dir: Path
) -> None:
    project_store.update_design_run(project_id, run_id, status="running")
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=int(os.getenv("MIRA_DESIGN_TIMEOUT_SECONDS", "3600")),
            check=False,
        )
        if completed.returncode != 0:
            project_store.update_design_run(
                project_id,
                run_id,
                status="failed",
                error=(completed.stderr or completed.stdout or f"Exited with {completed.returncode}")[-4000:],
            )
            return
        generated_ids = []
        for path in sorted(output_dir.rglob("*")):
            if path.suffix.lower() not in SUPPORTED_STRUCTURE_SUFFIXES or not path.is_file():
                continue
            _, structure = project_store.save_structure(project_id, path.name, path.read_bytes())
            generated_ids.append(structure.id)
        project_store.update_design_run(
            project_id,
            run_id,
            status="completed",
            generated_structure_ids=generated_ids,
            error=None,
        )
    except Exception as exc:
        project_store.update_design_run(project_id, run_id, status="failed", error=str(exc))


def _design_command_template(library: str) -> str | None:
    env_name = f"MIRA_DESIGN_{library.upper()}_COMMAND"
    return os.getenv(env_name) or os.getenv("MIRA_DESIGN_COMMAND")


def _render_design_command(
    template: str | None,
    *,
    project: ProjectRecord,
    run_id: str,
    target_path: Path,
    output_dir: Path,
    chain_id: str,
    num_designs: int,
    prompt: str,
) -> str | None:
    if not template:
        return None
    return template.format(
        project_id=shlex.quote(project.id),
        run_id=shlex.quote(run_id),
        target_path=shlex.quote(str(target_path)),
        output_dir=shlex.quote(str(output_dir)),
        chain_id=shlex.quote(chain_id),
        num_designs=num_designs,
        prompt=shlex.quote(prompt),
    )


def _design_configuration_message(library: str) -> str:
    return (
        f"{library} design is not configured on this server. Set "
        f"MIRA_DESIGN_{library.upper()}_COMMAND or MIRA_DESIGN_COMMAND with placeholders "
        "{target_path}, {output_dir}, {num_designs}, {chain_id}, and {prompt}."
    )


def _tool_error(project: ProjectRecord, tool_name: str, message: str) -> tuple[ProjectRecord, ToolResult]:
    return (
        project,
        ToolResult(success=False, data=message, raw={"status": "failed"}, error=message, tool_name=tool_name),
    )


def _unique_copy_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise ValueError(f"Could not create unique filename for {path.name}")


def _short_text(value: object, limit: int = 1200) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _pdb_id_from_name(filename: str) -> str:
    return Path(filename).stem.upper()
