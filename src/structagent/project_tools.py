"""Constrained project tools for hosted chat workflows."""

from __future__ import annotations

import os
import re
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from structagent.design_adapters import DesignRequest, execute_design, prepare_design
from structagent.jobs.models import JobConfig
from structagent.jobs.runner import JobRunner
from structagent.jobs.store import JobStore
from structagent.projects import ProjectRecord, ProjectStore
from structagent.registry import ToolRegistry, ToolResult
from structagent.tool_metadata import TOOL_SCHEMAS


PDB_ID_PATTERN = re.compile(r"\b([0-9][A-Za-z0-9]{3})\b")
SUPPORTED_STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif"}


@dataclass
class ProjectToolRuntime:
    project_store: ProjectStore
    job_store: JobStore | None = None
    runner: JobRunner | None = None
    background_tasks: Any = None
    llm_api_key: str | None = None


_PROJECT_NATIVE_TOOL_SCHEMAS: list[dict[str, Any]] = [
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
        "name": "identify_hotspots",
        "description": (
            "Identify likely binder-design hotspot residues on a selected target surface. "
            "Use this when the user asks to find hotspots, epitopes, binding patches, or promising design regions."
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
                    "description": "Optional chain to analyze. If omitted, MIRA scans up to four protein chains.",
                },
                "min_relative_sasa_percent": {
                    "type": "number",
                    "default": 45.0,
                    "description": "Minimum relative SASA percent for hotspot candidates.",
                },
                "max_residues": {
                    "type": "integer",
                    "default": 12,
                    "description": "Maximum number of hotspot residues to keep as clickable evidence.",
                },
            },
            "required": [],
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
            "Invoke a configured generative design library such as FoldingDiff, BindCraft, RFdiffusion, "
            "ProteinMPNN, or LigandMPNN. Use FoldingDiff for CPU de novo backbone/structure generation. "
            "If the library command is not configured, create a design setup record explaining what is missing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "library": {
                    "type": "string",
                    "enum": ["foldingdiff", "bindcraft", "rfdiffusion", "proteinmpnn", "ligandmpnn", "custom"],
                    "default": "custom",
                },
                "target_structure_id_or_pdb_id": {
                    "type": "string",
                    "description": (
                        "Target structure id, 'target', or PDB/file stem. Defaults to selected/target. "
                        "Not required for unconditional FoldingDiff backbone generation."
                    ),
                },
                "chain_id": {"type": "string", "description": "Optional target chain or receptor chain."},
                "num_designs": {"type": "integer", "default": 8},
                "length": {
                    "type": "integer",
                    "default": 80,
                    "description": "Approximate residue length for unconditional FoldingDiff backbone generation.",
                },
                "temperature": {"type": "string", "default": "0.1"},
                "seed": {"type": "integer", "default": 0},
                "contigs": {
                    "type": "string",
                    "description": "RFdiffusion contig map such as '[A1-100/0 80-120]'.",
                },
                "hotspot_residues": {
                    "type": "string",
                    "description": "RFdiffusion hotspot residues such as '[A45,A67]'.",
                },
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


_PROJECT_NATIVE_TOOL_NAMES = {schema["name"] for schema in _PROJECT_NATIVE_TOOL_SCHEMAS}
_REGISTRY_PROJECT_TOOL_NAMES = {schema["name"] for schema in TOOL_SCHEMAS}


def _project_chat_tool_schemas() -> list[dict[str, Any]]:
    schemas = list(_PROJECT_NATIVE_TOOL_SCHEMAS)
    schemas.extend(_project_wrapped_registry_schema(schema) for schema in TOOL_SCHEMAS)
    return schemas


def _project_wrapped_registry_schema(schema: dict[str, Any]) -> dict[str, Any]:
    wrapped = deepcopy(schema)
    params = wrapped.setdefault("parameters", {"type": "object", "properties": {}, "required": []})
    params.setdefault("type", "object")
    properties = params.setdefault("properties", {})
    params.setdefault("required", [])
    params["additionalProperties"] = False
    if "pdb_path" in properties or "input_path" in properties:
        properties.setdefault(
            "structure_id_or_pdb_id",
            {
                "type": "string",
                "description": "Project structure id, 'target', or visible file/PDB stem. Defaults to selected project structure.",
            },
        )
    if "pdb_path_1" in properties:
        properties.setdefault(
            "structure_id_or_pdb_id_1",
            {
                "type": "string",
                "description": "First project structure id, 'target', or file/PDB stem. Defaults to selected project structure.",
            },
        )
    if "pdb_path_2" in properties:
        properties.setdefault(
            "structure_id_or_pdb_id_2",
            {
                "type": "string",
                "description": "Second project structure id, 'target', or file/PDB stem.",
            },
        )
    wrapped["description"] = (
        f"{wrapped.get('description', '')} In project chat, omit local path fields to use the selected project structure."
    )
    return wrapped


PROJECT_CHAT_TOOL_SCHEMAS: list[dict[str, Any]] = _project_chat_tool_schemas()


def message_may_need_project_tool(message: str) -> bool:
    lowered = message.lower()
    if message_is_results_status_query(message):
        return bool(extract_pdb_id(message))
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
        "hotspot",
        "hotspots",
        "epitope",
        "epitopes",
        "binding patch",
        "binding site",
        "residue",
        "ramachandran",
        "batch",
        "rank",
        "screen",
        "candidate",
        "binder",
        "design",
        "generate",
        "foldingdiff",
        "folding diff",
        "backbone",
        "de novo",
        "rfdiffusion",
        "bindcraft",
        "proteinmpnn",
        "ligandmpnn",
        "secondary structure",
        "conservation",
        "annotation",
        "functional",
        "homolog",
        "foldseek",
        "align",
        "alignment",
        "rmsd",
        "renumber",
        "relax",
        "interface energy",
        "score interface",
        "normal mode",
        "cross correlation",
        "dynamics",
        "hinge",
        "perturbation",
        "allosteric",
    )
    return bool(PDB_ID_PATTERN.search(message)) or any(word in lowered for word in action_words)


def message_is_results_status_query(message: str) -> bool:
    lowered = message.lower()
    status_words = (
        "result",
        "results",
        "status",
        "progress",
        "finished",
        "complete",
        "completed",
        "done",
        "what happened",
        "what did",
        "how did",
        "generated",
        "generations",
        "sequence design",
        "sequence designs",
    )
    mutating_phrases = (
        "generate",
        "design me",
        "make ",
        "create ",
        "start ",
        "run ",
        "launch ",
        "redesign ",
        "design a",
        "design new",
    )
    if not any(word in lowered for word in status_words):
        return False
    return not any(phrase in lowered for phrase in mutating_phrases)


def fallback_project_tool_calls(message: str) -> list[dict[str, Any]]:
    lowered = message.lower()
    pdb_id = extract_pdb_id(message)
    calls: list[dict[str, Any]] = []
    if pdb_id:
        calls.append({"tool": "load_pdb_id", "args": {"pdb_id": pdb_id}, "purpose": "Detected explicit PDB ID."})
    if message_is_results_status_query(message):
        return calls[:1]
    direct_tool = _fallback_registry_tool_name(lowered)
    if (
        any(word in lowered for word in ("analyze", "analyse", "flexible", "sasa", "charge", "ramachandran"))
        and not direct_tool
    ):
        calls.append({"tool": "analyze_structure", "args": {}, "purpose": "Detected target-analysis request."})
    if any(word in lowered for word in ("batch", "rank", "screen")):
        calls.append({"tool": "start_batch_from_project", "args": {}, "purpose": "Detected batch-screening request."})
    if direct_tool:
        args = _fallback_registry_args(message, direct_tool)
        calls.append({"tool": direct_tool, "args": args, "purpose": f"Detected {direct_tool} request."})
    if _message_requests_generation(lowered):
        library = "proteinmpnn"
        if "foldingdiff" in lowered or "folding diff" in lowered or _message_requests_backbone_generation(lowered):
            library = "foldingdiff"
        elif "bindcraft" in lowered:
            library = "bindcraft"
        elif "rfdiffusion" in lowered or "rf diffusion" in lowered:
            library = "rfdiffusion"
        elif "ligandmpnn" in lowered or "ligand mpnn" in lowered:
            library = "ligandmpnn"
        elif "proteinmpnn" in lowered or "protein mpnn" in lowered:
            library = "proteinmpnn"
        generation_args: dict[str, Any] = {"library": library, "design_prompt": message}
        requested_count = _extract_requested_design_count(lowered)
        requested_length = _extract_requested_design_length(lowered)
        if requested_count:
            generation_args["num_designs"] = requested_count
        if requested_length and library == "foldingdiff":
            generation_args["length"] = requested_length
        calls.append(
            {
                "tool": "generate_design_candidates",
                "args": generation_args,
                "purpose": "Detected candidate design request.",
            }
        )
    return calls[:6]


def _fallback_registry_tool_name(lowered: str) -> str | None:
    phrase_map = [
        (("hotspot", "hotspots", "epitope", "epitopes", "binding patch", "binding site"), "identify_hotspots"),
        (("secondary structure",), "get_secondary_structure"),
        (("conservation", "conserved"), "get_conservation_scores"),
        (("annotation", "functional"), "get_functional_annotations"),
        (("homolog", "foldseek"), "search_structural_homologs"),
        (("align", "alignment", "rmsd"), "align_structures"),
        (("renumber",), "renumber_pdb"),
        (("relax",), "fast_relax"),
        (("interface energy", "interface energies"), "analyze_interface_energies"),
        (("score interface", "interface score"), "score_interface"),
        (("normal mode", "normal modes"), "compute_normal_modes"),
        (("cross correlation", "cross correlations"), "compute_cross_correlations"),
        (("hinge",), "predict_hinge_regions"),
        (("perturbation", "allosteric"), "compute_perturbation_response"),
    ]
    for phrases, tool_name in phrase_map:
        if any(phrase in lowered for phrase in phrases):
            return tool_name
    return None


def _fallback_registry_args(message: str, tool_name: str) -> dict[str, Any]:
    args: dict[str, Any] = {}
    chain = _extract_chain_id(message)
    if chain and tool_name == "identify_hotspots":
        args["chain_id"] = chain
    if chain and _tool_accepts_parameter(tool_name, "chain_id"):
        args["chain_id"] = chain
    if chain and tool_name == "analyze_interface_energies":
        args["binder_chain"] = chain
    if chain and tool_name == "score_interface":
        args["binder_chains"] = chain
    residue = _extract_residue_number(message)
    if residue is not None and tool_name == "compute_perturbation_response":
        args["source_residue"] = residue
    return args


def _message_requests_generation(lowered: str) -> bool:
    explicit_library = any(
        word in lowered
        for word in (
            "generate",
            "foldingdiff",
            "folding diff",
            "rfdiffusion",
            "rf diffusion",
            "bindcraft",
            "proteinmpnn",
            "ligandmpnn",
        )
    )
    explicit_creation = any(
        phrase in lowered
        for phrase in (
            "design me",
            "design a",
            "design an",
            "design new",
            "design candidate",
            "design candidates",
            "design binder",
            "design binders",
            "redesign",
            "make ",
            "create ",
        )
    )
    counted_design = re.search(
        r"\bdesign\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        lowered,
    )
    return (
        explicit_library or explicit_creation or bool(counted_design) or _message_requests_backbone_generation(lowered)
    )


def _message_requests_backbone_generation(lowered: str) -> bool:
    return any(
        phrase in lowered
        for phrase in (
            "backbone design",
            "backbone designs",
            "backbone generation",
            "generate backbone",
            "generate backbones",
            "generate structure",
            "generate structures",
            "structure design",
            "structure designs",
            "de novo backbone",
            "de novo structure",
            "new backbone",
            "new structure",
        )
    )


def _extract_requested_design_count(lowered: str) -> int | None:
    match = re.search(
        r"\b(?:generate|make|create|design|redesign)\s+"
        r"(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        lowered,
    )
    if not match:
        return None
    return _small_number(match.group("count"))


def _extract_requested_design_length(lowered: str) -> int | None:
    patterns = [
        r"\b(?P<length>\d{2,4})\s*(?:aa|residue|residues|amino acid|amino acids)\b",
        r"\blength\s+(?:of\s+)?(?P<length>\d{2,4})\b",
        r"\b(?P<length>\d{2,4})\s*(?:mer|mers)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return int(match.group("length"))
    return None


def _small_number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    return words.get(value)


def _tool_accepts_parameter(tool_name: str, parameter: str) -> bool:
    schema = next((schema for schema in TOOL_SCHEMAS if schema["name"] == tool_name), None)
    properties = ((schema or {}).get("parameters") or {}).get("properties") or {}
    return parameter in properties


def _extract_chain_id(message: str) -> str | None:
    match = re.search(r"\bchain\s+([A-Za-z0-9])\b", message, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def _extract_residue_number(message: str) -> int | None:
    match = re.search(r"\b(?:residue|res|position)\s+(\d+)\b", message, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


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
    if tool_name == "identify_hotspots":
        return _identify_hotspots(runtime, project, args)
    if tool_name == "analyze_interface":
        return _analyze_interface(runtime, project, args)
    if tool_name == "start_batch_from_project":
        return _start_batch_from_project(runtime, project, args)
    if tool_name == "generate_design_candidates":
        return _generate_design_candidates(runtime, project, args)
    if tool_name in _REGISTRY_PROJECT_TOOL_NAMES:
        return _run_registry_project_tool(runtime, project, tool_name, args)
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


def _identify_hotspots(
    runtime: ProjectToolRuntime, project: ProjectRecord, args: dict[str, Any]
) -> tuple[ProjectRecord, ToolResult]:
    ref = _resolve_structure(runtime.project_store, project, args.get("structure_id_or_pdb_id"))
    if not ref:
        return _tool_error(project, "identify_hotspots", "No selected project structure is available.")

    registry = _structure_registry()
    chain_filter = str(args.get("chain_id") or "").strip() or None
    min_relative_sasa = _bounded_float(args.get("min_relative_sasa_percent"), default=45.0, low=0.0, high=100.0)
    max_residues = int(_bounded_float(args.get("max_residues"), default=12.0, low=1.0, high=50.0))

    tool_events: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    features: dict[str, Any] = {}
    summaries: list[str] = []

    load_result = registry.call_tool("load_structure", pdb_path=str(ref["path"]))
    _append_tool_event(tool_events, load_result, {"structure_id": ref["id"]})
    if not load_result.success:
        analysis = runtime.project_store.save_analysis(
            project.id,
            kind="hotspot_analysis",
            query=str(args),
            status="failed",
            selected_structure_id=str(ref["id"]),
            tool_events=tool_events,
            summary=_short_text(load_result.data, 8000),
        )
        return (
            project,
            ToolResult(
                success=False,
                data=f"Could not load {ref['pdb_id']} for hotspot analysis.",
                raw={"status": "failed", "analysis_id": analysis.id, "tool_events": tool_events},
                error=load_result.error,
                tool_name="identify_hotspots",
            ),
        )

    chains = _chains_from_load_result(load_result, chain_filter)
    hotspot_candidates: list[dict[str, Any]] = []

    for chain_id in chains[:4]:
        sasa_result = registry.call_tool("compute_sasa", pdb_path=str(ref["path"]), chain_id=chain_id)
        _append_tool_event(tool_events, sasa_result, {"structure_id": ref["id"], "chain_id": chain_id})
        _merge_tool_output(metrics, features, sasa_result, chain_id=chain_id)
        if not sasa_result.success:
            summaries.append(_short_text(sasa_result.data))
            continue

        bfactor_result = registry.call_tool("analyze_bfactors", pdb_path=str(ref["path"]), chain_id=chain_id)
        _append_tool_event(tool_events, bfactor_result, {"structure_id": ref["id"], "chain_id": chain_id})
        _merge_tool_output(metrics, features, bfactor_result, chain_id=chain_id)
        summaries.append(_short_text(sasa_result.data))
        summaries.append(_short_text(bfactor_result.data))

        bfactor_by_residue = _bfactor_by_residue(bfactor_result)
        for residue in _raw_residue_list(sasa_result):
            residue_number = residue.get("residue_number")
            if residue_number is None:
                continue
            relative_sasa = _as_float(residue.get("relative_sasa_percent"))
            if relative_sasa is None or relative_sasa < min_relative_sasa:
                continue
            resname = str(residue.get("resname") or residue.get("residue_name") or "Residue")
            bfactor = bfactor_by_residue.get(int(residue_number))
            bfactor_class = str((bfactor or {}).get("classification") or "unknown")
            chemistry = _hotspot_residue_class(resname)
            score = _hotspot_candidate_score(relative_sasa, bfactor_class, chemistry)
            hotspot_candidates.append(
                {
                    "kind": "hotspots",
                    "chain": chain_id,
                    "residue_number": int(residue_number),
                    "residue_name": resname,
                    "label": f"{resname}-{residue_number} chain {chain_id}",
                    "score": score,
                    "relative_sasa_percent": round(relative_sasa, 1),
                    "absolute_sasa": residue.get("absolute_sasa"),
                    "surface_classification": residue.get("classification"),
                    "avg_bfactor": (bfactor or {}).get("avg_bfactor"),
                    "bfactor_classification": bfactor_class,
                    "chemistry": chemistry,
                }
            )

    hotspot_candidates.sort(
        key=lambda item: (float(item.get("score") or 0), float(item.get("relative_sasa_percent") or 0)), reverse=True
    )
    top_hotspots = hotspot_candidates[:max_residues]
    features["hotspots"] = top_hotspots
    metrics["hotspot_count"] = len(top_hotspots)
    if top_hotspots:
        metrics["top_hotspot_score"] = top_hotspots[0]["score"]
        metrics["mean_hotspot_relative_sasa_percent"] = round(
            sum(float(item.get("relative_sasa_percent") or 0) for item in top_hotspots) / len(top_hotspots),
            2,
        )

    summary = _hotspot_summary(ref["pdb_id"], top_hotspots, min_relative_sasa)
    if summaries:
        summary = f"{summary}\n\nSupporting local analyses:\n" + "\n\n".join(part for part in summaries if part)[:5000]
    success = bool(top_hotspots) or any(event.get("success") for event in tool_events)
    analysis = runtime.project_store.save_analysis(
        project.id,
        kind="hotspot_analysis",
        query=str(args),
        status="completed" if success else "failed",
        selected_structure_id=str(ref["id"]),
        tool_events=tool_events,
        metrics=metrics,
        features=features,
        summary=summary[:8000],
    )
    updated_project = runtime.project_store.set_selection(project.id, None, str(ref["id"]))
    return (
        updated_project,
        ToolResult(
            success=success,
            data=summary,
            raw={
                "status": "completed" if success else "failed",
                "analysis_id": analysis.id,
                "pdb_id": ref["pdb_id"],
                "selected_structure_id": ref["id"],
                "selected_job_id": None,
                "metrics": metrics,
                "features": {"hotspots": top_hotspots},
                "hotspots": top_hotspots,
                "tool_events": tool_events,
            },
            error=None if success else "No hotspot candidates could be scored.",
            tool_name="identify_hotspots",
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


def _run_registry_project_tool(
    runtime: ProjectToolRuntime, project: ProjectRecord, tool_name: str, args: dict[str, Any]
) -> tuple[ProjectRecord, ToolResult]:
    registry = _structure_registry()
    schema = next((schema for schema in TOOL_SCHEMAS if schema["name"] == tool_name), None)
    if not schema:
        return _tool_error(project, tool_name, f"Unknown registry tool: {tool_name}")

    kwargs, selected_ref, setup_error = _registry_tool_kwargs(runtime.project_store, project, schema, args)
    if setup_error:
        return _tool_error(project, tool_name, setup_error)

    result = registry.call_tool(tool_name, **kwargs)
    tool_events: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    features: dict[str, Any] = {}
    selected_structure_id = str(selected_ref["id"]) if selected_ref else project.selected_structure_id
    selected_pdb_id = str(selected_ref["pdb_id"]) if selected_ref else str(kwargs.get("pdb_id") or "")
    _append_tool_event(tool_events, result, {"tool_name": tool_name, **_redacted_registry_args(kwargs)})
    _merge_tool_output(metrics, features, result, chain_id=str(kwargs.get("chain_id") or "") or None)

    generated_structure_ids = []
    raw = result.raw if isinstance(result.raw, dict) else {}
    output_path = raw.get("output_path") or raw.get("relaxed_path")
    if isinstance(output_path, str):
        maybe_path = Path(output_path)
        if maybe_path.exists() and maybe_path.suffix.lower() in SUPPORTED_STRUCTURE_SUFFIXES:
            _, structure = runtime.project_store.save_structure(project.id, maybe_path.name, maybe_path.read_bytes())
            generated_structure_ids.append(structure.id)

    analysis = runtime.project_store.save_analysis(
        project.id,
        kind=f"tool_{tool_name}",
        query=str({"tool": tool_name, "args": _redacted_registry_args(kwargs)}),
        status="completed" if result.success else "failed",
        selected_structure_id=selected_structure_id,
        tool_events=tool_events,
        metrics=metrics,
        features=features,
        summary=_short_text(result.data, 8000),
    )
    updated_project = (
        runtime.project_store.set_selection(project.id, None, selected_structure_id)
        if selected_structure_id
        else project
    )
    return (
        updated_project,
        ToolResult(
            success=result.success,
            data=f"Ran `{tool_name}` for {selected_pdb_id or 'the selected project context'}.",
            raw={
                "status": "completed" if result.success else "failed",
                "analysis_id": analysis.id,
                "registry_tool": tool_name,
                "pdb_id": selected_pdb_id or kwargs.get("pdb_id"),
                "selected_structure_id": selected_structure_id,
                "selected_job_id": None,
                "metrics": metrics,
                "features": features,
                "generated_structure_ids": generated_structure_ids,
                "tool_events": tool_events,
            },
            error=result.error,
            tool_name=tool_name,
        ),
    )


def _registry_tool_kwargs(
    project_store: ProjectStore, project: ProjectRecord, schema: dict[str, Any], args: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    tool_name = schema["name"]
    properties = (schema.get("parameters") or {}).get("properties") or {}
    allowed = set(properties)
    internal_keys = {"structure_id_or_pdb_id", "structure_id_or_pdb_id_1", "structure_id_or_pdb_id_2"}
    kwargs = {
        key: value
        for key, value in args.items()
        if key in allowed and key not in internal_keys and value is not None and value != ""
    }
    for path_key in ("pdb_path", "input_path", "pdb_path_1", "pdb_path_2", "plot_output"):
        kwargs.pop(path_key, None)

    selected_ref = _resolve_structure(project_store, project, args.get("structure_id_or_pdb_id"))

    if "pdb_path" in properties:
        if selected_ref:
            kwargs["pdb_path"] = str(selected_ref["path"])
            kwargs.pop("pdb_id", None)
        elif not kwargs.get("pdb_path") and not kwargs.get("pdb_id"):
            return kwargs, selected_ref, "Select or upload a project structure before running this tool."
    if "input_path" in properties:
        if selected_ref:
            kwargs["input_path"] = str(selected_ref["path"])
        elif not kwargs.get("input_path"):
            return kwargs, selected_ref, "Select or upload a project structure before running this tool."

    if tool_name == "align_structures":
        first_ref = _resolve_structure(project_store, project, args.get("structure_id_or_pdb_id_1")) or selected_ref
        second_ref = _resolve_structure(project_store, project, args.get("structure_id_or_pdb_id_2"))
        if first_ref:
            kwargs["pdb_path_1"] = str(first_ref["path"])
            kwargs.pop("pdb_id_1", None)
            selected_ref = first_ref
        if second_ref:
            kwargs["pdb_path_2"] = str(second_ref["path"])
            kwargs.pop("pdb_id_2", None)
        if not (kwargs.get("pdb_path_1") or kwargs.get("pdb_id_1")):
            return kwargs, selected_ref, "Alignment needs a first structure."
        if not (kwargs.get("pdb_path_2") or kwargs.get("pdb_id_2")):
            return kwargs, selected_ref, "Alignment needs a second structure or PDB ID."

    if "plot_output" in properties and "plot_output" not in kwargs:
        output_dir = project_store.analysis_dir(project.id) / "tool_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        kwargs["plot_output"] = str(output_dir / f"{tool_name}_{os.getpid()}.png")

    if tool_name in {"get_functional_annotations", "get_conservation_scores", "search_structural_homologs"}:
        if not kwargs.get("pdb_id") and selected_ref:
            pdb_id = str(selected_ref.get("pdb_id") or "")
            if PDB_ID_PATTERN.fullmatch(pdb_id):
                kwargs["pdb_id"] = pdb_id
        if not kwargs.get("pdb_id"):
            return kwargs, selected_ref, f"`{tool_name}` requires an RCSB PDB ID."

    if tool_name == "renumber_pdb" and "pdb_path" not in kwargs:
        return kwargs, selected_ref, "Renumbering needs a selected local PDB project structure."

    return kwargs, selected_ref, None


def _redacted_registry_args(kwargs: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(kwargs)
    for key in list(redacted):
        if key.endswith("_path") or key in {"pdb_path", "input_path", "plot_output"}:
            redacted[key] = Path(str(redacted[key])).name
    return redacted


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
    library = str(args.get("library") or os.getenv("MIRA_DEFAULT_DESIGN_LIBRARY") or "proteinmpnn").strip().lower()
    target_optional = library in {"foldingdiff"}
    target_ref = _resolve_structure(
        runtime.project_store,
        project,
        args.get("target_structure_id_or_pdb_id") or project.selected_structure_id or "target",
    )
    if not target_ref and not target_optional:
        return _tool_error(project, "generate_design_candidates", "Upload or load a target structure before design.")

    prompt = str(args.get("design_prompt") or "Generate candidate binders for the selected target.")
    max_designs = int(os.getenv("MIRA_MAX_DESIGNS_PER_CHAT", "64"))
    if library == "foldingdiff":
        max_designs = min(max_designs, int(os.getenv("MIRA_FOLDINGDIFF_MAX_DESIGNS", "8")))
    num_designs = max(1, min(int(args.get("num_designs") or 8), max_designs))
    run = runtime.project_store.create_design_run(
        project.id,
        library=library,
        prompt=prompt,
        target_structure_id=str(target_ref["id"]) if target_ref else None,
        output_dir=None,
        command=None,
        num_designs=num_designs,
        parameters={},
        status="preparing",
        error=None,
    )
    output_dir = runtime.project_store.design_run_output_dir(project.id, run.id)
    output_dir.mkdir(parents=True, exist_ok=True)

    design_request = DesignRequest(
        library=library,
        target_path=Path(target_ref["path"]) if target_ref else None,
        output_dir=output_dir,
        project_id=project.id,
        run_id=run.id,
        prompt=prompt,
        chain_id=str(args.get("chain_id") or ""),
        num_designs=num_designs,
        seed=int(args.get("seed") or 0),
        temperature=str(args.get("temperature") or "0.1"),
        extra_args={
            key: value
            for key, value in args.items()
            if key
            not in {
                "library",
                "target_structure_id_or_pdb_id",
                "chain_id",
                "num_designs",
                "seed",
                "temperature",
                "design_prompt",
            }
        },
    )
    prepared = prepare_design(design_request)
    runtime.project_store.update_design_run(
        project.id,
        run.id,
        output_dir=str(output_dir),
        command=prepared.command,
        parameters=prepared.parameters,
        status=prepared.status,
        error=prepared.error,
    )

    if prepared.status == "configuration_required":
        return (
            project,
            ToolResult(
                success=True,
                data=prepared.error or f"{library} design backend is not configured.",
                raw={
                    "status": "configuration_required",
                    "design_run_id": run.id,
                    "library": library,
                    "backend": prepared.parameters.get("backend"),
                    "target_structure_id": target_ref["id"] if target_ref else None,
                    "num_designs": num_designs,
                },
                tool_name="generate_design_candidates",
            ),
        )

    if runtime.background_tasks is not None:
        runtime.background_tasks.add_task(
            _execute_design_adapter,
            runtime.project_store,
            project.id,
            run.id,
            output_dir,
        )
        data = f"Queued real {library} design run {run.id} for {num_designs} candidate(s)."
        status = "queued"
    else:
        _execute_design_adapter(runtime.project_store, project.id, run.id, output_dir)
        status = runtime.project_store.update_design_run(project.id, run.id).status
        data = f"Completed real {library} design run {run.id}."

    return (
        project,
        ToolResult(
            success=True,
            data=data,
            raw={
                "status": status,
                "design_run_id": run.id,
                "library": library,
                "backend": prepared.parameters.get("backend"),
                "target_structure_id": target_ref["id"] if target_ref else None,
                "output_dir": str(output_dir),
                "num_designs": num_designs,
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
    from structagent.tools import alignment as _alignment  # noqa: F401
    from structagent.tools import annotations as _annotations  # noqa: F401
    from structagent.tools import bfactor as _bfactor  # noqa: F401
    from structagent.tools import charge as _charge  # noqa: F401
    from structagent.tools import conservation as _conservation  # noqa: F401
    from structagent.tools import contacts as _contacts  # noqa: F401
    from structagent.tools import dynamics as _dynamics  # noqa: F401
    from structagent.tools import foldseek as _foldseek  # noqa: F401
    from structagent.tools import interface as _interface  # noqa: F401
    from structagent.tools import interface_energy as _interface_energy  # noqa: F401
    from structagent.tools import pyrosetta_interface as _pyrosetta_interface  # noqa: F401
    from structagent.tools import ramachandran as _ramachandran  # noqa: F401
    from structagent.tools import relaxation as _relaxation  # noqa: F401
    from structagent.tools import renumber_pdb as _renumber_pdb  # noqa: F401
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


def _chains_from_load_result(load_result: ToolResult, chain_filter: str | None) -> list[str]:
    if chain_filter:
        return [chain_filter]
    raw = load_result.raw if isinstance(load_result.raw, dict) else {}
    chains = raw.get("chains") if isinstance(raw.get("chains"), list) else []
    chain_ids = [str(chain.get("id")) for chain in chains if isinstance(chain, dict) and chain.get("id")]
    return chain_ids or ["A"]


def _raw_residue_list(result: ToolResult) -> list[dict[str, Any]]:
    raw = result.raw if isinstance(result.raw, dict) else {}
    residues = raw.get("residues")
    return [item for item in residues if isinstance(item, dict)] if isinstance(residues, list) else []


def _bfactor_by_residue(result: ToolResult) -> dict[int, dict[str, Any]]:
    by_residue: dict[int, dict[str, Any]] = {}
    for residue in _raw_residue_list(result):
        residue_number = residue.get("residue_number")
        if residue_number is None:
            continue
        try:
            by_residue[int(residue_number)] = residue
        except (TypeError, ValueError):
            continue
    return by_residue


def _as_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bounded_float(value: object, *, default: float, low: float, high: float) -> float:
    parsed = _as_float(value)
    if parsed is None:
        parsed = default
    return min(max(parsed, low), high)


def _hotspot_residue_class(resname: str) -> str:
    residue = resname.upper()
    if residue in {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TYR", "PRO"}:
        return "hydrophobic"
    if residue in {"LYS", "ARG", "HIS", "ASP", "GLU"}:
        return "charged"
    if residue in {"SER", "THR", "ASN", "GLN", "CYS"}:
        return "polar"
    return "mixed"


def _hotspot_candidate_score(relative_sasa: float, bfactor_class: str, chemistry: str) -> float:
    flexibility_weight = {
        "rigid": 0.72,
        "ordered": 1.0,
        "flexible": 0.88,
        "highly_flexible": 0.52,
        "unknown": 0.82,
    }.get(bfactor_class, 0.82)
    chemistry_weight = {
        "hydrophobic": 1.0,
        "charged": 0.92,
        "polar": 0.82,
        "mixed": 0.74,
    }.get(chemistry, 0.74)
    return round(relative_sasa * flexibility_weight * chemistry_weight, 2)


def _hotspot_summary(pdb_id: str, hotspots: list[dict[str, Any]], min_relative_sasa: float) -> str:
    if not hotspots:
        return (
            f"Hotspot analysis completed for `{pdb_id}`, but no surface residues exceeded "
            f"the {min_relative_sasa:.0f}% relative SASA threshold."
        )
    lines = [
        f"Identified `{len(hotspots)}` hotspot candidate residue(s) on `{pdb_id}`.",
        (
            "These are exposed residues scored from relative SASA, local B-factor class, "
            "and residue chemistry; treat them as binder-design epitope candidates."
        ),
        "Top candidates:",
    ]
    for hotspot in hotspots[:8]:
        lines.append(
            "- "
            f"{hotspot.get('residue_name')}-{hotspot.get('residue_number')} chain {hotspot.get('chain')}: "
            f"score {_fmt_number(hotspot.get('score'))}, "
            f"{_fmt_number(hotspot.get('relative_sasa_percent'))}% relative SASA, "
            f"{hotspot.get('chemistry')} chemistry, "
            f"{hotspot.get('bfactor_classification')} B-factor class."
        )
    return "\n".join(lines)


def _fmt_number(value: object) -> str:
    parsed = _as_float(value)
    return f"{parsed:.1f}" if parsed is not None else "n/a"


def _download_rcsb_cif(pdb_id: str) -> bytes:
    pdb_id = pdb_id.upper()
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    try:
        response = requests.get(url, timeout=30.0)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError(f"failed to download {pdb_id} from RCSB") from exc
    return response.content


def _execute_design_adapter(project_store: ProjectStore, project_id: str, run_id: str, output_dir: Path) -> None:
    project_store.update_design_run(project_id, run_id, status="running")
    try:
        run = project_store.update_design_run(project_id, run_id)
        result = execute_design(run.parameters, output_dir)
        generated_ids = []
        for path in result.generated_structure_paths:
            if path.suffix.lower() not in SUPPORTED_STRUCTURE_SUFFIXES or not path.is_file():
                continue
            _, structure = project_store.save_structure(project_id, path.name, path.read_bytes())
            generated_ids.append(structure.id)
        if not result.success:
            updated_run = project_store.update_design_run(
                project_id,
                run_id,
                status=result.status,
                generated_structure_ids=generated_ids,
                generated_sequences=result.generated_sequences,
                artifacts=result.artifacts,
                logs=result.logs,
                error=result.error,
            )
            _append_design_completion_message(project_store, project_id, updated_run)
            return
        updated_run = project_store.update_design_run(
            project_id,
            run_id,
            status="completed",
            generated_structure_ids=generated_ids,
            generated_sequences=result.generated_sequences,
            artifacts=result.artifacts,
            logs=result.logs,
            error=None,
        )
        _append_design_completion_message(project_store, project_id, updated_run)
    except Exception as exc:
        updated_run = project_store.update_design_run(project_id, run_id, status="failed", error=str(exc))
        _append_design_completion_message(project_store, project_id, updated_run)


def _append_design_completion_message(project_store: ProjectStore, project_id: str, run: Any) -> None:
    project = project_store.get_project(project_id)
    sequence_count = len(run.generated_sequences or [])
    structure_count = len(run.generated_structure_ids or [])
    fallback_count = len((run.parameters or {}).get("fallbacks") or [])
    conversion = (run.parameters or {}).get("target_conversion")
    lines = []
    if run.status == "completed":
        lines.append(f"Design run `{run.id}` completed with `{run.library}`.")
        if sequence_count:
            lines.append(f"Generated `{sequence_count}` sequence design(s).")
        if structure_count:
            lines.append(f"Saved `{structure_count}` generated structure file(s) into the project workspace.")
    else:
        lines.append(f"Design run `{run.id}` finished with status `{run.status}` for `{run.library}`.")
        if run.error:
            lines.append(f"Error: {run.error}")
    if conversion:
        lines.append(f"Prepared the target with `{conversion}` before running the model.")
    if fallback_count:
        lines.append(f"Tried `{fallback_count}` fallback path(s); see the run logs in Workspace for details.")
    sequence_preview = []
    for item in (run.generated_sequences or [])[:3]:
        sequence = str(item.get("sequence") or "")
        if sequence:
            sequence_preview.append(f"- `{item.get('id') or 'sequence'}`: `{sequence[:80]}`")
    if sequence_preview:
        lines.append("Top sequence preview:\n" + "\n".join(sequence_preview))
    lines.append("Open Workspace to inspect the saved generation folder and use the candidates for filtering.")
    project_store.append_chat_message(
        project_id,
        "assistant",
        "\n\n".join(lines),
        project.selected_job_id,
        project.selected_structure_id,
        tool_events=[
            {
                "tool": "generate_design_candidates",
                "success": run.status == "completed",
                "data": f"Design run {run.id} {run.status}.",
                "error": run.error,
                "raw": {
                    "status": run.status,
                    "design_run_id": run.id,
                    "library": run.library,
                    "generated_sequences": sequence_count,
                    "generated_structures": structure_count,
                    "fallback_count": fallback_count,
                },
            }
        ],
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
