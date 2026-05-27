"""Deterministic analysis profiles for local-first structure triage.

The profile runner is intentionally boring: it uses the existing tool registry
instead of an LLM plan so demos and tests can rank local PDB folders offline.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from structagent.agent import AgentRun, AgentStep
from structagent.batch import StructureResult
from structagent.metrics import extract_metrics_from_steps
from structagent.registry import ToolRegistry, get_registry


STRUCTURE_TOOL_MODULES = (
    "structure_io",
    "contacts",
    "sasa",
    "secondary_structure",
    "interface",
    "alignment",
    "annotations",
    "bfactor",
    "charge",
    "conservation",
    "ramachandran",
    "foldseek",
    "dynamics",
    "relaxation",
    "interface_energy",
    "pyrosetta_interface",
    "renumber_pdb",
)

FEATURE_LIMIT = 250
STANDARD_AMINO_ACIDS = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


@dataclass(frozen=True)
class AnalysisProfile:
    """A named deterministic workflow for batch triage."""

    name: str
    label: str
    description: str
    default_rank_by: str
    tools: tuple[str, ...]


@dataclass
class ProfileRunResult:
    """StructureResult plus UI evidence extracted from raw tool outputs."""

    structure_result: StructureResult
    profile: str
    chains: list[dict[str, Any]]
    features: dict[str, list[dict[str, Any]]]
    warnings: list[str]


ANALYSIS_PROFILES = {
    "triage_default": AnalysisProfile(
        name="triage_default",
        label="Batch triage",
        description="Balanced local screen for surface exposure, flexibility, charge, geometry, and interfaces.",
        default_rank_by="stability",
        tools=(
            "load_structure",
            "list_residues",
            "compute_sasa",
            "analyze_bfactors",
            "compute_charge_distribution",
            "check_ramachandran",
            "compute_interface",
        ),
    ),
    "interface": AnalysisProfile(
        name="interface",
        label="Interface screen",
        description="Prioritizes buried surface area, interface residues, and chain-pair evidence.",
        default_rank_by="buried_surface_area",
        tools=("load_structure", "list_residues", "compute_interface", "compute_sasa", "analyze_bfactors"),
    ),
    "quality": AnalysisProfile(
        name="quality",
        label="Model quality",
        description="Checks per-chain exposure, B-factors, charge clusters, and Ramachandran outliers.",
        default_rank_by="mean_bfactor",
        tools=(
            "load_structure",
            "list_residues",
            "compute_sasa",
            "analyze_bfactors",
            "compute_charge_distribution",
            "check_ramachandran",
        ),
    ),
    "stability": AnalysisProfile(
        name="stability",
        label="Stability screen",
        description="Focuses on compactness, buried/exposed counts, and flexible regions.",
        default_rank_by="stability",
        tools=("load_structure", "compute_sasa", "analyze_bfactors"),
    ),
}


def ensure_analysis_tools_registered() -> None:
    """Import tool modules so registry decorators have run."""
    registry = get_registry()
    reload_modules = not registry.is_tool_registered("load_structure")
    for module_name in STRUCTURE_TOOL_MODULES:
        module = importlib.import_module(f"structagent.tools.{module_name}")
        if reload_modules:
            importlib.reload(module)


def list_analysis_profiles() -> list[dict[str, Any]]:
    """Return public profile metadata for API/UI consumers."""
    return [
        {
            "name": profile.name,
            "label": profile.label,
            "description": profile.description,
            "default_rank_by": profile.default_rank_by,
            "tools": list(profile.tools),
        }
        for profile in ANALYSIS_PROFILES.values()
    ]


def run_analysis_profile(
    *,
    pdb_id: str,
    pdb_path: str | None,
    query: str,
    profile: str = "triage_default",
    chain_a: str | None = None,
    chain_b: str | None = None,
    registry: ToolRegistry | None = None,
) -> ProfileRunResult:
    """Run a deterministic profile against one structure and return metrics/evidence."""
    ensure_analysis_tools_registered()
    registry = registry or get_registry()
    profile_def = ANALYSIS_PROFILES.get(profile, ANALYSIS_PROFILES["triage_default"])
    start_time = time.time()
    steps: list[AgentStep] = []
    warnings: list[str] = []

    source_args = _source_args(pdb_id, pdb_path)
    load_step = _call_step(registry, "load_structure", source_args, "Load structure metadata")
    steps.append(load_step)

    if not load_step.tool_result or not load_step.tool_result.success:
        run = _build_agent_run(query, steps, "Could not load the structure.", start_time)
        error = load_step.tool_result.error if load_step.tool_result else "Structure load failed"
        return ProfileRunResult(
            structure_result=StructureResult(
                pdb_id=pdb_id,
                pdb_path=pdb_path,
                run=run,
                metrics={},
                success=False,
                error=error,
            ),
            profile=profile_def.name,
            chains=[],
            features=_empty_features(),
            warnings=[error or "Structure load failed"],
        )

    chains = _chain_records(load_step.tool_result.raw)
    selected_chain = _select_chain(chains, chain_a)
    interface_pair = _select_interface_pair(chains, selected_chain, chain_b)

    if "list_residues" in profile_def.tools and selected_chain:
        steps.append(
            _call_step(
                registry,
                "list_residues",
                {**source_args, "chain_id": selected_chain},
                f"List residues on chain {selected_chain}",
            )
        )

    if "compute_sasa" in profile_def.tools and selected_chain:
        steps.append(
            _call_step(
                registry,
                "compute_sasa",
                {**source_args, "chain_id": selected_chain},
                f"Compute solvent exposure on chain {selected_chain}",
            )
        )

    if "analyze_bfactors" in profile_def.tools and selected_chain:
        steps.append(
            _call_step(
                registry,
                "analyze_bfactors",
                {**source_args, "chain_id": selected_chain},
                f"Analyze B-factors on chain {selected_chain}",
            )
        )

    if "compute_charge_distribution" in profile_def.tools and selected_chain:
        steps.append(
            _call_step(
                registry,
                "compute_charge_distribution",
                {**source_args, "chain_id": selected_chain},
                f"Analyze charge distribution on chain {selected_chain}",
            )
        )

    if "check_ramachandran" in profile_def.tools and selected_chain:
        steps.append(
            _call_step(
                registry,
                "check_ramachandran",
                {**source_args, "chain_id": selected_chain},
                f"Check Ramachandran statistics on chain {selected_chain}",
            )
        )

    if "compute_interface" in profile_def.tools:
        if interface_pair:
            pair_a, pair_b = interface_pair
            steps.append(
                _call_step(
                    registry,
                    "compute_interface",
                    {**source_args, "chain_a": pair_a, "chain_b": pair_b},
                    f"Compute interface between chains {pair_a} and {pair_b}",
                )
            )
        else:
            warnings.append("No chain pair available for interface analysis.")

    failed_steps = [
        step.tool_name
        for step in steps
        if step.tool_result is not None and not step.tool_result.success and step.tool_name != "check_ramachandran"
    ]
    metrics = extract_metrics_from_steps(steps)
    features = extract_evidence_features(steps)
    final_answer = _summarize_run(pdb_id, profile_def, metrics, features, failed_steps, warnings)
    run = _build_agent_run(query, steps, final_answer, start_time)
    success = bool(metrics) and load_step.tool_result.success

    return ProfileRunResult(
        structure_result=StructureResult(
            pdb_id=pdb_id,
            pdb_path=pdb_path,
            run=run,
            metrics=metrics,
            success=success,
            error="; ".join(failed_steps) if failed_steps and not success else None,
        ),
        profile=profile_def.name,
        chains=chains,
        features=features,
        warnings=warnings,
    )


def extract_evidence_features(steps: list[AgentStep]) -> dict[str, list[dict[str, Any]]]:
    """Extract residue-level evidence for the web viewer."""
    features = _empty_features()

    for step in steps:
        result = step.tool_result
        if not result or not result.success or not result.raw:
            continue

        tool_name = step.tool_name
        raw = result.raw
        args = step.tool_args or {}

        if tool_name == "compute_interface":
            chain_a = args.get("chain_a")
            chain_b = args.get("chain_b")
            residues_a = {
                (residue.get("resname"), residue.get("seqid")) for residue in raw.get("interface_residues_a") or []
            }
            residues_b = {
                (residue.get("resname"), residue.get("seqid")) for residue in raw.get("interface_residues_b") or []
            }
            for residue in raw.get("interface_residues_a") or []:
                if not _is_standard_residue(residue):
                    continue
                features["interface_residues"].append(
                    _residue_feature("interface", chain_a, residue, "Interface residue")
                )
            for residue in raw.get("interface_residues_b") or []:
                if not _is_standard_residue(residue):
                    continue
                features["interface_residues"].append(
                    _residue_feature("interface", chain_b, residue, "Interface residue")
                )
            for residue in raw.get("hotspots") or []:
                if not _is_standard_residue(residue):
                    continue
                features["hotspots"].append(
                    _residue_feature(
                        "hotspot",
                        residue.get("chain")
                        or _infer_interface_chain(residue, residues_a, residues_b, chain_a, chain_b),
                        residue,
                        "Predicted interface hotspot",
                        score=residue.get("buried_sa"),
                    )
                )

        elif tool_name == "compute_sasa":
            chain = args.get("chain_id")
            for residue in raw.get("residues") or []:
                classification = residue.get("classification")
                if classification == "buried":
                    features["buried_residues"].append(_residue_feature("buried", chain, residue, "Buried residue"))
                elif classification == "exposed":
                    features["exposed_residues"].append(_residue_feature("exposed", chain, residue, "Exposed residue"))

        elif tool_name == "analyze_bfactors":
            chain = args.get("chain_id")
            for residue in raw.get("residues") or []:
                if not _is_standard_residue(residue):
                    continue
                classification = residue.get("classification")
                if classification in {"flexible", "highly_flexible"} or residue.get("potentially_disordered"):
                    features["high_bfactor_residues"].append(
                        _residue_feature(
                            "high_bfactor",
                            chain,
                            residue,
                            "Flexible or high B-factor residue",
                            score=residue.get("avg_bfactor"),
                        )
                    )

        elif tool_name == "compute_charge_distribution":
            for index, cluster in enumerate(raw.get("clusters") or [], start=1):
                residues = []
                for residue in cluster.get("residues") or []:
                    residues.append(_residue_feature("charge_cluster", args.get("chain_id"), residue, "Charge cluster"))
                features["charge_clusters"].append(
                    {
                        "id": f"charge-cluster-{index}",
                        "kind": "charge_cluster",
                        "label": f"Charge cluster {index}",
                        "count": cluster.get("count", len(residues)),
                        "total_charge": cluster.get("total_charge"),
                        "residues": residues,
                    }
                )

        elif tool_name == "check_ramachandran":
            chain = args.get("chain_id")
            for residue in raw.get("outliers") or []:
                features["ramachandran_outliers"].append(
                    _residue_feature("rama_outlier", chain, residue, "Ramachandran outlier")
                )

    for key, values in features.items():
        features[key] = values[:FEATURE_LIMIT]
    return features


def _source_args(pdb_id: str, pdb_path: str | None) -> dict[str, str]:
    if pdb_path:
        return {"pdb_path": pdb_path}
    return {"pdb_id": pdb_id}


def _call_step(registry: ToolRegistry, tool_name: str, args: dict[str, Any], purpose: str) -> AgentStep:
    result = registry.call_tool(tool_name, **args)
    return AgentStep(
        thought=purpose,
        tool_name=tool_name,
        tool_args=dict(args),
        tool_result=result,
        is_final=False,
        timestamp=time.time(),
    )


def _build_agent_run(query: str, steps: list[AgentStep], final_answer: str, start_time: float) -> AgentRun:
    return AgentRun(
        query=query,
        steps=steps,
        final_answer=final_answer,
        total_steps=len(steps),
        wall_time_seconds=time.time() - start_time,
        model="deterministic-profile",
    )


def _chain_records(load_raw: dict[str, Any]) -> list[dict[str, Any]]:
    chains = load_raw.get("chains") or []
    return [dict(chain) for chain in chains if chain.get("id")]


def _select_chain(chains: list[dict[str, Any]], requested: str | None) -> str | None:
    chain_ids = [chain["id"] for chain in chains]
    if requested and requested in chain_ids:
        return requested
    return chain_ids[0] if chain_ids else requested


def _select_interface_pair(
    chains: list[dict[str, Any]], selected_chain: str | None, requested_b: str | None
) -> tuple[str, str] | None:
    chain_ids = [chain["id"] for chain in chains]
    if selected_chain and requested_b and requested_b in chain_ids and requested_b != selected_chain:
        return selected_chain, requested_b
    if len(chain_ids) >= 2:
        if selected_chain in chain_ids:
            partner = next((chain_id for chain_id in chain_ids if chain_id != selected_chain), None)
            return (selected_chain, partner) if partner else None
        return chain_ids[0], chain_ids[1]
    return None


def _empty_features() -> dict[str, list[dict[str, Any]]]:
    return {
        "interface_residues": [],
        "hotspots": [],
        "buried_residues": [],
        "exposed_residues": [],
        "high_bfactor_residues": [],
        "charge_clusters": [],
        "ramachandran_outliers": [],
    }


def _residue_feature(
    kind: str,
    chain: str | None,
    residue: dict[str, Any],
    label: str,
    score: float | None = None,
) -> dict[str, Any]:
    residue_number = residue.get("seqid", residue.get("residue_number", residue.get("res_num")))
    residue_name = residue.get("resname", residue.get("residue_name", residue.get("res_name")))
    feature = {
        "kind": kind,
        "chain": chain,
        "residue_number": residue_number,
        "residue_name": residue_name,
        "label": label,
    }
    if score is not None:
        feature["score"] = score
    return feature


def _is_standard_residue(residue: dict[str, Any]) -> bool:
    residue_name = residue.get("resname", residue.get("residue_name", residue.get("res_name")))
    return residue_name in STANDARD_AMINO_ACIDS


def _infer_interface_chain(
    residue: dict[str, Any],
    residues_a: set[tuple[Any, Any]],
    residues_b: set[tuple[Any, Any]],
    chain_a: str | None,
    chain_b: str | None,
) -> str | None:
    key = (residue.get("resname"), residue.get("seqid"))
    if key in residues_a:
        return chain_a
    if key in residues_b:
        return chain_b
    return chain_a


def _summarize_run(
    pdb_id: str,
    profile: AnalysisProfile,
    metrics: dict[str, Any],
    features: dict[str, list[dict[str, Any]]],
    failed_steps: list[str],
    warnings: list[str],
) -> str:
    lines = [f"{profile.label} completed for {Path(pdb_id).stem or pdb_id}."]
    if metrics:
        metric_bits = []
        for key in ("buried_surface_area", "n_interface_residues", "mean_relative_sasa_percent", "mean_bfactor"):
            if key in metrics:
                value = metrics[key]
                metric_bits.append(
                    f"{key.replace('_', ' ')}={value:.2f}" if isinstance(value, float) else f"{key}={value}"
                )
        if metric_bits:
            lines.append("Key ranking signals: " + ", ".join(metric_bits) + ".")

    evidence_bits = []
    for key in ("interface_residues", "high_bfactor_residues", "ramachandran_outliers", "charge_clusters"):
        count = len(features.get(key) or [])
        if count:
            evidence_bits.append(f"{count} {key.replace('_', ' ')}")
    if evidence_bits:
        lines.append("Viewer evidence: " + ", ".join(evidence_bits) + ".")

    if failed_steps:
        lines.append("Some optional checks failed: " + ", ".join(failed_steps) + ".")
    if warnings:
        lines.append("Warnings: " + " ".join(warnings))
    return " ".join(lines)
