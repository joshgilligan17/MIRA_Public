"""Shared metric extraction and ranking definitions for batch analysis."""

from __future__ import annotations

from numbers import Real
from typing import Iterable, Any


RANKING_CRITERIA = {
    "stability": {"metric": "mean_relative_sasa_percent", "higher_is_better": False},
    "buried_surface_area": {"metric": "buried_surface_area", "higher_is_better": True},
    "n_interface_residues": {"metric": "n_interface_residues", "higher_is_better": True},
    "mean_bfactor": {"metric": "mean_bfactor", "higher_is_better": False},
    "std_bfactor": {"metric": "std_bfactor", "higher_is_better": False},
    "n_buried": {"metric": "n_buried", "higher_is_better": True},
    "n_exposed": {"metric": "n_exposed", "higher_is_better": False},
    "interface_energy": {"metric": "interface_energy", "higher_is_better": False},
    "shape_complementarity": {"metric": "shape_complementarity", "higher_is_better": True},
    "packstat": {"metric": "packstat", "higher_is_better": True},
}


def extract_metrics_from_steps(steps: Iterable[Any]) -> dict:
    """Extract rankable metrics from agent steps with successful tool results."""
    metrics: dict = {}

    for step in steps:
        tool_result = getattr(step, "tool_result", None)
        if not (tool_result and tool_result.success and tool_result.raw):
            continue

        raw = tool_result.raw
        tool_name = getattr(step, "tool_name", None)

        if tool_name == "compute_interface":
            buried_sa = raw.get("buried_sa_total") or raw.get("buried_surface_area")
            if buried_sa is not None:
                metrics["buried_surface_area"] = buried_sa

            residues_a = raw.get("interface_residues_a") or []
            residues_b = raw.get("interface_residues_b") or []
            n_interface = len(residues_a) + len(residues_b)
            if n_interface > 0:
                metrics["n_interface_residues"] = n_interface

        elif tool_name == "compute_sasa":
            residues = raw.get("residues") or []
            sasas = [r.get("relative_sasa_percent", 100) for r in residues]
            if sasas:
                metrics["mean_relative_sasa_percent"] = sum(sasas) / len(sasas)

            classifications = [r.get("classification", "") for r in residues]
            metrics["n_buried"] = sum(1 for c in classifications if c == "buried")
            metrics["n_partial"] = sum(1 for c in classifications if c == "partial")
            metrics["n_exposed"] = sum(1 for c in classifications if c == "exposed")

        elif tool_name == "analyze_bfactors":
            stats = raw.get("statistics", {})
            mean_bf = stats.get("mean")
            if mean_bf is not None:
                metrics["mean_bfactor"] = mean_bf
            std_bf = stats.get("std")
            if std_bf is not None:
                metrics["std_bfactor"] = std_bf

        elif tool_name == "compute_charge_distribution":
            total_charge = raw.get("total_charge")
            if total_charge is not None:
                metrics["total_charge"] = total_charge
            cluster_count = raw.get("cluster_count")
            if cluster_count is not None:
                metrics["charge_cluster_count"] = cluster_count

        elif tool_name == "score_interface":
            interface_dg = raw.get("interface_dG")
            if interface_dg is None:
                interface_dg = raw.get("dG")
            if interface_dg is not None:
                metrics["interface_energy"] = interface_dg

            sc = raw.get("shape_complementarity")
            if sc is not None:
                metrics["shape_complementarity"] = sc

            packstat = raw.get("packstat")
            if packstat is not None:
                metrics["packstat"] = packstat

        execution_time = getattr(tool_result, "execution_time_seconds", None)
        if isinstance(execution_time, Real):
            metrics["total_execution_time"] = metrics.get("total_execution_time", 0.0) + execution_time

    return metrics
