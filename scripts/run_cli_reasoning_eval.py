#!/usr/bin/env python3
"""Run a scaled planning/tool-selection eval for the MIRA CLI agent.

The eval is intentionally fast: it scores whether the agent can turn structural
biology requests into valid MIRA tool plans. This is a useful near-term proxy for
CLI reasoning quality because incorrect tool choice or invalid arguments break
downstream analysis before any expensive structure work begins.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import html
import io
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from structagent.agent import MiraAgent  # noqa: E402
from structagent.cli import initialize_tools, resolve_api_key  # noqa: E402
from structagent.registry import get_registry  # noqa: E402


CATEGORY_ORDER = [
    "active_site",
    "interface",
    "allostery",
    "stability",
    "design",
    "homology",
    "general",
]


CATEGORY_SPECS: dict[str, dict[str, Any]] = {
    "active_site": {
        "label": "Active-site reasoning",
        "queries": [
            (
                "For PDB {pdb_id}, plan an analysis that identifies likely catalytic or "
                "functional residues, their contacts, and whether those residues are buried "
                "or solvent exposed."
            ),
            (
                "For PDB {pdb_id}, determine which residues are most likely to explain the "
                "reported enzymatic function and what structural evidence should support them."
            ),
        ],
        "expected_tool_groups": [
            ["load_structure"],
            ["get_functional_annotations", "list_residues"],
            ["get_residue_contacts"],
            ["compute_sasa"],
        ],
    },
    "interface": {
        "label": "Interface reasoning",
        "queries": [
            (
                "For PDB {pdb_id}, plan a protein-protein interface analysis that finds the "
                "contacting residues, estimates buried surface area, and highlights likely "
                "binding hotspots."
            ),
            (
                "For PDB {pdb_id}, rank the evidence that chains A and B form a meaningful "
                "interaction and identify residues that should be inspected in the viewer."
            ),
        ],
        "expected_tool_groups": [
            ["load_structure"],
            ["compute_interface", "get_residue_contacts"],
            ["compute_sasa"],
            ["analyze_bfactors", "score_interface", "analyze_interface_energies"],
        ],
    },
    "allostery": {
        "label": "Dynamics/allostery reasoning",
        "queries": [
            (
                "For PDB {pdb_id}, plan an analysis of possible hinge regions, flexible "
                "segments, and allosteric motion that could influence function."
            ),
            (
                "For PDB {pdb_id}, identify the structural evidence needed to reason about "
                "conformational change and dynamic coupling across the protein."
            ),
        ],
        "expected_tool_groups": [
            ["load_structure"],
            ["compute_normal_modes", "predict_hinge_regions", "compute_cross_correlations"],
            ["analyze_bfactors"],
            ["get_secondary_structure"],
        ],
    },
    "stability": {
        "label": "Stability/quality reasoning",
        "queries": [
            (
                "For PDB {pdb_id}, plan a structure-quality and stability analysis covering "
                "secondary structure, solvent exposure, B-factors, and Ramachandran outliers."
            ),
            (
                "For PDB {pdb_id}, decide whether the fold looks reliable enough for design "
                "filtering and what quality metrics should be collected."
            ),
        ],
        "expected_tool_groups": [
            ["load_structure"],
            ["get_secondary_structure"],
            ["compute_sasa"],
            ["analyze_bfactors"],
            ["check_ramachandran"],
        ],
    },
    "design": {
        "label": "Design triage reasoning",
        "queries": [
            (
                "For PDB {pdb_id}, plan how MIRA should triage this designed structure for "
                "binder-design follow-up, including interface, exposure, and quality checks."
            ),
            (
                "For PDB {pdb_id}, identify evidence that would make this de novo design a "
                "strong or weak candidate for additional sequence or structure design."
            ),
        ],
        "expected_tool_groups": [
            ["load_structure"],
            ["compute_interface", "get_residue_contacts"],
            ["compute_sasa"],
            ["analyze_bfactors", "check_ramachandran"],
        ],
    },
    "homology": {
        "label": "Homology/context reasoning",
        "queries": [
            (
                "For PDB {pdb_id}, plan how to compare this structure to known homologs and "
                "use that context to interpret conserved functional regions."
            ),
            (
                "For PDB {pdb_id}, determine whether structural homolog searches and "
                "functional annotations would clarify the biological role of the protein."
            ),
        ],
        "expected_tool_groups": [
            ["load_structure"],
            ["search_structural_homologs"],
            ["get_functional_annotations", "get_conservation_scores"],
            ["align_structures"],
        ],
    },
    "general": {
        "label": "General structure reasoning",
        "queries": [
            (
                "For PDB {pdb_id}, plan a general structural analysis that summarizes chains, "
                "fold elements, solvent exposure, residue-level features, and limitations."
            ),
            (
                "For PDB {pdb_id}, produce an evidence-gathering plan for a first-pass "
                "structure review before deciding whether deeper analysis is needed."
            ),
        ],
        "expected_tool_groups": [
            ["load_structure"],
            ["list_residues"],
            ["get_secondary_structure"],
            ["compute_sasa", "analyze_bfactors"],
        ],
    },
}


TOOL_ARG_TEMPLATES: dict[str, dict[str, Any]] = {
    "load_structure": {"pdb_id": "{pdb_id}"},
    "list_residues": {"pdb_id": "{pdb_id}", "chain_id": "A"},
    "get_residue_contacts": {"pdb_id": "{pdb_id}", "chain_id": "A", "residue_number": 1},
    "compute_sasa": {"pdb_id": "{pdb_id}", "chain_id": "A"},
    "get_secondary_structure": {"pdb_id": "{pdb_id}", "chain_id": "A"},
    "compute_interface": {"pdb_id": "{pdb_id}", "chain_a": "A", "chain_b": "B"},
    "align_structures": {"pdb_id_1": "{pdb_id}", "chain_id_1": "A", "pdb_id_2": "1UBQ", "chain_id_2": "A"},
    "get_functional_annotations": {"pdb_id": "{pdb_id}", "chain_id": "A"},
    "analyze_bfactors": {"pdb_id": "{pdb_id}", "chain_id": "A"},
    "compute_charge_distribution": {"pdb_id": "{pdb_id}", "chain_id": "A"},
    "check_ramachandran": {"pdb_id": "{pdb_id}", "chain_id": "A"},
    "search_structural_homologs": {"pdb_id": "{pdb_id}", "chain_id": "A", "max_hits": 10},
    "compute_normal_modes": {"pdb_id": "{pdb_id}", "chain_id": "A", "n_modes": 10},
    "compute_cross_correlations": {"pdb_id": "{pdb_id}", "chain_id": "A", "n_modes": 10},
    "predict_hinge_regions": {"pdb_id": "{pdb_id}", "chain_id": "A", "n_modes": 10},
    "compute_perturbation_response": {"pdb_id": "{pdb_id}", "chain_id": "A", "source_residue": 1},
    "fast_relax": {"input_path": "{pdb_id}.pdb", "sidechain_only": True, "iterations": 5},
    "analyze_interface_energies": {"pdb_path": "{pdb_id}.pdb", "binder_chain": "B"},
    "score_interface": {"pdb_path": "{pdb_id}.pdb", "binder_chains": ["B"], "target_chain": "A"},
    "renumber_pdb": {"pdb_path": "{pdb_id}.pdb"},
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            pdb_id = str(record.get("pdb_id", "")).upper()
            if not pdb_id or pdb_id == "PLACEHOLDER":
                continue
            record["pdb_id"] = pdb_id
            records.append(record)
    return records


def infer_category(record: dict[str, Any]) -> str:
    category = str(record.get("category", "")).strip().lower()
    if category in CATEGORY_SPECS:
        return category

    text = " ".join(
        str(record.get(field, ""))
        for field in ("title", "abstract", "description", "journal", "organism")
    ).lower()

    if any(word in text for word in ("interface", "antibody", "complex", "binding", "receptor", "tcr", "mhc")):
        return "interface"
    if any(word in text for word in ("alloster", "flexib", "conformation", "motion", "hinge", "open", "closed")):
        return "allostery"
    if any(word in text for word in ("enzyme", "cataly", "active site", "substrate", "cofactor")):
        return "active_site"
    if any(word in text for word in ("homolog", "evolution", "conserved", "family", "fold")):
        return "homology"
    if any(word in text for word in ("design", "designed", "de novo", "binder")):
        return "design"
    if any(word in text for word in ("stability", "mutant", "variant", "packing", "b-factor", "resolution")):
        return "stability"
    return "general"


def make_context(record: dict[str, Any], category: str) -> str:
    title = record.get("title") or record.get("description") or "No title available"
    resolution = record.get("resolution")
    resolution_text = f"{resolution} A" if resolution else "unknown"
    return (
        f"Benchmark structure: {record['pdb_id']}. "
        f"Task family: {category}. "
        f"Known context: {title}. "
        f"Resolution: {resolution_text}. "
        "Assume the primary protein chain is A unless tool evidence says otherwise. "
        "For interface tasks, assume chains A and B are the candidate interaction pair. "
        "Use MIRA CLI tools to gather evidence rather than answering from memory alone."
    )


def build_tasks(records: list[dict[str, Any]], limit: int, tasks_per_record: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORY_ORDER}
    seen = set()
    for record in records:
        pdb_id = record["pdb_id"]
        if pdb_id in seen:
            continue
        seen.add(pdb_id)
        grouped.setdefault(infer_category(record), []).append(record)

    tasks = []
    cursors = {category: 0 for category in grouped}
    while len(tasks) < limit:
        made_progress = False
        for category in CATEGORY_ORDER:
            records_for_category = grouped.get(category, [])
            cursor = cursors.get(category, 0)
            if cursor >= len(records_for_category):
                continue

            record = records_for_category[cursor]
            cursors[category] = cursor + 1
            spec = CATEGORY_SPECS[category]
            for query_index, template in enumerate(spec["queries"][:tasks_per_record], start=1):
                tasks.append(
                    {
                        "id": f"{category}_{record['pdb_id']}_{query_index:02d}",
                        "pdb_id": record["pdb_id"],
                        "category": category,
                        "category_label": spec["label"],
                        "query": template.format(pdb_id=record["pdb_id"]),
                        "context": make_context(record, category),
                        "expected_tool_groups": spec["expected_tool_groups"],
                    }
                )
                made_progress = True
                if len(tasks) >= limit:
                    break
            if len(tasks) >= limit:
                break
        if not made_progress:
            break

    return tasks


def format_args(tool_name: str, pdb_id: str) -> dict[str, Any]:
    template = TOOL_ARG_TEMPLATES.get(tool_name, {})

    def replace(value: Any) -> Any:
        if isinstance(value, str):
            return value.format(pdb_id=pdb_id)
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    return replace(template)


def heuristic_plan(task: dict[str, Any]) -> dict[str, Any]:
    steps = []
    used = set()
    for group in task["expected_tool_groups"]:
        tool_name = group[0]
        if tool_name in used:
            continue
        used.add(tool_name)
        steps.append(
            {
                "tool": tool_name,
                "args": format_args(tool_name, task["pdb_id"]),
                "purpose": f"Collect evidence for {task['category_label'].lower()}.",
            }
        )
    return {
        "reasoning": "Deterministic offline baseline plan generated from the task rubric.",
        "steps": steps,
    }


def create_live_agent(args: argparse.Namespace) -> MiraAgent:
    api_key, base_url, model = resolve_api_key(
        provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )
    if not api_key:
        raise RuntimeError("No API key found. Set MINIMAX_API_KEY or pass --api-key for live mode.")

    return MiraAgent(
        model=model or args.model or "MiniMax-M2.7",
        base_url=base_url or args.base_url or "https://api.minimax.io/v1",
        api_key=api_key,
        max_steps=args.max_steps,
        verbose=False,
        mode="plan",
        display="verbose",
        timeout=args.timeout,
        temperature=args.temperature,
    )


def live_plan(agent: MiraAgent, task: dict[str, Any], probe_chains: bool) -> dict[str, Any] | None:
    if not probe_chains:
        agent._get_structure_chain_info = lambda _pdb_id: None  # type: ignore[method-assign]

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        return agent.create_plan(task["query"], context=task["context"])


def validate_plan_schema(plan: Any, registry: Any) -> tuple[bool, list[str]]:
    errors = []
    if not isinstance(plan, dict):
        return False, ["plan is not an object"]
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return False, ["plan has no steps"]

    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"step {index} is not an object")
            continue
        tool_name = step.get("tool")
        if not tool_name:
            errors.append(f"step {index} missing tool")
            continue
        if tool_name not in registry._tools:
            errors.append(f"step {index} unknown tool {tool_name}")
            continue

        args = step.get("args") or {}
        if not isinstance(args, dict):
            errors.append(f"step {index} args is not an object")
            continue

        schema = registry._tools[tool_name].parameters or {}
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        allowed = set(properties.keys())
        extra_args = sorted(set(args.keys()) - allowed)
        missing_args = sorted(required - set(args.keys()))
        if extra_args:
            errors.append(f"step {index} {tool_name} extra args: {extra_args}")
        if missing_args:
            errors.append(f"step {index} {tool_name} missing required args: {missing_args}")

    return not errors, errors


def score_plan(task: dict[str, Any], plan: Any, registry: Any, threshold: float) -> dict[str, Any]:
    schema_valid, schema_errors = validate_plan_schema(plan, registry)
    steps = plan.get("steps", []) if isinstance(plan, dict) else []
    planned_tools = [step.get("tool") for step in steps if isinstance(step, dict) and step.get("tool")]
    planned_tools = [tool for tool in planned_tools if isinstance(tool, str)]
    planned_unique = list(dict.fromkeys(planned_tools))

    expected_groups = task["expected_tool_groups"]
    group_hits = []
    for group in expected_groups:
        group_hits.append(any(tool in planned_unique for tool in group))

    expected_pool = {tool for group in expected_groups for tool in group}
    relevant = sum(1 for tool in planned_unique if tool in expected_pool)
    recall = sum(group_hits) / len(expected_groups) if expected_groups else 0.0
    precision = relevant / len(planned_unique) if planned_unique else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    unknown_tools = [tool for tool in planned_unique if tool not in registry._tools]
    extra_tools = [tool for tool in planned_unique if tool not in expected_pool]
    passed = bool(schema_valid and not unknown_tools and recall >= threshold)

    return {
        "planned_tools": planned_unique,
        "tool_count": len(planned_unique),
        "expected_group_hits": group_hits,
        "tool_recall": recall,
        "tool_precision": precision,
        "tool_f1": f1,
        "schema_valid": schema_valid,
        "schema_errors": schema_errors,
        "unknown_tools": unknown_tools,
        "extra_tools": extra_tools,
        "passed": passed,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for row in results if row["passed"])
    category_rows = {}
    for category in sorted({row["category"] for row in results}):
        rows = [row for row in results if row["category"] == category]
        category_rows[category] = {
            "n": len(rows),
            "pass_rate": safe_mean([1.0 if row["passed"] else 0.0 for row in rows]),
            "tool_recall": safe_mean([row["tool_recall"] for row in rows]),
            "tool_precision": safe_mean([row["tool_precision"] for row in rows]),
            "tool_f1": safe_mean([row["tool_f1"] for row in rows]),
            "schema_valid_rate": safe_mean([1.0 if row["schema_valid"] else 0.0 for row in rows]),
            "mean_latency_seconds": safe_mean([row["latency_seconds"] for row in rows]),
        }
    return {
        "n_tasks": total,
        "n_passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "tool_recall": safe_mean([row["tool_recall"] for row in results]),
        "tool_precision": safe_mean([row["tool_precision"] for row in results]),
        "tool_f1": safe_mean([row["tool_f1"] for row in results]),
        "schema_valid_rate": safe_mean([1.0 if row["schema_valid"] else 0.0 for row in results]),
        "mean_latency_seconds": safe_mean([row["latency_seconds"] for row in results]),
        "by_category": category_rows,
    }


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "id",
        "pdb_id",
        "category",
        "mode",
        "passed",
        "tool_recall",
        "tool_precision",
        "tool_f1",
        "schema_valid",
        "tool_count",
        "latency_seconds",
        "planned_tools",
        "schema_errors",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key]) if isinstance(row.get(key), (list, dict)) else row.get(key)
                    for key in columns
                }
            )


def write_report(path: Path, summary: dict[str, Any], run_config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    misses = [row for row in rows if not row["passed"]][:10]
    lines = [
        "# MIRA CLI Reasoning Eval",
        "",
        f"- Mode: `{run_config['mode']}`",
        f"- Tasks: `{summary['n_tasks']}`",
        f"- Pass rate: `{summary['pass_rate']:.1%}`",
        f"- Mean tool recall: `{summary['tool_recall']:.1%}`",
        f"- Mean tool precision: `{summary['tool_precision']:.1%}`",
        f"- Schema-valid plans: `{summary['schema_valid_rate']:.1%}`",
        f"- Mean latency: `{summary['mean_latency_seconds']:.2f}s`",
        "",
        "## By Category",
        "",
        "| Category | n | Pass | Recall | Precision | Schema valid | Latency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category, data in summary["by_category"].items():
        lines.append(
            "| {category} | {n} | {pass_rate:.1%} | {tool_recall:.1%} | "
            "{tool_precision:.1%} | {schema_valid_rate:.1%} | {mean_latency_seconds:.2f}s |".format(
                category=category,
                **data,
            )
        )
    lines.extend(["", "## First Misses", ""])
    if not misses:
        lines.append("No failed tasks at the configured threshold.")
    else:
        for row in misses:
            lines.append(
                f"- `{row['id']}` recall `{row['tool_recall']:.1%}`, "
                f"schema valid `{row['schema_valid']}`, tools `{', '.join(row['planned_tools'])}`"
            )
            if row["schema_errors"]:
                lines.append(f"  Schema errors: `{'; '.join(row['schema_errors'])}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This eval measures planning and tool-use readiness, not final biological correctness. "
            "It is meant to show whether the CLI agent can decompose common structure-reasoning "
            "questions into executable MIRA tool calls before running expensive analysis or design jobs.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def svg_start(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#17233f} .muted{fill:#5e6d86} .axis{stroke:#b9c7d6;stroke-width:1} .grid{stroke:#dbe6f0;stroke-width:1}</style>',
    ]


def svg_text(x: float, y: float, text: str, size: int = 16, anchor: str = "start", klass: str = "") -> str:
    class_attr = f' class="{klass}"' if klass else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}"{class_attr}>{html.escape(text)}</text>'


def save_bar_chart(path: Path, title: str, values: dict[str, float], ylabel: str) -> None:
    width, height = 1100, 680
    margin_left, margin_right, margin_top, margin_bottom = 110, 60, 90, 130
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    labels = list(values.keys())
    max_value = max(1.0, max(values.values(), default=1.0))
    bar_gap = 18
    bar_w = (plot_w - bar_gap * max(0, len(labels) - 1)) / max(1, len(labels))
    lines = svg_start(width, height)
    lines.append(svg_text(width / 2, 42, title, 28, "middle"))
    lines.append(svg_text(width / 2, 70, ylabel, 14, "middle", "muted"))

    for tick in range(0, 6):
        value = tick / 5 * max_value
        y = margin_top + plot_h - (value / max_value) * plot_h
        lines.append(f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}"/>')
        lines.append(svg_text(margin_left - 14, y + 5, f"{value:.0%}", 13, "end", "muted"))

    palette = ["#2474bc", "#22a6a1", "#f2a03d", "#7c62d6", "#d94f70", "#5f7d4e", "#8a9aa8"]
    for index, label in enumerate(labels):
        value = values[label]
        x = margin_left + index * (bar_w + bar_gap)
        bar_h = (value / max_value) * plot_h
        y = margin_top + plot_h - bar_h
        color = palette[index % len(palette)]
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="6" fill="{color}"/>')
        lines.append(svg_text(x + bar_w / 2, y - 8, f"{value:.0%}", 14, "middle"))
        lines.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{height - 78}" font-size="13" text-anchor="middle" transform="rotate(-32 {x + bar_w / 2:.1f} {height - 78})">{html.escape(label)}</text>'
        )
    lines.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_grouped_metric_chart(path: Path, title: str, category_data: dict[str, dict[str, float]]) -> None:
    width, height = 1200, 760
    margin_left, margin_right, margin_top, margin_bottom = 110, 70, 155, 145
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    categories = list(category_data.keys())
    metrics = [("tool_recall", "Recall", "#2474bc"), ("tool_precision", "Precision", "#22a6a1"), ("schema_valid_rate", "Schema", "#f2a03d")]
    group_gap = 28
    group_w = (plot_w - group_gap * max(0, len(categories) - 1)) / max(1, len(categories))
    bar_w = group_w / len(metrics) - 4
    lines = svg_start(width, height)
    lines.append(svg_text(width / 2, 42, title, 28, "middle"))
    lines.append(svg_text(width / 2, 70, "Mean score by task family", 14, "middle", "muted"))
    legend_start = width / 2 - 220
    legend_y = 105
    lines.append(
        f'<rect x="{legend_start - 22:.1f}" y="{legend_y - 25}" width="440" height="40" rx="10" fill="#ffffff" stroke="#dbe6f0"/>'
    )
    for metric_index, (_, label, color) in enumerate(metrics):
        legend_x = legend_start + metric_index * 145
        lines.append(f'<rect x="{legend_x:.1f}" y="{legend_y - 13}" width="16" height="16" rx="3" fill="{color}"/>')
        lines.append(svg_text(legend_x + 24, legend_y + 1, label, 14))
    for tick in range(0, 6):
        value = tick / 5
        y = margin_top + plot_h - value * plot_h
        lines.append(f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}"/>')
        lines.append(svg_text(margin_left - 14, y + 5, f"{value:.0%}", 13, "end", "muted"))
    for category_index, category in enumerate(categories):
        x0 = margin_left + category_index * (group_w + group_gap)
        for metric_index, (key, _, color) in enumerate(metrics):
            value = category_data[category].get(key, 0.0)
            bar_h = value * plot_h
            x = x0 + metric_index * (bar_w + 4)
            y = margin_top + plot_h - bar_h
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="5" fill="{color}"/>')
        label_x = x0 + group_w / 2
        lines.append(
            f'<text x="{label_x:.1f}" y="{height - 78}" font-size="13" text-anchor="middle" transform="rotate(-30 {label_x:.1f} {height - 78})">{html.escape(category)}</text>'
        )
    lines.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_latency_plot(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    width, height = 1200, 620
    margin_left, margin_right, margin_top, margin_bottom = 110, 60, 90, 90
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_latency = max([row["latency_seconds"] for row in rows] or [1.0])
    max_latency = max(max_latency, 0.01)
    lines = svg_start(width, height)
    lines.append(svg_text(width / 2, 42, title, 28, "middle"))
    lines.append(svg_text(width / 2, 70, "Each dot is one planning task", 14, "middle", "muted"))
    for tick in range(0, 6):
        value = tick / 5 * max_latency
        y = margin_top + plot_h - (value / max_latency) * plot_h
        lines.append(f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}"/>')
        lines.append(svg_text(margin_left - 14, y + 5, f"{value:.2f}s", 13, "end", "muted"))
    palette = {"passed": "#2474bc", "failed": "#d94f70"}
    for index, row in enumerate(rows):
        x = margin_left + (index / max(1, len(rows) - 1)) * plot_w
        y = margin_top + plot_h - (row["latency_seconds"] / max_latency) * plot_h
        color = palette["passed" if row["passed"] else "failed"]
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" opacity="0.82"/>')
    lines.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}"/>')
    lines.append(svg_text(width / 2, height - 35, "Task index", 15, "middle", "muted"))
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_tool_heatmap(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    tool_totals = Counter()
    for row in rows:
        category = row["category"]
        for tool in row["planned_tools"]:
            category_counts[category][tool] += 1
            tool_totals[tool] += 1
    tools = [tool for tool, _ in tool_totals.most_common(14)]
    categories = [category for category in CATEGORY_ORDER if category in category_counts]
    cell_w, cell_h = 72, 42
    width = 260 + cell_w * max(1, len(tools))
    height = 150 + cell_h * max(1, len(categories))
    lines = svg_start(width, height)
    lines.append(svg_text(width / 2, 42, title, 26, "middle"))
    lines.append(svg_text(width / 2, 70, "Tool selection frequency by category", 14, "middle", "muted"))
    for tool_index, tool in enumerate(tools):
        x = 220 + tool_index * cell_w + cell_w / 2
        lines.append(
            f'<text x="{x:.1f}" y="125" font-size="12" text-anchor="middle" transform="rotate(-45 {x:.1f} 125)">{html.escape(tool)}</text>'
        )
    for category_index, category in enumerate(categories):
        y = 145 + category_index * cell_h
        lines.append(svg_text(200, y + 27, category, 14, "end"))
        denom = max(1, sum(1 for row in rows if row["category"] == category))
        for tool_index, tool in enumerate(tools):
            count = category_counts[category][tool]
            value = count / denom
            blue = 245 - int(120 * value)
            green = 250 - int(75 * value)
            color = f"rgb({245 - int(210 * value)},{green},{blue})"
            x = 220 + tool_index * cell_w
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_w - 3}" height="{cell_h - 3}" fill="{color}" stroke="#e3edf5"/>')
            if count:
                lines.append(svg_text(x + cell_w / 2, y + 26, str(count), 13, "middle"))
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plots(run_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    category_pass = {category: data["pass_rate"] for category, data in summary["by_category"].items()}
    save_bar_chart(plots_dir / "pass_rate_by_category.svg", "MIRA CLI Eval Pass Rate", category_pass, "Pass rate")
    save_grouped_metric_chart(plots_dir / "reasoning_metrics_by_category.svg", "MIRA CLI Reasoning Metrics", summary["by_category"])
    save_latency_plot(plots_dir / "latency_by_task.svg", "MIRA CLI Planning Latency", rows)
    save_tool_heatmap(plots_dir / "tool_selection_heatmap.svg", "MIRA CLI Tool Coverage", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "benchmark_candidates.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "eval-results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--limit", type=int, default=72)
    parser.add_argument("--tasks-per-record", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.75, help="Minimum expected tool-group recall to pass")
    parser.add_argument("--provider", choices=["openai", "anthropic", "minimax", "azure"], default="minimax")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--probe-chains",
        action="store_true",
        help="Allow live planning to load PDB chain metadata before planning. Disabled by default to avoid network/tool work.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or datetime.now().strftime("cli-reasoning-%Y%m%d-%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(args.benchmark)
    tasks = build_tasks(records, limit=args.limit, tasks_per_record=max(1, args.tasks_per_record))

    initialize_tools()
    registry = get_registry()
    agent = create_live_agent(args) if args.mode == "live" else None

    rows = []
    for index, task in enumerate(tasks, start=1):
        started = time.time()
        error = None
        plan = None
        try:
            if args.mode == "offline":
                plan = heuristic_plan(task)
            else:
                assert agent is not None
                plan = live_plan(agent, task, probe_chains=args.probe_chains)
        except Exception as exc:  # pylint: disable=broad-except
            error = str(exc)
        latency = time.time() - started
        score = score_plan(task, plan, registry, args.threshold)
        row = {
            **task,
            **score,
            "mode": args.mode,
            "latency_seconds": latency,
            "error": error,
            "plan": plan,
        }
        rows.append(row)
        status = "PASS" if row["passed"] else "MISS"
        print(
            f"[{index:03d}/{len(tasks):03d}] {status} {task['id']} "
            f"recall={row['tool_recall']:.2f} schema={row['schema_valid']} latency={latency:.2f}s",
            flush=True,
        )

    summary = summarize(rows)
    run_config = {
        "mode": args.mode,
        "benchmark": str(args.benchmark),
        "limit": args.limit,
        "tasks_per_record": args.tasks_per_record,
        "threshold": args.threshold,
        "provider": args.provider if args.mode == "live" else None,
        "model": agent.model if agent else "offline-heuristic",
        "probe_chains": args.probe_chains,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    summary["run_config"] = run_config

    write_jsonl(run_dir / "results.jsonl", rows)
    write_csv(run_dir / "results.csv", rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_report(run_dir / "report.md", summary, run_config, rows)
    write_plots(run_dir, summary, rows)

    print("")
    print(f"Run directory: {run_dir}")
    print(f"Tasks: {summary['n_tasks']}")
    print(f"Pass rate: {summary['pass_rate']:.1%}")
    print(f"Mean recall: {summary['tool_recall']:.1%}")
    print(f"Mean precision: {summary['tool_precision']:.1%}")
    print(f"Schema valid: {summary['schema_valid_rate']:.1%}")


if __name__ == "__main__":
    main()
