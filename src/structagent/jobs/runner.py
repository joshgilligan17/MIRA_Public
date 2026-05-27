"""Execution engine for persisted local batch jobs."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import quote

from structagent.batch import ResultAggregator, SUPPORTED_STRUCTURE_SUFFIXES
from structagent.jobs.models import JobConfig
from structagent.jobs.store import JobStore
from structagent.providers import create_provider
from structagent.profiles import ProfileRunResult, run_analysis_profile

PROVIDER_DEFAULTS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "env_vars": ("OPENAI_API_KEY",),
    },
    "minimax": {
        "base_url": "https://api.minimax.io/v1",
        "model": "MiniMax-M2.7",
        "env_vars": ("MINIMAX_API_KEY",),
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "model": "claude-3-5-haiku-20241022",
        "env_vars": ("ANTHROPIC_API_KEY",),
    },
    "azure": {
        "base_url": None,
        "model": None,
        "env_vars": ("AZURE_OPENAI_KEY",),
    },
}


class JobRunner:
    """Runs queued jobs from a JobStore."""

    def __init__(self, store: JobStore | None = None):
        self.store = store or JobStore()

    def create_job(self, config: JobConfig) -> str:
        return self.store.create_job(config).id

    def run_job(self, job_id: str, llm_api_key: str | None = None) -> None:
        record = self.store.get_record(job_id)
        structures = _discover_structures(self.store.input_dir(job_id), record.config.glob_pattern)
        total_count = len(structures)
        self.store.update_record(job_id, total_count=total_count, completed_count=0, failed_count=0)

        if not structures:
            self.store.save_results(job_id, _empty_results(record.to_dict()))
            self.store.save_report(job_id, _build_report(record.to_dict(), [], []))
            self.store.set_status(job_id, "failed", "No structure files found.", error="No structure files found")
            return

        self.store.set_status(job_id, "running", f"Analyzing {total_count} structure file(s).")
        profile_results: list[ProfileRunResult] = []
        failed_count = 0

        try:
            max_workers = max(1, record.config.max_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._run_structure, record.config, pdb_id, pdb_path): (pdb_id, pdb_path)
                    for pdb_id, pdb_path in structures
                }
                for future in as_completed(futures):
                    pdb_id, _ = futures[future]
                    try:
                        result = future.result()
                        profile_results.append(result)
                        if result.structure_result.success:
                            message = f"{pdb_id} completed."
                            event_type = "structure_completed"
                        else:
                            failed_count += 1
                            message = f"{pdb_id} failed: {result.structure_result.error or 'unknown error'}"
                            event_type = "structure_failed"
                        completed_count = len(profile_results)
                        self.store.update_record(job_id, completed_count=completed_count, failed_count=failed_count)
                        self.store.append_event(
                            job_id,
                            event_type,
                            message,
                            structure_id=pdb_id,
                            completed_count=completed_count,
                            total_count=total_count,
                        )
                    except Exception as exc:  # pragma: no cover - defensive guard for worker crashes
                        failed_count += 1
                        completed_count = len(profile_results) + failed_count
                        self.store.update_record(job_id, completed_count=completed_count, failed_count=failed_count)
                        self.store.append_event(
                            job_id,
                            "structure_failed",
                            f"{pdb_id} crashed: {type(exc).__name__}: {exc}",
                            structure_id=pdb_id,
                            completed_count=completed_count,
                            total_count=total_count,
                        )

            ranking = _rank_results(profile_results, record.config.rank_by)
            structures_json = [
                _serialize_profile_result(item, job_id) for item in _sort_by_ranking(profile_results, ranking)
            ]
            results = {
                "job": self.store.get_record(job_id).to_dict(),
                "summary": _summary(structures_json, ranking, record.config.rank_by),
                "ranking": [
                    {"rank": index, "pdb_id": pdb_id, "score": score}
                    for index, (pdb_id, score) in enumerate(ranking, start=1)
                ],
                "structures": structures_json,
            }
            synthesis = _maybe_generate_llm_synthesis(results["job"], results["ranking"], structures_json, llm_api_key)
            results["report_synthesis"] = {
                "mode": synthesis["mode"],
                "provider": synthesis.get("provider"),
                "model": synthesis.get("model"),
                "error": synthesis.get("error"),
            }
            self.store.save_results(job_id, results)
            self.store.save_report(
                job_id,
                _build_report(
                    results["job"],
                    results["ranking"],
                    structures_json,
                    synthesis_lines=synthesis.get("lines"),
                    synthesis_meta=results["report_synthesis"],
                ),
            )
            if synthesis["mode"] == "llm":
                self.store.append_event(job_id, "llm_synthesis_completed", "Generated LLM synthesis.")
            elif synthesis.get("error"):
                self.store.append_event(
                    job_id,
                    "llm_synthesis_fallback",
                    f"Used deterministic report synthesis: {synthesis['error']}",
                )
            self.store.set_status(job_id, "completed", "Job completed.")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.store.set_status(job_id, "failed", f"Job failed: {message}", error=message)
            raise

    def _run_structure(self, config: JobConfig, pdb_id: str, pdb_path: str) -> ProfileRunResult:
        return run_analysis_profile(
            pdb_id=pdb_id,
            pdb_path=pdb_path,
            query=config.query,
            profile=config.profile,
            chain_a=config.chain_a,
            chain_b=config.chain_b,
        )


def _discover_structures(input_dir: Path, glob_pattern: str = "*") -> list[tuple[str, str]]:
    if glob_pattern and glob_pattern not in {"*", "*.*"}:
        candidates = input_dir.glob(glob_pattern)
    else:
        candidates = input_dir.rglob("*")

    structures = []
    seen: dict[str, int] = {}
    for path in sorted(candidates):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_STRUCTURE_SUFFIXES:
            continue
        pdb_id = path.stem.upper()
        count = seen.get(pdb_id, 0) + 1
        seen[pdb_id] = count
        if count > 1:
            pdb_id = f"{pdb_id}_{count}"
        structures.append((pdb_id, str(path.resolve())))
    return structures


def _rank_results(results: list[ProfileRunResult], rank_by: str) -> list[tuple[str, float]]:
    aggregator = ResultAggregator(rank_by)
    for result in results:
        aggregator.add_result(result.structure_result)
    return aggregator.get_ranking()


def _sort_by_ranking(results: list[ProfileRunResult], ranking: list[tuple[str, float]]) -> list[ProfileRunResult]:
    ranking_order = {pdb_id: index for index, (pdb_id, _) in enumerate(ranking)}
    return sorted(results, key=lambda result: ranking_order.get(result.structure_result.pdb_id, len(ranking_order)))


def _serialize_profile_result(result: ProfileRunResult, job_id: str) -> dict[str, Any]:
    structure = result.structure_result
    run = structure.run
    path = Path(structure.pdb_path) if structure.pdb_path else None
    return {
        "id": structure.pdb_id,
        "pdb_id": structure.pdb_id,
        "filename": path.name if path else structure.pdb_id,
        "source_path": str(path) if path else None,
        "job_id": job_id,
        "success": structure.success,
        "error": structure.error,
        "profile": result.profile,
        "chains": result.chains,
        "metrics": structure.metrics,
        "features": result.features,
        "warnings": result.warnings,
        "summary": run.final_answer if run else "",
        "structure_url": f"/api/jobs/{job_id}/structures/{structure.pdb_id}",
        "steps": [_serialize_step(step) for step in run.steps] if run else [],
    }


def _serialize_step(step) -> dict[str, Any]:
    result = step.tool_result
    return {
        "thought": step.thought,
        "tool_name": step.tool_name,
        "tool_args": step.tool_args,
        "success": result.success if result else None,
        "data": result.data if result else None,
        "raw": result.raw if result else None,
        "error": result.error if result else None,
        "execution_time_seconds": result.execution_time_seconds if result else None,
    }


def _summary(structures: list[dict[str, Any]], ranking: list[tuple[str, float]], rank_by: str) -> dict[str, Any]:
    successful = [item for item in structures if item["success"]]
    failed = [item for item in structures if not item["success"]]
    return {
        "structure_count": len(structures),
        "successful_count": len(successful),
        "failed_count": len(failed),
        "rank_by": rank_by,
        "top_structure": ranking[0][0] if ranking else None,
        "top_score": ranking[0][1] if ranking else None,
    }


def _empty_results(job: dict[str, Any]) -> dict[str, Any]:
    return {"job": job, "summary": _summary([], [], job["config"]["rank_by"]), "ranking": [], "structures": []}


def _build_report(
    job: dict[str, Any],
    ranking: list[dict[str, Any]],
    structures: list[dict[str, Any]],
    synthesis_lines: list[str] | None = None,
    synthesis_meta: dict[str, Any] | None = None,
) -> str:
    config = job["config"]
    lines = [
        "# MIRA Batch Report",
        "",
        f"- Job: `{job['id']}`",
        f"- Query: {config['query']}",
        f"- Profile: `{config['profile']}`",
        f"- Ranking criterion: `{config['rank_by']}`",
        f"- Synthesis mode: {_synthesis_mode_label(synthesis_meta)}",
        "",
        *(synthesis_lines or _build_screening_synthesis(job, ranking, structures)),
        "",
        "## Ranking",
        "",
    ]
    if ranking:
        lines.extend(["| Rank | Structure | Score |", "| ---: | --- | ---: |"])
        for row in ranking:
            lines.append(f"| {row['rank']} | {row['pdb_id']} | {row['score']:.3f} |")
    else:
        lines.append("No rankable structures were produced.")

    lines.extend(["", "## Structure Summaries", ""])
    for item in structures:
        status = "success" if item["success"] else f"failed: {item['error']}"
        lines.extend([f"### {item['pdb_id']}", "", f"- Status: {status}"])
        lines.extend(_structure_interpretation_lines(item, ranking, config["rank_by"]))
        evidence_lines = _evidence_report_lines(item)
        if evidence_lines:
            lines.extend(["", "#### Referenced Regions", "", *evidence_lines])
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _synthesis_mode_label(synthesis_meta: dict[str, Any] | None) -> str:
    if not synthesis_meta:
        return "deterministic"
    if synthesis_meta.get("mode") == "llm":
        provider = synthesis_meta.get("provider") or "provider"
        model = synthesis_meta.get("model") or "model"
        return f"LLM ({provider}/{model})"
    if synthesis_meta.get("error"):
        return f"deterministic fallback ({synthesis_meta['error']})"
    return "deterministic"


def _build_screening_synthesis(
    job: dict[str, Any], ranking: list[dict[str, Any]], structures: list[dict[str, Any]]
) -> list[str]:
    config = job["config"]
    successful = [item for item in structures if item.get("success")]
    failed = [item for item in structures if not item.get("success")]
    target_context = config["query"].strip() or "No explicit target context was supplied."
    rank_by = config["rank_by"]
    lines = [
        "## Synthesis",
        "",
        "### Target Context",
        "",
        (
            "The screening target and design intent are inferred from the user query: "
            f'"{target_context}". MIRA treats each uploaded structure as a candidate design or complex and '
            "uses local structural evidence to decide which candidates should move forward for closer review."
        ),
        "",
        "### Design and Filtering Strategy",
        "",
        (
            "The primary strategy is a structural triage screen: prioritize candidates with interpretable ranking "
            f"signal for {_criterion_description(rank_by)}, then cross-check that rank against interface size, "
            "surface exposure, B-factor flexibility, charge clustering, and backbone geometry. This gives a compact "
            "read on whether a design looks well packed, interface-competent, and free of obvious local liabilities."
        ),
    ]

    if not successful:
        lines.extend(
            [
                "",
                "### Batch Outcome",
                "",
                "No structures completed successfully, so no design-level synthesis could be generated.",
            ]
        )
        return lines

    top = _structure_by_id(structures, ranking[0]["pdb_id"]) if ranking else successful[0]
    top_metrics = top.get("metrics") or {}
    lines.extend(
        [
            "",
            "### Batch Outcome",
            "",
            (
                f"{len(successful)} of {len(structures)} structures completed successfully"
                + (f"; {len(failed)} failed during analysis" if failed else "")
                + ". "
                + _top_structure_sentence(top, ranking[0] if ranking else None, rank_by)
            ),
            "",
            "### Attribute-Level Interpretation",
            "",
            *_batch_attribute_summary(successful),
        ]
    )

    if top_metrics:
        lines.extend(
            [
                "",
                "### Lead Candidate Rationale",
                "",
                (
                    f"`{top['pdb_id']}` is the current lead because it best satisfies the selected ranking criterion "
                    f"while retaining supporting structural evidence: {_metric_clause(top_metrics)}."
                ),
            ]
        )

    return lines


def _maybe_generate_llm_synthesis(
    job: dict[str, Any],
    ranking: list[dict[str, Any]],
    structures: list[dict[str, Any]],
    api_key_override: str | None = None,
) -> dict[str, Any]:
    config = JobConfig(**job["config"])
    if not config.enable_llm_synthesis:
        return {"mode": "deterministic", "lines": None, "error": None}

    provider_name, model, base_url, api_key = _resolve_llm_config(config, api_key_override)
    if not api_key:
        return {"mode": "deterministic", "lines": None, "error": "no API key configured"}
    if not model:
        return {"mode": "deterministic", "lines": None, "error": "no model configured"}

    try:
        provider = create_provider(
            provider_name,
            api_key=api_key,
            base_url=base_url,
            timeout=120.0,
            temperature=config.llm_temperature,
        )
        response = provider.chat(
            messages=_llm_synthesis_messages(job, ranking, structures),
            model=model,
            temperature=config.llm_temperature,
        )
        lines = _normalize_llm_markdown(response.content)
        if not lines:
            raise ValueError("empty synthesis returned")
        return {"mode": "llm", "provider": provider_name, "model": model, "lines": lines, "error": None}
    except Exception as exc:
        return {
            "mode": "deterministic",
            "provider": provider_name,
            "model": model,
            "lines": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _resolve_llm_config(
    config: JobConfig, api_key_override: str | None
) -> tuple[str, str | None, str | None, str | None]:
    provider_name = config.llm_provider or os.getenv("MIRA_REPORT_PROVIDER") or _infer_provider_from_env()
    provider_name = provider_name.lower()
    defaults = PROVIDER_DEFAULTS.get(provider_name, PROVIDER_DEFAULTS["openai"])
    model = config.llm_model or os.getenv("MIRA_REPORT_MODEL") or defaults["model"]
    base_url = config.llm_base_url or os.getenv("MIRA_REPORT_BASE_URL") or defaults["base_url"]
    api_key = api_key_override or os.getenv("MIRA_REPORT_API_KEY")
    if not api_key:
        for env_var in defaults["env_vars"]:
            api_key = os.getenv(env_var)
            if api_key:
                break
    return provider_name, model, base_url, api_key


def _infer_provider_from_env() -> str:
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("MINIMAX_API_KEY"):
        return "minimax"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("AZURE_OPENAI_KEY"):
        return "azure"
    return "openai"


def _llm_synthesis_messages(
    job: dict[str, Any], ranking: list[dict[str, Any]], structures: list[dict[str, Any]]
) -> list[dict[str, str]]:
    config = job["config"]
    context = {
        "query": config["query"],
        "profile": config["profile"],
        "ranking_criterion": config["rank_by"],
        "ranking_criterion_description": _criterion_description(config["rank_by"]),
        "ranking": ranking,
        "structures": [_structure_context_for_llm(item, ranking) for item in structures[:8]],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are MIRA, a molecular structure reasoning agent writing a grounded screening report. "
                "Use only the supplied metrics and evidence links. Do not invent targets, residues, energies, "
                "affinities, wet-lab claims, or unsupported biological mechanisms. If you mention a residue or "
                "region, copy one of the supplied markdown evidence links exactly so the UI can highlight it. "
                "Do not describe buried surface area as binding affinity or thermodynamic favorability. "
                "Do not claim statistical significance, validation, thresholds, or functional outcomes unless "
                "provided in the context. Do not include hidden reasoning, analysis, or <think> blocks. "
                "Do not mention execution time. Use 'Ramachandran outliers' instead of 'violations'. "
                "Avoid the forbidden terms: affinity, thermodynamic, statistically, threshold, validation. "
                "Write concise scientific prose for a de novo protein-structure filtering workflow."
            ),
        },
        {
            "role": "user",
            "content": (
                "Write the top-level synthesis section for this report. Return markdown only, no code fence. "
                "Use exactly these headings in this order:\n"
                "## Synthesis\n"
                "### Target Context\n"
                "### Design and Filtering Strategy\n"
                "### Batch Outcome\n"
                "### Attribute-Level Interpretation\n"
                "### Lead Candidate Rationale\n\n"
                "Context JSON:\n"
                f"{json.dumps(context, indent=2, sort_keys=True)}"
            ),
        },
    ]


def _structure_context_for_llm(item: dict[str, Any], ranking: list[dict[str, Any]]) -> dict[str, Any]:
    rank = next((row for row in ranking if row["pdb_id"] == item["pdb_id"]), None)
    evidence = {
        "interface": _first_refs(item, "interface_residues", 5),
        "hotspots": _first_refs(item, "hotspots", 4),
        "flexible": _first_refs(item, "high_bfactor_residues", 4),
        "geometry_outliers": _first_refs(item, "ramachandran_outliers", 4),
        "charge_clusters": _first_charge_refs(item, 4),
    }
    return {
        "pdb_id": item["pdb_id"],
        "rank": rank,
        "success": item["success"],
        "error": item.get("error"),
        "metrics": _llm_metrics(item.get("metrics") or {}),
        "evidence_links": evidence,
    }


def _llm_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "total_execution_time"}


def _normalize_llm_markdown(markdown: str) -> list[str]:
    content = re.sub(r"<think>.*?</think>", "", markdown, flags=re.DOTALL | re.IGNORECASE).strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("markdown"):
            content = content[len("markdown") :].strip()
    synthesis_start = content.find("## Synthesis")
    if synthesis_start > 0:
        content = content[synthesis_start:].strip()
    lines = [line.rstrip() for line in content.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].strip() != "## Synthesis":
        lines = ["## Synthesis", "", *lines]
    return lines


def _structure_interpretation_lines(item: dict[str, Any], ranking: list[dict[str, Any]], rank_by: str) -> list[str]:
    metrics = item.get("metrics") or {}
    rank = next((row for row in ranking if row["pdb_id"] == item["pdb_id"]), None)
    lines = ["", "#### Synthesis", ""]
    if not item.get("success"):
        return [*lines, f"This structure did not complete successfully: {item.get('error') or 'unknown error'}."]

    rank_text = f"ranked #{rank['rank']} by {_criterion_description(rank_by)}" if rank else "was not rankable"
    lines.append(f"`{item['pdb_id']}` {rank_text}. Its main structural signals are {_metric_clause(metrics)}.")

    interface_refs = _first_refs(item, "interface_residues", limit=4)
    hotspot_refs = _first_refs(item, "hotspots", limit=3)
    flexible_refs = _first_refs(item, "high_bfactor_residues", limit=3)
    geometry_refs = _first_refs(item, "ramachandran_outliers", limit=3)
    charge_refs = _first_charge_refs(item, limit=4)

    lines.extend(["", "#### Specific Protein Attributes", ""])
    lines.append(
        "- Interface: "
        + _metric_sentence(metrics, "buried_surface_area", "buried surface area", "A^2")
        + "; "
        + _metric_sentence(metrics, "n_interface_residues", "interface residues")
        + (f". Referenced residues: {', '.join(interface_refs)}" if interface_refs else ".")
    )
    lines.append(
        "- Packing/stability: "
        + _metric_sentence(metrics, "mean_relative_sasa_percent", "mean relative SASA", "%")
        + "; "
        + _metric_sentence(metrics, "n_buried", "buried residues")
        + "; "
        + _metric_sentence(metrics, "n_exposed", "exposed residues")
        + "."
    )
    lines.append(
        "- Flexibility and geometry: "
        + _metric_sentence(metrics, "mean_bfactor", "mean B-factor")
        + "; "
        + _metric_sentence(metrics, "std_bfactor", "B-factor spread")
        + (f". Flexible residues: {', '.join(flexible_refs)}" if flexible_refs else ".")
        + (f" Geometry outliers: {', '.join(geometry_refs)}." if geometry_refs else "")
    )
    lines.append(
        "- Electrostatics: "
        + _metric_sentence(metrics, "total_charge", "net charge")
        + "; "
        + _metric_sentence(metrics, "charge_cluster_count", "charge clusters")
        + (f". Charge-region examples: {', '.join(charge_refs)}" if charge_refs else ".")
    )
    if hotspot_refs:
        lines.append(f"- Hotspot candidates: {', '.join(hotspot_refs)}.")
    if item.get("warnings"):
        lines.append("- Warnings: " + " ".join(item["warnings"]))
    return lines


def _evidence_report_lines(item: dict[str, Any]) -> list[str]:
    features = item.get("features") or {}
    sections = [
        ("interface_residues", "Interface residues"),
        ("hotspots", "Hotspots"),
        ("high_bfactor_residues", "Flexible/high B-factor residues"),
        ("ramachandran_outliers", "Ramachandran outliers"),
        ("buried_residues", "Buried residues"),
        ("exposed_residues", "Exposed residues"),
    ]
    lines: list[str] = []
    for key, label in sections:
        refs = [_region_link(key, feature) for feature in (features.get(key) or [])[:8]]
        refs = _dedupe([ref for ref in refs if ref])
        if refs:
            lines.append(f"- {label}: {', '.join(refs)}")

    cluster_refs = []
    for cluster in (features.get("charge_clusters") or [])[:3]:
        for residue in (cluster.get("residues") or [])[:4]:
            link = _region_link("charge_clusters", residue)
            if link:
                cluster_refs.append(link)
    if cluster_refs:
        lines.append(f"- Charge clusters: {', '.join(_dedupe(cluster_refs))}")
    return lines


def _batch_attribute_summary(successful: list[dict[str, Any]]) -> list[str]:
    metrics = [item.get("metrics") or {} for item in successful]
    bsa_values = [value for value in (_metric(item, "buried_surface_area") for item in metrics) if value is not None]
    interface_values = [
        value for value in (_metric(item, "n_interface_residues") for item in metrics) if value is not None
    ]
    sasa_values = [
        value for value in (_metric(item, "mean_relative_sasa_percent") for item in metrics) if value is not None
    ]
    bfactor_values = [value for value in (_metric(item, "mean_bfactor") for item in metrics) if value is not None]
    charge_values = [
        value for value in (_metric(item, "charge_cluster_count") for item in metrics) if value is not None
    ]

    lines = []
    if bsa_values or interface_values:
        lines.append(
            "- Interface evidence: "
            + _range_phrase(bsa_values, "buried surface area", "A^2")
            + "; "
            + _range_phrase(interface_values, "interface residue count")
            + "."
        )
    if sasa_values:
        lines.append(
            "- Surface/packing evidence: "
            + _range_phrase(sasa_values, "mean relative SASA", "%")
            + ". Lower values indicate candidates with more buried/compact surface character under this profile."
        )
    if bfactor_values:
        lines.append(
            "- Flexibility evidence: "
            + _range_phrase(bfactor_values, "mean B-factor")
            + ". Lower means are preferred when the goal is rigid, well-ordered designs."
        )
    if charge_values:
        lines.append(
            "- Electrostatic evidence: "
            + _range_phrase(charge_values, "charge cluster count")
            + ". Charge clusters are highlighted for follow-up because they can indicate designed polar patches or liabilities."
        )
    return lines or ["- No rankable metric distributions were available."]


def _top_structure_sentence(top: dict[str, Any], rank: dict[str, Any] | None, rank_by: str) -> str:
    score_text = f" with score {_fmt(rank['score'])}" if rank and rank.get("score") is not None else ""
    return (
        f"The leading structure is `{top['pdb_id']}`{score_text} under {_criterion_description(rank_by)}. "
        f"Its supporting attributes are {_metric_clause(top.get('metrics') or {})}."
    )


def _metric_clause(metrics: dict[str, Any]) -> str:
    clauses = []
    for key, label, suffix in [
        ("buried_surface_area", "buried surface area", " A^2"),
        ("n_interface_residues", "interface residues", ""),
        ("mean_relative_sasa_percent", "mean relative SASA", "%"),
        ("mean_bfactor", "mean B-factor", ""),
        ("charge_cluster_count", "charge clusters", ""),
    ]:
        value = _metric(metrics, key)
        if value is not None:
            clauses.append(f"{label} {_fmt(value)}{suffix}")
    return ", ".join(clauses) if clauses else "no rankable metrics available"


def _metric_sentence(metrics: dict[str, Any], key: str, label: str, suffix: str = "") -> str:
    value = _metric(metrics, key)
    if value is None:
        return f"{label} not available"
    return f"{label} {_fmt(value)}{_unit(suffix)}"


def _range_phrase(values: list[float], label: str, suffix: str = "") -> str:
    if not values:
        return f"{label} not available"
    unit = _unit(suffix)
    return f"{label} {_fmt(min(values))}{unit}-{_fmt(max(values))}{unit} (mean {_fmt(mean(values))}{unit})"


def _criterion_description(rank_by: str) -> str:
    descriptions = {
        "stability": "stability/compactness",
        "buried_surface_area": "buried interface surface area",
        "n_interface_residues": "interface residue count",
        "mean_bfactor": "low B-factor flexibility",
        "std_bfactor": "uniform B-factor profile",
        "n_buried": "buried residue count",
        "n_exposed": "low exposed residue count",
        "interface_energy": "interface energy",
        "shape_complementarity": "shape complementarity",
        "packstat": "packing quality",
    }
    return descriptions.get(rank_by, rank_by.replace("_", " "))


def _structure_by_id(structures: list[dict[str, Any]], pdb_id: str) -> dict[str, Any]:
    return next((item for item in structures if item["pdb_id"] == pdb_id), structures[0])


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _unit(suffix: str) -> str:
    if not suffix:
        return ""
    if suffix == "%":
        return suffix
    return f" {suffix}"


def _first_refs(item: dict[str, Any], evidence_key: str, limit: int) -> list[str]:
    refs = [
        _region_link(evidence_key, feature) for feature in (item.get("features") or {}).get(evidence_key, [])[:limit]
    ]
    return _dedupe([ref for ref in refs if ref])


def _first_charge_refs(item: dict[str, Any], limit: int) -> list[str]:
    refs = []
    for cluster in (item.get("features") or {}).get("charge_clusters", []):
        for residue in cluster.get("residues") or []:
            link = _region_link("charge_clusters", residue)
            if link and link not in refs:
                refs.append(link)
            if len(refs) >= limit:
                return refs
    return refs


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _region_link(evidence_key: str, feature: dict[str, Any]) -> str | None:
    residue_number = feature.get("residue_number")
    if residue_number is None:
        return None
    chain = feature.get("chain") or "any"
    residue_name = feature.get("residue_name") or "Residue"
    label = f"{residue_name}-{residue_number}"
    if chain != "any":
        label = f"{label} chain {chain}"
    href = f"mira://region/{quote(evidence_key)}/{quote(str(chain))}/{quote(str(residue_number))}"
    return f"[{label}]({href})"
