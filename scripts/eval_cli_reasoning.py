"""Run a compact live reasoning eval for the MIRA CLI agent.

The eval uses the same MiraAgent and CLI configuration path as `mira chat`,
but runs programmatically so results are reproducible and easy to export.
It is intentionally small enough to run before a demo or class submission.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from structagent.agent import MiraAgent
from structagent.cli import cli, initialize_tools, resolve_api_key


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "eval-results" / "cli_reasoning_eval.json"


@dataclass(frozen=True)
class EvalCase:
    """One CLI-agent reasoning eval case."""

    name: str
    kind: str
    query: str
    required_tools: tuple[str, ...]
    answer_terms: tuple[tuple[str, ...], ...] = ()
    context: str = ""
    structures: tuple[tuple[str, str], ...] = ()
    require_local_path: bool = True


def build_cases() -> list[EvalCase]:
    """Build eval cases with absolute local fixture paths."""
    fixture_dir = REPO_ROOT / "eval-results" / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    ubq_path = fixture_dir / "local_stability_fixture.pdb"
    complex_path = fixture_dir / "local_complex_fixture.pdb"
    shutil.copyfile(REPO_ROOT / "tests" / "data" / "batch" / "1ubq.pdb", ubq_path)
    shutil.copyfile(REPO_ROOT / "tests" / "data" / "local" / "mini_complex.pdb", complex_path)
    local_context = (
        "Use the supplied local pdb_path values. Prefer pdb_path over pdb_id. "
        "Do not download structures from RCSB for these eval cases."
    )
    return [
        EvalCase(
            name="local_stability_reasoning",
            kind="run",
            query=(
                f"Using pdb_path {ubq_path}, assess chain A stability. "
                "Load the local structure, compute solvent accessibility and B-factors, "
                "then explain whether the structure looks compact or flexible."
            ),
            context=local_context,
            required_tools=("load_structure", "compute_sasa", "analyze_bfactors"),
            answer_terms=(("sasa", "solvent"), ("b-factor", "bfactor", "temperature factor"), ("chain a", "chain")),
        ),
        EvalCase(
            name="interface_evidence_reasoning",
            kind="run",
            query=(
                f"Using pdb_path {complex_path}, identify the interface between chains A and B. "
                "Compute the interface and inspect contacts around residue A5, then summarize the strongest evidence."
            ),
            context=local_context,
            required_tools=("load_structure", "compute_interface", "get_residue_contacts"),
            answer_terms=(("interface", "contact"), ("chain a", "a5", "residue 5"), ("chain b",)),
        ),
        EvalCase(
            name="quality_triage_reasoning",
            kind="run",
            query=(
                f"Using pdb_path {complex_path}, triage local structure quality for chain A. "
                "Check Ramachandran geometry and B-factors, then state any limitations of the evidence."
            ),
            context=local_context,
            required_tools=("load_structure", "check_ramachandran", "analyze_bfactors"),
            answer_terms=(("ramachandran", "geometry"), ("b-factor", "bfactor"), ("limitation", "limited")),
        ),
        EvalCase(
            name="batch_folder_plan_reasoning",
            kind="batch_plan",
            query=(
                "Create a reusable batch plan to rank local candidate structures by stability. "
                "Use local file paths, solvent accessibility, and B-factor evidence."
            ),
            context=local_context,
            required_tools=("load_structure", "compute_sasa", "analyze_bfactors"),
            structures=(("local_stability", str(ubq_path)), ("local_complex", str(complex_path))),
            answer_terms=(),
        ),
    ]


def run_cli_smoke() -> list[dict[str, Any]]:
    """Verify that the public CLI commands still parse."""
    runner = CliRunner()
    checks = [
        ("mira --help", ["--help"]),
        ("mira chat --help", ["chat", "--help"]),
        ("mira batch --help", ["batch", "--help"]),
    ]
    results = []
    for label, args in checks:
        result = runner.invoke(cli, args)
        results.append(
            {
                "command": label,
                "exit_code": result.exit_code,
                "passed": result.exit_code == 0,
                "output_excerpt": result.output[:500],
            }
        )
    return results


def run_eval_case(case: EvalCase, agent: MiraAgent) -> dict[str, Any]:
    """Run and score one eval case."""
    start = time.time()
    if case.kind == "batch_plan":
        plan = agent.create_batch_plan(case.query, [(name, path) for name, path in case.structures])
        elapsed = time.time() - start
        steps = plan.get("steps", []) if isinstance(plan, dict) else []
        tools = [step.get("tool") for step in steps if step.get("tool")]
        args_blob = json.dumps([step.get("args", {}) for step in steps], sort_keys=True)
        score = score_plan_case(case, tools, args_blob)
        return {
            "name": case.name,
            "kind": case.kind,
            "score": score["score"],
            "passed": score["passed"],
            "elapsed_seconds": round(elapsed, 3),
            "tools": tools,
            "required_tools": list(case.required_tools),
            "missing_required_tools": score["missing_required_tools"],
            "local_path_grounded": score["local_path_grounded"],
            "plan": plan,
            "notes": score["notes"],
        }

    run = agent.run(case.query, context=case.context)
    elapsed = time.time() - start
    tools = [step.tool_name for step in run.steps if step.tool_name]
    successful_tools = [
        step.tool_name for step in run.steps if step.tool_name and step.tool_result and step.tool_result.success
    ]
    failed_tools = [
        {
            "tool": step.tool_name,
            "error": step.tool_result.error if step.tool_result else None,
        }
        for step in run.steps
        if step.tool_name and step.tool_result and not step.tool_result.success
    ]
    args_blob = json.dumps([step.tool_args or {} for step in run.steps], sort_keys=True)
    score = score_run_case(case, tools, successful_tools, failed_tools, args_blob, run.final_answer)
    return {
        "name": case.name,
        "kind": case.kind,
        "score": score["score"],
        "passed": score["passed"],
        "elapsed_seconds": round(elapsed, 3),
        "tools": tools,
        "successful_tools": successful_tools,
        "failed_tools": failed_tools,
        "required_tools": list(case.required_tools),
        "missing_required_tools": score["missing_required_tools"],
        "local_path_grounded": score["local_path_grounded"],
        "answer_term_hits": score["answer_term_hits"],
        "final_answer_excerpt": run.final_answer[:1200],
        "tokens": {
            "input": run.total_input_tokens,
            "output": run.total_output_tokens,
        },
        "notes": score["notes"],
    }


def score_plan_case(case: EvalCase, tools: list[str], args_blob: str) -> dict[str, Any]:
    """Score a plan-only batch case."""
    missing = [tool for tool in case.required_tools if tool not in tools]
    coverage = (len(case.required_tools) - len(missing)) / max(len(case.required_tools), 1)
    local_path_grounded = "$PDB_PATH" in args_blob or "pdb_path" in args_blob
    score = 0.8 * coverage + 0.2 * float(local_path_grounded)
    notes = []
    if missing:
        notes.append(f"Missing required tools: {', '.join(missing)}")
    if not local_path_grounded:
        notes.append("Plan did not clearly use $PDB_PATH/pdb_path for local structures.")
    return {
        "score": round(score, 3),
        "passed": score >= 0.75,
        "missing_required_tools": missing,
        "local_path_grounded": local_path_grounded,
        "notes": notes,
    }


def score_run_case(
    case: EvalCase,
    tools: list[str],
    successful_tools: list[str],
    failed_tools: list[dict[str, Any]],
    args_blob: str,
    final_answer: str,
) -> dict[str, Any]:
    """Score an end-to-end reasoning case."""
    missing = [tool for tool in case.required_tools if tool not in tools]
    missing_success = [tool for tool in case.required_tools if tool not in successful_tools]
    tool_coverage = (len(case.required_tools) - len(missing)) / max(len(case.required_tools), 1)
    success_coverage = (len(case.required_tools) - len(missing_success)) / max(len(case.required_tools), 1)
    local_path_grounded = "pdb_path" in args_blob
    answer_text = final_answer.lower()
    answer_hits = [any(alias.lower() in answer_text for alias in aliases) for aliases in case.answer_terms]
    answer_score = sum(answer_hits) / max(len(answer_hits), 1)
    no_failures = not failed_tools
    score = (
        0.35 * tool_coverage
        + 0.25 * success_coverage
        + 0.15 * float(local_path_grounded)
        + 0.15 * answer_score
        + 0.10 * float(no_failures)
    )
    notes = []
    if missing:
        notes.append(f"Missing required tools: {', '.join(missing)}")
    if missing_success:
        notes.append(f"Required tools did not all succeed: {', '.join(missing_success)}")
    if not local_path_grounded:
        notes.append("Tool arguments did not clearly use local pdb_path.")
    if answer_score < 1.0:
        notes.append("Final answer missed one or more expected evidence terms.")
    if failed_tools:
        notes.append(f"Failed tool calls: {failed_tools}")
    return {
        "score": round(score, 3),
        "passed": score >= 0.75,
        "missing_required_tools": missing,
        "local_path_grounded": local_path_grounded,
        "answer_term_hits": answer_hits,
        "notes": notes,
    }


def resolve_agent(provider: str, model: str | None, base_url: str | None, api_key: str | None) -> MiraAgent:
    """Create a MiraAgent using the same provider resolution as the CLI."""
    resolved_api_key, resolved_base_url, resolved_model = resolve_api_key(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )
    if not resolved_api_key:
        raise RuntimeError("No API key found. Set MINIMAX_API_KEY/OPENAI_API_KEY or pass --api-key.")
    initialize_tools()
    return MiraAgent(
        model=resolved_model or model or "MiniMax-M2.7",
        base_url=resolved_base_url or base_url or "https://api.minimax.io/v1",
        api_key=resolved_api_key,
        max_steps=6,
        verbose=False,
        mode="plan",
        display="normal",
        timeout=120.0,
        temperature=0.0,
    )


def summarize(results: list[dict[str, Any]], cli_smoke: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact summary block."""
    eval_scores = [result["score"] for result in results]
    return {
        "cli_smoke_passed": all(item["passed"] for item in cli_smoke),
        "eval_cases": len(results),
        "eval_passed": sum(1 for result in results if result["passed"]),
        "mean_score": round(sum(eval_scores) / max(len(eval_scores), 1), 3),
        "min_score": round(min(eval_scores), 3) if eval_scores else 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compact live reasoning evals for the MIRA CLI agent.")
    parser.add_argument("--provider", default="minimax", choices=["minimax", "openai", "anthropic", "azure"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-live", action="store_true", help="Only run CLI command smoke checks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cli_smoke = run_cli_smoke()
    results: list[dict[str, Any]] = []

    if not args.skip_live:
        agent = resolve_agent(args.provider, args.model, args.base_url, args.api_key)
        for case in build_cases():
            results.append(run_eval_case(case, agent))

    report = {
        "created_at_unix": time.time(),
        "provider": args.provider,
        "model": args.model or "provider_default",
        "summary": summarize(results, cli_smoke),
        "cli_smoke": cli_smoke,
        "reasoning_cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
