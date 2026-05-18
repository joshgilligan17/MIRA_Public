"""Orchestrator agent for managing subagent-based parallel execution.

The OrchestratorAgent coordinates multiple subagents to analyze structures
in parallel, providing better scalability than the original single-plan
batch approach.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from structagent.metrics import extract_metrics_from_steps
from structagent.registry import get_registry


@dataclass
class SubagentResult:
    """Result from a single subagent execution."""

    pdb_id: str
    pdb_path: Optional[str]
    success: bool
    steps: list = field(default_factory=list)
    final_answer: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    error: Optional[str] = None
    metrics: dict = field(default_factory=dict)


class OrchestratorAgent:
    """Orchestrates multiple subagents for parallel structure analysis.

    The OrchestratorAgent manages a pool of subagents that can analyze
    structures in parallel. Each subagent is an independent MiraAgent
    instance that executes analysis for a subset of structures.

    Args:
        max_subagents: Maximum number of parallel subagents (default 4)
        model: Model name for subagents (default "MiniMax-M2.7")
        base_url: API base URL (default "https://api.minimax.io/v1")
        api_key: API key for authentication
        max_steps: Maximum steps per subagent (default 15)
        timeout: Request timeout in seconds (default 120)
        temperature: Sampling temperature (default 0.0)
        toolsets: List of toolset names to enable (None = all)
        verbose: Whether to print step logs (default False)
    """

    def __init__(
        self,
        max_subagents: int = 4,
        model: str = "MiniMax-M2.7",
        base_url: str = "https://api.minimax.io/v1",
        api_key: Optional[str] = None,
        max_steps: int = 15,
        timeout: float = 120.0,
        temperature: float = 0.0,
        toolsets: Optional[list[str]] = None,
        verbose: bool = False,
        **kwargs,
    ):
        self.max_subagents = max_subagents
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_steps = max_steps
        self.timeout = timeout
        self.temperature = temperature
        self.toolsets = toolsets
        self.verbose = verbose
        self.kwargs = kwargs
        self.registry = get_registry()

    def _create_subagent(self):
        """Create a new subagent instance.

        Returns:
            A new MiraAgent instance configured for this orchestrator.
        """
        from structagent.agent import MiraAgent

        return MiraAgent(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            max_steps=self.max_steps,
            toolsets=self.toolsets,
            verbose=self.verbose,
            mode="plan",
            timeout=self.timeout,
            temperature=self.temperature,
        )

    def _extract_metrics_from_steps(self, steps: list) -> dict:
        """Extract rankable metrics from agent steps.

        Mirrors the logic in MiraAgent._extract_metrics_from_run to ensure
        consistent metric extraction for ranking.

        Args:
            steps: List of AgentStep objects

        Returns:
            Dictionary of extracted metrics
        """
        return extract_metrics_from_steps(steps)

    def _run_single_structure(
        self,
        pdb_id: str,
        pdb_path: Optional[str],
        query: str,
        plan: dict = None,
    ) -> SubagentResult:
        """Run analysis for a single structure using a subagent.

        Args:
            pdb_id: PDB identifier
            pdb_path: Optional path to local PDB file
            query: The analysis query
            plan: Optional pre-validated plan to execute (for global plan mode)

        Returns:
            SubagentResult with analysis results
        """
        from structagent.agent import AgentStep

        agent = self._create_subagent()

        try:
            if plan is not None:
                # Use pre-validated global plan (no planning needed)
                run = agent.execute_batch_plan(plan, query, pdb_id, pdb_path)
            else:
                # Fall back to individual planning
                pdb_context = f"Analyze PDB '{pdb_id}'."
                if pdb_path:
                    pdb_context += f" Structure file: {pdb_path}"
                run = agent.run(query, context=pdb_context)

            # Extract metrics
            metrics = self._extract_metrics_from_steps(run.steps)

            return SubagentResult(
                pdb_id=pdb_id,
                pdb_path=pdb_path,
                success=True,
                steps=run.steps,
                final_answer=run.final_answer,
                total_input_tokens=run.total_input_tokens,
                total_output_tokens=run.total_output_tokens,
                metrics=metrics,
            )
        except Exception as e:
            return SubagentResult(
                pdb_id=pdb_id,
                pdb_path=pdb_path,
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
            )

    def run(
        self,
        query: str,
        pdb_ids: list[str],
        pdb_paths: list[Optional[str]],
    ) -> list[SubagentResult]:
        """Execute parallel analysis for multiple structures using subagents.

        This method:
        1. Creates ONE global plan using create_batch_plan()
        2. Distributes that plan to all subagents
        3. Each subagent executes the SAME plan via execute_batch_plan()

        Args:
            query: The analysis query
            pdb_ids: List of PDB identifiers
            pdb_paths: List of optional paths to local PDB files

        Returns:
            List of SubagentResult objects, one per structure
        """
        # Pair pdb_ids with paths
        structures = list(zip(pdb_ids, pdb_paths))

        # Create ONE global plan upfront (this is the key difference from before)
        planner_agent = self._create_subagent()
        global_plan = planner_agent.create_batch_plan(query, structures)

        if global_plan is None:
            # Fall back to individual planning if global planning fails
            global_plan = None

        # Determine number of subagents to use
        num_structures = len(structures)
        num_subagents = min(self.max_subagents, num_structures)

        # For single subagent or single structure, run directly
        if num_subagents == 1 or num_structures == 1:
            results = []
            for pdb_id, pdb_path in structures:
                result = self._run_single_structure(pdb_id, pdb_path, query, plan=global_plan)
                results.append(result)
            return results

        # Distribute structures across subagents
        # Each subagent gets roughly ceil(num_structures / num_subagents) structures
        structures_per_subagent = (num_structures + num_subagents - 1) // num_subagents

        results: list[SubagentResult] = []

        def run_batch_for_subagent(subagent_structures: list) -> list:
            """Run analysis for a batch of structures using a single subagent.

            All subagents execute the SAME global_plan.
            """
            subagent_results = []
            for pdb_id, pdb_path in subagent_structures:
                result = self._run_single_structure(pdb_id, pdb_path, query, plan=global_plan)
                subagent_results.append(result)
            return subagent_results

        # Execute in thread pool for parallel I/O-bound work
        with ThreadPoolExecutor(max_workers=num_subagents) as executor:
            # Submit batches to each subagent
            futures = []
            for i in range(num_subagents):
                start_idx = i * structures_per_subagent
                end_idx = min(start_idx + structures_per_subagent, num_structures)
                if start_idx >= num_structures:
                    break
                subagent_structures = structures[start_idx:end_idx]
                future = executor.submit(run_batch_for_subagent, subagent_structures)
                futures.append(future)

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    subagent_results = future.result()
                    results.extend(subagent_results)
                except Exception as e:
                    # If a subagent batch fails entirely, create error results
                    pass

        return results

    def run_synchronous(
        self,
        query: str,
        pdb_ids: list[str],
        pdb_paths: list[Optional[str]],
    ) -> list[SubagentResult]:
        """Synchronous wrapper for run() - provided for backward compatibility.

        Args:
            query: The analysis query
            pdb_ids: List of PDB identifiers
            pdb_paths: List of optional paths to local PDB files

        Returns:
            List of SubagentResult objects
        """
        return self.run(query, pdb_ids, pdb_paths)


def orchestrator_from_agent_config(
    agent_config: dict,
    max_subagents: int = 4,
) -> OrchestratorAgent:
    """Create an OrchestratorAgent from an agent configuration dictionary.

    Args:
        agent_config: Dictionary with agent configuration (model, base_url, api_key, etc.)
        max_subagents: Maximum number of parallel subagents

    Returns:
        Configured OrchestratorAgent instance
    """
    return OrchestratorAgent(
        max_subagents=max_subagents,
        model=agent_config.get("model", "MiniMax-M2.7"),
        base_url=agent_config.get("base_url", "https://api.minimax.io/v1"),
        api_key=agent_config.get("api_key"),
        max_steps=agent_config.get("max_steps", 15),
        timeout=agent_config.get("timeout", 120.0),
        temperature=agent_config.get("temperature", 0.0),
        toolsets=agent_config.get("toolsets"),
        verbose=agent_config.get("verbose", False),
    )
