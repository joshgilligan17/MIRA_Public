"""Batch analysis infrastructure for parallel PDB structure processing."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Optional

from .agent import AgentRun
from .metrics import RANKING_CRITERIA as DEFAULT_RANKING_CRITERIA
from .metrics import extract_metrics_from_steps


SUPPORTED_STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif"}


@dataclass
class StructureResult:
    """Result for a single structure in batch run."""

    pdb_id: str
    pdb_path: Optional[str]
    run: AgentRun
    metrics: dict  # Extracted rankable metrics
    success: bool
    error: Optional[str] = None


@dataclass
class BatchResult:
    """Complete result of batch analysis."""

    query: str
    structure_results: list[StructureResult]
    ranking: list[tuple[str, float]]  # (pdb_id, score) sorted
    ranking_criterion: str
    synthesis: str  # LLM comparative analysis
    total_wall_time: float
    total_tokens: int
    # New fields for subagent support
    orchestrator_used: bool = False  # True if OrchestratorAgent was used
    subagent_results: list = field(default_factory=list)  # Raw subagent results if available
    # Field for binder-design mode target analysis context
    target_analysis: Optional[Any] = None  # TargetAnalysisReport from Stage 1


class ResultAggregator:
    """Extracts metrics and ranks structures."""

    RANKING_CRITERIA = DEFAULT_RANKING_CRITERIA

    def __init__(self, criterion: str):
        self.criterion = criterion
        self.results: list[StructureResult] = []

    def add_result(self, result: StructureResult):
        """Add a structure result for aggregation."""
        self.results.append(result)

    def extract_metrics(self, result: StructureResult) -> dict:
        """Extract rankable metrics from StructureResult.run.steps.

        Mirrors the logic in agent._extract_metrics_from_run to ensure
        consistent metric extraction for ranking.
        """
        if result.run is None:  # Handle failed results
            return {}
        return extract_metrics_from_steps(result.run.steps)

    def get_ranking(self) -> list[tuple[str, float]]:
        """Return structures sorted by ranking criterion."""
        ranked = []
        for result in self.results:
            extracted = self.extract_metrics(result)
            criterion_info = self.RANKING_CRITERIA.get(self.criterion, {})
            metric_name = criterion_info.get("metric", self.criterion)
            value = extracted.get(metric_name)

            if value is not None:
                ranked.append((result.pdb_id, float(value)))

        criterion_info = self.RANKING_CRITERIA.get(self.criterion, {})
        higher_is_better = criterion_info.get("higher_is_better", True)
        ranked.sort(key=lambda x: x[1], reverse=higher_is_better)
        return ranked


class BatchRunner:
    """Orchestrates parallel batch execution with optional subagent support.

    The BatchRunner supports two execution modes:
    1. Legacy mode (use_subagents=False): Uses MiraAgent.create_batch_plan()
       for single-plan parallel execution
    2. Subagent mode (use_subagents=True): Uses OrchestratorAgent to spawn
       multiple subagents for parallel execution

    Args:
        agent: Optional MiraAgent instance for legacy mode
        max_workers: Maximum parallel workers (used in legacy mode)
        max_subagents: Maximum parallel subagents (used in subagent mode)
        model: Model name for subagents (used in subagent mode or if no agent provided)
        use_subagents: If True, use OrchestratorAgent for execution
        **kwargs: Additional arguments passed to OrchestratorAgent
    """

    def __init__(
        self,
        agent=None,
        max_workers: int = 4,
        max_subagents: int = 4,
        model: str = "MiniMax-M2.7",
        use_subagents: bool = False,
        **kwargs,
    ):
        self.agent = agent
        self.max_workers = max_workers
        self.max_subagents = max_subagents
        self.model = model
        self.use_subagents = use_subagents
        self.kwargs = kwargs

    def discover_pdbs(self, folder: str, glob_pattern: str = "*") -> list[tuple[str, str]]:
        """Discover PDB files in folder. Returns list of (pdb_id, pdb_path)."""
        path = Path(folder)
        pdbs = []
        for f in path.glob(glob_pattern):
            if f.suffix.lower() not in SUPPORTED_STRUCTURE_SUFFIXES:
                continue
            # Use filename stem as PDB ID
            pdb_id = f.stem.upper()
            pdbs.append((pdb_id, str(f.absolute())))
        return sorted(pdbs, key=lambda x: x[0])

    def _run_with_subagents(
        self,
        query: str,
        pdb_ids: list[str],
        pdb_paths: list[Optional[str]],
        rank_by: str,
    ) -> BatchResult:
        """Execute batch analysis using OrchestratorAgent with subagents.

        Args:
            query: The analysis query
            pdb_ids: List of PDB identifiers
            pdb_paths: List of optional paths to local PDB files
            rank_by: Ranking criterion

        Returns:
            BatchResult with analysis results
        """
        from .orchestrator import OrchestratorAgent

        start_time = time.time()

        # Create orchestrator with appropriate settings
        orchestrator = OrchestratorAgent(
            max_subagents=self.max_subagents,
            model=self.model,
            **self.kwargs,
        )

        # Run with orchestrator
        subagent_results = orchestrator.run_synchronous(query, pdb_ids, pdb_paths)

        # Convert SubagentResult to StructureResult for compatibility
        aggregator = ResultAggregator(rank_by)
        results: list[StructureResult] = []
        total_tokens = 0

        for sar in subagent_results:
            # Convert AgentSteps to AgentRun
            if sar.success:
                run = AgentRun(
                    query=query,
                    steps=sar.steps,
                    final_answer=sar.final_answer,
                    total_steps=len(sar.steps),
                    total_input_tokens=sar.total_input_tokens,
                    total_output_tokens=sar.total_output_tokens,
                    wall_time_seconds=0.0,
                    model=self.model,
                )
                total_tokens += sar.total_input_tokens + sar.total_output_tokens
            else:
                run = None

            structure_result = StructureResult(
                pdb_id=sar.pdb_id,
                pdb_path=sar.pdb_path,
                run=run,
                metrics=sar.metrics,
                success=sar.success,
                error=sar.error,
            )
            results.append(structure_result)
            aggregator.add_result(structure_result)

        # Sort results to match ranking order
        ranking_dict = dict(aggregator.get_ranking())
        results.sort(key=lambda r: ranking_dict.get(r.pdb_id, float("inf")))

        total_time = time.time() - start_time

        return BatchResult(
            query=query,
            structure_results=results,
            ranking=aggregator.get_ranking(),
            ranking_criterion=rank_by,
            synthesis="",  # Will be filled by BatchSynthesisEngine
            total_wall_time=total_time,
            total_tokens=total_tokens,
            orchestrator_used=True,
            subagent_results=subagent_results,
        )

    def run(
        self, query: str, pdb_ids: list[str], pdb_paths: list[Optional[str]], rank_by: str = "stability"
    ) -> BatchResult:
        """Execute batch analysis for all structures.

        If use_subagents=True, delegates to OrchestratorAgent for subagent-based
        parallel execution. Otherwise, uses the legacy single-plan approach
        with MiraAgent.create_batch_plan().

        Args:
            query: The analysis query
            pdb_ids: List of PDB identifiers
            pdb_paths: List of optional paths to local PDB files
            rank_by: Ranking criterion (default "stability")

        Returns:
            BatchResult with analysis results
        """
        # Use subagent mode if explicitly requested or if no agent provided
        if self.use_subagents or self.agent is None:
            return self._run_with_subagents(query, pdb_ids, pdb_paths, rank_by)

        # Legacy mode with MiraAgent
        return self._run_with_legacy_agent(query, pdb_ids, pdb_paths, rank_by)

    def _run_with_legacy_agent(
        self,
        query: str,
        pdb_ids: list[str],
        pdb_paths: list[Optional[str]],
        rank_by: str,
    ) -> BatchResult:
        """Execute batch analysis using legacy single-plan approach.

        This method:
        1. Creates ONE plan for all structures via create_batch_plan()
        2. Executes that plan in parallel for each structure via execute_batch_plan()
        3. Collects results for ranking and synthesis

        Args:
            query: The analysis query
            pdb_ids: List of PDB identifiers
            pdb_paths: List of optional paths to local PDB files
            rank_by: Ranking criterion

        Returns:
            BatchResult with analysis results
        """
        start_time = time.time()
        results: list[StructureResult] = []
        aggregator = ResultAggregator(rank_by)

        # Step 1: ONE plan for all structures
        structures = list(zip(pdb_ids, pdb_paths))
        plan = self.agent.create_batch_plan(query, structures)
        if not plan:
            raise RuntimeError("Failed to create batch plan")

        # Step 2: Parallel execution using the SAME plan
        def run_single(pdb_id: str, pdb_path: Optional[str]) -> StructureResult:
            return self.agent.execute_batch_plan(plan, query, pdb_id, pdb_path)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(run_single, pid, path): pid for pid, path in zip(pdb_ids, pdb_paths)}
            for future in as_completed(futures):
                pdb_id = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    # Create a failed StructureResult on crash
                    result = StructureResult(
                        pdb_id=pdb_id,
                        pdb_path=None,
                        run=None,
                        metrics={},
                        success=False,
                        error=f"Execution failed: {type(e).__name__}: {str(e)}",
                    )
                results.append(result)
                aggregator.add_result(result)

        # Sort results to match ranking order
        ranking_dict = dict(aggregator.get_ranking())
        results.sort(key=lambda r: ranking_dict.get(r.pdb_id, float("inf")))

        total_tokens = sum(r.run.total_input_tokens + r.run.total_output_tokens for r in results if r.success and r.run)
        total_time = time.time() - start_time

        return BatchResult(
            query=query,
            structure_results=results,
            ranking=aggregator.get_ranking(),
            ranking_criterion=rank_by,
            synthesis="",  # Will be filled by BatchSynthesisEngine
            total_wall_time=total_time,
            total_tokens=total_tokens,
            orchestrator_used=False,
            subagent_results=[],
        )


class BatchSynthesisEngine:
    """Generates comparative analysis from batch results."""

    def synthesize(self, batch_result: BatchResult, provider, model: str, temperature: float = 0.0) -> str:
        """Generate joint analysis prompt and return LLM response."""
        from .prompts import build_batch_synthesis_prompt

        prompt = build_batch_synthesis_prompt(
            batch_result.query, batch_result.structure_results, batch_result.ranking, batch_result.ranking_criterion
        )

        messages = [
            {"role": "system", "content": "You are MIRA, an expert structural biologist."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = provider.chat(messages=messages, model=model, temperature=temperature)
            return response.content or ""
        except Exception as e:
            return f"Error generating synthesis: {type(e).__name__}: {str(e)}"

    def extract_metrics_for_synthesis(self, result: StructureResult) -> dict:
        """Extract all metrics for synthesis prompt."""
        if result.run is None:
            return {"pdb_id": result.pdb_id, "success": result.success, "error": result.error, "steps": []}
        return {
            "pdb_id": result.pdb_id,
            "success": result.success,
            "error": result.error,
            "steps": [
                {
                    "tool": step.tool_name,
                    "result": step.tool_result.data[:500] if step.tool_result else "",
                    "raw": step.tool_result.raw if step.tool_result else {},
                }
                for step in result.run.steps
                if step.tool_name
            ],
        }
