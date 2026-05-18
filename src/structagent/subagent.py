"""Lightweight subagent for parallel batch processing.

Each SubAgent has its own isolated MiraAgent instance with no shared mutable
state, making it safe to run in parallel across multiple subprocesses or threads.
Supports both:
- Self-planned analysis via analyze_batch() (each structure gets its own plan)
- Shared folder-plan execution via run_batch() (all structures use same plan)
"""

from dataclasses import dataclass
from typing import Optional
import time

from .agent import MiraAgent
from .batch import StructureResult


@dataclass
class SubAgentResult:
    """Result from a subagent execution."""

    agent_id: str
    structure_results: list[StructureResult]  # StructureResult list
    total_tokens: int
    wall_time: float


class SubAgent:
    """Lightweight agent for parallel batch processing.

    Each SubAgent has its own isolated MiraAgent instance with no shared
    mutable state, making it safe to run in parallel across multiple
    subprocesses or threads.

    Supports two modes of operation:
    - analyze_batch(): Each structure gets its own plan via run_for_batch()
    - run_batch(): All structures execute a shared folder-level plan
    """

    def __init__(
        self,
        agent_id: str,
        model: str = "MiniMax-M2.7",
        max_steps: int = 15,
        toolsets: Optional[list[str]] = None,
        api_key: Optional[str] = None,
        base_url: str = "https://api.minimax.io/v1",
    ):
        """Initialize the subagent.

        Args:
            agent_id: Unique identifier for this subagent instance.
            model: Model name to use for chat completions.
            max_steps: Maximum ReAct loop iterations per structure.
            toolsets: List of toolset names to enable (None = all).
            api_key: API key (reads from env if not provided).
            base_url: API base URL.
        """
        self.agent_id = agent_id
        self.agent = MiraAgent(
            model=model,
            max_steps=max_steps,
            toolsets=toolsets,
            api_key=api_key,
            base_url=base_url,
            verbose=False,
        )

    async def analyze_batch(
        self,
        query: str,
        structures: list[tuple[str, Optional[str]]],
    ) -> SubAgentResult:
        """Analyze a batch of structures with isolated agent state.

        Each structure is processed sequentially within this subagent,
        but multiple subagents can run in parallel for true parallelism.
        Uses run_for_batch() which creates its own plan for each structure.

        Args:
            query: The user's query for analyzing the structures.
            structures: List of (pdb_id, pdb_path) tuples to analyze.

        Returns:
            SubAgentResult containing all structure results and metrics.
        """
        import asyncio

        start_time = time.time()
        structure_results: list[StructureResult] = []

        # Use asyncio.gather() for concurrent processing of all structures
        import asyncio

        async def process_one(pdb_id: str, pdb_path: Optional[str]) -> StructureResult:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._run_single_structure,
                query,
                pdb_id,
                pdb_path,
            )

        # Gather all results concurrently
        structure_results = await asyncio.gather(*[process_one(pdb_id, pdb_path) for pdb_id, pdb_path in structures])

        wall_time = time.time() - start_time

        # Sum tokens across all structure results
        total_tokens = sum(r.run.total_input_tokens + r.run.total_output_tokens for r in structure_results if r.success)

        return SubAgentResult(
            agent_id=self.agent_id,
            structure_results=structure_results,
            total_tokens=total_tokens,
            wall_time=wall_time,
        )

    async def run_batch(
        self,
        query: str,
        folder_plan: dict,
        structures: list[tuple[str, Optional[str]]],
    ) -> SubAgentResult:
        """Execute a shared folder-level plan on a batch of structures.

        All structures in this batch use the same pre-validated folder_plan
        for execution, which is more efficient when all binders need the same
        analysis pipeline.

        Args:
            query: The user's query for analyzing the structures.
            folder_plan: Pre-validated folder-level plan dict to execute.
            structures: List of (pdb_id, pdb_path) tuples to analyze.

        Returns:
            SubAgentResult containing all structure results and metrics.
        """
        import asyncio

        start_time = time.time()
        structure_results: list[StructureResult] = []

        # Use asyncio.gather() for concurrent processing of all structures
        import asyncio

        async def process_one(pdb_id: str, pdb_path: Optional[str]) -> StructureResult:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._execute_single_with_plan,
                query,
                folder_plan,
                pdb_id,
                pdb_path,
            )

        # Gather all results concurrently
        structure_results = await asyncio.gather(*[process_one(pdb_id, pdb_path) for pdb_id, pdb_path in structures])

        wall_time = time.time() - start_time

        # Sum tokens across all structure results
        total_tokens = sum(r.run.total_input_tokens + r.run.total_output_tokens for r in structure_results if r.success)

        return SubAgentResult(
            agent_id=self.agent_id,
            structure_results=structure_results,
            total_tokens=total_tokens,
            wall_time=wall_time,
        )

    def _run_single_structure(
        self,
        query: str,
        pdb_id: str,
        pdb_path: Optional[str],
    ) -> StructureResult:
        """Run agent analysis for a single structure using self-planned approach.

        Args:
            query: The user's query.
            pdb_id: PDB identifier.
            pdb_path: Optional path to local PDB file.

        Returns:
            StructureResult for this structure.
        """
        return self.agent.run_for_batch(
            query=query,
            pdb_id=pdb_id,
            pdb_path=pdb_path,
        )

    def _execute_single_with_plan(
        self,
        query: str,
        folder_plan: dict,
        pdb_id: str,
        pdb_path: Optional[str],
    ) -> StructureResult:
        """Execute folder plan for a single structure.

        Args:
            query: The user's query.
            folder_plan: Pre-validated folder-level plan to execute.
            pdb_id: PDB identifier.
            pdb_path: Optional path to local PDB file.

        Returns:
            StructureResult for this structure.
        """
        return self.agent.execute_batch_plan(
            plan=folder_plan,
            query=query,
            pdb_id=pdb_id,
            pdb_path=pdb_path,
        )
