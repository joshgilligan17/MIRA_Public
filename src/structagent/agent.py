"""MIRA: ReAct agent loop for structural biology reasoning."""

import json
import re
import time
import importlib.util
import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Any, Tuple, List
from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.markup import escape

from structagent.registry import get_registry, ToolResult
from structagent.prompts import build_system_prompt, build_planning_prompt
from structagent.tool_metadata import get_tool_schemas_for_planning, build_compact_tool_list
from structagent.display import get_display_strategy, DisplayStrategy
from structagent.metrics import extract_metrics_from_steps

if TYPE_CHECKING:
    from structagent.batch import StructureResult


console = Console()


@dataclass
class AgentStep:
    """A single step in an agent run."""

    thought: Optional[str] = None  # The model's reasoning
    tool_name: Optional[str] = None  # Tool called (None if final)
    tool_args: Optional[dict] = None  # Arguments passed
    tool_result: Optional[ToolResult] = None  # Result from registry.call_tool
    is_final: bool = False  # True if this is the final answer step
    timestamp: float = 0.0  # Unix timestamp


@dataclass
class AgentRun:
    """Complete record of an agent execution."""

    query: str  # Original user query
    steps: list[AgentStep] = field(default_factory=list)  # All steps taken
    final_answer: str = ""  # The final text response
    total_steps: int = 0  # Number of steps taken
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    wall_time_seconds: float = 0.0
    model: str = ""

    def to_dict(self) -> dict:
        """Serialize for saving trajectories to JSON."""
        return {
            "query": self.query,
            "final_answer": self.final_answer,
            "total_steps": self.total_steps,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "wall_time_seconds": self.wall_time_seconds,
            "model": self.model,
            "steps": [
                {
                    "thought": s.thought,
                    "tool_name": s.tool_name,
                    "tool_args": s.tool_args,
                    "tool_result": s.tool_result.data if s.tool_result else None,
                    "is_final": s.is_final,
                }
                for s in self.steps
            ],
        }


class MiraAgent:
    """MIRA - ReAct agent for structural biology reasoning."""

    def __init__(
        self,
        model: str = "MiniMax-M2.7",
        base_url: str = "https://api.minimax.io/v1",
        max_steps: int = 15,
        toolsets: Optional[list[str]] = None,
        api_key: Optional[str] = None,
        verbose: bool = True,
        mode: str = "plan",
        display: str = "verbose",
        timeout: float = 120.0,
        temperature: float = 0.0,
    ):
        """Initialize the agent.

        Args:
            model: Model name to use for chat completions.
            base_url: API base URL.
            max_steps: Maximum ReAct loop iterations.
            toolsets: List of toolset names to enable (None = all).
            api_key: API key (reads from env if not provided).
            verbose: Whether to print step logs to console.
            mode: Execution mode - "plan" for planning-first, "react" for pure ReAct.
            display: Display mode - "normal" for compact, "verbose" for detailed.
            timeout: Request timeout in seconds (default 120).
            temperature: Sampling temperature for responses (default 0.0).
        """
        self.model = model
        self.base_url = base_url
        self.max_steps = max_steps
        self.toolsets = toolsets
        self.verbose = verbose
        self.mode = mode
        self.timeout = timeout
        self.temperature = temperature
        self.display_strategy: DisplayStrategy = get_display_strategy(display)

        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

        self.registry = get_registry()
        self._steps: list[AgentStep] = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._last_chain_info: Optional[dict] = None  # For chain validation

    def run(self, query: str, context: Optional[str] = None) -> AgentRun:
        """Execute a single query through the agent loop.

        In "plan" mode (default): Uses planning-first execution with adaptation.
        In "react" mode: Uses pure ReAct loop (backwards compatible).

        Args:
            query: The user's question.
            context: Optional additional context for the system prompt.

        Returns:
            AgentRun with all steps and the final answer.
        """
        if self.mode == "react":
            return self.run_react(query, context)

        start_time = time.time()
        self._steps = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._last_chain_info = None  # Reset chain info for each run

        # PHASE 1: Planning
        plan = self._create_plan(query, context)
        if not plan:
            final_answer = self._synthesize("Could not create plan", query, context)
            wall_time = time.time() - start_time
            return AgentRun(
                query=query,
                steps=self._steps,
                final_answer=final_answer,
                total_steps=len(self._steps),
                total_input_tokens=self._total_input_tokens,
                total_output_tokens=self._total_output_tokens,
                wall_time_seconds=wall_time,
                model=self.model,
            )

        # PHASE 2: Validate (max 2 revisions)
        validated_plan = self._validate_plan(plan, max_revisions=2)
        if not validated_plan:
            final_answer = self._synthesize("Could not validate plan", query, context)
            wall_time = time.time() - start_time
            return AgentRun(
                query=query,
                steps=self._steps,
                final_answer=final_answer,
                total_steps=len(self._steps),
                total_input_tokens=self._total_input_tokens,
                total_output_tokens=self._total_output_tokens,
                wall_time_seconds=wall_time,
                model=self.model,
            )

        # PHASE 3: Execute with adaptation
        results = self._execute_with_adaptation(validated_plan, query, context)

        # PHASE 4: Synthesize
        final_answer = self._synthesize(results, query, context)
        wall_time = time.time() - start_time

        return AgentRun(
            query=query,
            steps=self._steps,
            final_answer=final_answer,
            total_steps=len(self._steps),
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            wall_time_seconds=wall_time,
            model=self.model,
        )

    def run_for_batch(
        self, query: str, pdb_id: str, pdb_path: Optional[str] = None, context: Optional[str] = None
    ) -> "StructureResult":
        """Execute analysis for single structure in batch context.

        Injects PDB context into query so agent knows which structure.
        Extracts rankable metrics from tool results.

        Args:
            query: The user's query (will have PDB context injected)
            pdb_id: PDB identifier (e.g., '1UBQ')
            pdb_path: Optional path to local file (for pdb_path-based tools)
            context: Optional additional context

        Returns:
            StructureResult with the agent run and extracted metrics
        """
        # Import at runtime to avoid circular import
        from structagent.batch import StructureResult

        # Build PDB context
        pdb_context = f"Analyze PDB '{pdb_id}'."
        if pdb_path:
            pdb_context += f" Structure file: {pdb_path}"
        full_context = f"{context or ''}\n\n{pdb_context}".strip()

        # Run agent
        run = self.run(query, context=full_context)

        # Extract metrics
        metrics = self._extract_metrics_from_run(run)

        return StructureResult(
            pdb_id=pdb_id,
            pdb_path=pdb_path,
            run=run,
            metrics=metrics,
            success=run.final_answer is not None and "error" not in (run.final_answer or "").lower(),
            error=None,
        )

    def execute_batch_plan(
        self,
        plan: dict,
        query: str,
        pdb_id: str,
        pdb_path: Optional[str] = None,
    ) -> "StructureResult":
        """Execute a pre-validated plan for a single structure without planning.

        Takes a plan created by create_batch_plan() and executes it for one
        structure. No planning is done - this is purely execution.

        Args:
            plan: Pre-validated plan dict from create_batch_plan().
            query: The user's original query.
            pdb_id: PDB identifier (e.g., '1UBQ').
            pdb_path: Optional path to local file (for pdb_path-based tools).

        Returns:
            StructureResult with the agent run and extracted metrics.
        """
        from structagent.batch import StructureResult

        start_time = time.time()
        self._steps = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._last_chain_info = None

        # Build PDB context for tools
        pdb_context = f"Analyze PDB '{pdb_id}'."
        if pdb_path:
            pdb_context += f" Structure file: {pdb_path}"

        # Execute the pre-validated plan (no planning needed)
        resolved_plan = self._resolve_batch_placeholders(plan, pdb_id, pdb_path)
        results = self._execute_with_adaptation(resolved_plan, query, pdb_context)

        # Synthesize final answer from results
        final_answer = self._synthesize(results, query, pdb_context)

        wall_time = time.time() - start_time

        # Build AgentRun from the execution
        run = AgentRun(
            query=query,
            steps=self._steps,
            final_answer=final_answer,
            total_steps=len(self._steps),
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            wall_time_seconds=wall_time,
            model=self.model,
        )

        # Extract metrics from tool results
        metrics = self._extract_metrics_from_run(run)

        return StructureResult(
            pdb_id=pdb_id,
            pdb_path=pdb_path,
            run=run,
            metrics=metrics,
            success=run.final_answer is not None and "error" not in (run.final_answer or "").lower(),
            error=None,
        )

    def _resolve_batch_placeholders(
        self,
        plan: dict,
        pdb_id: str,
        pdb_path: Optional[str],
    ) -> dict:
        """Substitute per-structure placeholders in a batch plan."""

        def resolve(value):
            if value == "$PDB_ID":
                return pdb_id
            if value == "$PDB_PATH":
                return pdb_path
            if isinstance(value, dict):
                return {k: resolve(v) for k, v in value.items()}
            if isinstance(value, list):
                return [resolve(v) for v in value]
            return value

        return resolve(copy.deepcopy(plan))

    def _schema_properties_for_tool(self, tool_name: str) -> set[str]:
        """Return normalized parameter names for a registered tool."""
        entry = self.registry._tools.get(tool_name)
        if not entry or not entry.parameters:
            return set()
        parameters = entry.parameters
        if "properties" in parameters:
            return set(parameters.get("properties", {}).keys())
        return set(parameters.keys())

    def _normalize_batch_plan_placeholders(
        self,
        plan: dict,
        structures: list[tuple[str, Optional[str]]],
    ) -> dict:
        """Normalize a batch plan so it can be executed for each structure."""
        normalized = copy.deepcopy(plan)
        pdb_ids = {pdb_id for pdb_id, _ in structures if pdb_id}
        pdb_paths = {pdb_path for _, pdb_path in structures if pdb_path}

        def replace_known_values(value):
            if isinstance(value, str):
                if value in pdb_ids:
                    return "$PDB_ID"
                if value in pdb_paths:
                    return "$PDB_PATH"
            if isinstance(value, dict):
                return {k: replace_known_values(v) for k, v in value.items()}
            if isinstance(value, list):
                return [replace_known_values(v) for v in value]
            return value

        for step in normalized.get("steps", []):
            tool_name = step.get("tool")
            args = replace_known_values(step.get("args") or {})
            props = self._schema_properties_for_tool(tool_name)

            if pdb_paths and "pdb_path" in props:
                args["pdb_path"] = "$PDB_PATH"
                if "pdb_id" in args:
                    args.pop("pdb_id")
            elif "pdb_id" in props and "pdb_id" not in args:
                args["pdb_id"] = "$PDB_ID"

            step["args"] = args

        return normalized

    def _extract_metrics_from_run(self, run: AgentRun) -> dict:
        """Extract rankable metrics from AgentRun steps.

        Extracts meaningful metrics from tool results for ranking structures.
        Metrics are only extracted if present in the tool's raw output.
        """
        return extract_metrics_from_steps(run.steps)

    def run_react(self, query: str, context: Optional[str] = None) -> AgentRun:
        """Execute a single query through the pure ReAct loop.

        This is the original implementation kept for backwards compatibility
        when mode="react" is set.

        Args:
            query: The user's question.
            context: Optional additional context for the system prompt.

        Returns:
            AgentRun with all steps and the final answer.
        """
        start_time = time.time()
        self._steps = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        # Build system prompt and get tool schemas
        system_prompt = build_system_prompt(context)
        tool_schemas = self.registry.get_tool_schemas(self.toolsets)

        # Initialize messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        final_answer = ""
        step_count = 0
        max_steps_exceeded = False

        try:
            while step_count < self.max_steps:
                step_count += 1

                # Call the model
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tool_schemas if tool_schemas else None,
                    temperature=0.0,
                )

                # Track tokens
                if response.usage:
                    self._total_input_tokens += response.usage.prompt_tokens or 0
                    self._total_output_tokens += response.usage.completion_tokens or 0

                # Get the assistant's message
                assistant_msg = response.choices[0].message
                finish_reason = response.choices[0].finish_reason

                # If the model stopped without tool calls, return the answer
                if finish_reason == "stop":
                    final_answer = assistant_msg.content or ""
                    self._log_step(
                        thought=None,
                        tool_name=None,
                        tool_args=None,
                        tool_result=None,
                        is_final=True,
                    )
                    break

                # Handle tool calls
                if assistant_msg.tool_calls:
                    # Append assistant message with tool calls
                    messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_msg.content,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in assistant_msg.tool_calls
                            ],
                        }
                    )

                    # Process each tool call
                    for tc in assistant_msg.tool_calls:
                        tool_name = tc.function.name
                        raw_args = tc.function.arguments

                        # Parse arguments from JSON string
                        try:
                            tool_args = json.loads(raw_args) if raw_args else {}
                        except json.JSONDecodeError:
                            tool_args = {}

                        # Call the tool
                        result = self.registry.call_tool(tool_name, **tool_args)

                        # Append tool result message
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result.data,
                            }
                        )

                        # Log the step
                        self._log_step(
                            thought=assistant_msg.content,
                            tool_name=tool_name,
                            tool_args=tool_args,
                            tool_result=result,
                            is_final=False,
                        )
                else:
                    # No tool calls but also not "stop" - treat as final answer
                    final_answer = assistant_msg.content or ""
                    self._log_step(
                        thought=assistant_msg.content,
                        tool_name=None,
                        tool_args=None,
                        tool_result=None,
                        is_final=True,
                    )
                    break
            else:
                # max_steps exceeded
                max_steps_exceeded = True

            # If max_steps was exceeded, try to synthesize a response
            if max_steps_exceeded and not final_answer:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have reached the maximum number of steps. "
                            "Please provide a synthesis of what you have found so far."
                        ),
                    }
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=None,  # No tools on synthesis call
                    temperature=0.0,
                )
                final_answer = response.choices[0].message.content or ""
                if response.usage:
                    self._total_input_tokens += response.usage.prompt_tokens or 0
                    self._total_output_tokens += response.usage.completion_tokens or 0

        except Exception as e:
            console.print(f"[bold #ff0066]Error during agent run[/bold]")
            final_answer = "An error occurred during analysis. Please try again or rephrase your query."

        wall_time = time.time() - start_time

        return AgentRun(
            query=query,
            steps=self._steps,
            final_answer=final_answer,
            total_steps=len(self._steps),
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            wall_time_seconds=wall_time,
            model=self.model,
        )

    def _extract_pdb_id(self, text: str) -> Optional[str]:
        """Extract PDB ID from query or context text.

        Looks for 4-character PDB IDs (alphanumeric, starting with digit).

        Args:
            text: Query or context to search.

        Returns:
            PDB ID if found, None otherwise.
        """
        # PDB ID pattern: starts with digit, followed by 3 alphanumeric chars
        # Common formats: "1BRC", "1abc", "6VXX", etc.
        pattern = r"\b([0-9][0-9A-Za-z]{3})\b"
        matches = re.findall(pattern, text)
        # Filter out likely non-PDB matches (like years, version numbers)
        for match in matches:
            upper_match = match.upper()
            # Skip if it looks like a year (e.g., "2023")
            if upper_match[0] == "2" and upper_match[1] == "0":
                continue
            # Skip if it looks like a number that happens to be 4 digits
            if upper_match.isdigit():
                continue
            return upper_match
        return None

    def _get_structure_chain_info(self, pdb_id: str) -> Optional[str]:
        """Load structure and get chain composition info.

        Also stores raw chain data in self._last_chain_info for validation.

        Args:
            pdb_id: PDB identifier.

        Returns:
            Human-readable string describing chains, or None if load failed.
        """
        try:
            result = self.registry.call_tool("load_structure", pdb_id=pdb_id)
            if not result.success:
                return None

            chains = result.raw.get("chains", [])
            if not chains:
                return None

            # Store raw chain info for validation
            self._last_chain_info = {"pdb_id": pdb_id.upper(), "chains": chains}

            chain_descriptions = []
            for ch in chains:
                chain_id = ch.get("id", "?")
                length = ch.get("length", 0)
                first_res = ch.get("first_residue")
                last_res = ch.get("last_residue")
                if first_res and last_res:
                    chain_descriptions.append(f"Chain {chain_id}: {length} residues (residues {first_res}-{last_res})")
                else:
                    chain_descriptions.append(f"Chain {chain_id}: {length} residues")

            chain_text = "; ".join(chain_descriptions)
            return f"Structure {pdb_id.upper()} has {len(chains)} chain(s): {chain_text}."
        except Exception:
            return None

    def _create_plan(self, query: str, context: Optional[str] = None) -> Optional[dict]:
        """Create an analysis plan via a single LLM call with no tools.

        Args:
            query: The user's question.
            context: Optional additional context.

        Returns:
            Parsed JSON plan dict or None if planning failed.
        """
        self.display_strategy.start_thinking("planning")

        # Auto-detect chain composition if PDB ID is found in query/context
        chain_info = None
        combined_text = f"{query} {context or ''}"
        pdb_id = self._extract_pdb_id(combined_text)
        if pdb_id:
            chain_info = self._get_structure_chain_info(pdb_id)
            if chain_info:
                console.print(f"[dim cyan]Detected structure info: {chain_info}[/dim cyan]")

        # Build compact tool list for the planning prompt using tool_metadata
        tool_list = build_compact_tool_list(self.toolsets)
        planning_prompt = build_planning_prompt(context, tool_list, chain_info=chain_info)

        messages = [
            {"role": "system", "content": planning_prompt},
            {"role": "user", "content": query},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )

            if response.usage:
                self._total_input_tokens += response.usage.prompt_tokens or 0
                self._total_output_tokens += response.usage.completion_tokens or 0

            content = response.choices[0].message.content
            if not content:
                self.display_strategy.finish_thinking("planning")
                return None

            # Parse JSON from response
            # Try to extract JSON from the content (it might have surrounding text)
            json_str = content.strip()

            # Handle various markdown code block formats
            # Find JSON between first { and last }
            json_start = json_str.find("{")
            json_end = json_str.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = json_str[json_start:json_end]

            plan = json.loads(json_str)
            self.display_strategy.finish_thinking("planning")
            return plan

        except json.JSONDecodeError as e:
            console.print(f"[bold cyan]Failed to parse plan JSON:[/bold cyan] {escape(str(e))}")
            self.display_strategy.finish_thinking("planning")
            return None
        except Exception as e:
            console.print(f"[bold #ff0066]Error during plan creation[/bold]")
            self.display_strategy.finish_thinking("planning")
            return None

    def _validate_chain_ids(self, steps: list[dict]) -> list[str]:
        """Validate chain IDs in plan steps against known structure chains.

        Args:
            steps: List of plan steps.

        Returns:
            List of error messages for chain ID issues.
        """
        errors = []

        # Only validate if we have chain info from structure loading
        if not self._last_chain_info:
            return errors

        chains_info = self._last_chain_info.get("chains", [])
        if not chains_info:
            return errors

        # Build set of valid chain IDs
        valid_chain_ids = {ch.get("id") for ch in chains_info if ch.get("id")}

        # Tools that use chain_a/chain_b pairs
        chain_pair_tools = {"compute_interface", "score_interface", "analyze_interface_energies"}

        # Tools that use chain_id
        chain_id_tools = {
            "list_residues",
            "get_residue_contacts",
            "compute_sasa",
            "get_secondary_structure",
            "analyze_bfactors",
            "compute_charge_distribution",
            "get_conservation_scores",
            "check_ramachandran",
            "search_structural_homologs",
            "compute_normal_modes",
            "compute_cross_correlations",
            "predict_hinge_regions",
            "compute_perturbation_response",
        }

        for i, step in enumerate(steps):
            tool_name = step.get("tool")
            if not tool_name:
                continue

            tool_args = step.get("args", {})

            # Check chain_a/chain_b pairs
            if tool_name in chain_pair_tools:
                chain_a = tool_args.get("chain_a")
                chain_b = tool_args.get("chain_b")

                if chain_a and chain_a not in valid_chain_ids:
                    # Try to suggest a correction
                    suggestion = ""
                    if valid_chain_ids:
                        # Find chains with similar names or just list available
                        available = ", ".join(sorted(valid_chain_ids))
                        suggestion = f" Available chains: {available}"
                        # Try simple heuristic: if user typed uppercase and we have match
                        if chain_a.upper() in valid_chain_ids:
                            suggestion = f" Did you mean '{chain_a.upper()}'?"
                    errors.append(f"Step {i}: '{tool_name}' has invalid chain_a='{chain_a}'.{suggestion}")

                if chain_b and chain_b not in valid_chain_ids:
                    suggestion = ""
                    if valid_chain_ids:
                        available = ", ".join(sorted(valid_chain_ids))
                        suggestion = f" Available chains: {available}"
                        if chain_b.upper() in valid_chain_ids:
                            suggestion = f" Did you mean '{chain_b.upper()}'?"
                    errors.append(f"Step {i}: '{tool_name}' has invalid chain_b='{chain_b}'.{suggestion}")

            # Check single chain_id
            if tool_name in chain_id_tools:
                chain_id = tool_args.get("chain_id")
                if chain_id and chain_id not in valid_chain_ids:
                    suggestion = ""
                    if valid_chain_ids:
                        available = ", ".join(sorted(valid_chain_ids))
                        suggestion = f" Available chains: {available}"
                        if chain_id.upper() in valid_chain_ids:
                            suggestion = f" Did you mean '{chain_id.upper()}'?"
                    errors.append(f"Step {i}: '{tool_name}' has invalid chain_id='{chain_id}'.{suggestion}")

        return errors

    def _validate_plan(self, plan: dict, max_revisions: int = 2) -> Optional[dict]:
        """Validate a plan by checking tool availability and dependencies.

        Args:
            plan: The plan dict to validate.
            max_revisions: Maximum number of revision attempts.

        Returns:
            Validated plan dict or None if validation failed.
        """
        available_tools = set(self.registry.list_tools())
        errors = []
        planned_tool_names = set()

        # Validate each step's tool exists and args are valid
        steps = plan.get("steps", [])
        for i, step in enumerate(steps):
            tool_name = step.get("tool")
            if not tool_name:
                errors.append(f"Step {i}: Missing tool name")
                continue

            planned_tool_names.add(tool_name)
            if tool_name not in available_tools:
                errors.append(f"Step {i}: Tool '{tool_name}' not found in registry")
                continue

            # Validate arguments against tool's parameter schema
            tool_args = step.get("args", {})
            if tool_name in self.registry._tools:
                entry = self.registry._tools[tool_name]
                param_schema = entry.parameters
                if param_schema and "properties" in param_schema:
                    valid_params = set(param_schema["properties"].keys())
                    required_params = set(param_schema.get("required", []))
                    for arg_name in tool_args:
                        if arg_name not in valid_params:
                            errors.append(
                                f"Step {i}: '{tool_name}' has no parameter '{arg_name}'. Valid params: {list(valid_params)}"
                            )
                    missing_required = required_params - set(tool_args.keys())
                    if missing_required:
                        errors.append(
                            f"Step {i}: '{tool_name}' missing required parameter(s): {sorted(missing_required)}"
                        )

        # Check dependencies ONLY for planned tools (not all tools in registry)
        for tool_name in planned_tool_names:
            if tool_name not in self.registry._tools:
                continue
            entry = self.registry._tools[tool_name]
            if entry.check_fn is not None:
                try:
                    if not entry.check_fn():
                        errors.append(f"Tool '{tool_name}' dependency check failed")
                except Exception as e:
                    errors.append(f"Tool '{tool_name}' dependency check error")

        # Validate chain IDs if we have structure info
        chain_errors = self._validate_chain_ids(steps)
        errors.extend(chain_errors)

        if errors:
            console.print(f"[bold cyan]Plan validation errors:[/bold cyan]")
            for err in errors:
                console.print(f"  [bold cyan]- {escape(err)}[/bold cyan]")
            if len(steps) <= max_revisions:
                return self._revise_plan(plan, errors)

        if errors:
            console.print(
                f"[bold #ff0066]Plan validation failed after {max_revisions} revisions, proceeding with plan anyway[/bold #ff0066]"
            )
            # Return the plan even though validation failed - let execution try
            return plan

        return plan

    def _revise_plan(self, plan: dict, errors: list[str]) -> Optional[dict]:
        """Attempt to revise a plan to fix validation errors.

        Args:
            plan: The original plan dict.
            errors: List of validation error messages.

        Returns:
            Revised plan dict or None if revision failed.
        """
        console.print("[cyan]Attempting to revise plan...[/cyan]")

        error_summary = "\n".join([f"- {err}" for err in errors])

        # Build tool schema reference for correct parameter names
        tool_schemas = {}
        for tool_name in self.registry.list_tools():
            if tool_name in self.registry._tools:
                entry = self.registry._tools[tool_name]
                param_props = entry.parameters.get("properties", {}) if entry.parameters else {}
                tool_schemas[tool_name] = {"description": entry.description, "parameters": list(param_props.keys())}

        tool_ref = "\n".join(
            [f"- {name}: params {', '.join(info['parameters'])}" for name, info in tool_schemas.items()]
        )

        # Add chain info if available
        chain_info_section = ""
        if self._last_chain_info:
            chains = self._last_chain_info.get("chains", [])
            if chains:
                chain_descriptions = []
                for ch in chains:
                    chain_id = ch.get("id", "?")
                    length = ch.get("length", 0)
                    first_res = ch.get("first_residue")
                    last_res = ch.get("last_residue")
                    if first_res and last_res:
                        chain_descriptions.append(
                            f"Chain {chain_id}: {length} residues (residues {first_res}-{last_res})"
                        )
                    else:
                        chain_descriptions.append(f"Chain {chain_id}: {length} residues")

                chain_text = "; ".join(chain_descriptions)
                pdb_id = self._last_chain_info.get("pdb_id", "?")
                chain_info_section = (
                    f"\n\nSTRUCTURE CHAIN INFORMATION for {pdb_id}:\n"
                    f"{chain_text}.\n"
                    f"IMPORTANT: When specifying chain IDs in tool arguments, you MUST use the chain IDs "
                    f"exactly as they appear above (e.g., 'E', 'I', not 'A', 'B')."
                )

        revision_prompt = f"""The following plan has validation errors:

{json.dumps(plan, indent=2)}

Errors:
{error_summary}

Available tools with their CORRECT parameter names:
{tool_ref}
{chain_info_section}

IMPORTANT: Use EXACTLY the parameter names listed above (e.g., pdb_id not pdbid, chain_id not chainId).
Output ONLY the revised JSON plan.
"""

        messages = [
            {
                "role": "system",
                "content": "You are a planning assistant. Fix the plan errors using exact parameter names.",
            },
            {"role": "user", "content": revision_prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )

            if response.usage:
                self._total_input_tokens += response.usage.prompt_tokens or 0
                self._total_output_tokens += response.usage.completion_tokens or 0

            content = response.choices[0].message.content
            if not content:
                return None

            json_str = content.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                json_str = json_str.strip()

            revised_plan = json.loads(json_str)
            return revised_plan

        except Exception as e:
            console.print(f"[bold #ff0066]Error revising plan[/bold]")
            return None

    def _execute_with_adaptation(
        self,
        plan: dict,
        query: str,
        context: Optional[str] = None,
    ) -> list[dict]:
        """Execute validated plan steps deterministically through the registry.

        Args:
            plan: The validated plan dict.
            query: The original user query.
            context: Optional additional context.

        Returns:
            List of execution result dicts with tool_name, args, result.
        """
        start_time = time.time()
        self._wall_time = 0.0
        results = []

        self.display_strategy.start_thinking("executing")
        self.display_strategy.show_plan(plan)

        try:
            for step_index, step in enumerate(plan.get("steps", [])[: self.max_steps]):
                tool_name = step.get("tool")
                if not tool_name:
                    result = ToolResult(
                        success=False,
                        data=f"Plan step {step_index} has no tool name.",
                        raw={},
                        error="Missing tool name",
                    )
                    tool_args = {}
                else:
                    tool_args = step.get("args") or {}
                    result = self.registry.call_tool(tool_name, **tool_args)

                results.append(
                    {
                        "tool_name": tool_name or "unknown",
                        "args": tool_args,
                        "result": result,
                    }
                )

                self._log_step(
                    thought=step.get("purpose"),
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=result,
                    is_final=False,
                )

                if not result.success:
                    break

        except Exception as e:
            console.print(f"[bold #ff0066]Error during execution[/bold]")
            results.append(
                {
                    "tool_name": "error",
                    "args": {},
                    "result": ToolResult(
                        success=False,
                        data=f"Execution error: {str(e)}",
                        raw={},
                        error=str(e),
                    ),
                }
            )

        self._wall_time = time.time() - start_time
        self.display_strategy.finish_thinking("executing")
        return results

    def _synthesize(
        self,
        results: Any,
        query: str,
        context: Optional[str] = None,
    ) -> str:
        """Synthesize a final answer from execution results.

        Args:
            results: Either a list of result dicts (plan mode) or error string (fallback).
            query: The original user query.
            context: Optional additional context.

        Returns:
            Final answer string.
        """
        # Format results for synthesis
        if isinstance(results, list):
            formatted_results = []
            for r in results:
                if isinstance(r, dict):
                    tool_name = r.get("tool_name", "unknown")
                    result_data = r.get("result")
                    if hasattr(result_data, "data"):
                        data = result_data.data
                    else:
                        data = str(result_data)
                    formatted_results.append(f"- {tool_name}: {data}")
                else:
                    formatted_results.append(f"- {r}")

            results_text = "\n".join(formatted_results) if formatted_results else "No results obtained."
        else:
            results_text = str(results)

        synthesis_prompt = f"""Based on the following execution results, provide a comprehensive answer to the user's query.

User Query: {query}

Execution Results:
{results_text}

Please synthesize a detailed response that:
1. Directly addresses the user's question
2. References specific findings from the tool outputs
3. Integrates structural biology knowledge
4. Notes any limitations or uncertainties
"""

        messages = [
            {"role": "system", "content": build_system_prompt(context)},
            {"role": "user", "content": synthesis_prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )

            if response.usage:
                self._total_input_tokens += response.usage.prompt_tokens or 0
                self._total_output_tokens += response.usage.completion_tokens or 0

            final_answer = response.choices[0].message.content or ""
            return final_answer

        except Exception as e:
            console.print(f"[bold #ff0066]Error during synthesis[/bold]")
            return "An error occurred during synthesis."

    def chat(
        self,
        query: str,
        message_history: list[dict],
        context: Optional[str] = None,
    ) -> AgentRun:
        """Execute a multi-turn conversation.

        In "plan" mode (default): Uses planning-first execution with adaptation.
        In "react" mode: Uses pure ReAct loop (backwards compatible).

        Args:
            query: The current user message.
            message_history: Full conversation history with role and content.
            context: Optional additional context for the system prompt.

        Returns:
            AgentRun with all steps and the final answer.
        """
        if self.mode == "react":
            return self.chat_react(query, message_history, context)

        start_time = time.time()
        self._steps = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        # PHASE 1: Planning
        plan = self._create_plan(query, context)
        if not plan:
            final_answer = self._synthesize("Could not create plan", query, context)
            wall_time = time.time() - start_time
            return AgentRun(
                query=query,
                steps=self._steps,
                final_answer=final_answer,
                total_steps=len(self._steps),
                total_input_tokens=self._total_input_tokens,
                total_output_tokens=self._total_output_tokens,
                wall_time_seconds=wall_time,
                model=self.model,
            )

        # PHASE 2: Validate (max 2 revisions)
        validated_plan = self._validate_plan(plan, max_revisions=2)
        if not validated_plan:
            final_answer = self._synthesize("Could not validate plan", query, context)
            wall_time = time.time() - start_time
            return AgentRun(
                query=query,
                steps=self._steps,
                final_answer=final_answer,
                total_steps=len(self._steps),
                total_input_tokens=self._total_input_tokens,
                total_output_tokens=self._total_output_tokens,
                wall_time_seconds=wall_time,
                model=self.model,
            )

        # PHASE 3: Execute with adaptation
        results = self._execute_with_adaptation(validated_plan, query, context)

        # PHASE 4: Synthesize
        final_answer = self._synthesize(results, query, context)
        wall_time = time.time() - start_time

        return AgentRun(
            query=query,
            steps=self._steps,
            final_answer=final_answer,
            total_steps=len(self._steps),
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            wall_time_seconds=wall_time,
            model=self.model,
        )

    def chat_react(
        self,
        query: str,
        message_history: list[dict],
        context: Optional[str] = None,
    ) -> AgentRun:
        """Execute a multi-turn conversation using pure ReAct loop.

        This is the original implementation kept for backwards compatibility
        when mode="react" is set.

        Args:
            query: The current user message.
            message_history: Full conversation history with role and content.
            context: Optional additional context for the system prompt.

        Returns:
            AgentRun with all steps and the final answer.
        """
        start_time = time.time()
        self._steps = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        # Build system prompt and get tool schemas
        system_prompt = build_system_prompt(context)
        tool_schemas = self.registry.get_tool_schemas(self.toolsets)

        # Build messages: system + history + current query
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(message_history)
        messages.append({"role": "user", "content": query})

        final_answer = ""
        step_count = 0
        max_steps_exceeded = False

        try:
            while step_count < self.max_steps:
                step_count += 1

                # Call the model
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tool_schemas if tool_schemas else None,
                    temperature=0.0,
                )

                # Track tokens
                if response.usage:
                    self._total_input_tokens += response.usage.prompt_tokens or 0
                    self._total_output_tokens += response.usage.completion_tokens or 0

                # Get the assistant's message
                assistant_msg = response.choices[0].message
                finish_reason = response.choices[0].finish_reason

                # If the model stopped without tool calls, return the answer
                if finish_reason == "stop":
                    final_answer = assistant_msg.content or ""
                    self._log_step(
                        thought=None,
                        tool_name=None,
                        tool_args=None,
                        tool_result=None,
                        is_final=True,
                    )
                    break

                # Handle tool calls
                if assistant_msg.tool_calls:
                    # Append assistant message with tool calls
                    messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_msg.content,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in assistant_msg.tool_calls
                            ],
                        }
                    )

                    # Process each tool call
                    for tc in assistant_msg.tool_calls:
                        tool_name = tc.function.name
                        raw_args = tc.function.arguments

                        # Parse arguments from JSON string
                        try:
                            tool_args = json.loads(raw_args) if raw_args else {}
                        except json.JSONDecodeError:
                            tool_args = {}

                        # Call the tool
                        result = self.registry.call_tool(tool_name, **tool_args)

                        # Append tool result message
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result.data,
                            }
                        )

                        # Log the step
                        self._log_step(
                            thought=assistant_msg.content,
                            tool_name=tool_name,
                            tool_args=tool_args,
                            tool_result=result,
                            is_final=False,
                        )
                else:
                    # No tool calls but also not "stop" - treat as final answer
                    final_answer = assistant_msg.content or ""
                    self._log_step(
                        thought=assistant_msg.content,
                        tool_name=None,
                        tool_args=None,
                        tool_result=None,
                        is_final=True,
                    )
                    break
            else:
                # max_steps exceeded
                max_steps_exceeded = True

            # If max_steps was exceeded, try to synthesize a response
            if max_steps_exceeded and not final_answer:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have reached the maximum number of steps. "
                            "Please provide a synthesis of what you have found so far."
                        ),
                    }
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=None,
                    temperature=0.0,
                )
                final_answer = response.choices[0].message.content or ""
                if response.usage:
                    self._total_input_tokens += response.usage.prompt_tokens or 0
                    self._total_output_tokens += response.usage.completion_tokens or 0

        except Exception as e:
            console.print(f"[bold #ff0066]Error during chat run[/bold]")
            final_answer = "An error occurred during chat. Please try again."

        wall_time = time.time() - start_time

        return AgentRun(
            query=query,
            steps=self._steps,
            final_answer=final_answer,
            total_steps=len(self._steps),
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            wall_time_seconds=wall_time,
            model=self.model,
        )

    def create_plan(self, query: str, context: Optional[str] = None) -> dict:
        """Create and return an analysis plan without executing it.

        Args:
            query: The user's question.
            context: Optional additional context for the system prompt.

        Returns:
            Plan dict with 'reasoning' and 'steps' keys.
        """
        return self._create_plan(query, context)

    def _create_and_validate_plan(self, query: str, context: Optional[str] = None) -> Optional[dict]:
        """Create a plan and validate it, returning the validated plan or None.

        Args:
            query: The user's question.
            context: Optional additional context for the system prompt.

        Returns:
            Validated plan dict or None if creation or validation failed.
        """
        plan = self._create_plan(query, context)
        if plan is None:
            return None
        return self._validate_plan(plan)

    def create_batch_plan(self, query: str, structures: list[tuple[str, Optional[str]]]) -> Optional[dict]:
        """Create and validate a single plan for multiple structures.

        Args:
            query: The user's question.
            structures: List of (pdb_id, pdb_path) tuples for structures to analyze.

        Returns:
            Validated plan dict for all structures, or None if planning failed.
        """
        if not structures:
            return None

        # Build context describing all structures
        structure_contexts = []
        all_chains = []

        for pdb_id, pdb_path in structures:
            # Load structure to get chain info (use pdb_path if local file, otherwise pdb_id)
            if pdb_path:
                result = self.registry.call_tool("load_structure", pdb_path=pdb_path)
            else:
                result = self.registry.call_tool("load_structure", pdb_id=pdb_id)
            if result.success:
                chains = result.raw.get("chains", [])
                if chains:
                    chain_descriptions = []
                    for ch in chains:
                        chain_id = ch.get("id", "?")
                        length = ch.get("length", 0)
                        first_res = ch.get("first_residue")
                        last_res = ch.get("last_residue")
                        if first_res and last_res:
                            chain_descriptions.append(
                                f"Chain {chain_id}: {length} residues (residues {first_res}-{last_res})"
                            )
                        else:
                            chain_descriptions.append(f"Chain {chain_id}: {length} residues")
                        all_chains.append(ch)

                    chain_text = "; ".join(chain_descriptions)
                    structure_contexts.append(f"Structure {pdb_id.upper()}: {chain_text}")

        if not structure_contexts:
            return None

        # Build combined context string
        combined_context = (
            f"You are analyzing {len(structures)} structure(s). "
            + " ".join(structure_contexts)
            + "\n\nCreate one reusable plan for every structure. "
            "Use $PDB_ID for the current structure identifier and $PDB_PATH for "
            "the current local structure file path. Prefer $PDB_PATH whenever a "
            "tool accepts pdb_path so local folder runs do not require RCSB downloads."
        )

        # Store chain info for validation
        if all_chains:
            self._last_chain_info = {"chains": all_chains}

        plan = self._create_plan(query, combined_context)
        if plan is None:
            return None
        normalized_plan = self._normalize_batch_plan_placeholders(plan, structures)
        return self._validate_plan(normalized_plan)

    def create_informed_batch_plan(
        self,
        query: str,
        structures: list[tuple[str, Optional[str]]],
        target_report: Any,
    ) -> Optional[dict]:
        """Create an informed plan for multiple structures using target analysis context.

        Stage 2 of binder-design workflow. Creates a batch plan that incorporates
        the target's identified hotspots, flexible regions, and recommended focus
        areas into the analysis query.

        Args:
            query: The user's question.
            structures: List of (pdb_id, pdb_path) tuples for structures to analyze.
            target_report: TargetAnalysisReport from Stage 1.

        Returns:
            Validated plan dict for all structures, or None if planning failed.
        """
        if not structures:
            return None

        # Build target context for the planning prompt
        target = target_report

        # Build hotspot context
        hotspot_lines = []
        for hs in target.hotspots[:5]:
            hotspot_lines.append(
                f"- Chain {hs.chain_id}, residues {hs.residue_range} "
                f"({hs.classification}): buried SA contribution {hs.buried_sa_contribution:.1f} A^2"
            )
        hotspots_context = "\n".join(hotspot_lines) if hotspot_lines else "  None identified"

        # Build flexible region context
        flex_lines = []
        for fr in target.flexible_regions[:5]:
            hinge_note = " (hinge region)" if fr.is_hinge_region else ""
            flex_lines.append(
                f"- Chain {fr.chain_id}, residues {fr.residue_range} "
                f"({fr.classification}, mean B-factor {fr.mean_bfactor:.1f}){hinge_note}"
            )
        flex_context = "\n".join(flex_lines) if flex_lines else "  None identified"

        # Build surface region context
        surf_lines = []
        for sr in target.surface_regions[:5]:
            surf_lines.append(
                f"- Chain {sr.chain_id}, residues {sr.residue_range} "
                f"({sr.classification}, mean rel SASA {sr.mean_relative_sasa:.1f}%)"
            )
        surf_context = "\n".join(surf_lines) if surf_lines else "  None identified"

        # Build recommended focus context
        focus_context = (
            ", ".join(target.recommended_analysis_focus) if target.recommended_analysis_focus else "general analysis"
        )

        # Build structure contexts
        structure_contexts = []
        all_chains = []

        for pdb_id, pdb_path in structures:
            # Load structure to get chain info
            if pdb_path:
                result = self.registry.call_tool("load_structure", pdb_path=pdb_path)
            else:
                result = self.registry.call_tool("load_structure", pdb_id=pdb_id)
            if result.success:
                chains = result.raw.get("chains", [])
                if chains:
                    chain_descriptions = []
                    for ch in chains:
                        chain_id = ch.get("id", "?")
                        length = ch.get("length", 0)
                        first_res = ch.get("first_residue")
                        last_res = ch.get("last_residue")
                        if first_res and last_res:
                            chain_descriptions.append(
                                f"Chain {chain_id}: {length} residues (residues {first_res}-{last_res})"
                            )
                        else:
                            chain_descriptions.append(f"Chain {chain_id}: {length} residues")
                        all_chains.append(ch)

                    chain_text = "; ".join(chain_descriptions)
                    structure_contexts.append(f"Structure {pdb_id.upper()}: {chain_text}")

        if not structure_contexts:
            return None

        # Build informed context
        informed_context = f"""You are analyzing {len(structures)} candidate binder structure(s) for binding to target {target.target_id}.

## Target Analysis Context (Stage 1 Results)
Design Strategy: "{target.design_strategy}"

### Identified Hotspots (high-contact interface residues):
{hotspots_context}

### Flexible Regions:
{flex_context}

### Surface-Exposed Regions:
{surf_context}

### Recommended Analysis Focus: {focus_context}

Candidate Structures:
{" ".join(structure_contexts)}"""

        # Store chain info for validation
        if all_chains:
            self._last_chain_info = {"chains": all_chains}

        # Create and validate the plan
        return self._create_and_validate_plan(query, informed_context)

    def _log_step(
        self,
        thought: Optional[str],
        tool_name: Optional[str],
        tool_args: Optional[dict],
        tool_result: Optional[ToolResult],
        is_final: bool,
    ):
        """Log a step with verbose rich output."""
        step = AgentStep(
            thought=thought,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            is_final=is_final,
            timestamp=time.time(),
        )
        self._steps.append(step)

        if not self.verbose:
            return

        # Use display strategy
        if is_final:
            content = thought or (tool_result.data if tool_result else "")
            self.display_strategy.show_final_answer(content)
        elif tool_name:
            self.display_strategy.start_tool(tool_name, tool_args)
            if tool_result:
                self.display_strategy.finish_tool(tool_name, tool_result.data, tool_result.execution_time_seconds)
