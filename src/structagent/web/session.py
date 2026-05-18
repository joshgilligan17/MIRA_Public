"""Agent session manager for web interface."""

import asyncio
from typing import Optional, AsyncGenerator, Any

from structagent.agent import MiraAgent, AgentRun, AgentStep
from structagent.web.tool_processor import ToolResultProcessor, ViewerUpdate


class AgentSession:
    """Wraps MiraAgent for async web usage with event streaming."""

    def __init__(
        self,
        model: str = "MiniMax-M2.7",
        base_url: str = "https://api.minimax.io/v1",
        max_steps: int = 15,
        toolsets: Optional[list[str]] = None,
        api_key: Optional[str] = None,
        mode: str = "plan",
        timeout: float = 120.0,
        temperature: float = 0.0,
    ):
        """Initialize an agent session.

        Args:
            model: Model name to use for chat completions.
            base_url: API base URL.
            max_steps: Maximum ReAct loop iterations.
            toolsets: List of toolset names to enable (None = all).
            api_key: API key (reads from env if not provided).
            mode: Execution mode - "plan" for planning-first, "react" for pure ReAct.
            timeout: Request timeout in seconds.
            temperature: Sampling temperature for responses.
        """
        self.agent = MiraAgent(
            model=model,
            base_url=base_url,
            max_steps=max_steps,
            toolsets=toolsets,
            api_key=api_key,
            verbose=False,
            mode=mode,
            display="normal",
            timeout=timeout,
            temperature=temperature,
        )
        self.current_pdb_id: Optional[str] = None
        self._processor = ToolResultProcessor()

    async def chat(self, query: str, context: Optional[str] = None) -> AsyncGenerator[dict, None]:
        """Execute a chat query and yield events.

        Args:
            query: The user's question.
            context: Optional additional context.

        Yields:
            Event dictionaries with keys: event_type, data
            - tool_execution: When a tool is called, contains tool_name, tool_args, result
            - viewer_update: When a tool produces relevant viewer data, contains ViewerUpdate
            - chat_response: Final response from the agent
            - error: Error message if something fails
        """
        try:
            # Run agent in executor to avoid blocking
            loop = asyncio.get_event_loop()
            run = await loop.run_in_executor(None, self.agent.run, query, context)

            # Process each step
            for step in run.steps:
                if step.tool_name and step.tool_result:
                    # Emit tool execution event
                    yield {
                        "event_type": "tool_execution",
                        "data": {
                            "tool_name": step.tool_name,
                            "tool_args": step.tool_args,
                            "result": step.tool_result.data,
                            "success": step.tool_result.success,
                        },
                    }

                    # Try to convert tool result to viewer update
                    if step.tool_result.success and step.tool_result.raw:
                        viewer_update = self._processor.process(
                            step.tool_name, step.tool_result.raw
                        )
                        if viewer_update:
                            # Update current PDB ID if available
                            if viewer_update.pdb_id:
                                self.current_pdb_id = viewer_update.pdb_id

                            yield {
                                "event_type": "viewer_update",
                                "data": {
                                    "action": viewer_update.action,
                                    "pdb_id": viewer_update.pdb_id or self.current_pdb_id,
                                    "highlight": viewer_update.highlight,
                                    "message": viewer_update.message,
                                },
                            }

            # Emit final chat response
            yield {
                "event_type": "chat_response",
                "data": {
                    "response": run.final_answer,
                    "total_steps": run.total_steps,
                },
            }

        except Exception as e:
            yield {
                "event_type": "error",
                "data": {
                    "message": str(e),
                },
            }

    def set_pdb_id(self, pdb_id: str) -> None:
        """Set the current PDB ID for the session."""
        self.current_pdb_id = pdb_id
