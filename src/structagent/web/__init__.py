"""Web GUI module for MIRA with chat + PDB viewer."""

from structagent.web.server import app
from structagent.web.session import AgentSession
from structagent.web.tool_processor import ToolResultProcessor, ViewerUpdate

__all__ = ["app", "AgentSession", "ToolResultProcessor", "ViewerUpdate"]
