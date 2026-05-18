"""Display strategies for MIRA TUI."""

import sys
from abc import ABC, abstractmethod
from rich.console import Console
from rich.control import Control
from rich.panel import Panel

console = Console()


class DisplayStrategy(ABC):
    """Abstract base class for display strategies."""

    @abstractmethod
    def start_tool(self, tool_name: str, args: dict):
        """Called when a tool starts executing."""
        pass

    @abstractmethod
    def finish_tool(self, tool_name: str, result_data: str, duration: float):
        """Called when a tool finishes."""
        pass

    @abstractmethod
    def start_thinking(self, phase: str):
        """Called when agent starts thinking (planning/executing)."""
        pass

    @abstractmethod
    def finish_thinking(self, phase: str):
        """Called when agent finishes thinking."""
        pass

    @abstractmethod
    def show_plan(self, plan: dict):
        """Called to display the analysis plan."""
        pass

    @abstractmethod
    def show_final_answer(self, answer: str):
        """Called to display the final answer."""
        pass


class NormalDisplay(DisplayStrategy):
    """Compact normal mode with loading bar and byline."""

    def __init__(self):
        self.current_tool = None  # (tool_name, status)
        self.phase = "Planning"
        self._panel = None

    def start_tool(self, tool_name: str, args: dict):
        self.current_tool = (tool_name, "running...")
        self._render()

    def finish_tool(self, tool_name: str, result_data: str, duration: float):
        if self.current_tool and self.current_tool[0] == tool_name:
            summary = result_data[:40].replace("\n", " ") if result_data else "done"
            self.current_tool = (tool_name, summary)
        self._render()

    def start_thinking(self, phase: str):
        self.phase = "Planning" if phase == "planning" else "Analysis"
        self.current_tool = None
        self._render()

    def finish_thinking(self, phase: str):
        self.phase = "Planning" if phase == "planning" else "Analysis"
        self._render()

    def show_plan(self, plan: dict):
        steps = plan.get("steps", [])
        step_names = [s["tool"] for s in steps]
        self.current_tool = (f"📋 {' → '.join(step_names)}", "")
        self._render()

    def show_final_answer(self, answer: str):
        console.print(answer)

    def _render(self):
        """Render current state as a compact panel."""
        phase_icons = {
            "Planning": "🧠",
            "Analysis": "⚙️",
        }
        icon = phase_icons.get(self.phase, "⏳")

        if self.current_tool:
            name, summary = self.current_tool
            byline = f"{name}" + (f": {summary}" if summary else "")
        else:
            byline = "Working..."

        status = f"{icon} [{self.phase}] {byline}"

        self._panel = Panel(status, title="[bold #ff00ff]MIRA[/bold #ff00ff]", border_style="#ff00ff")

        # Clear previous panel and print new one in place
        if self._panel is not None:
            # Move up 3 lines (panel is 3 lines) and clear to end of screen
            sys.stdout.write("\033[3A")
            sys.stdout.write("\033[J")
            sys.stdout.flush()

        console.print(self._panel)


class VerboseDisplay(DisplayStrategy):
    """Current verbose mode with full tool-by-tool output."""

    def __init__(self):
        pass

    def start_tool(self, tool_name: str, args: dict):
        import json

        args_str = json.dumps(args) if args else "{}"
        console.print(f"[bold cyan]Tool:[/bold cyan] {tool_name}({args_str})")

    def finish_tool(self, tool_name: str, result_data: str, duration: float):
        result_preview = result_data[:200] + "..." if len(result_data) > 200 else result_data
        console.print(f"  [dim]→ {result_preview}[/dim]")
        if duration:
            console.print(f"  [dim]✓ {duration:.2f}s[/dim]")

    def start_thinking(self, phase: str):
        display_phase = "Planning" if phase == "planning" else "Analysis"
        console.print(f"[dim]🤔 {display_phase}...[/dim]")

    def finish_thinking(self, phase: str):
        display_phase = "Planning" if phase == "planning" else "Analysis"
        console.print(f"[dim]✓ {display_phase} complete[/dim]")

    def show_plan(self, plan: dict):
        steps = plan.get("steps", [])
        console.print(f"[bold cyan]Plan ({len(steps)} steps):[/bold cyan]")
        for i, step in enumerate(steps, 1):
            console.print(f"  {i}. {step['tool']} — {step.get('purpose', '')}")

    def show_final_answer(self, answer: str):
        pass  # Verbose mode handles final answer elsewhere


def get_display_strategy(mode: str) -> DisplayStrategy:
    """Get display strategy by name."""
    if mode == "normal":
        return NormalDisplay()
    return VerboseDisplay()
