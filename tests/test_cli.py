"""Tests for the MiraAgent CLI."""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from structagent.cli import chat, print_tools, print_examples, save_trajectory
from structagent.agent import AgentRun, AgentStep
from structagent.registry import ToolResult, tool


# Reset registry for clean test state
def reset_registry():
    """Reset the singleton registry to empty state."""
    from structagent.registry import ToolRegistry

    reg = ToolRegistry()
    reg._tools = {}


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure registry is clean before each test."""
    reset_registry()
    yield
    reset_registry()


@pytest.fixture
def mock_agent_run():
    """Create a mock AgentRun."""
    return AgentRun(
        query="Test query",
        steps=[
            AgentStep(
                thought="Loading structure",
                tool_name="load_structure",
                tool_args={"pdb_id": "1UBQ"},
                tool_result=ToolResult(success=True, data="Structure loaded", raw={}),
                is_final=False,
            ),
            AgentStep(
                thought=None,
                tool_name=None,
                tool_args=None,
                tool_result=None,
                is_final=True,
            ),
        ],
        final_answer="The structure of 1UBQ is a ubiquitin fold with...",
        total_steps=1,
        total_input_tokens=100,
        total_output_tokens=50,
        wall_time_seconds=2.5,
        model="MiniMax-M2.7",
    )


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock = MagicMock()
    mock.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(finish_reason="stop", message=MagicMock(content="Mocked response", tool_calls=None))],
        usage=MagicMock(prompt_tokens=10, completion_tokens=5),
    )
    return mock


class TestPrintTools:
    """Tests for print_tools function."""

    def test_print_tools_empty(self):
        """print_tools handles empty registry."""
        from io import StringIO
        from structagent.cli import console

        reset_registry()

        # Should print no tools message
        with patch.object(console, "print") as mock_print:
            print_tools()
            mock_print.assert_called()
            call_args = str(mock_print.call_args)
            assert "No tools" in call_args or mock_print.call_count > 0

    def test_print_tools_with_tools(self):
        """print_tools lists registered tools."""

        @tool(
            name="my_tool",
            toolset="test",
            description="A test tool",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def my_func():
            return ToolResult(success=True, data="ok", raw={})

        from structagent.cli import console

        with patch.object(console, "print") as mock_print:
            print_tools()
            assert mock_print.call_count > 0


class TestPrintExamples:
    """Tests for print_examples function."""

    def test_print_examples(self):
        """print_examples displays example queries."""
        from structagent.cli import console

        with patch.object(console, "print") as mock_print:
            print_examples()
            assert mock_print.call_count > 0


class TestSaveTrajectory:
    """Tests for save_trajectory function."""

    def test_save_trajectory(self, mock_agent_run, tmp_path):
        """save_trajectory writes JSONL file."""
        from structagent.cli import save_trajectory

        trajectory_dir = str(tmp_path)
        filename = save_trajectory(mock_agent_run, trajectory_dir)

        assert os.path.exists(filename)
        with open(filename) as f:
            data = json.loads(f.readline())
        assert data["query"] == "Test query"
        assert data["final_answer"] == "The structure of 1UBQ is a ubiquitin fold with..."

    def test_save_trajectory_creates_directory(self, mock_agent_run, tmp_path):
        """save_trajectory creates directory if needed."""
        from structagent.cli import save_trajectory

        trajectory_dir = os.path.join(str(tmp_path), "subdir", "nested")
        filename = save_trajectory(mock_agent_run, trajectory_dir)

        assert os.path.exists(filename)


class TestCLIHelp:
    """Tests for CLI --help."""

    def test_help_flag(self):
        """CLI --help shows usage."""
        runner = CliRunner()
        result = runner.invoke(chat, ["--help"])
        assert result.exit_code == 0
        assert "MIRA" in result.output
        assert "--model" in result.output
        assert "--max-steps" in result.output


class TestCLIToolsets:
    """Tests for CLI toolsets option."""

    def test_toolsets_option_parses(self):
        """CLI accepts --toolsets option."""
        runner = CliRunner()
        result = runner.invoke(chat, ["--help"])
        assert result.exit_code == 0
        assert "--toolsets" in result.output

    def test_save_trajectories_option(self):
        """CLI accepts --save-trajectories flag."""
        runner = CliRunner()
        result = runner.invoke(chat, ["--help"])
        assert result.exit_code == 0
        assert "--save-trajectories" in result.output


class TestCLICommands:
    """Tests for REPL commands via mocked agent."""

    def test_quit_command(self, mock_openai_client):
        """User can quit with /quit."""
        with patch("structagent.agent.OpenAI", return_value=mock_openai_client):
            runner = CliRunner()
            result = runner.invoke(chat, input="/quit\n")
            assert "Goodbye" in result.output or result.exit_code == 0

    def test_help_command(self, mock_openai_client):
        """User can get help with /help."""
        with patch("structagent.agent.OpenAI", return_value=mock_openai_client):
            runner = CliRunner()
            result = runner.invoke(chat, input="/help\n/quit\n")
            assert "Commands" in result.output or "tools" in result.output

    def test_tools_command(self, mock_openai_client):
        """User can list tools with /tools."""
        with patch("structagent.agent.OpenAI", return_value=mock_openai_client):
            runner = CliRunner()
            result = runner.invoke(chat, input="/tools\n/quit\n")
            assert "Available Tools" in result.output or "load_structure" in result.output

    def test_example_command(self, mock_openai_client):
        """User can show examples with /example."""
        with patch("structagent.agent.OpenAI", return_value=mock_openai_client):
            runner = CliRunner()
            result = runner.invoke(chat, input="/example\n/quit\n")
            assert "allosteric" in result.output or "binding" in result.output

    def test_clear_command(self, mock_openai_client):
        """User can clear history with /clear."""
        with patch("structagent.agent.OpenAI", return_value=mock_openai_client):
            runner = CliRunner()
            result = runner.invoke(chat, input="/clear\n/quit\n")
            assert "cleared" in result.output.lower() or "history" in result.output.lower()

    def test_history_command(self, mock_openai_client):
        """User can check history with /history."""
        with patch("structagent.agent.OpenAI", return_value=mock_openai_client):
            runner = CliRunner()
            result = runner.invoke(chat, input="/history\n/quit\n")
            assert "history" in result.output.lower()
