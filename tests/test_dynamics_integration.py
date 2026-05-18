"""Integration tests for ProDy dynamics tools in the full agentic loop.

These tests verify that the type coercion for n_modes works correctly
when the LLM passes n_modes as a string (e.g., "10") instead of an integer.
"""

import importlib
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch
from openai import OpenAI

from structagent.agent import MiraAgent
from structagent.registry import tool, ToolResult

LOCAL_PDB_PATH = str(Path(__file__).parent / "data" / "local" / "mini_complex.pdb")


# Reset registry for clean test state
def reset_registry():
    """Reset the singleton registry to empty state."""
    from structagent.registry import ToolRegistry

    reg = ToolRegistry()
    reg._tools = {}


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure registry is clean and tools are re-registered before each test."""
    reset_registry()
    # Re-import and reload dynamics module to re-register tools
    import structagent.tools.dynamics as dyn_mod

    importlib.reload(dyn_mod)
    yield
    reset_registry()


@pytest.fixture
def mock_client():
    """Create a mock OpenAI client."""
    return MagicMock(spec=OpenAI)


def make_tool_call_response(tool_name: str, tool_args: dict, finish_reason: str = "tool_calls"):
    """Make a mock response with a tool call."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].finish_reason = finish_reason
    mock_response.choices[0].message.content = "Processing..."
    mock_response.choices[0].message.tool_calls = [MagicMock()]
    mock_response.choices[0].message.tool_calls[0].id = "call_123"
    mock_response.choices[0].message.tool_calls[0].function.name = tool_name
    mock_response.choices[0].message.tool_calls[0].function.arguments = __import__("json").dumps(tool_args)
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 20
    mock_response.usage.completion_tokens = 10
    return mock_response


def make_final_response(content: str):
    """Make a mock final response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].finish_reason = "stop"
    mock_response.choices[0].message.content = content
    mock_response.choices[0].message.tool_calls = None
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 30
    mock_response.usage.completion_tokens = 5
    return mock_response


class TestComputeNormalModesWithStringNModes:
    """Test compute_normal_modes handles string n_modes from LLM."""

    def test_compute_normal_modes_with_string_n_modes(self, mock_client):
        """Agent loop completes successfully when LLM passes n_modes as string."""
        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: LLM requests tool with n_modes as STRING
                return make_tool_call_response(
                    "compute_normal_modes", {"pdb_path": LOCAL_PDB_PATH, "chain_id": "A", "n_modes": "10"}
                )
            else:
                # Second call: return final answer
                return make_final_response("Normal mode analysis complete.")

        mock_client.chat.completions.create = mock_create

        with patch.object(OpenAI, "__init__", lambda self, **kw: None):
            with patch.object(OpenAI, "__enter__", lambda self: mock_client):
                with patch.object(OpenAI, "__exit__", lambda self, *a: None):
                    agent = MiraAgent(verbose=False, mode="react")
                    agent.client = mock_client
                    run = agent.run("Compute normal modes for 1UBQ chain A")

        assert run.final_answer == "Normal mode analysis complete."
        assert run.steps[0].tool_name == "compute_normal_modes"
        # Verify n_modes was coerced from string to int
        assert run.steps[0].tool_args.get("n_modes") == "10"  # Original args preserved
        assert run.steps[0].tool_result.success is True


class TestComputeCrossCorrelationsWithStringNModes:
    """Test compute_cross_correlations handles string n_modes from LLM."""

    def test_compute_cross_correlations_with_string_n_modes(self, mock_client):
        """Agent loop completes successfully when LLM passes n_modes as string."""
        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return make_tool_call_response(
                    "compute_cross_correlations", {"pdb_path": LOCAL_PDB_PATH, "chain_id": "A", "n_modes": "15"}
                )
            else:
                return make_final_response("Cross-correlation analysis complete.")

        mock_client.chat.completions.create = mock_create

        with patch.object(OpenAI, "__init__", lambda self, **kw: None):
            with patch.object(OpenAI, "__enter__", lambda self: mock_client):
                with patch.object(OpenAI, "__exit__", lambda self, *a: None):
                    agent = MiraAgent(verbose=False, mode="react")
                    agent.client = mock_client
                    run = agent.run("Compute cross correlations for 1UBQ chain A")

        assert run.final_answer == "Cross-correlation analysis complete."
        assert run.steps[0].tool_name == "compute_cross_correlations"
        assert run.steps[0].tool_result.success is True


class TestPredictHingeRegionsWithStringNModes:
    """Test predict_hinge_regions handles string n_modes from LLM."""

    def test_predict_hinge_regions_with_string_n_modes(self, mock_client):
        """Agent loop completes successfully when LLM passes n_modes as string."""
        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return make_tool_call_response(
                    "predict_hinge_regions", {"pdb_path": LOCAL_PDB_PATH, "chain_id": "A", "n_modes": "5"}
                )
            else:
                return make_final_response("Hinge prediction complete.")

        mock_client.chat.completions.create = mock_create

        with patch.object(OpenAI, "__init__", lambda self, **kw: None):
            with patch.object(OpenAI, "__enter__", lambda self: mock_client):
                with patch.object(OpenAI, "__exit__", lambda self, *a: None):
                    agent = MiraAgent(verbose=False, mode="react")
                    agent.client = mock_client
                    run = agent.run("Predict hinge regions for 1UBQ chain A")

        assert run.final_answer == "Hinge prediction complete."
        assert run.steps[0].tool_name == "predict_hinge_regions"
        assert run.steps[0].tool_result.success is True


class TestComputePerturbationResponseWithStringNModes:
    """Test compute_perturbation_response handles string n_modes from LLM."""

    def test_compute_perturbation_response_with_string_n_modes(self, mock_client):
        """Agent loop completes successfully when LLM passes n_modes as string."""
        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return make_tool_call_response(
                    "compute_perturbation_response",
                    {"pdb_path": LOCAL_PDB_PATH, "chain_id": "A", "source_residue": 10, "n_modes": "20"},
                )
            else:
                return make_final_response("Perturbation response analysis complete.")

        mock_client.chat.completions.create = mock_create

        with patch.object(OpenAI, "__init__", lambda self, **kw: None):
            with patch.object(OpenAI, "__enter__", lambda self: mock_client):
                with patch.object(OpenAI, "__exit__", lambda self, *a: None):
                    agent = MiraAgent(verbose=False, mode="react")
                    agent.client = mock_client
                    run = agent.run("Compute perturbation response for 1UBQ chain A")

        assert run.final_answer == "Perturbation response analysis complete."
        assert run.steps[0].tool_name == "compute_perturbation_response"
        assert run.steps[0].tool_result.success is True
