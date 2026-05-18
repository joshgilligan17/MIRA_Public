"""Tests for the MiraAgent ReAct loop."""

import pytest
from unittest.mock import MagicMock, patch
from openai import OpenAI

from structagent.agent import MiraAgent, AgentRun, AgentStep
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
def mock_client():
    """Create a mock OpenAI client."""
    return MagicMock(spec=OpenAI)


class TestAgentStep:
    """Tests for AgentStep dataclass."""

    def test_agent_step_creation(self):
        """AgentStep can be created with all fields."""
        step = AgentStep(
            thought="I should load the structure first",
            tool_name="load_structure",
            tool_args={"pdb_id": "1UBQ"},
            tool_result=None,
            is_final=False,
            timestamp=1234567890.0,
        )
        assert step.thought == "I should load the structure first"
        assert step.tool_name == "load_structure"
        assert step.tool_args == {"pdb_id": "1UBQ"}
        assert step.is_final is False

    def test_agent_step_optional_fields(self):
        """AgentStep can be created with minimal fields."""
        step = AgentStep()
        assert step.thought is None
        assert step.tool_name is None
        assert step.tool_args is None
        assert step.tool_result is None
        assert step.is_final is False


class TestAgentRun:
    """Tests for AgentRun dataclass."""

    def test_agent_run_creation(self):
        """AgentRun can be created with required fields."""
        run = AgentRun(
            query="What is the structure of 1UBQ?",
            steps=[],
            final_answer="1UBQ is a ubiquitin structure.",
            total_steps=1,
            total_input_tokens=100,
            total_output_tokens=50,
            wall_time_seconds=1.5,
            model="MiniMax-M2.7",
        )
        assert run.query == "What is the structure of 1UBQ?"
        assert run.final_answer == "1UBQ is a ubiquitin structure."
        assert run.total_steps == 1

    def test_to_dict(self):
        """to_dict serializes the run correctly."""
        step = AgentStep(
            thought="Loading structure",
            tool_name="load_structure",
            tool_args={"pdb_id": "1UBQ"},
            tool_result=ToolResult(success=True, data="Structure loaded", raw={}),
            is_final=False,
        )
        run = AgentRun(
            query="Test query",
            steps=[step],
            final_answer="Done",
            total_steps=1,
            model="test",
        )
        d = run.to_dict()
        assert d["query"] == "Test query"
        assert d["final_answer"] == "Done"
        assert len(d["steps"]) == 1
        assert d["steps"][0]["tool_name"] == "load_structure"


class TestSimpleOneStep:
    """Test 1: Simple 1-step (model returns text, no tool calls)."""

    def test_one_step_no_tools(self, mock_client):
        """Agent gets a direct text response without any tool calls."""

        # Register a mock tool so registry isn't empty
        @tool(
            name="test_tool",
            toolset="test",
            description="A test tool",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def test_func():
            return ToolResult(success=True, data="test", raw={})

        # Mock the chat completion response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.content = "This is the final answer."
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_client.chat.completions.create = MagicMock(return_value=mock_response)

        with patch.object(OpenAI, "__init__", lambda self, **kw: None):
            with patch.object(OpenAI, "__enter__", lambda self: mock_client):
                with patch.object(OpenAI, "__exit__", lambda self, *a: None):
                    agent = MiraAgent(verbose=False, mode="react")
                    agent.client = mock_client
                    run = agent.run("What is 1+1?")

        assert run.final_answer == "This is the final answer."
        assert run.total_steps == 1
        assert run.steps[0].is_final is True


class TestTwoStep:
    """Test 2: 2-step (model calls tool, gets result, responds)."""

    def test_two_step_with_tool_call(self, mock_client):
        """Agent calls a tool and then responds with the final answer."""

        # Register a mock tool
        @tool(
            name="add_numbers",
            toolset="test",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        )
        def add_numbers(a: float, b: float):
            return ToolResult(success=True, data=f"The sum is {a + b}", raw={"sum": a + b})

        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: model requests tool
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].finish_reason = "tool_calls"
                mock_response.choices[0].message.content = "Let me calculate that."
                mock_response.choices[0].message.tool_calls = [MagicMock()]
                mock_response.choices[0].message.tool_calls[0].id = "call_123"
                mock_response.choices[0].message.tool_calls[0].function.name = "add_numbers"
                mock_response.choices[0].message.tool_calls[0].function.arguments = '{"a": 2, "b": 3}'
                mock_response.usage = MagicMock()
                mock_response.usage.prompt_tokens = 20
                mock_response.usage.completion_tokens = 10
                return mock_response
            else:
                # Second call: model returns final answer
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].finish_reason = "stop"
                mock_response.choices[0].message.content = "The answer is 5."
                mock_response.choices[0].message.tool_calls = None
                mock_response.usage = MagicMock()
                mock_response.usage.prompt_tokens = 30
                mock_response.usage.completion_tokens = 5
                return mock_response

        mock_client.chat.completions.create = mock_create

        with patch.object(OpenAI, "__init__", lambda self, **kw: None):
            with patch.object(OpenAI, "__enter__", lambda self: mock_client):
                with patch.object(OpenAI, "__exit__", lambda self, *a: None):
                    agent = MiraAgent(verbose=False, mode="react")
                    agent.client = mock_client
                    run = agent.run("What is 2+3?")

        assert run.final_answer == "The answer is 5."
        assert run.total_steps == 2
        assert run.steps[0].tool_name == "add_numbers"
        assert run.steps[0].tool_args == {"a": 2, "b": 3}
        assert run.steps[1].is_final is True


class TestMaxStepsEnforcement:
    """Test 3: max_steps enforcement."""

    def test_max_steps_stops_loop(self, mock_client):
        """Agent stops after reaching max_steps."""

        # Register a tool that always gets called
        @tool(
            name="dummy",
            toolset="test",
            description="A dummy tool",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def dummy():
            return ToolResult(success=True, data="done", raw={})

        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1

            # Always request tool call
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].finish_reason = "tool_calls"
            mock_response.choices[0].message.content = "Thinking..."
            mock_response.choices[0].message.tool_calls = [MagicMock()]
            mock_response.choices[0].message.tool_calls[0].id = f"call_{call_count}"
            mock_response.choices[0].message.tool_calls[0].function.name = "dummy"
            mock_response.choices[0].message.tool_calls[0].function.arguments = "{}"
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 10
            return mock_response

        mock_client.chat.completions.create = mock_create

        with patch.object(OpenAI, "__init__", lambda self, **kw: None):
            with patch.object(OpenAI, "__enter__", lambda self: mock_client):
                with patch.object(OpenAI, "__exit__", lambda self, *a: None):
                    agent = MiraAgent(max_steps=3, verbose=False, mode="react")
                    agent.client = mock_client
                    run = agent.run("Keep calling tools")

        # Should stop at max_steps (3 tool calls + 1 synthesis = 4 total calls)
        # Actually: 3 iterations of the loop, then synthesis call
        assert len(run.steps) == 3
        # And there should be a final synthesis message
        assert run.final_answer != ""


class TestErrorHandling:
    """Test 4: Error handling when tool fails."""

    def test_tool_error_is_handled(self, mock_client):
        """Agent handles tool errors gracefully."""

        @tool(
            name="failing_tool",
            toolset="test",
            description="A tool that fails",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def failing_tool():
            raise RuntimeError("Something went wrong")

        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: model requests tool
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].finish_reason = "tool_calls"
                mock_response.choices[0].message.content = "Let me try that."
                mock_response.choices[0].message.tool_calls = [MagicMock()]
                mock_response.choices[0].message.tool_calls[0].id = "call_1"
                mock_response.choices[0].message.tool_calls[0].function.name = "failing_tool"
                mock_response.choices[0].message.tool_calls[0].function.arguments = "{}"
                mock_response.usage = MagicMock()
                mock_response.usage.prompt_tokens = 10
                mock_response.usage.completion_tokens = 5
                return mock_response
            else:
                # Second call: model responds after seeing error
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].finish_reason = "stop"
                mock_response.choices[0].message.content = "I see the tool failed, but continuing."
                mock_response.choices[0].message.tool_calls = None
                mock_response.usage = MagicMock()
                mock_response.usage.prompt_tokens = 20
                mock_response.usage.completion_tokens = 8
                return mock_response

        mock_client.chat.completions.create = mock_create

        with patch.object(OpenAI, "__init__", lambda self, **kw: None):
            with patch.object(OpenAI, "__enter__", lambda self: mock_client):
                with patch.object(OpenAI, "__exit__", lambda self, *a: None):
                    agent = MiraAgent(verbose=False, mode="react")
                    agent.client = mock_client
                    run = agent.run("Use the failing tool")

        # Should have recorded the failed tool call
        assert run.steps[0].tool_name == "failing_tool"
        assert run.steps[0].tool_result.success is False
        assert "RuntimeError" in run.steps[0].tool_result.data


class TestMultipleToolCalls:
    """Test 5: Multiple tool calls in single response."""

    def test_multiple_tools_in_one_response(self, mock_client):
        """Agent handles multiple tool calls from a single response."""

        @tool(
            name="tool_a",
            toolset="test",
            description="Tool A",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def tool_a():
            return ToolResult(success=True, data="Result A", raw={})

        @tool(
            name="tool_b",
            toolset="test",
            description="Tool B",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def tool_b():
            return ToolResult(success=True, data="Result B", raw={})

        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: model requests multiple tools
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].finish_reason = "tool_calls"
                mock_response.choices[0].message.content = "Calling multiple tools."
                mock_response.choices[0].message.tool_calls = [
                    MagicMock(),
                    MagicMock(),
                ]
                mock_response.choices[0].message.tool_calls[0].id = "call_1"
                mock_response.choices[0].message.tool_calls[0].function.name = "tool_a"
                mock_response.choices[0].message.tool_calls[0].function.arguments = "{}"
                mock_response.choices[0].message.tool_calls[1].id = "call_2"
                mock_response.choices[0].message.tool_calls[1].function.name = "tool_b"
                mock_response.choices[0].message.tool_calls[1].function.arguments = "{}"
                mock_response.usage = MagicMock()
                mock_response.usage.prompt_tokens = 10
                mock_response.usage.completion_tokens = 10
                return mock_response
            else:
                # Second call: model returns final answer
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].finish_reason = "stop"
                mock_response.choices[0].message.content = "Both tools completed."
                mock_response.choices[0].message.tool_calls = None
                mock_response.usage = MagicMock()
                mock_response.usage.prompt_tokens = 30
                mock_response.usage.completion_tokens = 5
                return mock_response

        mock_client.chat.completions.create = mock_create

        with patch.object(OpenAI, "__init__", lambda self, **kw: None):
            with patch.object(OpenAI, "__enter__", lambda self: mock_client):
                with patch.object(OpenAI, "__exit__", lambda self, *a: None):
                    agent = MiraAgent(verbose=False, mode="react")
                    agent.client = mock_client
                    run = agent.run("Call both tools")

        assert run.final_answer == "Both tools completed."
        # 2 tool call steps + 1 final answer step = 3 total
        assert len(run.steps) == 3
        assert run.steps[0].tool_name == "tool_a"
        assert run.steps[1].tool_name == "tool_b"
        assert run.steps[2].is_final is True
