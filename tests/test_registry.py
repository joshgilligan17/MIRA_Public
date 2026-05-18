import pytest
from structagent.registry import ToolRegistry, ToolResult, tool, get_registry


# Reset registry for clean test state
def reset_registry():
    """Reset the singleton registry to empty state."""
    reg = ToolRegistry()
    reg._tools = {}


@pytest.fixture(autouse=True)
def clean_registry():
    reset_registry()
    yield
    reset_registry()


def test_empty_registry():
    """Registry starts empty."""
    reg = get_registry()
    assert reg.list_tools() == []


def test_tool_decorator_registers():
    """A decorated function gets registered."""

    @tool(
        name="test_tool",
        toolset="test",
        description="A test tool",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    def my_test_func():
        return ToolResult(success=True, data="ok", raw={})

    reg = get_registry()
    assert "test_tool" in reg.list_tools()


def test_get_tool_schemas_format():
    """get_tool_schemas returns valid OpenAI function-calling format."""

    @tool(
        name="my_tool",
        toolset="test",
        description="Does something",
        parameters={"type": "object", "properties": {"arg1": {"type": "string"}}, "required": ["arg1"]},
    )
    def another_func(arg1: str):
        return ToolResult(success=True, data="ok", raw={})

    reg = get_registry()
    schemas = reg.get_tool_schemas()

    assert len(schemas) == 1
    schema = schemas[0]
    assert schema["type"] == "function"
    assert "function" in schema
    assert schema["function"]["name"] == "my_tool"
    assert schema["function"]["description"] == "Does something"
    assert schema["function"]["parameters"]["type"] == "object"


def test_call_tool_dispatches():
    """call_tool dispatches correctly and returns ToolResult."""

    @tool(
        name="dispatch_test",
        toolset="test",
        description="Test dispatch",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    def dispatched_func():
        return ToolResult(success=True, data="dispatched!", raw={"key": "value"})

    reg = get_registry()
    result = reg.call_tool("dispatch_test")

    assert result.success is True
    assert result.data == "dispatched!"
    assert result.raw == {"key": "value"}


def test_call_tool_error_wrapping():
    """call_tool wraps errors in ToolResult."""

    @tool(
        name="error_tool",
        toolset="test",
        description="Raises error",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    def error_func():
        raise ValueError("test error")

    reg = get_registry()
    result = reg.call_tool("error_tool")

    assert result.success is False
    assert "test error" in result.error
    assert "ValueError" in result.data


def test_call_tool_unknown():
    """Calling unknown tool returns error."""
    reg = get_registry()
    result = reg.call_tool("nonexistent_tool")

    assert result.success is False
    assert "not found" in result.data


def test_check_fn_filtering():
    """Tools with failing check_fn are excluded from schemas."""

    def always_fail():
        return False

    def always_pass():
        return True

    @tool(
        name="filtered_tool",
        toolset="test",
        description="Should be filtered",
        parameters={"type": "object", "properties": {}, "required": []},
        check_fn=always_fail,
    )
    def filtered_func():
        return ToolResult(success=True, data="ok", raw={})

    @tool(
        name="kept_tool",
        toolset="test",
        description="Should be kept",
        parameters={"type": "object", "properties": {}, "required": []},
        check_fn=always_pass,
    )
    def kept_func():
        return ToolResult(success=True, data="ok", raw={})

    reg = get_registry()
    schemas = reg.get_tool_schemas()
    tool_names = [s["function"]["name"] for s in schemas]

    assert "filtered_tool" not in tool_names
    assert "kept_tool" in tool_names


def test_tool_schemas_filters_by_toolset():
    """get_tool_schemas filters by toolset when specified."""

    @tool(
        name="tool_a",
        toolset="group_a",
        description="Group A tool",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    def tool_a():
        return ToolResult(success=True, data="a", raw={})

    @tool(
        name="tool_b",
        toolset="group_b",
        description="Group B tool",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    def tool_b():
        return ToolResult(success=True, data="b", raw={})

    reg = get_registry()

    all_schemas = reg.get_tool_schemas()
    assert len(all_schemas) == 2

    a_only = reg.get_tool_schemas(toolsets=["group_a"])
    assert len(a_only) == 1
    assert a_only[0]["function"]["name"] == "tool_a"
