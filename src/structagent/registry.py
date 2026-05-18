import importlib
import importlib.util
import types
from dataclasses import dataclass, field
from typing import Callable, Optional
import time


@dataclass
class ToolEntry:
    name: str
    toolset: str
    description: str
    parameters: dict
    handler: Callable
    check_fn: Optional[Callable[[], bool]] = None
    module_path: Optional[str] = None
    function_name: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)


@dataclass
class LazyToolEntry:
    """Tool entry that supports lazy loading of handler functions."""

    name: str
    toolset: str
    description: str
    parameters: dict
    module_path: str
    function_name: str
    dependencies: list[str] = field(default_factory=list)
    check_fn: Optional[Callable[[], bool]] = None
    handler: Optional[Callable] = None
    _loaded_module: Optional[types.ModuleType] = None


@dataclass
class ToolResult:
    success: bool
    data: str  # LLM-readable narrative text
    raw: dict  # machine-readable data
    error: Optional[str] = None
    tool_name: str = ""
    execution_time_seconds: float = 0.0


class ToolRegistry:
    _instance = None
    _tools: dict = field(default_factory=dict)
    _loaded_modules: dict = field(default_factory=dict)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
            cls._instance._loaded_modules = {}
        return cls._instance

    def register(
        self,
        name: str,
        toolset: str,
        description: str,
        parameters: dict,
        handler: Callable,
        check_fn: Optional[Callable[[], bool]] = None,
        module_path: Optional[str] = None,
        function_name: Optional[str] = None,
        dependencies: list[str] = None,
    ):
        """Register a tool at import time."""
        self._tools[name] = ToolEntry(
            name=name,
            toolset=toolset,
            description=description,
            parameters=self._normalize_parameters(parameters),
            handler=handler,
            check_fn=check_fn,
            module_path=module_path,
            function_name=function_name,
            dependencies=dependencies or [],
        )

    def register_lazy(
        self,
        name: str,
        toolset: str,
        description: str,
        parameters: dict,
        module_path: str,
        function_name: str,
        dependencies: list[str] = None,
        check_fn: Optional[Callable[[], bool]] = None,
        handler: Optional[Callable] = None,
    ):
        """Register a tool for lazy loading."""
        self._tools[name] = LazyToolEntry(
            name=name,
            toolset=toolset,
            description=description,
            parameters=self._normalize_parameters(parameters),
            module_path=module_path,
            function_name=function_name,
            dependencies=dependencies or [],
            check_fn=check_fn,
            handler=handler,
        )

    def _normalize_parameters(self, parameters: Optional[dict]) -> dict:
        """Return an OpenAI-compatible JSON Schema object for tool arguments."""
        if not parameters:
            return {"type": "object", "properties": {}, "required": []}
        if parameters.get("type") == "object":
            normalized = dict(parameters)
            normalized.setdefault("properties", {})
            normalized.setdefault("required", [])
            return normalized
        return {
            "type": "object",
            "properties": parameters,
            "required": [],
        }

    def _check_dependency_available(self, dep: str) -> bool:
        """Check if a dependency is available via importlib without actually importing."""
        return importlib.util.find_spec(dep) is not None

    def _is_tool_available(self, entry) -> bool:
        """Check if a tool is available (dependencies met or check_fn passes)."""
        # Check dependencies via importlib.util.find_spec (no actual import)
        deps_available = True
        for dep in entry.dependencies:
            if not self._check_dependency_available(dep):
                deps_available = False
                break

        if not deps_available:
            return False

        # If check_fn is provided, call it regardless of dependencies
        if entry.check_fn is not None:
            return entry.check_fn()

        return True

    def get_available_tools(self, toolsets: Optional[list[str]] = None) -> list[dict]:
        """Returns OpenAI-compatible function-calling tool schemas for available tools.

        Checks dependencies via importlib.util.find_spec (no actual import).
        Falls back to check_fn for runtime checks.
        """
        schemas = []
        for name, entry in self._tools.items():
            if not self._is_tool_available(entry):
                continue
            if toolsets and entry.toolset not in toolsets:
                continue
            schema = {
                "type": "function",
                "function": {"name": entry.name, "description": entry.description, "parameters": entry.parameters},
            }
            schemas.append(schema)
        return schemas

    def get_tool_schemas(self, toolsets: Optional[list[str]] = None) -> list[dict]:
        """Returns OpenAI-compatible function-calling tool schemas."""
        # For backwards compatibility, delegate to get_available_tools
        return self.get_available_tools(toolsets)

    def _load_handler(self, entry: LazyToolEntry) -> Callable:
        """Lazy load the handler function from module_path and function_name."""
        # Check if already cached
        if entry._loaded_module is not None:
            return getattr(entry._loaded_module, entry.function_name)

        # Check if module is cached in registry
        if entry.module_path in self._loaded_modules:
            module = self._loaded_modules[entry.module_path]
        else:
            # Import the module
            module = importlib.import_module(entry.module_path)
            self._loaded_modules[entry.module_path] = module

        # Cache on entry for future lookups
        entry._loaded_module = module

        return getattr(module, entry.function_name)

    def call_tool(self, name: str, **kwargs) -> ToolResult:
        """Dispatches to handler with error wrapping."""
        start = time.time()
        if name not in self._tools:
            return ToolResult(
                success=False,
                data=f"Tool '{name}' not found",
                raw={},
                error=f"Unknown tool: {name}",
                tool_name=name,
                execution_time_seconds=time.time() - start,
            )
        entry = self._tools[name]

        # Lazy load handler if needed
        handler = entry.handler
        if handler is None:
            if isinstance(entry, LazyToolEntry):
                try:
                    handler = self._load_handler(entry)
                except Exception as e:
                    return ToolResult(
                        success=False,
                        data=f"Error loading tool '{name}': {type(e).__name__}: {str(e)}",
                        raw={},
                        error=str(e),
                        tool_name=name,
                        execution_time_seconds=time.time() - start,
                    )
            else:
                return ToolResult(
                    success=False,
                    data=f"Tool '{name}' has no handler and does not support lazy loading",
                    raw={},
                    error="No handler available",
                    tool_name=name,
                    execution_time_seconds=time.time() - start,
                )

        try:
            result = handler(**kwargs)
            if isinstance(result, ToolResult):
                result.tool_name = name
                result.execution_time_seconds = time.time() - start
                return result
            # Assume handler returned a ToolResult
            return result
        except Exception as e:
            return ToolResult(
                success=False,
                data=f"Error executing {name}: {type(e).__name__}: {str(e)}",
                raw={},
                error=str(e),
                tool_name=name,
                execution_time_seconds=time.time() - start,
            )

    def list_tools(self) -> list[str]:
        """List available tool names."""
        return list(self._tools.keys())

    def is_tool_registered(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools


def get_registry() -> ToolRegistry:
    return ToolRegistry()


def tool(
    name: str,
    toolset: str,
    description: str,
    parameters: dict,
    check_fn: Optional[Callable[[], bool]] = None,
    module_path: Optional[str] = None,
    function_name: Optional[str] = None,
    dependencies: list[str] = None,
):
    """Decorator that registers a function as an agent tool.

    Args:
        name: Tool name for registration
        toolset: Toolset category
        description: Human-readable description
        parameters: OpenAI-compatible parameters schema
        check_fn: Optional function to check if tool is available at runtime
        module_path: Module path for lazy loading (e.g., 'structagent.tools.pyrosetta_interface')
        function_name: Function name for lazy loading
        dependencies: List of dependency names to check via importlib.util.find_spec
    """

    def decorator(func: Callable):
        registry = get_registry()
        registry.register(
            name=name,
            toolset=toolset,
            description=description,
            parameters=parameters,
            handler=func,
            check_fn=check_fn,
            module_path=module_path,
            function_name=function_name,
            dependencies=dependencies,
        )
        return func

    return decorator


def lazy_tool(
    name: str,
    toolset: str,
    description: str,
    parameters: dict,
    module_path: str,
    function_name: str,
    dependencies: list[str] = None,
    check_fn: Optional[Callable[[], bool]] = None,
):
    """Decorator that registers a tool for lazy loading.

    The handler is not imported until the tool is actually called.

    Args:
        name: Tool name for registration
        toolset: Toolset category
        description: Human-readable description
        parameters: OpenAI-compatible parameters schema
        module_path: Module path to import (e.g., 'structagent.tools.pyrosetta_interface')
        function_name: Function name to get from module
        dependencies: List of dependency names to check via importlib.util.find_spec
        check_fn: Optional function for additional runtime checks
    """

    def decorator(func: Callable):
        registry = get_registry()
        registry.register_lazy(
            name=name,
            toolset=toolset,
            description=description,
            parameters=parameters,
            module_path=module_path,
            function_name=function_name,
            dependencies=dependencies,
            check_fn=check_fn,
            handler=func,  # Also store handler for backwards compatibility
        )
        return func

    return decorator
