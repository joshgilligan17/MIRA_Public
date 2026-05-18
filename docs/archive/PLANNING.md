# Planning: Planning-First Lazy-Loading Architecture for MIRA

## Overview

This document describes the planned refactoring of MIRA to support a **planning-first** approach with **lazy tool loading**. Currently, all tool modules are eagerly imported at startup, causing significant slowdowns from expensive dependencies like PyRosetta and ProDy.

---

## Current Architecture Problems

### Problem A: Eager Import at Module Load Time

In `tools/__init__.py` (lines 1-18), ALL tool modules are imported at once:

```python
from structagent.tools import structure_io
from structagent.tools import pyrosetta_interface  # EXPENSIVE
from structagent.tools import dynamics             # EXPENSIVE (ProDy)
# ... 17+ modules total
```

This causes `pyrosetta` (huge C++ binding), `prody`, `scipy`, etc. to load even if never needed.

### Problem B: `@tool` Decorator Fires at Import Time

In `registry.py` (lines 103-111), the `@tool` decorator registers tools at import time:

```python
def tool(name: str, toolset: str, description: str, parameters: dict,
         check_fn: Optional[Callable[[], bool]] = None):
    def decorator(func: Callable):
        registry = get_registry()
        registry.register(...)  # <-- Registration happens at import time
        return func
    return decorator
```

### Problem C: Planning and Execution Are Interleaved

In `agent.py` (lines 134-217), the ReAct loop handles thought and tool calls in the same LLM response. There is no separate planning phase.

### Problem D: CLI Inconsistency

`cli.py` only imports 6 tools, but `agent.py` imports all 17+. This inconsistency suggests partial awareness of the problem but no systematic solution.

---

## Proposed Architecture

```
+-------------------+
|   User Query     |
+--------+----------+
         |
         v
+--------+----------+
|  PLANNING PHASE  |  <-- LLM produces step-by-step plan (no tools executed)
+--------+----------+
         |
         v
+--------+----------+
|  Plan Validation |  <-- Verify plan is valid, check tool availability
+--------+----------+
         |
         v
+--------+----------+
| EXECUTION PHASE  |  <-- Execute tools based on validated plan
+--------+----------+
         |
         v
+--------+----------+
|  Synthesis       |  <-- Final answer from accumulated results
+-------------------+
```

---

## Key Component Designs

### 1. Lazy Tool Registry (`registry.py` changes)

#### Current `ToolEntry` (lines 5-12):

```python
@dataclass
class ToolEntry:
    name: str
    toolset: str
    description: str
    parameters: dict
    handler: Callable
    check_fn: Optional[Callable[[], bool]] = None
```

#### Proposed `LazyToolEntry`:

```python
@dataclass
class LazyToolEntry:
    name: str
    toolset: str
    description: str
    parameters: dict
    module_path: str        # e.g., "structagent.tools.pyrosetta_interface"
    class_name: str         # e.g., "score_interface" (function name)
    dependencies: list[str]  # e.g., ["pyrosetta"]
    check_fn: Optional[Callable[[], bool]] = None
    _handler: Optional[Callable] = None  # Lazily initialized
```

#### New `LazyToolRegistry` class:

```python
class LazyToolRegistry:
    _instance = None
    _entries: dict[str, LazyToolEntry] = {}
    _loaded_modules: set[str] = {}

    def register_lazy(
        self,
        name: str,
        toolset: str,
        description: str,
        parameters: dict,
        module_path: str,
        class_name: str,
        dependencies: list[str],
        check_fn: Optional[Callable[[], bool]] = None
    ):
        """Register tool metadata WITHOUT importing the module."""
        self._entries[name] = LazyToolEntry(...)

    def _load_tool_handler(self, name: str) -> Callable:
        """Actually import and return the tool handler."""
        entry = self._entries[name]

        # Check dependencies first
        for dep in entry.dependencies:
            if not self._check_dependency(dep):
                raise ImportError(f"Missing dependency '{dep}' for tool '{name}'")

        # Import the module
        if entry.module_path not in self._loaded_modules:
            import importlib
            importlib.import_module(entry.module_path)
            self._loaded_modules.add(entry.module_path)

        # Get the handler from the module
        module = importlib.import_module(entry.module_path)
        handler = getattr(module, entry.class_name)
        entry._handler = handler
        return handler

    def call_tool(self, name: str, **kwargs) -> ToolResult:
        """Lazily load and call a tool."""
        if name not in self._entries:
            return ToolResult(success=False, data=f"Tool '{name}' not found", ...)

        entry = self._entries[name]

        # Run check_fn if present
        if entry.check_fn is not None and not entry.check_fn():
            return ToolResult(success=False, data=f"Tool '{name}' dependencies not available", ...)

        # Lazy load
        handler = self._load_tool_handler(name)

        # Call handler
        start = time.time()
        try:
            result = handler(**kwargs)
            # ... rest of error handling
        except Exception as e:
            return ToolResult(success=False, ...)
```

---

### 2. New `@lazy_tool` Decorator

```python
def lazy_tool(
    name: str,
    toolset: str,
    description: str,
    parameters: dict,
    dependencies: list[str] = None,
    check_fn: Optional[Callable[[], bool]] = None
):
    """Decorator that registers tool metadata WITHOUT importing the module.

    Usage:
        @lazy_tool(
            name="score_interface",
            toolset="structure",
            description="...",
            parameters={...},
            dependencies=["pyrosetta"],
            check_fn=_check_pyrosetta
        )
        def score_interface(...):
            ...
    """
    def decorator(func: Callable):
        registry = get_registry()
        registry.register_lazy(
            name=name,
            toolset=toolset,
            description=description,
            parameters=parameters,
            module_path=func.__module__,
            class_name=func.__name__,
            dependencies=dependencies or [],
            check_fn=check_fn
        )
        return func  # Return original function (not loaded yet)
    return decorator
```

**Key insight**: The decorated function is returned as-is. Only metadata is registered. The actual import happens when `call_tool()` is invoked.

---

### 3. Tool Metadata Registry (New File: `tool_metadata.py`)

To enable planning before ANY tool is loaded, we need a separate metadata registry containing ONLY schema information:

```python
"""Tool metadata registry - schemas only, no handler imports."""

# Tool metadata: (name, toolset, description, parameters, dependencies, check_fn_name)
# This file should have ZERO imports of expensive packages

TOOL_METADATA = {
    "score_interface": {
        "toolset": "structure",
        "description": "Compute comprehensive interface scoring metrics...",
        "parameters": {
            "type": "object",
            "properties": {
                "pdb_path": {"type": "string", "description": "..."},
            },
            "required": ["pdb_path"]
        },
        "dependencies": ["pyrosetta"],
        "check_fn": "_check_pyrosetta"
    },
    "load_structure": {
        "toolset": "structure",
        "description": "Load a protein structure from the PDB...",
        "parameters": {...},
        "dependencies": ["gemmi", "requests"],
        "check_fn": None  # Always available
    },
    # ... all tools
}

def get_tool_schemas_for_planning(toolsets: list[str] = None) -> list[dict]:
    """Return schemas for ALL registered tools (for planning phase).

    This does NOT check dependencies - it's purely for the LLM to know
    what tools exist. Tool availability is checked at execution time.
    """
    schemas = []
    for name, meta in TOOL_METADATA.items():
        if toolsets and meta["toolset"] not in toolsets:
            continue
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": meta["description"],
                "parameters": meta["parameters"]
            }
        })
    return schemas
```

---

### 4. Planning Prompt Templates (`prompts.py`)

```python
PLANNING_SYSTEM_PROMPT = '''You are MIRA, a planning-first structural biology reasoning agent.

## Your Process

1. **Understand**: Read the user's question carefully.

2. **Plan**: Before taking ANY actions, create a step-by-step plan:
   - Step 1: [Tool name] - [What it accomplishes and why needed]
   - Step 2: [Tool name] - [What it accomplishes and why needed]
   - ...

3. **Execute**: Execute tools in your planned order.

4. **Revise**: If a tool fails, revise your plan.

## Important Constraints
- You MUST call tools through their exact names
- Each tool call is EXECUTED IMMEDIATELY before you see the next step
- Report results from each tool call before continuing
- If a tool fails, acknowledge the error and propose an alternative approach

## Planning Format
When given a query, first output:

**PLAN:**
1. [Step description with tool name]
2. [Step description with tool name]
...

Then execute.
'''

PLAN_VALIDATION_PROMPT = '''Review the user's query and the proposed plan.

Query: {query}
Proposed Plan:
{plan}

Does this plan make sense? If not, suggest improvements.
Be conservative - only include tools that are clearly necessary.
'''
```

---

### 5. Agent Loop Changes (`agent.py`)

#### New `MiraAgent` Architecture

```python
def run(self, query: str, context: Optional[str] = None) -> AgentRun:
    """Execute a single query with planning-first approach."""
    start_time = time.time()
    self._steps = []

    # PHASE 1: Planning (no tools executed)
    plan = self._create_plan(query, context)
    if not plan:
        return self._synthesize_final_answer(query, context, "Could not create plan")

    # PHASE 2: Validate plan
    if not self._validate_plan(plan):
        plan = self._revise_plan(query, plan)

    # PHASE 3: Execute plan
    execution_result = self._execute_plan(plan, query, context)

    # PHASE 4: Synthesize final answer
    final_answer = self._synthesize_from_execution(execution_result, query)

    return AgentRun(...)

def _create_plan(self, query: str, context: Optional[str]) -> list[dict]:
    """Send query to LLM and get back a step-by-step plan."""
    system_prompt = build_planning_system_prompt(context)

    # Get ALL tool schemas for planning (metadata only - no imports)
    tool_schemas = get_tool_metadata_registry().get_all_schemas()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    response = self.client.chat.completions.create(
        model=self.model,
        messages=messages,
        tools=tool_schemas,
        temperature=0.0,
    )

    # Parse the response to extract plan steps
    return self._parse_plan_from_response(response)

def _parse_plan_from_response(self, response) -> list[dict]:
    """Extract planned steps from LLM response.

    The LLM may respond with:
    1. Tool calls representing planned steps (preferred)
    2. Structured text "PLAN:\n1. tool_name: arg\n2. ..."
    """
    assistant_msg = response.choices[0].message

    # If tool_calls provided, use them as the plan
    if assistant_msg.tool_calls:
        plan = []
        for tc in assistant_msg.tool_calls:
            plan.append({
                "tool_name": tc.function.name,
                "tool_args": json.loads(tc.function.arguments or "{}"),
                "purpose": assistant_msg.content  # LLM explains why
            })
        return plan

    # Otherwise parse from content
    return []

def _validate_plan(self, plan: list[dict]) -> bool:
    """Validate that all tools in plan are available.

    This checks dependencies WITHOUT loading the actual tools.
    """
    registry = get_registry()
    for step in plan:
        tool_name = step["tool_name"]
        if not registry.is_tool_available(tool_name):
            return False
    return True

def _execute_plan(self, plan: list[dict], query: str, context: Optional[str]) -> dict:
    """Execute tools one-by-one based on the plan."""
    results = []

    for i, step in enumerate(plan):
        tool_name = step["tool_name"]
        tool_args = step["tool_args"]

        # Log the planned step
        self._log_step(
            thought=step.get("purpose", f"Executing step {i+1}: {tool_name}"),
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=None,
            is_final=False,
        )

        # Execute the tool (lazy-loaded)
        result = self.registry.call_tool(tool_name, **tool_args)

        self._log_step(
            thought=None,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=result,
            is_final=False,
        )

        results.append({
            "step": i + 1,
            "tool_name": tool_name,
            "args": tool_args,
            "result": result,
            "success": result.success
        })

        # If tool failed, decide whether to continue or revise
        if not result.success:
            # Could trigger replanning here
            pass

    return {"steps": results, "query": query}
```

---

## Tool Migration Guide

### Before (current `pyrosetta_interface.py`):

```python
from structagent.registry import tool, ToolResult

@tool(
    name="score_interface",
    toolset="structure",
    description="...",
    parameters={...},
    check_fn=_check_pyrosetta
)
def score_interface(pdb_path: str, ...):
    # ... implementation
```

### After:

```python
# No module-level imports of expensive packages!
from structagent.registry import lazy_tool, ToolResult

def _check_pyrosetta() -> bool:
    try:
        import pyrosetta
        return True
    except ImportError:
        return False

@lazy_tool(
    name="score_interface",
    toolset="structure",
    description="...",
    parameters={...},
    dependencies=["pyrosetta"],  # NEW: declare dependencies
    check_fn=_check_pyrosetta
)
def score_interface(pdb_path: str, ...):
    # ... implementation (pyrosetta imported INSIDE the function)
```

---

## Phase-by-Phase Flow

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: PLANNING                                           │
├─────────────────────────────────────────────────────────────┤
│ Input: User query                                           │
│ Tool: LLM with ALL tool schemas (metadata only)             │
│ Output: Structured plan (tool sequence)                     │
│                                                             │
│ Messages:                                                   │
│   system: PLANNING_SYSTEM_PROMPT                            │
│   user: {query}                                             │
│                                                             │
│ LLM Response: "PLAN: 1. load_structure for 1ABC..."         │
└─────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: PLAN VALIDATION (fast, no tool execution)          │
├─────────────────────────────────────────────────────────────┤
│ Check each planned tool:                                     │
│   1. Does tool exist in registry?                            │
│   2. Are dependencies available? (importlib check only)      │
│   3. Does check_fn pass?                                    │
│                                                             │
│ If validation fails:                                        │
│   - Ask LLM to revise plan                                  │
│   - OR return error to user                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: EXECUTION                                          │
├─────────────────────────────────────────────────────────────┤
│ For each step in validated plan:                            │
│   1. Lazy-load the tool handler (import module)              │
│   2. Execute the tool                                        │
│   3. Log the result                                         │
│   4. If failure: mark step, potentially trigger replanning   │
│   5. Continue to next step                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: SYNTHESIS                                          │
├─────────────────────────────────────────────────────────────┤
│ Input: All tool results + original query                     │
│ Tool: LLM (no tools)                                        │
│ Output: Final natural language answer                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Tool Dependency Chart

| Tool Name               | Dependencies         | Heavy? |
|-------------------------|---------------------|--------|
| load_structure          | gemmi, requests     | No     |
| get_residue_contacts    | scipy, Bio.PDB, gemmi| Medium |
| score_interface         | pyrosetta           | **YES**|
| analyze_interface_...   | pyrosetta           | **YES**|
| fast_relax               | pyrosetta           | **YES**|
| compute_normal_modes    | prody               | **YES**|
| compute_cross_...       | prody               | **YES**|
| predict_hinge_regions   | prody               | **YES**|
| compute_perturbation... | prody               | **YES**|
| renumber_pdb           | (none)              | No     |
| secondary_structure    | (none)              | No     |
| sasa                   | (none)              | No     |
| bfactor                | (none)              | No     |

---

## Files to Modify

| File | Lines | Change |
|------|-------|--------|
| `registry.py` | 1-112 | Add `LazyToolEntry`, `LazyToolRegistry`, `lazy_tool()` decorator |
| `prompts.py` | 1-150 | Add `PLANNING_SYSTEM_PROMPT`, `PLAN_VALIDATION_PROMPT` |
| `agent.py` | 64-270 | Refactor into `_create_plan()`, `_validate_plan()`, `_execute_plan()`, `_synthesize()` |
| `tools/__init__.py` | 1-18 | Remove eager imports, only load metadata |
| `tools/pyrosetta_interface.py` | 292-326 | Change `@tool` to `@lazy_tool`, declare dependencies |
| `tools/interface_energy.py` | 277-310 | Change `@tool` to `@lazy_tool`, declare dependencies |
| `tools/relaxation.py` | 29-52 | Change `@tool` to `@lazy_tool`, declare dependencies |
| `tools/dynamics.py` | 102-127 | Change `@tool` to `@lazy_tool`, declare dependencies |

---

## New Files

| File | Purpose |
|------|---------|
| `tool_metadata.py` | All tool schemas with zero imports |
| `dependency_manager.py` | Handles dependency checking (optional) |

---

## Potential Pitfalls and Mitigations

### Pitfall 1: Circular Imports
- **Problem**: `tools/__init__.py` imports modules that import from registry
- **Mitigation**: `tool_metadata.py` must have ZERO imports from tools/ or registry/

### Pitfall 2: Planning Without Knowing What's Available
- **Problem**: LLM plans tools in planning phase that aren't installed
- **Mitigation**: Phase 2 validates the plan before execution; if invalid, asks LLM to revise

### Pitfall 3: Backwards Compatibility
- **Problem**: Existing `@tool` decorator users need to migrate
- **Mitigation**: Keep `@tool` as alias to `@lazy_tool` with empty dependencies list (for light tools)

### Pitfall 4: Debugging Lazy Loading Failures
- **Problem**: If a tool fails to load, the error comes at execution time, not registration
- **Mitigation**: Add `registry.debug_tool(name)` method to force-load and see errors early

### Pitfall 5: Multiple Tools in One Response
- **Problem**: Current ReAct allows multiple tool calls per response
- **Mitigation**: Planning phase outputs sequence; execution phase runs sequentially, accumulating results

### Pitfall 6: Dynamic Tool Discovery
- **Problem**: User might ask for something that requires a tool not in metadata
- **Mitigation**: Phase 2 can detect "unknown tool" and either add it to plan dynamically or report back

---

## Implementation Sequence

1. **Phase 1**: Create `tool_metadata.py` with all tool schemas (no imports)
2. **Phase 2**: Add `LazyToolEntry` and `lazy_tool` to `registry.py`
3. **Phase 3**: Add planning prompts to `prompts.py`
4. **Phase 4**: Refactor `agent.py` to support planning phase
5. **Phase 5**: Migrate heavy tools (`pyrosetta_interface`, `interface_energy`, `relaxation`, `dynamics`) to `@lazy_tool`
6. **Phase 6**: Update `tools/__init__.py` to only import metadata
7. **Phase 7**: Update CLI to use new registry

---

## Benefits

1. **Fast startup**: No expensive imports until needed
2. **Planning visibility**: Users can see/review the plan before execution
3. **On-demand loading**: Tools load exactly when called
4. **Dynamic tool discovery**: Can extend plan mid-execution if needed
