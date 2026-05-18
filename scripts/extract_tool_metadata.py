#!/usr/bin/env python3
"""Extract @tool decorator metadata from tool files using AST parsing.

This script parses tool files without importing them to extract metadata
about each tool's name, toolset, description, parameters, and check_fn.
"""

import ast
import os
from pathlib import Path
from typing import Any


def extract_tool_metadata(source_path: str) -> list[dict[str, Any]]:
    """Extract tool metadata from a source file using AST parsing.

    Args:
        source_path: Path to the tool source file

    Returns:
        List of tool metadata dictionaries
    """
    with open(source_path, 'r') as f:
        source_code = f.read()

    tree = ast.parse(source_code)
    tools = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                # Check if decorator is a Call (e.g., @tool(...))
                if isinstance(decorator, ast.Call):
                    # Check if the function being called is 'tool'
                    func = decorator.func
                    is_tool_call = False

                    if isinstance(func, ast.Name) and func.id == 'tool':
                        is_tool_call = True
                    elif isinstance(func, ast.Attribute) and func.attr == 'tool':
                        is_tool_call = True

                    if is_tool_call:
                        metadata = _extract_decorator_args(decorator)
                        metadata['function_name'] = node.name
                        tools.append(metadata)

    return tools


def _extract_decorator_args(decorator: ast.Call) -> dict[str, Any]:
    """Extract arguments from a @tool(...) decorator call.

    Args:
        decorator: AST Call node representing the decorator

    Returns:
        Dictionary of extracted metadata
    """
    metadata = {
        'name': None,
        'toolset': None,
        'description': None,
        'parameters': None,
        'check_fn': None
    }

    if not decorator.keywords:
        return metadata

    for keyword in decorator.keywords:
        key = keyword.arg
        value = keyword.value

        if key == 'name':
            metadata['name'] = _extract_string_value(value)
        elif key == 'toolset':
            metadata['toolset'] = _extract_string_value(value)
        elif key == 'description':
            metadata['description'] = _extract_string_value(value)
        elif key == 'parameters':
            metadata['parameters'] = _extract_dict_value(value)
        elif key == 'check_fn':
            metadata['check_fn'] = _extract_func_name(value)

    return metadata


def _extract_string_value(node: ast.AST) -> str | None:
    """Extract string value from an AST node."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return node.value
    elif isinstance(node, ast.Str):  # Python 3.7 compatibility
        return node.s
    return None


def _extract_dict_value(node: ast.AST) -> dict | None:
    """Extract dict value from an AST node using ast.literal_eval or ast.unparse."""
    if isinstance(node, ast.Dict):
        # Try to reconstruct the dict using ast.unparse (Python 3.9+)
        try:
            import sys
            if sys.version_info >= (3, 9):
                return ast.literal_eval(ast.unparse(node))
        except (ValueError, SyntaxError):
            pass

        # Fall back to manual extraction for simple cases
        result = {}
        for key, val in zip(node.keys, node.values):
            if key is not None:
                k = _extract_string_value(key)
                v = _extract_literal_value(val)
                if k is not None:
                    result[k] = v
        return result if result else None
    elif isinstance(node, ast.Call):
        # Handle cases where parameters is passed as a variable reference
        if isinstance(node.func, ast.Name):
            return {'_type': 'variable', 'name': node.func.id}
    return None


def _extract_literal_value(node: ast.AST) -> Any:
    """Extract a literal value from an AST node."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Str):  # Python 3.7 compatibility
        return node.s
    elif isinstance(node, ast.Num):  # Python 3.7 compatibility
        return node.n
    elif isinstance(node, ast.NameConstant):  # Python 3.7 compatibility
        return node.value
    elif isinstance(node, ast.Dict):
        result = {}
        for k, v in zip(node.keys, node.values):
            if k is not None:
                key = _extract_string_value(k)
                val = _extract_literal_value(v)
                if key is not None:
                    result[key] = val
        return result
    elif isinstance(node, ast.List):
        return [_extract_literal_value(elt) for elt in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(_extract_literal_value(elt) for elt in node.elts)
    return None


def _extract_func_name(node: ast.AST) -> str | None:
    """Extract function name from an AST node (for check_fn)."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return node.attr
    return None


def get_tools_dir() -> Path:
    """Get the path to the tools directory."""
    # Navigate from scripts/ to src/structagent/tools/
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    return project_dir / "src" / "structagent" / "tools"


def get_tool_files() -> list[Path]:
    """Get list of tool files to parse."""
    tools_dir = get_tools_dir()
    tool_files = [
        "structure_io.py",
        "contacts.py",
        "sasa.py",
        "secondary_structure.py",
        "interface.py",
        "alignment.py",
        "annotations.py",
        "bfactor.py",
        "charge.py",
        "conservation.py",
        "ramachandran.py",
        "foldseek.py",
        "relaxation.py",
        "interface_energy.py",
        "pyrosetta_interface.py",
        "renumber_pdb.py",
        "dynamics.py",
    ]
    return [tools_dir / f for f in tool_files if (tools_dir / f).exists()]


def generate_tool_metadata_file(tools: list[dict], output_path: Path) -> None:
    """Generate the tool_metadata.py file with tool schemas.

    Args:
        tools: List of tool metadata dictionaries
        output_path: Path to write the output file
    """
    lines = [
        "# Auto-generated tool metadata - DO NOT EDIT",
        "# Generated by scripts/extract_tool_metadata.py",
        "",
        "",
        "TOOL_SCHEMAS = [",
    ]

    for tool in tools:
        lines.append("    {")
        lines.append(f"        'name': {repr(tool.get('name'))},")
        lines.append(f"        'toolset': {repr(tool.get('toolset'))},")
        lines.append(f"        'description': {repr(tool.get('description'))},")

        # Handle parameters
        params = _normalize_parameters(tool.get('parameters'))
        if params:
            params_repr = _format_parameters_for_output(params)
            lines.append(f"        'parameters': {params_repr},")
        else:
            lines.append("        'parameters': None,")

        # Handle check_fn
        check_fn = tool.get('check_fn')
        if check_fn:
            lines.append(f"        'check_fn': {repr(check_fn)},")
        else:
            lines.append("        'check_fn': None,")

        lines.append("    },")

    lines.append("]")
    lines.append("")
    lines.append("")
    lines.append("def get_tool_schemas_for_planning(toolsets=None):")
    lines.append("    \"\"\"Get tool schemas filtered by toolsets.")
    lines.append("")
    lines.append("    Args:")
    lines.append("        toolsets: Optional list of toolset names to filter by.")
    lines.append("                 If None, returns all tools.")
    lines.append("")
    lines.append("    Returns:")
    lines.append("        List of tool schema dicts matching the specified toolsets.")
    lines.append("    \"\"\"")
    lines.append("    if toolsets is None:")
    lines.append("        return TOOL_SCHEMAS")
    lines.append("    return [t for t in TOOL_SCHEMAS if t.get('toolset') in toolsets]")
    lines.append("")
    lines.append("")
    lines.append("def build_compact_tool_list(toolsets=None):")
    lines.append("    \"\"\"Build a compact tool list for planning prompts.\"\"\"")
    lines.append("    schemas = get_tool_schemas_for_planning(toolsets)")
    lines.append("    lines = []")
    lines.append("    for schema in schemas:")
    lines.append("        name = schema.get('name', 'unknown')")
    lines.append("        desc = (schema.get('description') or 'No description').split('\\n')[0][:80]")
    lines.append("        params = schema.get('parameters') or {}")
    lines.append("        props = params.get('properties', {})")
    lines.append("        param_names = list(props.keys())")
    lines.append("        if param_names:")
    lines.append("            lines.append(f\"- {name}: {desc} (params: {', '.join(param_names)})\")")
    lines.append("        else:")
    lines.append("            lines.append(f\"- {name}: {desc}\")")
    lines.append("    return '\\n'.join(lines) if lines else 'No tools available'")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def _format_parameters_for_output(params: Any) -> str:
    """Format parameters dict for output in the generated file."""
    return repr(params)


def _normalize_parameters(params: Any) -> dict:
    """Ensure extracted parameters are valid JSON Schema objects."""
    if not params:
        return {"type": "object", "properties": {}, "required": []}
    if isinstance(params, dict) and params.get("type") == "object":
        normalized = dict(params)
        normalized.setdefault("properties", {})
        normalized.setdefault("required", [])
        return normalized
    if isinstance(params, dict) and params.get("_type") == "variable":
        return {"type": "object", "properties": {}, "required": []}
    if isinstance(params, dict):
        return {"type": "object", "properties": params, "required": []}
    return {"type": "object", "properties": {}, "required": []}


def main():
    """Main entry point for the extraction script."""
    tools_dir = get_tools_dir()
    tool_files = get_tool_files()

    all_tools = []

    for tool_file in tool_files:
        print(f"Parsing: {tool_file.name}")
        try:
            tools = extract_tool_metadata(str(tool_file))
            all_tools.extend(tools)
            print(f"  Found {len(tools)} tool(s)")
        except Exception as e:
            print(f"  Error parsing {tool_file.name}: {e}")

    print(f"\nTotal tools found: {len(all_tools)}")

    # Generate output file
    output_path = tools_dir.parent / "tool_metadata.py"
    generate_tool_metadata_file(all_tools, output_path)
    print(f"\nGenerated: {output_path}")


if __name__ == "__main__":
    main()
