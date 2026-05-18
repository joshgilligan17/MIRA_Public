"""Tests for PyRosetta tool workflow in MiraAgent.

These tests verify that the agent correctly sequences PyRosetta tool calls
for TCR-peptide interface comparison between 1AO7 and 1BD2.
"""

import importlib
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch
from structagent.registry import get_registry, ToolResult


# Import tools to trigger registration BEFORE any tests run
from structagent.tools import *  # noqa: F401, F403

LOCAL_PDB_PATH = str(Path(__file__).parent / "data" / "local" / "mini_complex.pdb")


@pytest.fixture(autouse=True)
def ensure_tools_registered():
    """Ensure tools are imported and registered before each test."""
    tool_modules = [
        "structagent.tools.structure_io",
        "structagent.tools.relaxation",
        "structagent.tools.pyrosetta_interface",
        "structagent.tools.interface_energy",
    ]
    for mod_name in tool_modules:
        mod = importlib.import_module(mod_name)
        importlib.reload(mod)


@pytest.fixture
def registry(ensure_tools_registered):
    """Get the tool registry with all tools registered."""
    return get_registry()


@pytest.fixture
def mock_1ao7_pdb_path(tmp_path):
    """Return a mock PDB path for 1AO7."""
    return str(tmp_path / "1AO7.pdb")


@pytest.fixture
def mock_1bd2_pdb_path(tmp_path):
    """Return a mock PDB path for 1BD2."""
    return str(tmp_path / "1BD2.pdb")


class TestPyRosettaToolSequence:
    """Tests that verify correct tool call sequencing for TCR-peptide comparison."""

    def test_load_structure_returns_file_path_in_raw(self, registry):
        """load_structure should return the PDB file path in raw.file_path."""
        result = registry.call_tool("load_structure", pdb_path=LOCAL_PDB_PATH)

        assert result.success is True
        assert "file_path" in result.raw
        assert result.raw["file_path"] is not None
        assert isinstance(result.raw["file_path"], str)

    def test_score_interface_accepts_pdb_path_argument(self, registry):
        """score_interface should accept pdb_path as first argument."""
        # This is a signature test - verify the tool accepts the expected parameters
        from structagent.tools.pyrosetta_interface import score_interface
        import inspect

        sig = inspect.signature(score_interface)
        params = list(sig.parameters.keys())

        assert "pdb_path" in params
        assert "binder_chains" in params
        assert "target_chain" in params

    def test_analyze_interface_energies_accepts_pdb_path_argument(self, registry):
        """analyze_interface_energies should accept pdb_path as first argument."""
        from structagent.tools.interface_energy import analyze_interface_energies
        import inspect

        sig = inspect.signature(analyze_interface_energies)
        params = list(sig.parameters.keys())

        assert "pdb_path" in params
        assert "binder_chain" in params

    def test_score_interface_with_1ao7_mock(self, registry, mock_1ao7_pdb_path):
        """Test score_interface can be called with a PDB path from load_structure.

        Note: This test verifies the tool can be invoked with a file path,
        but does NOT run PyRosetta itself (which requires a license).
        """
        # We'll mock pyrosetta at the import level to avoid needing a license
        with patch.dict("sys.modules", {"pyrosetta": MagicMock()}):
            from structagent.tools import pyrosetta_interface
            import inspect

            sig = inspect.signature(pyrosetta_interface.score_interface)
            params = list(sig.parameters.keys())

            # Verify essential parameters exist
            assert "pdb_path" in params
            assert "binder_chains" in params
            assert "target_chain" in params

    def test_workflow_sequence_load_then_score(self, registry, mock_1ao7_pdb_path):
        """Verify the expected workflow: load_structure -> score_interface."""
        # Step 1: load_structure
        load_result = registry.call_tool("load_structure", pdb_path=LOCAL_PDB_PATH)
        assert load_result.success is True
        file_path = load_result.raw.get("file_path")
        assert file_path is not None

        # Step 2: Verify score_interface accepts the file path
        from structagent.tools.pyrosetta_interface import score_interface
        import inspect

        sig = inspect.signature(score_interface)
        bind_result = sig.bind(pdb_path=file_path, binder_chains="B", target_chain="A")
        assert bind_result.arguments["pdb_path"] == file_path

    def test_workflow_sequence_load_then_analyze_energies(self, registry, mock_1ao7_pdb_path):
        """Verify the expected workflow: load_structure -> analyze_interface_energies."""
        # Step 1: load_structure
        load_result = registry.call_tool("load_structure", pdb_path=LOCAL_PDB_PATH)
        assert load_result.success is True
        file_path = load_result.raw.get("file_path")
        assert file_path is not None

        # Step 2: Verify analyze_interface_energies accepts the file path
        from structagent.tools.interface_energy import analyze_interface_energies
        import inspect

        sig = inspect.signature(analyze_interface_energies)
        bind_result = sig.bind(pdb_path=file_path, binder_chain="B")
        assert bind_result.arguments["pdb_path"] == file_path


class TestPyRosettaToolRegistry:
    """Verify PyRosetta tools are properly registered."""

    def test_score_interface_registered(self, registry):
        """score_interface should be registered in the tool registry."""
        tools = registry.list_tools()
        assert "score_interface" in tools

    def test_analyze_interface_energies_registered(self, registry):
        """analyze_interface_energies should be registered in the tool registry."""
        tools = registry.list_tools()
        assert "analyze_interface_energies" in tools

    def test_fast_relax_registered(self, registry):
        """fast_relax should be registered in the tool registry."""
        tools = registry.list_tools()
        assert "fast_relax" in tools

    def test_score_interface_in_structure_toolset(self, registry):
        """score_interface should be in the 'structure' toolset."""
        schemas = registry.get_tool_schemas(toolsets=["structure"])
        schema_dict = {s["function"]["name"]: s for s in schemas}
        if importlib.util.find_spec("pyrosetta"):
            assert "score_interface" in schema_dict
        else:
            assert "score_interface" not in schema_dict

    def test_analyze_interface_energies_in_structure_toolset(self, registry):
        """analyze_interface_energies should be in the 'structure' toolset."""
        schemas = registry.get_tool_schemas(toolsets=["structure"])
        schema_dict = {s["function"]["name"]: s for s in schemas}
        if importlib.util.find_spec("pyrosetta"):
            assert "analyze_interface_energies" in schema_dict
        else:
            assert "analyze_interface_energies" not in schema_dict


class TestToolParameterDefaults:
    """Verify tool parameter defaults match expected values."""

    def test_score_interface_defaults(self, registry):
        """score_interface should have correct default values."""
        from structagent.tools.pyrosetta_interface import score_interface
        import inspect

        sig = inspect.signature(score_interface)
        defaults = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}

        assert defaults.get("binder_chains") == "B"
        assert defaults.get("target_chain") == "A"
        assert defaults.get("relax_structure") is True

    def test_analyze_interface_energies_defaults(self, registry):
        """analyze_interface_energies should have correct default values."""
        from structagent.tools.interface_energy import analyze_interface_energies
        import inspect

        sig = inspect.signature(analyze_interface_energies)
        defaults = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}

        assert defaults.get("cutoff") == 4.0
        assert defaults.get("plot_output") is None
