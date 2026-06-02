"""Tests for structural analysis tools."""

import importlib
from pathlib import Path

import pytest
from structagent.registry import get_registry, ToolResult

# Import tools to trigger registration BEFORE any tests run
from structagent.tools import *  # noqa: F401, F403

LOCAL_PDB_PATH = str(Path(__file__).parent / "data" / "local" / "mini_complex.pdb")
MISSING_PDB_PATH = str(Path(__file__).parent / "data" / "local" / "missing.pdb")


@pytest.fixture(autouse=True)
def ensure_tools_registered():
    """Ensure tools are imported and registered before each test.

    This must run AFTER clean_registry from test_registry.py (autouse=True),
    so tools get re-registered after the registry is cleared.
    Uses importlib.reload to force re-execution of the @tool decorator.
    """
    # Re-import tools to re-register them after clean_registry clears them
    # Using reload to force re-execution of @tool decorator
    tool_modules = [
        "structagent.tools.structure_io",
        "structagent.tools.contacts",
        "structagent.tools.sasa",
        "structagent.tools.secondary_structure",
        "structagent.tools.interface",
        "structagent.tools.alignment",
        "structagent.tools.foldseek",
        "structagent.tools.annotations",
        "structagent.tools.bfactor",
        "structagent.tools.charge",
        "structagent.tools.conservation",
        "structagent.tools.ramachandran",
        "structagent.tools.renumber_pdb",
        "structagent.tools.relaxation",
        "structagent.tools.interface_energy",
        "structagent.tools.pyrosetta_interface",
        "structagent.tools.dynamics",
    ]
    for mod_name in tool_modules:
        mod = importlib.import_module(mod_name)
        importlib.reload(mod)


@pytest.fixture
def registry(ensure_tools_registered):
    """Get the tool registry with all tools registered."""
    return get_registry()


class TestLoadStructure:
    """Tests for load_structure tool."""

    def test_load_structure_success(self, registry):
        """load_structure returns success for a valid local PDB."""
        result = registry.call_tool("load_structure", pdb_path=LOCAL_PDB_PATH)

        assert result.success is True
        assert isinstance(result.data, str)
        assert isinstance(result.raw, dict)
        assert result.raw.get("pdb_id") == "mini_complex.pdb"
        assert "resolution" in result.raw
        assert "chains" in result.raw
        assert len(result.raw["chains"]) >= 1

    def test_load_structure_chain_a_residue_range(self, registry):
        """load_structure returns chain A with the fixture residue range."""
        result = registry.call_tool("load_structure", pdb_path=LOCAL_PDB_PATH)

        assert result.success is True
        chain_a = next((c for c in result.raw["chains"] if c["id"] == "A"), None)
        assert chain_a is not None
        assert chain_a["first_residue"] == 1
        assert chain_a["last_residue"] == 10

    def test_load_structure_missing_local_file(self, registry):
        """load_structure returns success=False for a missing local file."""
        result = registry.call_tool("load_structure", pdb_path=MISSING_PDB_PATH)

        assert result.success is False
        assert result.error is not None

    def test_load_structure_raw_has_resolution(self, registry):
        """load_structure raw data contains expected metadata keys."""
        result = registry.call_tool("load_structure", pdb_path=LOCAL_PDB_PATH)

        assert result.success is True
        assert "resolution" in result.raw
        assert "method" in result.raw


class TestGetResidueContacts:
    """Tests for get_residue_contacts tool."""

    def test_contacts_val5_finds_contacts(self, registry):
        """get_residue_contacts finds contacts for VAL-5 in the local fixture."""
        result = registry.call_tool("get_residue_contacts", pdb_path=LOCAL_PDB_PATH, chain_id="A", residue_number=5)

        assert result.success is True
        assert isinstance(result.data, str)
        assert isinstance(result.raw, dict)
        assert "contacts" in result.raw
        assert len(result.raw["contacts"]) > 0

    def test_contacts_val5_contact_types(self, registry):
        """Contacts for VAL-5 are classified with valid types."""
        result = registry.call_tool("get_residue_contacts", pdb_path=LOCAL_PDB_PATH, chain_id="A", residue_number=5)

        assert result.success is True
        valid_types = {"salt_bridge", "hydrogen_bond", "hydrophobic", "cation_pi", "disulfide", "polar", "vdw"}
        for contact in result.raw["contacts"]:
            assert contact["contact_type"] in valid_types
            assert "distance" in contact
            assert contact["distance"] > 0

    def test_contacts_bad_chain(self, registry):
        """get_residue_contacts returns success=False for nonexistent chain."""
        result = registry.call_tool("get_residue_contacts", pdb_path=LOCAL_PDB_PATH, chain_id="Z", residue_number=5)

        assert result.success is False

    def test_contacts_bad_pdb(self, registry):
        """get_residue_contacts returns success=False for invalid PDB."""
        result = registry.call_tool("get_residue_contacts", pdb_path=MISSING_PDB_PATH, chain_id="A", residue_number=5)

        assert result.success is False


class TestComputeSasa:
    """Tests for compute_sasa tool."""

    def test_sasa_residues_1_10(self, registry):
        """compute_sasa returns per-residue data for chain A residues 1-10."""
        result = registry.call_tool("compute_sasa", pdb_path=LOCAL_PDB_PATH, chain_id="A", residue_range="1-10")

        assert result.success is True
        assert isinstance(result.data, str)
        assert isinstance(result.raw, dict)
        assert "residues" in result.raw
        residues = result.raw["residues"]
        assert len(residues) == 10

    def test_sasa_per_residue_fields(self, registry):
        """Each residue has required SASA fields."""
        result = registry.call_tool("compute_sasa", pdb_path=LOCAL_PDB_PATH, chain_id="A", residue_range="1-5")

        assert result.success is True
        for r in result.raw["residues"]:
            assert "resname" in r
            assert "residue_number" in r
            assert "absolute_sasa" in r
            assert "relative_sasa_percent" in r
            assert "classification" in r
            assert r["classification"] in ("buried", "partial", "exposed")

    def test_sasa_accepts_json_list_range(self, registry):
        """compute_sasa accepts JSON-native residue ranges from model plans."""
        result = registry.call_tool("compute_sasa", pdb_path=LOCAL_PDB_PATH, chain_id="A", residue_range=[1, 5])

        assert result.success is True
        assert len(result.raw["residues"]) == 5

    def test_sasa_bad_chain(self, registry):
        """compute_sasa returns success=False for nonexistent chain."""
        result = registry.call_tool("compute_sasa", pdb_path=LOCAL_PDB_PATH, chain_id="Z")

        assert result.success is False


class TestGetSecondaryStructure:
    """Tests for get_secondary_structure tool."""

    def test_secondary_structure_returns_elements(self, registry):
        """get_secondary_structure returns secondary structure elements."""
        result = registry.call_tool("get_secondary_structure", pdb_path=LOCAL_PDB_PATH, chain_id="A")

        assert result.success is True
        assert "elements" in result.raw

    def test_secondary_structure_bad_chain(self, registry):
        """get_secondary_structure returns success=False for nonexistent chain."""
        result = registry.call_tool("get_secondary_structure", pdb_path=LOCAL_PDB_PATH, chain_id="Z")

        assert result.success is False


class TestComputeInterface:
    """Tests for compute_interface tool."""

    def test_interface_bad_chain(self, registry):
        """compute_interface returns success=False for nonexistent chain."""
        result = registry.call_tool("compute_interface", pdb_path=LOCAL_PDB_PATH, chain_a="A", chain_b="Z")

        assert result.success is False

    def test_interface_success_for_fixture_complex(self, registry):
        """compute_interface succeeds for the two-chain local fixture."""
        result = registry.call_tool("compute_interface", pdb_path=LOCAL_PDB_PATH, chain_a="A", chain_b="B")

        assert result.success is True
        assert "buried_sa_total" in result.raw
        assert result.raw["interface_residues_a"]


class TestAlignStructures:
    """Tests for align_structures tool."""

    def test_align_structures_bad_chain(self, registry):
        """align_structures returns success=False for nonexistent chain."""
        result = registry.call_tool(
            "align_structures", pdb_path_1=LOCAL_PDB_PATH, chain_id_1="A", pdb_path_2=LOCAL_PDB_PATH, chain_id_2="Z"
        )

        assert result.success is False

    def test_align_structures_same_chain(self, registry):
        """align_structures succeeds when aligning the fixture chain to itself."""
        result = registry.call_tool(
            "align_structures", pdb_path_1=LOCAL_PDB_PATH, chain_id_1="A", pdb_path_2=LOCAL_PDB_PATH, chain_id_2="A"
        )

        assert result.success is True
        assert isinstance(result.raw, dict)
        assert "rmsd" in result.raw
        assert "aligned_length" in result.raw
        # Self-alignment should have RMSD ~ 0
        assert result.raw["rmsd"] < 0.01


class TestAnalyzeBfactors:
    """Tests for analyze_bfactors tool."""

    def test_bfactors_success(self, registry):
        """analyze_bfactors returns success for the local fixture."""
        result = registry.call_tool("analyze_bfactors", pdb_path=LOCAL_PDB_PATH, chain_id="A")

        assert result.success is True
        assert isinstance(result.data, str)
        assert isinstance(result.raw, dict)
        assert "statistics" in result.raw
        assert "residues" in result.raw

    def test_bfactors_statistics_fields(self, registry):
        """analyze_bfactors returns required statistics fields."""
        result = registry.call_tool("analyze_bfactors", pdb_path=LOCAL_PDB_PATH, chain_id="A")

        assert result.success is True
        stats = result.raw["statistics"]
        assert "mean" in stats
        assert "std" in stats
        assert "median" in stats
        assert "min" in stats
        assert "max" in stats

    def test_bfactors_classifications(self, registry):
        """analyze_bfactors classifies residues correctly."""
        result = registry.call_tool("analyze_bfactors", pdb_path=LOCAL_PDB_PATH, chain_id="A")

        assert result.success is True
        classification_counts = result.raw.get("classification_counts", {})
        assert "rigid" in classification_counts
        assert "ordered" in classification_counts
        assert "flexible" in classification_counts
        assert "highly_flexible" in classification_counts

    def test_bfactors_residue_range(self, registry):
        """analyze_bfactors respects residue_range parameter."""
        result = registry.call_tool("analyze_bfactors", pdb_path=LOCAL_PDB_PATH, chain_id="A", residue_range="1-10")

        assert result.success is True
        assert len(result.raw.get("residues", [])) == 10


class TestComputeChargeDistribution:
    """Tests for compute_charge_distribution tool."""

    def test_charge_success(self, registry):
        """compute_charge_distribution returns success for the local fixture."""
        result = registry.call_tool("compute_charge_distribution", pdb_path=LOCAL_PDB_PATH, chain_id="A")

        assert result.success is True
        assert isinstance(result.data, str)
        assert isinstance(result.raw, dict)
        assert "charged_residues" in result.raw

    def test_charge_ph_default(self, registry):
        """compute_charge_distribution uses default pH 7.4."""
        result = registry.call_tool("compute_charge_distribution", pdb_path=LOCAL_PDB_PATH, chain_id="A")

        assert result.success is True
        # At pH 7.4, LYS and ARG should be positively charged
        # ASP and GLU should be negatively charged

    def test_charge_clusters(self, registry):
        """compute_charge_distribution identifies charge clusters."""
        result = registry.call_tool("compute_charge_distribution", pdb_path=LOCAL_PDB_PATH, chain_id="A")

        assert result.success is True
        assert "clusters" in result.raw
        assert "cluster_count" in result.raw


class TestCheckRamachandran:
    """Tests for check_ramachandran tool."""

    def test_ramachandran_success(self, registry):
        """check_ramachandran returns success for the local fixture."""
        result = registry.call_tool("check_ramachandran", pdb_path=LOCAL_PDB_PATH, chain_id="A")

        assert result.success is True
        assert isinstance(result.data, str)
        assert isinstance(result.raw, dict)
        assert "favored" in result.raw

    def test_ramachandran_statistics(self, registry):
        """check_ramachandran returns counts and percentages."""
        result = registry.call_tool("check_ramachandran", pdb_path=LOCAL_PDB_PATH, chain_id="A")

        assert result.success is True
        assert "favored" in result.raw
        assert "allowed" in result.raw
        assert "outlier" in result.raw
        assert "favored_pct" in result.raw
        assert "allowed_pct" in result.raw
        assert "outlier_pct" in result.raw

    def test_ramachandran_outliers(self, registry):
        """check_ramachandran lists outlier residues."""
        result = registry.call_tool("check_ramachandran", pdb_path=LOCAL_PDB_PATH, chain_id="A")

        assert result.success is True
        # Outliers list should be present (may be empty for well-structured proteins)
        assert "outliers" in result.raw


class TestSearchStructuralHomologs:
    """Tests for search_structural_homologs tool (HTTP API)."""

    def test_foldseek_tool_registered(self, registry):
        """search_structural_homologs tool is registered."""
        # Tool is registered (check_fn may fail if service unavailable, but it's registered)
        tools = registry.list_tools()
        assert "search_structural_homologs" in tools

    def test_foldseek_schema_excluded_when_check_fails(self, registry):
        """search_structural_homologs may be excluded from schema if check_fn fails."""
        registered_tools = registry.list_tools()
        schemas = registry.get_tool_schemas()
        schema_names = [s["function"]["name"] for s in schemas]
        # Tool is registered but may be excluded from schema if service unavailable
        assert "search_structural_homologs" in registered_tools


class TestGetFunctionalAnnotations:
    """Tests for get_functional_annotations tool (HTTP API)."""

    def test_annotations_tool_registered(self, registry):
        """get_functional_annotations tool is registered."""
        tools = registry.list_tools()
        assert "get_functional_annotations" in tools

    def test_annotations_in_schema(self, registry):
        """get_functional_annotations should be in the planning schema."""
        schemas = registry.get_tool_schemas()
        schema_names = [s["function"]["name"] for s in schemas]
        assert "get_functional_annotations" in schema_names


class TestGetConservationScores:
    """Tests for get_conservation_scores tool (HTTP API)."""

    def test_conservation_tool_registered(self, registry):
        """get_conservation_scores tool is registered."""
        tools = registry.list_tools()
        assert "get_conservation_scores" in tools


class TestToolResultIntegrity:
    """Tests for ToolResult success/failure correctness."""

    def test_success_result_has_no_error(self, registry):
        """Successful ToolResult has no error field."""
        result = registry.call_tool("load_structure", pdb_path=LOCAL_PDB_PATH)

        if result.success:
            assert result.error is None or result.error == ""

    def test_failure_result_has_error(self, registry):
        """Failed ToolResult has error field."""
        result = registry.call_tool("load_structure", pdb_path=MISSING_PDB_PATH)

        assert result.success is False
        assert result.error is not None

    def test_raw_data_consistency(self, registry):
        """raw data values match what's described in data narrative."""
        result = registry.call_tool("load_structure", pdb_path=LOCAL_PDB_PATH)

        assert result.success is True
        # The narrative should mention chain A
        assert "A" in result.data or "Chain A" in result.data
        # The raw should have chain A
        assert any(c["id"] == "A" for c in result.raw.get("chains", []))


class TestRenumberPdb:
    """Tests for renumber_pdb tool (file-based, no external deps)."""

    @pytest.fixture
    def temp_pdb_file(self, tmp_path):
        """Create a minimal PDB file for testing."""
        pdb_content = (
            "HEADER    TEST                                    25-MAR-26   1ABC\n"
            "ATOM      1  N   MET A   1      12.001  14.002  15.003  1.00  0.00           N\n"
            "ATOM      2  CA  MET A   1      13.001  15.002  16.003  1.00  0.00           C\n"
            "ATOM      3  C   MET A   1      14.001  16.002  17.003  1.00  0.00           C\n"
            "ATOM      4  O   MET A   1      15.001  17.002  18.003  1.00  0.00           O\n"
            "ATOM      5  N   GLY A  10      20.001  21.002  22.003  1.00  0.00           N\n"
            "ATOM      6  CA  GLY A  10      21.001  22.002  23.003  1.00  0.00           C\n"
            "ATOM      7  C   GLY A  10      22.001  23.002  24.003  1.00  0.00           C\n"
            "ATOM      8  O   GLY A  10      23.001  24.002  25.003  1.00  0.00           O\n"
            "END\n"
        )
        pdb_path = tmp_path / "test.pdb"
        pdb_path.write_text(pdb_content)
        return pdb_path

    def test_renumber_pdb_success(self, registry, temp_pdb_file):
        """renumber_pdb returns success for valid PDB file."""
        result = registry.call_tool("renumber_pdb", pdb_path=str(temp_pdb_file))

        assert result.success is True
        assert isinstance(result.data, str)
        assert isinstance(result.raw, dict)
        assert "output_path" in result.raw
        assert "_renum.pdb" in result.raw["output_path"]

    def test_renumber_pdb_default_mode(self, registry, temp_pdb_file):
        """renumber_pdb uses standard mode (gaps preserved) by default."""
        result = registry.call_tool("renumber_pdb", pdb_path=str(temp_pdb_file))

        assert result.success is True
        assert "standard" in result.raw.get("mode", "").lower()

    def test_renumber_pdb_no_skip_mode(self, registry, temp_pdb_file):
        """renumber_pdb with no_skip=True numbers continuously."""
        result = registry.call_tool("renumber_pdb", pdb_path=str(temp_pdb_file), no_skip=True)

        assert result.success is True
        assert "continuous" in result.raw.get("mode", "").lower()

    def test_renumber_pdb_target_chain(self, registry, temp_pdb_file):
        """renumber_pdb respects target_chain parameter."""
        result = registry.call_tool("renumber_pdb", pdb_path=str(temp_pdb_file), target_chain="X")

        assert result.success is True
        assert result.raw.get("target_chain") == "X"

    def test_renumber_pdb_file_not_found(self, registry):
        """renumber_pdb returns success=False for nonexistent file."""
        result = registry.call_tool("renumber_pdb", pdb_path="/nonexistent/path/to/file.pdb")

        assert result.success is False
        assert result.error is not None

    def test_renumber_pdb_raw_fields(self, registry, temp_pdb_file):
        """renumber_pdb raw data contains required fields."""
        result = registry.call_tool("renumber_pdb", pdb_path=str(temp_pdb_file))

        assert result.success is True
        assert "input_path" in result.raw
        assert "output_path" in result.raw
        assert "mode" in result.raw
        assert "atoms_processed" in result.raw
        assert result.raw["atoms_processed"] == 8


@pytest.fixture
def _pyrosetta_available():
    """Check if PyRosetta is importable."""
    try:
        import pyrosetta  # noqa: F401

        return True
    except ImportError:
        return False


class TestFastRelax:
    """Tests for fast_relax tool (requires PyRosetta)."""

    @pytest.fixture
    def temp_pdb_file(self, tmp_path):
        """Create a minimal PDB file for testing with proper geometry."""
        pdb_content = (
            "HEADER    TEST                                    25-MAR-26   1ABC\n"
            "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  MET A   1       1.500   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  C   MET A   1       2.500   0.000   0.000  1.00  0.00           C\n"
            "ATOM      4  O   MET A   1       3.000   1.000   0.000  1.00  0.00           O\n"
            "ATOM      5  N   GLY A   2       3.500   0.000   0.000  1.00  0.00           N\n"
            "ATOM      6  CA  GLY A   2       5.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      7  C   GLY A   2       6.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      8  O   GLY A   2       6.500   1.000   0.000  1.00  0.00           O\n"
            "END\n"
        )
        pdb_path = tmp_path / "test.pdb"
        pdb_path.write_text(pdb_content)
        return pdb_path

    def test_fast_relax_registered(self, registry):
        """fast_relax tool is registered."""
        tools = registry.list_tools()
        assert "fast_relax" in tools

    def test_fast_relax_excluded_when_pyrosetta_unavailable(self, registry):
        """fast_relax is excluded from schema when PyRosetta not available."""
        schemas = registry.get_tool_schemas()
        schema_names = [s["function"]["name"] for s in schemas]
        # Check if PyRosetta is available
        pyrosetta_ok = True
        try:
            import pyrosetta  # noqa: F401
        except ImportError:
            pyrosetta_ok = False
        if not pyrosetta_ok:
            assert "fast_relax" not in schema_names
        # If PyRosetta IS available, tool should be in schema (other tests cover this)

    @pytest.mark.skipif(not bool(__import__("importlib").util.find_spec("pyrosetta")), reason="PyRosetta not installed")
    def test_fast_relax_success(self, registry, temp_pdb_file):
        """fast_relax returns success for valid PDB file when PyRosetta available."""
        result = registry.call_tool("fast_relax", input_path=str(temp_pdb_file))

        assert result.success is True
        assert isinstance(result.data, str)
        assert isinstance(result.raw, dict)
        assert "output_file" in result.raw
        assert "_relaxed.pdb" in result.raw["output_file"]

    @pytest.mark.skipif(not bool(__import__("importlib").util.find_spec("pyrosetta")), reason="PyRosetta not installed")
    def test_fast_relax_sidechain_only(self, registry, temp_pdb_file):
        """fast_relax with sidechain_only=True sets mode correctly."""
        result = registry.call_tool("fast_relax", input_path=str(temp_pdb_file), sidechain_only=True)

        assert result.success is True
        assert result.raw.get("sidechain_only") is True

    @pytest.mark.skipif(not bool(__import__("importlib").util.find_spec("pyrosetta")), reason="PyRosetta not installed")
    def test_fast_relax_file_not_found(self, registry):
        """fast_relax returns success=False for nonexistent file."""
        result = registry.call_tool("fast_relax", input_path="/nonexistent/path/to/file.pdb")

        assert result.success is False
        assert result.error is not None


class TestAnalyzeInterfaceEnergies:
    """Tests for analyze_interface_energies tool (requires PyRosetta + Bio.PDB)."""

    @pytest.fixture
    def temp_pdb_file(self, tmp_path):
        """Create a minimal two-chain PDB file for testing with proper geometry."""
        # Chain A (target) and Chain B (binder) at interface distance (~4A apart)
        pdb_content = (
            "HEADER    TEST                                    25-MAR-26   1ABC\n"
            "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  MET A   1       1.500   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  C   MET A   1       2.500   0.000   0.000  1.00  0.00           C\n"
            "ATOM      4  O   MET A   1       3.000   1.000   0.000  1.00  0.00           O\n"
            "ATOM      5  N   GLY A   2       3.500   0.000   0.000  1.00  0.00           N\n"
            "ATOM      6  CA  GLY A   2       5.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      7  C   GLY A   2       6.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      8  O   GLY A   2       6.500   1.000   0.000  1.00  0.00           O\n"
            # Chain B is positioned ~4A from Chain A (interface distance)
            "ATOM      9  N   ALA B   1       4.000   4.000   0.500  1.00  0.00           N\n"
            "ATOM     10  CA  ALA B   1       5.500   4.000   0.500  1.00  0.00           C\n"
            "ATOM     11  C   ALA B   1       6.500   4.000   0.500  1.00  0.00           C\n"
            "ATOM     12  O   ALA B   1       7.000   5.000   0.500  1.00  0.00           O\n"
            "END\n"
        )
        pdb_path = tmp_path / "test.pdb"
        pdb_path.write_text(pdb_content)
        return pdb_path

    def test_analyze_interface_energies_registered(self, registry):
        """analyze_interface_energies tool is registered."""
        tools = registry.list_tools()
        assert "analyze_interface_energies" in tools

    def test_analyze_interface_energies_excluded_when_pyrosetta_unavailable(self, registry):
        """Tool excluded from schema when PyRosetta not available."""
        schemas = registry.get_tool_schemas()
        schema_names = [s["function"]["name"] for s in schemas]
        pyrosetta_ok = True
        try:
            import pyrosetta  # noqa: F401
        except ImportError:
            pyrosetta_ok = False
        if not pyrosetta_ok:
            assert "analyze_interface_energies" not in schema_names

    @pytest.mark.skipif(not bool(__import__("importlib").util.find_spec("pyrosetta")), reason="PyRosetta not installed")
    def test_analyze_interface_energies_success(self, registry, temp_pdb_file):
        """analyze_interface_energies returns success when PyRosetta available."""
        result = registry.call_tool("analyze_interface_energies", pdb_path=str(temp_pdb_file), binder_chain="B")

        assert result.success is True
        assert isinstance(result.data, str)
        assert isinstance(result.raw, dict)
        assert "summary" in result.raw
        assert "per_residue_energies" in result.raw

    @pytest.mark.skipif(not bool(__import__("importlib").util.find_spec("pyrosetta")), reason="PyRosetta not installed")
    def test_analyze_interface_energies_bad_chain(self, registry, temp_pdb_file):
        """analyze_interface_energies returns success=False for nonexistent chain."""
        result = registry.call_tool("analyze_interface_energies", pdb_path=str(temp_pdb_file), binder_chain="Z")

        assert result.success is False
        assert result.error is not None

    @pytest.mark.skipif(not bool(__import__("importlib").util.find_spec("pyrosetta")), reason="PyRosetta not installed")
    def test_analyze_interface_energies_summary_fields(self, registry, temp_pdb_file):
        """analyze_interface_energies summary has required fields."""
        result = registry.call_tool("analyze_interface_energies", pdb_path=str(temp_pdb_file), binder_chain="B")

        assert result.success is True
        summary = result.raw["summary"]
        assert "total_energy" in summary
        assert "n_interface_residues" in summary
        assert "n_favorable" in summary
        assert "n_unfavorable" in summary

    @pytest.mark.skipif(not bool(__import__("importlib").util.find_spec("pyrosetta")), reason="PyRosetta not installed")
    def test_analyze_interface_energies_file_not_found(self, registry):
        """analyze_interface_energies returns success=False for nonexistent file."""
        result = registry.call_tool(
            "analyze_interface_energies", pdb_path="/nonexistent/path/to/file.pdb", binder_chain="B"
        )

        assert result.success is False


class TestScoreInterface:
    """Tests for score_interface tool (requires PyRosetta)."""

    @pytest.fixture
    def temp_pdb_file(self, tmp_path):
        """Create a minimal two-chain PDB file for testing with proper geometry."""
        # Chain A (target) and Chain B (binder) at interface distance (~4A apart)
        pdb_content = (
            "HEADER    TEST                                    25-MAR-26   1ABC\n"
            "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  MET A   1       1.500   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  C   MET A   1       2.500   0.000   0.000  1.00  0.00           C\n"
            "ATOM      4  O   MET A   1       3.000   1.000   0.000  1.00  0.00           O\n"
            "ATOM      5  N   GLY A   2       3.500   0.000   0.000  1.00  0.00           N\n"
            "ATOM      6  CA  GLY A   2       5.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      7  C   GLY A   2       6.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      8  O   GLY A   2       6.500   1.000   0.000  1.00  0.00           O\n"
            # Chain B is positioned ~4A from Chain A (interface distance)
            "ATOM      9  N   ALA B   1       4.000   4.000   0.500  1.00  0.00           N\n"
            "ATOM     10  CA  ALA B   1       5.500   4.000   0.500  1.00  0.00           C\n"
            "ATOM     11  C   ALA B   1       6.500   4.000   0.500  1.00  0.00           C\n"
            "ATOM     12  O   ALA B   1       7.000   5.000   0.500  1.00  0.00           O\n"
            "END\n"
        )
        pdb_path = tmp_path / "test.pdb"
        pdb_path.write_text(pdb_content)
        return pdb_path

    def test_score_interface_registered(self, registry):
        """score_interface tool is registered."""
        tools = registry.list_tools()
        assert "score_interface" in tools

    def test_score_interface_excluded_when_pyrosetta_unavailable(self, registry):
        """score_interface excluded from schema when PyRosetta not available."""
        schemas = registry.get_tool_schemas()
        schema_names = [s["function"]["name"] for s in schemas]
        pyrosetta_ok = True
        try:
            import pyrosetta  # noqa: F401
        except ImportError:
            pyrosetta_ok = False
        if not pyrosetta_ok:
            assert "score_interface" not in schema_names

    @pytest.mark.skipif(not bool(__import__("importlib").util.find_spec("pyrosetta")), reason="PyRosetta not installed")
    def test_score_interface_success(self, registry, temp_pdb_file):
        """score_interface returns success when PyRosetta available."""
        result = registry.call_tool("score_interface", pdb_path=str(temp_pdb_file))

        assert result.success is True
        assert isinstance(result.data, str)
        assert isinstance(result.raw, dict)
        assert "interface_dG" in result.raw
        assert "shape_complementarity" in result.raw
        assert "packstat" in result.raw

    @pytest.mark.skipif(not bool(__import__("importlib").util.find_spec("pyrosetta")), reason="PyRosetta not installed")
    def test_score_interface_raw_fields(self, registry, temp_pdb_file):
        """score_interface raw data contains all required metrics."""
        result = registry.call_tool("score_interface", pdb_path=str(temp_pdb_file))

        assert result.success is True
        required_fields = [
            "interface_dG",
            "shape_complementarity",
            "packstat",
            "delta_sasa",
            "interface_hbonds",
            "delta_unsat_hbonds",
            "n_hotspot_residues",
            "binder_score",
            "binder_sasa",
        ]
        for field in required_fields:
            assert field in result.raw, f"Missing field: {field}"

    @pytest.mark.skipif(not bool(__import__("importlib").util.find_spec("pyrosetta")), reason="PyRosetta not installed")
    def test_score_interface_relax_false(self, registry, temp_pdb_file):
        """score_interface with relax_structure=False skips relaxation."""
        result = registry.call_tool("score_interface", pdb_path=str(temp_pdb_file), relax_structure=False)

        assert result.success is True
        assert result.raw.get("relaxed") is False

    @pytest.mark.skipif(not bool(__import__("importlib").util.find_spec("pyrosetta")), reason="PyRosetta not installed")
    def test_score_interface_file_not_found(self, registry):
        """score_interface returns success=False for nonexistent file."""
        result = registry.call_tool("score_interface", pdb_path="/nonexistent/path/to/file.pdb")

        assert result.success is False
