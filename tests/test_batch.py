"""Tests for batch analysis mode."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from structagent.batch import ResultAggregator, BatchRunner, StructureResult, BatchResult
from structagent.agent import AgentRun, AgentStep


class TestResultAggregator:
    """Tests for ResultAggregator."""

    def test_aggregator_initialization(self):
        """Test aggregator initializes with criterion."""
        agg = ResultAggregator("stability")
        assert agg.criterion == "stability"
        assert len(agg.results) == 0

    def test_extracts_interface_energy_metrics(self):
        """Test extraction of interface energy metrics."""
        agg = ResultAggregator("interface_energy")

        # Create mock StructureResult
        mock_run = Mock(spec=AgentRun)
        mock_step = Mock(spec=AgentStep)
        mock_step.tool_name = "score_interface"
        mock_step.tool_result = Mock()
        mock_step.tool_result.success = True
        mock_step.tool_result.raw = {"dG": -15.3, "shape_complementarity": 0.72, "buried_sa": 1200.0, "packstat": 0.85}
        mock_run.steps = [mock_step]

        sr = StructureResult(pdb_id="1BRS", pdb_path="/path/to/1brs.pdb", run=mock_run, metrics={}, success=True)

        metrics = agg.extract_metrics(sr)
        assert metrics["interface_energy"] == -15.3
        assert metrics["shape_complementarity"] == 0.72

    def test_ranking_interface_energy(self):
        """Test ranking by interface energy (lower is better)."""
        agg = ResultAggregator("interface_energy")

        for pdb_id, dG in [("1BRS", -15.0), ("1BRC", -8.0), ("1BRD", -20.0)]:
            mock_run = Mock(spec=AgentRun)
            mock_step = Mock(spec=AgentStep)
            mock_step.tool_name = "score_interface"
            mock_step.tool_result = Mock()
            mock_step.tool_result.success = True
            mock_step.tool_result.raw = {"dG": dG}
            mock_run.steps = [mock_step]

            sr = StructureResult(pdb_id=pdb_id, pdb_path=None, run=mock_run, metrics={}, success=True)
            agg.add_result(sr)

        ranking = agg.get_ranking()
        # Lower interface energy = better = first in list
        assert ranking[0][0] == "1BRD"  # -20.0
        assert ranking[1][0] == "1BRS"  # -15.0
        assert ranking[2][0] == "1BRC"  # -8.0

    def test_ranking_buried_sa_higher_is_better(self):
        """Test ranking by buried SA (higher is better)."""
        agg = ResultAggregator("buried_surface_area")

        for pdb_id, buried_sa in [("1BRS", 1000.0), ("1BRC", 1500.0), ("1BRD", 800.0)]:
            mock_run = Mock(spec=AgentRun)
            mock_step = Mock(spec=AgentStep)
            mock_step.tool_name = "compute_interface"
            mock_step.tool_result = Mock()
            mock_step.tool_result.success = True
            mock_step.tool_result.raw = {"buried_sa_total": buried_sa}
            mock_run.steps = [mock_step]

            sr = StructureResult(pdb_id=pdb_id, pdb_path=None, run=mock_run, metrics={}, success=True)
            agg.add_result(sr)

        ranking = agg.get_ranking()
        # Higher buried SA = better = first in list
        assert ranking[0][0] == "1BRC"  # 1500.0
        assert ranking[1][0] == "1BRS"  # 1000.0
        assert ranking[2][0] == "1BRD"  # 800.0

    def test_handles_missing_metrics(self):
        """Test that aggregator handles structures with missing metrics."""
        agg = ResultAggregator("interface_energy")

        # Structure with no tool results
        mock_run = Mock(spec=AgentRun)
        mock_run.steps = []
        sr = StructureResult(pdb_id="TEST", pdb_path=None, run=mock_run, metrics={}, success=True)
        agg.add_result(sr)

        ranking = agg.get_ranking()
        # Should not crash, TEST should not appear in ranking (no metrics)
        assert len(ranking) == 0


class TestBatchRunner:
    """Tests for BatchRunner."""

    def test_discovers_pdbs_from_folder(self, tmp_path):
        """Test PDB discovery from folder."""
        # Create test PDB files
        (tmp_path / "1ubq.pdb").touch()
        (tmp_path / "2hhI.pdb").touch()
        (tmp_path / "notapdb.txt").touch()

        # Import and test
        from structagent.batch import BatchRunner

        mock_agent = Mock()
        runner = BatchRunner(mock_agent)

        pdbs = runner.discover_pdbs(str(tmp_path), "*.pdb")
        pdb_ids = [p[0] for p in pdbs]

        assert "1UBQ" in pdb_ids
        assert "2HHI" in pdb_ids
        assert len(pdb_ids) == 2  # Not .txt file

    def test_discovers_with_custom_glob(self, tmp_path):
        """Test PDB discovery with custom glob pattern."""
        (tmp_path / "test_1.pdb").touch()
        (tmp_path / "test_2.pdb").touch()
        (tmp_path / "other.pdb").touch()

        from structagent.batch import BatchRunner

        mock_agent = Mock()
        runner = BatchRunner(mock_agent)

        pdbs = runner.discover_pdbs(str(tmp_path), "test_*.pdb")
        pdb_ids = [p[0] for p in pdbs]

        assert len(pdb_ids) == 2
        assert "TEST_1" in pdb_ids
        assert "TEST_2" in pdb_ids


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_batch_result_creation(self):
        """Test BatchResult can be created."""
        from structagent.batch import BatchResult, StructureResult

        mock_run = Mock(spec=AgentRun)
        sr = StructureResult(
            pdb_id="1UBQ", pdb_path="/path/to/1ubq.pdb", run=mock_run, metrics={"interface_energy": -10.0}, success=True
        )

        result = BatchResult(
            query="analyze interface",
            structure_results=[sr],
            ranking=[("1UBQ", -10.0)],
            ranking_criterion="interface_energy",
            synthesis="Test synthesis",
            total_wall_time=5.0,
            total_tokens=1000,
        )

        assert result.query == "analyze interface"
        assert len(result.structure_results) == 1
        assert result.ranking_criterion == "interface_energy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
