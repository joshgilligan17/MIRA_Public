"""Tests for deterministic analysis profiles."""

from pathlib import Path

from structagent.profiles import list_analysis_profiles, run_analysis_profile


FIXTURE = Path("tests/data/local/mini_complex.pdb")


def test_lists_analysis_profiles():
    profiles = list_analysis_profiles()
    names = {profile["name"] for profile in profiles}
    assert "triage_default" in names
    assert "interface" in names


def test_run_analysis_profile_extracts_metrics_and_evidence():
    result = run_analysis_profile(
        pdb_id="MINI_COMPLEX",
        pdb_path=str(FIXTURE),
        query="Rank these structures for de novo filtering.",
        profile="triage_default",
    )

    structure = result.structure_result
    assert structure.success
    assert structure.metrics["mean_relative_sasa_percent"] > 0
    assert structure.metrics["buried_surface_area"] > 0
    assert structure.metrics["n_interface_residues"] > 0
    assert result.features["interface_residues"]
    assert result.features["buried_residues"]
    assert result.chains[0]["id"] == "A"
