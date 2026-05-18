"""Tests for prompts.py."""

import pytest
from structagent.prompts import (
    build_system_prompt,
    SYSTEM_PROMPT,
    EXAMPLE_QUERIES,
)


class TestBuildSystemPrompt:
    """Tests for build_system_prompt function."""

    def test_returns_string(self):
        """build_system_prompt returns a string."""
        result = build_system_prompt()
        assert isinstance(result, str)

    def test_contains_key_sections(self):
        """build_system_prompt returns string with key sections."""
        result = build_system_prompt()

        # Check for key sections from SYSTEM_PROMPT
        assert "MIRA" in result
        assert "expert structural biologist" in result
        assert "Your Expertise" in result
        assert "How You Work" in result
        assert "Orient" in result
        assert "Hypothesize" in result
        assert "Investigate" in result
        assert "Synthesize" in result
        assert "Qualify" in result
        assert "Guidelines" in result

    def test_no_tool_schemas_in_prompt(self):
        """System prompt should not contain tool schemas (they go in tools= parameter)."""
        result = build_system_prompt()

        # Tool schemas have "type": "function" and "parameters" - check for these patterns
        # The word "function" appears in "protein function" so we check for schema-like patterns
        assert '"type": "function"' not in result
        assert '"parameters"' not in result

    def test_without_context(self):
        """build_system_prompt without context returns SYSTEM_PROMPT only."""
        result = build_system_prompt()
        assert result == SYSTEM_PROMPT

    def test_with_context(self):
        """build_system_prompt appends context when provided."""
        context = "User is asking about ubiquitin mutant analysis."
        result = build_system_prompt(context=context)

        assert context in result
        assert result.startswith(SYSTEM_PROMPT)
        assert result.endswith(context)

    def test_context_appended_after_prompt(self):
        """Context is appended at the end after SYSTEM_PROMPT."""
        context = "Additional context: focus on hydrophobic core."
        result = build_system_prompt(context=context)

        # Context should appear at the end
        assert result.endswith(context)

    def test_empty_string_context(self):
        """build_system_prompt with empty string context returns SYSTEM_PROMPT."""
        result = build_system_prompt(context="")
        assert result == SYSTEM_PROMPT


class TestSystemPrompt:
    """Tests for SYSTEM_PROMPT constant."""

    def test_is_string(self):
        """SYSTEM_PROMPT is a string."""
        assert isinstance(SYSTEM_PROMPT, str)

    def test_is_not_empty(self):
        """SYSTEM_PROMPT is not empty."""
        assert len(SYSTEM_PROMPT) > 0

    def test_mentions_residue_citation(self):
        """SYSTEM_PROMPT mentions proper residue citation format."""
        assert "ARG-152 on chain A" in SYSTEM_PROMPT

    def test_mentions_tools_not_fabricated(self):
        """SYSTEM_PROMPT says to report distances from tool outputs."""
        assert "never fabricate numbers" in SYSTEM_PROMPT.lower() or "never fabricate" in SYSTEM_PROMPT.lower()

    def test_mentions_allosteric_tracing(self):
        """SYSTEM_PROMPT mentions tracing allosteric pathways via contact queries."""
        assert "contact queries" in SYSTEM_PROMPT.lower() or "sequential" in SYSTEM_PROMPT.lower()


class TestExampleQueries:
    """Tests for EXAMPLE_QUERIES constant."""

    def test_is_dict(self):
        """EXAMPLE_QUERIES is a dictionary."""
        assert isinstance(EXAMPLE_QUERIES, dict)

    def test_has_five_keys(self):
        """EXAMPLE_QUERIES has 5 example queries."""
        assert len(EXAMPLE_QUERIES) == 5

    def test_all_values_are_strings(self):
        """All EXAMPLE_QUERIES values are strings."""
        for key, value in EXAMPLE_QUERIES.items():
            assert isinstance(value, str), f"Value for {key} is not a string"

    def test_all_values_contain_placeholders(self):
        """All EXAMPLE_QUERIES values contain {placeholder} markers."""
        for key, value in EXAMPLE_QUERIES.items():
            assert "{" in value, f"Value for {key} lacks placeholders"

    def test_has_required_keys(self):
        """EXAMPLE_QUERIES has all required keys."""
        required_keys = {
            "allosteric_trace",
            "binding_interface",
            "active_site",
            "stability",
            "mutation_impact",
        }
        assert set(EXAMPLE_QUERIES.keys()) == required_keys

    def test_allosteric_trace_format(self):
        """allosteric_trace query mentions source_residue and pdb_id."""
        query = EXAMPLE_QUERIES["allosteric_trace"]
        assert "{source_residue}" in query
        assert "{pdb_id}" in query

    def test_binding_interface_format(self):
        """binding_interface query mentions chain_a, chain_b, and pdb_id."""
        query = EXAMPLE_QUERIES["binding_interface"]
        assert "{chain_a}" in query
        assert "{chain_b}" in query
        assert "{pdb_id}" in query

    def test_active_site_format(self):
        """active_site query mentions pdb_id."""
        query = EXAMPLE_QUERIES["active_site"]
        assert "{pdb_id}" in query

    def test_stability_format(self):
        """stability query mentions pdb_id."""
        query = EXAMPLE_QUERIES["stability"]
        assert "{pdb_id}" in query

    def test_mutation_impact_format(self):
        """mutation_impact query mentions residue and pdb_id."""
        query = EXAMPLE_QUERIES["mutation_impact"]
        assert "{residue}" in query
        assert "{pdb_id}" in query
