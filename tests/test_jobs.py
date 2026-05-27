"""Tests for persisted batch jobs."""

from pathlib import Path
from shutil import copyfile

from structagent.jobs.models import JobConfig
from structagent.jobs.runner import JobRunner
from structagent.jobs.store import JobStore
from structagent.providers import ProviderResponse


FIXTURE = Path("tests/data/local/mini_complex.pdb")


def test_job_runner_persists_results_and_report(tmp_path):
    store = JobStore(tmp_path / "jobs")
    runner = JobRunner(store)
    record = store.create_job(JobConfig(query="Rank these structures.", max_workers=1, enable_llm_synthesis=False))
    copyfile(FIXTURE, store.input_dir(record.id) / "mini_complex.pdb")
    record.input_files = ["mini_complex.pdb"]
    store.write_record(record)

    runner.run_job(record.id)

    final_record = store.get_record(record.id)
    results = store.load_results(record.id)
    report = store.report_path(record.id).read_text()

    assert final_record.status == "completed"
    assert final_record.completed_count == 1
    assert results["summary"]["top_structure"] == "MINI_COMPLEX"
    assert results["structures"][0]["features"]["interface_residues"]
    assert "MIRA Batch Report" in report
    assert "mira://region/interface_residues" in report


def test_job_runner_uses_mocked_llm_synthesis(tmp_path, monkeypatch):
    class FakeProvider:
        def chat(self, messages, model, **kwargs):
            assert model == "fake-model"
            assert "Context JSON" in messages[-1]["content"]
            return ProviderResponse(
                content=(
                    "<think>internal draft that must not be saved</think>\n"
                    "## Synthesis\n\n"
                    "### Target Context\n\n"
                    "Mock LLM target context.\n\n"
                    "### Design and Filtering Strategy\n\n"
                    "Mock filtering strategy.\n\n"
                    "### Batch Outcome\n\n"
                    "Mock batch outcome.\n\n"
                    "### Attribute-Level Interpretation\n\n"
                    "- Mock attribute interpretation.\n\n"
                    "### Lead Candidate Rationale\n\n"
                    "Mock lead rationale."
                ),
                input_tokens=10,
                output_tokens=20,
            )

    def fake_create_provider(provider_name, api_key, base_url=None, timeout=120.0, temperature=0.0):
        assert provider_name == "openai"
        assert api_key == "test-key"
        return FakeProvider()

    monkeypatch.setattr("structagent.jobs.runner.create_provider", fake_create_provider)
    store = JobStore(tmp_path / "jobs")
    runner = JobRunner(store)
    record = store.create_job(
        JobConfig(
            query="Rank these structures.",
            max_workers=1,
            enable_llm_synthesis=True,
            llm_provider="openai",
            llm_model="fake-model",
        )
    )
    copyfile(FIXTURE, store.input_dir(record.id) / "mini_complex.pdb")
    record.input_files = ["mini_complex.pdb"]
    store.write_record(record)

    runner.run_job(record.id, llm_api_key="test-key")

    results = store.load_results(record.id)
    report = store.report_path(record.id).read_text()
    assert results["report_synthesis"]["mode"] == "llm"
    assert "Mock LLM target context" in report
    assert "<think>" not in report
    assert "internal draft" not in report
