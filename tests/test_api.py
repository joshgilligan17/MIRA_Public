"""Tests for the local FastAPI batch API."""

from fastapi.testclient import TestClient

from structagent.api import server
from structagent.jobs.runner import JobRunner
from structagent.jobs.store import JobStore
from structagent.projects import ProjectStore
from structagent.providers import ProviderResponse


def test_api_accepts_upload_and_serves_results(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs")
    runner = JobRunner(store)
    monkeypatch.setattr(server, "STORE", store)
    monkeypatch.setattr(server, "RUNNER", runner)
    client = TestClient(server.app)

    with open("tests/data/local/mini_complex.pdb", "rb") as handle:
        response = client.post(
            "/api/jobs",
            data={
                "query": "Rank these structures.",
                "profile": "triage_default",
                "rank_by": "stability",
                "enable_llm_synthesis": "false",
            },
            files={"files": ("mini_complex.pdb", handle, "chemical/x-pdb")},
        )

    assert response.status_code == 200
    job_id = response.json()["job_id"]

    status = client.get(f"/api/jobs/{job_id}")
    results = client.get(f"/api/jobs/{job_id}/results")
    structure = client.get(f"/api/jobs/{job_id}/structures/MINI_COMPLEX")

    assert status.status_code == 200
    assert status.json()["job"]["status"] == "completed"
    assert results.status_code == 200
    assert results.json()["ranking"][0]["pdb_id"] == "MINI_COMPLEX"
    assert structure.status_code == 200
    assert b"ATOM" in structure.content


def test_project_crud_target_upload_and_project_job(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs")
    projects = ProjectStore(tmp_path / "projects")
    runner = JobRunner(store)
    monkeypatch.setattr(server, "STORE", store)
    monkeypatch.setattr(server, "PROJECTS", projects)
    monkeypatch.setattr(server, "RUNNER", runner)
    client = TestClient(server.app)

    created = client.post("/api/projects", json={"name": "TREM2 screen", "description": "class project"})
    assert created.status_code == 200
    project_id = created.json()["project"]["id"]

    with open("tests/data/local/mini_complex.pdb", "rb") as handle:
        target = client.post(
            f"/api/projects/{project_id}/target",
            files={"file": ("target.pdb", handle, "chemical/x-pdb")},
        )

    assert target.status_code == 200
    assert target.json()["project"]["target_structure"]["structure_url"] == f"/api/projects/{project_id}/target"
    target_file = client.get(f"/api/projects/{project_id}/target")
    assert target_file.status_code == 200
    assert b"ATOM" in target_file.content

    with open("tests/data/local/mini_complex.pdb", "rb") as handle:
        job_response = client.post(
            f"/api/projects/{project_id}/jobs",
            data={
                "query": "Rank candidate binders.",
                "profile": "triage_default",
                "rank_by": "stability",
                "enable_llm_synthesis": "false",
            },
            files={"files": ("mini_complex.pdb", handle, "chemical/x-pdb")},
        )

    assert job_response.status_code == 200
    job_id = job_response.json()["job_id"]
    job = client.get(f"/api/jobs/{job_id}").json()["job"]
    project_jobs = client.get(f"/api/projects/{project_id}/jobs").json()["jobs"]
    project = client.get(f"/api/projects/{project_id}").json()["project"]

    assert job["project_id"] == project_id
    assert project_jobs[0]["id"] == job_id
    assert project["job_ids"] == [job_id]


def test_project_chat_uses_mocked_synthesis_provider(tmp_path, monkeypatch):
    class FakeProvider:
        def chat(self, messages, model, **kwargs):
            assert model == "fake-model"
            assert "Project context JSON" in messages[-1]["content"]
            return ProviderResponse(
                content=(
                    "<think>hidden</think>\n"
                    "The selected design has a referenced interface residue "
                    "[ALA-1](mira://region/interface_residues/A/1)."
                ),
                input_tokens=12,
                output_tokens=18,
            )

    def fake_create_provider(provider_name, api_key, base_url=None, timeout=120.0, temperature=0.0):
        assert provider_name == "openai"
        assert api_key == "test-key"
        return FakeProvider()

    store = JobStore(tmp_path / "jobs")
    projects = ProjectStore(tmp_path / "projects")
    runner = JobRunner(store)
    monkeypatch.setattr(server, "STORE", store)
    monkeypatch.setattr(server, "PROJECTS", projects)
    monkeypatch.setattr(server, "RUNNER", runner)
    monkeypatch.setattr(server, "create_provider", fake_create_provider)
    monkeypatch.setenv("MIRA_REPORT_PROVIDER", "openai")
    monkeypatch.setenv("MIRA_REPORT_MODEL", "fake-model")
    monkeypatch.setenv("MIRA_REPORT_API_KEY", "test-key")
    client = TestClient(server.app)

    project_id = client.post("/api/projects", json={"name": "Chat screen"}).json()["project"]["id"]
    with open("tests/data/local/mini_complex.pdb", "rb") as handle:
        job_response = client.post(
            f"/api/projects/{project_id}/jobs",
            data={
                "query": "Rank candidate binders.",
                "profile": "triage_default",
                "rank_by": "stability",
                "enable_llm_synthesis": "false",
            },
            files={"files": ("mini_complex.pdb", handle, "chemical/x-pdb")},
        )
    job_id = job_response.json()["job_id"]
    results = client.get(f"/api/jobs/{job_id}/results").json()
    structure_id = results["structures"][0]["id"]

    chat = client.post(
        f"/api/projects/{project_id}/chat",
        json={"message": "What stands out?", "selected_job_id": job_id, "selected_structure_id": structure_id},
    )

    assert chat.status_code == 200
    messages = chat.json()["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert "mira://region/interface_residues" in messages[-1]["content"]
    assert "<think>" not in messages[-1]["content"]
