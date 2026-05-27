"""Tests for the local FastAPI batch API."""

from fastapi.testclient import TestClient

from structagent.api import server
from structagent.jobs.runner import JobRunner
from structagent.jobs.store import JobStore


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
