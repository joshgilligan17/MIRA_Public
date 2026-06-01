"""Tests for the local FastAPI batch API."""

import sys

from fastapi.testclient import TestClient

from structagent.api import server
from structagent.jobs import runner as jobs_runner
from structagent.jobs.runner import JobRunner
from structagent.jobs.store import JobStore
from structagent import project_tools
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
    assert target.json()["project"]["selected_job_id"] is None
    assert target.json()["project"]["selected_structure_id"] == "target"
    target_file = client.get(f"/api/projects/{project_id}/target")
    assert target_file.status_code == 200
    assert b"ATOM" in target_file.content

    with open("tests/data/local/mini_complex.pdb", "rb") as handle:
        chat_structure = client.post(
            f"/api/projects/{project_id}/structures",
            files={"file": ("chat_structure.pdb", handle, "chemical/x-pdb")},
        )

    assert chat_structure.status_code == 200
    structure_id = chat_structure.json()["structure"]["id"]
    assert (
        chat_structure.json()["structure"]["structure_url"] == f"/api/projects/{project_id}/structures/{structure_id}"
    )
    assert chat_structure.json()["project"]["selected_job_id"] is None
    assert chat_structure.json()["project"]["selected_structure_id"] == structure_id
    structure_file = client.get(f"/api/projects/{project_id}/structures/{structure_id}")
    assert structure_file.status_code == 200
    assert b"ATOM" in structure_file.content

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

    cleared = client.patch(
        f"/api/projects/{project_id}",
        json={"selected_job_id": None, "selected_structure_id": structure_id},
    )
    assert cleared.status_code == 200
    assert cleared.json()["project"]["selected_job_id"] is None
    assert cleared.json()["project"]["selected_structure_id"] == structure_id

    deleted = client.delete(f"/api/projects/{project_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert not projects.project_dir(project_id).exists()
    assert client.get(f"/api/projects/{project_id}").status_code == 404
    assert all(item["id"] != project_id for item in client.get("/api/projects").json()["projects"])


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
    monkeypatch.setattr(jobs_runner, "create_provider", fake_create_provider)
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


def test_project_chat_can_use_uploaded_structure_without_batch(tmp_path, monkeypatch):
    class FakeProvider:
        def chat(self, messages, model, **kwargs):
            prompt = messages[-1]["content"]
            assert model == "fake-model"
            assert '"selected_job": null' in prompt
            assert '"pdb_id": "CHAT_STRUCTURE"' in prompt
            return ProviderResponse(
                content="I am looking at `CHAT_STRUCTURE` in the structure panel.",
                input_tokens=12,
                output_tokens=18,
            )

    def fake_create_provider(provider_name, api_key, base_url=None, timeout=120.0, temperature=0.0):
        return FakeProvider()

    projects = ProjectStore(tmp_path / "projects")
    monkeypatch.setattr(server, "PROJECTS", projects)
    monkeypatch.setattr(server, "create_provider", fake_create_provider)
    monkeypatch.setenv("MIRA_REPORT_PROVIDER", "openai")
    monkeypatch.setenv("MIRA_REPORT_MODEL", "fake-model")
    monkeypatch.setenv("MIRA_REPORT_API_KEY", "test-key")
    client = TestClient(server.app)

    project_id = client.post("/api/projects", json={"name": "Standalone chat"}).json()["project"]["id"]
    with open("tests/data/local/mini_complex.pdb", "rb") as handle:
        upload = client.post(
            f"/api/projects/{project_id}/structures",
            files={"file": ("chat_structure.pdb", handle, "chemical/x-pdb")},
        )
    structure_id = upload.json()["structure"]["id"]

    chat = client.post(
        f"/api/projects/{project_id}/chat",
        json={"message": "What is this?", "selected_job_id": None, "selected_structure_id": structure_id},
    )

    assert chat.status_code == 200
    assert chat.json()["messages"][-1]["content"] == "I am looking at `CHAT_STRUCTURE` in the structure panel."


def test_project_chat_can_pull_up_pdb_id_from_message(tmp_path, monkeypatch):
    class FakeProvider:
        def chat(self, messages, model, **kwargs):
            if "project tool router" in messages[0]["content"]:
                return ProviderResponse(
                    content='{"tool_calls":[{"tool":"load_pdb_id","args":{"pdb_id":"1UBQ"},"purpose":"Open requested structure"}]}',
                    input_tokens=12,
                    output_tokens=18,
                )
            prompt = messages[-1]["content"]
            assert '"pdb_id": "1UBQ"' in prompt
            assert '"selected_job": null' in prompt
            assert '"status": "loaded"' in prompt
            return ProviderResponse(
                content="`1UBQ` is now visible in the structure panel.",
                input_tokens=12,
                output_tokens=18,
            )

    def fake_create_provider(provider_name, api_key, base_url=None, timeout=120.0, temperature=0.0):
        return FakeProvider()

    def fake_download(pdb_id):
        assert pdb_id == "1UBQ"
        return b"data_1ubq"

    projects = ProjectStore(tmp_path / "projects")
    monkeypatch.setattr(server, "PROJECTS", projects)
    monkeypatch.setattr(server, "create_provider", fake_create_provider)
    monkeypatch.setattr(project_tools, "_download_rcsb_cif", fake_download)
    monkeypatch.setenv("MIRA_REPORT_PROVIDER", "openai")
    monkeypatch.setenv("MIRA_REPORT_MODEL", "fake-model")
    monkeypatch.setenv("MIRA_REPORT_API_KEY", "test-key")
    client = TestClient(server.app)

    project_id = client.post("/api/projects", json={"name": "Target chat"}).json()["project"]["id"]

    chat = client.post(
        f"/api/projects/{project_id}/chat",
        json={"message": "Pull up 1ubq and tell me what target we are looking at."},
    )

    assert chat.status_code == 200
    body = chat.json()
    structure = body["project"]["structures"][0]
    assert structure["pdb_id"] == "1UBQ"
    assert body["project"]["selected_job_id"] is None
    assert body["project"]["selected_structure_id"] == structure["id"]
    assert body["messages"][-1]["selected_structure_id"] == structure["id"]
    assert body["messages"][-1]["content"] == "`1UBQ` is now visible in the structure panel."
    structure_file = client.get(structure["structure_url"])
    assert structure_file.status_code == 200
    assert structure_file.content == b"data_1ubq"


def test_project_tool_router_keeps_fallback_action_coverage():
    planned = [{"tool": "load_pdb_id", "args": {"pdb_id": "1UBQ"}, "purpose": "Load requested PDB."}]
    fallback = project_tools.fallback_project_tool_calls("Pull up 1UBQ and analyze the selected structure.")

    merged = server._merge_tool_plan_with_fallback(planned, fallback)

    assert [call["tool"] for call in merged] == ["load_pdb_id", "analyze_structure"]


def test_project_tool_router_does_not_start_generation_for_results_question():
    assert project_tools.message_is_results_status_query("What are the ProteinMPNN design results?")
    assert project_tools.fallback_project_tool_calls("What are the ProteinMPNN design results?") == []

    planned = [
        {
            "tool": "generate_design_candidates",
            "args": {"library": "proteinmpnn"},
            "purpose": "Incorrect fresh generation.",
        }
    ]
    fallback = project_tools.fallback_project_tool_calls("What are the design results?")
    filtered = [
        call
        for call in planned
        if str(call.get("tool") or call.get("name") or "")
        not in {"generate_design_candidates", "start_batch_from_project"}
    ]

    assert server._merge_tool_plan_with_fallback(filtered, fallback) == []


def test_project_tool_router_falls_back_to_hotspot_analysis():
    calls = project_tools.fallback_project_tool_calls("Identify hotspots on chain A for binder design.")

    assert [call["tool"] for call in calls] == ["identify_hotspots"]
    assert calls[0]["args"]["chain_id"] == "A"


def test_project_tool_router_falls_back_to_foldingdiff_backbone_generation():
    calls = project_tools.fallback_project_tool_calls("Generate three de novo backbone structures around 55 residues.")

    assert calls[-1]["tool"] == "generate_design_candidates"
    assert calls[-1]["args"]["library"] == "foldingdiff"
    assert calls[-1]["args"]["num_designs"] == 3
    assert calls[-1]["args"]["length"] == 55


def test_clean_chat_response_handles_non_string_content():
    assert server._clean_chat_response(["Hotspot", "analysis"]) == "['Hotspot', 'analysis']"


def test_project_chat_identifies_hotspots_when_router_abstains(tmp_path, monkeypatch):
    class FakeProvider:
        def chat(self, messages, model, **kwargs):
            if "project tool router" in messages[0]["content"]:
                tool_names = {tool["function"]["name"] for tool in kwargs.get("tools", [])}
                assert "identify_hotspots" in tool_names
                return ProviderResponse(content='{"tool_calls":[]}', input_tokens=20, output_tokens=4)
            prompt = messages[-1]["content"]
            assert '"tool": "identify_hotspots"' in prompt
            assert '"hotspot_count"' in prompt
            assert "mira://region/hotspots" in prompt
            return ProviderResponse(
                content="Hotspot analysis completed with clickable residue evidence.",
                input_tokens=20,
                output_tokens=16,
            )

    def fake_create_provider(provider_name, api_key, base_url=None, timeout=120.0, temperature=0.0):
        return FakeProvider()

    projects = ProjectStore(tmp_path / "projects")
    monkeypatch.setattr(server, "PROJECTS", projects)
    monkeypatch.setattr(server, "create_provider", fake_create_provider)
    monkeypatch.setenv("MIRA_REPORT_PROVIDER", "openai")
    monkeypatch.setenv("MIRA_REPORT_MODEL", "fake-model")
    monkeypatch.setenv("MIRA_REPORT_API_KEY", "test-key")
    client = TestClient(server.app)

    project_id = client.post("/api/projects", json={"name": "Hotspot chat"}).json()["project"]["id"]
    with open("tests/data/local/mini_complex.pdb", "rb") as handle:
        upload = client.post(
            f"/api/projects/{project_id}/target",
            files={"file": ("target.pdb", handle, "chemical/x-pdb")},
        )
    assert upload.status_code == 200

    chat = client.post(
        f"/api/projects/{project_id}/chat",
        json={"message": "Identify hotspots on chain A for binder design."},
    )

    assert chat.status_code == 200
    body = chat.json()
    event = body["messages"][-1]["tool_events"][0]
    assert event["tool"] == "identify_hotspots"
    assert event["success"] is True
    assert event["raw"]["metrics"]["hotspot_count"] > 0
    assert body["project"]["analyses"][0]["kind"] == "hotspot_analysis"
    target = body["project"]["target_structure"]
    assert target["features"]["hotspots"]


def test_project_chat_exposes_registry_tools_for_selected_structure(tmp_path, monkeypatch):
    class FakeProvider:
        def chat(self, messages, model, **kwargs):
            if "project tool router" in messages[0]["content"]:
                tool_names = {tool["function"]["name"] for tool in kwargs.get("tools", [])}
                assert "analyze_bfactors" in tool_names
                assert "compute_normal_modes" in tool_names
                assert kwargs["tool_choice"] == "auto"
                return ProviderResponse(
                    content="",
                    input_tokens=30,
                    output_tokens=20,
                    tool_calls=[{"name": "analyze_bfactors", "arguments": '{"chain_id":"A"}'}],
                )
            prompt = messages[-1]["content"]
            assert '"registry_tool": "analyze_bfactors"' in prompt
            assert '"mean_bfactor"' in prompt
            return ProviderResponse(
                content="B-factor analysis completed from the registry tool.", input_tokens=20, output_tokens=18
            )

    def fake_create_provider(provider_name, api_key, base_url=None, timeout=120.0, temperature=0.0):
        return FakeProvider()

    projects = ProjectStore(tmp_path / "projects")
    monkeypatch.setattr(server, "PROJECTS", projects)
    monkeypatch.setattr(server, "create_provider", fake_create_provider)
    monkeypatch.setenv("MIRA_REPORT_PROVIDER", "openai")
    monkeypatch.setenv("MIRA_REPORT_MODEL", "fake-model")
    monkeypatch.setenv("MIRA_REPORT_API_KEY", "test-key")
    client = TestClient(server.app)

    project_id = client.post("/api/projects", json={"name": "Registry tools"}).json()["project"]["id"]
    with open("tests/data/local/mini_complex.pdb", "rb") as handle:
        upload = client.post(
            f"/api/projects/{project_id}/target",
            files={"file": ("target.pdb", handle, "chemical/x-pdb")},
        )
    assert upload.status_code == 200

    chat = client.post(
        f"/api/projects/{project_id}/chat",
        json={"message": "Analyze B-factors for chain A using the selected target."},
    )

    assert chat.status_code == 200
    event = chat.json()["messages"][-1]["tool_events"][0]
    assert event["tool"] == "analyze_bfactors"
    assert event["raw"]["registry_tool"] == "analyze_bfactors"
    assert event["raw"]["metrics"]["mean_bfactor"] > 0
    project = client.get(f"/api/projects/{project_id}").json()["project"]
    assert project["analyses"][0]["kind"] == "tool_analyze_bfactors"
    assert project["selected_structure_id"] == "target"


def test_project_chat_can_analyze_uploaded_structure(tmp_path, monkeypatch):
    class FakeProvider:
        def chat(self, messages, model, **kwargs):
            if "project tool router" in messages[0]["content"]:
                return ProviderResponse(
                    content=(
                        '{"tool_calls":[{"tool":"analyze_structure",'
                        '"args":{"analyses":["load_structure","bfactors","sasa","charge","ramachandran"],'
                        '"chain_id":"A"},"purpose":"Analyze selected target"}]}'
                    ),
                    input_tokens=20,
                    output_tokens=22,
                )
            prompt = messages[-1]["content"]
            assert '"analysis_id"' in prompt
            assert '"mean_bfactor"' in prompt
            return ProviderResponse(
                content="I analyzed the selected structure and saved target-analysis evidence.",
                input_tokens=20,
                output_tokens=22,
            )

    def fake_create_provider(provider_name, api_key, base_url=None, timeout=120.0, temperature=0.0):
        return FakeProvider()

    projects = ProjectStore(tmp_path / "projects")
    monkeypatch.setattr(server, "PROJECTS", projects)
    monkeypatch.setattr(server, "create_provider", fake_create_provider)
    monkeypatch.setenv("MIRA_REPORT_PROVIDER", "openai")
    monkeypatch.setenv("MIRA_REPORT_MODEL", "fake-model")
    monkeypatch.setenv("MIRA_REPORT_API_KEY", "test-key")
    client = TestClient(server.app)

    project_id = client.post("/api/projects", json={"name": "Analyze target"}).json()["project"]["id"]
    with open("tests/data/local/mini_complex.pdb", "rb") as handle:
        upload = client.post(
            f"/api/projects/{project_id}/structures",
            files={"file": ("mini_complex.pdb", handle, "chemical/x-pdb")},
        )
    structure_id = upload.json()["structure"]["id"]

    chat = client.post(
        f"/api/projects/{project_id}/chat",
        json={"message": "Analyze the selected target structure.", "selected_structure_id": structure_id},
    )

    assert chat.status_code == 200
    body = chat.json()
    assert body["messages"][-1]["tool_events"][0]["tool"] == "analyze_structure"
    assert body["project"]["analyses"][0]["kind"] == "structure_analysis"
    updated_structure = body["project"]["structures"][0]
    assert updated_structure["metrics"]["mean_bfactor"] is not None


def test_project_chat_can_start_batch_from_project_structures(tmp_path, monkeypatch):
    class FakeProvider:
        def chat(self, messages, model, **kwargs):
            if "project tool router" in messages[0]["content"]:
                return ProviderResponse(
                    content='{"tool_calls":[{"tool":"start_batch_from_project","args":{"rank_by":"stability"},"purpose":"Screen loaded candidates"}]}',
                    input_tokens=20,
                    output_tokens=22,
                )
            return ProviderResponse(content="Started the project batch screen.", input_tokens=20, output_tokens=22)

    def fake_create_provider(provider_name, api_key, base_url=None, timeout=120.0, temperature=0.0):
        return FakeProvider()

    store = JobStore(tmp_path / "jobs")
    projects = ProjectStore(tmp_path / "projects")
    runner = JobRunner(store)
    monkeypatch.setattr(server, "STORE", store)
    monkeypatch.setattr(server, "PROJECTS", projects)
    monkeypatch.setattr(server, "RUNNER", runner)
    monkeypatch.setattr(server, "create_provider", fake_create_provider)
    monkeypatch.setattr(jobs_runner, "create_provider", fake_create_provider)
    monkeypatch.setenv("MIRA_REPORT_PROVIDER", "openai")
    monkeypatch.setenv("MIRA_REPORT_MODEL", "fake-model")
    monkeypatch.setenv("MIRA_REPORT_API_KEY", "test-key")
    client = TestClient(server.app)

    project_id = client.post("/api/projects", json={"name": "Batch chat"}).json()["project"]["id"]
    for filename in ["candidate_a.pdb", "candidate_b.pdb"]:
        with open("tests/data/local/mini_complex.pdb", "rb") as handle:
            client.post(
                f"/api/projects/{project_id}/structures",
                files={"file": (filename, handle, "chemical/x-pdb")},
            )

    chat = client.post(
        f"/api/projects/{project_id}/chat",
        json={"message": "Run a batch screen over these candidate binders."},
    )

    assert chat.status_code == 200
    body = chat.json()
    event = body["messages"][-1]["tool_events"][0]
    assert event["tool"] == "start_batch_from_project"
    assert event["raw"]["structure_count"] == 2
    job_id = event["raw"]["job_id"]
    assert client.get(f"/api/jobs/{job_id}").json()["job"]["project_id"] == project_id


def test_project_chat_can_create_design_run_setup_record(tmp_path, monkeypatch):
    class FakeProvider:
        def chat(self, messages, model, **kwargs):
            if "project tool router" in messages[0]["content"]:
                return ProviderResponse(
                    content=(
                        '{"tool_calls":[{"tool":"generate_design_candidates",'
                        '"args":{"library":"bindcraft","num_designs":4,"design_prompt":"make compact binders"},'
                        '"purpose":"Start design library"}]}'
                    ),
                    input_tokens=20,
                    output_tokens=22,
                )
            return ProviderResponse(content="Created the design setup record.", input_tokens=20, output_tokens=22)

    def fake_create_provider(provider_name, api_key, base_url=None, timeout=120.0, temperature=0.0):
        return FakeProvider()

    projects = ProjectStore(tmp_path / "projects")
    monkeypatch.setattr(server, "PROJECTS", projects)
    monkeypatch.setattr(server, "create_provider", fake_create_provider)
    monkeypatch.delenv("MIRA_DESIGN_BINDCRAFT_COMMAND", raising=False)
    monkeypatch.delenv("MIRA_DESIGN_COMMAND", raising=False)
    monkeypatch.setenv("MIRA_REPORT_PROVIDER", "openai")
    monkeypatch.setenv("MIRA_REPORT_MODEL", "fake-model")
    monkeypatch.setenv("MIRA_REPORT_API_KEY", "test-key")
    client = TestClient(server.app)

    project_id = client.post("/api/projects", json={"name": "Design chat"}).json()["project"]["id"]
    with open("tests/data/local/mini_complex.pdb", "rb") as handle:
        upload = client.post(
            f"/api/projects/{project_id}/target",
            files={"file": ("target.pdb", handle, "chemical/x-pdb")},
        )
    assert upload.status_code == 200

    chat = client.post(
        f"/api/projects/{project_id}/chat",
        json={"message": "Design four candidate binders with BindCraft."},
    )

    assert chat.status_code == 200
    body = chat.json()
    event = body["messages"][-1]["tool_events"][0]
    assert event["tool"] == "generate_design_candidates"
    assert event["raw"]["status"] == "configuration_required"
    assert body["project"]["design_runs"][0]["library"] == "bindcraft"
    assert body["project"]["design_runs"][0]["status"] == "configuration_required"


def test_project_chat_runs_configured_foldingdiff_structure_generation(tmp_path, monkeypatch):
    fake_repo = tmp_path / "FoldingDiff"
    fake_bin = fake_repo / "bin"
    fake_bin.mkdir(parents=True)
    (fake_bin / "sample.py").write_text(
        """
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-l", nargs=2, type=int)
parser.add_argument("-n", type=int)
parser.add_argument("-b", type=int)
parser.add_argument("--device", default="cpu")
parser.add_argument("--outdir", default=".")
parser.add_argument("--nopsea", action="store_true")
args, _ = parser.parse_known_args()

out = Path(args.outdir) / "sampled_pdb"
out.mkdir(parents=True, exist_ok=True)
for index in range(args.n):
    with (out / f"generated_{index}.pdb").open("w") as handle:
        handle.write("ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 10.00           N\\n")
        handle.write("ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00 10.00           C\\n")
        handle.write("ATOM      3  C   GLY A   1       2.028   1.410   0.000  1.00 10.00           C\\n")
        handle.write("TER\\nEND\\n")
print(f"generated {args.n} FoldingDiff backbones at length {args.l[0]} on {args.device}")
""".strip()
    )

    class FakeProvider:
        def chat(self, messages, model, **kwargs):
            if "project tool router" in messages[0]["content"]:
                return ProviderResponse(
                    content=(
                        '{"tool_calls":[{"tool":"generate_design_candidates",'
                        '"args":{"library":"foldingdiff","num_designs":3,"length":55,'
                        '"design_prompt":"generate compact de novo backbones"},'
                        '"purpose":"Run local backbone generation"}]}'
                    ),
                    input_tokens=20,
                    output_tokens=22,
                )
            return ProviderResponse(
                content="Started real FoldingDiff backbone generation.", input_tokens=20, output_tokens=22
            )

    def fake_create_provider(provider_name, api_key, base_url=None, timeout=120.0, temperature=0.0):
        return FakeProvider()

    projects = ProjectStore(tmp_path / "projects")
    monkeypatch.setattr(server, "PROJECTS", projects)
    monkeypatch.setattr(server, "create_provider", fake_create_provider)
    monkeypatch.setenv("MIRA_FOLDINGDIFF_REPO", str(fake_repo))
    monkeypatch.setenv("MIRA_FOLDINGDIFF_PYTHON", sys.executable)
    monkeypatch.setenv("MIRA_FOLDINGDIFF_MAX_DESIGNS", "8")
    monkeypatch.delenv("MIRA_FOLDINGDIFF_COMMAND", raising=False)
    monkeypatch.delenv("MIRA_DESIGN_FOLDINGDIFF_COMMAND", raising=False)
    monkeypatch.delenv("MIRA_DESIGN_COMMAND", raising=False)
    monkeypatch.setenv("MIRA_REPORT_PROVIDER", "openai")
    monkeypatch.setenv("MIRA_REPORT_MODEL", "fake-model")
    monkeypatch.setenv("MIRA_REPORT_API_KEY", "test-key")
    client = TestClient(server.app)

    project_id = client.post("/api/projects", json={"name": "FoldingDiff chat"}).json()["project"]["id"]

    chat = client.post(
        f"/api/projects/{project_id}/chat",
        json={"message": "Generate three 55 residue backbone structures with FoldingDiff."},
    )

    assert chat.status_code == 200
    event = chat.json()["messages"][-1]["tool_events"][0]
    assert event["tool"] == "generate_design_candidates"
    assert event["raw"]["backend"] == "local"
    assert event["raw"]["library"] == "foldingdiff"
    assert event["raw"]["target_structure_id"] is None

    project = client.get(f"/api/projects/{project_id}").json()["project"]
    run = project["design_runs"][0]
    assert run["library"] == "foldingdiff"
    assert run["status"] == "completed"
    assert run["target_structure_id"] is None
    assert run["parameters"]["target_path"] is None
    assert run["parameters"]["length"] == 55
    assert run["parameters"]["device"] == "cpu"
    assert len(run["generated_structure_ids"]) == 3
    assert len(project["structures"]) == 3
    messages = client.get(f"/api/projects/{project_id}/chat").json()["messages"]
    assert any("Saved `3` generated structure file(s)" in message["content"] for message in messages)


def test_project_chat_runs_configured_proteinmpnn_sequence_design(tmp_path, monkeypatch):
    fake_repo = tmp_path / "ProteinMPNN"
    fake_repo.mkdir()
    (fake_repo / "protein_mpnn_run.py").write_text(
        """
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--pdb_path")
parser.add_argument("--out_folder")
parser.add_argument("--num_seq_per_target", type=int)
parser.add_argument("--sampling_temp")
parser.add_argument("--batch_size")
parser.add_argument("--seed")
parser.add_argument("--pdb_path_chains", default="")
args, _ = parser.parse_known_args()

seq_dir = Path(args.out_folder) / "seqs"
seq_dir.mkdir(parents=True, exist_ok=True)
stem = Path(args.pdb_path).stem
with (seq_dir / f"{stem}.fa").open("w") as handle:
    handle.write(f">{stem}, score=1.0, global_score=1.0\\n")
    handle.write("TARGETSEQ\\n")
    for index in range(args.num_seq_per_target):
        handle.write(f">design_{index}|temp={args.sampling_temp}|chains={args.pdb_path_chains}\\n")
        handle.write("ACDEFGHIKLMNPQRSTVWY\\n")
print(f"generated {args.num_seq_per_target} ProteinMPNN sequences")
""".strip()
    )

    class FakeProvider:
        def chat(self, messages, model, **kwargs):
            if "project tool router" in messages[0]["content"]:
                return ProviderResponse(
                    content=(
                        '{"tool_calls":[{"tool":"generate_design_candidates",'
                        '"args":{"library":"proteinmpnn","num_designs":5,"chain_id":"A",'
                        '"temperature":"0.2","design_prompt":"redesign target sequence"},'
                        '"purpose":"Run local sequence design"}]}'
                    ),
                    input_tokens=20,
                    output_tokens=22,
                )
            return ProviderResponse(
                content="Started real ProteinMPNN sequence design.", input_tokens=20, output_tokens=22
            )

    def fake_create_provider(provider_name, api_key, base_url=None, timeout=120.0, temperature=0.0):
        return FakeProvider()

    projects = ProjectStore(tmp_path / "projects")
    monkeypatch.setattr(server, "PROJECTS", projects)
    monkeypatch.setattr(server, "create_provider", fake_create_provider)
    monkeypatch.setenv("MIRA_PROTEINMPNN_REPO", str(fake_repo))
    monkeypatch.setenv("MIRA_PROTEINMPNN_PYTHON", sys.executable)
    monkeypatch.delenv("MIRA_DESIGN_COMMAND", raising=False)
    monkeypatch.setenv("MIRA_REPORT_PROVIDER", "openai")
    monkeypatch.setenv("MIRA_REPORT_MODEL", "fake-model")
    monkeypatch.setenv("MIRA_REPORT_API_KEY", "test-key")
    client = TestClient(server.app)

    project_id = client.post("/api/projects", json={"name": "ProteinMPNN chat"}).json()["project"]["id"]
    with open("tests/data/local/mini_complex.pdb", "rb") as handle:
        upload = client.post(
            f"/api/projects/{project_id}/target",
            files={"file": ("target.pdb", handle, "chemical/x-pdb")},
        )
    assert upload.status_code == 200

    chat = client.post(
        f"/api/projects/{project_id}/chat",
        json={"message": "Generate five ProteinMPNN sequences for chain A."},
    )

    assert chat.status_code == 200
    event = chat.json()["messages"][-1]["tool_events"][0]
    assert event["tool"] == "generate_design_candidates"
    assert event["raw"]["backend"] == "local"

    project = client.get(f"/api/projects/{project_id}").json()["project"]
    run = project["design_runs"][0]
    assert run["library"] == "proteinmpnn"
    assert run["status"] == "completed"
    assert run["parameters"]["model"] == "proteinmpnn"
    assert len(run["generated_sequences"]) == 5
    assert run["generated_sequences"][0]["sequence"] == "ACDEFGHIKLMNPQRSTVWY"
    messages = client.get(f"/api/projects/{project_id}/chat").json()["messages"]
    assert any("completed with `proteinmpnn`" in message["content"] for message in messages)


def test_project_chat_proteinmpnn_converts_cif_and_retries_without_chain(tmp_path, monkeypatch):
    fake_repo = tmp_path / "ProteinMPNN"
    fake_repo.mkdir()
    (fake_repo / "protein_mpnn_run.py").write_text(
        """
import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--pdb_path")
parser.add_argument("--out_folder")
parser.add_argument("--num_seq_per_target", type=int)
parser.add_argument("--sampling_temp")
parser.add_argument("--batch_size")
parser.add_argument("--seed")
parser.add_argument("--pdb_path_chains", default="")
args, _ = parser.parse_known_args()

if Path(args.pdb_path).suffix.lower() != ".pdb":
    print("ProteinMPNN parser expected a PDB input", file=sys.stderr)
    raise SystemExit(12)
if args.pdb_path_chains:
    print("requested chain could not be parsed", file=sys.stderr)
    raise SystemExit(13)

seq_dir = Path(args.out_folder) / "seqs"
seq_dir.mkdir(parents=True, exist_ok=True)
stem = Path(args.pdb_path).stem
with (seq_dir / f"{stem}.fa").open("w") as handle:
    handle.write(f">{stem}, score=1.0, global_score=1.0\\n")
    handle.write("TARGETSEQ\\n")
    for index in range(args.num_seq_per_target):
        handle.write(f">fallback_design_{index}|temp={args.sampling_temp}\\n")
        handle.write("YYYYYVVVVV\\n")
print(f"generated {args.num_seq_per_target} fallback ProteinMPNN sequences")
""".strip()
    )

    class FakeProvider:
        def chat(self, messages, model, **kwargs):
            if "project tool router" in messages[0]["content"]:
                return ProviderResponse(
                    content=(
                        '{"tool_calls":[{"tool":"generate_design_candidates",'
                        '"args":{"library":"proteinmpnn","num_designs":3,"chain_id":"chain A.",'
                        '"temperature":"0.1","design_prompt":"redesign chain A"},'
                        '"purpose":"Run local sequence design"}]}'
                    ),
                    input_tokens=20,
                    output_tokens=22,
                )
            return ProviderResponse(content="Started ProteinMPNN retry design.", input_tokens=20, output_tokens=22)

    def fake_create_provider(provider_name, api_key, base_url=None, timeout=120.0, temperature=0.0):
        return FakeProvider()

    from Bio.PDB import MMCIFIO, PDBParser

    cif_path = tmp_path / "target.cif"
    structure = PDBParser(QUIET=True).get_structure("mini", "tests/data/local/mini_complex.pdb")
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(str(cif_path))

    projects = ProjectStore(tmp_path / "projects")
    monkeypatch.setattr(server, "PROJECTS", projects)
    monkeypatch.setattr(server, "create_provider", fake_create_provider)
    monkeypatch.setenv("MIRA_PROTEINMPNN_REPO", str(fake_repo))
    monkeypatch.setenv("MIRA_PROTEINMPNN_PYTHON", sys.executable)
    monkeypatch.delenv("MIRA_PROTEINMPNN_DISABLE_FALLBACKS", raising=False)
    monkeypatch.setenv("MIRA_REPORT_PROVIDER", "openai")
    monkeypatch.setenv("MIRA_REPORT_MODEL", "fake-model")
    monkeypatch.setenv("MIRA_REPORT_API_KEY", "test-key")
    client = TestClient(server.app)

    project_id = client.post("/api/projects", json={"name": "ProteinMPNN CIF retry"}).json()["project"]["id"]
    with cif_path.open("rb") as handle:
        upload = client.post(
            f"/api/projects/{project_id}/target",
            files={"file": ("target.cif", handle, "chemical/x-cif")},
        )
    assert upload.status_code == 200

    chat = client.post(
        f"/api/projects/{project_id}/chat",
        json={"message": "Redesign chain A with ProteinMPNN."},
    )

    assert chat.status_code == 200
    project = client.get(f"/api/projects/{project_id}").json()["project"]
    run = project["design_runs"][0]
    assert run["status"] == "completed"
    assert run["parameters"]["target_conversion"] == "mmcif_to_pdb"
    assert run["parameters"]["target_path"].endswith("proteinmpnn_input.pdb")
    assert len(run["parameters"]["fallbacks"]) == 1
    assert "retry_without_chain_selection" in run["logs"]
    assert len(run["generated_sequences"]) == 3
    assert run["generated_sequences"][0]["sequence"] == "YYYYYVVVVV"
