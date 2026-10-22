"""Exercise the web API end to end in offline mode."""

import importlib
import os
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(settings):
    os.environ["PP_DATA_DIR"] = str(settings.data_dir)
    os.environ["PP_PROVIDER"] = "offline"
    import policypilot.api.server as server

    importlib.reload(server)  # pick up the test settings
    return TestClient(server.app)


def test_health_and_listing(client):
    h = client.get("/api/health").json()
    assert h["mode"] == "offline" and h["clauses"] > 40
    regs = client.get("/api/regulations").json()
    assert len(regs) == 6
    assert regs[0]["requirement_list"][0] == {"id": "R1", "title": "Reporting of personal data breaches"}
    assert regs[0]["meta"]["effective"] == "1 January 2027"
    assert client.get("/").status_code == 200


def test_search_endpoint(client):
    hits = client.get("/api/search", params={"q": "card tokens", "k": 3}).json()
    assert hits[0]["clause"]["id"] == "info-security§3.2"
    assert client.get("/api/search", params={"q": "x", "method": "nope"}).status_code == 400


def test_analyze_then_decide(client):
    job_id = client.post("/api/analyze", json={"regulation_id": "reg-04-card-tokenisation"}).json()["job_id"]
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            break
        time.sleep(0.1)
    assert job["status"] == "done", job
    report = job["report"]
    assert report["findings"] and all(f["verified"] for f in report["findings"])
    assert any(e["kind"] == "verify" for e in job["events"])

    f = report["findings"][0]
    key = f"{f['requirement_id']}|{f['clause_id']}"
    res = client.post("/api/decisions", json={"run_id": job_id, "key": key, "verdict": "approved"}).json()
    assert res["decisions"][key] == "approved"


def test_pasted_text_and_bad_input(client):
    assert client.post("/api/analyze", json={}).status_code == 400
    job_id = client.post("/api/analyze", json={"text": "R1. Logs must be kept for 180 days at least."}).json()["job_id"]
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            break
        time.sleep(0.1)
    assert job["status"] == "done"


def test_upload_policy(client, settings):
    body = b"# Travel Policy\n\n## 1.1 Approvals\n\nAll travel requires approval from a line manager in advance.\n"
    res = client.post("/api/policies", files={"file": ("travel-policy.md", body, "text/markdown")}).json()
    assert res["clauses_added"] == 1
    (settings.policies_dir / "travel-policy.md").unlink()
    client.post("/api/policies", files={"file": ("bad.exe", b"x", "application/octet-stream")})


def test_bundled_example_ai_report_is_served(client):
    job = client.get("/api/jobs/example-ai").json()
    assert job["status"] == "done"
    assert job["report"]["mode"] == "groq" and job["report"]["findings"]
    assert all(f["verified"] for f in job["report"]["findings"])


def test_read_only_demo_refuses_uploads(client, monkeypatch):
    import policypilot.api.server as server

    monkeypatch.setattr(server.settings, "read_only", True)
    assert client.get("/api/health").json()["read_only"] is True
    res = client.post("/api/policies", files={"file": ("x.md", b"# X\n\n## 1.1 A\n\nBody text here.\n", "text/markdown")})
    assert res.status_code == 403
