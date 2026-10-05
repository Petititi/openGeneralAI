"""The web app, with a scripted LLM (no network, no API key)."""

import json
from pathlib import Path

import pytest

from opengeneralai.agent.orchestrator import Orchestrator
from opengeneralai.config import AppConfig
from opengeneralai.errors import LLMAuthError
from opengeneralai.tools.registry import ToolRegistry
from opengeneralai.web.app import create_app

REPO_ROOT = Path(__file__).parent.parent
PLAN_DONE = json.dumps({"plan_steps": [{"descr": "Say <b>hello</b>", "status": "done"}]})


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"provider": "mistral", "model": "mistral/mistral-small-latest"}))
    (tmp_path / ".env").write_text("MISTRAL_API_KEY=sk-secret\n")
    return AppConfig(tmp_path / "config.json", tmp_path / ".env")


def client(root, cfg, make_orchestrator):
    return create_app(root=root, cfg=cfg, make_orchestrator=make_orchestrator).test_client()


def test_ask_answers_with_the_plan(tmp_path, cfg, fake_llm):
    c = client(tmp_path, cfg, lambda cfg: Orchestrator(fake_llm([PLAN_DONE]), ToolRegistry()))

    r = c.post("/ask", json={"question": "hello?"})

    assert r.status_code == 200 and r.get_json()["ok"]
    answer = r.get_json()["answer"]
    assert "✅ Say <b>hello</b>" in answer and "cost: $" in answer


def test_llm_errors_are_reported_as_bad_gateway(tmp_path, cfg):
    class FailingLLM:
        model = "fake/failing"

        def complete(self, messages):
            raise LLMAuthError("Authentication error: check the API key of the provider.")

        def cost(self, usage):
            return 0.0

    c = client(tmp_path, cfg, lambda cfg: Orchestrator(FailingLLM(), ToolRegistry()))

    r = c.post("/ask", json={"question": "hello?"})

    assert r.status_code == 502
    assert r.get_json() == {"ok": False, "answer": "Authentication error: check the API key of the provider."}


def test_unexpected_errors_do_not_leak_details(tmp_path, cfg):
    def broken(cfg):
        raise RuntimeError("secret internal detail")

    r = client(tmp_path, cfg, broken).post("/ask", json={"question": "hello?"})

    assert r.status_code == 500
    assert "secret" not in r.get_data(as_text=True)


@pytest.mark.parametrize("question", ["", "   ", "x" * 10001])
def test_invalid_questions_are_rejected(tmp_path, cfg, question):
    r = client(tmp_path, cfg, lambda cfg: pytest.fail("no run expected")).post("/ask", json={"question": question})
    assert r.status_code == 400


def test_project_files_are_not_served(tmp_path, cfg):
    c = client(tmp_path, cfg, lambda cfg: None)
    for path in ["/.env", "/config.json", "/traces/../.env", "/traces/notes.txt"]:
        assert c.get(path).status_code == 404, path


def test_trace_viewer_and_traces_are_served(cfg):
    c = client(REPO_ROOT, cfg, lambda cfg: None)

    viewer = c.get("/viewer/")
    assert viewer.status_code == 200
    assert "script-src 'self' https://cdn.jsdelivr.net" in viewer.headers["Content-Security-Policy"]
    assert c.get("/traces/examples/2025-09-10_file_editing_tools.json").status_code == 200


def test_index_redirects_to_config_without_api_key(tmp_path, cfg, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    r = client(tmp_path, cfg, lambda cfg: None).get("/")
    assert r.status_code == 302 and r.headers["Location"].endswith("/config")
