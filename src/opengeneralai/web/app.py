"""Web app: chat page, configuration page and trace viewer.

Nothing happens at import time: create_app() builds the app (and indexes the memory), main()
runs it. Tests give create_app() their own configuration and orchestrator.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional

import litellm
from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from opengeneralai.agent.context import MemoryContext
from opengeneralai.agent.orchestrator import Orchestrator, RunResult
from opengeneralai.agent.plan import StepStatus
from opengeneralai.config import AppConfig
from opengeneralai.errors import LLMError, OpenGeneralAIError
from opengeneralai.llm import catalog
from opengeneralai.llm.client import LiteLLMClient
from opengeneralai.memory.longterm_memory import LongTermMemory
from opengeneralai.tools.memory_search import SearchContext
from opengeneralai.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Root of the repository (src/opengeneralai/web/app.py -> repository root): config.json, .env,
# viewer/ and traces/ live there
ROOT_FOLDER = Path(__file__).resolve().parents[3]
MAX_QUESTION_LENGTH = 10000

# Content-Security-Policy: no inline script/style in the app pages (see static/)
CDN = "https://cdn.jsdelivr.net"
APP_CSP = (
    f"default-src 'self'; script-src 'self' {CDN}; style-src 'self' {CDN}; "
    f"font-src 'self' {CDN}; img-src 'self' data:; frame-ancestors 'self'"
)
# The trace viewer loads Cytoscape from the CDN; its JSON viewer injects <style> elements
TRACE_VIEWER_CSP = (
    f"default-src 'self'; script-src 'self' {CDN}; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; connect-src 'self'; frame-ancestors 'self'"
)

# Builds the orchestrator of one request from the current configuration (the model can change)
OrchestratorFactory = Callable[[AppConfig], Orchestrator]


def default_orchestrator_factory(root: Path, cfg: AppConfig) -> OrchestratorFactory:
    """Agent with the code of the repository in its memory and a tool to search it."""
    ltm = LongTermMemory(db_path=str(root / cfg.db_path), faiss_index_path=str(root / cfg.faiss_index_path))
    ltm.add_folder(str(root))
    memory = MemoryContext(ltm)
    tools = ToolRegistry()
    tools.register(SearchContext(ltm))

    def make(cfg: AppConfig) -> Orchestrator:
        return Orchestrator(LiteLLMClient(cfg.model), tools, user_lang=cfg.user_lang, memory=memory)

    return make


def format_answer(model: str, result: RunResult) -> str:
    """Render the result of a run as plain text for the chat UI."""
    lines = [f"[{model}]"]
    for step in result.plan.steps:
        lines.append(f"{'✅' if step.status == StepStatus.DONE else '⬜'} {step.descr}")
    if not result.plan.steps:
        lines.append("No plan was produced.")
    if result.error:
        lines.append(f"Stopped: {result.error}")
    elif not result.done:
        lines.append(f"Stopped after {result.turns} turns without finishing.")
    lines.append(f"cost: ${result.cost:.4f}")
    return "\n".join(lines)


def create_app(
    root: Path = ROOT_FOLDER,
    cfg: Optional[AppConfig] = None,
    make_orchestrator: Optional[OrchestratorFactory] = None,
) -> Flask:
    # LiteLLM debug output: it logs full requests, so it is opt-in (LITELLM_DEBUG=1)
    if os.environ.get("LITELLM_DEBUG"):
        litellm._turn_on_debug()

    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    # Cross-origin access: the UI is served by this app, so no CORS is needed by default.
    # Set ALLOWED_ORIGINS to a comma-separated list of origins to allow another front-end.
    allowed_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    if allowed_origins:
        CORS(app, resources={r"/*": {"origins": allowed_origins}})

    cfg = cfg or AppConfig(root / "config.json", root / ".env")
    make_orchestrator = make_orchestrator or default_orchestrator_factory(root, cfg)
    viewer_folder, traces_folder = root / "viewer", root / "traces"

    @app.after_request
    def add_security_headers(resp):
        # Only pages from this server may embed the UI in an iframe
        resp.headers["X-Frame-Options"] = "SAMEORIGIN"
        # Scripts and styles come from /static and the Bootstrap CDN; routes may set another policy
        resp.headers.setdefault("Content-Security-Policy", APP_CSP)
        return resp

    @app.errorhandler(OpenGeneralAIError)
    def user_error(e: OpenGeneralAIError):
        # The LLM provider failed (502), or the request cannot be served as is (400)
        status = 502 if isinstance(e, LLMError) else 400
        return jsonify({"ok": False, "answer": str(e)}), status

    @app.errorhandler(Exception)
    def internal_error(e: Exception):
        if isinstance(e, HTTPException):
            return e
        logger.exception("Unexpected error on %s", request.path)
        return jsonify({"ok": False, "answer": "Internal error: see the server log."}), 500

    @app.route("/")
    def index():
        if cfg.need_configuration:
            return redirect(url_for("config_page"))
        return render_template("index.html")

    @app.get("/config")
    def config_page():
        return render_template("config.html")

    @app.get("/api/config")
    def get_config():
        providers, provider_to_models = catalog.get_providers_and_models()
        _, has_key = cfg.get_env_info(providers)
        return jsonify({
            "ok": True,
            "config": {"provider": cfg.provider, "model": cfg.model},
            "providers": providers,
            "models": provider_to_models,
            "has_key": has_key,
        })

    @app.post("/api/config")
    def update_config():
        data = request.get_json(silent=True) or {}
        provider = (data.get("provider") or "").strip()
        model = (data.get("model") or "").strip()
        api_key = (data.get("apiKey") or "").strip() if "apiKey" in data else None
        result = cfg.update_config(provider, model, api_key)
        return jsonify(result), (200 if result["ok"] else 400)

    @app.post("/ask")
    def ask():
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or request.form.get("question") or "").strip()
        if not question:
            return jsonify({"ok": False, "answer": "Please ask a question."}), 400
        if len(question) > MAX_QUESTION_LENGTH:
            return jsonify({"ok": False,
                            "answer": f"Question exceeds maximum length of {MAX_QUESTION_LENGTH} characters."}), 400

        result = make_orchestrator(cfg).run(question)
        return jsonify({"ok": True, "answer": format_answer(cfg.model, result)})

    @app.get("/viewer/")
    @app.get("/viewer/<path:filename>")
    def trace_viewer(filename="index.html"):
        """Serve the trace viewer: /viewer/?trace=/traces/<name>.json (see viewer/README.md).

        Only the viewer and traces folders are exposed: never serve the project root, which holds .env.
        """
        resp = send_from_directory(viewer_folder, filename)
        resp.headers["Content-Security-Policy"] = TRACE_VIEWER_CSP
        return resp

    @app.get("/traces/<path:filename>")
    def trace_files(filename):
        """Serve the traces written by the tests (JSON files only)."""
        if not filename.endswith(".json"):
            abort(404)
        return send_from_directory(traces_folder, filename)

    return app


def main():
    port = int(os.environ.get("PORT", 12000))
    # Listen on localhost only; set HOST=0.0.0.0 to expose the server on your network
    host = os.environ.get("HOST", "127.0.0.1")
    create_app().run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
