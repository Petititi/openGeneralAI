import os
from flask import Flask, abort, render_template, request, jsonify, redirect, url_for, send_from_directory
from flask_cors import CORS
from pathlib import Path

import litellm

from opengeneralai.llm import catalog as llm_interactions
from opengeneralai.config import AppConfig
from opengeneralai.agent.context import MemoryContext
from opengeneralai.agent.orchestrator import Orchestrator
from opengeneralai.llm.client import LiteLLMClient
from opengeneralai.tools import registry as tools_module
from opengeneralai.tools import memory_search as memory_tools_module
from opengeneralai.memory.longterm_memory import LongTermMemory

# --- LiteLLM debug output: it logs full requests, so it is opt-in (LITELLM_DEBUG=1)
if os.environ.get("LITELLM_DEBUG"):
    litellm._turn_on_debug()

# --- Various global config:
# Root of the repository (src/opengeneralai/web/app.py -> repository root): config.json, .env,
# viewer/ and traces/ live there
ROOT_FOLDER = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT_FOLDER / "config.json"
ENV_PATH = ROOT_FOLDER / ".env"
VIEWER_FOLDER = ROOT_FOLDER / "viewer"
TRACES_FOLDER = ROOT_FOLDER / "traces"

# --- Cross-origin access: the UI is served by this app, so no CORS is needed by default.
# Set ALLOWED_ORIGINS to a comma-separated list of origins to allow another front-end.
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

# --- Content-Security-Policy: no inline script/style in the app pages (see static/)
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

# --- server config/init:
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True
if ALLOWED_ORIGINS:
    CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# --- Configuration creation
cfg = AppConfig(CONFIG_PATH, ENV_PATH)

ltm = LongTermMemory(
    db_path=cfg.db_path,
    faiss_index_path=cfg.faiss_index_path
)
ltm.add_folder(str(ROOT_FOLDER))
memory = MemoryContext(ltm)
tools_registry = tools_module.ToolRegistry()
tools_registry.register(memory_tools_module.SearchContext(ltm))


@app.after_request
def add_security_headers(resp):
    # Only pages from this server may embed the UI in an iframe
    resp.headers["X-Frame-Options"] = "SAMEORIGIN"
    # Scripts and styles come from /static and the Bootstrap CDN; routes may set a stricter or looser policy
    resp.headers.setdefault("Content-Security-Policy", APP_CSP)
    return resp


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
    providers, provider_to_models = llm_interactions.get_providers_and_models()
    _, has_key = cfg.get_env_info(providers)
    return jsonify(
        {
            "ok": True,
            "config": {"provider": cfg.provider, "model": cfg.model},
            "providers": providers,
            "models": provider_to_models,
            "has_key": has_key,
        }
    )


@app.post("/api/config")
def update_config():
    data = request.get_json(silent=True) or {}
    provider = (data.get("provider") or "").strip()
    model = (data.get("model") or "").strip()
    if "apiKey" in data:
        apiKey = (data.get("apiKey") or "").strip()
    else:
        apiKey = None

    result = cfg.update_config(provider, model, apiKey)
    if result["ok"]:
        return jsonify(result)
    else:
        return jsonify(result), 400


@app.post("/ask")
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or request.form.get("question") or "").strip()

    if not question:
        return jsonify({"ok": False, "answer": "Please ask a question."}), 400
    
    # Input validation and sanitization
    # 1. Length check
    MAX_QUESTION_LENGTH = 10000
    if len(question) > MAX_QUESTION_LENGTH:
        return jsonify({"ok": False, "answer": f"Question exceeds maximum length of {MAX_QUESTION_LENGTH} characters."}), 400
    
    # 2. Basic sanitization - remove potentially dangerous content
    # Note: The LLM itself handles most safety concerns, but we add a basic layer
    sanitized_question = question.strip()
    
    # 3. Check for empty or whitespace-only after sanitization
    if not sanitized_question or not sanitized_question.replace("\n", "").replace("\r", "").replace(" ", ""):
        return jsonify({"ok": False, "answer": "Question cannot be empty or whitespace only."}), 400
    
    try:
        # The model may change through /api/config: build the (stateless) orchestrator per request
        orchestrator = Orchestrator(LiteLLMClient(cfg.model), tools_registry, user_lang=cfg.user_lang, memory=memory)
        result = orchestrator.run(sanitized_question)
    except Exception as e:
        return jsonify({"ok": False, "answer": str(e)}), 400

    return jsonify({"ok": True, "answer": format_answer(result.plan, result.cost)})


def format_answer(plan: dict, cost: float) -> str:
    """Render the final plan returned by the orchestrator as plain text for the chat UI."""
    lines = [f"[{cfg.model}]"]
    steps = plan.get("plan_steps", []) if isinstance(plan, dict) else []
    for step in steps:
        if isinstance(step, dict):
            mark = "✅" if str(step.get("status", "")).lower() == "done" else "⬜"
            lines.append(f"{mark} {step.get('descr', '')}")
        else:
            lines.append(f"• {step}")
    if not steps:
        lines.append("No plan was produced.")
    lines.append(f"cost: ${cost:.4f}")
    return "\n".join(lines)


@app.get("/viewer/")
@app.get("/viewer/<path:filename>")
def trace_viewer(filename="index.html"):
    """Serve the trace viewer: /viewer/?trace=/traces/<name>.json (see viewer/README.md).

    Only the viewer and traces folders are exposed: never serve the project root, which holds .env.
    """
    resp = send_from_directory(VIEWER_FOLDER, filename)
    resp.headers["Content-Security-Policy"] = TRACE_VIEWER_CSP
    return resp


@app.get("/traces/<path:filename>")
def trace_files(filename):
    """Serve the traces written by the tests (JSON files only)."""
    if not filename.endswith(".json"):
        abort(404)
    return send_from_directory(TRACES_FOLDER, filename)


def main():
    port = int(os.environ.get("PORT", 12000))
    # Listen on localhost only; set HOST=0.0.0.0 to expose the server on your network
    host = os.environ.get("HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
