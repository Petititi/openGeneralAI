import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from flask_cors import CORS
from pathlib import Path

import litellm

import llm_interactions
import configurator
from agents.orchestrator import Orchestrator
import agents.tools.ToolRegistry as tools_module

# --- LiteLLM debug output: it logs full requests, so it is opt-in (LITELLM_DEBUG=1)
if os.environ.get("LITELLM_DEBUG"):
    litellm._turn_on_debug()

# --- Various global config:
ROOT_FOLDER = Path(__file__).parent
CONFIG_PATH = ROOT_FOLDER / "config.json"
ENV_PATH = ROOT_FOLDER / ".env"
TEMPLATES_FOLDER = ROOT_FOLDER / "templates"

# --- Cross-origin access: the UI is served by this app, so no CORS is needed by default.
# Set ALLOWED_ORIGINS to a comma-separated list of origins to allow another front-end.
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

# --- server config/init:
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True
if ALLOWED_ORIGINS:
    CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# --- Configuration creation
cfg = configurator.AppConfig(CONFIG_PATH, ENV_PATH)

tools_registry = tools_module.ToolRegistry()
orchestrator = Orchestrator(cfg, tools_registry)


@app.after_request
def add_security_headers(resp):
    # Only pages from this server may embed the UI in an iframe
    resp.headers["X-Frame-Options"] = "SAMEORIGIN"
    resp.headers["Content-Security-Policy"] = "frame-ancestors 'self'"
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
    
    try:
        plan, cost = orchestrator.process_user_message(question)
    except Exception as e:
        return jsonify({"ok": False, "answer": str(e)}), 400

    return jsonify({"ok": True, "answer": format_answer(plan, cost)})


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


@app.get("/templates/<path:filename>")
def trace_viewer_files(filename):
    """Serve the trace viewer pages (templates/dbg*.html) and their JS/CSS for local debugging.

    Only the templates folder is exposed: never serve the project root, which holds .env.
    """
    return send_from_directory(TEMPLATES_FOLDER, filename)


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 12000))
    # Listen on localhost only; set HOST=0.0.0.0 to expose the server on your network
    host = os.environ.get("HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=False)
