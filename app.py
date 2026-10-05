import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
from pathlib import Path

import litellm

import llm_interactions
import configurator
from agents.orchestrator import Orchestrator
import agents.tools.ToolRegistry as tools_module

# --- LiteLLM debug output
litellm._turn_on_debug()

# --- Various global config:
CONFIG_PATH = Path(__file__).parent / "config.json"
ENV_PATH = Path(__file__).parent / ".env"

# --- server config/init:
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# --- Configuration creation
cfg = configurator.AppConfig(CONFIG_PATH, ENV_PATH)

tools_registry = tools_module.ToolRegistry()
orchestrator = Orchestrator(cfg, tools_registry)


@app.after_request
def add_security_headers(resp):
    # Allow embedding in iframes and enable broad CORS for this demo app
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    # Allow embedding in iframes (note: consider tightening for production)
    resp.headers["X-Frame-Options"] = "ALLOWALL"
    resp.headers["Content-Security-Policy"] = "frame-ancestors *"
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
        plan = orchestrator.process_user_message(question)
    except Exception as e:
        return jsonify({"ok": False, "answer": str(e)}), 400

    return jsonify({"ok": True, "answer": format_answer(plan)})


def format_answer(plan: dict) -> str:
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
    return "\n".join(lines)


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 12000))
    # Bind to all interfaces so the provided URL can reach it
    app.run(host="0.0.0.0", port=port, debug=False)
