import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from flask_cors import CORS
from pathlib import Path

import litellm

import llm_interactions
import configurator
from agents.orchestrator import Orchestrator
import agents.tools.ToolRegistry as tools_module
import agents.tools.memory_tool as memory_tools_module
from storage.longterm_memory import LongTermMemory

# --- LiteLLM debug output
litellm._turn_on_debug()

# --- Various global config:
ROOT_FOLDER = Path(__file__).parent
CONFIG_PATH = ROOT_FOLDER / "config.json"
ENV_PATH = ROOT_FOLDER / ".env"

# --- server config/init:
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# --- Configuration creation
cfg = configurator.AppConfig(CONFIG_PATH, ENV_PATH)

ltm = LongTermMemory(
    db_path=cfg.db_path,
    faiss_index_path=cfg.faiss_index_path
)
ltm.add_folder(str(ROOT_FOLDER))
tools_registry = tools_module.ToolRegistry()
orchestrator = Orchestrator(cfg, tools_registry, ltm=ltm)
tools_registry.register(memory_tools_module.SearchContext(ltm, ask_llm=orchestrator.safe_ask))


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
        llm_response = orchestrator.process_user_message(question)
    except Exception as e:
        return jsonify({"ok": False, "answer": str(e)}), 400

    answer = (
        f"[{cfg.model}] : {llm_response.choices[0].message['content']}\n"
        f"tokens: {llm_response.usage['prompt_tokens']}=>{llm_response.usage['completion_tokens']} (total: {llm_response.usage['total_tokens']})"
    )
    return jsonify({"ok": True, "answer": answer})

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 12000))
    # Bind to all interfaces so the provided URL can reach it
    app.run(host="0.0.0.0", port=port, debug=False)
