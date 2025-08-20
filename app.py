import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
from datetime import datetime
from pathlib import Path

import litellm
from litellm import AuthenticationError, RateLimitError, APIConnectionError, Timeout, BadRequestError

import llm_interactions
import configurator

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


def safe_ask(message: str) -> str:
    try:
        llm_response = litellm.completion(
            model=cfg.model,
            messages=message, timeout=30)
    except AuthenticationError:
        raise Exception("Authentification error : check your API_KEY.")
    except RateLimitError:
        raise Exception("Rate limit exceeded: try again later.")
    except Timeout:
        raise Exception("Can't get a response from the server.")
    except APIConnectionError:
        raise Exception("Network issue.")
    except BadRequestError as e:
        raise Exception(f"Invalid query : {e}")

    return llm_response

@app.post("/ask")
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or request.form.get("question") or "").strip()

    if not question:
        return jsonify({"ok": False, "answer": "Please ask a question."}), 400
    
    message = [{"role": "user", "content": question}]
    try:
        llm_response = safe_ask(message)
    except Exception as e:
        return jsonify({"ok": False, "answer": str(e)}), 400

    answer = (
        f"[{cfg.model}] : {llm_response.choices[0].message["content"]}\n"
        f"tokens: {llm_response.usage["prompt_tokens"]}=>{llm_response.usage["completion_tokens"]} (total: {llm_response.usage["total_tokens"]})"
    )
    return jsonify({"ok": True, "answer": answer})


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 12000))
    # Bind to all interfaces so the provided URL can reach it
    app.run(host="0.0.0.0", port=port, debug=False)
