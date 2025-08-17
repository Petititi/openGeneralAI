from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import json
from datetime import datetime
from pathlib import Path

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ---- Configuration constants ----
VERIFIED_PROVIDERS = ['openhands', 'anthropic', 'openai', 'mistral']

VERIFIED_OPENAI_MODELS = [
    'gpt-5-2025-08-07',
    'o4-mini',
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4-32k',
    'gpt-4.1',
    'gpt-4.1-2025-04-14',
    'o1-mini',
    'o3',
    'codex-mini-latest',
]

VERIFIED_ANTHROPIC_MODELS = [
    'claude-sonnet-4-20250514',
    'claude-opus-4-20250514',
    'claude-opus-4-1-20250805',
    'claude-3-7-sonnet-20250219',
    'claude-3-sonnet-20240229',
    'claude-3-opus-20240229',
    'claude-3-haiku-20240307',
    'claude-3-5-haiku-20241022',
    'claude-3-5-sonnet-20241022',
    'claude-3-5-sonnet-20240620',
    'claude-2.1',
    'claude-2',
]

VERIFIED_MISTRAL_MODELS = [
    'devstral-small-2505',
    'devstral-small-2507',
    'devstral-medium-2507',
]

VERIFIED_OPENHANDS_MODELS = [
    'claude-sonnet-4-20250514',
    'gpt-5-2025-08-07',
    'claude-opus-4-20250514',
    'claude-opus-4-1-20250805',
    'devstral-small-2507',
    'devstral-medium-2507',
    'o3',
    'o4-mini',
    'gemini-2.5-pro',
    'kimi-k2-0711-preview',
    'qwen3-coder-480b',
]

PROVIDER_TO_MODELS = {
    'openai': VERIFIED_OPENAI_MODELS,
    'anthropic': VERIFIED_ANTHROPIC_MODELS,
    'mistral': VERIFIED_MISTRAL_MODELS,
    'openhands': VERIFIED_OPENHANDS_MODELS,
}

CONFIG_PATH = Path(__file__).parent / 'config.json'
DEFAULT_CONFIG = {"provider": "openhands", "model": "o4-mini"}


def load_config():
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        # basic validation and fallback
        provider = cfg.get('provider')
        model = cfg.get('model')
        if provider not in VERIFIED_PROVIDERS:
            return DEFAULT_CONFIG
        if model not in PROVIDER_TO_MODELS.get(provider, []):
            # default to first model of provider
            return {"provider": provider, "model": PROVIDER_TO_MODELS[provider][0]}
        return {"provider": provider, "model": model}
    except Exception:
        return DEFAULT_CONFIG


def save_config(cfg: dict):
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


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
    return render_template("index.html")


@app.get("/config")
def config_page():
    return render_template("config.html")


@app.get("/api/config")
def get_config():
    cfg = load_config()
    return jsonify({
        "ok": True,
        "config": cfg,
        "providers": VERIFIED_PROVIDERS,
        "models": PROVIDER_TO_MODELS,
    })


@app.post("/api/config")
def update_config():
    data = request.get_json(silent=True) or {}
    provider = (data.get("provider") or "").strip()
    model = (data.get("model") or "").strip()

    if provider not in VERIFIED_PROVIDERS:
        return jsonify({"ok": False, "error": "Fournisseur invalide."}), 400
    allowed = PROVIDER_TO_MODELS.get(provider, [])
    if model not in allowed:
        return jsonify({"ok": False, "error": "Modèle invalide pour ce fournisseur."}), 400

    cfg = {"provider": provider, "model": model}
    save_config(cfg)
    return jsonify({"ok": True, "config": cfg})


@app.post("/ask")
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or request.form.get("question") or "").strip()

    if not question:
        return jsonify({"ok": False, "answer": "Veuillez entrer une question."}), 400

    # Demo answer logic (replace with your LLM/agent later)
    cfg = load_config()
    answer = (
        f"[{cfg['provider']} · {cfg['model']}] "
        f"Réponse simulée ({datetime.utcnow().strftime('%H:%M:%S UTC')}): "
        f"{question[::-1]}"
    )
    return jsonify({"ok": True, "answer": answer})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 12000))
    # Bind to all interfaces so the provided URL can reach it
    app.run(host="0.0.0.0", port=port, debug=False)