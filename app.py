from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv, set_key
import litellm

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['TEMPLATES_AUTO_RELOAD'] = True

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

CONFIG_PATH = Path(__file__).parent / 'config.json'
ENV_PATH = Path(__file__).parent / '.env'

load_dotenv(ENV_PATH)

def get_providers_and_models():
    providers = []
    provider_to_models = {}
    mbp = litellm.models_by_provider
    if isinstance(mbp, dict) and mbp:
        provider_to_models = {str(k): sorted(set(v)) for k, v in mbp.items() if v}
    else:
        # Fallback: derive mapping from litellm.model_list using get_llm_provider
        ml = litellm.model_list
        tmp = {}
        for m in ml:
            try:
                _m, prov, *_ = litellm.get_llm_provider(m)
                if prov:
                    tmp.setdefault(str(prov), []).append(m)
            except Exception:
                continue
        provider_to_models = {k: sorted(set(v)) for k, v in tmp.items()}
    providers = sorted(provider_to_models.keys())
    return providers, provider_to_models


def env_var_for_provider(provider: str) -> str:
    mapping = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "cohere": "COHERE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "huggingface": "HUGGINGFACEHUB_API_TOKEN",
        "replicate": "REPLICATE_API_TOKEN",
        "together_ai": "TOGETHERAI_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY",
        "xai": "XAI_API_KEY",
        "ollama": "OLLAMA_API_KEY",
        "azure": "AZURE_API_KEY",
        "azure_ai": "AZURE_AI_API_KEY",
        "vertex_ai": "VERTEXAI_API_KEY",
        "bedrock": "AWS_SECRET_ACCESS_KEY",  # complex auth; placeholder
    }
    if provider in mapping:
        return mapping[provider]
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in provider).upper()
    return f"{sanitized}_API_KEY"


def get_env_info(providers: list[str]):
    env_var_by_provider = {p: env_var_for_provider(p) for p in providers}
    has_key_by_provider = {p: bool(os.getenv(env_var_by_provider[p])) for p in providers}
    return env_var_by_provider, has_key_by_provider


def save_api_key(provider: str, api_key: str):
    if not api_key:
        return None
    var = env_var_for_provider(provider)
    try:
        set_key(str(ENV_PATH), var, api_key)
        # Reload env for current process usage
        load_dotenv(ENV_PATH, override=True)
        return var
    except Exception:
        return None

# Determine a sensible default based on litellm data
_PROVIDERS, _PROVIDER_TO_MODELS = get_providers_and_models()
_default_provider = "mistral" if "mistral" in _PROVIDERS else (_PROVIDERS[0] if _PROVIDERS else "mistral")
_default_model_list = _PROVIDER_TO_MODELS.get(_default_provider, [])
_default_model = _default_model_list[0] if _default_model_list else "mistral/devstral-medium-2507"
DEFAULT_CONFIG = {"provider": _default_provider, "model": _default_model}

def load_config():
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        provider = (cfg.get('provider') or '').strip()
        model = (cfg.get('model') or '').strip()
        providers, provider_to_models = get_providers_and_models()
        if provider not in providers:
            return DEFAULT_CONFIG
        allowed = provider_to_models.get(provider, [])
        if model not in allowed:
            # default to first model of provider
            if allowed:
                return {"provider": provider, "model": allowed[0]}
            return DEFAULT_CONFIG
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
    providers, provider_to_models = get_providers_and_models()
    _, has_key = get_env_info(providers)
    return jsonify({
        "ok": True,
        "config": cfg,
        "providers": providers,
        "has_key": has_key,
        "models": provider_to_models,
    })


@app.post("/api/config")
def update_config():
    data = request.get_json(silent=True) or {}
    provider = (data.get("provider") or "").strip()
    model = (data.get("model") or "").strip()

    providers, provider_to_models = get_providers_and_models()
    if provider not in providers:
        return jsonify({"ok": False, "error": "Fournisseur invalide."}), 400
    allowed = provider_to_models.get(provider, [])
    if model not in allowed:
        return jsonify({"ok": False, "error": "Modèle invalide pour ce fournisseur."}), 400

    cfg = {"provider": provider, "model": model}
    save_config(cfg)
    if "apiKey" in data:
        apiKey = (data.get("apiKey") or "").strip()
        type_of_key = save_api_key(provider, apiKey)
        if type_of_key is None:
            return jsonify({"ok": False, "error": "Modèle OK, mais clé inconnue."}), 400
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