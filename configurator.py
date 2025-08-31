import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

from dotenv import load_dotenv
from dotenv import set_key as _set_key  # from python-dotenv
from dotenv import unset_key as _unset_key  # from python-dotenv

import llm_interactions


logger = logging.getLogger(__name__)


class AppConfig:
    """
    Helper to manage the server's LLM provider configuration and store/reload API keys
    in a .env file.

    Responsibilities:
    - Read/write a small JSON config that contains at least {"provider": ..., "model": ...}
    - Validate provider/model pairs against available providers provided by `llm_interactions`
    - Save/clear provider API keys in an env file using python-dotenv
    - Provide convenience getters and a `need_configuration` property

    Parameters
    ----------
    config_path:
        Path to the JSON config file. If missing, a default config will be created.
    env_path:
        Path to the .env file used to store API keys. The file will be loaded into
        the process environment on initialization.

    Notes
    -----
    - The class intentionally avoids throwing on recoverable errors and will fall back to
      sensible defaults. Critical/unexpected exceptions are logged.
    """

    def __init__(self, config_path: Path | str, env_path: Path | str) -> None:
        self.CONFIG_PATH: Path = Path(config_path)
        self.ENV_PATH: Path = Path(env_path)

        # Load environment variables (so os.getenv sees them)
        load_dotenv(str(self.ENV_PATH))

        # Private storage of the loaded configuration
        self.reload_config()

        logger.debug("AppConfig initialized with config %s and env %s", self._cfg, self.ENV_PATH)

    # -----------------------
    # Public API / accessors
    # -----------------------
    @property
    def user_lang(self) -> str:
        """Configured user language (or default user language if missing)."""
        return self._cfg.get("user_lang", "English")
    @property
    def provider(self) -> str:
        """Configured provider name (or default provider if missing)."""
        return self._cfg.get("provider", "")

    @property
    def model(self) -> str:
        """Configured model name (or default model if missing)."""
        return self._cfg.get("model", "")

    @property
    def need_configuration(self) -> bool:
        """
        Whether an API key is missing for the current provider.
        """
        provider = self.provider
        if not provider:
            return True
        
        return not bool(os.getenv(self.get_env_name_for(provider)))

    # -----------------------
    # Config loading & saving
    # -----------------------
    def default_config(self) -> Dict[str, str]:
        """
        Determine a sensible default provider/model using llm_interactions.
        If no providers/models are available, fall back to a hard-coded fallback.
        """
        providers, provider_to_models = llm_interactions.get_providers_and_models()
        default_provider = "mistral" if "mistral" in providers else (providers[0] if providers else "mistral")
        model_list = provider_to_models.get(default_provider, [])
        default_model = model_list[0] if model_list else "mistral/devstral-small-2505"
        return {"provider": default_provider, "model": default_model, "user_lang": "English"}

    def reload_config(self) -> Dict[str, str]:
        """
        Load the JSON config from disk and validate provider/model.
        Returns a validated config dictionary; on any problem returns a default config.
        """
        try:
            if not self.CONFIG_PATH.exists():
                logger.info("Config file %s not found — creating default config.", self.CONFIG_PATH)
                default = self.default_config()
                self.save_config(default)
                return default

            with self.CONFIG_PATH.open("r", encoding="utf-8") as f:
                self._cfg = json.load(f)

            provider = (self._cfg.get("provider") or "").strip()
            model = (self._cfg.get("model") or "").strip()

            providers, provider_to_models = llm_interactions.get_providers_and_models()

            if provider not in providers:
                logger.warning("Provider %r not in available providers %s — using defaults.", provider, providers)
                return self.default_config()

            allowed_models = provider_to_models.get(provider, [])
            if model not in allowed_models:
                if allowed_models:
                    logger.warning("Model %r is not allowed for provider %r — defaulting to first allowed model.",
                                   model, provider)
                    return {"provider": provider, "model": allowed_models[0]}
                logger.warning("No allowed models for provider %r — using global default config.", provider)
                return self.default_config()

            return {"provider": provider, "model": model}
        except Exception as exc:
            logger.exception("Failed to load/validate config (%s). Falling back to default.", exc)
            return self.default_config()

    def save_config(self) -> None:
        """
        Persist the config to disk as JSON.

        Writes atomically by writing to a temporary file in the same directory then replacing.
        """
        parent = self.CONFIG_PATH.parent
        parent.mkdir(parents=True, exist_ok=True)

        # Create a secure temporary file in the same directory to allow atomic replace
        with tempfile.NamedTemporaryFile("w", dir=str(parent), delete=False, encoding="utf-8") as tmpf:
            json.dump(self._cfg, tmpf, ensure_ascii=False, indent=2, sort_keys=True)
            tmp_name = Path(tmpf.name)

        try:
            # Replace the config file atomically
            os.replace(str(tmp_name), str(self.CONFIG_PATH))
            logger.debug("Config saved atomically to %s", self.CONFIG_PATH)
        except Exception:
            logger.exception("Failed to move temp config %s into place. Cleaning up.", tmp_name)
            try:
                tmp_name.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    # -----------------------
    # Configuration updates
    # -----------------------
    def update_config(self, provider: str, model: str, api_key: Optional[str]) -> Dict[str, object]:
        """
        Validate and persist the new configuration.

        Returns a dict in the form {"ok": bool, "config": {...}} on success or {"ok": False, "error": str}.
        """
        providers, provider_to_models = llm_interactions.get_providers_and_models()

        if provider not in providers:
            return {"ok": False, "error": "Invalid provider"}

        allowed_models = provider_to_models.get(provider, [])
        if model not in allowed_models:
            return {"ok": False, "error": "Invalid model for this provider."}

        self._cfg["provider"] = provider
        self._cfg["model"] = model
        try:
            self.save_config()
        except Exception as exc:
            logger.exception("Failed to save config: %s", exc)
            return {"ok": False, "error": "Failed to persist configuration."}

        if api_key is not None:
            saved_var = self.save_api_key(provider, api_key)
            if saved_var is None:
                return {"ok": False, "error": "Model OK, but can't store the API key."}
            if saved_var is None:
                return {"ok": False, "error": "API key removed."}

        return {"ok": True, "config": self._cfg}

    # -----------------------
    # Environment / API key helpers
    # -----------------------
    def get_env_info(self, providers: list[str]) -> Tuple[Dict[str, str], Dict[str, bool]]:
        """
        For a list of provider names, return:
          - provider_key: mapping provider -> expected env var name
          - has_key_by_provider: mapping provider -> whether an env value exists now

        Example:
            return ({'mistral': 'MISTRAL_API_KEY'}, {'mistral': True})
        """
        provider_key = {p: self.get_env_name_for(p) for p in providers}
        has_key_by_provider = {p: bool(os.getenv(provider_key[p])) for p in providers}
        return provider_key, has_key_by_provider

    def save_api_key(self, provider: str, api_key: str) -> Optional[str]:
        """
        Save (or unset) the API key for `provider` in the .env file.

        - If `api_key` is falsy (empty/None/""), the env var is removed from the file and None is returned.
        - On success returns the env var name that was written (e.g. "MISTRAL_API_KEY")
        - On failure returns None.
        """
        if not provider:
            logger.debug("save_api_key called with empty provider.")
            return None

        var_name = self.get_env_name_for(provider)
        try:
            if not api_key:
                # Remove the variable from the .env file if present.
                _unset_key(str(self.ENV_PATH), var_name)
                # Also remove from process env
                os.environ.pop(var_name, None)
                logger.info("Unset API key env var %s from %s", var_name, self.ENV_PATH)
                # We intentionally return an empty string to indicate "no key now"
                return ""

            # Set/overwrite the key in the .env file
            _set_key(str(self.ENV_PATH), var_name, api_key)
            # Reload into current process
            load_dotenv(str(self.ENV_PATH), override=True)
            logger.info("Saved API key into %s (env var: %s)", self.ENV_PATH, var_name)
            return var_name
        except Exception:
            logger.exception("Failed to set/unset API key %s in %s", var_name, self.ENV_PATH)
            return None
        
    def get_env_name_for(self, provider):
        prefix = provider.upper()
        suffix = "API_KEY"
        return f"{prefix}_{suffix}"