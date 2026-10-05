"""AppConfig: config.json and the API keys in .env."""

import json

from opengeneralai.config import AppConfig


def test_missing_config_file_is_created_with_defaults(tmp_path):
    cfg = AppConfig(tmp_path / "config.json", tmp_path / ".env")

    saved = json.loads((tmp_path / "config.json").read_text())
    assert cfg.provider and cfg.model and saved["model"] == cfg.model
    assert cfg.user_lang == "English"


def test_invalid_provider_is_replaced_but_other_settings_are_kept(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"provider": "nope", "model": "x", "user_lang": "French"}))

    cfg = AppConfig(tmp_path / "config.json", tmp_path / ".env")

    assert cfg.provider != "nope" and cfg.model != "x"
    assert cfg.user_lang == "French"


def test_unknown_model_falls_back_to_a_model_of_the_provider(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"provider": "mistral", "model": "mistral/not-a-model"}))

    cfg = AppConfig(tmp_path / "config.json", tmp_path / ".env")

    assert cfg.provider == "mistral" and cfg.model.startswith("mistral/") and cfg.model != "mistral/not-a-model"


def test_api_key_is_saved_then_removed(tmp_path, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    (tmp_path / "config.json").write_text(json.dumps({"provider": "mistral", "model": "mistral/mistral-small-latest"}))
    cfg = AppConfig(tmp_path / "config.json", tmp_path / ".env")
    assert cfg.need_configuration

    assert cfg.update_config("mistral", "mistral/mistral-small-latest", "sk-test")["ok"]
    assert "MISTRAL_API_KEY" in (tmp_path / ".env").read_text() and not cfg.need_configuration

    assert cfg.update_config("mistral", "mistral/mistral-small-latest", "")["ok"]
    assert cfg.need_configuration
