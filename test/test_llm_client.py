"""LiteLLMClient: provider quirks, error mapping and usage, without calling any provider."""

from types import SimpleNamespace

import litellm
import pytest

from opengeneralai.errors import LLMAuthError, LLMBadRequestError, LLMUnavailableError
from opengeneralai.llm.client import LiteLLMClient, Usage, merge_system_messages


def test_merge_system_messages_for_models_without_system_role():
    messages = [
        {"role": "system", "content": "core. "},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "plan"},
        {"role": "system", "content": "act!"},
    ]
    assert merge_system_messages(messages) == [
        {"role": "user", "content": "core. act!"},
        {"role": "user", "content": "question"},
        {"role": "user", "content": "plan"},
    ]


def test_complete_returns_text_and_usage(monkeypatch):
    sent = {}

    def completion(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message={"content": "hello"})],
                               usage={"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15})

    monkeypatch.setattr(litellm, "completion", completion)
    response = LiteLLMClient("mistral/devstral-small-2505").complete(
        [{"role": "system", "content": "core"}, {"role": "user", "content": "hi"}])

    assert response.text == "hello"
    assert response.usage == Usage(12, 3)
    assert sent["messages"][0] == {"role": "user", "content": "core"}, "devstral: no system message sent"


@pytest.mark.parametrize("error, expected", [
    (litellm.AuthenticationError("bad key", llm_provider="mistral", model="m"), LLMAuthError),
    (litellm.Timeout("slow", model="m", llm_provider="mistral"), LLMUnavailableError),
    (litellm.APIConnectionError("down", llm_provider="mistral", model="m"), LLMUnavailableError),
    (litellm.InternalServerError("403 Forbidden", llm_provider="mistral", model="m"), LLMUnavailableError),
    (litellm.NotFoundError("no such model", llm_provider="mistral", model="m"), LLMBadRequestError),
])
def test_provider_errors_become_llm_errors(monkeypatch, error, expected):
    def completion(**kwargs):
        raise error

    monkeypatch.setattr(litellm, "completion", completion)
    with pytest.raises(expected):
        LiteLLMClient("mistral/mistral-small-latest").complete([{"role": "user", "content": "hi"}])


def test_cost_of_unknown_model_is_zero():
    assert LiteLLMClient("nobody/unknown-model").cost(Usage(1000, 1000)) == 0.0
