"""Access to the LLM.

The agent only knows the LLMClient interface: LiteLLMClient talks to the providers through
LiteLLM, and the tests use a scripted fake (test/conftest.py). Provider quirks and error
mapping live here, not in the agent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import litellm
import openai

from opengeneralai.errors import (
    LLMAuthError,
    LLMError,
    LLMBadRequestError,
    LLMRateLimitError,
    LLMUnavailableError,
)


@dataclass(frozen=True)
class Usage:
    """Tokens consumed by one or several LLM calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(self.prompt_tokens + other.prompt_tokens, self.completion_tokens + other.completion_tokens)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: Usage = field(default_factory=Usage)
    duration_s: float = 0.0


class LLMClient(Protocol):
    """What the agent needs from an LLM."""

    model: str

    def complete(self, messages: list[dict]) -> LLMResponse:
        """Answer the conversation; raise an LLMError subclass on failure."""
        ...

    def cost(self, usage: Usage) -> float:
        """Price in USD of the given usage (0 when unknown)."""
        ...


# Models that ignore the "system" role: their system messages are merged into the first message
MODELS_WITHOUT_SYSTEM_ROLE = ("mistral/devstral-small-250",)


def merge_system_messages(messages: list[dict]) -> list[dict]:
    """Rewrite a conversation for models that ignore the system role.

    The first message becomes a user message, later system messages are appended to it, and
    assistant messages become user messages.
    """
    merged: list[dict] = []
    for msg in messages:
        if not merged:
            merged.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "system":
            merged[0]["content"] += msg["content"]
        elif msg["role"] == "assistant":
            merged.append({"role": "user", "content": msg["content"]})
        else:
            merged.append(msg)
    return merged


class LiteLLMClient:
    """LLMClient for every provider supported by LiteLLM (model names like "mistral/...")."""

    def __init__(self, model: str, timeout: float = 30, num_retries: int = 0):
        self.model = model
        self.timeout = timeout
        self.num_retries = num_retries

    def complete(self, messages: list[dict]) -> LLMResponse:
        if self.model.startswith(MODELS_WITHOUT_SYSTEM_ROLE):
            messages = merge_system_messages(messages)
        start = time.time()
        try:
            response = litellm.completion(
                model=self.model, messages=messages, timeout=self.timeout, num_retries=self.num_retries
            )
        except (litellm.AuthenticationError, litellm.PermissionDeniedError) as e:
            raise LLMAuthError("Authentication error: check the API key of the provider.") from e
        except litellm.RateLimitError as e:
            raise LLMRateLimitError("Rate limit exceeded: try again later.") from e
        except litellm.Timeout as e:
            raise LLMUnavailableError("The LLM did not answer in time.") from e
        except litellm.APIConnectionError as e:
            raise LLMUnavailableError("Cannot reach the LLM provider (network issue).") from e
        except (litellm.InternalServerError, litellm.ServiceUnavailableError) as e:
            raise LLMUnavailableError("The LLM provider is unavailable (server error).") from e
        except (litellm.BadRequestError, litellm.NotFoundError) as e:
            raise LLMBadRequestError(f"Invalid request: {e}") from e
        except openai.APIError as e:  # base class of every LiteLLM API error
            raise LLMError(f"LLM provider error: {e}") from e
        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message["content"] or "",
            usage=Usage(usage["prompt_tokens"], usage["completion_tokens"]),
            duration_s=time.time() - start,
        )

    def cost(self, usage: Usage) -> float:
        try:
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=self.model, prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens
            )
        except Exception:  # unknown model price
            return 0.0
        return prompt_cost + completion_cost
