"""
Pytest configuration for the project.
Now uses real implementations since LongTermMemory works without heavy dependencies.
"""

import sys
from pathlib import Path

# Ensure the project root is in the path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Check if embeddings are available
EMBEDDINGS_AVAILABLE = False
try:
    from storage.EmbeddingManagement import EmbeddingManager
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    pass

import copy
from types import SimpleNamespace

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-llm", action="store_true", default=False,
        help="also run the tests marked 'llm': they call the configured LLM for real (API key needed, costs money)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests that need what this run does not have: embeddings, or a real LLM (--run-llm)."""
    if not EMBEDDINGS_AVAILABLE:
        skip_embedding = pytest.mark.skip(reason="Embeddings not available (sentence_transformers not installed)")
        for item in items:
            if "semantic" in item.name.lower() or "embedding" in item.name.lower():
                item.add_marker(skip_embedding)
    if not config.getoption("--run-llm"):
        skip_llm = pytest.mark.skip(reason="calls a real LLM: run with --run-llm")
        for item in items:
            if "llm" in item.keywords:
                item.add_marker(skip_llm)


class FakeLLM:
    """Scripted stand-in for litellm.completion.

    Returns the given replies in order and records the messages of each call, so that
    the agent loop can be tested without network, API key or randomness.
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def completion(self, model, messages, **kwargs):
        if not self.replies:
            raise AssertionError(f"FakeLLM: no scripted reply left for call #{len(self.calls) + 1}")
        self.calls.append(copy.deepcopy(messages))
        content = self.replies.pop(0)
        prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4
        completion_tokens = len(content) // 4
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                 "total_tokens": prompt_tokens + completion_tokens}
        return SimpleNamespace(choices=[SimpleNamespace(message={"content": content})], usage=usage)


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the LLM used by the orchestrator: fake_llm([reply1, reply2, ...]) returns the FakeLLM."""
    def install(replies):
        fake = FakeLLM(replies)
        monkeypatch.setattr("agents.orchestrator.litellm.completion", fake.completion)
        monkeypatch.setattr("agents.orchestrator.cost_per_token",
                            lambda model, prompt_tokens, completion_tokens: (prompt_tokens * 1e-6, completion_tokens * 2e-6))
        return fake
    return install
