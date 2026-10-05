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
    from opengeneralai.memory.embeddings import EmbeddingManager
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    pass

import copy

import pytest

from opengeneralai.llm.client import LLMResponse, Usage


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
    """Scripted LLMClient: returns the given replies in order and records the messages of each
    call, so that the agent loop runs without network, API key or randomness."""

    model = "fake/scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages):
        if not self.replies:
            raise AssertionError(f"FakeLLM: no scripted reply left for call #{len(self.calls) + 1}")
        self.calls.append(copy.deepcopy(messages))
        text = self.replies.pop(0)
        prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4
        return LLMResponse(text, Usage(prompt_tokens, len(text) // 4), duration_s=0.01)

    def cost(self, usage):
        return usage.prompt_tokens * 1e-6 + usage.completion_tokens * 2e-6


@pytest.fixture
def fake_llm():
    """fake_llm([reply1, reply2, ...]) returns a FakeLLM to give to the Orchestrator."""
    return FakeLLM
