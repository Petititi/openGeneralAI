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

import pytest

# Skip tests that require embeddings if not available
def pytest_collection_modifyitems(config, items):
    """Skip tests that require embeddings if they're not available."""
    if not EMBEDDINGS_AVAILABLE:
        skip_embedding = pytest.mark.skip(reason="Embeddings not available (sentence_transformers not installed)")
        for item in items:
            if "semantic" in item.name.lower() or "embedding" in item.name.lower():
                item.add_marker(skip_embedding)
