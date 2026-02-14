"""
Pytest configuration to mock heavy dependencies before imports.
"""

import sys
from unittest.mock import Mock, MagicMock

# Create mock modules before any imports that would load storage.longterm_memory
# This avoids needing sentence_transformers and faiss for these tests

def pytest_configure(config):
    """Called before any tests run - set up mocks here"""
    # Create mock for storage module
    mock_storage = Mock()
    mock_storage.longterm_memory = Mock()
    mock_storage.DatabaseManagement = Mock()
    mock_storage.EmbeddingManagement = Mock()
    
    # Add RESERVED_KEYWORD_CODE to the mock
    mock_storage.longterm_memory.RESERVED_KEYWORD_CODE = [
        "def", "class", "function", "var", "let", "const", "import", "from",
        "public", "private", "protected", "interface", "struct", "enum", "package",
        "return", "if", "else", "switch", "case", "for", "while", "do", "try",
        "catch", "finally", "throw", "new", "this", "super", "extends", "implements",
    ]
    
    sys.modules['storage'] = mock_storage
    sys.modules['storage.longterm_memory'] = mock_storage.longterm_memory
    sys.modules['storage.DatabaseManagement'] = mock_storage.DatabaseManagement
    sys.modules['storage.EmbeddingManagement'] = mock_storage.EmbeddingManagement
