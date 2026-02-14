"""
Tests for the enhanced RAG (Retrieval-Augmented Generation) features in orchestrator.py

These tests verify:
1. LLM-based query analysis for determining if memory search is useful
2. Query reformulation for better semantic search
3. Comprehensive symbol research (classes, functions, methods)
"""

import sys
from pathlib import Path
import pytest
from unittest.mock import Mock, MagicMock, patch
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.orchestrator import Orchestrator
from storage.longterm_memory import LongTermMemory


# Sample test code for indexing
TEST_CODE = '''
class Calculator:
    """A simple calculator for testing"""
    
    def add(self, a, b):
        """Add two numbers"""
        return a + b
    
    def subtract(self, a, b):
        """Subtract two numbers"""
        return a - b

class DatabaseManager:
    """Manages database operations"""
    
    def __init__(self, db_path):
        self.db_path = db_path
    
    def get_class_code(self, class_name):
        """Get code for a specific class"""
        pass

def standalone_function():
    """A standalone function"""
    print("Hello World")
'''


@pytest.fixture
def mock_ltm():
    """Create a mocked LongTermMemory instance"""
    ltm = Mock()
    
    # Mock database lookups
    ltm.db = Mock()
    
    # Mock get_class_code
    ltm.db.get_class_code = Mock(return_value=[
        {
            "chunk_id": 1,
            "document_id": "doc1",
            "start_line": 1,
            "end_line": 20,
            "content": "class Calculator:\n    \"\"\"A simple calculator\"\"\"\n    pass",
            "class_name": "Calculator",
            "source_path": "/test/calculator.py",
            "language": "python"
        }
    ])
    
    # Mock get_function_code
    ltm.db.get_function_code = Mock(return_value=[
        {
            "chunk_id": 2,
            "document_id": "doc1",
            "start_line": 25,
            "end_line": 30,
            "content": "def standalone_function():\n    pass",
            "function_name": "standalone_function",
            "source_path": "/test/calculator.py",
            "language": "python"
        }
    ])
    
    # Mock get_method_code
    ltm.db.get_method_code = Mock(return_value=[
        {
            "chunk_id": 3,
            "document_id": "doc1",
            "start_line": 5,
            "end_line": 8,
            "content": "def add(self, a, b):\n    return a + b",
            "method_name": "add",
            "class_name": "Calculator",
            "source_path": "/test/calculator.py",
            "language": "python"
        }
    ])
    
    # Mock search methods
    ltm.search = Mock(return_value=[
        {
            "chunk_id": 10,
            "document_id": "doc1",
            "start_line": 1,
            "end_line": 20,
            "content": "class Calculator:\n    pass",
            "chunk_type": "class",
            "chunk_name": "Calculator",
            "source_path": "/test/calculator.py",
            "language": "python",
            "score": 0.95
        }
    ])
    
    return ltm


@pytest.fixture
def mock_config():
    """Create a mock configuration"""
    cfg = Mock()
    cfg.model = "gpt-4"
    cfg.user_lang = "en"
    return cfg


@pytest.fixture
def mock_tools():
    """Create a mock tool registry"""
    tools = Mock()
    tools.names = Mock(return_value=["file_editor", "generic_search"])
    return tools


@pytest.fixture
def orchestrator_with_mock_ltm(mock_config, mock_tools, mock_ltm):
    """Create an Orchestrator with mocked LTM"""
    with patch('agents.orchestrator.TrajectoryLogger'):
        orch = Orchestrator(
            cfg=mock_config,
            tools=mock_tools,
            ltm=mock_ltm,
            max_turn=3,
            max_context_tokens=2000
        )
    return orch


class TestQueryAnalysis:
    """Tests for LLM-based query analysis"""
    
    def test_analyze_query_for_memory_code_question(self, orchestrator_with_mock_ltm):
        """Test analysis of a code-related question"""
        # Mock the safe_ask to return a controlled response
        query = "How does the Calculator class work?"
        
        expected_analysis = {
            "should_search": True,
            "search_queries": ["Calculator class implementation", "how class works"],
            "search_type": "semantic",
            "symbols": ["Calculator"],
            "reasoning": "Query asks about code implementation"
        }
        
        # Create a mock LLM response
        llm_response = json.dumps(expected_analysis)
        
        with patch.object(orchestrator_with_mock_ltm, 'safe_ask', return_value=llm_response):
            result = orchestrator_with_mock_ltm._analyze_query_for_memory(query)
        
        assert result["should_search"] is True
        assert "Calculator" in result["symbols"]
        assert len(result["search_queries"]) > 0
    
    def test_analyze_query_for_memory_non_code_question(self, orchestrator_with_mock_ltm):
        """Test analysis of a non-code question (should not search)"""
        query = "What's the weather today?"
        
        expected_analysis = {
            "should_search": False,
            "search_queries": [],
            "search_type": "semantic",
            "symbols": [],
            "reasoning": "Not a code-related question"
        }
        
        llm_response = json.dumps(expected_analysis)
        
        with patch.object(orchestrator_with_mock_ltm, 'safe_ask', return_value=llm_response):
            result = orchestrator_with_mock_ltm._analyze_query_for_memory(query)
        
        assert result["should_search"] is False
        assert result["symbols"] == []
    
    def test_analyze_query_fallback_on_error(self, orchestrator_with_mock_ltm):
        """Test fallback behavior when LLM fails"""
        query = "Find the Orchestrator class"
        
        # Mock safe_ask to raise an exception
        with patch.object(orchestrator_with_mock_ltm, 'safe_ask', side_effect=Exception("LLM Error")):
            result = orchestrator_with_mock_ltm._analyze_query_for_memory(query)
        
        # Should fall back to local extraction
        assert result["should_search"] is True
        assert "Orchestrator" in result["symbols"] or len(result["symbols"]) > 0
        assert result["search_type"] == "hybrid"


class TestSymbolExtraction:
    """Tests for symbol extraction from queries"""
    
    def test_extract_camelcase_symbols(self, orchestrator_with_mock_ltm):
        """Test extraction of CamelCase identifiers"""
        query = "How does Orchestrator work with MemoryTool?"
        
        symbols = orchestrator_with_mock_ltm._extract_symbols_from_query(query)
        
        assert "Orchestrator" in symbols
        assert "MemoryTool" in symbols
    
    def test_extract_snake_case_symbols(self, orchestrator_with_mock_ltm):
        """Test extraction of snake_case identifiers"""
        query = "Find the get_class_code function"
        
        symbols = orchestrator_with_mock_ltm._extract_symbols_from_query(query)
        
        assert "get_class_code" in symbols
    
    def test_filter_reserved_keywords(self, orchestrator_with_mock_ltm):
        """Test that reserved keywords are filtered out"""
        query = "How to use class and function in Python"
        
        symbols = orchestrator_with_mock_ltm._extract_symbols_from_query(query)
        
        assert "class" not in symbols
        assert "function" not in symbols


class TestComprehensiveSymbolSearch:
    """Tests for comprehensive symbol search"""
    
    def test_search_exact_class_match(self, orchestrator_with_mock_ltm):
        """Test finding exact class match via database"""
        results = orchestrator_with_mock_ltm._search_symbol_comprehensive("Calculator")
        
        assert len(results) > 0
        # Should have called get_class_code
        orchestrator_with_mock_ltm.ltm.db.get_class_code.assert_called()
    
    def test_search_exact_function_match(self, orchestrator_with_mock_ltm):
        """Test finding exact function match via database"""
        results = orchestrator_with_mock_ltm._search_symbol_comprehensive("standalone_function")
        
        assert len(results) > 0
        # Should have called get_function_code
        orchestrator_with_mock_ltm.ltm.db.get_function_code.assert_called()
    
    def test_search_method_with_parent_class(self, orchestrator_with_mock_ltm):
        """Test finding method within a class"""
        results = orchestrator_with_mock_ltm._search_symbol_comprehensive("add")
        
        assert len(results) > 0
        # Should have called get_method_code
        orchestrator_with_mock_ltm.ltm.db.get_method_code.assert_called()
    
    def test_fallback_to_keyword_search(self, orchestrator_with_mock_ltm):
        """Test fallback to keyword search when DB returns few results"""
        # Reset mocks
        orchestrator_with_mock_ltm.ltm.db.get_class_code.return_value = []
        orchestrator_with_mock_ltm.ltm.db.get_function_code.return_value = []
        orchestrator_with_mock_ltm.ltm.db.get_method_code.return_value = []
        
        results = orchestrator_with_mock_ltm._search_symbol_comprehensive("UnknownSymbol")
        
        # Should fall back to search
        orchestrator_with_mock_ltm.ltm.search.assert_called()
    
    def test_filter_reserved_keywords_in_search(self, orchestrator_with_mock_ltm):
        """Test that reserved keywords don't trigger searches"""
        results = orchestrator_with_mock_ltm._search_symbol_comprehensive("class")
        
        # Should return empty list for reserved keywords
        assert results == []


class TestGetRelevantContext:
    """Tests for the main context retrieval method"""
    
    def test_get_relevant_context_with_valid_ltm(self, orchestrator_with_mock_ltm):
        """Test context retrieval when LTM is available"""
        query = "How does Calculator work?"
        
        expected_analysis = {
            "should_search": True,
            "search_queries": ["Calculator implementation"],
            "search_type": "semantic",
            "symbols": ["Calculator"]
        }
        
        llm_response = json.dumps(expected_analysis)
        
        with patch.object(orchestrator_with_mock_ltm, 'safe_ask', return_value=llm_response):
            context = orchestrator_with_mock_ltm._get_relevant_context(query)
        
        # Should return non-empty context
        assert context is not None
        assert len(context) > 0
    
    def test_get_relevant_context_no_search_needed(self, orchestrator_with_mock_ltm):
        """Test that context is empty when should_search is False"""
        query = "What's the capital of France?"
        
        expected_analysis = {
            "should_search": False,
            "search_queries": [],
            "search_type": "semantic",
            "symbols": []
        }
        
        llm_response = json.dumps(expected_analysis)
        
        with patch.object(orchestrator_with_mock_ltm, 'safe_ask', return_value=llm_response):
            context = orchestrator_with_mock_ltm._get_relevant_context(query)
        
        # Should return empty context
        assert context == ""
    
    def test_get_relevant_context_no_ltm(self, mock_config, mock_tools):
        """Test context retrieval when no LTM is available"""
        with patch('agents.orchestrator.TrajectoryLogger'):
            orch = Orchestrator(
                cfg=mock_config,
                tools=mock_tools,
                ltm=None  # No LTM
            )
        
        context = orch._get_relevant_context("Test query")
        
        assert context == ""
    
    def test_get_relevant_context_uses_symbols_first(self, orchestrator_with_mock_ltm):
        """Test that symbol search is prioritized"""
        query = "Show me the Calculator class and add method"
        
        expected_analysis = {
            "should_search": True,
            "search_queries": ["Calculator class implementation"],
            "search_type": "hybrid",
            "symbols": ["Calculator", "add"]
        }
        
        llm_response = json.dumps(expected_analysis)
        
        with patch.object(orchestrator_with_mock_ltm, 'safe_ask', return_value=llm_response):
            context = orchestrator_with_mock_ltm._get_relevant_context(query)
        
        # Symbol search should have been called first
        calls = orchestrator_with_mock_ltm.ltm.db.get_class_code.call_count
        assert calls > 0


class TestContextFormatting:
    """Tests for formatting search results"""
    
    def test_format_context_results_with_valid_results(self, orchestrator_with_mock_ltm):
        """Test formatting of search results"""
        results = [
            {
                "chunk_id": 1,
                "source_path": "/test/calculator.py",
                "start_line": 1,
                "end_line": 10,
                "chunk_type": "class",
                "chunk_name": "Calculator",
                "content": "class Calculator:\n    pass",
                "language": "python"
            },
            {
                "chunk_id": 2,
                "source_path": "/test/calculator.py",
                "start_line": 15,
                "end_line": 20,
                "chunk_type": "function",
                "chunk_name": "standalone_function",
                "content": "def standalone_function():\n    pass",
                "language": "python"
            }
        ]
        
        formatted = orchestrator_with_mock_ltm._format_context_results(results)
        
        assert "Calculator" in formatted
        assert "standalone_function" in formatted
        assert "/test/calculator.py" in formatted
    
    def test_format_context_filters_short_content(self, orchestrator_with_mock_ltm):
        """Test that very short content is filtered out"""
        results = [
            {
                "chunk_id": 1,
                "source_path": "/test/file.py",
                "start_line": 1,
                "end_line": 1,
                "chunk_type": "code",
                "content": "x",  # Too short
                "language": "python"
            }
        ]
        
        formatted = orchestrator_with_mock_ltm._format_context_results(results)
        
        # Short content should be skipped
        assert formatted == ""
    
    def test_format_context_truncates_long_content(self, orchestrator_with_mock_ltm):
        """Test that very long content is truncated"""
        long_content = "x" * 3000  # Very long content
        
        results = [
            {
                "chunk_id": 1,
                "source_path": "/test/file.py",
                "start_line": 1,
                "end_line": 100,
                "chunk_type": "code",
                "content": long_content,
                "language": "python"
            }
        ]
        
        formatted = orchestrator_with_mock_ltm._format_context_results(results)
        
        # Should be truncated
        assert "..." in formatted


class TestIntegration:
    """Integration tests for the full RAG pipeline"""
    
    def test_full_query_to_context_pipeline(self, orchestrator_with_mock_ltm):
        """Test the complete pipeline from query to context"""
        query = "How do I use the Calculator class add method?"
        
        # This tests the integration of all the new methods
        expected_analysis = {
            "should_search": True,
            "search_queries": ["Calculator add method usage", "class method implementation"],
            "search_type": "semantic",
            "symbols": ["Calculator", "add"]
        }
        
        llm_response = json.dumps(expected_analysis)
        
        with patch.object(orchestrator_with_mock_ltm, 'safe_ask', return_value=llm_response):
            context = orchestrator_with_mock_ltm._get_relevant_context(query)
        
        # Verify the pipeline worked
        assert context is not None
        # Symbols should have been searched
        orchestrator_with_mock_ltm.ltm.db.get_class_code.assert_called()
        orchestrator_with_mock_ltm.ltm.db.get_method_code.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
