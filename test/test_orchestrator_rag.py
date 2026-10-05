"""
Tests for the code context retrieved from the long-term memory (agent/context.py)

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

from opengeneralai.agent.context import MemoryContext
from opengeneralai.errors import LLMUnavailableError
from opengeneralai.memory.longterm_memory import LongTermMemory


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
def memory_context(mock_ltm):
    """MemoryContext over the mocked LTM"""
    return MemoryContext(mock_ltm, max_context_tokens=2000)


def failing_ask(messages):
    raise LLMUnavailableError("LLM Error")


class TestQueryAnalysis:
    """Tests for LLM-based query analysis"""
    
    def test_analyze_query_for_memory_code_question(self, memory_context):
        """Test analysis of a code-related question"""
        # The LLM answer is given by the ask callable
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
        
        result = memory_context.analyze_query(query, ask=lambda messages: llm_response)
        
        assert result["should_search"] is True
        assert "Calculator" in result["symbols"]
        assert len(result["search_queries"]) > 0
    
    def test_analyze_query_for_memory_non_code_question(self, memory_context):
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
        
        result = memory_context.analyze_query(query, ask=lambda messages: llm_response)
        
        assert result["should_search"] is False
        assert result["symbols"] == []
    
    def test_analyze_query_fallback_on_error(self, memory_context):
        """Test fallback behavior when LLM fails"""
        query = "Find the Orchestrator class"
        
        # The LLM call fails
        result = memory_context.analyze_query(query, ask=failing_ask)
        
        # Should fall back to local extraction
        assert result["should_search"] is True
        assert "Orchestrator" in result["symbols"] or len(result["symbols"]) > 0
        assert result["search_type"] == "hybrid"


class TestSymbolExtraction:
    """Tests for symbol extraction from queries"""
    
    def test_extract_camelcase_symbols(self, memory_context):
        """Test extraction of CamelCase identifiers"""
        query = "How does Orchestrator work with MemoryTool?"
        
        symbols = memory_context.extract_symbols(query)
        
        assert "Orchestrator" in symbols
        assert "MemoryTool" in symbols
    
    def test_extract_snake_case_symbols(self, memory_context):
        """Test extraction of snake_case identifiers"""
        query = "Find the get_class_code function"
        
        symbols = memory_context.extract_symbols(query)
        
        assert "get_class_code" in symbols
    
    def test_filter_reserved_keywords(self, memory_context):
        """Test that reserved keywords are filtered out"""
        query = "How to use class and function in Python"
        
        symbols = memory_context.extract_symbols(query)
        
        assert "class" not in symbols
        assert "function" not in symbols


class TestComprehensiveSymbolSearch:
    """Tests for comprehensive symbol search"""
    
    def test_search_exact_class_match(self, memory_context):
        """Test finding exact class match via database"""
        results = memory_context.search_symbol("Calculator")
        
        assert len(results) > 0
        # Should have called get_class_code
        memory_context.ltm.db.get_class_code.assert_called()
    
    def test_search_exact_function_match(self, memory_context):
        """Test finding exact function match via database"""
        results = memory_context.search_symbol("standalone_function")
        
        assert len(results) > 0
        # Should have called get_function_code
        memory_context.ltm.db.get_function_code.assert_called()
    
    def test_search_method_with_parent_class(self, memory_context):
        """Test finding method within a class"""
        results = memory_context.search_symbol("add")
        
        assert len(results) > 0
        # Should have called get_method_code
        memory_context.ltm.db.get_method_code.assert_called()
    
    def test_fallback_to_keyword_search(self, memory_context):
        """Test fallback to keyword search when DB returns few results"""
        # Reset mocks
        memory_context.ltm.db.get_class_code.return_value = []
        memory_context.ltm.db.get_function_code.return_value = []
        memory_context.ltm.db.get_method_code.return_value = []
        
        results = memory_context.search_symbol("UnknownSymbol")
        
        # Should fall back to search
        memory_context.ltm.search.assert_called()
    
    def test_filter_reserved_keywords_in_search(self, memory_context):
        """Test that reserved keywords don't trigger searches"""
        results = memory_context.search_symbol("class")
        
        # Should return empty list for reserved keywords
        assert results == []


class TestGetRelevantContext:
    """Tests for the main context retrieval method"""
    
    def test_get_relevant_context_with_valid_ltm(self, memory_context):
        """Test context retrieval when LTM is available"""
        query = "How does Calculator work?"
        
        expected_analysis = {
            "should_search": True,
            "search_queries": ["Calculator implementation"],
            "search_type": "semantic",
            "symbols": ["Calculator"]
        }
        
        llm_response = json.dumps(expected_analysis)
        
        context = memory_context.retrieve(query, ask=lambda messages: llm_response)
        
        # Should return non-empty context
        assert context is not None
        assert len(context) > 0
    
    def test_get_relevant_context_no_search_needed(self, memory_context):
        """Test that context is empty when should_search is False"""
        query = "What's the capital of France?"
        
        expected_analysis = {
            "should_search": False,
            "search_queries": [],
            "search_type": "semantic",
            "symbols": []
        }
        
        llm_response = json.dumps(expected_analysis)
        
        context = memory_context.retrieve(query, ask=lambda messages: llm_response)
        
        # Should return empty context
        assert context == ""
    
    def test_get_relevant_context_uses_symbols_first(self, memory_context):
        """Test that symbol search is prioritized"""
        query = "Show me the Calculator class and add method"
        
        expected_analysis = {
            "should_search": True,
            "search_queries": ["Calculator class implementation"],
            "search_type": "hybrid",
            "symbols": ["Calculator", "add"]
        }
        
        llm_response = json.dumps(expected_analysis)
        
        context = memory_context.retrieve(query, ask=lambda messages: llm_response)
        
        # Symbol search should have been called first
        calls = memory_context.ltm.db.get_class_code.call_count
        assert calls > 0


class TestContextFormatting:
    """Tests for formatting search results"""
    
    def test_format_context_results_with_valid_results(self, memory_context):
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
        
        formatted = memory_context.format_results(results)
        
        assert "Calculator" in formatted
        assert "standalone_function" in formatted
        assert "/test/calculator.py" in formatted
    
    def test_format_context_filters_short_content(self, memory_context):
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
        
        formatted = memory_context.format_results(results)
        
        # Short content should be skipped
        assert formatted == ""
    
    def test_format_context_truncates_long_content(self, memory_context):
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
        
        formatted = memory_context.format_results(results)
        
        # Should be truncated
        assert "..." in formatted


class TestIntegration:
    """Integration tests for the full RAG pipeline"""
    
    def test_full_query_to_context_pipeline(self, memory_context):
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
        
        context = memory_context.retrieve(query, ask=lambda messages: llm_response)
        
        # Verify the pipeline worked
        assert context is not None
        # Symbols should have been searched
        memory_context.ltm.db.get_class_code.assert_called()
        memory_context.ltm.db.get_method_code.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
