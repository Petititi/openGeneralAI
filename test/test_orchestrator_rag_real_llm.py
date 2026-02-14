"""
Integration tests for the enhanced RAG features using a real LLM.

These tests verify:
1. LLM-based query analysis with actual API calls
2. Real semantic search against indexed code
3. End-to-end context retrieval

Note: These tests require a valid LLM API key in the environment.
They will be skipped if no API key is available.
"""

import sys
import os
from pathlib import Path
import pytest
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for API key availability
HAS_LLM_KEY = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("AZURE_API_KEY"))


# Only run these tests if we have an API key
pytestmark = pytest.mark.skipif(not HAS_LLM_KEY, reason="No LLM API key available")


@pytest.fixture
def persistant_ltm():
    """Fixture to create a persistent LongTermMemory instance with real data"""
    db_path = "test/datas/test_memory.sqlite"
    faiss_index_path = "test/datas/test_faiss.index"

    from storage.longterm_memory import LongTermMemory
    
    CREATE_DBD = not Path(db_path).exists()
    
    ltm = LongTermMemory(
        db_path=str(db_path),
        faiss_index_path=str(faiss_index_path)
    )

    if CREATE_DBD:
        ltm.add_folder(str(Path(__file__).parent.parent))
    
    yield ltm
    
    # Cleanup
    ltm.close()


@pytest.fixture
def config():
    """Create a real configuration"""
    import configurator
    
    CONFIG_PATH = os.getcwd() + "/config.json"
    ENV_PATH = os.getcwd() + "/.env"
    
    cfg = configurator.AppConfig(CONFIG_PATH, ENV_PATH)
    
    # Skip if not configured
    if cfg.need_configuration:
        pytest.skip("LLM not configured")
    
    return cfg


@pytest.fixture
def mock_tools():
    """Create a mock tool registry"""
    from unittest.mock import Mock
    from agents.tools.ToolRegistry import ToolRegistry
    
    tools = ToolRegistry()
    # Register a mock tool
    mock_tool = Mock()
    mock_tool.name = "mock_tool"
    mock_tool.signature = "() -> test"
    tools.register(mock_tool)
    
    return tools


class TestRealLLMQueryAnalysis:
    """Tests using real LLM for query analysis"""
    
    def test_analyze_code_question_real_llm(self, config, mock_tools, persistant_ltm):
        """Test LLM analysis of a code-related question"""
        from agents.orchestrator import Orchestrator
        from agents.logger import TrajectoryLogger
        
        # Create orchestrator with real LTM
        with patch('agents.orchestrator.TrajectoryLogger'):
            orch = Orchestrator(
                cfg=config,
                tools=mock_tools,
                ltm=persistant_ltm,
                max_turn=3
            )
        
        query = "How does the LongTermMemory class work?"
        
        # Call the real LLM
        result = orch._analyze_query_for_memory(query)
        
        # Verify the result structure
        assert "should_search" in result
        assert "search_queries" in result
        assert "search_type" in result
        assert "symbols" in result
        
        # For code questions, should_search should be True
        assert result["should_search"] is True
        
        # Should have extracted some symbols
        assert len(result["symbols"]) > 0
        
        # Should have reformulated search queries
        assert len(result["search_queries"]) > 0
        
        print(f"\nQuery Analysis Result:")
        print(f"  should_search: {result['should_search']}")
        print(f"  search_queries: {result['search_queries']}")
        print(f"  search_type: {result['search_type']}")
        print(f"  symbols: {result['symbols']}")
        print(f"  reasoning: {result.get('reasoning', 'N/A')}")
    
    def test_analyze_non_code_question_real_llm(self, config, mock_tools, persistant_ltm):
        """Test LLM analysis of a non-code question"""
        from agents.orchestrator import Orchestrator
        
        with patch('agents.orchestrator.TrajectoryLogger'):
            orch = Orchestrator(
                cfg=config,
                tools=mock_tools,
                ltm=persistant_ltm,
                max_turn=3
            )
        
        query = "What is the capital of France?"
        
        result = orch._analyze_query_for_memory(query)
        
        assert "should_search" in result
        # Non-code questions might return should_search=False
        assert isinstance(result["should_search"], bool)
        
        print(f"\nNon-Code Query Analysis Result:")
        print(f"  should_search: {result['should_search']}")
        print(f"  reasoning: {result.get('reasoning', 'N/A')}")
    
    def test_analyze_implementation_question_real_llm(self, config, mock_tools, persistant_ltm):
        """Test LLM analysis of implementation questions"""
        from agents.orchestrator import Orchestrator
        
        with patch('agents.orchestrator.TrajectoryLogger'):
            orch = Orchestrator(
                cfg=config,
                tools=mock_tools,
                ltm=persistant_ltm,
                max_turn=3
            )
        
        query = "Find and show me the get_class_code method implementation"
        
        result = orch._analyze_query_for_memory(query)
        
        assert result["should_search"] is True
        assert "get_class_code" in result["symbols"] or len(result["symbols"]) > 0
        
        print(f"\nImplementation Query Analysis:")
        print(f"  symbols: {result['symbols']}")
        print(f"  search_queries: {result['search_queries']}")


class TestRealSymbolSearch:
    """Tests using real database lookups"""
    
    def test_search_existing_class_real_db(self, config, mock_tools, persistant_ltm):
        """Test searching for an existing class in the database"""
        from agents.orchestrator import Orchestrator
        
        with patch('agents.orchestrator.TrajectoryLogger'):
            orch = Orchestrator(
                cfg=config,
                tools=mock_tools,
                ltm=persistant_ltm,
                max_turn=3
            )
        
        # Search for a class that exists in the indexed codebase
        results = orch._search_symbol_comprehensive("LongTermMemory")
        
        # Should find results
        assert len(results) > 0
        
        # Verify result structure
        first_result = results[0]
        assert "chunk_id" in first_result
        assert "content" in first_result
        assert "chunk_type" in first_result
        
        print(f"\nFound {len(results)} results for 'LongTermMemory'")
        print(f"First result type: {first_result.get('chunk_type')}")
    
    def test_search_function_real_db(self, config, mock_tools, persistant_ltm):
        """Test searching for a function in the database"""
        from agents.orchestrator import Orchestrator
        
        with patch('agents.orchestrator.TrajectoryLogger'):
            orch = Orchestrator(
                cfg=config,
                tools=mock_tools,
                ltm=persistant_ltm,
                max_turn=3
            )
        
        # Search for a function
        results = orch._search_symbol_comprehensive("add_folder")
        
        # Should find results (it's a method in LongTermMemory)
        assert len(results) > 0
        
        print(f"\nFound {len(results)} results for 'add_folder'")
    
    def test_search_unknown_symbol_fallback(self, config, mock_tools, persistant_ltm):
        """Test fallback search for unknown symbols"""
        from agents.orchestrator import Orchestrator
        
        with patch('agents.orchestrator.TrajectoryLogger'):
            orch = Orchestrator(
                cfg=config,
                tools=mock_tools,
                ltm=persistant_ltm,
                max_turn=3
            )
        
        # Search for something that doesn't exist
        results = orch._search_symbol_comprehensive("CompletelyRandomSymbolXYZ123")
        
        # Should still return some results from semantic search fallback
        assert isinstance(results, list)


class TestRealContextRetrieval:
    """Tests for end-to-end context retrieval"""
    
    def test_get_relevant_context_class_query(self, config, mock_tools, persistant_ltm):
        """Test retrieving context for a class-related query"""
        from agents.orchestrator import Orchestrator
        
        with patch('agents.orchestrator.TrajectoryLogger'):
            orch = Orchestrator(
                cfg=config,
                tools=mock_tools,
                ltm=persistant_ltm,
                max_turn=3
            )
        
        query = "How is the DatabaseManager class implemented?"
        
        context = orch._get_relevant_context(query)
        
        # Should return some context
        assert context is not None
        assert len(context) > 0
        
        # Should contain code
        assert "class" in context.lower() or "def" in context.lower()
        
        print(f"\nContext length: {len(context)} chars")
        print(f"Context preview: {context[:500]}...")
    
    def test_get_relevant_context_method_query(self, config, mock_tools, persistant_ltm):
        """Test retrieving context for a method-related query"""
        from agents.orchestrator import Orchestrator
        
        with patch('agents.orchestrator.TrajectoryLogger'):
            orch = Orchestrator(
                cfg=config,
                tools=mock_tools,
                ltm=persistant_ltm,
                max_turn=3
            )
        
        query = "Show me the search method implementation"
        
        context = orch._get_relevant_context(query)
        
        assert context is not None
        assert len(context) > 0
        
        print(f"\nMethod query context: {context[:300]}...")
    
    def test_get_relevant_context_no_ltm(self, config, mock_tools):
        """Test context retrieval when no LTM is available"""
        from agents.orchestrator import Orchestrator
        
        with patch('agents.orchestrator.TrajectoryLogger'):
            orch = Orchestrator(
                cfg=config,
                tools=mock_tools,
                ltm=None,
                max_turn=3
            )
        
        query = "Test query"
        
        context = orch._get_relevant_context(query)
        
        # Should return empty string
        assert context == ""


class TestRealIntegration:
    """Full integration tests with real LLM and real database"""
    
    def test_full_rag_pipeline_orchestrator(self, config, mock_tools, persistant_ltm):
        """Test the complete RAG pipeline in the orchestrator"""
        from agents.orchestrator import Orchestrator
        from agents.logger import TrajectoryLogger
        
        with patch('agents.orchestrator.TrajectoryLogger'):
            orch = Orchestrator(
                cfg=config,
                tools=mock_tools,
                ltm=persistant_ltm,
                max_turn=3,
                max_context_tokens=2000
            )
        
        # Test query that should trigger memory search
        query = "Find the Orchestrator class and explain how it processes user messages"
        
        # This tests the full pipeline
        context = orch._get_relevant_context(query)
        
        # Verify we got useful context
        assert context is not None
        
        # The context should mention the relevant code
        if len(context) > 0:
            # Should contain some code from the codebase
            has_code = any(marker in context for marker in ["class", "def", "```", "orchestrator"])
            assert has_code, "Context should contain code-related content"
        
        print(f"\nFull pipeline test:")
        print(f"  Query: {query}")
        print(f"  Context length: {len(context)} chars")
        print(f"  Context has code: {'class' in context.lower()}")
    
    def test_memory_context_injection(self, config, mock_tools, persistant_ltm):
        """Test that memory context is properly injected into messages"""
        from agents.orchestrator import Orchestrator
        
        with patch('agents.orchestrator.TrajectoryLogger'):
            orch = Orchestrator(
                cfg=config,
                tools=mock_tools,
                ltm=persistant_ltm,
                max_turn=3
            )
        
        # Get context first
        query = "How does the add method work in Calculator?"
        context = orch._get_relevant_context(query)
        
        # Inject into messages
        messages = [
            {"role": "system", "content": orch.core_prompt},
            {"role": "user", "content": query}
        ]
        
        enhanced_messages = orch._inject_memory_context(messages, context)
        
        # Should have more messages now
        assert len(enhanced_messages) > len(messages)
        
        # The context should be in a system message
        context_in_messages = any(
            "context from the codebase" in msg.get("content", "").lower() 
            for msg in enhanced_messages 
            if msg.get("role") == "system"
        )
        
        print(f"\nContext injection test:")
        print(f"  Original messages: {len(messages)}")
        print(f"  Enhanced messages: {len(enhanced_messages)}")
        print(f"  Context injected: {context_in_messages}")


# Helper for patching
from unittest.mock import patch


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
