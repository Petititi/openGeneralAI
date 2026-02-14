"""
Integration tests for the enhanced RAG features using a real LLM.

These tests verify:
1. LLM-based query analysis with actual API calls
2. Database lookups with mocked LTM
3. End-to-end context retrieval

Note: These tests require a valid LLM API key in the environment.
They will be skipped if no API key is available.
"""

import sys
import os
from pathlib import Path
import pytest
import json
from unittest.mock import Mock, MagicMock, patch, create_autospec

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for API key availability
HAS_LLM_KEY = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("AZURE_API_KEY") or os.getenv("MISTRAL_API_KEY"))

# Only run these tests if we have an API key
pytestmark = pytest.mark.skipif(not HAS_LLM_KEY, reason="No LLM API key available")


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
    from agents.tools.ToolRegistry import ToolRegistry
    
    tools = ToolRegistry()
    mock_tool = Mock()
    mock_tool.name = "mock_tool"
    mock_tool.signature = "() -> test"
    tools.register(mock_tool)
    return tools


@pytest.fixture
def mock_ltm():
    """Create a mocked LongTermMemory instance with database lookups"""
    # Create a mock for the RESERVED_KEYWORD_CODE constant
    RESERVED_KEYWORD_CODE = [
        "def", "class", "function", "var", "let", "const", "import", "from",
        "public", "private", "protected", "interface", "struct", "enum", "package",
        "return", "if", "else", "switch", "case", "for", "while", "do", "try",
    ]
    
    ltm = Mock()
    ltm.db = MagicMock()
    
    # Mock get_class_code to return sample data
    def mock_get_class_code(class_name):
        return [
            {
                "chunk_id": 1,
                "document_id": "doc_orchestrator",
                "start_line": 17,
                "end_line": 50,
                "content": """class Orchestrator:
    def __init__(self, cfg, tools, ltm=None, max_turn=10):
        self.cfg = cfg
        self.tools = tools
        self.ltm = ltm
        self.max_turn = max_turn""",
                "class_name": "Orchestrator",
                "source_path": "/workspace/project/openGeneralAI/agents/orchestrator.py",
                "language": "python"
            }
        ]
    
    def mock_get_function_code(func_name):
        return [
            {
                "chunk_id": 10,
                "document_id": "doc_orchestrator",
                "start_line": 114,
                "end_line": 163,
                "content": """def _get_relevant_context(self, query: str) -> str:
    if self.ltm is None:
        return \"\"
    # LLM-based query analysis""",
                "function_name": "_get_relevant_context",
                "source_path": "/workspace/project/openGeneralAI/agents/orchestrator.py",
                "language": "python"
            }
        ]
    
    def mock_get_method_code(method_name, class_name=None):
        return [
            {
                "chunk_id": 20,
                "document_id": "doc_orchestrator",
                "start_line": 165,
                "end_line": 216,
                "content": """def _analyze_query_for_memory(self, query: str) -> Dict[str, Any]:
    prompt = f\"Analyze this user query for code context retrieval.\"""",
                "method_name": "_analyze_query_for_memory",
                "class_name": "Orchestrator",
                "source_path": "/workspace/project/openGeneralAI/agents/orchestrator.py",
                "language": "python"
            }
        ]
    
    ltm.db.get_class_code = Mock(side_effect=mock_get_class_code)
    ltm.db.get_function_code = Mock(side_effect=mock_get_function_code)
    ltm.db.get_method_code = Mock(side_effect=mock_get_method_code)
    
    ltm.search = Mock(return_value=[
        {
            "chunk_id": 1,
            "document_id": "doc_orchestrator",
            "start_line": 17,
            "end_line": 50,
            "content": "class Orchestrator: ...",
            "chunk_type": "class",
            "chunk_name": "Orchestrator",
            "source_path": "/workspace/project/openGeneralAI/agents/orchestrator.py",
            "language": "python",
            "score": 0.95
        }
    ])
    
    return ltm


class TestRealLLMQueryAnalysis:
    """Tests using real LLM for query analysis"""
    
    def test_analyze_code_question_real_llm(self, config, mock_tools, mock_ltm):
        """Test LLM analysis of a code-related question"""
        from agents.orchestrator import Orchestrator
        
        with patch('agents.orchestrator.TrajectoryLogger'):
            orch = Orchestrator(
                cfg=config,
                tools=mock_tools,
                ltm=mock_ltm,
                max_turn=3
            )
        
        query = "How does the Orchestrator class work?"
        
        # Call the real LLM for query analysis
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
    
    def test_analyze_non_code_question_real_llm(self, config, mock_tools, mock_ltm):
        """Test LLM analysis of a non-code question"""
        from agents.orchestrator import Orchestrator
        
        with patch('agents.orchestrator.TrajectoryLogger'):
            orch = Orchestrator(
                cfg=config,
                tools=mock_tools,
                ltm=mock_ltm,
                max_turn=3
            )
        
        query = "What is the capital of France?"
        
        result = orch._analyze_query_for_memory(query)
        
        assert "should_search" in result
        assert isinstance(result["should_search"], bool)
        
        print(f"\nNon-Code Query Analysis Result:")
        print(f"  should_search: {result['should_search']}")
        print(f"  reasoning: {result.get('reasoning', 'N/A')}")
    
    def test_analyze_implementation_question_real_llm(self, config, mock_tools, mock_ltm):
        """Test LLM analysis of implementation questions"""
        from agents.orchestrator import Orchestrator
        
        with patch('agents.orchestrator.TrajectoryLogger'):
            orch = Orchestrator(
                cfg=config,
                tools=mock_tools,
                ltm=mock_ltm,
                max_turn=3
            )
        
        query = "Find and show me the get_class_code method implementation"
        
        result = orch._analyze_query_for_memory(query)
        
        assert result["should_search"] is True
        assert len(result["symbols"]) > 0
        
        print(f"\nImplementation Query Analysis:")
        print(f"  symbols: {result['symbols']}")
        print(f"  search_queries: {result['search_queries']}")


class TestRealSymbolSearch:
    """Tests using mocked database lookups"""
    
    def test_search_existing_class(self, config, mock_tools, mock_ltm):
        """Test searching for a class"""
        from agents.orchestrator import Orchestrator
        
        with patch('agents.orchestrator.TrajectoryLogger'):
            orch = Orchestrator(
                cfg=config,
                tools=mock_tools,
                ltm=mock_ltm,
                max_turn=3
            )
        
        results = orch._search_symbol_comprehensive("Orchestrator")
        
        assert len(results) > 0
        assert results[0].get("chunk_type") == "class"
        print(f"\nFound {len(results)} results for 'Orchestrator'")
    
    def test_search_function(self, config, mock_tools, mock_ltm):
        """Test searching for a function"""
        from agents.orchestrator import Orchestrator
        
        with patch('agents.orchestrator.TrajectoryLogger'):
            orch = Orchestrator(
                cfg=config,
                tools=mock_tools,
                ltm=mock_ltm,
                max_turn=3
            )
        
        results = orch._search_symbol_comprehensive("_get_relevant_context")
        
        assert len(results) > 0
        print(f"\nFound {len(results)} results for '_get_relevant_context'")


class TestRealContextRetrieval:
    """Tests for end-to-end context retrieval"""
    
    def test_get_relevant_context_class_query(self, config, mock_tools, mock_ltm):
        """Test retrieving context for a class-related query"""
        from agents.orchestrator import Orchestrator
        
        with patch('agents.orchestrator.TrajectoryLogger'):
            orch = Orchestrator(
                cfg=config,
                tools=mock_tools,
                ltm=mock_ltm,
                max_turn=3
            )
        
        query = "How is the Orchestrator class implemented?"
        
        context = orch._get_relevant_context(query)
        
        assert context is not None
        assert len(context) > 0
        assert "Orchestrator" in context
        
        print(f"\nContext length: {len(context)} chars")
    
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
        
        assert context == ""


class TestRealIntegration:
    """Full integration tests with real LLM"""
    
    def test_full_rag_pipeline(self, config, mock_tools, mock_ltm):
        """Test the complete RAG pipeline"""
        from agents.orchestrator import Orchestrator
        
        with patch('agents.orchestrator.TrajectoryLogger'):
            orch = Orchestrator(
                cfg=config,
                tools=mock_tools,
                ltm=mock_ltm,
                max_turn=3,
                max_context_tokens=2000
            )
        
        query = "Find the Orchestrator class and explain how it processes user messages"
        
        context = orch._get_relevant_context(query)
        
        assert context is not None
        assert len(context) > 0
        
        # Should contain code
        has_code = any(marker in context for marker in ["class", "def", "```", "Orchestrator"])
        assert has_code, "Context should contain code-related content"
        
        print(f"\nFull pipeline test:")
        print(f"  Query: {query}")
        print(f"  Context length: {len(context)} chars")
    
    def test_memory_context_injection(self, config, mock_tools, mock_ltm):
        """Test that memory context is properly injected into messages"""
        from agents.orchestrator import Orchestrator
        
        with patch('agents.orchestrator.TrajectoryLogger'):
            orch = Orchestrator(
                cfg=config,
                tools=mock_tools,
                ltm=mock_ltm,
                max_turn=3
            )
        
        query = "How does the add method work?"
        context = orch._get_relevant_context(query)
        
        messages = [
            {"role": "system", "content": orch.core_prompt},
            {"role": "user", "content": query}
        ]
        
        enhanced_messages = orch._inject_memory_context(messages, context)
        
        assert len(enhanced_messages) > len(messages)
        
        print(f"\nContext injection test:")
        print(f"  Original messages: {len(messages)}")
        print(f"  Enhanced messages: {len(enhanced_messages)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
