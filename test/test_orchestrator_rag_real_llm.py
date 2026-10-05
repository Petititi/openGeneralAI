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

from opengeneralai.agent.context import MemoryContext

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for API key availability
HAS_LLM_KEY = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("AZURE_API_KEY") or os.getenv("MISTRAL_API_KEY"))

# Real LLM calls: only with --run-llm (see conftest.py), and only if an API key is available
pytestmark = [pytest.mark.llm, pytest.mark.skipif(not HAS_LLM_KEY, reason="No LLM API key available")]


@pytest.fixture
def config():
    """Create a real configuration"""
    from opengeneralai.config import AppConfig
    
    CONFIG_PATH = os.getcwd() + "/config.json"
    ENV_PATH = os.getcwd() + "/.env"
    
    cfg = AppConfig(CONFIG_PATH, ENV_PATH)
    
    # Skip if not configured
    if cfg.need_configuration:
        pytest.skip("LLM not configured")
    
    return cfg


def real_ask(config):
    """Send messages to the configured LLM (these tests used to fall back without calling it:
    the trace the orchestrator needed was not initialized)."""
    from opengeneralai.llm.client import LiteLLMClient
    client = LiteLLMClient(config.model)
    return lambda messages: client.complete(messages).text


@pytest.fixture
def mock_tools():
    """Create a mock tool registry"""
    from opengeneralai.tools.registry import ToolRegistry
    
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
        orch = MemoryContext(mock_ltm)
        ask = real_ask(config)
        
        query = "How does the Orchestrator class work?"
        
        # Call the real LLM for query analysis
        result = orch.analyze_query(query, ask)
        
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
        orch = MemoryContext(mock_ltm)
        ask = real_ask(config)
        
        query = "What is the capital of France?"
        
        result = orch.analyze_query(query, ask)
        
        assert "should_search" in result
        assert isinstance(result["should_search"], bool)
        
        print(f"\nNon-Code Query Analysis Result:")
        print(f"  should_search: {result['should_search']}")
        print(f"  reasoning: {result.get('reasoning', 'N/A')}")
    
    def test_analyze_implementation_question_real_llm(self, config, mock_tools, mock_ltm):
        """Test LLM analysis of implementation questions"""
        orch = MemoryContext(mock_ltm)
        ask = real_ask(config)
        
        query = "Find and show me the get_class_code method implementation"
        
        result = orch.analyze_query(query, ask)
        
        assert result["should_search"] is True
        assert len(result["symbols"]) > 0
        
        print(f"\nImplementation Query Analysis:")
        print(f"  symbols: {result['symbols']}")
        print(f"  search_queries: {result['search_queries']}")


class TestRealSymbolSearch:
    """Tests using mocked database lookups"""
    
    def test_search_existing_class(self, config, mock_tools, mock_ltm):
        """Test searching for a class"""
        orch = MemoryContext(mock_ltm)
        ask = real_ask(config)
        
        results = orch.search_symbol("Orchestrator")
        
        assert len(results) > 0
        assert results[0].get("chunk_type") == "class"
        print(f"\nFound {len(results)} results for 'Orchestrator'")
    
    def test_search_function(self, config, mock_tools, mock_ltm):
        """Test searching for a function"""
        orch = MemoryContext(mock_ltm)
        ask = real_ask(config)
        
        results = orch.search_symbol("_get_relevant_context")
        
        assert len(results) > 0
        print(f"\nFound {len(results)} results for '_get_relevant_context'")


class TestRealContextRetrieval:
    """Tests for end-to-end context retrieval"""
    
    def test_get_relevant_context_class_query(self, config, mock_tools, mock_ltm):
        """Test retrieving context for a class-related query"""
        orch = MemoryContext(mock_ltm)
        ask = real_ask(config)
        
        query = "How is the Orchestrator class implemented?"
        
        context = orch.retrieve(query, ask)
        
        assert context is not None
        assert len(context) > 0
        assert "Orchestrator" in context
        
        print(f"\nContext length: {len(context)} chars")


class TestRealIntegration:
    """Full integration tests with real LLM"""
    
    def test_full_rag_pipeline(self, config, mock_tools, mock_ltm):
        """Test the complete RAG pipeline"""
        orch = MemoryContext(mock_ltm, max_context_tokens=2000)
        ask = real_ask(config)
        
        query = "Find the Orchestrator class and explain how it processes user messages"
        
        context = orch.retrieve(query, ask)
        
        assert context is not None
        assert len(context) > 0
        
        # Should contain code
        has_code = any(marker in context for marker in ["class", "def", "```", "Orchestrator"])
        assert has_code, "Context should contain code-related content"
        
        print(f"\nFull pipeline test:")
        print(f"  Query: {query}")
        print(f"  Context length: {len(context)} chars")
    
    def test_memory_context_injection(self, config, mock_tools, mock_ltm):
        """The memory context reaches the planning prompt, right after the core prompt"""
        from opengeneralai.agent.orchestrator import Orchestrator
        from opengeneralai.llm.client import LiteLLMClient

        orchestrator = Orchestrator(LiteLLMClient(config.model), mock_tools, memory=MemoryContext(mock_ltm), max_turns=1)
        result = orchestrator.run("How does the add method work?")

        plan_node = next(n for n in result.trace.nodes.values() if n.phase == "plan/create")
        assert plan_node.prompt_messages[1]["content"].startswith("Relevant context from the codebase")

        print("\nContext injection test:")
        print(f"  Planning prompt: {len(plan_node.prompt_messages)} messages")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
