"""
Constants Module - Centralized constants for the application.

This module consolidates all magic strings and numbers used across the codebase
to improve maintainability and reduce duplication.
"""

# ===================
# Agent Response Status Strings
# ===================
class AgentStatus:
    """Status strings used in agent responses."""
    STOP = "STOP"
    FINAL = "FINAL"
    FINAL_PREFIX = "FINAL:"  # Prefix for final responses
    PLAN_PREFIX = "PLAN:"    # Prefix for plan responses
    NEXT_TASK = "NEXT_TASK"
    NEW_PLAN = "NEW_PLAN"
    DONE = "DONE"
    ERROR = "error"


class StepStatus:
    """Status values for plan steps."""
    DONE = "done"
    CURRENT = "current"
    TODO = "todo"
    ERROR = "error"


# ===================
# Orchestrator Configuration
# ===================
class OrchestratorConfig:
    """Default configuration values for the Orchestrator."""
    # Default max tokens for context (roughly 1000 words to limit costs)
    DEFAULT_CONTEXT_MAX_TOKENS = 2000
    # Maximum number of turns per task
    DEFAULT_MAX_TURN = 10
    # Multiplier for calculating max characters from tokens
    CONTEXT_CHARS_MULTIPLIER = 4
    # Maximum content length for individual chunks
    MAX_CHUNK_CONTENT_LENGTH = 2000
    # Number of search queries to analyze
    MAX_SEARCH_QUERIES = 3
    # Maximum symbols to search
    MAX_SYMBOLS_TO_FIND = 5
    # Number of results per search
    DEFAULT_TOP_K = 5


# ===================
# Code Parser Configuration
# ===================
class CodeParserConfig:
    """Configuration values for code parsing."""
    # Maximum characters per chunk
    MAX_CHARS = 6000
    # Default max chars for text extraction
    DEFAULT_TEXT_MAX_CHARS = 3000
    # Default max bytes for reading files
    DEFAULT_MAX_FILE_BYTES = 10_000_000


# ===================
# Search Configuration
# ===================
class SearchConfig:
    """Configuration values for search functionality."""
    # Default number of results
    DEFAULT_TOP_K = 5
    # Minimum results threshold
    MIN_RESULTS_THRESHOLD = 3


# ===================
# Input Validation
# ===================
class InputValidation:
    """Input validation limits."""
    MAX_QUESTION_LENGTH = 10000
    MAX_ANSWER_LENGTH = 50000


# ===================
# Database Configuration
# ===================
class DatabaseConfig:
    """Database-related configuration."""
    DEFAULT_DB_PATH = "memory.sqlite"
    DEFAULT_STORAGE_DIR = "storage"
    DEFAULT_FAISS_INDEX_PATH = "faiss.index"


# ===================
# Embedding Configuration
# ===================
class EmbeddingConfig:
    """Configuration for embedding models."""
    DEFAULT_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
    # Hybrid search alpha weight (0 = keyword, 1 = semantic)
    DEFAULT_HYBRID_ALPHA = 0.5
    # Embedding dimension (for validation)
    EMBEDDING_DIMENSION = 1024


# ===================
# API Configuration
# ===================
class APIConfig:
    """API-related configuration."""
    DEFAULT_PORT = 12000
    DEFAULT_HOST = "0.0.0.0"
    # Timeout for LLM requests (seconds)
    LLM_TIMEOUT = 30


# ===================
# Logging Configuration
# ===================
class LoggingConfig:
    """Logging configuration."""
    DEFAULT_LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
