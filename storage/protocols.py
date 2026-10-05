"""
Protocols Module - Abstract base classes for swappable backends.

This module defines protocol classes (interfaces) for various backends
to enable swapping implementations (e.g., FAISS -> Pinecone, Qdrant).

Responsibility: Abstract interface definitions.
No concrete implementations.
"""

from typing import Protocol, List, Dict, Any, Optional, runtime_checkable
from pathlib import Path
import numpy as np


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector storage backends.
    
    This defines the interface for any vector store implementation
    (FAISS, Pinecone, Qdrant, Weaviate, etc.)
    """
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """Add vectors to the store.
        
        Args:
            vectors: NxD array of vectors
            metadata: Optional metadata for each vector
            
        Returns:
            List of vector IDs
        """
        ...
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector
            top_k: Number of results
            filters: Optional metadata filters
            
        Returns:
            List of search results with metadata and scores
        """
        ...
    
    def delete(self, vector_ids: List[int]) -> None:
        """Delete vectors by ID.
        
        Args:
            vector_ids: List of vector IDs to delete
        """
        ...
    
    def save(self, path: str) -> None:
        """Save the index to disk.
        
        Args:
            path: Path to save the index
        """
        ...
    
    def load(self, path: str) -> None:
        """Load the index from disk.
        
        Args:
            path: Path to load the index from
        """
        ...


@runtime_checkable
class SearchBackend(Protocol):
    """Protocol for search engine backends.
    
    This defines the interface for any search backend
    (SQLite FTS, Elasticsearch, Meilisearch, etc.)
    """
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for documents.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Optional filters
            
        Returns:
            List of search results
        """
        ...
    
    def index_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Index a document.
        
        Args:
            doc_id: Document ID
            content: Document content
            metadata: Optional metadata
        """
        ...
    
    def delete_document(self, doc_id: str) -> None:
        """Delete a document.
        
        Args:
            doc_id: Document ID to delete
        """
        ...


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Protocol for embedding model backends.
    
    This defines the interface for any embedding model
    (OpenAI, HuggingFace, Cohere, etc.)
    """
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress
            
        Returns:
            NxD array of embeddings
        """
        ...
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        ...
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        ...


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for storage backends.
    
    This defines the interface for any persistent storage
    (SQLite, PostgreSQL, MongoDB, etc.)
    """
    
    def connect(self) -> None:
        """Establish connection to the storage."""
        ...
    
    def close(self) -> None:
        """Close the connection."""
        ...
    
    def execute(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> Any:
        """Execute a query.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            Query results
        """
        ...
    
    def commit(self) -> None:
        """Commit the current transaction."""
        ...
    
    def rollback(self) -> None:
        """Rollback the current transaction."""
        ...
