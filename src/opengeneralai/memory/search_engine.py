"""
Search Engine Module - Handles hybrid search (keyword + semantic).

This module is responsible for:
- Coordinating keyword search (FTS5) and semantic search (FAISS)
- Implementing hybrid search with score fusion
- Providing a clean search API

Responsibility: Pure search operations.
No database writes, parsing, or file handling.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np


class SearchEngine:
    """
    Hybrid search engine coordinating keyword and semantic search.
    
    This class handles the search logic without directly managing
    database connections or embedding computations.
    """
    
    def __init__(
        self,
        db_manager,
        embedding_manager: Optional[object] = None,
        hybrid_alpha: float = 0.5
    ):
        """
        Initialize the search engine.
        
        Args:
            db_manager: Database manager instance for FTS and metadata queries
            embedding_manager: Optional embedding manager for semantic search
            hybrid_alpha: Weight for keyword search in hybrid mode (0-1).
                         Higher = more weight on keyword search.
        """
        self.db = db_manager
        self.embedding_manager = embedding_manager
        self.hybrid_alpha = hybrid_alpha
    
    @property
    def embeddings_available(self) -> bool:
        """Check if embeddings are available for semantic search."""
        return self.embedding_manager is not None
    
    def search(
        self,
        query: str,
        top_k: int = 8,
        mode: str = "hybrid"
    ) -> List[Dict]:
        """
        Search for chunks matching the query.
        
        Args:
            query: The search query string
            top_k: Maximum number of results to return
            mode: Search mode - "keyword", "semantic", or "hybrid"
        
        Returns:
            List of result dictionaries with chunk information
        """
        mode = mode.lower()
        if mode not in {"keyword", "semantic", "hybrid"}:
            mode = "hybrid"

        kw_results = []
        sem_results = []

        # Keyword search
        if mode in {"keyword", "hybrid"}:
            kw_results = self.db.keyword_search(query, top_k * 3)

        # Semantic search
        if mode in {"semantic", "hybrid"}:
            if self.embedding_manager is None:
                if mode == "semantic":
                    # No embeddings available, return empty for semantic-only
                    pass
                # Fall back to keyword-only for hybrid
            else:
                q_emb = self.embedding_manager.encode(
                    [query], normalize=True, prompt_name="query"
                )
                sims, ids = self.embedding_manager.search(q_emb, top_k * 3)
                ids = ids[0]
                sims = sims[0]
                
                # Map faiss_id -> chunk
                if len(ids) > 0 and ids[0] != -1:
                    mapping = self.db.get_faiss_chunk_mapping([int(i) for i in ids])
                    for faiss_id, sim in zip(ids, sims):
                        if faiss_id == -1:
                            continue
                        chunk_id = mapping.get(int(faiss_id))
                        if chunk_id is None:
                            continue
                        row = self.db.get_chunk_by_id(chunk_id)
                        if row:
                            sem_results.append((*row, float(sim)))

        def assemble(rows):
            """Assemble search results with document metadata."""
            out = []
            for cid, did, s, e, chunk_type, name, content, score in rows:
                d = self.db.get_document_info(did)
                out.append({
                    "chunk_id": int(cid),
                    "document_id": did,
                    "start_line": int(s),
                    "end_line": int(e),
                    "chunk_type": chunk_type,
                    "chunk_name": name,
                    "content": content,
                    "score": float(score),
                    "source_path": d[0] if d else None,
                    "language": d[1] if d else None
                })
            return out

        if mode == "keyword":
            return assemble(kw_results)[:top_k]
        if mode == "semantic":
            return assemble(sem_results)[:top_k]

        # HYBRID: fusion by min-max normalization + alpha
        def norm_scores(lst):
            if not lst:
                return {}
            scores = [x[-1] for x in lst]
            lo, hi = min(scores), max(scores)
            if hi - lo < 1e-12:
                return {x[0]: 1.0 for x in lst}
            return {x[0]: (x[-1] - lo) / (hi - lo) for x in lst}

        kw_norm = norm_scores(kw_results)
        sem_norm = norm_scores(sem_results)

        # Gather by chunk_id
        all_ids = set(kw_norm.keys()) | set(sem_norm.keys())
        fused = []
        for cid in all_ids:
            k = kw_norm.get(cid, 0.0)
            s = sem_norm.get(cid, 0.0)
            score = self.hybrid_alpha * k + (1 - self.hybrid_alpha) * s
            
            # Get corresponding row (from kw or sem results)
            row = None
            for r in kw_results:
                if r[0] == cid:
                    row = r
                    break
            if row is None:
                for r in sem_results:
                    if r[0] == cid:
                        row = r
                        break
            if row:
                fused.append((*row[:-1], score))

        fused.sort(key=lambda r: r[-1], reverse=True)
        return assemble(fused)[:top_k]
    
    def keyword_search_only(
        self,
        query: str,
        top_k: int = 8
    ) -> List[Dict]:
        """Perform only keyword (FTS5) search."""
        return self.search(query, top_k, mode="keyword")
    
    def semantic_search_only(
        self,
        query: str,
        top_k: int = 8
    ) -> List[Dict]:
        """Perform only semantic (FAISS) search."""
        return self.search(query, top_k, mode="semantic")


class SearchResultFormatter:
    """Utility class for formatting search results."""
    
    @staticmethod
    def format_result(result: Dict, max_content_length: int = 200) -> str:
        """Format a single search result as a string."""
        lines = [
            f"Type: {result.get('chunk_type', 'unknown')}",
            f"Name: {result.get('chunk_name', 'N/A')}",
            f"Source: {result.get('source_path', 'N/A')}:{result.get('start_line', '?')}-{result.get('end_line', '?')}",
            f"Score: {result.get('score', 0):.4f}",
            "---",
        ]
        
        content = result.get('content', '')
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        lines.append(content)
        
        return "\n".join(lines)
    
    @staticmethod
    def format_results(results: List[Dict], max_content_length: int = 200) -> str:
        """Format multiple search results as a string."""
        if not results:
            return "No results found."
        
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"\n{'='*60}")
            formatted.append(f"Result {i}")
            formatted.append(SearchResultFormatter.format_result(result, max_content_length))
        
        return "\n".join(formatted)
