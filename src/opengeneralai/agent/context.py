"""Code context from the long-term memory, added to the prompts before planning.

An LLM call first decides whether the question is about code and how to search for it; the
symbols it names are looked up first, then the reformulated queries.
"""

import json
import logging
import re
from typing import Any, Callable, Dict, List

from opengeneralai.errors import LLMError
from opengeneralai.memory.longterm_memory import RESERVED_KEYWORD_CODE, LongTermMemory

logger = logging.getLogger(__name__)

# Sends messages to the LLM and returns its answer (Run.ask)
Ask = Callable[[List[dict]], str]


class MemoryContext:
    def __init__(self, ltm: LongTermMemory, max_context_tokens: int = 2000):
        self.ltm = ltm
        self.max_context_tokens = max_context_tokens

    def retrieve(self, query: str, ask: Ask) -> str:
        """Code context for the question, as text ("" when nothing relevant was found)."""
        try:
            # Step 1: Use LLM to analyze if we should search memory and how
            analysis = self.analyze_query(query, ask)
            
            if not analysis.get("should_search", True):
                return ""
            
            search_queries = analysis.get("search_queries", [query])
            search_type = analysis.get("search_type", "hybrid")
            symbols_to_find = analysis.get("symbols", [])
            
            all_results = []
            seen_chunks = set()
            
            # Step 2: Search for specific symbols first (highest priority)
            for symbol in symbols_to_find[:5]:
                symbol_results = self.search_symbol(symbol)
                for r in symbol_results:
                    if r.get("chunk_id") not in seen_chunks:
                        seen_chunks.add(r.get("chunk_id"))
                        all_results.append(r)
            
            # Step 3: Execute enhanced search queries
            for sq in search_queries[:3]:
                mode = "semantic" if search_type == "semantic" else "hybrid"
                results = self.ltm.search(sq, top_k=5, mode=mode)
                for r in results:
                    if r.get("chunk_id") not in seen_chunks:
                        seen_chunks.add(r.get("chunk_id"))
                        all_results.append(r)
            
            # Fallback: if no results, try direct query with keyword mode
            if not all_results:
                results = self.ltm.search(query, top_k=5, mode="keyword")
                all_results = results
            
            if not all_results:
                return ""
            
            return self.format_results(all_results)
            
        except Exception:
            # The context only enriches the prompt: answer without it rather than fail
            logger.exception("Memory context error")
            return ""

    def analyze_query(self, query: str, ask: Ask) -> Dict[str, Any]:
        """Use LLM to analyze query for memory search.
        
        This method determines:
        1. Whether it's useful to search memory (should_search)
        2. How to format the search queries (search_queries)
        3. What search mode to use (search_type)
        4. What code symbols to look up (symbols)
        """
        prompt = f"""Analyze this user query for code context retrieval.

User Query: {query}

You must respond with ONLY valid JSON (no markdown, no explanation). Determine:
1. should_search: Is this query asking about code? (true for any code-related question like "how does X work", "find Y", "show me Z", "what is class/function W", "implement feature", "fix bug in X")
2. search_queries: Reformulate the query to be more effective for semantic search. Include conceptual variations (e.g., "authentication implementation" from "how to add login").
3. search_type: "semantic" for conceptual questions, "keyword" for exact names, "hybrid" for both
4. symbols: List of specific code symbols (classes, functions, methods) mentioned or implied - extract CamelCase and snake_case identifiers

JSON schema:
{{"should_search": bool, "search_queries": [str], "search_type": "semantic"|"keyword"|"hybrid", "symbols": [str], "reasoning": str}}"""
        
        messages = [{"role": "system", "content": "You are a precise query analyzer. Always respond with valid JSON only."}, 
                   {"role": "user", "content": prompt}]
        
        try:
            response = ask(messages).strip()
            # Clean potential markdown code blocks
            if response.startswith("```"):
                parts = response.split("```")
                response = parts[1] if len(parts) > 1 else response
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()
            
            analysis = json.loads(response)
            return {
                "should_search": analysis.get("should_search", True),
                "search_queries": analysis.get("search_queries", [query]),
                "search_type": analysis.get("search_type", "hybrid"),
                "symbols": analysis.get("symbols", []),
                "reasoning": analysis.get("reasoning", "")
            }
        except (ValueError, LLMError) as e:
            # Invalid JSON or failed LLM call: extract the symbols locally and use hybrid search
            logger.warning("Query analysis failed, falling back to local symbol extraction: %s", e)
            return {
                "should_search": True,
                "search_queries": [query],
                "search_type": "hybrid",
                "symbols": self.extract_symbols(query),
                "reasoning": f"Fallback due to: {e}"
            }

    def extract_symbols(self, query: str) -> List[str]:
        """Extract code symbols from query using regex patterns."""
        # Extract CamelCase identifiers (e.g., MyClass, Orchestrator)
        camel = re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b', query)
        
        # Extract snake_case identifiers (e.g., my_function, get_data)
        snake = re.findall(r'\b[a-z_][a-z0-9_]*\b', query)
        
        # Filter out reserved keywords and short names
        return [s for s in camel + snake if s not in RESERVED_KEYWORD_CODE and len(s) > 2][:5]

    def search_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Comprehensive search for code symbols using multiple strategies.
        
        This method:
        1. First tries exact name match in database (highest priority)
        2. Then uses keyword search
        3. Finally uses semantic search as fallback
        
        Returns class, function, method and related code.
        """
        if not symbol or symbol in RESERVED_KEYWORD_CODE:
            return []
        
        all_results = []
        seen_chunks = set()
        
        # Strategy 1: Direct database lookups for exact matches
        try:
            # Try to get class code
            class_results = self.ltm.db.get_class_code(symbol)
            for r in class_results:
                chunk_id = r.get("chunk_id")
                if chunk_id and chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    all_results.append({
                        "chunk_id": chunk_id,
                        "document_id": r.get("document_id"),
                        "start_line": r.get("start_line"),
                        "end_line": r.get("end_line"),
                        "content": r.get("content"),
                        "chunk_type": "class",
                        "chunk_name": r.get("class_name"),
                        "source_path": r.get("source_path"),
                        "language": r.get("language")
                    })
        except Exception:
            pass
        
        try:
            # Try to get function code
            func_results = self.ltm.db.get_function_code(symbol)
            for r in func_results:
                chunk_id = r.get("chunk_id")
                if chunk_id and chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    all_results.append({
                        "chunk_id": chunk_id,
                        "document_id": r.get("document_id"),
                        "start_line": r.get("start_line"),
                        "end_line": r.get("end_line"),
                        "content": r.get("content"),
                        "chunk_type": "function",
                        "chunk_name": r.get("function_name"),
                        "source_path": r.get("source_path"),
                        "language": r.get("language")
                    })
        except Exception:
            pass
        
        try:
            # Try to get method code (could be in a class)
            method_results = self.ltm.db.get_method_code(symbol)
            for r in method_results:
                chunk_id = r.get("chunk_id")
                if chunk_id and chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    all_results.append({
                        "chunk_id": chunk_id,
                        "document_id": r.get("document_id"),
                        "start_line": r.get("start_line"),
                        "end_line": r.get("end_line"),
                        "content": r.get("content"),
                        "chunk_type": "method",
                        "chunk_name": r.get("method_name"),
                        "parent_class": r.get("class_name"),
                        "source_path": r.get("source_path"),
                        "language": r.get("language")
                    })
        except Exception:
            pass
        
        # Strategy 2: Keyword search for additional context
        if len(all_results) < 3:
            keyword_results = list(self.ltm.search(symbol, top_k=5, mode="keyword"))
            for r in keyword_results:
                chunk_id = r.get("chunk_id")
                if chunk_id and chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    all_results.append(r)
        
        # Strategy 3: Semantic search for broader context
        if len(all_results) < 3:
            semantic_results = list(self.ltm.search(symbol, top_k=5, mode="semantic"))
            for r in semantic_results:
                chunk_id = r.get("chunk_id")
                if chunk_id and chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    all_results.append(r)
        
        return all_results[:5]

    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results."""
        parts, total, max_chars = [], 0, self.max_context_tokens * 4
        for r in results:
            src = r.get("source_path", "unknown")
            ln = f"{r.get('start_line', 0)}-{r.get('end_line', 0)}"
            ct = r.get("chunk_type", "code")
            cn = r.get("chunk_name", "")
            header = f"[{ct}] {cn} @ {src}:{ln}" if cn else f"[{ct}] {src}:{ln}"
            content = r.get("content", "")
            if len(content) < 20:
                continue
            if len(content) > 2000:
                content = content[:2000] + "..."
            lang = r.get("language", "")
            backtick = "`"
            part = header + "\n" + backtick*3 + lang + "\n" + content + "\n" + backtick*3
            if total + len(part) > max_chars:
                break
            parts.append(part)
            total += len(part)
        return "\n\n---\n\n".join(parts) if parts else ""
