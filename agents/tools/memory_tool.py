from agents.tools.ToolRegistry import Tool, ToolResult
from storage.longterm_memory import LongTermMemory, RESERVED_KEYWORD_CODE
from typing import Callable, List, Dict, Any
import json
import re


class SearchContext(Tool):
    name = "generic_search"
    signature = "(query: str, type: str = 'any|class|function') -> content retrieved from context"
    
    # Search strategy definitions
    STRATEGY_PROMPT = """You are a query analyzer for a code search system. Analyze the user's query and determine the SINGLE best search strategy.

Available strategies:
1. CODE_DISCOVERY: Find code implementing specific functionality or algorithms
   - Use when: "how to...", "find implementation of...", "code that does..."
   - Examples: "authentication logic", "file upload handler", "sorting algorithm"

2. SYMBOL_INSPECTION: Look up specific symbols (classes, functions, methods)
   - Use when: exact names or CamelCase/snake_case identifiers are mentioned
   - Examples: "SearchContext class", "run method", "process_user_message function"

3. FILE_ANALYSIS: Analyze file structure, imports, or dependencies
   - Use when: asking about file organization, modules, imports
   - Examples: "what files use X?", "imports in module Y", "file structure"

4. CATALOG_BROWSE: Browse categories, list available symbols, explore codebase
   - Use when: open-ended exploration or listing requests
   - Examples: "list all classes", "available tools", "what's in the project"

Query: "{query}"
Type filter: {type_filter}

Respond with ONLY a JSON object:
{{"strategy": "CODE_DISCOVERY|SYMBOL_INSPECTION|FILE_ANALYSIS|CATALOG_BROWSE", "reasoning": "brief explanation"}}"""
    
    def __init__(self, ltm: LongTermMemory, ask_llm: Callable[[List[dict]], str]):
        self.ltm = ltm
        self.ask_llm = ask_llm
    
    def _analyze_query(self, query: str, type_filter: str) -> Dict[str, str]:
        """Use LLM to determine the best search strategy."""
        messages = [
            {
                "role": "system",
                "content": "You are a precise query analyzer. Always respond with valid JSON only."
            },
            {
                "role": "user",
                "content": self.STRATEGY_PROMPT.format(query=query, type_filter=type_filter)
            }
        ]
        
        try:
            response = self.ask_llm(messages)
            # Clean potential markdown code blocks
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()
            
            analysis = json.loads(response)
            return analysis
        except Exception as e:
            # Fallback to CODE_DISCOVERY on error
            return {"strategy": "CODE_DISCOVERY", "reasoning": f"Analysis failed: {e}"}
    
    def _search_code_discovery(self, query: str, type_filter: str) -> List[Dict[str, Any]]:
        """Search for code implementing specific functionality."""
        # Use semantic search for conceptual queries
        results = self.ltm.search(query, top_k=5, mode="semantic")
        
        # Filter by type if specified
        if type_filter != "any":
            results = [r for r in results if type_filter in r['chunk_type'] or 
                      (type_filter == "function" and "method" in r['chunk_type'])]
        
        return results
    
    def _search_symbol_inspection(self, query: str, type_filter: str) -> List[Dict[str, Any]]:
        """Look up specific symbols by name."""
        # Extract potential symbol names (CamelCase, snake_case)
        symbols = re.findall(r'\b[A-Z][a-zA-Z0-9_]*|[a-z_][a-z0-9_]*\b', query)
        
        results = []
        for symbol in symbols[:3]:  # Check top 3 candidates
            if symbol in RESERVED_KEYWORD_CODE:
                continue
            # Try exact keyword match first
            symbol_results = self.ltm.search(symbol, top_k=3, mode="keyword")
            results.extend(symbol_results)
        
        # Deduplicate by chunk_id
        seen = set()
        unique_results = []
        for r in results:
            if r.get('chunk_id') not in seen:
                seen.add(r.get('chunk_id'))
                unique_results.append(r)
        
        # Apply type filter
        if type_filter != "any":
            unique_results = [r for r in unique_results if type_filter in r['chunk_type'] or
                            (type_filter == "function" and "method" in r['chunk_type'])]
        
        return unique_results[:5]
    
    def _search_file_analysis(self, query: str, type_filter: str) -> List[Dict[str, Any]]:
        """Analyze file structure and dependencies."""
        # Use hybrid search for file-related queries
        results = self.ltm.search(query, top_k=5, mode="hybrid")
        
        # Prioritize imports and file-level structures
        imports = [r for r in results if r['chunk_type'] == 'import']
        classes = [r for r in results if r['chunk_type'] == 'class']
        others = [r for r in results if r['chunk_type'] not in ['import', 'class']]
        
        # Combine with imports first
        prioritized = imports + classes + others
        return prioritized[:5]
    
    def _search_catalog_browse(self, query: str, type_filter: str) -> List[Dict[str, Any]]:
        """Browse and list symbols in the codebase."""
        # Use database queries for catalog browsing
        results = []
        
        if type_filter == "class" or "class" in query.lower():
            # List classes
            classes = self.ltm.db.list_all_classes()
            for cls_name in classes[:5]:
                class_code = self.ltm.db.get_class_code(cls_name)
                if class_code:
                    results.append(class_code)
        elif type_filter == "function" or "function" in query.lower() or "method" in query.lower():
            # List functions/methods - use keyword search
            results = self.ltm.search(query, top_k=5, mode="keyword")
            results = [r for r in results if r['chunk_type'] in ['function', 'method']]
        else:
            # General catalog - use hybrid search
            results = self.ltm.search(query, top_k=5, mode="hybrid")
        
        return results
    
    def _format_results(self, results: List[Dict[str, Any]], strategy: str) -> str:
        """Format search results for display."""
        if not results:
            return "No relevant content found."
        
        content_parts = []
        for idx, r in enumerate(results[:3], 1):
            if r['chunk_type'] == 'class':
                summary = self.ltm.format_code_summary(r['chunk_name'], 'class')
                content_parts.append(f"[Result {idx}] {summary}")
            elif r['chunk_type'] in ['function', 'method']:
                summary = self.ltm.format_code_summary(
                    r['chunk_name'], 
                    r['chunk_type'], 
                    r.get('parent_class')
                )
                content_parts.append(f"[Result {idx}] {summary}")
            else:
                header = f"[Result {idx}] {r['language']} code from {r['source_path']} (lines {r['start_line']}-{r['end_line']})"
                content_parts.append(f"{header}\n```{r['language']}\n{r['content'][:500]}\n```")
        
        return "\n\n" + "\n\n---\n\n".join(content_parts)
    
    def run(self, **kwargs) -> ToolResult:
        query = kwargs.get("query")
        type_filter = kwargs.get("type", "any")
        
        if not query:
            return ToolResult(False, content="Missing 'query'", meta={"error": "Missing 'query'"})
        
        try:
            # Step 1: Analyze query to determine strategy
            analysis = self._analyze_query(query, type_filter)
            strategy = analysis.get("strategy", "CODE_DISCOVERY")
            reasoning = analysis.get("reasoning", "")
            
            # Step 2: Execute appropriate search strategy
            if strategy == "CODE_DISCOVERY":
                results = self._search_code_discovery(query, type_filter)
            elif strategy == "SYMBOL_INSPECTION":
                results = self._search_symbol_inspection(query, type_filter)
            elif strategy == "FILE_ANALYSIS":
                results = self._search_file_analysis(query, type_filter)
            elif strategy == "CATALOG_BROWSE":
                results = self._search_catalog_browse(query, type_filter)
            else:
                # Fallback
                results = self._search_code_discovery(query, type_filter)
            
            # Step 3: Format results
            content = self._format_results(results, strategy)
            
            return ToolResult(
                ok=True,
                content=content,
                meta={
                    "query": query,
                    "strategy": strategy,
                    "reasoning": reasoning,
                    "type": type_filter,
                    "count": str(len(results))
                }
            )
        except Exception as e:
            return ToolResult(
                ok=False,
                content=f"Search error: {e}",
                meta={"error": str(e)}
            )
