"""
LLM Extractor Module - Shared LLM-based information extraction.

This module provides a generic LLM-based information extractor that can be used
by both LongTermMemory and SouvenirAssistant for extracting structured 
information from unstructured text.

Responsibility: Pure LLM-based text analysis and extraction.
No storage, database, or embedding concerns.
"""

import json
import re
from typing import List, Dict, Any, Optional, Callable

# Optional import - litellm for LLM interactions
try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


class LLMExtractor:
    """Generic LLM-based information extractor for any text content.
    
    This class uses an LLM to extract structured, meaningful information from
    unstructured text like emails, messages, documents, or code.
    """
    
    def __init__(self, model: str = "gpt-4o-mini", **kwargs):
        """Initialize LLM extractor.
        
        Args:
            model: The LLM model to use for extraction.
                   Defaults to "gpt-4o-mini" for cost efficiency.
            **kwargs: Additional model parameters (temperature, etc.)
        """
        self.model = model
        self.kwargs = kwargs
    
    def extract(
        self, 
        content: str, 
        extraction_type: str = "general",
        custom_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract meaningful information from content using LLM.
        
        Args:
            content: The text content to extract information from
            extraction_type: Type of extraction ('general', 'email', 'conversation', 'meeting', 'code')
            custom_prompt: Optional custom prompt for specialized extraction
            system_prompt: Optional custom system prompt
            
        Returns:
            Dictionary with extracted information fields, or {"ok": False, "error": ...}
        """
        if not LITELLM_AVAILABLE:
            return {"ok": False, "error": "litellm not installed"}
        
        if not content or not content.strip():
            return {"ok": False, "error": "Empty content"}
        
        # Build extraction prompt based on type
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = self._build_extraction_prompt(content, extraction_type)
        
        # Default system prompt
        if system_prompt is None:
            system_prompt = (
                "You are an expert information extraction assistant. "
                "Extract structured, meaningful information from the given content. "
                "Return valid JSON only. Respect language of content."
            )
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.kwargs.get("temperature", 0.3),
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            
            # Parse JSON response
            try:
                extracted = json.loads(result)
                return {"ok": True, "data": extracted}
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    extracted = json.loads(json_match.group())
                    return {"ok": True, "data": extracted}
                return {"ok": False, "error": "Could not parse JSON from LLM response"}
                
        except Exception as e:
            return {"ok": False, "error": str(e)}
    
    def _build_extraction_prompt(self, content: str, extraction_type: str) -> str:
        """Build extraction prompt based on type.
        
        Args:
            content: The content to extract from (will be truncated if too long)
            extraction_type: Type of extraction
            
        Returns:
            Formatted prompt string
        """
        # Truncate content if too long
        max_content_length = {
            "general": 3000,
            "email": 4000,
            "conversation": 5000,
            "meeting": 4000,
            "code": 6000,
        }.get(extraction_type, 3000)
        
        truncated_content = content[:max_content_length]
        
        # Build prompt based on extraction type
        prompts = {
            "general": """
Content:
---
{truncated_content}
---

Return a JSON object with these fields:
{{
    "summary": "2-3 sentence summary of the content, same language as content",
    "participants": ["list of participants"],
    "entities": {{"people": [], "organizations": [], "locations": []}},
    "key_points": ["main points discussed"],
    "action_items": ["any tasks or actions mentioned"],
    "important_dates": ["any dates or deadlines mentioned (DD-MM-YYYY hh:mm format)"],
    "sentiment": "positive, neutral, or negative",
    "urgency": "high, medium, or low",
    "follow_ups": ["items that need follow-up"]
}}""",
            
            "email": """
Email Content:
---
{truncated_content}
---

Analyze this email and return a JSON object with:
{{
    "summary": "2-3 sentence summary of the email",
    "sender": "email address or name of sender",
    "recipients": ["list of recipients"],
    "subject": "email subject line",
    "entities": {{"people": [], "organizations": [], "locations": []}},
    "key_points": ["main points in the email"],
    "action_items": ["any tasks or actions requested"],
    "urgency": "high, medium, or low",
    "requires_response": true or false,
    "follow_ups": ["items that need follow-up"]
}}""",
            
            "conversation": """
Conversation Transcript:
---
{truncated_content}
---

Analyze this conversation and return a JSON object with:
{{
    "summary": "2-3 sentence summary of the conversation",
    "participants": ["list of participants"],
    "entities": {{"people": [], "organizations": [], "locations": []}},
    "key_points": ["main topics discussed"],
    "decisions": ["any decisions made"],
    "action_items": ["tasks assigned or mentioned"],
    "sentiment": "positive, neutral, or negative",
    "follow_ups": ["items that need follow-up"]
}}""",
            
            "meeting": """
Meeting Notes:
---
{truncated_content}
---

Analyze this meeting and return a JSON object with:
{{
    "summary": "2-3 sentence summary of the meeting",
    "attendees": ["list of attendees"],
    "agenda_items": ["topics discussed"],
    "decisions": ["decisions made"],
    "action_items": ["tasks assigned with owner and deadline"],
    "next_meeting": "date/time if mentioned",
    "key_points": ["important takeaways"],
    "follow_ups": ["items that need follow-up"]
}}""",
            
            "code": """
Code Content:
---
{truncated_content}
---

Analyze this code and return a JSON object with:
{{
    "summary": "2-3 sentence summary of what this code does",
    "language": "programming language",
    "functions": ["list of function/method names"],
    "classes": ["list of class names"],
    "dependencies": ["external libraries or modules used"],
    "purpose": "main purpose of this code",
    "complexity": "low, medium, or high"
}}""",
        }
        
        prompt_template = prompts.get(extraction_type, prompts["general"])
        return prompt_template.format(truncated_content=truncated_content)
    
    def extract_batch(
        self, 
        contents: List[str], 
        extraction_type: str = "general",
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """Extract information from multiple contents.
        
        Args:
            contents: List of text contents to process
            extraction_type: Type of extraction
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            List of extraction results
        """
        results = []
        total = len(contents)
        
        for i, content in enumerate(contents):
            result = self.extract(content, extraction_type)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
    
    def summarize(self, content: str, max_length: int = 200) -> str:
        """Generate a simple summary of the content.
        
        Args:
            content: Content to summarize
            max_length: Maximum length of summary in characters
            
        Returns:
            Summary string
        """
        result = self.extract(content, extraction_type="general")
        if result.get("ok"):
            data = result.get("data", {})
            summary = data.get("summary", "")
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            return summary
        return ""


class CodeSummaryExtractor(LLMExtractor):
    """Specialized LLM extractor for code analysis and summarization."""
    
    def __init__(self, model: str = "gpt-4o-mini", **kwargs):
        super().__init__(model, **kwargs)
    
    def extract_summary(
        self, 
        code: str, 
        language: str = "python",
        include_docstring: bool = True
    ) -> Dict[str, Any]:
        """Extract a summary of code functionality.
        
        Args:
            code: The code content to analyze
            language: Programming language
            include_docstring: Whether to extract docstring if present
            
        Returns:
            Dictionary with code analysis
        """
        prompt = f"""
Analyze this {language} code and return a JSON object:

{{
    "summary": "2-3 sentence summary of what this code does",
    "function_name": "main function or class name if applicable",
    "parameters": ["list of parameters"],
    "return_value": "description of return value",
    "dependencies": ["external libraries/modules imported"],
    "side_effects": ["any side effects or I/O operations"],
    "complexity": "low, medium, or high",
    "suggested_tests": ["potential test cases"]
}}
"""
        return self.extract(code, extraction_type="code", custom_prompt=prompt)
    
    def explain_error(self, code: str, error_message: str) -> Dict[str, Any]:
        """Explain a code error and suggest fixes.
        
        Args:
            code: The code that produced an error
            error_message: The error message
            
        Returns:
            Dictionary with error explanation
        """
        prompt = f"""
Code:
---
{code}
---

Error:
{error_message}

Explain this error and suggest a fix. Return JSON:
{{
    "error_type": "type of error",
    "explanation": "why this error occurred",
    "suggested_fix": "how to fix the error",
    "corrected_code": "the corrected code if possible"
}}
"""
        return self.extract(prompt, extraction_type="code")
