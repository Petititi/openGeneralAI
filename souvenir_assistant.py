#!/usr/bin/env python3
"""
Souvenir - A personal memory assistant powered by LLM.

This application helps users remember things by:
1. Adding persistent souvenirs (notes, memories, facts)
2. Retrieving details about stored souvenirs using semantic search

Usage:
    python souvenir_assistant.py add "My first day at work"
    python souvenir_assistant.py search "What did I do on my first day?"
    python souvenir_assistant.py list
    python souvenir_assistant.py interactive
"""

import argparse
import sys
from typing import List, Dict
import hashlib
from email_reply_parser import EmailReplyParser
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import configurator
import litellm
from storage.longterm_memory import LongTermMemory
from storage.DatabaseManagement import DatabaseManager
from typing import Optional, List, Dict, Any, Callable
import re
import json


class LLMExtractor:
    """Generic LLM-based information extractor for any text content.
    
    This class uses an LLM to extract structured, meaningful information from
    unstructured text like emails, messages, or documents.
    """
    
    def __init__(self, cfg: Optional[Any] = None):
        """Initialize LLM extractor.
        
        Args:
            cfg: Optional configuration object with 'model' attribute. 
                 If not provided, uses default litellm settings.
        """
        self.cfg = cfg
        self.model = cfg.model if cfg else "gpt-4o-mini"
    
    def clean_email_thread(self, text):
        text = re.sub(r'\[cid:.*?\]', '', text)
        text = re.sub(r'\[https?://.*?\]', '', text)
        
        text = re.sub(r'<mailto:.*?>', '', text)
        
        text = re.sub(r'[ \t]+', ' ', text)
        
        text = re.sub(r'\n{2,}', '\n\n', text)
        
        text = re.sub(r'.*Office 365.*', '', text)
        text = re.sub(r'.*Unknown To address.*', '', text)
        text = re.sub(r'.*Action Required.*', '', text)
        
        text = text.strip()
        
        return text
    
    def extract(self, content: str, extraction_type: str = "general", 
                custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Extract meaningful information from content using LLM.
        
        Args:
            content: The text content to extract information from
            extraction_type: Type of extraction ('general', 'email', 'conversation', 'meeting')
            custom_prompt: Optional custom prompt for specialized extraction
            
        Returns:
            Dictionary with extracted information fields
        """
        reduced_content = self.clean_email_thread(content)
        if not reduced_content:
            return {"ok": False, "error": "Empty content"}
        
        # Build extraction prompt based on type
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = self._build_extraction_prompt(reduced_content, extraction_type)
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert information extraction assistant. Extract structured, meaningful information from the given content. Return valid JSON only. Respect language of content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
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
        """Build extraction prompt based on type."""
        
        base_prompt = f"""
Content:
---
{content[:3000]}
---

Return a JSON object with these fields:
"""
        base_prompt = base_prompt + """
{
    "summary": "2-3 sentence summary of the content, same language as content",
    "participants": ["list of participants"],
    "entities": {"people": [], "organizations": [], "locations": []},
    "key_points": ["main points discussed"],
    "action_items": ["any tasks or actions mentioned"],
    "important_dates": ["any dates or deadlines mentioned (DD-MM-YYYY hh:mm format)"],
    "sentiment": "positive, neutral, or negative",
    "urgency": "high, medium, or low",
    "follow_ups": ["items that need follow-up"]
}"""
        return base_prompt
    
    def extract_batch(self, contents: List[str], extraction_type: str = "general",
                      progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Dict[str, Any]]:
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


class ThreadBuilder:
    """Generic message thread builder that reconstructs conversations.
    
    Supports multiple strategies for linking messages:
    - References/In-Reply-To headers (email)
    - Subject line matching (Re:, Fwd:, etc.)
    - Content similarity (body text appears in other messages)
    """
    
    def __init__(self, id_field: str = "id", parent_id_field: str = "parent_id",
                 date_field: str = "date", subject_field: str = "subject",
                 body_field: str = "body", references_field: str = "references",
                 in_reply_to_field: str = "in_reply_to", message_id_field: str = "message_id"):
        """Initialize thread builder.
        
        Args:
            id_field: Field name for message ID
            parent_id_field: Field name for parent message ID
            date_field: Field name for message date
            subject_field: Field name for subject
            body_field: Field name for message body
            references_field: Field name for References header
            in_reply_to_field: Field name for In-Reply-To header
            message_id_field: Field name for Message-ID header
        """
        self.id_field = id_field
        self.parent_id_field = parent_id_field
        self.date_field = date_field
        self.subject_field = subject_field
        self.body_field = body_field
        self.references_field = references_field
        self.in_reply_to_field = in_reply_to_field
        self.message_id_field = message_id_field
    
    def build_threads(self, messages: List[Dict], 
                     use_subject_matching: bool = True,
                     use_body_similarity: bool = True,
                     similarity_threshold: float = 0.3) -> Dict[str, Dict]:
        """Build thread structure from a list of messages.
        
        Args:
            messages: List of message dictionaries
            use_subject_matching: Whether to use subject matching for threading
            use_body_similarity: Whether to detect quoted/forwarded content
            similarity_threshold: Threshold for body similarity detection
            
        Returns:
            Dictionary mapping thread_id -> thread data
        """
        if not messages:
            return {}
        
        # First, group by thread_id (Gmail's thread ID for emails)
        thread_map = self._group_by_thread_id(messages)
        
        # For messages without thread_id, try to link using other methods
        # Now merge small threads (<=2 messages) that share subject or body content
        thread_map = self._merge_small_threads(
            thread_map, 
            use_subject_matching=use_subject_matching,
            use_body_similarity=use_body_similarity,
            similarity_threshold=similarity_threshold
        )
        
        # Sort messages within each thread by date
        for thread_id in thread_map:
            thread_map[thread_id]['emails'].sort(key=lambda m: m.get(self.date_field, ''))
        
        return thread_map
    
    def _merge_small_threads(self, thread_map: Dict[str, Dict],
                            use_subject_matching: bool = True,
                            use_body_similarity: bool = True,
                            similarity_threshold: float = 0.3) -> Dict[str, Dict]:
        """Merge small threads (<=2 messages) that share subject or body content.
        
        Args:
            thread_map: Current thread mapping
            use_subject_matching: Whether to use subject matching
            use_body_similarity: Whether to use body similarity
            similarity_threshold: Threshold for body similarity
            
        Returns:
            Updated thread mapping with merged threads
        """
        # Find small threads (1 or 2 messages)
        small_threads = []
        for thread_id, thread_data in thread_map.items():
            if thread_data.get('size', 0) <= 2:
                small_threads.append((thread_id, thread_data))
        
        if not small_threads:
            return thread_map
        
        # Collect all messages from small threads
        all_small_messages = []
        for thread_id, thread_data in small_threads:
            all_small_messages.extend(thread_data.get('emails', []))
        
        # Remove small threads from map
        for thread_id, _ in small_threads:
            del thread_map[thread_id]
        
        # Re-group small messages using subject and body similarity
        if all_small_messages:
            # Group by subject first
            if use_subject_matching:
                subject_groups = self._group_by_subject(all_small_messages)
                # Add these as threads
                for thread_id, thread_data in subject_groups.items():
                    thread_map[thread_id] = thread_data
                
                # Get remaining ungrouped messages
                remaining = [m for m in all_small_messages 
                           if not m.get('_linked_by_subject')]
            else:
                remaining = all_small_messages
            
            # Then group remaining by body similarity
            if use_body_similarity and remaining:
                body_groups = self._group_by_body_similarity(remaining, similarity_threshold)
                for thread_id, thread_data in body_groups.items():
                    thread_map[thread_id] = thread_data
            
            # Any remaining messages become single-message threads
            used_ids = set()
            for thread_data in thread_map.values():
                for email in thread_data.get('emails', []):
                    used_ids.add(email.get(self.id_field))
            
            for msg in remaining:
                msg_id = msg.get(self.id_field)
                if msg_id not in used_ids:
                    thread_id = f"orphan_{msg_id}"
                    thread_map[thread_id] = {
                        'emails': [msg],
                        'size': 1,
                        'method': 'remaining'
                    }
                    used_ids.add(msg_id)
        
        return thread_map
    
    def _group_by_thread_id(self, messages: List[Dict]) -> Dict[str, Dict]:
        """Group messages by native thread ID."""
        from collections import defaultdict
        
        thread_map = defaultdict(list)
        for msg in messages:
            thread_id = msg.get('thread_id', '')
            if thread_id:
                thread_map[thread_id].append(msg)
        
        # Convert to proper format
        threads = {}
        for thread_id, msgs in thread_map.items():
            threads[thread_id] = {
                'emails': msgs,
                'size': len(msgs),
                'method': 'thread_id'
            }
        
        return threads
    
    def _group_by_subject(self, messages: List[Dict]) -> Dict[str, Dict]:
        """Group messages by subject line patterns (Re:, Fwd:, etc.)."""
        from collections import defaultdict
        
        # Normalize subjects by removing Re:, Fwd:, etc.
        def normalize_subject(subject: str) -> str:
            if not subject:
                return ""
            # Remove common prefixes
            normalized = re.sub(r'^(Re:|Fwd:|Re\[?\d*\]?:?)\s*', '', subject, flags=re.IGNORECASE)
            return normalized.strip().lower()
        
        # Group by normalized subject
        subject_map = defaultdict(list)
        for msg in messages:
            subject = msg.get(self.subject_field, '')
            if subject:
                normalized = normalize_subject(subject)
                if normalized:
                    subject_map[normalized].append(msg)
        
        # Filter to only subjects with multiple messages
        threads = {}
        for subject, msgs in subject_map.items():
            if len(msgs) > 1:
                # Create unique thread ID from subject
                thread_id = f"subject_{hash(subject)}"
                
                # Build parent-child relationships
                roots, replies = self._build_tree_from_references(msgs)
                
                threads[thread_id] = {
                    'emails': msgs,
                    'root_emails': roots,
                    'replies': replies,
                    'size': len(msgs),
                    'method': 'subject'
                }
                
                # Mark messages as linked
                for m in msgs:
                    m['_linked_by_subject'] = True
        
        return threads
    
    def _group_by_body_similarity(self, messages: List[Dict], 
                                   threshold: float = 0.3) -> Dict[str, Dict]:
        """Group messages by body text similarity (quoted/forwarded content)."""
        from collections import defaultdict
        
        # Simple similarity: check if one message's body contains 
        # significant portion of another message's body
        threads = []
        used = set()
        
        for i, msg in enumerate(messages):
            if msg.get(self.id_field) in used:
                continue
            
            body = msg.get(self.body_field, '')
            if not body or len(body) < 50:
                continue
            
            # Find similar messages
            similar = [msg]
            for j, other in enumerate(messages):
                if i == j or other.get(self.id_field) in used:
                    continue
                
                other_body = other.get(self.body_field, '')
                if not other_body:
                    continue
                
                # Check if one contains the other
                if len(body) > 100 and len(other_body) > 100:
                    # Simple containment check
                    smaller, larger = (body, other_body) if len(body) < len(other_body) else (other_body, body)
                    
                    # Check for significant overlap (at least 30% of smaller)
                    if smaller in larger or self._calculate_similarity(smaller, larger) > threshold:
                        similar.append(other)
            
            if len(similar) > 1:
                thread_id = f"body_sim_{msg.get(self.id_field, i)}"
                threads.append({
                    'id': thread_id,
                    'emails': similar,
                    'size': len(similar),
                    'method': 'body_similarity'
                })
                for m in similar:
                    used.add(m.get(self.id_field))
        
        # Convert to dict
        return {t['id']: {k: v for k, v in t.items() if k != 'id'} for t in threads}
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two texts."""
        # Tokenize and compute Jaccard similarity
        def tokenize(text):
            return set(text.lower().split())
        
        set1, set2 = tokenize(text1), tokenize(text2)
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _build_tree_from_references(self, messages: List[Dict]) -> tuple:
        """Build parent-child tree from References/In-Reply-To headers."""
        from collections import defaultdict
        
        roots = []
        replies = defaultdict(list)
        
        for msg in messages:
            refs = msg.get(self.references_field, '').split()
            in_reply_to = msg.get(self.in_reply_to_field, '').strip('<>')
            
            parent_found = False
            for ref in refs:
                ref = ref.strip('<>')
                if ref:
                    for potential_parent in messages:
                        parent_msg_id = potential_parent.get(self.message_id_field, '').strip('<>')
                        if parent_msg_id == ref:
                            replies[potential_parent.get(self.id_field)].append(msg)
                            parent_found = True
                            break
            
            if not parent_found:
                roots.append(msg)
        
        # If no roots found, use first message as root
        if not roots and messages:
            roots = [messages[0]]
            for msg in messages[1:]:
                replies[messages[0].get(self.id_field)].append(msg)
        
        return roots, dict(replies)
    
    def _merge_thread_maps(self, base: Dict, additional: Dict) -> Dict:
        """Merge two thread maps."""
        for thread_id, thread_data in additional.items():
            if thread_id not in base:
                base[thread_id] = thread_data
        return base


class SouvenirAssistant:
    """A personal memory assistant using LLM and LongTermMemory."""
    
    def __init__(self, db_path: Optional[str] = None, enable_embeddings: bool = False):
        """Initialize the souvenir assistant with LongTermMemory storage."""
        
        # Load configuration using configurator
        root_folder = Path(__file__).parent
        config_path = root_folder / "config.json"
        env_path = root_folder / ".env"
        self.cfg = configurator.AppConfig(config_path, env_path)
        
        # Use config for db_path if not provided
        if db_path is None:
            db_path = self.cfg.souvenir_db_path
        
        self.db_path = db_path
        
        # Initialize LongTermMemory with optional embeddings
        try:
            self.ltm = LongTermMemory(
                db_path=db_path,
                enable_embeddings=enable_embeddings
            )
            print(f"✓ LongTermMemory initialized (embeddings: {enable_embeddings})")
        except Exception as e:
            print(f"❌ Failed to initialize LongTermMemory: {e}")
            raise
        
        if self.cfg.need_configuration:
            print("Warning: LLM not configured. Some features may not work.")
        
        # Initialize LLM extractor for rich information extraction
        self.extractor = LLMExtractor(self.cfg)
        
        # Initialize thread builder for conversation reconstruction
        self.thread_builder = ThreadBuilder()
    
    def extract_information(self, content: str, extraction_type: str = "general") -> Dict[str, Any]:
        """Extract meaningful information from content using LLM.
        
        This method uses an LLM to intelligently extract structured information
        from unstructured text like emails, messages, or documents.
        
        Args:
            content: The text content to extract information from
            extraction_type: Type of extraction ('general', 'email', 'conversation', 'meeting')
            
        Returns:
            Dictionary with extracted information fields (summary, topics, entities, etc.)
        """
        return self.extractor.extract(content, extraction_type)
    
    def build_threads(self, messages: List[Dict], 
                      use_subject_matching: bool = True,
                      use_body_similarity: bool = True) -> Dict[str, Dict]:
        """Build thread structure from a list of messages.
        
        Uses multiple strategies:
        - Native thread ID (e.g., Gmail thread ID)
        - Subject line matching (Re:, Fwd:, etc.)
        - Body text similarity (quoted/forwarded content)
        
        Args:
            messages: List of message dictionaries
            use_subject_matching: Whether to use subject matching for threading
            use_body_similarity: Whether to detect quoted/forwarded content
            
        Returns:
            Dictionary mapping thread_id -> thread data with 'emails', 'size', etc.
        """
        return self.thread_builder.build_threads(
            messages, 
            use_subject_matching=use_subject_matching,
            use_body_similarity=use_body_similarity
        )
    
    def process_conversation_for_memory(self, messages: List[Dict]) -> Dict[str, Any]:
        """Process a conversation/thread and extract meaningful information using LLM.
        
        Args:
            messages: List of message dictionaries in chronological order
            
        Returns:
            Dictionary with extracted info: topic, participants, key_points, etc.
        """
        if not messages:
            return {"ok": False, "error": "No messages provided"}
        
        # Build conversation content
        conversation_content = self._format_conversation_for_extraction(messages)
        
        # Use LLM to extract information
        result = self.extractor.extract(conversation_content, extraction_type="conversation")
        
        if not result.get("ok"):
            return {
                "ok": False,
                "error": result.get("error", "Extraction failed"),
                "topic": "",
                "participants": [],
                "key_points": [],
                "tags": ["conversation"]
            }
        
        data = result.get("data", {})
        
        # Build tags
        tags = ["conversation"]
        if data.get("action_items"):
            tags.append("action_required")
        
        return {
            "ok": True,
            "topic": data.get("summary", ""),
            "participants": data.get("participants", []),
            "entities": data.get("entities", {'people':[], 'organizations':[], 'locations':[]}),
            "key_points": data.get("key_points", []),
            "action_items": data.get("action_items", []),
            "important_dates": data.get("important_dates", []),
            "follow_ups": data.get("follow_ups", []),
            "sentiment": data.get("sentiment", "neutral"),
            "urgency": data.get("urgency", ""),
            "tags": list(set(tags))
        }
    
    def _format_email_for_extraction(self, email: Dict) -> str:
        """Format an email for LLM extraction."""
        lines = []
        
        if email.get("subject"):
            lines.append(f"Subject: {email['subject']}")
        
        if email.get("sender"):
            lines.append(f"From: {email['sender']}")
        
        if email.get("to"):
            lines.append(f"To: {email['to']}")
        
        if email.get("date"):
            lines.append(f"Date: {email['date']}")
        
        lines.append("")
        lines.append("Content:")
        lines.append(email.get("body", "")[:3000])
        
        return "\n".join(lines)
    
    def _strip_quoted_text(self, body: str) -> str:
        return EmailReplyParser.parse_reply(body)

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into meaningful paragraphs.
        """
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _hash_paragraph(self, text: str) -> str:
        """
        Generate stable hash for a paragraph (normalized).
        """
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()

    def _format_conversation_for_extraction(self, messages: List[Dict]) -> str:
        """
        Format conversation for LLM extraction.
        Deduplicate content at paragraph level instead of message level.
        """
        lines = []
        seen_paragraph_hashes = set()
        for i, msg in enumerate(messages):
            raw_body = msg.get("body", "")
            clean_body = self._strip_quoted_text(raw_body)

            paragraphs = self._split_into_paragraphs(clean_body)
            new_paragraphs = []
            for paragraph in paragraphs:
                p_hash = self._hash_paragraph(paragraph)

                if p_hash not in seen_paragraph_hashes or len(paragraph) < 15:
                    seen_paragraph_hashes.add(p_hash)
                    new_paragraphs.append(paragraph)
                else:
                    break


            if not new_paragraphs:
                continue

            lines.append(f"--- Message {i+1} ---")

            if msg.get("sender") and msg.get("to"):
                lines.append(f"From: {msg['sender']} To: {msg['to']}")

            if msg.get("date"):
                lines.append(f"Date: {msg['date']}")

            if msg.get("subject"):
                lines.append(f"Subject: {msg['subject']}")

            lines.append("")
            lines.append("\n\n".join(new_paragraphs)[:1000])
            lines.append("")

        return "\n".join(lines)

    def _format_conversation_for_extraction_legacy(self, messages: List[Dict]) -> str:
        """Format a conversation for LLM extraction."""
        lines = []
        
        for i, msg in enumerate(messages):
            lines.append(f"--- Message {i+1} ---")
            
            if msg.get("sender"):
                lines.append(f"From: {msg['sender']}")
            
            if msg.get("date"):
                lines.append(f"Date: {msg['date']}")
            
            if msg.get("subject"):
                lines.append(f"Subject: {msg['subject']}")
            
            lines.append("")
            lines.append(msg.get("body", "")[:1000])
            lines.append("")
        
        return "\n".join(lines)
    
    def analyze_memories(self, question: str, category_filter: Optional[str] = None, 
                        limit: int = 50) -> Dict[str, Any]:
        """Analyze stored memories to answer high-level questions using LLM.
        
        This method retrieves relevant memories and uses LLM to analyze them
        and answer questions about the stored information.
        
        Args:
            question: The question to answer about the memories
            category_filter: Optional category to filter memories
            limit: Maximum number of memories to retrieve
            
        Returns:
            Dictionary with answer and sources
        """
        # Get relevant memories
        souvenirs = self.list_souvenirs(category=category_filter, limit=limit)
        
        if not souvenirs:
            return {
                "ok": True,
                "answer": "No memories found to analyze.",
                "sources": []
            }
        
        # Format memories for LLM analysis
        context = self._format_memories_for_analysis(souvenirs)
        
        # Build prompt for analysis
        prompt = f"""Based on the following stored memories, answer this question: {question}

Memories:
---
{context}
---

Provide a detailed answer based on the information available in these memories. If the information is not available, say so clearly.
"""
        
        try:
            response = litellm.completion(
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes personal memories and emails. Answer questions based on the stored information. Be specific and reference the sources when possible."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            return {
                "ok": True,
                "answer": answer,
                "sources": [s.get("id", "unknown") for s in souvenirs]
            }
        except Exception as e:
            return {
                "ok": False,
                "error": str(e)
            }
    
    def _format_memories_for_analysis(self, souvenirs: List[Dict]) -> str:
        """Format memories for LLM analysis."""
        lines = []
        
        for i, s in enumerate(souvenirs):
            lines.append(f"--- Memory {i+1} ---")
            lines.append(f"Title: {s.get('title', 'Untitled')}")
            lines.append(f"Category: {s.get('category', 'general')}")
            lines.append(f"Tags: {', '.join(s.get('tags', []))}")
            lines.append(f"Content: {s.get('content', '')[:500]}")
            lines.append("")
        
        return "\n".join(lines)
    
    def extract_insights_from_memories(self, category_filter: Optional[str] = None,
                                        limit: int = 100) -> Dict[str, Any]:
        """Extract high-level insights from stored memories using LLM.
        
        Analyzes memories to find:
        - Common themes and topics
        - Key contacts and their frequency
        - Pending action items
        - Upcoming events/dates
        
        Args:
            category_filter: Optional category to filter memories
            limit: Maximum number of memories to analyze
            
        Returns:
            Dictionary with extracted insights
        """
        souvenirs = self.list_souvenirs(category=category_filter, limit=limit)
        
        if not souvenirs:
            return {"ok": False, "error": "No memories found"}
        
        # Format memories
        context = self._format_memories_for_analysis(souvenirs)
        
        prompt = f"""Analyze the following memories and extract high-level insights.

Provide a JSON response with these fields:
{{
    "key_themes": ["list of main themes/topics found"],
    "important_contacts": [{{"name": "name", "role": "role", "frequency": number}}],
    "pending_actions": ["list of action items found"],
    "upcoming_events": ["list of events/dates mentioned"],
    "summary": "brief overall summary of what these memories reveal"
}}

Memories:
---
{context}
---
"""
        
        try:
            response = litellm.completion(
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing personal memories and communications. Extract meaningful insights and return structured JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            
            try:
                insights = json.loads(result)
                return {"ok": True, "insights": insights, "memories_analyzed": len(souvenirs)}
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    insights = json.loads(json_match.group())
                    return {"ok": True, "insights": insights, "memories_analyzed": len(souvenirs)}
                return {"ok": False, "error": "Could not parse JSON from LLM response"}
                
        except Exception as e:
            return {"ok": False, "error": str(e)}
    
    def add_souvenir(self, content: str, title: Optional[str] = None, category: str = 'general', tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Add a new souvenir/memory to the storage."""
        try:
            import uuid
            
            # Generate a unique ID for the souvenir
            souvenir_id = str(uuid.uuid4())[:8]
            
            # Use the DatabaseManager directly to insert a souvenir
            self.ltm.db.insert_souvenir(
                doc_id=souvenir_id,
                content=content,
                title=title,
                category=category,
                tags=tags
            )
            
            return {
                "ok": True,
                "id": souvenir_id,
                "title": title or "Untitled Memory",
                "category": category,
                "message": f"Souvenir added successfully! (ID: {souvenir_id})"
            }
        except Exception as e:
            return {
                "ok": False,
                "error": str(e)
            }

    def find_similar_documents(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.95
    ) -> List[Dict[str, Any]]:
        """Find documents with similar content using semantic search.
        
        Args:
            query_text: The text to search for
            top_k: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0-1) to consider a match
            
        Returns:
            List of similar documents with their similarity scores
        """
        if self.ltm.embeddings is None:
            return []
        
        try:
            # Encode the query
            query_emb = self.ltm.embeddings.encode([query_text], normalize=True)
            
            # Search in FAISS index
            distances, indices = self.ltm.embeddings.search(query_emb, top_k)
            
            # Get chunk information for matches above threshold
            similar_docs = []
            for dist, idx in zip(distances[0], indices[0]):
                if dist >= similarity_threshold and idx >= 0:
                    # Get chunk info from FAISS mapping
                    chunk_info = self.ltm.db.get_chunk_by_id(int(idx))
                    if chunk_info:
                        chunk_id, doc_id, start_line, end_line, chunk_type, chunk_name, content = chunk_info
                        # Get full document details
                        doc_details = self.ltm.db.get_document_details(doc_id)
                        if doc_details:
                            similar_docs.append({
                                "chunk_id": chunk_id,
                                "document_id": doc_id,
                                "document": doc_details["document"],
                                "chunk_content": content,
                                "chunk_type": chunk_type,
                                "similarity_score": float(dist)
                            })
            
            return similar_docs
        except Exception as e:
            print(f"Error searching for similar documents: {e}")
            return []
    
    def embedding_exists(
        self,
        text: str,
        similarity_threshold: float = 0.9999
    ) -> Optional[int]:
        """Check if an embedding for the given text already exists.
        
        Uses exact match (cosine similarity = 1.0 for normalized embeddings).
        
        Args:
            text: The text to check
            similarity_threshold: Minimum similarity to consider as exact match
            
        Returns:
            FAISS ID if found, None otherwise
        """
        if self.ltm.embeddings is None or self.ltm.embeddings.ntotal == 0:
            return None
        
        try:
            # Encode the text
            text_emb = self.ltm.embeddings.encode([text], normalize=True)
            
            # Search for exact match
            distances, indices = self.ltm.embeddings.search(text_emb, 1)
            
            # Check if we have an exact match
            if distances[0][0] >= similarity_threshold and indices[0][0] >= 0:
                return int(indices[0][0])
            
            return None
        except Exception as e:
            print(f"Error checking embedding existence: {e}")
            return None
    
    def add_email_memory(
        self,
        thread_emails: List[Dict],
        conversation_info: Dict,
        category: str = 'email_conversation',
        tags: Optional[List[str]] = None,
        allow_duplicates: bool = False
    ) -> Dict[str, Any]:
        """Add an email thread as a memory with multiple chunks.
        
        Instead of flattening all information into a single content string,
        this method stores different aspects of the email conversation as
        separate chunks, enabling better retrieval and search.
        
        Args:
            thread_emails: List of email dictionaries in the thread
            conversation_info: Information extracted by LLM (topic, participants, key_points, etc.)
            category: Category for the memory
            tags: Optional tags for the memory
            allow_duplicates: If False, check for similar existing documents first
            
        Returns:
            Dictionary with operation result
        """
        try:
            import uuid
            import hashlib
            
            # Build content for similarity check
            topic = conversation_info.get("topic", "")
            key_points = conversation_info.get("key_points", [])
            content_for_check = topic + " " + " ".join(key_points)
            
            # Check for similar existing documents if duplicates not allowed
            if not allow_duplicates and self.ltm.embeddings is not None and content_for_check.strip():
                similar_docs = self.find_similar_documents(content_for_check, top_k=3, similarity_threshold=0.90)
                if similar_docs:
                    # Return the most similar document instead of creating a duplicate
                    best_match = similar_docs[0]
                    return {
                        "ok": True,
                        "id": best_match["document_id"],
                        "title": best_match["document"].get("extra", {}).get("title", "Email Conversation"),
                        "category": category,
                        "chunks_created": 0,
                        "message": "Similar email memory already exists",
                        "existing_document": best_match["document"],
                        "similarity_score": best_match["similarity_score"]
                    }
            
            # Generate a unique ID for the email memory
            memory_id = str(uuid.uuid4())[:8]
            
            # Build title from topic or subject
            if conversation_info.get("topic"):
                title = f"📧 {conversation_info['topic']}"
            else:
                subject = thread_emails[0].get('subject', 'Email Conversation')
                title = f"📧 Thread: {subject[:45]}{'...' if len(subject) > 45 else ''}"
            
            # Prepare extra metadata
            extra = {
                'title': title,
                'tags': tags or [],
                'email_thread': {
                    'subject': thread_emails[0].get('subject', ''),
                    'message_count': len(thread_emails),
                    'participants': conversation_info.get('participants', []),
                }
            }
            
            # Insert the document
            content_preview = conversation_info.get('topic', '')[:500]
            self.ltm.db.insert_document(
                doc_id=memory_id,
                source_path=f"email_memory:{memory_id}",
                rel_path=None,
                media_type='text/plain',
                language='en',
                sha256='',
                size_bytes=len(content_preview),
                category=category,
                extra=extra
            )
            
            # Create multiple chunks for different aspects of the email thread
            chunk_ordinal = 0
            
            # Chunk 1: Topic/Summary
            if conversation_info.get("topic"):
                topic_content = f"Topic: {conversation_info['topic']}"
                topic_chunk_id = self.ltm.db.insert_chunk(
                    memory_id, chunk_ordinal, 1, len(topic_content.split('\n')),
                    topic_content, 'email_topic', conversation_info['topic'][:50], None
                )
                chunk_ordinal += 1
                
                # Add embedding for topic (check for duplicates)
                if self.ltm.embeddings is not None:
                    existing_faiss_id = self.embedding_exists(topic_content)
                    if existing_faiss_id is None:
                        emb = self.ltm.embeddings.encode([topic_content], normalize=True)
                        faiss_ids = self.ltm.embeddings.add_embeddings(emb)
                        self.ltm.db.insert_faiss_mapping(faiss_ids[0], topic_chunk_id)
                    else:
                        # Reuse existing embedding
                        self.ltm.db.insert_faiss_mapping(existing_faiss_id, topic_chunk_id)
            
            # Chunk 2: Participants
            participants = conversation_info.get("participants", [])
            if participants:
                participants_content = "Participants:\n" + "\n".join(f"  - {p}" for p in participants)
                participants_chunk_id = self.ltm.db.insert_chunk(
                    memory_id, chunk_ordinal, 1, len(participants_content.split('\n')),
                    participants_content, 'email_participants', None, None
                )
                chunk_ordinal += 1
            
            # Chunk 3: Key Points
            key_points = conversation_info.get("key_points", [])
            if key_points:
                key_points_content = "Key Points:\n" + "\n".join(f"  - {point}" for point in key_points)
                key_points_chunk_id = self.ltm.db.insert_chunk(
                    memory_id, chunk_ordinal, 1, len(key_points_content.split('\n')),
                    key_points_content, 'email_key_points', None, None
                )
                chunk_ordinal += 1
                
                # Add embedding for key points (check for duplicates)
                if self.ltm.embeddings is not None:
                    existing_faiss_id = self.embedding_exists(key_points_content)
                    if existing_faiss_id is None:
                        emb = self.ltm.embeddings.encode([key_points_content], normalize=True)
                        faiss_ids = self.ltm.embeddings.add_embeddings(emb)
                        self.ltm.db.insert_faiss_mapping(faiss_ids[0], key_points_chunk_id)
                    else:
                        # Reuse existing embedding
                        self.ltm.db.insert_faiss_mapping(existing_faiss_id, key_points_chunk_id)
            
            # Chunk 4: Action Items
            action_items = conversation_info.get("action_items", [])
            if action_items:
                action_items_content = "Action Items:\n" + "\n".join(f"  - {item}" for item in action_items)
                action_items_chunk_id = self.ltm.db.insert_chunk(
                    memory_id, chunk_ordinal, 1, len(action_items_content.split('\n')),
                    action_items_content, 'email_action_items', None, None
                )
                chunk_ordinal += 1
                
                # Add embedding for action items (check for duplicates)
                if self.ltm.embeddings is not None:
                    existing_faiss_id = self.embedding_exists(action_items_content)
                    if existing_faiss_id is None:
                        emb = self.ltm.embeddings.encode([action_items_content], normalize=True)
                        faiss_ids = self.ltm.embeddings.add_embeddings(emb)
                        self.ltm.db.insert_faiss_mapping(faiss_ids[0], action_items_chunk_id)
                    else:
                        # Reuse existing embedding
                        self.ltm.db.insert_faiss_mapping(existing_faiss_id, action_items_chunk_id)
            
            # Chunk 5: Important Dates
            important_dates = conversation_info.get("important_dates", [])
            if important_dates:
                dates_content = "Important Dates:\n" + "\n".join(f"  - {date}" for date in important_dates)
                dates_chunk_id = self.ltm.db.insert_chunk(
                    memory_id, chunk_ordinal, 1, len(dates_content.split('\n')),
                    dates_content, 'email_dates', None, None
                )
                chunk_ordinal += 1
            
            # Chunk 6: Entities (people, organizations, locations)
            entities = conversation_info.get("entities", {})
            if entities:
                entities_lines = []
                people = entities.get("people", [])
                orgs = entities.get("organizations", [])
                locations = entities.get("locations", [])
                
                if people:
                    entities_lines.append("People:")
                    for p in people:
                        entities_lines.append(f"  - {p}")
                if orgs:
                    entities_lines.append("Organizations:")
                    for o in orgs:
                        entities_lines.append(f"  - {o}")
                if locations:
                    entities_lines.append("Locations:")
                    for loc in locations:
                        entities_lines.append(f"  - {loc}")
                
                if entities_lines:
                    entities_content = "\n".join(entities_lines)
                    entities_chunk_id = self.ltm.db.insert_chunk(
                        memory_id, chunk_ordinal, 1, len(entities_content.split('\n')),
                        entities_content, 'email_entities', None, None
                    )
                    chunk_ordinal += 1
            
            # Chunk 7: Follow-ups
            follow_ups = conversation_info.get("follow_ups", [])
            if follow_ups:
                follow_ups_content = "Follow-ups:\n" + "\n".join(f"  - {follow_up}" for follow_up in follow_ups)
                follow_ups_chunk_id = self.ltm.db.insert_chunk(
                    memory_id, chunk_ordinal, 1, len(follow_ups_content.split('\n')),
                    follow_ups_content, 'email_follow_ups', None, None
                )
                chunk_ordinal += 1
                
                # Add embedding for follow-ups (check for duplicates)
                if self.ltm.embeddings is not None:
                    existing_faiss_id = self.embedding_exists(follow_ups_content)
                    if existing_faiss_id is None:
                        emb = self.ltm.embeddings.encode([follow_ups_content], normalize=True)
                        faiss_ids = self.ltm.embeddings.add_embeddings(emb)
                        self.ltm.db.insert_faiss_mapping(faiss_ids[0], follow_ups_chunk_id)
                    else:
                        # Reuse existing embedding
                        self.ltm.db.insert_faiss_mapping(existing_faiss_id, follow_ups_chunk_id)
            
            # Chunk 8: Original Email Content (truncated for each email)
            if thread_emails:
                for i, email in enumerate(thread_emails):
                    email_details = []
                    email_details.append(f"Message {i}:")
                    email_details.append(f"  From: {email.get('sender', 'Unknown')}")
                    email_details.append(f"  Date: {email.get('date', 'Unknown')}")
                    body = email.get('body', '')
                    if body:
                        preview =self.extractor.clean_email_thread(body)
                        email_details.append(preview)
                
                    email_content = "\n".join(email_details)
                    email_chunk_id = self.ltm.db.insert_chunk(
                        memory_id, chunk_ordinal, 1, len(email_content.split('\n')),
                        email_content, 'email_original', thread_emails[0].get('subject', '')[:50], None
                    )
                    chunk_ordinal += 1
                    
                    # Add embedding for original emails (check for duplicates)
                    if self.ltm.embeddings is not None:
                        existing_faiss_id = self.embedding_exists(email_content)
                        if existing_faiss_id is None:
                            emb = self.ltm.embeddings.encode([email_content], normalize=True)
                            faiss_ids = self.ltm.embeddings.add_embeddings(emb)
                            self.ltm.db.insert_faiss_mapping(faiss_ids[0], email_chunk_id)
                        else:
                            # Reuse existing embedding
                            self.ltm.db.insert_faiss_mapping(existing_faiss_id, email_chunk_id)
            
            # Persist embeddings if available
            if self.ltm.embeddings is not None:
                self.ltm.embeddings.persist()
            
            return {
                "ok": True,
                "id": memory_id,
                "title": title,
                "category": category,
                "chunks_created": chunk_ordinal,
                "message": f"Email memory added successfully with {chunk_ordinal} chunks! (ID: {memory_id})"
            }
        except Exception as e:
            return {
                "ok": False,
                "error": str(e)
            }
    
    def get_email_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get an email memory by ID with all its chunks.
        
        Args:
            memory_id: The ID of the email memory
            
        Returns:
            Dictionary with memory details and all chunks, or None if not found
        """
        try:
            details = self.ltm.db.get_document_details(memory_id)
            if not details:
                return None
            
            doc = details["document"]
            chunks = details["chunks"]
            
            # Organize chunks by type for easier access
            chunks_by_type = {}
            for chunk in chunks:
                chunk_type = chunk.get("chunk_type", "unknown")
                if chunk_type not in chunks_by_type:
                    chunks_by_type[chunk_type] = []
                chunks_by_type[chunk_type].append(chunk)
            
            return {
                "ok": True,
                "id": memory_id,
                "title": doc.get("extra", {}).get("title", ""),
                "category": doc.get("category", "general"),
                "tags": doc.get("extra", {}).get("tags", []),
                "email_thread": doc.get("extra", {}).get("email_thread", {}),
                "chunks": chunks,
                "chunks_by_type": chunks_by_type,
                "created_at": doc.get("created_at", "")
            }
        except Exception as e:
            return {
                "ok": False,
                "error": str(e)
            }
    
    def search_souvenirs(self, query: str, category: Optional[str] = None, top_k: int = 15) -> List[Dict[str, Any]]:
        """Search for souvenirs matching the query."""
        try:
            results = self.ltm.db.search_souvenirs(query, category=category, limit=top_k)
            
            souvenirs = []
            for r in results:
                extra = json.loads(r.get('extra', '{}')) if r.get('extra') else {}
                souvenirs.append({
                    "id": r["id"],
                    "title": extra.get('title', 'Untitled'),
                    "content": r.get('content', ''),
                    "category": r.get('category', 'general'),
                    "tags": extra.get('tags', []),
                    "created_at": r.get('created_at', ''),
                    "similarity_score": r.get('similarity_score', 0)
                })
            return souvenirs
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def search_by_chunk_type(self, chunk_types: List[str], query: Optional[str] = None, 
                              top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for souvenirs by specific chunk types.
        
        Args:
            chunk_types: List of chunk types to search (e.g., ['email_action_items', 'email_key_points'])
            query: Optional text query to further filter results
            top_k: Maximum number of results
            
        Returns:
            List of souvenirs matching the criteria
        """
        try:
            with self.ltm.db._lock:
                cur = self.ltm.db.conn.cursor()
                
                placeholders = ','.join('?' * len(chunk_types))
                sql = f"""
                    SELECT d.id, d.source_path, d.category, d.created_at, d.extra, c.content, c.chunk_type, c.chunk_name
                    FROM documents d
                    JOIN chunks c ON d.id = c.document_id
                    WHERE c.chunk_type IN ({placeholders})
                """
                
                params = list(chunk_types)
                
                if query:
                    sql += " AND c.content LIKE ?"
                    params.append(f'%{query}%')
                
                sql += " ORDER BY d.created_at DESC LIMIT ?"
                params.append(top_k)
                
                cur.execute(sql, params)
                rows = cur.fetchall()
            
            # Group results by document
            doc_results = {}
            for row in rows:
                doc_id = row[0]
                if doc_id not in doc_results:
                    extra = json.loads(row[4] or '{}') if row[4] else {}
                    doc_results[doc_id] = {
                        "id": doc_id,
                        "title": extra.get('title', 'Untitled'),
                        "content": row[5],  # First chunk content
                        "category": row[2],
                        "tags": extra.get('tags', []),
                        "created_at": row[3],
                        "matching_chunks": []
                    }
                doc_results[doc_id]["matching_chunks"].append({
                    "content": row[5],
                    "chunk_type": row[6],
                    "chunk_name": row[7]
                })
            
            return list(doc_results.values())
        except Exception as e:
            print(f"Search by chunk type error: {e}")
            return []
    
    def search_semantic(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for souvenirs using semantic embeddings.
        
        Args:
            query: The search query
            top_k: Maximum number of results
            
        Returns:
            List of souvenirs with similarity scores
        """
        if self.ltm.embeddings is None:
            return []
        
        try:
            # Encode the query
            query_emb = self.ltm.embeddings.encode([query], normalize=True)
            
            # Search in FAISS index
            distances, indices = self.ltm.embeddings.search(query_emb, top_k)
            
            # Get chunk information
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0:
                    chunk_info = self.ltm.db.get_chunk_by_id(int(idx))
                    if chunk_info:
                        chunk_id, doc_id, start_line, end_line, chunk_type, chunk_name, content = chunk_info
                        doc_details = self.ltm.db.get_document_details(doc_id)
                        if doc_details:
                            extra = doc_details["document"].get("extra", {})
                            results.append({
                                "id": doc_id,
                                "title": extra.get("title", "Untitled"),
                                "content": content,
                                "category": doc_details["document"].get("category", "general"),
                                "tags": extra.get("tags", []),
                                "created_at": doc_details["document"].get("created_at", ""),
                                "chunk_type": chunk_type,
                                "similarity_score": float(dist)
                            })
            
            return results
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []
    
    def _analyze_question_for_search(self, question: str) -> Dict[str, Any]:
        """Analyze a question to determine the best search strategy.
        
        Uses LLM to understand what type of information is being requested
        and how to best search for it in the database.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary with search strategy and parameters
        """
        # First, check if we have embeddings for semantic search
        has_embeddings = self.ltm.embeddings is not None and self.ltm.embeddings.ntotal > 0
        
        # Define chunk types that can be searched
        chunk_type_info = {
            "email_topic": "Main topic/theme of email conversations",
            "email_participants": "People involved in email conversations", 
            "email_key_points": "Main points discussed in emails",
            "email_action_items": "Tasks, todos, and action items from emails",
            "email_dates": "Important dates and deadlines mentioned in emails",
            "email_entities": "People, organizations, locations from emails",
            "email_follow_ups": "Items needing follow-up from emails",
            "email_original": "Original email content"
        }
        
        chunk_types_json = json.dumps(chunk_type_info, indent=2)
        
        try:
            prompt = f"""Analyze this question to determine the best way to search a personal memory database.

Question: {question}

The database has the following chunk types (each document is split into multiple chunks):
{chunk_types_json}

Return a JSON object with:
{{
    "search_strategy": "semantic" or "keyword" or "chunk_type" or "hybrid",
    "chunk_types": ["list of relevant chunk types if chunk_type or hybrid strategy"],
    "keywords": ["important keywords for search"],
    "reasoning": "brief explanation of why this strategy was chosen"
}}

Choose "chunk_type" or "hybrid" if the question specifically asks about:
- action items, tasks, todos -> email_action_items
- dates, deadlines, when -> email_dates  
- participants, who -> email_participants
- key points, main points -> email_key_points
- topic, about -> email_topic
- follow-ups -> email_follow_ups
- entities, people, organizations -> email_entities

Choose "semantic" if the question is conversational/natural language and embeddings are available ({has_embeddings}).
Choose "keyword" for simple factual queries or if no embeddings."""

            response = litellm.completion(
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": "You are a search strategy assistant. Analyze questions to determine optimal database search approaches. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            strategy = json.loads(result)
            
            # Ensure required fields exist
            return {
                "search_strategy": strategy.get("search_strategy", "hybrid"),
                "chunk_types": strategy.get("chunk_types", []),
                "keywords": strategy.get("keywords", []),
                "reasoning": strategy.get("reasoning", ""),
                "has_embeddings": has_embeddings
            }
            
        except Exception as e:
            print(f"Question analysis error: {e}")
            # Fallback to simple keyword extraction
            keywords = question.lower().split()
            return {
                "search_strategy": "keyword" if not has_embeddings else "semantic",
                "chunk_types": [],
                "keywords": keywords,
                "reasoning": "Fallback due to error",
                "has_embeddings": has_embeddings
            }
    
    def ask_about_souvenirs(self, question: str) -> Dict[str, Any]:
        """Ask a question about souvenirs using intelligent search based on question analysis."""
        # Analyze the question to determine best search strategy
        search_strategy = self._analyze_question_for_search(question)
        
        all_results = []
        
        # Execute search based on strategy
        if search_strategy["search_strategy"] in ["semantic", "hybrid"]:
            semantic_results = self.search_semantic(question, top_k=5)
            all_results.extend(semantic_results)
        
        if search_strategy["search_strategy"] in ["keyword", "hybrid"]:
            keywords = search_strategy.get("keywords", [])
            if keywords:
                keyword_query = " ".join(keywords[:5])
                keyword_results = self.search_souvenirs(keyword_query, top_k=15)
                all_results.extend(keyword_results)
        
        if search_strategy["search_strategy"] == "chunk_type":
            chunk_types = search_strategy.get("chunk_types", [])
            if chunk_types:
                # Combine keywords if any
                query = " ".join(search_strategy.get("keywords", []))
                chunk_results = self.search_by_chunk_type(chunk_types, query=query if query else None, top_k=15)
                all_results.extend(chunk_results)
        
        # Deduplicate results by document ID
        seen_ids = set()
        unique_results = []
        for r in all_results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                unique_results.append(r)
        
        # Re-rank by similarity score if available
        if any("similarity_score" in r for r in unique_results):
            unique_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        souvenirs = unique_results[:5]
        
        if not souvenirs:
            return {
                "ok": True,
                "answer": "I don't have any souvenirs that match your question. Try adding some first!",
                "sources": []
            }
        
        # Build context with information about data structure
        context = self._format_souvenirs_for_llm_enhanced(souvenirs, search_strategy)
        
        # Create prompt for LLM with database structure info
        prompt = self._build_question_prompt_enhanced(question, context, search_strategy)
        
        try:
            response = litellm.completion(
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that helps users remember their past experiences. Answer the user's question based on the provided souvenirs/memories. If you can't find relevant information, say so."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            return {
                "ok": True,
                "answer": answer,
                "sources": [s.get("id", "unknown") for s in souvenirs]
            }
        except Exception as e:
            return {
                "ok": False,
                "error": str(e)
            }
    
    def _format_souvenirs_for_llm_enhanced(self, souvenirs: List[Dict[str, Any]], 
                                           search_strategy: Dict[str, Any]) -> str:
        """Format souvenirs for inclusion in LLM prompt with chunk type info."""
        chunk_type_labels = {
            "email_topic": "Topic",
            "email_participants": "Participants",
            "email_key_points": "Key Points",
            "email_action_items": "Action Items",
            "email_dates": "Important Dates",
            "email_entities": "Entities",
            "email_follow_ups": "Follow-ups",
            "email_original": "Original Content"
        }
        
        formatted = []
        for i, s in enumerate(souvenirs, 1):
            created = s.get('created_at', 'Unknown date')
            title = s.get('title', 'Untitled')
            category = s.get('category', 'general')
            chunk_type = s.get('chunk_type', 'unknown')
            chunk_label = chunk_type_labels.get(chunk_type, chunk_type)
            
            formatted.append(f"""
--- Memory {i} ---
Title: {title}
Category: {category}
Type: {chunk_label}
Date: {created}
Content:
{s['content']}
""")
            
            # Include matching chunks if available
            if "matching_chunks" in s and s["matching_chunks"]:
                formatted.append("Additional matching sections:")
                for mc in s["matching_chunks"][:3]:  # Limit to 3 additional chunks
                    mc_type = mc.get("chunk_type", "unknown")
                    mc_label = chunk_type_labels.get(mc_type, mc_type)
                    formatted.append(f"  [{mc_label}]: {mc.get('content', '')[:200]}")
        
        return "\n".join(formatted)
    
    def _build_question_prompt_enhanced(self, question: str, context: str, 
                                       search_strategy: Dict[str, Any]) -> str:
        """Build enhanced prompt with database structure information."""
        chunk_type_info = """
The memories are stored with different types of information:
- Topic: Main theme/subject of the memory
- Participants: People involved
- Key Points: Main discussion points
- Action Items: Tasks and todos
- Important Dates: Deadlines and dates mentioned
- Entities: People, organizations, locations mentioned
- Follow-ups: Items needing follow-up
- Original Content: Raw content of the memory
"""
        
        reasoning = search_strategy.get("reasoning", "")
        strategy = search_strategy.get("search_strategy", "hybrid")
        
        return f"""Based on the following souvenirs/memories, please answer my question.

The search was performed using: {strategy} strategy
Reasoning: {reasoning}

{chunk_type_info}

Memories:
{context}

My question: {question}

Please provide a detailed answer based on the memories above. If the question asks about specific types of information (like action items, dates, participants), prioritize those sections in your answer:"""
    
    def list_souvenirs(self, category: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """List all stored souvenirs, optionally filtered by category."""
        try:
            results = self.ltm.db.list_souvenirs(category=category, limit=limit)
            
            souvenirs = []
            for r in results:
                extra = json.loads(r.get('extra', '{}')) if r.get('extra') else {}
                souvenirs.append({
                    "id": r["id"],
                    "title": extra.get('title', 'Untitled'),
                    "content": r.get('content', ''),
                    "category": r.get('category', 'general'),
                    "tags": extra.get('tags', []),
                    "created_at": r.get('created_at', ''),
                })
            return souvenirs
        except Exception as e:
            print(f"Error listing souvenirs: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            stats = self.ltm.stats()
            # Get category counts
            categories = self.ltm.db.list_categories()
            
            return {
                "documents": stats.get("documents", 0),
                "chunks": stats.get("chunks", 0),
                "db_path": self.db_path,
                "categories": categories,
                "ltm_available": True
            }
        except Exception as e:
            return {"error": str(e)}
    
    def list_categories(self) -> List[Dict[str, Any]]:
        """List all available categories."""
        try:
            return self.ltm.db.list_categories()
        except Exception as e:
            print(f"Error listing categories: {e}")
            return []
    
    def create_category(self, name: str, description: str = '', color: str = '#6B7280', icon: str = '📂') -> Dict[str, Any]:
        """Create a new category."""
        try:
            cat_id = self.ltm.db.create_category(name, description, color, icon)
            return {"ok": True, "id": cat_id, "message": f"Category '{name}' created"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    
    def _format_souvenirs_for_llm(self, souvenirs: List[Dict[str, Any]]) -> str:
        """Format souvenirs for inclusion in LLM prompt."""
        formatted = []
        for i, s in enumerate(souvenirs, 1):
            created = s.get('created_at', 'Unknown date')
            title = s.get('title', 'Untitled')
            category = s.get('category', 'general')
            formatted.append(f"\n--- Souvenir {i} ({title}) - Category: {category} - {created} ---\n{s['content']}\n")
        return "\n".join(formatted)
    
    def _build_question_prompt(self, question: str, context: str) -> str:
        """Build the prompt for asking a question."""
        return f"""Based on the following souvenirs/memories, please answer my question.

Souvenirs:
{context}

My question: {question}

Please provide a detailed answer based on the souvenirs above:"""
    
    def _extract_keywords_from_question(self, question: str) -> str:
        """Extract key terms from a question for searching using LLM.
        
        This method uses an LLM to intelligently extract meaningful keywords
        from the question, supporting multiple languages and semantic understanding
        rather than simple pattern matching.
        
        Args:
            question: The user's question
            
        Returns:
            Space-separated string of extracted keywords for search
        """
        try:
            prompt = f"""Extract the most important keywords from the following question for searching a personal memory database.

Question: {question}

Return a JSON object with a "keywords" field containing an array of 3-7 important keywords or key phrases that would help find relevant memories. 
- Include nouns, verbs, and important concepts
- Skip very common question words (what, who, where, when, why, how)
- Keep keywords in the same language as the question
- Use singular forms when possible

Example output format:
{{"keywords": ["meeting", "project deadline", "team celebration"]}}"""

            response = litellm.completion(
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": "You are a keyword extraction assistant. Extract meaningful search keywords from questions. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            
            # Parse JSON response
            try:
                extracted = json.loads(result)
                keywords = extracted.get("keywords", [])
                if keywords:
                    # Join keywords with spaces, multi-word phrases with underscores
                    formatted = []
                    for kw in keywords:
                        # Convert spaces to underscores for multi-word key phrases
                        formatted.append(kw.replace(" ", "_"))
                    return " ".join(formatted[:7])
            except json.JSONDecodeError:
                # Fallback: try to extract JSON from the response
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    extracted = json.loads(json_match.group())
                    keywords = extracted.get("keywords", [])
                    if keywords:
                        formatted = [kw.replace(" ", "_") for kw in keywords[:7]]
                        return " ".join(formatted)
        
        except Exception as e:
            # Log the error but don't crash - fall back to simple extraction
            print(f"Warning: LLM keyword extraction failed: {e}")
        
        # Fallback to simple rule-based extraction if LLM fails
        return self._fallback_keyword_extraction(question)
    
    def _fallback_keyword_extraction(self, question: str) -> str:
        """Fallback rule-based keyword extraction for when LLM is unavailable."""
        import re
        # Remove question words, keep more content words (reduced stop words)
        stop_words = r'\b(what|who|where|when|why|how)\b'
        cleaned = re.sub(stop_words, '', question.lower())
        
        # Extract meaningful words (alphanumeric, length > 2)
        keywords = re.findall(r'\b[a-z0-9]{3,}\b', cleaned)
        
        # If no keywords, fall back to the original question
        if not keywords:
            keywords = re.findall(r'\b[a-z0-9]{3,}\b', question.lower())
        
        # Simple stemming for common suffixes
        stemmed = []
        for kw in keywords:
            # Remove common suffixes to improve matching
            if kw.endswith('ies'):
                stemmed.append(kw[:-3] + 'y')  # meetings → meeting
            elif kw.endswith('ed') and len(kw) > 4:
                stemmed.append(kw[:-2])  # discussed → discuss
            elif kw.endswith('ing') and len(kw) > 5:
                stemmed.append(kw[:-3])  # discussing → discuss
            elif kw.endswith('s') and len(kw) > 3 and kw not in ['this', 'thus', 'yes', 'class']:
                stemmed.append(kw[:-1])  # trips → trip
            else:
                stemmed.append(kw)
        
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for kw in stemmed:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)
        
        return ' '.join(unique[:5])
    
    def close(self):
        """Close the memory storage."""
        if self.ltm:
            self.ltm.close()


def interactive_mode(assistant: SouvenirAssistant):
    """Run the assistant in interactive mode."""
    print("=" * 60)
    print("🎒 Souvenir Assistant - Interactive Mode")
    print("=" * 60)
    print("Commands:")
    print("  add <text>   - Add a new souvenir")
    print("  ask <question> - Ask about your souvenirs")
    print("  search <query> - Search souvenirs")
    print("  list         - List all souvenirs")
    print("  stats        - Show storage statistics")
    print("  quit         - Exit the program")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n🎒 > ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye! 👋")
                break
            
            # Parse command
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if command == "add":
                if not args:
                    print("Please provide content to add: add <text>")
                    continue
                result = assistant.add_souvenir(args)
                print(f"✅ {result.get('message', result.get('error', 'Done'))}")
            
            elif command == "ask":
                if not args:
                    print("Please provide a question: ask <question>")
                    continue
                result = assistant.ask_about_souvenirs(args)
                if result.get("ok"):
                    print(f"\n💬 Answer:\n{result['answer']}")
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
            elif command == "search":
                if not args:
                    print("Please provide a search query: search <query>")
                    continue
                results = assistant.search_souvenirs(args)
                if results:
                    print(f"\n🔍 Found {len(results)} souvenirs:\n")
                    for i, r in enumerate(results, 1):
                        print(f"{i}. {r['content'][:150]}...")
                        print(f"   📅 {r['created_at']}\n")
                else:
                    print("No souvenirs found matching your query.")
            
            elif command == "list":
                souvenirs = assistant.list_souvenirs()
                if souvenirs:
                    print(f"\n📋 Found {len(souvenirs)} souvenirs:\n")
                    for i, s in enumerate(souvenirs, 1):
                        print(f"{i}. {s['content'][:100]}...")
                else:
                    print("No souvenirs stored yet.")
            
            elif command == "stats":
                stats = assistant.get_stats()
                print(f"\n📊 Storage Statistics:")
                print(f"   Documents: {stats.get('documents', 'N/A')}")
                print(f"   Chunks: {stats.get('chunks', 'N/A')}")
                print(f"   Database: {stats['db_path']}")
                print(f"\n📁 Categories:")
                for cat in stats.get('categories', []):
                    icon = cat.get('icon', '📂')
                    name = cat['name']
                    desc = cat.get('description', '')
                    system = '(system)' if cat.get('is_system') else ''
                    print(f"   {icon} {name} {system} - {desc}")
            
            elif command == "categories":
                categories = assistant.list_categories()
                print(f"\n📁 Available Categories:")
                for cat in categories:
                    icon = cat.get('icon', '📂')
                    name = cat['name']
                    desc = cat.get('description', '')
                    system = '(system)' if cat.get('is_system') else ''
                    print(f"   {icon} {name} {system} - {desc}")
            
            elif command == "help":
                print("Available commands: add, ask, search, list, stats, categories, quit")
            
            else:
                print(f"Unknown command: {command}")
                print("Use 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point for the souvenir assistant."""
    parser = argparse.ArgumentParser(
        description="🎒 Souvenir - Your personal memory assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python souvenir_assistant.py add "Today I learned to cook pasta al dente"
  python souvenir_assistant.py ask "What did I learn about cooking?"
  python souvenir_assistant.py search "cooking"
  python souvenir_assistant.py list
  python souvenir_assistant.py interactive
        """
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        choices=["add", "ask", "search", "list", "stats", "categories", "interactive"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "text",
        nargs="?",
        help="Text content or query for the command"
    )
    
    parser.add_argument(
        "--title", "-t",
        help="Title for added souvenirs"
    )
    
    parser.add_argument(
        "--category", "-c",
        default="general",
        help="Category for added souvenirs"
    )
    
    parser.add_argument(
        "--tags",
        nargs="*",
        help="Tags for added souvenirs"
    )
    
    parser.add_argument(
        "--db",
        default="souvenir_memory.sqlite",
        help="Path to the SQLite database"
    )
    
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Enable semantic embeddings for search"
    )
    
    args = parser.parse_args()
    
    # Initialize assistant
    assistant = SouvenirAssistant(db_path=args.db, enable_embeddings=args.embeddings)
    
    try:
        if args.command == "interactive":
            interactive_mode(assistant)
        
        elif args.command == "add":
            if not args.text:
                print("Error: Please provide content to add")
                print("Usage: python souvenir_assistant.py add \"Your memory here\"")
                sys.exit(1)
            
            result = assistant.add_souvenir(args.text, title=args.title, category=args.category, tags=args.tags)
            if result.get("ok"):
                print(f"✅ {result['message']}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        
        elif args.command == "ask":
            if not args.text:
                print("Error: Please provide a question")
                print("Usage: python souvenir_assistant.py ask \"Your question\"")
                sys.exit(1)
            
            result = assistant.ask_about_souvenirs(args.text)
            if result.get("ok"):
                print(f"\n💬 Answer:\n{result['answer']}")
                if result.get("sources"):
                    print(f"\n📚 Sources: {', '.join(result['sources'])}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        
        elif args.command == "search":
            if not args.text:
                print("Error: Please provide a search query")
                print("Usage: python souvenir_assistant.py search \"query\"")
                sys.exit(1)
            
            results = assistant.search_souvenirs(args.text)
            if results:
                print(f"\n🔍 Found {len(results)} souvenirs:\n")
                for i, r in enumerate(results, 1):
                    print(f"{i}. {r['content'][:200]}...")
                    print(f"   📅 {r['created_at']}")
                    print(f"   🏷️  Tags: {', '.join(r.get('tags', [])) or 'None'}\n")
            else:
                print("No souvenirs found.")
        
        elif args.command == "list":
            souvenirs = assistant.list_souvenirs()
            if souvenirs:
                print(f"\n📋 Found {len(souvenirs)} souvenirs:\n")
                for i, s in enumerate(souvenirs, 1):
                    print(f"{i}. {s['content']}")
            else:
                print("No souvenirs stored yet.")
        
        elif args.command == "stats":
            stats = assistant.get_stats()
            print(f"\n📊 Storage Statistics:")
            print(f"   Documents: {stats.get('documents', 'N/A')}")
            print(f"   Chunks: {stats.get('chunks', 'N/A')}")
            print(f"   Database: {stats['db_path']}")
            print(f"\n📁 Categories:")
            for cat in stats.get('categories', []):
                icon = cat.get('icon', '📂')
                name = cat['name']
                desc = cat.get('description', '')
                system = '(system)' if cat.get('is_system') else ''
                print(f"   {icon} {name} {system} - {desc}")
        
        elif args.command == "categories":
            categories = assistant.list_categories()
            print(f"\n📁 Available Categories:")
            for cat in categories:
                icon = cat.get('icon', '📂')
                name = cat['name']
                desc = cat.get('description', '')
                system = '(system)' if cat.get('is_system') else ''
                print(f"   {icon} {name} {system} - {desc}")
        
        else:
            parser.print_help()
    
    finally:
        assistant.close()


if __name__ == "__main__":
    main()
