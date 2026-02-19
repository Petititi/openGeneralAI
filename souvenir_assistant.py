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
        if not content or not content.strip():
            return {"ok": False, "error": "Empty content"}
        
        # Build extraction prompt based on type
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = self._build_extraction_prompt(content, extraction_type)
        
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
        tags.extend(data.get("key_points", [])[:5])
        
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
            "tags": list(set(tags))[:15]
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
            old_messages = False
            for paragraph in paragraphs:
                p_hash = self._hash_paragraph(paragraph)

                if p_hash not in seen_paragraph_hashes:
                    seen_paragraph_hashes.add(p_hash)
                    new_paragraphs.append(paragraph)
                else:
                    old_messages = True
                    break


            if not new_paragraphs:
                continue

            lines.append(f"--- Message {i+1} ---")

            if msg.get("sender"):
                lines.append(f"From: {msg['sender']}")

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
    
    def search_souvenirs(self, query: str, category: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
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
                })
            return souvenirs
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def ask_about_souvenirs(self, question: str) -> Dict[str, Any]:
        """Ask a question about souvenirs using LLM with context from memory."""
        # Extract keywords from the question for better search
        search_query = self._extract_keywords_from_question(question)
        
        # Get relevant souvenirs as context
        souvenirs = self.search_souvenirs(search_query, top_k=5)
        
        if not souvenirs:
            return {
                "ok": True,
                "answer": "I don't have any souvenirs that match your question. Try adding some first!",
                "sources": []
            }
        
        # Build context from souvenirs
        context = self._format_souvenirs_for_llm(souvenirs)
        
        # Create prompt for LLM
        prompt = self._build_question_prompt(question, context)
        
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
