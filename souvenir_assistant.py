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
from typing import Optional, List, Dict, Any, Set
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import configurator
import litellm
from storage.longterm_memory import LongTermMemory
from storage.DatabaseManagement import DatabaseManager
from storage.llm_extractor import LLMExtractor  # Shared LLM extractor
from typing import Optional, List, Dict, Any, Callable
import re
import json



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


class EntityContextBuilder:
    """Builds entity context from email content for richer memory storage.
    
    This class handles:
    - Finding related entities in the database before storing emails
    - Building context from previous related entity memories
    - Creating or updating entity documents
    - Cross-referencing emails with entities
    """
    
    def __init__(self, souvenir_assistant):
        """Initialize with a reference to SouvenirAssistant.
        
        Args:
            souvenir_assistant: Instance of SouvenirAssistant for database operations
        """
        self.souvenir_assistant = souvenir_assistant
        self.ltm = souvenir_assistant.ltm
    
    def find_related_entities(
        self,
        email_content: str,
        participants: List[str],
        entity_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Find entities related to an email based on content and participants.
        
        Args:
            email_content: The email body/content to search for entity mentions
            participants: List of email participants (senders, recipients)
            entity_types: Optional list of entity types to filter (person/org/project/service)
            
        Returns:
            List of related entity documents with their details
        """
        if entity_types is None:
            entity_types = ['person', 'organization', 'project', 'service']
        
        related_entities = []
        
        # Search for people by email or name
        for participant in participants:
            # Try to find person by email address
            if '@' in participant:
                # Search for existing person entity with this email
                person_results = self._find_entity_by_value('person', 'email', participant)
                related_entities.extend(person_results)
            
            # Also try to find by name (extract name from email or use full string)
            name = participant.split('@')[0].replace('.', ' ').replace('_', ' ')
            if name and len(name) > 2:
                person_results = self._find_entity_by_value('person', 'name', name)
                related_entities.extend(person_results)
        
        # Search for organizations mentioned in content
        org_results = self._search_entities_in_content(
            email_content, 
            ['organization', 'company', 'inc', 'llc', 'corp']
        )
        related_entities.extend(org_results)
        
        # Search for projects/services mentioned in content
        project_service_results = self._search_entities_in_content(
            email_content,
            ['project', 'service', 'product', 'platform']
        )
        related_entities.extend(project_service_results)
        
        # Deduplicate by entity ID
        seen_ids = set()
        unique_entities = []
        for e in related_entities:
            if e.get('id') not in seen_ids:
                seen_ids.add(e.get('id'))
                unique_entities.append(e)
        
        return unique_entities
    
    def _find_entity_by_value(self, entity_type: str, field: str, value: str) -> List[Dict[str, Any]]:
        """Find entity by a specific field value.
        
        Args:
            entity_type: Type of entity (person, organization, etc.)
            field: Field to search (email, name, etc.)
            value: Value to search for
            
        Returns:
            List of matching entity documents
        """
        if not value or len(value) < 3:
            return []
        
        try:
            # Search using FTS for entities with matching field
            results = self.ltm.search_souvenirs(value, category='entity', limit=10)
            
            entities = []
            for r in results:
                doc_id = r.get('document_id')
                if doc_id:
                    details = self.ltm.get_memory(doc_id)
                    if details:
                        doc = details.get('document', {})
                        # Check if this is the right entity type
                        if doc.get('media_type') == entity_type:
                            extra = doc.get('extra', {})
                            entity_data = extra.get('entity_data', {})
                            
                            # Check if the field matches
                            if field == 'email':
                                entity_email = entity_data.get('email', '').lower()
                                if entity_email and value.lower() in entity_email:
                                    entities.append({
                                        'id': doc_id,
                                        'title': extra.get('title', ''),
                                        'entity_type': entity_type,
                                        'entity_data': entity_data,
                                        'linked_memories': extra.get('linked_memories', [])
                                    })
                            elif field == 'name':
                                entity_name = entity_data.get('name', '').lower()
                                if entity_name and value.lower() in entity_name:
                                    entities.append({
                                        'id': doc_id,
                                        'title': extra.get('title', ''),
                                        'entity_type': entity_type,
                                        'entity_data': entity_data,
                                        'linked_memories': extra.get('linked_memories', [])
                                    })
            
            return entities
        except Exception as e:
            print(f"Error finding entity by {field}: {e}")
            return []
    
    def _search_entities_in_content(self, content: str, keywords: List[str]) -> List[Dict[str, Any]]:
        """Search for entities mentioned in content based on keywords.
        
        Args:
            content: Text content to search
            keywords: Keywords to look for (entity types, company suffixes, etc.)
            
        Returns:
            List of potential entity matches
        """
        content_lower = content.lower()
        
        # Check if any keywords are in content
        found_keywords = [kw for kw in keywords if kw.lower() in content_lower]
        if not found_keywords:
            return []
        
        # Search for entities in database
        entities = []
        try:
            # Search for any entity documents
            results = self.ltm.search_souvenirs(content[:500], category='entity', limit=20)
            
            for r in results:
                doc_id = r.get('document_id')
                if doc_id:
                    details = self.ltm.get_memory(doc_id)
                    if details:
                        doc = details.get('document', {})
                        extra = doc.get('extra', {})
                        entity_data = extra.get('entity_data', {})
                        
                        entities.append({
                            'id': doc_id,
                            'title': extra.get('title', ''),
                            'entity_type': doc.get('media_type', ''),
                            'entity_data': entity_data,
                            'linked_memories': extra.get('linked_memories', [])
                        })
        except Exception as e:
            print(f"Error searching entities in content: {e}")
        
        return entities
    
    def build_entity_context(self, related_entities: List[Dict[str, Any]], max_mem: int = 3) -> str:
        """Build a context string from related entity documents.
        
        Args:
            related_entities: List of entity documents to build context from
            max_memories_per_entity: Maximum number of memory contexts to include per entity
            
        Returns:
            Context string summarizing related entities and their memories
        """
        if not related_entities:
            return ""
        
        context_parts = []
        
        for entity in related_entities:
            entity_title = entity.get('title', 'Unknown')
            entity_type = entity.get('entity_type', 'person')
            
            souvenirs = self.souvenir_assistant.get_souvenirs_by_title(title=entity_title, category=entity_type, limit=max_mem)
            if souvenirs:
                context_parts.append(f"--- Related {entity_type}: {entity_title} ---")
                for souvenir in souvenirs:
                    context_parts.append(souvenir["description"])
        
        return "\n".join(context_parts)
    
    def update_or_create_entity(
        self,
        entity_type: str,
        entity_name: str,
        entity_data: Dict[str, Any],
        new_content: str,
        linked_memory_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Find existing entity or create a new one.
        
        Args:
            entity_type: Type of entity (person, organization, project, service)
            entity_name: Name of the entity
            entity_data: Entity data dictionary
            linked_memory_id: Optional memory ID to link to this entity
            
        Returns:
            Dictionary with entity information (existing or newly created)
        """
        # convert entity_data to string:
        entity_data_str = json.dumps(entity_data)

        # ask an llm to improve the entity data with the available context:
        new_datas=self.souvenir_assistant.extractor.extract(entity_data_str,"entity_info",additional_info=new_content)
        if new_datas.get("ok"):
            data = new_datas.get("data", {})
            if data.get("validity") == "false":
                return {"ok": False, "error": "Entity data not valid or useful according to LLM", "description": data.get("description")}
            elif data.get("completeness") == "false" and data.get("description"):
                entity_data["souvenirs"] = data.get("description")

                result = self.souvenir_assistant.add_entity_memory(
                    entity_type=entity_type,
                    entity_name=entity_name,
                    entity_data=entity_data,
                    allow_duplicates=False,  # This will find existing or create new
                    linked_memory_id=linked_memory_id
                )
        
        return result


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
        self.extractor = LLMExtractor(self.cfg.model)
        
        # Initialize thread builder for conversation reconstruction
        self.thread_builder = ThreadBuilder()
        
        # Initialize entity context builder for entity memory management
        self.entity_context_builder = EntityContextBuilder(self)
    
    def get_stored_thread_ids(self) -> Set[str]:
        """Get all thread_ids already stored in the database.
        
        Queries all email-related documents and extracts their thread_ids
        from the extra.email_thread metadata for deduplication purposes.
        
        Returns:
            Set of thread_id strings already stored in the database
        """
        stored_thread_ids = set()
        
        try:
            # Get all documents from the database
            all_docs = self.ltm.db_manager.get_all_documents()
            
            for doc in all_docs:
                doc_id = doc[0]  # First element is the document ID
                
                # Get document details including extra metadata
                doc_details = self.ltm.db_manager.get_document_details(doc_id)
                
                if doc_details and doc_details.get("document"):
                    extra = doc_details["document"].get("extra", {})
                    email_thread = extra.get("email_thread", {})
                    
                    # Extract thread_id if present
                    thread_id = email_thread.get("thread_id")
                    if thread_id:
                        stored_thread_ids.add(thread_id)
                    else:
                        stored_thread_ids.add(doc_id)
            
            return stored_thread_ids
            
        except Exception as e:
            print(f"Warning: Failed to get stored thread_ids: {e}")
            return set()
    
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
    
    def process_conversation_for_memory(self, messages: List[Dict], context: str) -> Dict[str, Any]:
        """Process a conversation/thread and extract meaningful information using LLM.
        
        Args:
            messages: List of message dictionaries in chronological order
            context: optional informations about participants
            
        Returns:
            Dictionary with extracted info: topic, participants, key_points, etc.
        """
        if not messages:
            return {"ok": False, "error": "No messages provided"}
        
        # Build conversation content
        conversation_content = self._format_conversation_for_extraction(messages, context)
        
        # Use LLM to extract information
        result = self.extractor.extract(conversation_content, extraction_type="general")
        
        if not result.get("ok"):
            return {
                "ok": False,
                "error": result.get("error", "Extraction failed"),
            }
        
        data = result.get("data", {})
        
        return {
            "ok": True,
            "summary": data.get("summary", ""),
            "entities": data.get("entities", {'people':[], 'organizations':[], 'locations':[]}),
            "key_points": data.get("key_points", []),
            "important_dates": data.get("important_dates", []),
            "action_items": data.get("action_items", []),
            "sentiment": data.get("sentiment", "neutral"),
            "follow_ups": data.get("follow_ups", []),
            "urgency": data.get("urgency", ""),
            "long_term_impact": data.get("long_term_impact", ""),
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
        response = EmailReplyParser.parse_reply(body)
        # remove from response all css/html tags:
        # Handles edge cases where ">" might appear inside quoted attributes
        response = re.sub(r'<[^>]*(?:"[^"]*"[^>]*)*(?:\'[^\']*\'[^>]*)*>', '', response)
        return response

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

    def _format_conversation_for_extraction(self, messages: List[Dict], context: str) -> str:
        """
        Format conversation for LLM extraction.
        Deduplicate content at paragraph level instead of message level.
        """
        lines = []
        if context:
            lines.append(context)
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
            self.ltm.insert_souvenir(
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
        
        Delegates to LongTermMemory.find_similar_documents()
        """
        return self.ltm.find_similar_documents(
            query_text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
    
    def embedding_exists(
        self,
        text: str,
        similarity_threshold: float = 0.9999
    ) -> Optional[int]:
        """Check if an embedding for the given text already exists.
        
        Delegates to LongTermMemory.embedding_exists()
        """
        return self.ltm.embedding_exists(
            text=text,
            similarity_threshold=similarity_threshold
        )
    
    def add_email_memory(
        self,
        thread_emails: List[Dict],
        conversation_info: Dict,
        category: str = 'email_conversation',
        tags: Optional[List[str]] = None,
        allow_duplicates: bool = False,
        is_single_email: bool = False,
        thread_id: Optional[str] = None
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
            is_single_email: If True, use lower threshold (0.70) for single emails
            thread_id: Optional Gmail thread ID for deduplication tracking
            
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
            
            # Determine duplicate threshold - lower for single emails (0.70) to allow more storage
            # For threads: 0.75, for singles: 0.70
            duplicate_threshold = 0.70 if is_single_email else 0.75
            
            # Check for similar existing documents if duplicates not allowed
            # Lowered threshold from 0.90 to allow storing more variants
            if not allow_duplicates and self.ltm.embeddings is not None and content_for_check.strip():
                similar_docs = self.find_similar_documents(content_for_check, top_k=3, similarity_threshold=duplicate_threshold)
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
            
            # Step 9: Create content hash for improved deduplication
            # Use full email content (body, sender, date) for more accurate deduplication
            full_content_parts = []
            for email in thread_emails:
                full_content_parts.append(email.get('body', ''))
                full_content_parts.append(email.get('from', ''))
                full_content_parts.append(email.get('date', ''))
            content_hash = hashlib.sha256(' '.join(full_content_parts).encode()).hexdigest()[:16]
            
            # Build title from topic or subject
            if conversation_info.get("topic"):
                title = f"📧 {conversation_info['topic']}"
            else:
                subject = thread_emails[0].get('subject', 'Email Conversation')
                title = f"📧 Thread: {subject[:45]}{'...' if len(subject) > 45 else ''}"
            
            # Prepare extra metadata - include content hash for deduplication tracking
            # Also include thread_id if provided for future deduplication
            email_thread_info = {
                'subject': thread_emails[0].get('subject', ''),
                'message_count': len(thread_emails),
                'participants': conversation_info.get('participants', []),
                'content_hash': content_hash  # Store for deduplication reference
            }
            if thread_id:
                email_thread_info['thread_id'] = thread_id
            
            extra = {
                'title': title,
                'tags': tags or [],
                'email_thread': email_thread_info
            }
            
            # Insert the document
            content_preview = conversation_info.get('topic', '')[:500]
            self.ltm.db_manager.insert_document(
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
                topic_chunk_id = self.ltm.add_memory_chunk(
                    memory_id, chunk_ordinal, 1, len(topic_content.split('\n')),
                    topic_content, 'email_topic', conversation_info['topic'][:50], None
                )
                chunk_ordinal += 1
            
            # Chunk 2: Participants (no embeddings - contains email addresses)
            participants = conversation_info.get("participants", [])
            if participants:
                participants_content = "\n".join(f"  - {p}" for p in participants)
                participants_chunk_id = self.ltm.add_memory_chunk(
                    memory_id, chunk_ordinal, 1, len(participants_content.split('\n')),
                    participants_content, 'email_participants', None, None, embed=False
                )
                chunk_ordinal += 1
            
            # Chunk 3: Key Points
            key_points = conversation_info.get("key_points", [])
            if key_points:
                key_points_content = "Key Points:\n" + "\n".join(f"  - {point}" for point in key_points)
                key_points_chunk_id = self.ltm.add_memory_chunk(
                    memory_id, chunk_ordinal, 1, len(key_points_content.split('\n')),
                    key_points_content, 'email_key_points', None, None
                )
                chunk_ordinal += 1
            
            # Chunk 4: Action Items
            action_items = conversation_info.get("action_items", [])
            if action_items:
                action_items_content = "Action Items:\n" + "\n".join(f"  - {item}" for item in action_items)
                action_items_chunk_id = self.ltm.add_memory_chunk(
                    memory_id, chunk_ordinal, 1, len(action_items_content.split('\n')),
                    action_items_content, 'email_action_items', None, None
                )
                chunk_ordinal += 1
            
            # Chunk 5: Important Dates
            important_dates = conversation_info.get("important_dates", [])
            if important_dates:
                dates_content = "Important Dates:\n" + "\n".join(f"  - {date}" for date in important_dates)
                dates_chunk_id = self.ltm.add_memory_chunk(
                    memory_id, chunk_ordinal, 1, len(dates_content.split('\n')),
                    dates_content, 'email_dates', None, None, embed=False
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
                    entities_chunk_id = self.ltm.add_memory_chunk(
                        memory_id, chunk_ordinal, 1, len(entities_content.split('\n')),
                        entities_content, 'email_entities', None, None, embed=False
                    )
                    chunk_ordinal += 1
            
            # Chunk 7: Follow-ups
            follow_ups = conversation_info.get("follow_ups", [])
            if follow_ups:
                follow_ups_content = "Follow-ups:\n" + "\n".join(f"  - {follow_up}" for follow_up in follow_ups)
                follow_ups_chunk_id = self.ltm.add_memory_chunk(
                    memory_id, chunk_ordinal, 1, len(follow_ups_content.split('\n')),
                    follow_ups_content, 'email_follow_ups', None, None
                )
                chunk_ordinal += 1
            
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
                    email_chunk_id = self.ltm.add_memory_chunk(
                        memory_id, chunk_ordinal, 1, len(email_content.split('\n')),
                        email_content, 'email_original', thread_emails[0].get('subject', '')[:50], None
                    )
                    chunk_ordinal += 1
            
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

    def add_entity_memory(
        self,
        entity_type: str,
        entity_name: str,
        entity_data: Dict[str, Any],
        category: str = 'entity',
        tags: Optional[List[str]] = None,
        allow_duplicates: bool = False,
        linked_memory_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add an entity (person, organization, project, service) as a memory document.
        
        Entities are stored with specific media_type to enable querying by entity type.
        The extra JSON field stores entity-specific metadata and cross-references to email memories.
        
        Args:
            entity_type: Type of entity ('person', 'organization', 'project', 'service')
            entity_name: Name/identifier of the entity
            entity_data: Dictionary with entity-specific data (email, role, description, etc.)
            category: Category for the memory (default: 'entity')
            tags: Optional tags for the entity
            allow_duplicates: If False, check for similar existing entities first
            linked_memory_id: Optional ID of email memory to link to this entity
            
        Returns:
            Dictionary with operation result
        """
        try:
            import uuid
            
            # Validate entity type
            valid_types = ['person', 'organization', 'project', 'service']
            if entity_type not in valid_types:
                return {"ok": False, "error": f"Invalid entity_type. Must be one of: {valid_types}"}
            
            # Build content for similarity check
            content_for_check = entity_name + " " + entity_data.get("description", "")
            
            # Check for similar existing entities if duplicates not allowed
            if not allow_duplicates and self.ltm.embeddings is not None and content_for_check.strip():
                similar_docs = self.find_similar_documents(
                    content_for_check, 
                    top_k=3, 
                    similarity_threshold=0.85
                )
                # Filter to same entity type
                for sim_doc in similar_docs:
                    doc = sim_doc.get("document", {})
                    if doc.get("media_type") == entity_type:
                        # Check if it's really the same entity (name similarity)
                        doc_extra = doc.get("extra", {})
                        existing_name = doc_extra.get("entity_data", {}).get("name", "")
                        if existing_name.lower() in entity_name.lower() or entity_name.lower() in existing_name.lower():
                            # Update entity with new memory link instead of creating duplicate
                            if linked_memory_id:
                                update_result = self.update_entity_with_memory(
                                    sim_doc["document_id"], 
                                    linked_memory_id
                                )
                                return {
                                    "ok": True,
                                    "id": sim_doc["document_id"],
                                    "title": doc_extra.get("title", entity_name),
                                    "category": category,
                                    "chunks_created": 0,
                                    "message": "Entity already exists, updated with new memory link",
                                    "updated": True,
                                    "existing_document": doc
                                }
            
            # Generate a unique ID for the entity
            entity_id = str(uuid.uuid4())[:8]
            
            # Build title
            entity_icons = {
                'person': '👤',
                'organization': '🏢',
                'project': '📁',
                'service': '🔧'
            }
            icon = entity_icons.get(entity_type, '📄')
            title = f"{icon} {entity_name}"
            
            # Prepare extra metadata with entity data
            import datetime
            now = datetime.datetime.utcnow().isoformat() + "Z"
            
            # Build linked_memories list if provided
            linked_memories = []
            if linked_memory_id:
                linked_memories.append({
                    "memory_id": linked_memory_id,
                    "linked_at": now
                })
            
            extra = {
                'title': title,
                'tags': tags or [entity_type],
                'entity_type': entity_type,
                'entity_data': {
                    'name': entity_name,
                    **entity_data
                },
                'first_seen': now,
                'last_seen': now,
                'linked_memories': linked_memories
            }
            
            # Insert the document with entity type as media_type
            content_preview = entity_data.get("description", entity_name)[:500]
            self.ltm.db_manager.insert_document(
                doc_id=entity_id,
                source_path=f"entity:{entity_type}:{entity_id}",
                rel_path=None,
                media_type=entity_type,  # Store entity type as media_type
                language='en',
                sha256='',
                size_bytes=len(content_preview),
                category=category,
                extra=extra
            )
            
            # Create chunks for entity data
            chunk_ordinal = 0
            
            # Chunk 1: Entity Summary
            summary_content = f"Entity: {entity_name}\nType: {entity_type}"
            if entity_data.get("description"):
                summary_content += f"\nDescription: {entity_data['description']}"
            self.ltm.add_memory_chunk(
                entity_id, chunk_ordinal, 1, len(summary_content.split('\n')),
                summary_content, 'entity_summary', entity_name[:50], None
            )
            chunk_ordinal += 1
            
            # Chunk 2: Entity Details (role, email, etc.)
            details_lines = []
            if entity_data.get("role"):
                details_lines.append(f"Role: {entity_data['role']}")
            if entity_data.get("email"):
                details_lines.append(f"Email: {entity_data['email']}")
            if entity_data.get("organization"):
                details_lines.append(f"Organization: {entity_data['organization']}")
            if entity_data.get("status"):
                details_lines.append(f"Status: {entity_data['status']}")
            if entity_data.get("provider"):
                details_lines.append(f"Provider: {entity_data['provider']}")
            
            if details_lines:
                details_content = "\n".join(details_lines)
                self.ltm.add_memory_chunk(
                    entity_id, chunk_ordinal, 1, len(details_lines),
                    details_content, 'entity_details', None, None, embed=False
                )
                chunk_ordinal += 1
            
            # Chunk 3: Related Information
            if entity_data.get("related_info"):
                related_content = f"Related Info: {entity_data['related_info']}"
                self.ltm.add_memory_chunk(
                    entity_id, chunk_ordinal, 1, len(related_content.split('\n')),
                    related_content, 'entity_related', None, None
                )
                chunk_ordinal += 1
            
            return {
                "ok": True,
                "id": entity_id,
                "title": title,
                "category": category,
                "entity_type": entity_type,
                "chunks_created": chunk_ordinal,
                "message": f"Entity '{entity_name}' added successfully as {entity_type}! (ID: {entity_id})"
            }
        except Exception as e:
            return {
                "ok": False,
                "error": str(e)
            }
    
    def update_entity_with_memory(
        self,
        entity_id: str,
        memory_id: str,
        memory_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update an entity document with a new linked email memory.
        
        Args:
            entity_id: The ID of the entity document to update
            memory_id: The ID of the email memory to link
            memory_context: Optional context summary of the memory
            
        Returns:
            Dictionary with operation result
        """
        try:
            import datetime
            
            # Get current entity document
            details = self.ltm.get_memory(entity_id)
            if not details:
                return {"ok": False, "error": "Entity not found"}
            
            doc = details["document"]
            extra = doc.get("extra", {})
            
            # Update linked_memories
            linked_memories = extra.get("linked_memories", [])
            now = datetime.datetime.utcnow().isoformat() + "Z"
            
            # Check if this memory is already linked
            for link in linked_memories:
                if link.get("memory_id") == memory_id:
                    # Update existing link
                    link["linked_at"] = now
                    if memory_context:
                        link["context"] = memory_context
                    break
            else:
                # Add new link
                new_link = {
                    "memory_id": memory_id,
                    "linked_at": now
                }
                if memory_context:
                    new_link["context"] = memory_context
                linked_memories.append(new_link)
            
            # Update last_seen
            extra["last_seen"] = now
            extra["linked_memories"] = linked_memories
            
            # Update document in database
            self.ltm.db_manager.update_document(
                entity_id,
                extra=extra
            )
            
            return {
                "ok": True,
                "id": entity_id,
                "message": f"Entity updated with memory link"
            }
        except Exception as e:
            return {
                "ok": False,
                "error": str(e)
            }
    
    def get_entity_memory(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get an entity memory by ID with all its details.
        
        Args:
            entity_id: The ID of the entity
            
        Returns:
            Dictionary with entity details and linked memories, or None if not found
        """
        try:
            details = self.ltm.get_memory(entity_id)
            if not details:
                return None
            
            doc = details["document"]
            chunks = details["chunks"]
            
            return {
                "ok": True,
                "id": entity_id,
                "title": doc.get("extra", {}).get("title", ""),
                "entity_type": doc.get("media_type", ""),
                "entity_data": doc.get("extra", {}).get("entity_data", {}),
                "category": doc.get("category", "entity"),
                "tags": doc.get("extra", {}).get("tags", []),
                "first_seen": doc.get("extra", {}).get("first_seen", ""),
                "last_seen": doc.get("extra", {}).get("last_seen", ""),
                "linked_memories": doc.get("extra", {}).get("linked_memories", []),
                "chunks": chunks,
                "created_at": doc.get("created_at", "")
            }
        except Exception as e:
            return {
                "ok": False,
                "error": str(e)
            }
        
    def get_souvenirs_by_title(self, title, category=None, media_type=None, limit: int = 15):
        """Get a souvenir using a media type and a title
        
        Args:
            memory_id: The ID of the email memory
            
        Returns:
            Dictionary with memory details and all chunks, or None if not found
        """
        output = []
        docs_ids = self.ltm.db_manager.search_document_by_title(category=category, media_type=media_type, query=title, limit=limit)
        for doc in docs_ids:
            if doc["similarity_score"] < 0.9:
                continue
            output.append(doc)

        return output


    
    def get_email_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get an email memory by ID with all its chunks.
        
        Args:
            memory_id: The ID of the email memory
            
        Returns:
            Dictionary with memory details and all chunks, or None if not found
        """
        try:
            details = self.ltm.get_memory(memory_id)
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
            results = self.ltm.search_souvenirs(query, category=category, limit=top_k)

            souvenirs = []
            for r in results:
                souvenirs.append({
                    "id": r["document_id"],
                    "title": r.get('title', 'Untitled'),
                    "participants": r.get('participants', 'Unknown'),
                    "category": r.get('category', 'general'),
                    "tags": r.get('tags', []),
                    "similarity_score": r.get('score', 0)
                })
            return souvenirs
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def search_by_chunk_type(self, chunk_types: List[str], query: Optional[str] = None, 
                              top_k: int = 30) -> List[Dict[str, Any]]:
        """Search for souvenirs by specific chunk types.
        
        Args:
            chunk_types: List of chunk types to search
            query: Optional text query for FTS search
            top_k: Maximum results (default increased from 5 to 30)
            
        Returns:
            List of souvenirs matching the criteria
        """
        return self.ltm.search_by_chunk_type(
            chunk_types=chunk_types,
            query=query,
            top_k=top_k
        )
    
    def search_by_participant(self, participant_name: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Search for emails by participant/sender name.
        
        Args:
            participant_name: Name of the sender or participant to search for
            top_k: Maximum number of results
            
        Returns:
            List of souvenirs where the participant was involved
        """
        # Search specifically in email_participants chunk type
        return self.ltm.search_by_chunk_type(
            chunk_types=["email_participants"],
            query=participant_name,
            top_k=top_k
        )
    
    # ========== Entity Query Methods ==========
    
    def get_person_knowledge(self, name_or_email: str) -> Dict[str, Any]:
        """Get all known information about a person from entity memories.
        
        Args:
            name_or_email: Name or email address of the person
            
        Returns:
            Dictionary with person's known information and linked memories
        """
        # Try to find by email first
        if '@' in name_or_email:
            entities = self.entity_context_builder._find_entity_by_value('person', 'email', name_or_email)
            if entities:
                return self._build_entity_knowledge_response(entities[0])
        
        # Try to find by name
        entities = self.entity_context_builder._find_entity_by_value('person', 'name', name_or_email)
        if entities:
            return self._build_entity_knowledge_response(entities[0])
        
        # Also search in linked memories from email participants
        email_results = self.search_by_participant(name_or_email, top_k=10)
        
        return {
            "ok": True,
            "found": False,
            "name": name_or_email,
            "message": "No entity found. Showing email memories with this participant instead.",
            "email_memories": email_results
        }
    
    def get_organization_info(self, org_name: str) -> Dict[str, Any]:
        """Get organization-related memories.
        
        Args:
            org_name: Name of the organization
            
        Returns:
            Dictionary with organization's known information and linked memories
        """
        entities = self.entity_context_builder._find_entity_by_value('organization', 'name', org_name)
        if entities:
            return self._build_entity_knowledge_response(entities[0])
        
        # Search in email content for organization mentions
        search_results = self.search_souvenirs(org_name, top_k=10)
        
        return {
            "ok": True,
            "found": False,
            "name": org_name,
            "message": "No organization entity found. Showing related email memories.",
            "related_memories": search_results
        }
    
    def get_project_info(self, project_name: str) -> Dict[str, Any]:
        """Get project-related memories.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Dictionary with project's known information and linked memories
        """
        entities = self.entity_context_builder._find_entity_by_value('project', 'name', project_name)
        if entities:
            return self._build_entity_knowledge_response(entities[0])
        
        search_results = self.search_souvenirs(project_name, top_k=10)
        
        return {
            "ok": True,
            "found": False,
            "name": project_name,
            "message": "No project entity found. Showing related email memories.",
            "related_memories": search_results
        }
    
    def list_entities(self, entity_type: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List all entities of a specific type or all entities.
        
        Args:
            entity_type: Optional entity type to filter (person/organization/project/service)
            limit: Maximum number of entities to return
            
        Returns:
            List of entity documents
        """
        try:
            # Get all documents with entity category
            all_docs = self.ltm.db_manager.get_all_documents()
            
            entities = []
            for doc in all_docs:
                doc_id = doc[0]
                details = self.ltm.get_memory(doc_id)
                if details:
                    doc_data = details.get("document", {})
                    media_type = doc_data.get("media_type", "")
                    
                    # Filter by entity type if specified
                    if entity_type and media_type != entity_type:
                        continue
                    
                    # Only include entity types
                    if media_type in ['person', 'organization', 'project', 'service']:
                        extra = doc_data.get("extra", {})
                        entities.append({
                            "id": doc_id,
                            "title": extra.get("title", ""),
                            "entity_type": media_type,
                            "entity_data": extra.get("entity_data", {}),
                            "first_seen": extra.get("first_seen", ""),
                            "last_seen": extra.get("last_seen", ""),
                            "linked_memories_count": len(extra.get("linked_memories", []))
                        })
            
            return entities[:limit]
        except Exception as e:
            print(f"Error listing entities: {e}")
            return []
    
    def search_entities(self, query: str, entity_types: Optional[List[str]] = None, 
                       top_k: int = 20) -> List[Dict[str, Any]]:
        """Search across all entity documents.
        
        Args:
            query: Search query
            entity_types: Optional list of entity types to filter
            top_k: Maximum number of results
            
        Returns:
            List of matching entity documents
        """
        # Search in entity category
        results = self.ltm.search_souvenirs(query, category='entity', limit=top_k)
        
        entities = []
        for r in results:
            doc_id = r.get('document_id')
            if doc_id:
                details = self.ltm.get_memory(doc_id)
                if details:
                    doc_data = details.get("document", {})
                    media_type = doc_data.get("media_type", "")
                    
                    # Filter by entity type if specified
                    if entity_types and media_type not in entity_types:
                        continue
                    
                    extra = doc_data.get("extra", {})
                    entities.append({
                        "id": doc_id,
                        "title": extra.get("title", ""),
                        "entity_type": media_type,
                        "entity_data": extra.get("entity_data", {}),
                        "linked_memories": extra.get("linked_memories", []),
                        "similarity_score": r.get("score", 0)
                    })
        
        return entities
    
    def _build_entity_knowledge_response(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Build a comprehensive knowledge response for an entity.
        
        Args:
            entity: Entity dictionary from find_related_entities
            
        Returns:
            Formatted response with entity knowledge
        """
        entity_id = entity.get('id')
        entity_type = entity.get('entity_type', 'entity')
        entity_data = entity.get('entity_data', {})
        linked_memories = entity.get('linked_memories', [])
        
        # Get linked memory details
        linked_memories_details = []
        for memory_link in linked_memories[:10]:
            memory_id = memory_link.get('memory_id')
            if memory_id:
                try:
                    memory_details = self.get_email_memory(memory_id)
                    if memory_details:
                        linked_memories_details.append({
                            "id": memory_id,
                            "title": memory_details.get("title", ""),
                            "category": memory_details.get("category", ""),
                            "created_at": memory_details.get("created_at", ""),
                            "context": memory_link.get("context", "")
                        })
                except:
                    pass
        
        return {
            "ok": True,
            "found": True,
            "id": entity_id,
            "entity_type": entity_type,
            "entity_data": entity_data,
            "linked_memories": linked_memories_details,
            "total_memories": len(linked_memories)
        }
    
    def search_semantic(self, query: str, top_k: int = 20, similarity_threshold: float = 0.75) -> List[Dict[str, Any]]:
        """Search for souvenirs using semantic embeddings.
        
        Args:
            query: The search query
            top_k: Maximum number of results (default increased from 5 to 20)
            similarity_threshold: Minimum similarity score (default lowered from 0.95 to 0.75)
            
        Returns:
            List of souvenirs with similarity scores
        """
        if not self.ltm.is_embeddings_available():
            return []
        
        try:
            # Encode the query
            query_emb = self.ltm.encode_text(query, normalize=True)
            
            # Search in FAISS index - request more results to filter by threshold
            search_k = max(top_k * 2, 40)  # Get more results to apply threshold filtering
            distances, indices = self.ltm.search_embeddings(query_emb, search_k)
            
            # Get chunk information and apply threshold filtering
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0:
                    chunk_info = self.ltm.get_chunk_by_id(int(idx))
                    if chunk_info:
                        chunk_id, doc_id, start_line, end_line, chunk_type, chunk_name, content = chunk_info
                        # Apply similarity threshold filtering
                        if float(dist) >= similarity_threshold:
                            doc_details = self.ltm.get_memory(doc_id)
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
            
            # Limit to requested top_k after filtering
            return results[:top_k]
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
        has_embeddings = self.ltm.embeddings is not None and self.ltm.embeddings_count() > 0
        
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
    "search_strategy": "semantic" or "keyword" or "chunk_type" or "hybrid" or "participant",
    "chunk_types": ["list of relevant chunk types if chunk_type or hybrid strategy"],
    "keywords": ["important keywords for search"],
    "participant_name": "person name if the question is about a specific person (e.g., 'what did John email me', 'emails from Sarah')",
    "entities": ["people, organizations, locations mentioned in the question for query expansion"],
    "dates": ["dates or time periods mentioned (e.g., 'last week', 'this month', 'January')"],
    "action_items_implied": true/false - if question asks about tasks or todos,
    "reasoning": "brief explanation of why this strategy was chosen"
}}

Step 4: Query Expansion - Extract entities and dates from question for broadening search
- Extract person names, organizations, locations mentioned
- Identify any date references (last week, this month, specific dates)
- Note if the question implies looking for action items/tasks

Choose "participant" or include participant_name if the question asks about:
- emails from/to a specific person
- what someone emailed
- communications from a specific person
- "emails from X", "what did Y send me", "messages between me and Z"

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
            
            # Ensure required fields exist - including query expansion fields
            return {
                "search_strategy": strategy.get("search_strategy", "hybrid"),
                "chunk_types": strategy.get("chunk_types", []),
                "keywords": strategy.get("keywords", []),
                "participant_name": strategy.get("participant_name", ""),
                "entities": strategy.get("entities", []),
                "dates": strategy.get("dates", []),
                "action_items_implied": strategy.get("action_items_implied", False),
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
                "participant_name": "",
                "entities": [],
                "dates": [],
                "action_items_implied": False,
                "reasoning": "Fallback due to error",
                "has_embeddings": has_embeddings
            }
    
    def _apply_recency_boost(self, results: List[Dict[str, Any]], decay_factor: float = 0.95, decay_days: int = 30) -> List[Dict[str, Any]]:
        """Apply recency boosting to search results.
        
        Calculates days since email creation and applies a decay factor to boost
        scores for more recent memories.
        
        Args:
            results: List of search results
            decay_factor: Multiplication factor per decay_days (default 0.95)
            decay_days: Number of days for each decay step (default 30)
            
        Returns:
            List of results with combined scores that factor in recency
        """
        from datetime import datetime
        
        try:
            current_time = datetime.now()
        except Exception:
            return results
        
        for result in results:
            created_at = result.get("created_at", "")
            days_old = 0
            
            if created_at:
                try:
                    # Try parsing ISO format
                    if isinstance(created_at, str):
                        created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        # For naive datetime, assume UTC
                        if created_date.tzinfo is None:
                            created_date = created_date
                        # Calculate days difference
                        days_old = (current_time - created_date.replace(tzinfo=None)).days
                except Exception:
                    # Try parsing other common formats
                    try:
                        created_date = datetime.strptime(created_at[:10], "%Y-%m-%d")
                        days_old = (current_time - created_date).days
                    except Exception:
                        pass
            
            # Calculate recency boost: newer items get higher boost
            # decay_factor^(days_old/decay_days) gives 1.0 for today, 0.95 for 30 days ago, etc.
            import math
            recency_multiplier = math.pow(decay_factor, days_old / decay_days) if days_old > 0 else 1.0
            
            # Get original similarity score (default to 0.5 for keyword results without scores)
            original_score = result.get("similarity_score", 0.5)
            
            # Combine scores: weighted average of similarity and recency
            combined_score = (0.8 * original_score) + (0.2 * recency_multiplier)
            
            result["combined_score"] = combined_score
            result["recency_boost"] = recency_multiplier
            result["days_old"] = days_old
        
        return results
    
    def ask_about_souvenirs(self, question: str) -> Dict[str, Any]:
        """Ask a question about souvenirs using intelligent search based on question analysis."""
        # Analyze the question to determine best search strategy
        search_strategy = self._analyze_question_for_search(question)
        
        all_results = []
        
        # Execute search based on strategy - using improved parameters for better recall
        if search_strategy["search_strategy"] in ["semantic", "hybrid"]:
            semantic_results = self.search_semantic(question, top_k=20, similarity_threshold=0.75)
            all_results.extend(semantic_results)
        
        if search_strategy["search_strategy"] in ["keyword", "hybrid"]:
            keywords = search_strategy.get("keywords", [])
            if keywords:
                # Increased from 5 to 10 keywords for better coverage
                keyword_query = " ".join(keywords[:10])
                # Increased from 15 to 30 results for more candidates
                keyword_results = self.search_souvenirs(keyword_query, top_k=30)
                all_results.extend(keyword_results)
        
        if search_strategy["search_strategy"] == "chunk_type":
            chunk_types = search_strategy.get("chunk_types", [])
            if chunk_types:
                # Combine keywords if any - increased from 5 to 10
                query = " ".join(search_strategy.get("keywords", [])[:10])
                # Increased from 15 to 30 results for more candidates
                chunk_results = self.search_by_chunk_type(chunk_types, query=query if query else None, top_k=30)
                all_results.extend(chunk_results)
        
        # Step 2: Add explicit participant/sender search
        participant_name = search_strategy.get("participant_name", "")
        if participant_name:
            participant_results = self.search_by_participant(participant_name, top_k=20)
            all_results.extend(participant_results)
        
        # Step 4: Query Expansion - broaden search using extracted entities and dates
        entities = search_strategy.get("entities", [])
        dates = search_strategy.get("dates", [])
        action_items_implied = search_strategy.get("action_items_implied", False)
        
        # Expand keywords with entities
        expanded_keywords = list(search_strategy.get("keywords", []))
        expanded_keywords.extend(entities)
        
        # If entities or dates are found, do additional searches
        if entities or dates:
            # Search with expanded keywords
            expanded_query = " ".join(expanded_keywords[:15])
            if expanded_query:
                expanded_results = self.search_souvenirs(expanded_query, top_k=20)
                all_results.extend(expanded_results)
            
            # If action items are implied, search action items chunk type
            if action_items_implied:
                action_results = self.search_by_chunk_type(
                    ["email_action_items"],
                    query=expanded_query if expanded_query else None,
                    top_k=20
                )
                all_results.extend(action_results)
            
            # If dates are mentioned, search dates chunk type
            if dates:
                date_query = " ".join(dates)
                date_results = self.search_by_chunk_type(
                    ["email_dates"],
                    query=date_query,
                    top_k=15
                )
                all_results.extend(date_results)
        
        # Deduplicate results by document ID
        seen_ids = set()
        unique_results = []
        for r in all_results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                unique_results.append(r)
        
        # Step 5: Fallback Search Chain - ensure results even when primary search fails
        if len(unique_results) < 3:
            # Fallback 1: Try semantic search with lower threshold
            fallback_semantic = self.search_semantic(question, top_k=15, similarity_threshold=0.5)
            for r in fallback_semantic:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    unique_results.append(r)
        
        if len(unique_results) < 3:
            # Fallback 2: Try chunk-type specific search across all email types
            fallback_chunk = self.search_by_chunk_type(
                ["email_topic", "email_key_points", "email_original"],
                query=question,
                top_k=20
            )
            for r in fallback_chunk:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    unique_results.append(r)
        
        if len(unique_results) < 3:
            # Fallback 3: Full-text keyword search across all chunks
            fallback_keyword = self.search_souvenirs(question, top_k=25)
            for r in fallback_keyword:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    unique_results.append(r)
        
        if not unique_results:
            # Fallback 4: List recent souvenirs as last resort
            recent_memories = self.ltm.list_souvenirs(limit=10)
            for mem in recent_memories:
                doc_id = mem.get("document_id")
                if doc_id:
                    # Get full memory details
                    memory_details = self.ltm.get_memory(doc_id)
                    unique_results.append({
                        "id": doc_id,
                        "title": memory_details.get("document", {}).get("extra", {}).get("title", "Untitled") if memory_details else mem.get("title", "Untitled"),
                        "content": memory_details.get("document", {}).get("content", "")[:500] if memory_details else "",
                        "category": memory_details.get("document", {}).get("category", "general") if memory_details else mem.get("category", "general"),
                        "created_at": memory_details.get("document", {}).get("created_at", "") if memory_details else "",
                        "chunk_type": "fallback_recent",
                        "similarity_score": 0.1
                    })
        
        # Step 3: Apply recency boosting - prioritize recent emails
        unique_results = self._apply_recency_boost(unique_results)
        
        # keep only results that have a combined_score above 0.5 to ensure relevance, but allow some flexibility
        unique_results = [r for r in unique_results if r.get("combined_score", r.get("similarity_score", 0)) >= 0.5]
        if any("similarity_score" in r for r in unique_results):
            unique_results.sort(key=lambda x: x.get("combined_score", x.get("similarity_score", 0)), reverse=True)
        
        # Increased from 5 to 15 to provide more candidates to the LLM
        if len(unique_results) > 10:
            souvenirs = unique_results[:10]
        else:
            souvenirs = unique_results
        
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
            participants = s.get('participants', 'Unknown')
            title = s.get('title', 'Untitled')
            chunk_type = s.get('chunk_type', 'unknown')
            
            formatted.append(f"""
--- Memory {i} ---
Title: {title}
Participants: {participants}
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

Please provide a detailed answer using question language based on the memories above. If the question asks about specific types of information (like action items, dates, participants), prioritize those sections in your answer:"""
    
    def list_souvenirs(self, category: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """List all stored souvenirs, optionally filtered by category."""
        try:
            results = self.ltm.list_souvenirs(category=category, limit=limit)
            
            souvenirs = []
            for r in results:
                souvenirs.append({
                    "id": r["document_id"],
                    "title": r.get('title', 'Untitled'),
                    "category": r.get('category', 'general'),
                    "tags": r.get('tags', []),
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
            categories = self.ltm.list_categories()
            
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
            return self.ltm.list_categories()
        except Exception as e:
            print(f"Error listing categories: {e}")
            return []
    
    def create_category(self, name: str, description: str = '', color: str = '#6B7280', icon: str = '📂') -> Dict[str, Any]:
        """Create a new category."""
        try:
            cat_id = self.ltm.create_category_if_not_exists(name, description, color, icon)
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
