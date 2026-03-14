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
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import configurator
import litellm
from storage.longterm_memory import LongTermMemory
from storage.DatabaseManagement import DatabaseManager


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
        self.enable_embeddings = enable_embeddings
        
        # Initialize LongTermMemory with optional embeddings
        try:
            self.ltm = LongTermMemory(
                db_path=db_path,
                enable_embeddings=enable_embeddings
            )
            
            # Enable souvenir embeddings in the database
            if enable_embeddings:
                self.ltm.db.enable_souvenir_embeddings = True
                self.ltm.db._init_souvenir_embeddings()
            
            print(f"✓ LongTermMemory initialized (embeddings: {enable_embeddings})")
        except Exception as e:
            print(f"❌ Failed to initialize LongTermMemory: {e}")
            raise
        
        if self.cfg.need_configuration:
            print("Warning: LLM not configured. Some features may not work.")
    
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
        """Search for souvenirs matching the query using hybrid search."""
        try:
            # Use hybrid search (combines keyword, FTS5, and semantic search)
            results = self.ltm.db.search_souvenirs_hybrid(query, category=category, limit=top_k)
            
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
                    "score": r.get('rrf_score', 0),  # Include RRF score for reference
                })
            return souvenirs
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def ask_about_souvenirs(self, question: str) -> Dict[str, Any]:
        """Ask a question about souvenirs using LLM with context from memory.
        
        Uses enhanced keyword extraction, query expansion, and hybrid search
        for improved souvenir recall.
        """
        # Extract keywords from the question for better search
        search_query = self._extract_keywords_from_question(question)
        
        # Expand query with synonyms for better recall
        expanded_query = self._expand_query_with_synonyms(search_query)
        
        # Combine original question context with extracted keywords
        # This helps capture semantic meaning beyond keywords
        combined_query = f"{question} {expanded_query}"
        
        # Get relevant souvenirs using hybrid search
        souvenirs = self.search_souvenirs(combined_query, top_k=5)
        
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
        """Extract key terms from a question using LLM for better multilingual support.
        
        Uses LLM to intelligently extract search keywords that will help find
        relevant souvenirs. This approach works well for any language and handles
        complex queries better than rule-based extraction.
        
        Falls back to simple extraction if LLM is not available.
        """
        import re
        
        # Try to use LLM for keyword extraction
        try:
            # Check if model is configured
            if not hasattr(self.cfg, 'model') or not self.cfg.model:
                return self._extract_keywords_fallback(question)
            
            response = litellm.completion(
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": """You are a keyword extraction assistant. Your task is to extract 5-10 important search keywords from the user's question that will help find relevant souvenirs/memories.

Extract keywords that describe:
- What happened (events, activities)
- Who was involved (people names or roles)
- Where it happened (locations)
- When it happened (time expressions like yesterday, last week, first day)
- Emotional context (celebration, meeting, trip)

IMPORTANT:
1. Output ONLY keywords, one per line, no numbering
2. Do NOT include question words (what, who, where, when, why, how)
3. Do NOT include articles (the, a, an)
4. Use base form of verbs (working -> work, met -> meet)
5. Include specific names or proper nouns as they are
6. Handle ANY language - extract keywords from the language the question is written in
7. If the question mentions negation (NOT something), include both positive and negative concepts

Example:
Input: "What did I do on my first day at work?"
Output:
first day
work
office
colleague
boss
"""},
                    {"role": "user", "content": f"Extract keywords from this question:\n{question}"}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            # Parse the LLM response
            keywords_text = response.choices[0].message.content.strip()
            keywords = [k.strip() for k in keywords_text.split('\n') if k.strip()]
            
            # Filter out any non-keyword content
            keywords = [k for k in keywords if len(k) > 1 and not k.startswith(('1', '2', '3', '4', '5', '6', '7', '8', '9', '0'))]
            
            return ' '.join(keywords[:8])
            
        except Exception as e:
            # Fall back to simple extraction if LLM fails
            print(f"LLM keyword extraction failed ({e}), using fallback")
            return self._extract_keywords_fallback(question)
    
    def _extract_keywords_fallback(self, question: str) -> str:
        """Fallback keyword extraction using simple rules.
        
        This is a simplified version kept for when LLM is not available.
        """
        import re
        from collections import OrderedDict
        
        question_lower = question.lower()
        
        # Define comprehensive stop words (question words + common filler words)
        stop_words = {
            'what', 'who', 'where', 'when', 'why', 'how', 'which', 'whom',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'do', 'does', 'did', 'have', 'has', 'had',
            'can', 'could', 'would', 'should', 'may', 'might', 'must',
            'i', 'me', 'my', 'mine', 'we', 'our', 'ours',
            'you', 'your', 'yours', 'he', 'she', 'it', 'they', 'them', 'their',
            'this', 'that', 'these', 'those',
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else',
            'to', 'for', 'of', 'in', 'on', 'at', 'by', 'with', 'about',
            'from', 'up', 'out', 'over', 'under', 'again', 'further',
            'so', 'very', 'just', 'only', 'also', 'now', 'here', 'there',
            'tell', 'ask', 'remember', 'recall', 'find', 'get', 'show'
        }
        
        # Simple lemmatization dictionary for common irregular words
        lemmatization_map = {
            'meeting': 'meet', 'meetings': 'meet',
            'working': 'work', 'worked': 'work',
            'going': 'go', 'went': 'go',
            'having': 'have', 'had': 'have',
            'doing': 'do', 'did': 'do',
            'saying': 'say', 'said': 'say',
            'thinking': 'think', 'thought': 'think',
            'eating': 'eat', 'ate': 'eat',
            'buying': 'buy', 'bought': 'buy',
            'seeing': 'see', 'saw': 'see',
            'getting': 'get', 'got': 'get',
            'making': 'make', 'made': 'make',
            'taking': 'take', 'took': 'take',
            'giving': 'give', 'gave': 'give',
            'knowing': 'know', 'knew': 'know',
            'coming': 'come', 'came': 'come',
            'using': 'use', 'used': 'use',
            'learning': 'learn', 'learned': 'learn', 'learnt': 'learn',
            'talking': 'talk', 'talked': 'talk',
            'walking': 'walk', 'walked': 'walk',
            'running': 'run', 'ran': 'run',
            'traveling': 'travel', 'travelled': 'travel',
            'writing': 'write', 'wrote': 'write',
            'reading': 'read', 'read': 'read',
            'listening': 'listen', 'listened': 'listen',
            'watching': 'watch', 'watched': 'watch',
            'playing': 'play', 'played': 'play',
            'working': 'work', 'works': 'work',
            'visiting': 'visit', 'visited': 'visit',
            'celebrating': 'celebrate', 'celebrated': 'celebrate',
            'graduating': 'graduate', 'graduated': 'graduate',
            'starting': 'start', 'started': 'start',
            'finishing': 'finish', 'finished': 'finish',
            'beginning': 'begin', 'began': 'begin',
            'purchasing': 'purchase', 'purchased': 'purchase',
            'ordered': 'order', 'ordering': 'order',
            'received': 'receive', 'receiving': 'receive',
        }
        
        # Simple suffix-stripping lemmatization for regular words
        def simple_lemmatize(word: str) -> str:
            """Apply simple rule-based lemmatization."""
            # Check dictionary first
            if word in lemmatization_map:
                return lemmatization_map[word]
            
            # Handle common suffixes
            if word.endswith('ies') and len(word) > 4:
                return word[:-3] + 'y'
            elif word.endswith('ied') and len(word) > 4:
                return word[:-3] + 'y'
            elif word.endswith('es') and len(word) > 4:
                return word[:-2]
            elif word.endswith('ed') and len(word) > 4:
                # Check for double consonant
                if len(word) > 3 and word[-3] == word[-4]:
                    return word[:-3]
                return word[:-2]
            elif word.endswith('ing') and len(word) > 5:
                # Handle doubled consonants (running -> run)
                if len(word) > 5 and word[-4] == word[-5]:
                    return word[:-4]
                return word[:-3]
            elif word.endswith('s') and len(word) > 3 and word[-2] not in 'su':
                return word[:-1]
            elif word.endswith('er') and len(word) > 4:
                return word[:-2]
            elif word.endswith('est') and len(word) > 5:
                return word[:-3]
            
            return word
        
        # Detect negation patterns
        negation_words = {'not', 'no', "n't", 'never', 'none', 'nothing', 'neither', 'nobody', 'nowhere'}
        has_negation = any(neg in question_lower for neg in negation_words)
        
        # Extract time expressions for boosting
        time_patterns = [
            (r'\byesterday\b', 'yesterday'),
            (r'\btoday\b', 'today'),
            (r'\blast week\b', 'last_week'),
            (r'\blist month\b', 'last_month'),
            (r'\byesterday\b', 'yesterday'),
            (r'\bweek ago\b', 'week_ago'),
            (r'\bmonth ago\b', 'month_ago'),
            (r'\byears? ago\b', 'years_ago'),
            (r'\bfirst day\b', 'first_day'),
            (r'\blast year\b', 'last_year'),
        ]
        
        time_expressions = []
        for pattern, label in time_patterns:
            if re.search(pattern, question_lower):
                time_expressions.append(label)
        
        # Extract named entities patterns (people, places)
        entity_patterns = [
            (r'\b(mr|mrs|ms|dr|prof)\s+\w+\b', 'person'),
            (r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', 'person'),
            (r'\b(?:at|in|to|from)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', 'place'),
        ]
        
        # Clean the question and extract tokens
        # Remove question words and punctuation
        cleaned = re.sub(r'\b(what|who|where|when|why|how|which|whom)\b', '', question_lower)
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Extract words
        words = cleaned.split()
        
        # Filter stopwords and lemmatize
        keywords = []
        for word in words:
            if len(word) > 2 and word not in stop_words:
                lemma = simple_lemmatize(word)
                if lemma not in stop_words:
                    keywords.append(lemma)
        
        # Add time expressions if found
        keywords.extend(time_expressions)
        
        # Deduplicate while preserving order using OrderedDict
        unique_keywords = list(OrderedDict.fromkeys(keywords))
        
        # Limit to top keywords
        final_keywords = unique_keywords[:8]
        
        return ' '.join(final_keywords)
    
    def _expand_query_with_synonyms(self, query: str) -> str:
        """Expand query with synonyms and related terms for better recall.
        
        Uses a simple synonym dictionary for common terms related to
        memories, events, people, and activities.
        """
        # Synonym dictionary for common memory-related terms
        synonyms = {
            # Work/Career
            'work': ['job', 'career', 'professional', 'office', 'boss', 'colleague', 'coworker'],
            'meeting': ['meet', 'met', 'discussion', 'standup', 'one-on-one'],
            'project': ['task', 'assignment', 'deliverable', 'deadline'],
            
            # Social/People
            'meet': ['met', 'introduced', 'introduce', 'greet', 'greeting'],
            'friend': ['buddy', 'companion', 'acquaintance', 'pal', 'mate'],
            'colleague': ['coworker', 'teammate', 'associate', 'partner'],
            'boss': ['manager', 'supervisor', 'lead', 'chief', 'director'],
            
            # Events/Activities
            'event': ['occasion', 'function', 'gathering', 'party', 'celebration'],
            'conference': ['convention', 'summit', 'seminar', 'workshop', 'talk'],
            'lunch': ['meal', 'dinner', 'breakfast', 'food', 'eating'],
            'travel': ['trip', 'journey', 'visit', 'vacation', 'honeymoon'],
            
            # Memory/Learning
            'learn': ['learned', 'learnt', 'study', 'studied', 'discover', 'discovered'],
            'remember': ['recall', 'recollect', 'remind', 'reminded', 'memory'],
            'forget': ['forgot', 'forgotten', 'missed', 'miss'],
            
            # Time expressions
            'first': ['initial', 'initial', 'beginning', 'beginning'],
            'last': ['previous', 'past', 'recent', 'recently'],
            'yesterday': ['previous', 'prior', 'last day'],
            'today': ['now', 'current', 'this day'],
            'week': ['seven days', 'weekend', 'weekday'],
            
            # Emotions/States
            'happy': ['glad', 'pleased', 'delighted', 'thrilled', 'excited'],
            'sad': ['unhappy', 'disappointed', 'upset', 'down'],
            'excited': ['thrilled', 'eager', 'enthusiastic', 'pumped'],
            
            # Achievements
            'promotion': ['advance', 'raised', 'elevation', 'step up'],
            'award': ['prize', 'recognition', 'honor', 'accolade'],
            'graduate': ['graduation', 'degree', 'diploma', 'completed'],
        }
        
        # Add expansion
        expanded_terms = []
        query_lower = query.lower()
        words = query_lower.split()
        
        for word in words:
            expanded_terms.append(word)
            if word in synonyms:
                # Add a subset of synonyms (not all to avoid query explosion)
                for syn in synonyms[word][:3]:
                    if syn not in query_lower:  # Only add if not already in query
                        expanded_terms.append(syn)
        
        return ' '.join(expanded_terms)
    
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
