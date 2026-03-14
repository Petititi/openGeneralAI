import json
import sqlite3
import datetime as dt
import threading
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# --------------------------
# DatabaseManager
# --------------------------

class DatabaseManager:
    """
    Gère toutes les opérations de base de données SQLite pour la mémoire à long terme.
    Responsable de la persistance des documents, chunks, imports et métadonnées.
    Thread-safe grâce à l'utilisation d'un verrou pour synchroniser les accès.
    """
    def __init__(self, db_path: str = "memory.sqlite", enable_souvenir_embeddings: bool = False):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like row access
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self._lock = threading.RLock()
        
        # For souvenir embeddings
        self.enable_souvenir_embeddings = enable_souvenir_embeddings
        self._souvenir_embedding_manager = None
        self._souvenir_faiss_index_path = db_path.replace('.sqlite', '_souvenirs.faiss')
        
        if enable_souvenir_embeddings:
            self._init_souvenir_embeddings()
        
        self._ensure_schema()

    def _ensure_schema(self):
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            rel_path TEXT,
            media_type TEXT,
            language TEXT,
            sha256 TEXT,
            size_bytes INTEGER,
            created_at TEXT,
            category TEXT DEFAULT 'general',
            extra JSON
            )""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            color TEXT,
            icon TEXT,
            created_at TEXT,
            is_system INTEGER DEFAULT 0
            )""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS document_imports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
            import_statement TEXT
            )""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
            ord INTEGER,
            start_line INTEGER,
            end_line INTEGER,
            content TEXT,
            token_count INTEGER DEFAULT NULL,
            chunk_type TEXT,
            chunk_name TEXT,
            parent_class TEXT
            )""")
            cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            content, 
            chunk_id UNINDEXED, 
            document_id UNINDEXED, 
            tokenize='porter',
            content='chunks',
            content_rowid='id'
            )""")
            cur.executescript("""
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, content, chunk_id, document_id)
            VALUES (new.id, new.content, new.id, new.document_id);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid) VALUES('delete', old.id);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, content, chunk_id, document_id)
            VALUES('delete', old.id, old.content, old.id, old.document_id);
            INSERT INTO chunks_fts(rowid, content, chunk_id, document_id)
            VALUES (new.id, new.content, new.id, new.document_id);
            END;
            """)
            
            # Create FTS5 virtual table specifically for souvenirs
            cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS souvenirs_fts USING fts5(
            title,
            content,
            tags,
            tokenize='porter'
            )""")
            
            # Create table to map souvenir IDs to FTS rowids
            cur.execute("""
            CREATE TABLE IF NOT EXISTS souvenirs_faiss_map (
            souvenir_id TEXT PRIMARY KEY,
            faiss_id INTEGER
            )""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS faiss_map (
            faiss_id INTEGER PRIMARY KEY,
            chunk_id INTEGER UNIQUE REFERENCES chunks(id) ON DELETE CASCADE
            )""")
            cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunk_type ON chunks(chunk_type)
            """)
            cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunk_name ON chunks(chunk_name)
            """)
            cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_parent_class ON chunks(parent_class)
            """)
            cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_category ON documents(category)
            """)
            
            # Initialize default categories
            self._init_default_categories()
            
            self.conn.commit()
    
    def _init_souvenir_embeddings(self):
        """Initialize FAISS index for souvenir embeddings."""
        try:
            model_name = "mixedbread-ai/mxbai-embed-large-v1"
            self._souvenir_embedding_manager = SentenceTransformer(model_name)
            self._souvenir_dim = self._souvenir_embedding_manager.get_sentence_embedding_dimension()
            
            # Load existing index or create new one
            if os.path.exists(self._souvenir_faiss_index_path):
                self._souvenir_faiss_index = faiss.read_index(self._souvenir_faiss_index_path)
            else:
                self._souvenir_faiss_index = faiss.IndexFlatIP(self._souvenir_dim)
            
            # Load existing souvenir-FAISS mapping from database
            self._load_souvenir_faiss_map()
        except Exception as e:
            print(f"Warning: Failed to initialize souvenir embeddings: {e}")
            self.enable_souvenir_embeddings = False
    
    def _load_souvenir_faiss_map(self):
        """Load the mapping of souvenir IDs to FAISS IDs."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT souvenir_id, faiss_id FROM souvenirs_faiss_map WHERE faiss_id IS NOT NULL")
            rows = cur.fetchall()
            self._souvenir_faiss_id_map = {}  # faiss_id -> souvenir_id
            self._souvenir_id_faiss_map = {}  # souvenir_id -> faiss_id
            for row in rows:
                if hasattr(row, '__getitem__'):
                    souvenir_id, faiss_id = row[0], row[1]
                    self._souvenir_faiss_id_map[faiss_id] = souvenir_id
                    self._souvenir_id_faiss_map[souvenir_id] = faiss_id

    def insert_document(self, doc_id: str, source_path: str, rel_path: Optional[str],
                       media_type: str, language: Optional[str], sha256: str,
                       size_bytes: int, category: str = 'general', extra: Optional[Dict] = None) -> None:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
            INSERT OR IGNORE INTO documents(id, source_path, rel_path, media_type, language, sha256, size_bytes, created_at, category, extra)
            VALUES(?,?,?,?,?,?,?,?,?,?)
        """, (doc_id, source_path, rel_path, media_type, language, sha256, size_bytes, now_iso(), category, json.dumps(extra or {})))
            self.conn.commit()
    
    def _init_default_categories(self):
        """Initialize default system categories."""
        cur = self.conn.cursor()
        default_categories = [
            {'name': 'general', 'description': 'General notes and memories', 'color': '#6B7280', 'icon': '📝', 'is_system': 1},
            {'name': 'appointment', 'description': 'Appointments and scheduled events', 'color': '#3B82F6', 'icon': '📅', 'is_system': 1},
            {'name': 'work', 'description': 'Work-related memories', 'color': '#10B981', 'icon': '💼', 'is_system': 1},
            {'name': 'object', 'description': 'Objects and items', 'color': '#F59E0B', 'icon': '📦', 'is_system': 1},
            {'name': 'person', 'description': 'People and contacts', 'color': '#EC4899', 'icon': '👤', 'is_system': 1},
            {'name': 'location', 'description': 'Places and locations', 'color': '#8B5CF6', 'icon': '📍', 'is_system': 1},
            {'name': 'idea', 'description': 'Ideas and thoughts', 'color': '#06B6D4', 'icon': '💡', 'is_system': 1},
            {'name': 'travel', 'description': 'Travel and trips', 'color': '#EF4444', 'icon': '✈️', 'is_system': 1},
        ]
        
        for cat in default_categories:
            cur.execute("""
                INSERT OR IGNORE INTO categories (name, description, color, icon, created_at, is_system)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (cat['name'], cat['description'], cat['color'], cat['icon'], now_iso(), cat['is_system']))
        self.conn.commit()
    
    # ---------- Category Management ----------
    
    def create_category(self, name: str, description: str = '', color: str = '#6B7280', icon: str = '📂') -> int:
        """Create a new category. Returns the category id."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO categories (name, description, color, icon, created_at, is_system)
                VALUES (?, ?, ?, ?, ?, 0)
            """, (name, description, color, icon, now_iso()))
            self.conn.commit()
            return cur.lastrowid
    
    def get_category(self, name: str) -> Optional[Dict]:
        """Get a category by name."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM categories WHERE name = ?", (name,))
            row = cur.fetchone()
            if row:
                return dict(row)
            return None
    
    def list_categories(self) -> List[Dict]:
        """List all categories."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM categories ORDER BY is_system DESC, name")
            results = []
            for row in cur.fetchall():
                # Handle both tuple and Row objects
                if hasattr(row, 'keys'):
                    results.append(dict(row))
                else:
                    # Get column names from cursor description
                    columns = [desc[0] for desc in cur.description]
                    results.append(dict(zip(columns, row)))
            return results
    
    def update_category(self, name: str, description: str = None, color: str = None, icon: str = None) -> bool:
        """Update a category. Returns True if successful."""
        with self._lock:
            cur = self.conn.cursor()
            updates = []
            params = []
            if description is not None:
                updates.append("description = ?")
                params.append(description)
            if color is not None:
                updates.append("color = ?")
                params.append(color)
            if icon is not None:
                updates.append("icon = ?")
                params.append(icon)
            
            if not updates:
                return False
            
            params.append(name)
            cur.execute(f"UPDATE categories SET {', '.join(updates)} WHERE name = ? AND is_system = 0", params)
            self.conn.commit()
            return cur.rowcount > 0
    
    def delete_category(self, name: str) -> bool:
        """Delete a category. Only non-system categories can be deleted."""
        with self._lock:
            cur = self.conn.cursor()
            # First, move documents from this category to 'general'
            cur.execute("UPDATE documents SET category = 'general' WHERE category = ?", (name,))
            # Then delete the category
            cur.execute("DELETE FROM categories WHERE name = ? AND is_system = 0", (name,))
            self.conn.commit()
            return cur.rowcount > 0
    
    # ---------- Souvenir/Document helpers ----------
    
    def insert_souvenir(self, doc_id: str, content: str, title: str = None, category: str = 'general', tags: List[str] = None) -> None:
        """Insert a souvenir (convenience method)."""
        extra = {'title': title, 'tags': tags or []}
        self.insert_document(
            doc_id=doc_id,
            source_path=f"souvenir:{doc_id}",
            rel_path=None,
            media_type='text/plain',
            language='en',
            sha256='',
            size_bytes=len(content),
            category=category,
            extra=extra
        )
        # Insert content as a single chunk
        self.insert_chunk(doc_id, 0, 1, len(content.split('\n')), content, 'souvenir', title, None)
        
        # Also add to souvenirs FTS5 index
        with self._lock:
            cur = self.conn.cursor()
            tags_str = ' '.join(tags) if tags else ''
            cur.execute("""
                INSERT INTO souvenirs_fts(title, content, tags)
                VALUES (?, ?, ?)
            """, (title or '', content, tags_str))
            
            # Get the rowid of the inserted FTS entry
            fts_rowid = cur.lastrowid
            
            # Update the souvenirs_faiss_map (initially with None for faiss_id)
            cur.execute("""
                INSERT OR REPLACE INTO souvenirs_faiss_map(souvenir_id, faiss_id)
                VALUES (?, ?)
            """, (doc_id, fts_rowid))
            
            self.conn.commit()
        
        # Add embedding for semantic search if enabled
        if self.enable_souvenir_embeddings and self._souvenir_embedding_manager is not None:
            self._add_souvenir_embedding(doc_id, title or '', content, ' '.join(tags) if tags else '')
    
    def get_souvenir(self, doc_id: str) -> Optional[Dict]:
        """Get a souvenir by ID."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cur.fetchone()
            if not row:
                return None
            
            doc = dict(row)
            # Get the chunk content
            cur.execute("SELECT content FROM chunks WHERE document_id = ?", (doc_id,))
            chunk_row = cur.fetchone()
            if chunk_row:
                doc['content'] = chunk_row[0]
            return doc
    
    def list_souvenirs(self, category: str = None, limit: int = 20) -> List[Dict]:
        """List all souvenirs, optionally filtered by category."""
        with self._lock:
            cur = self.conn.cursor()
            if category:
                cur.execute("""
                    SELECT d.*, c.content 
                    FROM documents d
                    JOIN chunks c ON d.id = c.document_id
                    WHERE d.category = ?
                    ORDER BY d.created_at DESC
                    LIMIT ?
                """, (category, limit))
            else:
                cur.execute("""
                    SELECT d.*, c.content 
                    FROM documents d
                    JOIN chunks c ON d.id = c.document_id
                    ORDER BY d.created_at DESC
                    LIMIT ?
                """, (limit,))
            
            results = []
            for row in cur.fetchall():
                # Handle both tuple and Row objects
                if hasattr(row, 'keys'):
                    doc = dict(row)
                else:
                    # Get column names from cursor description
                    columns = [desc[0] for desc in cur.description]
                    doc = dict(zip(columns, row))
                results.append(doc)
            return results
    
    def search_souvenirs(self, query: str, category: str = None, limit: int = 10) -> List[Dict]:
        """Search souvenirs by content using simple LIKE search with OR logic."""
        with self._lock:
            cur = self.conn.cursor()
            
            # Split query into keywords and search with OR logic
            keywords = query.lower().split()
            
            if len(keywords) == 1:
                # Single keyword - simple search
                if category:
                    cur.execute("""
                        SELECT d.*, c.content 
                        FROM documents d
                        JOIN chunks c ON d.id = c.document_id
                        WHERE (c.content LIKE ? OR d.extra LIKE ?) AND d.category = ?
                        ORDER BY d.created_at DESC
                        LIMIT ?
                    """, (f'%{query}%', f'%{query}%', category, limit))
                else:
                    cur.execute("""
                        SELECT d.*, c.content 
                        FROM documents d
                        JOIN chunks c ON d.id = c.document_id
                        WHERE c.content LIKE ? OR d.extra LIKE ?
                        ORDER BY d.created_at DESC
                        LIMIT ?
                    """, (f'%{query}%', f'%{query}%', limit))
            else:
                # Multiple keywords - OR logic for any match
                conditions = " OR ".join(["c.content LIKE ?" for _ in keywords]) + " OR " + " OR ".join(["d.extra LIKE ?" for _ in keywords])
                params = [f'%{kw}%' for kw in keywords] + [f'%{kw}%' for kw in keywords]
                
                if category:
                    sql = f"""
                        SELECT d.*, c.content 
                        FROM documents d
                        JOIN chunks c ON d.id = c.document_id
                        WHERE ({conditions}) AND d.category = ?
                        ORDER BY d.created_at DESC
                        LIMIT ?
                    """
                    params.extend([category, limit])
                else:
                    sql = f"""
                        SELECT d.*, c.content 
                        FROM documents d
                        JOIN chunks c ON d.id = c.document_id
                        WHERE {conditions}
                        ORDER BY d.created_at DESC
                        LIMIT ?
                    """
                    params.append(limit)
                
                cur.execute(sql, params)
            
            results = []
            for row in cur.fetchall():
                # Handle both tuple and Row objects
                if hasattr(row, 'keys'):
                    doc = dict(row)
                else:
                    # Get column names from cursor description
                    columns = [desc[0] for desc in cur.description]
                    doc = dict(zip(columns, row))
                results.append(doc)
            return results
    
    def search_souvenirs_fts(self, query: str, category: str = None, limit: int = 10) -> List[Dict]:
        """Search souvenirs using FTS5 full-text search with BM25 scoring.
        
        This provides much better search quality than simple LIKE queries.
        """
        with self._lock:
            cur = self.conn.cursor()
            
            # Prepare the FTS5 query - join keywords with OR for flexible matching
            keywords = query.lower().split()
            fts_query = ' OR '.join(keywords)
            
            # Build the SQL query with BM25 scoring
            if category:
                sql = """
                    SELECT d.*, c.content, bm25(souvenirs_fts) as rank
                    FROM souvenirs_fts sfts
                    JOIN documents d ON d.id = (
                        SELECT souvenir_id FROM souvenirs_faiss_map WHERE faiss_id = sfts.rowid
                    )
                    JOIN chunks c ON d.id = c.document_id
                    WHERE souvenirs_fts MATCH ? AND d.category = ?
                    ORDER BY rank
                    LIMIT ?
                """
                cur.execute(sql, (fts_query, category, limit))
            else:
                sql = """
                    SELECT d.*, c.content, bm25(souvenirs_fts) as rank
                    FROM souvenirs_fts sfts
                    JOIN documents d ON d.id = (
                        SELECT souvenir_id FROM souvenirs_faiss_map WHERE faiss_id = sfts.rowid
                    )
                    JOIN chunks c ON d.id = c.document_id
                    WHERE souvenirs_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """
                cur.execute(sql, (fts_query, limit))
            
            results = []
            for row in cur.fetchall():
                if hasattr(row, 'keys'):
                    doc = dict(row)
                else:
                    columns = [desc[0] for desc in cur.description]
                    doc = dict(zip(columns, row))
                results.append(doc)
            return results
    
    def _add_souvenir_embedding(self, doc_id: str, title: str, content: str, tags: str) -> None:
        """Add a souvenir embedding to the FAISS index."""
        if not self.enable_souvenir_embeddings or self._souvenir_embedding_manager is None:
            return
        
        try:
            # Create combined text for embedding
            combined_text = f"{title} {content} {tags}".strip()
            
            # Encode the text
            embedding = self._souvenir_embedding_manager.encode(
                [combined_text],
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype("float32")
            
            # Add to FAISS index
            faiss_id = self._souvenir_faiss_index.ntotal
            self._souvenir_faiss_index.add(embedding)
            
            # Update mapping
            self._souvenir_faiss_id_map[faiss_id] = doc_id
            self._souvenir_id_faiss_map[doc_id] = faiss_id
            
            # Persist the FAISS index
            faiss.write_index(self._souvenir_faiss_index, self._souvenir_faiss_index_path)
            
            # Update the database mapping
            with self._lock:
                cur = self.conn.cursor()
                cur.execute("""
                    UPDATE souvenirs_faiss_map SET faiss_id = ? WHERE souvenir_id = ?
                """, (faiss_id, doc_id))
                self.conn.commit()
        except Exception as e:
            print(f"Error adding souvenir embedding: {e}")
    
    def search_souvenirs_semantic(self, query: str, category: str = None, limit: int = 10) -> List[Dict]:
        """Search souvenirs using semantic embeddings (vector search).
        
        Returns souvenirs most similar to the query using cosine similarity.
        """
        if not self.enable_souvenir_embeddings or self._souvenir_embedding_manager is None:
            return []
        
        try:
            # Encode the query
            query_embedding = self._souvenir_embedding_manager.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype("float32")
            
            # Search in FAISS
            distances, indices = self._souvenir_faiss_index.search(query_embedding, min(limit * 2, self._souvenir_faiss_index.ntotal))
            
            # Collect results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                souvenir_id = self._souvenir_faiss_id_map.get(int(idx))
                if souvenir_id is None:
                    continue
                
                # Get the souvenir details from the database
                with self._lock:
                    cur = self.conn.cursor()
                    cur.execute("""
                        SELECT d.*, c.content
                        FROM documents d
                        JOIN chunks c ON d.id = c.document_id
                        WHERE d.id = ?
                    """, (souvenir_id,))
                    row = cur.fetchone()
                    if row:
                        if hasattr(row, 'keys'):
                            doc = dict(row)
                        else:
                            columns = [desc[0] for desc in cur.description]
                            doc = dict(zip(columns, row))
                        doc['score'] = float(dist)
                        results.append(doc)
                
                if len(results) >= limit:
                    break
            
            # Apply category filter if specified
            if category:
                results = [r for r in results if r.get('category') == category]
            
            return results[:limit]
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def search_souvenirs_hybrid(self, query: str, category: str = None, limit: int = 10) -> List[Dict]:
        """Hybrid search combining keyword (LIKE), FTS5, and semantic search using Reciprocal Rank Fusion.
        
        This approach provides robust recall by combining multiple search signals:
        - Keyword search (LIKE queries) - traditional text matching
        - FTS5 full-text search - better ranking with BM25
        - Semantic search - concept matching via embeddings
        
        Results are fused using RRF: score = 1 / (k + rank)
        """
        from collections import defaultdict
        
        # RRF constant - higher k reduces the impact of ranking differences
        k = 60
        
        # Get results from each search method
        keyword_results = self.search_souvenirs(query, category=category, limit=limit * 3)
        fts_results = self.search_souvenirs_fts(query, category=category, limit=limit * 3)
        semantic_results = self.search_souvenirs_semantic(query, category=category, limit=limit * 3)
        
        # Build RRF scores
        rrf_scores = defaultdict(float)
        doc_details = {}
        
        # Process keyword results
        for rank, doc in enumerate(keyword_results):
            doc_id = doc.get('id')
            if doc_id:
                rrf_scores[doc_id] += 1.0 / (k + rank + 1)
                if doc_id not in doc_details:
                    doc_details[doc_id] = doc
        
        # Process FTS results
        for rank, doc in enumerate(fts_results):
            doc_id = doc.get('id')
            if doc_id:
                rrf_scores[doc_id] += 1.0 / (k + rank + 1)
                if doc_id not in doc_details:
                    doc_details[doc_id] = doc
        
        # Process semantic results (with similarity score weighting)
        for rank, doc in enumerate(semantic_results):
            doc_id = doc.get('id')
            if doc_id:
                # Weight semantic scores by their similarity (already 0-1 range)
                semantic_weight = doc.get('score', 0.5)
                rrf_scores[doc_id] += semantic_weight * (1.0 / (k + rank + 1))
                if doc_id not in doc_details:
                    doc_details[doc_id] = doc
        
        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build final results
        results = []
        for doc_id, score in sorted_docs[:limit]:
            doc = doc_details.get(doc_id)
            if doc:
                doc['rrf_score'] = score
                results.append(doc)
        
        return results

    def insert_imports(self, doc_id: str, imports: List[str]) -> None:
        with self._lock:
            cur = self.conn.cursor()
            for imp in imports:
                cur.execute("""
                INSERT INTO document_imports(document_id, import_statement)
                VALUES(?,?)
            """, (doc_id, imp))
            self.conn.commit()

    def insert_chunk(self, doc_id: str, ord: int, start_line: int, end_line: int,
                    content: str, chunk_type: Optional[str], chunk_name: Optional[str],
                    parent_class: Optional[str]) -> int:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
            INSERT INTO chunks(document_id, ord, start_line, end_line, content, chunk_type, chunk_name, parent_class)
            VALUES(?,?,?,?,?,?,?,?)
        """, (doc_id, ord, start_line, end_line, content, chunk_type, chunk_name, parent_class))
            chunk_id = cur.lastrowid
            self.conn.commit()
            return chunk_id

    def get_chunk_content(self, chunk_id: int) -> Optional[str]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT content FROM chunks WHERE id=?", (chunk_id,))
            row = cur.fetchone()
            return row[0] if row else None

    def get_all_chunks(self) -> List[Tuple[int, str, Optional[int]]]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT c.id, c.content, fm.faiss_id FROM chunks c LEFT JOIN faiss_map fm ON c.id = fm.chunk_id ORDER BY c.id")
            return cur.fetchall()

    def insert_faiss_mapping(self, faiss_id: int, chunk_id: int) -> None:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("INSERT INTO faiss_map(faiss_id, chunk_id) VALUES(?,?)", (faiss_id, chunk_id))
            self.conn.commit()

    def clear_faiss_mappings(self) -> None:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM faiss_map")
            self.conn.commit()

    def get_faiss_mapping_count(self) -> int:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT COUNT(1) FROM faiss_map")
            return cur.fetchone()[0]
        
    def keyword_search(self, query: str, top_k: int) -> List[Tuple]:
        with self._lock:
            cur = self.conn.cursor()
            
            # First, try exact match on chunk_name (class, function, method names)
            cur.execute("""
                SELECT c.id, c.document_id, c.start_line, c.end_line, c.chunk_type, c.chunk_name, c.content,
                    1000.0 AS score
                FROM chunks c
                WHERE c.chunk_name = ? AND c.chunk_name IS NOT NULL
                ORDER BY c.chunk_type DESC
                LIMIT ?
            """, (query, top_k))
            exact_matches = cur.fetchall()
            
            # If we have exact matches, return them first
            if exact_matches:
                remaining = top_k - len(exact_matches)
                if remaining <= 0:
                    return exact_matches
                
                # Get FTS results excluding exact matches
                exact_ids = [row[0] for row in exact_matches]
                placeholders = ','.join('?' * len(exact_ids))
                cur.execute(f"""
                    SELECT c.id, c.document_id, c.start_line, c.end_line, c.chunk_type, c.chunk_name, c.content,
                        bm25(chunks_fts) AS score
                    FROM chunks_fts
                    JOIN chunks c ON c.id = chunks_fts.rowid
                    WHERE chunks_fts MATCH ? AND c.id NOT IN ({placeholders})
                    ORDER BY score LIMIT ?
                """, (query, *exact_ids, remaining))
                fts_results = cur.fetchall()
                
                return exact_matches + fts_results
            else:
                # No exact matches, use FTS only
                cur.execute("""
                    SELECT c.id, c.document_id, c.start_line, c.end_line, c.chunk_type, c.chunk_name, c.content,
                        bm25(chunks_fts) AS score
                    FROM chunks_fts
                    JOIN chunks c ON c.id = chunks_fts.rowid
                    WHERE chunks_fts MATCH ?
                    ORDER BY score LIMIT ?
                """, (query, top_k))
                return cur.fetchall()

    def get_chunk_by_id(self, chunk_id: int) -> Optional[Tuple]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
            SELECT id, document_id, start_line, end_line, chunk_type, chunk_name, content FROM chunks WHERE id=?
        """, (chunk_id,))
            return cur.fetchone()

    def get_faiss_chunk_mapping(self, faiss_ids: List[int]) -> Dict[int, int]:
        with self._lock:
            cur = self.conn.cursor()
            q = f"SELECT faiss_id, chunk_id FROM faiss_map WHERE faiss_id IN ({','.join('?'*len(faiss_ids))})"
            cur.execute(q, [int(i) for i in faiss_ids])
            return {row[0]: row[1] for row in cur.fetchall()}

    def get_document_info(self, document_id: str) -> Optional[Tuple]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT source_path, language FROM documents WHERE id=?", (document_id,))
            return cur.fetchone()

    def get_document_details(self, document_id: str) -> Optional[Dict]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
            SELECT id, source_path, media_type, language, size_bytes, created_at, extra
            FROM documents WHERE id=?
        """, (document_id,))
            d = cur.fetchone()
            if not d:
                return None
            
            source_path = Path(d[1])
            file_exists = source_path.exists()
            
            cur.execute("""
            SELECT id, ord, start_line, end_line, content FROM chunks WHERE document_id=?
            ORDER BY ord ASC
        """, (document_id,))
            chunks = [{
                "chunk_id": r[0], "ord": r[1], "start_line": r[2], "end_line": r[3], "content": r[4]
            } for r in cur.fetchall()]
            
            return {
                "document": {
                    "id": d[0], "source_path": d[1], "media_type": d[2],
                    "language": d[3], "size_bytes": d[4], "created_at": d[5],
                    "extra": json.loads(d[6] or "{}"),
                    "file_exists": file_exists
                },
                "chunks": chunks
            }

    def get_all_documents(self) -> List[Tuple[str, str, str]]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT id, source_path, sha256 FROM documents")
            return cur.fetchall()

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            self.conn.commit()

    def get_stats(self) -> Dict:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM documents")
            n_docs = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM chunks")
            n_chunks = cur.fetchone()[0]
            return {"documents": n_docs, "chunks": n_chunks}

    def get_methods_by_class(self, class_name: str) -> List[Dict]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
            SELECT c.id, c.document_id, c.start_line, c.end_line, c.content, 
                   c.chunk_name, c.parent_class, d.source_path, d.language
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.parent_class = ? AND c.chunk_type = 'method'
            ORDER BY c.start_line
        """, (class_name,))
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "chunk_id": row[0], "document_id": row[1], "start_line": row[2],
                    "end_line": row[3], "content": row[4], "method_name": row[5],
                    "class_name": row[6], "source_path": row[7], "language": row[8]
                })
            return results

    def get_class_code(self, class_name: str) -> List[Dict]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
            SELECT c.id, c.document_id, c.start_line, c.end_line, c.content,
                   c.chunk_name, d.source_path, d.language
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.chunk_name = ? AND c.chunk_type = 'class'
            ORDER BY c.start_line
        """, (class_name,))
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "chunk_id": row[0], "document_id": row[1], "start_line": row[2],
                    "end_line": row[3], "content": row[4], "class_name": row[5],
                    "source_path": row[6], "language": row[7]
                })
            return results

    def get_method_code(self, method_name: str, class_name: Optional[str] = None) -> List[Dict]:
        with self._lock:
            cur = self.conn.cursor()
            if class_name:
                cur.execute("""
                SELECT c.id, c.document_id, c.start_line, c.end_line, c.content,
                       c.chunk_name, c.parent_class, d.source_path, d.language
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.chunk_name = ? AND c.parent_class = ? AND c.chunk_type = 'method'
                ORDER BY c.start_line
            """, (method_name, class_name))
            else:
                cur.execute("""
                SELECT c.id, c.document_id, c.start_line, c.end_line, c.content,
                       c.chunk_name, c.parent_class, d.source_path, d.language
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.chunk_name = ? AND c.chunk_type IN ('method', 'function')
                ORDER BY c.start_line
            """, (method_name,))
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "chunk_id": row[0], "document_id": row[1], "start_line": row[2],
                    "end_line": row[3], "content": row[4], "method_name": row[5],
                    "class_name": row[6], "source_path": row[7], "language": row[8]
                })
            return results

    def get_function_code(self, function_name: str) -> List[Dict]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
            SELECT c.id, c.document_id, c.start_line, c.end_line, c.content,
                   c.chunk_name, d.source_path, d.language
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.chunk_name = ? AND c.chunk_type = 'function' AND c.parent_class IS NULL
            ORDER BY c.start_line
        """, (function_name,))
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "chunk_id": row[0], "document_id": row[1], "start_line": row[2],
                    "end_line": row[3], "content": row[4], "function_name": row[5],
                    "source_path": row[6], "language": row[7]
                })
            return results

    def get_document_imports(self, document_id: str) -> List[str]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
            SELECT import_statement
            FROM document_imports
            WHERE document_id = ?
            ORDER BY id
        """, (document_id,))
            return [row[0] for row in cur.fetchall()]

    def get_imports_by_file(self, file_path: str) -> List[str]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
            SELECT di.import_statement
            FROM document_imports di
            JOIN documents d ON di.document_id = d.id
            WHERE d.source_path = ?
            ORDER BY di.id
        """, (file_path,))
            return [row[0] for row in cur.fetchall()]

    def list_all_classes(self) -> List[Dict]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
            SELECT DISTINCT c.chunk_name, d.source_path, d.language, c.document_id
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.chunk_type = 'class' AND c.chunk_name IS NOT NULL
            ORDER BY c.chunk_name
        """)
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "class_name": row[0], "source_path": row[1],
                    "language": row[2], "document_id": row[3]
                })
            return results

    def list_all_functions(self) -> List[Dict]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
            SELECT DISTINCT c.chunk_name, d.source_path, d.language, c.document_id
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.chunk_type = 'function' AND c.parent_class IS NULL AND c.chunk_name IS NOT NULL
            ORDER BY c.chunk_name
        """)
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "function_name": row[0], "source_path": row[1],
                    "language": row[2], "document_id": row[3]
                })
            return results

    def get_chunk_metadata(self, chunk_id: int) -> Optional[Dict]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
            SELECT c.id, c.document_id, c.ord, c.start_line, c.end_line, 
                   c.content, c.chunk_type, c.chunk_name, c.parent_class,
                   d.source_path, d.language, d.media_type
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.id = ?
        """, (chunk_id,))
            
            row = cur.fetchone()
            if not row:
                return None
            
            return {
                "chunk_id": row[0], "document_id": row[1], "order": row[2],
                "start_line": row[3], "end_line": row[4], "content": row[5],
                "chunk_type": row[6], "chunk_name": row[7], "parent_class": row[8],
                "source_path": row[9], "language": row[10], "media_type": row[11]
            }

    def search_by_type(self, chunk_type: str, name_pattern: Optional[str] = None) -> List[Dict]:
        with self._lock:
            cur = self.conn.cursor()
            if name_pattern:
                cur.execute("""
                SELECT c.id, c.document_id, c.start_line, c.end_line, c.content,
                       c.chunk_name, c.parent_class, c.chunk_type, d.source_path, d.language
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.chunk_type = ? AND c.chunk_name LIKE ?
                ORDER BY c.chunk_name
            """, (chunk_type, f"%{name_pattern}%"))
            else:
                cur.execute("""
                SELECT c.id, c.document_id, c.start_line, c.end_line, c.content,
                       c.chunk_name, c.parent_class, c.chunk_type, d.source_path, d.language
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.chunk_type = ?
                ORDER BY c.chunk_name
            """, (chunk_type,))
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "chunk_id": row[0], "document_id": row[1], "start_line": row[2],
                    "end_line": row[3], "content": row[4], "name": row[5],
                    "parent_class": row[6], "chunk_type": row[7],
                    "source_path": row[8], "language": row[9]
                })
            return results

    def close(self):
        with self._lock:
            self.conn.close()

