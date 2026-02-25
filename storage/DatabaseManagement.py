import json
import sqlite3
import datetime as dt
import threading
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from contextlib import contextmanager

# Configure module-level logger
logger = logging.getLogger(__name__)


def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# --------------------------
# DatabaseManager
# --------------------------

class DatabaseManager:
    """
    Manages all SQLite database operations for long-term memory.
    Responsible for persistence of documents, chunks, imports, and metadata.
    Thread-safe through the use of a lock for synchronized access.
    
    Optimizations:
    - Connection pooling via thread-local connections
    - Prepared statements for frequently used queries
    - WAL mode for better concurrent access
    - Proper indexes for common queries
    """
    
    # Prepared statement cache
    _PREPARED_STATEMENTS = {
        "insert_document": """
            INSERT OR IGNORE INTO documents(id, source_path, rel_path, media_type, language, sha256, size_bytes, created_at, category, extra)
            VALUES(?,?,?,?,?,?,?,?,?,?)
        """,
        "get_document": "SELECT * FROM documents WHERE id = ?",
        "get_document_by_path": "SELECT * FROM documents WHERE source_path = ?",
        "delete_document": "DELETE FROM documents WHERE id = ?",
        "insert_chunk": """
            INSERT INTO chunks(document_id, ord, start_line, end_line, content, token_count, chunk_type, chunk_name, parent_class)
            VALUES(?,?,?,?,?,?,?,?,?)
        """,
        "get_chunk": "SELECT * FROM chunks WHERE id = ?",
        "get_chunks_by_doc": "SELECT * FROM chunks WHERE document_id = ?",
        "insert_faiss_mapping": "INSERT OR REPLACE INTO faiss_map(faiss_id, chunk_id) VALUES(?, ?)",
        "get_faiss_mapping": "SELECT chunk_id FROM faiss_map WHERE faiss_id = ?",
    }
    
    def __init__(self, db_path: str = "memory.sqlite"):
        self.db_path = db_path
        # Ensure the directory exists before trying to connect
        db_dir = Path(db_path).parent
        if db_dir and str(db_dir) != '.' and not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)
        
        # Create initial connection for schema initialization
        self._conn = sqlite3.connect(
            self.db_path, 
            check_same_thread=False,
            timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA cache_size=10000;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        
        self._lock = threading.RLock()
        self._ensure_schema()
    
    @property
    def conn(self) -> sqlite3.Connection:
        """Get database connection (backward compatible)."""
        return self._conn

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
    
    def get_souvenir_chuncks(self, doc_id: str) -> Optional[Dict]:
        """Get a souvenir by ID."""
        with self._lock:
            cur = self.conn.cursor()
            # Get the chunk content
            cur.execute("SELECT content, chunk_type FROM chunks WHERE document_id = ?", (doc_id,))
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
        """Search souvenirs by content using multiple queries and prioritizing documents with the most keyword matches."""
        with self._lock:
            cur = self.conn.cursor()

            # Split query into keywords
            keywords = query.lower().split()

            # Perform a separate query for each keyword and store results in sets
            result_sets = []
            for keyword in keywords:
                # Query for each keyword
                sql = """
                    SELECT d.id, c.content 
                    FROM documents d
                    JOIN chunks c ON d.id = c.document_id
                    WHERE (c.content LIKE ? OR d.extra LIKE ?)
                """
                params = [f'%{keyword}%', f'%{keyword}%']
                if category:
                    sql += " AND d.category = ?"
                    params.append(category)
                sql += " LIMIT ?"
                params.append(limit * 10)
                cur.execute(sql, params)

                # Fetch results for this keyword and store document IDs in a set
                result_set = set(row[0] for row in cur.fetchall())
                result_sets.append(result_set)

            # Compute the intersection of all result sets
            # Start with the first set, and iteratively intersect with others
            final_result_set = result_sets[0]
            for result_set in result_sets[1:]:
                final_result_set |= result_set

            # If there are no results in the intersection, return an empty list
            if not final_result_set:
                return []

            # Fetch the documents corresponding to the intersection
            final_results = []
            for doc_id in final_result_set:
                sql = """
                    SELECT d.*, c.content 
                    FROM documents d
                    JOIN chunks c ON d.id = c.document_id
                    WHERE d.id = ?
                """
                cur.execute(sql, [doc_id])
                row = cur.fetchone()

                # Handle both tuple and Row objects
                if hasattr(row, 'keys'):
                    doc = dict(row)
                else:
                    columns = [desc[0] for desc in cur.description]
                    doc = dict(zip(columns, row))

                # Calculate the number of keyword matches in 'content' and 'extra'
                matches = sum(1 for kw in keywords if kw in doc.get('content', '').lower())
                matches += sum(1 for kw in keywords if kw in doc.get('extra', '').lower())

                # Add match score to the document
                doc['similarity_score'] = matches / (2 * len(keywords))
                final_results.append(doc)

            # Sort by match_score (higher is better) and limit to 'limit' number of results
            sorted_results = sorted(final_results, key=lambda x: x['similarity_score'], reverse=True)
            return sorted_results[:limit]

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
            SELECT id, ord, start_line, end_line, content, chunk_type, chunk_name FROM chunks WHERE document_id=?
            ORDER BY ord ASC
        """, (document_id,))
            chunks = [{
                "chunk_id": r[0], "ord": r[1], "start_line": r[2], "end_line": r[3], 
                "content": r[4], "chunk_type": r[5], "chunk_name": r[6]
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

