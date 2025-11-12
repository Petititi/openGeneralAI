import json
import sqlite3
import datetime as dt
from pathlib import Path
from typing import List, Dict, Tuple, Optional

def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# --------------------------
# DatabaseManager
# --------------------------

class DatabaseManager:
    """
    Gère toutes les opérations de base de données SQLite pour la mémoire à long terme.
    Responsable de la persistance des documents, chunks, imports et métadonnées.
    """
    def __init__(self, db_path: str = "memory.sqlite"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self._ensure_schema()

    def _ensure_schema(self):
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
            extra JSON
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
        self.conn.commit()

    def insert_document(self, doc_id: str, source_path: str, rel_path: Optional[str],
                       media_type: str, language: Optional[str], sha256: str,
                       size_bytes: int, extra: Optional[Dict] = None) -> None:
        cur = self.conn.cursor()
        cur.execute("""
            INSERT OR IGNORE INTO documents(id, source_path, rel_path, media_type, language, sha256, size_bytes, created_at, extra)
            VALUES(?,?,?,?,?,?,?,?,?)
        """, (doc_id, source_path, rel_path, media_type, language, sha256, size_bytes, now_iso(), json.dumps(extra or {})))
        self.conn.commit()

    def insert_imports(self, doc_id: str, imports: List[str]) -> None:
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
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO chunks(document_id, ord, start_line, end_line, content, chunk_type, chunk_name, parent_class)
            VALUES(?,?,?,?,?,?,?,?)
        """, (doc_id, ord, start_line, end_line, content, chunk_type, chunk_name, parent_class))
        chunk_id = cur.lastrowid
        self.conn.commit()
        return chunk_id

    def get_chunk_content(self, chunk_id: int) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT content FROM chunks WHERE id=?", (chunk_id,))
        row = cur.fetchone()
        return row[0] if row else None

    def get_all_chunks(self) -> List[Tuple[int, str]]:
        cur = self.conn.cursor()
        cur.execute("SELECT id, content FROM chunks ORDER BY id")
        return cur.fetchall()

    def insert_faiss_mapping(self, faiss_id: int, chunk_id: int) -> None:
        cur = self.conn.cursor()
        cur.execute("INSERT INTO faiss_map(faiss_id, chunk_id) VALUES(?,?)", (faiss_id, chunk_id))
        self.conn.commit()

    def clear_faiss_mappings(self) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM faiss_map")
        self.conn.commit()

    def get_faiss_mapping_count(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(1) FROM faiss_map")
        return cur.fetchone()[0]

    def keyword_search(self, query: str, top_k: int) -> List[Tuple]:
        cur = self.conn.cursor()
        cur.execute("""
            SELECT c.id, c.document_id, c.start_line, c.end_line, c.content,
                   bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY score LIMIT ?
        """, (query, top_k))
        return cur.fetchall()

    def get_chunk_by_id(self, chunk_id: int) -> Optional[Tuple]:
        cur = self.conn.cursor()
        cur.execute("""
            SELECT id, document_id, start_line, end_line, content FROM chunks WHERE id=?
        """, (chunk_id,))
        return cur.fetchone()

    def get_faiss_chunk_mapping(self, faiss_ids: List[int]) -> Dict[int, int]:
        cur = self.conn.cursor()
        q = f"SELECT faiss_id, chunk_id FROM faiss_map WHERE faiss_id IN ({','.join('?'*len(faiss_ids))})"
        cur.execute(q, [int(i) for i in faiss_ids])
        return {row[0]: row[1] for row in cur.fetchall()}

    def get_document_info(self, document_id: str) -> Optional[Tuple]:
        cur = self.conn.cursor()
        cur.execute("SELECT source_path, language FROM documents WHERE id=?", (document_id,))
        return cur.fetchone()

    def get_document_details(self, document_id: str) -> Optional[Dict]:
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
        cur = self.conn.cursor()
        cur.execute("SELECT id, source_path, sha256 FROM documents")
        return cur.fetchall()

    def delete_document(self, doc_id: str) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        self.conn.commit()

    def get_stats(self) -> Dict:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM documents")
        n_docs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks")
        n_chunks = cur.fetchone()[0]
        return {"documents": n_docs, "chunks": n_chunks}

    def get_methods_by_class(self, class_name: str) -> List[Dict]:
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
        cur = self.conn.cursor()
        cur.execute("""
            SELECT import_statement
            FROM document_imports
            WHERE document_id = ?
            ORDER BY id
        """, (document_id,))
        return [row[0] for row in cur.fetchall()]

    def get_imports_by_file(self, file_path: str) -> List[str]:
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
        self.conn.close()

