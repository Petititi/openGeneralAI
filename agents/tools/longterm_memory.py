import os
import io
import sys
import json
import time
import math
import hashlib
import shutil
import sqlite3
import datetime as dt
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from tree_sitter_languages import get_parser  # bundlé (py, js, ts, c, cpp, java, go, rust, etc.)

# --------------------------
# Utilitaires
# --------------------------

SUPPORTED_CODE_EXT = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".php": "php",
    ".rb": "ruby",
    ".kt": "kotlin",
    ".swift": "swift",
    ".m": "objective-c",
    ".mm": "objective-cpp",
    ".cs": "c_sharp",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".sql": "sql",
    ".r": "r",
    ".scala": "scala",
    ".hs": "haskell",
}

TEXT_EXT = {".md", ".txt", ".rst", ".log", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".json"}

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def normalize(v: np.ndarray) -> np.ndarray:
    # L2-normalize rows
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms

def read_text_safe(path: Path, max_bytes: int = 10_000_000) -> str:
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")

# --------------------------
# Chunking via tree-sitter
# --------------------------

def extract_code_chunks(source: str, language: str) -> List[Tuple[int, int, str]]:
    """
    Retourne des chunks (start_line, end_line, text) pour fonctions/classes.
    Si rien de structuré trouvé, fallback: chunk par ~120 lignes.
    """
    try:
        parser = get_parser(language)
    except Exception:
        # parser non dispo -> fallback lignes
        lines = source.splitlines()
        chunks = []
        step = 120
        for i in range(0, len(lines), step):
            part = "\n".join(lines[i:i+step])
            if part.strip():
                chunks.append((i+1, min(i+step, len(lines)), part))
        return chunks

    tree = parser.parse(bytes(source, "utf-8"))
    root = tree.root_node

    # Noms de noeuds typiques par langage
    CANDIDATE_TYPES = {
        "function_definition",
        "function_declaration",
        "method_definition",
        "class_definition",
        "class_declaration",
        "interface_declaration",
        "struct_specifier",
        "enum_specifier",
        "module_declaration",
    }

    chunks = []
    def walk(node):
        # Sélectionne les noeuds "importants"
        if node.type in CANDIDATE_TYPES:
            start = node.start_point[0] + 1
            end = node.end_point[0] + 1
            text = source[node.start_byte:node.end_byte]
            if text.strip():
                chunks.append((start, end, text))
        for c in node.children:
            walk(c)

    walk(root)

    if not chunks:
        # fallback: grands blocs
        lines = source.splitlines()
        step = 120
        for i in range(0, len(lines), step):
            part = "\n".join(lines[i:i+step])
            if part.strip():
                chunks.append((i+1, min(i+step, len(lines)), part))

    # Limiter la taille de chunk (~1500 tokens équiv) par nombre de caractères
    MAX_CHARS = 6000
    final = []
    for s, e, t in chunks:
        if len(t) <= MAX_CHARS:
            final.append((s, e, t))
        else:
            # re-split par lignes
            lines = t.splitlines()
            buf, start = [], s
            count = 0
            for idx, line in enumerate(lines):
                buf.append(line)
                count += len(line) + 1
                if count >= MAX_CHARS:
                    segment = "\n".join(buf)
                    final.append((start, s + idx, segment))
                    buf, start, count = [], s + idx + 1, 0
            if buf:
                final.append((start, e, "\n".join(buf)))
    return final

def extract_text_chunks(text: str, max_chars: int = 3000) -> List[Tuple[int, int, str]]:
    # Split par paragraphes, puis pack jusqu’à ~max_chars
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buf = []
    size = 0
    start = 1
    line_counter = 1
    for p in paras:
        if size + len(p) + 2 > max_chars and buf:
            chunk = "\n\n".join(buf)
            end = line_counter
            chunks.append((start, end, chunk))
            buf, size, start = [], 0, line_counter + 1
        buf.append(p)
        size += len(p) + 2
        line_counter += p.count("\n") + 2
    if buf:
        chunks.append((start, line_counter, "\n\n".join(buf)))
    return chunks

# --------------------------
# LongTermMemory
# --------------------------

class LongTermMemory:
    def __init__(
        self,
        db_path: str = "memory.sqlite",
        storage_dir: str = "storage",
        faiss_index_path: str = "faiss.index",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hybrid_alpha: float = 0.5,
    ):
        self.db_path = db_path
        self.storage_dir = Path(storage_dir)
        self.faiss_index_path = faiss_index_path
        self.hybrid_alpha = hybrid_alpha

        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self._ensure_schema()

        # Embedding model
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

        # FAISS index (cosine via dot-product sur vecteurs normalisés)
        self.index = faiss.IndexFlatIP(self.dim)
        self._load_faiss()

    # ---------- Schema & FAISS ----------

    def _ensure_schema(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            source_path TEXT,
            rel_path TEXT,
            media_type TEXT,
            language TEXT,
            sha256 TEXT,
            size_bytes INTEGER,
            created_at TEXT,
            extra JSON
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
            ord INTEGER,
            start_line INTEGER,
            end_line INTEGER,
            content TEXT,
            token_count INTEGER DEFAULT NULL
        )""")
        # FTS5 avec contenu externe pour sync facile
        cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            content, 
            chunk_id UNINDEXED, 
            document_id UNINDEXED, 
            tokenize='porter',
            content='chunks',
            content_rowid='id'
        )""")
        # Triggers pour garder FTS en phase
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
        self.conn.commit()

    def _load_faiss(self):
        # Recharger un index existant si présent, sinon construire à partir de la DB
        if os.path.exists(self.faiss_index_path):
            self.index = faiss.read_index(self.faiss_index_path)
            return

        # (Re)construire depuis zéro si faiss_map non vide sans index — cas rare
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(1) FROM faiss_map")
        if cur.fetchone()[0] == 0:
            # Pas de vecteurs encore
            return
        # Si on a une map mais pas l’index, on reconstruit à partir des embeddings stockés ?
        # Ici, par simplicité, on considère que l’on recompute lors du prochain add() si besoin.
        # (Alternative: stocker les embeddings en BLOB pour reconstruction.)
        return

    def _persist_faiss(self):
        faiss.write_index(self.index, self.faiss_index_path)

    # ---------- API publique ----------

    def add(self, path: str, rel_to: Optional[str] = None, extra: Optional[Dict] = None) -> str:
        """
        Ajoute un document fichier : copie le brut, crée des chunks, indexe FTS + FAISS.
        Retourne document_id (sha256).
        """
        p = Path(path)
        data = p.read_bytes()
        doc_id = sha256_bytes(data)

        # Copie content-addressed
        sub = self.storage_dir / doc_id[:2]
        sub.mkdir(parents=True, exist_ok=True)
        raw_path = sub / doc_id
        if not raw_path.exists():
            raw_path.write_bytes(data)

        # Déterminer type/langue
        ext = p.suffix.lower()
        if ext in SUPPORTED_CODE_EXT:
            language = SUPPORTED_CODE_EXT[ext]
            media_type = "code"
            text = read_text_safe(p)
            chunks = extract_code_chunks(text, language)
        elif ext in TEXT_EXT:
            language = None
            media_type = "text"
            text = read_text_safe(p)
            chunks = extract_text_chunks(text)
        else:
            # Fallback: on ignore les binaires ici pour rester simple
            raise ValueError(f"Type de fichier non géré pour l’indexation: {ext}")

        rel_path = str(Path(rel_to).resolve().joinpath(p).resolve()) if rel_to else None

        cur = self.conn.cursor()
        cur.execute("""
            INSERT OR IGNORE INTO documents(id, source_path, rel_path, media_type, language, sha256, size_bytes, created_at, extra)
            VALUES(?,?,?,?,?,?,?,?,?)
        """, (
            doc_id, str(p.resolve()), rel_path, media_type, language, doc_id, len(data), now_iso(), json.dumps(extra or {})
        ))

        # Insérer chunks
        for i, (start, end, content) in enumerate(chunks):
            cur.execute("""
                INSERT INTO chunks(document_id, ord, start_line, end_line, content)
                VALUES(?,?,?,?,?)
            """, (doc_id, i, start, end, content))
            chunk_id = cur.lastrowid

            # Embedding & FAISS
            emb = self.model.encode([content], batch_size=1, convert_to_numpy=True, normalize_embeddings=True)
            # emb est déjà normalisé par sentence-transformers si normalize_embeddings=True,
            # mais normalisons pour être sûrs si l’impl change :
            emb = normalize(emb.astype("float32"))
            faiss_id = self.index.ntotal
            self.index.add(emb)
            cur.execute("INSERT INTO faiss_map(faiss_id, chunk_id) VALUES(?,?)", (faiss_id, chunk_id))

        self.conn.commit()
        self._persist_faiss()
        return doc_id

    def add_folder(self, folder: str):
        """
        Parcourt un dossier, indexe les fichiers supportés (code + texte).
        """
        root = Path(folder)
        files = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            if ext in SUPPORTED_CODE_EXT or ext in TEXT_EXT:
                files.append(path)

        # Encodage batch possible, mais on garde simple & robuste (un par un)
        for f in files:
            try:
                self.add(str(f), rel_to=str(root))
            except Exception as e:
                print(f"[WARN] Skip {f}: {e}", file=sys.stderr)

    def search(self, query: str, top_k: int = 8, mode: str = "hybrid") -> List[Dict]:
        """
        mode in {"keyword", "semantic", "hybrid"}
        Retour: liste de dicts {chunk_id, document_id, score, start_line, end_line, content, source_path, language}
        """
        mode = mode.lower()
        if mode not in {"keyword", "semantic", "hybrid"}:
            mode = "hybrid"

        kw_results = []
        sem_results = []

        cur = self.conn.cursor()

        if mode in {"keyword", "hybrid"}:
            # FTS5: bm25-like; on prend un peu plus puis on tronque
            cur.execute("""
                SELECT c.id, c.document_id, c.start_line, c.end_line, c.content,
                       bm25(chunks_fts) AS score
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY score LIMIT ?
            """, (query, top_k * 3))
            kw_results = cur.fetchall()

        if mode in {"semantic", "hybrid"}:
            q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
            sims, ids = self.index.search(q_emb, top_k * 3)  # dot product (cosine) scores
            ids = ids[0]
            sims = sims[0]
            # map faiss_id -> chunk
            if len(ids) > 0 and ids[0] != -1:
                q = f"SELECT faiss_id, chunk_id FROM faiss_map WHERE faiss_id IN ({','.join('?'*len(ids))})"
                cur.execute(q, [int(i) for i in ids])
                mapping = {row[0]: row[1] for row in cur.fetchall()}
                for faiss_id, sim in zip(ids, sims):
                    if faiss_id == -1: 
                        continue
                    chunk_id = mapping.get(int(faiss_id))
                    if chunk_id is None:
                        continue
                    cur.execute("""
                        SELECT id, document_id, start_line, end_line, content FROM chunks WHERE id=?
                    """, (chunk_id,))
                    row = cur.fetchone()
                    if row:
                        sem_results.append((*row, float(sim)))

        def assemble(rows):
            out = []
            for cid, did, s, e, content, score in rows:
                cur.execute("SELECT source_path, language FROM documents WHERE id=?", (did,))
                d = cur.fetchone()
                out.append({
                    "chunk_id": int(cid),
                    "document_id": did,
                    "start_line": int(s),
                    "end_line": int(e),
                    "content": content,
                    "score": float(score),
                    "source_path": d[0] if d else None,
                    "language": d[1] if d else None
                })
            return out

        if mode == "keyword":
            return assemble(kw_results)[:top_k]
        if mode == "semantic":
            return assemble(sem_results)[:top_k]

        # HYBRID: fusion par min-max normalization + alpha
        def norm_scores(lst):
            if not lst:
                return {}
            scores = [x[-1] for x in lst]
            lo, hi = min(scores), max(scores)
            if hi - lo < 1e-12:
                return {x[0]: 1.0 for x in lst}
            return {x[0]: (x[-1] - lo) / (hi - lo) for x in lst}

        kw_norm = norm_scores(kw_results)
        sem_norm = norm_scores(sem_results)

        # rassemblement par chunk_id
        all_ids = set(kw_norm.keys()) | set(sem_norm.keys())
        fused = []
        for cid in all_ids:
            k = kw_norm.get(cid, 0.0)
            s = sem_norm.get(cid, 0.0)
            score = self.hybrid_alpha * k + (1 - self.hybrid_alpha) * s
            # récupérer la ligne correspondante (depuis sem ou kw, au choix)
            row = None
            for r in kw_results:
                if r[0] == cid:
                    row = r
                    break
            if row is None:
                for r in sem_results:
                    if r[0] == cid:
                        row = r
                        break
            if row:
                fused.append((*row[:-1], score))

        fused.sort(key=lambda r: r[-1], reverse=True)
        return assemble(fused[:top_k])

    def get(self, document_id: str) -> Dict:
        cur = self.conn.cursor()
        cur.execute("""
            SELECT id, source_path, media_type, language, size_bytes, created_at, extra
            FROM documents WHERE id=?
        """, (document_id,))
        d = cur.fetchone()
        if not d:
            return {}
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
                "extra": json.loads(d[6] or "{}")
            },
            "chunks": chunks
        }

    def stats(self) -> Dict:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM documents")
        n_docs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks")
        n_chunks = cur.fetchone()[0]
        return {
            "documents": n_docs,
            "chunks": n_chunks,
            "faiss_ntotal": int(self.index.ntotal),
            "embedding_dim": int(self.dim)
        }

    def close(self):
        self._persist_faiss()
        self.conn.close()

# --------------------------
# CLI simple
# --------------------------

def _cmd_index(folder: str):
    ltm = LongTermMemory()
    ltm.add_folder(folder)
    print(json.dumps(ltm.stats(), indent=2))
    ltm.close()

def _cmd_search(query: str, mode: str = "hybrid"):
    ltm = LongTermMemory()
    res = ltm.search(query, top_k=8, mode=mode)
    for i, r in enumerate(res, 1):
        print(f"\n[{i}] score={r['score']:.4f}  {r['source_path']}:{r['start_line']}-{r['end_line']}")
        print("-" * 80)
        snippet = r["content"]
        print(snippet[:800] + ("..." if len(snippet) > 800 else ""))
    ltm.close()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    ap_idx = sub.add_parser("index", help="Indexer un dossier")
    ap_idx.add_argument("folder", type=str)

    ap_s = sub.add_parser("search", help="Rechercher")
    ap_s.add_argument("query", type=str)
    ap_s.add_argument("--mode", type=str, default="hybrid", choices=["keyword", "semantic", "hybrid"])

    args = ap.parse_args()
    if args.cmd == "index":
        _cmd_index(args.folder)
    elif args.cmd == "search":
        _cmd_search(args.query, args.mode)
    else:
        ap.print_help()
