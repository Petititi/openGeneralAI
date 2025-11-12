import os
import sys
import json
import hashlib
import sqlite3
import datetime as dt
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from tree_sitter_language_pack import get_parser

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

def read_text_safe(path: Path, max_bytes: int = 10_000_000) -> bytes:
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return data

# --------------------------
# Chunking via tree-sitter
# --------------------------

def extract_name_from_node(node, source: bytes) -> Optional[str]:
    """Extrait le nom d'une fonction/classe/méthode depuis un noeud tree-sitter."""
    for child in node.children:
        if child.type in {"identifier", "name"}:
            return source[child.start_byte:child.end_byte].decode("utf-8")
    return None

def extract_imports(root_node, source: bytes, _language: str) -> List[str]:
    """Extrait tous les imports/includes d'un fichier."""
    imports = []
    
    IMPORT_TYPES = {
        "import_statement",
        "import_from_statement",
        "preproc_include",
        "using_directive",
        "package_declaration",
    }
    
    def walk_imports(node):
        if node.type in IMPORT_TYPES:
            import_text = source[node.start_byte:node.end_byte].strip()
            imports.append(import_text)
        for child in node.children:
            walk_imports(child)
    
    walk_imports(root_node)
    return imports

def extract_code_chunks(source: bytes, language: str) -> Tuple[List[Tuple[int, int, str, Dict]], List[str], Dict[str, List[int]]]:
    """
    Retourne des chunks (start_line, end_line, text, metadata) pour fonctions/classes.
    metadata contient: {type, name, class_name, parent_class}
    Si rien de structuré trouvé, fallback: chunk par ~120 lignes.
    
    Retourne: (chunks, imports, class_methods_map)
    class_methods_map: Dict[class_name, List[chunk_index]] pour lier classes et méthodes
    """
    try:
        parser = get_parser(language)
    except LookupError as e:  # type: ignore
        print(f"[WARN] extract_code_chunks: pas de parser pour '{language}': {e}", file=sys.stderr)
        # parser non dispo -> fallback lignes
        lines = source.decode("utf-8").splitlines()
        chunks = []
        step = 120
        for i in range(0, len(lines), step):
            part = "\n".join(lines[i:i+step])
            if part.strip():
                chunks.append((i+1, min(i+step, len(lines)), part, {"type": "block"}))
        return chunks, [], {}
    tree = parser.parse(source)
    root = tree.root_node

    # Extraire les imports une fois pour tout le fichier
    imports = extract_imports(root, source, language)

    # Noms de noeuds typiques par langage
    FUNCTION_TYPES = {
        "function_definition",
        "function_declaration",
    }
    
    METHOD_TYPES = {
        "method_definition",
    }
    
    CLASS_TYPES = {
        "class_definition",
        "class_declaration",
        "interface_declaration",
        "struct_specifier",
        "enum_specifier",
    }
    
    MODULE_TYPES = {
        "module_declaration",
    }

    chunks = []
    class_methods_map = {}  # {class_name: [chunk_indices]}
    
    def is_docstring(node):
        if node.type == "string":
            parent = node.parent
            if parent is None or parent.type != "expression_statement":
                return False
            grandparent = parent.parent
            if grandparent is None:
                return False
            # Vérifie que c'est le premier élément du bloc
            first_child = grandparent.children[0] if grandparent.children else None
            return first_child == parent and grandparent.type in ("module", "class_definition", "function_definition")
        elif node.type == "comment" and node.parent is not None:
            parent = node.parent
            next_node = node.next_sibling
            return next_node is not None and next_node.type == "class_definition"

    
    def walk(node, parent_class=None):
        # Identifier le type de noeud
        chunk_type = None
        if node.type in CLASS_TYPES:
            chunk_type = "class"
        elif node.type in METHOD_TYPES:
            chunk_type = "method"
        elif node.type in FUNCTION_TYPES:
            # Différencier méthode (dans une classe) de fonction (standalone)
            if parent_class:
                chunk_type = "method"
            else:
                chunk_type = "function"
        elif node.type in MODULE_TYPES:
            chunk_type = "module"
        elif node.type == "comment":
            if is_docstring(node):
                print('found ' + source[node.start_byte:node.end_byte].decode("utf-8"))
        
        if chunk_type:
            start = node.start_point[0] + 1
            end = node.end_point[0] + 1
            text = source[node.start_byte:node.end_byte].decode("utf-8")
            
            if text.strip():
                name = extract_name_from_node(node, source)
                metadata = {
                    "type": chunk_type,
                    "name": name,
                    "parent_class": parent_class
                }
                chunk_index = len(chunks)
                chunks.append((start, end, text, metadata))
                
                # Si c'est une méthode, l'ajouter à la map de sa classe
                if chunk_type == "method" and parent_class:
                    if parent_class not in class_methods_map:
                        class_methods_map[parent_class] = []
                    class_methods_map[parent_class].append(chunk_index)
                
                # Si c'est une classe, on marque les méthodes enfants avec ce parent
                if chunk_type == "class":
                    parent_class = name
        
        # Continuer la marche récursive
        for c in node.children:
            walk(c, parent_class)

    walk(root)

    if not chunks:
        # fallback: grands blocs
        lines = source.decode("utf-8").splitlines()
        step = 120
        for i in range(0, len(lines), step):
            part = "\n".join(lines[i:i+step])
            if part.strip():
                chunks.append((i+1, min(i+step, len(lines)), part, {"type": "block"}))
        return chunks, imports, {}

    # Limiter la taille de chunk (~1500 tokens équiv) par nombre de caractères
    MAX_CHARS = 6000
    final = []
    for s, e, t, meta in chunks:
        if len(t) <= MAX_CHARS:
            final.append((s, e, t, meta))
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
                    final.append((start, s + idx, segment, {**meta, "type": "partial"}))
                    buf, start, count = [], s + idx + 1, 0
            if buf:
                final.append((start, e, "\n".join(buf), meta))
    
    # Mettre à jour class_methods_map avec les nouveaux indices après split
    # Note: les indices peuvent avoir changé, mais on garde la structure originale
    # car le split ne devrait affecter que les chunks trop longs
    
    # Retourner aussi les imports et le mapping classe->méthodes
    return final, imports, class_methods_map

def extract_text_chunks(source: bytes, max_chars: int = 3000) -> Tuple[List[Tuple[int, int, str, Dict]], List[str], Dict[str, List[int]]]:
    # Split par paragraphes, puis pack jusqu'à ~max_chars
    text = source.decode("utf-8")
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
            chunks.append((start, end, chunk, {"type": "text"}))
            buf, size, start = [], 0, line_counter + 1
        buf.append(p)
        size += len(p) + 2
        line_counter += p.count("\n") + 2
    if buf:
        chunks.append((start, line_counter, "\n\n".join(buf), {"type": "text"}))
    return chunks, [], {}

# --------------------------
# LongTermMemory
# --------------------------

class LongTermMemory:
    """
    Système de mémoire à long terme basé sur l'indexation de fichiers.
    
    Version modifiée pour ne pas copier les fichiers analysés mais stocker
    seulement leurs chemins d'accès. Cela économise l'espace disque mais
    nécessite que les fichiers restent à leur emplacement original.
    
    Fonctionnalités:
    - Indexation de code (via tree-sitter) et de texte
    - Recherche hybride (mots-clés + sémantique)
    - Vérification de l'intégrité des fichiers
    - Nettoyage automatique des références cassées
    """
    def __init__(
        self,
        db_path: str = "memory.sqlite",
        storage_dir: str = "storage",
        faiss_index_path: str = "faiss.index",
        model_name: str = "mixedbread-ai/mxbai-embed-large-v1",
        hybrid_alpha: float = 0.5,
    ):
        self.db_path = db_path
        self.storage_dir = Path(storage_dir)  # Conservé pour compatibilité mais non utilisé
        self.faiss_index_path = faiss_index_path
        self.hybrid_alpha = hybrid_alpha

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
            source_path TEXT NOT NULL,
            rel_path TEXT,
            media_type TEXT,
            language TEXT,
            sha256 TEXT,
            size_bytes INTEGER,
            created_at TEXT,
            extra JSON
        )""")
        # Table pour stocker les imports d'un document
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
        # Index pour rechercher efficacement par type de chunk, nom, classe parente
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
        Ajoute un document fichier : analyse le contenu, crée des chunks, indexe FTS + FAISS.
        Retourne document_id (sha256 du contenu).
        """
        p = Path(path)
        data = p.read_bytes()
        doc_id = sha256_bytes(data)

        # Plus de copie dans storage_dir, on garde seulement le chemin d'accès original

        # Déterminer type/langue
        ext = p.suffix.lower()
        class_methods_map = {}
        if ext in SUPPORTED_CODE_EXT:
            language = SUPPORTED_CODE_EXT[ext]
            media_type = "code"
            text = read_text_safe(p)
            chunks, imports, class_methods_map = extract_code_chunks(text, language)
        elif ext in TEXT_EXT:
            language = None
            media_type = "text"
            text = read_text_safe(p)
            chunks, imports, class_methods_map = extract_text_chunks(text)
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

        # Insérer les imports pour ce document
        for imp in imports:
            cur.execute("""
                INSERT INTO document_imports(document_id, import_statement)
                VALUES(?,?)
            """, (doc_id, imp))

        # Première passe: insérer tous les chunks et calculer les embeddings des non-classes
        chunk_ids = []
        chunk_embeddings = {}
        
        for i, (start, end, content, metadata) in enumerate(chunks):
            cur.execute("""
                INSERT INTO chunks(document_id, ord, start_line, end_line, content, chunk_type, chunk_name, parent_class)
                VALUES(?,?,?,?,?,?,?,?)
            """, (doc_id, i, start, end, content, 
                  metadata.get("type"), metadata.get("name"), metadata.get("parent_class")))
            chunk_id = cur.lastrowid
            chunk_ids.append(chunk_id)
            
            chunk_type = metadata.get("type")
            chunk_name = metadata.get("name")
            
            # Pour les classes, on ne calcule pas encore l'embedding
            if chunk_type == "class":
                chunk_embeddings[i] = None  # Sera calculé plus tard
            else:
                # Embedding normal pour fonctions, méthodes, etc.
                emb = self.model.encode([content], batch_size=1, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
                chunk_embeddings[i] = emb[0]
        
        # Deuxième passe: calculer les embeddings des classes comme somme pondérée des méthodes
        for i, (start, end, content, metadata) in enumerate(chunks):
            chunk_type = metadata.get("type")
            chunk_name = metadata.get("name")
            
            if chunk_type == "class" and chunk_name in class_methods_map:
                method_indices = class_methods_map[chunk_name]
                method_embeddings = []
                method_weights = []
                # Classe sans méthodes ou non trouvée dans la map
                class_emb = self.model.encode([f"This is a {language} class named {chunk_name}"], batch_size=1, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
                method_embeddings.append(class_emb[0])
                method_weights.append(500) # arbitrary weight for class description
                
                # Collecter les embeddings des méthodes de cette classe avec leurs tailles
                for method_idx in method_indices:
                    if method_idx < len(chunk_embeddings) and chunk_embeddings[method_idx] is not None:
                        method_embeddings.append(chunk_embeddings[method_idx])
                        # Poids basé sur la taille en bytes du contenu de la méthode
                        method_content = chunks[method_idx][2]
                        method_weights.append(len(method_content.encode('utf-8')))
                
                if method_embeddings:
                    # Calculer la moyenne pondérée par la taille (bytes) de chaque méthode
                    method_embeddings_array = np.array(method_embeddings)
                    method_weights_array = np.array(method_weights, dtype="float32")
                    
                    # Normaliser les poids pour qu'ils somment à 1
                    weights_sum = np.sum(method_weights_array)
                    if weights_sum > 0:
                        normalized_weights = method_weights_array / weights_sum
                        # Moyenne pondérée: somme des embeddings multipliés par leurs poids
                        class_emb = np.sum(method_embeddings_array * normalized_weights[:, np.newaxis], axis=0).astype("float32")
                    else:
                        # Fallback sur moyenne simple si problème avec les poids
                        class_emb = np.mean(method_embeddings_array, axis=0).astype("float32")
                    
                    # Re-normaliser
                    norm = np.linalg.norm(class_emb)
                    if norm > 1e-12:
                        class_emb = class_emb / norm
                    chunk_embeddings[i] = class_emb
                else:
                    # Pas de méthodes trouvées, utiliser l'embedding du contenu de la classe
                    emb = self.model.encode([content], batch_size=1, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
                    chunk_embeddings[i] = emb[0]
            elif chunk_type == "class":
                # Classe sans méthodes ou non trouvée dans la map
                emb = self.model.encode([content], batch_size=1, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
                chunk_embeddings[i] = emb[0]
        
        # Troisième passe: ajouter tous les embeddings à FAISS
        for i, chunk_id in enumerate(chunk_ids):
            if i in chunk_embeddings and chunk_embeddings[i] is not None:
                emb = chunk_embeddings[i].reshape(1, -1)
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
            except Exception as e:  # type: ignore
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
            q_emb = self.model.encode([query], prompt_name="query", convert_to_numpy=True, normalize_embeddings=True).astype("float32")
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
        
        # Vérifier que le fichier source existe encore
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

    def check_file_integrity(self) -> Dict:
        """
        Vérifie l'intégrité des fichiers indexés.
        Retourne des statistiques sur les fichiers existants/manquants/modifiés.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT id, source_path, sha256 FROM documents")
        docs = cur.fetchall()
        
        existing = 0
        missing = 0
        modified = 0
        
        for _doc_id, source_path, expected_sha256 in docs:
            path = Path(source_path)
            if not path.exists():
                missing += 1
            else:
                try:
                    current_data = path.read_bytes()
                    current_sha256 = sha256_bytes(current_data)
                    if current_sha256 == expected_sha256:
                        existing += 1
                    else:
                        modified += 1
                except Exception:  # type: ignore
                    missing += 1
        
        return {
            "total_documents": len(docs),
            "existing_unchanged": existing,
            "missing": missing,
            "modified": modified
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
            "embedding_dim": int(self.dim) if self.dim is not None else 0
        }

    def cleanup_missing_files(self) -> int:
        """
        Supprime de la base de données les références vers les fichiers qui n'existent plus.
        Retourne le nombre de documents supprimés.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT id, source_path FROM documents")
        docs = cur.fetchall()
        
        deleted_count = 0
        for doc_id, source_path in docs:
            if not Path(source_path).exists():
                # Supprimer le document (les chunks et entrées FAISS seront supprimés en cascade)
                cur.execute("DELETE FROM documents WHERE id=?", (doc_id,))
                deleted_count += 1
        
        if deleted_count > 0:
            self.conn.commit()
            # Reconstruire l'index FAISS pour éliminer les vecteurs orphelins
            self._rebuild_faiss_index()
        
        return deleted_count

    def _rebuild_faiss_index(self):
        """Reconstruit l'index FAISS à partir des chunks restants."""
        # Créer un nouvel index
        self.index = faiss.IndexFlatIP(self.dim)
        
        # Récupérer tous les chunks restants
        cur = self.conn.cursor()
        cur.execute("SELECT id, content FROM chunks ORDER BY id")
        chunks = cur.fetchall()
        
        # Vider la table de mapping
        cur.execute("DELETE FROM faiss_map")
        
        # Réencoder et réindexer tous les chunks
        if chunks:
            contents = [chunk[1] for chunk in chunks]
            embeddings = self.model.encode(contents, convert_to_numpy=True, normalize_embeddings=True)
            embeddings = embeddings.astype("float32")
            
            # Ajouter à FAISS
            self.index.add(embeddings)
            
            # Mettre à jour la table de mapping
            for i, (chunk_id, _) in enumerate(chunks):
                cur.execute("INSERT INTO faiss_map(faiss_id, chunk_id) VALUES(?,?)", (i, chunk_id))
        
        self.conn.commit()
        self._persist_faiss()

    def get_methods_by_class(self, class_name: str) -> List[Dict]:
        """
        Retourne toutes les méthodes d'une classe spécifique.
        """
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
                "chunk_id": row[0],
                "document_id": row[1],
                "start_line": row[2],
                "end_line": row[3],
                "content": row[4],
                "method_name": row[5],
                "class_name": row[6],
                "source_path": row[7],
                "language": row[8]
            })
        return results

    def get_class_code(self, class_name: str) -> List[Dict]:
        """
        Retourne le code complet d'une classe spécifique.
        """
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
                "chunk_id": row[0],
                "document_id": row[1],
                "start_line": row[2],
                "end_line": row[3],
                "content": row[4],
                "class_name": row[5],
                "source_path": row[6],
                "language": row[7]
            })
        return results

    def get_method_code(self, method_name: str, class_name: Optional[str] = None) -> List[Dict]:
        """
        Retourne le code d'une méthode spécifique. 
        Si class_name est fourni, recherche dans cette classe uniquement.
        """
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
                "chunk_id": row[0],
                "document_id": row[1],
                "start_line": row[2],
                "end_line": row[3],
                "content": row[4],
                "method_name": row[5],
                "class_name": row[6],
                "source_path": row[7],
                "language": row[8]
            })
        return results

    def get_function_code(self, function_name: str) -> List[Dict]:
        """
        Retourne le code d'une fonction spécifique (pas de classe parente).
        """
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
                "chunk_id": row[0],
                "document_id": row[1],
                "start_line": row[2],
                "end_line": row[3],
                "content": row[4],
                "function_name": row[5],
                "source_path": row[6],
                "language": row[7]
            })
        return results

    def get_document_imports(self, document_id: str) -> List[str]:
        """
        Retourne tous les imports d'un document spécifique.
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT import_statement
            FROM document_imports
            WHERE document_id = ?
            ORDER BY id
        """, (document_id,))
        
        return [row[0] for row in cur.fetchall()]

    def get_imports_by_file(self, file_path: str) -> List[str]:
        """
        Retourne tous les imports d'un fichier par son chemin.
        """
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
        """
        Liste toutes les classes indexées avec leurs informations.
        """
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
                "class_name": row[0],
                "source_path": row[1],
                "language": row[2],
                "document_id": row[3]
            })
        return results

    def list_all_functions(self) -> List[Dict]:
        """
        Liste toutes les fonctions indexées (hors méthodes de classe).
        """
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
                "function_name": row[0],
                "source_path": row[1],
                "language": row[2],
                "document_id": row[3]
            })
        return results

    def get_chunk_metadata(self, chunk_id: int) -> Optional[Dict]:
        """
        Retourne toutes les métadonnées d'un chunk spécifique.
        """
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
            "chunk_id": row[0],
            "document_id": row[1],
            "order": row[2],
            "start_line": row[3],
            "end_line": row[4],
            "content": row[5],
            "chunk_type": row[6],
            "chunk_name": row[7],
            "parent_class": row[8],
            "source_path": row[9],
            "language": row[10],
            "media_type": row[11]
        }

    def search_by_type(self, chunk_type: str, name_pattern: Optional[str] = None) -> List[Dict]:
        """
        Recherche des chunks par type (class, method, function, etc.) 
        avec option de filtre par pattern de nom.
        """
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
                "chunk_id": row[0],
                "document_id": row[1],
                "start_line": row[2],
                "end_line": row[3],
                "content": row[4],
                "name": row[5],
                "parent_class": row[6],
                "chunk_type": row[7],
                "source_path": row[8],
                "language": row[9]
            })
        return results

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

    classes = ltm.list_all_classes()
    for cls in classes[:10]:  # Top 10
        print(f"   - {cls['class_name']:30} [{cls['language']:10}] {cls['source_path']}")


    res = ltm.search(query, top_k=8, mode=mode)
    for i, r in enumerate(res, 1):
        print(f"\n[{i}] score={r['score']:.4f}  {r['source_path']}:{r['start_line']}-{r['end_line']}")
        print("-" * 80)
        snippet = r["content"]
        linebreak = snippet.find("\n")
        if linebreak > 0:
            print(snippet[:linebreak] + "...")
        else:
            print(snippet[:100] + ("..." if len(snippet) > 100 else ""))
    ltm.close()

def _cmd_integrity():
    ltm = LongTermMemory()
    integrity = ltm.check_file_integrity()
    print(json.dumps(integrity, indent=2))
    ltm.close()

def _cmd_cleanup():
    ltm = LongTermMemory()
    deleted = ltm.cleanup_missing_files()
    print(f"Supprimé {deleted} documents avec fichiers manquants")
    print(json.dumps(ltm.stats(), indent=2))
    ltm.close()

def _cmd_list_classes():
    ltm = LongTermMemory()
    classes = ltm.list_all_classes()
    print(f"Found {len(classes)} classes:")
    for cls in classes:
        print(f"  - {cls['class_name']} in {cls['source_path']} ({cls['language']})")
    ltm.close()

def _cmd_list_functions():
    ltm = LongTermMemory()
    functions = ltm.list_all_functions()
    print(f"Found {len(functions)} functions:")
    for func in functions:
        print(f"  - {func['function_name']} in {func['source_path']} ({func['language']})")
    ltm.close()

def _cmd_get_class(class_name: str):
    ltm = LongTermMemory()
    results = ltm.get_class_code(class_name)
    if not results:
        print(f"No class found with name: {class_name}")
    else:
        for r in results:
            print(f"\n{'='*80}")
            print(f"Class: {r['class_name']}")
            print(f"File: {r['source_path']}:{r['start_line']}-{r['end_line']}")
            print(f"Language: {r['language']}")
            print(f"{'='*80}")
            print(r['content'])
    ltm.close()

def _cmd_get_methods(class_name: str):
    ltm = LongTermMemory()
    methods = ltm.get_methods_by_class(class_name)
    if not methods:
        print(f"No methods found for class: {class_name}")
    else:
        print(f"Found {len(methods)} methods in class {class_name}:")
        for m in methods:
            print(f"\n{'-'*80}")
            print(f"Method: {m['method_name']} (lines {m['start_line']}-{m['end_line']})")
            print(f"File: {m['source_path']}")
            print(f"{'-'*80}")
            print(m['content'][:500] + ("..." if len(m['content']) > 500 else ""))
    ltm.close()

def _cmd_get_function(function_name: str):
    ltm = LongTermMemory()
    results = ltm.get_function_code(function_name)
    if not results:
        print(f"No function found with name: {function_name}")
    else:
        for r in results:
            print(f"\n{'='*80}")
            print(f"Function: {r['function_name']}")
            print(f"File: {r['source_path']}:{r['start_line']}-{r['end_line']}")
            print(f"Language: {r['language']}")
            print(f"{'='*80}")
            print(r['content'])
    ltm.close()

def _cmd_get_imports(file_path: str):
    ltm = LongTermMemory()
    imports = ltm.get_imports_by_file(file_path)
    if not imports:
        print(f"No imports found for file: {file_path}")
    else:
        print(f"Imports in {file_path}:")
        for imp in imports:
            print(f"  {imp}")
    ltm.close()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    ap_idx = sub.add_parser("index", help="Indexer un dossier (stocke seulement les chemins)")
    ap_idx.add_argument("folder", type=str)

    ap_s = sub.add_parser("search", help="Rechercher")
    ap_s.add_argument("query", type=str)
    ap_s.add_argument("--mode", type=str, default="hybrid", choices=["keyword", "semantic", "hybrid"])

    ap_int = sub.add_parser("integrity", help="Vérifier l'intégrité des fichiers indexés")
    
    ap_clean = sub.add_parser("cleanup", help="Supprimer les références vers les fichiers manquants")

    ap_list_cls = sub.add_parser("list-classes", help="Lister toutes les classes indexées")
    
    ap_list_fn = sub.add_parser("list-functions", help="Lister toutes les fonctions indexées")
    
    ap_get_cls = sub.add_parser("get-class", help="Obtenir le code d'une classe")
    ap_get_cls.add_argument("class_name", type=str)
    
    ap_get_methods = sub.add_parser("get-methods", help="Obtenir toutes les méthodes d'une classe")
    ap_get_methods.add_argument("class_name", type=str)
    
    ap_get_fn = sub.add_parser("get-function", help="Obtenir le code d'une fonction")
    ap_get_fn.add_argument("function_name", type=str)
    
    ap_get_imp = sub.add_parser("get-imports", help="Obtenir les imports d'un fichier")
    ap_get_imp.add_argument("file_path", type=str)

    args = ap.parse_args()
    if args.cmd == "index":
        _cmd_index(args.folder)
    elif args.cmd == "search":
        _cmd_search(args.query, args.mode)
    elif args.cmd == "integrity":
        _cmd_integrity()
    elif args.cmd == "cleanup":
        _cmd_cleanup()
    elif args.cmd == "list-classes":
        _cmd_list_classes()
    elif args.cmd == "list-functions":
        _cmd_list_functions()
    elif args.cmd == "get-class":
        _cmd_get_class(args.class_name)
    elif args.cmd == "get-methods":
        _cmd_get_methods(args.class_name)
    elif args.cmd == "get-function":
        _cmd_get_function(args.function_name)
    elif args.cmd == "get-imports":
        _cmd_get_imports(args.file_path)
    else:
        ap.print_help()
