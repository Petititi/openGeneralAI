import os
import sys
import json
import hashlib
import datetime as dt
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

from tree_sitter_language_pack import get_parser

from .DatabaseManagement import DatabaseManager
from .EmbeddingManagement import EmbeddingManager

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
        self.storage_dir = Path(storage_dir)  # Conservé pour compatibilité
        self.hybrid_alpha = hybrid_alpha

        # Initialiser les managers
        self.db = DatabaseManager(db_path)
        self.embeddings = EmbeddingManager(faiss_index_path, model_name)

    # Propriétés de compatibilité pour accès legacy
    @property
    def conn(self):
        return self.db.conn
    
    @property
    def index(self):
        return self.embeddings.index
    
    @property
    def model(self):
        return self.embeddings.model
    
    @property
    def dim(self):
        return self.embeddings.dim

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

        # Insérer le document
        self.db.insert_document(
            doc_id, str(p.resolve()), rel_path, media_type, language, doc_id, len(data), extra
        )

        # Insérer les imports
        self.db.insert_imports(doc_id, imports)

        # Première passe: insérer tous les chunks et calculer les embeddings des non-classes
        chunk_ids = []
        chunk_embeddings = {}
        
        for i, (start, end, content, metadata) in enumerate(chunks):
            chunk_id = self.db.insert_chunk(
                doc_id, i, start, end, content,
                metadata.get("type"), metadata.get("name"), metadata.get("parent_class")
            )
            chunk_ids.append(chunk_id)
            
            chunk_type = metadata.get("type")
            chunk_name = metadata.get("name")
            
            # Pour les classes, on ne calcule pas encore l'embedding
            if chunk_type == "class":
                chunk_embeddings[i] = None  # Sera calculé plus tard
            else:
                # Embedding normal pour fonctions, méthodes, etc.
                emb = self.embeddings.encode([content], normalize=True)
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
                class_emb = self.embeddings.encode([f"This is a {language} class named {chunk_name}"], normalize=True)
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
                    emb = self.embeddings.encode([content], normalize=True)
                    chunk_embeddings[i] = emb[0]
            elif chunk_type == "class":
                # Classe sans méthodes ou non trouvée dans la map
                emb = self.embeddings.encode([content], normalize=True)
                chunk_embeddings[i] = emb[0]
        
        # Troisième passe: ajouter tous les embeddings à FAISS
        for i, chunk_id in enumerate(chunk_ids):
            if i in chunk_embeddings and chunk_embeddings[i] is not None:
                emb = chunk_embeddings[i].reshape(1, -1)
                faiss_ids = self.embeddings.add_embeddings(emb)
                self.db.insert_faiss_mapping(faiss_ids[0], chunk_id)

        self.embeddings.persist()
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
            kw_results = self.db.keyword_search(query, top_k * 3)

        if mode in {"semantic", "hybrid"}:
            q_emb = self.embeddings.encode([query], normalize=True, prompt_name="query")
            sims, ids = self.embeddings.search(q_emb, top_k * 3)
            ids = ids[0]
            sims = sims[0]
            # map faiss_id -> chunk
            if len(ids) > 0 and ids[0] != -1:
                mapping = self.db.get_faiss_chunk_mapping([int(i) for i in ids])
                for faiss_id, sim in zip(ids, sims):
                    if faiss_id == -1: 
                        continue
                    chunk_id = mapping.get(int(faiss_id))
                    if chunk_id is None:
                        continue
                    row = self.db.get_chunk_by_id(chunk_id)
                    if row:
                        sem_results.append((*row, float(sim)))

        def assemble(rows):
            out = []
            for cid, did, s, e, content, score in rows:
                d = self.db.get_document_info(did)
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
        result = self.db.get_document_details(document_id)
        return result if result else {}

    def check_file_integrity(self) -> Dict:
        """
        Vérifie l'intégrité des fichiers indexés.
        Retourne des statistiques sur les fichiers existants/manquants/modifiés.
        """
        docs = self.db.get_all_documents()
        
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
        db_stats = self.db.get_stats()
        return {
            **db_stats,
            "faiss_ntotal": self.embeddings.ntotal,
            "embedding_dim": self.embeddings.embedding_dim
        }

    def cleanup_missing_files(self) -> int:
        """
        Supprime de la base de données les références vers les fichiers qui n'existent plus.
        Retourne le nombre de documents supprimés.
        """
        docs = self.db.get_all_documents()
        
        deleted_count = 0
        for doc_id, source_path, _sha256 in docs:
            if not Path(source_path).exists():
                # Supprimer le document (les chunks et entrées FAISS seront supprimés en cascade)
                self.db.delete_document(doc_id)
                deleted_count += 1
        
        if deleted_count > 0:
            # Reconstruire l'index FAISS pour éliminer les vecteurs orphelins
            self._rebuild_faiss_index()
        
        return deleted_count

    def _rebuild_faiss_index(self):
        """Reconstruit l'index FAISS à partir des chunks restants."""
        # Récupérer tous les chunks restants
        chunks = self.db.get_all_chunks()
        
        # Vider la table de mapping
        self.db.clear_faiss_mappings()
        
        # Réencoder et réindexer tous les chunks
        if chunks:
            contents = [chunk[1] for chunk in chunks]
            embeddings = self.embeddings.encode(contents, normalize=True)
            
            # Reconstruire l'index
            self.embeddings.rebuild_index(embeddings)
            
            # Mettre à jour la table de mapping
            for i, (chunk_id, _) in enumerate(chunks):
                self.db.insert_faiss_mapping(i, chunk_id)
        
        self.embeddings.persist()

    # Méthodes de délégation vers DatabaseManager
    def get_methods_by_class(self, class_name: str) -> List[Dict]:
        return self.db.get_methods_by_class(class_name)

    def get_class_code(self, class_name: str) -> List[Dict]:
        return self.db.get_class_code(class_name)

    def get_method_code(self, method_name: str, class_name: Optional[str] = None) -> List[Dict]:
        return self.db.get_method_code(method_name, class_name)

    def get_function_code(self, function_name: str) -> List[Dict]:
        return self.db.get_function_code(function_name)

    def get_document_imports(self, document_id: str) -> List[str]:
        return self.db.get_document_imports(document_id)

    def get_imports_by_file(self, file_path: str) -> List[str]:
        return self.db.get_imports_by_file(file_path)

    def list_all_classes(self) -> List[Dict]:
        return self.db.list_all_classes()

    def list_all_functions(self) -> List[Dict]:
        return self.db.list_all_functions()

    def get_chunk_metadata(self, chunk_id: int) -> Optional[Dict]:
        return self.db.get_chunk_metadata(chunk_id)

    def search_by_type(self, chunk_type: str, name_pattern: Optional[str] = None) -> List[Dict]:
        return self.db.search_by_type(chunk_type, name_pattern)

    def close(self):
        self.embeddings.persist()
        self.db.close()

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
