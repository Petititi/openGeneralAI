import os
import sys
import json
import hashlib
import datetime as dt
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np

# Import the new code_parser module (tree-sitter parsing)
from .code_parser import (
    parse_file,
    extract_code_chunks,
    extract_text_chunks,
    SUPPORTED_CODE_EXT,
    TEXT_EXT,
    read_text_safe as _read_text_safe,
    is_supported_code_file,
    is_text_file,
)

from .DatabaseManagement import DatabaseManager

# Try to import EmbeddingManager, but make it optional
try:
    from .EmbeddingManagement import EmbeddingManager
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    EmbeddingManager = None

# Import SearchEngine for better search separation
from .search_engine import SearchEngine

# --------------------------
# Utilitaires
# --------------------------

RESERVED_KEYWORD_CODE = [
    "def", "class", "function", "var", "let", "const", "import", "from",
    "public", "private", "protected", "interface", "struct", "enum", "package",
    "return", "if", "else", "switch", "case", "for", "while", "do", "try",
    "catch", "finally", "throw", "new", "this", "super", "extends", "implements",
]

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
    """Read file content safely with size limit. Wrapper for backward compatibility."""
    return _read_text_safe(path, max_bytes)


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
        enable_embeddings: bool = True,
    ):
        self.storage_dir = Path(storage_dir)  # Conservé pour compatibilité
        self.hybrid_alpha = hybrid_alpha
        self._embeddings = None
        self._embedding_manager_class = EmbeddingManager

        need_consistency_check = not Path(db_path).exists()
        # Initialiser les managers
        self.db = DatabaseManager(db_path)
        
        # Initialize embeddings only if available and enabled
        if enable_embeddings and EMBEDDING_AVAILABLE and EmbeddingManager is not None:
            try:
                self._embeddings = EmbeddingManager(faiss_index_path, model_name)
            except Exception as e:
                print(f"[WARNING] LongTermMemory: Could not initialize embeddings: {e}", file=sys.stderr)
                self._embeddings = None

        if need_consistency_check:
            report = self.check_file_integrity()
            if report["total_documents"] != report["existing_unchanged"]:
                print(f"[INFO] LongTermMemory: Consistency check found issues: {report}", file=sys.stderr)

    @property
    def embeddings(self):
        """Property to access embeddings, returns None if not available."""
        return self._embeddings

    # Propriétés de compatibilité pour accès legacy
    @property
    def conn(self):
        return self.db.conn
    
    @property
    def index(self):
        if self.embeddings is None:
            return None
        return self.embeddings.index
    
    @property
    def model(self):
        if self.embeddings is None:
            return None
        return self.embeddings.model
    
    @property
    def dim(self):
        if self.embeddings is None:
            return None
        return self.embeddings.dim

    # ---------- API publique ----------

    def add(self, path: str, rel_to: Optional[str] = None, extra: Optional[Dict] = None) -> Tuple[str, bool]:
        """
        Ajoute un document fichier : analyse le contenu, crée des chunks, indexe FTS + FAISS.
        Retourne document_id (sha256 du contenu).
        
        Si le document existe déjà avec le même hash, skip le traitement.
        Si le document existe mais avec un hash différent, met à jour.
        """
        should_update_FAISS = False
        p = Path(path)
        data = p.read_bytes()
        doc_id = sha256_bytes(data)

        # Vérifier si le document existe déjà avec le même contenu
        existing_doc = self.db.get_document_info(doc_id)
        if existing_doc:
            # Document déjà indexé
            return doc_id, should_update_FAISS
        
        # Vérifier si un document avec ce chemin existe déjà (mais hash différent)
        # Dans ce cas, on doit le supprimer avant de réindexer
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT id, sha256 FROM documents WHERE source_path = ?", (str(p.resolve()),))
        old_doc = cursor.fetchone()
        if old_doc and old_doc[1] != doc_id:
            # Le fichier a été modifié - supprimer l'ancienne version
            self.db.delete_document(old_doc[0])
            should_update_FAISS = True
        
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
            elif self.embeddings is not None:
                # Embedding normal pour fonctions, méthodes, etc. (only if embeddings available)
                emb = self.embeddings.encode([content], normalize=True)
                chunk_embeddings[i] = emb[0]
            else:
                # No embeddings - skip embedding calculation
                chunk_embeddings[i] = None
        
        # Deuxième passe: calculer les embeddings des classes comme somme pondérée des méthodes et attributs
        # Only do this if embeddings are available
        if self.embeddings is not None:
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
                    
                    # Collecter les embeddings des méthodes et attributs de cette classe avec leurs tailles
                    for method_idx in method_indices:
                        if method_idx < len(chunk_embeddings) and chunk_embeddings[method_idx] is not None:
                            method_embeddings.append(chunk_embeddings[method_idx])
                            # Poids basé sur la taille en bytes du contenu de la méthode/attribut
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
        
        # Troisième passe: ajouter tous les embeddings à FAISS (only if embeddings available)
        if self.embeddings is not None:
            for i, chunk_id in enumerate(chunk_ids):
                if i in chunk_embeddings and chunk_embeddings[i] is not None:
                    emb = chunk_embeddings[i].reshape(1, -1)
                    faiss_ids = self.embeddings.add_embeddings(emb)
                    self.db.insert_faiss_mapping(faiss_ids[0], chunk_id)
            
            self.embeddings.persist()
        
        return doc_id,should_update_FAISS

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
        should_update_FAISS = False
        for f in files:
            try:
                _, updated = self.add(str(f), rel_to=str(root))
                if updated:
                    should_update_FAISS = True
            except Exception as e:  # type: ignore
                print(f"[WARN] Skip {f}: {e}", file=sys.stderr)
        if should_update_FAISS:
            self._rebuild_faiss_index()

    def search(self, query: str, top_k: int = 8, mode: str = "hybrid") -> List[Dict]:
        """
        mode in {"keyword", "semantic", "hybrid"}
        Retour: liste de dicts {chunk_id, document_id, score, start_line, end_line, chunk_type, chunk_name, content, source_path, language}
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
            # Check if embeddings are available
            if self.embeddings is None:
                if mode == "semantic":
                    # No embeddings available, return empty results for semantic-only
                    pass
                # Fall back to keyword-only for hybrid
            else:
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
            for cid, did, s, e, chunk_type, name, content, score in rows:
                d = self.db.get_document_info(did)
                out.append({
                    "chunk_id": int(cid),
                    "document_id": did,
                    "start_line": int(s),
                    "end_line": int(e),
                    "chunk_type": chunk_type,
                    "chunk_name": name,
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
        result = {**db_stats}
        
        # Handle case where embeddings are not available
        if self.embeddings is not None:
            result["faiss_ntotal"] = self.embeddings.ntotal
            result["embedding_dim"] = self.embeddings.embedding_dim
        else:
            result["faiss_ntotal"] = 0
            result["embedding_dim"] = 0
            
        return result

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
        """
        Reconstruit l'index FAISS et ne ré-encode que les chunks qui 
        n'ont pas d'embedding dans FAISS.
        Préserve les embeddings existants pour les chunks inchangés.
        """
        # Récupérer tous les chunks et leurs mappings FAISS actuels
        all_chunk_data = self.db.get_all_chunks()
        
        if not all_chunk_data:
            # nothing to do, empty database
            return
        
        chunks_with_embeddings = []  # (chunk_id, content, old_faiss_id)
        chunks_without_embeddings = []  # (chunk_id, content)
        for chunk_id, content, faiss_id in all_chunk_data:
            if faiss_id is not None:
                chunks_with_embeddings.append((chunk_id, content, faiss_id))
            else:
                chunks_without_embeddings.append((chunk_id, content))
        
        # Récupérer les embeddings existants depuis FAISS
        old_embeddings = {}
        if chunks_with_embeddings and self.embeddings.ntotal > 0:
            # Extraire les embeddings de l'ancien index
            old_faiss_ids = [faiss_id for _, _, faiss_id in chunks_with_embeddings]
            max_faiss_id = max(old_faiss_ids)
            
            if max_faiss_id < self.embeddings.ntotal:
                # Récupérer les vecteurs de l'index actuel
                dim = self.embeddings.dim if self.embeddings.dim else 1024
                for chunk_id, content, old_faiss_id in chunks_with_embeddings:
                    try:
                        # Extraire le vecteur depuis FAISS
                        vector = np.zeros(dim, dtype="float32")
                        self.embeddings.index.reconstruct(int(old_faiss_id), vector)
                        old_embeddings[chunk_id] = vector
                    except Exception:
                        # Si on ne peut pas récupérer le vecteur, on devra le ré-encoder
                        chunks_without_embeddings.append((chunk_id, content))
        
        # Encoder uniquement les nouveaux chunks
        new_embeddings = {}
        if chunks_without_embeddings:
            contents_to_encode = [content for _, content in chunks_without_embeddings]
            encoded = self.embeddings.encode(contents_to_encode, normalize=True)
            for i, (chunk_id, _) in enumerate(chunks_without_embeddings):
                new_embeddings[chunk_id] = encoded[i]
        
        # Construire le nouvel index avec tous les embeddings
        all_embeddings = []
        chunk_id_order = []
        
        for chunk_id, content, _ in all_chunk_data:
            if chunk_id in old_embeddings:
                all_embeddings.append(old_embeddings[chunk_id])
                chunk_id_order.append(chunk_id)
            elif chunk_id in new_embeddings:
                all_embeddings.append(new_embeddings[chunk_id])
                chunk_id_order.append(chunk_id)
        
        # Reconstruire l'index FAISS
        if all_embeddings:
            embeddings_array = np.array(all_embeddings, dtype="float32")
            self.embeddings.rebuild_index(embeddings_array)
            
            # Mettre à jour la table de mapping
            self.db.clear_faiss_mappings()
            for new_faiss_id, chunk_id in enumerate(chunk_id_order):
                self.db.insert_faiss_mapping(new_faiss_id, chunk_id)
        else:
            # Pas d'embeddings, créer un index vide
            dim = self.embeddings.dim if self.embeddings.dim else 1024
            self.embeddings.rebuild_index(np.array([], dtype="float32").reshape(0, dim))
            self.db.clear_faiss_mappings()
        
        self.embeddings.persist()

    # Méthodes de délégation vers DatabaseManager
    def get_methods_by_class(self, class_name: str) -> List[Dict]:
        return self.db.get_methods_by_class(class_name)

    def get_class_code(self, class_name: str) -> List[Dict]:
        return self.db.get_class_code(class_name)
    
    def search_symbols(self, pattern: str, symbol_type: Optional[str] = None, 
                      language: Optional[str] = None, case_sensitive: bool = False) -> List[Dict]:
        """
        Recherche flexible de symboles (classes, fonctions, méthodes) avec pattern matching.
        
        Args:
            pattern: Pattern de recherche (supporte SQL LIKE avec % et _)
            symbol_type: Type de symbole ('class', 'function', 'method', None pour tous)
            language: Filtrer par langage (None pour tous)
            case_sensitive: Recherche sensible à la casse
        
        Returns:
            Liste de symboles trouvés avec métadonnées
        """
        with self.db._lock:
            cur = self.db.conn.cursor()
            
            # Construire la requête dynamiquement
            conditions = []
            params = []
            
            if not case_sensitive:
                pattern = pattern.lower()
                name_expr = "LOWER(c.chunk_name)"
            else:
                name_expr = "c.chunk_name"
            
            conditions.append(f"{name_expr} LIKE ?")
            params.append(f"%{pattern}%")
            
            if symbol_type:
                conditions.append("c.chunk_type = ?")
                params.append(symbol_type)
            else:
                conditions.append("c.chunk_type IN ('class', 'function', 'method')")
            
            if language:
                conditions.append("d.language = ?")
                params.append(language)
            
            conditions.append("c.chunk_name IS NOT NULL")
            
            query = f"""
                SELECT c.chunk_name, c.chunk_type, c.parent_class, c.start_line, c.end_line,
                       d.source_path, d.language, c.id, c.document_id
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE {' AND '.join(conditions)}
                ORDER BY c.chunk_name, c.start_line
            """
            
            cur.execute(query, params)
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "name": row[0],
                    "type": row[1],
                    "parent_class": row[2],
                    "start_line": row[3],
                    "end_line": row[4],
                    "source_path": row[5],
                    "language": row[6],
                    "chunk_id": row[7],
                    "document_id": row[8]
                })
            return results

    def get_class_with_members(self, class_name: str, include_code: bool = True) -> Optional[Dict]:
        """
        Récupère une classe avec TOUS ses membres (méthodes, attributs, etc.).
        
        Args:
            class_name: Nom de la classe
            include_code: Inclure le code complet ou seulement les signatures
        
        Returns:
            Dict structuré avec class_definition, methods, attributes, etc.
        """
        class_defs = self.db.get_class_code(class_name)
        if not class_defs:
            return None
        
        class_def = class_defs[0]
        
        # Récupérer toutes les méthodes
        methods = self.db.get_methods_by_class(class_name)
        
        # Récupérer les attributs
        with self.db._lock:
            cur = self.db.conn.cursor()
            cur.execute("""
                SELECT c.chunk_name, c.content, c.start_line, c.end_line, c.chunk_type
                FROM chunks c
                WHERE c.parent_class = ? AND c.chunk_type LIKE '%attribute%'
                ORDER BY c.start_line
            """, (class_name,))
            attribute_rows = cur.fetchall()
        
        # Formater les résultats
        result = {
            "class_name": class_name,
            "source_path": class_def["source_path"],
            "language": class_def["language"],
            "start_line": class_def["start_line"],
            "end_line": class_def["end_line"],
            "document_id": class_def["document_id"],
        }
        
        if include_code:
            result["class_definition"] = class_def["content"]
            result["methods"] = [
                {
                    "name": m["method_name"],
                    "code": m["content"],
                    "start_line": m["start_line"],
                    "end_line": m["end_line"]
                } for m in methods
            ]
            result["attributes"] = [
                {
                    "name": attr[0],
                    "code": attr[1],
                    "start_line": attr[2],
                    "end_line": attr[3],
                    "type": attr[4]
                } for attr in attribute_rows
            ]
        else:
            # Seulement les signatures
            result["class_signature"] = class_def["content"].split('\n')[0]
            result["methods"] = [
                {
                    "name": m["method_name"],
                    "signature": m["content"].split('\n')[0],
                    "start_line": m["start_line"]
                } for m in methods
            ]
            result["attributes"] = [
                {
                    "name": attr[0],
                    "signature": attr[1].split('\n')[0],
                    "start_line": attr[2]
                } for attr in attribute_rows
            ]
        
        return result

    def get_dependencies_for_file(self, file_path: str) -> Dict:
        """
        Récupère les dépendances complètes d'un fichier : imports + symboles définis.
        
        Args:
            file_path: Chemin vers le fichier
        
        Returns:
            Dict avec imports, classes_defined, functions_defined
        """
        # Normaliser le chemin
        file_path = str(Path(file_path).resolve())
        
        # Récupérer le document_id
        with self.db._lock:
            cur = self.db.conn.cursor()
            cur.execute("SELECT id FROM documents WHERE source_path = ?", (file_path,))
            doc_row = cur.fetchone()
            if not doc_row:
                return {"error": f"File not found: {file_path}"}
            
            doc_id = doc_row[0]
        
        # Imports
        imports = self.db.get_document_imports(doc_id)
        
        # Symboles définis
        with self.db._lock:
            cur = self.db.conn.cursor()
            
            # Classes
            cur.execute("""
                SELECT chunk_name, start_line FROM chunks
                WHERE document_id = ? AND chunk_type = 'class' AND chunk_name IS NOT NULL
                ORDER BY start_line
            """, (doc_id,))
            classes = [{"name": r[0], "line": r[1]} for r in cur.fetchall()]
            
            # Fonctions (top-level seulement)
            cur.execute("""
                SELECT chunk_name, start_line FROM chunks
                WHERE document_id = ? AND chunk_type = 'function' 
                AND parent_class IS NULL AND chunk_name IS NOT NULL
                ORDER BY start_line
            """, (doc_id,))
            functions = [{"name": r[0], "line": r[1]} for r in cur.fetchall()]
        
        return {
            "file_path": file_path,
            "document_id": doc_id,
            "imports": imports,
            "classes_defined": classes,
            "functions_defined": functions
        }

    def find_symbol_usage(self, symbol_name: str, symbol_type: Optional[str] = None) -> Dict:
        """
        Trouve tous les endroits où un symbole est potentiellement utilisé.
        Utilise la recherche full-text pour trouver les mentions du symbole.
        
        Args:
            symbol_name: Nom du symbole à chercher
            symbol_type: Type optionnel pour filtrer la définition
        
        Returns:
            Dict avec definitions et usages
        """
        # D'abord, trouver la définition
        definitions = self.search_symbols(symbol_name, symbol_type=symbol_type, case_sensitive=True)
        
        # Ensuite, chercher les usages via FTS
        usage_results = self.db.keyword_search(symbol_name, top_k=100)
        
        usages = []
        for chunk_id, doc_id, start_line, end_line, chunk_type, chunk_name, content, score in usage_results:
            # Exclure les définitions elles-mêmes
            is_definition = any(
                d["chunk_id"] == chunk_id for d in definitions
            )
            
            if not is_definition:
                doc_info = self.db.get_document_info(doc_id)
                usages.append({
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "source_path": doc_info[0] if doc_info else None,
                    "language": doc_info[1] if doc_info else None,
                    "start_line": start_line,
                    "end_line": end_line,
                    "chunk_type": chunk_type,
                    "chunk_name": chunk_name,
                    "score": score,
                    "preview": content[:200] + "..." if len(content) > 200 else content
                })
        
        return {
            "symbol": symbol_name,
            "definitions": definitions,
            "usages": usages
        }

    def get_related_symbols(self, class_name: str) -> Dict:
        """
        Récupère tous les symboles liés à une classe : classes parentes, classes filles,
        symboles importés dans le même fichier, etc.
        
        Args:
            class_name: Nom de la classe
        
        Returns:
            Dict avec related_classes, same_file_symbols, imports
        """
        class_defs = self.db.get_class_code(class_name)
        if not class_defs:
            return {"error": f"Class {class_name} not found"}
        
        class_def = class_defs[0]
        doc_id = class_def["document_id"]
        source_path = class_def["source_path"]
        
        # Imports du fichier
        imports = self.db.get_document_imports(doc_id)
        
        # Autres symboles dans le même fichier
        with self.db._lock:
            cur = self.db.conn.cursor()
            cur.execute("""
                SELECT DISTINCT chunk_name, chunk_type, parent_class
                FROM chunks
                WHERE document_id = ? AND chunk_name IS NOT NULL AND chunk_name != ?
                AND chunk_type IN ('class', 'function')
                ORDER BY chunk_type, chunk_name
            """, (doc_id, class_name))
            
            same_file = [{"name": r[0], "type": r[1], "parent": r[2]} for r in cur.fetchall()]
        
        # Chercher les références à cette classe dans d'autres fichiers
        references = self.find_symbol_usage(class_name, symbol_type="class")
        
        return {
            "class_name": class_name,
            "source_path": source_path,
            "imports": imports,
            "same_file_symbols": same_file,
            "references": references
        }

    def format_code_summary(self, name: str, code_type: str, parent_class: Optional[str] = None) -> str:
        """
        Formate un résumé de code pour le LLM.
        
        Args:
            name: Nom de la classe/fonction/méthode
            code_type: Type ('class', 'function', 'method')
            parent_class: Nom de la classe parente (pour les méthodes)
        
        Returns:
            Résumé formaté avec signature et structure
        """
        if code_type == 'class':
            class_info = self.get_class_with_members(name, include_code=False)
            if not class_info:
                return f"Class `{name}` not found"
            
            language = class_info['language']
            source_path = class_info['source_path']
            
            # Construire le résumé
            summary = f"**{language} Class: `{name}`** (from `{source_path}`)"
            summary += f"\n\n```{language}\n{class_info['class_signature']}\n```\n"
            
            # Lister les attributs
            if class_info['attributes']:
                summary += "\n**Attributes:**\n"
                for attr in class_info['attributes'][:5]:
                    summary += f"  - `{attr['name']}`: {attr['signature']}\n"
                if len(class_info['attributes']) > 5:
                    summary += f"  - ... and {len(class_info['attributes']) - 5} more\n"
            
            # Lister les méthodes
            if class_info['methods']:
                summary += "\n**Methods:**\n"
                for method in class_info['methods'][:10]:
                    summary += f"  - `{method['name']}`: {method['signature']}\n"
                if len(class_info['methods']) > 10:
                    summary += f"  - ... and {len(class_info['methods']) - 10} more methods\n"
            
            return summary
        
        elif code_type in ['function', 'method']:
            # Récupérer le code de la fonction/méthode
            if code_type == 'method' and parent_class:
                code_defs = self.get_method_code(name, parent_class)
            else:
                code_defs = self.get_function_code(name)
            
            if not code_defs:
                return f"{code_type.capitalize()} `{name}` not found"
            
            code_def = code_defs[0]
            language = code_def.get('language', 'unknown')
            source_path = code_def.get('source_path', 'unknown')
            content = code_def['content']
            
            # Extraire la signature (première ligne + docstring si présent)
            lines = content.split('\n')
            signature = lines[0].strip()
            
            # Chercher une docstring
            docstring = ""
            if len(lines) > 1:
                # Python docstring
                if language == 'python' and lines[1].strip().startswith(('"""', "'''")):
                    doc_lines = []
                    in_doc = False
                    for line in lines[1:]:
                        if '"""' in line or "'''" in line:
                            if not in_doc:
                                in_doc = True
                                doc_lines.append(line.strip())
                            else:
                                doc_lines.append(line.strip())
                                break
                        elif in_doc:
                            doc_lines.append(line.strip())
                    if doc_lines:
                        docstring = " ".join(doc_lines)[:200]  # Max 200 chars
                # JSDoc, JavaDoc, etc.
                elif lines[1].strip().startswith(('/*', '//', '#')):
                    doc_lines = [lines[1].strip()]
                    for line in lines[2:5]:  # Max 3 lignes
                        if line.strip().startswith(('*', '//', '#')):
                            doc_lines.append(line.strip())
                        else:
                            break
                    docstring = " ".join(doc_lines)[:200]
            
            prefix = f"Method of class `{parent_class}`" if parent_class else "Function"
            summary = f"**{language} {prefix}: `{name}`** (from `{source_path}`)"
            summary += f"\n\n```{language}\n{signature}\n```"
            
            if docstring:
                summary += f"\n\n{docstring}"
            
            return summary
        
        else:
            return f"Unknown code type: {code_type}"

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

    # ============================================================
    # Souvenir/Memory API - High-level methods for souvenir_assistant
    # These wrap db + embeddings operations for cleaner interface
    # ============================================================

    def add_memory_chunk(
        self,
        document_id: str,
        ord: int,
        start_line: int,
        end_line: int,
        content: str,
        chunk_type: str = "text",
        chunk_name: Optional[str] = None,
        parent_class: Optional[str] = None,
        embed: bool = True,
    ) -> int:
        """
        Add a chunk to the database with automatic embedding handling.
        
        Handles embedding computation, duplicate detection, and reuse automatically.
        
        Args:
            document_id: The document this chunk belongs to
            ord: Ordinal position
            start_line: Starting line number
            end_line: Ending line number
            content: The text content of the chunk
            chunk_type: Type of chunk (text, class, method, function, etc.)
            chunk_name: Optional name for the chunk
            parent_class: Optional parent class name (for methods/attributes)
            embed: Whether to compute and store embeddings (default: True)
            
        Returns:
            The chunk_id of the inserted chunk
        """
        # Insert chunk into database
        chunk_id = self.db.insert_chunk(
            document_id, ord, start_line, end_line, content,
            chunk_type, chunk_name, parent_class
        )
        
        # Add embedding with automatic duplicate detection and reuse
        if embed and self.embeddings is not None and content.strip():
            existing_faiss_id = self.embedding_exists(content)
            if existing_faiss_id is not None:
                # Reuse existing embedding
                self.db.insert_faiss_mapping(existing_faiss_id, chunk_id)
            else:
                # Compute new embedding
                emb = self.embeddings.encode([content], normalize=True)
                faiss_ids = self.embeddings.add_embeddings(emb)
                self.db.insert_faiss_mapping(faiss_ids[0], chunk_id)
                self.embeddings.persist()
        
        return chunk_id

    def add_memory(
        self,
        content: str,
        source_path: str = "memory",
        title: Optional[str] = None,
        category: str = "general",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Add a memory/souvenir to the database.
        
        This is a high-level method that handles document creation,
        chunk creation, and embedding indexing.
        
        Args:
            content: The memory content
            source_path: Source path or identifier
            title: Optional title
            category: Category (default: general)
            tags: Optional list of tags
            
        Returns:
            Dict with 'ok' and 'document_id' or 'error'
        """
        import hashlib
        
        # Generate document ID from content
        doc_id = hashlib.sha256(content.encode()).hexdigest()
        
        # Check if document already exists
        existing = self.db.get_document_info(doc_id)
        if existing:
            return {"ok": True, "document_id": doc_id, "message": "Memory already exists"}
        
        # Insert document
        self.db.insert_document(
            doc_id=doc_id,
            source_path=source_path,
            rel_path=None,
            media_type="text",
            language=None,
            sha256=doc_id,
            size_bytes=len(content),
            category=category,
            extra={"title": title, "tags": tags or []}
        )
        
        # Add chunk with embedding
        self.add_memory_chunk(
            document_id=doc_id,
            content=content,
            chunk_type="memory",
            chunk_name=title,
        )
        
        return {"ok": True, "document_id": doc_id, "message": "Memory added successfully"}

    def get_memory(self, memory_id: str) -> Optional[Dict]:
        """Get a memory by its ID."""
        return self.db.get_document_details(memory_id)

    def search_memories(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 10,
        mode: str = "hybrid",
    ) -> List[Dict]:
        """
        Search memories using hybrid search.
        
        Args:
            query: Search query
            category: Optional category filter
            top_k: Number of results
            mode: Search mode (keyword, semantic, hybrid)
            
        Returns:
            List of memory results
        """
        results = self.search(query, top_k=top_k, mode=mode)
        
        # Filter by category if specified
        if category:
            results = [r for r in results if r.get("category") == category]
        
        return results

    def get_memory_chunks(self, memory_id: str) -> List[Dict]:
        """Get all chunks for a memory document."""
        with self.db._lock:
            cur = self.db.conn.cursor()
            cur.execute("""
                SELECT id, ord, start_line, end_line, content, chunk_type, chunk_name
                FROM chunks
                WHERE document_id = ?
                ORDER BY ord
            """, (memory_id,))
            
            return [
                {
                    "chunk_id": row[0],
                    "ord": row[1],
                    "start_line": row[2],
                    "end_line": row[3],
                    "content": row[4],
                    "chunk_type": row[5],
                    "chunk_name": row[6],
                }
                for row in cur.fetchall()
            ]

    # ============================================================
    # Category Management
    # ============================================================

    def list_categories(self) -> List[Dict]:
        """List all categories."""
        return self.db.list_categories()

    def create_category(
        self,
        name: str,
        description: str = "",
        color: str = "#6B7280",
        icon: str = "📂",
    ) -> int:
        """Create a new category."""
        return self.db.create_category(name, description, color, icon)

    def get_category(self, name: str) -> Optional[Dict]:
        """Get a category by name."""
        return self.db.get_category(name)

    # ============================================================
    # Embedding utilities
    # ============================================================

    def encode_text(self, text: str, normalize: bool = True) -> Optional[np.ndarray]:
        """
        Encode text to embedding vector.
        
        Args:
            text: Text to encode
            normalize: Whether to normalize the vector
            
        Returns:
            Embedding vector or None if embeddings not available
        """
        if self.embeddings is None:
            return None
        return self.embeddings.encode([text], normalize=normalize)[0]

    def find_similar(
        self,
        text: str,
        top_k: int = 5,
        exclude_doc_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Find similar chunks using semantic search.
        
        Args:
            text: Query text
            top_k: Number of results
            exclude_doc_id: Optional document ID to exclude from results
            
        Returns:
            List of similar chunks with scores
        """
        if self.embeddings is None or self.embeddings.ntotal == 0:
            return []
        
        query_emb = self.embeddings.encode([text], normalize=True)
        distances, indices = self.embeddings.search(query_emb, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            
            # Get chunk from mapping
            mapping = self.db.get_faiss_chunk_mapping([int(idx)])
            chunk_id = mapping.get(int(idx))
            if chunk_id is None:
                continue
            
            chunk = self.db.get_chunk_by_id(chunk_id)
            if not chunk:
                continue
            
            # Optionally exclude a document
            if exclude_doc_id and chunk[1] == exclude_doc_id:
                continue
            
            results.append({
                "chunk_id": chunk[0],
                "document_id": chunk[1],
                "content": chunk[6],
                "score": float(dist),
                "chunk_type": chunk[4],
                "chunk_name": chunk[5],
            })
        
        return results

    def is_embeddings_available(self) -> bool:
        """Check if embeddings are available."""
        return self.embeddings is not None

    def embeddings_count(self) -> int:
        """Get the number of embeddings in the index."""
        if self.embeddings is None:
            return 0
        return self.embeddings.ntotal

    def search_embeddings(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search embeddings index.
        
        Args:
            query_embedding: The query vector
            top_k: Number of results
            
        Returns:
            Tuple of (distances, indices)
        """
        if self.embeddings is None:
            return np.array([]), np.array([-1])
        return self.embeddings.search(query_embedding.reshape(1, -1), top_k)

    def add_embedding(
        self,
        content: str,
        chunk_id: int,
    ) -> Optional[int]:
        """
        Add an embedding for content and map it to a chunk.
        
        Args:
            content: Text content to embed
            chunk_id: The chunk ID to map the embedding to
            
        Returns:
            FAISS ID or None if embeddings not available
        """
        if self.embeddings is None or not content.strip():
            return None
        
        emb = self.embeddings.encode([content], normalize=True)
        faiss_ids = self.embeddings.add_embeddings(emb)
        self.db.insert_faiss_mapping(faiss_ids[0], chunk_id)
        return faiss_ids[0]

    def add_embedding_for_existing_chunk(
        self,
        content: str,
        chunk_id: int,
        existing_faiss_id: Optional[int] = None,
    ) -> int:
        """
        Add embedding for an existing chunk, optionally using existing FAISS ID.
        
        Args:
            content: Text content
            chunk_id: Chunk ID
            existing_faiss_id: Optional existing FAISS ID to reuse
            
        Returns:
            FAISS ID used
        """
        if existing_faiss_id is not None and existing_faiss_id >= 0:
            self.db.insert_faiss_mapping(existing_faiss_id, chunk_id)
            return existing_faiss_id
        
        return self.add_embedding(content, chunk_id) or -1

    def persist_embeddings(self):
        """Persist embeddings to disk."""
        if self.embeddings is not None:
            self.embeddings.persist()

    # ============================================================
    # Search API - Comprehensive search methods that combine
    # embeddings and database lookups
    # ============================================================

    def find_similar_documents(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.95,
    ) -> List[Dict[str, Any]]:
        """
        Find documents with similar content using semantic search.
        
        Args:
            query_text: The text to search for
            top_k: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0-1) to consider a match
            
        Returns:
            List of similar documents with their similarity scores
        """
        if not self.is_embeddings_available():
            return []
        
        try:
            # Encode the query
            query_emb = self.encode_text(query_text, normalize=True)
            
            # Search in FAISS index
            distances, indices = self.search_embeddings(query_emb, top_k)
            
            # Get chunk information for matches above threshold
            similar_docs = []
            for dist, idx in zip(distances[0], indices[0]):
                if dist >= similarity_threshold and idx >= 0:
                    # Get chunk info from database
                    chunk_info = self.get_chunk_by_id(int(idx))
                    if chunk_info:
                        chunk_id, doc_id, start_line, end_line, chunk_type, chunk_name, content = chunk_info
                        # Get full document details
                        doc_details = self.get_memory(doc_id)
                        if doc_details:
                            similar_docs.append({
                                "chunk_id": chunk_id,
                                "document_id": doc_id,
                                "document": doc_details.get("document"),
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
        """
        Check if an embedding for the given text already exists.
        
        Uses exact match (cosine similarity = 1.0 for normalized embeddings).
        
        Args:
            text: The text to check
            similarity_threshold: Minimum similarity to consider as exact match
            
        Returns:
            FAISS ID if found, None otherwise
        """
        if not self.is_embeddings_available() or self.embeddings_count() == 0:
            return None
        
        try:
            # Encode the text
            text_emb = self.encode_text(text, normalize=True)
            
            # Search for exact match
            distances, indices = self.search_embeddings(text_emb, 1)
            
            # Check if we have an exact match
            if distances[0][0] >= similarity_threshold and indices[0][0] >= 0:
                return int(indices[0][0])
            
            return None
        except Exception as e:
            print(f"Error checking embedding existence: {e}")
            return None

    def search_by_chunk_type(
        self,
        chunk_types: List[str],
        query: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents by specific chunk types.
        
        Args:
            chunk_types: List of chunk types to search (e.g., ['email_action_items', 'email_key_points'])
            query: Optional text query to further filter results
            top_k: Maximum number of results
            
        Returns:
            List of documents matching the criteria
        """
        try:
            with self.db._lock:
                cur = self.db.conn.cursor()
                
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

    def search_semantic(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for souvenirs using semantic embeddings.
        
        Args:
            query: The search query
            top_k: Maximum number of results
            
        Returns:
            List of souvenirs with similarity scores
        """
        if not self.is_embeddings_available():
            return []
        
        try:
            # Encode query
            query_emb = self.encode_text(query, normalize=True)
            
            # Search embeddings
            distances, indices = self.search_embeddings(query_emb, top_k)
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0:
                    # Get chunk info
                    chunk_info = self.get_chunk_by_id(int(idx))
                    if chunk_info:
                        chunk_id, doc_id, start_line, end_line, chunk_type, chunk_name, content = chunk_info
                        
                        # Get document info
                        doc_info = self.db.get_document_info(doc_id)
                        if doc_info:
                            extra = json.loads(doc_info.get('extra', '{}')) if doc_info.get('extra') else {}
                            results.append({
                                "id": doc_id,
                                "title": extra.get('title', 'Untitled'),
                                "content": content,
                                "category": doc_info.get('category', 'general'),
                                "tags": extra.get('tags', []),
                                "created_at": doc_info.get('created_at', ''),
                                "similarity_score": float(dist),
                                "chunk_type": chunk_type,
                            })
            
            return results
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []

    # ============================================================
    # Legacy/Souvenir API - For backward compatibility with souvenir_assistant
    # These wrap common db + embeddings patterns
    # ============================================================

    def insert_souvenir(
        self,
        content: str,
        category: str = "general",
        title: Optional[str] = None,
        source: str = "manual",
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Insert a souvenir/memory (legacy method).
        
        Args:
            content: The memory content
            category: Category name
            title: Optional title
            source: Source identifier
            tags: Optional tags
            
        Returns:
            Document ID
        """
        import hashlib
        doc_id = hashlib.sha256(content.encode()).hexdigest()
        
        # Check if exists
        existing = self.db.get_document_info(doc_id)
        if existing:
            return doc_id
        
        self.db.insert_document(
            doc_id=doc_id,
            source_path=source,
            rel_path=None,
            media_type="text",
            language=None,
            sha256=doc_id,
            size_bytes=len(content),
            category=category,
            extra={"title": title, "tags": tags or []}
        )
        
        # Add chunk
        self.add_memory_chunk(
            document_id=doc_id,
            content=content,
            chunk_type="souvenir",
            chunk_name=title,
        )
        
        return doc_id

    def list_souvenirs(
        self,
        category: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """
        List souvenirs (legacy method).
        
        Args:
            category: Optional category filter
            limit: Maximum number to return
            offset: Offset for pagination
            
        Returns:
            List of souvenirs
        """
        with self.db._lock:
            cur = self.db.conn.cursor()
            
            query = "SELECT doc_id, source_path, category, extra FROM documents WHERE 1=1"
            params = []
            
            if category:
                query += " AND category = ?"
                params.append(category)
            
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cur.execute(query, params)
            rows = cur.fetchall()
            
            return [
                {
                    "document_id": row[0],
                    "source_path": row[1],
                    "category": row[2],
                    "title": (row[3].get("title") if row[3] else None),
                    "tags": (row[3].get("tags") if row[3] else []),
                }
                for row in rows
            ]

    def search_souvenirs(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """
        Search souvenirs using keyword search.
        
        Args:
            query: Search query
            category: Optional category filter
            limit: Maximum results
            
        Returns:
            List of matching souvenirs
        """
        # Use hybrid search from parent method
        results = self.search(query, top_k=limit, mode="hybrid")
        
        souvenirs = []
        for r in results:
            doc = r.get("document", {})
            if category and doc.get("category") != category:
                continue
            souvenirs.append({
                "document_id": doc.get("doc_id"),
                "title": doc.get("extra", {}).get("title"),
                "category": doc.get("category"),
                "source_path": doc.get("source_path"),
                "content": r.get("chunk_content"),
                "score": r.get("score"),
            })
        
        return souvenirs

    def get_document_info(self, doc_id: str) -> Optional[Dict]:
        """Get document info by ID."""
        return self.db.get_document_info(doc_id)

    def get_chunk_by_id(self, chunk_id: int) -> Optional[Tuple]:
        """Get chunk by ID."""
        return self.db.get_chunk_by_id(chunk_id)

    def create_category_if_not_exists(
        self,
        name: str,
        description: str = "",
        color: str = "#6B7280",
        icon: str = "📂",
    ) -> int:
        """Create category if it doesn't exist, return ID."""
        existing = self.db.get_category(name)
        if existing:
            return existing.get("id", 0)
        return self.db.create_category(name, description, color, icon)

    @property
    def connection(self):
        """Get database connection (legacy property)."""
        return self.db.conn

    @property
    def db_lock(self):
        """Get database lock (legacy property)."""
        return self.db._lock

    # ============================================================
    # Low-level access (for backward compatibility)
    # ============================================================

    @property
    def db_manager(self):
        """Access the database manager directly (backward compatibility)."""
        return self.db

    @property
    def embedding_manager(self):
        """Access the embedding manager directly (backward compatibility)."""
        return self.embeddings

    def cluster_memories(self, n_clusters: int = 5, chunk_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Cluster memories using embeddings.
        
        Uses KMeans-like clustering on the embeddings to group related memories.
        
        Args:
            n_clusters: Number of clusters to create
            chunk_types: Optional list of chunk types to include (e.g., ['memory', 'souvenir'])
            
        Returns:
            Dictionary with clustering results and statistics
        """
        if self.embeddings is None:
            return {"error": "Embeddings not available"}
        
        # Get all embeddings and their associated chunks
        cursor = self.db.conn.cursor()
        
        # Build query to get chunks with embeddings
        query = """
            SELECT c.id, c.document_id, c.content, c.chunk_type, c.chunk_name, d.source_path, d.category
            FROM chunks c
            JOIN documents d ON c.document_id = d.doc_id
            WHERE c.id IN (SELECT chunk_id FROM faiss_mappings)
        """
        
        if chunk_types:
            placeholders = ",".join("?" * len(chunk_types))
            query += f" AND c.chunk_type IN ({placeholders})"
        
        if chunk_types:
            cursor.execute(query, chunk_types)
        else:
            cursor.execute(query)
        
        rows = cursor.fetchall()
        
        if len(rows) < n_clusters:
            return {
                "error": f"Not enough chunks ({len(rows)}) for {n_clusters} clusters",
                "total_chunks": len(rows)
            }
        
        # Build embeddings matrix and store chunk info
        chunk_infos = []
        embeddings_matrix = []
        
        for row in rows:
            chunk_id, doc_id, content, chunk_type, chunk_name, source_path, category = row
            
            # Get FAISS index for this chunk
            cursor.execute("SELECT faiss_id FROM faiss_mappings WHERE chunk_id = ?", (chunk_id,))
            faiss_row = cursor.fetchone()
            if not faiss_row:
                continue
            
            faiss_id = faiss_row[0]
            
            # Get embedding from FAISS index
            try:
                embedding = self.embeddings.index.reconstruct(faiss_id)
                embeddings_matrix.append(embedding)
                chunk_infos.append({
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "content": content[:100],  # Truncate for display
                    "chunk_type": chunk_type,
                    "chunk_name": chunk_name,
                    "source_path": source_path,
                    "category": category,
                    "faiss_id": faiss_id,
                })
            except Exception as e:
                continue
        
        if len(embeddings_matrix) < n_clusters:
            return {
                "error": f"Not enough valid embeddings ({len(embeddings_matrix)}) for {n_clusters} clusters",
                "total_embeddings": len(embeddings_matrix)
            }
        
        embeddings_matrix = np.array(embeddings_matrix).astype("float32")
        
        # Normalize embeddings for clustering
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True) + 1e-12
        embeddings_matrix = embeddings_matrix / norms
        
        # Simple KMeans implementation
        # Initialize cluster centers randomly
        np.random.seed(42)
        indices = np.random.choice(len(embeddings_matrix), n_clusters, replace=False)
        centers = embeddings_matrix[indices].copy()
        
        # Run KMeans iterations
        max_iter = 50
        for _ in range(max_iter):
            # Assign each point to nearest center
            distances = np.dot(embeddings_matrix, centers.T)
            labels = np.argmax(distances, axis=1)
            
            # Update centers
            new_centers = np.zeros_like(centers)
            for k in range(n_clusters):
                mask = labels == k
                if np.sum(mask) > 0:
                    new_centers[k] = np.mean(embeddings_matrix[mask], axis=0)
                    # Re-normalize center
                    norm = np.linalg.norm(new_centers[k])
                    if norm > 1e-12:
                        new_centers[k] = new_centers[k] / norm
            
            # Check convergence
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
        
        # Assign labels to all chunks
        distances = np.dot(embeddings_matrix, centers.T)
        labels = np.argmax(distances, axis=1)
        
        # Build cluster information
        clusters = {}
        for i, info in enumerate(chunk_infos):
            label = int(labels[i])
            if label not in clusters:
                clusters[label] = {
                    "cluster_id": label,
                    "chunks": [],
                    "documents": set(),
                    "categories": set(),
                    "chunk_types": set(),
                }
            clusters[label]["chunks"].append(info)
            clusters[label]["documents"].add(info["document_id"])
            if info["category"]:
                clusters[label]["categories"].add(info["category"])
            clusters[label]["chunk_types"].add(info["chunk_type"])
        
        # Calculate statistics for each cluster
        results = {
            "n_clusters": n_clusters,
            "total_chunks": len(chunk_infos),
            "clusters": [],
        }
        
        for cluster_id in sorted(clusters.keys()):
            cluster = clusters[cluster_id]
            
            # Get top terms (simplified - using first few chars of content)
            sample_contents = [c["content"][:50] for c in cluster["chunks"][:5]]
            
            cluster_info = {
                "cluster_id": cluster_id,
                "size": len(cluster["chunks"]),
                "unique_documents": len(cluster["documents"]),
                "categories": sorted(list(cluster["categories"])) if cluster["categories"] else [],
                "chunk_types": sorted(list(cluster["chunk_types"])),
                "sample_topics": sample_contents,
            }
            results["clusters"].append(cluster_info)
        
        # Sort clusters by size
        results["clusters"].sort(key=lambda x: x["size"], reverse=True)
        
        return results

    def print_cluster_summary(self, n_clusters: int = 5, chunk_types: Optional[List[str]] = None):
        """
        Print a summary of clustered memories.
        
        Args:
            n_clusters: Number of clusters
            chunk_types: Optional filter for chunk types
        """
        print(f"\n{'='*60}")
        print(f"Memory Clustering Analysis")
        print(f"{'='*60}")
        
        result = self.cluster_memories(n_clusters=n_clusters, chunk_types=chunk_types)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        print(f"\nTotal chunks analyzed: {result['total_chunks']}")
        print(f"Number of clusters: {result['n_clusters']}")
        print(f"\n{'='*60}")
        
        for cluster in result["clusters"]:
            print(f"\n📌 Cluster {cluster['cluster_id']}")
            print(f"   Size: {cluster['size']} chunks")
            print(f"   Unique documents: {cluster['unique_documents']}")
            if cluster["categories"]:
                print(f"   Categories: {', '.join(cluster['categories'])}")
            print(f"   Chunk types: {', '.join(cluster['chunk_types'])}")
            print(f"   Sample topics:")
            for topic in cluster["sample_topics"]:
                print(f"      - {topic}...")
        
        print(f"\n{'='*60}\n")

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

def _cmd_sync(verbose: bool = False):
    ltm = LongTermMemory()
    stats = ltm.sync_with_filesystem(verbose=verbose)
    print(f"\nSynchronization completed:")
    print(f"  Removed: {stats['removed']}")
    print(f"  Updated: {stats['updated']}")
    print(f"  Unchanged: {stats['unchanged']}")
    print(f"  Failed: {stats['failed']}")
    print(f"\nDatabase stats:")
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

def _cmd_search_symbols(pattern: str, symbol_type: Optional[str] = None, language: Optional[str] = None):
    ltm = LongTermMemory()
    results = ltm.search_symbols(pattern, symbol_type=symbol_type, language=language)
    print(f"Found {len(results)} symbols matching '{pattern}':")
    for r in results:
        parent = f" (in {r['parent_class']})" if r['parent_class'] else ""
        print(f"  - {r['type']:10} {r['name']:30}{parent:30} @ {r['source_path']}:{r['start_line']}")
    ltm.close()

def _cmd_get_class_full(class_name: str, with_code: bool = False):
    ltm = LongTermMemory()
    result = ltm.get_class_with_members(class_name, include_code=with_code)
    if not result:
        print(f"Class {class_name} not found")
    else:
        print(json.dumps(result, indent=2))
    ltm.close()

def _cmd_get_dependencies(file_path: str):
    ltm = LongTermMemory()
    deps = ltm.get_dependencies_for_file(file_path)
    print(json.dumps(deps, indent=2))
    ltm.close()

def _cmd_find_usage(symbol_name: str, symbol_type: Optional[str] = None):
    ltm = LongTermMemory()
    results = ltm.find_symbol_usage(symbol_name, symbol_type=symbol_type)
    print(f"\nSymbol: {results['symbol']}")
    print(f"\nDefinitions ({len(results['definitions'])})")
    for d in results['definitions']:
        print(f"  - {d['type']} in {d['source_path']}:{d['start_line']}")
    print(f"\nUsages ({len(results['usages'])})")
    for u in results['usages'][:20]:  # Limit to 20
        print(f"  - {u['source_path']}:{u['start_line']} (score: {u['score']:.2f})")
        print(f"    {u['preview'][:100]}")
    ltm.close()

def _cmd_get_related(class_name: str):
    ltm = LongTermMemory()
    related = ltm.get_related_symbols(class_name)
    print(json.dumps(related, indent=2))
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
    
    ap_sync = sub.add_parser("sync", help="Synchroniser la base avec le système de fichiers (met à jour fichiers modifiés)")
    ap_sync.add_argument("--verbose", "-v", action="store_true", help="Afficher les détails")

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
    
    ap_search_sym = sub.add_parser("search-symbols", help="Rechercher des symboles avec pattern")
    ap_search_sym.add_argument("pattern", type=str)
    ap_search_sym.add_argument("--type", type=str, choices=["class", "function", "method"], help="Type de symbole")
    ap_search_sym.add_argument("--language", type=str, help="Filtrer par langage")
    
    ap_class_full = sub.add_parser("get-class-full", help="Obtenir une classe avec tous ses membres")
    ap_class_full.add_argument("class_name", type=str)
    ap_class_full.add_argument("--with-code", action="store_true", help="Inclure le code complet")
    
    ap_deps = sub.add_parser("get-dependencies", help="Obtenir les dépendances d'un fichier")
    ap_deps.add_argument("file_path", type=str)
    
    ap_usage = sub.add_parser("find-usage", help="Trouver où un symbole est utilisé")
    ap_usage.add_argument("symbol_name", type=str)
    ap_usage.add_argument("--type", type=str, choices=["class", "function", "method"], help="Type de symbole")
    
    ap_related = sub.add_parser("get-related", help="Obtenir les symboles liés à une classe")
    ap_related.add_argument("class_name", type=str)

    args = ap.parse_args()
    if args.cmd == "index":
        _cmd_index(args.folder)
    elif args.cmd == "search":
        _cmd_search(args.query, args.mode)
    elif args.cmd == "integrity":
        _cmd_integrity()
    elif args.cmd == "cleanup":
        _cmd_cleanup()
    elif args.cmd == "sync":
        _cmd_sync(verbose=args.verbose)
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
    elif args.cmd == "search-symbols":
        _cmd_search_symbols(args.pattern, symbol_type=args.type, language=args.language)
    elif args.cmd == "get-class-full":
        _cmd_get_class_full(args.class_name, with_code=args.with_code)
    elif args.cmd == "get-dependencies":
        _cmd_get_dependencies(args.file_path)
    elif args.cmd == "find-usage":
        _cmd_find_usage(args.symbol_name, symbol_type=args.type)
    elif args.cmd == "get-related":
        _cmd_get_related(args.class_name)
    else:
        ap.print_help()
