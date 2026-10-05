import os
from typing import List, Tuple, Optional

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# --------------------------
# EmbeddingManager
# --------------------------

class EmbeddingManager:
    """
    Gère l'indexation vectorielle FAISS et les embeddings via SentenceTransformer.
    Responsable de l'encodage des textes et des recherches sémantiques.
    """
    def __init__(self, faiss_index_path: str = "faiss.index",
                 model_name: str = "mixedbread-ai/mxbai-embed-large-v1"):
        self.faiss_index_path = faiss_index_path
        self.model = SentenceTransformer(model_name, local_files_only=True)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)
        self._load_faiss()

    def _load_faiss(self):
        if os.path.exists(self.faiss_index_path):
            self.index = faiss.read_index(self.faiss_index_path)

    def persist(self):
        faiss.write_index(self.index, self.faiss_index_path)

    def encode(self, texts: List[str], normalize: bool = True, prompt_name: Optional[str] = None) -> np.ndarray:
        if prompt_name:
            emb = self.model.encode(
                texts, 
                prompt_name=prompt_name,
                convert_to_numpy=True, 
                normalize_embeddings=normalize
            )
        else:
            emb = self.model.encode(
                texts, 
                convert_to_numpy=True, 
                normalize_embeddings=normalize
            )
        return emb.astype("float32")

    def add_embeddings(self, embeddings: np.ndarray) -> List[int]:
        """Ajoute des embeddings à l'index FAISS et retourne les IDs FAISS assignés."""
        start_id = self.index.ntotal
        self.index.add(embeddings)
        return list(range(start_id, self.index.ntotal))

    def search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Recherche les top_k vecteurs les plus similaires."""
        return self.index.search(query_embedding, top_k)

    def rebuild_index(self, embeddings: np.ndarray) -> None:
        """Reconstruit complètement l'index FAISS avec de nouveaux embeddings."""
        self.index = faiss.IndexFlatIP(self.dim)
        if len(embeddings) > 0:
            self.index.add(embeddings)

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    @property
    def embedding_dim(self) -> int:
        return int(self.dim) if self.dim is not None else 0
