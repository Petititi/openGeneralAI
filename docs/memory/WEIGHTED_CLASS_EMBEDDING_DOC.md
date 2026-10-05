# Modification: Embedding Pondéré des Classes

## Résumé

Le système `longterm_memory.py` a été modifié pour que l'embedding d'une classe soit calculé comme une **somme pondérée des embeddings de ses méthodes** plutôt que directement à partir du texte complet de la classe.

## Motivation

Cette approche présente plusieurs avantages :

1. **Meilleure représentation sémantique** : L'embedding d'une classe reflète mieux ses fonctionnalités réelles à travers ses méthodes
2. **Recherche plus pertinente** : Les requêtes sur les fonctionnalités trouvent les bonnes classes
3. **Réduction du bruit** : Les commentaires et docstrings de la classe n'influencent pas trop l'embedding
4. **Cohérence** : Les méthodes similaires dans différentes classes produisent des embeddings de classe similaires

## Modifications Apportées

### 1. Signature de `extract_code_chunks`

**Avant:**
```python
def extract_code_chunks(source: bytes, language: str) -> Tuple[List[Tuple[int, int, str, Dict]], List[str]]:
```

**Après:**
```python
def extract_code_chunks(source: bytes, language: str) -> Tuple[List[Tuple[int, int, str, Dict]], List[str], Dict[str, List[int]]]:
```

Retourne maintenant un troisième élément : `class_methods_map` qui est un dictionnaire mappant chaque nom de classe vers la liste des indices de chunks de ses méthodes.

### 2. Signature de `extract_text_chunks`

Mise à jour pour cohérence avec la nouvelle signature (retourne un dictionnaire vide pour `class_methods_map`).

### 3. Tracking des Méthodes par Classe

Dans `extract_code_chunks`, ajout d'un dictionnaire `class_methods_map` qui enregistre l'association classe → méthodes :

```python
chunks = []
class_methods_map = {}  # {class_name: [chunk_indices]}
```

### 4. Distinction Méthode vs Fonction

Modification de la fonction `walk` pour distinguer correctement les méthodes (fonctions dans une classe) des fonctions standalone :

```python
elif node.type in FUNCTION_TYPES:
    # Différencier méthode (dans une classe) de fonction (standalone)
    if parent_class:
        chunk_type = "method"
    else:
        chunk_type = "function"
```

### 5. Calcul de l'Embedding Pondéré

Dans la méthode `add`, le processus d'embedding se fait maintenant en trois passes :

**Passe 1 - Insertion et embedding des non-classes :**
```python
for i, (start, end, content, metadata) in enumerate(chunks):
    # Insérer le chunk dans la DB
    # Si c'est une classe, marquer l'embedding comme None (sera calculé plus tard)
    # Sinon, calculer l'embedding normalement
```

**Passe 2 - Calcul des embeddings des classes :**
```python
for i, (start, end, content, metadata) in enumerate(chunks):
    chunk_type = metadata.get("type")
    chunk_name = metadata.get("name")
    
    if chunk_type == "class" and chunk_name in class_methods_map:
        method_indices = class_methods_map[chunk_name]
        method_embeddings = []
        
        # Collecter les embeddings des méthodes
        for method_idx in method_indices:
            if method_idx < len(chunk_embeddings) and chunk_embeddings[method_idx] is not None:
                method_embeddings.append(chunk_embeddings[method_idx])
        
        if method_embeddings:
            # Calculer la moyenne des embeddings des méthodes
            class_emb = np.mean(method_embeddings, axis=0).astype("float32")
            # Re-normaliser L2
            norm = np.linalg.norm(class_emb)
            if norm > 1e-12:
                class_emb = class_emb / norm
            chunk_embeddings[i] = class_emb
```

**Passe 3 - Ajout à FAISS :**
```python
for i, chunk_id in enumerate(chunk_ids):
    if i in chunk_embeddings and chunk_embeddings[i] is not None:
        emb = chunk_embeddings[i].reshape(1, -1)
        faiss_id = self.index.ntotal
        self.index.add(emb)
        cur.execute("INSERT INTO faiss_map(faiss_id, chunk_id) VALUES(?,?)", (faiss_id, chunk_id))
```

## Pondération

Actuellement, une **moyenne simple** est utilisée :
```python
class_emb = np.mean(method_embeddings, axis=0)
```

Cette approche peut être facilement modifiée pour utiliser une pondération différente :
- Pondération par taille de méthode
- Pondération par importance (méthodes publiques vs privées)
- Pondération inversement proportionnelle au nombre de méthodes
- Etc.

## Résultats des Tests

Les tests montrent que l'embedding pondéré fonctionne correctement :

### Test 1: Recherche "data cleaning and normalization"
1. **DataProcessor** (score: 0.7814) ← Classe correctement identifiée
2. clean_data (score: 0.7499)
3. normalize_data (score: 0.7181)

### Test 2: Recherche "file reading and writing operations"
1. **FileManager** (score: 0.6851) ← Classe correctement identifiée
2. read_file (score: 0.6573)
3. write_file (score: 0.6140)

### Test 3: Recherche "mathematical operations like addition"
1. add (score: 0.8013)
2. **Calculator** (score: 0.7851) ← Classe correctement identifiée
3. multiply (score: 0.7134)

## Cas Particuliers

### Classes sans méthodes
Si une classe n'a pas de méthodes ou si elles ne sont pas trouvées dans `class_methods_map`, l'embedding est calculé normalement à partir du contenu de la classe.

### Méthodes non indexées
Seules les méthodes effectivement indexées (présentes dans `chunk_embeddings`) sont utilisées pour le calcul.

## Compatibilité

Les modifications sont rétrocompatibles :
- Les fichiers texte continuent de fonctionner normalement
- Les langages sans support tree-sitter utilisent le fallback par lignes
- Aucun changement dans l'API publique de `LongTermMemory`

## Fichiers Modifiés

- `agents/tools/longterm_memory.py` : Logique principale
- Tests créés :
  - `test_weighted_class_embedding.py` : Test basique
  - `test_advanced_weighted_embedding.py` : Test démonstratif avancé
  - `test_tree_sitter_debug.py` : Debug de la structure AST
