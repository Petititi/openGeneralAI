# Long Term Memory - Fonctionnalités Structurelles

## Vue d'ensemble

Le module `longterm_memory.py` a été amélioré pour capturer et interroger des informations structurelles sur le code indexé. Au lieu de simplement stocker des chunks de texte, le système maintenant extrait et indexe :

- **Types de chunks** : classe, méthode, fonction, module, block, text
- **Noms** : noms de classes, méthodes, fonctions
- **Hiérarchie** : quelle classe parente pour les méthodes
- **Imports** : tous les imports/includes d'un fichier

## Nouvelles métadonnées stockées

### Table `chunks` - Nouvelles colonnes :
- `chunk_type` : type du chunk (class, method, function, module, block, text)
- `chunk_name` : nom de la classe/méthode/fonction
- `parent_class` : nom de la classe parente pour les méthodes

### Table `document_imports` :
- `document_id` : référence au document
- `import_statement` : texte complet de l'import

## Nouvelles méthodes API

### 1. Recherche par classe

```python
# Obtenir le code complet d'une classe
results = ltm.get_class_code("MyClass")

# Obtenir toutes les méthodes d'une classe
methods = ltm.get_methods_by_class("MyClass")
```

### 2. Recherche par méthode/fonction

```python
# Obtenir une méthode spécifique d'une classe
method = ltm.get_method_code("process_data", class_name="DataProcessor")

# Obtenir une fonction (hors classe)
func = ltm.get_function_code("calculate_sum")
```

### 3. Recherche par imports

```python
# Obtenir tous les imports d'un fichier
imports = ltm.get_imports_by_file("/path/to/file.py")

# Obtenir les imports par document_id
imports = ltm.get_document_imports(doc_id)
```

### 4. Lister les symboles

```python
# Lister toutes les classes indexées
classes = ltm.list_all_classes()

# Lister toutes les fonctions indexées
functions = ltm.list_all_functions()
```

### 5. Recherche par type avec pattern

```python
# Trouver toutes les méthodes avec "test" dans le nom
test_methods = ltm.search_by_type("method", name_pattern="test")

# Trouver toutes les classes avec "Manager" dans le nom
managers = ltm.search_by_type("class", name_pattern="Manager")
```

### 6. Métadonnées détaillées

```python
# Obtenir toutes les métadonnées d'un chunk
metadata = ltm.get_chunk_metadata(chunk_id)
# Retourne: chunk_type, chunk_name, parent_class, start_line, end_line, etc.
```

## Utilisation en ligne de commande

### Lister les classes
```bash
python longterm_memory.py list-classes
```

### Lister les fonctions
```bash
python longterm_memory.py list-functions
```

### Obtenir le code d'une classe
```bash
python longterm_memory.py get-class MyClass
```

### Obtenir les méthodes d'une classe
```bash
python longterm_memory.py get-methods MyClass
```

### Obtenir le code d'une fonction
```bash
python longterm_memory.py get-function my_function
```

### Obtenir les imports d'un fichier
```bash
python longterm_memory.py get-imports /path/to/file.py
```

## Exemple d'utilisation programmatique

```python
from agents.tools.longterm_memory import LongTermMemory

# Initialiser
ltm = LongTermMemory()

# Indexer un projet
ltm.add_folder("./my_project")

# Trouver toutes les classes "Manager"
managers = ltm.search_by_type("class", name_pattern="Manager")

for cls in managers:
    print(f"Classe: {cls['name']}")
    
    # Obtenir toutes ses méthodes
    methods = ltm.get_methods_by_class(cls['name'])
    for method in methods:
        print(f"  - {method['method_name']}")
    
    # Obtenir les imports du fichier
    imports = ltm.get_imports_by_file(cls['source_path'])
    print(f"  Imports: {len(imports)}")

# Recherche hybride (mots-clés + sémantique) toujours disponible
results = ltm.search("comment gérer les erreurs", mode="hybrid")

# Fermer
ltm.close()
```

## Format des résultats

Toutes les méthodes de recherche retournent des dictionnaires avec :

- `chunk_id` : ID unique du chunk
- `document_id` : ID du document source
- `start_line`, `end_line` : positions dans le fichier
- `content` : code complet
- `source_path` : chemin du fichier source
- `language` : langage de programmation
- `chunk_type` : type (class, method, function, etc.)
- `chunk_name` ou `method_name`, `class_name`, `function_name` : selon le type
- `parent_class` : classe parente pour les méthodes

## Langages supportés

Le parsing structurel fonctionne avec tous les langages supportés par tree-sitter :

- Python, JavaScript, TypeScript, TSX
- C, C++, C#
- Java, Go, Rust
- PHP, Ruby, Kotlin, Swift
- Objective-C, Lua, Bash, SQL
- R, Scala, Haskell

Pour les fichiers texte (.md, .txt, etc.), les chunks sont créés par paragraphes sans métadonnées structurelles.

## Migration

Les bases de données existantes seront automatiquement migrées au premier lancement :
- Les nouvelles colonnes `chunk_type`, `chunk_name`, `parent_class` seront créées
- La table `document_imports` sera créée
- Les index seront créés

Les chunks existants auront des valeurs NULL pour ces nouveaux champs. Pour bénéficier pleinement des nouvelles fonctionnalités, il est recommandé de ré-indexer les documents.

## Performance

- Les index SQL sont créés sur `chunk_type`, `chunk_name`, et `parent_class` pour des recherches rapides
- La recherche hybride (mots-clés + sémantique) reste inchangée
- L'extraction structurelle ajoute ~10-20% de temps d'indexation mais permet des requêtes beaucoup plus précises
