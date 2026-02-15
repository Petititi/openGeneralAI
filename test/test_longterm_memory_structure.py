"""
Tests pour vérifier les fonctionnalités structurelles de longterm_memory.py
"""

import sys
from pathlib import Path
import pytest

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.longterm_memory import LongTermMemory

# Import EMBEDDINGS_AVAILABLE from conftest
try:
    from test.conftest import EMBEDDINGS_AVAILABLE
except ImportError:
    EMBEDDINGS_AVAILABLE = False


@pytest.fixture
def persistant_ltm():
    """Fixture pour créer une instance temporaire de LongTermMemory"""
    db_path = "test/datas/test_memory.sqlite"
    faiss_index_path = "test/datas/test_faiss.index"

    CREATE_DBD = not Path(db_path).exists()
    
    ltm = LongTermMemory(
        db_path=str(db_path),
        faiss_index_path=str(faiss_index_path)
    )

    if CREATE_DBD:
        ltm.add_folder(str(Path(__file__).parent.parent))
    
    yield ltm
    
    # Cleanup
    ltm.close()

# Créer un fichier de test Python avec une classe
TEST_CODE = '''
class Calculator:
    """A simple calculator"""
    
    def add(self, a, b):
        """Add two numbers"""
        return a + b
    
    def subtract(self, a, b):
        """Subtract two numbers"""
        return a - b
    
    def multiply(self, a, b):
        """Multiply two numbers"""
        return a * b
    
    def divide(self, a, b):
        """Divide two numbers"""
        if b == 0:
            raise ValueError("Division by zero")
        return a / b

def standalone_function():
    """A standalone function"""
    print("Hello World")
'''

@pytest.fixture
def temp_ltm(tmp_path):
    """Fixture pour créer une instance temporaire de LongTermMemory"""
    db_path = tmp_path / "test_memory.sqlite"
    faiss_index_path = tmp_path / "test_faiss.index"
    
    test_file = Path(tmp_path) / "calculator.py"
    test_file.write_text(TEST_CODE, encoding='utf-8')
    
    # Check if embeddings are available
    enable_embeddings = EMBEDDINGS_AVAILABLE
    
    ltm = LongTermMemory(
        db_path=str(db_path),
        faiss_index_path=str(faiss_index_path),
        enable_embeddings=enable_embeddings
    )
    
    yield ltm
    
    # Cleanup
    ltm.close()


# Also update persistant_ltm fixture
@pytest.fixture
def persistant_ltm():
    """Fixture pour créer une instance temporaire de LongTermMemory"""
    db_path = "test/datas/test_memory.sqlite"
    faiss_index_path = "test/datas/test_faiss.index"

    CREATE_DBD = not Path(db_path).exists()
    
    # Check if embeddings are available
    enable_embeddings = EMBEDDINGS_AVAILABLE
    
    ltm = LongTermMemory(
        db_path=str(db_path),
        faiss_index_path=str(faiss_index_path),
        enable_embeddings=enable_embeddings
    )

    if CREATE_DBD:
        ltm.add_folder(str(Path(__file__).parent.parent))
    
    yield ltm
    
    # Cleanup
    ltm.close()


def test_indexation(tmp_path, temp_ltm):
    initial_stats = temp_ltm.stats()
    assert initial_stats['documents'] == 0
    assert initial_stats['chunks'] == 0
    doc_id = temp_ltm.add(str(tmp_path / "calculator.py"))
    assert doc_id is not None
    
    stats = temp_ltm.stats()
    assert stats['documents'] == 1
    assert stats['chunks'] > 0
    
    # Only check faiss_ntotal if embeddings are available
    if EMBEDDINGS_AVAILABLE:
        assert stats['faiss_ntotal'] > 0

def test_stats(persistant_ltm):
    """Test de récupération des statistiques"""
    stats = persistant_ltm.stats()
    
    assert isinstance(stats, dict)
    assert len(stats) > 0
    #relatively big codebase:
    assert stats['chunks'] > 200
    
    class_name = "LongTermMemory"
    
    # Obtenir le code de la classe
    class_code = persistant_ltm.get_class_code(class_name)
    assert class_code is not None


def test_list_content(persistant_ltm):
    classes = persistant_ltm.list_all_classes()
    
    assert isinstance(classes, list) and len(classes) > 0
    assert 'class_name' in classes[0]
    assert 'language' in classes[0]

    functions = persistant_ltm.list_all_functions()
    
    assert isinstance(functions, list) and len(functions) > 0
    assert 'function_name' in functions[0]
    assert 'language' in functions[0]
    
    test_class = "LongTermMemory"
    methods = persistant_ltm.get_methods_by_class(test_class)
    
    assert isinstance(methods, list) and len(methods) > 0
    assert 'method_name' in methods[0]
    assert 'start_line' in methods[0]
    assert 'end_line' in methods[0]

    file_to_check = str(Path("./storage/longterm_memory.py").resolve())
    imports = persistant_ltm.get_imports_by_file(file_to_check)
    
    assert isinstance(imports, list)
    assert len(imports) > 10


def test_semantic_search(persistant_ltm):
    """Test de recherche sémantique"""
    results = persistant_ltm.search("add file to index", top_k=3, mode="semantic")
    
    assert isinstance(results, list)
    assert len(results) == 3
    
    if results:
        assert 'score' in results[0]
        assert 'source_path' in results[0]
        assert 'start_line' in results[0]
        assert 'end_line' in results[0]
        assert 'content' in results[0]
    # longterm_memory.py in source_path in at least one result:
    assert any("longterm_memory.py" in result['source_path'] for result in results)

def test_get_chunk_metadata(persistant_ltm):
    """Test de récupération des métadonnées d'un chunk"""
    # Use keyword mode if embeddings not available
    mode = "semantic" if EMBEDDINGS_AVAILABLE else "keyword"
    results = persistant_ltm.search("add file to index", top_k=1, mode=mode)
    
    if not results:
        pytest.skip("No search results available")
    
    chunk_id = results[0]['chunk_id']
    metadata = persistant_ltm.get_chunk_metadata(chunk_id)
    
    assert metadata is not None
    assert isinstance(metadata, dict)


def test_search_by_type(persistant_ltm):
    """Test de recherche par type"""
    memory_classes = persistant_ltm.search_by_type("class", name_pattern="Memory")

    assert isinstance(memory_classes, list)
    assert len(memory_classes) > 0
    

def test_search_calculator_class(persistant_ltm):
    """Test la recherche de la classe Calculator"""
    
    # Use keyword mode if embeddings not available
    mode = "semantic" if EMBEDDINGS_AVAILABLE else "keyword"
    results = persistant_ltm.search("Calculator class with math operations", top_k=2, mode=mode)
    
    if not results:
        pytest.skip("No search results available")
    
    for r in results:
        assert 'score' in r
        assert 'start_line' in r
        assert 'end_line' in r
        assert 'content' in r
        # Score can be negative for semantic search, just check it exists
        assert 'score' in r
    # Vérifier que Calculator est dans les résultats
    assert any("Calculator" in r['content'] for r in results)
