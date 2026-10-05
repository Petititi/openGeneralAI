"""Debug tree-sitter pour comprendre la structure du code Python"""

import pytest
from tree_sitter_language_pack import get_parser


@pytest.fixture
def sample_code():
    """Fixture providing sample Python code for testing"""
    return '''
class Motocultor:
    """A simple motocultor class"""
    
    def start(self, a, b):
        """Start the motocultor"""
        return a + b
    
    def end(self, a, b):
        """Stop the motocultor"""
        return a - b

def standalone_function():
    """A standalone function"""
    print("Hello World")
'''


@pytest.fixture
def parser():
    """Fixture providing a Python parser"""
    return get_parser("python")


def print_tree(node, indent=0):
    """Helper function to print the tree structure"""
    print("  " * indent + f"{node.type} [{node.start_point[0]}:{node.start_point[1]} - {node.end_point[0]}:{node.end_point[1]}]")
    for child in node.children:
        print_tree(child, indent + 1)


def test_tree_sitter_parsing(parser, sample_code, capsys):
    """Test tree-sitter parsing of Python code structure"""
    tree = parser.parse(sample_code.encode("utf-8"))
    
    # Verify that the tree was parsed successfully
    assert tree is not None
    assert tree.root_node is not None
    
    # Print the tree structure
    print_tree(tree.root_node)
    
    # Capture and verify output was produced
    captured = capsys.readouterr()
    assert "module" in captured.out
    assert "class_definition" in captured.out
    assert "function_definition" in captured.out


def test_tree_structure_depth(parser, sample_code):
    """Test that the parsed tree has expected structure"""
    tree = parser.parse(sample_code.encode("utf-8"))
    root = tree.root_node
    
    # Verify root node type
    assert root.type == "module"
    
    # Verify we have children nodes
    assert len(root.children) > 0


def test_class_detection(parser, sample_code):
    """Test detection of class definitions in the code"""
    tree = parser.parse(sample_code.encode("utf-8"))
    
    # Find class definitions
    def find_nodes_by_type(node, node_type):
        """Recursively find all nodes of a specific type"""
        nodes = []
        if node.type == node_type:
            nodes.append(node)
        for child in node.children:
            nodes.extend(find_nodes_by_type(child, node_type))
        return nodes
    
    classes = find_nodes_by_type(tree.root_node, "class_definition")
    assert len(classes) >= 1
    

def test_function_detection(parser, sample_code):
    """Test detection of function definitions in the code"""
    tree = parser.parse(sample_code.encode("utf-8"))
    
    def find_nodes_by_type(node, node_type):
        """Recursively find all nodes of a specific type"""
        nodes = []
        if node.type == node_type:
            nodes.append(node)
        for child in node.children:
            nodes.extend(find_nodes_by_type(child, node_type))
        return nodes
    
    functions = find_nodes_by_type(tree.root_node, "function_definition")
    # Should find methods in Calculator class and standalone_function
    assert len(functions) >= 3
