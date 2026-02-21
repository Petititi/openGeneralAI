"""
Code Parser Module - Tree-sitter based code analysis.

This module is responsible for:
- Parsing source code files using tree-sitter
- Extracting structural elements (classes, functions, methods, attributes)
- Extracting imports and dependencies
- Creating text chunks for indexing

Responsibility: Pure code parsing and text extraction.
No database, storage, or embedding concerns.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Tree-sitter imports with graceful fallback
try:
    from tree_sitter_language_pack import get_parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    get_parser = None

# Supported code file extensions to language mapping
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

# Text file extensions
TEXT_EXT = {".md", ".txt", ".rst", ".log", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".json"}


def is_supported_code_file(path: Path) -> bool:
    """Check if a file is a supported code file based on extension."""
    return path.suffix.lower() in SUPPORTED_CODE_EXT


def is_text_file(path: Path) -> bool:
    """Check if a file is a text file based on extension."""
    return path.suffix.lower() in TEXT_EXT


def get_language_for_file(path: Path) -> Optional[str]:
    """Get the language for a file based on its extension."""
    return SUPPORTED_CODE_EXT.get(path.suffix.lower())


def read_text_safe(path: Path, max_bytes: int = 10_000_000) -> bytes:
    """Read file content safely with size limit."""
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return data


# --------------------------
# Tree-sitter parsing functions
# --------------------------

def extract_name_from_node(node, source: bytes) -> Optional[str]:
    """Extract the name of a function/class/method from a tree-sitter node."""
    for child in node.children:
        if child.type in {"identifier", "name"}:
            return source[child.start_byte:child.end_byte].decode("utf-8")
    return None


def extract_class_attributes(root_node, source: bytes, language: str) -> List[Tuple[int, int, str, Dict]]:
    """
    Extract class attributes (class variables, properties, fields, etc.).
    Returns a list of tuples (start_line, end_line, text, metadata).
    """
    attributes = []
    
    # Node types for attributes by language
    ATTRIBUTE_TYPES = {
        "python": {"assignment", "expression_statement"},
        "javascript": {"field_definition", "public_field_definition"},
        "typescript": {"field_definition", "public_field_definition", "property_signature"},
        "tsx": {"field_definition", "public_field_definition", "property_signature"},
        "java": {"field_declaration"},
        "c_sharp": {"field_declaration", "property_declaration"},
        "cpp": {"field_declaration"},
        "c": {"field_declaration"},
        "go": {"field_declaration"},
        "rust": {"field_declaration"},
        "php": {"property_declaration"},
        "ruby": {"assignment", "instance_variable"},
        "kotlin": {"property_declaration"},
        "swift": {"property_declaration"},
    }
    
    CLASS_TYPES = {
        "class_definition",
        "class_declaration",
        "interface_declaration",
        "struct_specifier",
    }
    
    attribute_types = ATTRIBUTE_TYPES.get(language, set())
    if not attribute_types:
        return []
    
    def is_class_level_attribute(node, parent_class_node):
        """Check if a node is a class-level attribute (not inside a method)."""
        if parent_class_node is None:
            return False
        
        # Traverse up to verify we're directly in the class body
        current = node.parent
        while current and current != parent_class_node:
            # If we cross a function/method, it's not a class attribute
            if current.type in {"function_definition", "method_definition", "function_declaration"}:
                return False
            current = current.parent
        
        return current == parent_class_node
    
    def walk_for_attributes(node, parent_class_node=None):
        # Detect if it's a class
        if node.type in CLASS_TYPES:
            parent_class_node = node
            class_name = extract_name_from_node(node, source)
        
        # Check if it's a class attribute
        if node.type in attribute_types and parent_class_node:
            if is_class_level_attribute(node, parent_class_node):
                # Extract information
                start = node.start_point[0] + 1
                end = node.end_point[0] + 1
                text = source[node.start_byte:node.end_byte].decode("utf-8").strip()
                
                if text:
                    # Extract attribute name
                    attr_name = None
                    if language == "python":
                        # For Python: look for pattern "name = value"
                        for child in node.children:
                            if child.type == "assignment":
                                left = child.child_by_field_name("left")
                                if left and left.type == "identifier":
                                    attr_name = source[left.start_byte:left.end_byte].decode("utf-8")
                                    break
                            elif child.type == "identifier":
                                attr_name = source[child.start_byte:child.end_byte].decode("utf-8")
                                break
                    else:
                        # For other languages: look for identifier or declarator
                        for child in node.children:
                            if child.type in {"identifier", "variable_declarator", "property_identifier"}:
                                if child.type == "variable_declarator":
                                    name_node = child.child_by_field_name("name")
                                    if name_node:
                                        attr_name = source[name_node.start_byte:name_node.end_byte].decode("utf-8")
                                else:
                                    attr_name = source[child.start_byte:child.end_byte].decode("utf-8")
                                break
                    
                    parent_class_name = extract_name_from_node(parent_class_node, source)
                    metadata = {
                        "type": "class_attribute",
                        "name": attr_name,
                        "parent_class": parent_class_name
                    }
                    attributes.append((start, end, text, metadata))
        
        # Continue traversal
        for child in node.children:
            walk_for_attributes(child, parent_class_node)
    
    walk_for_attributes(root_node)
    return attributes


def extract_imports(root_node, source: bytes, _language: str) -> List[str]:
    """Extract all imports/includes from a file."""
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
    Extract code chunks (functions/classes/attributes) from source code.
    
    Returns:
        - chunks: List of (start_line, end_line, text, metadata) tuples
        - imports: List of import statements
        - class_methods_map: Dict mapping class names to list of chunk indices
    
    metadata contains: {type, name, parent_class}
    If no structured elements found, fallback: chunk by ~120 lines.
    """
    if not TREE_SITTER_AVAILABLE or get_parser is None:
        # Fallback when tree-sitter not available
        lines = source.decode("utf-8").splitlines()
        chunks = []
        step = 120
        for i in range(0, len(lines), step):
            part = "\n".join(lines[i:i+step])
            if part.strip():
                chunks.append((i+1, min(i+step, len(lines)), part, {"type": "text"}))
        return chunks, [], {}
    
    try:
        parser = get_parser(language)
    except LookupError as e:
        print(f"[WARN] extract_code_chunks: no parser for '{language}': {e}", file=sys.stderr)
        # Fallback lines
        lines = source.decode("utf-8").splitlines()
        chunks = []
        step = 120
        for i in range(0, len(lines), step):
            part = "\n".join(lines[i:i+step])
            if part.strip():
                chunks.append((i+1, min(i+step, len(lines)), part, {"type": "text"}))
        return chunks, [], {}
    
    tree = parser.parse(source)
    root = tree.root_node

    # Extract imports once for the entire file
    imports = extract_imports(root, source, language)
    
    # Extract class attributes
    class_attributes = extract_class_attributes(root, source, language)

    # Node type names by language
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
            # Check that it's the first element in the block
            first_child = grandparent.children[0] if grandparent.children else None
            return first_child == parent and grandparent.type in ("module", "class_definition", "function_definition")
        elif node.type == "comment" and node.parent is not None:
            parent = node.parent
            next_node = node.next_sibling
            return next_node is not None and next_node.type == "class_definition"

    
    def walk(node, parent_class=None):
        # Identify node type
        chunk_type = None
        if node.type in CLASS_TYPES:
            chunk_type = "class"
        elif node.type in METHOD_TYPES:
            chunk_type = "method"
        elif node.type in FUNCTION_TYPES:
            if parent_class:
                chunk_type = "method"
            else:
                chunk_type = "function"
        elif node.type in MODULE_TYPES:
            chunk_type = "module"
        
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
                
                # If it's a method, add it to its class map
                if chunk_type == "method" and parent_class:
                    if parent_class not in class_methods_map:
                        class_methods_map[parent_class] = []
                    class_methods_map[parent_class].append(chunk_index)
                
                # If it's a class, mark child methods with this parent
                if chunk_type == "class":
                    parent_class = name
        
        # Continue recursive walk
        for c in node.children:
            walk(c, parent_class)

    walk(root)
    
    # Add extracted class attributes to chunks
    for start, end, text, metadata in class_attributes:
        chunk_index = len(chunks)
        chunks.append((start, end, text, metadata))
        
        # Add attribute to its parent class map
        parent_class = metadata.get("parent_class")
        if parent_class:
            if parent_class not in class_methods_map:
                class_methods_map[parent_class] = []
            class_methods_map[parent_class].append(chunk_index)

    if not chunks:
        # Fallback: large blocks
        lines = source.decode("utf-8").splitlines()
        step = 120
        for i in range(0, len(lines), step):
            part = "\n".join(lines[i:i+step])
            if part.strip():
                chunks.append((i+1, min(i+step, len(lines)), part, {"type": "block"}))
        return chunks, imports, {}

    # Limit chunk size (~1500 tokens equivalent) by character count
    MAX_CHARS = 6000
    final = []
    for s, e, t, meta in chunks:
        if len(t) <= MAX_CHARS:
            final.append((s, e, t, meta))
        else:
            # re-split by lines
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
    
    return final, imports, class_methods_map


def extract_text_chunks(source: bytes, max_chars: int = 3000) -> Tuple[List[Tuple[int, int, str, Dict]], List[str], Dict[str, List[int]]]:
    """
    Extract text chunks from a text file.
    
    Returns:
        - chunks: List of (start_line, end_line, text, metadata) tuples
        - imports: Empty list (no imports in text files)
        - class_methods_map: Empty dict (no classes in text files)
    """
    # Split by paragraphs, then pack up to ~max_chars
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


def parse_file(path: Path) -> Tuple[List[Tuple[int, int, str, Dict]], List[str], Dict[str, List[int]]]:
    """
    Parse a file and extract chunks, imports, and class-method mapping.
    
    Automatically detects whether the file is code or text based on extension.
    
    Returns:
        - chunks: List of (start_line, end_line, text, metadata) tuples
        - imports: List of import statements
        - class_methods_map: Dict mapping class names to chunk indices
    """
    ext = path.suffix.lower()
    
    if ext in SUPPORTED_CODE_EXT:
        language = SUPPORTED_CODE_EXT[ext]
        source = read_text_safe(path)
        return extract_code_chunks(source, language)
    elif ext in TEXT_EXT:
        source = read_text_safe(path)
        return extract_text_chunks(source)
    else:
        raise ValueError(f"Unsupported file type for indexing: {ext}")
