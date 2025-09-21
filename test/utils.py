import time
from agents.tools.ToolRegistry import Tool, ToolResult

import difflib
from typing import Dict, Optional, Tuple, Callable

# ---------- In-memory FileSystem ----------
class InMemoryFS:
    """
    InMemoryFS provides a simple in-memory file system abstraction.

    Attributes:
        files (Dict[str, str]): A dictionary mapping file paths to their contents.

    Methods:
        __init__(files: Optional[Dict[str, str]] = None):
            Initializes the in-memory file system with optional initial files.

        read(path: str) -> str:
            Reads and returns the content of the file at the given path.
            Raises FileNotFoundError if the file does not exist.

        write(path: str, content: str) -> str:
            Writes content to the file at the given path.
            Returns a unified diff string representing the changes from the previous content.
    """
    def __init__(self, files: Optional[Dict[str, str]] = None):
        self.files = dict(files or {})
    def read(self, path: str) -> str:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]
    def write(self, path: str, content: str) -> str:
        old = self.files.get(path, "")
        self.files[path] = content
        diff = difflib.unified_diff(
            old.splitlines(), content.splitlines(),
            fromfile="before", tofile="after", lineterm=""
        )
        readable_diff = "\n".join(diff)
        return readable_diff


# ---------- Concrete tools (using FS) ----------
class ReadFile(Tool):
    name = "read_file"
    signature = "(filepath:str) → content of any file"
    def __init__(self, fs: InMemoryFS):
        super().__init__()
        self.fs = fs
    def run(self, **kwargs) -> ToolResult:
        path = kwargs.get("filepath")
        if not path:
            return ToolResult(False, meta={"error": "Missing 'filepath'"})
        try:
            content = self.fs.read(path)
            return ToolResult(True, content, meta={"file_path": path})
        except FileNotFoundError as e:
            return ToolResult(False, meta={"error": f"{path}: {str(e)}"})

class EditFile(Tool):
    name = "edit_file"
    signature = "(filepath:str, full_content:str) → diff (old vs new)"

    def __init__(self, fs: InMemoryFS):
        super().__init__()
        self.fs = fs
    def run(self, **kwargs) -> ToolResult:
        path = kwargs.get("filepath")
        content = kwargs.get("full_content")
        if not path or content is None:
            return ToolResult(False, meta={"error": "Missing 'filepath' or 'full_content'"})
        diff = self.fs.write(path, content)
        return ToolResult(True, diff, meta={"file_path": path})

class RunProg(Tool):
    name = "bash"
    signature = "(cmd) → output of bash command"
    def __init__(self, fs: InMemoryFS, validation_lambda: Callable[[str], Tuple[bool, str]]):
        super().__init__()
        self.fs = fs
        self.validation_lambda = validation_lambda

    def run(self, **kwargs) -> ToolResult:
        cmd = kwargs.get("cmd")
        if not cmd:
            return ToolResult(False, meta={"error": "Missing 'cmd'"})

        valid, output = self.validation_lambda(cmd)
        if not valid:
            return ToolResult(False, "", meta={"error": output})
        return ToolResult(True, output, meta={"cmd": cmd})