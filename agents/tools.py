from dataclasses import dataclass


@dataclass
class ToolResult:
    ok: bool
    content: str
    meta: dict


class Tool:
    name: str
    schema: str

    def run(self, **kwargs) -> ToolResult: ...


class ReadFile(Tool):
    def __init__(self):
        self.name = "read_file"
        self.schema = "(path:str) → content of any file"

    def run(self, **kwargs) -> ToolResult:
        if "path" not in kwargs:
            return ToolResult(ok=False, content="", meta={"error": "Missing 'path' argument"})
        file_path = kwargs.get("path", "")
        return ToolResult(ok=True, content="print('time:' + time.time())\n", meta={"file_path": file_path})


class EditFile(Tool):
    def __init__(self):
        self.name = "edit_file"
        self.schema = "(path:str, content:str) → None"

    def run(self, **kwargs) -> ToolResult:
        if "path" not in kwargs:
            return ToolResult(ok=False, content="", meta={"error": "Missing 'path' argument"})
        if "content" not in kwargs:
            return ToolResult(ok=False, content="", meta={"error": "Missing 'content' argument"})
        file_path = kwargs.get("path", "")
        content = kwargs.get("content", "")
        return ToolResult(ok=True, content="---", meta={"file_path": file_path, "content": content})


class RunProg(Tool):
    def __init__(self):
        self.name = "run"
        self.schema = "(cmd) → None"

    def run(self, **kwargs) -> ToolResult:
        if "cmd" not in kwargs:
            return ToolResult(ok=False, content="", meta={"error": "Missing 'cmd' argument"})
        cmd = kwargs.get("cmd", "")
        return ToolResult(ok=True, content="---", meta={"cmd": cmd})


TOOLS = {
    "read_file": ReadFile(),
    "edit_file": EditFile(),
    "run": RunProg(),
}
