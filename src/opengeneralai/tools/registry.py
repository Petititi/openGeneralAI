# tool_runtime.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol
from abc import ABC, abstractmethod
import time as _time

# ---------- Core types ----------
@dataclass
class ToolResult:
    """
    Represents the result of a tool execution.

    Attributes:
        ok (bool): Indicates whether the tool execution was successful.
        content (str): The main content or output produced by the tool. Defaults to an empty string.
        meta (Dict[str, Any]): Additional metadata related to the tool result.
    """
    ok: bool
    content: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

class Tool(ABC):
    name: str
    signature: str

    @abstractmethod
    def run(self, **kwargs) -> ToolResult: ...

class ToolRegistry:
    """
    A registry for managing and retrieving Tool instances by name.

    Attributes:
        _tools (Dict[str, Tool]): Internal dictionary mapping tool names to Tool instances.

    Methods:
        register(tool: Tool):
            Registers a Tool instance in the registry.

        get(name: str) -> Tool:
            Retrieves a Tool instance by its name.

        names() -> List[str]:
            Returns a list of all registered tool names.
    """
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    def register(self, tool: Tool):
        self._tools[tool.name] = tool
    def get(self, name: str) -> Tool:
        return self._tools[name]
    def names(self) -> List[str]:
        return list(self._tools.keys())
    def signatures(self) -> str:
        output = ""
        for tool in self._tools.values():
            output += f"{tool.name}{tool.signature}\n"
        return output
