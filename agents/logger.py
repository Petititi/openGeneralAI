from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple, Literal
import time, uuid, json, hashlib
import copy
from pathlib import Path

Phase = Literal["start", "plan/create", "plan/update", "plan/recover", "act/run", "done"]

def _digest_messages(messages: List[Dict[str, Any]]) -> str:
    joined = "\n".join(f"{m.get('role')}:{m.get('content','')}" for m in messages)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()

@dataclass
class LogNode:
    id: str
    parent_id: Optional[str]
    turn: int
    phase: Phase
    timestamp: float

    # Messages & réponses
    prompt_messages: Optional[List[Dict[str, Any]]] = None      # List of "system" + "user" messages
    raw_response: Optional[str] = None                          # response of LLM

    # Plan / état
    plan_snapshot: Optional[Dict[str, str]] = None
    is_done: Optional[bool] = None

    # Action/outils
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None
    tool_output: Optional[str] = None

    # Meta
    token_usage: Optional[List[int]] = None                 # {"prompt":..., "completion":...}
    duration_s: Optional[float] = None
    error: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)


class TrajectoryLogger:
    def get_current_interaction(self) -> List[dict]:
        """
        Returns a list containing the prompt messages and the raw response from the current node.
        Equivalent to clean_message_history(scratchpad[-1]) for the current node.
        """
        node = self.current_node()
        result = []
        if node.prompt_messages:
            result.extend(copy.deepcopy(node.prompt_messages))
        if node.raw_response is not None:
            result.append({"role": "assistant", "content": node.raw_response})
        if node.tool_name:
            if node.tool_output and node.tool_output.meta:
                params = ",".join(node.tool_output.meta.values())
                result.append({"role": "user", "content": f"Tool success: '{node.tool_name}'({params})\nContent:\n```\n{node.tool_output.content}\n```"})
            elif node.error:
                result.append({"role": "user", "content": node.error})
            else:
                assert False, "Logger node has tool_name but no tool_output or error"
        return result
    
    """Logger structuré en arbre de décision pour tracer chaque chemin de l'agent."""
    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or str(uuid.uuid4())
        self.nodes: Dict[str, LogNode] = {}
        self.children: Dict[str, List[str]] = {}
        self.root_id: Optional[str] = None
        # Last node added: the next node becomes its child. Kept per trace, so that a new
        # trace never hangs its nodes under the last node of a previous one.
        self.current_node_id: Optional[str] = None

    # --- création & mise à jour ------------------------------------------------
    def _new_id(self) -> str:
        return str(uuid.uuid4())

    def add_node(
        self,
        *,
        phase: Phase,
        turn: int,
        tags: Optional[Dict[str, Any]] = None,
        token_usage: Optional[List[int]] = None,
    ) -> str:
        node_id = self._new_id()
        parent_id = self.current_node_id
        node = LogNode(
            id=node_id,
            parent_id=parent_id,
            turn=turn,
            phase=phase,
            timestamp=time.time(),
            tags=tags or {},
            token_usage=token_usage,
        )
        self.nodes[node_id] = node
        if parent_id:
            self.children.setdefault(parent_id, []).append(node_id)
        else:
            self.root_id = node_id
        self.current_node_id = node_id
        return node_id

    def current_node(self) -> LogNode:
        node_id = self.current_node_id
        if node_id is None:
            raise Exception("No current node: call add_node() first")
        return self.nodes[node_id]

    def set_questions(self, prompt_messages: List[Dict[str, Any]]):
        n = self.current_node()
        n.prompt_messages = copy.deepcopy(prompt_messages)

    def set_response(self, raw_response: str, token_usage: Optional[List[int]] = None, duration_s: Optional[float] = None):
        n = self.current_node()
        n.raw_response = raw_response
        n.token_usage = token_usage
        n.duration_s = duration_s

    def set_plan(self, plan: Dict[str, Any], is_done: bool):
        n = self.current_node()
        n.plan_snapshot = copy.deepcopy(plan)
        n.is_done = is_done

    def set_tool(self, tool_name: str, tool_input: Any, tool_output: Any = None, error: Optional[str] = None):
        n = self.current_node()
        n.tool_name = tool_name
        n.tool_input = tool_input
        n.tool_output = tool_output
        n.error = error

    def tag(self, **kv):
        n = self.current_node()
        n.tags.update(kv)

    # --- exports ---------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "root_id": self.root_id,
            "nodes": {nid: asdict(n) for nid, n in self.nodes.items()},
            "children": self.children,
        }

    def to_json(self) -> str:
        """Serialize the trace; objects that are not JSON (e.g. tool results) are written as text."""
        def _default(o):
            try:
                return str(o)
            except Exception:
                return "<non-serializable>"
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=1, default=_default)

    def save(self, path: Path | str) -> Path:
        """Write the trace as JSON, e.g. traces/my_run.json.

        Open it with the viewer: http://localhost:12000/viewer/?trace=/traces/my_run.json
        (see viewer/README.md).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path
