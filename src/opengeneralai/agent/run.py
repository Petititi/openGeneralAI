"""State of one answer to one question.

The orchestrator itself keeps no state between questions: everything that belongs to one
question (trace, token usage, current turn) lives in a Run, so that two questions answered
at the same time cannot mix their traces or their costs.
"""

from __future__ import annotations

from opengeneralai.llm.client import LLMClient, Usage
from opengeneralai.tracing.logger import TrajectoryLogger


class Run:
    def __init__(self, llm: LLMClient, trace: TrajectoryLogger | None = None):
        self.llm = llm
        self.trace = trace or TrajectoryLogger()
        self.usage = Usage()
        self.turn = 0

    def ask(self, messages: list[dict], node_id: str | None = None) -> str:
        """Call the LLM and record the messages, the answer and its usage in the trace node
        (the current node by default)."""
        self.trace.set_questions(messages, node_id=node_id)
        response = self.llm.complete(messages)
        self.usage += response.usage
        u = response.usage
        self.trace.set_response(
            response.text,
            token_usage=[u.total_tokens, u.prompt_tokens, u.completion_tokens],
            duration_s=response.duration_s,
            node_id=node_id,
        )
        return response.text

    def ask_from_tool(self, messages: list[dict]) -> str:
        """LLM call made by a tool: recorded in its own node, under the action node."""
        node_id = self.trace.add_side_node(phase="act/llm", turn=self.turn)
        return self.ask(messages, node_id=node_id)

    @property
    def cost(self) -> float:
        return self.llm.cost(self.usage)
