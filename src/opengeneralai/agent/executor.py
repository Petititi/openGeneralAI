"""Actions: the LLM picks one tool call per turn, the executor runs it."""

from __future__ import annotations

import logging
from typing import Any

from opengeneralai.agent import prompts
from opengeneralai.agent.run import Run
from opengeneralai.errors import ToolCallError
from opengeneralai.json_parsing import JSONParseError, parse_llm_json
from opengeneralai.tools.registry import ToolRegistry
from opengeneralai.tracing.logger import _digest_messages

logger = logging.getLogger(__name__)


def parse_action(raw: str) -> tuple[str, dict[str, Any]]:
    """Tool name and arguments of an action like {"tool_name": "...", "arguments": {...}}."""
    text = raw.strip()
    if text.startswith("ACTION:"):
        text = text[len("ACTION:"):]
    try:
        data = parse_llm_json(text)
    except JSONParseError as e:
        raise ToolCallError(f"the action is not valid JSON ({e})") from e
    tool_name = str(data.get("tool_name", "")).strip()
    arguments = data.get("arguments", {})
    if not tool_name:
        raise ToolCallError('the action has no "tool_name"')
    if not isinstance(arguments, dict):
        raise ToolCallError('"arguments" must be a JSON object', tool_name)
    return tool_name, arguments


class Executor:
    def __init__(self, tools: ToolRegistry):
        self.tools = tools
        self.prompt = prompts.ACTION.format(tool_signatures=tools.signatures())

    def act(self, run: Run, messages: list[dict]) -> list[dict]:
        """Ask for one action and run it.

        Returns the messages to add to the conversation: the action as the LLM wrote it, then
        its result (or why it could not run), so that the LLM sees both at the next turn.
        """
        messages = messages + [{"role": "system", "content": self.prompt}]
        run.trace.add_node(phase="act/run", turn=run.turn, tags={"digest": _digest_messages(messages)})
        raw = run.ask(messages)
        action = {"role": "assistant", "content": raw}

        try:
            tool_name, arguments = parse_action(raw)
            if tool_name not in self.tools.names():
                raise ToolCallError(f"unknown tool '{tool_name}', use one of: {', '.join(self.tools.names())}",
                                    tool_name)
        except ToolCallError as e:
            run.trace.set_tool(tool_name=e.tool_name or "?", tool_input=raw, error=str(e))
            return [action, {"role": "user", "content": prompts.INVALID_ACTION.format(error=e)}]

        tool = self.tools.get(tool_name)
        call_args = dict(arguments)
        if tool.needs_llm:
            call_args["ask_llm"] = run.ask_from_tool
        try:
            result = tool.run(**call_args)
        except Exception as e:  # a tool crash is reported to the agent, which can try something else
            logger.exception("Tool %s crashed", tool_name)
            error = f"{type(e).__name__}: {e}"
            run.trace.set_tool(tool_name=tool_name, tool_input=raw, error=error)
            return [action, {"role": "user", "content": prompts.TOOL_FAILED.format(tool_name=tool_name, error=error)}]

        if not result.ok:
            error = result.meta.get("error", "unknown error")
            run.trace.set_tool(tool_name=tool_name, tool_input=raw, tool_output=result, error=error)
            return [action, {"role": "user", "content": prompts.TOOL_FAILED.format(tool_name=tool_name, error=error)}]

        run.trace.set_tool(tool_name=tool_name, tool_input=raw, tool_output=result)
        return [action, {"role": "user", "content": prompts.TOOL_SUCCEEDED.format(tool_name=tool_name,
                                                                                   output=result.content)}]
