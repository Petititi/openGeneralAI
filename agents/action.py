import json5
import configurator
from typing import Tuple, Callable, List
import json

import agents.tools.ToolRegistry as tools_module
from agents.logger import TrajectoryLogger, _digest_messages

class AgentError(Exception):
    def __init__(self, message, tool_name):
        super().__init__(message)

        # Now for your custom code...
        self.tool_name = tool_name

class ActionAgent:
    def __init__(self, cfg: configurator.AppConfig, logger: TrajectoryLogger, tools: tools_module.ToolRegistry, ask_llm: Callable[[List[dict]], str]):
        self.cfg = cfg
        self.tools = tools
        self.logger = logger
        self.ask_llm = ask_llm

        self.core_prompt = """Tool discipline:
* Use only explicitly named tools.
* One tool ACTION per turn (no text); make it idempotent when possible.

Only output this JSON:

{
"tool_name": "...",
"arguments": { ... }
}

TOOLS:
"""
        self.core_prompt += self.tools.signatures()

    def get_tool_message(self) -> dict:
        return {"role": "system", "content": self.core_prompt}

    def execute_action_with_callback(self, messages: List[dict], turn: int = 0) -> Tuple[str, str]:
        """Execute a tool action using the provided LLM callback function."""
        messages_to_send = messages.copy()
        tool_msg = self.get_tool_message()
        messages_to_send.append(tool_msg)
        
        # Create trace node for action execution
        self.logger.add_node(
            phase="act/run",
            turn=turn,
            tags={"digest": _digest_messages(messages_to_send)}
        )
        
        raw_response = self.ask_llm(messages_to_send)
        
        try:
            result, tool_name = self.execute_tool(raw_response)
            return json.dumps(result), tool_name
        except AgentError as e:
            return str(e), e.tool_name
        except Exception as e:
            return str(e), "unknown"
    
    def safe_json_parsing(self, text: str) -> Tuple[str, dict]:
        try:
            response_json = json5.loads(text)
        except ValueError:
            try:
                # second try: handle escaped unicode characters
                fixed_text = text.encode("utf-8").decode("unicode_escape")
                response_json = json5.loads(fixed_text)
            except ValueError as exc:
                raise AgentError(str(exc), "not valid JSON") from exc
        
        if not isinstance(response_json, dict):
            raise AgentError("Parsed response is not a dict", "not valid JSON")
        tool_name = response_json.get("tool_name", "")
        arguments = response_json.get("arguments", {})
        return tool_name, arguments

    def execute_tool(self, raw_response: str) -> Tuple[str, str]:
        node_id = self.logger.current_node().id
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:-3]
        if raw_response.startswith("```"):
            raw_response = raw_response[3:-3]
        if raw_response.startswith("ACTION:"):
            raw_response = raw_response[7:]
        # parse the raw_response to extract the tool_name and arguments
        tool_name, arguments = self.safe_json_parsing(raw_response)

        if tool_name not in self.tools.names():
            self.logger.set_tool(tool_name=tool_name, tool_input=raw_response, error=f"Tool '{tool_name}' is not recognized.")
            raise AgentError(f"Tool '{tool_name}' is not recognized.", tool_name)

        tool = self.tools.get(tool_name)
        result = tool.run(**arguments)
        if not result.ok:
            if self.logger and node_id is not None:
                self.logger.set_tool(tool_name=tool_name, tool_input=raw_response, error=f"Tool '{tool_name}'({arguments}) execution failed: {result.meta.get('error', 'Unknown error')}")
            raise AgentError(f"Tool '{tool_name}'({arguments}) execution failed: {result.meta.get('error', 'Unknown error')}", tool_name)
        params = ",".join(result.meta.values())
        if self.logger and node_id is not None:
            self.logger.set_tool(tool_name=tool_name, tool_input=raw_response, tool_output=result)
        return f"Tool used: '{tool_name}'({params})\nContent:\n```\n{result.content}\n```", tool_name