import json5
import configurator
from typing import Tuple

import agents.tools.ToolRegistry as tools_module

class AgentError(Exception):
    def __init__(self, message, tool_name):
        super().__init__(message)

        # Now for your custom code...
        self.tool_name = tool_name

class ActionAgent:
    def __init__(self, cfg: configurator.AppConfig, tools: tools_module.ToolRegistry):
        self.cfg = cfg
        self.tools = tools

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
    
    def safe_json_parsing(self, text: str) -> Tuple[str, dict]:
        try:
            response_json = json5.loads(text)
        except ValueError as e:
            # remove '\'' and '"' from text and try again:
            text = text.replace("\\'", "'")
            try:
                response_json = json5.loads(text)
            except ValueError:
                raise AgentError(str(e), "not valid JSON")
        tool_name = response_json.get("tool_name", "")
        arguments = response_json.get("arguments", {})
        return tool_name, arguments

    def execute_tool(self, raw_response: str) -> Tuple[str, str]:
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:-3]
        if raw_response.startswith("```"):
            raw_response = raw_response[3:-3]
        if raw_response.startswith("ACTION:"):
            raw_response = raw_response[7:]
        # parse the raw_response to extract the tool_name and arguments
        tool_name, arguments = self.safe_json_parsing(raw_response)

        if tool_name not in self.tools.names():
            raise AgentError(f"Tool '{tool_name}' is not recognized.", tool_name)

        tool = self.tools.get(tool_name)
        result = tool.run(**arguments)
        if not result.ok:
            raise AgentError(f"Tool '{tool_name}'({arguments}) execution failed: {result.meta.get('error', 'Unknown error')}", tool_name)
        params = ",".join(result.meta.values())
        return f"Tool used: '{tool_name}'({params})\nContent:\n```\n{result.content}\n```", tool_name