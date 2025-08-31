

import json
import configurator
import agents.tools.ToolRegistry as tools_module


class ActionAgent:
    def __init__(self, cfg: configurator.AppConfig, tools: tools_module.ToolRegistry):
        self.cfg = cfg
        self.tools = tools

        self.core_prompt = """Tool discipline:
* Use only explicitly named tools.
* One tool ACTION per turn (no text); make it idempotent when possible.

Only output the JSON object for the ACTION.

ACTION:
{
"tool_name": "...",
"arguments": { ... }
}

TOOLS:
"""
        self.core_prompt += self.tools.signatures()

    def get_tool_message(self) -> dict:
        return {"role": "system", "content": self.core_prompt}

    def execute_tool(self, raw_response: str) -> str:
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:-3]
        if raw_response.startswith("```"):
            raw_response = raw_response[3:-3]
        if raw_response.startswith("ACTION:"):
            raw_response = raw_response[7:]
        # parse the raw_response to extract the tool_name and arguments
        try:
            response_json = json.loads(raw_response)
            tool_name = response_json.get("tool_name", "")
            arguments = response_json.get("arguments", {})
        except json.JSONDecodeError:
            raise ValueError("Response is not valid JSON")

        if tool_name not in self.tools.names():
            raise ValueError(f"Tool '{tool_name}' is not recognized.")

        tool = self.tools.get(tool_name)
        result = tool.run(**arguments)
        if not result.ok:
            raise ValueError(f"Tool '{tool_name}' execution failed: {result.meta.get('error', 'Unknown error')}")

        return f"Tool '{tool_name}': {result.meta}\nContent:\n```\n{result.content}\n```"