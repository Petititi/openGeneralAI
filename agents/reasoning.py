import configurator
import json5
from typing import Tuple, Callable, List

from agents.logger import TrajectoryLogger, _digest_messages

class ReasoningAgent:
    def __init__(self, cfg: configurator.AppConfig, logger: TrajectoryLogger, ask_llm: Callable[[List[dict]], str]):
        self.cfg = cfg
        self.logger = logger
        self.ask_llm = ask_llm

        self.core_prompt = """Operating principles:
* Produce a brief, checkable PLAN (no raw chain-of-thought) where each steps have a small description, a status and a criteria needed to go to the next step.
* Keep outputs precise and minimal; follow formats exactly.
* Use the status "done" only when proof exists and ensure the last step is a validation of success.

For the output, use only this JSON schemas:
PLAN:
{
"plan_steps": ["descr": "...", "status": done|todo, "next_step_criteria": "..."]
}
"""
        self.continue_plan_prompt = """Operating principles:
* From the previous interactions choose whether to proceed to the NEXT_TASK or build a NEW_PLAN.
* No elaboration.

CURRENT → {task_description}
NEXT_STEP_CRITERIA → {next_step_criteria}

NEXT_TASK → {next_task_description}
NEW_PLAN → Current and next tasks are not working; need a new plan.

OUTPUT SCHEMA:
{
"status": "NEXT_TASK|NEW_PLAN"
}
"""
        self.last_plan_prompt = """Operating principles:
* From the previous interactions, evaluate if we succeeded, according to next step criteria.
* Use either "NEW_PLAN" or "DONE" to indicate the status
* If "DONE", add a description of the success in the output.
* No elaboration.

CUR_TASK → {task_description}
NEXT_STEP_CRITERIA → {next_step_criteria}

NEW_PLAN → Need to adjust plan according to previous interactions.

OUTPUT SCHEMA:
{
"status": "DONE"|"NEW_PLAN"
}
"""
        self.update_plan_prompt = """Operating principles:
* Produce a brief, checkable PLAN (no raw chain-of-thought) where each steps have a small description, a status and a success criteria.
* Keep outputs precise and minimal; follow formats exactly.
* Use the status "done" only when proof of success exists, use "todo" otherwise.
* Ensure the last step is a validation of success.

For the output, use only this JSON schemas:
PLAN:
{
"plan_steps": ["descr": "...", "status": done|todo, "next_step_criteria": "..."]
}
"""

    def planning_message(self) -> dict:
        return {"role": "system", "content": self.core_prompt}

    def create_plan(self, messages: List[dict], turn: int = 0) -> Tuple[dict, bool]:
        """Create a new plan using the provided LLM callback function."""
        messages_to_send = messages.copy()
        messages_to_send.append(self.planning_message())
        
        # Create trace node for plan creation
        self.logger.add_node(
            phase="plan/create",
            turn=turn,
            tags={"digest": _digest_messages(messages_to_send)}
        )

        raw_response = self.ask_llm(messages_to_send)
        return self.get_plan(raw_response)

    def update_plan(self, messages: List[dict], plan: dict, improve_plan: bool = False, turn: int = 0) -> Tuple[dict, bool]:
        try:
            messages_to_send = messages.copy()
            messages_to_send.append(self.reevaluate_plan_message(plan, improve_plan))
            
            # Create trace node for plan update
            phase = "plan/update" if not improve_plan else "plan/recover"
            extra_tags = {"reason": "update_failed"} if improve_plan else {}
            tags = {"digest": _digest_messages(messages_to_send)}
            tags.update(extra_tags)
            
            self.logger.add_node(
                phase=phase,
                turn=turn,
                tags=tags
            )
            
            raw_response = self.ask_llm(messages_to_send)
            if improve_plan:
                return self.get_plan(raw_response)
            else:
                return self._update_plan(plan, raw_response)
        except ValueError:
            # Fallback to improved plan creation
            messages_to_send = messages.copy()
            messages_to_send.append(self.reevaluate_plan_message(plan, improve_plan=True))
            
            # Create trace node for plan recovery
            self.logger.add_node(
                phase="plan/recover",
                turn=turn,
                tags={"digest": _digest_messages(messages_to_send), "reason": "update_failed"}
            )
            
            raw_response = self.ask_llm(messages_to_send)
            return self.get_plan(raw_response)

    def get_plan(self, raw_response: str) -> Tuple[dict, bool]:
        node_id = self.logger.current_node().id
        # two possibilities: raw_response start with "FINAL" or "PLAN"
        json_data = None
        is_done = False
        if raw_response.startswith("STOP"):
            return {}, True
        if raw_response.startswith("FINAL:"):
            json_data = raw_response[len("FINAL:"):].strip()
            is_done = True
        elif raw_response.startswith("PLAN:"):
            json_data = raw_response[len("PLAN:"):].strip()
        if json_data is None:
            # try with the full response:
            json_data = raw_response.strip()

        # Check if the answer is well-formed according to the expected JSON schema
        try:
            parsed_json = json5.loads(json_data)
            # Defensive: ensure parsed_json is a dict
            if not isinstance(parsed_json, dict):
                raise ValueError("Parsed response is not a dict")
            done_flag = is_done or parsed_json.get("done", False) or all(isinstance(step, dict) and step.get("status") == "done" for step in parsed_json.get("plan_steps", []))
            if self.logger and node_id is not None:
                self.logger.set_plan(plan=parsed_json, is_done=done_flag)
            return parsed_json, done_flag
        except ValueError as exc:
            raise ValueError("Response is not valid JSON") from exc

    def get_current_step_idx(self, plan: dict) -> int:
        first_todo_step = -1
        steps = plan.get("plan_steps", [])
        for i, step in enumerate(steps):
            if step.get("status", "").lower() == "current":
                return i
            elif step.get("status", "").lower() != "done" and first_todo_step < 0:
                first_todo_step = i
        return first_todo_step

    def get_step_description(self, plan: dict) -> dict:
        output = "Action plan:\n"
        # construct the history of done steps:
        for step in plan.get("plan_steps", []):
            if step.get("status", "").lower() == "done":
                output += f"Done: {step.get('descr', '')}\n"

        todo_step_idx = self.get_current_step_idx(plan)
        output += f"TODO: {plan['plan_steps'][todo_step_idx].get('descr', '')}\n"
        return {"role": "assistant", "content": output}

    def reevaluate_plan_message(self, plan: dict, improve_plan: bool=False) -> dict:
        todo_step_idx = self.get_current_step_idx(plan)
        if todo_step_idx < 0:
            raise ValueError("plan doesn't have a current step:" + json5.dumps(plan))

        content_message = ""
        if improve_plan:
            content_message = self.update_plan_prompt.replace(
                "{existing_plan}", json5.dumps(plan))
        else:
            cur_task = plan['plan_steps'][todo_step_idx]
            if todo_step_idx + 1 < len(plan['plan_steps']):
                next_task = plan['plan_steps'][todo_step_idx + 1].get('descr', '')
                content_message = self.continue_plan_prompt.replace(
                    "{task_description}", cur_task.get('descr', '')).replace(
                    "{next_task_description}", next_task).replace(
                    "{next_step_criteria}", cur_task.get('next_step_criteria', ''))
            else:
                content_message = self.last_plan_prompt.replace(
                    "{task_description}", cur_task.get('descr', '')).replace(
                    "{next_step_criteria}", cur_task.get('next_step_criteria', ''))
        return {"role": "system", "content": content_message}

    def _update_plan(self, plan: dict, raw_response: str) -> Tuple[dict, bool]:
        node_id = self.logger.current_node().id
        try:
            parsed_json = json5.loads(raw_response)
            if not isinstance(parsed_json, dict):
                raise ValueError("Parsed response is not a dict")
            if "status" not in parsed_json:
                raise ValueError("Response JSON is missing 'status' field")
            status = str(parsed_json["status"]).strip().upper()
        except ValueError as exc:
            raise ValueError("Response is not valid JSON") from exc

        if status == "NEXT_TASK":
            # Mark the current step as done
            current_step = plan['plan_steps'][self.get_current_step_idx(plan)]
            current_step["status"] = "done"
        elif status == "NEW_PLAN":
            current_step = self.get_current_step_idx(plan)
            plan['plan_steps'][current_step]["status"] = "error"
            if current_step>0:
                plan['plan_steps'][current_step-1]["status"] = "error"
            raise ValueError("Need a 'NEW_PLAN'.")
        elif status == "DONE":
            # Mark all step as done
            for step in plan['plan_steps']:
                step["status"] = "done"
        else:
            raise ValueError("Expecting either 'DONE', 'NEXT_TASK', or 'NEW_PLAN' from the reasoning agent.")

        # Logging
        if self.logger and node_id is not None:
            self.logger.set_plan(plan=plan, is_done=all(step.get("status") == "done" for step in plan["plan_steps"]))
        return plan, all(step.get("status") == "done" for step in plan["plan_steps"])
