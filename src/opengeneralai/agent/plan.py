"""The plan the LLM writes and the agent follows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from opengeneralai.errors import PlanParseError
from opengeneralai.json_parsing import JSONParseError, parse_llm_json


class StepStatus(str, Enum):
    TODO = "todo"
    CURRENT = "current"
    DONE = "done"
    ERROR = "error"


class PlanDecision(str, Enum):
    """What the LLM decides after an action."""

    NEXT_TASK = "NEXT_TASK"   # the current step succeeded
    NEW_PLAN = "NEW_PLAN"     # the current step failed: write a new plan
    DONE = "DONE"             # the whole task succeeded


@dataclass
class Step:
    descr: str
    status: StepStatus = StepStatus.TODO
    next_step_criteria: str = ""


@dataclass
class Plan:
    steps: list[Step] = field(default_factory=list)
    # The LLM answered that the task was already done (FINAL) or that it cannot comply (STOP)
    final: bool = False

    @property
    def is_done(self) -> bool:
        return self.final or all(step.status == StepStatus.DONE for step in self.steps)

    def current_index(self) -> Optional[int]:
        """The step marked "current", else the first step not done; None when all are done."""
        for i, step in enumerate(self.steps):
            if step.status == StepStatus.CURRENT:
                return i
        for i, step in enumerate(self.steps):
            if step.status != StepStatus.DONE:
                return i
        return None

    def description(self) -> str:
        """Steps done and the step to do now, for the action prompt."""
        lines = ["Action plan:"]
        lines += [f"Done: {step.descr}" for step in self.steps if step.status == StepStatus.DONE]
        index = self.current_index()
        if index is not None:
            lines.append(f"TODO: {self.steps[index].descr}")
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {"plan_steps": [
            {"descr": s.descr, "status": s.status.value, "next_step_criteria": s.next_step_criteria}
            for s in self.steps
        ]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: Any) -> Plan:
        if not isinstance(data, dict) or not isinstance(data.get("plan_steps"), list):
            raise PlanParseError('Expected a JSON object with a "plan_steps" list')
        steps = []
        for raw in data["plan_steps"]:
            if not isinstance(raw, dict) or not str(raw.get("descr", "")).strip():
                raise PlanParseError('Each plan step needs a "descr"')
            try:
                status = StepStatus(str(raw.get("status", "todo")).strip().lower())
            except ValueError:
                status = StepStatus.TODO
            steps.append(Step(str(raw["descr"]), status, str(raw.get("next_step_criteria", ""))))
        return cls(steps, final=bool(data.get("done", False)))


def parse_plan(raw: str) -> Plan:
    """Plan from an LLM answer: a PLAN JSON, optionally prefixed by "PLAN:" or "FINAL:", or "STOP"."""
    text = raw.strip()
    if text.startswith("STOP"):
        return Plan(final=True)
    final = text.startswith("FINAL:")
    for prefix in ("FINAL:", "PLAN:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    try:
        plan = Plan.from_dict(parse_llm_json(text))
    except JSONParseError as e:
        raise PlanParseError(f"The plan is not valid JSON: {e}") from e
    plan.final = plan.final or final
    return plan


def parse_decision(raw: str) -> PlanDecision:
    """Decision from an LLM answer like {"status": "NEXT_TASK"}."""
    try:
        data = parse_llm_json(raw)
    except JSONParseError as e:
        raise PlanParseError(f"The decision is not valid JSON: {e}") from e
    status = str(data.get("status", "")).strip().upper()
    try:
        return PlanDecision(status)
    except ValueError:
        raise PlanParseError('Expected "status" to be NEXT_TASK, NEW_PLAN or DONE') from None
