"""Planning: write the plan, follow it step by step, write a new one when a step fails."""

from __future__ import annotations

from typing import Callable

from opengeneralai.agent import prompts
from opengeneralai.agent.plan import Plan, PlanDecision, StepStatus, parse_decision, parse_plan
from opengeneralai.agent.run import Run
from opengeneralai.errors import PlanParseError
from opengeneralai.tracing.logger import Phase, _digest_messages

# Attempts for one planning answer: the first one plus retries after an invalid answer
MAX_ATTEMPTS = 2


def system(content: str) -> dict:
    return {"role": "system", "content": content}


class Planner:
    def create(self, run: Run, messages: list[dict]) -> Plan:
        """First plan of the task."""
        return self._ask_plan(run, "plan/create", messages + [system(prompts.CREATE_PLAN.format())])

    def update(self, run: Run, messages: list[dict], plan: Plan) -> Plan:
        """After an action: move to the next step, finish, or write a new plan."""
        index = plan.current_index()
        if index is None:
            return plan
        step = plan.steps[index]
        if index + 1 < len(plan.steps):
            prompt = prompts.CONTINUE_PLAN.format(
                task_description=step.descr, next_step_criteria=step.next_step_criteria,
                next_task_description=plan.steps[index + 1].descr)
        else:
            prompt = prompts.LAST_STEP.format(task_description=step.descr, next_step_criteria=step.next_step_criteria)

        try:
            decision = self._ask(run, "plan/update", messages + [system(prompt)], parse_decision)
        except PlanParseError as e:
            return self.recover(run, messages, plan, reason=f"invalid decision: {e}")

        if decision == PlanDecision.NEXT_TASK:
            step.status = StepStatus.DONE
        elif decision == PlanDecision.DONE:
            for s in plan.steps:
                s.status = StepStatus.DONE
        else:  # NEW_PLAN: the current step (and the one before, which led to it) failed
            step.status = StepStatus.ERROR
            if index > 0:
                plan.steps[index - 1].status = StepStatus.ERROR
            run.trace.set_plan(plan.to_dict(), plan.is_done)
            return self.recover(run, messages, plan, reason="NEW_PLAN")
        run.trace.set_plan(plan.to_dict(), plan.is_done)
        return plan

    def recover(self, run: Run, messages: list[dict], plan: Plan, reason: str) -> Plan:
        """New plan, written from the existing one (failed steps marked "error")."""
        prompt = prompts.NEW_PLAN.format(existing_plan=plan.to_json())
        return self._ask_plan(run, "plan/recover", messages + [system(prompt)], tags={"reason": reason})

    def _ask_plan(self, run: Run, phase: Phase, messages: list[dict], tags: dict | None = None) -> Plan:
        plan = self._ask(run, phase, messages, parse_plan, tags)
        run.trace.set_plan(plan.to_dict(), plan.is_done)
        return plan

    def _ask(self, run: Run, phase: Phase, messages: list[dict], parse: Callable, tags: dict | None = None):
        """Ask until the answer parses, one trace node per attempt; raise PlanParseError at the end."""
        for attempt in range(MAX_ATTEMPTS):
            node_tags = {"digest": _digest_messages(messages), **(tags or {})}
            if attempt:
                node_tags["retry"] = attempt
            run.trace.add_node(phase=phase, turn=run.turn, tags=node_tags)
            raw = run.ask(messages)
            try:
                return parse(raw)
            except PlanParseError as e:
                error = e
                messages = messages + [{"role": "assistant", "content": raw},
                                       {"role": "user", "content": prompts.INVALID_ANSWER.format(error=e)}]
        raise error
