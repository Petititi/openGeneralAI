"""The plan -> act -> check loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from opengeneralai.agent import prompts
from opengeneralai.agent.context import MemoryContext
from opengeneralai.agent.executor import Executor
from opengeneralai.agent.plan import Plan
from opengeneralai.agent.planner import Planner, system
from opengeneralai.agent.run import Run
from opengeneralai.errors import AgentError
from opengeneralai.llm.client import LLMClient, Usage
from opengeneralai.tools.registry import ToolRegistry
from opengeneralai.tracing.logger import TrajectoryLogger

logger = logging.getLogger(__name__)

DEFAULT_MAX_TURNS = 10
# Messages of the conversation sent at each turn, besides the question: the last actions and results
DEFAULT_HISTORY_SIZE = 12


@dataclass
class RunResult:
    """Outcome of one question."""

    plan: Plan
    done: bool                  # False when the turn limit was reached, or on error
    turns: int
    usage: Usage
    cost: float                 # USD, 0 when the model price is unknown
    trace: TrajectoryLogger
    error: Optional[str] = None  # why the agent stopped, when it could not follow its protocol


class Orchestrator:
    """Answers questions with the plan/act/check loop.

    It keeps no state between questions (see Run): one instance can serve several requests.
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        *,
        user_lang: str = "English",
        memory: Optional[MemoryContext] = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        history_size: int = DEFAULT_HISTORY_SIZE,
    ):
        self.llm = llm
        self.memory = memory
        self.max_turns = max_turns
        self.history_size = history_size
        self.core_prompt = prompts.core_prompt(user_lang)
        self.planner = Planner()
        self.executor = Executor(tools)

    def run(self, question: str) -> RunResult:
        """Answer one question; the trace of the run is in the result."""
        run = Run(self.llm)
        trace = run.trace
        trace.add_node(phase="start", turn=0, tags={"QUESTION": question})

        # Code context from the memory, retrieved once: its LLM call has its own node
        context = ""
        if self.memory is not None:
            context_node = trace.add_side_node(phase="context", turn=0)
            context = self.memory.retrieve(question, ask=lambda messages: run.ask(messages, node_id=context_node))
            trace.nodes[context_node].tags["context_length"] = len(context)

        # The conversation: the question, then each action of the agent and its result
        question_msg = {"role": "user", "content": question}
        history: list[dict] = []
        trace.set_questions(self._prompt(context, question_msg, history))

        plan: Optional[Plan] = None
        error = None
        try:
            while run.turn < self.max_turns and not (plan and plan.is_done):
                run.turn += 1
                messages = self._prompt(context, question_msg, history)
                plan = self.planner.create(run, messages) if plan is None else self.planner.update(run, messages, plan)
                if plan.is_done:
                    break
                step = {"role": "assistant", "content": plan.description()}
                history += self.executor.act(run, messages + [step])
        except AgentError as e:
            logger.warning("Run stopped: %s", e)
            error = str(e)

        done = error is None and plan is not None and plan.is_done
        plan = plan or Plan()
        cost = run.cost
        trace.add_node(phase="done", turn=run.turn, tags={"cost": cost})
        trace.set_response(
            raw_response="DONE" if done else (error or "too much interactions"),
            duration_s=0,
            token_usage=[run.usage.total_tokens, run.usage.prompt_tokens, run.usage.completion_tokens],
        )
        if error:
            trace.current_node().error = error
        return RunResult(plan=plan, done=done, turns=run.turn, usage=run.usage, cost=cost, trace=trace, error=error)

    def _prompt(self, context: str, question: dict, history: list[dict]) -> list[dict]:
        """System prompt, memory context, question and the last messages of the conversation."""
        messages = [system(self.core_prompt)]
        if context:
            messages.append(system(prompts.MEMORY_CONTEXT.format(context=context)))
        return messages + [question] + history[-self.history_size:]
