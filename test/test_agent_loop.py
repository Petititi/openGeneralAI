"""
Deterministic test of the plan -> act -> check loop, with a scripted LLM (see FakeLLM in conftest.py).

Same scenario as test_simple_scenario.py::test_file_editing, without network nor API key:
the trace it writes in traces/ can be opened with the viewer (see viewer/README.md).
"""

import json
import time
from pathlib import Path

from unittest.mock import Mock

import utils
from opengeneralai.agent.context import MemoryContext
from opengeneralai.agent.orchestrator import Orchestrator
from opengeneralai.tools.registry import ToolRegistry

TRACES_DIR = Path(__file__).parent.parent / "traces"

BROKEN_LOADER = "print('time:' + time.time())"
FIXED_LOADER = "import time\nprint('time:' + str(time.time()))"


def make_tools(fs):
    def run_python(cmd):
        if cmd.split() != ["python", "loader.py"]:
            return False, "Invalid command"
        content = fs.read("loader.py")
        if "import time" in content and "str(" in content:
            return True, f"time:{time.time()}"
        if "import time" in content:
            return False, "TypeError: can only concatenate str (not 'float') to str"
        return False, "NameError: name 'time' is not defined"

    tools = ToolRegistry()
    tools.register(utils.ReadFile(fs))
    tools.register(utils.EditFile(fs))
    tools.register(utils.RunProg(fs, run_python))
    return tools


def test_plan_act_check_loop_fixes_the_script(fake_llm):
    fs = utils.InMemoryFS({"loader.py": BROKEN_LOADER})
    llm = fake_llm([
        # turn 1: create the plan, then act on step 1
        json.dumps({"plan_steps": [
            {"descr": "Read loader.py", "status": "todo", "next_step_criteria": "content known"},
            {"descr": "Import time and convert the float to str", "status": "todo", "next_step_criteria": "file edited"},
            {"descr": "Run loader.py", "status": "todo", "next_step_criteria": "prints the time"},
        ]}),
        json.dumps({"tool_name": "read_file", "arguments": {"filepath": "loader.py"}}),
        # turn 2: step 1 is done, act on step 2
        json.dumps({"status": "NEXT_TASK"}),
        json.dumps({"tool_name": "edit_file", "arguments": {"filepath": "loader.py", "full_content": FIXED_LOADER}}),
        # turn 3: step 2 is done, act on step 3
        json.dumps({"status": "NEXT_TASK"}),
        json.dumps({"tool_name": "bash", "arguments": {"cmd": "python loader.py"}}),
        # turn 4: the last step succeeded
        json.dumps({"status": "DONE"}),
    ])
    orchestrator = Orchestrator(llm, make_tools(fs))

    result = orchestrator.run("Script `loader.py` doesn't start. Fix the mistake.")

    trace_file = result.trace.save(TRACES_DIR / "fake_file_editing.json")

    assert fs.read("loader.py") == FIXED_LOADER
    assert result.done
    assert [step["status"] for step in result.plan["plan_steps"]] == ["done", "done", "done"]
    assert len(llm.calls) == 7, "one LLM call per plan update and per action"
    assert result.turns == 4
    assert result.usage.total_tokens > 0 and result.cost > 0

    trace = json.loads(trace_file.read_text())
    phases = [node["phase"] for node in sorted(trace["nodes"].values(), key=lambda n: n["timestamp"])]
    assert phases == ["start", "plan/create", "act/run", "plan/update", "act/run",
                      "plan/update", "act/run", "plan/update", "done"]
    tools_used = [n["tool_name"] for n in sorted(trace["nodes"].values(), key=lambda n: n["timestamp"]) if n["tool_name"]]
    assert tools_used == ["read_file", "edit_file", "bash"]


def test_memory_context_is_retrieved_and_traced(fake_llm):
    """The query analysis of the memory goes through the LLM and gets its own trace node."""
    ltm = Mock()
    ltm.search.return_value = [{
        "chunk_id": 1, "content": "class Calculator:\n    def add(self, a, b):\n        return a + b",
        "chunk_type": "class", "chunk_name": "Calculator", "source_path": "calc.py",
        "start_line": 1, "end_line": 3, "language": "python",
    }]
    analysis = json.dumps({"should_search": True, "search_queries": ["calculator add"],
                           "search_type": "keyword", "symbols": []})
    llm = fake_llm([analysis, json.dumps({"plan_steps": [{"descr": "Explain add()", "status": "done"}]})])

    result = Orchestrator(llm, ToolRegistry(), memory=MemoryContext(ltm)).run("How does Calculator add?")

    context_nodes = [n for n in result.trace.nodes.values() if n.phase == "context"]
    assert len(context_nodes) == 1
    assert context_nodes[0].raw_response == analysis
    planning_prompt = llm.calls[1]
    assert "class Calculator" in planning_prompt[1]["content"], "the context follows the core prompt"
    assert result.done and result.turns == 1


def test_orchestrator_keeps_no_state_between_questions(fake_llm):
    """Two questions on the same orchestrator: separate traces and usage."""
    plan_done = json.dumps({"plan_steps": [{"descr": "Answer", "status": "done"}]})
    llm = fake_llm([plan_done, plan_done])
    orchestrator = Orchestrator(llm, ToolRegistry())

    first = orchestrator.run("Question one")
    second = orchestrator.run("Question two")

    assert first.trace is not second.trace
    root = second.trace.nodes[second.trace.root_id]
    assert root.phase == "start" and root.parent_id is None
    assert root.tags["QUESTION"] == "Question two"
    assert first.usage == second.usage, "same-size questions: each result counts its own call only"
    assert not any(n.phase == "context" for n in second.trace.nodes.values()), "no memory, no context node"
