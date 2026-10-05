"""
Deterministic test of the plan -> act -> check loop, with a scripted LLM (see FakeLLM in conftest.py).

Same scenario as test_simple_scenario.py::test_file_editing, without network nor API key:
the trace it writes in traces/ can be opened with the viewer (see viewer/README.md).
"""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import utils
from agents.orchestrator import Orchestrator
from agents.tools.ToolRegistry import ToolRegistry

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
    cfg = SimpleNamespace(model="mistral/mistral-small-latest", user_lang="English")
    orchestrator = Orchestrator(cfg, make_tools(fs))

    plan, cost = orchestrator.process_user_message("Script `loader.py` doesn't start. Fix the mistake.")

    trace_file = orchestrator.last_trace.save(TRACES_DIR / "fake_file_editing.json")

    assert fs.read("loader.py") == FIXED_LOADER
    assert [step["status"] for step in plan["plan_steps"]] == ["done", "done", "done"]
    assert len(llm.calls) == 7, "one LLM call per plan update and per action"
    assert cost > 0

    trace = json.loads(trace_file.read_text())
    phases = [node["phase"] for node in sorted(trace["nodes"].values(), key=lambda n: n["timestamp"])]
    assert phases == ["start", "plan/create", "act/run", "plan/update", "act/run",
                      "plan/update", "act/run", "plan/update", "done"]
    tools_used = [n["tool_name"] for n in sorted(trace["nodes"].values(), key=lambda n: n["timestamp"]) if n["tool_name"]]
    assert tools_used == ["read_file", "edit_file", "bash"]
