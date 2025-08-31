import os
from pathlib import Path
import time
import pytest

from agents.orchestrator import Orchestrator
import configurator
import utils
from agents.tools.ToolRegistry import ToolRegistry

def test_file_editing():

    # 1) FS initial:
    fs = utils.InMemoryFS({"loader.py":"print('time:' + time.time())"})

    # 2) final lambda validation that try to interpret python modifications:
    def validation_lambda(cmd):
        program, params = cmd.split(" ")[0], cmd.split(" ")[1:]
        if "python" in program:
            # look for loader.py in params:
            if "loader.py" in params:
                try:
                    content = fs.read("loader.py")
                    if "import time" in content and "str(" in content:
                        return True, f"time:{time.time()}"
                    if "import time" in content:
                        return False, "TypeError: can only concatenate str (not 'float') to str"
                    return False, "NameError: name 'time' is not defined"
                except FileNotFoundError as e:
                    return False, f"File not found: {e}"
        return False, "Invalid command"

    # 3) Tools creations:
    tools_registry = ToolRegistry()
    tools_registry.register(utils.ReadFile(fs))
    tools_registry.register(utils.EditFile(fs))
    tools_registry.register(utils.RunProg(fs, validation_lambda))

    # 4) start the test:

    # --- Various global config:
    CONFIG_PATH = os.getcwd() + "/config.json"
    ENV_PATH = os.getcwd() + "/.env"

    # --- Configuration creation
    cfg = configurator.AppConfig(CONFIG_PATH, ENV_PATH)

    orchestrator = Orchestrator(cfg, tools_registry)
    result = orchestrator.process_user_message("Le script `loader.py` ne se lance pas. Corrige l'erreur.")
    assert result is not None
    assert "plan_steps" in result
    # all steps should be "done":
    for step in result["plan_steps"]:
        assert step["status"] == "done"
    # the file should also be modified correctly:
    assert validation_lambda("python loader.py")[0] == True


def test_file_editing_only_bash():

    # 1) FS initial:
    fs = utils.InMemoryFS({"loader.py":"print('time:' + time.time())"})

    # 2) final lambda validation that try to interpret python modifications:
    def validation_lambda(cmd):
        program, params = cmd.split(" ")[0], cmd.split(" ")[1:]
        if "cat" in program:
            try:
                content = fs.read(params[0])
                return True, content
            except FileNotFoundError as e:
                return False, f"{params[0]}: {str(e)}"
        if program.startswith("echo") and " > " in cmd:
            escape_char = cmd[5]
            if escape_char == "'" or escape_char == '"':
                new_content = cmd[6:cmd.rfind(escape_char)]
            else:
                new_content = cmd[5:cmd.rfind(" > ")].strip()
            return True, fs.write(cmd[cmd.rfind(" > ") + 3:], new_content)
        if "sed" in program:
            try:
                import re
                m = re.search(r"'(?:(\d+)s|s)/([^/]*)/([^/]*)/'", cmd)
                if m:
                    line_number = int(m.group(1)) if m.group(1) else None
                    pattern     = m.group(2)
                    replacement = m.group(3)
                    replacement = replacement.encode('utf-8').decode('unicode_escape')
                    content = fs.read(params[-1])
                    if line_number:
                        # Remplacement uniquement sur la ligne donnée
                        lines = content.splitlines(keepends=True)
                        idx = int(line_number) - 1
                        if 0 <= idx < len(lines):
                            lines[idx] = re.sub(re.escape(pattern) if pattern != "^" else pattern, replacement, lines[idx])
                        content = "".join(lines)
                    else:
                        # Remplacement global
                        content = re.sub(re.escape(pattern) if pattern != "^" else pattern, replacement, content)
                    return True, fs.write(params[-1], content)
                else:
                    return False, f"Invalid sed pattern"
            except FileNotFoundError as e:
                return False, f"{cmd}: {str(e)}"
        if "python" in program:
            # look for loader.py in params:
            if "loader.py" in params:
                try:
                    content = fs.read("loader.py")
                    if "import time" in content and "str(" in content:
                        return True, f"time:{time.time()}"
                    if "import time" in content:
                        return False, "python: TypeError: can only concatenate str (not 'float') to str"
                    return False, "python: NameError: name 'time' is not defined"
                except FileNotFoundError as e:
                    return False, f"python: File not found: {e}"
        return False, "Invalid command"

    # 3) Tools creations:
    tools_registry = ToolRegistry()
    tools_registry.register(utils.RunProg(fs, validation_lambda))

    # 4) start the test:

    # --- Various global config:
    CONFIG_PATH = os.getcwd() + "/config.json"
    ENV_PATH = os.getcwd() + "/.env"

    # --- Configuration creation
    cfg = configurator.AppConfig(CONFIG_PATH, ENV_PATH)

    orchestrator = Orchestrator(cfg, tools_registry)
    result = orchestrator.process_user_message("Le script `loader.py` ne se lance pas. Corrige l'erreur.")
    assert result is not None
    assert "plan_steps" in result
    # all steps should be "done":
    for step in result["plan_steps"]:
        assert step["status"] == "done"
    # the file should also be modified correctly:
    assert validation_lambda("python loader.py")[0] == True
