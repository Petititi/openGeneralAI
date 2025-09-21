import os
import time

from agents.orchestrator import Orchestrator
import configurator
import graphviz
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
    result, cost = orchestrator.process_user_message("Script `loader.py` doesn't start. Fix the mistake.")

    with open("templates/dbg.html", "w", encoding="utf-8") as f:
        f.write(orchestrator.last_trace.to_html())

    assert result is not None
    assert "plan_steps" in result
    # all steps should be "done":
    for step in result["plan_steps"]:
        assert step["status"] == "done"
    # the file should also be modified correctly:
    assert validation_lambda("python loader.py")[0] == True

import re

def parse_sed_expr(expr: str):
    r"""Parse l'expression à l'intérieur des quotes."""
    i = 0; n = len(expr)
    while i < n and expr[i].isspace(): i += 1
    addr = None
    m = re.match(r'(\d+)', expr[i:])
    if m:
        addr = int(m.group(1)); i += m.end(0)
    while i < n and expr[i].isspace(): i += 1
    if i >= n:
        raise ValueError("Empty sed expression")
    cmd = expr[i]; i += 1

    def parse_field(start_idx: int, delim: str):
        j = start_idx; out = []
        while j < n:
            ch = expr[j]
            if ch == '\\' and j+1 < n:
                out.append('\\'); out.append(expr[j+1]); j += 2
            elif ch == delim:
                return ''.join(out), j+1
            else:
                out.append(ch); j += 1
        raise ValueError("Delimiter not closed")

    if cmd == 's':
        if i >= n: raise ValueError("Missing delimiter for s")
        delim = expr[i]; i += 1
        pattern, i = parse_field(i, delim)
        replacement, i = parse_field(i, delim)
        flags = expr[i:].strip()
        return ('s', addr, pattern, replacement, flags)
    elif cmd in ('i', 'a'):
        if expr[i].isspace():
            i += 1
        text = expr[i:]
        if text.startswith('\\'):
            text = text[1:]
        return (cmd, addr, text)
    else:
        raise ValueError(f"Unsupported sed command '{cmd}'")

def _unescape_replacement(s: str) -> str:
    r"""Decode \n, \t, \xNN, \uNNNN... but preserve \1, \2 backreferences (as '\1')."""
    out = []; i = 0; n = len(s)
    while i < n:
        c = s[i]
        if c == '\\' and i+1 < n:
            nxt = s[i+1]
            if nxt == 'n': out.append('\n'); i += 2
            elif nxt == 't': out.append('\t'); i += 2
            elif nxt == 'r': out.append('\r'); i += 2
            elif nxt == '\\': out.append('\\'); i += 2
            elif nxt == 'f': out.append('\f'); i += 2
            elif nxt == 'v': out.append('\v'); i += 2
            elif nxt == 'a': out.append('\a'); i += 2
            elif nxt == 'b': out.append('\b'); i += 2
            elif nxt in '0123456789':
                out.append('\\' + nxt); i += 2  # backreference -> keep as \N
            elif nxt == 'x' and i+3 < n and all(ch in "0123456789abcdefABCDEF" for ch in s[i+2:i+4]):
                out.append(chr(int(s[i+2:i+4],16))); i += 4
            elif nxt == 'u' and i+5 < n and all(ch in "0123456789abcdefABCDEF" for ch in s[i+2:i+6]):
                out.append(chr(int(s[i+2:i+6],16))); i += 6
            else:
                out.append(nxt); i += 2
        else:
            out.append(c); i += 1
    return ''.join(out)

def _unescape_pattern(s: str) -> str:
    # Pour les patterns on applique le même décodage (utile pour \n dans pattern, etc.)
    return _unescape_replacement(s)

def apply_sed(cmd: str, fs, params: list):
    """Retourne (ok:bool, result). result = fs.write(...) ou message d'erreur."""
    mq = re.search(r"'(.*?)'", cmd, re.DOTALL)
    if not mq:
        return False, "Invalid sed pattern (no quotes)"
    expr = mq.group(1)
    try:
        parsed = parse_sed_expr(expr)
    except ValueError as e:
        return False, f"Invalid sed expression: {e}"

    content = fs.read(params[-1])
    lines = content.splitlines(keepends=True)

    if parsed[0] in ('i', 'a'):
        cmdc, addr, text = parsed
        if addr is None:
            return False, "Insert/append requires a line number address"
        insert_text = _unescape_replacement(text)
        if not insert_text.endswith('\n'):
            insert_text += '\n'
        idx = addr - 1
        if cmdc == 'i':
            if idx <= len(lines): lines.insert(idx, insert_text)
            else: lines.append(insert_text)
        else:
            if 0 <= idx < len(lines): lines.insert(idx+1, insert_text)
            else: lines.append(insert_text)
        return True, fs.write(params[-1], "".join(lines))

    # substitution
    _, addr, pattern_raw, repl_raw, flags = parsed
    pattern = _unescape_pattern(pattern_raw)
    repl = _unescape_replacement(repl_raw)

    count_per_line = 0 if ('g' in flags) else 1
    re_flags = 0
    if 'i' in flags or 'I' in flags:
        re_flags |= re.IGNORECASE

    if addr is not None:
        idx = addr - 1
        if 0 <= idx < len(lines):
            lines[idx] = re.sub(pattern, repl, lines[idx], count=count_per_line, flags=re_flags)
    else:
        # appliquer per-line (comme sed)
        lines = [re.sub(re.escape(pattern), repl, ln, count=count_per_line, flags=re_flags) for ln in lines]
    output = fs.write(params[-1], "".join(lines))
    return True, output


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
            return apply_sed(cmd, fs, params)
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
            if "-c" in params:
                return True, "" # no output for this simple unit test...

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
    result, cost = orchestrator.process_user_message("Le script `loader.py` ne se lance pas. Corrige l'erreur.")

    with open("templates/dbg_bash.html", "w", encoding="utf-8") as f:
        f.write(orchestrator.last_trace.to_html())

    assert result is not None
    assert "plan_steps" in result
    # all steps should be "done":
    for step in result["plan_steps"]:
        assert step["status"] == "done"
    # the file should also be modified correctly:
    assert validation_lambda("python loader.py")[0] == True
