# openGeneralAI - AI Coding Agent Instructions

## Project Overview

A **LLM reasoning agent** (plan → act → check) supporting the providers of LiteLLM, with a
long-term memory of code and a trace of every run. It is the companion code of the blog series
*Building LLM Coding Agents*: each article has a git tag (see `README.md`). Read
`docs/ARCHITECTURE.md` first: layout, dependency rule, steps of a run, errors.

## Core Architecture

`src/opengeneralai/` is an installable package (`pip install -e ".[dev]"`).

### Agent loop (`agent/`)
`Orchestrator.run(question) -> RunResult` (plan, done, turns, usage, cost, trace, error):

1. `Planner` (`planner.py`) creates the plan, then after each action asks for a decision
   (`NEXT_TASK`, `DONE`, `NEW_PLAN`) and writes a new plan from the existing one when needed.
2. `Executor` (`executor.py`) asks for **one** tool call per turn and runs it.
3. Everything that belongs to one question lives in a `Run` (`run.py`): trace, usage, turn.
   **The orchestrator keeps no state between questions**: never store per-question data on it.

The plan is typed (`plan.py`: `Plan`, `Step`, `StepStatus`, `PlanDecision`). Parse LLM answers
with `parse_plan`, `parse_decision`, `parse_action`, or `json_parsing.parse_llm_json`: they
raise `PlanParseError` / `ToolCallError`, never return half-parsed dicts.

### Conversation
The prompt of each LLM call is: core system prompt, memory context (if any), the question, the
last `history_size` messages (each action as an `assistant` message, followed by its result as
a `user` message), then the system prompt of the step. Keep this role discipline.

### Prompts (`agent/prompts.py`)
Every prompt of the loop is there, as a `str.format()` template (literal braces doubled).
Do not write prompts inline in other modules. A tool owns the prompts of its own LLM calls.

### LLM access (`llm/client.py`)
The agent only knows `LLMClient` (`complete(messages) -> LLMResponse`, `cost(usage)`).
`LiteLLMClient` holds the provider quirks (e.g. `mistral/devstral-small-250x` ignores system
messages: `merge_system_messages`) and maps provider errors to `LLMError` subclasses. Test new
providers against these quirks here, not in the agent.

## Long-Term Memory (`memory/`)

- `database.py` (SQLite + FTS5): documents, chunks with tree-sitter metadata, imports.
  Stores paths, not copies of the files.
- `embeddings.py` (FAISS + SentenceTransformer, optional extra `embeddings`): normalized vectors,
  class embedding = weighted average of its methods (`docs/memory/`). Call `persist()` after
  bulk adds.
- `longterm_memory.py`: `LongTermMemory`, hybrid search (keyword + semantic), `add_folder()`.
  CLI: `python -m opengeneralai.memory.longterm_memory search "query" --mode hybrid`.
- `agent/context.py`: `MemoryContext.retrieve(question, ask)` builds the code context added to
  the prompts; its LLM call is traced in a `context` node.

## Tool System (`tools/registry.py`)

```python
class Tool(ABC):
    name: str
    signature: str          # shown to the LLM: must match the run() arguments
    needs_llm: bool = False # True: run() also receives ask_llm(messages) -> str
    def run(self, **kwargs) -> ToolResult: ...  # ToolResult(ok, content, meta={"error": ...})
```

Register tools in a `ToolRegistry` given to the `Orchestrator`. Tools must not import `agent/`.
Examples: `test/utils.py` (`ReadFile`, `EditFile`, `RunProg`), `tools/memory_search.py`.

## Configuration

`config.py` (`AppConfig`): `config.json` (provider, model, user_lang, memory paths) and the API
keys in `.env` (`{PROVIDER}_API_KEY`). A missing `config.json` is created with defaults; an
unknown provider or model is replaced, the other settings are kept. Only the web app and
`apps/` use it: the agent receives plain values.

## Web app (`web/app.py`)

`create_app(root, cfg, make_orchestrator)` builds the Flask app; nothing happens at import
time. Errors meant for the user (`errors.py`) are returned as JSON (502 for LLM errors), other
exceptions as a generic 500. Only `viewer/` and `traces/*.json` are served from the repository:
never serve the project root (it holds `.env`). No inline script or style in the pages (CSP).

## Tracing (`tracing/logger.py`)

`TrajectoryLogger`: one node per step (`start`, `context`, `plan/create`, `plan/update`,
`plan/recover`, `act/run`, `act/llm`, `done`) with the messages sent, the raw answer, the plan,
the tool call, tokens and duration. `result.trace.save("traces/x.json")`, then open
http://localhost:12000/viewer/?trace=/traces/x.json (`viewer/README.md`).

## Development Workflows

```bash
pip install -e ".[dev]"
ruff check .
pytest              # no network, no API key: FakeLLM in test/conftest.py
pytest --run-llm    # also the tests marked llm (real LLM, API key in .env)
python app.py       # http://localhost:12000/
```

Test pattern (`test/test_agent_loop.py`):
```python
llm = fake_llm([plan_json, action_json, '{"status": "DONE"}'])  # one reply per LLM call
result = Orchestrator(llm, tools).run("Task...")
result.trace.save(TRACES_DIR / "my_scenario.json")
assert result.done and len(llm.calls) == 3
```
Scenario tools never execute the code written by the LLM (`simulate_loader` reads it with `ast`).

## Critical Conventions

1. **No state on the orchestrator**: per-question data goes in `Run`.
2. **Typed errors**: raise `errors.py` classes for what the user can act upon; do not swallow
   exceptions silently (log them), do not return error strings instead of raising.
3. **User language**: `user_lang` goes into the core prompt; JSON keys are never translated.
4. **Dependencies**: `web -> agent -> llm, tools, tracing, memory -> errors, json_parsing`.
5. **Tests**: every bug fix comes with a test that fails without it.

## Common Pitfalls

1. **Tool signature mismatches**: the LLM sees `signature`, it must match `run(**kwargs)`.
2. **Prompt braces**: templates go through `.format()`; double the literal braces.
3. **Normalized embeddings**: use `normalize=True` in `encode()` for cosine similarity.
4. **Tree-sitter language codes**: use the keys of `SUPPORTED_CODE_EXT` (`"c_sharp"`, not `"csharp"`).
5. **apps/souvenir/** is work in progress with known bugs (see its README); `gmail_memory_assistant.py`
   is excluded from the lint.
