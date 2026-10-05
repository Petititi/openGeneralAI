# Architecture

## Layout

```
src/opengeneralai/
  agent/         the plan -> act -> check loop
    orchestrator.py   Orchestrator.run(question) -> RunResult
    run.py            Run: state of one question (trace, token usage, turn)
    planner.py        create, update and recover the plan
    executor.py       one tool call per turn
    plan.py           Plan, Step, StepStatus, PlanDecision, parsing
    prompts.py        every prompt of the loop
    context.py        MemoryContext: code context from the long-term memory
  llm/
    client.py         LLMClient interface, LiteLLMClient (provider quirks, error mapping, cost)
    catalog.py        providers and models known to LiteLLM
  tools/
    registry.py       Tool, ToolResult, ToolRegistry
    memory_search.py  SearchContext tool (searches the long-term memory)
  tracing/
    logger.py         TrajectoryLogger: tree of the steps of a run, saved as JSON
  memory/             long-term memory of code: tree-sitter parsing, SQLite FTS5, FAISS
  web/                Flask app (create_app), templates/, static/
  config.py           AppConfig: config.json and the API keys in .env
  errors.py           errors meant for the user
  json_parsing.py     tolerant parsing of the JSON answers of the LLM
apps/souvenir/        personal memory assistant and Gmail import (work in progress)
viewer/               trace viewer (static page)
traces/               traces written by the tests; examples/ is versioned
test/
```

## Dependencies between modules

```
web -> agent -> llm, tools, tracing, memory -> errors, json_parsing
web -> config
```

The agent never imports `web` or `config`: it receives an `LLMClient`, a `ToolRegistry`,
an optional `MemoryContext` and plain values (`user_lang`, `max_turns`). Tools do not import
`agent`: a tool that needs the LLM declares `needs_llm = True` and receives `ask_llm` from the
executor.

## A run

`Orchestrator.run(question)` creates a `Run`, then:

1. **start** node: the question.
2. **context** node (beside start), when a memory is given: the LLM decides whether and how
   to search the code; the result is added to every prompt after the core system prompt.
3. Up to `max_turns` turns:
   - **plan/create** (first turn) or **plan/update**: the LLM writes the plan, or decides
     `NEXT_TASK`, `DONE` or `NEW_PLAN` for the current step;
   - **plan/recover** after `NEW_PLAN` or an unusable answer: a new plan written from the
     existing one, whose failed steps are marked `error`;
   - **act/run**: the LLM picks one tool call, the executor runs it. A tool that calls the
     LLM itself gets **act/llm** nodes under the action.
4. **done** node: total tokens, cost, and the error if the agent could not follow its protocol.

Prompts are built from the core system prompt, the memory context, the question and the last
`history_size` messages of the conversation: each action as the LLM wrote it (assistant
message) followed by its result (user message). An answer that cannot be parsed is retried
once with the error; then the run stops and `RunResult.error` says why.

The orchestrator keeps no state between questions: one instance can serve concurrent
requests. The web app builds one per request anyway, since the model can change through
`/api/config`.

## Errors

`errors.py` holds the errors whose message is meant for the user:

| Error | Raised by | Web answer |
| --- | --- | --- |
| `LLMError` and subclasses | `LiteLLMClient` | 502 with the message |
| `PlanParseError`, `ToolCallError` (`AgentError`) | planner, executor | in `RunResult.error` |
| anything else | bugs | 500, details in the server log only |

## Tests

- `pytest` runs without network or API key: `FakeLLM` (`test/conftest.py`) is an `LLMClient`
  that returns scripted replies and records the prompts.
- `pytest --run-llm` also runs the tests marked `llm`, which call the configured LLM.
- The scenario tools never execute the code the LLM writes: `simulate_loader`
  (`test/utils.py`) reads it with `ast`.

## Next steps

- Native tool calling (`tools=` of LiteLLM), tools described with JSON Schema and
  validated arguments, results sent with the `tool` role.
- JSONL logs with the run id; budgets in time and tokens, not only in turns.
- Memory: split `LongTermMemory` (indexing, metadata store, vector store behind an
  interface), make `memory/llm_extractor.py` use `LLMClient`, replace the `print` calls by
  logging, stop indexing the whole repository when the web app starts.
- Configuration: read-only settings separated from the API key storage; use LiteLLM's own
  environment variable names (`litellm.validate_environment`).
- Souvenir and Gmail import (`apps/souvenir/`): decide where they live, then fix them (see
  their README).
