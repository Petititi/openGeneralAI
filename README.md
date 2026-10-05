# openGeneralAI — Towards Production-Ready LLM Reasoning Agents

This repository provides a clear, open-source, and pragmatic foundation for building advanced reasoning agents across multiple model providers (OpenAI, Anthropic, Mistral, OpenHands, etc.). Our goal is to deliver simple yet robust examples, proven best practices, and reusable components—including planning, tool execution, memory management, and evaluation—to help teams transition from proof-of-concept (POC) to production-grade agents.

## Blog series and tags

This repository is the companion code of the blog series [Building LLM Coding Agents](https://charlie-soft.com/llm-agent/). Each article has a git tag with the code it describes:

| Article | Tag |
| --- | --- |
| 01-a Initialize the project | `01-initialize-project` |
| 01-b Plug a real LLM | `01-b-real-LLM` |
| 02-a Reasoning tools | `02-a-reasoning-tools` |
| 02-b Debugging tools | `02-b-debugging-tools` |
| 02-c Hardening and restructuring | `02-c-hardening` |

```bash
git checkout -b my-branch tags/02-a-reasoning-tools
```

`master` is ahead of the published articles (long-term memory, Souvenir and Gmail import in `apps/souvenir/`).

> **Security note**: from commit `f6c2664` (September 2025) until the fix of October 2026, `app.py` served every file of the project folder over HTTP, including `.env` and its API keys. If you ran one of these versions on a machine reachable from a network, revoke your API keys. The tags above are not affected.

## What is in the repository

- `src/opengeneralai/`: the agent, an installable package
  - `agent/`: the plan → act → check loop (`Orchestrator.run(question)`)
  - `llm/`: access to the LLMs through LiteLLM (`LLMClient`)
  - `tools/`: tool interface and the memory search tool
  - `tracing/`: trace of each run, as a tree saved in JSON
  - `memory/`: long-term memory of code (tree-sitter, SQLite FTS5, FAISS)
  - `web/`: Flask app (chat page, configuration page, trace viewer)
- `apps/souvenir/`: personal memory assistant and Gmail import (work in progress)
- `viewer/` and `traces/`: trace viewer and example traces
- `docs/ARCHITECTURE.md`: how the pieces fit together

## Quick Start

```bash
pip install -r requirements.txt    # everything; "pip install -e ." for the core only
python app.py                      # http://localhost:12000/
```

- Home: http://localhost:12000/ (redirects to the configuration page until an API key is set)
- Config: http://localhost:12000/config
- Trace viewer: http://localhost:12000/viewer/?trace=/traces/examples/2025-09-10_file_editing_tools.json (see [viewer/README.md](viewer/README.md))

From Python:

```python
from opengeneralai.agent.orchestrator import Orchestrator
from opengeneralai.llm.client import LiteLLMClient
from opengeneralai.tools.registry import ToolRegistry

result = Orchestrator(LiteLLMClient("mistral/mistral-small-latest"), ToolRegistry()).run("Explain what a Python decorator is")
print(result.done, [step.descr for step in result.plan.steps], result.cost)
result.trace.save("traces/my_run.json")
```

Environment variables:

- `HOST` (default `127.0.0.1`): the server only listens on localhost. Use `HOST=0.0.0.0` only on a trusted network: anyone who can reach the server can spend your API credits and change your API keys.
- `ALLOWED_ORIGINS` (default: none): comma-separated origins allowed to call the API from another front-end (CORS).
- `LITELLM_DEBUG=1`: print the full LiteLLM requests (prompts included) for debugging.

## Tests

```bash
pip install -e ".[dev]"
ruff check .
pytest              # deterministic tests: the agent loop runs against a scripted LLM (test/conftest.py)
pytest --run-llm    # also the scenarios that call the configured LLM for real (API key, costs)
```

The scenario tests write their traces in `traces/`; open them with the trace viewer.

## Roadmap Highlights

    Integrate Major Providers: OpenAI, Anthropic, Mistral, and others.

    Agent Loop: Planning → tool execution → verification.

    Memory Systems: Short/long-term memory + RAG-enhanced context.

    Evaluation Framework: Task benchmarks, metrics, canary testing, and CI integration.

    Security: Sandboxed tooling + safe-by-default policies.

    Observability: Trace visualization, run analytics, and debugging tools.

    Deployment: Dockerization and reproducible environments.

## Why Are Reasoning Agents Challenging to Build?

While basic chatbots are straightforward, creating agents capable of complex reasoning, tool interaction, self-correction, and task execution presents significant hurdles. Key practical challenges include:

- Task Decomposition & Planning
    Breaking goals into logical substeps, selecting actionable plans, and dynamically adjusting strategies.

- Tool Orchestration
    Choosing appropriate tools, formatting inputs/outputs (schemas), interpreting results, and chaining multi-tool workflows.

- Uncertainty & Error Recovery
    Detecting mistakes, estimating confidence levels, retrying steps, and implementing fallback heuristics.

- Long-Context & State Management
    Handling context windows, recalling historical data, and persisting state (working/long-term memory).

- Non-Determinism vs. Robustness
    Balancing creativity with reproducibility; ensuring traceable, repeatable executions.

- Hallucinations & Grounding
    Mitigating unsubstantiated claims by anchoring decisions in reliable sources and tool outputs.

- Safety & Compliance
    Preventing data exfiltration, prompt injections, and unauthorized actions; sandboxing code/tool execution.

- Observability & Debugging
    Tracing every step (prompts, outputs, tool calls) and diagnosing failure root causes.

- Evaluation & Metrics
    Defining task-specific benchmarks, business-aligned metrics, and regression tests.

- Cost/Latency Optimization
    Implementing caching, fallback policies, intelligent scaling, and spend governance.

- Model Heterogeneity
    Bridging differences in capabilities (function calling vs. tool use), context limits, and provider-specific behaviors.

- Prompt Engineering
    Structuring prompts (scratchpads, chain-of-thought, constraints); versioning and testing iterations.

- Multi-Agent Coordination
    Avoiding deadlocks, synchronizing specialized roles, and sharing reliable global states.

## Design Principles

- Modular & Provider-Agnostic
    Unified adapters for seamless integration with diverse model APIs.

- Strict Tool Contracts
    Typed schemas, input/output validation, and explicit error handling.

- Reasoning-First Execution
    Plan-then-act strategies (scratchpads, self-reflection) with built-in verification.

- Observability by Default
    Comprehensive logging (prompts, tools, traces), session IDs, and metadata for analytics.

- Reproducibility
    Explicit configurations, seed management, and state snapshots for replayable runs.

- Continuous Evaluation
    Task suites, regression testing, performance scorecards, and regression alerts.

## Contributing

We welcome contributions: bug fixes, model/tool integrations, examples, documentation, and evaluation suites.

## Acknowledgments

We thank the vibrant AI community and model providers driving agent innovation. This project aims to consolidate practical knowledge for building reliable, safe, and impactful reasoning agents in real-world applications.
