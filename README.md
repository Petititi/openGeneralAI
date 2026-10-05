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

```bash
git checkout -b my-branch tags/02-a-reasoning-tools
```

`master` is ahead of the published articles (long-term memory, Souvenir, Gmail import).

> **Security note**: from commit `f6c2664` (September 2025) until the fix of October 2026, `app.py` served every file of the project folder over HTTP, including `.env` and its API keys. If you ran one of these versions on a machine reachable from a network, revoke your API keys. The tags above are not affected.

## Current Implementation (Minimal Example)

A lightweight Flask server with Bootstrap UI:

- Homepage: Submit queries and view dynamic responses.

- Configuration: Select verified providers/models via dropdown.

- API Endpoints: /ask (query handling), /api/config (persists settings to config.json).

Note: This intentionally minimal example lacks full agent loops or real tools. It serves as a foundation for integrating models and tooled execution.

## Quick Start (Flask Example)

Install: pip install -r requirements.txt

Launch: PORT=12000 python app.py

Access:

- Home: http://localhost:12000/

- Config: http://localhost:12000/config

- Trace viewer (written by the tests): http://localhost:12000/templates/dbg.html

Environment variables:

- `HOST` (default `127.0.0.1`): the server only listens on localhost. Use `HOST=0.0.0.0` only on a trusted network: anyone who can reach the server can spend your API credits and change your API keys.
- `ALLOWED_ORIGINS` (default: none): comma-separated origins allowed to call the API from another front-end (CORS).
- `LITELLM_DEBUG=1`: print the full LiteLLM requests (prompts included) for debugging.

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
