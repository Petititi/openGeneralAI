# openGeneralAI - AI Coding Agent Instructions

## Project Overview

This is a **production-ready LLM reasoning agent framework** supporting multiple providers (OpenAI, Anthropic, Mistral, etc.) with a focus on structured agent loops, long-term memory, and observability. The system uses a **reasoning-then-action** architecture with trajectory logging and tree-based decision tracking.

## Core Architecture

### Agent Loop (Orchestrator Pattern)
The system follows a **plan → act → verify** cycle coordinated by `Orchestrator`:

1. **ReasoningAgent** creates/updates plans as JSON with structured steps (`descr`, `status`, `next_step_criteria`)
2. **ActionAgent** executes tool calls strictly one-per-turn, outputting only JSON
3. **TrajectoryLogger** maintains a tree of execution nodes with full prompt/response history
4. Loop continues max 10 turns or until all plan steps are `"done"`

**Key file**: `agents/orchestrator.py` - Study `process_user_message()` for the complete flow.

### Message Scratchpad Pattern
Messages accumulate as a "scratchpad" through `logger.get_current_interaction()`:
- System prompts define agent behavior (reasoning rules, tool schemas)
- User messages provide context and task descriptions  
- Assistant responses contain plans or tool calls
- Tool results feed back as user messages

**Critical**: `clean_message_history()` filters/limits context to prevent overflow. Always maintain message role discipline.

### Strict JSON Communication
All agent responses MUST be valid JSON5:
- **ReasoningAgent** outputs: `{"plan_steps": [{"descr": "...", "status": "done|todo", "next_step_criteria": "..."}]}`
- **ActionAgent** outputs: `{"tool_name": "...", "arguments": {...}}`
- No mixed text+JSON. Use `clean_raw_response()` to strip markdown code blocks.
- Parsing failures trigger recovery via `improve_plan=True`

### Provider-Specific Quirks
**Mistral devstral-small-250X models ignore system messages** - see `safe_ask()` in `orchestrator.py`:
```python
# Workaround: merge system messages into first user message
if self.cfg.model.startswith("mistral/devstral-small-250"):
    # Concatenate all system content to first user message
```
Always test new providers against this pattern.

## Long-Term Memory System

**File**: `storage/longterm_memory.py` - Three-layer architecture:

### 1. DatabaseManager (SQLite + FTS5)
- Stores documents, chunks (with tree-sitter parsed metadata), imports
- Full-text search via `chunks_fts` virtual table
- Metadata queries: `get_class_code()`, `get_methods_by_class()`, `list_all_classes()`
- **No file copying** - stores only paths; verify integrity with `check_file_integrity()`

### 2. EmbeddingManager (FAISS + SentenceTransformer)
- Uses `mixedbread-ai/mxbai-embed-large-v1` by default
- L2-normalized vectors for cosine similarity via `IndexFlatIP`
- **Class embedding strategy**: Weighted average of method embeddings (see `WEIGHTED_CLASS_EMBEDDING_DOC.md`)
- Always call `persist()` after bulk adds

### 3. LongTermMemory (Orchestrator)
- **Hybrid search** (α=0.5): Min-max normalized keyword + semantic fusion
- Tree-sitter parsing extracts: functions, classes, methods, imports for 15+ languages
- CLI available: `python storage/longterm_memory.py search "query" --mode hybrid`

**When indexing new files**: Use `add_folder()` for robustness - skips unparseable files with warnings.

## Tool System

**Registry pattern** in `agents/tools/ToolRegistry.py`:
```python
class Tool(ABC):
    name: str
    signature: str  # Shown to LLM in prompt
    def run(self, **kwargs) -> ToolResult
```

**ToolResult contract**:
- `ok: bool` - Success flag (checked in `execute_action()`)
- `content: str` - Main output returned to agent
- `meta: Dict[str, Any]` - Logged metadata (params, errors)

**Creating new tools**:
1. Inherit from `Tool`, define `name` and `signature` (function-like syntax for LLM)
2. Implement `run()` returning `ToolResult(ok=True, content="...", meta={...})`
3. Register in orchestrator: `tools_registry.register(YourTool())`

See `test/utils.py` for minimal examples: `ReadFile`, `EditFile`, `RunProg`.

## Configuration & Environment

**Two-file system**: `config.json` + `.env`

- `AppConfig` (configurator.py) validates provider/model against LiteLLM's registry
- API keys stored in `.env` as `{PROVIDER}_API_KEY` (e.g., `MISTRAL_API_KEY`)
- Provider detection: `llm_interactions.get_providers_and_models()` queries LiteLLM

**Validation flow**:
1. Check provider in `litellm.models_by_provider`
2. Verify model in provider's model list
3. Ensure API key exists via `get_env_name_for(provider)`
4. `need_configuration` property gates UI redirects

**Never hardcode API keys** - use `save_api_key()` to write to `.env` atomically.

## Trajectory Logging & Debugging

**Tree-based execution traces** via `TrajectoryLogger`:

- Each `add_node()` creates a decision point with: `phase`, `turn`, `prompt_messages`, `raw_response`, `plan_snapshot`, `tool_name/input/output`
- Context variable `_current_node_id` maintains tree traversal state
- Export to HTML: `orchestrator.last_trace.to_html()` → interactive Cytoscape graph

**Debug workflow**:
1. Run orchestrator, save trace: `orchestrator.last_trace`
2. Write HTML: `open("debug.html", "w").write(trace.to_html())`
3. Inspect tree in browser - nodes show full prompts, plans, tool calls
4. Template: `templates/template_logger.html` (Cytoscape + custom rendering)

**Test pattern** (see `test/test_simple_scenario.py`):
```python
orchestrator = Orchestrator(cfg, tools_registry)
result, cost = orchestrator.process_user_message("Task...")
with open("templates/dbg.html", "w") as f:
    f.write(orchestrator.last_trace.to_html())
assert all(step["status"] == "done" for step in result["plan_steps"])
```

## Development Workflows

### Running the Flask UI
```bash
PORT=12000 python app.py
# Visit http://localhost:12000/ for chat UI
# Visit http://localhost:12000/config for provider selection
```

### Testing with Custom Tools
Create in-memory filesystem and validation lambdas (see `test/test_simple_scenario.py`):
```python
fs = InMemoryFS({"file.py": "content"})
tools = ToolRegistry()
tools.register(ReadFile(fs))
orchestrator = Orchestrator(cfg, tools)
```

### Indexing Codebase for Memory
```bash
cd storage
python longterm_memory.py index ../agents
python longterm_memory.py search "tool execution" --mode hybrid
python longterm_memory.py list-classes  # See all indexed classes
```

### Running Tests
```bash
python -m pytest test/test_simple_scenario.py -v
```

## Critical Conventions

### 1. Turn Budget Enforcement
**Max 6 turns per task** - hardcoded in reasoning agent prompts. Plans must include validation as final step.

### 2. User Language Handling
- Primary output language: `cfg.user_lang` (default: "English")
- JSON keys never translated - only values shown to users
- System prompts specify: `"Always reply in {self.cfg.user_lang}"`

### 3. Token Usage Tracking
Orchestrator accumulates: `self.token_count`, `self.in_token`, `self.out_token`
- Per-response tracked in `safe_ask()` via `llm_response.usage`
- Cost calculated: `litellm.cost_per_token(model, prompt_tokens, completion_tokens)`

### 4. Error Recovery
- **ReasoningAgent**: If plan update fails, retry with `improve_plan=True` (full re-plan)
- **ActionAgent**: Tool failures set `error` field, feed back as user message
- LiteLLM exceptions: Wrap in descriptive strings (`"Rate limit exceeded: try again later"`)

### 5. Message Role Discipline
- **system**: Instructions, tool schemas, plan templates
- **user**: Tasks, tool results, context
- **assistant**: Plans, tool calls (JSON only)

Never mix roles - breaks model context understanding.

## File Organization

```
agents/
  orchestrator.py      # Main loop coordinator
  reasoning.py         # Plan creation/update logic
  action.py            # Tool execution wrapper
  logger.py            # Trajectory tree + HTML export
  tools/ToolRegistry.py  # Tool registration

storage/
  longterm_memory.py   # Main API (add, search, get)
  DatabaseManagement.py  # SQLite + FTS5 ops
  EmbeddingManagement.py # FAISS + embeddings

templates/
  index.html           # Chat UI
  config.html          # Provider selection
  template_logger.html # Debug visualization

test/
  test_simple_scenario.py  # Integration tests
  utils.py             # Mock tools (ReadFile, EditFile, RunProg)
```

## Common Pitfalls

1. **Forgetting to normalize embeddings** - Always use `normalize=True` in `encode()` for cosine similarity
2. **Not persisting FAISS index** - Call `embeddings.persist()` after adds
3. **Tool signature mismatches** - LLM sees `signature`, must match `run(**kwargs)` params exactly
4. **Ignoring turn limits** - Plans that don't finish in 6 turns fail silently (no FINAL emitted)
5. **Tree-sitter language codes** - Use exact keys from `SUPPORTED_CODE_EXT` (e.g., `"c_sharp"` not `"csharp"`)

## When Adding Features

1. **New provider**: Add to LiteLLM models, test system message handling in `safe_ask()`
2. **New tool**: Define schema in `signature`, implement `run()`, add to registry in `app.py`
3. **New language**: Add to `SUPPORTED_CODE_EXT`, ensure tree-sitter-languages supports it
4. **New search mode**: Extend `search()` hybrid fusion logic in `longterm_memory.py`

## References

- **LiteLLM docs**: https://docs.litellm.ai/ (unified API for providers)
- **Tree-sitter**: https://tree-sitter.github.io/ (code parsing)
- **FAISS**: https://github.com/facebookresearch/faiss (vector search)
- **Design principles**: See `README.md` sections on "Why Are Reasoning Agents Challenging" and "Design Principles"

## Quick Start for New Contributors

1. Set API key: `echo MISTRAL_API_KEY=your_key > .env`
2. Run server: `python app.py`
3. Test agent: Visit UI, submit "Calculate 5 + 3" (requires Calculator tool)
4. Inspect trace: Check `templates/dbg.html` after test run
5. Read code flow: `app.py` → `orchestrator.py` → `reasoning.py` + `action.py`
