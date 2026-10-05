# Trace viewer

Interactive view of the traces written by `TrajectoryLogger` (`agents/logger.py`): one column per
step of the agent loop (plan, action, ...), with the messages sent to the LLM, its answer, the plan
and the tool calls of each step.

The viewer is a static page: it reads the trace from the JSON file given in `?trace=`.

| Page | Use |
| --- | --- |
| `index.html` | full viewer: filters, node details, layouts, JSON export |
| `embed.html` | compact view for iframes (used by the blog articles) |

## Open a trace

Traces are written in `traces/` by the scenario tests (`traces/examples/` holds a few real runs):

```python
orchestrator.last_trace.save("traces/my_run.json")
```

With the web app running (`python app.py`):

- <http://localhost:12000/viewer/?trace=/traces/fake_file_editing.json>
- <http://localhost:12000/viewer/?trace=/traces/examples/2025-09-10_file_editing_tools.json>

Without the app, any static server at the root of the repository works:

```bash
python -m http.server 8000
# http://localhost:8000/viewer/?trace=../traces/examples/2025-09-10_file_editing_tools.json
```

Opening `index.html` directly from the disk (`file://`) does not work: browsers block ES modules
and `fetch` there.

## Notes

- Only traces from the same site are loaded (`js/load.js`): a trace holds prompts and tool outputs.
- Cytoscape, dagre and the JSON viewer come from jsDelivr, pinned to a version with
  Subresource Integrity hashes. To upgrade one, update the URL and its `integrity` hash
  (`openssl dgst -sha384 -binary file.js | openssl base64 -A`).
- The blog copies this folder and `traces/examples/` with its `scripts/sync-viewer.sh`.
