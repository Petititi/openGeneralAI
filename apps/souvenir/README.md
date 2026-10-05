# Souvenir and Gmail import (work in progress)

A personal memory assistant built on the long-term memory of `opengeneralai`:

- `souvenir_assistant.py`: add, search and list "souvenirs" (notes, facts), interactive mode.
- `gmail_memory_assistant.py`: fetch Gmail threads (OAuth) and turn them into souvenirs.

```bash
pip install -e ".[embeddings,souvenir]"
python apps/souvenir/souvenir_assistant.py interactive
```

These scripts handle personal data (OAuth token, e-mails): keep `gmail_token.pickle`,
`mails.pickle` and the `*.sqlite` files out of git (see `.gitignore`).

## Known problems

The code comes from the `gmail` branch (March 2026) and is not finished:

- `LongTermMemory.insert_souvenir()` and `DatabaseManager.insert_souvenir()` call
  `insert_document()` with wrong arguments (`rel_path`, missing `description`), and
  `insert_souvenir()` calls `add_memory_chunk()` without its line arguments: every call raises
  `TypeError`.
- `gmail_memory_assistant.py` uses 14 names that are not defined, between lines 900
  and 1050 (left from a refactoring): it is excluded from the lint (`pyproject.toml`).
- The `memory` branch has one more commit (`9e1a928`, semantic and hybrid search of the
  souvenirs) that conflicts with this version.
- Whether Souvenir stays in this repository or gets its own is still to decide.
