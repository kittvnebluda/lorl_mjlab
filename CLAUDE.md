# CLAUDE.md

## Development Workflow

**Always use `uv run`, not python**.

```sh

# 1. Make changes.

# 2. Type check.
uv run ty check  # Fast
uv run pyright  # More thorough, but slower

# 3. Run tests.
uv run pytest tests/  # Single suite
uv run pytest tests/<test_file>.py  # Specific file

# 4. Format and lint before committing.
uv run ruff format
uv run ruff check --fix
```

Common commands are bundled into a Makefile for convenience.

```sh
make format     # Format and lint
make type       # Type-check
make check      # make format && make type
make test       # Run the full test suite
```

Always run `make check` before committing. This runs formatting, linting,
and type checking. Do not commit code that fails type checking.

## Commits

- Line length limit is 120 columns. This applies to code, comments, and docstrings.
- Avoid local imports unless they are strictly necessary (e.g. circular imports).
- Tests should follow these principles:
  - Use functions and fixtures; do not use test classes.
  - Favor targeted, efficient tests over exhaustive edge-case coverage.
  - Prefer running individual tests rather than the full test suite to improve iteration speed.

Follow **Conventional Commits** (`type(scope): subject`).

- **Types:** `feat`, `fix`, `chore`, `docs`, `refactor`
- **Scopes:** Package/area name (e.g., `hardware`, `nav2_params`, `bringup/launch`)

**Rules:**

1. **Atomic commits:** One logical change per commit; build/pass tests before committing.
2. **Isolate changes:** Separate mechanical edits (format/rename) and `docs`/`chore` from behavioral `feat`/`fix`.
3. **Selective staging:** Always stage specific files (`git add <file>` or `-p`); never `git add -A`.
4. **On-demand only:** Only commit when explicitly asked; never bundle extra unrequested edits.
