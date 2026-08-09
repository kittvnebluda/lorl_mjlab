.PHONY: sync
sync:
	uv sync --all-packages --extra cu130 --group dev

.PHONY: sync-cpu
sync-cpu:
	uv sync --all-packages --extra cpu --group dev

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: type
type:
	uv run ty check
	uv run pyright

.PHONY: stubs
stubs:
	uv run --no-sync pybind11-stubgen mujoco -o typings --ignore-all-errors

.PHONY: test
test:
	uv run pytest

.PHONY: check
check: format type
