# Contributing to MIRA

Thank you for your interest in contributing to MIRA (Molecular Intelligence and Reasoning Agent).

## Development Setup

1. **Clone the private repository**
   ```bash
   git clone <private-repo-url>
   cd structagent
   ```

2. **Create a virtual environment** (Python 3.11+ required)
   ```bash
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   ```

3. **Install the package in editable mode with dev dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks** (optional but recommended)
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Running Tests

```bash
# Run the offline default suite
venv/bin/python -m pytest -q

# Run a specific test file
venv/bin/python -m pytest tests/test_registry.py -v

# Optional tests that may need network or licensed/heavy dependencies
venv/bin/python -m pytest -q -m network
venv/bin/python -m pytest -q -m pyrosetta
venv/bin/python -m pytest -q -m prody
```

## Code Style

- This project uses **Ruff** for linting and formatting
- Run Ruff checks before committing:
  ```bash
  venv/bin/python -m ruff check src tests
  venv/bin/python -m ruff format --check src tests
  ```

## Adding a New Tool

1. Create a new file in `src/structagent/tools/` (e.g., `my_tool.py`)
2. Implement your tool function with proper type annotations
3. Decorate it with `@tool` and provide a valid OpenAI-compatible JSON Schema object for parameters
4. Add the module to `TOOL_MODULES` in `src/structagent/cli.py` if the CLI should load it
5. Regenerate planning metadata:
   ```bash
   venv/bin/python scripts/extract_tool_metadata.py
   ```
6. Add focused tests in `tests/`, using local fixtures under `tests/data/` whenever possible
7. Update `README.md` if the user-facing batch baseline changes

## Project Structure

- `src/structagent/` — main package source
- `src/structagent/tools/` — individual tool implementations
- `src/structagent/agent.py` — core agent logic
- `src/structagent/batch.py` — batch discovery, ranking, and synthesis
- `src/structagent/metrics.py` — shared metric extraction and ranking criteria
- `src/structagent/cli.py` — command-line interface
- `tests/data/` — small intentional local fixtures
- `docs/archive/` — historical planning docs and experimental notes

## Submitting Changes

1. Create a feature branch (`git checkout -b codex/my-feature`)
3. Make your changes and run tests
4. Commit with a clear message
5. Push and open a pull request in the private repo

## Reporting Issues

Please report bugs and feature requests via GitHub Issues. Include:
- Python version
- MIRA version (`mira --version`)
- Steps to reproduce
- Expected vs actual behavior
