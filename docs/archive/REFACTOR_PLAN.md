# MIRA Public Release Refactor Plan

**Project:** MIRA (Molecular Intelligence and Reasoning Agent)
**Goal:** Clean up codebase for public open-source release
**Created:** 2026-04-08

---

## Phase 0: Pre-Flight Audit

Before making any changes, document the current state.

- [ ] Run full test suite and record which tests pass/fail
- [ ] Verify all imports work (`python -c "import structagent"`)
- [ ] Count files, lines of code, test coverage estimate
- [ ] Check for any hardcoded secrets, API keys, or debug artifacts
- [ ] Verify CLI entry point works (`mira --help`)

---

## Phase 1: Repository Hygiene

### 1.1 Add LICENSE file
- MIT license (matches README mention)
- Add `license = {text = "MIT"}` to `pyproject.toml`

### 1.2 Clean .gitignore
- Already solid — covers `.env`, `.venv`, `__pycache__`, etc.
- Consider adding: `*.pdb` (debug builds), `.ruff_cache/`, `.mypy_cache/`

### 1.3 pyproject.toml polish
Missing fields for PyPI:
- [ ] `license = "MIT"`
- [ ] `authors = [{name = "...", email = "..."}]`
- [ ] `urls = {Repository = "...", Homepage = "..."}`
- [ ] `readme = "README.md"`
- [ ] `keywords = ["protein", "structure", "biology", "agent", "AI"]`
- [ ] `classifiers` (Python version, license, intended audience)

### 1.4 Remove experimental/ship items
- [ ] Review `src/structagent/tools/ToDo/` — does it ship? Should it?
  - If experimental: move to repo root `ToDo/` or delete
  - Update README architecture diagram if removed
- [ ] Check `src/structagent/web/` — is the web server ready for shipping or experimental?
  - `pyproject.toml` already has `web` as optional dependency — good
  - Review web server code for debug prints, hardcoded URLs

---

## Phase 2: Code Quality

### 2.1 Remove debug artifacts
- [ ] `results.md` swap file (`.results.md.swp`) — is this in git?
- [ ] Any `print()` statements left in production code?
- [ ] `test_binders/` and `test_binders_small/` — are these test fixtures or ad-hoc runs?
    - If ad-hoc: add to `.gitignore` or delete
    - If fixtures: move to `tests/data/`

### 2.2 TODO/FIXME audit
- [ ] Review all TODO/FIXME/HACK comments — are any resolved?
- [ ] `prompts.py` has XXX markers (lines 320, 596) — what are these?
- [ ] `binder_design.py`, `subagent.py` flagged by grep — review and address

### 2.3 Naming consistency
- [ ] Package name: `structagent` (internal) vs `MIRA` (product name)
    - README already uses MIRA throughout — consistent
    - CLI entry point `mira` is good
    - `pyproject.toml` name is `mira` — good

### 2.4 Imports and dependencies
- [ ] Verify all imports resolve in fresh venv
- [ ] Check for transitive dependencies not listed in `pyproject.toml`
- [ ] Optional deps: `prody`, `pyrosetta` — are they actually lazy-loaded?
    - Review `tools/dynamics.py`, `tools/pyrosetta_interface.py`

---

## Phase 3: Documentation

### 3.1 README polish
- [ ] Architecture diagram shows `ToDo/` — update if removing
- [ ] Add badges: CI status, PyPI version, Python version
- [ ] Add `CONTRIBUTING.md` link
- [ ] Add `LICENSE` link

### 3.2 Add CONTRIBUTING.md
- [ ] How to run tests
- [ ] How to add a new tool
- [ ] Development setup (venv, dependencies)
- [ ] Style guide (ruff? black?)

### 3.3 Add .github/ for CI/CD
- [ ] `.github/workflows/test.yml` — run tests on push/PR
- [ ] `.github/workflows/lint.yml` — ruff/mypy checks

### 3.4 Changelog
- [ ] `CHANGELOG.md` — even a stub for v0.1.0

---

## Phase 4: Test Coverage

- [ ] Check which tests actually run vs collect-only
- [ ] Identify untested core modules: `registry.py`, `agent.py`, `batch.py`
- [ ] Add tests for registry tool discovery
- [ ] Add tests for batch runner with synthetic data
- [ ] Verify tests are parametrized and deterministic

---

## Phase 5: OSS Checklist

### Required for public release
- [x] README.md — exists, comprehensive
- [ ] LICENSE file — **MISSING**
- [ ] `pyproject.toml` fully populated — partial
- [ ] Clean git history — review `.git` for large files/blobs
- [ ] No secrets in history — check `git log`, `.git/config`
- [ ] CONTRIBUTING.md — **MISSING**
- [ ] `.github/workflows/` — **MISSING**

### Nice to have
- [ ] `CHANGELOG.md`
- [ ] `SECURITY.md` (if any security-sensitive features)
- [ ] Demo GIF or screenshot
- [ ] `docs/` folder with API reference

---

## Estimated Timeline

| Phase | Effort | Risk |
|-------|--------|------|
| Phase 0: Audit | 15 min | Low |
| Phase 1: Repo Hygiene | 30 min | Low |
| Phase 2: Code Quality | 45 min | Medium |
| Phase 3: Documentation | 30 min | Low |
| Phase 4: Test Coverage | 45 min | Medium |
| Phase 5: OSS Checklist | 15 min | Low |

**Total: ~3 hours** (most of it low-risk cleanup)

---

## Parallelization Strategy

**Track A (Repo Hygiene):** LICENSE, pyproject.toml, .gitignore — independent, fast
**Track B (Code Audit):** TODO/FIXME, debug artifacts, import audit — independent, moderate
**Track C (Documentation):** CONTRIBUTING.md, workflows, CHANGELOG — independent, fast

Track A and C can run in parallel subagents. Track B needs more judgment — do sequentially.

---

## Post-Refactor Steps

1. Tag v0.1.0
2. Create GitHub repo (if not already)
3. Push and verify CI passes
4. Create PyPI account and publish test package (`twine check`)
5. Announce

---

## Open Questions

1. **ToDo/ directory** — is this experimental code that should be removed, or is it part of the story?
2. **test_binders/** — are these fixtures or ad-hoc runs? Can they be regenerated?
3. **GitHub repo** — does one exist yet, or are we starting from local only?
4. **Author name/email** — what should appear in `pyproject.toml` authors?
