# Changelog

All notable changes to MIRA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Cleaned the baseline around `mira batch` for local `.pdb`, `.cif`, and `.mmcif` folder analysis.
- Batch plans now use `$PDB_ID` and `$PDB_PATH` placeholders and execute deterministically through `ToolRegistry`.
- Shared metric extraction and ranking criteria across agent, batch aggregation, and orchestrator code.
- Historical planning docs and experimental notes moved under `docs/archive/`.

### Fixed
- Rich CLI markup errors in help/interactive output.
- Tool schemas normalized to valid OpenAI function parameter JSON Schema.
- Offline tests now use local fixtures instead of requiring live RCSB downloads by default.

## [0.1.0] — 2024-04-08

### Added
- Initial release of MIRA (Molecular Intelligence and Reasoning Agent)
- Core agent with plan-then-act loop for protein structure analysis
- Tool registry with automatic tool discovery
- Interactive CLI (`mira`) with REPL mode
- Batch analysis mode for analyzing multiple PDB structures
- Subagent mode for parallel structure analysis
- Two-stage binder design workflow (target analysis + candidate evaluation)
- Rich terminal output with themes and formatted tables
- Comprehensive test suite

### Tools
- `compute_sasa` — Solvent Accessible Surface Area analysis
- `compute_interface` — Interface detection between chains
- `align_structures` — Structure alignment using ProDy
- `get_functional_annotations` — Functional site annotation via FDAP
- `analyze_bfactors` — B-factor analysis for flexibility
- `compute_charge_distribution` — Electrostatic charge analysis
- `get_conservation_scores` — Sequence conservation analysis
- `check_ramachandran` — Ramachandran plot analysis
- `search_structural_homologs` — Structural homolog search via FoldSeek
- `compute_normal_modes` — Normal mode analysis via ProDy
- `cross_correlations` — Cross-correlation analysis
- `fast_relax` — Energy minimization / fast relaxation
- `analyze_interface_energies` — Interface energy analysis
- `score_interface` — Rosetta-based interface scoring
- `get_secondary_structure` — DSSP secondary structure assignment
- `renumber_pdb` — PDB chain renumbering
- `structure_io` — Structure file I/O (PDB/CIF)

### Optional Dependencies
- `web` — FastAPI web server for serving MIRA as an API

[0.1.0]: https://github.com/joshgilligan17/MIRA/releases/tag/v0.1.0
