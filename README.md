
# MIRA - Molecular Intelligence and Reasoning Agent

A tool-augmented reasoning agent for biomolecular structure analysis with a planning-first execution loop and registry-based tool system. The cleaned baseline focuses on reliable `mira batch` runs over local folders of `.pdb`, `.cif`, and `.mmcif` files for structure filtering, ranking, and comparative analysis.

## Features

### Planning-First Execution
MIRA analyzes your query, creates an analysis plan, validates it against the actual PDB chain structure, then executes - giving you visibility into what tools will be used before they run. Chain IDs are automatically detected and validated to prevent planning errors.

### Deterministic Tool Execution
Validated plan steps are executed through `ToolRegistry` in order. Adaptive replanning is reserved for validation or execution failures, so batch runs are reproducible and do not ask the model to re-decide tool calls after planning.

### 20+ Built-in Tools
- **Structure I/O**: `load_structure`, `renumber_pdb`
- **Interface Analysis**: `compute_interface`, `analyze_interface_energies`, `score_interface` (PyRosetta)
- **Surface Properties**: `compute_sasa`, `compute_charge_distribution`
- **Structural Quality**: `analyze_bfactors`, `check_ramachandran`, `get_secondary_structure`
- **Dynamics**: `compute_normal_modes`, `compute_cross_correlations`, `predict_hinge_regions`
- **Conservation**: `get_conservation_scores`, `get_functional_annotations`
- **Structural Search**: `search_structural_homologs`, `align_structures`

### Batch Analysis Mode
Analyze local folders of `.pdb`, `.cif`, and `.mmcif` files with parallel execution and joint comparative synthesis. Folder runs pass `pdb_path` to tools by default, so offline tests and private structure filtering do not require RCSB downloads. RCSB access is still available when you provide PDB IDs explicitly.

### Batch Dashboard
A local FastAPI + React dashboard supports upload-based triage jobs, ranked result tables, residue-level evidence, 3D structure inspection, and markdown report export. It uses deterministic analysis profiles for demo-reliable offline runs over uploaded PDB/CIF/mmCIF files.

### Flexible Display Modes
- **Verbose Mode** (default): Full step-by-step output showing all tool calls, arguments, and results
- **Normal Mode**: Compact single-line display with phase indicator and tool byline

### Lazy Tool Loading
Heavy dependencies (PyRosetta, ProDy) are only loaded when needed, ensuring fast startup.

## Installation

```bash
pip install -e .
```

**Requirements:**
- Python 3.11+
- MINIMAX_API_KEY or OPENAI_API_KEY environment variable (or use `--api-key` flag)
- Node.js 20+ for the optional dashboard frontend

## Usage

### Interactive Mode (Single Structure)

```bash
# Start interactive session

source venv/bin/activate
mira

# Or start a session with a prompt
mira "Analyze the TCR-pMHC interface in 1AO7"

# Plan only (see what would be done without executing)
mira --plan-only "Analyze 1UBQ"

# Skip planning and use direct ReAct (backwards compatible)
mira --no-plan "Compute SASA for 1UBQ chain A"
```

### Batch Mode Example Usage

```bash
# Analyze a local folder of PDB/CIF/mmCIF files with parallel execution
mira batch --folder ./binders "Analyze the interfaces and identify key binding residues"

# Rank by a specific metric
mira batch --folder ./antibodies --rank-by interface_energy "Analyze CDR loop positioning"

# Use specific PDB IDs from RCSB
mira batch --pdb-ids "1BRS,1BRC,1BRD" "Compare interface energetics"

# Sequential execution (for debugging)
mira batch --folder ./test_pdbs --sequential "Analyze structures"

# Custom glob pattern
mira batch --folder ./structures --glob "*.cif" "Analyze all CIF files"
```

### Batch Ranking Criteria

| Criterion | Description | Higher is Better |
|-----------|-------------|------------------|
| `stability` | Mean relative SASA percent | No |
| `buried_surface_area` | Total interface buried surface area (Å²) | Yes |
| `n_interface_residues` | Number of residues at interface | Yes |
| `n_buried` | Count of buried residues | Yes |
| `n_exposed` | Count of exposed residues | No |
| `mean_bfactor` | Mean B-factor of structure | No |
| `std_bfactor` | B-factor variability | No |
| `interface_energy` | PyRosetta interface energy | No |

Legacy web UI code, `binder-design`, and `batch-analyze` are currently experimental/archive paths and are hidden from the primary CLI help. The supported baseline is `mira`, `mira chat`, `mira batch`, and the local dashboard API below.

### Local Dashboard

```bash
# Backend API
pip install -e ".[web]"
uvicorn structagent.api.server:app --reload --port 8000

# Frontend
cd webapp
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. Uploaded jobs are stored under `.mira/jobs/`.

The historical `src/structagent/web/` UI remains archived/experimental. The supported dashboard entrypoint is `structagent.api.server:app` plus `webapp/`.

Project chat can route bounded tool calls for structure loading, target analysis, contact/interface checks, batch screening from project structures, and real design-model execution. Local Apple Silicon development should start with ProteinMPNN or LigandMPNN sequence design. GPU-only backbones and binder pipelines such as RFdiffusion and BindCraft are queued through the same design-run interface but should run on a CUDA worker.

```bash
# Local sequence-design adapters.
MIRA_MODEL_DIR=/data/mira/models
MIRA_PROTEINMPNN_REPO=/path/to/ProteinMPNN
MIRA_PROTEINMPNN_PYTHON=/path/to/proteinmpnn-env/bin/python
MIRA_LIGANDMPNN_REPO=/path/to/LigandMPNN
MIRA_LIGANDMPNN_PYTHON=/path/to/ligandmpnn-env/bin/python

# GPU design adapters.
MIRA_RFDIFFUSION_REPO=/path/to/RFdiffusion
MIRA_RFDIFFUSION_CONTIGS='[A1-100/0 80-120]'
MIRA_BINDCRAFT_REPO=/path/to/BindCraft
MIRA_BINDCRAFT_SETTINGS=/path/to/settings.json

# Admin-supplied fallback command template for custom backends.
MIRA_DESIGN_COMMAND='custom-design --target {target_path} --out {output_dir}'
MIRA_DESIGN_TIMEOUT_SECONDS=3600
```

Supported custom-command placeholders are `{project_id}`, `{run_id}`, `{target_path}`, `{output_dir}`, `{chain_id}`, `{num_designs}`, and `{prompt}`. If a real backend is not configured, chat creates a `configuration_required` design run instead of pretending generation happened.

On the Docker deployment, install the CPU ProteinMPNN backend with:

```bash
cd /opt/mira
scripts/install_proteinmpnn_cpu.sh
docker compose up -d
```

### DigitalOcean Deployment

The supported hosted prototype path is a Dockerized FastAPI + React app on a DigitalOcean Droplet, with MiniMax synthesis for now and Cloudflare Workers AI planned as the next provider.

```bash
cp .env.example .env
docker compose up -d --build
```

See `docs/deploy-digitalocean.md` for the full Droplet setup, auth, DNS, and update flow.

### Display Modes

```bash
# Verbose mode (default) - full step-by-step output
mira --display verbose "Analyze 1UBQ"

# Normal mode - compact single-line display
mira --display normal "Analyze 1UBQ"
```

## Architecture

MIRA employs a **planning-first ReAct loop**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Planning Phase                               │
│  1. Extract PDB ID from query                                        │
│  2. Auto-detect chain composition via load_structure                 │
│  3. Generate plan with correct chain IDs (validated against structure) │
│  4. Revise plan if validation fails                                 │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         Validation Phase                             │
│  • Check tool availability and dependencies                          │
│  • Validate chain IDs against actual structure                       │
│  • Validate parameter names and types                               │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         Execution Phase                              │
│  • Execute tools via ToolRegistry in validated order                 │
│  • Replan only if validation/execution fails                         │
│  • Extract rankable metrics from tool outputs                       │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         Synthesis Phase                              │
│  • LLM synthesizes final answer from results                         │
│  • In batch mode: comparative ranking + joint analysis              │
└─────────────────────────────────────────────────────────────────────┘
```

### Batch Analysis Flow

```
User Query + PDB Folder or PDB IDs
         ↓
┌──────────────────────────────────────────────────────────────┐
│                    BatchRunner                                 │
│  • Discover .pdb/.cif/.mmcif files via glob pattern           │
│  • Create one batch plan with $PDB_ID/$PDB_PATH placeholders  │
│  • Execute validated plan per structure in parallel           │
│  • Collect StructureResult with extracted metrics              │
└──────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────────┐
│               ResultAggregator                                │
│  • Extract metrics from ToolResult.raw dicts                  │
│  • Rank structures by specified criterion                     │
└──────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────────┐
│            BatchSynthesisEngine                              │
│  • Build comparative analysis prompt                          │
│  • LLM generates ranking + narrative explanation              │
└──────────────────────────────────────────────────────────────┘
```

### ToolRegistry

A singleton registry manages tools. Tools self-register at import time via the `@tool` decorator with lazy loading support for heavy dependencies.

## Project Structure

```
structagent/
├── pyproject.toml              # Package configuration
├── README.md                   # This file
├── src/structagent/
│   ├── __init__.py            # Public API exports
│   ├── cli.py                 # CLI with chat and batch commands
│   ├── agent.py               # MiraAgent with planning-first ReAct
│   ├── registry.py            # ToolRegistry with lazy loading
│   ├── prompts.py            # System prompts (planning, synthesis)
│   ├── display.py            # Display strategies (normal, verbose)
│   ├── metrics.py            # Shared metric extraction/ranking criteria
│   ├── tool_metadata.py      # Tool schemas (extracted via AST)
│   ├── batch.py             # BatchRunner, ResultAggregator, Synthesis
│   └── tools/
│       ├── structure_io.py    # load_structure, renumber_pdb
│       ├── contacts.py        # get_residue_contacts
│       ├── sasa.py           # compute_sasa
│       ├── secondary_structure.py  # get_secondary_structure
│       ├── interface.py      # compute_interface
│       ├── alignment.py      # align_structures
│       ├── annotations.py    # get_functional_annotations
│       ├── bfactor.py       # analyze_bfactors
│       ├── charge.py         # compute_charge_distribution
│       ├── conservation.py    # get_conservation_scores
│       ├── ramachandran.py   # check_ramachandran
│       ├── foldseek.py       # search_structural_homologs
│       ├── dynamics.py       # compute_normal_modes, cross_correlations
│       ├── relaxation.py      # fast_relax
│       ├── interface_energy.py # analyze_interface_energies
│       ├── pyrosetta_interface.py # score_interface
└── tests/
    ├── test_batch.py         # Batch analysis tests
    └── data/                 # Small intentional test fixtures
```

## Batch Analysis Deep Dive

### How Metrics Are Extracted

MIRA automatically extracts rankable metrics from tool outputs stored in `ToolResult.raw`:

| Tool | Extracted Metrics |
|------|-------------------|
| `compute_sasa` | `mean_relative_sasa_percent`, `n_buried`, `n_partial`, `n_exposed` |
| `compute_interface` | `buried_surface_area`, `n_interface_residues` |
| `analyze_bfactors` | `mean_bfactor`, `std_bfactor` |
| `compute_charge_distribution` | `total_charge`, `charge_cluster_count` |

### Chain Auto-Detection

Before executing a plan, MIRA:
1. Resolves either a local `pdb_path` or an explicit RCSB `pdb_id`
2. Calls `load_structure` to get actual chain composition where available
3. Includes chain IDs and batch placeholders in the planning prompt
4. Validates generated plans against registered tool schemas and known chains

This prevents errors like `compute_interface(chain_a="A", chain_b="D")` when the structure has chains "E" and "I".

### Parallel Execution

Batch mode uses `ThreadPoolExecutor` to run multiple structure analyses in parallel:
- Default: 4 workers
- Configurable via `--max-workers`
- Each structure executes the same validated batch plan with per-structure placeholders resolved

### Comparative Synthesis

After ranking, MIRA's LLM generates a comparative analysis:
- Explains why each structure ranks where it does
- Identifies key structural differences driving the ranking
- Highlights notable residues or anomalies
- Provides biological insights

## Key Dependencies

- **gemmi**: Structure parsing (PDB/mmCIF)
- **freesasa**: SASA computation
- **openai**: API client (MiniMax/M2.7 default)
- **biopython**: Biological file formats
- **scipy/numpy**: Computational utilities
- **rich/click**: CLI interface
- **pyfiglet**: ASCII art banners
- **prody**: Normal mode analysis (optional)
- **pyrosetta**: Interface scoring (optional)

## Configuration

### Environment Variables

```bash
export MINIMAX_API_KEY=your_key_here  # Default API
export OPENAI_API_KEY=your_key_here  # Alternative
export MIRA_JOB_ROOT=.mira/jobs
export MIRA_REPORT_PROVIDER=minimax
export MIRA_REPORT_API_KEY=your_key_here
export MIRA_BASIC_AUTH_USERNAME=mira
export MIRA_BASIC_AUTH_PASSWORD=use-a-long-random-password
```

### CLI Options

**Chat Mode:**
```bash
mira --model MiniMax-M2.7 \
     --base-url https://api.minimax.io/v1 \
     --max-steps 15 \
     --toolsets structure,analysis \
     --display normal \
     --save-trajectories \
     --trajectory-dir ./runs
```

**Batch Mode:**
```bash
mira batch --folder ./pdbs \
           --rank-by buried_surface_area \
           --max-workers 4 \
           --max-steps 10 \
           --save-trajectories
```

## Examples

### Single Structure Analysis

```
You > Analyze the interface residues in 1BRC

MIRA >
🤔 Planning...
Detected structure info: Structure 1BRC has 2 chain(s): Chain E: 345 residues; Chain I: 75 residues.

✓ Planning complete
Plan (4 steps):
  1. compute_interface — Identify interface between chains E and I
  2. compute_sasa — Calculate surface exposure
  3. analyze_bfactors — Identify flexible regions
  4. get_residue_contacts — Map specific contacts

Tool: compute_interface(...)
  → Interface: 1318.7 Å² buried surface area, 35+19 interface residues
```

### Batch Analysis with Ranking

```
$ mira batch --folder ./tcr_pmhc --rank-by buried_surface_area "Compare TCR-pMHC interfaces"

MIRA Batch Analysis
Analyzing 4 structures...
Ranking by: buried_surface_area

✓ Planning complete
✓ Planning complete
✓ Planning complete
✓ Planning complete

Generating comparative analysis...

╭────────────────────────────────── Results ───────────────────────────────────╮
│ Ranking by buried_surface_area:                                              │
│   1. 1AO7: 1842.3                                                         │
│   2. 1BRS: 1654.7                                                         │
│   3. 1BRC: 1318.7                                                         │
│   4. 2HHI: 987.2                                                          │
╰──────────────────────────────────────────────────────────────────────────────╯

Comparative Analysis Summary

The buried surface area analysis reveals 1AO7 has the most extensive interface
(1842.3 Å²), correlating with its higher-affinity TCR-pMHC interaction.
1BRS follows closely at 1654.7 Å², while 1BRC and 2HHI show progressively
smaller interfaces...

4 structures analyzed | 98.6s | 236,615 tokens
```

## License

MIT
