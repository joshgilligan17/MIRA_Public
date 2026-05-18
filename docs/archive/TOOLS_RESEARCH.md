# MIRA Tool Expansion: In Silico Protein Structure Screening

**Goal:** Expand MIRA's toolset to support high-throughput virtual screening for protein engineering and therapeutic protein design.
**Date:** 2026-04-08

---

## Executive Summary

MIRA currently has ~17 tools covering structure I/O, interface analysis, surface properties, structural quality, dynamics, conservation, and structural search. For in silico protein structure screening, the following gaps are most significant:

| Priority | Gap | Value |
|----------|-----|-------|
| P0 | Thermostability / ΔΔG prediction | Core to protein engineering |
| P0 | Aggregation propensity prediction | Therapeutic protein liabilities |
| P1 | DMS data analysis | Interpret variant effects |
| P1 | Immunogenicity / HLA binding | Therapeutic protein safety |
| P2 | Protein-protein docking | Structural mechanism |
| P2 | Binding affinity estimation | Lead optimization |
| P2 | Structural quality validation | Model confidence |

---

## P0: Must-Have Tools

### 1. Thermostability / ΔΔG Prediction

#### Option A: `predict_thermostability` (ThermoNet-based)
- **What:** Neural network-based prediction of ΔΔG from protein structure. ThermoNet uses 3D convolutional neural networks on protein structures.
- **How:** Local inference using a pretrained model (available via BioPython integration or standalone)
- **Complexity:** Medium
- **Implementation:** Load structure with gemmi → compute features → run ThermoNet inference → return ΔΔG
- **Why valuable:** Core to every protein engineering campaign. Directly tells you which mutations stabilize/destabilize.

#### Option B: `predict_thermostability_ddgun` (DDGun)
- **What:** Sequence-based ddG prediction. Fast, no structure needed (but can use structure).
- **How:** Python package `ddgun` available via pip
- **Complexity:** Low
- **Dependencies:** `ddgun` pip package
- **Why valuable:** Fast enough for screening hundreds of variants.

#### Option C: `predict_thermostability_rosetta` (Rosetta ΔΔG)
- **What:** High-accuracy physics-based ΔΔG using Rosetta `ddg` protocol.
- **How:** PyRosetta already in MIRA (via `pyrosetta_interface`). Already partially implemented in `ToDo/fastrelaxer.py`.
- **Complexity:** Medium (PyRosetta already available)
- **Why valuable:** Gold standard accuracy, but slow. Use for validation of hits, not screening.

#### Recommendation: Implement all three with different complexity/accuracy tradeoffs.
- `predict_thermostability_fast` → DDGun (P0, easy)
- `predict_thermostability` → ThermoNet (P0, medium)
- `predict_thermostability_accurate` → Rosetta ddG (P1, use only on small sets)

---

### 2. Aggregation Propensity Prediction

#### `predict_aggregation` (TANGO + Aggrescan3D)
- **What:** Identifies aggregation-prone regions (APR) in protein sequences/structures. Critical for therapeutic proteins (aggregation → immunogenicity, loss of function).
- **How:**
  - **TANGO:** REST API at `services.nki.uu.nl/tango` (free, no API key) OR standalone command-line
  - **Aggrescan3D:** Web server at `biotools.uem.mz/anderson`
  - **Aggregation Toolbox:** Python library available
- **Complexity:** Low-Medium
- **Why valuable:** Therapeutic proteins must be screened for aggregation liability. Required for any IND filing.

#### `predict_aggregation_hotspots`
- **What:** Returns specific residues predicted to be in aggregation-prone regions, with scores.
- **Output:** List of residue ranges + confidence scores
- **Integration:** Can chain with `compute_sasa` to identify solvent-exposed APRs (more concerning for therapeutics)

---

## P1: High-Value Tools

### 3. Deep Mutational Scanning (DMS) Analysis

#### `query_dms_data` (ProteinGym / Canary)
- **What:** Pulls DMS fitness data for a protein to see which positions tolerate mutation and which don't.
- **How:**
  - **ProteinGym:** Large DMS benchmark (DeMiguel et al., 2023). Data available as downloadable TSV files. Subset via AWS Open Data or direct download.
  - **Canary:** Python library for DMS data access and analysis (from the Havranek lab).
  - **API:** No public REST API, but data files are downloadable.
- **Complexity:** Low-Medium (data access pattern)
- **Why valuable:** Lets MIRA say "position X is known to be highly sensitive to mutations from DMS data" — very powerful for engineering context.

#### `predict_variant_effect_from_dms` (ESM-2 / ProteinMAE)
- **What:** Use protein language model embeddings to predict variant effects, trained on DMS data.
- **How:** ESM-2 embeddings + regression head. Can approximate DMS fitness scores.
- **Complexity:** Medium-High
- **Why valuable:** Extend experimental DMS data to variants not yet tested.

---

### 4. Immunogenicity / HLA Binding Prediction

#### `predict_mhc_binding` (NetMHC / MHCflurry)
- **What:** Predicts MHC peptide binding affinity — key for assessing immunogenicity risk of therapeutic proteins.
- **How:**
  - **NetMHC:** Standalone + Python bindings (requires license for full version, free for academic from legepidemia.org)
  - **MHCflurry:** Open-source Python library (mhcflurry-py)
  - **HLAthena:** REST API + Python client
- **Complexity:** Medium
- **Dependencies:** `mhcflurry` or `netmhc` (MHCflurry is free, easier to install)
- **Why valuable:** Therapeutic proteins must be screened for T-cell epitope liability. Directly addresses a key safety concern.

#### `predict_immunogenicity_risk`
- **What:** High-level summary: identifies potential HLA-binding peptide hotspots in a therapeutic protein sequence.
- **Agent-facing:** Accepts protein sequence or PDB chain, returns risk regions with MHC IC50 scores.

---

## P2: Supporting Tools

### 5. Protein-Protein Docking (for mechanism)

#### `dock_structures` (HADDOCK / ClusPro API)
- **What:** Predicts how two proteins interact in 3D. Useful for understanding mechanism of binding.
- **How:**
  - **HADDOCK:** REST API at `haddock.science.uu.nl` (free tier available)
  - **ClusPro:** Web server at `cluspro.bu.edu` (has API access)
  - **RosettaDock:** Local via PyRosetta (already available in MIRA)
- **Complexity:** Medium (API integration)
- **Why valuable:** When screening mutants, docking lets MIRA explain *why* a mutation affects binding — structural mechanism.

#### `dock_peptide` (HADDOCK)
- **What:** Peptide-protein docking for neoepitope prediction.
- **Specialty:** Different from protein-protein — shorter peptides need specialized protocols.

---

### 6. Binding Affinity Estimation

#### `predict_binding_affinity` (PRODIGY)
- **What:** Predicts binding affinity (Kd/Ki) from a PDB structure of a complex.
- **How:** PRODIGY web server at `projects.vu.nl/prodigy` + Python API
- **Complexity:** Low-Medium
- **Why valuable:** For comparing mutant vs. wild-type binding, ΔΔG of binding can be compared.

#### `score_interface_quality`
- **What:** Uses geometry and interaction fingerprints to score interface quality beyond just buried surface area.
- **How:** gemmi + scipy-based local calculation
- **Integration:** Can reuse `compute_interface` data and add scoring layer.

---

### 7. Structural Quality Validation

#### `validate_structure_quality` (MolProbity / ProQ3)
- **What:** Comprehensive stereochemistry and geometry validation (Ramachandran outliers, rotamer issues, clashes).
- **How:**
  - **MolProbity:** Command-line + web server. Can be called via ` phenix.molprobity` or web API.
  - **gemmi:** Has some quality checks built-in (Ramachandran already covered in MIRA)
- **Complexity:** Medium
- **Why valuable:** For AI-generated or designed proteins, structural quality validation is critical before experimental testing.

---

## Implementation Priority Matrix

| Tool | Difficulty | Impact | Priority | Dependencies |
|------|-----------|--------|----------|--------------|
| `predict_thermostability_fast` (DDGun) | Low | High | P0 | `ddgun` pip |
| `predict_aggregation_hotspots` (TANGO) | Low | High | P0 | REST API call |
| `query_dms_data` (ProteinGym) | Medium | High | P1 | Download data files |
| `predict_mhc_binding` (MHCflurry) | Medium | High | P1 | `mhcflurry` pip |
| `predict_thermostability` (ThermoNet) | Medium | High | P0 | TensorFlow/PyTorch |
| `predict_variant_effect` (ESM-2) | High | Medium | P2 | `transformers` |
| `dock_proteins` (HADDOCK) | Medium | Medium | P2 | REST API |
| `predict_binding_affinity` (PRODIGY) | Medium | Medium | P2 | REST API |
| `validate_structure_quality` | Medium | Medium | P2 | MolProbity CLI |

---

## Architecture Notes

### Tool Pattern (from `registry.py`)
```python
from structagent.registry import tool, ToolResult

@tool(
    name="predict_thermostability",
    toolset="screening",
    description="Predict thermostability changes from mutations...",
    parameters={...},
    dependencies=["torch", "gemmi"]
)
def predict_thermostability(pdb_id: str, chain_id: str, mutations: list[dict]) -> ToolResult:
    # Implementation
    return ToolResult(success=True, data=..., raw={...})
```

### New toolset: `screening`
Currently MIRA has toolsets: `structure`, `analysis`, `web`
New: `screening` for thermostability, aggregation, immunogenicity tools.

### Lazy loading
All these tools involve heavy dependencies (PyTorch for ESM-2, pandas for DMS data, etc.). Use `@lazy_tool` decorator with `dependencies` list checked via `importlib.util.find_spec`.

### Batch integration
For screening workflows, the `batch.py` module should be extended to support ranking by thermostability, aggregation score, MHC IC50, etc.

---

## Research Notes

### REST APIs available:
- **FireProt:** `https://loschmidt.chemi.muni.cz/fireprotws/api/v1/` — swagger docs available
- **TANGO:** REST at `services.nki.uu.nl/tango` (free, no key)
- **HADDOCK:** `https://haddock.science.uu.nl/rest-api/`
- **PRODIGY:** `https://bianaconda.ewi.utwente.nl:8443/api/v1/`
- **IEDB:** `https://www.iedb.org/documentation-api` — MHC binding predictions
- **PDB RCSB GraphQL:** Already used in MIRA for structure loading

### Key datasets:
- **ProteinGym:** DMS benchmarks (DeMiguel et al., 2023) — downloadable from HuggingFace
- **S669:** Thermostability dataset of 669 proteins
- **ProTherm:** Thermodynamics database of protein mutations
- **BioLiP:** Ligand-binding residue database

### Important caveats:
- All prediction tools have error bars — should be flagged in agent output
- Rosetta ddG is slow (minutes per mutant) — not suitable for large-scale screening without GPU
- ThermoNet requires structure — for sequence-only use DDGun or ESM-2 embeddings
- MHCflurry needs HLA allele panel configuration — default panel is comprehensive
