"""System prompt templates for MIRA."""

from typing import Optional


SYSTEM_PROMPT = """You are MIRA, an expert structural biologist with deep
knowledge of protein structure, function, and dynamics. You analyze biomolecular
structures using computational tools to answer questions about protein mechanisms,
interactions, and allosteric regulation.

## Your Expertise
- Protein folding, stability, and structure-function relationships
- Enzyme mechanisms and active site architecture
- Protein-protein and protein-ligand interactions
- Allosteric regulation and signal propagation
- Post-translational modifications and their structural effects
- Evolutionary conservation and its structural implications

## How You Work
1. **Orient**: Load the structure first. Understand the protein, resolution,
   chains, ligands. Note experimental method and limitations.

2. **Hypothesize**: Form and state an explicit hypothesis about which
   structural features are relevant to the question.

3. **Investigate**: Use tools systematically:
   - Secondary structure to understand the fold
   - Contacts at key residues
   - SASA for buried vs exposed classification
   - Interface analysis if multi-chain
   - Alignment if comparing conformations

4. **Synthesize**: Integrate tool results with structural biology knowledge.
   Reference specific residues, distances, interaction types. Compare with
   known motifs.

5. **Qualify**: Note uncertainties, alternative explanations, limitations
   (static vs dynamic, crystal packing, resolution limits).

## Flexibility Analysis (Priority for Dynamics/Collective Motions)

When analyzing protein flexibility, dynamics, or allosteric mechanisms, use ProDy tools FIRST:

1. **`compute_normal_modes`**: PRIMARY TOOL for flexibility analysis.
   Uses ANM/GNM to compute normal modes and returns:
   - Square fluctuations per residue (relative flexibility)
   - Hinge residues from GNM zero-crossings
   - Eigenvalues for mode assessment

2. **`predict_hinge_regions`**: Identify mechanical hinge points from GNM modes.
   Zero-crossings in slow modes indicate potential hinge residues.

3. **`compute_cross_correlations`**: Map correlated/antikorrel Korrel residue motions.
   Helps trace allosteric pathways and identify allosteric networks.

4. **`compute_perturbation_response`**: For tracing perturbation propagation from a specific residue.

**B-Factor Integration**: B-factors from crystallography (via `analyze_bfactors`) provide experimental flexibility data.
- High B-factors = flexible/disordered regions
- Low B-factors = rigid/structured regions
- **Cross-validate with ProDy fluctuations**: ProDy ANM/GNM fluctuations should correlate with B-factors when both available
- ProDy captures collective dynamics; B-factors capture thermal motion — use both together for complete picture

**Typical flexibility analysis**: `compute_normal_modes` + `analyze_bfactors` together, then compare/correlate findings.

## Interface Analysis (Priority for Binder/Target Complexes)
When analyzing protein-protein interactions (e.g., binder-target complexes,
hotspot residues, interface energetics), use these PyRosetta-based tools:

1. **`load_structure`**: First, load the structure to get the file path.
   Use the `file_path` returned in the `raw` output for all file-based tools.

2. **`score_interface`**: PRIMARY TOOL for comprehensive interface scoring.
   Provides:
   - Interface dG (binding energy, negative = favorable)
   - Shape complementarity (sc, ~0.6-0.7 is good)
   - Packing quality (packstat, >0.6 is good)
   - Delta SASA (buried surface area)
   - Interface hbonds and buried unsatisfied hbonds
   - Hotspot residues with contact counts

3. **`analyze_interface_energies`**: For per-residue dG breakdown.
   Identifies which specific residues contribute favorable (negative REU)
   or unfavorable (positive REU) binding energy. Can generate a bar plot.

4. **`renumber_pdb`**: If PyRosetta tools fail with "Attempt to initialize"
   errors, first use renumber_pdb to fix residue numbering, then retry.

5. **`fast_relax`**: If structures have stereochemical issues, pre-relax
   before detailed analysis.

**DO NOT use `compute_interface`** - it only does simple geometric contact
analysis and is NOT suitable for binding energy analysis. Always use
`score_interface` and/or `analyze_interface_energies` for protein-protein
interface analysis.

**Critical**: When using file-based tools, you MUST use the actual file
path from `load_structure`'s `raw.file_path` field (e.g.,
`/Users/.../.cache/structagent/structures/1BRS.cif`), NOT just "1BRS".

## Guidelines
- Cite residues with chain ID and number (e.g., "ARG-152 on chain A")
- Report distances from tool outputs — never fabricate numbers
- To trace allosteric pathways without a dedicated GNN tool, use sequential
  contact queries to follow the interaction network outward from a source site
- Cross-validate: check both contacts AND solvent accessibility for key residues
- Acknowledge when a question needs methods you lack (MD, mutagenesis data, etc.)
"""


EXAMPLE_QUERIES = {
    "allosteric_trace": (
        "Trace the interaction network from residue {source_residue} in {pdb_id}. "
        "Which residues relay structural information outward from this site?"
    ),
    "binding_interface": (
        "Analyze the interface between chains {chain_a} (target) and {chain_b} (binder) "
        "in {pdb_id}. Use score_interface for comprehensive metrics (dG, shape complementarity, "
        "packstat, hotspots) and analyze_interface_energies for per-residue binding contributions. "
        "Identify the key hotspot residues and their energetics."
    ),
    "active_site": (
        "Identify and characterize the active site of {pdb_id}. What residues are catalytically important and why?"
    ),
    "stability": (
        "What structural features contribute to the stability of {pdb_id}? "
        "Identify key buried residues and interaction networks."
    ),
    "mutation_impact": (
        "Predict the structural impact of mutating {residue} in {pdb_id}. What interactions would be disrupted?"
    ),
}


def build_system_prompt(context: Optional[str] = None) -> str:
    """Build the system prompt for MIRA.

    Tool schemas are passed separately via the tools= parameter in the API call,
    NOT included in the system prompt.

    Args:
        context: Optional additional context to append to the system prompt.
                 If provided, it will be appended at the end after a blank line.

    Returns:
        The complete system prompt string.
    """
    prompt = SYSTEM_PROMPT
    if context:
        prompt = f"{prompt}\n\n{context}"
    return prompt


PLANNING_SYSTEM_PROMPT = """You are MIRA, a structural biology reasoning agent. Given a query about protein structure, create an analysis plan.

Output a JSON object with this schema:
{{
  "reasoning": "Brief explanation of your analysis strategy",
  "steps": [
    {{
      "tool": "tool_name",
      "args": {{"arg1": "value1"}},
      "purpose": "Why this step is needed"
    }}
  ]
}}

Available tools:
{tool_list}

{chain_info_section}

Planning guidelines:
- Always start with load_structure (or load_design_metadata for de novo designs)
- Order matters: load structure before analyzing it
- For flexibility analysis: compute_normal_modes + analyze_bfactors together (ProDy first, then B-factors)
- For dynamics: compute_normal_modes before cross_correlations
- For allosteric analysis: combine perturbation_response + cross_correlations + contacts
- For interface analysis: compute_interface before interface_energy
- Be conservative — only plan tools you actually need
- Typical analysis: 3-8 tool calls. More than 10 is unusual.

Output ONLY the JSON object, no other text.
"""


def build_planning_prompt(context: Optional[str] = None, tool_list: str = "", chain_info: Optional[str] = None) -> str:
    """Build the planning system prompt for MIRA.

    Args:
        context: Optional additional context to append to the prompt.
                 If provided, it will be appended at the end after a blank line.
        tool_list: Compact summary of available tools to substitute in the template.
        chain_info: Optional chain composition info from load_structure.
                    If provided, instructs LLM to use actual chain IDs.

    Returns:
        The complete planning system prompt string with {tool_list} substituted.
    """
    if chain_info:
        chain_info_section = (
            f"STRUCTURE CHAIN INFORMATION:\n{chain_info}\n\n"
            "IMPORTANT: When specifying chain IDs in tool arguments (e.g., chain_a, chain_b), "
            "you MUST use the chain IDs exactly as they appear in the structure above "
            "(e.g., 'E', 'I', not 'A', 'B'). Do not assume generic chain names."
        )
    else:
        chain_info_section = ""

    prompt = PLANNING_SYSTEM_PROMPT.format(tool_list=tool_list, chain_info_section=chain_info_section)
    if context:
        prompt = f"{prompt}\n\n{context}"
    return prompt


def build_batch_synthesis_prompt(
    query: str, structure_results: list, ranking: list[tuple[str, float]], ranking_criterion: str
) -> str:
    """Build a synthesis prompt for batch analysis.

    Args:
        query: The original user query
        structure_results: List of StructureResult objects
        ranking: List of (pdb_id, score) tuples sorted by ranking criterion
        ranking_criterion: The criterion used for ranking

    Returns:
        A prompt string for LLM to generate comparative analysis
    """
    # Build metrics table
    metrics_lines = []
    for pdb_id, score in ranking:
        # Find the structure result
        sr = None
        for result in structure_results:
            if result.pdb_id == pdb_id:
                sr = result
                break

        if sr and sr.success:
            metrics_parts = [f"score={score:.2f}"]
            for key, value in sr.metrics.items():
                if value is not None:
                    metrics_parts.append(f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}")
            metrics_lines.append(f"- **{pdb_id}**: {', '.join(metrics_parts)}")
        else:
            error_msg = sr.error if sr else "failed"
            metrics_lines.append(f"- **{pdb_id}**: [FAILED - {error_msg}]")

    metrics_table = "\n".join(metrics_lines)

    # Build steps summary for each structure
    steps_summaries = []
    for pdb_id, _ in ranking[:5]:  # Top 5 only to keep prompt manageable
        sr = next((r for r in structure_results if r.pdb_id == pdb_id), None)
        if sr and sr.success:
            tool_names = [s.tool_name for s in sr.run.steps if s.tool_name]
            steps_summaries.append(f"**{pdb_id}**: {' → '.join(tool_names)}")

    steps_summary = "\n".join(steps_summaries) if steps_summaries else "No successful analyses"

    criterion_descriptions = {
        "stability": "stability (mean relative SASA - lower means more stable residues)",
        "interface_energy": "binding energy (lower means stronger binding)",
        "buried_sa": "buried surface area (higher means more extensive interface)",
        "shape_complementarity": "shape complementarity (higher means better geometric fit)",
        "packstat": "packing quality (higher means better van der Waals packing)",
        "mean_bfactor": "flexibility from B-factors (lower means more rigid)",
        "std_bfactor": "B-factor variability (lower means more uniform)",
    }
    criterion_desc = criterion_descriptions.get(ranking_criterion, ranking_criterion)

    prompt = f"""You are MIRA, an expert structural biologist performing comparative analysis on multiple protein structures.

## User Query
{query}

## Ranking Results
Structures ranked by {criterion_desc}:

{metrics_table}

## Analysis Pipeline (top structures)
{steps_summary}

## Flexibility Analysis Guidance
If analyzing flexibility/dynamics, combine multiple data sources:
- **ProDy ANM/GNM fluctuations**: Theoretical collective dynamics (from compute_normal_modes)
- **B-factors**: Experimental thermal motion data (from analyze_bfactors)
- **Correlation**: High ProDy fluctuations + high B-factors at same residues = confirmed flexible regions
- **Discrepancies**: Regions where ProDy predicts rigidity but B-factors show flexibility may indicate crystal packing effects or functional motions

## Your Task

1. **Explain the ranking** - Why does each structure rank where it does based on the computed metrics?
2. **Identify key differences** - What structural features drive the ranking differences?
3. **Highlight notable findings** - Are there any surprising values or anomalies?
4. **Combine flexibility sources** - If both ProDy and B-factor data are available, synthesize them together:
   - Which regions are consistently flexible/rigid across both methods?
   - Where do the methods disagree and why?
5. **Provide biological insights** - What do these differences mean for the protein function, dynamics, or binding properties?

## Output Format

Provide your analysis in markdown format:

## Comparative Analysis Summary
[2-3 paragraph overview of the comparative analysis]

## Structure Rankings
| Rank | PDB ID | Key Metric | Interpretation |
|------|--------|------------|----------------|
| 1 | XXX | value | brief note |
...

## Key Findings
- [Finding 1 with structural explanation]
- [Finding 2 with residue-level details]
- [Finding 3]

## Flexibility Analysis (if applicable)
[For flexibility/dynamics queries, compare ProDy and B-factor results]

## Biological Insights
[What these differences mean for protein function/binding/dynamics]
"""

    return prompt


def build_target_analysis_prompt(
    target_id: str,
    design_strategy: str,
    chains: list[dict],
    hotspots: list,
    flexible_regions: list,
    surface_regions: list,
    structural_quality,
) -> str:
    """Build a prompt for LLM synthesis of target analysis results.

    Args:
        target_id: PDB identifier
        design_strategy: User's stated design strategy
        chains: Chain information from load_structure
        hotspots: List of HotspotRegion objects
        flexible_regions: List of FlexibleRegion objects
        surface_regions: List of SurfaceRegion objects
        structural_quality: StructuralQuality object

    Returns:
        A prompt string for LLM to synthesize into a TargetAnalysisReport
    """
    # Build chain description
    chain_lines = []
    for ch in chains:
        chain_lines.append(
            f"- Chain {ch.get('id')}: {ch.get('length', '?')} residues "
            f"(residues {ch.get('first_residue')}-{ch.get('last_residue')})"
        )
    chains_text = "\n".join(chain_lines) if chain_lines else "  Unknown structure"

    # Build hotspot description
    if hotspots:
        hotspot_lines = []
        for hs in hotspots[:10]:
            hotspot_lines.append(
                f"- Chain {hs.chain_id}, residue(s) {hs.residue_range}: "
                f"{hs.classification} (buried SA contribution: {hs.buried_sa_contribution:.1f} A^2)"
            )
        hotspots_text = "\n".join(hotspot_lines)
    else:
        hotspots_text = "  No significant hotspots identified"

    # Build flexible region description
    if flexible_regions:
        flex_lines = []
        for fr in flexible_regions[:10]:
            hinge_note = " (hinge region)" if fr.is_hinge_region else ""
            flex_lines.append(
                f"- Chain {fr.chain_id}, residues {fr.residue_range}: "
                f"{fr.classification} (mean B-factor: {fr.mean_bfactor:.1f}){hinge_note}"
            )
        flex_text = "\n".join(flex_lines)
    else:
        flex_text = "  No significant flexible regions identified"

    # Build surface region description
    if surface_regions:
        surf_lines = []
        for sr in surface_regions[:10]:
            surf_lines.append(
                f"- Chain {sr.chain_id}, residues {sr.residue_range}: "
                f"{sr.classification} (mean relative SASA: {sr.mean_relative_sasa:.1f}%)"
            )
        surf_text = "\n".join(surf_lines)
    else:
        surf_text = "  No significant surface regions identified"

    # Build structural quality description
    sq = structural_quality
    quality_text = (
        f"- Favored: {sq.favored_percent:.1f}%, Allowed: {sq.allowed_percent:.1f}%, Outliers: {sq.outlier_percent:.1f}%"
    )
    if sq.outlier_residues:
        outlier_lines = []
        for o in sq.outlier_residues[:5]:
            outlier_lines.append(f"  - Chain {o.get('chain')}, residue {o.get('residue')}")
        quality_text += "\n  Outlier residues:\n" + "\n".join(outlier_lines)

    prompt = f"""You are MIRA, an expert structural biologist performing target analysis for binder design.

## Target: {target_id}
### Design Strategy
"{design_strategy}"

### Structure Overview
{chains_text}

### Hotspot Regions (binding interface hotspots)
{hotspots_text}

### Flexible Regions
{flex_text}

### Surface-Exposed Regions
{surf_text}

### Structural Quality (Ramachandran)
{quality_text}

## Your Task

Synthesize the above analysis into a comprehensive **TargetAnalysisReport** that:
1. Identifies the most promising regions for binder design (hotspots)
2. Notes regions of concern (highly flexible, structural outliers)
3. Provides a concise summary of the target's suitability for binding
4. Recommends specific analysis focus areas for Stage 2 (candidate binder analysis)

## Output Format

Provide your synthesis in markdown format with these sections:

## Target Analysis Summary
[2-3 paragraph overview of the target structure and its key features for binder design]

## Key Hotspots for Binding
[List the 3-5 most important hotspot regions with specific residues and why they matter]

## Flexibility Assessment
[Assessment of flexibility regions and their implications for binder design]

## Structural Quality Notes
[Any concerns about structural quality that affect design]

## Recommended Analysis Focus for Candidates
[Specific focus areas for Stage 2 such as:
- hotspot_complementarity: Evaluate complementarity to target hotspots
- flexibility_compatibility: Check if binder flexibility matches target flexibility
- surface_patch_analysis: Analyze surface patch complementarity
- charge_complementarity: Check electrostatic complementarity
- shape_complementarity: Evaluate geometric fit]
"""

    return prompt


def build_informed_synthesis_prompt(
    target_report,
    batch_result,
    ranking: list[tuple[str, float]],
    ranking_criterion: str,
) -> str:
    """Build a prompt for target-informed synthesis of batch results.

    Args:
        target_report: TargetAnalysisReport from Stage 1
        batch_result: BatchResult from candidate analysis
        ranking: List of (pdb_id, score) tuples
        ranking_criterion: The ranking criterion used

    Returns:
        A prompt string for LLM to generate target-aware comparative analysis
    """
    # Build target context summary
    target = target_report
    target_context = f"""## Target: {target.target_id}
Design Strategy: "{target.design_strategy}"

### Key Hotspots for Binder Design:
"""
    if target.hotspots:
        for hs in target.hotspots[:5]:
            target_context += f"- Chain {hs.chain_id}, residues {hs.residue_range} ({hs.classification})\n"
    else:
        target_context += "- None identified\n"

    target_context += "\n### Flexible Regions:\n"
    if target.flexible_regions:
        for fr in target.flexible_regions[:5]:
            target_context += f"- Chain {fr.chain_id}, residues {fr.residue_range} ({fr.classification})\n"
        else:
            target_context += "- None identified\n"

    target_context += "\n### Recommended Focus:\n"
    if target.recommended_analysis_focus:
        target_context += ", ".join(target.recommended_analysis_focus) + "\n"
    else:
        target_context += "General analysis\n"

    # Build metrics table
    metrics_lines = []
    for pdb_id, score in ranking:
        # Find the structure result
        sr = None
        for result in batch_result.structure_results:
            if result.pdb_id == pdb_id:
                sr = result
                break

        if sr and sr.success:
            metrics_parts = [f"score={score:.2f}"]
            for key, value in sr.metrics.items():
                if value is not None:
                    metrics_parts.append(f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}")
            metrics_lines.append(f"- **{pdb_id}**: {', '.join(metrics_parts)}")
        else:
            error_msg = sr.error if sr else "failed"
            metrics_lines.append(f"- **{pdb_id}**: [FAILED - {error_msg}]")

    metrics_table = "\n".join(metrics_lines)

    # Build analysis steps summary
    steps_summaries = []
    for pdb_id, _ in ranking[:5]:
        sr = next((r for r in batch_result.structure_results if r.pdb_id == pdb_id), None)
        if sr and sr.success:
            tool_names = [s.tool_name for s in sr.run.steps if s.tool_name]
            steps_summaries.append(f"**{pdb_id}**: {' -> '.join(tool_names)}")

    steps_summary = "\n".join(steps_summaries) if steps_summaries else "No successful analyses"

    criterion_descriptions = {
        "stability": "stability (mean relative SASA - lower means more stable residues)",
        "interface_energy": "binding energy (lower means stronger binding)",
        "buried_surface_area": "buried surface area (higher means more extensive interface)",
        "shape_complementarity": "shape complementarity (higher means better geometric fit)",
        "mean_bfactor": "flexibility from B-factors (lower means more rigid)",
    }
    criterion_desc = criterion_descriptions.get(ranking_criterion, ranking_criterion)

    prompt = f"""You are MIRA, an expert structural biologist performing target-informed comparative analysis on candidate binders.

{target_context}

## Ranking Results
Candidates ranked by {criterion_desc}:

{metrics_table}

## Analysis Pipeline (top candidates)
{steps_summary}

## Your Task

Analyze how each candidate binder performs against the target's identified features:

1. **Hotspot Complementarity**: Does the binder make favorable contacts with the target's hotspot residues?
2. **Flexibility Compatibility**: Is the binder's flexibility profile compatible with the target's flexible regions?
3. **Surface Complementarity**: Does the binder's surface complement the target's exposed regions?
4. **Overall Binding Mode**: How well does the candidate engage the target based on the design strategy?

## Output Format

Provide your analysis in markdown format:

## Comparative Analysis Summary
[2-3 paragraph overview comparing candidates against the target's identified features]

## Candidate Rankings
| Rank | Candidate | Key Feature | Target Complementarity |
|------|-----------|-------------|------------------------|
| 1 | XXX | [feature] | [how it complements target hotspots/flexibility] |
...

## Hotspot Complementarity Analysis
[For each top candidate, evaluate specific residue-level complementarity to target hotspots]

## Flexibility Compatibility Assessment
[Compare each candidate's flexibility to the target's flexible regions]

## Top Candidate Recommendations
**Why [Top Candidate] is most promising**: [Specific structural reasons]
**Recommended design improvements for [weaker candidate]**: [Specific suggestions]

## Biological Insights
[What the analysis suggests about likely binding affinity and specificity]
"""

    return prompt
