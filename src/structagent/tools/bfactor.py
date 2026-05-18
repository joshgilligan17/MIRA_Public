"""Tools for analyzing B-factors (temperature factors) in protein structures."""

import os

import numpy as np

from structagent.registry import tool, ToolResult
from structagent.tools.structure_io import get_structure


def _parse_residue_range(range_str: str) -> tuple[int, int]:
    """
    Parse a residue range string like '1-50' into start and end integers.

    Args:
        range_str: String in format 'start-end' (e.g., '1-50')

    Returns:
        Tuple of (start_residue, end_residue)

    Raises:
        ValueError: If the format is invalid
    """
    parts = range_str.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid residue range format: '{range_str}'. Expected 'start-end' (e.g., '1-50')")
    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid residue range values: '{range_str}'. Must be integers.")
    if start > end:
        raise ValueError(f"Invalid residue range: start ({start}) > end ({end})")
    return start, end


def _classify_bfactor(b: float, mean: float, std: float) -> str:
    """
    Classify a B-factor value based on statistics.

    Classification:
    - rigid: B < mean - std
    - ordered: mean - std <= B <= mean + std
    - flexible: mean + std < B <= mean + 2*std
    - highly_flexible: B > mean + 2*std
    """
    if b < mean - std:
        return "rigid"
    elif b <= mean + std:
        return "ordered"
    elif b <= mean + 2 * std:
        return "flexible"
    else:
        return "highly_flexible"


@tool(
    name="analyze_bfactors",
    toolset="structure",
    description="Analyze per-residue B-factors (temperature factors) in a protein structure to identify rigid, ordered, and flexible regions.",
    parameters={
        "pdb_id": {
            "type": "string",
            "description": "4-character PDB identifier (e.g., '1ABC', '6VXX'). Use either this or pdb_path.",
        },
        "pdb_path": {"type": "string", "description": "Path to local PDB file. Use either this or pdb_id."},
        "chain_id": {"type": "string", "description": "Chain identifier (e.g., 'A', 'B')"},
        "residue_range": {
            "type": "string",
            "description": "Optional residue range to analyze in format 'start-end' (e.g., '1-50'). If not provided, analyzes the entire chain.",
            "default": None,
        },
    },
)
def analyze_bfactors(
    pdb_id: str = None, pdb_path: str = None, chain_id: str = None, residue_range: str | None = None
) -> ToolResult:
    """
    Analyze B-factors (temperature factors) for residues in a PDB structure.

    This tool computes per-residue average B-factors from all atoms in each residue,
    calculates overall statistics (mean, std, median, min, max), and classifies
    each residue as:
    - rigid: B < mean - std (very stable, low thermal motion)
    - ordered: mean - std <= B <= mean + std (typical ordered region)
    - flexible: mean + std < B <= mean + 2*std (some flexibility)
    - highly_flexible: B > mean + 2*std (disordered/highly mobile)

    Additionally:
    - Flags residues with B > 80 A^2 as potentially disordered
    - Warns if the structure appears to be cryo-EM (typically has uniform B-factors)

    Parameters
    ----------
    pdb_id : str
        4-character PDB identifier
    chain_id : str
        Chain identifier (e.g., 'A')
    residue_range : str, optional
        Residue range to analyze in format 'start-end' (e.g., '1-50').
        If None, analyzes the entire chain.

    Returns
    -------
    ToolResult
        success: bool indicating if the operation succeeded
        data: Human-readable narrative description of B-factor analysis
        raw: Dict with per-residue B-factors, statistics, and classifications
    """
    try:
        structure = get_structure(pdb_id=pdb_id, pdb_path=pdb_path)

        # Parse residue range if provided
        range_start = None
        range_end = None
        if residue_range is not None:
            range_start, range_end = _parse_residue_range(residue_range)

        # Collect per-residue B-factors for the specified chain
        residue_bfactors: dict[int, list[float]] = {}  # res_num -> list of B values

        for model in structure:
            for chain in model:
                if chain.name != chain_id:
                    continue

                for residue in chain:
                    res_num = residue.seqid.num

                    # Filter by range if specified
                    if range_start is not None and range_end is not None:
                        if res_num < range_start or res_num > range_end:
                            continue

                    # Collect all B-factors for this residue
                    b_values = []
                    for atom in residue:
                        b = atom.b_iso
                        if b is not None and b > 0:
                            b_values.append(b)

                    if b_values:
                        residue_bfactors[res_num] = b_values

        if not residue_bfactors:
            pdb_label = pdb_id if pdb_id else os.path.basename(pdb_path) if pdb_path else "unknown"
            return ToolResult(
                success=False,
                data=f"No B-factor data found for chain {chain_id} of {pdb_label}"
                + (f" in residue range {residue_range}" if residue_range else ""),
                raw={},
                error="No B-factor data found",
            )

        # Compute per-residue average B-factors
        residue_avg_bfactors: dict[int, float] = {}
        all_b_values: list[float] = []

        for res_num, b_values in residue_bfactors.items():
            avg_b = sum(b_values) / len(b_values)
            residue_avg_bfactors[res_num] = avg_b
            all_b_values.extend(b_values)

        # Compute overall statistics
        all_b_values = np.array(all_b_values)
        overall_mean = float(np.mean(all_b_values))
        overall_std = float(np.std(all_b_values))
        overall_median = float(np.median(all_b_values))
        overall_min = float(np.min(all_b_values))
        overall_max = float(np.max(all_b_values))

        # Detect cryo-EM structures (typically have uniform B-factors around 60-80)
        is_likely_cryoem = False
        if overall_std < 10.0 and overall_mean > 40.0 and overall_mean < 100.0:
            is_likely_cryoem = True

        # Classify each residue
        residues_data: list[dict] = []
        classification_counts = {"rigid": 0, "ordered": 0, "flexible": 0, "highly_flexible": 0}
        disordered_count = 0

        for res_num in sorted(residue_avg_bfactors.keys()):
            avg_b = residue_avg_bfactors[res_num]
            classification = _classify_bfactor(avg_b, overall_mean, overall_std)
            is_disordered = avg_b > 80.0

            residue_info = {
                "residue_number": res_num,
                "avg_bfactor": round(avg_b, 2),
                "classification": classification,
                "potentially_disordered": is_disordered,
            }

            # Get residue name from structure
            for model in structure:
                for chain in model:
                    if chain.name == chain_id:
                        for residue in chain:
                            if residue.seqid.num == res_num:
                                residue_info["residue_name"] = residue.name
                                break

            residues_data.append(residue_info)
            classification_counts[classification] += 1
            if is_disordered:
                disordered_count += 1

        # Build narrative
        lines = []

        range_desc = f" in residues {range_start}-{range_end}" if residue_range else ""
        pdb_label = pdb_id.upper() if pdb_id else os.path.basename(pdb_path) if pdb_path else "unknown"
        lines.append(f"B-factor analysis for chain {chain_id} of {pdb_label}{range_desc}")
        lines.append(
            f"Statistics: mean={overall_mean:.1f}, std={overall_std:.1f}, median={overall_median:.1f}, min={overall_min:.1f}, max={overall_max:.1f} A^2"
        )
        lines.append(f"Total residues analyzed: {len(residues_data)}")

        if is_likely_cryoem:
            lines.append(
                "Note: This structure appears to be cryo-EM (uniform B-factors ~60-80 A^2). B-factor analysis may be less meaningful for identifying flexibility."
            )

        lines.append("")
        lines.append("Classification summary:")
        lines.append(f"  Rigid (B < mean - std): {classification_counts['rigid']} residues")
        lines.append(f"  Ordered (mean - std <= B <= mean + std): {classification_counts['ordered']} residues")
        lines.append(f"  Flexible (mean + std < B <= mean + 2*std): {classification_counts['flexible']} residues")
        lines.append(f"  Highly flexible (B > mean + 2*std): {classification_counts['highly_flexible']} residues")

        if disordered_count > 0:
            lines.append(f"Potentially disordered (B > 80 A^2): {disordered_count} residues")

        # List highly flexible and rigid residues
        lines.append("")
        lines.append("Rigid regions (B < {:.1f} A^2):".format(overall_mean - overall_std))
        rigid_res = [r for r in residues_data if r["classification"] == "rigid"]
        if rigid_res:
            rigid_strs = [f"{r['residue_name']}-{r['residue_number']} ({r['avg_bfactor']} A^2)" for r in rigid_res[:10]]
            lines.append("  " + ", ".join(rigid_strs))
            if len(rigid_res) > 10:
                lines.append(f"  ... and {len(rigid_res) - 10} more rigid residues")
        else:
            lines.append("  None")

        lines.append("")
        lines.append("Highly flexible regions (B > {:.1f} A^2):".format(overall_mean + 2 * overall_std))
        flexible_res = [r for r in residues_data if r["classification"] == "highly_flexible"]
        if flexible_res:
            flex_strs = [
                f"{r['residue_name']}-{r['residue_number']} ({r['avg_bfactor']} A^2)" for r in flexible_res[:10]
            ]
            lines.append("  " + ", ".join(flex_strs))
            if len(flexible_res) > 10:
                lines.append(f"  ... and {len(flexible_res) - 10} more highly flexible residues")
        else:
            lines.append("  None")

        # List potentially disordered residues
        disordered_res = [r for r in residues_data if r["potentially_disordered"]]
        if disordered_res:
            lines.append("")
            lines.append("Potentially disordered residues (B > 80 A^2):")
            disc_strs = [
                f"{r['residue_name']}-{r['residue_number']} ({r['avg_bfactor']} A^2)" for r in disordered_res[:10]
            ]
            lines.append("  " + ", ".join(disc_strs))
            if len(disordered_res) > 10:
                lines.append(f"  ... and {len(disordered_res) - 10} more disordered residues")

        data = "\n".join(lines)

        raw = {
            "pdb_id": pdb_id.upper() if pdb_id else os.path.basename(pdb_path) if pdb_path else "unknown",
            "chain_id": chain_id,
            "residue_range": residue_range,
            "statistics": {
                "mean": round(overall_mean, 2),
                "std": round(overall_std, 2),
                "median": round(overall_median, 2),
                "min": round(overall_min, 2),
                "max": round(overall_max, 2),
                "total_atoms": int(len(all_b_values)),
            },
            "classification_counts": classification_counts,
            "likely_cryoem": is_likely_cryoem,
            "potentially_disordered_count": disordered_count,
            "residues": residues_data,
        }

        return ToolResult(success=True, data=data, raw=raw)

    except ValueError as e:
        return ToolResult(
            success=False, data=f"Error analyzing B-factors for chain {chain_id}: {str(e)}", raw={}, error=str(e)
        )
    except Exception as e:
        return ToolResult(
            success=False,
            data=f"Unexpected error analyzing B-factors: {type(e).__name__}: {str(e)}",
            raw={},
            error=str(e),
        )
