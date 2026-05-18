"""Tools for analyzing charge distribution and charge clusters in protein structures."""

from typing import Optional

import numpy as np
from scipy.spatial import KDTree

from structagent.registry import tool, ToolResult
from structagent.tools.structure_io import get_structure


# Standard pKa values at neutral pH (Henderson-Hasselbalch reference)
PKA_VALUES = {
    "ASP": 3.65,
    "GLU": 4.25,
    "HIS": 6.00,
    "CYS": 8.18,
    "TYR": 10.07,
    "LYS": 10.53,
    "ARG": 12.48,
    "NTERM": 9.69,  # N-terminus
    "CTERM": 2.34,  # C-terminus
}

# Charged residue types
CHARGED_RESIDUES = {"ASP", "GLU", "HIS", "CYS", "TYR", "LYS", "ARG"}

# pH-dependent charge calculation using Henderson-Hasselbalch equation
# charge = 1 / (1 + 10^(pKa - pH)) for acidic (ASP, GLU, CYS, TYR)
# charge = 1 / (1 + 10^(pH - pKa)) for basic (LYS, ARG, HIS)
# Simplified: charge = 1 / (1 + 10^(pKa - pH)) works for all if pKa is for deprotonation


def _compute_residue_charge(res_name: str, ph: float) -> float:
    """
    Compute the charge of a residue at a given pH using Henderson-Hasselbalch.

    For acidic residues (ASP, GLU): charge = 1 / (1 + 10^(pKa - pH))
    For basic residues (LYS, ARG, HIS): charge = 1 / (1 + 10^(pH - pKa))
    For neutral with ionizable sidechains (CYS, TYR): charge = 1 / (1 + 10^(pKa - pH))

    Args:
        res_name: 3-letter residue code
        ph: pH value

    Returns:
        Fractional charge (0 to 1)
    """
    if res_name not in PKA_VALUES:
        return 0.0

    pka = PKA_VALUES[res_name]

    # Henderson-Hasselbalch: fraction deprotonated = 1 / (1 + 10^(pKa - pH))
    # For basic residues, we invert since we're computing protonated state
    if res_name in ("LYS", "ARG", "HIS"):
        # For basic: protonated = 1 / (1 + 10^(pH - pKa))
        charge = 1 / (1 + 10 ** (ph - pka))
    else:
        # For acidic and neutral ionizable: deprotonated = 1 / (1 + 10^(pKa - pH))
        charge = 1 / (1 + 10 ** (pka - ph))

    return charge


def _parse_residue_range(range_str: Optional[str]) -> Optional[tuple[int, int]]:
    """Parse residue range string like '1-50' into (start, end)."""
    if not range_str:
        return None
    try:
        parts = range_str.split("-")
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    except ValueError:
        return None
    return None


def _find_charge_clusters(charged_residues: list[dict], distance_threshold: float = 8.0) -> list[list[dict]]:
    """
    Identify charge clusters: groups of >= 3 charged residues within distance_threshold (Cα).

    Args:
        charged_residues: List of dicts with 'chain', 'res_num', 'res_name', 'ca_coord', 'charge'
        distance_threshold: Maximum Cα-Cα distance in Angstroms for clustering

    Returns:
        List of clusters, where each cluster is a list of residue dicts
    """
    if len(charged_residues) < 3:
        return []

    # Extract Cα coordinates
    coords = []
    for res in charged_residues:
        coords.append(res["ca_coord"])

    coords = np.array(coords)

    # Build KDTree for efficient distance queries
    kdtree = KDTree(coords)

    # Find all pairs within threshold
    clusters = []
    used = set()

    for i, res in enumerate(charged_residues):
        if i in used:
            continue

        # Find all residues within threshold of this one
        indices = kdtree.query_ball_point(coords[i], distance_threshold)
        indices = [idx for idx in indices if idx != i]

        if len(indices) >= 2:  # Need at least 3 total (including current)
            # Build cluster
            cluster_indices = {i} | set(indices)
            cluster = [charged_residues[j] for j in cluster_indices]
            clusters.append(cluster)

            # Mark all as used
            used.update(cluster_indices)

    return clusters


@tool(
    name="compute_charge_distribution",
    toolset="analysis",
    description="Compute the charge distribution of a protein structure at a given pH, identifying charged residues and charge clusters (3+ residues within 8 Angstroms).",
    parameters={
        "type": "object",
        "properties": {
            "pdb_id": {
                "type": "string",
                "description": "4-character PDB identifier (e.g., '1ABC', '6VXX'). Use either this or pdb_path.",
            },
            "pdb_path": {"type": "string", "description": "Path to local PDB/mmCIF file. Use either this or pdb_id."},
            "chain_id": {"type": "string", "description": "Chain identifier (e.g., 'A', 'B')"},
            "ph": {"type": "number", "description": "pH value for charge calculation (default: 7.4)", "default": 7.4},
            "residue_range": {
                "type": "string",
                "description": "Optional residue range to analyze, format 'start-end' (e.g., '1-50')",
                "default": None,
            },
        },
        "required": ["chain_id"],
    },
)
def compute_charge_distribution(
    pdb_id: str = None, pdb_path: str = None, chain_id: str = None, ph: float = 7.4, residue_range: Optional[str] = None
) -> ToolResult:
    """
    Compute the charge distribution of a protein at a given pH.

    Uses the Henderson-Hasselbalch equation to calculate the fractional charge
    of each ionizable residue at the specified pH. Identifies charge clusters
    where 3 or more charged residues have C-alpha atoms within 8 Angstroms.

    Parameters
    ----------
    pdb_id : str
        4-character PDB identifier
    chain_id : str
        Chain identifier (e.g., 'A')
    ph : float, optional
        pH value for charge calculation (default 7.4)
    residue_range : str, optional
        Residue range to analyze, format 'start-end' (e.g., '1-50')

    Returns
    -------
    ToolResult
        success: bool indicating if the operation succeeded
        data: Human-readable narrative description of charge distribution
        raw: Dict with charged_residues, total_charge, clusters, ph, pka_values
    """
    try:
        structure = get_structure(pdb_id=pdb_id, pdb_path=pdb_path)
        pdb_label = pdb_id.upper() if pdb_id else (pdb_path if pdb_path else "unknown")

        # Find target chain
        target_chain = None
        for model in structure:
            for chain in model:
                if chain.name == chain_id:
                    target_chain = chain
                    break
            if target_chain:
                break

        if target_chain is None:
            return ToolResult(
                success=False,
                data=f"Chain {chain_id} not found in {pdb_label}",
                raw={},
                error=f"Chain {chain_id} not found",
            )

        # Parse residue range if provided
        range_tuple = _parse_residue_range(residue_range)

        # Collect charged residues
        charged_residues = []
        nterm_found = False
        cterm_found = False

        prev_residue = None
        for residue in target_chain:
            res_num = int(residue.seqid.num)  # Ensure numeric comparison
            res_name = residue.name

            # Filter by residue range if specified
            if range_tuple:
                start, end = range_tuple
                if res_num < start or res_num > end:
                    continue

            # Find C-alpha atom
            ca_atom = None
            for atom in residue:
                if atom.name == "CA":
                    ca_atom = atom
                    break

            if ca_atom is None:
                continue

            # Check for charged residues
            if res_name in CHARGED_RESIDUES:
                charge = _compute_residue_charge(res_name, ph)
                charged_residues.append(
                    {
                        "chain": chain_id,
                        "res_num": res_num,
                        "res_name": res_name,
                        "ca_coord": np.array([ca_atom.pos.x, ca_atom.pos.y, ca_atom.pos.z]),
                        "charge": charge,
                        "pka": PKA_VALUES.get(res_name),
                    }
                )

            # Check for N-terminus (only once at the beginning of chain)
            if not nterm_found and prev_residue is None:
                # First residue in chain is N-terminus
                nterm_found = True
                charge = _compute_residue_charge("NTERM", ph)
                charged_residues.append(
                    {
                        "chain": chain_id,
                        "res_num": res_num,
                        "res_name": "NTERM",
                        "ca_coord": np.array([ca_atom.pos.x, ca_atom.pos.y, ca_atom.pos.z]),
                        "charge": charge,
                        "pka": PKA_VALUES.get("NTERM"),
                    }
                )

            prev_residue = residue

        # Handle C-terminus (last residue)
        if charged_residues and not cterm_found:
            # C-terminus is at the end of the chain - use last residue
            last_res = charged_residues[-1]
            # Find the actual last residue in the chain
            for residue in target_chain:
                last_res = residue
            # Check if we already added it as NTERM (for single residue chains)
            if int(last_res.seqid.num) != charged_residues[0]["res_num"] or charged_residues[0]["res_name"] != "NTERM":
                cterm_found = True
                ca_atom = None
                for atom in last_res:
                    if atom.name == "CA":
                        ca_atom = atom
                        break
                if ca_atom:
                    charge = _compute_residue_charge("CTERM", ph)
                    charged_residues.append(
                        {
                            "chain": chain_id,
                            "res_num": int(last_res.seqid.num),
                            "res_name": "CTERM",
                            "ca_coord": np.array([ca_atom.pos.x, ca_atom.pos.y, ca_atom.pos.z]),
                            "charge": charge,
                            "pka": PKA_VALUES.get("CTERM"),
                        }
                    )

        if not charged_residues:
            return ToolResult(
                success=False,
                data=f"No charged residues found in chain {chain_id} of {pdb_label}",
                raw={},
                error="No charged residues",
            )

        # Calculate total charge
        total_charge = sum(r["charge"] for r in charged_residues)

        # Separate positive and negative charges
        positive_charges = [r for r in charged_residues if r["res_name"] in ("LYS", "ARG", "HIS", "NTERM")]
        negative_charges = [r for r in charged_residues if r["res_name"] in ("ASP", "GLU", "CYS", "TYR", "CTERM")]

        total_positive = sum(r["charge"] for r in positive_charges)
        total_negative = sum(r["charge"] for r in negative_charges)

        # Find charge clusters
        clusters = _find_charge_clusters(charged_residues, distance_threshold=8.0)

        # Format charged residues for output (remove numpy arrays)
        output_residues = []
        for r in charged_residues:
            output_residues.append(
                {
                    "chain": r["chain"],
                    "res_num": r["res_num"],
                    "res_name": r["res_name"],
                    "charge": round(r["charge"], 3),
                    "pka": r["pka"],
                }
            )

        # Format clusters for output
        output_clusters = []
        for cluster in clusters:
            cluster_sum = sum(r["charge"] for r in cluster)
            output_clusters.append(
                {
                    "residues": [
                        {"res_name": r["res_name"], "res_num": r["res_num"], "charge": round(r["charge"], 3)}
                        for r in cluster
                    ],
                    "total_charge": round(cluster_sum, 3),
                    "count": len(cluster),
                }
            )

        # Generate narrative
        lines = []
        lines.append(f"Charge distribution analysis for {pdb_label} chain {chain_id} at pH {ph}")

        if residue_range:
            lines.append(f"Analyzing residue range: {residue_range}")

        lines.append(f"\nTotal charged residues: {len(charged_residues)}")
        lines.append(f"Net charge: {total_charge:.2f} (positive: {total_positive:.2f}, negative: {total_negative:.2f})")

        # Breakdown by residue type
        residue_type_counts = {}
        for r in charged_residues:
            if r["res_name"] not in ("NTERM", "CTERM"):
                residue_type_counts[r["res_name"]] = residue_type_counts.get(r["res_name"], 0) + 1

        if residue_type_counts:
            type_lines = []
            for res_type in sorted(residue_type_counts.keys()):
                count = residue_type_counts[res_type]
                pka = PKA_VALUES.get(res_type)
                type_lines.append(f"{res_type}(pKa={pka}): {count}")
            lines.append(f"Residue breakdown: {', '.join(type_lines)}")

        # Charge clusters
        if clusters:
            lines.append(f"\nCharge clusters found: {len(clusters)} (3+ residues within 8 A)")
            for i, cluster in enumerate(output_clusters, 1):
                res_strs = [f"{r['res_name']}-{r['res_num']}" for r in cluster["residues"]]
                lines.append(f"  Cluster {i}: {', '.join(res_strs)} (net charge: {cluster['total_charge']:+.2f})")
        else:
            lines.append("\nNo charge clusters found (3+ charged residues within 8 A)")

        data = "\n".join(lines)

        raw = {
            "pdb_id": pdb_label,
            "chain_id": chain_id,
            "ph": ph,
            "charged_residues": output_residues,
            "total_charge": round(total_charge, 3),
            "total_positive": round(total_positive, 3),
            "total_negative": round(total_negative, 3),
            "clusters": output_clusters,
            "cluster_count": len(clusters),
            "pka_values": PKA_VALUES,
        }

        return ToolResult(success=True, data=data, raw=raw)

    except ValueError as e:
        return ToolResult(
            success=False,
            data=f"Error computing charge distribution for {pdb_id or pdb_path} chain {chain_id}: {str(e)}",
            raw={},
            error=str(e),
        )
    except Exception as e:
        return ToolResult(success=False, data=f"Unexpected error: {type(e).__name__}: {str(e)}", raw={}, error=str(e))
