"""Tools for analyzing Ramachandran plot statistics."""

from typing import Optional
import math

import gemmi

from structagent.registry import tool, ToolResult
from structagent.tools.structure_io import get_structure


# Simplified Ramachandran boundaries (in degrees)
# General (non-Gly, non-Pro) residues
GENERAL_FAVORED_PHI_MIN = -180
GENERAL_FAVORED_PHI_MAX = 180
GENERAL_FAVORED_PSI_MIN = -180
GENERAL_FAVORED_PSI_MAX = 180

# General favored region (approximate alpha-helix and beta-sheet regions)
# Alpha helix: phi ~ -57, psi ~ -47
# Beta sheet: phi ~ -135, psi ~ 135
GENERAL_FAVORED = {
    "alpha": {"phi": (-90, -30), "psi": (-90, -10)},
    "beta": {"phi": (-180, -90), "psi": (90, 180)},
    "left_handed": {"phi": (30, 90), "psi": (-90, 90)},
}

# Allowed region extends ~15-20 degrees beyond favored
GENERAL_ALLOWED_BUFFER = 20

# Glycine boundaries (more permissive)
GLY_FAVORED_BUFFER = 40

# Proline boundaries (restricted phi)
PRO_FAVORED_PHI_MIN = -75
PRO_FAVORED_PHI_MAX = -45
PRO_FAVORED_PSI_MIN = -60
PRO_FAVORED_PSI_MAX = -10


def _is_in_region(phi: float, psi: float, region: dict) -> bool:
    """Check if phi/psi is within a given region definition."""
    phi_min, phi_max = region["phi"]
    psi_min, psi_max = region["psi"]
    return phi_min <= phi <= phi_max and psi_min <= psi <= psi_max


def _is_in_general_favored(phi: float, psi: float) -> bool:
    """Check if phi/psi is in the general favored region."""
    for region in GENERAL_FAVORED.values():
        if _is_in_region(phi, psi, region):
            return True
    return False


def _is_in_general_allowed(phi: float, psi: float) -> bool:
    """Check if phi/psi is in the general allowed region (favored + buffer)."""
    if _is_in_general_favored(phi, psi):
        return True

    # Check extended regions with buffer
    for region_name, region in GENERAL_FAVORED.items():
        phi_min, phi_max = region["phi"]
        psi_min, psi_max = region["psi"]

        # Apply buffer
        extended_phi_min = phi_min - GENERAL_ALLOWED_BUFFER
        extended_phi_max = phi_max + GENERAL_ALLOWED_BUFFER
        extended_psi_min = psi_min - GENERAL_ALLOWED_BUFFER
        extended_psi_max = psi_max + GENERAL_ALLOWED_BUFFER

        if extended_phi_min <= phi <= extended_phi_max and extended_psi_min <= psi <= extended_psi_max:
            return True

    return False


def _classify_general(phi: float, psi: float) -> str:
    """Classify a general (non-Gly, non-Pro) residue."""
    if _is_in_general_favored(phi, psi):
        return "favored"
    elif _is_in_general_allowed(phi, psi):
        return "allowed"
    else:
        return "outlier"


def _classify_glycine(phi: float, psi: float) -> str:
    """Classify a glycine residue (more permissive boundaries)."""
    # Glycine can adopt conformations that are forbidden for other residues
    # Extended buffer for favored regions
    for region_name, region in GENERAL_FAVORED.items():
        phi_min, phi_max = region["phi"]
        psi_min, psi_max = region["psi"]

        # Apply larger buffer for glycine
        extended_phi_min = phi_min - GLY_FAVORED_BUFFER
        extended_phi_max = phi_max + GLY_FAVORED_BUFFER
        extended_psi_min = psi_min - GLY_FAVORED_BUFFER
        extended_psi_max = psi_max + GLY_FAVORED_BUFFER

        if extended_phi_min <= phi <= extended_phi_max and extended_psi_min <= psi <= extended_psi_max:
            return "favored"

    # Check allowed (even more permissive)
    if -180 <= phi <= 180 and -180 <= psi <= 180:
        # Most glycine conformations are at least allowed
        return "allowed" if not _is_in_general_allowed(phi, psi) else "favored"

    return "outlier"


def _classify_proline(phi: float, psi: float) -> str:
    """Classify a proline residue (restricted phi)."""
    # Proline has a restricted phi angle (~-60)
    phi_outlier = (
        phi < PRO_FAVORED_PHI_MIN - GENERAL_ALLOWED_BUFFER or phi > PRO_FAVORED_PHI_MAX + GENERAL_ALLOWED_BUFFER
    )

    if phi_outlier:
        return "outlier"

    # Check if psi is in the proline allowed range
    proline_allowed_psi = (
        PRO_FAVORED_PSI_MIN - GENERAL_ALLOWED_BUFFER <= psi <= PRO_FAVORED_PSI_MAX + GENERAL_ALLOWED_BUFFER
    )

    # Favored: phi and psi both in proline favored range
    if PRO_FAVORED_PHI_MIN <= phi <= PRO_FAVORED_PHI_MAX and PRO_FAVORED_PSI_MIN <= psi <= PRO_FAVORED_PSI_MAX:
        return "favored"
    elif proline_allowed_psi:
        return "allowed"
    else:
        return "outlier"


def _classify_residue(res_name: str, phi: float, psi: float) -> str:
    """Classify a residue based on its phi/psi angles."""
    if res_name == "GLY":
        return _classify_glycine(phi, psi)
    elif res_name == "PRO":
        return _classify_proline(phi, psi)
    else:
        return _classify_general(phi, psi)


def _parse_residue_range(residue_range: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    """
    Parse a residue range string like '1-100' into start and end integers.
    Returns (None, None) if residue_range is None or invalid.
    """
    if not residue_range:
        return None, None

    try:
        if "-" in residue_range:
            parts = residue_range.split("-")
            if len(parts) == 2:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
                return start, end
        else:
            # Single residue
            res_num = int(residue_range.strip())
            return res_num, res_num
    except ValueError:
        pass

    return None, None


@tool(
    name="check_ramachandran",
    toolset="structure",
    description="Analyze Ramachandran plot statistics for a protein chain, classifying each residue as favored, allowed, or outlier based on its phi/psi angles. Handles GLY and PRO specially.",
    parameters={
        "pdb_id": {
            "type": "string",
            "description": "4-character PDB identifier (e.g., '1ABC', '6VXX'). Use either this or pdb_path.",
        },
        "pdb_path": {"type": "string", "description": "Path to local PDB file. Use either this or pdb_id."},
        "chain_id": {"type": "string", "description": "Chain identifier (e.g., 'A', 'B')"},
        "residue_range": {
            "type": "string",
            "description": "Optional residue range to analyze (e.g., '1-100' or '50'). If not specified, analyzes the entire chain.",
            "default": None,
        },
    },
)
def check_ramachandran(
    pdb_id: str = None, pdb_path: str = None, chain_id: str = None, residue_range: Optional[str] = None
) -> ToolResult:
    """
    Analyze Ramachandran plot statistics for a protein chain.

    Calculates phi and psi angles for each residue and classifies them into:
    - favored: phi/psi in the most common protein secondary structure regions
    - allowed: phi/psi slightly outside favored but still plausible
    - outlier: phi/psi in sterically disallowed regions

    Special handling:
    - GLY (glycine): More permissive boundaries since it lacks a side chain
    - PRO (proline): Restricted phi angle (~-60) due to its ring structure

    Parameters
    ----------
    pdb_id : str
        4-character PDB identifier
    chain_id : str
        Chain identifier (e.g., 'A')
    residue_range : str, optional
        Optional residue range to analyze (e.g., '1-100' or '50')

    Returns
    -------
    ToolResult
        success: bool indicating if the operation succeeded
        data: Human-readable narrative description of Ramachandran statistics
        raw: Dict with counts, percentages, and per-residue classification
    """
    try:
        structure = get_structure(pdb_id=pdb_id, pdb_path=pdb_path)

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
                success=False, data=f"Chain {chain_id} not found in structure", raw={}, error=f"Chain not found"
            )

        # Parse residue range if provided
        range_start, range_end = _parse_residue_range(residue_range)

        # Collect phi/psi angles for each residue
        residue_data = []
        outliers = []
        stats = {"favored": 0, "allowed": 0, "outlier": 0}
        total = 0

        # Use gemmi to calculate phi/psi angles
        # gemmi provides phi/psi via the Ramachandran plot functionality
        for residue in target_chain:
            # Skip waters and non-standard residues
            if residue.is_water():
                continue
            # Skip ligands (non-polymer entities)
            if hasattr(residue, "entity_type") and str(residue.entity_type) != "EntityType.Polymer":
                continue

            res_name = residue.name
            res_num = residue.seqid.num

            # Skip if outside range
            if range_start is not None and range_end is not None:
                if res_num < range_start or res_num > range_end:
                    continue

            # Get CA, N, C atoms to calculate phi/psi
            n_atom = None
            ca_atom = None
            c_atom = None

            for atom in residue:
                if atom.name == "N":
                    n_atom = atom
                elif atom.name == "CA":
                    ca_atom = atom
                elif atom.name == "C":
                    c_atom = atom

            # Need at least N, CA, C to calculate phi/psi
            if n_atom is None or ca_atom is None or c_atom is None:
                continue

            # For phi: need previous residue's N, C, CA, and current N, CA, C
            # For psi: need current N, CA, C and next residue's N, CA, C
            # We need to look at neighboring residues

            # Get residue list for neighbor lookup
            res_list = list(target_chain)
            res_idx = None
            for i, r in enumerate(res_list):
                if r.seqid == residue.seqid:
                    res_idx = i
                    break

            next_residue = None
            if res_idx is not None and res_idx + 1 < len(res_list):
                next_residue = res_list[res_idx + 1]

            prev_residue = None
            if res_idx is not None and res_idx - 1 >= 0:
                prev_residue = res_list[res_idx - 1]

            # Calculate phi/psi using gemmi's function
            try:
                phi_psi = gemmi.calculate_phi_psi(prev_residue, residue, next_residue)
                if phi_psi is None or len(phi_psi) < 2:
                    continue
                phi, psi = phi_psi[0], phi_psi[1]
                if phi is None or psi is None:
                    continue
                # Convert from radians to degrees (gemmi returns radians)
                phi = math.degrees(phi)
                psi = math.degrees(psi)
            except (ValueError, RuntimeError):
                # Cannot calculate phi/psi for terminal residues without valid neighbors
                continue

            # Classify the residue
            classification = _classify_residue(res_name, phi, psi)
            stats[classification] += 1
            total += 1

            residue_info = {
                "res_num": res_num,
                "res_name": res_name,
                "phi": round(phi, 1),
                "psi": round(psi, 1),
                "classification": classification,
            }
            residue_data.append(residue_info)

            if classification == "outlier":
                outliers.append(residue_info)

        # Generate narrative
        if total == 0:
            pdb_label = pdb_id if pdb_id else (pdb_path if pdb_path else "unknown")
            data = f"No valid residues found in chain {chain_id} of {pdb_label}"
            if residue_range:
                data += f" for the specified range {residue_range}"
            return ToolResult(success=False, data=data, raw={}, error="No valid residues")

        # Calculate percentages
        favored_pct = (stats["favored"] / total) * 100 if total > 0 else 0
        allowed_pct = (stats["allowed"] / total) * 100 if total > 0 else 0
        outlier_pct = (stats["outlier"] / total) * 100 if total > 0 else 0

        # Build narrative
        range_str = f" residues {range_start}-{range_end}" if range_start is not None and range_end is not None else ""
        pdb_label = pdb_id if pdb_id else (pdb_path if pdb_path else "unknown")
        lines = [
            f"Ramachandran analysis for {pdb_label} chain {chain_id}{range_str}:",
            f"Total residues analyzed: {total}",
            f"Favored: {stats['favored']} ({favored_pct:.1f}%)",
            f"Allowed: {stats['allowed']} ({allowed_pct:.1f}%)",
            f"Outliers: {stats['outlier']} ({outlier_pct:.1f}%)",
        ]

        # List outliers
        if outliers:
            lines.append("")
            lines.append("Outlier residues:")
            for out in outliers:
                lines.append(f"  {out['res_name']}-{out['res_num']}: phi={out['phi']:.1f}, psi={out['psi']:.1f}")

        data = "\n".join(lines)

        raw = {
            "pdb_id": pdb_id.upper() if pdb_id else (pdb_path if pdb_path else "unknown"),
            "chain_id": chain_id,
            "total": total,
            "favored": stats["favored"],
            "allowed": stats["allowed"],
            "outlier": stats["outlier"],
            "favored_pct": round(favored_pct, 1),
            "allowed_pct": round(allowed_pct, 1),
            "outlier_pct": round(outlier_pct, 1),
            "residues": residue_data,
            "outliers": outliers,
        }

        return ToolResult(success=True, data=data, raw=raw)

    except ValueError as e:
        return ToolResult(
            success=False, data=f"Error analyzing Ramachandran for chain {chain_id}: {str(e)}", raw={}, error=str(e)
        )
    except Exception as e:
        return ToolResult(success=False, data=f"Unexpected error: {type(e).__name__}: {str(e)}", raw={}, error=str(e))
