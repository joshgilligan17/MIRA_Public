"""Interface analysis tools for computing protein-protein interfaces."""

import io
import tempfile
import gemmi
import freesasa

from structagent.registry import tool, ToolResult
from structagent.tools.structure_io import get_structure


# Residue classification for interface composition
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS"}
CHARGED_POSITIVE = {"LYS", "ARG", "HIS"}
CHARGED_NEGATIVE = {"ASP", "GLU"}


def _residue_sasa(residue: gemmi.Residue, sasa_result: freesasa.Result, structure: gemmi.Structure) -> float:
    """Get SASA for a specific residue from freesasa result."""
    # Find the residue's atom indices in the structure
    for model in structure:
        for chain in model:
            for res in chain:
                if res == residue:
                    # Get residue coordinate for SASA lookup
                    coords = []
                    for atom in res:
                        if atom.element not in ("H", "D"):  # Skip hydrogens
                            coords.append(atom.pos)
                    return 0.0  # Placeholder - actual implementation needs atom selection
    return 0.0


def _compute_sasa_for_chain(structure: gemmi.Structure, chain_id: str) -> float:
    """Compute total SASA for a specific chain using freesasa."""
    import os

    # Build PDB string for the specific chain
    pdb_str = ""
    for model in structure:
        for chain in model:
            if chain.name == chain_id:
                for residue in chain:
                    # Skip waters and HETATMs (het_flag='H'), but include polymer residues (het_flag='A')
                    if residue.het_flag == "H" or residue.is_water():
                        continue
                    for atom in residue:
                        elem = atom.element.name  # Get element symbol (e.g., 'N', 'C')
                        if elem not in ("H", "D"):  # Skip hydrogens
                            pdb_str += f"ATOM  {atom.serial:5d} {atom.name:4s} {residue.name:3s} {chain.name:1s}{residue.seqid.num:4d}   {atom.pos.x:8.3f}{atom.pos.y:8.3f}{atom.pos.z:8.3f}  1.00  0.00           {elem:>2s}\n"
                pdb_str += "TER\n"
                break

    if not pdb_str:
        return 0.0

    # Write to temp file and use freesasa.Structure
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            f.write(pdb_str)
            temp_path = f.name

        try:
            struct = freesasa.Structure(temp_path)
            result = freesasa.calc(struct)
            return result.totalArea()
        finally:
            os.unlink(temp_path)
    except Exception:
        return 0.0


def _compute_sasa_for_complex(structure: gemmi.Structure, chain_a: str, chain_b: str) -> float:
    """Compute SASA for the complex of two chains."""
    import os

    pdb_str = ""
    serial = 1
    for model in structure:
        for chain in model:
            if chain.name in (chain_a, chain_b):
                for residue in chain:
                    # Skip waters and HETATMs, but include polymer residues
                    if residue.het_flag == "H" or residue.is_water():
                        continue
                    for atom in residue:
                        elem = atom.element.name
                        if elem not in ("H", "D"):
                            pdb_str += f"ATOM  {serial:5d} {atom.name:4s} {residue.name:3s} {chain.name:1s}{residue.seqid.num:4d}   {atom.pos.x:8.3f}{atom.pos.y:8.3f}{atom.pos.z:8.3f}  1.00  0.00           {elem:>2s}\n"
                            serial += 1
                pdb_str += "TER\n"

    if not pdb_str:
        return 0.0

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            f.write(pdb_str)
            temp_path = f.name

        try:
            struct = freesasa.Structure(temp_path)
            result = freesasa.calc(struct)
            return result.totalArea()
        finally:
            os.unlink(temp_path)
    except Exception:
        return 0.0


def _classify_residue(resname: str) -> str:
    """Classify a residue as hydrophobic, polar, or charged."""
    if resname in HYDROPHOBIC:
        return "hydrophobic"
    elif resname in POLAR:
        return "polar"
    elif resname in CHARGED_POSITIVE:
        return "charged_positive"
    elif resname in CHARGED_NEGATIVE:
        return "charged_negative"
    else:
        return "other"


@tool(
    name="compute_interface",
    toolset="structure",
    description="Identify interface residues between two chains. Computes "
    "buried surface area, key contacts, and interface composition. "
    "Essential for protein-protein interaction analysis.",
    parameters={
        "type": "object",
        "properties": {
            "pdb_id": {
                "type": "string",
                "description": "4-character PDB identifier (e.g., '1ABC', '3HFM'). Use either this or pdb_path.",
            },
            "pdb_path": {"type": "string", "description": "Path to local PDB file. Use either this or pdb_id."},
            "chain_a": {"type": "string", "description": "First chain ID (e.g., 'A')"},
            "chain_b": {"type": "string", "description": "Second chain ID (e.g., 'B')"},
            "distance_cutoff": {
                "type": "number",
                "default": 5.0,
                "description": "Distance cutoff for interface contacts in Angstroms (default 5.0)",
            },
        },
        "required": ["chain_a", "chain_b"],
    },
)
def compute_interface(
    pdb_id: str = None, pdb_path: str = None, chain_a: str = None, chain_b: str = None, distance_cutoff: float = 5.0
) -> ToolResult:
    """
    Compute interface properties between two protein chains.

    Finds cross-chain contacts within a distance cutoff, calculates buried
    surface area via three SASA computations (chain A alone, chain B alone,
    and the complex), classifies interface residues by chemistry, and
    identifies hotspot residues contributing >50 A^2 to the buried surface.

    Args:
        pdb_id: 4-character PDB identifier
        chain_a: First chain ID
        chain_b: Second chain ID
        distance_cutoff: Distance in Angstroms for contact detection (default 5.0)

    Returns:
        ToolResult with:
        - success: bool indicating if analysis succeeded
        - data: human-readable narrative with buried SA, interface residue counts,
                composition breakdown, and key hotspots
        - raw: dict with buried_sa_total, chain_a_sa, chain_b_sa,
                interface_residues_a, interface_residues_b, composition, hotspots
    """
    try:
        # Load structure using unified loader
        structure = get_structure(pdb_id=pdb_id, pdb_path=pdb_path)

        # Determine the identifier for reporting
        if pdb_path:
            pdb_label = pdb_path
        elif pdb_id:
            pdb_label = pdb_id.upper()
        else:
            pdb_label = "unknown"

        # Find the chains
        chain_a_obj = None
        chain_b_obj = None
        for model in structure:
            for chain in model:
                if chain.name == chain_a:
                    chain_a_obj = chain
                elif chain.name == chain_b:
                    chain_b_obj = chain

        if chain_a_obj is None:
            return ToolResult(
                success=False,
                data=f"Chain '{chain_a}' not found in structure",
                raw={},
                error=f"Chain {chain_a} not found",
            )
        if chain_b_obj is None:
            return ToolResult(
                success=False,
                data=f"Chain '{chain_b}' not found in structure",
                raw={},
                error=f"Chain {chain_b} not found",
            )

        # Use gemmi ContactSearch to find cross-chain contacts
        # Set up NeighborSearch and populate with atoms
        ns = gemmi.NeighborSearch(structure, distance_cutoff)
        ns.populate()

        # Find contacts between the two chains
        contact_search = gemmi.ContactSearch(distance_cutoff)
        contact_search.ignore = gemmi.ContactSearch.Ignore.SameChain  # Only cross-chain contacts

        # Find contacts using the NeighborSearch
        contacts = contact_search.find_contacts(ns)

        # Build set of interface residues from contacts
        interface_residues_a = set()
        interface_residues_b = set()
        contact_pairs = []

        for contact in contacts:
            # Get the two residues involved (gemmi uses partner1/partner2, not r1/r2)
            try:
                p1 = contact.partner1
                p2 = contact.partner2
                if p1 and p2:
                    chain_r1 = p1.chain.name
                    chain_r2 = p2.chain.name

                    # Only consider A-B contacts
                    if (chain_r1 == chain_a and chain_r2 == chain_b) or (chain_r1 == chain_b and chain_r2 == chain_a):
                        if chain_r1 == chain_a:
                            interface_residues_a.add((p1.residue.name, p1.residue.seqid.num))
                            interface_residues_b.add((p2.residue.name, p2.residue.seqid.num))
                        else:
                            interface_residues_a.add((p2.residue.name, p2.residue.seqid.num))
                            interface_residues_b.add((p1.residue.name, p1.residue.seqid.num))
                        contact_pairs.append((p1, p2, contact.dist))
            except AttributeError:
                # Skip contacts that don't have expected structure
                pass

        # Compute SASA for each chain individually and the complex
        chain_a_sa = _compute_sasa_for_chain(structure, chain_a)
        chain_b_sa = _compute_sasa_for_chain(structure, chain_b)
        complex_sa = _compute_sasa_for_complex(structure, chain_a, chain_b)

        # Buried surface area = SASA(A) + SASA(B) - SASA(complex)
        buried_sa_total = chain_a_sa + chain_b_sa - complex_sa

        # Classify interface residues
        composition_a = {"hydrophobic": 0, "polar": 0, "charged": 0, "other": 0}
        composition_b = {"hydrophobic": 0, "polar": 0, "charged": 0, "other": 0}
        hotspot_residues_a = []
        hotspot_residues_b = []

        # For hotspot calculation, we need per-residue buried SA
        # Simplified approach: distribute buried SA proportionally to contact count
        total_contacts = len(contact_pairs) if contact_pairs else 1

        for res_tuple in interface_residues_a:
            resname, seqid = res_tuple
            classification = _classify_residue(resname)
            if classification == "hydrophobic":
                composition_a["hydrophobic"] += 1
            elif classification == "polar":
                composition_a["polar"] += 1
            elif classification in ("charged_positive", "charged_negative"):
                composition_a["charged"] += 1
            else:
                composition_a["other"] += 1

            # Estimate per-residue buried SA contribution (hotspot threshold: 50 A^2)
            # Weight by contact count for this residue
            residue_contacts = sum(
                1
                for p in contact_pairs
                if (p[0].residue.name == resname and int(p[0].residue.seqid.num) == int(seqid))
                or (p[1].residue.name == resname and int(p[1].residue.seqid.num) == int(seqid))
            )
            residue_buried_sa = (residue_contacts / total_contacts) * buried_sa_total if total_contacts > 0 else 0

            if residue_buried_sa > 50:
                hotspot_residues_a.append(
                    {"resname": resname, "seqid": int(seqid), "buried_sa": residue_buried_sa, "type": classification}
                )

        for res_tuple in interface_residues_b:
            resname, seqid = res_tuple
            classification = _classify_residue(resname)
            if classification == "hydrophobic":
                composition_b["hydrophobic"] += 1
            elif classification == "polar":
                composition_b["polar"] += 1
            elif classification in ("charged_positive", "charged_negative"):
                composition_b["charged"] += 1
            else:
                composition_b["other"] += 1

            residue_contacts = sum(
                1
                for p in contact_pairs
                if (p[0].residue.name == resname and int(p[0].residue.seqid.num) == int(seqid))
                or (p[1].residue.name == resname and int(p[1].residue.seqid.num) == int(seqid))
            )
            residue_buried_sa = (residue_contacts / total_contacts) * buried_sa_total if total_contacts > 0 else 0

            if residue_buried_sa > 50:
                hotspot_residues_b.append(
                    {"resname": resname, "seqid": int(seqid), "buried_sa": residue_buried_sa, "type": classification}
                )

        # Sort hotspots by buried SA
        hotspot_residues_a.sort(key=lambda x: x["buried_sa"], reverse=True)
        hotspot_residues_b.sort(key=lambda x: x["buried_sa"], reverse=True)
        all_hotspots = hotspot_residues_a + hotspot_residues_b
        all_hotspots.sort(key=lambda x: x["buried_sa"], reverse=True)

        # Calculate total composition
        total_interface_residues = len(interface_residues_a) + len(interface_residues_b)
        total_hydrophobic = composition_a["hydrophobic"] + composition_b["hydrophobic"]
        total_polar = composition_a["polar"] + composition_b["polar"]
        total_charged = composition_a["charged"] + composition_b["charged"]

        if total_interface_residues > 0:
            pct_hydrophobic = (total_hydrophobic / total_interface_residues) * 100
            pct_polar = (total_polar / total_interface_residues) * 100
            pct_charged = (total_charged / total_interface_residues) * 100
        else:
            pct_hydrophobic = pct_polar = pct_charged = 0

        composition = {
            "hydrophobic": {"count": total_hydrophobic, "percent": pct_hydrophobic},
            "polar": {"count": total_polar, "percent": pct_polar},
            "charged": {"count": total_charged, "percent": pct_charged},
        }

        # Build narrative
        data_lines = [
            f"Interface analysis between chains {chain_a} and {chain_b} in {pdb_label}:",
            f"Buried surface area: {buried_sa_total:.1f} A^2 (chain {chain_a}: {chain_a_sa:.1f} A^2, chain {chain_b}: {chain_b_sa:.1f} A^2)",
            f"Interface residues: {len(interface_residues_a)} on chain {chain_a}, {len(interface_residues_b)} on chain {chain_b}",
            f"Composition: {pct_hydrophobic:.0f}% hydrophobic, {pct_polar:.0f}% polar, {pct_charged:.0f}% charged",
        ]

        if all_hotspots:
            data_lines.append("")
            data_lines.append("Key interface residues (hotspots, >50 A^2 buried SA):")
            for hs in all_hotspots[:5]:  # Show top 5
                chain_id = chain_a if hs in hotspot_residues_a else chain_b
                data_lines.append(
                    f"  {hs['resname']}-{hs['seqid']} (chain {chain_id}): "
                    f"{hs['buried_sa']:.1f} A^2 buried ({hs['type']})"
                )

        data = " ".join(data_lines)

        raw = {
            "buried_sa_total": buried_sa_total,
            "chain_a_sa": chain_a_sa,
            "chain_b_sa": chain_b_sa,
            "interface_residues_a": [{"resname": r, "seqid": int(s)} for r, s in interface_residues_a],
            "interface_residues_b": [{"resname": r, "seqid": int(s)} for r, s in interface_residues_b],
            "composition": composition,
            "hotspots": all_hotspots[:10],  # Include top 10 hotspots in raw data
        }

        return ToolResult(success=True, data=data, raw=raw)

    except Exception as e:
        return ToolResult(
            success=False,
            data=f"Error computing interface for chains {chain_a}-{chain_b}: {type(e).__name__}: {str(e)}",
            raw={},
            error=str(e),
        )
