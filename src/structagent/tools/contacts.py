"""Tools for analyzing residue contacts and interactions."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import KDTree

from structagent.registry import tool, ToolResult
from structagent.tools.structure_io import get_structure

# Module-level cache for KDTree and aromatic centroids per structure
# Key: (pdb_id or pdb_path), Value: (kdtree, all_atom_coords, all_atom_info, aromatic_centroids)
_CONTACT_CACHE: dict[str, tuple] = {}


class ResidueNotFoundError(Exception):
    def __init__(self, chain_id, residue_number, available_range, suggestions=None):
        self.chain_id = chain_id
        self.residue_number = residue_number
        self.available_range = available_range
        self.suggestions = suggestions or []

        msg = (
            f"Residue {residue_number} not found in chain {chain_id}. "
            f"Chain {chain_id} contains residues {available_range[0]}-{available_range[1]}"
        )

        if suggestions:
            msg += f". Did you mean: {', '.join(str(s) for s in suggestions)}?"

        super().__init__(msg)


# Residue name sets
HYDROPHOBIC_RES = {"ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET", "PRO"}
AROMATIC_RES = {"PHE", "TYR", "TRP"}
CHARGED_POSITIVE = {"ARG", "LYS"}
CHARGED_NEGATIVE = {"ASP", "GLU"}
POLAR_RES = {"SER", "THR", "ASN", "GLN", "HIS", "CYS", "TYR"}

# Salt bridge atom names
SALT_BRIDGE_POSITIVE = {"NH1", "NH2", "NE", "NZ"}  # ARG and LYS sidechain N
SALT_BRIDGE_NEGATIVE = {"OD1", "OD2", "OE1", "OE2"}  # ASP and GLU sidechain O

# Hbond donor/acceptor atoms (excluding backbone which is handled separately)
HBOND_DONOR_SIDECHAIN = {"ND1", "NE2", "OG", "OG1", "OH", "SG", "NE", "NH1", "NH2"}
HBOND_ACCEPTOR_SIDECHAIN = {"OD1", "OD2", "OE1", "OE2", "OG", "OG1", "O", "S"}

# Backbone atom names
BACKBONE_ATOMS = {"N", "CA", "C", "O", "H", "HA", "HA2", "HA3"}

# Heavy elements (exclude H)
HEAVY_ELEMENTS = {
    "C",
    "N",
    "O",
    "S",
    "P",
    "FE",
    "MG",
    "CA",
    "MN",
    "ZN",
    "CU",
    "NI",
    "CO",
    "NA",
    "K",
    "CL",
    "BR",
    "I",
    "F",
    "SE",
    "B",
    "SI",
}

# Sidechain carbon atoms for hydrophobic contact
SIDECHAIN_CARBON = {
    "ALA": {"CB"},
    "VAL": {"CB", "CG1", "CG2"},
    "LEU": {"CB", "CG", "CD1", "CD2"},
    "ILE": {"CB", "CG1", "CG2", "CD1"},
    "PHE": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "MET": {"CB", "CG", "SD", "CE"},
    "PRO": {"CB", "CG", "CD"},
}

# Aromatic ring atoms for centroid calculation
AROMATIC_ATOMS = {
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
}


@dataclass
class ContactInfo:
    residue: str
    chain: str
    res_num: int
    contact_type: str
    distance: float
    atom_pair: str


def _is_heavy_atom(element) -> bool:
    """Check if atom element is heavy (not hydrogen)."""
    elem_str = str(element) if not isinstance(element, str) else element
    return elem_str.upper() not in ("H", "D", "T")


def _is_backbone_atom(atom_name: str) -> bool:
    """Check if atom is a backbone atom."""
    return atom_name in BACKBONE_ATOMS


def _is_hbond_donor(atom_name: str, res_name: str, element: str) -> bool:
    """Check if atom can act as H-bond donor."""
    if element in ("N", "O", "S"):
        # Backbone N is donor
        if atom_name == "N" and res_name not in ("PRO",):
            return True
        # Sidechain donors
        if atom_name in HBOND_DONOR_SIDECHAIN:
            return True
    return False


def _is_hbond_acceptor(atom_name: str, res_name: str, element: str) -> bool:
    """Check if atom can act as H-bond acceptor."""
    if element in ("N", "O", "S"):
        # Backbone O is acceptor
        if atom_name == "O":
            return True
        # Sidechain acceptors
        if atom_name in HBOND_ACCEPTOR_SIDECHAIN:
            return True
    return False


def _get_aromatic_centroid(residue) -> Optional[np.ndarray]:
    """Calculate centroid of aromatic ring atoms in a residue."""
    res_name = residue.name
    if res_name not in AROMATIC_ATOMS:
        return None

    aromatic_atoms = AROMATIC_ATOMS[res_name]
    positions = []

    for atom in residue:
        if atom.name in aromatic_atoms and _is_heavy_atom(atom.element):
            pos = atom.pos
            positions.append(np.array([pos.x, pos.y, pos.z]))

    if len(positions) < 3:
        return None

    return np.mean(positions, axis=0)


def _find_residue(structure, chain_id: str, residue_number: int, insertion_code: str = None) -> tuple:
    """Find a residue robustly, handling PDB numbering edge cases.

    Lookup strategy (in order):
    1. Exact match: chain_id + seqid.num == residue_number + insertion_code
    2. Fuzzy match: if residue_number is close to available numbers, suggest closest

    Returns:
        (residue, chain) tuple
    Raises:
        ResidueNotFoundError with helpful message
    """
    # Find the chain
    model = structure[0]
    chain = None
    for c in model:
        if c.name == chain_id:
            chain = c
            break

    if chain is None:
        available_chains = [c.name for c in model]
        raise ValueError(f"Chain '{chain_id}' not found. Available chains: {available_chains}")

    # Collect all residues with their sequence IDs
    residues_in_chain = []
    for res in chain:
        if res.is_water():
            continue
        residues_in_chain.append(
            {
                "seqid_num": res.seqid.num,
                "insertion_code": res.seqid.icode.strip(),
                "name": res.name,
            }
        )

    if not residues_in_chain:
        raise ResidueNotFoundError(chain_id, residue_number, (0, 0), [])

    # Strategy 1: exact match on seqid.num
    target = None
    for res in chain:
        if res.seqid.num == residue_number:
            if insertion_code is None or res.seqid.icode.strip() == insertion_code:
                if not res.is_water():
                    target = res
                    break

    # Strategy 2: fuzzy match — find closest residue number
    if target is None:
        all_nums = [r["seqid_num"] for r in residues_in_chain]
        if all_nums:
            closest = min(all_nums, key=lambda x: abs(x - residue_number))
            if abs(closest - residue_number) <= 5:
                suggestions = [closest]
            else:
                suggestions = []
            raise ResidueNotFoundError(chain_id, residue_number, (min(all_nums), max(all_nums)), suggestions)

    return target, chain


def _extract_heavy_atoms(
    structure, chain_ids: list[str] = None, include_hetatm: bool = True, include_water: bool = False
) -> tuple:
    """Extract heavy atom coordinates with proper handling of alternate conformations.

    Returns:
        coords: (N, 3) numpy array of coordinates
        atom_info: list of dicts with chain, resname, resnum, atom_name, is_sidechain
    """
    coords = []
    atom_info = []
    model = structure[0]

    for chain in model:
        if chain_ids and chain.name not in chain_ids:
            continue
        for residue in chain:
            # Skip water unless requested
            if residue.is_water() and not include_water:
                continue
            # Skip non-polymer HETATM unless requested
            if not include_hetatm and residue.het_flag == "H":
                continue

            for atom in residue:
                # Skip hydrogens and deuterium
                if atom.element.name in ("H", "D"):
                    continue
                # Handle alternate conformations: use '' (no altloc) or 'A' only
                if atom.altloc not in ("", "\x00", "A"):
                    continue
                # Skip atoms with zero or negative occupancy
                if atom.occ <= 0:
                    continue

                is_sidechain = atom.name not in ("N", "CA", "C", "O", "OXT")
                coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                atom_info.append(
                    {
                        "chain": chain.name,
                        "res_name": residue.name,
                        "res_seqid": residue.seqid.num,
                        "atom_name": atom.name,
                        "element": atom.element.name,
                    }
                )

    return np.array(coords), atom_info


def _classify_contact(
    res1_name: str,
    atom1_name: str,
    atom1_element: str,
    res2_name: str,
    atom2_name: str,
    atom2_element: str,
    distance: float,
    aromatic_centroids: dict,
) -> str:
    """Classify the type of contact between two residues."""

    # Disulfide: CYS SG within 2.5 Å
    if res1_name == "CYS" and res2_name == "CYS":
        if "SG" in (atom1_name, atom2_name) and distance <= 2.5:
            return "disulfide"

    # Salt bridge: charged positive within 4.0 Å of charged negative
    pos_atoms = SALT_BRIDGE_POSITIVE
    neg_atoms = SALT_BRIDGE_NEGATIVE

    res1_pos = res1_name in CHARGED_POSITIVE and atom1_name in pos_atoms
    res1_neg = res1_name in CHARGED_NEGATIVE and atom1_name in neg_atoms
    res2_pos = res2_name in CHARGED_POSITIVE and atom2_name in pos_atoms
    res2_neg = res2_name in CHARGED_NEGATIVE and atom2_name in neg_atoms

    if (res1_pos and res2_neg) or (res1_neg and res2_pos):
        if distance <= 4.0:
            return "salt_bridge"

    # Cation-pi: charged ARG/LYS within 6.0 Å of aromatic centroid
    if res1_name in CHARGED_POSITIVE and res2_name in AROMATIC_RES:
        if distance <= 6.0:
            return "cation_pi"
    if res2_name in CHARGED_POSITIVE and res1_name in AROMATIC_RES:
        if distance <= 6.0:
            return "cation_pi"

    # Hydrogen bond: N/O donor within 3.5 Å of N/O/S acceptor
    donor1 = _is_hbond_donor(atom1_name, res1_name, atom1_element)
    acceptor1 = _is_hbond_acceptor(atom1_name, res1_name, atom1_element)
    donor2 = _is_hbond_donor(atom2_name, res2_name, atom2_element)
    acceptor2 = _is_hbond_acceptor(atom2_name, res2_name, atom2_element)

    if donor1 and acceptor2 and distance <= 3.5:
        return "hydrogen_bond"
    if donor2 and acceptor1 and distance <= 3.5:
        return "hydrogen_bond"

    # Hydrophobic: both hydrophobic residues, sidechain C within 4.0 Å
    if res1_name in HYDROPHOBIC_RES and res2_name in HYDROPHOBIC_RES:
        atom1_sc_c = atom1_name in SIDECHAIN_CARBON.get(res1_name, set())
        atom2_sc_c = atom2_name in SIDECHAIN_CARBON.get(res2_name, set())
        if (atom1_sc_c or atom2_sc_c) and distance <= 4.0:
            return "hydrophobic"

    # Polar: other N/O/S contact within 3.5 Å
    if atom1_element in ("N", "O", "S") and atom2_element in ("N", "O", "S"):
        if distance <= 3.5:
            return "polar"

    # Default: van der Waals
    return "vdw"


def _format_atom_pair(res1_name: str, atom1_name: str, res2_name: str, atom2_name: str) -> str:
    """Format atom pair for narrative."""
    return f"{res1_name}-{atom1_name}...{res2_name}-{atom2_name}"


@tool(
    name="get_residue_contacts",
    toolset="structure",
    description="Find all atoms within a cutoff distance of a target residue's atoms, classify the contact type, and return a detailed report.",
    parameters={
        "pdb_id": {
            "type": "string",
            "description": "4-character PDB identifier (e.g., '1ABC', '6VXX'). Use either this or pdb_path.",
        },
        "pdb_path": {"type": "string", "description": "Path to local PDB file. Use either this or pdb_id."},
        "chain_id": {"type": "string", "description": "Chain identifier (e.g., 'A', 'B')"},
        "residue_number": {"type": "integer", "description": "Residue sequence number in the chain"},
        "cutoff_angstroms": {
            "type": "number",
            "description": "Distance cutoff in Angstroms for contact detection (default: 4.5)",
            "default": 4.5,
        },
    },
)
def get_residue_contacts(
    pdb_id: str = None,
    pdb_path: str = None,
    chain_id: str = None,
    residue_number: int = None,
    cutoff_angstroms: float = 4.5,
) -> ToolResult:
    """
    Find and classify contacts for a specific residue in a PDB structure.

    Uses a KDTree to efficiently find all heavy atoms within the cutoff distance
    of any atom in the target residue. Contacts are classified as:
    - salt_bridge: Charged sidechain N within 4.0 Å of charged sidechain O
    - hydrogen_bond: N/O/S donor within 3.5 Å of acceptor
    - hydrophobic: Both hydrophobic residues, sidechain C within 4.0 Å
    - cation_pi: Charged ARG/LYS within 6.0 Å of aromatic centroid
    - disulfide: CYS SG within 2.5 Å of CYS SG
    - polar: Other N/O/S contacts within 3.5 Å
    - vdw: Everything else

    Parameters
    ----------
    pdb_id : str
        4-character PDB identifier
    chain_id : str
        Chain identifier (e.g., 'A')
    residue_number : int
        Residue sequence number
    cutoff_angstroms : float, optional
        Maximum distance for contact detection (default 4.5)

    Returns
    -------
    ToolResult
        success: bool indicating if the operation succeeded
        data: Human-readable narrative description of contacts
        raw: Dict with 'contacts' list containing contact information
    """
    try:
        structure = get_structure(pdb_id=pdb_id, pdb_path=pdb_path)

        # Find target residue using robust lookup
        try:
            target_residue, target_chain = _find_residue(structure, chain_id, residue_number)
        except ResidueNotFoundError as e:
            return ToolResult(
                success=False,
                data=f"Residue {residue_number} not found in chain {chain_id}. Chain {chain_id} contains residues {e.available_range[0]}-{e.available_range[1]}. If you're looking for a specific residue, check the numbering by calling load_structure first.",
                raw={"error": "residue_not_found", "available_range": list(e.available_range)},
                error=str(e),
            )
        except ValueError as e:
            available_chains = []
            for chain in structure[0]:
                available_chains.append(chain.name)
            return ToolResult(
                success=False,
                data=f"Chain '{chain_id}' not found in structure. Available chains: {available_chains}.",
                raw={"error": "chain_not_found", "available_chains": available_chains},
                error=str(e),
            )

        # Build cache key
        cache_key = pdb_id if pdb_id else (pdb_path if pdb_path else id(structure))
        cache_key = str(cache_key)

        # Check cache for KDTree and aromatic centroids
        if cache_key in _CONTACT_CACHE:
            cached = _CONTACT_CACHE[cache_key]
            kdtree = cached[0]
            all_atom_coords = cached[1]
            all_atom_info = cached[2]
            aromatic_centroids = cached[3]
        else:
            # Extract heavy atoms from all residues for KDTree
            all_atom_coords, all_atom_info = _extract_heavy_atoms(structure)

            if not all_atom_coords.size:
                return ToolResult(
                    success=False, data=f"No heavy atoms found in structure", raw={}, error="No heavy atoms"
                )

            # Build KDTree
            kdtree = KDTree(all_atom_coords)

            # Pre-compute aromatic centroids for cation-pi classification
            aromatic_centroids = {}
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.name in AROMATIC_RES:
                            centroid = _get_aromatic_centroid(residue)
                            if centroid is not None:
                                key = (chain.name, residue.seqid)
                                aromatic_centroids[key] = centroid

            # Cache the computed data
            _CONTACT_CACHE[cache_key] = (kdtree, all_atom_coords, all_atom_info, aromatic_centroids)

        if not all_atom_coords.size:
            return ToolResult(success=False, data=f"No heavy atoms found in structure", raw={}, error="No heavy atoms")

        # Find atoms in target residue for query
        target_atoms = []
        target_coords = []
        for atom in target_residue:
            if not _is_heavy_atom(atom.element):
                continue
            pos = atom.pos
            target_coords.append([pos.x, pos.y, pos.z])
            target_atoms.append({"atom_name": atom.name, "element": str(atom.element)})

        if not target_coords:
            return ToolResult(
                success=False,
                data=f"No heavy atoms found in target residue {residue_number}",
                raw={},
                error="No heavy atoms in target",
            )

        target_coords = np.array(target_coords)

        # Find all atoms within cutoff of ANY atom in target residue
        # We need to query with each target atom
        nearby_indices = set()
        for tc in target_coords:
            indices = kdtree.query_ball_point(tc, cutoff_angstroms)
            nearby_indices.update(indices)

        # Classify contacts
        contacts: list[ContactInfo] = []

        # Group by residue and track min distance per atom pair
        residue_contacts: dict[tuple, dict] = {}

        for idx in nearby_indices:
            info = all_atom_info[idx]
            atom_chain = info["chain"]
            atom_res_name = info["res_name"]
            atom_res_seqid = info["res_seqid"]
            atom_name = info["atom_name"]
            atom_element = info["element"]

            # Skip if same residue
            if atom_chain == chain_id and atom_res_seqid == residue_number:
                continue

            # Get coordinate
            coord = all_atom_coords[idx]

            # Find min distance to any target atom
            min_dist = float("inf")
            min_target_atom = None
            for i, tc in enumerate(target_coords):
                dist = np.linalg.norm(coord - tc)
                if dist < min_dist:
                    min_dist = dist
                    min_target_atom = target_atoms[i]

            if min_target_atom is None:
                continue

            # Classify contact
            target_atom_name = min_target_atom["atom_name"]
            target_element = min_target_atom["element"]
            target_res_name = target_residue.name

            contact_type = _classify_contact(
                target_res_name,
                target_atom_name,
                target_element,
                atom_res_name,
                atom_name,
                atom_element,
                min_dist,
                aromatic_centroids,
            )

            # Create atom pair string
            atom_pair = _format_atom_pair(target_res_name, target_atom_name, atom_res_name, atom_name)

            # Group by residue
            key = (atom_chain, atom_res_name, atom_res_seqid)
            if key not in residue_contacts:
                residue_contacts[key] = {"contact_type": contact_type, "distance": min_dist, "atom_pair": atom_pair}
            else:
                # Take the closest contact for this residue
                if min_dist < residue_contacts[key]["distance"]:
                    residue_contacts[key] = {"contact_type": contact_type, "distance": min_dist, "atom_pair": atom_pair}

        # Build list of contacts sorted by distance
        contact_list = []
        for (chain, res_name, res_seqid), info in residue_contacts.items():
            contact_list.append(
                {
                    "residue": f"{res_name}-{res_seqid}",
                    "chain": chain,
                    "contact_type": info["contact_type"],
                    "distance": round(info["distance"], 2),
                    "atom_pair": info["atom_pair"],
                }
            )

        contact_list.sort(key=lambda x: x["distance"])

        # Generate narrative
        if not contact_list:
            data = f"No contacts found for {target_residue.name}-{residue_number} (chain {chain_id}) within {cutoff_angstroms} Å. This residue may be at the terminus or in a disordered region. Try increasing the cutoff distance."
            raw = {
                "error": "no_contacts",
                "residue": f"{target_residue.name}-{residue_number}",
                "cutoff": cutoff_angstroms,
            }
            return ToolResult(success=True, data=data, raw=raw)
        else:
            # Group by contact type
            type_counts: dict[str, int] = {}
            for c in contact_list:
                ct = c["contact_type"]
                type_counts[ct] = type_counts.get(ct, 0) + 1

            type_descriptions = []
            for ct, count in sorted(type_counts.items()):
                type_descriptions.append(f"{count} {ct.replace('_', ' ')}")

            summary = ", ".join(type_descriptions)

            # Build narrative lines
            lines = [
                f"{target_residue.name}-{residue_number} (chain {chain_id}): {len(contact_list)} contacts found ({summary})"
            ]

            # Add closest contacts
            for c in contact_list[:5]:
                ct = c["contact_type"].replace("_", " ")
                atom_pair = c["atom_pair"]
                dist = c["distance"]
                # Format based on contact type
                if c["contact_type"] == "hydrogen_bond":
                    lines.append(
                        f"  {c['residue']} (chain {c['chain']}): {ct}, {dist} Å ({atom_pair.replace('...', '—')})"
                    )
                elif c["contact_type"] == "salt_bridge":
                    lines.append(
                        f"  {c['residue']} (chain {c['chain']}): {ct}, {dist} Å ({atom_pair.replace('...', '—')})"
                    )
                else:
                    lines.append(f"  {c['residue']} (chain {c['chain']}): {ct}, {dist} Å")

            if len(contact_list) > 5:
                lines.append(f"  ... and {len(contact_list) - 5} more contacts")

            data = "\n".join(lines)

        raw = {"contacts": contact_list}

        return ToolResult(success=True, data=data, raw=raw)

    except ValueError as e:
        return ToolResult(
            success=False,
            data=f"Error finding contacts for chain {chain_id} residue {residue_number}: {str(e)}",
            raw={},
            error=str(e),
        )
    except Exception as e:
        return ToolResult(success=False, data=f"Unexpected error: {type(e).__name__}: {str(e)}", raw={}, error=str(e))
