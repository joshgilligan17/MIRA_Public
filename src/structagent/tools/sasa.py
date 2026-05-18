"""Solvent Accessible Surface Area (SASA) calculation tool."""

import io
import tempfile
import freesasa
from structagent.registry import tool, ToolResult
from structagent.tools.structure_io import get_structure


# Gly-X-Gly max SASA values (in Å²)
GLYCINE_X_GLY_MAX = {
    "ALA": 129,
    "ARG": 274,
    "ASN": 195,
    "ASP": 193,
    "CYS": 167,
    "GLN": 225,
    "GLU": 223,
    "GLY": 104,
    "HIS": 224,
    "ILE": 197,
    "LEU": 201,
    "LYS": 236,
    "MET": 224,
    "PHE": 240,
    "PRO": 159,
    "SER": 155,
    "THR": 172,
    "TRP": 285,
    "TYR": 263,
    "VAL": 174,
}


def classify(relative_percent: float) -> str:
    """Classify residue as buried, partial, or exposed."""
    if relative_percent < 20:
        return "buried"
    elif relative_percent <= 50:
        return "partial"
    else:
        return "exposed"


def parse_residue_range(residue_range: str, chain_residues: list) -> list:
    """Parse residue_range string into list of residues.

    Supports:
    - 'start-end': range from start to end (inclusive)
    - comma-separated list: '1,3,5' or '1, 3, 5'
    - empty/None: all residues
    """
    if not residue_range:
        return chain_residues

    import re

    residue_range = residue_range.strip()

    if "-" in residue_range and "," not in residue_range:
        # Range format: 'start-end'
        parts = residue_range.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid residue range format: '{residue_range}'. Expected 'start-end'.")
        # Extract numeric part only (handles insertion codes like "116A" -> 116)
        start_match = re.match(r"^(\d+)", parts[0].strip())
        end_match = re.match(r"^(\d+)", parts[1].strip())
        if not start_match or not end_match:
            raise ValueError(f"Invalid residue range format: '{residue_range}'. Residue numbers must be numeric.")
        start = int(start_match.group(1))
        end = int(end_match.group(1))
        return [r for r in chain_residues if start <= r.seqid.num <= end]
    else:
        # Comma-separated list
        selected_nums = set()
        for part in residue_range.split(","):
            part = part.strip()
            if part:
                # Extract numeric part only (handles insertion codes)
                match = re.match(r"^(\d+)", part)
                if match:
                    selected_nums.add(int(match.group(1)))
        return [r for r in chain_residues if r.seqid.num in selected_nums]


@tool(
    name="compute_sasa",
    toolset="structure",
    description="Calculate solvent accessible surface area (SASA) for protein residues. "
    "Returns absolute SASA values, relative percent to Gly-X-Gly max, "
    "and classification (buried <20%, partial 20-50%, exposed >50%).",
    parameters={
        "type": "object",
        "properties": {
            "pdb_id": {
                "type": "string",
                "description": "PDB ID code (e.g., '1abc'). Use either this or pdb_path.",
            },
            "pdb_path": {
                "type": "string",
                "description": "Path to local PDB file. Use either this or pdb_id.",
            },
            "chain_id": {
                "type": "string",
                "description": "Chain identifier (e.g., 'A', 'B')",
            },
            "residue_range": {
                "type": "string",
                "description": "Residue range to analyze. Can be 'start-end' (e.g., '1-50') or comma-separated list (e.g., '1,3,5'). If not specified, analyzes all residues.",
            },
        },
        "required": ["chain_id"],
    },
)
def compute_sasa(
    pdb_id: str = None, pdb_path: str = None, chain_id: str = None, residue_range: str = None
) -> ToolResult:
    """Compute SASA for a protein chain.

    Args:
        pdb_id: PDB identifier (e.g., '1abc')
        chain_id: Chain identifier (e.g., 'A')
        residue_range: Optional residue range ('start-end' or comma-separated list)

    Returns:
        ToolResult with narrative description and per-residue SASA data
    """
    try:
        # Get structure using the unified loader
        structure = get_structure(pdb_id=pdb_id, pdb_path=pdb_path)

        # Find the requested chain
        chain = None
        for c in structure[0]:
            if c.name == chain_id:
                chain = c
                break

        if chain is None:
            return ToolResult(
                success=False,
                data=f"Chain '{chain_id}' not found in structure",
                raw={},
            )

        # Get all residues from the chain
        chain_residues = list(chain)

        # Filter by residue range if specified
        selected_residues = parse_residue_range(residue_range, chain_residues)

        if not selected_residues:
            return ToolResult(
                success=True,
                data=f"No residues found in the specified range '{residue_range}'.",
                raw={"residues": []},
            )

        # Convert gemmi structure to PDB string
        pdb_string = structure.make_pdb_string()

        # Write PDB string to a temporary file for freesasa
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            f.write(pdb_string)
            temp_path = f.name

        # Parse PDB file with freesasa, filtering for the requested chain
        freesasa_structure = freesasa.Structure(temp_path)

        # Calculate SASA
        result = freesasa.calc(freesasa_structure)

        # Get per-residue SASA from result
        residue_areas = result.residueAreas()

        # Map (chain_name, residue_number) to SASA data to handle potential
        # residue number collisions across different chains
        residue_sasa_map = {}
        for chain_name, chain_residues in residue_areas.items():
            for res_num_str, ra in chain_residues.items():
                # ra is a ResidueArea object with .total, .residueType, etc.
                # Handle insertion codes like "116A" - extract numeric part
                if isinstance(res_num_str, str):
                    # Strip any insertion code suffix (e.g., "116A" -> 116)
                    import re

                    match = re.match(r"^(\d+)", res_num_str)
                    if match:
                        res_num = int(match.group(1))
                    else:
                        try:
                            res_num = int(res_num_str)
                        except ValueError:
                            continue
                else:
                    res_num = res_num_str
                residue_sasa_map[(chain_name, res_num)] = {
                    "res_name": ra.residueType,
                    "abs_sasa": ra.total,
                }

        # Build results for selected residues
        residues = []
        narrative_parts = []

        for residue in selected_residues:
            resname = residue.name
            if resname not in GLYCINE_X_GLY_MAX:
                continue  # Skip non-standard residues

            res_num = residue.seqid.num

            if (chain_id, res_num) not in residue_sasa_map:
                continue

            sasa_data = residue_sasa_map[(chain_id, res_num)]
            absolute_sasa = sasa_data["abs_sasa"]
            max_sasa = GLYCINE_X_GLY_MAX[resname]
            relative_percent = (absolute_sasa / max_sasa) * 100.0 if max_sasa > 0 else 0.0
            classification = classify(relative_percent)

            residues.append(
                {
                    "resname": resname,
                    "residue_number": res_num,
                    "absolute_sasa": round(absolute_sasa, 2),
                    "relative_sasa_percent": round(relative_percent, 1),
                    "classification": classification,
                }
            )

            narrative_parts.append(
                f"{resname}-{res_num}: {absolute_sasa:.1f} Å² ({relative_percent:.0f}% relative, {classification})"
            )

        if not residues:
            return ToolResult(
                success=True,
                data="No standard amino acid residues found in the specified range.",
                raw={"residues": []},
            )

        # Build summary statistics
        n_buried = sum(1 for r in residues if r["classification"] == "buried")
        n_partial = sum(1 for r in residues if r["classification"] == "partial")
        n_exposed = sum(1 for r in residues if r["classification"] == "exposed")
        mean_relative = sum(r["relative_sasa_percent"] for r in residues) / len(residues)

        # Format narrative with header
        residue_narrative = "\n  ".join(narrative_parts)
        pdb_label = pdb_id if pdb_id else (pdb_path if pdb_path else "unknown")
        narrative = (
            f"SASA for chain {chain_id} in {pdb_label}:\n"
            f"  {residue_narrative}\n"
            f"Summary: {n_buried} buried, {n_partial} partially exposed, {n_exposed} exposed.\n"
            f"Mean relative SASA: {mean_relative:.1f}%"
        )

        return ToolResult(
            success=True,
            data=narrative,
            raw={"residues": residues},
        )

    except Exception as e:
        return ToolResult(
            success=False,
            data=f"Error computing SASA: {type(e).__name__}: {str(e)}",
            raw={},
            error=str(e),
        )
