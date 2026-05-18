"""Tools for loading and working with structural data from the PDB."""

import os
import gemmi
import requests
from pathlib import Path

from structagent.registry import tool, ToolResult


# Module-level cache for parsed structures (LRU cache with maxsize=20)
_STRUCTURE_CACHE: dict[str, gemmi.Structure] = {}


def get_cached_structure(pdb_id: str) -> gemmi.Structure:
    """
    Download and parse a structure from the PDB, caching the result.

    Downloads mmCIF from https://files.rcsb.org/download/{pdb_id}.cif,
    saves it to ~/.cache/structagent/structures/{pdb_id}.cif, and parses
    it with gemmi.

    Args:
        pdb_id: 4-character PDB identifier (e.g., '1ABC')

    Returns:
        Parsed gemmi.Structure object

    Raises:
        ValueError: If download or parsing fails
    """
    pdb_id = pdb_id.upper()

    # Check cache first
    if pdb_id in _STRUCTURE_CACHE:
        return _STRUCTURE_CACHE[pdb_id]

    # Set up cache directory
    cache_dir = Path.home() / ".cache" / "structagent" / "structures"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cif_path = cache_dir / f"{pdb_id}.cif"

    # Download mmCIF file
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        cif_content = response.text
    except requests.RequestException as e:
        raise ValueError(f"Failed to download {pdb_id} from {url}: {e}")

    # Save to cache
    try:
        cif_path.write_text(cif_content)
    except OSError as e:
        raise ValueError(f"Failed to save {pdb_id} to cache at {cif_path}: {e}")

    # Parse structure
    try:
        structure = gemmi.read_structure(str(cif_path))
    except Exception as e:
        raise ValueError(f"Failed to parse {pdb_id} structure: {e}")

    # Cache and return
    # Implement simple LRU eviction if cache exceeds 200 entries
    if len(_STRUCTURE_CACHE) >= 200:
        # Remove oldest entry (first key in dict, FIFO)
        _STRUCTURE_CACHE.pop(next(iter(_STRUCTURE_CACHE)))

    _STRUCTURE_CACHE[pdb_id] = structure
    return structure


def get_structure_file_path(pdb_id: str) -> Path:
    """Get the cached file path for a PDB ID without re-downloading.

    Args:
        pdb_id: 4-character PDB identifier

    Returns:
        Path to the cached CIF file
    """
    pdb_id = pdb_id.upper()
    cache_dir = Path.home() / ".cache" / "structagent" / "structures"
    return cache_dir / f"{pdb_id}.cif"


def get_structure(pdb_id: str = None, pdb_path: str = None) -> gemmi.Structure:
    """Get a gemmi Structure from either RCSB (by PDB ID) or local file.

    Args:
        pdb_id: 4-character RCSB PDB identifier (e.g., '1UBQ')
        pdb_path: Path to local PDB/mmCIF file

    Returns:
        gemmi.Structure object

    Priority:
        1. If pdb_path is a valid file, use it directly
        2. Otherwise use pdb_id to download from RCSB
    """
    if pdb_path and os.path.isfile(pdb_path):
        return gemmi.read_structure(pdb_path)

    if pdb_id:
        return get_cached_structure(pdb_id)

    raise ValueError("Must provide either pdb_id or pdb_path (valid file)")


@tool(
    name="list_residues",
    toolset="structure",
    description="List all residues in a chain with their sequence numbers, names, and any insertion codes. Use this to check residue numbering before calling other residue-specific tools. Also reports gaps in numbering and modified residues.",
    parameters={
        "type": "object",
        "properties": {
            "pdb_id": {
                "type": "string",
                "description": "4-character PDB identifier (e.g., '1ABC', '6VXX'). Use either this or pdb_path.",
            },
            "pdb_path": {"type": "string", "description": "Path to local PDB file. Use either this or pdb_id."},
            "chain_id": {"type": "string", "description": "Chain identifier (e.g., 'A', 'B')"},
        },
        "required": ["chain_id"],
    },
)
def list_residues(pdb_id: str = None, pdb_path: str = None, chain_id: str = None) -> ToolResult:
    """List all residues in a chain with their numbering.

    Returns a detailed report including:
    - Total residue count
    - Numbering range
    - Insertion code residues
    - Gaps in numbering
    - Modified residues (HETATM)
    - First 10 and last 5 residues for quick reference
    """
    try:
        structure = get_structure(pdb_id=pdb_id, pdb_path=pdb_path)

        # Determine the identifier for reporting
        if pdb_path and os.path.isfile(pdb_path):
            pdb_id_upper = os.path.basename(pdb_path)
        else:
            pdb_id_upper = pdb_id.upper() if pdb_id else "unknown"

        # Find the requested chain
        chain = None
        for ch in structure[0]:  # First model
            if ch.name == chain_id:
                chain = ch
                break

        if chain is None:
            return ToolResult(
                success=False,
                data=f"Chain '{chain_id}' not found in structure {pdb_id_upper}",
                raw={},
                error=f"Chain '{chain_id}' not found",
            )

        # Collect all residue information
        residues = []
        insertion_code_residues = []
        modified_residues = []
        gaps = []
        prev_seqnum = None

        for res in chain:
            seqnum = res.seqid.num
            icode = res.seqid.icode.strip() if res.seqid.icode else ""
            resname = res.name

            # Build residue identifier (e.g., "ALA-52A")
            res_id = f"{resname}-{seqnum}{icode}" if icode else f"{resname}-{seqnum}"
            residues.append(
                {
                    "seqnum": seqnum,
                    "icode": icode,
                    "resname": resname,
                    "res_id": res_id,
                    "is_modified": res.het_flag == "H",
                }
            )

            # Check for insertion code (only non-empty after stripping)
            if icode:
                insertion_code_residues.append({"seqnum": seqnum, "icode": icode, "resname": resname, "res_id": res_id})

            # Check for modified residue
            if res.het_flag == "H":
                modified_residues.append({"seqnum": seqnum, "resname": resname, "res_id": res_id})

            # Check for gaps in numbering
            if prev_seqnum is not None:
                if seqnum - prev_seqnum > 1:
                    gaps.append((prev_seqnum + 1, seqnum - 1))
            prev_seqnum = seqnum

        # Calculate statistics
        total_residues = len(residues)
        seqnums = [r["seqnum"] for r in residues]
        min_seqnum = min(seqnums) if seqnums else 0
        max_seqnum = max(seqnums) if seqnums else 0

        # Build the narrative string
        data_parts = [f"Residues in chain {chain_id} of {pdb_id_upper}:"]

        # Total count
        if insertion_code_residues:
            data_parts.append(
                f"  Total: {total_residues} residues (including {len(insertion_code_residues)} with insertion codes)"
            )
        else:
            data_parts.append(f"  Total: {total_residues} residues")

        # Range
        if gaps:
            gap_strs = [f"{g[0]}-{g[1]}" for g in gaps]
            gaps_str = ", ".join(gap_strs)
            if insertion_code_residues:
                icode_strs = [r["res_id"] for r in insertion_code_residues]
                icodes_str = ", ".join(icode_strs)
                data_parts.append(
                    f"  Range: {min_seqnum}-{max_seqnum} (with gaps at {gaps_str}, insertion codes at {icodes_str})"
                )
            else:
                data_parts.append(f"  Range: {min_seqnum}-{max_seqnum} (with gaps at {gaps_str})")
        else:
            if insertion_code_residues:
                icode_strs = [r["res_id"] for r in insertion_code_residues]
                icodes_str = ", ".join(icode_strs)
                data_parts.append(f"  Range: {min_seqnum}-{max_seqnum} (insertion codes at {icodes_str})")
            else:
                data_parts.append(f"  Range: {min_seqnum}-{max_seqnum}")

        # First 10 residues
        first_10 = residues[:10]
        first_10_strs = [r["res_id"] for r in first_10]
        data_parts.append(f"  First 10: {', '.join(first_10_strs)}")

        # Last 5 residues
        last_5 = residues[-5:]
        last_5_strs = [r["res_id"] for r in last_5]
        data_parts.append(f"  Last 5: {', '.join(last_5_strs)}")

        # Insertion code residues detail
        if insertion_code_residues:
            data_parts.append("  Insertion code residues:")
            for r in insertion_code_residues:
                data_parts.append(f"    {r['res_id']}")

        # Gaps detail
        if gaps:
            gap_strs = [f"{g[0]}-{g[1]}" for g in gaps]
            data_parts.append(f"  Gaps in numbering: {', '.join(gap_strs)} (disordered loop, not resolved)")
        else:
            data_parts.append("  Gaps in numbering: none")

        # Modified residues
        if modified_residues:
            mod_strs = [r["res_id"] for r in modified_residues]
            data_parts.append(f"  Modified residues: {', '.join(mod_strs)}")
        else:
            data_parts.append("  Modified residues: none")

        data = "\n".join(data_parts)

        # Build raw data
        raw = {
            "pdb_id": pdb_id_upper,
            "chain_id": chain_id,
            "total_residues": total_residues,
            "range": (min_seqnum, max_seqnum),
            "insertion_code_residues": insertion_code_residues,
            "gaps": gaps,
            "modified_residues": modified_residues,
            "first_10": first_10_strs,
            "last_5": last_5_strs,
        }

        return ToolResult(success=True, data=data, raw=raw)

    except ValueError as e:
        return ToolResult(success=False, data=f"Failed to list residues: {str(e)}", raw={}, error=str(e))
    except Exception as e:
        return ToolResult(
            success=False, data=f"Unexpected error listing residues: {type(e).__name__}: {str(e)}", raw={}, error=str(e)
        )


@tool(
    name="load_structure",
    toolset="structure",
    description="Load a protein structure from the PDB by ID or from a local file. Returns structure details including resolution, method, chains, and ligands.",
    parameters={
        "type": "object",
        "properties": {
            "pdb_id": {
                "type": "string",
                "description": "4-character PDB identifier (e.g., '1ABC', '6VXX'). Use either this or pdb_path.",
            },
            "pdb_path": {"type": "string", "description": "Path to local PDB/mmCIF file. Use either this or pdb_id."},
        },
        "required": [],
    },
)
def load_structure(pdb_id: str = None, pdb_path: str = None) -> ToolResult:
    """
    Load a structure from the PDB and return a detailed summary.

    This tool fetches the mmCIF file from RCSB PDB, parses it, and returns
    a comprehensive summary of the structure suitable for a structural
    biologist's lab notebook.

    Args:
        pdb_id: 4-character PDB identifier

    Returns:
        ToolResult with:
        - success: bool indicating if load succeeded
        - data: human-readable narrative description
        - raw: dict with pdb_id, resolution, method, space_group, chains,
                ligands, ligand_count, assembly_count
    """
    try:
        structure = get_structure(pdb_id=pdb_id, pdb_path=pdb_path)

        # Determine the identifier for reporting
        if pdb_path and os.path.isfile(pdb_path):
            pdb_id_upper = os.path.basename(pdb_path)
            cif_path = pdb_path
            file_format = Path(pdb_path).suffix.lstrip(".").upper() or "structure"
        else:
            pdb_id_upper = pdb_id.upper() if pdb_id else "unknown"
            cif_path = get_structure_file_path(pdb_id) if pdb_id else None
            file_format = "mmCIF"
        resolution = structure.resolution if structure.resolution else None
        method = "Unknown"
        space_group = structure.spacegroup_hm if structure.spacegroup_hm else "Unknown"

        # Get chains information
        chains = []
        for chain in structure[0]:  # First model
            chain_id = chain.name
            # Get sequence length
            seq_length = sum(1 for _ in chain)
            # Get first and last residue numbers
            first_res = None
            last_res = None
            res_nums = []
            for residue in chain:
                res_nums.append(residue.seqid.num)
            if res_nums:
                first_res = min(res_nums)
                last_res = max(res_nums)
            chains.append({"id": chain_id, "length": seq_length, "first_residue": first_res, "last_residue": last_res})

        # Get ligands (non-standard residues / HETATM entries)
        ligands = []
        for chain in structure[0]:
            for residue in chain:
                # Check if it's a HET residue (ligand)
                if residue.het_flag not in (None, "."):
                    ligands.append({"chain": chain.name, "resname": residue.name, "seqid": residue.seqid})

        # Get assembly information
        assembly_count = 0
        if structure.assemblies:
            assembly_count = len(structure.assemblies)

        ligand_count = len(ligands)

        # Build human-readable narrative
        method_lower = method.lower() if method != "Unknown" else "unknown method"
        res_str = f"{resolution:.2f} Angstrom" if resolution else "resolution not available"

        chain_descriptions = []
        for ch in chains:
            if ch["first_residue"] and ch["last_residue"]:
                chain_descriptions.append(
                    f"Chain {ch['id']}: {ch['length']} residues (residues {ch['first_residue']}-{ch['last_residue']})"
                )
            else:
                chain_descriptions.append(f"Chain {ch['id']}: {ch['length']} residues")

        chain_text = "; ".join(chain_descriptions) if chain_descriptions else "No chains found"

        ligand_descriptions = []
        seen_ligands = {}
        for lig in ligands:
            key = (lig["chain"], lig["resname"])
            if key not in seen_ligands:
                seen_ligands[key] = lig
                ligand_descriptions.append(f"{lig['resname']} in chain {lig['chain']} at position {lig['seqid']}")

        ligand_text = ", ".join(ligand_descriptions) if ligand_descriptions else "No ligands bound"

        data_lines = [
            f"Structure {pdb_id_upper} was solved by {method_lower} at {res_str}.",
            f"The crystal belongs to space group {space_group}.",
            f"The asymmetric unit contains {len(chains)} chain{'s' if len(chains) != 1 else ''}: {chain_text}.",
            f"Bound ligands include: {ligand_text}.",
            f"There {('are ' + str(assembly_count) + ' biological assembly' + ('s' if assembly_count != 1 else '') + ' available') if assembly_count else 'is no biological assembly information available'}.",
            f"The structure file is cached at: {cif_path} (use this path for file-based tools like fast_relax, score_interface, renumber_pdb, analyze_interface_energies).",
        ]

        data = " ".join(data_lines)

        raw = {
            "pdb_id": pdb_id_upper,
            "resolution": resolution,
            "method": method,
            "space_group": space_group,
            "chains": chains,
            "ligands": ligands,
            "ligand_count": ligand_count,
            "assembly_count": assembly_count,
            "file_path": str(cif_path),
            "file_format": file_format,
        }

        return ToolResult(success=True, data=data, raw=raw)

    except ValueError as e:
        return ToolResult(success=False, data=f"Failed to load structure: {str(e)}", raw={}, error=str(e))
    except Exception as e:
        return ToolResult(
            success=False,
            data=f"Unexpected error loading structure: {type(e).__name__}: {str(e)}",
            raw={},
            error=str(e),
        )
