import os
import shutil
import tempfile
from typing import Optional

import gemmi

from structagent.registry import ToolResult, tool
from structagent.tools.structure_io import get_structure


def _dssp_to_secondary(dssp_code: str) -> str:
    """Map DSSP code to secondary structure type.

    H: alpha helix
    G: 3-helix (glycine)
    I: 5-helix (pi helix)
    E: beta strand
    B: beta bridge
    T: turn
    S: bend
    - or other: coil
    """
    if dssp_code in ("H", "G", "I"):
        return "helix"
    elif dssp_code in ("E", "B"):
        return "strand"
    else:
        return "coil"


def _format_element(element_type: str, start_res: str, end_res: str, count: int, element_num: int) -> str:
    """Format a secondary structure element as a lab notebook narrative."""
    if element_type == "helix":
        return f"Helix {element_num}: {start_res} to {end_res}, {count} residues"
    elif element_type == "strand":
        return f"β-strand {element_num}: {start_res} to {end_res}, {count} residues"
    else:
        return f"Coil {element_num}: {start_res} to {end_res}, {count} residues"


@tool(
    name="get_secondary_structure",
    toolset="structure",
    description="Assign secondary structure (helix, strand, coil) to a protein chain using DSSP. "
    "Returns grouped structural elements with residue ranges.",
    parameters={
        "type": "object",
        "properties": {
            "pdb_id": {"type": "string", "description": "PDB identifier (e.g., '1abc'). Use either this or pdb_path."},
            "pdb_path": {"type": "string", "description": "Path to local PDB file. Use either this or pdb_id."},
            "chain_id": {"type": "string", "description": "Chain identifier (e.g., 'A')"},
            "residue_range": {
                "type": "string",
                "description": "Optional residue range in 'start-end' format (e.g., '1-100')",
                "default": None,
            },
        },
        "required": ["chain_id"],
    },
)
def get_secondary_structure(
    pdb_id: str = None, pdb_path: str = None, chain_id: str = None, residue_range: Optional[str] = None
) -> ToolResult:
    """Get secondary structure assignment for a protein chain.

    Uses BioPython DSSP with the mkdssp binary to compute secondary structure
    assignments, then groups consecutive residues with the same assignment into
    elements (helices, strands, coils).
    """
    try:
        # Load structure using unified loader
        structure = get_structure(pdb_id=pdb_id, pdb_path=pdb_path)
        if structure is None:
            return ToolResult(success=False, data=f"Could not load structure", raw={})

        # Find the specified chain
        chain = None
        for c in structure[0]:
            if c.name == chain_id:
                chain = c
                break

        if chain is None:
            return ToolResult(success=False, data=f"Chain {chain_id} not found in structure", raw={})

        # Check if mkdssp binary is available
        # First try shutil.which, then check common conda/mamba paths
        mkdssp_path = shutil.which("mkdssp")
        if mkdssp_path is None:
            # Check common conda environments
            conda_paths = [
                "/opt/homebrew/Caskroom/miniconda/base/bin/mkdssp",
                "/opt/conda/bin/mkdssp",
                "/usr/local/bin/mkdssp",
            ]
            for path in conda_paths:
                if os.path.isfile(path) and os.access(path, os.X_OK):
                    mkdssp_path = path
                    break

        # Set LIBCIFPP_DATA_DIR if mkdssp was found and env var not already set
        if mkdssp_path and not os.environ.get("LIBCIFPP_DATA_DIR"):
            # Try to find the libcifpp data directory
            possible_libcifpp_dirs = [
                "/opt/homebrew/Caskroom/miniconda/base/share/libcifpp",
                "/opt/conda/share/libcifpp",
            ]
            for d in possible_libcifpp_dirs:
                if os.path.isdir(d):
                    os.environ["LIBCIFPP_DATA_DIR"] = d
                    break

        if mkdssp_path is None:
            return ToolResult(
                success=False,
                data="DSSP requires the external 'mkdssp' binary which is not installed. "
                "Install it with: conda install -c conda-forge dssp (recommended) or "
                "brew install DSSP (macOS) or apt install dssp (Linux). "
                "Then restart the application.",
                raw={"error": "mkdssp_not_found"},
                error="mkdssp binary not found",
            )

        # Use BioPython DSSP with mkdssp
        try:
            from Bio.PDB import PDBParser, DSSP as BioDSSP

            # Write structure to temp file for DSSP
            with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
                f.write(structure.make_pdb_string())
                temp_path = f.name

            try:
                parser = PDBParser(QUIET=True)
                model = parser.get_structure("temp", temp_path)[0]
                dssp_result = BioDSSP(model, temp_path, dssp=mkdssp_path)

                # Build residue -> DSSP code mapping
                # dssp_result is a dict with keys: (chain_id, (space, resnum, icode))
                # and values: (resnum, resname, ss_code, ASA, ...)
                res_dssp = {}
                for key in dssp_result.keys():
                    chain_id_dssp = key[0]
                    if chain_id_dssp != chain_id:
                        continue
                    res_id_tuple = key[1]
                    dssp_data = dssp_result[key]
                    ss_code = dssp_data[2]
                    resname = dssp_data[1]
                    resnum = res_id_tuple[1]
                    inscode = res_id_tuple[2] if res_id_tuple[2] and res_id_tuple[2] != " " else ""
                    label = f"{resname}-{resnum}{inscode}"
                    res_dssp[resnum] = (label, ss_code)
            finally:
                os.unlink(temp_path)

        except ImportError:
            return ToolResult(
                success=False,
                data="BioPython is required for DSSP calculation. Please install it with: pip install biopython",
                raw={"error": "biopython_not_found"},
                error="BioPython not available",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=f"DSSP calculation failed: {type(e).__name__}: {str(e)}",
                raw={"error": "dssp_failed"},
                error=str(e),
            )

        if not res_dssp:
            return ToolResult(success=False, data=f"No residues found for chain {chain_id} in structure", raw={})

        # Parse residue range if provided
        start_range = None
        end_range = None
        if residue_range and "-" in residue_range:
            try:
                parts = residue_range.split("-")
                start_range = int(parts[0])
                end_range = int(parts[1])
            except ValueError:
                return ToolResult(
                    success=False,
                    data=f"Invalid residue_range format: {residue_range}. Use 'start-end' format.",
                    raw={},
                )

        # Filter residues by range if specified
        if start_range is not None and end_range is not None:
            res_dssp = {k: v for k, v in res_dssp.items() if start_range <= k <= end_range}

        if not res_dssp:
            return ToolResult(
                success=False, data=f"No residues found in range {residue_range} for chain {chain_id}", raw={}
            )

        # Group consecutive residues with same secondary structure
        sorted_resnums = sorted(res_dssp.keys())
        elements = []
        current_type = None
        current_start = None
        current_start_label = None
        current_count = 0

        helix_count = 0
        strand_count = 0
        coil_count = 0

        for resnum in sorted_resnums:
            label, dssp_code = res_dssp[resnum]
            sec_struct = _dssp_to_secondary(dssp_code)

            # Check if this residue is consecutive to the current element
            if current_type == sec_struct:
                current_count += 1
            else:
                # Save previous element if it exists
                if current_type is not None:
                    end_label = res_dssp.get(sorted_resnums[sorted_resnums.index(resnum) - 1])[0]
                    elements.append(
                        {
                            "type": current_type,
                            "start_residue": current_start_label,
                            "end_residue": end_label,
                            "count": current_count,
                        }
                    )
                    if current_type == "helix":
                        helix_count += 1
                    elif current_type == "strand":
                        strand_count += 1
                    else:
                        coil_count += 1

                # Start new element
                current_type = sec_struct
                current_start = resnum
                current_start_label = label
                current_count = 1

        # Don't forget the last element
        if current_type is not None:
            last_label = res_dssp[sorted_resnums[-1]][0]
            elements.append(
                {
                    "type": current_type,
                    "start_residue": current_start_label,
                    "end_residue": last_label,
                    "count": current_count,
                }
            )
            if current_type == "helix":
                helix_count += 1
            elif current_type == "strand":
                strand_count += 1
            else:
                coil_count += 1

        # Build narrative
        narrative_parts = []
        helix_num = 0
        strand_num = 0
        coil_num = 0

        for elem in elements:
            if elem["type"] == "helix":
                helix_num += 1
                narrative_parts.append(
                    _format_element("helix", elem["start_residue"], elem["end_residue"], elem["count"], helix_num)
                )
            elif elem["type"] == "strand":
                strand_num += 1
                narrative_parts.append(
                    _format_element("strand", elem["start_residue"], elem["end_residue"], elem["count"], strand_num)
                )
            else:
                coil_num += 1
                narrative_parts.append(
                    _format_element("coil", elem["start_residue"], elem["end_residue"], elem["count"], coil_num)
                )

        narrative = "; ".join(narrative_parts)

        # Build raw data with summary
        raw = {
            "elements": elements,
            "summary": {
                "helices": helix_count,
                "strands": strand_count,
                "coils": coil_count,
                "total_elements": len(elements),
            },
        }

        return ToolResult(success=True, data=narrative, raw=raw)

    except Exception as e:
        return ToolResult(
            success=False,
            data=f"Error computing secondary structure: {type(e).__name__}: {str(e)}",
            raw={},
            error=str(e),
        )
