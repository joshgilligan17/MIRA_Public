"""Tools for structural alignment and superposition."""

import gemmi

from structagent.registry import tool, ToolResult
from structagent.tools.structure_io import get_structure


@tool(
    name="align_structures",
    toolset="structure",
    description="Align two protein chains by Cα superposition and compute per-residue deviations. Returns RMSD, rotation matrix, shift vector, aligned length, and flags regions with >2A deviation.",
    parameters={
        "pdb_id_1": {
            "type": "string",
            "description": "4-character PDB identifier for the first structure (e.g., '1ABC'). Use either this or pdb_path_1.",
        },
        "pdb_path_1": {
            "type": "string",
            "description": "Path to local PDB file for the first structure. Use either this or pdb_id_1.",
        },
        "chain_id_1": {"type": "string", "description": "Chain identifier in the first structure (e.g., 'A')"},
        "pdb_id_2": {
            "type": "string",
            "description": "4-character PDB identifier for the second structure (e.g., '6VXX'). Use either this or pdb_path_2.",
        },
        "pdb_path_2": {
            "type": "string",
            "description": "Path to local PDB file for the second structure. Use either this or pdb_id_2.",
        },
        "chain_id_2": {"type": "string", "description": "Chain identifier in the second structure (e.g., 'A')"},
    },
)
def align_structures(
    pdb_id_1: str = None,
    pdb_path_1: str = None,
    chain_id_1: str = None,
    pdb_id_2: str = None,
    pdb_path_2: str = None,
    chain_id_2: str = None,
) -> ToolResult:
    """
    Align two protein chains using Cα superposition.

    Loads both structures from the PDB, extracts the specified chains,
    performs sequence-aware Cα superposition, and computes per-residue
    deviations after superposition.

    Args:
        pdb_id_1: 4-character PDB identifier for the first structure
        chain_id_1: Chain identifier in the first structure
        pdb_id_2: 4-character PDB identifier for the second structure
        chain_id_2: Chain identifier in the second structure

    Returns:
        ToolResult with:
        - success: bool indicating if alignment succeeded
        - data: narrative description with RMSD, aligned residues, flagged regions
        - raw: dict with rmsd, aligned_length, total_length, rotation_matrix,
               shift_vector, per_residue_deviations, flagged_regions
    """
    try:
        # Load both structures using unified loader
        structure_1 = get_structure(pdb_id=pdb_id_1, pdb_path=pdb_path_1)
        structure_2 = get_structure(pdb_id=pdb_id_2, pdb_path=pdb_path_2)

        # Get the first model (model index 0)
        model_1 = structure_1[0]
        model_2 = structure_2[0]

        # Find the chains (using [] access for gemmi 0.7.5 compatibility)
        try:
            chain_1 = model_1[chain_id_1]
        except KeyError:
            pdb1_label = pdb_id_1.upper() if pdb_id_1 else (pdb_path_1 if pdb_path_1 else "unknown")
            return ToolResult(
                success=False,
                data=f"Chain '{chain_id_1}' not found in structure {pdb1_label}",
                raw={},
                error=f"Chain not found: {chain_id_1}",
            )

        try:
            chain_2 = model_2[chain_id_2]
        except KeyError:
            pdb2_label = pdb_id_2.upper() if pdb_id_2 else (pdb_path_2 if pdb_path_2 else "unknown")
            return ToolResult(
                success=False,
                data=f"Chain '{chain_id_2}' not found in structure {pdb2_label}",
                raw={},
                error=f"Chain not found: {chain_id_2}",
            )

        # Get polymer sequences for each chain
        polymer_1 = chain_1.get_polymer()
        polymer_2 = chain_2.get_polymer()

        pdb1_label = pdb_id_1.upper() if pdb_id_1 else (pdb_path_1 if pdb_path_1 else "unknown")
        pdb2_label = pdb_id_2.upper() if pdb_id_2 else (pdb_path_2 if pdb_path_2 else "unknown")

        if polymer_1 is None:
            return ToolResult(
                success=False,
                data=f"Chain '{chain_id_1}' in structure {pdb1_label} is not a polymer",
                raw={},
                error="Not a polymer chain",
            )

        if polymer_2 is None:
            return ToolResult(
                success=False,
                data=f"Chain '{chain_id_2}' in structure {pdb2_label} is not a polymer",
                raw={},
                error="Not a polymer chain",
            )

        # Determine polymer type
        ptype = gemmi.PolymerType.PeptideL

        # Perform Cα superposition
        sup_result = gemmi.calculate_superposition(polymer_1, polymer_2, ptype, gemmi.SupSelect.CaP)

        # Extract RMSD and transformation
        rmsd = sup_result.rmsd
        rotation_matrix = sup_result.transform.mat
        shift_vector = sup_result.transform.vec

        # Apply the superposition transformation to chain 2's polymer
        # (this modifies the coordinates in place for deviation calculation)
        polymer_2.transform_pos_and_adp(sup_result.transform)

        # Calculate per-residue Cα deviations
        per_residue_deviations = []
        flagged_regions = []

        # Iterate through both polymers in parallel (they should be aligned)
        for res_1, res_2 in zip(polymer_1, polymer_2):
            # Get Cα atoms from each residue
            ca_1 = res_1.get_ca()
            ca_2 = res_2.get_ca()

            if ca_1 is None or ca_2 is None:
                continue

            # Calculate Euclidean distance between Cα atoms
            pos_1 = ca_1.pos
            pos_2 = ca_2.pos
            deviation = (pos_2 - pos_1).length()

            res_info = {"residue_1": str(res_1.seqid), "residue_2": str(res_2.seqid), "deviation": round(deviation, 3)}
            per_residue_deviations.append(res_info)

            # Flag regions with >2A deviation
            if deviation > 2.0:
                flagged_regions.append({"residue": str(res_2.seqid), "deviation": round(deviation, 3)})

        # Aligned length is the number of residue pairs compared
        aligned_length = len(per_residue_deviations)

        # Total length is the number of residues in the reference chain (chain_1)
        total_length = sum(1 for _ in polymer_1)

        # Build rotation matrix as nested list for JSON serialization
        rot_matrix_list = [[round(v, 6) for v in row] for row in rotation_matrix.tolist()]

        # Build shift vector as list
        shift_vec_list = [round(shift_vector[i], 6) for i in range(3)]

        # Build human-readable narrative
        flagged_count = len(flagged_regions)
        flagged_summary = []
        if flagged_regions:
            # Group consecutive flagged residues for cleaner reporting
            flagged_res_nums = [f["residue"] for f in flagged_regions]
            flagged_summary = flagged_res_nums[:10]  # Show first 10
            if len(flagged_res_nums) > 10:
                flagged_summary.append(f"... and {len(flagged_res_nums) - 10} more")

        data_parts = [
            f"Aligned chain '{chain_id_1}' from {pdb1_label} against chain '{chain_id_2}' from {pdb2_label}.",
            f"Overall RMSD: {rmsd:.3f} Angstrom over {aligned_length} residue pairs.",
            f"Reference chain length: {total_length} residues.",
        ]

        if flagged_regions:
            flagged_text = ", ".join(str(r) for r in flagged_summary)
            data_parts.append(
                f"Flagged {flagged_count} region{'s' if flagged_count != 1 else ''} "
                f"with >2A Cα deviation: {flagged_text}."
            )
        else:
            data_parts.append("No regions with >2A Cα deviation detected.")

        data = " ".join(data_parts)

        # Build raw output
        raw = {
            "rmsd": round(rmsd, 3),
            "aligned_length": aligned_length,
            "total_length": total_length,
            "rotation_matrix": rot_matrix_list,
            "shift_vector": shift_vec_list,
            "per_residue_deviations": per_residue_deviations,
            "flagged_regions": flagged_regions,
        }

        return ToolResult(success=True, data=data, raw=raw)

    except ValueError as e:
        return ToolResult(success=False, data=f"Alignment failed: {str(e)}", raw={}, error=str(e))
    except Exception as e:
        return ToolResult(
            success=False, data=f"Unexpected error during alignment: {type(e).__name__}: {str(e)}", raw={}, error=str(e)
        )
