"""Tool for renumbering PDB files with gap and chain control."""

import os
from pathlib import Path

from structagent.registry import tool, ToolResult


def _process_pdb_lines(lines, merge_chains=False, no_skip=False, target_chain="A"):
    """
    Processes PDB lines to renumber residues with gap and chain control.

    Args:
        lines: List of PDB lines
        merge_chains: If True, merge all chains into one with a 50-residue gap
        no_skip: If True, number residues 1-N continuously ignoring gaps
        target_chain: Force a specific final chain ID for the whole file

    Returns:
        List of processed PDB lines
    """
    output_lines = []

    prev_orig_chain = None
    last_orig_resnum = None
    current_new_resnum = 0

    current_chain_idx = 0
    chain_ids = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            if line.startswith("ANISOU"):
                continue
            output_lines.append(line)
            continue

        orig_chain = line[21]
        try:
            # Standard PDB format: residues are at columns 22-26
            orig_resnum = int(line[22:26])
        except ValueError:
            output_lines.append(line)
            continue

        # 1. DETECT CHAIN TRANSITION
        if prev_orig_chain is not None and orig_chain != prev_orig_chain:
            if merge_chains:
                current_new_resnum += 50
            else:
                # Reset numbering for the new chain
                current_new_resnum = 1
                current_chain_idx += 1

            last_orig_resnum = orig_resnum

        # 2. INITIALIZE START OF FIRST CHAIN
        elif last_orig_resnum is None:
            current_new_resnum = 1
            last_orig_resnum = orig_resnum

        # 3. HANDLE RESIDUE INCREMENTS (Within the same chain)
        else:
            if orig_resnum != last_orig_resnum:
                if no_skip:
                    # Ignore the gap in original numbering, just increment by 1
                    current_new_resnum += 1
                else:
                    # Respect the original gap size
                    res_diff = orig_resnum - last_orig_resnum
                    if res_diff > 0:
                        current_new_resnum += res_diff
                last_orig_resnum = orig_resnum

        # Determine final Chain ID
        # If the user forced a specific chain with --chain, use it.
        # If merging, everything becomes one chain.
        # Otherwise, use the incrementing chain ID list.
        if (target_chain != "A" and target_chain is not None) or merge_chains:
            new_chain_char = target_chain if target_chain else "A"
        else:
            new_chain_char = chain_ids[min(current_chain_idx, 25)]

        # Reconstruct the line using f-string formatting for fixed-width columns
        new_line = line[:21] + new_chain_char + f"{current_new_resnum:4d}" + line[26:72] + new_chain_char + line[73:]

        output_lines.append(new_line)
        prev_orig_chain = orig_chain

    return output_lines


@tool(
    name="renumber_pdb",
    toolset="structure",
    description="Renumber residues in a PDB file with control over gap handling and chain ID assignment. Saves output as {stem}_renum.pdb.",
    parameters={
        "pdb_path": {"type": "string", "description": "Path to the input PDB file"},
        "merge_chains": {
            "type": "boolean",
            "description": "If True, merge all chains into one with a 50-residue gap between chains",
            "default": False,
        },
        "no_skip": {
            "type": "boolean",
            "description": "If True, number residues 1-N continuously ignoring all gaps in the original numbering",
            "default": False,
        },
        "target_chain": {
            "type": "string",
            "description": "Force a specific final chain ID for the whole file (default: 'A')",
            "default": "A",
        },
    },
    check_fn=None,
)
def renumber_pdb(
    pdb_path: str, merge_chains: bool = False, no_skip: bool = False, target_chain: str = "A"
) -> ToolResult:
    """
    Renumber residues in a PDB file with gap and chain ID control.

    This tool processes a PDB file and outputs a renumbered version where
    residues are sequentially numbered starting from 1. It provides control
    over how gaps in the original numbering are handled and allows
    specification of the output chain ID.

    Args:
        pdb_path: Path to the input PDB file
        merge_chains: If True, merge all chains into one with a 50-residue gap
        no_skip: If True, number residues 1-N continuously ignoring gaps
        target_chain: Force a specific final chain ID for the whole file

    Returns:
        ToolResult with:
        - success: bool indicating if renumbering succeeded
        - data: human-readable narrative description of the operation
        - raw: dict with input_path, output_path, mode, chains_processed,
               atoms_processed, and residue_count
    """
    try:
        # Validate input file
        input_path = Path(pdb_path)
        if not input_path.exists():
            return ToolResult(
                success=False, data=f"Input PDB file not found: {pdb_path}", raw={}, error=f"File not found: {pdb_path}"
            )

        if not input_path.is_file():
            return ToolResult(
                success=False, data=f"Input path is not a file: {pdb_path}", raw={}, error=f"Not a file: {pdb_path}"
            )

        # Read input file
        with open(input_path, "r") as f:
            original_lines = f.readlines()

        # Count input stats
        input_atoms = sum(1 for line in original_lines if line.startswith(("ATOM", "HETATM")))
        input_chains = set(
            line[21] for line in original_lines if line.startswith(("ATOM", "HETATM")) and len(line) > 21
        )

        # Process PDB lines
        final_lines = _process_pdb_lines(
            original_lines, merge_chains=merge_chains, no_skip=no_skip, target_chain=target_chain
        )

        # Determine output path
        stem = input_path.stem
        suffix = "_renum"
        output_path = input_path.parent / f"{stem}{suffix}{input_path.suffix}"

        # Write output file
        with open(output_path, "w") as f:
            f.writelines(final_lines)

        # Count output stats
        output_atoms = sum(1 for line in final_lines if line.startswith(("ATOM", "HETATM")))
        output_chains = set(line[21] for line in final_lines if line.startswith(("ATOM", "HETATM")) and len(line) > 21)

        # Determine mode string
        if no_skip:
            mode_str = "continuous (per chain, gaps ignored)"
        elif merge_chains:
            mode_str = "merged chains (50-residue gap between chains)"
        else:
            mode_str = "standard (gaps preserved)"

        # Build narrative
        chain_list_in = sorted(input_chains) if input_chains else ["?"]
        chain_list_out = sorted(output_chains) if output_chains else ["?"]

        data_lines = [
            f"Renumbered PDB file saved to: {output_path.name}",
            f"Mode: {mode_str}",
            f"Processed {input_atoms} atoms across {len(input_chains)} input chain(s) ({', '.join(chain_list_in)})",
            f"Output contains {output_atoms} atoms in {len(output_chains)} chain(s) ({', '.join(chain_list_out)})",
            f"Target chain ID was set to '{target_chain}'.",
        ]

        data = " ".join(data_lines)

        raw = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "mode": mode_str,
            "merge_chains": merge_chains,
            "no_skip": no_skip,
            "target_chain": target_chain,
            "chains_processed": len(input_chains),
            "input_chains": sorted(input_chains),
            "output_chains": sorted(output_chains),
            "atoms_processed": input_atoms,
            "residue_count": output_atoms,  # approximate, atoms include all HETATM
        }

        return ToolResult(success=True, data=data, raw=raw)

    except Exception as e:
        return ToolResult(
            success=False, data=f"Error renumbering PDB file: {type(e).__name__}: {str(e)}", raw={}, error=str(e)
        )
