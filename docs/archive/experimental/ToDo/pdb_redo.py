#!/usr/bin/env python3
import sys
import os
import argparse

def process_pdb(lines, merge_chains=False, no_skip=False, target_chain='A'):
    """
    Processes PDB lines to renumber residues with gap and chain control.
    """
    output_lines = []
    
    prev_orig_chain = None
    last_orig_resnum = None
    current_new_resnum = 0
    
    current_chain_idx = 0
    chain_ids = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    for line in lines:
        if not line.startswith(('ATOM', 'HETATM')):
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
        if (target_chain != 'A' and target_chain is not None) or merge_chains:
            new_chain_char = target_chain if target_chain else 'A'
        else:
            new_chain_char = chain_ids[min(current_chain_idx, 25)]
        
        # Reconstruct the line using f-string formatting for fixed-width columns
        new_line = (
            line[:21] + 
            new_chain_char + 
            f"{current_new_resnum:4d}" + 
            line[26:72] + 
            new_chain_char + 
            line[73:]
        )
        
        output_lines.append(new_line)
        prev_orig_chain = orig_chain

    return output_lines

def main():
    parser = argparse.ArgumentParser(description="Renumber PDBs with gap and chain ID control.")
    parser.add_argument("input", help="Path to the input PDB file")
    # Changed type to action="store_true" for standard boolean flag behavior
    parser.add_argument("--merge", action="store_true", help="Merge all chains into one with a 50-residue gap")
    parser.add_argument("--noskip", action="store_true", help="Number residues 1-N continuously, ignoring all gaps")
    parser.add_argument("--chain", type=str, default="A", help="Force a specific final chain ID for the whole file")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File '{args.input}' not found.")
        sys.exit(1)

    filename, ext = os.path.splitext(args.input)
    suffix = "_renum"
    output_path = f"{filename}{suffix}{ext}"

    with open(args.input, 'r') as f:
        original_lines = f.readlines()

    final_lines = process_pdb(
        original_lines, 
        merge_chains=args.merge, 
        no_skip=args.noskip, 
        target_chain=args.chain
    )

    with open(output_path, 'w') as f:
        f.writelines(final_lines)

    mode_str = "Continuous (per chain)" if args.noskip else "Merged" if args.merge else "Standard"
    print(f"Mode: {mode_str}")
    print(f"Success! Output: {output_path}")

if __name__ == "__main__":
    main()