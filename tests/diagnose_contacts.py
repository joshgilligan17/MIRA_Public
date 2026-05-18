#!/usr/bin/env python3
"""Diagnostic script for get_residue_contacts tool against various PDB structures."""

import sys
import traceback

# Import the function to test
from structagent.tools.contacts import get_residue_contacts

# Import gemmi and structure_io for diagnostics
import gemmi
from structagent.tools import structure_io


test_cases = [
    ("1UBQ", "A", 48, "LYS-48 in ubiquitin, simple case"),
    ("3HFM", "H", 31, "Antibody heavy chain, starts at residue 1"),
    ("2HHB", "A", 87, "Hemoglobin alpha, HIS-87 (F8 proximal histidine)"),
    ("2HHB", "B", 92, "Hemoglobin beta, HIS-92 (F8 proximal histidine)"),
    ("1UBQ", "A", 999, "Non-existent residue — should fail gracefully"),
    ("1UBQ", "X", 48, "Non-existent chain — should fail gracefully"),
]


def get_chain_residue_numbers(pdb_id: str, chain_id: str) -> list[int] | None:
    """Get list of residue sequence numbers for a chain using gemmi."""
    try:
        structure = structure_io.get_structure(pdb_id=pdb_id)
        for model in structure:
            for chain in model:
                if chain.name == chain_id:
                    res_nums = []
                    for residue in chain:
                        res_nums.append(int(residue.seqid.num))
                    return res_nums
        return None
    except Exception as e:
        return None


def print_chain_residue_info(pdb_id: str, chain_id: str) -> None:
    """Print diagnostic info about residues in a chain."""
    res_nums = get_chain_residue_numbers(pdb_id, chain_id)

    if res_nums is None:
        print(f"    Could not retrieve chain {chain_id} from {pdb_id}")
        return

    if not res_nums:
        print(f"    Chain {chain_id} has no residues")
        return

    # Show first 5 and last 5 residues
    sorted_nums = sorted(res_nums)
    total = len(sorted_nums)

    if total <= 10:
        print(f"    Residues in chain {chain_id}: {sorted_nums}")
    else:
        first_5 = sorted_nums[:5]
        last_5 = sorted_nums[-5:]
        print(f"    Residues in chain {chain_id} ({total} total):")
        print(f"      First 5: {first_5}")
        print(f"      Last 5: {last_5}")


def run_diagnostic(pdb_id: str, chain_id: str, residue_number: int, description: str) -> None:
    """Run a single diagnostic test case."""
    print("=" * 70)
    print(f"PDB: {pdb_id}, Chain: {chain_id}, Residue: {residue_number}")
    print(f"Description: {description}")
    print("-" * 70)

    try:
        result = get_residue_contacts(
            pdb_id=pdb_id,
            chain_id=chain_id,
            residue_number=residue_number,
            cutoff_angstroms=4.5
        )

        if result.success:
            print(f"  STATUS: SUCCESS")
            # Count contacts from raw data
            contacts = result.raw.get("contacts", [])
            print(f"  Contacts found: {len(contacts)}")
            if contacts:
                print(f"  Top 3 contacts:")
                for c in contacts[:3]:
                    print(f"    - {c['residue']} (chain {c['chain']}): {c['contact_type']} at {c['distance']} A")
        else:
            print(f"  STATUS: FAILED")
            print(f"  Error: {result.error}")

            # Show what residues are in the chain to help diagnose
            if "not found" in result.error.lower() or "residue" in result.error.lower():
                print(f"\n  Diagnostic info for chain {chain_id}:")
                print_chain_residue_info(pdb_id, chain_id)

    except Exception as e:
        print(f"  STATUS: EXCEPTION RAISED")
        print(f"  Exception: {type(e).__name__}: {e}")
        print(f"  Traceback:")
        traceback.print_exc()
        print(f"\n  Diagnostic info for chain {chain_id}:")
        print_chain_residue_info(pdb_id, chain_id)

    print()


def main() -> None:
    """Run all diagnostic test cases."""
    print("\n" + "=" * 70)
    print("RESIDUE CONTACTS DIAGNOSTIC SCRIPT")
    print("=" * 70)
    print()

    for pdb_id, chain_id, residue_number, description in test_cases:
        run_diagnostic(pdb_id, chain_id, residue_number, description)

    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
