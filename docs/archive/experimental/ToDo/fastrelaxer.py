#!/usr/bin/env python3
import sys
import os

import pyrosetta
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.kinematics import MoveMap


def run_fast_relax(input_pdb):
    # 1. Initialize PyRosetta
    # -ignore_unrecognized_res: prevents crashing on non-standard residues
    # -mute all: reduces console noise (optional, remove if you want full logs)
    pyrosetta.init(extra_options="-ignore_unrecognized_res")

    # 2. Load the PDB file into a Pose object
    if not os.path.exists(input_pdb):
        print(f"Error: The file '{input_pdb}' does not exist.")
        sys.exit(1)
        
    print(f"Loading PDB file: {input_pdb}")
    pose = pyrosetta.pose_from_pdb(input_pdb)

    # 4. Explicitly Setup the MoveMap
    movemap = MoveMap()
    movemap.set_bb(False)
    movemap.set_chi(True)
    movemap.set_jump(False)

    # 5. Configure FastRelax
    fast_relax = FastRelax()
    scorefxn = pyrosetta.get_fa_scorefxn()
    fast_relax.set_scorefxn(scorefxn)
    fast_relax.set_movemap(movemap)
    fast_relax.max_iter(250)
    fast_relax.min_type("lbfgs_armijo_nonmonotone")
    fast_relax.constrain_relax_to_start_coords(True)

    print("Starting FastRelax (this may take a moment)...")
    fast_relax.apply(pose)

    # 7. Output the file
    base_name = os.path.splitext(input_pdb)[0]
    output_pdb = f"{base_name}_relaxed.pdb"
    
    print(f"Saving relaxed structure to: {output_pdb}")
    pose.dump_pdb(output_pdb)
    print("Done.")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python3 fastrelaxer.py <pdb_file>")
        sys.exit(1)
    
    pdb_file_arg = sys.argv[1]
    run_fast_relax(pdb_file_arg)