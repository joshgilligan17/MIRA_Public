#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import signal
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pyrosetta import *
import pyrosetta as pr
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.core.pose import append_pose_to_pose
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from biopython_utils import hotspot_residues

def init_worker():
    """Initializes Rosetta in each worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Use the absolute path to DAlphaBall you provided
    pr.init("-holes:dalphaball /home/seanwang/bin/DAlphaBall.gcc -mute all")

def score_interface(pdb_file, target_chains_input, binder_chains_input):
    temp_pdb = f"tmp_{os.getpid()}_{os.path.basename(pdb_file)}"
    
    try:
        full_pose = pr.pose_from_pdb(pdb_file)
        target_list = [c.strip() for c in target_chains_input.split(',')]
        binder_list = [c.strip() for c in binder_chains_input.split(',')]

        def get_merged_chains(source_pose, chain_list):
            merged = None
            for c in chain_list:
                p = source_pose.clone()
                for i in reversed(range(1, p.total_residue() + 1)):
                    if p.pdb_info().chain(i) != c:
                        p.delete_residue_slow(i)
                if merged is None: merged = p
                else: append_pose_to_pose(merged, p, new_chain=False)
            return merged

        # 1. Merge Poses
        pose_target = get_merged_chains(full_pose, target_list)
        pose_binder = get_merged_chains(full_pose, binder_list)
        clean_pose = pose_target.clone()
        append_pose_to_pose(clean_pose, pose_binder, new_chain=True)
        
        # 2. FIXED RENUMBERING: Forces both Chain A and B to start at residue 1
        target_len = pose_target.total_residue()
        binder_len = pose_binder.total_residue()
        pdb_info = clean_pose.pdb_info()
        
        # Renumber Chain A (residues 1 to target_len)
        for i in range(1, target_len + 1):
            pdb_info.number(i, i)
            pdb_info.chain(i, 'A')
            
        # Renumber Chain B (residues target_len + 1 to total) to start at 1
        for i in range(1, binder_len + 1):
            ros_idx = target_len + i
            pdb_info.number(ros_idx, i) # This makes it B1, B2, B3...
            pdb_info.chain(ros_idx, 'B')
        
        clean_pose.pdb_info(pdb_info)

        # 3. Reload for clean internal pointers
        clean_pose.dump_pdb(temp_pdb)
        pose = pr.pose_from_pdb(temp_pdb)
        scorefxn = pr.get_fa_scorefxn()

        # 4. YOUR INTEGRATED FASTRELAX BLOCK
        movemap = MoveMap()
        movemap.set_chi(True)
        movemap.set_bb(True)
        
        fast_relax = FastRelax(standard_repeats=1)
        fast_relax.set_scorefxn(scorefxn)
        fast_relax.set_movemap(movemap)
        fast_relax.max_iter(200)
        fast_relax.min_type("lbfgs_armijo_nonmonotone")
        fast_relax.constrain_relax_to_start_coords(True)
        fast_relax.apply(pose)

        # 5. Interface Analysis
        iam = InterfaceAnalyzerMover()
        iam.set_interface("A_B")
        iam.set_scorefunction(scorefxn)
        iam.set_compute_packstat(True)
        iam.set_compute_interface_energy(True)
        iam.apply(pose)
        interfacescore = iam.get_all_data()

        # 6. Metrics & Composition
        exp_apol_count = 0
        total_count = 0
        for i in range(1, pose.total_residue() + 1):
            res_name = pose.residue(i).name3()
            if res_name in ['PHE', 'ILE', 'LEU', 'VAL', 'MET', 'TRP', 'TYR']:
                exp_apol_count += 1
            total_count += 1
        surf_hydro = exp_apol_count / total_count if total_count > 0 else 0

        # Hotspots (using the renumbered temp file)
        pose.dump_pdb(temp_pdb) # Update temp file after relax for hotspot accuracy
        res_data = hotspot_residues(temp_pdb, "B")
        res_ids = [f"B{num}" for num in res_data.keys()]
        
        interface_AA = {aa: 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}
        hydro_aa = set('ACFILMPVWY')
        h_count = 0
        for aa in res_data.values():
            interface_AA[aa] += 1
            if aa in hydro_aa: h_count += 1

        # 7. Binder Score & Fractions
        sel_b = ChainSelector("B")
        tem = pr.rosetta.core.simple_metrics.metrics.TotalEnergyMetric(sel_b, scorefxn)
        bsasa_m = pr.rosetta.core.simple_metrics.metrics.SasaMetric(sel_b)
        b_sasa = bsasa_m.calculate(pose)
        
        # 8. Buried Unsats
        buns_xml = '<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" ignore_surface_res="false" use_ddG_style="true" dalphaball_sasa="1" probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />'
        buns_filter = XmlObjects.static_get_filter(buns_xml)
        d_unsat = buns_filter.report_sm(pose)

        # 9. Dictionary Compilation
        results = {
            'ID': os.path.basename(pdb_file),
            'binder_score': tem.calculate(pose),
            'surface_hydrophobicity': surf_hydro,
            'interface_sc': interfacescore.sc_value,
            'interface_packstat': iam.get_interface_packstat(),
            'interface_dG': iam.get_interface_dG(),
            'interface_dSASA': iam.get_interface_delta_sasa(),
            'interface_fraction': (iam.get_interface_delta_sasa() / b_sasa * 100) if b_sasa > 0 else 0,
            'interface_nres': len(res_ids),
            'interface_interface_hbonds': interfacescore.interface_hbonds,
            'interface_delta_unsat_hbonds': d_unsat,
            'interface_residues': ",".join(res_ids)
        }
        results.update({f"AA_{k}": v for k, v in interface_AA.items()})

        if os.path.exists(temp_pdb): os.remove(temp_pdb)
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in results.items()}

    except Exception as e:
        if os.path.exists(temp_pdb): os.remove(temp_pdb)
        print(f"FAILED: {pdb_file} -> {e}")
        return None

def worker_wrapper(args):
    return score_interface(*args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdb_dir")
    parser.add_argument("--target_chain", required=True)
    parser.add_argument("--binder_chain", required=True)
    parser.add_argument("--out", default="relaxed_results.xlsx")
    parser.add_argument("--cpus", type=int, default=cpu_count())
    args = parser.parse_args()

    pdb_paths = [os.path.join(args.pdb_dir, f) for f in os.listdir(args.pdb_dir) if f.endswith(".pdb")]
    tasks = [(p, args.target_chain, args.binder_chain) for p in pdb_paths]

    pool = Pool(processes=args.cpus, initializer=init_worker, maxtasksperchild=1)
    
    results = []
    # Use apply_async to survive worker segfaults
    jobs = [pool.apply_async(worker_wrapper, (t,)) for t in tasks]
    
    for job in tqdm(jobs, desc="Relaxing & Scoring"):
        try:
            res = job.get(timeout=1200) # 20 mins timeout for Relax
            if res: results.append(res)
        except Exception as e:
            print(f"\nTask timed out or crashed. Continuing...")

    pool.close()
    pool.join()

    if results:
        pd.DataFrame(results).to_excel(args.out, index=False)
        print(f"Success! Results saved to {args.out}")

if __name__ == "__main__":
    main()