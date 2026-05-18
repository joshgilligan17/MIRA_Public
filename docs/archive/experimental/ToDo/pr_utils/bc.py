#!/usr/bin/env python3
import os
import argparse
import glob
import csv
import multiprocessing as mp
from tqdm import tqdm
import pyrosetta as pr
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector, OrResidueSelector
from pyrosetta.rosetta.core.simple_metrics.metrics import TotalEnergyMetric, SasaMetric
from biopython_utils import hotspot_residues

def convert_cif_to_pdb(cif_path):
    pdb_path = cif_path.replace(".cif", ".pdb")
    pose = pr.pose_from_file(cif_path)
    pose.dump_pdb(pdb_path)
    return pdb_path

def run_relax(pose):
    scorefxn = pr.get_fa_scorefxn()
    fr = FastRelax(scorefxn, 1)
    fr.max_iter(200)
    fr.apply(pose)
    return pose

def score_interface(pdb_file, binder_chains=["B"], target_chain="A"):
    pose = pr.pose_from_pdb(pdb_file)
    pose = run_relax(pose)

    # 1. Define Interface: Target vs All Binder Chains (e.g., A_BC)
    binder_str = "".join(binder_chains)
    interface_definition = f"{target_chain}_{binder_str}"
    
    iam = InterfaceAnalyzerMover()
    iam.set_interface(interface_definition)
    scorefxn = pr.get_fa_scorefxn()
    iam.set_scorefunction(scorefxn)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_energy(True)
    iam.set_calc_dSASA(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_separated(True)
    iam.apply(pose)

    # 2. Hotspot Calculation (Combined for all binder chains)
    total_hotspots = 0
    for chain in binder_chains:
        hotspots = hotspot_residues(pdb_file, chain)
        total_hotspots += len(hotspots)

    # 3. Buried Unsat Filter
    buns_xml = '<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" ignore_surface_res="false" use_ddG_style="true" probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />'
    buns_filter = XmlObjects.static_get_filter(buns_xml)
    interface_delta_unsat_hbonds = buns_filter.report_sm(pose)

    # 4. Multi-Chain Binder Metrics (Total Energy of B + C)
    # Build an OR selector for multiple chains
    if len(binder_chains) > 1:
        selector = OrResidueSelector()
        for chain in binder_chains:
            selector.add_residue_selector(ChainSelector(chain))
    else:
        selector = ChainSelector(binder_chains[0])

    tem = TotalEnergyMetric()
    tem.set_scorefunction(scorefxn)
    tem.set_residue_selector(selector)
    
    interfacescore = iam.get_all_data()

    res = {
        'design_name': os.path.basename(pdb_file).replace(".pdb", ""),
        'binder_chains': binder_str,
        'binder_score': tem.calculate(pose),
        'interface_sc': interfacescore.sc_value,
        'interface_dG': iam.get_interface_dG(),
        'interface_dSASA': iam.get_interface_delta_sasa(),
        'interface_nres': total_hotspots,
        'interface_delta_unsat_hbonds': interface_delta_unsat_hbonds,
    }
    return {k: round(v, 2) if isinstance(v, (float, int)) else v for k, v in res.items()}

def worker_init():
    pr.init("-ex1 -ex2 -use_input_sc -ignore_unrecognized_res -mute all")

def process_folder(folder_args):
    folder, binder_chains, target_chain = folder_args
    results = []
    cif_files = glob.glob(os.path.join(folder, "*.cif"))
    for cif in cif_files:
        try:
            pdb_path = convert_cif_to_pdb(cif)
            scores = score_interface(pdb_path, binder_chains, target_chain)
            results.append(scores)
        except Exception as e:
            tqdm.write(f"Error processing {cif}: {e}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Parallel BindCraft Interface Scorer")
    parser.add_argument("--dirs", required=True, help="Root directory")
    parser.add_argument("--binder_chain", default="B,C", help="Comma-separated chains (e.g. B,C)")
    parser.add_argument("--target_chain", default="A")
    parser.add_argument("--cpus", type=int, default=mp.cpu_count())
    parser.add_argument("--output", default="interface_scores.csv")
    args = parser.parse_args()

    # Split the input string into a list: ['B', 'C']
    binder_chains = [c.strip() for c in args.binder_chain.split(",")]

    subdirs = [d for d in glob.glob(os.path.join(args.dirs, "*")) if os.path.isdir(d)]
    task_list = [(d, binder_chains, args.target_chain) for d in subdirs]

    final_data = []
    with mp.Pool(processes=args.cpus, initializer=worker_init) as pool:
        for folder_results in tqdm(pool.imap_unordered(process_folder, task_list), 
                                   total=len(task_list), 
                                   desc="Scoring Complexes"):
            final_data.extend(folder_results)

    if final_data:
        keys = final_data[0].keys()
        with open(args.output, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(final_data)
        print(f"\nSaved {len(final_data)} entries to {args.output}")

if __name__ == "__main__":
    main()