"""PyRosetta-based interface scoring tools for protein-protein interactions."""

import os
import numpy as np
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser, Selection

from structagent.registry import tool, ToolResult


# Three-letter to one-letter amino acid code mapping
THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def _check_pyrosetta() -> bool:
    """Check if PyRosetta is available and can be imported."""
    try:
        import pyrosetta

        return True
    except ImportError:
        return False


def _get_pyrosetta_install_message() -> str:
    """Return installation instructions for PyRosetta."""
    return (
        "PyRosetta is not installed. To install:\n"
        "1. Go to https://www.pyrosetta.org/ and request a license\n"
        "2. Download PyRosetta from Rosetta Commons (rosettacommons.github.io)\n"
        "3. Follow installation instructions: https://www.pyrosetta.org/docs/home\n"
        "   Note: PyRosetta requires a license agreement and is not available via pip/conda."
    )


def _hotspot_residues(pdb_path: str, binder_chain: str, target_chain: str, atom_distance_cutoff: float = 4.0) -> dict:
    """Identify interacting hotspot residues at the binder interface.

    Uses Bio.PDB and scipy cKDTree to find residues on the binder chain
    that are within atom_distance_cutoff of any atom on the target chain.

    Args:
        pdb_path: Path to PDB file
        binder_chain: Chain identifier for the binder (e.g., 'B')
        target_chain: Chain identifier for the target (e.g., 'A')
        atom_distance_cutoff: Distance cutoff in Angstroms (default 4.0)

    Returns:
        Dict mapping residue numbers to {res: single_letter_code, contacts: set of target res nums}
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)

    # Get atoms for both chains
    binder_atoms = Selection.unfold_entities(structure[0][binder_chain], "A")
    binder_coords = np.array([atom.coord for atom in binder_atoms])

    target_atoms = Selection.unfold_entities(structure[0][target_chain], "A")
    target_coords = np.array([atom.coord for atom in target_atoms])

    # Build KD trees for both chains
    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)

    # Find pairs of atoms within distance cutoff
    pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

    # Collect interacting residues
    interacting_residues = {}

    for binder_idx, close_indices in enumerate(pairs):
        if not close_indices:
            continue

        binder_residue = binder_atoms[binder_idx].get_parent()
        binder_resname = binder_residue.get_resname()

        if binder_resname in THREE_TO_ONE:
            aa_single_letter = THREE_TO_ONE[binder_resname]
            binder_resnum = binder_residue.id[1]

            if binder_resnum not in interacting_residues:
                interacting_residues[binder_resnum] = {"res": aa_single_letter, "contacts": set()}

            for close_idx in close_indices:
                target_residue = target_atoms[close_idx].get_parent()
                interacting_residues[binder_resnum]["contacts"].add(target_residue.id[1])

    return interacting_residues


def _strip_modified_terminal_residues(pdb_path: str) -> str:
    """Strip modified terminal residues that PyRosetta cannot handle.

    Uses Bio.PDB to parse and identify standard vs modified residues,
    then writes only standard residues to a cleaned PDB file.

    Args:
        pdb_path: Path to input PDB file

    Returns:
        Path to cleaned PDB file
    """
    from Bio.PDB import PDBParser, PDBIO, Polypeptide

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)

    # Standard 20 amino acids + common modified ones we can handle
    standard_resnames = {
        "ALA",
        "CYS",
        "ASP",
        "GLU",
        "PHE",
        "GLY",
        "HIS",
        "ILE",
        "LYS",
        "LEU",
        "MET",
        "ASN",
        "PRO",
        "GLN",
        "ARG",
        "SER",
        "THR",
        "VAL",
        "TRP",
        "TYR",
        # Common modified residues that PyRosetta can handle
        "MSE",  # Selenomethionine
        "PTR",  # Phosphotyrosine
        "SEP",  # Phosphoserine
        "TPO",  # Phosphothreonine
        "CSO",  # S-hydroxycysteine
        "NLE",  # Norleucine
        "BVA",  # Beta-valine
    }

    class ResidueSelect(Polypeptide.PolypeptideSequencer):
        """Select only standard residues."""

        def accept_residue(self, residue):
            resname = residue.get_resname()
            # Include only standard amino acids
            if resname in standard_resnames:
                return True
            return False

    # Create a new structure with only standard residues
    # We need to manually filter atoms
    output_lines = []

    # Read original file and keep only ATOM/HETATM for standard residues
    chain_last_res = {}  # (chain_id, resnum) -> last atom line seen

    with open(pdb_path, "r") as f:
        lines = f.readlines()

    # First pass: identify problematic modified residues
    # Look for residues like PRO:CtermProteinFull, MET:Cext, etc.
    modified_residues = set()  # (chain, resnum) tuples to skip

    current_chain = None
    current_resnum = None
    residue_atoms = []

    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            chain = line[21]
            resnum = int(line[22:26])
            resname = line[17:20].strip()

            if resname not in standard_resnames:
                modified_residues.add((chain, resnum))

    # Second pass: write only standard residues
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            output_lines.append(line)
            continue

        chain = line[21]
        resnum = int(line[22:26])
        resname = line[17:20].strip()

        if (chain, resnum) in modified_residues:
            continue

        output_lines.append(line)

    # Write cleaned PDB
    cleaned_path = pdb_path.replace(".pdb", "_cleaned.pdb")
    with open(cleaned_path, "w") as f:
        f.writelines(output_lines)

    return cleaned_path


def _convert_cif_to_pdb(cif_path: str) -> str:
    """Convert mmCIF file to PDB format using PyRosetta.

    Following the original bc.py approach, converting to PDB format
    can help handle some modified residues better than direct CIF loading.

    Args:
        cif_path: Path to input CIF file

    Returns:
        Path to the converted PDB file (in same directory as CIF)
    """
    import pyrosetta as pr

    pdb_path = cif_path.replace(".cif", ".pdb")

    # Try to convert CIF to PDB
    try:
        pr.init("-ignore_unrecognized_res -mute all", silent=True)
        pose = pr.pose_from_file(cif_path)
        pose.dump_pdb(pdb_path)
    except Exception as e:
        # If direct conversion fails, try using Bio.PDB to convert
        from Bio.PDB import PDBParser, PDBIO

        parser = PDBParser()
        structure = parser.get_structure("structure", cif_path)
        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_path)

    return pdb_path


def _extract_interface_chains(pose, target_chain: str, binder_chains: list) -> tuple:
    """Extract target and binder chains into a clean 2-chain pose.

    PyRosetta's InterfaceAnalyzerMover with use_ddG_style has issues with
    poses > 3 chains. This function extracts only the relevant chains
    and renumbers them to create a clean 2-chain pose.

    Args:
        pose: Full PyRosetta pose
        target_chain: Target chain ID
        binder_chains: List of binder chain IDs

    Returns:
        Tuple of (clean_pose, target_len, binder_len) where clean_pose
        has target as chain A and binder(s) as chain B.
    """
    from pyrosetta.rosetta.core.pose import append_pose_to_pose

    def get_chain_pose(source_pose, chain_id):
        """Extract a single chain from the pose."""
        p = source_pose.clone()
        for i in reversed(range(1, p.total_residue() + 1)):
            if p.pdb_info().chain(i) != chain_id:
                p.delete_residue_slow(i)
        return p

    # Extract target chain
    pose_target = get_chain_pose(pose, target_chain)

    # Extract and merge binder chains
    pose_binder = None
    for bc in binder_chains:
        chain_pose = get_chain_pose(pose, bc)
        if pose_binder is None:
            pose_binder = chain_pose
        else:
            append_pose_to_pose(pose_binder, chain_pose, new_chain=False)

    # Build clean pose: target (A) + binder (B)
    clean_pose = pose_target.clone()
    append_pose_to_pose(clean_pose, pose_binder, new_chain=True)

    # Renumber: Chain A (target) -> residues 1 to target_len
    #           Chain B (binder) -> residues target_len+1 to total (renumbered to 1-N)
    target_len = pose_target.total_residue()
    binder_len = pose_binder.total_residue()
    pdb_info = clean_pose.pdb_info()

    # Renumber Chain A
    for i in range(1, target_len + 1):
        pdb_info.number(i, i)
        pdb_info.chain(i, "A")

    # Renumber Chain B (binder) to start at 1
    for i in range(1, binder_len + 1):
        ros_idx = target_len + i
        pdb_info.number(ros_idx, i)
        pdb_info.chain(ros_idx, "B")

    clean_pose.pdb_info(pdb_info)

    return clean_pose, target_len, binder_len


@tool(
    name="score_interface",
    toolset="structure",
    description="Compute comprehensive interface scoring metrics for protein-protein complexes "
    "using PyRosetta's InterfaceAnalyzerMover. Calculates interface dG, shape "
    "complementarity (sc), packstat, buried unsatisfied hbonds, and delta SASA. "
    "Optionally applies FastRelax before scoring. Also identifies hotspot residues "
    "at the interface using Bio.PDB and scipy cKDTree.",
    parameters={
        "type": "object",
        "properties": {
            "pdb_path": {
                "type": "string",
                "description": "Path to the PDB or mmCIF file containing the protein complex",
            },
            "binder_chains": {
                "type": "string",
                "default": "B",
                "description": "Chain identifier(s) for the binder (e.g., 'B' or 'B,C' for multiple)",
            },
            "target_chain": {
                "type": "string",
                "default": "A",
                "description": "Chain identifier for the target (e.g., 'A')",
            },
            "relax_structure": {
                "type": "boolean",
                "default": True,
                "description": "Whether to apply FastRelax before scoring (recommended for most cases)",
            },
        },
        "required": ["pdb_path"],
    },
    check_fn=_check_pyrosetta,
)
def score_interface(
    pdb_path: str, binder_chains: str = "B", target_chain: str = "A", relax_structure: bool = True
) -> ToolResult:
    """Score a protein-protein interface using PyRosetta.

    Loads a PDB file, optionally relaxes it with FastRelax, and computes
    comprehensive interface metrics using InterfaceAnalyzerMover.

    Args:
        pdb_path: Path to the PDB file
        binder_chains: Chain ID(s) for the binder (default "B", can be "B,C" for multi-chain)
        target_chain: Chain ID for the target (default "A")
        relax_structure: Whether to apply FastRelax before scoring (default True)

    Returns:
        ToolResult with:
        - success: bool indicating if analysis succeeded
        - data: Human-readable narrative with all scoring metrics
        - raw: Machine-readable dict with all metrics
        - error: Error message if unsuccessful
    """
    try:
        import pyrosetta as pr
        from pyrosetta.rosetta.protocols.relax import FastRelax
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
        from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
        from pyrosetta.rosetta.core.select.residue_selector import ChainSelector, OrResidueSelector
        from pyrosetta.rosetta.core.simple_metrics.metrics import TotalEnergyMetric, SasaMetric

        # Initialize PyRosetta with comprehensive flags matching original bc.py
        # -ex1 -ex2: Extra rotamer sampling for better packing
        # -use_input_sc: Use input sidechain conformations
        # -ignore_unrecognized_res: Don't crash on non-standard residues
        # -mute all: Reduce console noise
        pr.init("-ex1 -ex2 -use_input_sc -ignore_unrecognized_res -mute all", silent=True)

        # Set up scorefunction with disulfide scoring disabled BEFORE loading pose
        # This prevents errors from malformed disulfides in the PDB
        scorefxn = pr.get_fa_scorefxn()
        scorefxn.set_weight(pr.rosetta.core.scoring.ScoreType.dslf_fa13, 0.0)

        # Parse binder chains list
        binder_list = [c.strip() for c in binder_chains.split(",")]

        # Load pose - follow original bc.py approach: use full pose without chain extraction
        # Chain extraction breaks polymer bonds when there are modified terminal residues
        working_pdb = pdb_path
        if pdb_path.lower().endswith(".cif"):
            working_pdb = _convert_cif_to_pdb(pdb_path)

        # Load pose from PDB (full pose, no chain extraction like original bc.py)
        pose = pr.pose_from_pdb(working_pdb)

        # Run FastRelax first (like original bc.py does)
        movemap = pr.rosetta.core.kinematics.MoveMap()
        movemap.set_chi(True)
        movemap.set_bb(True)
        movemap.set_jump(False)

        fast_relax = FastRelax()
        fast_relax.set_scorefxn(scorefxn)
        fast_relax.set_movemap(movemap)
        fast_relax.max_iter(200)
        fast_relax.min_type("lbfgs_armijo_nonmonotone")
        fast_relax.constrain_relax_to_start_coords(True)
        fast_relax.apply(pose)

        # Define interface using original chain IDs (e.g., "A_DE" for chains D,E)
        binder_str = "".join(binder_list)
        interface_definition = f"{target_chain}_{binder_str}"

        # Set up InterfaceAnalyzerMover with comprehensive scoring
        iam = InterfaceAnalyzerMover()
        iam.set_interface(interface_definition)
        # Use the same modified scorefunction (with dslf_fa13 disabled)
        iam.set_scorefunction(scorefxn)
        iam.set_compute_packstat(True)
        iam.set_compute_interface_energy(True)
        iam.set_calc_dSASA(True)
        iam.set_compute_interface_sc(True)
        iam.set_pack_separated(True)
        iam.apply(pose)

        # Get interface data
        interface_data = iam.get_all_data()

        # Calculate binder total energy using selector
        # Use actual binder chain IDs (e.g., "D", "E", or "D,E" for multi-chain binder)
        if len(binder_list) > 1:
            selector = OrResidueSelector()
            for chain_id in binder_list:
                selector.add_residue_selector(ChainSelector(chain_id))
        else:
            selector = ChainSelector(binder_list[0])

        tem = TotalEnergyMetric()
        tem.set_scorefunction(scorefxn)
        tem.set_residue_selector(selector)
        binder_score = tem.calculate(pose)

        # Calculate binder SASA
        bsasa_m = SasaMetric(selector)
        binder_sasa = bsasa_m.calculate(pose)

        # Calculate buried unsatisfied hbonds filter
        # use_ddG_style is not compatible with poses > 3 chains, so we disable it
        # for multi-chain complexes like TCR-MHC-peptide (5 chains)
        use_ddG = "true" if pose.num_chains() <= 3 else "false"
        buns_xml = (
            '<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" '
            'scorefxn="scorefxn" ignore_surface_res="false" '
            f'use_ddG_style="{use_ddG}" probe_radius="1.1" '
            'burial_cutoff_apo="0.2" confidence="0" />'
        )
        buns_filter = XmlObjects.static_get_filter(buns_xml)
        delta_unsat_hbonds = buns_filter.report_sm(pose)

        # Calculate hotspot residues using the relaxed pose
        # Save the relaxed pose to a temp file for accurate hotspot calculation
        import tempfile
        import os as _os

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            temp_pdb = tmp.name
        pose.dump_pdb(temp_pdb)

        # Calculate hotspot residues for each binder chain against target
        total_hotspots = 0
        hotspot_details = []
        for binder_chain in binder_list:
            hotspots = _hotspot_residues(temp_pdb, target_chain, binder_chain)
            total_hotspots += len(hotspots)
            for resnum, info in hotspots.items():
                hotspot_details.append(
                    {
                        "chain": binder_chain,
                        "resnum": resnum,
                        "resname": info["res"],
                        "n_contacts": len(info["contacts"]),
                    }
                )

        # Clean up temp file
        try:
            _os.remove(temp_pdb)
        except OSError:
            pass

        # Extract core metrics
        interface_dG = iam.get_interface_dG()
        shape_complementarity = interface_data.sc_value
        packstat = iam.get_interface_packstat()
        delta_sasa = iam.get_interface_delta_sasa()
        interface_hbonds = interface_data.interface_hbonds

        # Calculate interface fraction
        interface_fraction = (delta_sasa / binder_sasa * 100) if binder_sasa > 0 else 0

        # Build raw results dict
        raw = {
            "pdb_path": pdb_path,
            "target_chain": target_chain,
            "binder_chains": binder_chains,
            "relaxed": relax_structure,
            "binder_score": round(binder_score, 2),
            "binder_sasa": round(binder_sasa, 2),
            "interface_dG": round(interface_dG, 2),
            "shape_complementarity": round(shape_complementarity, 4),
            "packstat": round(packstat, 4),
            "delta_sasa": round(delta_sasa, 2),
            "interface_fraction_percent": round(interface_fraction, 2),
            "interface_hbonds": interface_hbonds,
            "delta_unsat_hbonds": round(delta_unsat_hbonds, 2),
            "n_hotspot_residues": total_hotspots,
            "hotspot_details": hotspot_details[:20],  # Limit to top 20 for raw output
        }

        # Build human-readable narrative
        narrative_parts = [
            f"Interface scoring results for {os.path.basename(pdb_path)}:",
            f"Target chain: {target_chain}, Binder chain(s): {binder_chains}",
            "",
            "Core Metrics:",
            f"  Interface dG: {interface_dG:.2f} REU",
            f"  Shape Complementarity (sc): {shape_complementarity:.4f}",
            f"  Packstat: {packstat:.4f}",
            f"  Delta SASA: {delta_sasa:.2f} A^2",
            f"  Interface Fraction: {interface_fraction:.1f}% of binder surface",
            "",
            "Binder Properties:",
            f"  Binder Total Energy: {binder_score:.2f} REU",
            f"  Binder SASA: {binder_sasa:.2f} A^2",
            "",
            "Interface Contacts:",
            f"  Interface HBonds: {interface_hbonds}",
            f"  Buried Unsatisfied HBonds: {delta_unsat_hbonds:.2f}",
            f"  Hotspot Residues: {total_hotspots}",
        ]

        if hotspot_details:
            narrative_parts.append("")
            narrative_parts.append("Top Hotspot Residues:")
            # Sort by number of contacts
            sorted_hotspots = sorted(hotspot_details, key=lambda x: x["n_contacts"], reverse=True)[:5]
            for hs in sorted_hotspots:
                narrative_parts.append(f"  {hs['chain']}-{hs['resnum']} ({hs['resname']}): {hs['n_contacts']} contacts")

        narrative_parts.append("")
        narrative_parts.append(
            f"Interpretation: sc={shape_complementarity:.4f} "
            f"({'good' if shape_complementarity > 0.6 else 'moderate' if shape_complementarity > 0.5 else 'poor'} "
            f"shape complementarity), packstat={packstat:.4f} "
            f"({'good' if packstat > 0.6 else 'moderate' if packstat > 0.5 else 'poor'} packing)"
        )

        narrative = "\n".join(narrative_parts)

        return ToolResult(success=True, data=narrative, raw=raw)

    except ImportError as e:
        return ToolResult(success=False, data=_get_pyrosetta_install_message(), raw={}, error=f"ImportError: {str(e)}")
    except Exception as e:
        error_str = str(e).lower()
        if "pyrosetta" in error_str or "pose.cc" in error_str or "initialize" in error_str:
            # PDB has structural issues - suggest preprocessing
            preprocessing_tip = (
                f"PyRosetta could not initialize the pose from {pdb_path}. "
                "This usually means the PDB file has structural issues (duplicate atoms, "
                "missing residues, chain breaks, or non-standard residues).\n\n"
                "Suggestions:\n"
                "1. Use the 'renumber_pdb' tool to fix residue numbering and merge chains\n"
                "2. Use the 'fast_relax' tool to prepare the structure for analysis\n"
                "3. Verify the PDB file is valid using a tool like PDBfixer\n"
                f"\nOriginal error: {type(e).__name__}: {str(e)[:500]}"
            )
            return ToolResult(success=False, data=preprocessing_tip, raw={}, error=str(e))
        return ToolResult(
            success=False,
            data=f"Error scoring interface for {pdb_path}: {type(e).__name__}: {str(e)}",
            raw={},
            error=str(e),
        )
