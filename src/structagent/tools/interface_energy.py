"""Interface energy analysis tools using PyRosetta and Bio.PDB."""

import os
import numpy as np
from scipy.spatial import cKDTree

from Bio.PDB import PDBParser, Selection

from structagent.registry import tool, ToolResult


# Three-letter to single-letter amino acid code map
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
    """Check if PyRosetta is importable."""
    try:
        import pyrosetta  # noqa: F401

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


def _hotspot_residues(pdb_path: str, target_chain: str, binder_chain: str, atom_distance_cutoff: float = 4.0) -> dict:
    """
    Find interface residues between binder and target chains using cKDTree.

    Uses Bio.PDB (matching original biopython_utils.py) rather than gemmi,
    since gemmi's het_flag filtering can exclude valid standard residues.

    Parameters
    ----------
    pdb_path : str
        Path to PDB or mmCIF file
    target_chain : str
        Target chain ID
    binder_chain : str
        Binder chain ID
    atom_distance_cutoff : float
        Heavy-atom distance cutoff in Angstroms

    Returns
    -------
    dict
        Dictionary mapping binder residue numbers to dicts with
        'res' (single-letter code) and 'contacts' (set of target residue numbers)
    """
    # Use Bio.PDB like original biopython_utils.py - gemmi's het_flag filtering
    # can exclude valid standard residues causing interface detection to fail
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)

    # Get atoms for both chains using Selection.unfold_entities (like original)
    binder_atoms = Selection.unfold_entities(structure[0][binder_chain], "A")
    binder_coords = np.array([atom.coord for atom in binder_atoms])

    target_atoms = Selection.unfold_entities(structure[0][target_chain], "A")
    target_coords = np.array([atom.coord for atom in target_atoms])

    # Build KD trees
    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)

    # Find pairs within cutoff
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


def _energy_interacting_residues(
    pdb_path: str, target_chain: str, binder_chain: str, interacting_residues: dict
) -> dict:
    """
    Compute per-residue Rosetta energies for interface residues.

    Parameters
    ----------
    pdb_path : str
        Path to PDB file
    target_chain : str
        Target chain ID
    binder_chain : str
        Binder chain ID
    interacting_residues : dict
        Dictionary of interacting residues from _hotspot_residues

    Returns
    -------
    dict
        Dictionary mapping binder residue numbers to total dG (REU)
    """
    import pyrosetta as pr

    # Initialize PyRosetta
    options = """
    -ignore_unrecognized_res
    -include_sugars
    -auto_detect_glycan_connections
    -maintain_links
    -alternate_3_letter_codes pdb_sugar
    -write_glycan_pdb_codes
    -ignore_zero_occupancy false
    -load_PDB_components false
    -no_fconfig
    -mute all
    """
    pr.init(" ".join(options.split("\n")), silent=True)

    # Load pose - handles both PDB and mmCIF formats
    if pdb_path.lower().endswith(".cif"):
        pose = pr.pose_from_file(pdb_path)
    else:
        pose = pr.pose_from_pdb(pdb_path)
    scorefxn = pr.get_fa_scorefxn()
    scorefxn(pose)
    pdb_info = pose.pdb_info()
    energy_graph = pose.energies().energy_graph()

    results_dg = {}
    for binder_res, data in interacting_residues.items():
        b_idx = pdb_info.pdb2pose(binder_chain, binder_res)
        t_indices = [
            pdb_info.pdb2pose(target_chain, r) for r in data["contacts"] if pdb_info.pdb2pose(target_chain, r) != 0
        ]

        if b_idx == 0 or not t_indices:
            continue

        dg = 0.0
        for t_idx in t_indices:
            edge = energy_graph.find_edge(b_idx, t_idx)
            if edge is not None:
                dg += edge.dot(scorefxn.weights())

        results_dg[binder_res] = dg

    return results_dg


def _get_all_chains(pdb_path: str) -> list:
    """Get list of all chain IDs in a PDB or mmCIF file."""
    import gemmi

    structure = gemmi.read_structure(pdb_path)
    return [chain.name for chain in structure[0]]


def _plot_interface_energies(
    energy_data: dict, label_data: dict, binder_chain: str, input_file: str, output_path: str
) -> None:
    """
    Generate matplotlib bar plot of interface energies.

    Parameters
    ----------
    energy_data : dict
        Dictionary mapping residue numbers to energies
    label_data : dict
        Dictionary mapping residue numbers to single-letter codes
    binder_chain : str
        Binder chain ID
    input_file : str
        Input PDB filename for title
    output_path : str
        Output plot filename
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    energies = energy_data
    labels = label_data

    if not energies:
        return

    sorted_resnums = sorted(energies.keys())
    x_labels = [f"{labels.get(r, '?')}{r}" for r in sorted_resnums]
    y_values = [energies[r] for r in sorted_resnums]
    x_pos = np.arange(len(sorted_resnums)) * 0.55

    pos_color = "#d73027"  # red - unfavorable
    neg_color = "#4575b4"  # blue - favorable

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(
        f"Interface Residue Energies\n{os.path.basename(input_file)} - Chain {binder_chain}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    colors = [pos_color if v >= 0 else neg_color for v in y_values]
    bars = ax.bar(x_pos, y_values, color=colors, edgecolor="black", linewidth=0.4, width=0.4)

    # Expand y limits for annotations
    y_range = max(y_values) - min(y_values) if max(y_values) != min(y_values) else 1.0
    ax.set_ylim(min(y_values) - 0.18 * y_range, max(y_values) + 0.18 * y_range)

    # Annotate each bar
    for bar, val in zip(bars, y_values):
        offset = 0.02 * y_range
        va = "bottom" if val >= 0 else "top"
        y_text = val + offset if val >= 0 else val - offset
        ax.text(
            bar.get_x() + bar.get_width() / 2, y_text, f"{val:.2f}", ha="center", va=va, fontsize=6.5, fontweight="bold"
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=60, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_ylabel("Rosetta Energy (REU)", fontsize=10)
    ax.set_xlabel("Binder Interface Residues", fontsize=10)
    ax.set_title(
        f"Binder Chain {binder_chain} - {len(sorted_resnums)} interface residues", fontsize=12, fontweight="bold"
    )
    ax.yaxis.grid(True, alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)

    # Legend
    legend_handles = [
        mpatches.Patch(color=neg_color, label="Favorable (< 0 REU)"),
        mpatches.Patch(color=pos_color, label="Unfavorable (>= 0 REU)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")


@tool(
    name="analyze_interface_energies",
    toolset="structure",
    description="Analyze per-residue interface energies between a binder chain "
    "and all other chains in a protein complex using PyRosetta. "
    "Identifies interface residues within a distance cutoff and "
    "computes Rosetta dG for each. Optionally generates a bar plot "
    "showing favorable (blue, negative REU) and unfavorable "
    "(red, positive REU) residues.",
    parameters={
        "type": "object",
        "properties": {
            "pdb_path": {"type": "string", "description": "Path to PDB or mmCIF file on disk"},
            "binder_chain": {"type": "string", "description": "Binder chain ID (e.g., 'A', 'B')"},
            "cutoff": {
                "type": "number",
                "default": 4.0,
                "description": "Heavy-atom distance cutoff in Angstroms for interface detection (default: 4.0)",
            },
            "plot_output": {
                "type": "string",
                "description": "Optional path for matplotlib plot output (e.g., 'interface_energies.png')",
            },
        },
        "required": ["pdb_path", "binder_chain"],
    },
    check_fn=_check_pyrosetta,
)
def analyze_interface_energies(
    pdb_path: str, binder_chain: str, cutoff: float = 4.0, plot_output: str = None
) -> ToolResult:
    """
    Analyze per-residue interface energies for a binder chain against all
    other chains in a protein complex.

    Uses Bio.PDB and scipy cKDTree to find interface residues within the
    specified cutoff distance, then PyRosetta to compute per-residue dG
    using the full-atom score function.

    Parameters
    ----------
    pdb_path : str
        Path to PDB file
    binder_chain : str
        Binder chain ID (e.g., 'A')
    cutoff : float, optional
        Heavy-atom distance cutoff in Angstroms (default 4.0)
    plot_output : str, optional
        If provided, generate and save a matplotlib bar plot to this path

    Returns
    -------
    ToolResult
        success: bool indicating if analysis succeeded
        data: Human-readable narrative with interface summary and per-residue energies
        raw: Machine-readable dict with per-residue energies, labels, and summary stats
    """
    try:
        # Validate inputs
        if not os.path.isfile(pdb_path):
            return ToolResult(
                success=False, data=f"PDB file not found: {pdb_path}", raw={}, error=f"File not found: {pdb_path}"
            )

        # Get all chains
        all_chains = _get_all_chains(pdb_path)
        if binder_chain not in all_chains:
            return ToolResult(
                success=False,
                data=f"Binder chain '{binder_chain}' not found. Available chains: {all_chains}",
                raw={},
                error=f"Chain {binder_chain} not found",
            )

        target_chains = [c for c in all_chains if c != binder_chain]
        if not target_chains:
            return ToolResult(
                success=False,
                data=f"No target chains found (chains in file: {all_chains})",
                raw={},
                error="No target chains",
            )

        # Find interface residues and compute energies for each target chain
        combined_contacts = {}  # resnum -> {"res": aa, "contacts": set()}
        energies = {}  # resnum -> summed energy

        for target_chain in target_chains:
            contacts = _hotspot_residues(pdb_path, target_chain, binder_chain, cutoff)
            if not contacts:
                continue

            # Merge residue labels
            for resnum, data in contacts.items():
                if resnum not in combined_contacts:
                    combined_contacts[resnum] = {"res": data["res"], "contacts": set()}
                combined_contacts[resnum]["contacts"].update(data["contacts"])

            # Compute per-residue energies
            chain_energies = _energy_interacting_residues(pdb_path, target_chain, binder_chain, contacts)
            for resnum, e in chain_energies.items():
                energies[resnum] = energies.get(resnum, 0.0) + e

        labels = {r: d["res"] for r, d in combined_contacts.items()}

        # Generate plot if requested
        if plot_output:
            _plot_interface_energies(energies, labels, binder_chain, pdb_path, plot_output)

        # Build narrative
        if not energies:
            data = (
                f"No interface residues found between chain {binder_chain} "
                f"and other chains at {cutoff} Angstrom cutoff."
            )
        else:
            total = sum(energies.values())
            favorable = sum(v for v in energies.values() if v < 0)
            unfavorable = sum(v for v in energies.values() if v >= 0)
            n_favorable = sum(1 for v in energies.values() if v < 0)
            n_unfavorable = sum(1 for v in energies.values() if v >= 0)

            lines = [
                f"Interface energy analysis for chain {binder_chain} against {len(target_chains)} target chain(s):",
                f"Interface residues: {len(energies)} ({n_favorable} favorable, {n_unfavorable} unfavorable)",
                f"Total energy: {total:.3f} REU (favorable: {favorable:.3f}, unfavorable: {unfavorable:.3f})",
                "",
            ]

            # Top favorable residues
            if energies:
                sorted_by_energy = sorted(energies.items(), key=lambda x: x[1])
                lines.append("Most favorable residues:")
                for resnum, e in sorted_by_energy[:5]:
                    aa = labels.get(resnum, "?")
                    lines.append(f"  {aa}{resnum}: {e:.3f} REU")

                # Most unfavorable
                if len(sorted_by_energy) > 5:
                    lines.append("Most unfavorable residues:")
                    for resnum, e in sorted_by_energy[-5:]:
                        aa = labels.get(resnum, "?")
                        lines.append(f"  {aa}{resnum}: {e:.3f} REU")

            if plot_output:
                lines.append("")
                lines.append(f"Plot saved to: {plot_output}")

            data = "\n".join(lines)

        # Build raw data
        raw = {
            "binder_chain": binder_chain,
            "target_chains": target_chains,
            "cutoff": cutoff,
            "per_residue_energies": {
                f"{labels.get(r, '?')}{r}": {"resnum": r, "energy": e, "label": labels.get(r, "?"), "favorable": e < 0}
                for r, e in energies.items()
            },
            "summary": {
                "total_energy": sum(energies.values()) if energies else 0.0,
                "favorable_sum": sum(v for v in energies.values() if v < 0),
                "unfavorable_sum": sum(v for v in energies.values() if v >= 0),
                "n_interface_residues": len(energies),
                "n_favorable": sum(1 for v in energies.values() if v < 0),
                "n_unfavorable": sum(1 for v in energies.values() if v >= 0),
            },
        }

        return ToolResult(success=True, data=data, raw=raw)

    except Exception as e:
        error_str = str(e).lower()
        if (
            "pyrosetta" in error_str
            or "pose.cc" in error_str
            or "initialize" in error_str
            or isinstance(e, ImportError)
        ):
            # PDB has structural issues - suggest preprocessing
            preprocessing_tip = (
                f"PyRosetta could not initialize the pose from {pdb_path}. "
                "This usually means the PDB file has structural issues.\n\n"
                "Suggestions:\n"
                "1. Use the 'renumber_pdb' tool to fix residue numbering issues\n"
                "2. Use the 'fast_relax' tool to prepare the structure\n"
                f"\nOriginal error: {type(e).__name__}: {str(e)[:500]}"
            )
            return ToolResult(success=False, data=preprocessing_tip, raw={}, error=str(e))
        return ToolResult(
            success=False,
            data=f"Error analyzing interface energies: {type(e).__name__}: {str(e)}",
            raw={},
            error=str(e),
        )
