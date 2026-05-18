"""Tools for structure relaxation using PyRosetta FastRelax."""

import os
from pathlib import Path

from structagent.registry import tool, ToolResult


def _check_pyrosetta() -> bool:
    """Check if pyrosetta is importable."""
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


@tool(
    name="fast_relax",
    toolset="structure",
    description="Perform FastRelax energy minimization on a protein structure using PyRosetta. "
    "Relaxes the structure by optimizing sidechain and backbone positions to reduce "
    "steric clashes and improve stereochemistry. Supports both PDB and mmCIF input formats.",
    parameters={
        "input_path": {"type": "string", "description": "Path to input structure file (.pdb or .cif)"},
        "sidechain_only": {
            "type": "boolean",
            "description": "If True, only relax sidechain chi angles; if False, also relax backbone phi/psi angles",
            "default": True,
        },
        "iterations": {"type": "integer", "description": "Maximum number of minimization iterations", "default": 250},
    },
    check_fn=_check_pyrosetta,
)
def fast_relax(input_path: str, sidechain_only: bool = True, iterations: int = 250) -> ToolResult:
    """
    Perform FastRelax on a protein structure using PyRosetta.

    FastRelax iteratively relaxes the structure by applying a series of
    small perturbations and minimizing the energy. When sidechain_only=True,
    only chi (sidechain) angles are moved, preserving the backbone conformation.
    When False, both backbone and sidechain angles are optimized.

    Args:
        input_path: Path to input PDB or mmCIF file
        sidechain_only: Whether to only relax sidechains (default True)
        iterations: Maximum minimization iterations (default 250)

    Returns:
        ToolResult with:
        - success: bool indicating if relaxation succeeded
        - data: human-readable narrative description of the relaxation
        - raw: dict with input_file, output_file, initial_score, final_score,
               score_improvement, sidechain_only, iterations, and n_residues
    """
    try:
        import pyrosetta
        from pyrosetta.rosetta.protocols.relax import FastRelax
        from pyrosetta.rosetta.core.kinematics import MoveMap

        input_path = os.path.abspath(input_path)

        if not os.path.exists(input_path):
            return ToolResult(
                success=False, data=f"Input file not found: {input_path}", raw={}, error=f"File not found: {input_path}"
            )

        # Determine if input is CIF and convert to PDB if needed
        input_stem = Path(input_path).stem
        working_pdb = input_path

        if input_path.lower().endswith(".cif"):
            # Convert CIF to PDB using pyrosetta
            pyrosetta.init("-ex1 -ex2 -use_input_sc -ignore_unrecognized_res -mute all", silent=True)
            # pose_from_file handles both PDB and CIF formats
            pose = pyrosetta.pose_from_file(input_path)
            working_pdb = os.path.join(os.path.dirname(input_path), f"{input_stem}_temp.pdb")
            pose.dump_pdb(working_pdb)

        # Initialize pyrosetta if not already done
        pyrosetta.init("-ex1 -ex2 -use_input_sc -ignore_unrecognized_res -mute all", silent=True)

        # Load structure
        pose = pyrosetta.pose_from_pdb(working_pdb)

        # Set up MoveMap based on sidechain_only setting
        movemap = MoveMap()
        if sidechain_only:
            movemap.set_bb(False)
            movemap.set_chi(True)
        else:
            movemap.set_bb(True)
            movemap.set_chi(True)
        movemap.set_jump(False)

        # Configure FastRelax
        scorefxn = pyrosetta.get_fa_scorefxn()
        fast_relax = FastRelax()
        fast_relax.set_scorefxn(scorefxn)
        fast_relax.set_movemap(movemap)
        fast_relax.max_iter(iterations)
        fast_relax.min_type("lbfgs_armijo_nonmonotone")
        fast_relax.constrain_relax_to_start_coords(True)

        # Get initial score
        initial_score = scorefxn(pose)

        # Apply FastRelax
        fast_relax.apply(pose)

        # Get final score
        final_score = scorefxn(pose)
        score_improvement = initial_score - final_score  # Lower is better for Rosetta scores

        # Generate output path
        output_path = os.path.join(os.path.dirname(input_path), f"{input_stem}_relaxed.pdb")
        pose.dump_pdb(output_path)

        # Clean up temp file if created from CIF
        if input_path.lower().endswith(".cif") and os.path.exists(working_pdb):
            if working_pdb != output_path:
                try:
                    os.remove(working_pdb)
                except OSError:
                    pass  # Ignore cleanup errors

        n_residues = pose.total_residue()

        # Build human-readable narrative
        if sidechain_only:
            relax_type = "sidechain-only"
        else:
            relax_type = "full backbone and sidechain"

        data_lines = [
            f"FastRelax completed successfully on {input_path}.",
            f"The structure ({n_residues} residues) was relaxed using {relax_type} relaxation",
            f"with a maximum of {iterations} iterations.",
            f"Initial score: {initial_score:.2f} REU, final score: {final_score:.2f} REU.",
            f"Score improvement: {score_improvement:.2f} REU (lower is better).",
            f"Output saved to: {output_path}",
        ]
        data = " ".join(data_lines)

        raw = {
            "input_file": input_path,
            "output_file": output_path,
            "initial_score": round(initial_score, 2),
            "final_score": round(final_score, 2),
            "score_improvement": round(score_improvement, 2),
            "sidechain_only": sidechain_only,
            "iterations": iterations,
            "n_residues": n_residues,
        }

        return ToolResult(success=True, data=data, raw=raw)

    except ImportError as e:
        return ToolResult(success=False, data=_get_pyrosetta_install_message(), raw={}, error=f"ImportError: {str(e)}")
    except Exception as e:
        return ToolResult(
            success=False, data=f"Error during FastRelax: {type(e).__name__}: {str(e)}", raw={}, error=str(e)
        )
