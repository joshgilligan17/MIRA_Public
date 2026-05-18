"""Tools for normal mode analysis of protein structures using ProDy."""

import os

import numpy as np

from structagent.registry import tool, ToolResult
from structagent.tools.structure_io import get_structure


def _check_prody() -> bool:
    """Check if ProDy is available for import."""
    try:
        import prody

        return True
    except ImportError:
        return False


def _coerce_positive_int(value, param_name: str) -> int:
    """Coerce a value to a positive integer."""
    if value is None:
        raise ValueError(f"{param_name} is required")
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{param_name} must be an integer, got {type(value).__name__}")
    if int_value < 1:
        raise ValueError(f"{param_name} must be a positive integer, got {int_value}")
    return int_value


def _structure_label(pdb_id: str = None, pdb_path: str = None) -> str:
    """Return a compact label for user-facing dynamics output."""
    if pdb_id:
        return pdb_id.upper()
    if pdb_path:
        return os.path.basename(pdb_path)
    return "unknown"


def gemmi_to_prody(structure, chain_id: str):
    """Convert a gemmi Structure to a ProDy AtomGroup with Calpha atoms.

    Args:
        structure: gemmi Structure object
        chain_id: Chain identifier string (e.g., 'A', 'B')

    Returns:
        Tuple of (prody_atomgroup, residue_info_list) where residue_info_list
        contains dicts with keys: resname, resnum, index
    """
    from prody import AtomGroup

    # Find the matching chain in the first model
    chain = None
    for ch in structure[0]:
        if ch.name == chain_id:
            chain = ch
            break

    if chain is None:
        raise ValueError(f"Chain '{chain_id}' not found in structure")

    # Extract all Calpha atoms
    ca_coords = []
    residue_info = []

    for idx, residue in enumerate(chain):
        for atom in residue:
            if atom.name == "CA":
                ca_coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                residue_info.append({"resname": residue.name, "resnum": residue.seqid.num, "index": idx})
                break  # Only one CA per residue

    if not ca_coords:
        raise ValueError(f"No Calpha atoms found in chain '{chain_id}'")

    # Build numpy array of coordinates
    coords = np.array(ca_coords, dtype=np.float64)

    # Create ProDy AtomGroup
    ag = AtomGroup("CA_atoms")
    ag.setCoords(coords)

    return ag, residue_info


def _find_hinge_residues(eigenvector: np.ndarray) -> list:
    """Find hinge residues from GNM slowest mode eigenvector.

    Hinge residues are identified as zero-crossings in the eigenvector,
    detected by sign changes between consecutive indices.

    Args:
        eigenvector: 1D numpy array of eigenvector values

    Returns:
        List of residue indices where zero-crossings occur
    """
    hinge_indices = []
    for i in range(1, len(eigenvector)):
        # Check for sign change
        if eigenvector[i - 1] * eigenvector[i] < 0:
            hinge_indices.append(i)
    return hinge_indices


@tool(
    name="compute_normal_modes",
    toolset="dynamics",
    description="Compute normal modes (ANM/GNM) for a protein chain and analyze dynamics including fluctuations and hinge residues.",
    parameters={
        "pdb_id": {
            "type": "string",
            "description": "4-character PDB identifier (e.g., '1ABC', '6VXX'). Use either this or pdb_path.",
        },
        "pdb_path": {"type": "string", "description": "Path to local PDB/mmCIF file. Use either this or pdb_id."},
        "chain_id": {"type": "string", "description": "Chain identifier (e.g., 'A', 'B')"},
        "n_modes": {"type": "integer", "description": "Number of modes to compute", "default": 20},
        "cutoff": {
            "type": "number",
            "description": "Cutoff distance for ANM Hessian and GNM Kirchhoff (in Angstroms)",
            "default": 15.0,
        },
    },
    check_fn=_check_prody,
)
def compute_normal_modes(
    pdb_id: str = None,
    pdb_path: str = None,
    chain_id: str = None,
    n_modes: int = 20,
    cutoff: float = 15.0,
) -> ToolResult:
    """Compute ANM and GNM normal modes for a protein chain.

    Uses ProDy to perform anisotropic network model (ANM) and Gaussian network
    model (GNM) analysis. Returns eigenvalues, square fluctuations, hinge
    residues, and mode shapes.

    Args:
        pdb_id: 4-character PDB identifier
        pdb_path: Path to local PDB/mmCIF file
        chain_id: Chain identifier (e.g., 'A', 'B')
        n_modes: Number of modes to compute (default 20)
        cutoff: Cutoff distance in Angstroms (default 15.0)

    Returns:
        ToolResult with:
        - success: bool indicating if computation succeeded
        - data: human-readable narrative description
        - raw: dict with eigenvalues, fluctuations dict, hinge_residues list,
                and mode_shapes list
    """
    try:
        from prody import ANM, GNM, calcSqFlucts

        structure = get_structure(pdb_id=pdb_id, pdb_path=pdb_path)
        pdb_label = _structure_label(pdb_id, pdb_path)

        # Convert to ProDy format
        ag, residue_info = gemmi_to_prody(structure, chain_id)

        # Coerce n_modes to positive integer
        n_modes = _coerce_positive_int(n_modes, "n_modes")

        # Build ANM Hessian
        anm = ANM("protein")
        anm.buildHessian(ag.getCoords(), cutoff=cutoff)

        # Build GNM Kirchhoff
        gnm = GNM("protein")
        gnm.buildKirchhoff(ag.getCoords())

        # Compute modes
        anm.calcModes(n_modes)
        gnm.calcModes(n_modes)

        # Get square fluctuations from ANM modes (already normalized [0,1])
        sqflucts = calcSqFlucts(anm[:n_modes])

        # Get eigenvalues (first 10)
        eigenvalues = anm.getEigvals()[:10]

        # Get GNM slowest mode for hinge residue detection
        gnm_modes = gnm[:]
        if len(gnm_modes) > 0:
            # eigenvectors are stored as columns in getEigvecs()
            eigenvector = gnm.getEigvecs()[:, 0]
            hinge_indices = _find_hinge_residues(eigenvector)
        else:
            hinge_indices = []

        # Build residue->fluctuation dict
        fluctuations = {}
        for i, res_info in enumerate(residue_info):
            key = f"{res_info['resname']}-{res_info['resnum']}"
            fluctuations[key] = float(sqflucts[i])

        # Get first 3 mode shapes (eigenvectors as columns)
        eigvecs = anm.getEigvecs()
        mode_shapes = []
        for i in range(min(3, eigvecs.shape[1])):
            mode_shapes.append(eigvecs[:, i].tolist())

        # Build narrative
        n_residues = len(residue_info)
        top_ev = eigenvalues[0] if len(eigenvalues) > 0 else 0
        hinge_res_str = ", ".join([str(idx) for idx in hinge_indices[:10]])
        if len(hinge_indices) > 10:
            hinge_res_str += f", and {len(hinge_indices) - 10} more"

        narrative = (
            f"Normal mode analysis was performed on chain {chain_id} of structure {pdb_label} "
            f"containing {n_residues} C-alpha atoms. "
            f"Using a cutoff of {cutoff:.1f} Angstroms, {n_modes} modes were computed with ANM "
            f"and GNM. The slowest mode has an eigenvalue of {top_ev:.4f}. "
            f"Square fluctuations were calculated showing relative flexibility across residues. "
            f"Potential hinge residues were identified at indices: {hinge_res_str if hinge_res_str else 'none identified'}. "
            f"Analysis complete."
        )

        raw = {
            "eigenvalues": eigenvalues.tolist() if hasattr(eigenvalues, "tolist") else list(eigenvalues),
            "fluctuations": fluctuations,
            "hinge_residues": hinge_indices,
            "mode_shapes": mode_shapes,
            "n_residues": n_residues,
            "n_modes_computed": n_modes,
            "cutoff": cutoff,
        }

        return ToolResult(success=True, data=narrative, raw=raw)

    except ValueError as e:
        return ToolResult(success=False, data=f"Failed to compute normal modes: {str(e)}", raw={}, error=str(e))
    except Exception as e:
        return ToolResult(
            success=False,
            data=f"Unexpected error computing normal modes: {type(e).__name__}: {str(e)}",
            raw={},
            error=str(e),
        )


@tool(
    name="compute_cross_correlations",
    toolset="dynamics",
    description="Compute cross-correlation matrix from ANM modes to identify correlated and anti-correlated residue pairs.",
    parameters={
        "pdb_id": {
            "type": "string",
            "description": "4-character PDB identifier (e.g., '1ABC', '6VXX'). Use either this or pdb_path.",
        },
        "pdb_path": {"type": "string", "description": "Path to local PDB/mmCIF file. Use either this or pdb_id."},
        "chain_id": {"type": "string", "description": "Chain identifier (e.g., 'A', 'B')"},
        "source_residue": {
            "type": "integer",
            "description": "Optional source residue number to extract correlations for a specific residue",
            "required": False,
        },
        "n_modes": {
            "type": "integer",
            "description": "Number of modes to use for cross-correlation calculation",
            "default": 20,
        },
    },
    check_fn=_check_prody,
)
def compute_cross_correlations(
    pdb_id: str = None,
    pdb_path: str = None,
    chain_id: str = None,
    source_residue: int = None,
    n_modes: int = 20,
) -> ToolResult:
    """Compute cross-correlation matrix from ANM modes.

    Uses ProDy to compute ANM modes and then calculates the cross-correlation
    matrix between all pairs of C-alpha atoms. If source_residue is provided,
    extracts correlations for that residue. Otherwise, finds the most strongly
    correlated and anti-correlated pairs.

    Args:
        pdb_id: 4-character PDB identifier
        pdb_path: Path to local PDB/mmCIF file
        chain_id: Chain identifier (e.g., 'A', 'B')
        source_residue: Optional residue number to get correlations for
        n_modes: Number of modes to use (default 20)

    Returns:
        ToolResult with:
        - success: bool indicating if computation succeeded
        - data: human-readable narrative description
        - raw: dict with correlations dict, source_correlations dict (if source_residue), top_pairs list (if no source)
    """
    try:
        from prody import ANM, calcCrossCorr

        structure = get_structure(pdb_id=pdb_id, pdb_path=pdb_path)
        pdb_label = _structure_label(pdb_id, pdb_path)

        # Convert to ProDy format
        ag, residue_info = gemmi_to_prody(structure, chain_id)

        # Coerce n_modes to positive integer
        n_modes = _coerce_positive_int(n_modes, "n_modes")

        # Build ANM Hessian and compute modes
        anm = ANM("protein")
        anm.buildHessian(ag.getCoords(), cutoff=15.0)
        anm.calcModes(n_modes)

        # Compute cross-correlation matrix
        crosscorr = calcCrossCorr(anm[:n_modes])

        # Build residue key lookup
        def res_key(info):
            return f"{info['resname']}-{info['resnum']}"

        n_residues = len(residue_info)

        if source_residue is not None:
            # Find the index of the source residue
            source_idx = None
            for i, res_info in enumerate(residue_info):
                if res_info["resnum"] == source_residue:
                    source_idx = i
                    break

            if source_idx is None:
                return ToolResult(
                    success=False,
                    data=f"Source residue {source_residue} not found in chain {chain_id}",
                    raw={},
                    error=f"Residue {source_residue} not found",
                )

            # Extract correlations for source residue
            source_row = crosscorr[source_idx]
            source_correlations = {}

            for i, res_info in enumerate(residue_info):
                if i != source_idx:
                    key = res_key(res_info)
                    source_correlations[key] = float(source_row[i])

            # Find top 10 positively correlated and top 10 anti-correlated
            sorted_positive = sorted(source_correlations.items(), key=lambda x: x[1], reverse=True)[:10]
            sorted_negative = sorted(source_correlations.items(), key=lambda x: x[1])[:10]

            source_name = res_key(residue_info[source_idx])
            narrative = (
                f"Cross-correlation analysis was performed on chain {chain_id} of structure {pdb_label} "
                f"using {n_modes} ANM modes for {n_residues} C-alpha atoms. "
                f"Correlations were extracted for source residue {source_name}. "
                f"Top 10 positively correlated residues: {', '.join([f'{k} ({v:.3f})' for k, v in sorted_positive])}. "
                f"Top 10 anti-correlated residues: {', '.join([f'{k} ({v:.3f})' for k, v in sorted_negative])}."
            )

            raw = {
                "correlations": source_correlations,
                "source_correlations": source_correlations,
                "source_residue": source_name,
                "top_positive": sorted_positive,
                "top_anti": sorted_negative,
                "n_residues": n_residues,
                "n_modes": n_modes,
            }

        else:
            # Find top 20 most strongly correlated non-adjacent pairs (|i-j| > 5)
            # and top 20 most anti-correlated pairs
            all_pairs = []
            for i in range(n_residues):
                for j in range(i + 1, n_residues):
                    if abs(i - j) > 5:  # Non-adjacent
                        all_pairs.append((i, j, crosscorr[i, j]))

            # Sort by absolute correlation for strongest, and by value for anti
            sorted_by_abs = sorted(all_pairs, key=lambda x: abs(x[2]), reverse=True)
            sorted_by_value = sorted(all_pairs, key=lambda x: x[2])

            top_correlated = []
            for idx, jdx, corr in sorted_by_abs[:20]:
                top_correlated.append((res_key(residue_info[idx]), res_key(residue_info[jdx]), float(corr)))

            top_anti = []
            for idx, jdx, corr in sorted_by_value[:20]:
                top_anti.append((res_key(residue_info[idx]), res_key(residue_info[jdx]), float(corr)))

            # Build full correlations dict
            correlations = {}
            for i in range(n_residues):
                for j in range(i + 1, n_residues):
                    key1 = res_key(residue_info[i])
                    key2 = res_key(residue_info[j])
                    correlations[f"{key1}:{key2}"] = float(crosscorr[i, j])

            narrative = (
                f"Cross-correlation analysis was performed on chain {chain_id} of structure {pdb_label} "
                f"using {n_modes} ANM modes for {n_residues} C-alpha atoms. "
                f"Top 20 most strongly correlated non-adjacent residue pairs (|i-j| > 5): "
                f"{', '.join([f'({a}-{b}: {c:.3f})' for a, b, c in top_correlated[:5]])}. "
                f"Top 20 most anti-correlated pairs: "
                f"{', '.join([f'({a}-{b}: {c:.3f})' for a, b, c in top_anti[:5]])}."
            )

            raw = {
                "correlations": correlations,
                "top_correlated_pairs": top_correlated,
                "top_anti_correlated_pairs": top_anti,
                "top_pairs": top_correlated + top_anti,
                "n_residues": n_residues,
                "n_modes": n_modes,
            }

        return ToolResult(success=True, data=narrative, raw=raw)

    except ValueError as e:
        return ToolResult(success=False, data=f"Failed to compute cross-correlations: {str(e)}", raw={}, error=str(e))
    except Exception as e:
        return ToolResult(
            success=False,
            data=f"Unexpected error computing cross-correlations: {type(e).__name__}: {str(e)}",
            raw={},
            error=str(e),
        )


@tool(
    name="predict_hinge_regions",
    toolset="dynamics",
    description="Predict hinge regions in a protein chain using Gaussian Network Model (GNM) analysis. "
    "Identifies zero-crossings in slow mode eigenvectors as potential hinge points that "
    "separate protein domains or subdomains.",
    parameters={
        "type": "object",
        "properties": {
            "pdb_id": {
                "type": "string",
                "description": "4-character PDB identifier (e.g., '1ABC', '6VXX'). Use either this or pdb_path.",
            },
            "pdb_path": {"type": "string", "description": "Path to local PDB/mmCIF file. Use either this or pdb_id."},
            "chain_id": {"type": "string", "description": "Chain identifier (e.g., 'A', 'B')"},
            "n_modes": {
                "type": "integer",
                "description": "Number of slow modes to analyze for hinge detection",
                "default": 3,
            },
        },
        "required": ["chain_id"],
    },
    check_fn=_check_prody,
)
def predict_hinge_regions(
    pdb_id: str = None,
    pdb_path: str = None,
    chain_id: str = None,
    n_modes: int = 3,
) -> ToolResult:
    """Predict hinge regions in a protein using GNM analysis.

    Uses the Gaussian Network Model (GNM) to compute normal modes and identifies
    hinge regions by finding zero-crossings in the eigenvectors of the slowest
    (most collective) modes. These zero-crossings often correspond to hinge points
    that separate domains or subdomains in protein structures.

    Args:
        pdb_id: 4-character PDB identifier
        pdb_path: Path to local PDB/mmCIF file
        chain_id: Chain identifier (e.g., 'A')
        n_modes: Number of slow modes to analyze (default 3)

    Returns:
        ToolResult with:
        - success: bool indicating if analysis succeeded
        - data: human-readable narrative description of predicted hinges
        - raw: dict with hinges list and metadata
    """
    try:
        from prody import GNM

        structure = get_structure(pdb_id=pdb_id, pdb_path=pdb_path)
        pdb_label = _structure_label(pdb_id, pdb_path)
        ag, residue_info = gemmi_to_prody(structure, chain_id)

        # Coerce n_modes to positive integer
        n_modes = _coerce_positive_int(n_modes, "n_modes")

        n_residues = len(residue_info)

        if n_residues < 5:
            return ToolResult(
                success=False,
                data=f"Chain {chain_id} has too few residues ({n_residues}) for GNM analysis. Need at least 5 residues.",
                raw={},
            )

        # Build GNM Kirchhoff matrix from coordinates
        gnm = GNM("protein")
        gnm.buildKirchhoff(ag.getCoords())

        # Compute modes - get a few extra beyond requested for safety
        gnm.calcModes(n_modes=n_modes + 2)

        # Get eigenvectors (columns) and eigenvalues
        eigvecs = gnm.getEigvecs()
        eigvals = gnm.getEigvals()

        # Extract the slowest modes and find zero-crossings (hinge points)
        hinges = []

        for mode_idx in range(n_modes):
            eigenvector = eigvecs[:, mode_idx]
            eigenvalue = eigvals[mode_idx]

            # Find zero-crossings: indices where consecutive elements have opposite signs
            for j in range(len(eigenvector) - 1):
                if eigenvector[j] * eigenvector[j + 1] < 0:
                    # Zero-crossing found at index j (between residue j and j+1)
                    # Use the residue at j as the hinge point
                    if j < len(residue_info):
                        res_info = residue_info[j]
                        resnum = res_info["resnum"]

                        # Determine location in chain
                        position_fraction = j / max(1, n_residues - 1)
                        if position_fraction < 0.25:
                            location = "N-terminal domain"
                        elif position_fraction > 0.75:
                            location = "C-terminal domain"
                        else:
                            location = "mid-section"

                        # Simple loop conjecture: check if adjacent residues have high fluctuation
                        is_loop = False
                        if j > 0 and j < len(eigenvector) - 1:
                            max_fluctuation = max(abs(eigenvector[k]) for k in [j - 1, j, j + 1])
                            threshold = 0.5 * max(abs(eigenvector))
                            if max_fluctuation > threshold:
                                is_loop = True

                        # Determine what domains this hinge separates
                        if location == "mid-section":
                            separates = "separates N-terminal and C-terminal regions"
                        else:
                            separates = f"separates {location.lower()} from the rest of chain"

                        hinges.append(
                            {
                                "residue": resnum,
                                "resname": res_info["resname"],
                                "mode": mode_idx,
                                "separates": separates,
                                "location": location,
                                "is_loop_conjecture": is_loop,
                                "eigenvalue": float(eigenvalue),
                            }
                        )

        # Sort hinges by residue number
        hinges.sort(key=lambda x: x["residue"])

        # Build narrative description
        if hinges:
            # Deduplicate hinges at same residue (can occur from multiple modes)
            seen_residues = set()
            unique_hinges = []
            for h in hinges:
                if h["residue"] not in seen_residues:
                    seen_residues.add(h["residue"])
                    unique_hinges.append(h)

            hinge_descriptions = []
            for h in unique_hinges:
                loop_note = " (likely flexible loop)" if h["is_loop_conjecture"] else ""
                hinge_descriptions.append(
                    f"Residue {h['residue']} ({h['resname']}): mode {h['mode']} hinge in {h['location']}{loop_note}"
                )
            narrative = (
                f"GNM analysis of chain {chain_id} in {pdb_label} with {n_residues} residues "
                f"identified {len(unique_hinges)} potential hinge region(s) from the slowest {n_modes} modes. "
                + " ".join(hinge_descriptions)
                + ". "
                f"These hinges were detected as zero-crossings in the eigenvectors."
            )
        else:
            narrative = (
                f"GNM analysis of chain {chain_id} in {pdb_label} with {n_residues} residues "
                f"did not identify any clear hinge regions in the slowest {n_modes} modes. "
                f"This may indicate a rigid, single-domain structure or insufficient sampling."
            )

        raw = {
            "pdb_id": pdb_label,
            "chain_id": chain_id,
            "n_residues": n_residues,
            "n_modes_analyzed": n_modes,
            "hinges": hinges,
            "n_hinges_found": len(hinges),
        }

        return ToolResult(success=True, data=narrative, raw=raw)

    except ValueError as e:
        return ToolResult(
            success=False,
            data=f"Failed to predict hinge regions for {_structure_label(pdb_id, pdb_path)} chain {chain_id}: {str(e)}",
            raw={},
            error=str(e),
        )
    except Exception as e:
        return ToolResult(
            success=False, data=f"Error in GNM hinge analysis: {type(e).__name__}: {str(e)}", raw={}, error=str(e)
        )


@tool(
    name="compute_perturbation_response",
    toolset="dynamics",
    description="Perform perturbation response scanning (PRS) to predict how perturbing one residue "
    "(e.g., ligand binding) propagates through the structure. Uses ENM-based approximation "
    "of allosteric signal propagation. Returns which residues respond most strongly.",
    parameters={
        "type": "object",
        "properties": {
            "pdb_id": {
                "type": "string",
                "description": "4-character PDB identifier (e.g., '1ABC', '6VXX'). Use either this or pdb_path.",
            },
            "pdb_path": {"type": "string", "description": "Path to local PDB/mmCIF file. Use either this or pdb_id."},
            "chain_id": {"type": "string", "description": "Chain identifier (e.g., 'A', 'B')"},
            "source_residue": {
                "type": "integer",
                "description": "Residue number to perturb (e.g., ligand binding site)",
            },
            "n_modes": {"type": "integer", "description": "Number of modes to use for PRS calculation", "default": 20},
        },
        "required": ["chain_id", "source_residue"],
    },
    check_fn=_check_prody,
)
def compute_perturbation_response(
    pdb_id: str = None,
    pdb_path: str = None,
    chain_id: str = None,
    source_residue: int = None,
    n_modes: int = 20,
) -> ToolResult:
    """Perform perturbation response scanning (PRS) analysis.

    Predicts how a perturbation at one residue propagates through the structure
    using the ENM-based approach. The response at each residue is proportional
    to the magnitude of displacement when a unit force is applied at the source.

    Args:
        pdb_id: 4-character PDB identifier
        pdb_path: Path to local PDB/mmCIF file
        chain_id: Chain identifier (e.g., 'A', 'B')
        source_residue: Residue number to perturb
        n_modes: Number of modes to use (default 20)

    Returns:
        ToolResult with:
        - success: bool indicating if computation succeeded
        - data: human-readable narrative description
        - raw: dict with responses dict, pathway list, source_residue
    """
    try:
        from prody import ANM, calcPerturbResponse

        structure = get_structure(pdb_id=pdb_id, pdb_path=pdb_path)
        pdb_label = _structure_label(pdb_id, pdb_path)

        # Convert to ProDy format
        ag, residue_info = gemmi_to_prody(structure, chain_id)

        # Coerce n_modes to positive integer
        n_modes = _coerce_positive_int(n_modes, "n_modes")

        # Build ANM Hessian and compute modes
        anm = ANM("protein")
        anm.buildHessian(ag.getCoords(), cutoff=15.0)
        anm.calcModes(n_modes)

        # Compute full perturbation response matrix (returns tuple: matrix, effectiveness, sensitivity)
        prs_matrix = calcPerturbResponse(anm)[0]

        # Find source index
        source_idx = None
        for i, res_info in enumerate(residue_info):
            if res_info["resnum"] == source_residue:
                source_idx = i
                break

        if source_idx is None:
            return ToolResult(
                success=False,
                data=f"Source residue {source_residue} not found in chain {chain_id}",
                raw={},
                error=f"Residue {source_residue} not found",
            )

        # Extract responses for source residue from the PRS matrix
        # prs_matrix shape: (N, N) where prs_matrix[j, i] = response at j to perturbation at i
        source_responses = prs_matrix[:, source_idx]

        # Normalize to [0, 1]
        min_resp = source_responses.min()
        max_resp = source_responses.max()
        if max_resp > min_resp:
            normalized_responses = (source_responses - min_resp) / (max_resp - min_resp)
        else:
            normalized_responses = np.zeros_like(source_responses)

        # Build responses dict
        responses = {}
        for i, res_info in enumerate(residue_info):
            key = f"{res_info['resname']}-{res_info['resnum']}"
            responses[key] = float(normalized_responses[i])

        # Rank residues by response magnitude
        ranked = sorted(responses.items(), key=lambda x: x[1], reverse=True)
        top_10 = ranked[:10]

        # Trace propagation pathway: follow highest-response residues
        pathway = []
        visited = set()
        threshold = 0.3  # Only include residues with response > 0.3

        for res_key, resp_val in ranked:
            if resp_val < threshold:
                break
            if res_key not in visited:
                pathway.append(res_key)
                visited.add(res_key)

        source_name = f"{residue_info[source_idx]['resname']}-{residue_info[source_idx]['resnum']}"

        narrative = (
            f"Perturbation response scanning was performed on chain {chain_id} of structure {pdb_label} "
            f"using {n_modes} ANM modes for {len(residue_info)} C-alpha atoms. "
            f"Residue {source_name} was perturbed and responses were computed for all other residues. "
            f"Top 10 responding residues: {', '.join([f'{k} ({v:.3f})' for k, v in top_10])}. "
            f"Predicted propagation pathway (response > {threshold}): {' -> '.join(pathway) if pathway else 'none above threshold'}."
        )

        raw = {
            "responses": responses,
            "pathway": pathway,
            "source_residue": source_residue,
            "source_name": source_name,
            "top_responding": top_10,
            "n_residues": len(residue_info),
            "n_modes": n_modes,
            "threshold": threshold,
        }

        return ToolResult(success=True, data=narrative, raw=raw)

    except ValueError as e:
        return ToolResult(
            success=False, data=f"Failed to compute perturbation response: {str(e)}", raw={}, error=str(e)
        )
    except Exception as e:
        return ToolResult(
            success=False,
            data=f"Unexpected error computing perturbation response: {type(e).__name__}: {str(e)}",
            raw={},
            error=str(e),
        )
