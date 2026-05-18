"""Tools for retrieving and analyzing evolutionary conservation scores."""

import requests

from structagent.registry import tool, ToolResult
from structagent.tools.structure_io import get_cached_structure


CONSURF_DB_URL = "https://consurfdb.tau.ac.il"


def _check_consurfdb() -> bool:
    """Check if ConSurf-DB service is available."""
    try:
        response = requests.head(CONSURF_DB_URL, timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _classify_conservation(score: int) -> str:
    """Classify a ConSurf score into burial/exposure category."""
    if score <= 3:
        return "buried"
    elif score <= 6:
        return "intermediate"
    else:
        return "exposed"


@tool(
    name="get_conservation_scores",
    toolset="analysis",
    description="Retrieve evolutionary conservation scores for a protein structure from ConSurf-DB, classifying residues as buried (1-3), intermediate (4-6), or exposed (7-9). Falls back to RCSB GraphQL if ConSurf is unavailable.",
    parameters={
        "pdb_id": {"type": "string", "description": "4-character PDB identifier (e.g., '1ABC', '6VXX')"},
        "chain_id": {"type": "string", "description": "Chain identifier (e.g., 'A', 'B')"},
    },
    check_fn=_check_consurfdb,
)
def get_conservation_scores(pdb_id: str, chain_id: str) -> ToolResult:
    """
    Retrieve evolutionary conservation scores for a protein from ConSurf-DB.

    ConSurf-DB provides conservation scores based on evolutionary rate analysis.
    Scores range from 1 (fastest evolution, least conserved) to 9 (slowest
    evolution, most conserved).

    Classification:
    - Scores 1-3: Buried residues (low conservation)
    - Scores 4-6: Intermediate conservation
    - Scores 7-9: Exposed residues (high conservation)

    Parameters
    ----------
    pdb_id : str
        4-character PDB identifier
    chain_id : str
        Chain identifier (e.g., 'A')

    Returns
    -------
    ToolResult
        success: bool indicating if the operation succeeded
        data: Human-readable narrative description of conservation scores
        raw: Dict with 'scores' list containing per-residue conservation data
    """
    pdb_id = pdb_id.upper()
    chain_id = chain_id.upper()

    # Try ConSurf-DB first
    scores_data = _fetch_consurf_scores(pdb_id, chain_id)

    if scores_data is not None:
        # Build narrative from ConSurf data
        return _build_narrative(pdb_id, chain_id, scores_data)

    # Fall back to RCSB GraphQL
    return _fetch_rcsb_conservation(pdb_id, chain_id)


def _fetch_consurf_scores(pdb_id: str, chain_id: str) -> list[dict] | None:
    """Fetch conservation scores from ConSurf-DB."""
    url = f"{CONSURF_DB_URL}/results/{pdb_id}{chain_id}/consurf_summary.txt"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return _parse_consurf_file(response.text)
    except (requests.RequestException, ValueError):
        return None


def _parse_consurf_file(content: str) -> list[dict]:
    """Parse ConSurf summary file format.

    The file is space-delimited with columns:
    residue_number, AA, score (1-9), confidence, classification
    """
    scores = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        try:
            residue_number = int(parts[0])
            aa = parts[1]
            score = int(parts[2])
            confidence = float(parts[3])
            classification = parts[4]
        except (ValueError, IndexError):
            continue

        scores.append(
            {
                "residue_number": residue_number,
                "aa": aa,
                "score": score,
                "confidence": confidence,
                "classification": classification,
            }
        )

    return scores


def _fetch_rcsb_conservation(pdb_id: str, chain_id: str) -> ToolResult:
    """Fetch conservation data from RCSB GraphQL API as fallback."""
    query = (
        """
    {
        entry(entry_id: "%s") {
            polymer_entities {
                rcsb_conservation {
                    residue_number
                    aa
                    score
                    confidence
                }
            }
        }
    }
    """
        % pdb_id
    )

    try:
        response = requests.post("https://data.rcsb.org/graphql", json={"query": query}, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Extract conservation data from GraphQL response
        scores = []
        entries = data.get("data", {}).get("entry", {})
        polymer_entities = entries.get("polymer_entities", [])

        for entity in polymer_entities:
            consurf_data = entity.get("rcsb_conservation", [])
            for item in consurf_data:
                scores.append(
                    {
                        "residue_number": item.get("residue_number"),
                        "aa": item.get("aa"),
                        "score": item.get("score"),
                        "confidence": item.get("confidence"),
                        "classification": _classify_conservation(item.get("score", 5)),
                    }
                )

        if not scores:
            return ToolResult(
                success=False,
                data=f"No conservation data found in RCSB for PDB {pdb_id} chain {chain_id}",
                raw={},
                error="No conservation data available",
            )

        return _build_narrative(pdb_id, chain_id, scores)

    except requests.RequestException as e:
        return ToolResult(
            success=False, data=f"Failed to fetch conservation data from RCSB: {str(e)}", raw={}, error=str(e)
        )
    except (KeyError, ValueError) as e:
        return ToolResult(success=False, data=f"Error parsing RCSB GraphQL response: {str(e)}", raw={}, error=str(e))


def _build_narrative(pdb_id: str, chain_id: str, scores: list[dict]) -> ToolResult:
    """Build narrative and raw data from conservation scores list."""
    if not scores:
        return ToolResult(
            success=False,
            data=f"No conservation scores found for PDB {pdb_id} chain {chain_id}",
            raw={},
            error="Empty scores list",
        )

    # Classify scores
    buried = [s for s in scores if s["score"] <= 3]
    intermediate = [s for s in scores if 4 <= s["score"] <= 6]
    exposed = [s for s in scores if s["score"] >= 7]

    # Calculate average scores per category
    avg_buried = sum(s["score"] for s in buried) / len(buried) if buried else 0
    avg_intermediate = sum(s["score"] for s in intermediate) / len(intermediate) if intermediate else 0
    avg_exposed = sum(s["score"] for s in exposed) / len(exposed) if exposed else 0

    # Find most conserved residues (score >= 8)
    highly_conserved = [s for s in scores if s["score"] >= 8]
    highly_conserved.sort(key=lambda x: x["score"], reverse=True)

    # Find least conserved residues (score <= 2)
    least_conserved = [s for s in scores if s["score"] <= 2]
    least_conserved.sort(key=lambda x: x["score"])

    # Build narrative
    lines = [f"Conservation scores for PDB {pdb_id} chain {chain_id} ({len(scores)} residues):"]

    # Summary by category
    lines.append(f"  Buried (1-3): {len(buried)} residues, avg score {avg_buried:.1f}")
    lines.append(f"  Intermediate (4-6): {len(intermediate)} residues, avg score {avg_intermediate:.1f}")
    lines.append(f"  Exposed (7-9): {len(exposed)} residues, avg score {avg_exposed:.1f}")

    # Most conserved
    if highly_conserved:
        lines.append(f"\nMost conserved residues (score 8-9):")
        for s in highly_conserved[:5]:
            lines.append(
                f"  {s['aa']}-{s['residue_number']}: score {s['score']} (confidence {s.get('confidence', 'N/A')})"
            )
        if len(highly_conserved) > 5:
            lines.append(f"  ... and {len(highly_conserved) - 5} more")

    # Least conserved
    if least_conserved:
        lines.append(f"\nLeast conserved residues (score 1-2):")
        for s in least_conserved[:5]:
            lines.append(
                f"  {s['aa']}-{s['residue_number']}: score {s['score']} (confidence {s.get('confidence', 'N/A')})"
            )
        if len(least_conserved) > 5:
            lines.append(f"  ... and {len(least_conserved) - 5} more")

    data = "\n".join(lines)
    raw = {
        "scores": scores,
        "summary": {
            "total_residues": len(scores),
            "buried_count": len(buried),
            "intermediate_count": len(intermediate),
            "exposed_count": len(exposed),
            "avg_buried": round(avg_buried, 2),
            "avg_intermediate": round(avg_intermediate, 2),
            "avg_exposed": round(avg_exposed, 2),
            "highly_conserved_count": len(highly_conserved),
            "least_conserved_count": len(least_conserved),
        },
    }

    return ToolResult(success=True, data=data, raw=raw)
