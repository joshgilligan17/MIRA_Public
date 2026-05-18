"""Tools for searching structural homologs using Foldseek."""

import io
import tarfile
import tempfile
import time
from typing import Optional

import gemmi
import requests

from structagent.registry import tool, ToolResult
from structagent.tools.structure_io import get_cached_structure


def _check_foldseek() -> bool:
    """Check if Foldseek service is available."""
    try:
        response = requests.head("https://search.foldseek.com", timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _write_chain_to_pdb(structure: gemmi.Structure, chain_id: str) -> str:
    """
    Extract a specific chain from a structure and write it as a PDB file.

    Args:
        structure: gemmi Structure object
        chain_id: Chain identifier to extract

    Returns:
        Path to the temporary PDB file
    """
    # Find the chain
    chain = None
    for model in structure:
        for ch in model:
            if ch.name == chain_id:
                chain = ch
                break
        if chain:
            break

    if chain is None:
        raise ValueError(f"Chain {chain_id} not found in structure")

    # Write to temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".pdb")
    with open(temp_fd, "w") as f:
        # Use gemmi to write the chain as PDB
        gemmi.write_pdb(f, chain)

    return temp_path


def _submit_search(pdb_path: str, databases: list[str]) -> str:
    """
    Submit a search to Foldseek and return the ticket ID.

    Args:
        pdb_path: Path to the PDB file
        databases: List of databases to search

    Returns:
        Ticket ID for polling
    """
    url = "https://search.foldseek.com/api/ticket"

    with open(pdb_path, "rb") as f:
        files = {"q": ("query.pdb", f, "application/x-pdb")}
        data = {"database": ",".join(databases)}
        response = requests.post(url, files=files, data=data, timeout=60)

    response.raise_for_status()
    result = response.json()

    if "id" not in result:
        raise ValueError(f"No ticket ID returned: {result}")

    return result["id"]


def _poll_for_results(ticket_id: str, max_wait: int = 120) -> str:
    """
    Poll for search completion.

    Args:
        ticket_id: The ticket ID from submission
        max_wait: Maximum seconds to wait

    Returns:
        Status string

    Raises:
        TimeoutError: If polling times out
    """
    url = f"https://search.foldseek.com/api/ticket/{ticket_id}"
    start_time = time.time()

    while time.time() - start_time < max_wait:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        result = response.json()

        status = result.get("status", "")
        if status == "COMPLETE":
            return status
        elif status == "ERROR":
            raise RuntimeError(f"Foldseek search failed: {result}")

        time.sleep(2)

    raise TimeoutError(f"Foldseek polling timed out after {max_wait} seconds")


def _download_results(ticket_id: str) -> bytes:
    """
    Download the results tarball.

    Args:
        ticket_id: The ticket ID from submission

    Returns:
        Raw tarball bytes
    """
    url = f"https://search.foldseek.com/api/result/download/{ticket_id}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def _parse_tarball(tarball_bytes: bytes) -> list[dict]:
    """
    Parse the TSV from the tarball archive.

    Args:
        tarball_bytes: Raw tarball bytes

    Returns:
        List of homolog records
    """
    homologs = []

    with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".tsv"):
                tsv_file = tar.extractfile(member)
                if tsv_file is None:
                    continue

                content = tsv_file.read().decode("utf-8")
                lines = content.strip().split("\n")

                if not lines:
                    continue

                # Parse header to find column indices
                header = lines[0].split("\t")

                # Foldseek TSV typically has columns like:
                # query,target,fident,alnlen,evalue,bits,alntmscore,taxid,taxname,taxlineage
                for line in lines[1:]:
                    fields = line.split("\t")
                    record = {}

                    for i, field_name in enumerate(header):
                        if i < len(fields):
                            record[field_name] = fields[i]

                    if record:
                        homologs.append(record)

                break

    return homologs


@tool(
    name="search_structural_homologs",
    toolset="analysis",
    description="Search for structural homologs of a protein chain using Foldseek. Submit a PDB chain, search against databases (pdb100, afdb50), poll until complete, and return parsed homolog results with alignment statistics.",
    parameters={
        "pdb_id": {"type": "string", "description": "4-character PDB identifier (e.g., '1ABC', '6VXX')"},
        "chain_id": {"type": "string", "description": "Chain identifier (e.g., 'A', 'B')"},
        "databases": {
            "type": "array",
            "description": "List of Foldseek databases to search against",
            "default": ["pdb100", "afdb50"],
            "items": {"type": "string"},
        },
        "max_hits": {"type": "integer", "description": "Maximum number of hits to return", "default": 10},
    },
    check_fn=_check_foldseek,
)
def search_structural_homologs(
    pdb_id: str, chain_id: str, databases: list[str] = None, max_hits: int = 10
) -> ToolResult:
    """
    Search for structural homologs using Foldseek.

    This tool:
    1. Loads the structure from PDB
    2. Extracts the specified chain to a temporary PDB file
    3. Submits the chain to Foldseek for structural search
    4. Polls until the search completes (max 120s)
    5. Downloads and parses the results
    6. Returns homolog information

    Args:
        pdb_id: 4-character PDB identifier
        chain_id: Chain identifier (e.g., 'A')
        databases: List of Foldseek databases (default: ["pdb100", "afdb50"])
        max_hits: Maximum number of hits to return (default: 10)

    Returns:
        ToolResult with:
        - success: bool indicating if search succeeded
        - data: Human-readable narrative description of homologs
        - raw: Dict with 'homologs' list and search metadata
    """
    if databases is None:
        databases = ["pdb100", "afdb50"]

    try:
        # Step 1: Get structure and write chain to temp file
        structure = get_cached_structure(pdb_id)
        pdb_path = _write_chain_to_pdb(structure, chain_id)

        try:
            # Step 2: Submit search
            ticket_id = _submit_search(pdb_path, databases)

            # Step 3: Poll for completion
            status = _poll_for_results(ticket_id)

            if status != "COMPLETE":
                return ToolResult(
                    success=False,
                    data=f"Search did not complete: {status}",
                    raw={},
                    error=f"Unexpected status: {status}",
                )

            # Step 4: Download results
            tarball_bytes = _download_results(ticket_id)

            # Step 5: Parse results
            all_homologs = _parse_tarball(tarball_bytes)

            # Apply max_hits limit
            homologs = all_homologs[:max_hits]

            # Step 6: Build narrative and raw data
            if not homologs:
                data = f"No structural homologs found for {pdb_id} chain {chain_id} in databases {databases}"
                raw = {
                    "pdb_id": pdb_id.upper(),
                    "chain_id": chain_id,
                    "databases": databases,
                    "homologs": [],
                    "total_hits": 0,
                }
            else:
                # Build narrative
                lines = [
                    f"Found {len(all_homologs)} structural homologs for {pdb_id.upper()} chain {chain_id} "
                    f"(searched against {', '.join(databases)})."
                ]

                for i, h in enumerate(homologs, 1):
                    target = h.get("target", "Unknown")
                    evalue = h.get("evalue", "N/A")
                    alnlen = h.get("alnlen", "N/A")
                    bits = h.get("bits", "N/A")
                    alntmscore = h.get("alntmscore", "N/A")

                    lines.append(
                        f"{i}. {target}: E-value={evalue}, alignment_length={alnlen}, "
                        f"bits={bits}, TM-score={alntmscore}"
                    )

                data = "\n".join(lines)

                raw = {
                    "pdb_id": pdb_id.upper(),
                    "chain_id": chain_id,
                    "databases": databases,
                    "homologs": homologs,
                    "total_hits": len(all_homologs),
                }

            return ToolResult(success=True, data=data, raw=raw)

        finally:
            # Clean up temp file
            import os

            try:
                os.unlink(pdb_path)
            except OSError:
                pass

    except TimeoutError as e:
        return ToolResult(success=False, data=f"Foldseek search timed out: {str(e)}", raw={}, error=str(e))
    except ValueError as e:
        return ToolResult(success=False, data=f"Error with structure or chain: {str(e)}", raw={}, error=str(e))
    except requests.RequestException as e:
        return ToolResult(success=False, data=f"Foldseek API error: {str(e)}", raw={}, error=str(e))
    except Exception as e:
        return ToolResult(
            success=False,
            data=f"Unexpected error during structural homolog search: {type(e).__name__}: {str(e)}",
            raw={},
            error=str(e),
        )
