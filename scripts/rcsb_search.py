#!/usr/bin/env python3
"""
RCSB Search Module for candidate PDB structure discovery.

Uses the RCSB Search API at https://search.rcsb.org/search to search for
PDB structures meeting specific criteria.
"""

import json
import time
from typing import Any

import click
import requests


RCSB_SEARCH_API = "https://search.rcsb.org/rcsbsearch/v2/query"

# Rate limiting: 5 requests/sec -> 0.2 seconds between requests
RATE_LIMIT_DELAY = 0.2

# Allowed journal names (partial matches for RCSB query)
JOURNALS = [
    "Nature",
    "Science",
    "Cell",
    "Nature Structural & Molecular Biology",
    "NSMB",
    "Structure",
    "eLife",
    "PNAS",
    "Molecular Cell",
    "Journal of Molecular Biology",
    "JMB",
    "Journal of Biological Chemistry",
    "JBC",
]

# Journal abbreviations for RCSB query
JOURNAL_ABBREV = {
    "Nature": "Nature",
    "Science": "Science",
    "Cell": "Cell",
    "NSMB": "Nat Struct Mol Biol",
    "Structure": "Structure",
    "eLife": "Elife",
    "PNAS": "Proc Natl Acad Sci U S A",
    "Molecular Cell": "Mol Cell",
    "JMB": "J Mol Biol",
    "JBC": "J Biol Chem",
}


def search_rcsb_candidates(
    resolution_max: float = 2.5,
    method: str | list[str] = None,
    min_year: int = 2015,
    max_year: int = 2025,
    polymer_count_min: int = 1,
    polymer_count_max: int = 6,
    journals: list[str] = None,
    limit: int = 1000,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """
    Search for candidate PDB structures meeting specified criteria.

    Uses simple text queries and filters results in Python since the RCSB
    Search API v2 doesn't support complex boolean queries with all operators.

    Args:
        resolution_max: Maximum resolution in Angstroms (default: 2.5)
        method: Experimental method - "X-RAY" or "CRYOEM" or list of both
        min_year: Minimum release year (default: 2015)
        max_year: Maximum release year (default: 2025)
        polymer_count_min: Minimum number of polymer entities (default: 1)
        polymer_count_max: Maximum number of polymer entities (default: 6)
        journals: List of journal names to include (default: all allowed journals)
        limit: Maximum number of results to return (default: 1000)
        offset: Offset for pagination (default: 0)

    Returns:
        List of dicts containing: pdb_id, pmid, resolution, n_chains, journal, organism
    """
    if journals is None:
        journals = JOURNALS

    # Normalize method parameter
    method_list = []
    if method is None:
        method_list = ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"]
    elif isinstance(method, str):
        method_list = [method.upper().replace("X-RAY", "X-RAY DIFFRACTION").replace("CRYOEM", "ELECTRON MICROSCOPY")]
    else:
        for m in method:
            m_upper = m.upper()
            if "X-RAY" in m_upper or "XRAY" in m_upper:
                method_list.append("X-RAY DIFFRACTION")
            elif "CRYO" in m_upper or "EM" in m_upper:
                method_list.append("ELECTRON MICROSCOPY")

    results = []
    current_offset = offset

    # Fetch candidates in batches using simple resolution query
    while len(results) < limit:
        # Use simple text query for resolution
        query = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less",
                "value": resolution_max,
            },
        }

        # Always fetch 100 entries at a time for efficiency, even if limit is small
        # This helps us find matching entries faster by covering more ground
        batch_size = 100

        payload = {
            "query": query,
            "return_type": "entry",
            "request_options": {
                "paginate": {
                    "start": current_offset,
                    "rows": batch_size,
                },
                "sort": [{"sort_by": "score", "direction": "desc"}],
                "scoring_strategy": "combined",
            },
            "request_info": {
                "query_id": "benchmark-search",
            },
        }

        try:
            response = requests.post(
                RCSB_SEARCH_API,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            search_result = response.json()
        except requests.RequestException as e:
            print(f"Search API error: {e}")
            break

        # Extract entry IDs from response
        entries = _parse_search_response(search_result)

        if not entries:
            break

        # Fetch details for each entry with rate limiting and filter
        for entry in entries:
            time.sleep(RATE_LIMIT_DELAY)

            entry_id = entry.get("entry_id")
            if not entry_id:
                continue

            details = get_structure_details(entry_id)

            if not details:
                continue

            # Filter by criteria that couldn't be filtered in search API
            pmid = details.get("pmid")
            if not pmid:
                continue  # Skip entries without PubMed ID

            resolution = details.get("resolution")
            if resolution and resolution >= resolution_max:
                continue  # Skip entries above resolution threshold

            exptl_method = details.get("method", "")
            method_match = any(m.lower() in exptl_method.lower() for m in method_list)
            if method_list and not method_match:
                continue  # Skip entries not matching method filter

            # Filter by year (disabled for now - RCSB Search API v2 doesn't support date sorting/filtering well)
            # This is too slow to filter client-side since entries aren't sorted by date
            # release_year = details.get("release_year", 0)
            # if release_year:
            #     if release_year < min_year or release_year > max_year:
            #         continue

            # Filter by polymer count
            n_chains = details.get("n_chains", 0)
            if n_chains:
                if n_chains < polymer_count_min or n_chains > polymer_count_max:
                    continue

            # Filter by journal
            journal = details.get("journal", "") or ""
            journal_match = any(j.lower() in journal.lower() or journal.lower() in j.lower() for j in journals)
            if journals and not journal_match:
                continue

            results.append(details)

            if len(results) >= limit:
                break

        # Check if there are more results to fetch
        # If we got fewer entries than requested, we've reached the end (unless we found some results)
        if len(entries) < 100 and results:
            break

        # If we got fewer than 100 entries and found no results yet, continue to next batch
        if len(entries) < 100 and not results:
            current_offset += len(entries)
            continue

        # If we processed some entries but none passed filters, continue to next batch
        if not results:
            current_offset += len(entries)
            continue

        current_offset += len(entries)

    return results[:limit]


def _parse_search_response(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse RCSB search API response to extract entry IDs."""
    entries = []
    try:
        result_set = response.get("result_set", [])
        for item in result_set:
            entries.append({"entry_id": item.get("identifier")})
    except (KeyError, TypeError):
        pass
    return entries


def get_structure_details(pdb_id: str) -> dict[str, Any] | None:
    """
    Fetch detailed metadata for a specific PDB structure.

    Args:
        pdb_id: 4-character PDB identifier (e.g., '1ABC')

    Returns:
        Dict containing: pdb_id, pmid, resolution, n_chains, journal, organism, method
        Returns None if the structure is not found or an error occurs.
    """
    return get_entry_full(pdb_id)


@click.command()
@click.option(
    "--resolution-max",
    type=float,
    default=2.5,
    help="Maximum resolution in Angstroms.",
)
@click.option(
    "--method",
    type=str,
    default=None,
    help="Experimental method: X-RAY or CRYOEM (comma-separated for multiple).",
)
@click.option(
    "--min-year",
    type=int,
    default=2015,
    help="Minimum release year.",
)
@click.option(
    "--max-year",
    type=int,
    default=2025,
    help="Maximum release year.",
)
@click.option(
    "--limit",
    type=int,
    default=100,
    help="Maximum number of results.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file for results (JSON).",
)
def cli(
    resolution_max: float,
    method: str | None,
    min_year: int,
    max_year: int,
    limit: int,
    output: str | None,
) -> None:
    """Search RCSB PDB for candidate structures.

    Examples:

        Search for cryo-EM structures:
            rcsb_search.py --method CRYOEM --limit 50

        Search for high-resolution X-ray structures:
            rcsb_search.py --method X-RAY --resolution-max 1.5
    """
    # Parse method if provided
    method_list: str | list[str] | None = None
    if method:
        method_list = [m.strip() for m in method.split(",")]

    results = search_rcsb_candidates(
        resolution_max=resolution_max,
        method=method_list,
        min_year=min_year,
        max_year=max_year,
        limit=limit,
    )

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"Results written to {output}")
    else:
        click.echo(f"Found {len(results)} results:")
        for result in results:
            click.echo(f"  {result.get('pdb_id', 'N/A')} - "
                       f"res={result.get('resolution', 'N/A')} - "
                       f"pmid={result.get('pmid', 'N/A')}")


if __name__ == "__main__":
    cli()


def get_entry_full(pdb_id: str) -> dict[str, Any] | None:
    """
    Fetch full entry data from RCSB.

    Args:
        pdb_id: 4-character PDB identifier

    Returns:
        Full entry data as dict or None if not found.
    """
    time.sleep(RATE_LIMIT_DELAY)

    # Use the PDB API to get full entry data
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        entry_data = response.json()

        # Extract relevant fields
        pdb_id_out = entry_data.get("rcsb_id", pdb_id).upper()

        # Get experimental method
        exptl = entry_data.get("exptl", [{}])
        if isinstance(exptl, list) and len(exptl) > 0:
            exptl = exptl[0]
        method = exptl.get("method", "") if isinstance(exptl, dict) else ""

        # Get release year
        release_year = 0
        accession_info = entry_data.get("rcsb_accession_info", {})
        if isinstance(accession_info, dict):
            initial_release = accession_info.get("initial_release_date", "")
            if initial_release and len(initial_release) >= 4:
                try:
                    release_year = int(initial_release[:4])
                except ValueError:
                    pass

        # Get PubMed ID from citation
        pmid = None
        citation = entry_data.get("citation", [])
        if isinstance(citation, list) and len(citation) > 0:
            primary_citation = citation[0]
            if isinstance(primary_citation, dict):
                pmid = primary_citation.get("pdbx_database_id_pub_med")

        # Get resolution
        resolution = entry_data.get("rcsb_entry_info", {}).get(
            "resolution_combined", [None]
        )
        if isinstance(resolution, list):
            resolution = resolution[0]

        # Get number of chains (polymer entities)
        n_chains = entry_data.get("rcsb_entry_info", {}).get(
            "polymer_entity_count", 0
        )

        # Get journal info from citation
        journal = None
        citation = entry_data.get("citation", [])
        if isinstance(citation, list) and len(citation) > 0:
            primary_citation = citation[0]
            if isinstance(primary_citation, dict):
                journal = primary_citation.get("rcsb_journal_abbrev") or primary_citation.get("journal_abbrev")

        # Get organism info
        organism = None
        polymer_entities = entry_data.get("polymer_entities", [])
        if polymer_entities:
            first_entity = polymer_entities[0]
            if isinstance(first_entity, dict):
                organism = first_entity.get("rcsb_entity_host_org_common_name")
                if not organism:
                    organism = first_entity.get("rcsb_entity_host_org_scientific_name")

        return {
            "pdb_id": pdb_id_out,
            "pmid": pmid,
            "method": method,
            "release_year": release_year,
            "resolution": resolution,
            "n_chains": n_chains,
            "journal": journal,
            "organism": organism,
        }

    except requests.RequestException:
        return None
