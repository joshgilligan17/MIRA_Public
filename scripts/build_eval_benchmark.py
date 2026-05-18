"""Build evaluation benchmark dataset from canonical structures and RCSB candidates."""

import json
import os
import sys
from typing import Any

import click

# Add scripts directory to path for imports
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _scripts_dir)

# pylint: disable=import-error,wrong-import-position
from rcsb_search import get_structure_details, search_rcsb_candidates
from ncbi_fetch import fetch_pubmed_abstract


CANONICAL_SET = [
    # Enzyme mechanisms
    {"pdb_id": "1LYZ", "category": "active_site", "organism": "chicken",
     "description": "Hen egg-white lysozyme — GLU-35/ASP-52 catalytic pair"},
    {"pdb_id": "2CGA", "category": "active_site",
     "description": "Chymotrypsinogen — SER-195/HIS-57/ASP-102 catalytic triad"},
    {"pdb_id": "1A2P", "category": "active_site",
     "description": "Barnase — catalytic mechanism, well-studied interface with barstar"},

    # Allosteric mechanisms
    {"pdb_id": "2HHB", "category": "allostery",
     "description": "Deoxy hemoglobin — T-state, Perutz mechanism"},
    {"pdb_id": "1HHO", "category": "allostery",
     "description": "Oxy hemoglobin — R-state, compare with 2HHB"},
    {"pdb_id": "4AKE", "category": "allostery",
     "description": "Adenylate kinase open — classic hinge-bending enzyme"},

    # Protein-protein interfaces
    {"pdb_id": "1BRS", "category": "interface",
     "description": "Barnase-barstar — one of tightest known PPIs"},
    {"pdb_id": "1YCR", "category": "interface",
     "description": "p53-MDM2 — druggable PPI, hydrophobic triad"},
    {"pdb_id": "1AO7", "category": "interface",
     "description": "TCR-pMHC — A6 TCR/Tax/HLA-A2, canonical immune complex"},

    # Stability / fold features
    {"pdb_id": "1UBQ", "category": "stability",
     "description": "Ubiquitin — β-grasp fold, extremely stable and conserved"},
    {"pdb_id": "1CRN", "category": "stability",
     "description": "Crambin — tiny, 3 disulfides, very high resolution"},

    # De novo designs
    {"pdb_id": "8SK7", "category": "design",
     "description": "RFdiffusion HA binder — designed vs natural comparison"},

    # Placeholder entries (to be filled in by user)
    {"pdb_id": "PLACEHOLDER", "category": "tbd", "organism": "tbd",
     "description": "TODO: fill in"},
    {"pdb_id": "PLACEHOLDER", "category": "tbd", "organism": "tbd",
     "description": "TODO: fill in"},
    {"pdb_id": "PLACEHOLDER", "category": "tbd", "organism": "tbd",
     "description": "TODO: fill in"},
    {"pdb_id": "PLACEHOLDER", "category": "tbd", "organism": "tbd",
     "description": "TODO: fill in"},
    {"pdb_id": "PLACEHOLDER", "category": "tbd", "organism": "tbd",
     "description": "TODO: fill in"},
    {"pdb_id": "PLACEHOLDER", "category": "tbd", "organism": "tbd",
     "description": "TODO: fill in"},
    {"pdb_id": "PLACEHOLDER", "category": "tbd", "organism": "tbd",
     "description": "TODO: fill in"},
    {"pdb_id": "PLACEHOLDER", "category": "tbd", "organism": "tbd",
     "description": "TODO: fill in"},
    {"pdb_id": "PLACEHOLDER", "category": "tbd", "organism": "tbd",
     "description": "TODO: fill in"},
    {"pdb_id": "PLACEHOLDER", "category": "tbd", "organism": "tbd",
     "description": "TODO: fill in"},
    {"pdb_id": "PLACEHOLDER", "category": "tbd", "organism": "tbd",
     "description": "TODO: fill in"},
    {"pdb_id": "PLACEHOLDER", "category": "tbd", "organism": "tbd",
     "description": "TODO: fill in"},
    {"pdb_id": "PLACEHOLDER", "category": "tbd", "organism": "tbd",
     "description": "TODO: fill in"},
    {"pdb_id": "PLACEHOLDER", "category": "tbd", "organism": "tbd",
     "description": "TODO: fill in"},
    {"pdb_id": "PLACEHOLDER", "category": "tbd", "organism": "tbd",
     "description": "TODO: fill in"},
    {"pdb_id": "PLACEHOLDER", "category": "tbd", "organism": "tbd",
     "description": "TODO: fill in"},
    {"pdb_id": "PLACEHOLDER", "category": "tbd", "organism": "tbd",
     "description": "TODO: fill in"},
]


def build_canonical_record(entry: dict[str, Any]) -> dict[str, Any]:
    """Build a full record for a canonical set entry.

    Args:
        entry: Canonical set entry with pdb_id, category, organism, description.

    Returns:
        Record dict with all required fields (pmid/title/journal/abstract
        will be empty strings if not available).
    """
    return {
        "pdb_id": entry["pdb_id"],
        "pmid": "",
        "title": "",
        "journal": "",
        "abstract": "",
        "resolution": None,
        "n_chains": None,
        "organism": entry.get("organism", ""),
        "category": entry["category"],
        "description": entry["description"],
    }


def build_candidate_record(
    pdb_id: str,
    structure_details: dict[str, Any] | None,
    pubmed_data: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a full record for an RCSB candidate.

    Args:
        pdb_id: PDB ID.
        structure_details: Structure metadata from RCSB (may be None).
        pubmed_data: PubMed data from NCBI (may be None).

    Returns:
        Complete record dict for JSONL output.
    """
    return {
        "pdb_id": pdb_id,
        "pmid": pubmed_data.get("pmid", "") if pubmed_data else "",
        "title": pubmed_data.get("title", "") if pubmed_data else "",
        "journal": structure_details.get("journal", "") if structure_details else "",
        "abstract": pubmed_data.get("abstract", "") if pubmed_data else "",
        "resolution": structure_details.get("resolution") if structure_details else None,
        "n_chains": structure_details.get("n_chains") if structure_details else None,
        "organism": structure_details.get("organism", "") if structure_details else "",
    }


def write_jsonl_record(fh: Any, record: dict[str, Any]) -> None:
    """Write a single record as a JSONL line.

    Args:
        fh: File handle open for writing.
        record: Record dictionary to serialize.
    """
    fh.write(json.dumps(record) + "\n")


@click.command()
@click.option(
    "--output", "-o",
    default="benchmark_candidates.jsonl",
    help="Output JSONL file path.",
)
@click.option(
    "--max-candidates",
    default=200,
    help="Max number of RCSB candidates to fetch.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="If set, only test search without saving.",
)
def main(output: str, max_candidates: int, dry_run: bool) -> None:
    """Build evaluation benchmark dataset.

    Writes canonical set records first, then searches RCSB for additional
    candidates and writes them. Progress info is printed to stdout.
    """
    click.echo(f"Output file: {output}")
    click.echo(f"Max candidates: {max_candidates}")
    click.echo(f"Dry run: {dry_run}")
    click.echo("")

    # Write canonical set records
    click.echo("Writing canonical set records...")
    if dry_run:
        click.echo(f"  (dry-run) Would write {len(CANONICAL_SET)} canonical records")
        canonical_records = []
    else:
        with open(output, "w", encoding="utf-8") as fh:
            for entry in CANONICAL_SET:
                record = build_canonical_record(entry)
                write_jsonl_record(fh, record)
                click.echo(f"  Wrote canonical: {entry['pdb_id']} [{entry['category']}]")
        canonical_records = None  # type: ignore[assignment]

    # Search RCSB for candidates
    click.echo("")
    click.echo(f"Searching RCSB for up to {max_candidates} candidates...")
    try:
        candidates = list(search_rcsb_candidates(limit=max_candidates))
    except Exception as exc:  # pylint: disable=broad-except
        click.echo(f"  RCSB search failed: {exc}", err=True)
        candidates = []

    click.echo(f"  Found {len(candidates)} candidate PDB IDs")

    if dry_run:
        click.echo(f"  (dry-run) Would write {len(candidates)} candidate records")
        return

    if not candidates:
        click.echo("  No candidates to write.")
        return

    # Append candidate records to JSONL
    with open(output, "a", encoding="utf-8") as fh:
        for i, candidate in enumerate(candidates, start=1):
            pdb_id = candidate.get("pdb_id") if isinstance(candidate, dict) else candidate
            click.echo(f"  Processing candidate {i}/{len(candidates)}: {pdb_id}")

            # Get structure details from RCSB
            structure_details = None
            try:
                structure_details = get_structure_details(pdb_id)
            except Exception as exc:  # pylint: disable=broad-except
                click.echo(f"    Warning: failed to get structure details: {exc}")

            # Get PubMed data if PMID is available
            pubmed_data = None
            pmid = structure_details.get("pmid") if structure_details else None
            if pmid:
                try:
                    pubmed_data = fetch_pubmed_abstract(pmid)
                except Exception as exc:  # pylint: disable=broad-except
                    click.echo(f"    Warning: failed to fetch PubMed {pmid}: {exc}")

            # Build and write record
            record = build_candidate_record(pdb_id, structure_details, pubmed_data)
            write_jsonl_record(fh, record)

    click.echo("")
    click.echo(f"Done! Wrote canonical + {len(candidates)} candidates to {output}")


if __name__ == "__main__":
    main()
