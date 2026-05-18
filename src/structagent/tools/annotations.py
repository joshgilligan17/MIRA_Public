"""Tools for retrieving functional annotations from RCSB."""

import requests

from structagent.registry import tool, ToolResult
from structagent.tools.structure_io import get_cached_structure


RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"


# GraphQL queries for functional annotations

INSTANCE_FEATURES_QUERY = """
query GetInstanceFeatures($pdb_id: String!, $chain_id: String!) {
  polymer_entity_instances(pdb_id: $pdb_id) {
    rcsb_polymer_entity_instance_container_identifiers {
      entity_id
      asym_id
    }
    rcsb_instance_feature {
      feature_id
      feature_name
      feature_value
      feature_type
      provenance_source
    }
  }
}
"""

ENTITY_FEATURES_QUERY = """
query GetEntityFeatures($pdb_id: String!, $entity_id: String!) {
  polymer_entities(pdb_id: $pdb_id, entity_ids: [$entity_id]) {
    rcsb_polymer_entity_container_identifiers {
      entity_id
      entry_id
    }
    rcsb_polymer_entity_annotation {
      annotation_id
      type
      name
      description
      provenance_source
    }
    struct_ref {
      entity_id
      type
      db_name
      db_code
    }
    polymer_entity_annotation {
      entity_id
      type
      annotation_id
    }
    polymer_entity_go_annotation {
      go_term {
        id
        name
        definition
      }
      evidence_code
      qualifier
    }
  }
}
"""


def _parse_entity_id(entity_id: str) -> tuple[str, int]:
    """Parse entity ID format '{pdb_id}_{entity_number}' into components.

    Args:
        entity_id: Entity ID string (e.g., '1UBQ_1')

    Returns:
        Tuple of (pdb_id, entity_number)
    """
    parts = entity_id.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid entity ID format: {entity_id}")
    return parts[0], int(parts[1])


def _query_graphql(query: str, variables: dict) -> dict:
    """Execute a GraphQL query against the RCSB GraphQL endpoint.

    Args:
        query: GraphQL query string
        variables: Query variables

    Returns:
        Parsed JSON response

    Raises:
        ValueError: If the query fails
    """
    try:
        response = requests.post(
            RCSB_GRAPHQL_URL,
            json={"query": query, "variables": variables},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        if "errors" in result:
            raise ValueError(f"GraphQL errors: {result['errors']}")
        return result.get("data", {})
    except requests.RequestException as e:
        raise ValueError(f"RCSB GraphQL request failed: {e}")


def _get_instance_features(pdb_id: str, chain_id: str) -> dict:
    """Get instance features (CATH/SCOP) for a chain.

    Args:
        pdb_id: 4-character PDB identifier
        chain_id: Chain identifier

    Returns:
        Dict with instance features keyed by entity_id
    """
    data = _query_graphql(INSTANCE_FEATURES_QUERY, {"pdb_id": pdb_id, "chain_id": chain_id})

    features_by_entity = {}
    instances = data.get("polymer_entity_instances", [])

    for instance in instances:
        container_ids = instance.get("rcsb_polymer_entity_instance_container_identifiers", {})
        asym_id = container_ids.get("asym_id", "")
        entity_id = container_ids.get("entity_id", "")

        # Filter to requested chain
        if asym_id != chain_id:
            continue

        instance_features = instance.get("rcsb_instance_feature", [])
        features_by_entity[entity_id] = instance_features

    return features_by_entity


def _get_entity_annotations(pdb_id: str, entity_id: str) -> dict:
    """Get entity annotations (GO terms) for a specific entity.

    Args:
        pdb_id: 4-character PDB identifier
        entity_id: Entity ID (e.g., '1UBQ_1')

    Returns:
        Dict with entity annotations and cross-references
    """
    data = _query_graphql(ENTITY_FEATURES_QUERY, {"pdb_id": pdb_id, "entity_id": entity_id})

    entities = data.get("polymer_entities", [])
    if not entities:
        return {}

    entity = entities[0]

    # Get UniProt cross-reference
    uniprot_id = None
    struct_refs = entity.get("struct_ref", [])
    for ref in struct_refs:
        if ref.get("db_name") == "UniProt":
            uniprot_id = ref.get("db_code")
            break

    # Get GO annotations
    go_annotations = []
    go_data = entity.get("polymer_entity_go_annotation", [])
    for go_entry in go_data:
        go_term = go_entry.get("go_term", {})
        if go_term:
            go_annotations.append(
                {
                    "id": go_term.get("id"),
                    "name": go_term.get("name"),
                    "definition": go_term.get("definition"),
                    "evidence_code": go_entry.get("evidence_code"),
                    "qualifier": go_entry.get("qualifier"),
                }
            )

    # Get other annotations (CATH/SCOP from entity annotation)
    other_annotations = []
    annotation_data = entity.get("rcsb_polymer_entity_annotation", [])
    for ann in annotation_data:
        other_annotations.append(
            {
                "id": ann.get("annotation_id"),
                "type": ann.get("type"),
                "name": ann.get("name"),
                "description": ann.get("description"),
                "provenance_source": ann.get("provenance_source"),
            }
        )

    return {"uniprot_id": uniprot_id, "go_annotations": go_annotations, "other_annotations": other_annotations}


@tool(
    name="get_functional_annotations",
    toolset="structure",
    description="Retrieve functional annotations for a PDB structure chain including UniProt ID, GO terms, and CATH/SCOP domain classifications.",
    parameters={
        "pdb_id": {"type": "string", "description": "4-character PDB identifier (e.g., '1UBQ', '6VXX')"},
        "chain_id": {"type": "string", "description": "Chain identifier (e.g., 'A', 'B')"},
    },
)
def get_functional_annotations(pdb_id: str, chain_id: str) -> ToolResult:
    """
    Retrieve functional annotations for a specific chain in a PDB structure.

    Queries the RCSB GraphQL API to fetch:
    - UniProt cross-reference ID
    - Gene Ontology (GO) term annotations
    - CATH/SCOP domain classifications from instance features

    Parameters
    ----------
    pdb_id : str
        4-character PDB identifier (e.g., '1UBQ')
    chain_id : str
        Chain identifier (e.g., 'A')

    Returns
    -------
    ToolResult
        success: bool indicating if the operation succeeded
        data: Human-readable narrative description of functional annotations
        raw: Dict with uniprot_id, go_annotations, cath_domains, scop_domains
    """
    try:
        pdb_id = pdb_id.upper()
        chain_id = chain_id.upper()

        # First, get the structure to validate and map entity IDs
        structure = get_cached_structure(pdb_id)

        # Build mapping of chain -> entity_id from the structure
        # In gemmi, chains have names and we need to find the entity
        chain_to_entity = {}
        for model in structure:
            for chain in model:
                if chain.name == chain_id:
                    # Get entity ID from chain
                    # gemmi chains have a subchain name that corresponds to asym_id
                    asym_id = chain.name
                    # For mmCIF, entity IDs are stored separately
                    # We need to query the instance features to map
                    break

        # Query instance features to get entity_id mapping and CATH/SCOP
        instance_data = _get_instance_features(pdb_id, chain_id)

        # Query entity annotations (GO terms, UniProt)
        all_annotations = {}
        for entity_id in instance_data.keys():
            parsed_pdb, entity_num = _parse_entity_id(entity_id)
            all_annotations[entity_id] = _get_entity_annotations(pdb_id, entity_id)

        # Parse instance features for CATH/SCOP domains
        cath_domains = []
        scop_domains = []

        for entity_id, features in instance_data.items():
            for feature in features:
                feature_name = feature.get("feature_name", "")
                feature_type = feature.get("feature_type", "")
                provenance = feature.get("provenance_source", "")

                if "CATH" in provenance.upper() or "CATH" in feature_name.upper():
                    cath_domains.append(
                        {
                            "domain_id": feature.get("feature_id"),
                            "name": feature_name,
                            "value": feature.get("feature_value"),
                            "type": feature_type,
                        }
                    )
                elif "SCOP" in provenance.upper() or "SCOP" in feature_name.upper():
                    scop_domains.append(
                        {
                            "domain_id": feature.get("feature_id"),
                            "name": feature_name,
                            "value": feature.get("feature_value"),
                            "type": feature_type,
                        }
                    )

        # Compile GO terms from all entities
        all_go_terms = []
        for entity_id, annotations in all_annotations.items():
            for go in annotations.get("go_annotations", []):
                if go not in all_go_terms:
                    all_go_terms.append(go)

        # Compile UniProt IDs
        uniprot_ids = []
        for entity_id, annotations in all_annotations.items():
            up_id = annotations.get("uniprot_id")
            if up_id and up_id not in uniprot_ids:
                uniprot_ids.append(up_id)

        # Build raw result
        raw = {
            "pdb_id": pdb_id,
            "chain_id": chain_id,
            "uniprot_ids": uniprot_ids,
            "go_annotations": all_go_terms,
            "cath_domains": cath_domains,
            "scop_domains": scop_domains,
            "entities_with_annotations": list(all_annotations.keys()),
        }

        # Build narrative
        lines = [f"Functional annotations for {pdb_id} chain {chain_id}:"]

        if uniprot_ids:
            lines.append(f"UniProt: {', '.join(uniprot_ids)}")
        else:
            lines.append("UniProt: Not available")

        if all_go_terms:
            go_lines = []
            for go in all_go_terms[:10]:
                go_id = go.get("id", "")
                go_name = go.get("name", "")
                if go_id and go_name:
                    go_lines.append(f"{go_id} ({go_name})")
            if len(all_go_terms) > 10:
                lines.append(
                    f"GO terms ({len(all_go_terms)} total): {', '.join(go_lines[:10])} ... and {len(all_go_terms) - 10} more"
                )
            else:
                lines.append(f"GO terms ({len(all_go_terms)}): {', '.join(go_lines)}")
        else:
            lines.append("GO terms: Not available")

        if cath_domains:
            cath_lines = [f"{d.get('name', d.get('domain_id', 'Unknown'))}" for d in cath_domains[:5]]
            if len(cath_domains) > 5:
                lines.append(f"CATH domains: {', '.join(cath_lines)} ... and {len(cath_domains) - 5} more")
            else:
                lines.append(f"CATH domains: {', '.join(cath_lines)}")
        else:
            lines.append("CATH domains: Not available")

        if scop_domains:
            scop_lines = [f"{d.get('name', d.get('domain_id', 'Unknown'))}" for d in scop_domains[:5]]
            if len(scop_domains) > 5:
                lines.append(f"SCOP domains: {', '.join(scop_lines)} ... and {len(scop_domains) - 5} more")
            else:
                lines.append(f"SCOP domains: {', '.join(scop_lines)}")
        else:
            lines.append("SCOP domains: Not available")

        data = "\n".join(lines)

        return ToolResult(success=True, data=data, raw=raw)

    except ValueError as e:
        return ToolResult(
            success=False,
            data=f"Error retrieving annotations for {pdb_id} chain {chain_id}: {str(e)}",
            raw={},
            error=str(e),
        )
    except Exception as e:
        return ToolResult(success=False, data=f"Unexpected error: {type(e).__name__}: {str(e)}", raw={}, error=str(e))
