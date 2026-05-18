"""Tool result processor for converting tool outputs to viewer updates."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ViewerUpdate:
    """Represents an update to send to the molecular viewer."""
    action: str
    pdb_id: Optional[str] = None
    highlight: Optional[dict] = None
    message: Optional[str] = None


class ToolResultProcessor:
    """Processes tool results and converts them to viewer updates."""

    # Color mapping for secondary structure
    SS_COLORS = {
        "helix": "red",
        "strand": "yellow",
        "coil": "gray",
    }

    @staticmethod
    def process(tool_name: str, raw: dict) -> Optional[ViewerUpdate]:
        """Process a tool result and return a ViewerUpdate if applicable.

        Args:
            tool_name: Name of the tool that was executed.
            raw: Raw result dictionary from the tool.

        Returns:
            ViewerUpdate if the tool produces relevant structured data, None otherwise.
        """
        if tool_name == "load_structure":
            return ToolResultProcessor._process_load_structure(raw)
        elif tool_name == "get_secondary_structure":
            return ToolResultProcessor._process_secondary_structure(raw)
        elif tool_name == "get_residue_contacts":
            return ToolResultProcessor._process_residue_contacts(raw)
        elif tool_name == "compute_interface":
            return ToolResultProcessor._process_interface(raw)
        elif tool_name == "check_ramachandran":
            return ToolResultProcessor._process_ramachandran(raw)
        return None

    @staticmethod
    def _process_load_structure(raw: dict) -> ViewerUpdate:
        """Process load_structure result - highlight entire structure."""
        pdb_id = raw.get("pdb_id")
        return ViewerUpdate(
            action="load",
            pdb_id=pdb_id,
            highlight={"selection": "all", "color": None},
            message=f"Loaded structure {pdb_id}" if pdb_id else "Structure loaded",
        )

    @staticmethod
    def _process_secondary_structure(raw: dict) -> ViewerUpdate:
        """Process secondary structure result - color by ss element type."""
        elements = raw.get("secondary_structure", {}).get("elements", [])
        helix_residues = []
        strand_residues = []
        coil_residues = []

        for element in elements:
            ss_type = element.get("type", "").lower()
            residue = element.get("residue")
            chain = element.get("chain", "A")

            if ss_type == "helix":
                helix_residues.append(f"{chain}:{residue}")
            elif ss_type == "strand":
                strand_residues.append(f"{chain}:{residue}")
            else:
                coil_residues.append(f"{chain}:{residue}")

        highlight = {}
        if helix_residues:
            highlight["helix"] = {"residue_ids": helix_residues, "color": ToolResultProcessor.SS_COLORS["helix"]}
        if strand_residues:
            highlight["strand"] = {"residue_ids": strand_residues, "color": ToolResultProcessor.SS_COLORS["strand"]}
        if coil_residues:
            highlight["coil"] = {"residue_ids": coil_residues, "color": ToolResultProcessor.SS_COLORS["coil"]}

        return ViewerUpdate(
            action="color_ss",
            highlight=highlight if highlight else None,
            message="Secondary structure colored (helix=red, strand=yellow, coil=gray)",
        )

    @staticmethod
    def _process_residue_contacts(raw: dict) -> ViewerUpdate:
        """Process residue contacts result - highlight contacting residues."""
        contacts = raw.get("contacts", [])
        residue_ids = []

        for contact in contacts:
            residue_ids.append(f"{contact.get('chain_a', 'A')}:{contact.get('residue_a')}")
            residue_ids.append(f"{contact.get('chain_b', 'B')}:{contact.get('residue_b')}")

        # Remove duplicates
        residue_ids = list(dict.fromkeys(residue_ids))

        return ViewerUpdate(
            action="highlight_residues",
            highlight={"residue_ids": residue_ids, "color": "orange"},
            message=f"Highlighted {len(residue_ids)} contacting residues",
        )

    @staticmethod
    def _process_interface(raw: dict) -> ViewerUpdate:
        """Process interface result - show interface residues in different colors."""
        interface_residues = raw.get("interface_residues", {})
        chain_a_residues = interface_residues.get("chain_a", [])
        chain_b_residues = interface_residues.get("chain_b", [])

        highlight = {}
        if chain_a_residues:
            highlight["chain_a"] = {"residue_ids": chain_a_residues, "color": "yellow"}
        if chain_b_residues:
            highlight["chain_b"] = {"residue_ids": chain_b_residues, "color": "cyan"}

        return ViewerUpdate(
            action="show_interface",
            highlight=highlight if highlight else None,
            message=f"Interface: {len(chain_a_residues)} residues from chain A (yellow), {len(chain_b_residues)} from chain B (cyan)",
        )

    @staticmethod
    def _process_ramachandran(raw: dict) -> ViewerUpdate:
        """Process Ramachandran result - highlight outliers."""
        outliers = raw.get("outliers", [])

        outlier_residue_ids = [
            f"{o.get('chain', 'A')}:{o.get('residue')}" for o in outliers
        ]

        return ViewerUpdate(
            action="highlight_outliers",
            highlight={
                "residue_ids": outlier_residue_ids,
                "color": "red",
            } if outlier_residue_ids else None,
            message=f"Ramachandran outliers: {len(outlier_residue_ids)} residues highlighted in red" if outlier_residue_ids else "No Ramachandran outliers found",
        )
