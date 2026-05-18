"""NCBI PubMed abstract fetching module."""

import time
from typing import Any

import requests


def fetch_pubmed_abstract(pmid: str) -> dict[str, Any] | None:
    """Fetch a PubMed abstract by PMID.

    Args:
        pmid: PubMed ID to fetch.

    Returns:
        Dict with keys: pmid, title, abstract.
        Returns None on failure.
    """
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&rettype=abstract"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return None

    time.sleep(1.0)

    try:
        root = __parse_xml(response.text)
    except Exception:
        return None

    if root is None:
        return None

    return {
        "pmid": pmid,
        "title": root.get("title", ""),
        "abstract": root.get("abstract", ""),
    }


def __parse_xml(xml_text: str) -> dict[str, str] | None:
    """Parse PubMed XML and extract title and abstract.

    Args:
        xml_text: Raw XML response from efetch.

    Returns:
        Dict with 'title' and 'abstract' keys, or None on parse failure.
    """
    try:
        from xml.etree import ElementTree as ET
    except ImportError:
        return None

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    # Handle <MedlineTitle> or <ArticleTitle> for title
    title = _find_element_text(root, ["MedlineTitle", "ArticleTitle"])

    # Extract abstract text from <AbstractText> elements
    abstract_parts = []
    abstract_elem = root.find(".//Abstract")
    if abstract_elem is not None:
        for abstract_text in abstract_elem.findall("AbstractText"):
            if abstract_text.text:
                abstract_parts.append(abstract_text.text)
        # Also check for Label + AbstractText pattern (structured abstracts)
        for abstract_text in abstract_elem.findall("AbstractText"):
            label = abstract_text.get("Label")
            if label and abstract_text.text:
                abstract_parts.append(f"{label}: {abstract_text.text}")

    abstract = " ".join(abstract_parts) if abstract_parts else ""

    return {"title": title or "", "abstract": abstract}


def _find_element_text(root: Any, tags: list[str]) -> str | None:
    """Find first matching element and return its text content.

    Args:
        root: XML root element.
        tags: List of tag names to search for in order.

    Returns:
        Text content of first matching element, or None.
    """
    for tag in tags:
        elem = root.find(f".//{tag}")
        if elem is not None and elem.text:
            return elem.text
    return None
