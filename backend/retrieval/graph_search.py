"""Cypher traversal retrieval over Neo4j."""

from __future__ import annotations

from typing import Any

from backend.pipeline.graph_builder import Neo4jGraphStore
from backend.retrieval.query_classifier import QueryClassification

CYPHER_TEMPLATES = {
    "dependency": """
        MATCH (target)
        WHERE (target:Function OR target:Class)
          AND toLower(target.name) = toLower($entity)
        CALL {
          WITH target
          MATCH (caller)-[:CALLS|CONTAINS|INHERITS_FROM]->(target)
          RETURN caller.name AS name,
                 caller.file_path AS file,
                 labels(caller)[0] AS type,
                 "symbol" AS via
          UNION
          WITH target
          MATCH (target)-[:DEFINED_IN]->(target_file:File)
          MATCH (importer:File)-[:IMPORTS*1..2]->(target_file)
          RETURN importer.path AS name,
                 importer.path AS file,
                 "File" AS type,
                 "import" AS via
        }
        RETURN DISTINCT name, file, type, via
        LIMIT 20
    """,
    "impact": """
        MATCH (target)
        WHERE (target:Function OR target:Class)
          AND toLower(target.name) = toLower($entity)
        CALL {
          WITH target
          MATCH path = (dependent)-[:CALLS|CONTAINS|INHERITS_FROM*1..3]->(target)
          RETURN [node in nodes(path) | coalesce(node.name, node.path)] AS chain,
                 [node in nodes(path) | coalesce(node.file_path, node.path)] AS files
          UNION
          WITH target
          MATCH (target)-[:DEFINED_IN]->(target_file:File)
          MATCH path = (importer:File)-[:IMPORTS*1..3]->(target_file)
          RETURN [node in nodes(path) | coalesce(node.name, node.path)] AS chain,
                 [node in nodes(path) | coalesce(node.file_path, node.path)] AS files
        }
        RETURN DISTINCT chain, files
        LIMIT 20
    """,
    "definition": """
        MATCH (n)
        WHERE (n:Function OR n:Class OR n:Module) AND toLower(n.name) = toLower($entity)
        RETURN n.name AS name, n.file_path AS file,
               n.signature AS signature, n.docstring AS docstring,
               labels(n)[0] AS type, n.line_start AS line
        LIMIT 5
    """,
    "semantic": """
        MATCH (f)
        WHERE (f:Function OR f:Class)
          AND (toLower(f.name) CONTAINS toLower($entity)
               OR toLower(coalesce(f.docstring, "")) CONTAINS toLower($entity))
        RETURN f.name AS name, f.file_path AS file,
               f.docstring AS docstring, labels(f)[0] AS type
        LIMIT 10
    """,
}


def graph_search(classification: QueryClassification) -> dict[str, Any]:
    """Run the appropriate Cypher template for the classified query."""

    query = CYPHER_TEMPLATES[classification.type]
    try:
        with Neo4jGraphStore() as graph_store:
            records = graph_store.query(query, entity=classification.entity)
    except Exception:
        records = []
    return {"records": records, "formatted": _format_records(classification, records)}


def _format_records(classification: QueryClassification, records: list[dict[str, Any]]) -> list[str]:
    formatted: list[str] = []
    for record in sorted(records, key=lambda record: _path_penalty(str(record.get("file", "")))):
        if classification.type == "dependency":
            formatted.append(
                f"{record['name']} ({record['type']}, via={record.get('via', 'symbol')}) "
                f"-> {classification.entity} [{record['file']}]"
            )
        elif classification.type == "impact":
            chain = " -> ".join(record.get("chain", []))
            files = " -> ".join(record.get("files", []))
            formatted.append(f"{chain} [{files}]")
        elif classification.type == "definition":
            formatted.append(
                f"{record['type']} {record['name']} defined in {record['file']}:{record.get('line', 0)} | "
                f"{record.get('signature') or ''}"
            )
        else:
            formatted.append(
                f"{record['type']} {record['name']} in {record['file']} | {record.get('docstring') or ''}"
            )
    return formatted


def _path_penalty(file_path: str) -> float:
    lowered = file_path.lower()
    if "/src/" in lowered or lowered.startswith("src/"):
        return 0.0
    if "/tests/" in lowered or lowered.startswith("tests/"):
        return 0.8
    if "/scripts/" in lowered or lowered.startswith("scripts/"):
        return 0.95
    if "/docs/" in lowered or "/examples/" in lowered or "/docs_src/" in lowered:
        return 0.9
    return 0.2
