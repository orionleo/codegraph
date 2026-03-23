"""Neo4j graph persistence for parsed code structure."""

from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from neo4j import Driver, GraphDatabase

from backend.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from backend.pipeline.code_parser import ParsedClass, ParsedFile, ParsedFunction, ParsedImport


CALL_REL = "CALLS"
DEFINED_IN_REL = "DEFINED_IN"
CONTAINS_REL = "CONTAINS"
INHERITS_REL = "INHERITS_FROM"
IMPORTS_REL = "IMPORTS"
BELONGS_TO_REL = "BELONGS_TO"


class Neo4jGraphStore(AbstractContextManager["Neo4jGraphStore"]):
    """Neo4j wrapper for the code graph."""

    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD) -> None:
        self.driver: Driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=10,
        )

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Close the Neo4j driver."""

        self.driver.close()

    def ensure_constraints(self) -> None:
        """Create the graph constraints required by the prompt."""

        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Function) REQUIRE f.name IS NOT NULL",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Class) REQUIRE c.name IS NOT NULL",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:File) REQUIRE f.path IS NOT NULL",
        ]
        with self.driver.session() as session:
            for query in queries:
                session.run(query)

    def build_graph(self, parsed_files: list[ParsedFile]) -> dict[str, int]:
        """Write all nodes and relationships in two passes."""

        self.ensure_constraints()
        node_batches = _prepare_node_batches(parsed_files)
        edge_batches = _prepare_edge_batches(parsed_files)

        processed_files = 0
        node_count = 0
        edge_count = 0

        with self.driver.session() as session:
            for query, rows in node_batches:
                for batch in _chunked(rows, 100):
                    session.execute_write(lambda tx, q=query, b=batch: tx.run(q, rows=b).consume())
                    node_count += len(batch)
            for parsed_file in parsed_files:
                processed_files += 1
                print(
                    f"Processed {processed_files}/{len(parsed_files)} files, {node_count} nodes, {edge_count} edges created"
                )

            for query, rows in edge_batches:
                for batch in _chunked(rows, 100):
                    session.execute_write(lambda tx, q=query, b=batch: tx.run(q, rows=b).consume())
                    edge_count += len(batch)

        print(f"Processed {processed_files}/{len(parsed_files)} files, {node_count} nodes, {edge_count} edges created")
        return {
            "files_processed": len(parsed_files),
            "nodes_created": node_count,
            "edges_created": edge_count,
        }

    def delete_file(self, file_path: str) -> None:
        """Delete all graph nodes associated with a file path."""

        query = """
        MATCH (n)
        WHERE n.file_path = $path OR n.path = $path
        DETACH DELETE n
        """
        with self.driver.session() as session:
            session.run(query, path=file_path)

    def reset_graph(self) -> None:
        """Delete all nodes and relationships from the graph."""

        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def query(self, cypher: str, **params: object) -> list[dict[str, Any]]:
        """Run a read query and return plain dictionaries."""

        with self.driver.session() as session:
            result = session.run(cypher, **params)
            return [record.data() for record in result]

    def get_stats(self) -> dict[str, Any]:
        """Return graph stats for the frontend sidebar."""

        with self.driver.session() as session:
            record = session.run(
                """
                MATCH (f:Function) WITH count(f) AS total_functions
                MATCH (c:Class) WITH total_functions, count(c) AS total_classes
                MATCH (file:File) WITH total_functions, total_classes, count(file) AS total_files
                MATCH ()-[r]->() WITH total_functions, total_classes, total_files, count(r) AS total_edges
                MATCH (file:File)
                RETURN total_functions, total_classes, total_files, total_edges, collect(DISTINCT file.language) AS languages
                """
            ).single()
        if record is None:
            return {
                "total_functions": 0,
                "total_classes": 0,
                "total_files": 0,
                "total_edges": 0,
                "languages": [],
            }
        return dict(record)

    def explore(self, entity: str, depth: int = 2) -> dict[str, list[dict[str, Any]]]:
        """Return a small subgraph around an entity."""

        depth_limit = max(1, min(depth, 3))
        query = """
        MATCH (seed)
        WHERE seed.name = $entity OR seed.path = $entity
        OPTIONAL MATCH path = (seed)-[*1..DEPTH_LIMIT]-(neighbor)
        WITH seed, collect(path) AS paths
        WITH seed, [p IN paths WHERE p IS NOT NULL] AS valid_paths
        RETURN
          [{
            id: elementId(seed),
            label: coalesce(seed.name, seed.path),
            type: labels(seed)[0],
            file_path: coalesce(seed.file_path, seed.path)
          }] +
          reduce(all_nodes = [], p IN valid_paths |
            all_nodes + [n IN nodes(p) | {
              id: elementId(n),
              label: coalesce(n.name, n.path),
              type: labels(n)[0],
              file_path: coalesce(n.file_path, n.path)
            }]
          ) AS nodes,
          reduce(all_edges = [], p IN valid_paths |
            all_edges + [r IN relationships(p) | {
              source: elementId(startNode(r)),
              target: elementId(endNode(r)),
              type: type(r)
            }]
          ) AS edges
        """
        query = query.replace("DEPTH_LIMIT", str(depth_limit))
        records = self.query(query, entity=entity, depth=depth)
        if not records:
            return {"nodes": [], "edges": []}
        nodes = _dedupe_by_key(records[0].get("nodes", []), "id")
        edges = _dedupe_edges(records[0].get("edges", []))
        return {"nodes": nodes, "edges": edges}


def _prepare_node_batches(parsed_files: list[ParsedFile]) -> list[tuple[str, list[dict[str, Any]]]]:
    files: list[dict[str, Any]] = []
    modules: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []

    for parsed_file in parsed_files:
        files.append(
            {
                "path": parsed_file.file_path,
                "language": parsed_file.language,
                "summary": "",
            }
        )
        modules.append({"name": parsed_file.module_name, "file_path": parsed_file.file_path})
        for function in parsed_file.functions:
            functions.append(asdict(function))
        for parsed_class in parsed_file.classes:
            classes.append(asdict(parsed_class))

    return [
        (
            """
            UNWIND $rows AS row
            MERGE (f:File {path: row.path})
            SET f.language = row.language, f.summary = row.summary
            """,
            files,
        ),
        (
            """
            UNWIND $rows AS row
            MERGE (m:Module {name: row.name, file_path: row.file_path})
            """,
            modules,
        ),
        (
            """
            UNWIND $rows AS row
            MERGE (f:Function {name: row.name, file_path: row.file_path})
            SET f.docstring = row.docstring,
                f.signature = row.signature,
                f.line_start = row.line_start,
                f.line_end = row.line_end,
                f.is_method = row.is_method
            """,
            functions,
        ),
        (
            """
            UNWIND $rows AS row
            MERGE (c:Class {name: row.name, file_path: row.file_path})
            SET c.docstring = row.docstring,
                c.line_start = row.line_start,
                c.line_end = row.line_end
            """,
            classes,
        ),
    ]


def _prepare_edge_batches(parsed_files: list[ParsedFile]) -> list[tuple[str, list[dict[str, Any]]]]:
    contains_rows: list[dict[str, Any]] = []
    defined_rows: list[dict[str, Any]] = []
    file_module_rows: list[dict[str, Any]] = []
    inheritance_rows: list[dict[str, Any]] = []
    function_inheritance_rows: list[dict[str, Any]] = []
    import_rows: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []

    function_locations = _function_index(parsed_files)
    class_locations = _class_index(parsed_files)
    module_locations = _module_index(parsed_files)
    class_base_map = _class_base_index(parsed_files)

    for parsed_file in parsed_files:
        file_module_rows.append({"file_path": parsed_file.file_path, "module_name": parsed_file.module_name})
        for function in parsed_file.functions:
            defined_rows.append({"name": function.name, "file_path": function.file_path})
            if function.parent_class:
                contains_rows.append(
                    {
                        "class_name": function.parent_class,
                        "class_file_path": function.file_path,
                        "function_name": function.name,
                        "function_file_path": function.file_path,
                    }
                )
            for call_name in function.calls:
                target_file = _resolve_function_target(call_name, function_locations)
                call_rows.append(
                    {
                        "source_name": function.name,
                        "source_file_path": function.file_path,
                        "target_name": call_name.split(".")[-1],
                        "target_file_path": target_file or f"__external__::{call_name}",
                        "unresolved_name": call_name,
                    }
                )
            if function.parent_class:
                base_names = class_base_map.get((function.parent_class, function.file_path), [])
                for base in base_names:
                    base_name = base.split(".")[-1]
                    function_inheritance_rows.append(
                        {
                            "function_name": function.name,
                            "function_file_path": function.file_path,
                            "target_name": base_name,
                            "target_file_path": _resolve_class_target(base_name, class_locations)
                            or f"__external__::{base_name}",
                        }
                    )

        for parsed_class in parsed_file.classes:
            defined_rows.append({"name": parsed_class.name, "file_path": parsed_class.file_path, "label": "Class"})
            for base in parsed_class.bases:
                base_name = base.split(".")[-1]
                inheritance_rows.append(
                    {
                        "source_name": parsed_class.name,
                        "source_file_path": parsed_class.file_path,
                        "target_name": base_name,
                        "target_file_path": _resolve_class_target(base_name, class_locations)
                        or f"__external__::{base_name}",
                    }
                )

        for parsed_import in parsed_file.imports:
            resolved_file_path = _resolve_import_target(parsed_import, module_locations)
            import_rows.append(
                {
                    "source_path": parsed_import.source_file,
                    "target_path": resolved_file_path,
                    "external_module": "" if resolved_file_path else parsed_import.imported_module,
                }
            )

    placeholder_functions = [
        {
            "name": row["target_name"],
            "docstring": None,
            "signature": row["unresolved_name"],
            "line_start": 0,
            "line_end": 0,
            "is_method": False,
            "file_path": row["target_file_path"],
        }
        for row in call_rows
        if str(row["target_file_path"]).startswith("__external__::")
    ]

    placeholder_classes = [
        {
            "name": row["target_name"],
            "docstring": None,
            "line_start": 0,
            "line_end": 0,
            "file_path": row["target_file_path"],
        }
        for row in inheritance_rows
        if str(row["target_file_path"]).startswith("__external__::")
    ]
    placeholder_classes.extend(
        [
            {
                "name": row["target_name"],
                "docstring": None,
                "line_start": 0,
                "line_end": 0,
                "file_path": row["target_file_path"],
            }
            for row in function_inheritance_rows
            if str(row["target_file_path"]).startswith("__external__::")
        ]
    )

    batches: list[tuple[str, list[dict[str, Any]]]] = []
    if placeholder_functions:
        batches.append(
            (
                """
                UNWIND $rows AS row
                MERGE (f:Function {name: row.name, file_path: row.file_path})
                SET f.signature = row.signature,
                    f.docstring = row.docstring,
                    f.line_start = row.line_start,
                    f.line_end = row.line_end,
                    f.is_method = row.is_method
                """,
                placeholder_functions,
            )
        )
    if placeholder_classes:
        batches.append(
            (
                """
                UNWIND $rows AS row
                MERGE (c:Class {name: row.name, file_path: row.file_path})
                SET c.docstring = row.docstring,
                    c.line_start = row.line_start,
                    c.line_end = row.line_end
                """,
                placeholder_classes,
            )
        )

    batches.extend(
        [
            (
                """
                UNWIND $rows AS row
                MATCH (m:Module {name: row.module_name, file_path: row.file_path})
                MATCH (f:File {path: row.file_path})
                MERGE (f)-[:BELONGS_TO]->(m)
                """,
                file_module_rows,
            ),
            (
                """
                UNWIND $rows AS row
                MATCH (fn:Function {name: row.name, file_path: row.file_path})
                MATCH (f:File {path: row.file_path})
                MERGE (fn)-[:DEFINED_IN]->(f)
                """,
                [row for row in defined_rows if row.get("label") != "Class"],
            ),
            (
                """
                UNWIND $rows AS row
                MATCH (c:Class {name: row.name, file_path: row.file_path})
                MATCH (f:File {path: row.file_path})
                MERGE (c)-[:DEFINED_IN]->(f)
                """,
                [row for row in defined_rows if row.get("label") == "Class"],
            ),
            (
                """
                UNWIND $rows AS row
                MATCH (c:Class {name: row.class_name, file_path: row.class_file_path})
                MATCH (fn:Function {name: row.function_name, file_path: row.function_file_path})
                MERGE (c)-[:CONTAINS]->(fn)
                """,
                contains_rows,
            ),
            (
                """
                UNWIND $rows AS row
                MATCH (source:Function {name: row.source_name, file_path: row.source_file_path})
                MERGE (target:Function {name: row.target_name, file_path: row.target_file_path})
                ON CREATE SET target.signature = row.unresolved_name, target.line_start = 0, target.line_end = 0
                MERGE (source)-[:CALLS]->(target)
                """,
                call_rows,
            ),
            (
                """
                UNWIND $rows AS row
                MATCH (source:Class {name: row.source_name, file_path: row.source_file_path})
                MERGE (target:Class {name: row.target_name, file_path: row.target_file_path})
                ON CREATE SET target.line_start = 0, target.line_end = 0
                MERGE (source)-[:INHERITS_FROM]->(target)
                """,
                inheritance_rows,
            ),
            (
                """
                UNWIND $rows AS row
                MATCH (source:Function {name: row.function_name, file_path: row.function_file_path})
                MERGE (target:Class {name: row.target_name, file_path: row.target_file_path})
                ON CREATE SET target.line_start = 0, target.line_end = 0
                MERGE (source)-[:INHERITS_FROM]->(target)
                """,
                function_inheritance_rows,
            ),
            (
                """
                UNWIND $rows AS row
                MATCH (source:File {path: row.source_path})
                FOREACH (_ IN CASE WHEN row.target_path IS NULL THEN [] ELSE [1] END |
                    MERGE (target:File {path: row.target_path})
                    MERGE (source)-[r:IMPORTS]->(target)
                )
                FOREACH (_ IN CASE WHEN row.target_path IS NOT NULL THEN [] ELSE [1] END |
                    MERGE (source)-[r:IMPORTS {external_module: row.external_module}]->(:File {path: "__external__::" + row.external_module, language: "external", summary: ""})
                )
                """,
                import_rows,
            ),
        ]
    )
    return batches


def _function_index(parsed_files: list[ParsedFile]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for parsed_file in parsed_files:
        for function in parsed_file.functions:
            index[function.name].append(function.file_path)
            index[f"{Path(function.file_path).stem}.{function.name}"].append(function.file_path)
            if function.parent_class:
                index[f"{function.parent_class}.{function.name}"].append(function.file_path)
    return index


def _class_index(parsed_files: list[ParsedFile]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for parsed_file in parsed_files:
        for parsed_class in parsed_file.classes:
            index[parsed_class.name].append(parsed_class.file_path)
    return index


def _module_index(parsed_files: list[ParsedFile]) -> dict[str, str]:
    index: dict[str, str] = {}
    for parsed_file in parsed_files:
        index[parsed_file.module_name] = parsed_file.file_path
        index[Path(parsed_file.file_path).with_suffix("").as_posix()] = parsed_file.file_path
    return index


def _class_base_index(parsed_files: list[ParsedFile]) -> dict[tuple[str, str], list[str]]:
    index: dict[tuple[str, str], list[str]] = {}
    for parsed_file in parsed_files:
        for parsed_class in parsed_file.classes:
            index[(parsed_class.name, parsed_class.file_path)] = parsed_class.bases
    return index


def _resolve_function_target(call_name: str, function_locations: dict[str, list[str]]) -> str | None:
    exact = function_locations.get(call_name)
    if exact:
        return exact[0]
    short_name = call_name.split(".")[-1]
    exact = function_locations.get(short_name)
    if exact:
        return exact[0]
    fuzzy = next((paths[0] for key, paths in function_locations.items() if key.endswith(short_name)), None)
    return fuzzy


def _resolve_class_target(class_name: str, class_locations: dict[str, list[str]]) -> str | None:
    matches = class_locations.get(class_name)
    return matches[0] if matches else None


def _resolve_import_target(parsed_import: ParsedImport, module_locations: dict[str, str]) -> str | None:
    if parsed_import.imported_module in module_locations:
        return module_locations[parsed_import.imported_module]
    normalized = parsed_import.imported_module.replace(".", "/")
    return module_locations.get(normalized)


def _chunked(items: Iterable[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def _dedupe_by_key(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        value = str(item.get(key, ""))
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(item)
    return deduped


def _dedupe_edges(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        key = (str(item.get("source", "")), str(item.get("target", "")), str(item.get("type", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped
