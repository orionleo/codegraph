"""Incremental updates for a single changed file."""

from __future__ import annotations

from pathlib import Path

from backend.pipeline.code_parser import parse_repo_files
from backend.pipeline.embedder import CodeEmbedder
from backend.pipeline.graph_builder import Neo4jGraphStore
from backend.pipeline.repo_loader import RepoFile, load_repo_manifest


class DiffUpdater:
    """Re-index a single changed file without re-ingesting the full repo."""

    def update(self, file_path: str) -> dict[str, int | str]:
        """Re-parse one file and replace its graph/vector state."""

        manifest = load_repo_manifest()
        resolved = self._resolve_file(file_path, manifest)
        if resolved is None:
            raise FileNotFoundError(f"File not found in manifest or on disk: {file_path}")

        repo_file = RepoFile(
            file_path=resolved["relative_path"],
            language=resolved["language"],
            raw_content=Path(resolved["absolute_path"]).read_text(encoding="utf-8", errors="ignore"),
            absolute_path=resolved["absolute_path"],
        )
        parsed_files = parse_repo_files([repo_file])

        embedder = CodeEmbedder()
        with Neo4jGraphStore() as graph_store:
            graph_store.delete_file(repo_file.file_path)
            graph_stats = graph_store.build_graph(parsed_files)
        embedder.delete_file(repo_file.file_path)
        embed_stats = embedder.store(parsed_files)
        result = {
            "file_path": repo_file.file_path,
            "nodes_updated": int(graph_stats["nodes_created"]),
            "edges_updated": int(graph_stats["edges_created"]),
            "chunks_embedded": int(embed_stats["chunks_embedded"]),
        }
        print(f"Updated {result['nodes_updated']} nodes, {result['edges_updated']} edges for file: {repo_file.file_path}")
        return result

    def _resolve_file(self, file_path: str, manifest: dict[str, object]) -> dict[str, str] | None:
        """Resolve a file path from the saved ingest manifest."""

        files = manifest.get("files", [])
        absolute_candidate = Path(file_path).expanduser().resolve()
        for item in files if isinstance(files, list) else []:
            if not isinstance(item, dict):
                continue
            if item.get("file_path") == file_path or item.get("absolute_path") == str(absolute_candidate):
                return {
                    "relative_path": str(item["file_path"]),
                    "absolute_path": str(item["absolute_path"]),
                    "language": str(item["language"]),
                }
        if absolute_candidate.exists():
            return {
                "relative_path": absolute_candidate.name,
                "absolute_path": str(absolute_candidate),
                "language": _language_for_path(absolute_candidate),
            }
        return None


def _language_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".java": "java",
    }.get(suffix, "text")
