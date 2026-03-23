"""Repository loading for local folders and Git URLs."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from backend.config import REPO_MANIFEST_PATH, REPOS_DIR, ensure_runtime_dirs

try:
    from git import Repo
except ModuleNotFoundError:  # pragma: no cover - optional dependency at runtime
    Repo = None


LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
}
SKIP_DIRS = {
    "node_modules",
    "__pycache__",
    ".git",
    "dist",
    "build",
    "venv",
    ".venv",
    ".venv313",
    "docs",
    "docs_src",
    "tests",
    "test",
    "examples",
    ".pytest_cache",
    ".mypy_cache",
}
MAX_SOURCE_FILE_BYTES = 300_000


@dataclass(slots=True)
class RepoFile:
    """A source file selected for parsing."""

    file_path: str
    language: str
    raw_content: str
    absolute_path: str


@dataclass(slots=True)
class LoadedRepo:
    """Resolved repository and file set."""

    source: str
    repo_name: str
    repo_root: str
    revision: str
    files: list[RepoFile]


class RepoLoader:
    """Load code files from a local path or a cloned Git repository."""

    def load(self, source: str) -> LoadedRepo:
        """Resolve the source into a local repository tree and load supported files."""

        ensure_runtime_dirs()
        repo_root = self._resolve_source(source)
        repo_name = Path(repo_root).name
        files = self._walk_repo(repo_root)
        revision = self._repo_revision(repo_root, files)
        loaded = LoadedRepo(
            source=source,
            repo_name=repo_name,
            repo_root=str(repo_root),
            revision=revision,
            files=files,
        )
        return loaded

    def _resolve_source(self, source: str) -> Path:
        """Clone a remote repo or validate a local path."""

        if source.startswith(("http://", "https://", "git@")):
            return self._clone_repo(source)

        path = Path(source).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Repository path does not exist: {source}")
        return path

    def _clone_repo(self, source: str) -> Path:
        """Clone a Git URL into the managed repo cache."""

        if Repo is None:
            raise ModuleNotFoundError("gitpython is not installed. Run `pip install -r backend/requirements.txt`.")
        repo_name = _repo_name_from_source(source)
        target_dir = REPOS_DIR / repo_name
        if target_dir.exists():
            return target_dir
        Repo.clone_from(source, target_dir)
        return target_dir

    def _walk_repo(self, repo_root: Path) -> list[RepoFile]:
        """Walk a repository and return supported source files."""

        files: list[RepoFile] = []
        managed_repo_cache = (repo_root / "data" / "repos").resolve()
        for root, dirs, filenames in os.walk(repo_root):
            root_path = Path(root)
            filtered_dirs: list[str] = []
            for name in dirs:
                if name in SKIP_DIRS:
                    continue
                candidate = (root_path / name).resolve()
                # Avoid indexing cached demo repos when ingesting the CodeGraph project itself.
                if candidate == managed_repo_cache:
                    continue
                filtered_dirs.append(name)
            dirs[:] = filtered_dirs
            for filename in filenames:
                file_path = root_path / filename
                language = LANGUAGE_EXTENSIONS.get(file_path.suffix.lower())
                if not language:
                    continue
                try:
                    if file_path.stat().st_size > MAX_SOURCE_FILE_BYTES:
                        continue
                except OSError:
                    continue
                raw_content = file_path.read_text(encoding="utf-8", errors="ignore")
                relative_path = file_path.relative_to(repo_root).as_posix()
                files.append(
                    RepoFile(
                        file_path=relative_path,
                        language=language,
                        raw_content=raw_content,
                        absolute_path=str(file_path),
                    )
                )
        return sorted(files, key=lambda item: item.file_path)

    def _repo_revision(self, repo_root: Path, files: list[RepoFile]) -> str:
        """Compute a stable revision for cache validation."""

        if Repo is not None:
            try:
                repo = Repo(repo_root, search_parent_directories=True)
                if not repo.bare:
                    return str(repo.head.commit.hexsha)
            except Exception:
                pass
        digest = hashlib.sha256()
        for file in files:
            path = Path(file.absolute_path)
            stat = path.stat()
            digest.update(file.file_path.encode("utf-8"))
            digest.update(str(stat.st_mtime_ns).encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
        return digest.hexdigest()


def save_repo_manifest(loaded_repo: LoadedRepo) -> None:
    """Persist the last ingested repo manifest for incremental updates."""

    ensure_runtime_dirs()
    payload = {
        "source": loaded_repo.source,
        "repo_name": loaded_repo.repo_name,
        "repo_root": loaded_repo.repo_root,
        "revision": loaded_repo.revision,
        "files": [asdict(file) for file in loaded_repo.files],
    }
    REPO_MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_repo_manifest() -> dict[str, object]:
    """Load the last ingested repo manifest, if available."""

    if not REPO_MANIFEST_PATH.exists():
        return {}
    return json.loads(REPO_MANIFEST_PATH.read_text(encoding="utf-8"))


def update_repo_manifest(loaded_repo: LoadedRepo, ingest_result: dict[str, Any]) -> None:
    """Persist repo metadata plus the last successful ingest result."""

    save_repo_manifest(loaded_repo)
    payload = load_repo_manifest()
    payload["last_ingest_result"] = ingest_result
    REPO_MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_cached_ingest_result(source: str, revision: str) -> dict[str, Any] | None:
    """Return the previous ingest result if the repo source and revision match."""

    payload = load_repo_manifest()
    if payload.get("source") != source:
        return None
    if payload.get("revision") != revision:
        return None
    result = payload.get("last_ingest_result")
    return result if isinstance(result, dict) else None


def _repo_name_from_source(source: str) -> str:
    """Create a stable repo folder name from a Git URL."""

    parsed = urlparse(source)
    candidate = Path(parsed.path.rstrip("/")).stem or "repo"
    return candidate.removesuffix(".git")
