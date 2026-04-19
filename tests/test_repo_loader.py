"""Tests for repository loading and ingest cache helpers."""

from __future__ import annotations

from pathlib import Path

from backend.pipeline import repo_loader
from backend.pipeline.repo_loader import LoadedRepo, RepoFile, RepoLoader


def test_repo_loader_filters_supported_files_and_ignored_dirs(tmp_path: Path) -> None:
    repo = tmp_path / "demo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "docs").mkdir()
    (repo / "src" / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    (repo / "src" / "view.ts").write_text("export const view = () => 1;\n", encoding="utf-8")
    (repo / "src" / "README.md").write_text("# ignored\n", encoding="utf-8")
    (repo / "tests" / "test_app.py").write_text("def test_run(): pass\n", encoding="utf-8")
    (repo / "docs" / "conf.py").write_text("project = 'demo'\n", encoding="utf-8")

    loaded = RepoLoader().load(str(repo))

    assert loaded.repo_name == "demo"
    assert [file.file_path for file in loaded.files] == ["src/app.py", "src/view.ts"]
    assert [file.language for file in loaded.files] == ["python", "typescript"]
    assert loaded.revision


def test_repo_loader_skips_oversized_source_files(tmp_path: Path) -> None:
    repo = tmp_path / "demo"
    repo.mkdir()
    (repo / "small.py").write_text("def ok(): pass\n", encoding="utf-8")
    (repo / "large.py").write_text("x = 1\n" * 80_000, encoding="utf-8")

    loaded = RepoLoader().load(str(repo))

    assert [file.file_path for file in loaded.files] == ["small.py"]


def test_cached_ingest_result_requires_matching_source_and_revision(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setattr(repo_loader, "REPO_MANIFEST_PATH", manifest_path)

    repo = LoadedRepo(
        source="https://example.com/repo.git",
        repo_name="repo",
        repo_root="/tmp/repo",
        revision="abc123",
        files=[
            RepoFile(
                file_path="src/app.py",
                language="python",
                raw_content="def run(): pass",
                absolute_path="/tmp/repo/src/app.py",
            )
        ],
    )
    result = {"files_processed": 1, "cached": False}

    repo_loader.update_repo_manifest(repo, result)

    assert repo_loader.get_cached_ingest_result("https://example.com/repo.git", "abc123") == result
    assert repo_loader.get_cached_ingest_result("https://example.com/repo.git", "different") is None
    assert repo_loader.get_cached_ingest_result("https://example.com/other.git", "abc123") is None
