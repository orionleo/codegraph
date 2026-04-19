"""Tests for semantic chunk construction."""

from __future__ import annotations

from backend.pipeline.code_parser import ParsedClass, ParsedFile, ParsedFunction
from backend.pipeline.embedder import CodeEmbedder


def test_build_chunks_creates_function_class_and_file_summary_chunks() -> None:
    parsed_file = ParsedFile(
        file_path="src/auth.py",
        absolute_path="/repo/src/auth.py",
        language="python",
        module_name="src.auth",
        raw_content="class Auth:\n    def login(self):\n        return validate()\n",
        functions=[
            ParsedFunction(
                type="function",
                name="login",
                signature="def login(self):",
                docstring=None,
                body="def login(self):\n    return validate()",
                calls=["validate"],
                line_start=2,
                line_end=3,
                file_path="src/auth.py",
                is_method=True,
                parent_class="Auth",
            )
        ],
        classes=[
            ParsedClass(
                type="class",
                name="Auth",
                docstring="Authentication service.",
                methods=["login"],
                bases=[],
                line_start=1,
                line_end=3,
                file_path="src/auth.py",
            )
        ],
        imports=[],
    )

    chunks = CodeEmbedder().build_chunks([parsed_file])

    assert [chunk.metadata["type"] for chunk in chunks] == ["function", "class", "file_summary"]
    assert chunks[0].chunk_id == "src/auth.py::login"
    assert chunks[1].chunk_id == "src/auth.py::Auth"
    assert "def login(self):" in chunks[1].text
    assert "This file contains 1 functions and 1 classes" in chunks[2].text
