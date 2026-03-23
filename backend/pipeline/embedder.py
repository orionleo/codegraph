"""Embedding generation and ChromaDB storage for semantic code units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import chromadb
except ModuleNotFoundError:  # pragma: no cover - optional dependency at runtime
    chromadb = None

from backend.config import CHROMA_PATH, DEFAULT_CHROMA_COLLECTION, PROVIDER, USE_LLM_FILE_SUMMARIES, embed, llm_call
from backend.pipeline.code_parser import ParsedFile


@dataclass(slots=True)
class EmbeddingChunk:
    """A semantic code chunk stored in ChromaDB."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]


def get_chroma_collection(collection_name: str = DEFAULT_CHROMA_COLLECTION) -> Any:
    """Return a persistent Chroma collection."""

    if chromadb is None:
        raise ModuleNotFoundError("chromadb is not installed. Run `pip install -r backend/requirements.txt`.")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})


def reset_chroma_collection(collection_name: str = DEFAULT_CHROMA_COLLECTION) -> None:
    """Delete and recreate the active Chroma collection."""

    if chromadb is None:
        raise ModuleNotFoundError("chromadb is not installed. Run `pip install -r backend/requirements.txt`.")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})


class CodeEmbedder:
    """Build and store semantic code chunks."""

    def __init__(self, collection_name: str = DEFAULT_CHROMA_COLLECTION) -> None:
        self.collection_name = collection_name

    def build_chunks(
        self,
        parsed_files: list[ParsedFile],
        provider: str | None = None,
        progress_callback: Any | None = None,
    ) -> list[EmbeddingChunk]:
        """Create semantic chunks for functions, classes, and file summaries."""

        chunks: list[EmbeddingChunk] = []
        total_files = max(len(parsed_files), 1)
        for index, parsed_file in enumerate(parsed_files, start=1):
            for function in parsed_file.functions:
                chunks.append(
                    EmbeddingChunk(
                        chunk_id=f"{function.file_path}::{function.name}",
                        text=function.body,
                        metadata={
                            "chunk_id": f"{function.file_path}::{function.name}",
                            "type": "function",
                            "name": function.name,
                            "file_path": function.file_path,
                            "language": parsed_file.language,
                            "signature": function.signature,
                            "line_start": function.line_start,
                            "line_end": function.line_end,
                        },
                    )
                )

            for parsed_class in parsed_file.classes:
                method_signatures = []
                for function in parsed_file.functions:
                    if function.parent_class == parsed_class.name:
                        method_signatures.append(function.signature)
                class_text = "\n".join(
                    [f"class {parsed_class.name}", parsed_class.docstring or "", *method_signatures]
                ).strip()
                chunks.append(
                    EmbeddingChunk(
                        chunk_id=f"{parsed_class.file_path}::{parsed_class.name}",
                        text=class_text,
                        metadata={
                            "chunk_id": f"{parsed_class.file_path}::{parsed_class.name}",
                            "type": "class",
                            "name": parsed_class.name,
                            "file_path": parsed_class.file_path,
                            "language": parsed_file.language,
                            "signature": "",
                            "line_start": parsed_class.line_start,
                            "line_end": parsed_class.line_end,
                        },
                    )
                )

            summary = self._summarize_file(parsed_file, provider)
            chunks.append(
                EmbeddingChunk(
                    chunk_id=f"{parsed_file.file_path}::__file_summary__",
                    text=summary,
                    metadata={
                        "chunk_id": f"{parsed_file.file_path}::__file_summary__",
                        "type": "file_summary",
                        "name": parsed_file.file_path,
                        "file_path": parsed_file.file_path,
                        "language": parsed_file.language,
                        "signature": "",
                        "line_start": 1,
                        "line_end": max(1, len(parsed_file.raw_content.splitlines())),
                    },
                    )
                )
            if progress_callback is not None:
                progress_callback(index, total_files, len(chunks))
        return chunks

    def store(
        self,
        parsed_files: list[ParsedFile],
        provider: str | None = None,
        progress_callback: Any | None = None,
    ) -> dict[str, int]:
        """Embed semantic code chunks and persist them to ChromaDB."""

        collection = get_chroma_collection(self.collection_name)
        chunks = self.build_chunks(parsed_files, provider or PROVIDER)
        total_chunks = max(len(chunks), 1)
        for index, chunk in enumerate(chunks, start=1):
            collection.upsert(
                ids=[chunk.chunk_id],
                documents=[chunk.text],
                embeddings=[embed(chunk.text, provider or PROVIDER)],
                metadatas=[chunk.metadata],
            )
            if progress_callback is not None:
                progress_callback(index, total_chunks, chunk)
        return {"chunks_embedded": len(chunks)}

    def delete_file(self, file_path: str) -> None:
        """Delete all chunks associated with a file."""

        collection = get_chroma_collection(self.collection_name)
        try:
            collection.delete(where={"file_path": file_path})
        except Exception:
            pass

    def _summarize_file(self, parsed_file: ParsedFile, provider: str | None = None) -> str:
        """Generate a compact file summary for retrieval."""

        symbol_list = ", ".join(
            [function.name for function in parsed_file.functions[:8]]
            + [parsed_class.name for parsed_class in parsed_file.classes[:8]]
        )
        if not USE_LLM_FILE_SUMMARIES:
            if parsed_file.functions or parsed_file.classes:
                return (
                    f"This file contains {len(parsed_file.functions)} functions and "
                    f"{len(parsed_file.classes)} classes. Key symbols: {symbol_list or parsed_file.file_path}."
                )
            return f"This file contains {parsed_file.language} source code at {parsed_file.file_path}."
        prompt = (
            "Summarize this source file in 1-2 sentences. "
            "Start with 'This file contains...'. "
            f"Language: {parsed_file.language}\n"
            f"Path: {parsed_file.file_path}\n"
            f"Symbols: {symbol_list or 'None'}\n"
            f"Code:\n{parsed_file.raw_content[:6000]}"
        )
        try:
            return llm_call(prompt, system_prompt="You summarize code files precisely.", provider=provider or PROVIDER)
        except Exception:
            if parsed_file.functions or parsed_file.classes:
                return (
                    f"This file contains {len(parsed_file.functions)} functions and "
                    f"{len(parsed_file.classes)} classes: {symbol_list or parsed_file.file_path}."
                )
            return f"This file contains {parsed_file.language} source code at {parsed_file.file_path}."
