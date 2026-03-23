"""FastAPI entry point for CodeGraph."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import FRONTEND_DIR, PROVIDER, check_ollama_available, ensure_runtime_dirs, get_status
from backend.generation.generator import generate_answer
from backend.pipeline.code_parser import parse_repo_files
from backend.pipeline.diff_updater import DiffUpdater
from backend.pipeline.embedder import CodeEmbedder, get_chroma_collection, reset_chroma_collection
from backend.pipeline.graph_builder import Neo4jGraphStore
from backend.pipeline.repo_loader import RepoLoader, get_cached_ingest_result, update_repo_manifest
from backend.progress import INGEST_PROGRESS
from backend.retrieval.hybrid_merger import merge_hybrid_context, merge_vector_only_context

app = FastAPI(title="CodeGraph", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ensure_runtime_dirs()


class IngestRequest(BaseModel):
    """Incoming repo ingest request."""

    source: str = Field(..., min_length=1)


class QueryRequest(BaseModel):
    """Incoming question request."""

    question: str = Field(..., min_length=1)
    mode: str = Field("graphrag", pattern="^(graphrag|vanilla)$")


class UpdateRequest(BaseModel):
    """Incoming incremental update request."""

    file_path: str = Field(..., min_length=1)


if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")


@app.get("/")
async def root() -> FileResponse:
    """Serve the frontend."""

    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(index_path)


@app.post("/ingest")
async def ingest(payload: IngestRequest) -> dict[str, int | str]:
    """Load a repo, parse code, write Neo4j, and store semantic embeddings."""

    try:
        INGEST_PROGRESS.reset("Loading repository...")
        loaded_repo = RepoLoader().load(payload.source)
        cached_result = get_cached_ingest_result(payload.source, loaded_repo.revision)
        if cached_result is not None:
            result = dict(cached_result)
            result["cached"] = True
            INGEST_PROGRESS.finish(
                f"Using cached ingest for {loaded_repo.repo_name}.",
                **result,
            )
            return result
        INGEST_PROGRESS.update(
            "loading",
            10,
            f"Loaded repository with {len(loaded_repo.files)} source files.",
            files_discovered=len(loaded_repo.files),
        )
        parsed_files = parse_repo_files(loaded_repo.files)
        INGEST_PROGRESS.update("parsing", 35, f"Parsed {len(parsed_files)} files into AST structures.")
        with Neo4jGraphStore() as graph_store:
            graph_store.reset_graph()
            graph_stats = graph_store.build_graph(parsed_files)
        INGEST_PROGRESS.update(
            "graph",
            70,
            f"Graph written: {graph_stats['nodes_created']} nodes, {graph_stats['edges_created']} edges.",
            nodes_created=graph_stats["nodes_created"],
            edges_created=graph_stats["edges_created"],
        )
        reset_chroma_collection()
        embed_stats = CodeEmbedder().store(
            parsed_files,
            progress_callback=lambda current, total, _: INGEST_PROGRESS.update(
                "embedding",
                70 + int((current / max(total, 1)) * 29),
                f"Embedding chunks {current}/{total}...",
                chunks_processed=current,
                chunks_total=total,
            ),
        )
        result = {
            "source": payload.source,
            "repo_name": loaded_repo.repo_name,
            "revision": loaded_repo.revision,
            "files_processed": int(graph_stats["files_processed"]),
            "nodes_created": int(graph_stats["nodes_created"]),
            "edges_created": int(graph_stats["edges_created"]),
            "chunks_embedded": int(embed_stats["chunks_embedded"]),
            "cached": False,
        }
        update_repo_manifest(loaded_repo, result)
        INGEST_PROGRESS.finish(
            f"Ingested {result['files_processed']} files, {result['nodes_created']} nodes, {result['edges_created']} edges.",
            **result,
        )
        return result
    except Exception as exc:
        INGEST_PROGRESS.fail(str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/query")
async def query(payload: QueryRequest) -> dict[str, object]:
    """Answer a code question using GraphRAG or vanilla vector retrieval."""

    try:
        retrieval = (
            await merge_hybrid_context(payload.question)
            if payload.mode == "graphrag"
            else await merge_vector_only_context(payload.question)
        )
        classification = retrieval["classification"]
        answer = generate_answer(
            question=payload.question,
            merged_context=str(retrieval["context"]),
            query_type=classification.type,
            vector_hits_used=int(retrieval["metadata"]["vector_hits_count"]),
            graph_triples_used=int(retrieval["metadata"]["graph_triples_count"]),
        )
        answer["mode"] = payload.mode
        answer["entity"] = classification.entity
        answer["retrieval_context"] = retrieval["context"]
        answer["vector_results"] = retrieval["vector_hits"]
        answer["graph_results"] = retrieval["graph_records"]
        return answer
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/update")
async def update(payload: UpdateRequest) -> dict[str, int | str]:
    """Re-index a single changed file."""

    try:
        return DiffUpdater().update(payload.file_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/graph/stats")
async def graph_stats() -> dict[str, object]:
    """Expose graph stats for monitoring and the frontend."""

    try:
        with Neo4jGraphStore() as graph_store:
            return graph_store.get_stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/graph/explore")
async def graph_explore(entity: str = Query(..., min_length=1), depth: int = Query(2, ge=1, le=3)) -> dict[str, object]:
    """Return a small subgraph for frontend visualization."""

    try:
        with Neo4jGraphStore() as graph_store:
            return graph_store.explore(entity=entity, depth=depth)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
async def health() -> dict[str, object]:
    """Return best-effort service readiness."""

    status = get_status()
    try:
        with Neo4jGraphStore() as graph_store:
            graph_store.get_stats()
        neo4j_connected = True
    except Exception:
        neo4j_connected = False

    try:
        get_chroma_collection()
        chroma_ready = True
    except Exception:
        chroma_ready = False

    return {
        "provider": status.provider,
        "neo4j_connected": neo4j_connected,
        "chroma_ready": chroma_ready,
        "ollama_available": check_ollama_available() if PROVIDER == "ollama" else False,
    }


@app.get("/ingest/progress")
async def ingest_progress() -> dict[str, object]:
    """Return current ingest progress for the frontend."""

    return INGEST_PROGRESS.snapshot()
