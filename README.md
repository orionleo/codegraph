# CodeGraph

CodeGraph is a code-aware GraphRAG system for repository understanding.

Instead of treating a codebase like plain text, it parses source files with `tree-sitter`, builds a Neo4j graph of symbols and relationships, stores semantic code chunks in ChromaDB, and answers engineering questions with a hybrid graph + vector retrieval pipeline.

This project was built to answer the kinds of questions developers actually ask during onboarding, debugging, refactoring, and code reviews:

- Where is this class defined?
- What calls this function?
- What breaks if I change this module?
- How does this component work end to end?

## Why It Stands Out

- Code-aware retrieval, not generic document RAG
- Explicit graph reasoning over `CALLS`, `CONTAINS`, `DEFINED_IN`, `IMPORTS`, and `INHERITS_FROM`
- Hybrid retrieval that changes behavior based on query intent
- FastAPI backend, Neo4j graph layer, Chroma vector layer, single-file frontend
- Provider switch between local Ollama and cloud models via `.env`
- Built without LangChain or LlamaIndex

## Demo Snapshot

CodeGraph supports four practical codebase query types:

| Query Type | Example |
|---|---|
| `definition` | `Where is Signer defined?` |
| `semantic` | `How does Signer work?` |
| `dependency` | `What functions call unsign?` |
| `impact` | `What breaks if I change Serializer.loads?` |

This makes it a much better fit for real code analysis than vanilla RAG over chunks alone.

## Architecture

```text
Repository
   |
   v
tree-sitter parsing
   |
   +----------------------------+
   |                            |
   v                            v
Neo4j code graph          Chroma semantic index
   |                            |
   +------------+---------------+
                |
                v
        Query classifier
                |
                v
   Graph search + Vector search
                |
                v
         Context merger
                |
                v
          Answer generator
```

## What It Does

### Ingestion pipeline

CodeGraph can ingest either:

- a local repository path
- a GitHub repository URL

During ingest it:

1. walks supported source files
2. parses functions, classes, imports, and callsites with `tree-sitter`
3. writes graph nodes and edges into Neo4j
4. embeds semantic code units into ChromaDB
5. caches repeated ingests of the same unchanged repo

### Graph schema

The graph captures code structure with node types such as:

- `File`
- `Module`
- `Class`
- `Function`

and relationships such as:

- `CALLS`
- `DEFINED_IN`
- `CONTAINS`
- `IMPORTS`
- `INHERITS_FROM`
- `BELONGS_TO`

### Query flow

When a user asks a question, CodeGraph:

1. classifies the question type
2. extracts the main code entity
3. runs graph and vector retrieval with different weighting
4. merges the retrieved context
5. generates a grounded answer with confidence and source references

## Tech Stack

- `FastAPI` for the backend API
- `Neo4j` for structural code relationships
- `ChromaDB` for semantic retrieval
- `tree-sitter` for AST-based code parsing
- `Ollama`, `OpenAI`, or `Anthropic` for model access
- plain HTML/CSS/JS frontend with D3-based graph visualization

## Key Engineering Decisions

### 1. AST parsing over regex

Code is structured data. `tree-sitter` provides reliable symbol extraction and call detection, which is much more defensible than regex-based parsing.

### 2. Graph + vectors instead of only one

Vector search is useful for “what does this code do?” style questions.
Graph traversal is useful for “what calls this?” and “what breaks if I change this?” style questions.

CodeGraph combines both because real codebase understanding needs both.

### 3. Query-aware retrieval

A definition lookup and an impact-analysis question should not use the same retrieval strategy.

CodeGraph routes each query into a retrieval mode tuned for:

- exact lookups
- semantic explanation
- dependency tracing
- impact analysis

### 4. Fast local iteration

The project is designed for practical demoability:

- local Ollama support
- repo ingest progress reporting
- graph exploration endpoint
- incremental file update path
- cached re-ingest for unchanged repos

## Project Structure

```text
codegraph/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── progress.py
│   ├── pipeline/
│   │   ├── repo_loader.py
│   │   ├── code_parser.py
│   │   ├── graph_builder.py
│   │   ├── embedder.py
│   │   └── diff_updater.py
│   ├── retrieval/
│   │   ├── query_classifier.py
│   │   ├── vector_search.py
│   │   ├── graph_search.py
│   │   └── hybrid_merger.py
│   ├── generation/
│   │   └── generator.py
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   └── eval_questions.json
│   └── requirements.txt
├── frontend/
│   └── index.html
├── data/
├── evals/
├── .env.example
└── README.md
```

## Quick Start

From [/Users/jai/Documents/Projects_AI/codegraph](/Users/jai/Documents/Projects_AI/codegraph):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env
```

### Local Ollama path

```bash
ollama pull llama3.2
ollama pull nomic-embed-text

docker run -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

python -m uvicorn backend.main:app \
  --reload \
  --app-dir . \
  --reload-exclude 'data/*' \
  --reload-exclude 'chroma_db/*'
```

### OpenAI path

Set this in `.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

Then run:

```bash
python -m uvicorn backend.main:app --reload --app-dir .
```

## API Surface

- `POST /ingest`
- `GET /ingest/progress`
- `POST /query`
- `POST /update`
- `GET /graph/stats`
- `GET /graph/explore`
- `GET /health`

## Test Suite

CodeGraph includes a `pytest` suite that covers deterministic business logic and mocked API behavior without requiring live Neo4j, ChromaDB, Ollama, OpenAI, or Anthropic.

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=backend --cov-report=term-missing
```

The tests cover:

- repository filtering and ingest cache helpers
- query classification fallback behavior
- answer confidence/source extraction
- semantic chunk construction
- vector result ranking
- ingest progress state handling
- FastAPI route behavior with external services mocked

## Example Usage

### Ingest a repo

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source":"https://github.com/pallets/itsdangerous"}'
```

### Query the code graph

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Where is Signer defined?","mode":"graphrag"}'
```

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What functions call unsign?","mode":"graphrag"}'
```

## Good Repos To Demo

- [pallets/itsdangerous](https://github.com/pallets/itsdangerous)
- [pallets/werkzeug](https://github.com/pallets/werkzeug)
- [tiangolo/fastapi](https://github.com/tiangolo/fastapi)

## Recruiter-Friendly Summary

If someone scans only one section of this README, the takeaway should be:

> CodeGraph is a full-stack AI systems project that combines AST parsing, graph databases, vector retrieval, API design, and developer tooling into one practical application for codebase understanding.

It demonstrates:

- backend engineering
- systems integration
- AI application design
- retrieval architecture
- developer-focused product thinking

## Notes

- Re-ingesting the same unchanged repository returns a cached result.
- Ingest currently prioritizes speed and stability over exhaustive language coverage.
- Python, JavaScript, and TypeScript are the primary structured parse targets.
- The project is intentionally framework-light so the retrieval and graph logic stay transparent.
