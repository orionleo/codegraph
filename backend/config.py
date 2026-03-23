"""Configuration and provider-aware helpers for CodeGraph."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken
from dotenv import load_dotenv

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - optional dependency at runtime
    requests = None

try:
    from anthropic import Anthropic
except ModuleNotFoundError:  # pragma: no cover - optional dependency at runtime
    Anthropic = None

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - optional dependency at runtime
    OpenAI = None

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_LLM_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_LLM_TIMEOUT_SECONDS", "360"))
OLLAMA_LLM_RETRIES = int(os.getenv("OLLAMA_LLM_RETRIES", "2"))
OLLAMA_LLM_MAX_PROMPT_CHARS = int(os.getenv("OLLAMA_LLM_MAX_PROMPT_CHARS", "24000"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPOS_DIR = DATA_DIR / "repos"
REPO_MANIFEST_PATH = DATA_DIR / "repo_manifest.json"
EVALS_DIR = BASE_DIR / "evals"
EVAL_QUESTIONS_PATH = BASE_DIR / "backend" / "evaluation" / "eval_questions.json"
FRONTEND_DIR = BASE_DIR / "frontend"
DEFAULT_CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "codegraph_chunks")
USE_LLM_FILE_SUMMARIES = os.getenv("USE_LLM_FILE_SUMMARIES", "false").strip().lower() == "true"


@dataclass(slots=True)
class ProviderStatus:
    """Simple health payload for configured services."""

    provider: str
    llm_model: str
    embed_model: str
    ollama_base_url: str
    chroma_path: str
    neo4j_uri: str


def get_status() -> ProviderStatus:
    """Return the active provider configuration."""

    return ProviderStatus(
        provider=PROVIDER,
        llm_model=get_llm_model(PROVIDER),
        embed_model=get_embedding_model(PROVIDER),
        ollama_base_url=OLLAMA_BASE_URL,
        chroma_path=CHROMA_PATH,
        neo4j_uri=NEO4J_URI,
    )


def ensure_runtime_dirs() -> None:
    """Create the project directories that the pipeline depends on."""

    for path in (DATA_DIR, REPOS_DIR, EVALS_DIR, Path(CHROMA_PATH)):
        path.mkdir(parents=True, exist_ok=True)


def get_llm_model(provider: str | None = None) -> str:
    """Resolve the generation model for the given provider."""

    selected = (provider or PROVIDER).lower()
    if selected == "openai":
        return "gpt-4o-mini"
    if selected == "anthropic":
        return "claude-3-5-haiku-latest"
    return OLLAMA_MODEL


def get_embedding_model(provider: str | None = None) -> str:
    """Resolve the embedding model for the given provider."""

    selected = (provider or PROVIDER).lower()
    if selected == "openai":
        return "text-embedding-3-small"
    return OLLAMA_EMBED_MODEL


def get_token_encoding() -> tiktoken.Encoding:
    """Return a stable tokenizer for rough prompt sizing."""

    try:
        return tiktoken.encoding_for_model("gpt-4o-mini")
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def estimate_token_count(text: str) -> int:
    """Estimate token count for a block of text."""

    return len(get_token_encoding().encode(text))


def cosine_distance(vector_a: list[float], vector_b: list[float]) -> float:
    """Compute cosine distance without requiring numpy."""

    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimensionality.")
    dot = sum(a * b for a, b in zip(vector_a, vector_b))
    mag_a = math.sqrt(sum(a * a for a in vector_a))
    mag_b = math.sqrt(sum(b * b for b in vector_b))
    if mag_a == 0 or mag_b == 0:
        return 1.0
    return 1 - (dot / (mag_a * mag_b))


def safe_json_loads(raw_text: str) -> dict[str, Any]:
    """Parse model output into JSON, tolerating fenced blocks and prose noise."""

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return json.loads(cleaned[start : end + 1])


def _openai_client() -> OpenAI:
    """Create an OpenAI client."""

    if OpenAI is None:
        raise ModuleNotFoundError("openai is not installed. Run `pip install -r backend/requirements.txt`.")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=OPENAI_API_KEY)


def _anthropic_client() -> Anthropic:
    """Create an Anthropic client."""

    if Anthropic is None:
        raise ModuleNotFoundError("anthropic is not installed. Run `pip install -r backend/requirements.txt`.")
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not set.")
    return Anthropic(api_key=ANTHROPIC_API_KEY)


def _require_requests() -> None:
    """Ensure the requests dependency is available before HTTP calls."""

    if requests is None:
        raise ModuleNotFoundError("requests is not installed. Run `pip install -r backend/requirements.txt`.")


def llm_call(prompt: str, system_prompt: str | None = None, provider: str | None = None) -> str:
    """Call the configured LLM provider and return plain text output."""

    selected = (provider or PROVIDER).lower()
    if selected == "openai":
        client = _openai_client()
        response = client.chat.completions.create(
            model=get_llm_model(selected),
            messages=[
                {"role": "system", "content": system_prompt or "You are a precise assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    if selected == "anthropic":
        client = _anthropic_client()
        response = client.messages.create(
            model=get_llm_model(selected),
            system=system_prompt or "You are a precise assistant.",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1500,
        )
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")

    _require_requests()
    ollama_prompt = prompt[:OLLAMA_LLM_MAX_PROMPT_CHARS]
    last_error: Exception | None = None
    for attempt in range(OLLAMA_LLM_RETRIES + 1):
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
                json={
                    "model": get_llm_model(selected),
                    "prompt": ollama_prompt,
                    "system": system_prompt or "You are a precise assistant.",
                    "stream": False,
                    "options": {"temperature": 0},
                },
                timeout=OLLAMA_LLM_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
            return str(payload.get("response", "")).strip()
        except Exception as exc:
            last_error = exc
            if attempt < OLLAMA_LLM_RETRIES:
                time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"Ollama generation failed after retries: {last_error}")


def embed(text: str, provider: str | None = None) -> list[float]:
    """Create an embedding vector for the given text."""

    selected = (provider or PROVIDER).lower()
    if selected == "openai":
        client = _openai_client()
        response = client.embeddings.create(model=get_embedding_model(selected), input=text)
        return [float(value) for value in response.data[0].embedding]

    _require_requests()
    base_url = OLLAMA_BASE_URL.rstrip("/")
    # Keep embedding payload size stable across Ollama model/runtime versions.
    text_for_embedding = text[:8000]
    errors: list[str] = []

    for endpoint, body in (
        ("/api/embed", {"model": get_embedding_model(selected), "input": [text_for_embedding]}),
        ("/api/embeddings", {"model": get_embedding_model(selected), "prompt": text_for_embedding}),
    ):
        try:
            response = requests.post(f"{base_url}{endpoint}", json=body, timeout=180)
            if not response.ok:
                errors.append(f"{endpoint}: {response.status_code}")
                continue
            payload = response.json()
            embedding = payload.get("embedding")
            if isinstance(embedding, list):
                return [float(value) for value in embedding]
            embeddings = payload.get("embeddings")
            if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
                return [float(value) for value in embeddings[0]]
            errors.append(f"{endpoint}: missing embedding field")
        except Exception as exc:
            errors.append(f"{endpoint}: {exc}")
            continue

    raise ValueError("Ollama embedding endpoint was not found. Tried: " + ", ".join(errors))


def check_ollama_available() -> bool:
    """Return whether the Ollama HTTP endpoint is reachable."""

    if requests is None:
        return False
    try:
        response = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=10)
        return response.ok
    except Exception:
        return False
