"""LLM-based code query classification."""

from __future__ import annotations

from dataclasses import dataclass
import re

from backend.config import PROVIDER, llm_call, safe_json_loads

QUERY_TYPES = {
    "semantic": "How does X work / explain X / what is X",
    "dependency": "What calls X / what depends on X / what uses X",
    "impact": "What breaks if I change X / what would be affected",
    "definition": "Where is X defined / find X / show me X",
}

QUERY_WEIGHTS = {
    "semantic": {"vector_weight": 0.8, "graph_weight": 0.2, "vector_limit": 10, "graph_limit": 3},
    "dependency": {"vector_weight": 0.2, "graph_weight": 0.8, "vector_limit": 4, "graph_limit": 10},
    "impact": {"vector_weight": 0.1, "graph_weight": 0.9, "vector_limit": 3, "graph_limit": 10},
    "definition": {"vector_weight": 0.0, "graph_weight": 1.0, "vector_limit": 0, "graph_limit": 10},
}

PROMPT = """Classify the following code question into exactly one of these types:
semantic, dependency, impact, definition.

Return ONLY a JSON object: {"type": "<type>", "entity": "<main code entity mentioned>"}

Question: {question}
"""


@dataclass(slots=True)
class QueryClassification:
    """Routing output for a code question."""

    type: str
    entity: str
    vector_weight: float
    graph_weight: float
    vector_limit: int
    graph_limit: int


def classify_query(question: str, provider: str | None = None) -> QueryClassification:
    """Classify a question and return retrieval weights."""

    try:
        raw = llm_call(
            PROMPT.format(question=question),
            system_prompt="You are a precise classifier for codebase questions.",
            provider=provider or PROVIDER,
        )
        payload = safe_json_loads(raw)
        query_type = str(payload.get("type", "semantic")).strip().lower()
        entity = str(payload.get("entity", "")).strip() or _fallback_entity(question)
    except Exception:
        query_type = _fallback_type(question)
        entity = _fallback_entity(question)

    if query_type not in QUERY_WEIGHTS:
        query_type = _fallback_type(question)
    weights = QUERY_WEIGHTS[query_type]
    return QueryClassification(
        type=query_type,
        entity=entity,
        vector_weight=weights["vector_weight"],
        graph_weight=weights["graph_weight"],
        vector_limit=weights["vector_limit"],
        graph_limit=weights["graph_limit"],
    )


def _fallback_type(question: str) -> str:
    lowered = question.lower()
    if any(term in lowered for term in ("what calls", "depends on", "what uses", "used by", "caller")):
        return "dependency"
    if any(term in lowered for term in ("what breaks", "affected", "impact", "if i change")):
        return "impact"
    if any(term in lowered for term in ("where is", "find ", "show me", "defined")):
        return "definition"
    return "semantic"


def _fallback_entity(question: str) -> str:
    normalized = re.sub(r"[^\w./:()-]+", " ", question)
    tokens = [token.strip(" ?.,:") for token in normalized.split() if token.strip(" ?.,:")]

    preferred = []
    for token in tokens:
        if "_" in token:
            preferred.append(token)
            continue
        if re.search(r"[A-Z]", token[1:]):
            preferred.append(token)
            continue
        if token.isidentifier() and len(token) > 2 and token.lower() not in _STOPWORDS:
            preferred.append(token)

    if preferred:
        return preferred[-1]
    meaningful = [token for token in tokens if token.lower() not in _STOPWORDS]
    return meaningful[-1] if meaningful else ""


_STOPWORDS = {
    "how",
    "does",
    "work",
    "what",
    "where",
    "find",
    "show",
    "defined",
    "breaks",
    "change",
    "used",
    "uses",
    "call",
    "calls",
    "function",
    "functions",
    "class",
    "classes",
    "if",
    "i",
    "me",
    "the",
    "a",
    "an",
    "is",
    "are",
    "of",
    "to",
    "on",
    "in",
    "for",
    "and",
    "or",
}
