"""
Lightweight CodeGraph evaluator — no ragas, no langchain, works with Ollama.

Scoring approach:
  - Each answer is graded 0–3 by the LLM against the ground truth
  - Runs GraphRAG and VanillaRAG side by side
  - Prints a clean comparison table
  - Saves full results to evals/results.json
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from backend.config import EVALS_DIR, EVAL_QUESTIONS_PATH, llm_call
from backend.generation.generator import generate_answer
from backend.retrieval.hybrid_merger import merge_hybrid_context, merge_vector_only_context

SCORE_PROMPT = """\
You are an evaluation judge. Score the following answer against the ground truth.

Question: {question}
Ground truth: {ground_truth}
Answer to evaluate: {answer}

Score the answer on a scale of 0 to 3:
  3 = answer is correct and complete
  2 = answer is mostly correct with minor gaps
  1 = answer is partially correct but missing key information
  0 = answer is wrong or says the information is not available

Return ONLY a single JSON object like this, nothing else:
{{"score": 2, "reason": "one sentence explanation"}}
"""


def load_eval_questions(path: Path | None = None) -> list[dict[str, str]]:
    """Load eval questions from JSON file."""
    eval_path = path or EVAL_QUESTIONS_PATH
    return json.loads(eval_path.read_text(encoding="utf-8"))


def score_answer(question: str, ground_truth: str, answer: str) -> dict[str, Any]:
    """
    Ask the LLM to score an answer against ground truth.
    Returns {"score": int, "reason": str}.
    Falls back gracefully if the LLM returns malformed JSON.
    """
    prompt = SCORE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        answer=answer,
    )
    raw = llm_call(prompt)

    # Strip markdown fences if the model wraps output in ```json ... ```
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        return {
            "score": int(parsed.get("score", 0)),
            "reason": str(parsed.get("reason", "")),
        }
    except (json.JSONDecodeError, ValueError):
        # If the model wraps JSON in extra text, scan for first valid digit
        for char in cleaned:
            if char in "0123":
                return {"score": int(char), "reason": cleaned[:200]}
        return {"score": 0, "reason": f"Could not parse: {cleaned[:200]}"}


async def _run_mode(question: str, mode: str) -> dict[str, Any]:
    """Run a single question through either graphrag or vanilla pipeline."""
    retrieval = await (
        merge_hybrid_context(question)
        if mode == "graphrag"
        else merge_vector_only_context(question)
    )
    classification = retrieval["classification"]
    answer_payload = generate_answer(
        question=question,
        merged_context=str(retrieval["context"]),
        query_type=classification.type,
        vector_hits_used=int(retrieval["metadata"]["vector_hits_count"]),
        graph_triples_used=int(retrieval["metadata"]["graph_triples_count"]),
    )
    return {
        "answer": answer_payload["answer"],
        "query_type": classification.type,
        "vector_hits": int(retrieval["metadata"]["vector_hits_count"]),
        "graph_triples": int(retrieval["metadata"]["graph_triples_count"]),
    }


async def run_evaluation(output_path: Path | None = None) -> dict[str, Any]:
    """
    Run full evaluation suite.
    For each question: run graphrag + vanilla, score both, print live progress.
    """
    items = load_eval_questions()

    # Slice to 3 per query type — fast enough for local Ollama (~20 min total)
    grouped: dict[str, list[dict[str, str]]] = {}
    for item in items:
        grouped.setdefault(item["query_type"], []).append(item)
    sliced = [item for records in grouped.values() for item in records[:3]]

    print(f"\nRunning eval on {len(sliced)} questions (3 per query type)...\n")

    results: list[dict[str, Any]] = []

    for i, item in enumerate(sliced):
        question = item["question"]
        ground_truth = item["ground_truth"]
        query_type = item["query_type"]

        print(f"[{i+1}/{len(sliced)}] {query_type.upper()}: {question[:60]}...")

        # Run both pipelines sequentially — Ollama handles one request at a time
        graph_result = await _run_mode(question, "graphrag")
        vanilla_result = await _run_mode(question, "vanilla")

        # Score both answers using the LLM as judge
        graph_score = score_answer(question, ground_truth, graph_result["answer"])
        vanilla_score = score_answer(question, ground_truth, vanilla_result["answer"])

        print(f"  GraphRAG   {graph_score['score']}/3 — {graph_score['reason'][:80]}")
        print(f"  VanillaRAG {vanilla_score['score']}/3 — {vanilla_score['reason'][:80]}")
        print()

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "query_type": query_type,
            "graphrag": {
                "answer": graph_result["answer"],
                "score": graph_score["score"],
                "reason": graph_score["reason"],
                "vector_hits": graph_result["vector_hits"],
                "graph_triples": graph_result["graph_triples"],
            },
            "vanilla": {
                "answer": vanilla_result["answer"],
                "score": vanilla_score["score"],
                "reason": vanilla_score["reason"],
            },
        })

    # Aggregate scores per query type
    summary: dict[str, dict[str, Any]] = {}
    for query_type in grouped:
        type_results = [r for r in results if r["query_type"] == query_type]
        if not type_results:
            continue
        graph_avg = sum(r["graphrag"]["score"] for r in type_results) / len(type_results)
        vanilla_avg = sum(r["vanilla"]["score"] for r in type_results) / len(type_results)
        summary[query_type] = {
            "graphrag_avg": round(graph_avg, 2),
            "vanilla_avg": round(vanilla_avg, 2),
            "improvement": round(graph_avg - vanilla_avg, 2),
            "n": len(type_results),
        }

    # Overall averages
    all_graph = [r["graphrag"]["score"] for r in results]
    all_vanilla = [r["vanilla"]["score"] for r in results]
    overall_graph = sum(all_graph) / len(all_graph) if all_graph else 0
    overall_vanilla = sum(all_vanilla) / len(all_vanilla) if all_vanilla else 0
    overall_improvement_pct = (
        (overall_graph - overall_vanilla) / max(overall_vanilla, 0.01)
    ) * 100

    # Print final table
    print("\n" + "=" * 65)
    print("BENCHMARK RESULTS  (score out of 3, judged by LLM)")
    print("=" * 65)
    print(f"{'Query Type':<14} {'Vanilla RAG':>12} {'CodeGraph':>12} {'Δ':>8}")
    print("-" * 65)
    for query_type in ("semantic", "dependency", "impact", "definition"):
        if query_type not in summary:
            continue
        s = summary[query_type]
        delta = f"+{s['improvement']:.2f}" if s["improvement"] >= 0 else f"{s['improvement']:.2f}"
        print(f"{query_type:<14} {s['vanilla_avg']:>12.2f} {s['graphrag_avg']:>12.2f} {delta:>8}")
    print("-" * 65)
    print(f"{'Overall':<14} {overall_vanilla:>12.2f} {overall_graph:>12.2f}")
    print("=" * 65)
    print(f"\nCodeGraph scores {overall_improvement_pct:+.1f}% vs Vanilla RAG overall")
    print(f"(Based on {len(results)} questions scored 0–3 by LLM-as-judge)\n")

    # Save full results
    output = {
        "summary": summary,
        "overall": {
            "graphrag_avg": round(overall_graph, 2),
            "vanilla_avg": round(overall_vanilla, 2),
            "improvement_pct": round(overall_improvement_pct, 1),
            "n_questions": len(results),
        },
        "details": results,
    }
    target_path = output_path or EVALS_DIR / "results.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Full results saved to {target_path}")

    return output


def main() -> None:
    asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()