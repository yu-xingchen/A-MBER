"""
Find benchmark items where ALL specified runs score at or above a threshold.
These are candidates for "too easy" items that should be avoided in generation.

Usage:
    python scripts/langfuse_find_easy_items.py \
        --dataset-name emotion-memory-mini-s001-s005-pruned-v2-20260331 \
        --run-names "gemini25-session-local" "gemini31-session-local" "claude46-session-local" \
        --threshold 1.0 \
        --output-path docs/easy_items_analysis.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse_env import get_langfuse_host, require_env

load_dotenv()


def make_langfuse_client() -> Langfuse:
    public_key = require_env("LANGFUSE_PUBLIC_KEY", context="Langfuse client")
    secret_key = require_env("LANGFUSE_SECRET_KEY", context="Langfuse client")
    host = get_langfuse_host()
    return Langfuse(public_key=public_key, secret_key=secret_key, host=host)


def pick_effective_score(has_options: bool, scores: Dict[str, float]) -> Optional[float]:
    if has_options:
        return scores.get("exact_match")
    return scores.get("llm_judge_final")


def fetch_run_scores(
    langfuse: Langfuse,
    dataset_name: str,
    run_name: str,
) -> Dict[str, Dict[str, Any]]:
    """Returns {dataset_item_id: {effective_score, question_id, metadata, ...}}"""
    dataset = langfuse.get_dataset(dataset_name)
    items_by_id = {item.id: item for item in dataset.items}
    dataset_run = langfuse.get_dataset_run(dataset_name=dataset_name, run_name=run_name)

    results: Dict[str, Dict[str, Any]] = {}
    for run_item in dataset_run.dataset_run_items:
        dataset_item = items_by_id.get(run_item.dataset_item_id)
        if dataset_item is None:
            continue

        trace = langfuse.api.trace.get(run_item.trace_id)
        scores: Dict[str, float] = {}
        for score in trace.scores or []:
            name = getattr(score, "name", None)
            value = getattr(score, "value", None)
            if name and value is not None:
                try:
                    scores[str(name)] = float(value)
                except Exception:
                    pass

        input_payload = getattr(dataset_item, "input", {}) or {}
        metadata = getattr(dataset_item, "metadata", {}) or {}
        has_options = bool(input_payload.get("options"))
        effective_score = pick_effective_score(has_options, scores)

        results[run_item.dataset_item_id] = {
            "dataset_item_id": run_item.dataset_item_id,
            "question_id": metadata.get("qid") or metadata.get("question_id") or input_payload.get("question_id"),
            "question_text": input_payload.get("question_text"),
            "question_type": metadata.get("qtype") or metadata.get("question_type"),
            "content_type": metadata.get("content") or metadata.get("content_type"),
            "memory_level": metadata.get("memory") or metadata.get("memory_level"),
            "reasoning_structure": metadata.get("reasoning") or metadata.get("reasoning_structure"),
            "anchor_dia_id": metadata.get("anchor_dia_id"),
            "has_options": has_options,
            "gold_answer": (getattr(dataset_item, "expected_output", {}) or {}).get("gold_answer"),
            "gold_rationale": (getattr(dataset_item, "expected_output", {}) or {}).get("gold_rationale"),
            "effective_score": effective_score,
            "scores": scores,
        }

    return results


def analyze_easy_items(
    run_score_maps: List[Tuple[str, Dict[str, Dict[str, Any]]]],
    threshold: float,
) -> Dict[str, Any]:
    # Collect all item IDs across all runs
    all_item_ids = set()
    for _, score_map in run_score_maps:
        all_item_ids.update(score_map.keys())

    easy_items: List[Dict[str, Any]] = []
    partial_items: List[Dict[str, Any]] = []
    missing_in_some: List[str] = []

    for item_id in sorted(all_item_ids):
        run_scores: Dict[str, Optional[float]] = {}
        item_meta: Dict[str, Any] = {}

        for run_name, score_map in run_score_maps:
            if item_id in score_map:
                run_scores[run_name] = score_map[item_id]["effective_score"]
                if not item_meta:
                    item_meta = score_map[item_id]
            else:
                run_scores[run_name] = None
                missing_in_some.append(f"{item_id} missing in run {run_name}")

        valid_scores = [s for s in run_scores.values() if s is not None]
        if not valid_scores:
            continue

        all_above_threshold = all(s is not None and s >= threshold for s in run_scores.values())
        min_score = min(valid_scores)
        max_score = max(valid_scores)
        mean_score = sum(valid_scores) / len(valid_scores)

        entry = {
            "dataset_item_id": item_id,
            "question_id": item_meta.get("question_id"),
            "question_text": item_meta.get("question_text"),
            "question_type": item_meta.get("question_type"),
            "content_type": item_meta.get("content_type"),
            "memory_level": item_meta.get("memory_level"),
            "reasoning_structure": item_meta.get("reasoning_structure"),
            "anchor_dia_id": item_meta.get("anchor_dia_id"),
            "has_options": item_meta.get("has_options"),
            "gold_answer": item_meta.get("gold_answer"),
            "gold_rationale": item_meta.get("gold_rationale"),
            "run_scores": run_scores,
            "min_score": round(min_score, 3),
            "max_score": round(max_score, 3),
            "mean_score": round(mean_score, 3),
            "all_above_threshold": all_above_threshold,
        }

        if all_above_threshold:
            easy_items.append(entry)
        else:
            partial_items.append(entry)

    easy_items.sort(key=lambda x: (-x["mean_score"], x["question_id"] or ""))
    partial_items.sort(key=lambda x: (-x["mean_score"], x["question_id"] or ""))

    # Breakdown of easy items by dimension
    def breakdown(items: List[Dict[str, Any]], key: str) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for item in items:
            counts[str(item.get(key) or "unknown")] += 1
        return dict(sorted(counts.items()))

    return {
        "threshold": threshold,
        "total_items": len(all_item_ids),
        "easy_item_count": len(easy_items),
        "easy_item_pct": round(len(easy_items) / len(all_item_ids) * 100, 1) if all_item_ids else 0,
        "easy_by_question_type": breakdown(easy_items, "question_type"),
        "easy_by_memory_level": breakdown(easy_items, "memory_level"),
        "easy_by_content_type": breakdown(easy_items, "content_type"),
        "easy_by_reasoning": breakdown(easy_items, "reasoning_structure"),
        "easy_items": easy_items,
        "partial_items_sample": partial_items[:20],
        "missing_in_some_runs": missing_in_some,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Find items where all runs score above threshold.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--run-names", nargs="+", required=True, help="One or more run names to compare")
    parser.add_argument("--threshold", type=float, default=1.0, help="Score threshold (default: 1.0 = perfect score)")
    parser.add_argument("--output-path", default="docs/easy_items_analysis.json")
    args = parser.parse_args()

    langfuse = make_langfuse_client()

    run_score_maps: List[Tuple[str, Dict[str, Dict[str, Any]]]] = []
    for run_name in args.run_names:
        print(f"Fetching run: {run_name} ...")
        score_map = fetch_run_scores(langfuse, args.dataset_name, run_name)
        print(f"  -> {len(score_map)} items")
        run_score_maps.append((run_name, score_map))

    result = analyze_easy_items(run_score_maps, args.threshold)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nTotal items: {result['total_items']}")
    print(f"Easy items (all runs >= {args.threshold}): {result['easy_item_count']} ({result['easy_item_pct']}%)")
    print(f"\nBy question type: {result['easy_by_question_type']}")
    print(f"By memory level:  {result['easy_by_memory_level']}")
    print(f"By reasoning:     {result['easy_by_reasoning']}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
