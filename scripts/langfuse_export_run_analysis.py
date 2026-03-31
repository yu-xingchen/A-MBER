import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse_env import get_langfuse_host, require_env

load_dotenv()


def make_langfuse_client() -> Langfuse:
    public_key = require_env("LANGFUSE_PUBLIC_KEY", context="Langfuse client")
    secret_key = require_env("LANGFUSE_SECRET_KEY", context="Langfuse client")
    host = get_langfuse_host()
    return Langfuse(public_key=public_key, secret_key=secret_key, host=host)


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value)


def get_generation_observation(trace: Any) -> Optional[Any]:
    observations = list(trace.observations or [])
    generations = [obs for obs in observations if str(getattr(obs, "type", "")) == "GENERATION"]
    if not generations:
        return None
    return max(generations, key=lambda obs: getattr(obs, "latency", 0.0) or 0.0)


def get_span_observation(trace: Any) -> Optional[Any]:
    observations = list(trace.observations or [])
    spans = [obs for obs in observations if str(getattr(obs, "type", "")) == "SPAN"]
    if not spans:
        return None
    return max(spans, key=lambda obs: getattr(obs, "latency", 0.0) or 0.0)


def to_score_map(trace: Any) -> Dict[str, float]:
    score_map: Dict[str, float] = {}
    for score in trace.scores or []:
        name = getattr(score, "name", None)
        value = getattr(score, "value", None)
        if name is None or value is None:
            continue
        try:
            score_map[str(name)] = float(value)
        except Exception:
            continue
    return score_map


def get_model_output(trace: Any) -> str:
    output = getattr(trace, "output", None)
    if isinstance(output, dict):
        answer = output.get("answer")
        if isinstance(answer, str):
            return answer
        content = output.get("content")
        if isinstance(content, str):
            return content
    generation = get_generation_observation(trace)
    if generation is not None:
        gen_output = getattr(generation, "output", None)
        if isinstance(gen_output, dict):
            content = gen_output.get("content")
            if isinstance(content, str):
                return content
        if isinstance(gen_output, str):
            return gen_output
    return ""


def get_context_policy(trace: Any) -> str:
    output = getattr(trace, "output", None)
    if isinstance(output, dict) and isinstance(output.get("context_policy"), str):
        return output["context_policy"]
    return "unknown"


def compact_text(value: Optional[str], limit: int = 240) -> str:
    if not value:
        return ""
    value = " ".join(str(value).split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def pick_effective_score(row: Dict[str, Any]) -> Optional[float]:
    if row.get("has_options"):
        return row["scores"].get("exact_match")
    return row["scores"].get("llm_judge_final")


def get_metric_value(row: Dict[str, Any], metric: str) -> Optional[float]:
    if metric == "effective_score":
        value = row.get("effective_score")
    else:
        value = row["scores"].get(metric)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def group_average(rows: Iterable[Dict[str, Any]], key: str, metric: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        group_value = str(row.get(key) or "unknown")
        score = get_metric_value(row, metric)
        counts[group_value] += 1
        if score is not None:
            grouped[group_value].append(score)
    results: List[Dict[str, Any]] = []
    for group_value in sorted(counts):
        values = grouped.get(group_value, [])
        results.append(
            {
                "group": group_value,
                "count": counts[group_value],
                "mean": round(mean(values), 3) if values else None,
            }
        )
    return results


def build_rows(langfuse: Langfuse, dataset_name: str, run_name: str) -> List[Dict[str, Any]]:
    dataset = langfuse.get_dataset(dataset_name)
    items_by_id = {item.id: item for item in dataset.items}
    dataset_run = langfuse.get_dataset_run(dataset_name=dataset_name, run_name=run_name)

    rows: List[Dict[str, Any]] = []
    for run_item in dataset_run.dataset_run_items:
        dataset_item = items_by_id.get(run_item.dataset_item_id)
        if dataset_item is None:
            continue
        trace = langfuse.api.trace.get(run_item.trace_id)
        generation = get_generation_observation(trace)
        span = get_span_observation(trace)
        scores = to_score_map(trace)
        metadata = getattr(dataset_item, "metadata", {}) or {}
        input_payload = getattr(dataset_item, "input", {}) or {}
        expected_output = getattr(dataset_item, "expected_output", {}) or {}

        question_id = metadata.get("question_id") or metadata.get("qid") or input_payload.get("question_id")
        content_type = metadata.get("content_type") or metadata.get("content")
        question_type = metadata.get("question_type") or metadata.get("qtype")
        memory_level = metadata.get("memory_level") or metadata.get("memory")
        if isinstance(memory_level, str) and memory_level.isdigit():
            memory_level = f"Level {memory_level}"
        reasoning_structure = metadata.get("reasoning_structure") or metadata.get("reasoning")
        modality_condition = metadata.get("modality_condition") or metadata.get("modality")

        row = {
            "dataset_item_id": run_item.dataset_item_id,
            "trace_id": run_item.trace_id,
            "question_id": question_id,
            "unit_id": input_payload.get("unit_id"),
            "question_text": input_payload.get("question_text"),
            "question_type": question_type,
            "content_type": content_type,
            "memory_level": memory_level,
            "reasoning_structure": reasoning_structure,
            "modality_condition": modality_condition,
            "has_options": bool(input_payload.get("options")),
            "options": input_payload.get("options", []),
            "model_output": get_model_output(trace),
            "gold_answer": expected_output.get("gold_answer"),
            "acceptable_answers": expected_output.get("acceptable_answers", []),
            "gold_rationale": expected_output.get("gold_rationale"),
            "key_explanation_points": expected_output.get("key_explanation_points", []),
            "context_policy": get_context_policy(trace),
            "trace_latency_seconds": getattr(trace, "latency", None),
            "generation_latency_seconds": getattr(generation, "latency", None) if generation else None,
            "prompt_tokens": getattr(generation, "promptTokens", None) if generation else None,
            "completion_tokens": getattr(generation, "completionTokens", None) if generation else None,
            "total_tokens": getattr(generation, "totalTokens", None) if generation else None,
            "generation_model": getattr(generation, "model", None) if generation else None,
            "scores": scores,
            "effective_score": pick_effective_score({"has_options": bool(input_payload.get("options")), "scores": scores}),
            "trace_url": langfuse.get_trace_url(trace_id=trace.id),
            "anchor_text": ((input_payload.get("anchor") or {}).get("text")),
            "anchor_speaker": ((input_payload.get("anchor") or {}).get("speaker")),
            "span_output": getattr(span, "output", None) if span else None,
        }
        rows.append(row)

    rows.sort(key=lambda item: (item.get("question_id") or "", item.get("dataset_item_id") or ""))
    return rows


def build_summary(rows: List[Dict[str, Any]], dataset_name: str, run_name: str) -> Dict[str, Any]:
    metric_names = sorted({name for row in rows for name in row["scores"].keys()})
    overall_metrics = {}
    for metric in metric_names:
        values = [float(row["scores"][metric]) for row in rows if metric in row["scores"]]
        overall_metrics[metric] = round(mean(values), 3) if values else None

    effective_scores = [float(row["effective_score"]) for row in rows if row["effective_score"] is not None]
    overall_metrics["effective_score"] = round(mean(effective_scores), 3) if effective_scores else None

    low_items = sorted(
        rows,
        key=lambda row: (row["effective_score"] if row["effective_score"] is not None else 999.0, row.get("question_id") or ""),
    )[:5]

    return {
        "dataset_name": dataset_name,
        "run_name": run_name,
        "item_count": len(rows),
        "overall_metrics": overall_metrics,
        "by_question_type": group_average(rows, "question_type", "effective_score"),
        "by_memory_level": group_average(rows, "memory_level", "effective_score"),
        "by_content_type": group_average(rows, "content_type", "effective_score"),
        "lowest_items": [
            {
                "question_id": row.get("question_id"),
                "question_type": row.get("question_type"),
                "effective_score": row.get("effective_score"),
                "model_output": compact_text(row.get("model_output")),
                "gold_answer": compact_text(row.get("gold_answer")),
                "question_text": compact_text(row.get("question_text")),
                "trace_url": row.get("trace_url"),
            }
            for row in low_items
        ],
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "question_id",
        "question_type",
        "content_type",
        "memory_level",
        "reasoning_structure",
        "modality_condition",
        "has_options",
        "context_policy",
        "effective_score",
        "exact_match",
        "llm_judge_final",
        "judge_core_answer_correct",
        "judge_key_point_coverage",
        "judge_rationale_alignment",
        "judge_unsupported_claims",
        "judge_insufficiency_handling",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "generation_latency_seconds",
        "model_output",
        "gold_answer",
        "question_text",
        "trace_url",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "question_id": row.get("question_id"),
                    "question_type": row.get("question_type"),
                    "content_type": row.get("content_type"),
                    "memory_level": row.get("memory_level"),
                    "reasoning_structure": row.get("reasoning_structure"),
                    "modality_condition": row.get("modality_condition"),
                    "has_options": row.get("has_options"),
                    "context_policy": row.get("context_policy"),
                    "effective_score": row.get("effective_score"),
                    "exact_match": row["scores"].get("exact_match"),
                    "llm_judge_final": row["scores"].get("llm_judge_final"),
                    "judge_core_answer_correct": row["scores"].get("judge_core_answer_correct"),
                    "judge_key_point_coverage": row["scores"].get("judge_key_point_coverage"),
                    "judge_rationale_alignment": row["scores"].get("judge_rationale_alignment"),
                    "judge_unsupported_claims": row["scores"].get("judge_unsupported_claims"),
                    "judge_insufficiency_handling": row["scores"].get("judge_insufficiency_handling"),
                    "prompt_tokens": row.get("prompt_tokens"),
                    "completion_tokens": row.get("completion_tokens"),
                    "total_tokens": row.get("total_tokens"),
                    "generation_latency_seconds": row.get("generation_latency_seconds"),
                    "model_output": row.get("model_output"),
                    "gold_answer": row.get("gold_answer"),
                    "question_text": row.get("question_text"),
                    "trace_url": row.get("trace_url"),
                }
            )


def render_markdown_summary(summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(f"# Langfuse Run Analysis: {summary['run_name']}")
    lines.append("")
    lines.append(f"- Dataset: `{summary['dataset_name']}`")
    lines.append(f"- Items: `{summary['item_count']}`")
    lines.append("")
    lines.append("## Overall Metrics")
    lines.append("")
    for metric, value in summary["overall_metrics"].items():
        lines.append(f"- `{metric}`: `{value}`")
    lines.append("")
    for section_name, display_name in [
        ("by_question_type", "By Question Type"),
        ("by_memory_level", "By Memory Level"),
        ("by_content_type", "By Content Type"),
    ]:
        lines.append(f"## {display_name}")
        lines.append("")
        lines.append("| Group | Count | Mean Effective Score |")
        lines.append("| --- | ---: | ---: |")
        for item in summary[section_name]:
            mean_value = "" if item["mean"] is None else item["mean"]
            lines.append(f"| {item['group']} | {item['count']} | {mean_value} |")
        lines.append("")

    lines.append("## Lowest-Scoring Items")
    lines.append("")
    for item in summary["lowest_items"]:
        lines.append(f"### {item['question_id']} ({item['question_type']})")
        lines.append(f"- Effective score: `{item['effective_score']}`")
        lines.append(f"- Question: {item['question_text']}")
        lines.append(f"- Model output: {item['model_output']}")
        lines.append(f"- Gold answer: {item['gold_answer']}")
        lines.append(f"- Trace: {item['trace_url']}")
        lines.append("")

    no_option_rows = [row for row in rows if not row.get("has_options")]
    if no_option_rows:
        lines.append("## Open-Ended Error Notes")
        lines.append("")
        worst_open = sorted(
            no_option_rows,
            key=lambda row: (row["scores"].get("llm_judge_final", 999.0), row.get("question_id") or ""),
        )[:5]
        for row in worst_open:
            lines.append(f"### {row.get('question_id')}")
            lines.append(f"- `llm_judge_final`: `{row['scores'].get('llm_judge_final')}`")
            lines.append(f"- `judge_key_point_coverage`: `{row['scores'].get('judge_key_point_coverage')}`")
            lines.append(f"- `judge_rationale_alignment`: `{row['scores'].get('judge_rationale_alignment')}`")
            lines.append(f"- Question: {compact_text(row.get('question_text'), 400)}")
            lines.append(f"- Model output: {compact_text(row.get('model_output'), 400)}")
            lines.append(f"- Gold answer: {compact_text(row.get('gold_answer'), 400)}")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and analyze a Langfuse dataset run.")
    parser.add_argument("--dataset-name", required=True, help="Langfuse dataset name")
    parser.add_argument("--run-name", required=True, help="Langfuse dataset run name")
    parser.add_argument("--output-dir", default="docs/langfuse_exports", help="Output directory for exported analysis files")
    args = parser.parse_args()

    langfuse = make_langfuse_client()
    rows = build_rows(langfuse, args.dataset_name, args.run_name)
    summary = build_summary(rows, args.dataset_name, args.run_name)

    output_dir = Path(args.output_dir) / safe_name(args.dataset_name) / safe_name(args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    details_json_path = output_dir / "details.json"
    summary_json_path = output_dir / "summary.json"
    csv_path = output_dir / "items.csv"
    markdown_path = output_dir / "analysis.md"

    write_json(details_json_path, rows)
    write_json(summary_json_path, summary)
    write_csv(csv_path, rows)
    markdown_path.write_text(render_markdown_summary(summary, rows), encoding="utf-8")

    print(f"Saved detailed rows to: {details_json_path}")
    print(f"Saved summary JSON to: {summary_json_path}")
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved Markdown analysis to: {markdown_path}")


if __name__ == "__main__":
    main()
