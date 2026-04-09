import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse_env import get_langfuse_host, require_env

load_dotenv()


def load_units(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower()


def short_hash(*parts: Any, length: int = 12) -> str:
    payload = "||".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def make_langfuse_client() -> Langfuse:
    public_key = require_env("LANGFUSE_PUBLIC_KEY", context="Langfuse client")
    secret_key = require_env("LANGFUSE_SECRET_KEY", context="Langfuse client")
    host = get_langfuse_host()
    return Langfuse(public_key=public_key, secret_key=secret_key, host=host)


def build_dataset_item_input(unit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "unit_id": unit["unit_id"],
        "question_id": unit["question_id"],
        "question_text": unit["task_layer"]["question_text"],
        "options": unit["task_layer"].get("options", []),
        "anchor": {
            "speaker": unit["anchor"].get("speaker"),
            "text": unit["anchor"].get("text"),
            "timestamp": unit["anchor"].get("timestamp"),
            "voice_style": unit["anchor"].get("voice_style"),
        },
        "benchmark_views": {
            "session_local_context": unit["benchmark_views"].get("session_local_context", []),
            "full_history_context": unit["benchmark_views"].get("full_history_context", []),
            "gold_evidence_context": unit.get("history_evidence", []),
            "modality_conditioned_views": unit["benchmark_views"].get("modality_conditioned_views"),
        },
    }


def build_dataset_item_expected_output(unit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "gold_answer": unit["gold"].get("gold_answer"),
        "acceptable_answers": unit["gold"].get("acceptable_answers", []),
        "gold_rationale": unit["gold"].get("gold_rationale"),
        "key_explanation_points": unit["gold"].get("key_explanation_points", []),
    }


def build_source_marker(unit: Dict[str, Any], units_path: Path, index: int) -> str:
    anchor = unit.get("anchor", {})
    return "::".join(
        [
            units_path.stem,
            str(index),
            str(unit.get("conversation_id") or "conversation"),
            str(unit.get("question_id") or unit.get("unit_id") or "item"),
            str(anchor.get("dia_id") or "no-anchor"),
        ]
    )


def build_dataset_item_metadata(unit: Dict[str, Any], units_path: Path, index: int) -> Dict[str, Any]:
    task_layer = unit.get("task_layer", {})
    anchor = unit.get("anchor", {})
    memory_level = str(task_layer.get("memory_level") or "")
    memory_level_short = memory_level.replace("Level ", "").strip() if memory_level.startswith("Level ") else memory_level
    modality_condition = str(task_layer.get("modality_condition") or "normal").strip().lower() or "normal"
    modality_short_map = {
        "normal": "normal",
        "voice_style_missing": "missing",
        "voice_style_removed": "missing",
        "modality_missing": "missing",
        "voice_style_ambiguous": "low_quality",
        "modality_ambiguous": "low_quality",
        "text_ambiguous": "low_quality",
        "text_removed": "missing",
    }
    return {
        "qid": unit.get("question_id"),
        "unit_id": unit.get("unit_id"),
        "conversation_id": unit.get("conversation_id"),
        "anchor_dia_id": anchor.get("dia_id"),
        "source_units_file": units_path.name,
        "source_marker": build_source_marker(unit, units_path, index),
        "content": task_layer.get("content_type"),
        "qtype": task_layer.get("question_type"),
        "memory": memory_level_short,
        "reasoning": task_layer.get("reasoning_structure"),
        "modality": modality_short_map.get(modality_condition, modality_condition),
    }


def build_dataset_item_id(dataset_name: str, unit: Dict[str, Any]) -> str:
    anchor = unit.get("anchor", {})
    task_layer = unit.get("task_layer", {})
    identity_hash = short_hash(
        unit.get("unit_id"),
        unit.get("conversation_id"),
        unit.get("question_id"),
        anchor.get("dia_id"),
        anchor.get("text"),
        task_layer.get("question_text"),
    )
    return "__".join(
        [
            slugify(dataset_name),
            slugify(str(unit.get("conversation_id") or "conversation")),
            slugify(str(unit.get("question_id") or unit.get("unit_id") or "item")),
            identity_hash,
        ]
    )


def validate_unique_dataset_item_ids(units: List[Dict[str, Any]], dataset_name: str) -> None:
    collisions: Dict[str, List[str]] = {}
    for index, unit in enumerate(units):
        item_id = build_dataset_item_id(dataset_name, unit)
        descriptor = build_source_marker(unit, Path("unknown.json"), index)
        collisions.setdefault(item_id, []).append(descriptor)

    duplicates = {item_id: markers for item_id, markers in collisions.items() if len(markers) > 1}
    if duplicates:
        sample = next(iter(duplicates.items()))
        raise ValueError(
            "Duplicate Langfuse dataset item ids detected before upload. "
            f"Example collision: {sample[0]} -> {sample[1]}"
        )


def ensure_dataset(langfuse: Langfuse, name: str, description: str) -> None:
    try:
        langfuse.get_dataset(name)
    except Exception:
        langfuse.create_dataset(
            name=name,
            description=description,
        )


def upload_units(
    *,
    units_path: Path,
    dataset_name: str,
    description: str,
    max_items: int | None,
) -> None:
    langfuse = make_langfuse_client()
    ensure_dataset(langfuse, dataset_name, description)
    units = load_units(units_path)
    if max_items is not None:
        units = units[:max_items]

    validate_unique_dataset_item_ids(units, dataset_name)

    for index, unit in enumerate(units):
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            id=build_dataset_item_id(dataset_name, unit),
            input=build_dataset_item_input(unit),
            expected_output=build_dataset_item_expected_output(unit),
            metadata=build_dataset_item_metadata(unit, units_path, index),
        )

    print(f"Uploaded {len(units)} items to Langfuse dataset: {dataset_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload benchmark units to a Langfuse dataset.")
    parser.add_argument("--units-path", required=True, help="Path to all_units.json")
    parser.add_argument("--dataset-name", required=True, help="Langfuse dataset name")
    parser.add_argument("--description", default="Emotion memory benchmark dataset upload", help="Dataset description")
    parser.add_argument("--max-items", type=int, default=None, help="Optional limit for uploaded items")
    args = parser.parse_args()

    upload_units(
        units_path=Path(args.units_path),
        dataset_name=args.dataset_name,
        description=args.description,
        max_items=args.max_items,
    )


if __name__ == "__main__":
    main()
