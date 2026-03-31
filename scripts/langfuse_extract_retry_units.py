import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set


def load_units(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_item_blocks(report_text: str) -> List[str]:
    return re.findall(r"(?ms)^\d+\.\sItem\s\d+:.*?(?=^\d+\.\sItem\s\d+:|\Z)", report_text)


def extract_completed_unit_ids(report_text: str) -> Set[str]:
    completed: Set[str] = set()
    for block in extract_item_blocks(report_text):
        unit_id_match = re.search(r"'unit_id': '([^']+)'", block)
        if unit_id_match:
            completed.add(unit_id_match.group(1).strip())
    return completed


def extract_retry_unit_ids_by_answer_substrings(report_text: str, substrings: List[str]) -> Set[str]:
    if not substrings:
        return set()
    lowered_substrings = [s.lower() for s in substrings if s.strip()]
    retry_unit_ids: Set[str] = set()
    for block in extract_item_blocks(report_text):
        unit_id_match = re.search(r"'unit_id': '([^']+)'", block)
        actual_match = re.search(r"Actual:\s+(\{.*?\})(?:\s*\n\s*Scores:|\s*\n\s*Trace ID:)", block, re.S)
        if not unit_id_match or not actual_match:
            continue
        actual_text = actual_match.group(1).lower()
        if any(substring in actual_text for substring in lowered_substrings):
            retry_unit_ids.add(unit_id_match.group(1).strip())
    return retry_unit_ids


def build_retry_units(
    *,
    source_units: List[Dict[str, Any]],
    completed_unit_ids: Set[str],
    error_unit_ids: Set[str],
) -> List[Dict[str, Any]]:
    retry_units: List[Dict[str, Any]] = []
    for unit in source_units:
        unit_id = str(unit.get("unit_id") or "").strip()
        if not unit_id:
            continue
        if unit_id not in completed_unit_ids or unit_id in error_unit_ids:
            retry_units.append(unit)
    return retry_units


def extract_completed_keys(report_text: str) -> Set[tuple[str, str]]:
    blocks = re.findall(r"(?ms)^\d+\.\sItem\s\d+:.*?(?=^\d+\.\sItem\s\d+:|\Z)", report_text)
    completed: Set[tuple[str, str]] = set()
    for block in blocks:
        qtext_match = re.search(r"'question_text': ([\"'])(.+?)\1, 'benchmark_views'", block, re.S)
        anchor_text_match = re.search(r"'anchor': \{'text': ([\"'])(.+?)\1,", block, re.S)
        if not qtext_match:
            continue
        question_text = qtext_match.group(2).replace("\\'", "'").strip()
        anchor_text = anchor_text_match.group(2).replace("\\'", "'").strip() if anchor_text_match else ""
        completed.add((question_text, anchor_text))
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract retry units by diffing source units against a Langfuse local report.")
    parser.add_argument("--report-path", required=True, help="Path to the local Langfuse report txt")
    parser.add_argument("--source-units-path", required=True, help="Path to the source all_units.json used for the run")
    parser.add_argument("--out-path", required=True, help="Path to save missing/retry units json")
    parser.add_argument(
        "--retry-on-answer-substring",
        action="append",
        default=[],
        help="If Actual.answer contains this substring, include the unit in retry output. Repeatable.",
    )
    args = parser.parse_args()

    report_text = Path(args.report_path).read_text(encoding="utf-8", errors="ignore")
    source_units = load_units(Path(args.source_units_path))
    completed_unit_ids = extract_completed_unit_ids(report_text)
    error_unit_ids = extract_retry_unit_ids_by_answer_substrings(report_text, args.retry_on_answer_substring)
    source_unit_ids = {
        str(unit.get("unit_id") or "").strip()
        for unit in source_units
        if str(unit.get("unit_id") or "").strip()
    }

    # Backward-compatible fallback for old reports that may not carry stable unit ids.
    if not completed_unit_ids or len(completed_unit_ids) < len(source_units):
        completed_keys = extract_completed_keys(report_text)
        retry_units = []
        for unit in source_units:
            question_text = str(unit.get("task_layer", {}).get("question_text") or "").strip()
            anchor_text = str(unit.get("anchor", {}).get("text") or "").strip()
            if (question_text, anchor_text) not in completed_keys:
                retry_units.append(unit)
    else:
        retry_units = build_retry_units(
            source_units=source_units,
            completed_unit_ids=completed_unit_ids,
            error_unit_ids=error_unit_ids,
        )

    Path(args.out_path).write_text(json.dumps(retry_units, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Extracted {len(retry_units)} retry units to: {args.out_path}")


if __name__ == "__main__":
    main()
