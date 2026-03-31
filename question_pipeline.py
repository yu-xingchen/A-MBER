"""Question-generation orchestration.

This module sits between the top-level pipeline entrypoints and the lower-level
question helpers. It is responsible for stage sequencing only:

- build hard/support phase inputs
- run hard-core generation first
- resize support generation based on kept hard items
- merge, prune, validate, and save the final QA set

It should not contain prompt payload construction details or low-level retry
logic; those belong in ``question_generation_runtime.py``. It should also avoid
holding question-distribution policy constants that are better isolated in
``question_phase_planning.py``.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import question_curation as qc
from question_generation_runtime import (
    collect_invalid_question_reasons,
    drop_invalid_questions,
    generate_question_phase,
)
from question_phase_planning import build_question_phase_plan, resize_question_phase_plan
from validators import validate_questions


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def renumber_questions(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    renumbered: List[Dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        updated = dict(question)
        updated["question_id"] = f"Q{index:03d}"
        renumbered.append(updated)
    return renumbered


def prepare_question_generation_inputs(
    bundle: Dict[str, Any],
    conversation: Dict[str, Any],
    annotations: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    hard_plan = build_question_phase_plan(bundle["question_plan"], "hard_core")
    support_plan = build_question_phase_plan(bundle["question_plan"], "everything_else")
    closure_profile = str(bundle.get("global_outline", {}).get("closure_profile") or "")
    hard_annotations, preferred_anchor_ids, discouraged_anchor_ids = qc.build_hard_core_annotation_subset(
        annotations=annotations,
        conversation=conversation,
        question_plan=hard_plan,
        closure_profile=closure_profile,
    )
    hard_plan["preferred_anchor_dia_ids"] = preferred_anchor_ids
    if discouraged_anchor_ids:
        hard_plan["discouraged_anchor_dia_ids"] = discouraged_anchor_ids
    support_plan["avoid_anchor_dia_ids"] = preferred_anchor_ids
    return hard_plan, support_plan, hard_annotations


def run_hard_core_question_phase(
    client: Any,
    system_prompt: str,
    conversation: Dict[str, Any],
    hard_annotations: List[Dict[str, Any]],
    event_plan: List[Dict[str, Any]],
    hard_plan: Dict[str, Any],
    schemas_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    hard_questions = generate_question_phase(
        client=client,
        system_prompt=system_prompt,
        conversation=conversation,
        annotations=hard_annotations,
        event_plan=event_plan,
        phase_plan=hard_plan,
        schemas_dir=schemas_dir,
    )
    hard_minimum_keep = max(6, int(hard_plan.get("target_question_count", len(hard_questions))))
    return qc.postprocess_hard_core_questions(
        hard_questions,
        conversation,
        minimum_keep=hard_minimum_keep,
    )


def adjust_support_phase_plan(
    support_plan: Dict[str, Any],
    hard_questions: List[Dict[str, Any]],
    demoted_hard_questions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    support_target_count = min(
        int(support_plan.get("target_question_count", 0)),
        len(hard_questions),
    )
    adjusted_support_plan = resize_question_phase_plan(
        support_plan,
        new_target_count=max(0, support_target_count),
    )
    if demoted_hard_questions:
        demoted_anchor_ids = [
            str(question.get("anchor_dia_id") or "")
            for question in demoted_hard_questions
            if str(question.get("anchor_dia_id") or "").strip()
        ]
        support_avoid = [str(item) for item in adjusted_support_plan.get("avoid_anchor_dia_ids", []) if str(item).strip()]
        adjusted_support_plan["avoid_anchor_dia_ids"] = list(dict.fromkeys([*support_avoid, *demoted_anchor_ids]))
        print(
            "Demoted structurally soft hard-core questions: "
            + ", ".join(str(question.get("question_id") or "<unknown>") for question in demoted_hard_questions)
        )
    return adjusted_support_plan


def finalize_question_set(
    questions: List[Dict[str, Any]],
    conversation: Dict[str, Any],
    final_target_count: int,
    schemas_dir: Path,
    out_path: Path,
) -> List[Dict[str, Any]]:
    questions = renumber_questions(questions)
    questions = renumber_questions(
        qc.prune_questions_to_target(
            questions,
            conversation,
            target_count=final_target_count,
        )
    )
    invalid_reasons = collect_invalid_question_reasons(questions, conversation)
    if invalid_reasons:
        questions = renumber_questions(drop_invalid_questions(questions, invalid_reasons))
        if invalid_reasons:
            dropped_ids = ", ".join(sorted(invalid_reasons))
            print(f"Dropped invalid questions after phase merge: {dropped_ids}")
    validate_questions(questions, conversation, schemas_dir / "qa.schema.json")
    save_json(questions, out_path)
    return questions


def generate_questions_from_inputs(
    client: Any,
    system_prompt: str,
    schemas_dir: Path,
    bundle: Dict[str, Any],
    conversation: Dict[str, Any],
    annotations: List[Dict[str, Any]],
    out_path: Path,
) -> List[Dict[str, Any]]:
    hard_plan, support_plan, hard_annotations = prepare_question_generation_inputs(
        bundle=bundle,
        conversation=conversation,
        annotations=annotations,
    )
    hard_questions, demoted_hard_questions = run_hard_core_question_phase(
        client=client,
        system_prompt=system_prompt,
        conversation=conversation,
        hard_annotations=hard_annotations,
        event_plan=bundle["event_plan"],
        hard_plan=hard_plan,
        schemas_dir=schemas_dir,
    )
    support_plan = adjust_support_phase_plan(
        support_plan=support_plan,
        hard_questions=hard_questions,
        demoted_hard_questions=demoted_hard_questions,
    )
    support_questions = generate_question_phase(
        client=client,
        system_prompt=system_prompt,
        conversation=conversation,
        annotations=annotations,
        event_plan=bundle["event_plan"],
        phase_plan=support_plan,
        schemas_dir=schemas_dir,
    )
    final_target_count = len(hard_questions) + int(support_plan.get("target_question_count", len(support_questions)))
    return finalize_question_set(
        questions=hard_questions + support_questions,
        conversation=conversation,
        final_target_count=final_target_count,
        schemas_dir=schemas_dir,
        out_path=out_path,
    )
