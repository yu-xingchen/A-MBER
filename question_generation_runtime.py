"""Question-generation runtime helpers.

This module owns the operational part of QA generation:

- compact payload construction for model calls
- question enum/metadata normalization
- invalid-question detection
- retry payload construction
- per-phase generation with validation-aware retries

It intentionally does not decide phase sizing or final benchmark pruning.
Those concerns live in ``question_phase_planning.py`` and
``question_curation.py`` / ``question_pipeline.py``.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from conversation_utils import flatten_conversation_turns, normalize_level_label
from validators import (
    DataValidationError,
    contains_forbidden_question_reference,
    normalize_reasoning_structure,
    validate_questions,
)


QUESTION_TYPE_ALIASES = {
    "judgement": "judgment",
    "modality missing": "modality_missing",
    "modality ambiguous": "modality_ambiguous",
}

CONTENT_TYPE_ALIASES = {
    "instant emotion": "instant_emotion",
    "long term implicit emotion": "long_term_implicit_emotion",
    "long term explicit or semi explicit emotion": "long_term_explicit_or_semi_explicit_emotion",
    "long term fact": "long_term_fact",
    "near term fact": "near_term_fact",
    "relation state": "relation_state",
    "relation change": "relation_change",
    "modality conflict explanation": "modality_conflict_explanation",
}

MODALITY_CONDITION_ALIASES = {
    "voice style removed": "voice_style_removed",
    "text removed": "text_removed",
    "modality ambiguous": "modality_ambiguous",
    "voice style ambiguous": "voice_style_ambiguous",
    "text ambiguous": "text_ambiguous",
}

ADVERSARIAL_TYPE_ALIASES = {
    "pseudo relevant history": "pseudo_relevant_history",
    "insufficient evidence": "insufficient_evidence",
    "pseudo conflict": "pseudo_conflict",
}


def compact_json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def normalize_enum_token(value: Any, aliases: Dict[str, str], fallback: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    normalized = value.strip().lower().replace("-", " ").replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return aliases.get(normalized, normalized.replace(" ", "_"))


def infer_memory_level_from_evidence(question: Dict[str, Any], conversation: Dict[str, Any]) -> str:
    turn_map = flatten_conversation_turns(conversation)
    anchor_dia_id = question.get("anchor_dia_id")
    if anchor_dia_id not in turn_map:
        return str(question.get("memory_level") or "Level 0")

    anchor_turn = turn_map[anchor_dia_id]
    anchor_index = anchor_turn["turn_index_global"]
    anchor_session = int(anchor_dia_id.split(":")[0][1:])
    evidence_ids = [dia_id for dia_id in question.get("evidence_turn_ids", []) if dia_id in turn_map]
    prior_ids = [dia_id for dia_id in evidence_ids if turn_map[dia_id]["turn_index_global"] < anchor_index]
    if not prior_ids:
        return "Level 0"

    prior_sessions = {int(dia_id.split(":")[0][1:]) for dia_id in prior_ids}
    if any(session < anchor_session for session in prior_sessions):
        if len(prior_sessions) >= 2 or len(prior_ids) >= 3:
            return "Level 3"
        return "Level 2"
    return "Level 1"


def normalize_questions_metadata(questions: List[Dict[str, Any]], conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
    normalized_questions: List[Dict[str, Any]] = []
    for question in questions:
        if not isinstance(question, dict):
            normalized_questions.append(question)
            continue
        normalized_question = dict(question)
        normalized_question["memory_level"] = infer_memory_level_from_evidence(normalized_question, conversation)
        normalized_question["question_type"] = normalize_enum_token(
            normalized_question.get("question_type"),
            QUESTION_TYPE_ALIASES,
            "judgment",
        )
        normalized_question["content_type"] = normalize_enum_token(
            normalized_question.get("content_type"),
            CONTENT_TYPE_ALIASES,
            "long_term_implicit_emotion",
        )
        normalized_question["reasoning_structure"] = normalize_reasoning_structure(
            normalized_question.get("reasoning_structure")
        ) or "direct"
        if not isinstance(normalized_question.get("options"), list):
            normalized_question["options"] = []
        if not isinstance(normalized_question.get("acceptable_answers"), list):
            normalized_question["acceptable_answers"] = []
        if not isinstance(normalized_question.get("critical_event_ids"), list):
            normalized_question["critical_event_ids"] = []
        if not isinstance(normalized_question.get("evidence_turn_ids"), list):
            normalized_question["evidence_turn_ids"] = []
        if not isinstance(normalized_question.get("key_explanation_points"), list):
            normalized_question["key_explanation_points"] = []
        normalized_question["modality_condition"] = normalize_enum_token(
            normalized_question.get("modality_condition"),
            MODALITY_CONDITION_ALIASES,
            "normal",
        )
        if normalized_question.get("adversarial_flag"):
            normalized_question["adversarial_type"] = normalize_enum_token(
                normalized_question.get("adversarial_type"),
                ADVERSARIAL_TYPE_ALIASES,
                "",
            ) or None
        else:
            normalized_question["adversarial_type"] = None
        normalized_questions.append(normalized_question)
    return normalized_questions


def normalize_text_signature(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def collect_duplicate_question_reasons(questions: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    seen_by_signature: Dict[tuple[str, str], str] = {}
    duplicates: Dict[str, List[str]] = {}
    for question in questions:
        if not isinstance(question, dict):
            continue
        question_id = str(question.get("question_id") or "<unknown>")
        signature = (
            str(question.get("anchor_dia_id") or ""),
            normalize_text_signature(question.get("question_text")),
        )
        if not signature[0] or not signature[1]:
            continue
        if signature in seen_by_signature:
            duplicates[question_id] = [
                f"duplicates question wording for the same anchor as {seen_by_signature[signature]}"
            ]
            continue
        seen_by_signature[signature] = question_id
    return duplicates


def jaccard_overlap(left: List[str], right: List[str]) -> float:
    left_set = {str(item) for item in left if str(item).strip()}
    right_set = {str(item) for item in right if str(item).strip()}
    if not left_set and not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def is_closure_signal_question(question: Dict[str, Any], turn_map: Dict[str, Dict[str, Any]]) -> bool:
    anchor_dia_id = str(question.get("anchor_dia_id") or "")
    question_text = str(question.get("question_text") or "").lower()
    anchor_text = ""
    if anchor_dia_id in turn_map:
        anchor_text = str(turn_map[anchor_dia_id].get("text") or "").lower()

    closure_markers = [
        "goodbye",
        "see you",
        "see you next week",
        "thank you again",
        "casual goodbye",
        "relaxed goodbye",
        "final goodbye",
        "closing",
        "sign-off",
        "signoff",
    ]
    combined = f"{question_text} {anchor_text}"
    return any(marker in combined for marker in closure_markers)


def collect_signal_duplicate_question_reasons(
    questions: List[Dict[str, Any]],
    conversation: Dict[str, Any],
) -> Dict[str, List[str]]:
    turn_map = flatten_conversation_turns(conversation)
    duplicates: Dict[str, List[str]] = {}
    kept_closure_questions: List[Dict[str, Any]] = []

    for question in questions:
        if not isinstance(question, dict):
            continue
        question_id = str(question.get("question_id") or "<unknown>")
        if not is_closure_signal_question(question, turn_map):
            continue

        for prior in kept_closure_questions:
            same_type = str(prior.get("question_type") or "") == str(question.get("question_type") or "")
            overlap = jaccard_overlap(
                prior.get("evidence_turn_ids", []),
                question.get("evidence_turn_ids", []),
            )
            if same_type and overlap >= 0.5:
                duplicates[question_id] = [
                    f"duplicates closure/relationship-easing signal already covered by {prior.get('question_id')}"
                ]
                break
        else:
            kept_closure_questions.append(question)

    return duplicates


def build_question_curation_hints(
    questions: List[Dict[str, Any]],
    conversation: Dict[str, Any],
) -> Dict[str, Dict[str, List[str]]]:
    turn_map = flatten_conversation_turns(conversation)
    duplicate_wording_reasons = collect_duplicate_question_reasons(questions)
    duplicate_signal_reasons = collect_signal_duplicate_question_reasons(questions, conversation)
    hints: Dict[str, Dict[str, List[str]]] = {}

    for question in questions:
        if not isinstance(question, dict):
            continue
        question_id = str(question.get("question_id") or "<unknown>")
        tags: List[str] = []
        reasons: List[str] = []

        question_type = str(question.get("question_type") or "")
        content_type = str(question.get("content_type") or "")
        memory_level = str(question.get("memory_level") or "")
        reasoning_structure = str(question.get("reasoning_structure") or "")
        evidence_turn_ids = [str(item) for item in question.get("evidence_turn_ids", []) if str(item).strip()]

        if (
            memory_level in {"Level 0", "Level 1"}
            and reasoning_structure in {"direct", "single-hop"}
        ) or (
            question_type == "retrieval"
            and content_type in {"near_term_fact", "long_term_fact"}
            and memory_level == "Level 0"
        ):
            tags.append("likely_easy")
            reasons.append(
                "Low-memory and low-hop structure make this item more likely to be solvable without strong long-term integration."
            )

        if question_type == "retrieval" and (
            content_type == "near_term_fact"
            or memory_level == "Level 0"
            or reasoning_structure == "direct"
        ):
            tags.append("likely_low_value_retrieval")
            reasons.append(
                "This retrieval item appears closer to local factual recall than to long-horizon emotional memory."
            )

        if is_closure_signal_question(question, turn_map):
            tags.append("closure_signal")
            reasons.append(
                "This item is anchored on a closing / goodbye / relational easing signal, which can become repetitive in longer conversations."
            )

        if question_id in duplicate_wording_reasons:
            tags.append("likely_duplicate_wording")
            reasons.extend(duplicate_wording_reasons[question_id])

        if question_id in duplicate_signal_reasons:
            tags.append("likely_duplicate_signal")
            reasons.extend(duplicate_signal_reasons[question_id])

        if len(evidence_turn_ids) <= 1 and memory_level in {"Level 0", "Level 1"}:
            tags.append("thin_evidence")
            reasons.append(
                "This item relies on very little evidence and may function more as a local control question than a memory-intensive benchmark item."
            )

        hints[question_id] = {
            "tags": list(dict.fromkeys(tags)),
            "reasons": list(dict.fromkeys(reasons)),
        }

    return hints


def collect_invalid_question_reasons(questions: List[Dict[str, Any]], conversation: Dict[str, Any]) -> Dict[str, List[str]]:
    turn_map = flatten_conversation_turns(conversation)
    invalid: Dict[str, List[str]] = {}
    for question in questions:
        if not isinstance(question, dict):
            continue
        question_id = str(question.get("question_id") or "<unknown>")
        reasons: List[str] = []
        anchor_dia_id = question.get("anchor_dia_id")
        question_text = str(question.get("question_text") or "").lower()
        question_type = str(question.get("question_type") or "")
        adversarial_type = str(question.get("adversarial_type") or "")
        modality_condition = str(question.get("modality_condition") or "")

        if anchor_dia_id in turn_map:
            anchor_index = turn_map[anchor_dia_id]["turn_index_global"]
            for evidence_dia_id in question.get("evidence_turn_ids", []):
                if evidence_dia_id in turn_map and turn_map[evidence_dia_id]["turn_index_global"] > anchor_index:
                    reasons.append(f"uses future evidence turn {evidence_dia_id} after anchor {anchor_dia_id}")
                    break

        if question_type in {"modality_missing", "modality_ambiguous"} or modality_condition in {
            "voice_style_removed",
            "text_removed",
            "modality_ambiguous",
            "voice_style_ambiguous",
            "text_ambiguous",
        }:
            if "instead of" in question_text or re.search(r"\bif\b.+\bwas described as\b", question_text):
                reasons.append("injects replacement modality description")

        if adversarial_type == "pseudo_conflict":
            if re.search(r"\bif\b.+\bhad previously stated\b", question_text):
                reasons.append("injects fabricated prior dialogue")

        if question_type == "retrieval":
            if question_text.startswith("explain ") or " how does it relate" in question_text:
                reasons.append("retrieval-labeled item is actually explanatory")

        if contains_forbidden_question_reference(str(question.get("question_text") or "")):
            reasons.append("question text explicitly mentions session labels or turn IDs")

        if reasons:
            invalid[question_id] = reasons
    for question_id, reasons in collect_duplicate_question_reasons(questions).items():
        invalid.setdefault(question_id, []).extend(reasons)
    for question_id, reasons in collect_signal_duplicate_question_reasons(questions, conversation).items():
        invalid.setdefault(question_id, []).extend(reasons)
    return invalid


def drop_invalid_questions(questions: List[Dict[str, Any]], invalid_reasons: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    return [
        question for question in questions
        if str(question.get("question_id") or "<unknown>") not in invalid_reasons
    ]


def build_compact_question_conversation_payload(conversation: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {
        "conversation_id": conversation.get("conversation_id"),
        "speaker_a": conversation.get("speaker_a"),
        "speaker_b": conversation.get("speaker_b"),
        "target_speaker": conversation.get("target_speaker"),
    }
    for key, turns in conversation.items():
        if key.startswith("session_") and key.endswith("_date_time"):
            compact[key] = conversation.get(key)
            continue
        if not (key.startswith("session_") and isinstance(turns, list)):
            continue
        compact_turns: List[Dict[str, Any]] = []
        for turn in turns:
            compact_turns.append(
                {
                    "speaker": turn.get("speaker"),
                    "dia_id": turn.get("dia_id"),
                    "turn_index_global": turn.get("turn_index_global"),
                    "text": turn.get("text"),
                    "voice_style": turn.get("voice_style"),
                    "modality_available": turn.get("modality_available"),
                    "notes_for_dataset_builder": turn.get("notes_for_dataset_builder"),
                }
            )
        compact[key] = compact_turns
    return compact


def build_compact_annotation_payload(annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact_items: List[Dict[str, Any]] = []
    for item in annotations:
        compact_items.append(
            {
                "dia_id": item.get("dia_id"),
                "underlying_emotion": item.get("underlying_emotion"),
                "secondary_emotion": item.get("secondary_emotion"),
                "implicit_explicit": item.get("implicit_explicit"),
                "expression_style": item.get("expression_style"),
                "emotion_intensity": item.get("emotion_intensity"),
                "relation_state": item.get("relation_state"),
                "historical_memory_required": item.get("historical_memory_required"),
                "memory_dependency_level": item.get("memory_dependency_level"),
                "reasoning_structure": item.get("reasoning_structure"),
                "critical_event_ids": item.get("critical_event_ids"),
                "evidence_turn_ids": item.get("evidence_turn_ids"),
                "gold_rationale": item.get("gold_rationale"),
            }
        )
    return compact_items


def build_compact_event_plan_payload(event_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact_events: List[Dict[str, Any]] = []
    for event in event_plan:
        compact_events.append(
            {
                "event_id": event.get("event_id"),
                "session_id": event.get("session_id"),
                "event_type": event.get("event_type"),
                "description": event.get("description"),
                "relation_impact": event.get("relation_impact"),
                "can_be_critical_memory": event.get("can_be_critical_memory"),
                "can_be_distractor": event.get("can_be_distractor"),
            }
        )
    return compact_events


def build_question_user_payload(
    conversation: Dict[str, Any],
    annotations: List[Dict[str, Any]],
    event_plan: List[Dict[str, Any]],
    question_plan: Dict[str, Any],
) -> str:
    return f"""
Generate benchmark questions.

[conversation.json]
{compact_json_dumps(build_compact_question_conversation_payload(conversation))}

[annotation.json]
{compact_json_dumps(build_compact_annotation_payload(annotations))}

[event_plan.json]
{compact_json_dumps(build_compact_event_plan_payload(event_plan))}

[question_plan.json]
{compact_json_dumps(question_plan)}
""".strip()


def build_question_retry_user_payload(
    conversation: Dict[str, Any],
    annotations: List[Dict[str, Any]],
    event_plan: List[Dict[str, Any]],
    question_plan: Dict[str, Any],
    validation_error: str,
    previous_questions: List[Dict[str, Any]],
) -> str:
    base_payload = build_question_user_payload(conversation, annotations, event_plan, question_plan)
    return (
        f"{base_payload}\n\n"
        f"[previous_validation_error]\n{validation_error}\n\n"
        f"[previous_invalid_qa.json]\n{compact_json_dumps(previous_questions)}\n\n"
        "Regenerate the full question set so it passes validation. Keep the intended benchmark difficulty and distribution, "
        "but fix the invalid items. Do not introduce replacement modality cues, fabricated prior statements, future-evidence leakage, "
        "retrieval items that are actually explanations, or duplicate questions for the same anchor."
    )


def _format_invalid_reasons(invalid_reasons: Dict[str, List[str]]) -> str:
    return "; ".join(
        f"{question_id}: {', '.join(reasons)}"
        for question_id, reasons in invalid_reasons.items()
    )


def generate_question_phase(
    client: Any,
    system_prompt: str,
    conversation: Dict[str, Any],
    annotations: List[Dict[str, Any]],
    event_plan: List[Dict[str, Any]],
    phase_plan: Dict[str, Any],
    schemas_dir: Path,
) -> List[Dict[str, Any]]:
    user_payload = build_question_user_payload(
        conversation=conversation,
        annotations=annotations,
        event_plan=event_plan,
        question_plan=phase_plan,
    )
    last_error: str | None = None
    for attempt in range(3):
        questions = client.chat_json_array(system_prompt=system_prompt, user_prompt=user_payload)
        questions = normalize_questions_metadata(questions, conversation)
        invalid_reasons = collect_invalid_question_reasons(questions, conversation)
        try:
            if invalid_reasons:
                raise DataValidationError(f"qa content validation failed: {_format_invalid_reasons(invalid_reasons)}")
            validate_questions(questions, conversation, schemas_dir / "qa.schema.json")
            return questions
        except DataValidationError as exc:
            last_error = str(exc)
            if attempt == 2:
                pruned_questions = drop_invalid_questions(questions, invalid_reasons)
                if not pruned_questions:
                    raise
                validate_questions(pruned_questions, conversation, schemas_dir / "qa.schema.json")
                dropped_ids = ", ".join(sorted(invalid_reasons))
                print(f"Dropped invalid questions after retries: {dropped_ids}")
                return pruned_questions
            user_payload = build_question_retry_user_payload(
                conversation=conversation,
                annotations=annotations,
                event_plan=event_plan,
                question_plan=phase_plan,
                validation_error=str(exc),
                previous_questions=questions,
            )
    raise DataValidationError(last_error or "Question generation phase failed without a specific validation error.")
