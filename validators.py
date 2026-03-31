import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from jsonschema import Draft202012Validator


class DataValidationError(ValueError):
    """Raised when a generated file is structurally valid JSON but semantically unusable."""


REASONING_STRUCTURE_ALIASES = {
    "conflict_resolution": "conflict-resolution",
    "trajectory_based": "trajectory-based",
    "single_hop": "single-hop",
    "multi_hop": "multi-hop",
    "aggregation": "multi-hop",
    "cross-session": "multi-hop",
    "causal-chain": "multi-hop",
    "contrastive": "conflict-resolution",
    "contrast": "conflict-resolution",
}

VALID_REASONING_STRUCTURES = {
    "direct",
    "single-hop",
    "multi-hop",
    "conflict-resolution",
    "trajectory-based",
    *REASONING_STRUCTURE_ALIASES.keys(),
}

NON_EMOTION_TERMS = {
    "politeness",
    "self_minimization",
    "validation",
    "directness",
    "humor",
    "joking",
    "withdrawal",
    "people_pleasing",
}

QUESTION_TEXT_FORBIDDEN_REFERENCE_PATTERNS = [
    re.compile(r"\bsession\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bs\d+\b", re.IGNORECASE),
    re.compile(r"\bd\d+:\d+\b", re.IGNORECASE),
]

VOICE_STYLE_FORBIDDEN_RULES = [
    (
        "emotion_distress",
        re.compile(r"\b(anxious|anxiety|apprehensive|nervous|worried|stressed|overwhelmed)\b", re.IGNORECASE),
    ),
    (
        "emotion_negative",
        re.compile(r"\b(frustrated|frustration|defensive|disappointed|disappointment|ashamed|shame)\b", re.IGNORECASE),
    ),
    (
        "emotion_positive",
        re.compile(r"\b(relieved|relief|grateful|gratitude|content|confident|hopeful|hurt)\b", re.IGNORECASE),
    ),
    (
        "emotion_other",
        re.compile(r"\b(angry|anger|sad|fearful|afraid|resentful|embarrassed)\b", re.IGNORECASE),
    ),
    (
        "interpretive_verb",
        re.compile(r"\b(masking|showing|revealing|indicating|suggesting|signifying|reflecting)\b", re.IGNORECASE),
    ),
    (
        "hidden_state_reference",
        re.compile(r"\b(feeling|feels|emotion|emotional|underlying)\b", re.IGNORECASE),
    ),
]

VOICE_STYLE_FORBIDDEN_PATTERNS = [pattern for _, pattern in VOICE_STYLE_FORBIDDEN_RULES]


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise DataValidationError(message)


def has_non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def contains_forbidden_question_reference(text: str) -> bool:
    return any(pattern.search(text) for pattern in QUESTION_TEXT_FORBIDDEN_REFERENCE_PATTERNS)


def voice_style_leaks_hidden_state(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in VOICE_STYLE_FORBIDDEN_PATTERNS)


def get_voice_style_leak_matches(text: str) -> List[str]:
    normalized = str(text or "").strip()
    if not normalized:
        return []
    matches: List[str] = []
    for label, pattern in VOICE_STYLE_FORBIDDEN_RULES:
        if pattern.search(normalized):
            matches.append(label)
    return matches


def validate_emotion_terms(terms: List[str], label: str) -> None:
    for term in terms:
        normalized = str(term).strip().lower().replace(" ", "_")
        ensure(normalized not in NON_EMOTION_TERMS, f"{label} contains non-emotion term {term!r}")


def validate_against_schema(data: Any, schema_path: Path, label: str) -> None:
    """Validate against JSON Schema and surface the first error with a readable path."""
    schema = load_json(schema_path)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    if not errors:
        return

    first = errors[0]
    path = ".".join(str(part) for part in first.path) or "<root>"
    raise DataValidationError(f"{label} schema validation failed at {path}: {first.message}")


def normalize_reasoning_structure(value: Optional[str]) -> Optional[str]:
    """Accept a few historical aliases so older generated files still validate cleanly."""
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    return REASONING_STRUCTURE_ALIASES.get(normalized, normalized)


def normalize_reasoning_structure_in_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_items: List[Dict[str, Any]] = []
    for item in items:
        normalized_item = dict(item)
        normalized_item["reasoning_structure"] = normalize_reasoning_structure(item.get("reasoning_structure"))
        normalized_items.append(normalized_item)
    return normalized_items


def validate_generation_config(
    config: Dict[str, Any],
    schema_path: Path,
) -> None:
    """Validate batch generation config and basic cross-field constraints."""
    validate_against_schema(config, schema_path, "generation_config.json")
    ensure(
        config["turns_per_conversation"] >= config["sessions_per_conversation"] * 2,
        "turns_per_conversation must allow at least 2 turns per session",
    )
    ensure(
        config["stage_count"] <= config["sessions_per_conversation"],
        "stage_count cannot exceed sessions_per_conversation",
    )
    ensure(
        config["question_count"] <= config["turns_per_conversation"] * 2,
        "question_count is unrealistically large relative to turns_per_conversation",
    )


def validate_blueprint_bundle(
    personas: Dict[str, Any],
    global_outline: Dict[str, Any],
    session_scripts: List[Dict[str, Any]],
    event_plan: List[Dict[str, Any]],
    emotion_arc: List[Dict[str, Any]],
    question_plan: Dict[str, Any],
    schemas_dir: Path,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Validate generated planning artifacts and their cross-file consistency.

    Schema checks catch missing fields; the additional rules here catch planning bundles
    that are structurally valid JSON but logically unusable for benchmark generation.
    """
    validate_against_schema(personas, schemas_dir / "personas.schema.json", "personas.json")
    validate_against_schema(global_outline, schemas_dir / "global_outline.schema.json", "global_outline.json")
    validate_against_schema(session_scripts, schemas_dir / "session_scripts.schema.json", "session_scripts.json")
    validate_against_schema(event_plan, schemas_dir / "event_plan.schema.json", "event_plan.json")
    validate_against_schema(emotion_arc, schemas_dir / "emotion_arc.schema.json", "emotion_arc.json")
    validate_against_schema(question_plan, schemas_dir / "question_plan.schema.json", "question_plan.json")

    ensure("student" in personas and "teacher" in personas, "personas must include student and teacher")

    stages = global_outline.get("stages", [])
    stage_ids = [stage["stage_id"] for stage in stages]
    session_ids = [session["session_id"] for session in session_scripts]

    ensure(global_outline["total_sessions"] == len(session_scripts), "global_outline total_sessions must match session_scripts length")
    ensure(global_outline["total_turns"] == sum(session["turn_count"] for session in session_scripts), "global_outline total_turns must equal sum of session turn counts")
    ensure(len(stage_ids) > 0, "global_outline must contain at least one stage")
    ensure(len(set(stage_ids)) == len(stage_ids), "stage_id values must be unique")
    ensure(len(set(session_ids)) == len(session_ids), "session_id values must be unique")

    expected_session_ids = [f"S{i}" for i in range(1, len(session_scripts) + 1)]
    ensure(session_ids == expected_session_ids, f"session_ids must be sequential: expected {expected_session_ids}, got {session_ids}")

    covered_sessions: Set[int] = set()
    for stage in stages:
        span = stage["session_span"]
        ensure(span, f"stage {stage['stage_id']} session_span must not be empty")
        for session_number in span:
            ensure(1 <= session_number <= len(session_scripts), f"stage {stage['stage_id']} references out-of-range session {session_number}")
            covered_sessions.add(session_number)
    ensure(covered_sessions == set(range(1, len(session_scripts) + 1)), "global_outline stages must cover every session")

    for session in session_scripts:
        ensure(session["stage_id"] in stage_ids, f"session {session['session_id']} references unknown stage_id {session['stage_id']}")
        ensure(session["turn_count"] >= 2, f"session {session['session_id']} must have at least 2 turns")
        validate_emotion_terms(session.get("dominant_student_emotions", []), f"session {session['session_id']} dominant_student_emotions")

    for event in event_plan:
        ensure(event["session_id"] in session_ids, f"event {event['event_id']} references unknown session_id {event['session_id']}")

    event_counts_by_session = {session_id: 0 for session_id in session_ids}
    for event in event_plan:
        if event["session_id"] in event_counts_by_session:
            event_counts_by_session[event["session_id"]] += 1

    for session in session_scripts:
        min_event_count = 2 if session["turn_count"] >= 10 else 1
        ensure(
            event_counts_by_session[session["session_id"]] >= min_event_count,
            f"session {session['session_id']} should have at least {min_event_count} events for its turn count",
        )

    for arc_item in emotion_arc:
        ensure(arc_item["stage_id"] in stage_ids, f"emotion_arc references unknown stage_id {arc_item['stage_id']}")
        validate_emotion_terms(arc_item.get("student_dominant_emotions", []), f"emotion_arc {arc_item['stage_id']} student_dominant_emotions")
        validate_emotion_terms(arc_item.get("implicit_emotions_to_seed", []), f"emotion_arc {arc_item['stage_id']} implicit_emotions_to_seed")

    for stage in stages:
        validate_emotion_terms(stage.get("emotional_background", []), f"global_outline stage {stage['stage_id']} emotional_background")

    if config is not None:
        ensure(global_outline["total_sessions"] == config["sessions_per_conversation"], "blueprint total_sessions does not match generation config")
        ensure(global_outline["total_turns"] == config["turns_per_conversation"], "blueprint total_turns does not match generation config")
        ensure(len(stage_ids) == config["stage_count"], "blueprint stage count does not match generation config")
        ensure(question_plan["mvp_question_count"] == config["question_count"], "question_plan mvp_question_count does not match generation config")


def get_session_turns(conversation: Dict[str, Any]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Return only turn-bearing session arrays, ordered by session key."""
    session_items: List[Tuple[str, List[Dict[str, Any]]]] = []
    for key, value in conversation.items():
        if key.startswith("session_") and not key.endswith("_date_time") and isinstance(value, list):
            session_items.append((key, value))
    return sorted(session_items, key=lambda item: item[0])


def flatten_conversation_turns(conversation: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Index every dialogue turn by dia_id for fast cross-file consistency checks."""
    turn_map: Dict[str, Dict[str, Any]] = {}
    for key, value in conversation.items():
        if key.startswith("session_") and isinstance(value, list):
            for turn in value:
                dia_id = turn.get("dia_id")
                if not dia_id:
                    raise DataValidationError(f"Missing dia_id in turn under {key}: {turn}")
                turn_map[dia_id] = turn
    return turn_map


def build_annotation_map(annotations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    ann_map: Dict[str, Dict[str, Any]] = {}
    for ann in annotations:
        dia_id = ann.get("dia_id")
        if not dia_id:
            raise DataValidationError(f"Annotation missing dia_id: {ann}")
        ann_map[dia_id] = ann
    return ann_map


def validate_conversation(
    conversation: Dict[str, Any],
    schema_path: Path,
) -> None:
    """Validate conversation schema plus turn ordering, speaker identity, and modality metadata.

    This layer makes sure later annotation/question steps can safely treat the conversation
    as a contiguous chronological record instead of re-checking basic turn integrity.
    """
    validate_against_schema(conversation, schema_path, "conversation.json")

    allowed_speakers = {
        conversation.get("speaker_a"),
        conversation.get("speaker_b"),
    }
    ensure(
        conversation.get("target_speaker") in allowed_speakers,
        "conversation target_speaker must be one of speaker_a / speaker_b",
    )

    seen_dia_ids: Set[str] = set()
    seen_turn_indices: Set[int] = set()
    previous_turn_index = 0

    for session_key, turns in get_session_turns(conversation):
        session_id = session_key.split("_")[1]
        ensure(
            f"{session_key}_date_time" in conversation,
            f"{session_key} is missing its paired date field",
        )
        ensure(turns, f"{session_key} must not be empty")

        for turn in turns:
            dia_id = turn["dia_id"]
            turn_index_global = turn["turn_index_global"]

            ensure(dia_id not in seen_dia_ids, f"Duplicate dia_id found: {dia_id}")
            ensure(
                turn["speaker"] in allowed_speakers,
                f"{dia_id} uses unknown speaker {turn['speaker']!r}",
            )
            ensure(
                dia_id.startswith(f"D{session_id}:"),
                f"{dia_id} does not match session key {session_key}",
            )
            ensure(
                has_non_empty_text(turn.get("timestamp")),
                f"{dia_id} is missing timestamp",
            )
            ensure(
                turn_index_global not in seen_turn_indices,
                f"Duplicate turn_index_global found: {turn_index_global}",
            )
            ensure(
                turn_index_global == previous_turn_index + 1,
                f"turn_index_global must be contiguous; expected {previous_turn_index + 1}, got {turn_index_global} at {dia_id}",
            )
            ensure(
                "text" in turn["modality_available"],
                f"{dia_id} must include 'text' in modality_available",
            )
            ensure(
                has_non_empty_text(turn.get("voice_style")),
                f"{dia_id} is missing voice_style",
            )
            ensure(
                not voice_style_leaks_hidden_state(str(turn.get("voice_style", ""))),
                f"{dia_id} voice_style should describe delivery only, not emotion labels or interpretation",
            )

            audio_id = turn.get("audio_id")
            if isinstance(audio_id, str) and audio_id.strip():
                ensure(
                    audio_id.startswith(dia_id.replace(":", "_")),
                    f"{dia_id} audio_id {audio_id!r} should align with dia_id",
                )
            else:
                ensure(
                    "audio" not in turn["modality_available"],
                    f"{dia_id} lists audio modality but audio_id is empty",
                )

            seen_dia_ids.add(dia_id)
            seen_turn_indices.add(turn_index_global)
            previous_turn_index = turn_index_global

    ensure(seen_dia_ids, "conversation must contain at least one dialogue turn")


def validate_conversation_against_blueprint(
    conversation: Dict[str, Any],
    global_outline: Dict[str, Any],
    session_scripts: List[Dict[str, Any]],
) -> None:
    """Ensure the realized conversation actually matches the planned multi-session bundle."""
    session_turns = get_session_turns(conversation)
    expected_session_count = int(global_outline.get("total_sessions", len(session_scripts)))
    expected_total_turns = int(global_outline.get("total_turns", sum(int(session.get("turn_count", 0)) for session in session_scripts)))

    ensure(
        len(session_turns) == expected_session_count,
        f"conversation has {len(session_turns)} sessions but blueprint expects {expected_session_count}",
    )

    expected_session_keys = [f"session_{index}" for index in range(1, expected_session_count + 1)]
    actual_session_keys = [key for key, _ in session_turns]
    ensure(
        actual_session_keys == expected_session_keys,
        f"conversation session keys do not match blueprint order: expected {expected_session_keys}, got {actual_session_keys}",
    )

    actual_total_turns = sum(len(turns) for _, turns in session_turns)
    ensure(
        actual_total_turns == expected_total_turns,
        f"conversation has {actual_total_turns} turns but blueprint expects {expected_total_turns}",
    )

    ensure(
        len(session_scripts) == expected_session_count,
        "session_scripts length does not match blueprint total_sessions",
    )

    for index, ((session_key, turns), session_script) in enumerate(zip(session_turns, session_scripts), start=1):
        expected_turn_count = int(session_script.get("turn_count", 0))
        ensure(
            len(turns) == expected_turn_count,
            f"{session_key} has {len(turns)} turns but session_scripts expects {expected_turn_count}",
        )
        expected_session_id = f"S{index}"
        ensure(
            str(session_script.get("session_id")) == expected_session_id,
            f"session_scripts out of order: expected {expected_session_id}, got {session_script.get('session_id')!r}",
        )


def validate_annotations(
    annotations: List[Dict[str, Any]],
    conversation: Dict[str, Any],
    schema_path: Path,
) -> None:
    """Validate annotations against schema and ensure evidence links resolve back to conversation turns."""
    normalized_annotations = normalize_reasoning_structure_in_items(annotations)
    validate_against_schema(
        normalized_annotations,
        schema_path,
        "annotation.json",
    )

    turn_map = flatten_conversation_turns(conversation)
    seen_dia_ids: Set[str] = set()

    for ann in normalized_annotations:
        dia_id = ann["dia_id"]
        ensure(dia_id in turn_map, f"annotation dia_id not found in conversation: {dia_id}")
        ensure(dia_id not in seen_dia_ids, f"Duplicate annotation dia_id: {dia_id}")
        ensure(
            ann["speaker"] == turn_map[dia_id]["speaker"],
            f"annotation speaker mismatch for {dia_id}",
        )
        ensure(
            ann["reasoning_structure"] in VALID_REASONING_STRUCTURES,
            f"annotation {dia_id} has invalid reasoning_structure {ann['reasoning_structure']!r}",
        )
        ensure(
            ann["evidence_turn_ids"],
            f"annotation {dia_id} must include at least one evidence_turn_ids entry",
        )
        for evidence_dia_id in ann["evidence_turn_ids"]:
            ensure(
                evidence_dia_id in turn_map,
                f"annotation {dia_id} references missing evidence turn {evidence_dia_id}",
            )

        for quote in ann.get("evidence_turn_quotes", []):
            quote_dia_id = quote["dia_id"]
            ensure(
                quote_dia_id in ann["evidence_turn_ids"],
                f"annotation {dia_id} quote dia_id {quote_dia_id} is not present in evidence_turn_ids",
            )
            ensure(
                quote["speaker"] == turn_map[quote_dia_id]["speaker"],
                f"annotation {dia_id} quote speaker mismatch for {quote_dia_id}",
            )

        if ann["memory_dependency_level"] == "Level 0":
            ensure(
                not ann["historical_memory_required"],
                f"annotation {dia_id} is Level 0 but historical_memory_required is true",
            )

        seen_dia_ids.add(dia_id)


def validate_questions(
    questions: List[Dict[str, Any]],
    conversation: Dict[str, Any],
    schema_path: Path,
) -> None:
    """Validate questions and catch cross-turn issues that schema alone cannot express.

    Question text is allowed to be flexible, but evidence structure, adversarial flags,
    and memory-level claims still need to stay internally consistent.
    """
    normalized_questions = normalize_reasoning_structure_in_items(questions)
    validate_against_schema(
        normalized_questions,
        schema_path,
        "qa.json",
    )

    turn_map = flatten_conversation_turns(conversation)
    seen_question_ids: Set[str] = set()

    for question in normalized_questions:
        question_id = question["question_id"]
        anchor_dia_id = question["anchor_dia_id"]
        ensure(question_id not in seen_question_ids, f"Duplicate question_id: {question_id}")
        ensure(anchor_dia_id in turn_map, f"Question {question_id} anchor_dia_id not found: {anchor_dia_id}")
        ensure(
            question["reasoning_structure"] in VALID_REASONING_STRUCTURES,
            f"Question {question_id} has invalid reasoning_structure {question['reasoning_structure']!r}",
        )
        ensure(
            question["evidence_turn_ids"],
            f"Question {question_id} must include at least one evidence turn",
        )

        anchor_index = turn_map[anchor_dia_id]["turn_index_global"]
        has_prior_evidence = False
        for evidence_dia_id in question["evidence_turn_ids"]:
            ensure(
                evidence_dia_id in turn_map,
                f"Question {question_id} references missing evidence turn {evidence_dia_id}",
            )
            if turn_map[evidence_dia_id]["turn_index_global"] < anchor_index:
                has_prior_evidence = True

        if question["memory_level"] == "Level 3":
            ensure(
                has_prior_evidence,
                f"Question {question_id} is Level 3 but has no evidence turn before its anchor",
            )

        if question["adversarial_flag"]:
            ensure(
                bool(question.get("adversarial_type")),
                f"Question {question_id} is adversarial but adversarial_type is empty",
            )
        else:
            ensure(
                question.get("adversarial_type") in (None, ""),
                f"Question {question_id} is non-adversarial but adversarial_type is set",
            )

        if question["question_type"] == "judgment":
            ensure(
                question.get("options"),
                f"Question {question_id} is judgment-type and should provide options",
            )

        question_text = str(question.get("question_text", "")).lower()
        modality_condition = str(question.get("modality_condition") or "").lower()
        adversarial_type = question.get("adversarial_type")

        if question["question_type"] in {"modality_missing", "modality_ambiguous"} or modality_condition in {
            "voice_style_removed",
            "text_removed",
            "modality_ambiguous",
            "voice_style_ambiguous",
            "text_ambiguous",
        }:
            ensure(
                "instead of" not in question_text,
                f"Question {question_id} modality item should not replace the original cue with a new one",
            )
            ensure(
                not re.search(r"\bif\b.+\bwas described as\b", question_text),
                f"Question {question_id} modality item should not inject a hypothetical replacement description",
            )

        if adversarial_type == "pseudo_conflict":
            ensure(
                not re.search(r"\bif\b.+\bhad previously stated\b", question_text),
                f"Question {question_id} pseudo_conflict item should not inject fabricated prior dialogue",
            )

        ensure(
            not contains_forbidden_question_reference(str(question.get("question_text", ""))),
            f"Question {question_id} question_text should not mention session labels or turn IDs explicitly",
        )

        seen_question_ids.add(question_id)


def validate_units(
    units: List[Dict[str, Any]],
    conversation: Dict[str, Any],
    annotations: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
) -> None:
    """Validate that assembled benchmark units stay aligned with their source files."""
    turn_map = flatten_conversation_turns(conversation)
    annotation_map = build_annotation_map(annotations)
    question_map = {question["question_id"]: question for question in questions}
    seen_unit_ids: Set[str] = set()

    for unit in units:
        unit_id = unit["unit_id"]
        question_id = unit["question_id"]
        ensure(unit_id not in seen_unit_ids, f"Duplicate unit_id: {unit_id}")
        ensure(question_id in question_map, f"Unit {unit_id} references missing question_id {question_id}")
        ensure(
            unit["conversation_id"] == conversation.get("conversation_id"),
            f"Unit {unit_id} conversation_id mismatch",
        )
        ensure(unit["anchor"]["dia_id"] in turn_map, f"Unit {unit_id} anchor dia_id not found in conversation")

        expected_question = question_map[question_id]
        ensure(
            unit["anchor"]["dia_id"] == expected_question["anchor_dia_id"],
            f"Unit {unit_id} anchor dia_id does not match question anchor",
        )
        ensure(
            unit["gold"]["evidence_turn_ids"] == expected_question["evidence_turn_ids"],
            f"Unit {unit_id} gold evidence_turn_ids do not match source question",
        )
        ensure(
            len(unit["history_evidence"]) == len(expected_question["evidence_turn_ids"]),
            f"Unit {unit_id} history_evidence length does not match source question",
        )

        for evidence_turn in unit["history_evidence"]:
            dia_id = evidence_turn["dia_id"]
            ensure(dia_id in turn_map, f"Unit {unit_id} history evidence turn missing from conversation: {dia_id}")

        annotation = unit.get("annotation")
        if annotation is not None:
            anchor_dia_id = unit["anchor"]["dia_id"]
            ensure(
                anchor_dia_id in annotation_map,
                f"Unit {unit_id} anchor annotation missing for {anchor_dia_id}",
            )
            ensure(
                annotation["dia_id"] == anchor_dia_id,
                f"Unit {unit_id} annotation dia_id does not match anchor",
            )
