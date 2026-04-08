import math
from typing import Any, Dict, List, Tuple

from conversation_utils import (
    extract_session_index_from_dia_id,
    flatten_conversation_turns,
    normalize_level_label,
)


def get_session_count(conversation: Dict[str, Any]) -> int:
    return len(
        [
            key
            for key, value in conversation.items()
            if key.startswith("session_") and not key.endswith("_date_time") and isinstance(value, list)
        ]
    )


def is_late_summary_like_turn_text(text: str) -> bool:
    lowered = text.lower()
    summary_markers = [
        "thank you",
        "thanks again",
        "appreciate",
        "grateful",
        "empowering",
        "better equipped",
        "clear sense of direction",
        "clearer framework",
        "much more manageable",
        "really helped",
        "more confident",
        "less anxious",
        "true partnership",
        "huge difference",
    ]
    closure_markers = [
        "see you",
        "goodbye",
        "next week",
        "talk next week",
    ]
    return any(marker in lowered for marker in summary_markers + closure_markers)


def has_outcome_summary_language(text: str) -> bool:
    lowered = str(text or "").lower()
    markers = [
        "less anxious",
        "more confident",
        "better equipped",
        "clear sense of direction",
        "clear path forward",
        "clearer framework",
        "huge difference",
        "made a real difference",
        "true partnership",
        "manage it better",
        "trying to manage it better",
        "evolved by this turn",
        "overall demeanor",
        "proactive request",
        "group charter",
        "less like... hoping for the best",
        "less like hoping for the best",
        "weight has been lifted",
        "feel lighter",
        "it's okay not to be fine",
        "okay not to be 'fine'",
        "cautiously trusting",
        "emotional growth",
        "profound positive shift",
        "newfound coping",
        "newfound self-efficacy",
        "grew",
        "growth at this point",
        "impact of her conversation",
        "glad we talked",
        "stable",
        "managing",
        "much better",
        "work in progress",
        "relationship with",
        "reveals about her emotional state and growth",
    ]
    return any(marker in lowered for marker in markers)


def is_result_over_path_question(
    question: Dict[str, Any],
    conversation: Dict[str, Any],
) -> bool:
    turn_map = flatten_conversation_turns(conversation)
    anchor_dia_id = str(question.get("anchor_dia_id") or "")
    anchor_text = str(turn_map.get(anchor_dia_id, {}).get("text") or "")
    question_text = str(question.get("question_text") or "")
    gold_answer = str(question.get("gold_answer") or "")
    combined = " || ".join([anchor_text, question_text, gold_answer])
    return has_outcome_summary_language(combined)


def score_late_result_question_candidate(
    question: Dict[str, Any],
    conversation: Dict[str, Any],
) -> int:
    score = score_final_question_candidate(question, conversation)
    question_text = str(question.get("question_text") or "").lower()
    gold_answer = str(question.get("gold_answer") or "").lower()

    # Prefer late-session items that still preserve residual tension or contrast,
    # not ones that simply restate improved outcomes or trust.
    if any(marker in question_text or marker in gold_answer for marker in [
        "lingering",
        "still worry",
        "not wanting to end up",
        "hoping for the best",
        "actual approach before",
        "before this conversation",
    ]):
        score += 3

    if any(marker in question_text or marker in gold_answer for marker in [
        "relationship with",
        "emotional growth",
        "growth at this point",
        "overall demeanor",
        "proactive request",
        "cautiously trusting",
        "manage it better",
    ]):
        score -= 2

    if any(marker in question_text for marker in [
        "what does this reveal about her relationship",
        "overall demeanor and proactive request",
        "how has her approach evolved",
    ]):
        score -= 1

    return score


def score_hard_core_annotation_candidate(
    annotation: Dict[str, Any],
    turn_map: Dict[str, Dict[str, Any]],
    closure_profile: str | None,
    total_sessions: int,
) -> float:
    dia_id = str(annotation.get("dia_id") or "")
    turn_text = str(turn_map.get(dia_id, {}).get("text") or "")
    memory_level = normalize_level_label(annotation.get("memory_dependency_level"), "Level 0")
    reasoning = str(annotation.get("reasoning_structure") or "")
    implicit_explicit = str(annotation.get("implicit_explicit") or "").lower()
    relation_state = str(annotation.get("relation_state") or "")
    session_index = extract_session_index_from_dia_id(dia_id)

    score = 0.0
    score += {"Level 3": 6.0, "Level 2": 4.0, "Level 1": 1.5, "Level 0": 0.0}.get(memory_level, 0.0)
    score += {
        "trajectory-based": 2.0,
        "conflict-resolution": 2.0,
        "multi-hop": 1.5,
        "single-hop": 0.5,
        "direct": 0.0,
    }.get(reasoning, 0.0)
    if implicit_explicit == "implicit":
        score += 1.0
    elif implicit_explicit == "explicit":
        score -= 0.5
    if relation_state in {"mild_strain", "strained", "fragile"}:
        score += 1.0

    if session_index > 0 and total_sessions > 0:
        if session_index == total_sessions:
            score -= 1.5
        elif session_index == total_sessions - 1:
            score -= 0.5
        elif session_index <= 2:
            score += 1.0

    if is_late_summary_like_turn_text(turn_text):
        score -= 5.0
        if session_index == total_sessions:
            score -= 2.0

    if closure_profile in {"surface_closure_hidden_residue", "practical_progress_relational_residue"}:
        if session_index == total_sessions and any(
            marker in turn_text.lower()
            for marker in ["manageable", "helped", "fine", "back to normal", "work out", "clearer framework"]
        ):
            score -= 3.0

    return score


def build_hard_core_annotation_subset(
    annotations: List[Dict[str, Any]],
    conversation: Dict[str, Any],
    question_plan: Dict[str, Any],
    closure_profile: str | None,
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    turn_map = flatten_conversation_turns(conversation)
    total_sessions = get_session_count(conversation)
    base_target = int(question_plan.get("target_question_count", question_plan.get("mvp_question_count", 10)))
    target_count = max(base_target + 4, 10)

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        scored.append(
            (
                score_hard_core_annotation_candidate(annotation, turn_map, closure_profile, total_sessions),
                annotation,
            )
        )
    scored.sort(
        key=lambda item: (
            item[0],
            normalize_level_label(item[1].get("memory_dependency_level"), "Level 0"),
            str(item[1].get("reasoning_structure") or ""),
        ),
        reverse=True,
    )

    preferred: List[Dict[str, Any]] = []
    discouraged: List[str] = []
    covered_sessions: set[int] = set()
    for score, annotation in scored:
        dia_id = str(annotation.get("dia_id") or "")
        session_index = extract_session_index_from_dia_id(dia_id)
        if score < 0:
            discouraged.append(dia_id)
            continue
        if session_index > 0 and session_index not in covered_sessions and len(preferred) < target_count:
            preferred.append(annotation)
            covered_sessions.add(session_index)
            continue
        if len(preferred) < target_count:
            preferred.append(annotation)
        else:
            discouraged.append(dia_id)

    if len(preferred) < min(target_count, len(scored)):
        for _, annotation in scored:
            if annotation in preferred:
                continue
            preferred.append(annotation)
            if len(preferred) >= target_count:
                break

    preferred_ids = [str(item.get("dia_id") or "") for item in preferred if str(item.get("dia_id") or "").strip()]
    discouraged_ids = [dia_id for dia_id in discouraged if dia_id and dia_id not in preferred_ids]
    return preferred, preferred_ids, discouraged_ids


def get_question_evidence_sessions(question: Dict[str, Any]) -> set[int]:
    sessions: set[int] = set()
    for dia_id in question.get("evidence_turn_ids", []):
        session_index = extract_session_index_from_dia_id(dia_id)
        if session_index > 0:
            sessions.add(session_index)
    return sessions


def get_anchor_session(question: Dict[str, Any]) -> int:
    return extract_session_index_from_dia_id(str(question.get("anchor_dia_id") or ""))


def has_early_session_evidence(question: Dict[str, Any], min_session_gap: int = 2) -> bool:
    """Return True if evidence_turn_ids contains at least one turn from a session
    that is min_session_gap or more sessions before the anchor session."""
    anchor_session = get_anchor_session(question)
    if anchor_session <= 0:
        return False
    for dia_id in question.get("evidence_turn_ids", []):
        ev_session = extract_session_index_from_dia_id(str(dia_id))
        if ev_session > 0 and (anchor_session - ev_session) >= min_session_gap:
            return True
    return False


def score_hard_core_question_candidate(
    question: Dict[str, Any],
    conversation: Dict[str, Any],
) -> Tuple[int, List[str]]:
    turn_map = flatten_conversation_turns(conversation)
    anchor_dia_id = str(question.get("anchor_dia_id") or "")
    anchor_text = str(turn_map.get(anchor_dia_id, {}).get("text") or "")
    evidence_turn_ids = [str(item) for item in question.get("evidence_turn_ids", []) if str(item).strip()]
    evidence_sessions = get_question_evidence_sessions(question)
    memory_level = str(question.get("memory_level") or "")
    reasoning_structure = str(question.get("reasoning_structure") or "")
    question_type = str(question.get("question_type") or "")

    score = 0
    reasons: List[str] = []

    if len(evidence_sessions) >= 2:
        score += 3
    else:
        score -= 3
        reasons.append("single_session_evidence")

    # Hard constraint: Level 3 must have evidence from ≥2 sessions before anchor
    if memory_level == "Level 3":
        if has_early_session_evidence(question, min_session_gap=2):
            score += 4
        else:
            score -= 6
            reasons.append("level3_no_early_session_evidence")
    elif memory_level == "Level 2":
        if len(evidence_sessions) >= 2:
            score += 1
        else:
            score -= 3
            reasons.append("level2_single_session_evidence")
    elif memory_level == "Level 1":
        score -= 2
        reasons.append("low_memory_level")

    if reasoning_structure in {"multi-hop", "trajectory-based", "conflict-resolution"}:
        score += 1

    if is_late_summary_like_turn_text(anchor_text):
        score -= 3
        reasons.append("late_summary_anchor")

    if question_type == "retrieval" and len(evidence_turn_ids) <= 1 and memory_level in {"Level 0", "Level 1"}:
        score -= 4
        reasons.append("shallow_retrieval")

    return score, reasons


def postprocess_hard_core_questions(
    questions: List[Dict[str, Any]],
    conversation: Dict[str, Any],
    minimum_keep: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    scored: List[Tuple[int, Dict[str, Any], List[str]]] = []
    for question in questions:
        score, reasons = score_hard_core_question_candidate(question, conversation)
        scored.append((score, question, reasons))

    kept: List[Dict[str, Any]] = []
    demoted: List[Dict[str, Any]] = []
    for score, question, reasons in sorted(
        scored,
        key=lambda item: (
            item[0],
            str(item[1].get("memory_level") or ""),
            str(item[1].get("reasoning_structure") or ""),
        ),
        reverse=True,
    ):
        should_demote = score < 0 and len(kept) >= minimum_keep
        if should_demote:
            demoted_question = dict(question)
            demoted_question["_demotion_reasons"] = reasons
            demoted.append(demoted_question)
        else:
            kept.append(question)

    return kept, demoted


def score_final_question_candidate(question: Dict[str, Any], conversation: Dict[str, Any]) -> int:
    turn_map = flatten_conversation_turns(conversation)
    anchor_dia_id = str(question.get("anchor_dia_id") or "")
    anchor_text = str(turn_map.get(anchor_dia_id, {}).get("text") or "")
    anchor_session = extract_session_index_from_dia_id(anchor_dia_id)
    total_sessions = get_session_count(conversation)
    evidence_sessions = get_question_evidence_sessions(question)
    memory_level = str(question.get("memory_level") or "")
    reasoning_structure = str(question.get("reasoning_structure") or "")
    question_type = str(question.get("question_type") or "")

    score = 0
    score += {"Level 3": 6, "Level 2": 4, "Level 1": 1, "Level 0": -1}.get(memory_level, 0)
    # Hard constraint: Level 3 must have early-session evidence (≥2 sessions before anchor)
    if memory_level == "Level 3" and not has_early_session_evidence(question, min_session_gap=2):
        score -= 8
    if len(evidence_sessions) >= 2:
        score += 3
    elif memory_level in {"Level 2", "Level 3"}:
        score -= 3
    if reasoning_structure in {"multi-hop", "trajectory-based", "conflict-resolution"}:
        score += 2
    if reasoning_structure == "direct":
        score -= 1
    if question_type == "retrieval" and len(question.get("evidence_turn_ids", [])) <= 1:
        score -= 4
    if is_late_summary_like_turn_text(anchor_text):
        score -= 4
    if is_result_over_path_question(question, conversation):
        score -= 4
    if anchor_session == total_sessions and memory_level in {"Level 0", "Level 1"}:
        score -= 2
    if anchor_session == total_sessions and is_result_over_path_question(question, conversation):
        score -= 3
    if 2 <= anchor_session <= max(2, total_sessions - 1):
        score += 1
    return score


def dedupe_same_anchor_questions(
    questions: List[Dict[str, Any]],
    conversation: Dict[str, Any],
) -> List[Dict[str, Any]]:
    best_by_anchor: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    unanchored: List[Dict[str, Any]] = []

    for question in questions:
        anchor = str(question.get("anchor_dia_id") or "").strip()
        if not anchor:
            unanchored.append(question)
            continue

        score = score_final_question_candidate(question, conversation)
        if str(question.get("question_type") or "") == "retrieval":
            score -= 2
        if str(question.get("content_type") or "") in {
            "long_term_implicit_emotion",
            "relation_change",
            "relation_state",
        }:
            score += 1

        current = best_by_anchor.get(anchor)
        if current is None or score > current[0]:
            best_by_anchor[anchor] = (score, question)

    deduped = list(unanchored)
    deduped.extend(item[1] for item in best_by_anchor.values())
    return deduped


def score_obviously_easy_question(
    question: Dict[str, Any],
    conversation: Dict[str, Any],
) -> Tuple[int, List[str]]:
    turn_map = flatten_conversation_turns(conversation)
    anchor_dia_id = str(question.get("anchor_dia_id") or "")
    anchor_text = str(turn_map.get(anchor_dia_id, {}).get("text") or "")
    question_text = str(question.get("question_text") or "")
    gold_answer = str(question.get("gold_answer") or "")
    question_type = str(question.get("question_type") or "")
    memory_level = str(question.get("memory_level") or "")
    anchor_session = extract_session_index_from_dia_id(anchor_dia_id)
    total_sessions = get_session_count(conversation)
    evidence_turn_ids = [str(item) for item in question.get("evidence_turn_ids", []) if str(item).strip()]

    score = 0
    reasons: List[str] = []
    combined = " || ".join([anchor_text, question_text, gold_answer]).lower()

    if (
        anchor_session == total_sessions
        and is_result_over_path_question(question, conversation)
        and has_outcome_summary_language(combined)
    ):
        score += 4
        reasons.append("late_result_summary")

    if is_late_summary_like_turn_text(anchor_text) and anchor_session == total_sessions:
        score += 2
        reasons.append("late_summary_anchor")

    if question_type == "retrieval" and len(evidence_turn_ids) <= 1 and memory_level in {"Level 0", "Level 1"}:
        score += 5
        reasons.append("shallow_single_hop_retrieval")

    if question_type == "judgment" and anchor_session == total_sessions and any(
        marker in combined
        for marker in [
            "what does this signify",
            "what does this indicate",
            "what does this reveal about",
            "emotional state and growth",
            "evolution of their relationship",
            "long-term growth",
        ]
    ):
        score += 2
        reasons.append("late_generic_growth_judgment")

    return score, reasons


def prune_obviously_easy_questions(
    questions: List[Dict[str, Any]],
    conversation: Dict[str, Any],
    minimum_keep_score: int = 4,
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for question in questions:
        easy_score, _ = score_obviously_easy_question(question, conversation)
        if easy_score >= minimum_keep_score:
            continue
        kept.append(question)
    return kept


def prune_excess_late_result_questions(
    questions: List[Dict[str, Any]],
    conversation: Dict[str, Any],
    target_count: int,
    max_late_result: int = 3,
) -> List[Dict[str, Any]]:
    total_sessions = get_session_count(conversation)
    if total_sessions <= 0:
        return list(questions)

    protected: List[Dict[str, Any]] = []
    late_result: List[Tuple[int, Dict[str, Any]]] = []
    minimum_keep_score = 4

    for question in questions:
        anchor_session = extract_session_index_from_dia_id(question.get("anchor_dia_id"))
        if anchor_session == total_sessions and is_result_over_path_question(question, conversation):
            late_result.append((score_late_result_question_candidate(question, conversation), question))
        else:
            protected.append(question)

    filtered_late_result = [item for item in late_result if item[0] >= minimum_keep_score]

    if len(filtered_late_result) <= max_late_result:
        return protected + [question for _, question in filtered_late_result]

    def late_result_sort_key(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, int]:
        score, question = item
        question_text = str(question.get("question_text") or "").lower()
        generic_summary_penalty = 0
        if "overall demeanor" in question_text:
            generic_summary_penalty += 2
        if "relationship with" in question_text:
            generic_summary_penalty += 1
        if "how has" in question_text and "evolved" in question_text:
            generic_summary_penalty += 1
        return (score, -generic_summary_penalty)

    filtered_late_result.sort(key=late_result_sort_key, reverse=True)
    kept_late = [question for _, question in filtered_late_result[:max_late_result]]
    return protected + kept_late


def prune_questions_to_target(
    questions: List[Dict[str, Any]],
    conversation: Dict[str, Any],
    target_count: int,
) -> List[Dict[str, Any]]:
    questions = dedupe_same_anchor_questions(questions, conversation)
    questions = prune_obviously_easy_questions(questions, conversation)
    questions = prune_excess_late_result_questions(questions, conversation, target_count)
    if len(questions) <= target_count:
        return list(questions)

    session_buckets: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
    for question in questions:
        anchor_session = extract_session_index_from_dia_id(question.get("anchor_dia_id"))
        session_buckets.setdefault(anchor_session, []).append(
            (score_final_question_candidate(question, conversation), question)
        )

    kept: List[Dict[str, Any]] = []
    for session_index, bucket in session_buckets.items():
        if session_index <= 0:
            continue
        best_score, best_question = sorted(bucket, key=lambda item: item[0], reverse=True)[0]
        if best_question not in kept:
            kept.append(best_question)

    remaining_slots = max(0, target_count - len(kept))
    remaining: List[Tuple[int, Dict[str, Any]]] = []
    for bucket in session_buckets.values():
        for scored_question in bucket:
            if scored_question[1] not in kept:
                remaining.append(scored_question)

    remaining.sort(key=lambda item: item[0], reverse=True)
    kept.extend(question for _, question in remaining[:remaining_slots])
    return kept[:target_count]


def compute_oversampled_count(target_count: int, ratio: float = 0.25, minimum_extra: int = 2) -> int:
    return target_count + max(minimum_extra, int(math.ceil(target_count * ratio)))
