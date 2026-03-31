from copy import deepcopy
from typing import Any, Dict, List

import question_curation as qc


def _scaled_count(total: int, ratio: float, minimum: int = 0) -> int:
    return max(minimum, int(round(total * ratio)))


def _normalize_count_distribution(target_total: int, counts: Dict[str, int], priority_order: List[str]) -> Dict[str, int]:
    normalized = {key: max(0, int(value)) for key, value in counts.items()}
    current_total = sum(normalized.values())
    if current_total == target_total:
        return normalized
    if current_total < target_total:
        remaining = target_total - current_total
        order = priority_order or list(normalized.keys())
        index = 0
        while remaining > 0 and order:
            key = order[index % len(order)]
            normalized[key] = normalized.get(key, 0) + 1
            remaining -= 1
            index += 1
        return normalized

    overflow = current_total - target_total
    order = list(reversed(priority_order or list(normalized.keys())))
    index = 0
    while overflow > 0 and order:
        key = order[index % len(order)]
        if normalized.get(key, 0) > 0:
            normalized[key] -= 1
            overflow -= 1
        index += 1
    return normalized


def _build_hard_target_distributions(target_count: int) -> Dict[str, Dict[str, int]]:
    return {
        "content_distribution": _normalize_count_distribution(
            target_count,
            {
                "long_term_implicit_emotion": _scaled_count(target_count, 0.55, 4),
                "long_term_explicit_or_semi_explicit_emotion": _scaled_count(target_count, 0.25, 2),
                "long_term_fact": _scaled_count(target_count, 0.20, 1),
                "instant_emotion": 0,
                "near_term_fact": 0,
            },
            [
                "long_term_implicit_emotion",
                "long_term_explicit_or_semi_explicit_emotion",
                "long_term_fact",
            ],
        ),
        "memory_level_distribution": _normalize_count_distribution(
            target_count,
            {
                "Level_0": 0,
                "Level_1": 0,
                "Level_2": max(2, target_count // 3),
                "Level_3": max(5, target_count - max(2, target_count // 3)),
            },
            ["Level_3", "Level_2", "Level_1", "Level_0"],
        ),
        "format_distribution": _normalize_count_distribution(
            target_count,
            {
                "judgment": max(4, target_count // 2),
                "retrieval": 0,
                "explanation": max(3, target_count - max(4, target_count // 2)),
                "modality_missing_or_ambiguous": 0,
            },
            ["explanation", "judgment", "retrieval", "modality_missing_or_ambiguous"],
        ),
        "reasoning_distribution": _normalize_count_distribution(
            target_count,
            {
                "direct": 0,
                "single_hop": 1,
                "multi_hop": max(4, target_count // 2),
                "conflict_resolution": max(1, target_count // 5),
                "trajectory_based": max(2, target_count // 4),
            },
            ["multi_hop", "trajectory_based", "conflict_resolution", "single_hop", "direct"],
        ),
    }


def _subtract_distribution(original: Dict[str, Any], consumed: Dict[str, int], target_total: int) -> Dict[str, int]:
    return _normalize_count_distribution(
        target_total,
        {key: max(0, int(original.get(key, 0)) - int(consumed.get(key, 0))) for key in original},
        list(original.keys()),
    )


def build_question_phase_plan(question_plan: Dict[str, Any], phase_name: str) -> Dict[str, Any]:
    total = int(question_plan.get("mvp_question_count", 24))
    hard_count = max(8, min(total - 6, int(round(total * 0.5))))
    support_count = max(6, total - hard_count)
    hard_targets = _build_hard_target_distributions(hard_count)

    if phase_name == "hard_core":
        plan = deepcopy(question_plan)
        oversampled_count = qc.compute_oversampled_count(hard_count)
        oversampled_targets = _build_hard_target_distributions(int(oversampled_count))
        plan["generation_phase"] = "hard_core"
        plan["target_question_count"] = hard_count
        plan["mvp_question_count"] = oversampled_count
        plan.update(oversampled_targets)
        plan["adversarial_count"] = 0
        plan["adversarial_ratio"] = 0.0
        plan["adversarial_type_distribution"] = {
            "pseudo_relevant_history": 0,
            "insufficient_evidence": 0,
            "pseudo_conflict": 0,
        }
        return plan

    plan = deepcopy(question_plan)
    oversampled_count = qc.compute_oversampled_count(support_count)
    plan["generation_phase"] = "everything_else"
    plan["target_question_count"] = support_count
    plan["mvp_question_count"] = oversampled_count
    plan["content_distribution"] = _subtract_distribution(
        question_plan.get("content_distribution", {}),
        hard_targets["content_distribution"],
        int(oversampled_count),
    )
    plan["memory_level_distribution"] = _subtract_distribution(
        question_plan.get("memory_level_distribution", {}),
        hard_targets["memory_level_distribution"],
        int(oversampled_count),
    )
    plan["format_distribution"] = _subtract_distribution(
        question_plan.get("format_distribution", {}),
        hard_targets["format_distribution"],
        int(oversampled_count),
    )
    plan["reasoning_distribution"] = _subtract_distribution(
        question_plan.get("reasoning_distribution", {}),
        hard_targets["reasoning_distribution"],
        int(oversampled_count),
    )
    plan["adversarial_count"] = int(question_plan.get("adversarial_count", 0))
    plan["adversarial_ratio"] = question_plan.get("adversarial_ratio", 0.0)
    plan["adversarial_type_distribution"] = deepcopy(question_plan.get("adversarial_type_distribution", {}))
    return plan


def resize_question_phase_plan(plan: Dict[str, Any], new_target_count: int) -> Dict[str, Any]:
    resized = deepcopy(plan)
    oversampled_count = qc.compute_oversampled_count(new_target_count)
    resized["target_question_count"] = new_target_count
    resized["mvp_question_count"] = oversampled_count
    resized["content_distribution"] = _normalize_count_distribution(
        int(oversampled_count),
        resized.get("content_distribution", {}),
        list((resized.get("content_distribution") or {}).keys()),
    )
    resized["memory_level_distribution"] = _normalize_count_distribution(
        int(oversampled_count),
        resized.get("memory_level_distribution", {}),
        list((resized.get("memory_level_distribution") or {}).keys()),
    )
    resized["format_distribution"] = _normalize_count_distribution(
        int(oversampled_count),
        resized.get("format_distribution", {}),
        list((resized.get("format_distribution") or {}).keys()),
    )
    resized["reasoning_distribution"] = _normalize_count_distribution(
        int(oversampled_count),
        resized.get("reasoning_distribution", {}),
        list((resized.get("reasoning_distribution") or {}).keys()),
    )
    return resized
