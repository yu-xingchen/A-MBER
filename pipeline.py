import argparse
import asyncio
import json
import math
import os
import random
import re
import time
import unicodedata
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from conversation_utils import (
    extract_session_index_from_dia_id,
    flatten_conversation_turns as shared_flatten_conversation_turns,
    normalize_level_label,
)
from blueprint_pipeline import (
    generate_blueprint_bundle as generate_blueprint_bundle_stage,
    repair_blueprint_bundle as repair_blueprint_bundle_stage,
)
from interaction_pipeline import (
    generate_annotations_from_bundle as generate_annotations_from_bundle_stage,
    generate_conversation_from_bundle as generate_conversation_from_bundle_stage,
)
from question_generation_runtime import (
    build_question_curation_hints,
)
from question_pipeline import generate_questions_from_inputs

from validators import (
    DataValidationError,
    get_voice_style_leak_matches,
    load_json as validator_load_json,
    validate_annotations,
    validate_blueprint_bundle,
    validate_conversation,
    validate_conversation_against_blueprint,
    validate_generation_config,
    validate_questions,
    validate_units,
)

load_dotenv()


def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_prompt_language(config: Optional[Dict[str, Any]]) -> str:
    return str((config or {}).get("prompt_language", "en")).strip().lower() or "en"


def render_prompt_template(prompts_dir: Path, template_name: str, prompt_language: str) -> str:
    env = Environment(
        loader=FileSystemLoader(str(prompts_dir)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    normalized = "zh" if prompt_language.startswith("zh") else "en"
    return env.get_template(template_name).render(
        prompt_language=normalized,
        is_zh=normalized == "zh",
        is_en=normalized == "en",
    ).strip()


def load_json(path: Path) -> Any:
    return validator_load_json(path)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def compact_json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def save_bundle_json_files(bundle: Dict[str, Any], out_dir: Path) -> None:
    save_json(bundle["personas"], out_dir / "personas.json")
    save_json(bundle["global_outline"], out_dir / "global_outline.json")
    save_json(bundle["session_scripts"], out_dir / "session_scripts.json")
    save_json(bundle["event_plan"], out_dir / "event_plan.json")
    save_json(bundle["emotion_arc"], out_dir / "emotion_arc.json")
    save_json(bundle["question_plan"], out_dir / "question_plan.json")


def load_bundle_json_files(bundle_dir: Path) -> Dict[str, Any]:
    return {
        "personas": load_json(bundle_dir / "personas.json"),
        "global_outline": load_json(bundle_dir / "global_outline.json"),
        "session_scripts": load_json(bundle_dir / "session_scripts.json"),
        "event_plan": load_json(bundle_dir / "event_plan.json"),
        "emotion_arc": load_json(bundle_dir / "emotion_arc.json"),
        "question_plan": load_json(bundle_dir / "question_plan.json"),
    }


def distribute_turns_across_sessions(total_turns: int, session_count: int) -> List[int]:
    base_turns = total_turns // session_count
    remainder = total_turns % session_count
    return [base_turns + (1 if index < remainder else 0) for index in range(session_count)]


def derive_turn_variant_bundle(bundle: Dict[str, Any], total_turns: int) -> Dict[str, Any]:
    """Clone a blueprint bundle while changing only turn-budget metadata.

    This keeps personas, stages, themes, events, and emotion arcs unchanged so that
    length-based experiments compare realizations of the same scenario rather than
    separately generated blueprints.
    """
    variant = deepcopy(bundle)
    sessions = variant["session_scripts"]
    session_turns = distribute_turns_across_sessions(total_turns, len(sessions))
    for session, turn_count in zip(sessions, session_turns):
        session["turn_count"] = turn_count
    variant["global_outline"]["total_turns"] = total_turns
    return variant


def derive_turn_variant_scenarios(base_scenario_dir: Path, output_batch_dir: Path, turn_totals: List[int], schemas_dir: Path) -> List[Path]:
    base_bundle = load_bundle_json_files(base_scenario_dir)
    variant_dirs: List[Path] = []
    for total_turns in turn_totals:
        variant_dir = output_batch_dir / f"scenario_turns_{total_turns}"
        save_bundle_json_files(derive_turn_variant_bundle(base_bundle, total_turns), variant_dir)
        validate_blueprint_bundle(
            load_json(variant_dir / "personas.json"),
            load_json(variant_dir / "global_outline.json"),
            load_json(variant_dir / "session_scripts.json"),
            load_json(variant_dir / "event_plan.json"),
            load_json(variant_dir / "emotion_arc.json"),
            load_json(variant_dir / "question_plan.json"),
            schemas_dir,
        )
        variant_dirs.append(variant_dir)
    return variant_dirs


class ChatClient:
    def __init__(self, llm_config: Dict[str, Any]) -> None:
        self.provider = llm_config["provider"]
        self.model = llm_config["model"]
        self.api_key_env = llm_config["api_key_env"]
        self.api_key = os.getenv(self.api_key_env)
        self.base_url = llm_config.get("base_url", os.getenv("GATEWAY_BASE_URL", "https://api.portkey.ai/v1")).rstrip("/")
        if not self.api_key:
            raise ValueError(f"Missing API key in environment variable {self.api_key_env}.")

    def _post_with_retry(self, url: str, headers: dict, payload: dict, timeout: int = 600, max_retries: int = 3) -> dict:
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
                if attempt == max_retries - 1:
                    raise
                wait = 30 * (2 ** attempt)
                print(f"[Retry {attempt + 1}/{max_retries}] Timeout, retrying in {wait}s...")
                time.sleep(wait)
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code >= 500 and attempt < max_retries - 1:
                    wait = 30 * (2 ** attempt)
                    print(f"[Retry {attempt + 1}/{max_retries}] Server error {e.response.status_code}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, timeout: int = 600) -> Any:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = self._post_with_retry(url, headers, payload, timeout=timeout)
        return json.loads(data["choices"][0]["message"]["content"])

    def chat_json_array(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> Any:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = self._post_with_retry(url, headers, payload)
        content = data["choices"][0]["message"]["content"].strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            if "```" in content:
                content = content.replace("```json", "").replace("```", "").strip()
                return json.loads(content)
            raise ValueError("Model output is not valid JSON or JSON array.")


def load_generation_config(config_path: Path, schemas_dir: Path) -> Dict[str, Any]:
    config = load_json(config_path)
    validate_generation_config(config, schemas_dir / "generation_config.schema.json")
    return config


def get_conversation_generation_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    conversation_config = (config or {}).get("conversation_generation", {})
    return {
        "mode": conversation_config.get("mode", "single_agent"),
        "framework": conversation_config.get("framework", "autogen"),
        "temperature": conversation_config.get("temperature", 0.2),
        "max_turn_retries": conversation_config.get("max_turn_retries", 2),
        "request_timeout_seconds": conversation_config.get("request_timeout_seconds", 1200),
        "prompt_language": get_prompt_language(config),
    }


def select_persona_pair(persona_pool: Dict[str, Any], scenario_index: int, mode: str, seed: int) -> Dict[str, Any]:
    students = persona_pool["students"]
    teachers = persona_pool["teachers"]
    if mode == "round_robin":
        student = students[(scenario_index - 1) % len(students)]
        teacher = teachers[(scenario_index - 1) % len(teachers)]
    elif mode == "random_seeded":
        rng = random.Random(seed + scenario_index)
        student = students[rng.randrange(len(students))]
        teacher = teachers[rng.randrange(len(teachers))]
    elif mode == "random":
        # Keep "random" as a backward-compatible alias, but make it reproducible.
        rng = random.Random(seed + scenario_index)
        student = students[rng.randrange(len(students))]
        teacher = teachers[rng.randrange(len(teachers))]
    else:
        raise ValueError(f"Unsupported persona selection mode: {mode}")
    return {"student": student, "teacher": teacher}


CLOSURE_PROFILE_POOL = [
    "partial_repair_with_caution",
    "practical_progress_relational_residue",
    "clearer_relational_improvement_one_unresolved_thread",
    "mixed_or_stalled_progress",
    "surface_closure_hidden_residue",
    "recalibrated_working_alliance",
]


def _closure_profile_weights(selected_personas: Dict[str, Any]) -> Dict[str, float]:
    student = selected_personas.get("student", {})
    teacher = selected_personas.get("teacher", {})
    student_text = " ".join(
        [
            str(student.get("archetype", "")),
            str(student.get("expression_style", "")),
            str(student.get("interaction_risk", "")),
            " ".join(str(x) for x in student.get("coping_pattern", [])),
            " ".join(str(x) for x in student.get("misattunement_triggers", [])),
        ]
    ).lower()
    teacher_text = " ".join(
        [
            str(teacher.get("archetype", "")),
            str(teacher.get("support_style", "")),
            str(teacher.get("misattunement_pattern", "")),
            str(teacher.get("repair_style", "")),
        ]
    ).lower()

    weights = {profile: 1.0 for profile in CLOSURE_PROFILE_POOL}
    weights["practical_progress_relational_residue"] = 0.72
    weights["partial_repair_with_caution"] = 1.10

    if any(token in student_text for token in ("precision", "analyst", "logistics", "practical details", "intellectualize", "self-critical")):
        weights["recalibrated_working_alliance"] += 0.65
        weights["mixed_or_stalled_progress"] += 0.20

    if any(token in student_text for token in ("socially smooth", "approval-seeking", "cooperative while quietly pulling away", "tests whether support is genuine", "hides panic behind politeness", "uses humor to deflect")):
        weights["surface_closure_hidden_residue"] += 0.70
        weights["clearer_relational_improvement_one_unresolved_thread"] += 0.15

    if any(token in student_text for token in ("asks for help late", "health", "fatigue", "withdraws", "overwhelmed", "close to emotional overload")):
        weights["partial_repair_with_caution"] += 0.35
        weights["mixed_or_stalled_progress"] += 0.25

    if any(token in teacher_text for token in ("repair-minded", "warm reflector", "slows down", "returns to the student's original meaning", "names the rupture directly")):
        weights["clearer_relational_improvement_one_unresolved_thread"] += 0.45

    if any(token in teacher_text for token in ("open-ended", "ambiguity", "procedural", "plainspoken framing", "practical stabilizer")):
        weights["recalibrated_working_alliance"] += 0.25
        weights["practical_progress_relational_residue"] += 0.10

    return weights


def choose_closure_profile(selected_personas: Dict[str, Any], scenario_index: int, config: Dict[str, Any]) -> str:
    persona_selection = config.get("persona_selection", {})
    seed = int(persona_selection.get("seed", 0))
    rng = random.Random(seed * 9973 + scenario_index * 7919 + 17)
    weights = _closure_profile_weights(selected_personas)
    population = list(CLOSURE_PROFILE_POOL)
    weight_values = [max(weights.get(profile, 1.0), 0.01) for profile in population]
    return rng.choices(population, weights=weight_values, k=1)[0]

def generate_blueprint_bundle(
        client: ChatClient,
        prompts_dir: Path,
        templates_dir: Path,
        config: Dict[str, Any],
        selected_personas: Dict[str, Any],
        scenario_index: int,
        prior_batch_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return generate_blueprint_bundle_stage(
        client=client,
        prompts_dir=prompts_dir,
        templates_dir=templates_dir,
        config=config,
        selected_personas=selected_personas,
        scenario_index=scenario_index,
        prior_batch_summaries=prior_batch_summaries,
        load_json_fn=load_json,
        get_prompt_language_fn=get_prompt_language,
        render_prompt_template_fn=render_prompt_template,
        choose_closure_profile_fn=choose_closure_profile,
        build_question_plan_from_blueprint_fn=build_question_plan_from_blueprint,
    )


def repair_blueprint_bundle(
        client: ChatClient,
        prompts_dir: Path,
        config: Dict[str, Any],
        bundle: Dict[str, Any],
        validation_error: str,
) -> Dict[str, Any]:
    return repair_blueprint_bundle_stage(
        client=client,
        prompts_dir=prompts_dir,
        config=config,
        bundle=bundle,
        validation_error=validation_error,
        render_prompt_template_fn=render_prompt_template,
        get_prompt_language_fn=get_prompt_language,
    )


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def ensure_list_of_strings(value: Any, fallback: List[str]) -> List[str]:
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if cleaned:
            return cleaned
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return list(fallback)


MEASURABLE_POINT_ALIASES = {
    "conflict_resolution": "conflict resolution",
    "trajectory_based_relation_change": "trajectory-based relation change",
    "semi_explicit_emotion": "semi-explicit emotion",
    "long_term_implicit_emotion": "long-term implicit emotion",
    "long_term_fact": "long-term fact",
    "near_term_fact": "near-term fact",
}


def normalize_measurable_points(value: Any, fallback: List[str]) -> List[str]:
    normalized_points: List[str] = []
    for point in ensure_list_of_strings(value, fallback):
        key = point.strip().lower().replace("_", " ")
        canonical = MEASURABLE_POINT_ALIASES.get(point.strip().lower(), key)
        canonical = canonical.replace("  ", " ").strip()
        if canonical in {"modality missing", "modality ambiguous"}:
            continue
        if canonical and canonical not in normalized_points:
            normalized_points.append(canonical)
    return normalized_points or list(fallback)


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


def filter_emotion_terms(value: Any, fallback: List[str]) -> List[str]:
    candidates = ensure_list_of_strings(value, fallback)
    filtered = [
        term for term in candidates
        if term.strip().lower().replace(" ", "_") not in NON_EMOTION_TERMS
    ]
    return filtered or list(fallback)


def build_default_session_span(index: int, stage_count: int, sessions_per_conversation: int) -> List[int]:
    start = round(index * sessions_per_conversation / stage_count) + 1
    end = round((index + 1) * sessions_per_conversation / stage_count)
    if end < start:
        end = start
    end = min(end, sessions_per_conversation)
    return list(range(start, end + 1))


def infer_stage_name(stage: Dict[str, Any], index: int) -> str:
    for key in ("stage_name", "name", "theme", "label"):
        value = stage.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"stage_{index + 1}"


def infer_stage_goal(stage: Dict[str, Any], stage_name: str, fallback_stage: Dict[str, Any], index: int) -> str:
    for key in ("goal", "objective", "purpose"):
        value = stage.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    fallback = fallback_stage.get("goal")
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip()
    return f"Advance the emotional and memory arc during {stage_name or f'stage {index + 1}'}"


def infer_stage_relationship_state(stage: Dict[str, Any], fallback_stage: Dict[str, Any]) -> str:
    for key in ("relationship_state", "relation_state", "relationship_status"):
        value = stage.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    fallback = fallback_stage.get("relationship_state")
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip()
    return "support_building"


def infer_stage_emotional_background(stage: Dict[str, Any], fallback_stage: Dict[str, Any]) -> List[str]:
    for key in ("emotional_background", "student_dominant_emotions", "student_emotions"):
        if key in stage:
            return filter_emotion_terms(stage.get(key), fallback_stage.get("emotional_background", ["anxiety"]))
    return filter_emotion_terms(None, fallback_stage.get("emotional_background", ["anxiety"]))


def infer_stage_key_function(stage: Dict[str, Any], fallback_stage: Dict[str, Any], stage_name: str) -> str:
    for key in ("key_function", "memory_function", "benchmark_function"):
        value = stage.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    fallback = fallback_stage.get("key_function")
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip()
    return f"Plant evidence for later questions during {stage_name}"


def infer_session_measurable_points(session: Dict[str, Any], stage: Dict[str, Any], session_index: int) -> List[str]:
    existing = ensure_list_of_strings(session.get("measurable_points"), [])
    if existing:
        return normalize_measurable_points(existing, ["long-term implicit emotion"])

    stage_name = infer_stage_name(stage, session_index).lower()
    relation_state = str(stage.get("relationship_state", "")).lower()
    emotion_background = " ".join(ensure_list_of_strings(stage.get("emotional_background"), []))
    session_theme = str(session.get("session_theme") or session.get("theme") or "").lower()
    signal_text = " ".join([stage_name, relation_state, emotion_background.lower(), session_theme])

    if any(token in signal_text for token in ("repair", "re_engagement", "relief", "re-engagement")):
        return normalize_measurable_points(
            ["trajectory-based relation change", "long-term fact", "long-term implicit emotion"],
            ["long-term implicit emotion"],
        )
    if any(token in signal_text for token in ("misunderstanding", "tension", "strain", "withdraw", "grievance", "defensive")):
        return normalize_measurable_points(
            ["long-term implicit emotion", "conflict resolution", "long-term fact"],
            ["long-term implicit emotion"],
        )
    return normalize_measurable_points(
        ["near-term fact", "semi-explicit emotion", "long-term implicit emotion"],
        ["long-term implicit emotion"],
    )


def classify_scenario_skeleton(bundle: Dict[str, Any]) -> Dict[str, Any]:
    sessions = bundle.get("session_scripts", [])
    session_themes = [session.get("session_theme", "") for session in sessions]
    session_text = " ".join(session_themes).lower()
    event_text = " ".join(event.get("event_type", "") + " " + event.get("description", "") for event in bundle.get("event_plan", [])).lower()

    if any(token in event_text for token in ("health", "doctor", "fatigue", "unwell", "attendance")):
        stressor = "health_or_capacity"
    elif any(token in event_text for token in ("scholarship", "financial", "money")):
        stressor = "financial_or_scholarship"
    elif any(token in event_text for token in ("group project", "peer_conflict", "team", "teammate", "club")):
        stressor = "peer_or_group_conflict"
    elif any(token in event_text for token in ("family", "parent")):
        stressor = "family_expectation"
    else:
        stressor = "academic_pressure"

    if any(token in event_text for token in ("push through", "speak up", "take charge", "solution", "procedural")):
        misattunement = "advice_too_fast"
    elif any(token in event_text for token in ("feeling", "emotional state", "too focused on feelings")):
        misattunement = "emotion_focus_misses_practical_need"
    elif any(token in event_text for token in ("generic praise", "encouragement", "over-praise")):
        misattunement = "reassurance_misses_complexity"
    else:
        misattunement = "none_or_light"

    return {
        "student_role_id": bundle["personas"]["student"].get("role_id"),
        "teacher_role_id": bundle["personas"]["teacher"].get("role_id"),
        "closure_profile": bundle.get("global_outline", {}).get("closure_profile"),
        "stressor_family": stressor,
        "misattunement_family": misattunement,
        "session_themes": session_themes,
    }


def expected_event_count_for_session(turn_count: int) -> int:
    if turn_count >= 16:
        return 3
    if turn_count >= 10:
        return 2
    return 1


def get_target_event_count_for_session(session: Dict[str, Any], config: Dict[str, Any]) -> int:
    fixed_count = config.get("blueprint_generation", {}).get("fixed_events_per_session")
    if isinstance(fixed_count, int) and fixed_count >= 1:
        return fixed_count
    return expected_event_count_for_session(int(session["turn_count"]))


def slugify_event_type(value: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return compact or "memory_relevant_event"


def infer_supplemental_events_for_session(
        session: Dict[str, Any],
        existing_count: int,
        desired_count: int,
) -> List[Dict[str, Any]]:
    candidates: List[str] = []
    candidates.extend(ensure_list_of_strings(session.get("major_events"), []))
    candidates.extend(ensure_list_of_strings(session.get("future_memory_to_plant"), []))
    candidates.extend(ensure_list_of_strings(session.get("main_topics"), []))

    seen = set()
    deduped_candidates: List[str] = []
    for candidate in candidates:
        key = candidate.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped_candidates.append(candidate.strip())

    supplemental_events: List[Dict[str, Any]] = []
    for index in range(existing_count, desired_count):
        if index < len(deduped_candidates):
            seed_text = deduped_candidates[index]
        else:
            seed_text = f"{session['session_theme']} leaves a concrete trace for later memory questions"
        supplemental_events.append(
            {
                "session_id": session["session_id"],
                "event_type": slugify_event_type(seed_text.split(",")[0][:40]),
                "description": seed_text,
                "emotional_significance": "high" if index == 0 else "medium",
                "relation_impact": session["relationship_state"],
                "can_be_critical_memory": True,
                "can_be_distractor": index > 0,
            }
        )
    return supplemental_events


def stochastic_round_counts(raw_targets: Dict[str, float], total_count: int, rng: random.Random) -> Dict[str, int]:
    floors = {key: max(0, math.floor(value)) for key, value in raw_targets.items()}
    remainder = total_count - sum(floors.values())
    fractions = {key: max(0.0, raw_targets[key] - floors[key]) for key in raw_targets}

    if remainder > 0:
        available = {key: value for key, value in fractions.items() if value > 0}
        for _ in range(remainder):
            if not available:
                chosen_key = min(floors, key=floors.get)
                floors[chosen_key] += 1
                continue
            total_weight = sum(available.values())
            threshold = rng.random() * total_weight
            cumulative = 0.0
            chosen_key = next(iter(available))
            for key, weight in available.items():
                cumulative += weight
                if threshold <= cumulative:
                    chosen_key = key
                    break
            floors[chosen_key] += 1
            available.pop(chosen_key, None)
    elif remainder < 0:
        removable_keys = sorted(floors, key=floors.get, reverse=True)
        for _ in range(-remainder):
            for key in removable_keys:
                if floors[key] > 0:
                    floors[key] -= 1
                    break

    return floors


def realize_soft_distribution(
        ratio_distribution: Dict[str, Any],
        total_count: int,
        relative_flex: float,
        rng: random.Random,
) -> Dict[str, int]:
    if total_count <= 0:
        return {key: 0 for key in ratio_distribution}

    ratio_items = {
        key: float(value)
        for key, value in ratio_distribution.items()
        if isinstance(value, (int, float))
    }
    if not ratio_items:
        return {}

    raw_targets: Dict[str, float] = {}
    for key, ratio in ratio_items.items():
        base_target = max(0.0, ratio * total_count)
        jitter_scale = 1.0 + rng.uniform(-relative_flex, relative_flex)
        raw_targets[key] = max(0.0, base_target * jitter_scale)

    target_sum = sum(raw_targets.values())
    if target_sum <= 0:
        fallback_key = max(ratio_items, key=ratio_items.get)
        return {key: (total_count if key == fallback_key else 0) for key in ratio_items}

    normalized_targets = {
        key: value * total_count / target_sum
        for key, value in raw_targets.items()
    }
    return stochastic_round_counts(normalized_targets, total_count, rng)


def build_question_plan_from_blueprint(
        config: Dict[str, Any],
        bundle: Dict[str, Any],
        templates_dir: Path,
        scenario_index: int,
) -> Dict[str, Any]:
    reference_question_plan = load_json(templates_dir / "question_plan.template.json")
    tolerance = reference_question_plan["distribution_tolerance"]
    total_questions = config["question_count"]
    relative_flex = (
        tolerance["small_batch_relative_flex"]
        if total_questions < tolerance["large_batch_threshold"]
        else tolerance["large_batch_relative_flex"]
    )

    base_seed = config["persona_selection"].get("seed", 0)
    seed_material = "|".join(
        [
            str(base_seed),
            str(scenario_index),
            bundle["personas"]["student"].get("role_id", bundle["personas"]["student"].get("name", "student")),
            bundle["personas"]["teacher"].get("role_id", bundle["personas"]["teacher"].get("name", "teacher")),
            str(total_questions),
        ]
    )
    rng = random.Random(seed_material)

    question_plan = deepcopy(reference_question_plan)
    question_plan["distribution_mode"] = "soft_count_targets"
    question_plan["mvp_question_count"] = total_questions
    question_plan["source_note"] = "Counts computed from template ratios with controlled stochastic fluctuation."

    question_plan["content_distribution"] = realize_soft_distribution(
        reference_question_plan["content_distribution"],
        total_questions,
        relative_flex,
        rng,
    )
    question_plan["memory_level_distribution"] = realize_soft_distribution(
        reference_question_plan["memory_level_distribution"],
        total_questions,
        relative_flex,
        rng,
    )
    question_plan["format_distribution"] = realize_soft_distribution(
        reference_question_plan["format_distribution"],
        total_questions,
        relative_flex,
        rng,
    )
    question_plan["reasoning_distribution"] = realize_soft_distribution(
        reference_question_plan["reasoning_distribution"],
        total_questions,
        relative_flex,
        rng,
    )

    adversarial_ratio = float(reference_question_plan.get("adversarial_ratio", 0.0))
    expected_adversarial = adversarial_ratio * total_questions
    floor_count = math.floor(expected_adversarial)
    fractional_part = expected_adversarial - floor_count
    adversarial_count = floor_count + (1 if rng.random() < fractional_part else 0)
    question_plan["adversarial_count"] = adversarial_count
    question_plan["adversarial_type_distribution"] = realize_soft_distribution(
        reference_question_plan["adversarial_type_distribution"],
        adversarial_count,
        relative_flex,
        rng,
    )
    return question_plan


def normalize_question_plan(question_plan: Dict[str, Any], config: Dict[str, Any], templates_dir: Path) -> Dict[str, Any]:
    reference_question_plan = load_json(templates_dir / "question_plan.template.json")
    normalized = merge_dicts(reference_question_plan, question_plan)
    normalized["mvp_question_count"] = config["question_count"]

    if not isinstance(normalized.get("distribution_tolerance"), dict):
        normalized["distribution_tolerance"] = reference_question_plan["distribution_tolerance"]

    simplified_distribution = question_plan.get("distribution")
    if isinstance(simplified_distribution, list) and simplified_distribution:
        normalized["generated_distribution_notes"] = simplified_distribution

    return normalized


def normalize_blueprint_bundle(bundle: Dict[str, Any], config: Dict[str, Any], selected_personas: Dict[str, Any]) -> Dict[str, Any]:
    templates_dir = Path(__file__).resolve().parent / "templates"
    reference_global_outline = load_json(templates_dir / "global_outline.template.json")
    reference_session_scripts = load_json(templates_dir / "session_scripts.template.json")
    reference_event_plan = load_json(templates_dir / "event_plan.template.json")
    reference_emotion_arc = load_json(templates_dir / "emotion_arc.template.json")
    normalized = dict(bundle)

    personas = normalized.get("personas")
    if not isinstance(personas, dict):
        personas = {}
    personas.setdefault("student", selected_personas["student"])
    personas.setdefault("teacher", selected_personas["teacher"])
    normalized["personas"] = personas

    sessions_per_conversation = config["sessions_per_conversation"]
    stage_count = config["stage_count"]
    turn_total = config["turns_per_conversation"]

    global_outline = normalized.get("global_outline")
    if not isinstance(global_outline, dict):
        global_outline = {}
    global_outline.setdefault("scenario_type", config["scenario_type"])
    global_outline.setdefault("target_role", config["target_role"])
    global_outline.setdefault("total_sessions", sessions_per_conversation)
    global_outline.setdefault("total_turns", turn_total)
    closure_profile = str(global_outline.get("closure_profile") or "").strip()
    if closure_profile not in CLOSURE_PROFILE_POOL:
        closure_profile = choose_closure_profile(selected_personas, 0, config)
    global_outline["closure_profile"] = closure_profile
    if not isinstance(global_outline.get("overall_arc"), str) or not global_outline.get("overall_arc", "").strip():
        global_outline["overall_arc"] = reference_global_outline.get("overall_arc", "Trust building -> strain -> partial repair")
    raw_stages = global_outline.get("stages", [])
    if not isinstance(raw_stages, list):
        raw_stages = []
    normalized_stages: List[Dict[str, Any]] = []
    for index in range(stage_count):
        existing_stage = raw_stages[index] if index < len(raw_stages) and isinstance(raw_stages[index], dict) else {}
        fallback_stage = (
            reference_global_outline["stages"][index]
            if index < len(reference_global_outline.get("stages", []))
            else reference_global_outline["stages"][-1]
        )
        stage = dict(existing_stage)
        stage_name = infer_stage_name(stage, index)
        stage["stage_id"] = str(stage.get("stage_id") or fallback_stage.get("stage_id") or f"stage_{index + 1}")
        stage["stage_name"] = stage_name
        session_span = stage.get("session_span")
        if not isinstance(session_span, list) or not session_span:
            session_span = build_default_session_span(index, stage_count, sessions_per_conversation)
        stage["session_span"] = session_span
        stage["goal"] = infer_stage_goal(stage, stage_name, fallback_stage, index)
        stage["relationship_state"] = infer_stage_relationship_state(stage, fallback_stage)
        stage["emotional_background"] = infer_stage_emotional_background(stage, fallback_stage)
        stage["key_function"] = infer_stage_key_function(stage, fallback_stage, stage_name)
        normalized_stages.append(stage)
    global_outline["stages"] = normalized_stages
    normalized["global_outline"] = global_outline

    raw_sessions = normalized.get("session_scripts", [])
    if not isinstance(raw_sessions, list):
        raw_sessions = []
    base_turns = turn_total // sessions_per_conversation
    remainder = turn_total % sessions_per_conversation
    normalized_sessions: List[Dict[str, Any]] = []
    for index in range(sessions_per_conversation):
        existing_session = raw_sessions[index] if index < len(raw_sessions) and isinstance(raw_sessions[index], dict) else {}
        fallback_session = (
            reference_session_scripts[index]
            if index < len(reference_session_scripts)
            else reference_session_scripts[-1]
        )
        stage_id = next(
            (stage["stage_id"] for stage in normalized_stages if (index + 1) in stage["session_span"]),
            normalized_stages[min(index, len(normalized_stages) - 1)]["stage_id"],
        )
        stage = next(stage for stage in normalized_stages if stage["stage_id"] == stage_id)
        session = dict(existing_session)
        session["session_id"] = str(session.get("session_id") or f"S{index + 1}")
        session["stage_id"] = session.get("stage_id") or stage_id
        session["turn_count"] = base_turns + (1 if index < remainder else 0)
        session["session_theme"] = str(
            session.get("session_theme")
            or session.get("theme")
            or fallback_session.get("session_theme")
        )
        session["session_goal"] = str(
            session.get("session_goal")
            or session.get("goal")
            or fallback_session.get("session_goal")
            or f"Advance {stage['stage_name']}"
        )
        session["main_topics"] = ensure_list_of_strings(
            session.get("main_topics"),
            fallback_session.get("main_topics", stage["emotional_background"]),
        )
        session["major_events"] = ensure_list_of_strings(
            session.get("major_events"),
            fallback_session.get("major_events", [session["session_theme"]]),
        )
        session["dominant_student_emotions"] = ensure_list_of_strings(
            filter_emotion_terms(
                session.get("dominant_student_emotions") or session.get("student_emotions"),
                stage["emotional_background"],
            ),
            stage["emotional_background"],
        )
        session["relationship_state"] = str(
            session.get("relationship_state")
            or session.get("relation_state")
            or stage["relationship_state"]
        )
        session["measurable_points"] = infer_session_measurable_points(session, stage, index)
        session["future_memory_to_plant"] = ensure_list_of_strings(
            session.get("future_memory_to_plant"),
            fallback_session.get("future_memory_to_plant", [f"Later questions should reference {session['session_theme']}"]),
        )
        session["constraints"] = ensure_list_of_strings(
            session.get("constraints"),
            fallback_session.get("constraints", ["stay realistic and gradual"]),
        )
        normalized_sessions.append(session)
    normalized["session_scripts"] = normalized_sessions

    raw_event_plan = normalized.get("event_plan", [])
    if not isinstance(raw_event_plan, list):
        raw_event_plan = []
    events_by_session: Dict[str, List[Dict[str, Any]]] = {session["session_id"]: [] for session in normalized_sessions}
    for index, raw_event in enumerate(raw_event_plan):
        if not isinstance(raw_event, dict):
            continue
        fallback_event = reference_event_plan[index] if index < len(reference_event_plan) else reference_event_plan[-1]
        session_id = str(raw_event.get("session_id") or "")
        if session_id not in events_by_session:
            continue
        event = dict(raw_event)
        event["session_id"] = session_id
        event["event_type"] = str(event.get("event_type") or event.get("type") or fallback_event.get("event_type"))
        event["description"] = str(
            event.get("description")
            or fallback_event.get("description")
            or f"Student behavior in {session_id} reveals a memory-relevant emotional cue"
        )
        event["emotional_significance"] = str(event.get("emotional_significance") or fallback_event.get("emotional_significance") or "medium")
        session_state = next(session["relationship_state"] for session in normalized_sessions if session["session_id"] == session_id)
        event["relation_impact"] = str(event.get("relation_impact") or fallback_event.get("relation_impact") or session_state)
        event["can_be_critical_memory"] = bool(event.get("can_be_critical_memory", True))
        event["can_be_distractor"] = bool(event.get("can_be_distractor", True))
        session_script = next(session for session in normalized_sessions if session["session_id"] == session_id)
        event["interaction_targets"] = normalize_interaction_targets(event, session_script)
        events_by_session[session_id].append(event)

    normalized_events: List[Dict[str, Any]] = []
    for session in normalized_sessions:
        session_events = events_by_session[session["session_id"]]
        desired_count = get_target_event_count_for_session(session, config)
        if len(session_events) < desired_count:
            session_events.extend(
                infer_supplemental_events_for_session(
                    session=session,
                    existing_count=len(session_events),
                    desired_count=desired_count,
                )
            )
        normalized_events.extend(session_events[:desired_count] if len(session_events) > desired_count else session_events)

    for index, event in enumerate(normalized_events):
        event["event_id"] = f"E{index + 1}"
    normalized["event_plan"] = normalized_events

    raw_emotion_arc = normalized.get("emotion_arc", [])
    if not isinstance(raw_emotion_arc, list):
        raw_emotion_arc = []
    normalized_arc: List[Dict[str, Any]] = []
    for index, stage in enumerate(normalized_stages):
        existing_arc = raw_emotion_arc[index] if index < len(raw_emotion_arc) and isinstance(raw_emotion_arc[index], dict) else {}
        fallback_arc = reference_emotion_arc[index] if index < len(reference_emotion_arc) else reference_emotion_arc[-1]
        arc_item = dict(existing_arc)
        arc_item["stage_id"] = arc_item.get("stage_id") or stage["stage_id"]
        arc_item["student_dominant_emotions"] = ensure_list_of_strings(
            filter_emotion_terms(
                arc_item.get("student_dominant_emotions") or arc_item.get("student_emotions"),
                stage["emotional_background"],
            ),
            stage["emotional_background"],
        )
        arc_item["implicit_emotions_to_seed"] = filter_emotion_terms(
            arc_item.get("implicit_emotions_to_seed"),
            fallback_arc.get("implicit_emotions_to_seed", stage["emotional_background"][:1]),
        )
        arc_item["relation_state"] = str(arc_item.get("relation_state") or stage["relationship_state"])
        if index == 0:
            arc_item["change_from_previous"] = arc_item.get("change_from_previous")
        else:
            arc_item["change_from_previous"] = (
                arc_item.get("change_from_previous")
                or f"Shift toward {stage['relationship_state']} through {stage['stage_name']}"
            )
        normalized_arc.append(arc_item)
    normalized["emotion_arc"] = normalized_arc

    question_plan = normalized.get("question_plan")
    if not isinstance(question_plan, dict):
        question_plan = {}
    normalized["question_plan"] = normalize_question_plan(question_plan, config, templates_dir)
    return normalized


def build_batch_run_name(config: Dict[str, Any]) -> str:
    output_naming = config["output_naming"]
    if not output_naming["append_timestamp"]:
        return config["batch_name"]
    timestamp = datetime.now().strftime(output_naming["timestamp_format"])
    return f"{config['batch_name']}_{timestamp}"


def build_scenario_dir(output_root: Path, batch_run_name: str, scenario_index: int) -> Path:
    return output_root / batch_run_name / f"scenario_{scenario_index:03d}"


def summarize_scenario_bundle(scenario_index: int, scenario_dir: Path, config: Dict[str, Any], bundle: Dict[str, Any]) -> str:
    student = bundle["personas"]["student"].get("name", bundle["personas"]["student"].get("role_id", "student"))
    teacher = bundle["personas"]["teacher"].get("name", bundle["personas"]["teacher"].get("role_id", "teacher"))
    session_count = len(bundle["session_scripts"])
    total_turns = sum(session["turn_count"] for session in bundle["session_scripts"])
    return (
        f"Scenario {scenario_index:03d}: {student} x {teacher} | "
        f"{session_count} sessions | {total_turns} turns | "
        f"{bundle['question_plan'].get('mvp_question_count', config['question_count'])} questions | "
        f"{config['llm']['provider']}/{config['llm']['model']} | "
        f"{scenario_dir}"
    )


def build_session_datetime(session_index: int) -> str:
    base_dt = datetime(2024, 3, 10, 10, 0, 0, tzinfo=timezone(timedelta(hours=8)))
    session_dt = base_dt + timedelta(days=(session_index - 1) * 7)
    return session_dt.isoformat(timespec="seconds")


def parse_iso_datetime(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    return datetime.fromisoformat(normalized)


def format_iso_datetime(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def ensure_turn_timestamps(conversation: Dict[str, Any]) -> Dict[str, Any]:
    """Backfill deterministic per-turn timestamps from session start times."""
    normalized = dict(conversation)
    turn_spacing_seconds = 75

    for session_key, turns in get_session_turns(normalized):
        session_dt_key = f"{session_key}_date_time"
        session_dt_raw = normalized.get(session_dt_key)
        if not isinstance(session_dt_raw, str) or not session_dt_raw.strip():
            continue
        session_start = parse_iso_datetime(session_dt_raw)
        rebuilt_turns: List[Dict[str, Any]] = []
        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                rebuilt_turns.append(turn)
                continue
            normalized_turn = dict(turn)
            if not isinstance(normalized_turn.get("timestamp"), str) or not normalized_turn.get("timestamp", "").strip():
                normalized_turn["timestamp"] = format_iso_datetime(
                    session_start + timedelta(seconds=turn_index * turn_spacing_seconds)
                )
            rebuilt_turns.append(normalized_turn)
        normalized[session_key] = rebuilt_turns

    return normalized


def build_turn_generation_user_payload(
    bundle: Dict[str, Any],
    session_script: Dict[str, Any],
    session_events: List[Dict[str, Any]],
    stage_arc: Dict[str, Any],
    session_index: int,
    turn_number: int,
    global_turn_index: int,
    speaker_name: str,
    counterpart_name: str,
    history: List[Dict[str, Any]],
    retry_feedback: Optional[str] = None,
) -> str:
    recent_history = history[-8:]
    event_count = max(len(session_events), 1)
    event_slot_size = max(session_script["turn_count"] // event_count, 1)
    active_event_index = min((turn_number - 1) // event_slot_size, event_count - 1)
    active_event = session_events[active_event_index] if session_events else {}
    established_events = session_events[: active_event_index + 1]
    future_events = session_events[active_event_index + 1:]
    feedback_block = ""
    if retry_feedback:
        feedback_block = f"\n\n[retry_feedback]\n{retry_feedback}"
    return f"""
Generate exactly one dialogue turn as JSON only.

[personas.json]
{json.dumps(bundle["personas"], ensure_ascii=False, indent=2)}

[global_outline.json]
{json.dumps(bundle["global_outline"], ensure_ascii=False, indent=2)}

[session_script.json]
{json.dumps(session_script, ensure_ascii=False, indent=2)}

[session_events.json]
{json.dumps(session_events, ensure_ascii=False, indent=2)}

[stage_arc.json]
{json.dumps(stage_arc, ensure_ascii=False, indent=2)}

[session_progress]
{json.dumps({
    "session_theme": session_script.get("session_theme"),
    "session_goal": session_script.get("session_goal"),
    "main_topics": session_script.get("main_topics", []),
    "major_events": session_script.get("major_events", []),
    "future_memory_to_plant": session_script.get("future_memory_to_plant", []),
    "constraints": session_script.get("constraints", []),
    "active_event": active_event,
    "events_that_should_already_be_established": established_events,
    "events_reserved_for_later_in_this_session": future_events,
}, ensure_ascii=False, indent=2)}

[recent_history.json]
{json.dumps(recent_history, ensure_ascii=False, indent=2)}

[turn_target]
{json.dumps({
    "session_index": session_index,
    "turn_number_within_session": turn_number,
    "global_turn_index": global_turn_index,
    "speaker": speaker_name,
    "counterpart": counterpart_name,
}, ensure_ascii=False, indent=2)}
{feedback_block}
""".strip()


def ensure_turn_shape(
    generated_turn: Dict[str, Any],
    speaker_name: str,
    dia_id: str,
    global_turn_index: int,
) -> Dict[str, Any]:
    normalized_turn = dict(generated_turn)
    normalized_turn["speaker"] = speaker_name
    normalized_turn["dia_id"] = dia_id
    normalized_turn["turn_index_global"] = global_turn_index
    normalized_turn["audio_id"] = normalized_turn.get("audio_id")
    if not isinstance(normalized_turn.get("text"), str) or not normalized_turn["text"].strip():
        raise DataValidationError(f"Generated turn {dia_id} is missing text")
    if not isinstance(normalized_turn.get("voice_style"), str) or not normalized_turn["voice_style"].strip():
        raise DataValidationError(f"Generated turn {dia_id} is missing voice_style")
    notes = normalized_turn.get("notes_for_dataset_builder")
    if not isinstance(notes, dict):
        notes = {}
    linked_event_ids = notes.get("linked_event_ids")
    if not isinstance(linked_event_ids, list):
        linked_event_ids = []
    normalized_turn["notes_for_dataset_builder"] = {
        "linked_event_ids": linked_event_ids,
        "is_candidate_emotion_anchor": bool(notes.get("is_candidate_emotion_anchor", False)),
        "is_candidate_fact_anchor": bool(notes.get("is_candidate_fact_anchor", False)),
    }
    audio_id = normalized_turn.get("audio_id")
    if audio_id in ("", None):
        normalized_turn["audio_id"] = None if audio_id is None else ""
        normalized_turn["modality_available"] = ["text", "voice_style"]
    else:
        modality_available = normalized_turn.get("modality_available")
        if not isinstance(modality_available, list):
            modality_available = ["text", "voice_style", "audio"]
        if "text" not in modality_available:
            modality_available.append("text")
        if "voice_style" not in modality_available:
            modality_available.append("voice_style")
        if "audio" not in modality_available:
            modality_available.append("audio")
        normalized_turn["modality_available"] = modality_available
    return normalized_turn


def flatten_generated_history(conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    for session_key, turns in conversation.items():
        if session_key.startswith("session_") and isinstance(turns, list):
            history.extend(turns)
    history.sort(key=lambda turn: int(turn.get("turn_index_global", 0)))
    return history


def contains_any(text: str, keywords: List[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def derive_interaction_targets(active_event: Dict[str, Any], session_script: Dict[str, Any]) -> Dict[str, Any]:
    event_type = str(active_event.get("event_type") or "")
    topic_keywords: List[str] = []
    for topic in session_script.get("main_topics", []):
        topic_keywords.extend(str(topic).lower().split())

    blueprint_like_targets: Dict[str, Any] = {
        "teacher_move_target": None,
        "student_response_target": None,
        "must_include_signals": [],
        "avoid_signals": [],
        "anchoring_topics": [kw for kw in topic_keywords if len(kw) > 3],
    }
    if event_type == "health_disclosure":
        blueprint_like_targets["student_response_target"] = "mention ongoing fatigue or health strain without fully opening up"
        blueprint_like_targets["must_include_signals"] = ["tired", "fatigue", "drained", "exhaust", "run down", "sleep"]
    elif event_type == "support_offer":
        blueprint_like_targets["teacher_move_target"] = "keep the response practical and focused on immediate academic or logistical support"
        blueprint_like_targets["must_include_signals"] = ["ta", "professor", "email", "office hour", "extension", "lab", "option"]
    elif event_type == "self_minimization":
        blueprint_like_targets["student_response_target"] = "downplay the seriousness and keep the tone self-minimizing"
        blueprint_like_targets["must_include_signals"] = ["rough patch", "probably", "fine", "normal", "it'll pass", "back to normal", "push through", "i'm sure"]
    elif event_type == "struggle_to_implement":
        blueprint_like_targets["student_response_target"] = "show difficulty carrying out the practical steps because of exhaustion"
        blueprint_like_targets["must_include_signals"] = ["haven't", "too tired", "drained", "overwhelming", "hard to", "huge effort", "run down"]
    elif event_type == "relational_misattunement":
        blueprint_like_targets["teacher_move_target"] = "stay practical enough that the emotional mismatch remains visible"
        blueprint_like_targets["student_response_target"] = "sound polite but not genuinely relieved or fully understood"
        blueprint_like_targets["must_include_signals"] = ["email", "call", "time", "schedule", "plan", "ta", "professor", "office hour", "step"]
        blueprint_like_targets["avoid_signals"] = ["relief", "helpful", "that's better", "feel seen", "understand me"]
    elif event_type == "student_withdrawal":
        blueprint_like_targets["student_response_target"] = "become briefer, flatter, or more distant without escalating into open rupture"
        blueprint_like_targets["must_include_signals"] = ["quiet", "brief", "short", "whisper", "flat", "distant", "subdued", "pause"]
    elif event_type == "teacher_emotional_acknowledgement":
        blueprint_like_targets["teacher_move_target"] = "explicitly name the emotional burden before moving to solutions"
        blueprint_like_targets["must_include_signals"] = ["draining", "hard", "exhaust", "burden", "struggle", "unwell", "heavy", "tough"]
    elif event_type == "student_cautious_sharing":
        blueprint_like_targets["student_response_target"] = "open up more about the emotional weight, but still cautiously"
        blueprint_like_targets["must_include_signals"] = ["hard", "tired", "exhaust", "problem", "behind", "don't know", "drowning", "pretend", "carry", "overwhelmed"]
    elif event_type == "revised_support_plan":
        blueprint_like_targets["teacher_move_target"] = "offer a revised support plan that stays anchored to the existing scenario topics"
        blueprint_like_targets["must_include_signals"] = ["plan", "step", "contact", "email", "support", "service", "next", "check-in", "option", "help"]
    return blueprint_like_targets


def normalize_interaction_targets(active_event: Dict[str, Any], session_script: Dict[str, Any]) -> Dict[str, Any]:
    derived = derive_interaction_targets(active_event, session_script)
    existing = active_event.get("interaction_targets")
    if not isinstance(existing, dict):
        return derived
    normalized = dict(existing)
    normalized["teacher_move_target"] = str(
        normalized.get("teacher_move_target") or derived.get("teacher_move_target") or ""
    ).strip()
    normalized["student_response_target"] = str(
        normalized.get("student_response_target") or derived.get("student_response_target") or ""
    ).strip()
    normalized["must_include_signals"] = ensure_list_of_strings(
        normalized.get("must_include_signals"),
        derived.get("must_include_signals", []),
    )
    normalized["avoid_signals"] = ensure_list_of_strings(
        normalized.get("avoid_signals"),
        derived.get("avoid_signals", []),
    )
    normalized["anchoring_topics"] = ensure_list_of_strings(
        normalized.get("anchoring_topics"),
        derived.get("anchoring_topics", []),
    )
    return normalized


def collect_soft_event_feedback(
    recent_turns: List[Dict[str, Any]],
    session_script: Dict[str, Any],
    active_event: Dict[str, Any],
) -> Optional[str]:
    if not active_event:
        return None
    if not recent_turns:
        return None
    combined = " || ".join(
        f"{str(turn.get('text') or '').lower()} || {str(turn.get('voice_style') or '').lower()}"
        for turn in recent_turns
    )
    # New blueprints should already carry interaction_targets in event_plan.json.
    # Keep this fallback so older scenario directories can still run without being regenerated.
    targets = dict(active_event.get("interaction_targets") or {})
    if not targets:
        targets = derive_interaction_targets(active_event, session_script)
    must_include = [str(item).lower() for item in targets.get("must_include_signals", []) if str(item).strip()]
    avoid_signals = [str(item).lower() for item in targets.get("avoid_signals", []) if str(item).strip()]
    anchoring_topics = [str(item).lower() for item in targets.get("anchoring_topics", []) if str(item).strip()]
    teacher_move_target = str(targets.get("teacher_move_target") or "").strip()
    student_response_target = str(targets.get("student_response_target") or "").strip()

    if must_include and not contains_any(combined, must_include):
        if teacher_move_target and student_response_target:
            return f"Keep the beat closer to its intended shape: teacher should {teacher_move_target}, while the student should {student_response_target}."
        if teacher_move_target:
            return f"Keep the beat closer to its intended shape: teacher should {teacher_move_target}."
        if student_response_target:
            return f"Keep the beat closer to its intended shape: student should {student_response_target}."
    if avoid_signals and contains_any(combined, avoid_signals):
        return "Do not move this interaction beat too far forward yet. Keep the current tension or partial mismatch in place a little longer."
    if teacher_move_target and "anchored to the existing scenario topics" in teacher_move_target.lower():
        if anchoring_topics and not contains_any(combined, anchoring_topics):
            return "Keep the revised support move anchored to the blueprint's existing session topics rather than inventing a new support thread."
    return None


async def generate_conversation_with_autogen(
    client: ChatClient,
    prompts_dir: Path,
    bundle: Dict[str, Any],
    conversation_config: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_ext.models.openai import OpenAIChatCompletionClient
    except ImportError as exc:
        raise RuntimeError(
            "AutoGen conversation mode requires `autogen-agentchat` and `autogen-ext` to be installed. "
            "Switch `conversation_generation.mode` back to `single_agent`, or install the AutoGen packages first."
        ) from exc

    teacher_name = bundle["personas"]["teacher"]["name"]
    student_name = bundle["personas"]["student"]["name"]
    session_scripts = bundle["session_scripts"]
    event_plan = bundle["event_plan"]
    emotion_arc = {entry["stage_id"]: entry for entry in bundle["emotion_arc"]}

    model_client = OpenAIChatCompletionClient(
        model=client.model,
        api_key=client.api_key,
        base_url=client.base_url,
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": True,
            "family": "openai-compatible",
            "structured_output": True,
            "multiple_system_messages": True,
        },
    )
    prompt_language = conversation_config.get("prompt_language", "en")
    teacher_prompt = render_prompt_template(prompts_dir, "teacher_turn_generation_prompt_v1.j2", prompt_language)
    student_prompt = render_prompt_template(prompts_dir, "student_turn_generation_prompt_v1.j2", prompt_language)

    conversation: Dict[str, Any] = {
        "conversation_id": bundle["global_outline"].get("conversation_id", "multi_agent_conversation"),
        "scenario_type": bundle["global_outline"].get("scenario_type", "psychological_teacher_student"),
        "speaker_a": teacher_name,
        "speaker_b": student_name,
        "target_speaker": student_name,
    }
    global_turn_index = 1

    for session_script in session_scripts:
        session_id = str(session_script["session_id"])
        session_index = int(session_id.replace("S", ""))
        session_key = f"session_{session_index}"
        conversation[f"{session_key}_date_time"] = build_session_datetime(session_index)
        conversation[session_key] = []
        session_events = [dict(event) for event in event_plan if event.get("session_id") == session_id]
        stage_arc = emotion_arc.get(session_script["stage_id"], {})
        turn_count = int(session_script["turn_count"])
        enriched_session_script = dict(session_script)
        enriched_session_script["_teacher_name"] = teacher_name
        enriched_session_script["_student_name"] = student_name
        for event in session_events:
            # New blueprints persist interaction_targets during normalization. Recompute here only
            # as a backward-compatibility fallback for older scenario directories.
            event["interaction_targets"] = normalize_interaction_targets(event, enriched_session_script)
        event_count = max(len(session_events), 1)
        event_slot_size = max(turn_count // event_count, 1)
        session_feedback: Optional[str] = None

        for turn_number in range(1, turn_count + 1):
            is_teacher_turn = turn_number % 2 == 1
            speaker_name = teacher_name if is_teacher_turn else student_name
            counterpart_name = student_name if is_teacher_turn else teacher_name
            dia_id = f"D{session_index}:{turn_number}"
            active_event_index = min((turn_number - 1) // event_slot_size, event_count - 1)
            active_event = session_events[active_event_index] if session_events else {}
            last_error: Optional[Exception] = None
            generated_turn: Optional[Dict[str, Any]] = None
            retry_feedback: Optional[str] = session_feedback
            for _ in range(conversation_config["max_turn_retries"] + 1):
                try:
                    payload = build_turn_generation_user_payload(
                        bundle=bundle,
                        session_script=enriched_session_script,
                        session_events=session_events,
                        stage_arc=stage_arc,
                        session_index=session_index,
                        turn_number=turn_number,
                        global_turn_index=global_turn_index,
                        speaker_name=speaker_name,
                        counterpart_name=counterpart_name,
                        history=flatten_generated_history(conversation),
                        retry_feedback=retry_feedback,
                    )
                    agent = AssistantAgent(
                        name="teacher_agent" if is_teacher_turn else "student_agent",
                        model_client=model_client,
                        system_message=teacher_prompt if is_teacher_turn else student_prompt,
                    )
                    result = await agent.run(task=payload)
                    response_text = str(result.messages[-1].content).strip()
                    if "```" in response_text:
                        response_text = response_text.replace("```json", "").replace("```", "").strip()
                    generated_turn = json.loads(response_text)
                    generated_turn = ensure_turn_shape(generated_turn, speaker_name, dia_id, global_turn_index)
                    break
                except Exception as exc:  # pragma: no cover - depends on optional runtime
                    last_error = exc
            if generated_turn is None:
                raise RuntimeError(f"AutoGen failed to produce valid turn {dia_id}: {last_error}") from last_error
            conversation[session_key].append(generated_turn)

            window_turns = conversation[session_key][-4:]
            window_feedback = collect_soft_event_feedback(
                recent_turns=window_turns,
                session_script=enriched_session_script,
                active_event=active_event,
            )
            if window_feedback and turn_number < turn_count:
                session_feedback = (
                    f"Stay closer to the current interaction beat `{active_event.get('event_id')}` / "
                    f"`{active_event.get('event_type')}`. {window_feedback}"
                )
            else:
                session_feedback = None
            global_turn_index += 1

    return conversation


def normalize_conversation_metadata(conversation: Dict[str, Any], personas: Dict[str, Any]) -> Dict[str, Any]:
    # The dialogue model still occasionally emits legacy Teacher/Student labels.
    # Normalize them here so downstream validation always sees the resolved persona names.
    normalized = dict(conversation)
    teacher_name = personas["teacher"].get("name", "Teacher")
    student_name = personas["student"].get("name", "Student")
    known_teacher_tokens = {
        "",
        "...",
        "teacher",
        "psychological teacher",
        "teacher persona",
        teacher_name.lower(),
    }
    known_student_tokens = {
        "",
        "...",
        "student",
        "student persona",
        student_name.lower(),
    }

    normalized["speaker_a"] = teacher_name
    normalized["speaker_b"] = student_name

    target_speaker = normalized.get("target_speaker")
    if isinstance(target_speaker, str) and target_speaker.strip().lower() in {
        "teacher",
        "psychological teacher",
        teacher_name.lower(),
    }:
        normalized["target_speaker"] = teacher_name
    else:
        normalized["target_speaker"] = student_name

    next_turn_index_global = 1

    for session_key, turns in normalized.items():
        if not (session_key.startswith("session_") and isinstance(turns, list)):
            continue
        normalized_turns: List[Dict[str, Any]] = []
        for turn in turns:
            if not isinstance(turn, dict):
                normalized_turns.append(turn)
                continue
            normalized_turn = dict(turn)
            speaker = normalized_turn.get("speaker")
            if isinstance(speaker, str):
                lowered = speaker.strip().lower()
                if lowered in known_teacher_tokens:
                    normalized_turn["speaker"] = teacher_name
                elif lowered in known_student_tokens:
                    normalized_turn["speaker"] = student_name
            if not isinstance(normalized_turn.get("turn_index_global"), int):
                normalized_turn["turn_index_global"] = next_turn_index_global
            else:
                next_turn_index_global = int(normalized_turn["turn_index_global"])
            for text_field in ("text", "voice_style"):
                if isinstance(normalized_turn.get(text_field), str):
                    normalized_turn[text_field] = clean_mojibake_text(normalized_turn[text_field])
            if isinstance(normalized_turn.get("voice_style"), str):
                normalized_turn["voice_style"] = sanitize_voice_style(normalized_turn["voice_style"])
            normalized_turns.append(normalized_turn)
            next_turn_index_global = int(normalized_turn["turn_index_global"]) + 1
        normalized[session_key] = normalized_turns

    return ensure_turn_timestamps(normalized)


MOJIBAKE_REPLACEMENTS = {
    "鈥?": "'",
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€": '"',
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "Â ": " ",
    "Â": "",
}


def clean_mojibake_text(value: str) -> str:
    cleaned = value
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        cleaned = cleaned.replace(bad, good)
    # Normalize compatibility forms after deterministic replacements so common quote
    # corruption is fixed without rewriting the sentence semantics.
    return unicodedata.normalize("NFKC", cleaned)


VOICE_STYLE_SANITIZE_REPLACEMENTS = [
    (r"\ba hint of relief\b", "small exhale before the next phrase"),
    (r"\ba sense of relief and acceptance\b", "small exhale, softer closing words"),
    (r"\ba sense of relief\b", "small exhale"),
    (r"\ba clear expression of increased confidence\b", "steadier pace and firmer articulation"),
    (r"\bgenuine gratitude\b", "warmer tone, softer closing words"),
    (r"\bdeep gratitude\b", "warmer tone, softer closing words"),
    (r"\bheartfelt gratitude\b", "warmer tone, softer closing words"),
    (r"\bsincere\b", "steady, softened delivery"),
    (r"\bgrateful\b", "warmer, softer closing words"),
    (r"\bgratitude\b", "warmer, softer closing words"),
    (r"\brelief\b", "small exhale"),
    (r"\bdefensive\b", "firmer, clipped delivery"),
    (r"\bfrustrated\b", "tighter delivery, faster pace"),
    (r"\bfrustration\b", "tighter delivery"),
    (r"\bashamed\b", "quieter delivery, clipped ending"),
    (r"\bshame\b", "quieter delivery"),
    (r"\banxious\b", "quicker pace, tighter delivery"),
    (r"\banxiety\b", "quicker pace, tighter delivery"),
    (r"\bconcerned\b", "slower onset, lower pitch"),
    (r"\bnervous\b", "quicker pace, slightly tight delivery"),
    (r"\bvulnerable\b", "quieter, less guarded delivery"),
    (r"\bemotional\b", "audibly strained"),
    (r"\bthoughtful\b", "brief pause before speaking"),
    (r"\breflecting\b", "looking back"),
    (r"\bindicating\b", "marked by"),
    (r"\bsuggesting\b", "introducing"),
    (r"\bexplaining\b", "spelling out"),
    (r"\bexpressing\b", "with"),
    (r"\badmitting\b", "saying"),
    (r"\bacceptance\b", "more settled pacing"),
    (r"\boptimistic\b", "slightly lighter pace"),
    (r"\bhopeful\b", "slightly lighter pace"),
    (r"\bcontent\b", "more settled pacing"),
    (r"\bobserving\b", "measured, slightly quieter"),
    (r"\bfinality\b", "cleaner sentence ending"),
    (r"\boverwhelmed\b", "under pressure"),
    (r"\bmore emotion\b", "more audible strain"),
    (r"\bemotion\b", "audible strain"),
    (r"\bfeeling\b", ""),
    (r"\brelieved but still focused on\b", "small exhale before mentioning"),
    (r"\brelieved\b", "small exhale before the next phrase"),
    (r"\bappreciative\b", "softer volume, slower closing words"),
    (r"\bpatient\b", "steady pace, unhurried delivery"),
    (r"\battentive\b", "steady pacing, brief pauses"),
    (r"\bcurious\b", "brief pause before speaking"),
    (r"\bcuriosity\b", "extra emphasis on key terms"),
    (r"\bengaged\b", "steadier pace, more emphasis on key terms"),
    (r"\bopen\b", "unhurried onset"),
    (r"\backnowledging\b", "brief pause before the next clause"),
    (r"\baffirming\b", "softer volume, slightly brighter cadence"),
    (r"\binviting\b", "rising intonation at the end"),
    (r"\bprofessional\b", "even, controlled"),
    (r"\bcalm\b", "even pace, lower volume"),
    (r"\bsupportive\b", "softened volume, steadier pace"),
    (r"\breassuring\b", "slower onset, softer volume"),
    (r"\boffering practical help\b", "structured, step-by-step cadence"),
    (r"\bfocused on minute details\b", "extra emphasis on small technical words"),
    (r"\bfocused on\b", "extra emphasis on"),
    (r"\bslightly brighter cadence tone\b", "slightly brighter cadence"),
    (r"\bslightly tense\b", "slightly tight delivery"),
    (r"\bpractical help\b", "step-by-step guidance"),
    (r"\binstructional\b", "structured cadence"),
    (r"\bencouraging\b", "slightly brighter cadence"),
    (r"\ba hint of underlying tension\b", "slightly tight delivery"),
    (r"\bunderlying tension\b", "slightly tight delivery"),
    (r"\bsuggesting a strategy\b", "structured, directive cadence"),
    (r"\bsuggesting refinement\b", "measured, evaluative cadence"),
    (r"\badmitting to feeling overwhelmed\b", "longer pause before the final phrase"),
    (r"\bvalidating\b", "gentle, slower-paced"),
    (r"\breflective\b", "measured pacing"),
    (r"\bshowing understanding\b", "softened volume, brief pauses"),
    (r"\bshowing profound understanding\b", "softened volume, steady pacing"),
    (r"\bshowing genuine intellectual curiosity\b", "steadier pace, more emphasis on analytical terms"),
    (r"\bshowing initiative\b", "firmer articulation"),
    (r"\bshowing active problem-solving\b", "structured, step-by-step cadence"),
    (r"\bshowing clear rationale and agency\b", "clearer emphasis on the final phrase"),
    (r"\bshowing managed self-doubt\b", "slight pause before the final phrase"),
    (r"\bshowing agency\b", "firmer articulation"),
    (r"\bshowing genuine\b", "with"),
    (r"\bmore confident\b", "steadier pace"),
    (r"\bconfident\b", "steady, firmer articulation"),
    (r"\bself-assured\b", "steady, firmer articulation"),
    (r"\bindicating new perspective\b", "slight upward inflection at the end"),
    (r"\bcautious relief\b", "small exhale before the final words"),
    (r"\bempathetic\b", "gentle, softened delivery"),
    (r"\bprioritizing emotional check-in\b", "slower opening, gentle pacing"),
    (r"\backnowledging the difficulty\b", "slower onset, brief pause before the next clause"),
    (r"\brevealing his underlying fear of failure and wasting time\b", "voice tightens slightly, longer pause before the final clause"),
    (r"\bsummarizing his emotional state\b", "measured recap cadence"),
    (r"\brevealing lingering anxiety and guardedness\b", "hesitant start, clipped ending"),
    (r"\bemphasizing the emotional impact\b", "softer volume on the final phrase"),
    (r"\brevealing deeper fear\b", "voice turns quieter mid-sentence"),
    (r"\bvalidating his internal experience\b", "softened volume, deliberate pacing"),
    (r"\bexpressing vulnerability and shame\b", "quieter delivery, trailing final words"),
    (r"\baccepting the challenge with a new mindset\b", "steadier pace, cleaner sentence endings"),
    (r"\ba clear sense of relief and gratitude\b", "lighter pace, softer ending"),
    (r"\banalytical\b", "precise articulation"),
    (r"\bslightly academic\b", "more formal wording"),
    (r"\bslightly abstract\b", "slower onset"),
    (r"\bslightly dismissive\b", "flatter ending, clipped delivery"),
    (r"\bdismissive\b", "flatter ending"),
    (r"\bfocused\b", "firmer articulation on key words"),
    (r"\bdetermined\b", "firmer articulation"),
    (r"\bweariness\b", "longer pause before the final phrase"),
    (r"\bweary\b", "slower onset"),
    (r"\bfatigue\b", "slower pace"),
    (r"\bstrain\b", "slightly tight delivery"),
    (r"\bstressed\b", "slightly tight delivery"),
    (r"\bstress\b", "tighter delivery"),
    (r"\bconcern\b", "slower onset"),
    (r"\bunderlying concern\b", "slower onset on the last phrase"),
    (r"\bunderlying weariness\b", "longer pause before the final phrase"),
    (r"\baudibly strained content\b", "tighter delivery on the final phrase"),
    (r"\baudibly strained\b", "tighter delivery"),
    (r"\bunderlying\b", ""),
    (r"\bundertone of\b", ""),
    (r"\bseeking reassurance\b", "rising intonation at the end"),
    (r"\bseeking tone\b", "rising intonation at the end"),
    (r"\bwhen minimizing\b", "as the final words soften"),
    (r"\bminimizing\b", "softening the final words"),
    (r"\bmasked by\b", "followed by"),
    (r"\bnote of\b", ""),
    (r"\bsense of appreciation for his effort\b", "softer volume, slower closing words"),
    (r"\bforward-looking\b", "cleaner sentence ending"),
    (r"\bobjective\b", "even, clipped"),
]

VOICE_STYLE_FALLBACK_REPLACEMENTS = [
    (r"\b(anxious|anxiety|apprehensive|nervous|worried|stressed|overwhelmed)\b", "slightly tight delivery"),
    (r"\b(frustrated|frustration|defensive|disappointed|disappointment)\b", "firmer, clipped delivery"),
    (r"\b(ashamed|shame|embarrassed)\b", "quieter delivery"),
    (r"\b(relieved|relief|grateful|gratitude|content|hopeful)\b", "softer closing words"),
    (r"\b(confident|self-assured)\b", "firmer articulation"),
    (r"\b(angry|anger|sad|fearful|afraid|resentful|hurt)\b", "tighter delivery"),
    (r"\b(masking|showing|revealing|indicating|suggesting|signifying|reflecting)\b", ""),
    (r"\b(feeling|feels|emotion|emotional|underlying)\b", ""),
]


def _finalize_voice_style_text(value: str) -> str:
    cleaned = value
    for pattern, replacement in VOICE_STYLE_FALLBACK_REPLACEMENTS:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\ba hint of\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bslight\b", "slightly", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r",\s*,", ", ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = cleaned.strip(" ,")
    return cleaned


def collect_voice_style_leak_report(conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
    leaks: List[Dict[str, Any]] = []
    for session_key, turns in conversation.items():
        if not (session_key.startswith("session_") and isinstance(turns, list)):
            continue
        for turn in turns:
            voice_style = str(turn.get("voice_style") or "").strip()
            matches = get_voice_style_leak_matches(voice_style)
            if matches:
                leaks.append(
                    {
                        "session_key": session_key,
                        "dia_id": turn.get("dia_id"),
                        "speaker": turn.get("speaker"),
                        "voice_style": voice_style,
                        "matched_rules": matches,
                    }
                )
    return leaks


def sanitize_voice_style(value: str) -> str:
    sanitized = clean_mojibake_text(value)
    for pattern, replacement in VOICE_STYLE_SANITIZE_REPLACEMENTS:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    sanitized = _finalize_voice_style_text(sanitized)
    if get_voice_style_leak_matches(sanitized):
        sanitized = _finalize_voice_style_text(sanitized)
    return sanitized


def normalize_annotations_metadata(annotations: List[Dict[str, Any]], conversation: Dict[str, Any], personas: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Annotation generation is allowed to be a little "messy" in formatting.
    # This pass keeps the semantic content but trims it back to schema-safe fields.
    teacher_name = personas["teacher"].get("name", "Teacher")
    student_name = personas["student"].get("name", "Student")
    turn_map = flatten_conversation_turns(conversation)
    normalized_annotations: List[Dict[str, Any]] = []

    known_teacher_tokens = {"teacher", "psychological teacher", teacher_name.lower()}
    known_student_tokens = {"student", student_name.lower()}

    for annotation in annotations:
        if not isinstance(annotation, dict):
            normalized_annotations.append(annotation)
            continue
        normalized_annotation = dict(annotation)
        dia_id = normalized_annotation.get("dia_id")
        normalized_annotation["memory_dependency_level"] = normalize_level_label(
            normalized_annotation.get("memory_dependency_level"),
            "Level 0",
        )
        if not isinstance(normalized_annotation.get("critical_event_ids"), list):
            normalized_annotation["critical_event_ids"] = []
        if not isinstance(normalized_annotation.get("evidence_turn_ids"), list):
            normalized_annotation["evidence_turn_ids"] = []
        if not isinstance(normalized_annotation.get("evidence_turn_quotes"), list):
            normalized_annotation["evidence_turn_quotes"] = []
        if dia_id in turn_map:
            normalized_annotation["speaker"] = turn_map[dia_id]["speaker"]
        else:
            speaker = normalized_annotation.get("speaker")
            if isinstance(speaker, str):
                lowered = speaker.strip().lower()
                if lowered in known_teacher_tokens:
                    normalized_annotation["speaker"] = teacher_name
                elif lowered in known_student_tokens:
                    normalized_annotation["speaker"] = student_name

        normalized_annotation["target_role"] = "student"

        normalized_quotes: List[Dict[str, Any]] = []
        for quote in normalized_annotation.get("evidence_turn_quotes", []):
            if not isinstance(quote, dict):
                normalized_quotes.append(quote)
                continue
            normalized_quote = {
                "dia_id": quote.get("dia_id"),
                "speaker": quote.get("speaker"),
                "text_evidence": quote.get("text_evidence"),
            }
            quote_dia_id = normalized_quote.get("dia_id")
            if quote_dia_id in turn_map:
                normalized_quote["speaker"] = turn_map[quote_dia_id]["speaker"]
            else:
                quote_speaker = normalized_quote.get("speaker")
                if isinstance(quote_speaker, str):
                    lowered = quote_speaker.strip().lower()
                    if lowered in known_teacher_tokens:
                        normalized_quote["speaker"] = teacher_name
                    elif lowered in known_student_tokens:
                        normalized_quote["speaker"] = student_name
            normalized_quotes.append(normalized_quote)
        normalized_annotation["evidence_turn_quotes"] = normalized_quotes
        normalized_annotations.append(normalized_annotation)

    return normalized_annotations


def strip_multi_agent_fields_from_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    stripped = deepcopy(bundle)
    stripped_event_plan: List[Dict[str, Any]] = []
    for event in stripped.get("event_plan", []):
        if not isinstance(event, dict):
            stripped_event_plan.append(event)
            continue
        stripped_event = dict(event)
        stripped_event.pop("interaction_targets", None)
        stripped_event_plan.append(stripped_event)
    stripped["event_plan"] = stripped_event_plan
    return stripped


def generate_conversation_from_bundle(
    client: ChatClient,
    prompts_dir: Path,
    schemas_dir: Path,
    bundle: Dict[str, Any],
    out_path: Path,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return generate_conversation_from_bundle_stage(
        client=client,
        prompts_dir=prompts_dir,
        schemas_dir=schemas_dir,
        bundle=bundle,
        out_path=out_path,
        config=config,
        load_json_fn=load_json,
        save_json_fn=save_json,
        save_text_fn=save_text,
        collect_voice_style_leak_report_fn=collect_voice_style_leak_report,
        render_prompt_template_fn=render_prompt_template,
        get_prompt_language_fn=get_prompt_language,
        get_conversation_generation_config_fn=get_conversation_generation_config,
        strip_multi_agent_fields_from_bundle_fn=strip_multi_agent_fields_from_bundle,
        generate_conversation_with_autogen_fn=generate_conversation_with_autogen,
        normalize_conversation_metadata_fn=normalize_conversation_metadata,
        validate_conversation_fn=validate_conversation,
        validate_conversation_against_blueprint_fn=validate_conversation_against_blueprint,
        asyncio_run_fn=asyncio.run,
    )


def generate_annotations_from_bundle(
    client: ChatClient,
    prompts_dir: Path,
    schemas_dir: Path,
    bundle: Dict[str, Any],
    conversation_path: Path,
    out_path: Path,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    return generate_annotations_from_bundle_stage(
        client=client,
        prompts_dir=prompts_dir,
        schemas_dir=schemas_dir,
        bundle=bundle,
        conversation_path=conversation_path,
        out_path=out_path,
        config=config,
        load_json_fn=load_json,
        save_json_fn=save_json,
        render_prompt_template_fn=render_prompt_template,
        get_prompt_language_fn=get_prompt_language,
        normalize_annotations_metadata_fn=normalize_annotations_metadata,
        validate_annotations_fn=validate_annotations,
    )


def generate_questions_from_bundle(
    client: ChatClient,
    prompts_dir: Path,
    schemas_dir: Path,
    bundle: Dict[str, Any],
    conversation_path: Path,
    annotation_path: Path,
    out_path: Path,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    system_prompt = render_prompt_template(prompts_dir, "question_generation_prompt_v2.j2", get_prompt_language(config))
    conversation = load_json(conversation_path)
    annotations = load_json(annotation_path)
    return generate_questions_from_inputs(
        client=client,
        system_prompt=system_prompt,
        schemas_dir=schemas_dir,
        bundle=bundle,
        conversation=conversation,
        annotations=annotations,
        out_path=out_path,
    )


def flatten_conversation_turns(conversation: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    turn_map = shared_flatten_conversation_turns(conversation)
    for key, value in conversation.items():
        if key.startswith("session_") and isinstance(value, list):
            for turn in value:
                if not turn.get("dia_id"):
                    raise ValueError(f"Missing dia_id in turn under {key}: {turn}")
    return turn_map


def get_session_turns(conversation: Dict[str, Any]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Return only turn-bearing session arrays, ordered by session key."""
    session_items: List[Tuple[str, List[Dict[str, Any]]]] = []
    for key, value in conversation.items():
        if key.startswith("session_") and not key.endswith("_date_time") and isinstance(value, list):
            session_items.append((key, value))
    return sorted(session_items, key=lambda item: item[0])


def build_annotation_map(annotations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    ann_map: Dict[str, Dict[str, Any]] = {}
    for ann in annotations:
        dia_id = ann.get("dia_id")
        if not dia_id:
            raise ValueError(f"Annotation missing dia_id: {ann}")
        ann_map[dia_id] = ann
    return ann_map


def extract_turn_subset(turn_map: Dict[str, Dict[str, Any]], dia_ids: List[str]) -> List[Dict[str, Any]]:
    return [turn_map[dia_id] for dia_id in dia_ids]


def parse_dia_id(dia_id: str) -> tuple[int, int]:
    match = re.fullmatch(r"D(\d+):(\d+)", str(dia_id).strip())
    if not match:
        raise ValueError(f"Invalid dia_id format: {dia_id}")
    return int(match.group(1)), int(match.group(2))


def build_turn_payload(turn: Dict[str, Any], include_voice_style: bool = True) -> Dict[str, Any]:
    payload = {
        "speaker": turn.get("speaker"),
        "text": turn.get("text"),
        "timestamp": turn.get("timestamp"),
    }
    if include_voice_style:
        voice_style = turn.get("voice_style")
        if voice_style not in {None, ""}:
            payload["voice_style"] = voice_style
    return payload


def collect_context_turn_records(conversation: Dict[str, Any], anchor_dia_id: str, session_local_only: bool) -> List[Dict[str, Any]]:
    anchor_session, anchor_turn = parse_dia_id(anchor_dia_id)
    context_turns: List[Dict[str, Any]] = []
    for session_key, turns in sorted(get_session_turns(conversation), key=lambda item: item[0]):
        session_number = int(session_key.replace("session_", ""))
        if session_number > anchor_session:
            continue
        if session_local_only and session_number != anchor_session:
            continue
        for turn in turns:
            _, turn_number = parse_dia_id(turn["dia_id"])
            if session_number == anchor_session and turn_number > anchor_turn:
                break
            context_turns.append(turn)
    return context_turns


def collect_context_turns(conversation: Dict[str, Any], anchor_dia_id: str, session_local_only: bool) -> List[Dict[str, Any]]:
    return [
        build_turn_payload(turn)
        for turn in collect_context_turn_records(conversation, anchor_dia_id, session_local_only)
    ]


def build_retrieval_candidates(conversation: Dict[str, Any], anchor_dia_id: str) -> List[List[Dict[str, Any]]]:
    anchor_session, anchor_turn = parse_dia_id(anchor_dia_id)
    candidates: List[List[Dict[str, Any]]] = []
    for session_key, turns in sorted(get_session_turns(conversation), key=lambda item: item[0]):
        session_number = int(session_key.replace("session_", ""))
        if session_number > anchor_session:
            continue
        session_candidates: List[Dict[str, Any]] = []
        for turn in turns:
            _, turn_number = parse_dia_id(turn["dia_id"])
            if session_number == anchor_session and turn_number >= anchor_turn:
                break
            session_candidates.append(build_turn_payload(turn))
        for index in range(0, len(session_candidates), 2):
            window = session_candidates[index:index + 2]
            if window:
                candidates.append(window)
    return candidates


def get_local_modality_target_ids(anchor_dia_id: str) -> List[str]:
    """Mask only the local cue around the anchor, not the full evidence chain."""
    session_number, turn_number = parse_dia_id(anchor_dia_id)
    target_ids = [anchor_dia_id]
    if turn_number > 1:
        target_ids.insert(0, f"D{session_number}:{turn_number - 1}")
    return target_ids


def apply_modality_condition_to_turns(
    turns: List[Dict[str, Any]],
    target_dia_ids: set[str],
    condition_key: str,
) -> List[Dict[str, Any]]:
    conditioned_context: List[Dict[str, Any]] = []
    for turn in turns:
        payload = build_turn_payload(turn)
        if turn.get("dia_id") in target_dia_ids:
            if condition_key == "voice_style_missing":
                payload["voice_style"] = "unavailable"
            elif condition_key == "voice_style_ambiguous" and payload.get("voice_style"):
                payload["voice_style"] = "unclear"
        conditioned_context.append(payload)
    return conditioned_context


def build_modality_conditioned_views(conversation: Dict[str, Any], anchor_dia_id: str, question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    question_type = str(question.get("question_type") or "")
    modality_condition = str(question.get("modality_condition") or "normal")
    if question_type not in {"modality_missing", "modality_ambiguous"} and modality_condition == "normal":
        return None

    session_local_records = collect_context_turn_records(conversation, anchor_dia_id, session_local_only=True)
    full_history_records = collect_context_turn_records(conversation, anchor_dia_id, session_local_only=False)
    target_dia_ids = set(get_local_modality_target_ids(anchor_dia_id))
    condition_key = str(modality_condition or "").lower().replace("removed", "missing")
    if condition_key == "normal":
        condition_key = "voice_style_missing" if question_type == "modality_missing" else "voice_style_ambiguous"

    return {
        condition_key: {
            "session_local_context": apply_modality_condition_to_turns(session_local_records, target_dia_ids, condition_key),
            "full_history_context": apply_modality_condition_to_turns(full_history_records, target_dia_ids, condition_key),
        },
    }


def build_unit_from_question(
    conversation: Dict[str, Any],
    annotations: List[Dict[str, Any]],
    question: Dict[str, Any],
    curation_hint: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    turn_map = flatten_conversation_turns(conversation)
    annotation_map = build_annotation_map(annotations)
    anchor_turn = turn_map[question["anchor_dia_id"]]
    evidence_turns = extract_turn_subset(turn_map, question.get("evidence_turn_ids", []))
    session_local_context = collect_context_turns(conversation, question["anchor_dia_id"], session_local_only=True)
    full_history_context = collect_context_turns(conversation, question["anchor_dia_id"], session_local_only=False)
    retrieval_candidates = build_retrieval_candidates(conversation, question["anchor_dia_id"])
    modality_conditioned_views = build_modality_conditioned_views(conversation, question["anchor_dia_id"], question)
    task_layer = {
        "content_type": question.get("content_type"),
        "question_type": question.get("question_type"),
        "memory_level": question.get("memory_level"),
        "reasoning_structure": question.get("reasoning_structure"),
        "question_text": question.get("question_text"),
        "options": question.get("options", []),
        "modality_condition": question.get("modality_condition"),
        "adversarial_flag": question.get("adversarial_flag", False),
        "adversarial_type": question.get("adversarial_type"),
    }
    return {
        "unit_id": f"U_{question['question_id']}",
        "conversation_id": conversation.get("conversation_id"),
        "question_id": question["question_id"],
        "anchor": {
            "dia_id": anchor_turn.get("dia_id"),
            "speaker": anchor_turn.get("speaker"),
            "text": anchor_turn.get("text"),
            "timestamp": anchor_turn.get("timestamp"),
            "audio_id": anchor_turn.get("audio_id"),
            "voice_style": anchor_turn.get("voice_style"),
            "modality_available": anchor_turn.get("modality_available", []),
            "turn_index_global": anchor_turn.get("turn_index_global"),
        },
        "history_evidence": [
            {
                "dia_id": t.get("dia_id"),
                "speaker": t.get("speaker"),
                "text": t.get("text"),
                "timestamp": t.get("timestamp"),
                "audio_id": t.get("audio_id"),
                "voice_style": t.get("voice_style"),
                "modality_available": t.get("modality_available", []),
                "turn_index_global": t.get("turn_index_global"),
            }
            for t in evidence_turns
        ],
        "annotation": annotation_map.get(question["anchor_dia_id"]),
        "question": task_layer,
        "task_layer": task_layer,
        "gold": {
            "gold_answer": question.get("gold_answer"),
            "acceptable_answers": question.get("acceptable_answers", []),
            "critical_event_ids": question.get("critical_event_ids", []),
            "evidence_turn_ids": question.get("evidence_turn_ids", []),
            "gold_rationale": question.get("gold_rationale"),
            "key_explanation_points": question.get("key_explanation_points", []),
        },
        "benchmark_views": {
            "session_local_context": session_local_context,
            "full_history_context": full_history_context,
            "retrieval_candidates": retrieval_candidates,
            "modality_conditioned_views": modality_conditioned_views,
        },
        "curation_hints": curation_hint or {"tags": [], "reasons": []},
    }


def build_all_units(conversation_path: Path, annotation_path: Path, qa_path: Path, out_path: Path) -> List[Dict[str, Any]]:
    conversation = load_json(conversation_path)
    conversation = ensure_turn_timestamps(conversation)
    save_json(conversation, conversation_path)
    annotations = load_json(annotation_path)
    questions = load_json(qa_path)
    curation_hints = build_question_curation_hints(questions, conversation)
    units = [
        build_unit_from_question(
            conversation,
            annotations,
            q,
            curation_hint=curation_hints.get(str(q.get("question_id") or "<unknown>")),
        )
        for q in questions
    ]
    validate_units(units, conversation, annotations, questions)
    save_json(units, out_path)
    return units


def validate_scenario_dir(scenario_dir: Path, schemas_dir: Path) -> None:
    conversation_path = scenario_dir / "conversation.json"
    annotation_path = scenario_dir / "annotation.json"
    qa_path = scenario_dir / "qa.json"
    units_path = scenario_dir / "all_units.json"
    conversation = load_json(conversation_path)
    annotations = load_json(annotation_path)
    questions = load_json(qa_path)
    units = load_json(units_path)
    validate_conversation(conversation, schemas_dir / "conversation.schema.json")
    validate_conversation_against_blueprint(
        conversation,
        load_json(scenario_dir / "global_outline.json"),
        load_json(scenario_dir / "session_scripts.json"),
    )
    validate_annotations(annotations, conversation, schemas_dir / "annotation.schema.json")
    validate_questions(questions, conversation, schemas_dir / "qa.schema.json")
    validate_units(units, conversation, annotations, questions)


def run_existing_scenario_dir(client: ChatClient, prompts_dir: Path, schemas_dir: Path, scenario_dir: Path, config: Optional[Dict[str, Any]] = None) -> None:
    bundle = load_bundle_json_files(scenario_dir)
    conversation_path = scenario_dir / "conversation.json"
    annotation_path = scenario_dir / "annotation.json"
    qa_path = scenario_dir / "qa.json"
    units_path = scenario_dir / "all_units.json"
    generate_conversation_from_bundle(client, prompts_dir, schemas_dir, bundle, conversation_path, config=config)
    generate_annotations_from_bundle(client, prompts_dir, schemas_dir, bundle, conversation_path, annotation_path, config=config)
    generate_questions_from_bundle(client, prompts_dir, schemas_dir, bundle, conversation_path, annotation_path, qa_path, config=config)
    build_all_units(conversation_path, annotation_path, qa_path, units_path)
    validate_scenario_dir(scenario_dir, schemas_dir)


def run_existing_scenario_conversation(client: ChatClient, prompts_dir: Path, schemas_dir: Path, scenario_dir: Path, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    bundle = load_bundle_json_files(scenario_dir)
    conversation_path = scenario_dir / "conversation.json"
    return generate_conversation_from_bundle(client, prompts_dir, schemas_dir, bundle, conversation_path, config=config)


def run_existing_scenario_annotations(
    client: ChatClient,
    prompts_dir: Path,
    schemas_dir: Path,
    scenario_dir: Path,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    bundle = load_bundle_json_files(scenario_dir)
    conversation_path = scenario_dir / "conversation.json"
    annotation_path = scenario_dir / "annotation.json"
    return generate_annotations_from_bundle(client, prompts_dir, schemas_dir, bundle, conversation_path, annotation_path, config=config)


def run_existing_scenario_questions(
    client: ChatClient,
    prompts_dir: Path,
    schemas_dir: Path,
    scenario_dir: Path,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    bundle = load_bundle_json_files(scenario_dir)
    conversation_path = scenario_dir / "conversation.json"
    annotation_path = scenario_dir / "annotation.json"
    qa_path = scenario_dir / "qa.json"
    return generate_questions_from_bundle(client, prompts_dir, schemas_dir, bundle, conversation_path, annotation_path, qa_path, config=config)


def run_existing_scenario_units(schemas_dir: Path, scenario_dir: Path) -> List[Dict[str, Any]]:
    conversation_path = scenario_dir / "conversation.json"
    annotation_path = scenario_dir / "annotation.json"
    qa_path = scenario_dir / "qa.json"
    units_path = scenario_dir / "all_units.json"
    units = build_all_units(conversation_path, annotation_path, qa_path, units_path)
    validate_scenario_dir(scenario_dir, schemas_dir)
    return units


def run_existing_batch_dir(client: ChatClient, prompts_dir: Path, schemas_dir: Path, batch_dir: Path, config: Optional[Dict[str, Any]] = None) -> List[Path]:
    scenario_dirs = sorted(
        path for path in batch_dir.iterdir()
        if path.is_dir() and path.name.startswith("scenario_")
    )
    if not scenario_dirs:
        raise ValueError(f"No scenario_* directories found under {batch_dir}")
    for scenario_dir in scenario_dirs:
        run_existing_scenario_dir(client, prompts_dir, schemas_dir, scenario_dir, config=config)
    return scenario_dirs


def prepare_blueprints_from_config(client: ChatClient, prompts_dir: Path, templates_dir: Path, schemas_dir: Path, config_path: Path) -> List[Path]:
    config = load_generation_config(config_path, schemas_dir)
    persona_pool = load_json(templates_dir / "persona_pool.template.json")
    output_root = Path(config["output_root"])
    batch_run_name = build_batch_run_name(config)
    generated_dirs: List[Path] = []
    prior_batch_summaries: List[Dict[str, Any]] = []

    for scenario_index in range(1, config["dataset_count"] + 1):
        scenario_dir = build_scenario_dir(output_root, batch_run_name, scenario_index)
        selected_personas = select_persona_pair(
            persona_pool=persona_pool,
            scenario_index=scenario_index,
            mode=config["persona_selection"]["mode"],
            seed=config["persona_selection"]["seed"],
        )
        bundle = normalize_blueprint_bundle(
            bundle=generate_blueprint_bundle(
                client,
                prompts_dir,
                templates_dir,
                config,
                selected_personas,
                scenario_index,
                prior_batch_summaries,
            ),
            config=config,
            selected_personas=selected_personas,
        )
        bundle["question_plan"] = build_question_plan_from_blueprint(
            config=config,
            bundle=bundle,
            templates_dir=templates_dir,
            scenario_index=scenario_index,
        )

        try:
            validate_blueprint_bundle(
                bundle["personas"],
                bundle["global_outline"],
                bundle["session_scripts"],
                bundle["event_plan"],
                bundle["emotion_arc"],
                bundle["question_plan"],
                schemas_dir,
                config=config,
            )
        except Exception as exc:
            if not config["blueprint_generation"]["enable_repair"]:
                raise
            current_bundle = bundle
            current_error = str(exc)
            repaired = False
            for _ in range(config["blueprint_generation"]["max_repair_attempts"]):
                current_bundle = normalize_blueprint_bundle(
                    bundle=repair_blueprint_bundle(client, prompts_dir, config, current_bundle, current_error),
                    config=config,
                    selected_personas=selected_personas,
                )
                current_bundle["question_plan"] = build_question_plan_from_blueprint(
                    config=config,
                    bundle=current_bundle,
                    templates_dir=templates_dir,
                    scenario_index=scenario_index,
                )
                try:
                    validate_blueprint_bundle(
                        current_bundle["personas"],
                        current_bundle["global_outline"],
                        current_bundle["session_scripts"],
                        current_bundle["event_plan"],
                        current_bundle["emotion_arc"],
                        current_bundle["question_plan"],
                        schemas_dir,
                        config=config,
                    )
                    bundle = current_bundle
                    repaired = True
                    break
                except Exception as repair_exc:
                    current_error = str(repair_exc)
            if not repaired:
                raise

        saved_config = dict(config)
        saved_config["resolved_batch_run_name"] = batch_run_name
        save_json(saved_config, scenario_dir / "generation_config.json")
        save_bundle_json_files(bundle, scenario_dir)
        prior_batch_summaries.append(classify_scenario_skeleton(bundle))
        print(summarize_scenario_bundle(scenario_index, scenario_dir, config, bundle))
        generated_dirs.append(scenario_dir)
    return generated_dirs


def run_batch_from_config(client: ChatClient, prompts_dir: Path, templates_dir: Path, schemas_dir: Path, config_path: Path) -> List[Path]:
    config = load_json(config_path)
    scenario_dirs = prepare_blueprints_from_config(client, prompts_dir, templates_dir, schemas_dir, config_path)
    for scenario_dir in scenario_dirs:
        bundle = load_bundle_json_files(scenario_dir)
        conversation_path = scenario_dir / "conversation.json"
        annotation_path = scenario_dir / "annotation.json"
        qa_path = scenario_dir / "qa.json"
        units_path = scenario_dir / "all_units.json"
        generate_conversation_from_bundle(client, prompts_dir, schemas_dir, bundle, conversation_path, config=config)
        generate_annotations_from_bundle(client, prompts_dir, schemas_dir, bundle, conversation_path, annotation_path, config=config)
        generate_questions_from_bundle(client, prompts_dir, schemas_dir, bundle, conversation_path, annotation_path, qa_path, config=config)
        build_all_units(conversation_path, annotation_path, qa_path, units_path)
        validate_scenario_dir(scenario_dir, schemas_dir)
    return scenario_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    parser.add_argument("--config-path", type=str, default="configs/generation_config.example.json", help="Path to generation config")
    parser.add_argument("--scenario-dir", type=str, help="Scenario directory for validation")
    parser.add_argument("--base-scenario-dir", type=str, help="Base scenario directory used to derive turn-budget variants")
    parser.add_argument("--batch-dir", type=str, help="Existing batch directory containing scenario_* subdirectories")
    parser.add_argument("--output-dir", type=str, help="Output directory for derived scenarios")
    parser.add_argument("--turn-totals", type=str, help="Comma-separated total turn counts for derived length variants")
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=[
            "prepare_blueprints",
            "derive_turn_variants",
            "run_batch",
            "run_existing_batch",
            "run_existing_scenario",
            "generate_conversation",
            "generate_annotations",
            "generate_questions",
            "build_units",
            "validate_scenario",
        ],
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    prompts_dir = root / "prompts"
    schemas_dir = root / "schemas"
    templates_dir = root / "templates"
    config_path = (root / args.config_path).resolve()
    config = load_generation_config(config_path, schemas_dir) if args.step in {
        "prepare_blueprints",
        "run_batch",
        "run_existing_batch",
        "run_existing_scenario",
        "generate_conversation",
        "generate_annotations",
        "generate_questions",
    } else None

    if args.step == "prepare_blueprints":
        client = ChatClient(config["llm"])
        for scenario_dir in prepare_blueprints_from_config(client, prompts_dir, templates_dir, schemas_dir, config_path):
            print(f"Prepared {scenario_dir}")
    elif args.step == "derive_turn_variants":
        if not args.base_scenario_dir:
            raise ValueError("--base-scenario-dir is required for derive_turn_variants")
        if not args.output_dir:
            raise ValueError("--output-dir is required for derive_turn_variants")
        if not args.turn_totals:
            raise ValueError("--turn-totals is required for derive_turn_variants")
        base_scenario_dir = Path(args.base_scenario_dir).resolve()
        output_dir = Path(args.output_dir).resolve()
        turn_totals = [int(part.strip()) for part in args.turn_totals.split(",") if part.strip()]
        for scenario_dir in derive_turn_variant_scenarios(base_scenario_dir, output_dir, turn_totals, schemas_dir):
            print(f"Prepared {scenario_dir}")
    elif args.step == "run_batch":
        client = ChatClient(config["llm"])
        for scenario_dir in run_batch_from_config(client, prompts_dir, templates_dir, schemas_dir, config_path):
            print(f"Generated {scenario_dir}")
    elif args.step == "run_existing_batch":
        if not args.batch_dir:
            raise ValueError("--batch-dir is required for run_existing_batch")
        client = ChatClient(config["llm"])
        batch_dir = Path(args.batch_dir).resolve()
        for scenario_dir in run_existing_batch_dir(client, prompts_dir, schemas_dir, batch_dir, config=config):
            print(f"Generated {scenario_dir}")
    elif args.step == "run_existing_scenario":
        if not args.scenario_dir:
            raise ValueError("--scenario-dir is required for run_existing_scenario")
        client = ChatClient(config["llm"])
        scenario_dir = Path(args.scenario_dir).resolve()
        run_existing_scenario_dir(client, prompts_dir, schemas_dir, scenario_dir, config=config)
        print(f"Generated {scenario_dir}")
    elif args.step == "generate_conversation":
        if not args.scenario_dir:
            raise ValueError("--scenario-dir is required for generate_conversation")
        client = ChatClient(config["llm"])
        scenario_dir = Path(args.scenario_dir).resolve()
        run_existing_scenario_conversation(client, prompts_dir, schemas_dir, scenario_dir, config=config)
        print(f"Generated conversation for {scenario_dir}")
    elif args.step == "generate_annotations":
        if not args.scenario_dir:
            raise ValueError("--scenario-dir is required for generate_annotations")
        client = ChatClient(config["llm"])
        scenario_dir = Path(args.scenario_dir).resolve()
        run_existing_scenario_annotations(client, prompts_dir, schemas_dir, scenario_dir, config=config)
        print(f"Generated annotations for {scenario_dir}")
    elif args.step == "generate_questions":
        if not args.scenario_dir:
            raise ValueError("--scenario-dir is required for generate_questions")
        client = ChatClient(config["llm"])
        scenario_dir = Path(args.scenario_dir).resolve()
        run_existing_scenario_questions(client, prompts_dir, schemas_dir, scenario_dir, config=config)
        print(f"Generated questions for {scenario_dir}")
    elif args.step == "build_units":
        if not args.scenario_dir:
            raise ValueError("--scenario-dir is required for build_units")
        scenario_dir = Path(args.scenario_dir).resolve()
        run_existing_scenario_units(schemas_dir, scenario_dir)
        print(f"Built units for {scenario_dir}")
    elif args.step == "validate_scenario":
        if not args.scenario_dir:
            raise ValueError("--scenario-dir is required for validate_scenario")
        scenario_dir = Path(args.scenario_dir).resolve()
        validate_scenario_dir(scenario_dir, schemas_dir)
        print(f"Validated {scenario_dir}")


if __name__ == "__main__":
    main()
