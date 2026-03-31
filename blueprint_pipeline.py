"""Blueprint-stage orchestration helpers.

This module keeps the structure/detail/repair blueprint flow separate from the
top-level CLI orchestration in ``pipeline.py``. It owns stage sequencing for the
blueprint bundle, while callers still provide project-specific helpers such as
template loading and question-plan construction.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List

from generation_payloads import (
    build_blueprint_detail_user_payload,
    build_blueprint_repair_user_payload,
    build_blueprint_structure_user_payload,
)


def generate_blueprint_bundle(
    client: Any,
    prompts_dir: Path,
    templates_dir: Path,
    config: Dict[str, Any],
    selected_personas: Dict[str, Any],
    scenario_index: int,
    prior_batch_summaries: List[Dict[str, Any]],
    *,
    load_json_fn: Callable[[Path], Any],
    get_prompt_language_fn: Callable[[Dict[str, Any]], str],
    render_prompt_template_fn: Callable[[Path, str, str], str],
    choose_closure_profile_fn: Callable[[Dict[str, Any], int, Dict[str, Any]], str],
    build_question_plan_from_blueprint_fn: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    blueprint_seed = load_json_fn(templates_dir / "blueprint_seed.template.json")
    reference_global_outline = load_json_fn(templates_dir / "global_outline.template.json")
    reference_session_scripts = load_json_fn(templates_dir / "session_scripts.template.json")
    reference_event_plan = load_json_fn(templates_dir / "event_plan.template.json")
    reference_emotion_arc = load_json_fn(templates_dir / "emotion_arc.template.json")
    prompt_language = get_prompt_language_fn(config)
    selected_closure_profile = choose_closure_profile_fn(selected_personas, scenario_index, config)

    structure_prompt = render_prompt_template_fn(prompts_dir, "scenario_blueprint_structure_generation_prompt_v1.j2", prompt_language)
    structure_payload = build_blueprint_structure_user_payload(
        config,
        selected_personas,
        selected_closure_profile,
        blueprint_seed,
        reference_global_outline,
        reference_session_scripts,
        prior_batch_summaries,
    )
    structure_bundle = client.chat_json(system_prompt=structure_prompt, user_prompt=structure_payload)

    detail_prompt = render_prompt_template_fn(prompts_dir, "scenario_blueprint_detail_generation_prompt_v1.j2", prompt_language)
    detail_payload = build_blueprint_detail_user_payload(
        config=config,
        personas=structure_bundle.get("personas", selected_personas),
        global_outline=structure_bundle.get("global_outline", {}),
        session_scripts=structure_bundle.get("session_scripts", []),
        blueprint_seed=blueprint_seed,
        reference_event_plan=reference_event_plan,
        reference_emotion_arc=reference_emotion_arc,
    )
    detail_bundle = client.chat_json(system_prompt=detail_prompt, user_prompt=detail_payload)

    partial_bundle = {
        "personas": structure_bundle.get("personas", selected_personas),
        "global_outline": structure_bundle.get("global_outline", {}),
        "session_scripts": structure_bundle.get("session_scripts", []),
        "event_plan": detail_bundle.get("event_plan", []),
        "emotion_arc": detail_bundle.get("emotion_arc", []),
    }
    question_plan = build_question_plan_from_blueprint_fn(
        config=config,
        bundle=partial_bundle,
        templates_dir=templates_dir,
        scenario_index=scenario_index,
    )

    return {
        **partial_bundle,
        "question_plan": question_plan,
    }


def repair_blueprint_bundle(
    client: Any,
    prompts_dir: Path,
    config: Dict[str, Any],
    bundle: Dict[str, Any],
    validation_error: str,
    *,
    render_prompt_template_fn: Callable[[Path, str, str], str],
    get_prompt_language_fn: Callable[[Dict[str, Any]], str],
) -> Dict[str, Any]:
    system_prompt = render_prompt_template_fn(
        prompts_dir,
        "scenario_blueprint_repair_prompt_v1.j2",
        get_prompt_language_fn(config),
    )
    user_payload = build_blueprint_repair_user_payload(config, bundle, validation_error)
    return client.chat_json(system_prompt=system_prompt, user_prompt=user_payload)
