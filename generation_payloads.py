"""Prompt payload builders for blueprint, conversation, and annotation stages.

These functions are intentionally pure string builders. They keep long prompt
assembly blocks out of ``pipeline.py`` without owning any runtime logic.
"""

import json
from typing import Any, Dict, List


def build_blueprint_structure_user_payload(
    config: Dict[str, Any],
    selected_personas: Dict[str, Any],
    selected_closure_profile: str,
    blueprint_seed: Dict[str, Any],
    reference_global_outline: Dict[str, Any],
    reference_session_scripts: List[Dict[str, Any]],
    prior_batch_summaries: List[Dict[str, Any]],
) -> str:
    return f"""
Generate the interaction blueprint structure for one benchmark scenario.

Return JSON only.

[generation_config.json]
{json.dumps(config, ensure_ascii=False, indent=2)}

[selected_personas.json]
{json.dumps(selected_personas, ensure_ascii=False, indent=2)}

[selected_closure_profile]
{selected_closure_profile}

[blueprint_seed.template.json]
{json.dumps(blueprint_seed, ensure_ascii=False, indent=2)}

[reference_global_outline.template.json]
{json.dumps(reference_global_outline, ensure_ascii=False, indent=2)}

[reference_session_scripts.template.json]
{json.dumps(reference_session_scripts, ensure_ascii=False, indent=2)}

[prior_batch_summaries.json]
{json.dumps(prior_batch_summaries, ensure_ascii=False, indent=2)}
""".strip()


def build_blueprint_detail_user_payload(
    config: Dict[str, Any],
    personas: Dict[str, Any],
    global_outline: Dict[str, Any],
    session_scripts: List[Dict[str, Any]],
    blueprint_seed: Dict[str, Any],
    reference_event_plan: List[Dict[str, Any]],
    reference_emotion_arc: List[Dict[str, Any]],
) -> str:
    return f"""
Generate the detailed event and emotion blueprint for one benchmark scenario.

Return JSON only.

[generation_config.json]
{json.dumps(config, ensure_ascii=False, indent=2)}

[personas.json]
{json.dumps(personas, ensure_ascii=False, indent=2)}

[global_outline.json]
{json.dumps(global_outline, ensure_ascii=False, indent=2)}

[session_scripts.json]
{json.dumps(session_scripts, ensure_ascii=False, indent=2)}

[blueprint_seed.template.json]
{json.dumps(blueprint_seed, ensure_ascii=False, indent=2)}

[reference_event_plan.template.json]
{json.dumps(reference_event_plan, ensure_ascii=False, indent=2)}

[reference_emotion_arc.template.json]
{json.dumps(reference_emotion_arc, ensure_ascii=False, indent=2)}
""".strip()


def build_blueprint_repair_user_payload(config: Dict[str, Any], bundle: Dict[str, Any], validation_error: str) -> str:
    return f"""
Repair the current planning bundle so it passes validation.

[generation_config.json]
{json.dumps(config, ensure_ascii=False, indent=2)}

[validation_error]
{validation_error}

[current_bundle.json]
{json.dumps(bundle, ensure_ascii=False, indent=2)}
""".strip()


def build_dialogue_user_payload(
    personas: Dict[str, Any],
    global_outline: Dict[str, Any],
    session_scripts: List[Dict[str, Any]],
    event_plan: List[Dict[str, Any]],
    emotion_arc: List[Dict[str, Any]],
) -> str:
    return f"""
Task: Generate the structured conversation JSON.

[personas.json]
{json.dumps(personas, ensure_ascii=False, indent=2)}

[global_outline.json]
{json.dumps(global_outline, ensure_ascii=False, indent=2)}

[session_scripts.json]
{json.dumps(session_scripts, ensure_ascii=False, indent=2)}

[event_plan.json]
{json.dumps(event_plan, ensure_ascii=False, indent=2)}

[emotion_arc.json]
{json.dumps(emotion_arc, ensure_ascii=False, indent=2)}
""".strip()


def build_annotation_user_payload(conversation: Dict[str, Any], event_plan: List[Dict[str, Any]]) -> str:
    return f"""
Annotate the structured conversation.

[conversation.json]
{json.dumps(conversation, ensure_ascii=False, indent=2)}

[event_plan.json]
{json.dumps(event_plan, ensure_ascii=False, indent=2)}
""".strip()
