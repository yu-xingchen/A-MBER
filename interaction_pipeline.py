"""Conversation and annotation stage orchestration.

This module keeps the per-scenario conversation/annotation entrypoints separate
from the top-level CLI pipeline while still allowing the caller to inject
project-specific helpers and optional multi-agent behavior.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from generation_payloads import build_annotation_user_payload, build_dialogue_user_payload


def generate_conversation_from_bundle(
    client: Any,
    prompts_dir: Path,
    schemas_dir: Path,
    bundle: Dict[str, Any],
    out_path: Path,
    config: Optional[Dict[str, Any]] = None,
    *,
    load_json_fn: Callable[[Path], Any],
    save_json_fn: Callable[[Any, Path], None],
    save_text_fn: Callable[[str, Path], None],
    collect_voice_style_leak_report_fn: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
    render_prompt_template_fn: Callable[[Path, str, str], str],
    get_prompt_language_fn: Callable[[Optional[Dict[str, Any]]], str],
    get_conversation_generation_config_fn: Callable[[Optional[Dict[str, Any]]], Dict[str, Any]],
    strip_multi_agent_fields_from_bundle_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    generate_conversation_with_autogen_fn: Callable[..., Any],
    normalize_conversation_metadata_fn: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
    validate_conversation_fn: Callable[[Dict[str, Any], Path], None],
    validate_conversation_against_blueprint_fn: Callable[[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]], None],
    asyncio_run_fn: Callable[[Any], Any],
) -> Dict[str, Any]:
    if out_path.exists():
        print(f"Skipping conversation generation because output already exists: {out_path}")
        return load_json_fn(out_path)
    raw_out_path = out_path.with_name("conversation_raw.json")
    error_out_path = out_path.with_name("conversation_validation_error.txt")
    leak_json_path = out_path.with_name("conversation_voice_style_leaks.json")
    leak_text_path = out_path.with_name("conversation_voice_style_leaks.txt")

    conversation_config = get_conversation_generation_config_fn(config)
    if conversation_config["mode"] == "multi_agent":
        if conversation_config["framework"] != "autogen":
            raise ValueError(f"Unsupported conversation generation framework: {conversation_config['framework']}")
        conversation = asyncio_run_fn(
            generate_conversation_with_autogen_fn(
                client=client,
                prompts_dir=prompts_dir,
                bundle=bundle,
                conversation_config=conversation_config,
            )
        )
    else:
        single_agent_bundle = strip_multi_agent_fields_from_bundle_fn(bundle)
        system_prompt = render_prompt_template_fn(prompts_dir, "dialogue_generation_prompt_v2.j2", get_prompt_language_fn(config))
        user_payload = build_dialogue_user_payload(
            personas=single_agent_bundle["personas"],
            global_outline=single_agent_bundle["global_outline"],
            session_scripts=single_agent_bundle["session_scripts"],
            event_plan=single_agent_bundle["event_plan"],
            emotion_arc=single_agent_bundle["emotion_arc"],
        )
        conversation = client.chat_json(
            system_prompt=system_prompt,
            user_prompt=user_payload,
            timeout=int(conversation_config.get("request_timeout_seconds", 1200)),
        )
    conversation = normalize_conversation_metadata_fn(conversation, bundle["personas"])
    save_json_fn(conversation, raw_out_path)
    leak_report = collect_voice_style_leak_report_fn(conversation)
    save_json_fn(leak_report, leak_json_path)
    if leak_report:
        lines = []
        for item in leak_report:
            lines.append(
                f"{item.get('dia_id')} | {item.get('speaker')} | {', '.join(item.get('matched_rules', []))} | {item.get('voice_style')}"
            )
        save_text_fn("\n".join(lines), leak_text_path)
    else:
        save_text_fn("No voice_style leak candidates detected after normalization.", leak_text_path)
    try:
        validate_conversation_fn(conversation, schemas_dir / "conversation.schema.json")
        validate_conversation_against_blueprint_fn(
            conversation,
            bundle["global_outline"],
            bundle["session_scripts"],
        )
    except Exception as exc:
        save_text_fn(str(exc), error_out_path)
        raise
    save_json_fn(conversation, out_path)
    return conversation


def generate_annotations_from_bundle(
    client: Any,
    prompts_dir: Path,
    schemas_dir: Path,
    bundle: Dict[str, Any],
    conversation_path: Path,
    out_path: Path,
    config: Optional[Dict[str, Any]] = None,
    *,
    load_json_fn: Callable[[Path], Any],
    save_json_fn: Callable[[Any, Path], None],
    render_prompt_template_fn: Callable[[Path, str, str], str],
    get_prompt_language_fn: Callable[[Optional[Dict[str, Any]]], str],
    normalize_annotations_metadata_fn: Callable[[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]], List[Dict[str, Any]]],
    validate_annotations_fn: Callable[[List[Dict[str, Any]], Dict[str, Any], Path], None],
) -> List[Dict[str, Any]]:
    system_prompt = render_prompt_template_fn(prompts_dir, "annotation_prompt_v2.j2", get_prompt_language_fn(config))
    conversation = load_json_fn(conversation_path)
    user_payload = build_annotation_user_payload(conversation=conversation, event_plan=bundle["event_plan"])
    annotations = client.chat_json_array(system_prompt=system_prompt, user_prompt=user_payload)
    annotations = normalize_annotations_metadata_fn(annotations, conversation, bundle["personas"])
    validate_annotations_fn(annotations, conversation, schemas_dir / "annotation.schema.json")
    save_json_fn(annotations, out_path)
    return annotations
