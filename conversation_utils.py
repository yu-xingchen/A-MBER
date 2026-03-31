import re
from typing import Any, Dict


def normalize_level_label(value: Any, fallback: str = "Level 0") -> str:
    if not isinstance(value, str):
        return fallback
    match = re.search(r"level\D*([0-3])", value.strip().lower())
    if not match:
        return fallback
    return f"Level {match.group(1)}"


def extract_session_index_from_dia_id(dia_id: Any) -> int:
    if not isinstance(dia_id, str):
        return -1
    match = re.match(r"D(\d+):\d+$", dia_id.strip())
    return int(match.group(1)) if match else -1


def flatten_conversation_turns(conversation: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    turn_map: Dict[str, Dict[str, Any]] = {}
    for key, value in conversation.items():
        if key.startswith("session_") and isinstance(value, list):
            for turn in value:
                dia_id = turn.get("dia_id")
                if dia_id:
                    turn_map[dia_id] = turn
    return turn_map
