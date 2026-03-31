# Field mapping recommendations

## Core identifiers
- Old `turn` or `anchor_turn_id` -> new `dia_id` / `anchor_dia_id`
- Keep `event_id` only as internal support, not as sole benchmark evidence

## Evidence
- Replace event-only evidence with:
  - `critical_event_ids`
  - `evidence_turn_ids`
  - optional `evidence_turn_quotes`

## Conversation
Each turn must contain:
- `speaker`
- `dia_id`
- `turn_index_global`
- `text`
- `audio_id` (optional; may be null when no audio file exists)
- `voice_style` (required observable delivery metadata: pause, emphasis, pace, softness, hesitation, etc.)
- `modality_available`

## Annotation
Each annotation item must contain:
- `dia_id`
- `underlying_emotion`
- `memory_dependency_level`
- `reasoning_structure`
- `evidence_turn_ids`
- `gold_rationale`

## QA
Each question item must contain:
- `anchor_dia_id`
- `content_type`
- `question_type`
- `evidence_turn_ids`
- `gold_answer`
- `gold_rationale`
