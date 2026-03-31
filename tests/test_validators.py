import unittest
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validators import (
    DataValidationError,
    validate_blueprint_bundle,
    validate_conversation,
    validate_generation_config,
    validate_questions,
)
from pipeline import normalize_conversation_metadata
from pipeline import normalize_annotations_metadata
from pipeline import normalize_questions_metadata
from pipeline import collect_invalid_question_reasons
from pipeline import strip_multi_agent_fields_from_bundle
from pipeline import generate_conversation_from_bundle


def write_json(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def make_conversation() -> dict:
    return {
        "conversation_id": "conv_001",
        "scenario_type": "psychological_teacher_student",
        "speaker_a": "Teacher",
        "speaker_b": "Student",
        "target_speaker": "Student",
        "session_1_date_time": "2024-03-10T10:00:00Z",
        "session_1": [
            {
                "speaker": "Teacher",
                "dia_id": "D1:1",
                "timestamp": "2024-03-10T10:00:00Z",
                "turn_index_global": 1,
                "text": "How are you doing?",
                "audio_id": None,
                "voice_style": "gentle tone, medium pace, slight emphasis on 'doing'",
                "modality_available": ["text", "voice_style"],
            },
            {
                "speaker": "Student",
                "dia_id": "D1:2",
                "timestamp": "2024-03-10T10:01:15Z",
                "turn_index_global": 2,
                "text": "I am fine.",
                "audio_id": "",
                "voice_style": "short pause before answering, clipped delivery, light stress on 'fine'",
                "modality_available": ["text", "voice_style"],
            },
        ],
        "session_2_date_time": "2024-03-17T10:00:00Z",
        "session_2": [
            {
                "speaker": "Teacher",
                "dia_id": "D2:1",
                "timestamp": "2024-03-17T10:00:00Z",
                "turn_index_global": 3,
                "text": "How did the week go?",
                "audio_id": None,
                "voice_style": "warm tone, even pacing, slight upward inflection at the end",
                "modality_available": ["text", "voice_style"],
            },
            {
                "speaker": "Student",
                "dia_id": "D2:2",
                "timestamp": "2024-03-17T10:01:15Z",
                "turn_index_global": 4,
                "text": "Still a bit stressed, but managing.",
                "audio_id": None,
                "voice_style": "audible exhale first, slower pace on 'stressed', quieter finish on 'managing'",
                "modality_available": ["text", "voice_style"],
            },
        ],
    }


def make_questions() -> list[dict]:
    return [
        {
            "question_id": "Q001",
            "anchor_dia_id": "D2:2",
            "content_type": "long_term_implicit_emotion",
            "question_type": "judgment",
            "memory_level": "Level 3",
            "reasoning_structure": "multi-hop",
            "question_text": "What best describes the student's state in the current turn?",
            "options": ["Calm", "Mild stress"],
            "gold_answer": "Mild stress",
            "acceptable_answers": ["Mild stress"],
            "critical_event_ids": [],
            "evidence_turn_ids": ["D1:2", "D2:2"],
            "gold_rationale": "The later turn depends on earlier context.",
            "key_explanation_points": ["Prior context matters."],
            "modality_condition": "normal",
            "adversarial_flag": False,
            "adversarial_type": None,
        }
    ]


class ValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = Path(__file__).resolve().parent / ".tmp_test_schemas"
        cls.temp_dir.mkdir(parents=True, exist_ok=True)
        cls.schemas_dir = cls.temp_dir

        write_json(
            cls.schemas_dir / "conversation.schema.json",
            """
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["conversation_id", "scenario_type", "speaker_a", "speaker_b", "target_speaker"],
  "properties": {
    "conversation_id": {"type": "string"},
    "scenario_type": {"type": "string"},
    "speaker_a": {"type": "string"},
    "speaker_b": {"type": "string"},
    "target_speaker": {"type": "string"}
  },
  "patternProperties": {
    "^session_[0-9]+_date_time$": {"type": "string"},
    "^session_[0-9]+$": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["speaker", "dia_id", "timestamp", "turn_index_global", "text", "voice_style", "modality_available"],
        "properties": {
          "speaker": {"type": "string"},
          "dia_id": {"type": "string", "pattern": "^D[0-9]+:[0-9]+$"},
          "timestamp": {"type": "string"},
          "turn_index_global": {"type": "integer", "minimum": 1},
          "text": {"type": "string"},
          "audio_id": {"type": ["string", "null"]},
          "voice_style": {"type": "string", "minLength": 1},
          "modality_available": {"type": "array", "items": {"type": "string"}}
        }
      }
    }
  }
}
""".strip(),
        )

        write_json(
            cls.schemas_dir / "qa.schema.json",
            """
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "array",
  "items": {
    "type": "object",
    "required": [
      "question_id",
      "anchor_dia_id",
      "content_type",
      "question_type",
      "memory_level",
      "reasoning_structure",
      "question_text",
      "gold_answer",
      "evidence_turn_ids",
      "gold_rationale",
      "adversarial_flag"
    ],
    "properties": {
      "question_id": {"type": "string"},
      "anchor_dia_id": {"type": "string", "pattern": "^D[0-9]+:[0-9]+$"},
      "content_type": {"type": "string"},
      "question_type": {"type": "string"},
      "memory_level": {"type": "string"},
      "reasoning_structure": {
        "type": "string",
        "enum": ["direct", "single-hop", "multi-hop", "conflict-resolution", "trajectory-based"]
      },
      "question_text": {"type": "string"},
      "options": {"type": "array", "items": {"type": "string"}},
      "gold_answer": {},
      "acceptable_answers": {"type": "array"},
      "critical_event_ids": {"type": "array", "items": {"type": "string"}},
      "evidence_turn_ids": {"type": "array", "items": {"type": "string", "pattern": "^D[0-9]+:[0-9]+$"}},
      "gold_rationale": {"type": "string"},
      "key_explanation_points": {"type": "array", "items": {"type": "string"}},
      "modality_condition": {"type": ["string", "null"]},
      "adversarial_flag": {"type": "boolean"},
      "adversarial_type": {"type": ["string", "null"]}
    }
  }
}
""".strip(),
        )

        write_json(
            cls.schemas_dir / "generation_config.schema.json",
            """
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": [
    "batch_name",
    "dataset_count",
    "llm",
    "scenario_type",
    "target_role",
    "sessions_per_conversation",
    "turns_per_conversation",
    "stage_count",
    "question_count",
    "output_root",
    "output_naming",
    "persona_selection",
    "blueprint_generation"
  ],
  "properties": {
    "batch_name": {"type": "string"},
    "dataset_count": {"type": "integer", "minimum": 1},
    "llm": {
      "type": "object",
      "required": ["provider", "model", "api_key_env"],
      "properties": {
        "provider": {"type": "string"},
        "model": {"type": "string"},
        "base_url": {"type": "string"},
        "api_key_env": {"type": "string"}
      }
    },
    "scenario_type": {"type": "string"},
    "target_role": {"type": "string"},
    "sessions_per_conversation": {"type": "integer", "minimum": 1},
    "turns_per_conversation": {"type": "integer", "minimum": 2},
    "stage_count": {"type": "integer", "minimum": 1},
    "question_count": {"type": "integer", "minimum": 1},
    "output_root": {"type": "string"},
    "output_naming": {
      "type": "object",
      "required": ["append_timestamp", "timestamp_format"],
      "properties": {
        "append_timestamp": {"type": "boolean"},
        "timestamp_format": {"type": "string"}
      }
    },
    "persona_selection": {
      "type": "object",
      "required": ["mode", "seed"],
      "properties": {
        "mode": {"type": "string"},
        "seed": {"type": "integer"}
      }
    },
    "blueprint_generation": {
      "type": "object",
      "required": ["enable_repair", "max_repair_attempts"],
      "properties": {
        "enable_repair": {"type": "boolean"},
        "max_repair_attempts": {"type": "integer", "minimum": 0}
      }
    }
  }
}
""".strip(),
        )

        write_json(
            cls.schemas_dir / "personas.schema.json",
            """
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["student", "teacher"],
  "properties": {
    "student": {"type": "object"},
    "teacher": {"type": "object"}
  }
}
""".strip(),
        )

        write_json(
            cls.schemas_dir / "global_outline.schema.json",
            """
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["scenario_type", "target_role", "total_sessions", "total_turns", "overall_arc", "stages"],
  "properties": {
    "scenario_type": {"type": "string"},
    "target_role": {"type": "string"},
    "total_sessions": {"type": "integer"},
    "total_turns": {"type": "integer"},
    "overall_arc": {"type": "string"},
    "stages": {"type": "array"}
  }
}
""".strip(),
        )

        write_json(
            cls.schemas_dir / "session_scripts.schema.json",
            """
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "array",
  "items": {
    "type": "object",
    "required": ["session_id", "stage_id", "turn_count"]
  }
}
""".strip(),
        )

        write_json(
            cls.schemas_dir / "event_plan.schema.json",
            """
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "array",
  "items": {
    "type": "object",
    "required": ["event_id", "session_id", "event_type", "description", "emotional_significance", "relation_impact", "can_be_critical_memory", "can_be_distractor"]
  }
}
""".strip(),
        )

        write_json(
            cls.schemas_dir / "emotion_arc.schema.json",
            """
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "array",
  "items": {
    "type": "object",
    "required": ["stage_id", "student_dominant_emotions", "implicit_emotions_to_seed", "relation_state"]
  }
}
""".strip(),
        )

        write_json(
            cls.schemas_dir / "question_plan.schema.json",
            """
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "mvp_question_count": {"type": "integer"}
  }
}
""".strip(),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_validate_conversation_accepts_well_formed_data(self) -> None:
        validate_conversation(
            make_conversation(),
            self.schemas_dir / "conversation.schema.json",
        )

    def test_validate_conversation_rejects_non_contiguous_turn_index(self) -> None:
        conversation = make_conversation()
        conversation["session_2"][0]["turn_index_global"] = 5

        with self.assertRaisesRegex(DataValidationError, "turn_index_global must be contiguous"):
            validate_conversation(
                conversation,
                self.schemas_dir / "conversation.schema.json",
            )

    def test_validate_questions_rejects_missing_anchor(self) -> None:
        questions = make_questions()
        questions[0]["anchor_dia_id"] = "D9:9"

        with self.assertRaisesRegex(DataValidationError, "anchor_dia_id not found"):
            validate_questions(
                questions,
                make_conversation(),
                self.schemas_dir / "qa.schema.json",
            )

    def test_validate_questions_rejects_level3_without_prior_history(self) -> None:
        questions = make_questions()
        questions[0]["evidence_turn_ids"] = ["D2:2"]

        with self.assertRaisesRegex(DataValidationError, "Level 3 but has no evidence turn before its anchor"):
            validate_questions(
                questions,
                make_conversation(),
                self.schemas_dir / "qa.schema.json",
            )

    def test_validate_questions_accepts_legacy_reasoning_structure_alias(self) -> None:
        questions = make_questions()
        questions[0]["reasoning_structure"] = "trajectory_based"

        validate_questions(
            questions,
            make_conversation(),
            self.schemas_dir / "qa.schema.json",
        )

    def test_validate_questions_accepts_single_hop_reasoning_alias(self) -> None:
        questions = make_questions()
        questions[0]["reasoning_structure"] = "single_hop"

        validate_questions(
            questions,
            make_conversation(),
            self.schemas_dir / "qa.schema.json",
        )

    def test_validate_questions_accepts_title_case_reasoning_alias(self) -> None:
        questions = make_questions()
        questions[0]["reasoning_structure"] = "Single-hop"

        validate_questions(
            questions,
            make_conversation(),
            self.schemas_dir / "qa.schema.json",
        )

    def test_validate_questions_rejects_modality_item_with_replacement_cue(self) -> None:
        questions = make_questions()
        questions[0]["question_type"] = "modality_missing"
        questions[0]["modality_condition"] = "voice_style_removed"
        questions[0]["question_text"] = (
            "If the voice style in D2:2 was described as 'sarcastic and dismissive' instead of "
            "'hesitant and quiet', what would the emotion be?"
        )

        with self.assertRaisesRegex(DataValidationError, "should not replace the original cue"):
            validate_questions(
                questions,
                make_conversation(),
                self.schemas_dir / "qa.schema.json",
            )

    def test_validate_questions_rejects_pseudo_conflict_with_fabricated_prior_dialogue(self) -> None:
        questions = make_questions()
        questions[0]["adversarial_flag"] = True
        questions[0]["adversarial_type"] = "pseudo_conflict"
        questions[0]["question_text"] = (
            "If he had previously stated in D1:2 that he always acknowledges his feelings first, "
            "how would that conflict with D2:2?"
        )

        with self.assertRaisesRegex(DataValidationError, "should not inject fabricated prior dialogue"):
            validate_questions(
                questions,
                make_conversation(),
                self.schemas_dir / "qa.schema.json",
            )

    def test_validate_questions_rejects_explicit_session_or_turn_references_in_question_text(self) -> None:
        questions = make_questions()
        questions[0]["question_text"] = "Explain the shift from the end of Session 2 to D3:10."

        with self.assertRaisesRegex(DataValidationError, "should not mention session labels or turn IDs explicitly"):
            validate_questions(
                questions,
                make_conversation(),
                self.schemas_dir / "qa.schema.json",
            )

    def test_validate_conversation_rejects_missing_voice_style(self) -> None:
        conversation = make_conversation()
        conversation["session_1"][0]["voice_style"] = "   "

        with self.assertRaisesRegex(DataValidationError, "missing voice_style"):
            validate_conversation(
                conversation,
                self.schemas_dir / "conversation.schema.json",
            )

    def test_validate_conversation_rejects_audio_modality_without_audio_id(self) -> None:
        conversation = make_conversation()
        conversation["session_1"][0]["modality_available"] = ["text", "audio", "voice_style"]

        with self.assertRaisesRegex(DataValidationError, "lists audio modality but audio_id is empty"):
            validate_conversation(
                conversation,
                self.schemas_dir / "conversation.schema.json",
            )

    def test_normalize_conversation_metadata_repairs_legacy_speaker_labels(self) -> None:
        conversation = make_conversation()
        conversation["speaker_a"] = "..."
        conversation["speaker_b"] = ""
        conversation["target_speaker"] = "Student"
        conversation["session_1"][0]["speaker"] = "Teacher"
        conversation["session_1"][1]["speaker"] = "Student"
        conversation["session_2"][0]["speaker"] = "psychological teacher"
        conversation["session_2"][1]["speaker"] = "student"

        normalized = normalize_conversation_metadata(
            conversation,
            {
                "teacher": {"name": "Dr. Carter"},
                "student": {"name": "Ethan Brooks"},
            },
        )

        self.assertEqual(normalized["speaker_a"], "Dr. Carter")
        self.assertEqual(normalized["speaker_b"], "Ethan Brooks")
        self.assertEqual(normalized["target_speaker"], "Ethan Brooks")
        self.assertEqual(normalized["session_1"][0]["speaker"], "Dr. Carter")
        self.assertEqual(normalized["session_1"][1]["speaker"], "Ethan Brooks")
        self.assertEqual(normalized["session_2"][0]["speaker"], "Dr. Carter")
        self.assertEqual(normalized["session_2"][1]["speaker"], "Ethan Brooks")
        validate_conversation(
            normalized,
            self.schemas_dir / "conversation.schema.json",
        )

    def test_normalize_conversation_metadata_cleans_common_mojibake(self) -> None:
        conversation = make_conversation()
        conversation["session_1"][0]["text"] = "It鈥?s been a long week."
        conversation["session_1"][1]["voice_style"] = "soft â€¦ slightly hesitant"

        normalized = normalize_conversation_metadata(
            conversation,
            {
                "teacher": {"name": "Dr. Carter"},
                "student": {"name": "Ethan Brooks"},
            },
        )

        self.assertEqual(normalized["session_1"][0]["text"], "It's been a long week.")
        self.assertEqual(normalized["session_1"][1]["voice_style"], "soft ... slightly hesitant")

    def test_strip_multi_agent_fields_from_bundle_removes_interaction_targets(self) -> None:
        bundle = {
            "personas": {},
            "global_outline": {},
            "session_scripts": [],
            "emotion_arc": [],
            "event_plan": [
                {
                    "event_id": "E1",
                    "event_type": "support_offer",
                    "interaction_targets": {
                        "teacher_move_target": "stay practical",
                    },
                }
            ],
        }

        stripped = strip_multi_agent_fields_from_bundle(bundle)

        self.assertIn("interaction_targets", bundle["event_plan"][0])
        self.assertNotIn("interaction_targets", stripped["event_plan"][0])

    def test_generate_conversation_from_bundle_reuses_existing_output(self) -> None:
        scenario_dir = self.temp_dir / "reuse_case"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        out_path = scenario_dir / "conversation.json"
        existing = make_conversation()
        out_path.write_text(__import__("json").dumps(existing, ensure_ascii=False), encoding="utf-8")

        class DummyClient:
            pass

        bundle = {
            "personas": {
                "teacher": {"name": "Teacher"},
                "student": {"name": "Student"},
            },
            "global_outline": {},
            "session_scripts": [],
            "event_plan": [],
            "emotion_arc": [],
        }

        loaded = generate_conversation_from_bundle(
            client=DummyClient(),
            prompts_dir=PROJECT_ROOT / "prompts",
            schemas_dir=self.schemas_dir,
            bundle=bundle,
            out_path=out_path,
            config={"conversation_generation": {"mode": "single_agent", "framework": "autogen", "max_turn_retries": 0}},
        )

        self.assertEqual(loaded["conversation_id"], existing["conversation_id"])

    def test_normalize_annotations_metadata_repairs_legacy_speaker_labels(self) -> None:
        conversation = make_conversation()
        annotations = [
            {
                "dia_id": "D1:2",
                "speaker": "Student",
                "underlying_emotion": "stress",
                "secondary_emotion": None,
                "implicit_explicit": "Implicit",
                "expression_style": "guarded",
                "emotion_intensity": "Low",
                "relation_state": "guarded",
                "historical_memory_required": False,
                "memory_dependency_level": "level0",
                "reasoning_structure": "direct",
                "critical_event_ids": [],
                "evidence_turn_ids": ["D1:1", "D1:2"],
                "evidence_turn_quotes": [
                    {"dia_id": "D1:1", "speaker": "Teacher", "text_evidence": "How are you doing?", "voice_style": "extra"},
                    {"dia_id": "D1:2", "speaker": "Student", "text_evidence": "I am fine."},
                ],
                "gold_rationale": "test",
                "target_role": "Student",
            }
        ]

        normalized = normalize_annotations_metadata(
            annotations,
            conversation,
            {
                "teacher": {"name": "Dr. Carter"},
                "student": {"name": "Ethan Brooks"},
            },
        )

        self.assertEqual(normalized[0]["speaker"], "Student")
        self.assertEqual(normalized[0]["evidence_turn_quotes"][0]["speaker"], "Teacher")
        self.assertEqual(normalized[0]["evidence_turn_quotes"][1]["speaker"], "Student")
        self.assertEqual(normalized[0]["target_role"], "student")
        self.assertEqual(normalized[0]["memory_dependency_level"], "Level 0")
        self.assertNotIn("voice_style", normalized[0]["evidence_turn_quotes"][0])

    def test_normalize_questions_metadata_repairs_inconsistent_memory_level(self) -> None:
        questions = make_questions()
        questions[0]["memory_level"] = "Level 3"
        questions[0]["evidence_turn_ids"] = ["D2:2"]

        normalized = normalize_questions_metadata(questions, make_conversation())

        self.assertEqual(normalized[0]["memory_level"], "Level 0")

    def test_normalize_questions_metadata_replaces_null_arrays(self) -> None:
        questions = make_questions()
        questions[0]["options"] = None
        questions[0]["acceptable_answers"] = None
        questions[0]["critical_event_ids"] = None
        questions[0]["key_explanation_points"] = None

        normalized = normalize_questions_metadata(questions, make_conversation())

        self.assertEqual(normalized[0]["options"], [])
        self.assertEqual(normalized[0]["acceptable_answers"], [])
        self.assertEqual(normalized[0]["critical_event_ids"], [])
        self.assertEqual(normalized[0]["key_explanation_points"], [])

    def test_normalize_questions_metadata_repairs_common_enum_variants(self) -> None:
        questions = make_questions()
        questions[0]["question_type"] = "Judgement"
        questions[0]["content_type"] = "Long Term Implicit Emotion"
        questions[0]["modality_condition"] = "modality ambiguous"
        questions[0]["adversarial_flag"] = True
        questions[0]["adversarial_type"] = "pseudo conflict"

        normalized = normalize_questions_metadata(questions, make_conversation())

        self.assertEqual(normalized[0]["question_type"], "judgment")
        self.assertEqual(normalized[0]["content_type"], "long_term_implicit_emotion")
        self.assertEqual(normalized[0]["modality_condition"], "modality_ambiguous")
        self.assertEqual(normalized[0]["adversarial_type"], "pseudo_conflict")
        self.assertEqual(normalized[0]["reasoning_structure"], "multi-hop")

    def test_collect_invalid_question_reasons_flags_future_evidence(self) -> None:
        questions = make_questions()
        questions[0]["anchor_dia_id"] = "D1:2"
        questions[0]["evidence_turn_ids"] = ["D1:2", "D2:2"]

        invalid = collect_invalid_question_reasons(questions, make_conversation())

        self.assertIn("Q001", invalid)
        self.assertTrue(any("future evidence" in reason for reason in invalid["Q001"]))

    def test_collect_invalid_question_reasons_flags_retrieval_explanation_mismatch(self) -> None:
        questions = make_questions()
        questions[0]["question_type"] = "retrieval"
        questions[0]["question_text"] = "Explain how this relates to the student's earlier coping pattern."

        invalid = collect_invalid_question_reasons(questions, make_conversation())

        self.assertIn("Q001", invalid)
        self.assertTrue(any("retrieval-labeled" in reason for reason in invalid["Q001"]))

    def test_collect_invalid_question_reasons_flags_duplicate_question_text(self) -> None:
        questions = make_questions() + [
            {
                **make_questions()[0],
                "question_id": "Q002",
            }
        ]

        invalid = collect_invalid_question_reasons(questions, make_conversation())

        self.assertIn("Q002", invalid)
        self.assertTrue(any("duplicates question wording" in reason for reason in invalid["Q002"]))

    def test_collect_invalid_question_reasons_flags_explicit_session_or_turn_references(self) -> None:
        questions = make_questions()
        questions[0]["question_text"] = "What changed between Session 1 and S2 before D2:2?"

        invalid = collect_invalid_question_reasons(questions, make_conversation())

        self.assertIn("Q001", invalid)
        self.assertTrue(any("session labels or turn IDs" in reason for reason in invalid["Q001"]))

    def test_collect_invalid_question_reasons_flags_duplicate_closure_signal_questions(self) -> None:
        conversation = {
            **make_conversation(),
            "session_2": [
                {
                    "speaker": "Teacher",
                    "dia_id": "D2:1",
                    "timestamp": "2024-03-17T10:00:00Z",
                    "turn_index_global": 3,
                    "text": "See you next week.",
                    "audio_id": None,
                    "voice_style": "warm closing",
                    "modality_available": ["text", "voice_style"],
                },
                {
                    "speaker": "Student",
                    "dia_id": "D2:2",
                    "timestamp": "2024-03-17T10:01:15Z",
                    "turn_index_global": 4,
                    "text": "Goodbye.",
                    "audio_id": None,
                    "voice_style": "brief but softer than before",
                    "modality_available": ["text", "voice_style"],
                },
            ],
        }
        questions = [
            {
                **make_questions()[0],
                "question_id": "Q001",
                "anchor_dia_id": "D2:1",
                "question_text": "What does the casual goodbye reveal about the student's comfort level?",
                "evidence_turn_ids": ["D1:2", "D2:1"],
            },
            {
                **make_questions()[0],
                "question_id": "Q002",
                "anchor_dia_id": "D2:2",
                "question_text": "What does the goodbye signal about the student's relationship with the teacher?",
                "evidence_turn_ids": ["D1:2", "D2:1", "D2:2"],
            },
        ]

        invalid = collect_invalid_question_reasons(questions, conversation)

        self.assertIn("Q002", invalid)
        self.assertTrue(any("closure/relationship-easing signal" in reason for reason in invalid["Q002"]))


    def test_validate_generation_config_accepts_reasonable_scale(self) -> None:
        config = {
            "batch_name": "demo",
            "dataset_count": 3,
            "llm": {"provider": "gateway", "model": "demo-model", "base_url": "https://example.com/v1", "api_key_env": "DEMO_API_KEY"},
            "scenario_type": "psychological_teacher_student",
            "target_role": "student",
            "sessions_per_conversation": 3,
            "turns_per_conversation": 30,
            "stage_count": 3,
            "question_count": 10,
            "output_root": "data/generated_batches",
            "output_naming": {"append_timestamp": True, "timestamp_format": "%Y%m%d_%H%M%S"},
            "persona_selection": {"mode": "round_robin", "seed": 42},
            "blueprint_generation": {"enable_repair": True, "max_repair_attempts": 1},
        }

        validate_generation_config(
            config,
            self.schemas_dir / "generation_config.schema.json",
        )

    def test_validate_generation_config_accepts_random_persona_mode(self) -> None:
        config = {
            "batch_name": "demo",
            "dataset_count": 1,
            "llm": {"provider": "gateway", "model": "demo-model", "base_url": "https://example.com/v1", "api_key_env": "DEMO_API_KEY"},
            "scenario_type": "psychological_teacher_student",
            "target_role": "student",
            "sessions_per_conversation": 3,
            "turns_per_conversation": 30,
            "stage_count": 3,
            "question_count": 10,
            "output_root": "data/generated_batches",
            "output_naming": {"append_timestamp": True, "timestamp_format": "%Y%m%d_%H%M%S"},
            "persona_selection": {"mode": "random", "seed": 42},
            "blueprint_generation": {"enable_repair": True, "max_repair_attempts": 1},
        }

        validate_generation_config(
            config,
            self.schemas_dir / "generation_config.schema.json",
        )

    def test_validate_generation_config_rejects_too_many_stages(self) -> None:
        config = {
            "batch_name": "demo",
            "dataset_count": 1,
            "llm": {"provider": "gateway", "model": "demo-model", "base_url": "https://example.com/v1", "api_key_env": "DEMO_API_KEY"},
            "scenario_type": "psychological_teacher_student",
            "target_role": "student",
            "sessions_per_conversation": 2,
            "turns_per_conversation": 20,
            "stage_count": 3,
            "question_count": 8,
            "output_root": "data/generated_batches",
            "output_naming": {"append_timestamp": True, "timestamp_format": "%Y%m%d_%H%M%S"},
            "persona_selection": {"mode": "round_robin", "seed": 42},
            "blueprint_generation": {"enable_repair": False, "max_repair_attempts": 0},
        }

        with self.assertRaisesRegex(DataValidationError, "stage_count cannot exceed"):
            validate_generation_config(
                config,
                self.schemas_dir / "generation_config.schema.json",
            )

    def test_validate_blueprint_bundle_rejects_mismatched_turn_sum(self) -> None:
        config = {
            "sessions_per_conversation": 2,
            "turns_per_conversation": 12,
            "stage_count": 2,
            "question_count": 6,
        }
        personas = {"student": {"role_id": "student_01"}, "teacher": {"role_id": "teacher_01"}}
        global_outline = {
            "scenario_type": "psychological_teacher_student",
            "target_role": "student",
            "total_sessions": 2,
            "total_turns": 12,
            "overall_arc": "test arc",
            "stages": [
                {"stage_id": "stage_1", "stage_name": "one", "session_span": [1], "goal": "goal", "relationship_state": "state", "emotional_background": ["anxiety"], "key_function": "plant"},
                {"stage_id": "stage_2", "stage_name": "two", "session_span": [2], "goal": "goal", "relationship_state": "state", "emotional_background": ["relief"], "key_function": "repair"}
            ],
        }
        session_scripts = [
            {"session_id": "S1", "stage_id": "stage_1", "turn_count": 5},
            {"session_id": "S2", "stage_id": "stage_2", "turn_count": 5},
        ]
        event_plan = [
            {"event_id": "E1", "session_id": "S1", "event_type": "academic_pressure", "description": "x", "emotional_significance": "high", "relation_impact": "background_only", "can_be_critical_memory": True, "can_be_distractor": True}
        ]
        emotion_arc = [
            {"stage_id": "stage_1", "student_dominant_emotions": ["anxiety"], "implicit_emotions_to_seed": ["stress"], "relation_state": "cautious"},
            {"stage_id": "stage_2", "student_dominant_emotions": ["relief"], "implicit_emotions_to_seed": ["trust"], "relation_state": "repair"}
        ]
        question_plan = {"mvp_question_count": 6}

        with self.assertRaisesRegex(DataValidationError, "total_turns must equal sum of session turn counts"):
            validate_blueprint_bundle(
                personas,
                global_outline,
                session_scripts,
                event_plan,
                emotion_arc,
                question_plan,
                self.schemas_dir,
                config=config,
            )

    def test_validate_blueprint_bundle_rejects_sparse_event_plan(self) -> None:
        config = {
            "sessions_per_conversation": 2,
            "turns_per_conversation": 20,
            "stage_count": 2,
            "question_count": 6,
        }
        personas = {"student": {"role_id": "student_01"}, "teacher": {"role_id": "teacher_01"}}
        global_outline = {
            "scenario_type": "psychological_teacher_student",
            "target_role": "student",
            "total_sessions": 2,
            "total_turns": 20,
            "overall_arc": "test arc",
            "stages": [
                {"stage_id": "stage_1", "stage_name": "one", "session_span": [1], "goal": "goal", "relationship_state": "state", "emotional_background": ["anxiety"], "key_function": "plant"},
                {"stage_id": "stage_2", "stage_name": "two", "session_span": [2], "goal": "goal", "relationship_state": "state", "emotional_background": ["relief"], "key_function": "repair"}
            ],
        }
        session_scripts = [
            {"session_id": "S1", "stage_id": "stage_1", "turn_count": 10},
            {"session_id": "S2", "stage_id": "stage_2", "turn_count": 10},
        ]
        event_plan = [
            {"event_id": "E1", "session_id": "S1", "event_type": "academic_pressure", "description": "x", "emotional_significance": "high", "relation_impact": "background_only", "can_be_critical_memory": True, "can_be_distractor": True},
            {"event_id": "E2", "session_id": "S1", "event_type": "self_minimization", "description": "y", "emotional_significance": "medium", "relation_impact": "background_only", "can_be_critical_memory": True, "can_be_distractor": True},
        ]
        emotion_arc = [
            {"stage_id": "stage_1", "student_dominant_emotions": ["anxiety"], "implicit_emotions_to_seed": ["stress"], "relation_state": "cautious"},
            {"stage_id": "stage_2", "student_dominant_emotions": ["relief"], "implicit_emotions_to_seed": ["trust"], "relation_state": "repair"}
        ]
        question_plan = {"mvp_question_count": 6}

        with self.assertRaisesRegex(DataValidationError, "should have at least 2 events"):
            validate_blueprint_bundle(
                personas,
                global_outline,
                session_scripts,
                event_plan,
                emotion_arc,
                question_plan,
                self.schemas_dir,
                config=config,
            )

    def test_validate_blueprint_bundle_rejects_non_emotion_terms_in_emotion_fields(self) -> None:
        config = {
            "sessions_per_conversation": 2,
            "turns_per_conversation": 12,
            "stage_count": 2,
            "question_count": 6,
        }
        personas = {"student": {"role_id": "student_01"}, "teacher": {"role_id": "teacher_01"}}
        global_outline = {
            "scenario_type": "psychological_teacher_student",
            "target_role": "student",
            "total_sessions": 2,
            "total_turns": 12,
            "overall_arc": "test arc",
            "stages": [
                {"stage_id": "stage_1", "stage_name": "one", "session_span": [1], "goal": "goal", "relationship_state": "state", "emotional_background": ["anxiety"], "key_function": "plant"},
                {"stage_id": "stage_2", "stage_name": "two", "session_span": [2], "goal": "goal", "relationship_state": "state", "emotional_background": ["politeness"], "key_function": "repair"}
            ],
        }
        session_scripts = [
            {"session_id": "S1", "stage_id": "stage_1", "turn_count": 6, "dominant_student_emotions": ["anxiety"]},
            {"session_id": "S2", "stage_id": "stage_2", "turn_count": 6, "dominant_student_emotions": ["self_minimization"]},
        ]
        event_plan = [
            {"event_id": "E1", "session_id": "S1", "event_type": "academic_pressure", "description": "x", "emotional_significance": "high", "relation_impact": "background_only", "can_be_critical_memory": True, "can_be_distractor": True},
            {"event_id": "E2", "session_id": "S2", "event_type": "peer_conflict", "description": "y", "emotional_significance": "high", "relation_impact": "mild_strain", "can_be_critical_memory": True, "can_be_distractor": False},
        ]
        emotion_arc = [
            {"stage_id": "stage_1", "student_dominant_emotions": ["anxiety"], "implicit_emotions_to_seed": ["stress"], "relation_state": "cautious"},
            {"stage_id": "stage_2", "student_dominant_emotions": ["grievance"], "implicit_emotions_to_seed": ["politeness"], "relation_state": "repair"}
        ]
        question_plan = {"mvp_question_count": 6}

        with self.assertRaisesRegex(DataValidationError, "non-emotion term"):
            validate_blueprint_bundle(
                personas,
                global_outline,
                session_scripts,
                event_plan,
                emotion_arc,
                question_plan,
                self.schemas_dir,
                config=config,
            )


if __name__ == "__main__":
    unittest.main()
