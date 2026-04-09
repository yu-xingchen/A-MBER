# A-MBER: Affective Memory Benchmark for Emotion Recognition

A-MBER is a long-horizon affective memory benchmark that evaluates whether language models can leverage multi-session conversational history to interpret a user's current emotional state — especially when the emotion is **implicit** and the current utterance alone is insufficient.

## Motivation

In real-world agent–user interactions, understanding a user's emotional state often requires recalling past events, tracking relational dynamics, and integrating evidence scattered across multiple sessions. A-MBER targets this capability gap by constructing benchmark scenarios in a psychological teacher–student counseling setting, where:

- Emotions accumulate over sessions (disappointment, defensiveness, suppressed helplessness, etc.)
- Key evidence is distributed across distant dialogue turns
- Surface-level cues are often insufficient for correct interpretation

## Benchmark Structure

A-MBER uses a **core–diagnostic–stress-test** three-layer task design:

| Layer | Purpose | Examples |
|-------|---------|---------|
| Core | Main evaluation distribution | Implicit emotion judgment, retrieval, explanation |
| Diagnostic | Interpretable sub-capability probes | Relation-state judgment, trajectory-based items, multi-hop retrieval |
| Stress-test | Robustness under degraded input | Modality-missing, modality-ambiguous, comparison, ranking |

Adversarial items (pseudo-relevant history, insufficient evidence) are included as a horizontal tag across layers.

## Pipeline Overview

A-MBER is fully generated through a multi-stage pipeline:

```
Personas → Global Outline → Session Scripts → Event Plan → Emotion Arc
    → Question Plan → Conversation → Annotation → QA → Benchmark Units
```

Each scenario produces:

| File | Description |
|------|-------------|
| `personas.json` | Student–teacher persona pair |
| `global_outline.json` | Stage-level arc and narrative structure |
| `session_scripts.json` | Per-session goals, tensions, and signals |
| `event_plan.json` | Critical events supporting benchmark tasks |
| `emotion_arc.json` | Stage-level emotional progression |
| `question_plan.json` | Programmatic question distribution targets |
| `conversation.json` | Full multi-session dialogue |
| `annotation.json` | Turn-level emotional-memory annotations |
| `qa.json` | Task layer (questions + gold answers) |
| `all_units.json` | Final benchmark units with context views |

## Repository Layout

```
├── pipeline.py                    # Main CLI entrypoint
├── blueprint_pipeline.py          # Blueprint stage orchestration
├── interaction_pipeline.py        # Conversation & annotation orchestration
├── question_pipeline.py           # QA generation orchestration
├── question_phase_planning.py     # Question distribution sizing
├── question_generation_runtime.py # QA payload building, retry, validation
├── question_curation.py           # Post-processing, pruning, deduplication
├── generation_payloads.py         # Prompt payload builders
├── conversation_utils.py          # Turn flattening, dia_id, normalization
├── validators.py                  # Schema validation & consistency checks
├── configs/                       # Generation configs & model API profiles
├── prompts/                       # Jinja2 prompt templates
├── schemas/                       # JSON schemas for all data objects
├── templates/                     # Seed templates (persona pool, etc.)
├── scripts/                       # Langfuse eval workflow scripts
├── tests/                         # Unit tests
└── docs/                          # Design docs & methodology
```

## Quick Start

### 1. Generate interaction blueprints

```bash
python pipeline.py --step prepare_blueprints --config-path configs/generation_config.example.json
```

### 2. Run a full batch

```bash
python pipeline.py --step run_batch --config-path configs/generation_config.example.json
```

### 3. Continue from existing blueprints

```bash
python pipeline.py --step run_existing_batch --config-path configs/generation_config.example.json --batch-dir data/generated_batches/<batch_dir>
```

### 4. Re-run a single stage

```bash
python pipeline.py --step generate_conversation --config-path configs/generation_config.example.json --scenario-dir <scenario_dir>
python pipeline.py --step generate_annotations --config-path configs/generation_config.example.json --scenario-dir <scenario_dir>
python pipeline.py --step generate_questions --config-path configs/generation_config.example.json --scenario-dir <scenario_dir>
python pipeline.py --step build_units --scenario-dir <scenario_dir>
```

### 5. Validate a scenario

```bash
python pipeline.py --step validate_scenario --scenario-dir <scenario_dir>
```

## Evaluation with Langfuse

A-MBER includes a Langfuse-based evaluation workflow for running model comparisons:

```bash
# Upload benchmark units
python scripts/langfuse_upload_dataset.py \
  --units-path <all_units.json> \
  --dataset-name <dataset_name>

# Run experiment
python scripts/langfuse_run_experiment.py \
  --dataset-name <dataset_name> \
  --prompt-name <prompt_name> \
  --context-policy session_local \
  --task-profile <model_profile>
```

Two context policies are supported:
- `session_local`: only the current session up to the anchor turn
- `full_history`: complete conversation history up to the anchor turn

Built-in evaluators: `exact_match` (for option questions) and `llm_judge` (for open-ended questions).

### Langfuse Engineering Notes

- Dataset item ids must be globally unique across merged scenario files. Do not assume `conversation_id + question_id` is unique after combining scenarios.
- [scripts/langfuse_upload_dataset.py](/C:/Users/chen4/Desktop/红熊/emotion_memory_benchmark_package/emotion_memory_benchmark_package_pure_text/scripts/langfuse_upload_dataset.py) now builds collision-safe item ids from dataset name plus content-derived hashes and will raise an error before upload if duplicates remain.
- Upload metadata now includes `unit_id`, `conversation_id`, `anchor_dia_id`, `source_units_file`, and `source_marker` so merged datasets can be traced back to source scenarios.
- Local report based retry is part of the intended workflow. When provider-side failures happen, extract retry subsets with [scripts/langfuse_extract_retry_units.py](/C:/Users/chen4/Desktop/红熊/emotion_memory_benchmark_package/emotion_memory_benchmark_package_pure_text/scripts/langfuse_extract_retry_units.py), rerun only failed items, then merge main and retry reports at the item level.
- In this repository, `unit_id` is not globally unique across merged scenarios, so report merging and retry extraction should prefer `question_text + anchor.text` matching when a run mixes multiple scenarios.

## Key Design Decisions

- All benchmark evidence is grounded in **dialogue turn IDs**, not hidden event scripts
- `voice_style` is required for every turn as observable delivery metadata
- Question generation uses a two-phase approach: `hard_core` items first, then `everything_else`
- Post-generation curation includes deduplication, difficulty filtering, and structural pruning
- Conversation generation supports both `single_agent` and `multi_agent` (AutoGen-based) modes
- Evaluation reliability depends on provider retry handling; merged baselines should be computed from main + retry runs, not only from the first Langfuse run

## Documentation

- [Methodology & Pipeline Design](docs/methodology_rationale_and_pipeline.md)
- [Langfuse Evaluation Workflow](docs/langfuse_eval_workflow_zh.md)
- [Multi-Agent Conversation Experiments](docs/multi_agent_conversation_experiments.md)
- [Field Mapping Notes](docs/field_mapping_notes.md)

## License

TBD
