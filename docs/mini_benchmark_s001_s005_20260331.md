# Mini Benchmark S001-S005 (2026-03-31)

## Scope

This file records the first small benchmark set built from:

- `scenario_001`
- `scenario_002`
- `scenario_003`
- `scenario_004`
- `scenario_005`

The goal of this set is not to be the final benchmark. It is a compact, iterated testbed for checking:

- whether question difficulty is high enough
- whether late-session easy summary questions have been reduced
- whether the benchmark relies more on path/mechanism questions than result-summary questions
- whether the current question curation rules are stable enough to reuse

## Current combined dataset

Combined units file:

- `data/generated_batches/blueprint_review_5pack_closure_mix/all_units_s001_s005_pruned_20260331.json`

Current total:

- `88` items

## Scenario counts

- `scenario_001`: `19`
- `scenario_002`: `19`
- `scenario_003`: `17`
- `scenario_004`: `17`
- `scenario_005`: `16`

## Combined distribution

### Memory level

- `Level 3`: `65`
- `Level 2`: `4`
- `Level 1`: `12`
- `Level 0`: `7`

### Content type

- `long_term_implicit_emotion`: `51`
- `long_term_explicit_or_semi_explicit_emotion`: `18`
- `long_term_fact`: `9`
- `instant_emotion`: `7`
- `near_term_fact`: `2`
- `relation_change`: `1`

### Question type

- `judgment`: `45`
- `explanation`: `34`
- `retrieval`: `7`
- `modality_ambiguous`: `1`
- `modality_missing`: `1`

### Reasoning structure

- `multi-hop`: `40`
- `trajectory-based`: `17`
- `conflict-resolution`: `14`
- `single-hop`: `13`
- `direct`: `4`

### Anchor session

- `D1`: `8`
- `D2`: `21`
- `D3`: `26`
- `D4`: `22`
- `D5`: `11`

### Adversarial type

- `none`: `77`
- `pseudo_relevant_history`: `9`
- `insufficient_evidence`: `2`

## Current curation rules that matter most

The current mini benchmark is not prompt-only. It depends on post-generation curation.

The most important active rules are:

- same-anchor light deduplication
- late result-over-path question downweighting
- obvious-easy pruning
- hard-core structural filtering
- support count tied to kept hard-core count

Files:

- [question_curation.py](../question_curation.py)
- [question_pipeline.py](../question_pipeline.py)
- [question_phase_planning.py](../question_phase_planning.py)
- [question_generation_runtime.py](../question_generation_runtime.py)

## What was pruned most recently

The latest cleanup pass mainly removed:

- shallow single-hop retrieval questions
- late-session result-summary questions that were too easy

This particularly affected:

- `scenario_002`
- `scenario_003`

while `scenario_001`, `scenario_004`, and `scenario_005` were already relatively clean under the current rules.

## Intended use

This set should be treated as:

- the first reusable mini benchmark
- the baseline set for short evaluation cycles
- the reference set for future prompt and curation comparisons

It is small enough to iterate on quickly, but already structured enough to expose:

- overreliance on late-session summary cues
- weak retrieval controls
- false hard questions under `session_local`
- differences across closure profiles

## Corrected v2 Langfuse baselines

The original Langfuse upload used a collision-prone dataset item id and only surfaced `73` unique items in Langfuse UI. That bug was fixed in [langfuse_upload_dataset.py](../scripts/langfuse_upload_dataset.py), and the corrected evaluation dataset is:

- `emotion-memory-mini-s001-s005-pruned-v2-20260331`

All model comparisons below refer to this corrected `88`-item dataset.

### Gemini 2.5 Flash

Task model:

- `portkey_gemini25_flash`

Runs: `09370de1` (main), `3d3a75fd` (retry)

Combined result:

- total items: `88`
- open-ended judged items: `41`
- option / exact-match items: `48`
- `llm_judge_final`: `0.808`
- `judge_core_answer_correct`: `0.857`
- `exact_match`: `0.681`
- `judge_key_point_coverage`: `0.726`
- `judge_rationale_alignment`: `0.743`
- `judge_unsupported_claims`: `0.939`
- `judge_insufficiency_handling`: `0.951`

### Gemini 3.1 Pro High

Task model:

- `fanzhongzhuan_gemini31_pro_high`

Runs: `732c7486` (main), `0ca74963` (retry)

Combined result:

- total items: `88`
- open-ended judged items: `43`
- option / exact-match items: `49`
- `llm_judge_final`: `0.699`
- `judge_core_answer_correct`: `0.755`
- `exact_match`: `0.917`
- `judge_key_point_coverage`: `0.603`
- `judge_rationale_alignment`: `0.632`
- `judge_unsupported_claims`: `0.848`
- `judge_insufficiency_handling`: `0.872`

### Claude Sonnet 4.6

Task model:

- `fanzhongzhuan_claude_sonnet46`

Runs: `17fb257d` (main), `590989d5` (retry)

Combined result:

- total items: `88`
- open-ended judged items: `43`
- option / exact-match items: `49`
- `llm_judge_final`: `0.821`
- `judge_core_answer_correct`: `0.863`
- `exact_match`: `0.917`
- `judge_key_point_coverage`: `0.776`
- `judge_rationale_alignment`: `0.748`
- `judge_unsupported_claims`: `0.901`
- `judge_insufficiency_handling`: `0.921`

## Current model comparison on corrected v2

- `Claude Sonnet 4.6 (FAN key)` is currently strongest on this mini benchmark.
- `Gemini 2.5 Flash` is second.
- `Gemini 3.1 Pro High` is weakest on open-ended explanation quality, even though its `exact_match` is high.

Key comparison:

- `llm_judge_final`
  - Gemini 2.5: `0.808`
  - Gemini 3.1: `0.699`
  - Claude 4.6: `0.821`
- `judge_core_answer_correct`
  - Gemini 2.5: `0.857`
  - Gemini 3.1: `0.755`
  - Claude 4.6: `0.863`
- `exact_match`
  - Gemini 2.5: `0.681`
  - Gemini 3.1: `0.917`
  - Claude 4.6: `0.917`
- `judge_key_point_coverage`
  - Gemini 2.5: `0.726`
  - Gemini 3.1: `0.603`
  - Claude 4.6: `0.776`


## Structural breakdown by benchmark dimension

The table below uses a unified comparison rule:

- open-ended items use `llm_judge_final`
- option items use `exact_match`
- the reported average is this merged `primary_score`

### By memory level

- `Level 0`
  - Gemini 2.5: `0.600` (`n=5`)
  - Gemini 3.1: `1.000` (`n=5`)
  - Claude 4.6: `0.800` (`n=5`)
- `Level 1`
  - Gemini 2.5: `0.900` (`n=10`)
  - Gemini 3.1: `0.921` (`n=10`)
  - Claude 4.6: `0.957` (`n=10`)
- `Level 2`
  - Gemini 2.5: `0.816` (`n=4`)
  - Gemini 3.1: `0.619` (`n=4`)
  - Claude 4.6: `0.456` (`n=4`)
- `Level 3`
  - Gemini 2.5: `0.722` (`n=60`)
  - Gemini 3.1: `0.793` (`n=61`)
  - Claude 4.6: `0.895` (`n=61`)

### By reasoning structure

- `single-hop`
  - Gemini 2.5: `0.726`
  - Gemini 3.1: `0.829`
  - Claude 4.6: `0.682`
- `multi-hop`
  - Gemini 2.5: `0.786`
  - Gemini 3.1: `0.833`
  - Claude 4.6: `0.930`
- `trajectory-based`
  - Gemini 2.5: `0.584`
  - Gemini 3.1: `0.683`
  - Claude 4.6: `0.885`
- `conflict-resolution`
  - Gemini 2.5: `0.750`
  - Gemini 3.1: `0.858`
  - Claude 4.6: `0.827`

### By question type

- `retrieval`
  - Gemini 2.5: `0.888`
  - Gemini 3.1: `0.647`
  - Claude 4.6: `0.689`
- `judgment`
  - Gemini 2.5: `0.684`
  - Gemini 3.1: `0.921`
  - Claude 4.6: `0.921`
- `explanation`
  - Gemini 2.5: `0.793`
  - Gemini 3.1: `0.712`
  - Claude 4.6: `0.853`

### By anchor session

- `D1`
  - Gemini 2.5: `0.800`
  - Gemini 3.1: `0.890`
  - Claude 4.6: `1.000`
- `D2`
  - Gemini 2.5: `0.638`
  - Gemini 3.1: `0.796`
  - Claude 4.6: `0.820`
- `D3`
  - Gemini 2.5: `0.829`
  - Gemini 3.1: `0.880`
  - Claude 4.6: `0.895`
- `D4`
  - Gemini 2.5: `0.774`
  - Gemini 3.1: `0.763`
  - Claude 4.6: `0.873`
- `D5`
  - Gemini 2.5: `0.629`
  - Gemini 3.1: `0.754`
  - Claude 4.6: `0.878`

### Takeaways from the structural view

- The most discriminative parts of this mini benchmark are currently `Level 3`, `multi-hop`, and `trajectory-based` items.
- Claude 4.6 leads most clearly on `Level 3`, `multi-hop`, and `trajectory-based` questions.
- Gemini 3.1 is especially strong on `judgment`, but weaker than Claude on `explanation`.
- Gemini 2.5 is relatively strong on `retrieval`, but weaker on later-session and trajectory-heavy items.
